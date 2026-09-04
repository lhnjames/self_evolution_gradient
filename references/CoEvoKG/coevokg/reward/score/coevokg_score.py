import json
import os
import random
import re
import string
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

from coevokg.utils.llm_as_a_judge import get_global_judge
from coevokg.utils.api_provider_pool import FatalProviderPoolExhausted
from coevokg.utils.env import get_coevokg_env

# ---------------------------------------------------------------------------
# Solver format reward
# ---------------------------------------------------------------------------
_FORMAT_REWARD  =  0.1
_FORMAT_PENALTY = -0.1
_INFO_SPLIT_RE  = re.compile(r'<information>.*?</information>', re.DOTALL)
_ANSWER_RE = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
_THINK_RE = re.compile(r'<think>.*?</think>', re.DOTALL)
_SEARCH_RE = re.compile(r'<search>(.*?)</search>', re.DOTALL)
_TITLE_RE = re.compile(r'Title:\s*"([^"]+)"')


def _compute_solver_format_reward_hard(solution_str: str) -> float:
    """Per-turn format check: +0.1 if all assistant turns well-formed, -0.1 otherwise.

    Splits solution_str on <information>…</information> blocks (system-injected
    search results) to isolate each assistant turn, then verifies:
      1. Every turn has <think>…</think>  (prompt requires reasoning every turn)
      2. Last turn has <answer>…</answer> (task complete; absent = max-turns cutoff)
      3. All <search> tags are balanced   (no incomplete calls)
      4. No duplicate search queries      (case-insensitive)
    Returns a single one-time reward — not accumulated per turn.
    """
    segments = [s.strip() for s in _INFO_SPLIT_RE.split(solution_str) if s.strip()]
    if not segments:
        return _FORMAT_PENALTY

    # Rule 1: every assistant turn must contain <think>…</think>
    for seg in segments:
        if not re.search(r'<think>.*?</think>', seg, re.DOTALL):
            return _FORMAT_PENALTY

    # Rule 2: last turn must contain <answer>…</answer>, and exactly one globally
    if not re.search(r'<answer>.*?</answer>', segments[-1], re.DOTALL):
        return _FORMAT_PENALTY
    if len(re.findall(r'<answer>.*?</answer>', solution_str, re.DOTALL)) != 1:
        return _FORMAT_PENALTY

    # Rule 3: balanced <search> / </search> across full response
    if solution_str.count('<search>') != solution_str.count('</search>'):
        return _FORMAT_PENALTY

    # Rule 4: no duplicate search queries (case-insensitive strip)
    queries = [q.strip().lower()
               for q in re.findall(r'<search>(.*?)</search>', solution_str, re.DOTALL)]
    if len(queries) != len(set(queries)):
        return _FORMAT_PENALTY

    return _FORMAT_REWARD


def _compute_solver_format_reward(solution_str: str) -> float:
    """Soft one-time format reward in [-0.1, 0.1]."""
    segments = [s.strip() for s in _INFO_SPLIT_RE.split(solution_str) if s.strip()]
    if not segments:
        return _FORMAT_PENALTY

    score = 0.0

    answer_matches = list(_ANSWER_RE.finditer(solution_str))
    if len(answer_matches) == 1 and _ANSWER_RE.search(segments[-1]):
        score += 0.03
        answer_text = answer_matches[0].group(1).strip()
        if 0 < len(answer_text) <= 80 and "\n" not in answer_text:
            score += 0.02

    if solution_str.count('<search>') == solution_str.count('</search>'):
        score += 0.03
    else:
        score -= 0.03

    if len(segments) > 1:
        post_info_segments = [seg for seg in segments[1:] if seg]
        if post_info_segments and any(_THINK_RE.search(seg) for seg in post_info_segments):
            score += 0.02
    elif _THINK_RE.search(solution_str):
        score += 0.02

    queries = [q.strip().lower() for q in _SEARCH_RE.findall(solution_str)]
    duplicate_count = len(queries) - len(set(queries))
    if duplicate_count > 0:
        score -= min(0.06, 0.02 * duplicate_count)

    return max(_FORMAT_PENALTY, min(_FORMAT_REWARD, score))


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def subem_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            score = 1
            break
    return score


def _token_f1(norm_pred: str, norm_gold: str) -> float:
    """Token-level F1 in the SQuAD style for soft matching."""
    pred_tokens = norm_pred.split()
    gold_tokens = norm_gold.split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _parse_number(s: str) -> Optional[float]:
    """Parse a numeric value from text, including separators and unit words."""
    s = s.strip().replace(",", "").replace("$", "").replace("%", "")
    for word, mult in [("trillion", 1e12), ("billion", 1e9), ("million", 1e6), ("thousand", 1e3)]:
        if word in s.lower():
            s = re.sub(word, "", s, flags=re.IGNORECASE).strip()
            try:
                return float(s.split()[0]) * mult
            except (ValueError, IndexError):
                pass
    try:
        return float(s.split()[0])
    except (ValueError, IndexError):
        return None


def soft_em_check(prediction: str, golden_answers) -> bool:
    """
    Rule-enhanced matching between exact match and the LLM judge.
    It returns True only for high-confidence matches to avoid false positives.

    Covered cases:
      1. Containment: gold appears in the prediction, with length-ratio guards.
      2. Reverse containment: prediction appears in gold, for concise answers.
      3. Token F1 >= 0.9: near-complete overlap with minor wording differences.
      4. Numeric equivalence: equivalent values with units or separators.
    """
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]

    norm_pred = normalize_answer(prediction)
    pred_len = len(norm_pred)

    for gold in golden_answers:
        norm_gold = normalize_answer(gold)
        gold_len = len(norm_gold)

        # 1. Gold appears in prediction and is long enough to avoid short-token false hits.
        #    Length ratio >= 0.3 keeps gold from being buried in a long answer.
        if gold_len >= 4 and norm_gold in norm_pred and gold_len / max(pred_len, 1) >= 0.3:
            return True

        # 2. Prediction appears in gold when the model gives a concise form.
        #    Length ratio >= 0.5 avoids matching isolated word fragments.
        if pred_len >= 4 and norm_pred in norm_gold and pred_len / max(gold_len, 1) >= 0.5:
            return True

        # 3. Token F1 >= 0.9 for near-complete overlap with minor wording differences.
        if _token_f1(norm_pred, norm_gold) >= 0.9:
            return True

        # 4. Numeric equivalence within 1% tolerance, e.g. "1,500,000" == "1.5 million".
        pred_num = _parse_number(norm_pred)
        gold_num = _parse_number(norm_gold)
        if pred_num is not None and gold_num is not None and gold_num != 0:
            if abs(pred_num - gold_num) / abs(gold_num) < 0.01:
                return True

    return False


def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)
    # If there are 0  matches, return None
    if len(matches) < 1:
        return None
    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()


def extract_math_solution(solution_str):
    """Extract the answer from the solution string within <answer></answer> tags,
    returning boxed content if present, otherwise the raw content.
    If no <answer> tags are present, extract and return the last boxed content."""
    answer_pattern = r"<answer>(.*?)</answer>"
    answer_matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))

    if len(answer_matches) < 1:
        
        boxed_pattern = r"\\boxed\{(.*?)\}"
        boxed_matches = list(re.finditer(boxed_pattern, solution_str, re.DOTALL))
        if boxed_matches:
            return boxed_matches[-1].group(1).strip()
        else:
            return None

    answer_content = answer_matches[-1].group(1).strip()

    boxed_pattern = r"\\boxed\{(.*?)\}"
    boxed_match = re.search(boxed_pattern, answer_content)

    if boxed_match:
        return boxed_match.group(1).strip()
    else:
        return answer_content


def count_answer_tags(text):
    opening_tags = text.count("<answer>")
    closing_tags = text.count("</answer>")

    return opening_tags, closing_tags


def _solver_title_chain(solution_str: str, target: str) -> List[str]:
    """Deterministic evidence chain from the solver's retrieved document titles.

    Parses `Title: "..."` from the solution's <information> blocks in retrieval
    order (dedup, case-insensitive), then appends the answer entity. This is the
    LLM-free chain fed to the path-support process reward (mirrors the write-back
    extractor `_solver_chain_from_turns`). Returns [] when < 2 usable entities.
    """
    seen, seq = set(), []
    for title in _TITLE_RE.findall(solution_str or ""):
        k = title.strip().lower()
        if k and k not in seen:
            seen.add(k)
            seq.append(title.strip())
    tgt = str(target or "").strip()
    if tgt and tgt.lower() not in seen:
        seq.append(tgt)
    return seq


def _compute_process_score(extracted_chain: List[str], extra_info: dict) -> float:
    """
    Run PathSupportVerifier on the extracted thought chain.

    chain_data_path must be present in extra_info for path support scoring.
    Returns 0.0 on any error or when chain_data_path is absent.
    """
    if not extracted_chain or len(extracted_chain) < 2:
        return 0.0
    chain_data_path = (extra_info or {}).get("chain_data_path", "")
    if not chain_data_path:
        return 0.0
    try:
        from coevokg.utils.path_support_verifier import get_global_verifier
        verifier = get_global_verifier(chain_data_path)
        result = verifier.verify_chain(extracted_chain)
        return float(result.global_score)
    except Exception as e:
        import logging as _log
        _log.getLogger(__name__).warning(f"PathSupportVerifier failed: {e}")
        return 0.0


def _extract_target(ground_truth) -> str:
    """Normalize ground_truth to a plain target string.

    ground_truth arrives in several formats depending on the data source:
      - {"target": "answer"}  (standard)
      - "answer"              (legacy string)
      - None / anything else  (malformed — treated as empty)
    Centralising this avoids scattered KeyError/TypeError crashes throughout
    compute_score when data is non-standard.
    """
    if isinstance(ground_truth, dict):
        return ground_truth.get("target", "") or ""
    if isinstance(ground_truth, str):
        return ground_truth
    return ""


def _search_behavior_modifier(solution_str: str, data_source: Optional[str], correct: bool) -> float:
    """Score delta based on search behavior (auxiliary reward term R_aux).

    Disabled by default. Set COEVOKG_ENABLE_SEARCH_MODIFIER=1 to re-enable.
    When disabled, the solver reward reduces to R_ans + beta * S_path with no
    auxiliary behavioral-shaping term (max reward 1.3 on correct trajectories).

    When enabled:
    -0.2  no <search> at all — penalise skipping search entirely
    +0.2  feedback_pool + searched + correct — bonus for searching on recycled problems
     0.0  otherwise
    """
    if get_coevokg_env("ENABLE_SEARCH_MODIFIER", "0") != "1":
        return 0.0
    num_searches = (solution_str or "").count("<search>")
    if num_searches == 0:
        return -0.2
    if data_source == "feedback_pool" and correct:
        return 0.2
    return 0.0


def compute_score(
    data_source=None,
    solution_str=None,
    ground_truth=None,
    prompt_str=None,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    kg_sink=None,
):
    base_url = get_coevokg_env("BASE_URL")
    model = get_coevokg_env("MODEL")
    if not base_url or not model:
        raise ValueError("COEVOKG_BASE_URL or COEVOKG_MODEL is not set")

    judge = get_global_judge(
        base_url=base_url, api_key=get_coevokg_env("API_KEY", "dummy_api_key"), model=model
    )

    # Normalise before any access — ground_truth may be a dict, a plain string,
    # or None depending on data source.  All downstream comparisons use `target`.
    target = _extract_target(ground_truth)

    if data_source is not None and data_source in [
        "Algebra",
        "Intermediate Algebra",
        "Prealgebra",
        "Number Theory",
        "Geometry",
        "Precalculus",
        "Counting & Probability",
    ]:
        answer = extract_math_solution(solution_str=solution_str)
    else:
        answer = extract_solution(solution_str=solution_str)
    open_count, close_count = count_answer_tags(solution_str)
    do_print = random.randint(1, 64) == 1

    if do_print:
        print("--------------------------------")
        print(f"Golden answers: {target}")
        if answer is not None:
            print(f"Extracted answer is not None: {answer}")
        else:
            print("Extracted answer: None!")
        print(f"Solution string: {solution_str}")

    beta_proc = float((extra_info or {}).get("beta_proc", 0.3))
    use_chain_extraction = bool((extra_info or {}).get("chain_data_path", ""))

    if answer is None:
        print(f"[CoEvoKGScore]: No answer extracted, data_source: {data_source}")
        modifier = _search_behavior_modifier(solution_str, data_source, False)
        return 0.0 + modifier

    # Question for the unified judge+extract call.
    question = ""
    if prompt_str:
        if "Question: " in prompt_str:
            question = prompt_str.split("Question: ", 1)[1]
        elif "<｜User｜>" in prompt_str:
            question = prompt_str.replace("<｜User｜>", "")
        else:
            question = prompt_str

    # Process reward / write-back only meaningful with a chain DB and off recycled problems.
    proc_ok = use_chain_extraction and beta_proc > 0 and data_source != "feedback_pool"

    # ── Unified LLM-first: ONE call judges correctness AND extracts chain+relations. ──
    # The SAME chain feeds the path-support process reward here AND (via kg_sink
    # passback) the KG write-back — no second extraction call anywhere.
    start_time = time.time()
    # Strip <information> bodies (thousands of tokens → truncated JSON); judge needs
    # only the model's own reasoning. Bound length for the provider context window.
    solution_for_judge = re.sub(r'<information>.*?</information>', '', solution_str, flags=re.DOTALL).strip()
    _MAX_JUDGE_CHARS = int(get_coevokg_env("JUDGE_MAX_CHARS", "2500"))
    if len(solution_for_judge) > _MAX_JUDGE_CHARS:
        solution_for_judge = solution_for_judge[:_MAX_JUDGE_CHARS]
    titles = _solver_title_chain(solution_str, "")  # retrieved titles only (no target)
    correct, llm_chain, llm_rels = judge.judge_and_extract(
        question=question,
        golden_answer=target,
        final_answer=answer,
        reasoning=solution_for_judge,
        titles=titles,
    )
    used_llm = correct is not None
    elapsed = time.time() - start_time

    # ── Fallback: LLM failed/timed-out/parse-error → EM (+SoftEM) decides correctness. ──
    # judge_and_extract never raises for transient errors (returns None), so a slow or
    # failed judge falls back here instead of blocking or crashing reward computation.
    if correct is None:
        correct = em_check(answer, target) or soft_em_check(answer, target)

    if correct:
        # Process-score chain: the LLM canonical chain when usable, else the
        # deterministic retrieved-title chain (EM-fallback path / empty LLM chain).
        usable_llm_chain = used_llm and len(llm_chain) >= 2
        proc_chain = llm_chain if usable_llm_chain else _solver_title_chain(solution_str, target)
        process_score = _compute_process_score(proc_chain, extra_info) if proc_ok else 0.0
        cap = 0.25 if (open_count > 10 or close_count > 10) else 1.0
        score = cap * (1.0 + beta_proc * process_score)
        modifier = _search_behavior_modifier(solution_str, data_source, True)
        score += modifier
        # Stash the chain payload for KG write-back (reused there, not re-extracted).
        if kg_sink is not None and proc_ok:
            kg_sink["correct"] = True
            kg_sink["chain"] = proc_chain
            kg_sink["relations"] = llm_rels if usable_llm_chain else []
            kg_sink["path_support"] = round(float(process_score), 4)
        tag = "LLM-judge" if used_llm else "EM-fallback"
        print(
            f"[CoEvoKGScore]: {tag} correct, answer={answer}, gt={target}, "
            f"time={elapsed:.1f}s, process_score={process_score:.3f}, "
            f"chain_len={len(proc_chain)}, search_modifier={modifier:+.1f}, final={score:.3f}"
        )
        return score
    else:
        # Wrong answer -> no process reward under the unified scoring rule.
        if kg_sink is not None:
            kg_sink["correct"] = False
        modifier = _search_behavior_modifier(solution_str, data_source, False)
        tag = "LLM-judge" if used_llm else "EM-fallback"
        print(
            f"[CoEvoKGScore]: {tag} wrong, answer={answer}, gt={target}, "
            f"time={elapsed:.1f}s, search_modifier={modifier:+.1f}"
        )
        return 0.0 + modifier


def compute_score_batch(batch_data, score_coef: float = 1.0):
    results = [None] * len(batch_data)
    logger = __import__("logging").getLogger(__name__)

    def worker(batch_idx, data):
        try:
            kg_sink: dict = {}
            score = compute_score(
                data_source=data.get("data_source", ""),
                solution_str=data["response_str"],
                ground_truth=data["ground_truth"],
                prompt_str=data.get("prompt_str", None),
                extra_info=data.get("extra_info", {}),
                sandbox_fusion_url=data.get("sandbox_fusion_url", None),
                concurrent_semaphore=data.get("concurrent_semaphore", None),
                kg_sink=kg_sink,
            )
            result = {"score": score * score_coef, "idx": data["idx"]}
            # Pass the reward-stage chain payload back to the driver for KG write-back.
            # Serialised to a JSON string so validation-metric aggregation (which means
            # numeric vars and skips str vars) never chokes on list-valued fields.
            if kg_sink.get("chain"):
                result["kg"] = json.dumps(kg_sink, ensure_ascii=False)
            results[batch_idx] = result
        except FatalProviderPoolExhausted:
            raise
        except Exception as e:
            logger.error(
                f"[CoEvoKGScore] compute_score failed at batch_idx={batch_idx} "
                f"(data_source={data.get('data_source','?')}, "
                f"ground_truth={data.get('ground_truth')}): {e}",
                exc_info=True,
            )
            results[batch_idx] = {"score": 0.0, "idx": data["idx"]}

    start_time = time.time()
    futures = []
    with ThreadPoolExecutor(max_workers=int(get_coevokg_env("JUDGE_MAX_WORKERS", "32"))) as executor:
        for batch_idx, data in enumerate(batch_data):
            futures.append(executor.submit(worker, batch_idx, data))
        for fut in futures:
            fut.result()
    print(f"[CoEvoKGScore]: compute_score_batch time: {time.time() - start_time}s")
    return results
