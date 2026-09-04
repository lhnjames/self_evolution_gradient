"""
Offline evaluation for generated agent answers.

The script supports single-rollout JSONL files and pass@N evaluation over
multiple rollout files named iter1.jsonl, iter2.jsonl, ... . It can use exact
matching only, or an OpenAI-compatible judge endpoint configured by environment
variables.

Examples:
    # Single rollout.
    python evaluation/judge_eval.py --input outputs_eval/iter1.jsonl

    # pass@N: load iter1..iterN.jsonl from a directory.
    python evaluation/judge_eval.py --input-dir outputs_eval --rollout-count 3

    # pass@N: auto-detect all iter*.jsonl files.
    python evaluation/judge_eval.py --input-dir outputs_eval

    # Exact-match only, without calling an LLM judge.
    python evaluation/judge_eval.py --input-dir outputs_eval --exact-match-only

    # Resume an interrupted judge run.
    python evaluation/judge_eval.py --input outputs_eval/iter1.jsonl --resume

Required environment variables for LLM judge mode:
    COEVOKG_EVAL_BASE_URL      OpenAI-compatible endpoint URL
    COEVOKG_EVAL_JUDGE_MODEL   Judge model name
    COEVOKG_EVAL_API_KEY       API key, if required by the endpoint
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from math import comb
from pathlib import Path
from typing import Any, Optional


# ─── Judge prompt ─────────────────────────────────────────────────────────────

JUDGE_PROMPT = """You are an impartial judge evaluating whether an AI assistant's answer is correct.

[Question]
{question}

[Correct Answer]
{reference_answer}

[Assistant's Answer]
{assistant_answer}

Task: Determine if the assistant's answer is correct by comparing it to the correct answer.

Instructions:
1. The correct answer may be a short phrase, entity name, date, or number.
2. The assistant's answer may be long; extract its core claim or final answer.
3. Judge based on semantic equivalence, not exact string match.
4. For list-type answers (e.g. golden_answers is a list), the assistant is correct if it matches any one item.

Output format:
correct: [yes/no]
reasoning: [brief explanation]"""


# ─── LLM Judge ────────────────────────────────────────────────────────────────

class LLMJudge:
    def __init__(self, client, model: str):
        self.client = client
        self.model = model

    def judge(self, question: str, reference_answer: str, assistant_answer: str,
              max_retries: int = 3) -> dict[str, Any]:
        prompt = JUDGE_PROMPT.format(
            question=question,
            reference_answer=reference_answer,
            assistant_answer=assistant_answer,
        )
        last_err = None
        for attempt in range(max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=512,
                )
                judgment_text = (resp.choices[0].message.content or "").strip()
                break
            except Exception as e:
                last_err = e
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        else:
            return {"is_correct": False,
                    "judgment": f"Judge error: {last_err}",
                    "reasoning": str(last_err)}

        is_correct = False
        reasoning = ""
        for line in judgment_text.lower().split("\n"):
            stripped = line.strip()
            if stripped.startswith("correct:"):
                is_correct = "yes" in stripped
            elif stripped.startswith("reasoning:"):
                reasoning = stripped[len("reasoning:"):].strip()

        return {"is_correct": is_correct, "judgment": judgment_text, "reasoning": reasoning}


# Exact matching

def exact_match(prediction: str, reference: Any) -> bool:
    p = prediction.strip().lower()
    refs = reference if isinstance(reference, list) else [reference]
    for r in refs:
        r = str(r).strip().lower()
        if p == r or r in p or p in r:
            return True
    return False


def normalize_reference(answer: Any) -> str:
    if isinstance(answer, list):
        return " / ".join(str(a) for a in answer)
    return str(answer)


# Data loading

def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_env(path: str = ".env"):
    p = Path(path)
    if not p.exists():
        return
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def detect_iter_files(input_dir: str, rollout_count: Optional[int] = None) -> list[tuple[int, str]]:
    """Return sorted list of (rollout_idx, path) for iter*.jsonl files in input_dir."""
    dir_path = Path(input_dir)
    found: list[tuple[int, str]] = []
    for f in sorted(dir_path.glob("iter*.jsonl")):
        m = re.match(r"iter(\d+)", f.stem)
        if m:
            found.append((int(m.group(1)), str(f)))
    found.sort(key=lambda x: x[0])
    if rollout_count is not None:
        found = [(i, p) for i, p in found if i <= rollout_count]
    return found


def backfill_source(records: list[dict], eval_data_path: Path) -> None:
    """Fill missing source/id fields from eval_data.jsonl."""
    missing = [r for r in records if not r.get("source")]
    if not missing or not eval_data_path.exists():
        if missing:
            print(f"Warning: eval_data.jsonl not found at {eval_data_path}, source will be 'unknown'.")
        return
    q2meta = {
        item.get("question", ""): {"source": item.get("source", "unknown"), "id": item.get("id", "")}
        for item in load_jsonl(str(eval_data_path))
    }
    filled = 0
    for r in missing:
        meta = q2meta.get(r.get("question", ""))
        if meta:
            r["source"] = meta["source"]
            r.setdefault("id", meta["id"])
            filled += 1
    print(f"Back-filled source for {filled}/{len(missing)} records missing it.")


# Metric computation for one rollout

def calculate_metrics(results: list[dict]) -> dict[str, Any]:
    total = len(results)
    if total == 0:
        return {}

    judge_correct = sum(1 for r in results if r.get("judge_is_correct", False))
    em_correct    = sum(1 for r in results if r.get("em_correct", False))

    rounds_list = [r["rounds"] for r in results if r.get("rounds") is not None]
    time_list   = [r["time_taken"] for r in results if r.get("time_taken") is not None]

    term_dist: dict[str, int] = defaultdict(int)
    for r in results:
        term_dist[r.get("termination", "unknown")] += 1

    by_source: dict[str, dict] = defaultdict(lambda: {"total": 0, "correct": 0, "em": 0})
    for r in results:
        src = r.get("source", "unknown")
        by_source[src]["total"]   += 1
        by_source[src]["correct"] += int(r.get("judge_is_correct", False))
        by_source[src]["em"]      += int(r.get("em_correct", False))

    source_stats = {
        src: {
            "total":          v["total"],
            "judge_correct":  v["correct"],
            "judge_accuracy": v["correct"] / v["total"] if v["total"] else 0,
            "em_correct":     v["em"],
            "em_accuracy":    v["em"] / v["total"] if v["total"] else 0,
        }
        for src, v in sorted(by_source.items())
    }

    return {
        "total":              total,
        "judge_correct":      judge_correct,
        "judge_accuracy":     judge_correct / total,
        "em_correct":         em_correct,
        "em_accuracy":        em_correct / total,
        "avg_rounds":         statistics.mean(rounds_list) if rounds_list else 0,
        "avg_time_seconds":   statistics.mean(time_list) if time_list else 0,
        "termination_distribution": dict(term_dist),
        "by_source":          source_stats,
    }


def print_summary(metrics: dict[str, Any], judge_model: str):
    print("\n" + "=" * 68)
    print("  EVALUATION SUMMARY")
    print("=" * 68)
    print(f"  Judge model    : {judge_model}")
    print(f"  Total          : {metrics['total']}")
    print(f"  Judge accuracy : {metrics['judge_accuracy']:.2%}  "
          f"({metrics['judge_correct']}/{metrics['total']})")
    print(f"  Exact match    : {metrics['em_accuracy']:.2%}  "
          f"({metrics['em_correct']}/{metrics['total']})")
    print(f"  Avg rounds     : {metrics['avg_rounds']:.1f}")
    print(f"  Avg time (s)   : {metrics['avg_time_seconds']:.1f}")

    print("\n  Accuracy by source dataset:")
    print(f"  {'Source':<20} {'Total':>6} {'Judge%':>8} {'EM%':>8}")
    print(f"  {'-'*20} {'-'*6} {'-'*8} {'-'*8}")
    for src, v in metrics.get("by_source", {}).items():
        print(f"  {src:<20} {v['total']:>6} "
              f"{v['judge_accuracy']:>8.1%} {v['em_accuracy']:>8.1%}")

    print("\n  Termination distribution:")
    for reason, cnt in metrics.get("termination_distribution", {}).items():
        print(f"    {reason}: {cnt}")
    print("=" * 68)


# pass@k computation

def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator of pass@k (Chen et al., 2021).

    pass@k = 1 - C(n-c, k) / C(n, k)

    Args:
        n: total number of rollouts for this problem
        c: number of correct rollouts
        k: k in pass@k
    """
    if k > n:
        return float("nan")
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def calculate_pass_at_n_metrics(
    questions: list[str],
    correct_counts: dict[str, int],   # question → # correct rollouts
    n_rollouts: int,
    rollout_records: dict[int, list[dict]],  # rollout_idx → judged records
) -> dict[str, Any]:
    """Compute pass@k metrics for k=1..n_rollouts across all questions."""
    # per-question pass@k
    pass_at_k_per_q: dict[int, list[float]] = {k: [] for k in range(1, n_rollouts + 1)}
    for q in questions:
        c = correct_counts.get(q, 0)
        for k in range(1, n_rollouts + 1):
            pass_at_k_per_q[k].append(pass_at_k(n_rollouts, c, k))

    overall_pass_at_k = {
        k: statistics.mean(vals) for k, vals in pass_at_k_per_q.items() if vals
    }

    # per-rollout accuracy
    per_rollout_acc: dict[int, float] = {}
    for ri, records in sorted(rollout_records.items()):
        total = len(records)
        correct = sum(1 for r in records if r.get("judge_is_correct", False))
        per_rollout_acc[ri] = correct / total if total else 0.0

    # per-source pass@k
    by_source: dict[str, dict] = defaultdict(lambda: {"questions": [], "correct_counts": {}})
    for ri, records in rollout_records.items():
        for r in records:
            q   = r.get("question", "")
            src = r.get("source", "unknown")
            by_source[src]["questions"].append(q)
    # deduplicate questions per source
    source_questions: dict[str, list[str]] = {}
    for src, d in by_source.items():
        source_questions[src] = list(dict.fromkeys(d["questions"]))

    # correct count per question per source (use global correct_counts)
    source_pass_at_k: dict[str, dict[int, float]] = {}
    for src, qs in source_questions.items():
        src_pass: dict[int, list[float]] = {k: [] for k in range(1, n_rollouts + 1)}
        for q in qs:
            c = correct_counts.get(q, 0)
            for k in range(1, n_rollouts + 1):
                src_pass[k].append(pass_at_k(n_rollouts, c, k))
        source_pass_at_k[src] = {k: statistics.mean(vals) for k, vals in src_pass.items() if vals}

    return {
        "n_rollouts":        n_rollouts,
        "total_questions":   len(questions),
        "overall_pass_at_k": overall_pass_at_k,
        "per_rollout_acc":   per_rollout_acc,
        "source_pass_at_k":  {src: v for src, v in sorted(source_pass_at_k.items())},
    }


def print_pass_at_n_summary(metrics: dict[str, Any], judge_model: str):
    n = metrics["n_rollouts"]
    total_q = metrics["total_questions"]
    overall = metrics["overall_pass_at_k"]
    per_ro  = metrics["per_rollout_acc"]
    by_src  = metrics["source_pass_at_k"]

    ks = list(range(1, n + 1))

    print("\n" + "=" * 68)
    print(f"  PASS@N SUMMARY  (N={n})")
    print("=" * 68)
    print(f"  Judge model      : {judge_model}")
    print(f"  Total questions  : {total_q}")
    print(f"  Total rollouts   : {n}")

    print("\n  Overall pass@k (unbiased estimator):")
    for k in ks:
        v = overall.get(k, float("nan"))
        print(f"    pass@{k:<2d}: {v:.2%}")

    print("\n  Per-rollout accuracy:")
    for ri in sorted(per_ro):
        print(f"    Rollout {ri}: {per_ro[ri]:.2%}")

    if by_src:
        header = f"  {'Source':<20} {'N':>5}"
        for k in ks:
            header += f"  {'pass@'+str(k):>8}"
        print(f"\n  pass@k by source dataset:")
        print(header)
        print(f"  {'-'*20} {'-'*5}" + "  " + "  ".join(["-" * 8] * len(ks)))
        for src, src_pk in by_src.items():
            n_src = len([q for q in metrics.get("_source_questions", {}).get(src, [])])
            row = f"  {src:<20} {n_src:>5}"
            for k in ks:
                row += f"  {src_pk.get(k, float('nan')):>8.1%}"
            print(row)

    print("=" * 68)


# Save results

def save_results(judged: list[dict], metrics: dict, output_dir: str, suffix: str = "", model_name: str = ""):
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    # sanitize model name for use in filename
    model_slug = re.sub(r"[^\w\-]", "_", model_name) if model_name else ""
    name_part = f"_{model_slug}" if model_slug else ""

    results_file = Path(output_dir) / f"judge_results{suffix}{name_part}_{ts}.jsonl"
    with open(results_file, "w", encoding="utf-8") as f:
        for r in judged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    metrics_file = Path(output_dir) / f"judge_metrics{suffix}{name_part}_{ts}.json"
    # Remove internal helper key before saving
    metrics_out = {k: v for k, v in metrics.items() if not k.startswith("_")}
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, ensure_ascii=False, indent=2)

    print(f"\n  Results saved : {results_file}")
    print(f"  Metrics saved : {metrics_file}")


# Reusable single-file judge pipeline

def judge_records(
    records: list[dict],
    judge: Optional[LLMJudge],
    exact_match_only: bool,
    workers: int,
    max_retries: int,
    resume: bool = False,
    verbose: bool = False,
    progress_prefix: str = "",
) -> list[dict]:
    """Judge a list of records; returns list with judge_is_correct filled."""
    if resume:
        pending = [r for r in records if "judge_is_correct" not in r]
        done    = [r for r in records if "judge_is_correct" in r]
    else:
        pending = records
        done = []

    total = len(pending)
    judged_pending: list[Optional[dict]] = [None] * total
    completed = 0
    correct   = sum(1 for r in done if r.get("judge_is_correct", False))
    total_so_far = len(done)

    def judge_one(idx: int, record: dict) -> tuple[int, dict]:
        question   = record.get("question", "")
        answer_raw = record.get("answer", "")
        prediction = record.get("prediction", "")

        em = exact_match(prediction, answer_raw)
        reference_str = normalize_reference(answer_raw)

        result = dict(record)
        result["em_correct"] = em

        if exact_match_only or judge is None:
            result["judge_is_correct"] = em
            result["judge_judgment"]   = "exact_match"
            result["judge_reasoning"]  = ""
        elif not reference_str:
            result["judge_is_correct"] = False
            result["judge_judgment"]   = "no_reference"
            result["judge_reasoning"]  = ""
        else:
            j = judge.judge(question, reference_str, prediction, max_retries=max_retries)
            result["judge_is_correct"] = j["is_correct"]
            result["judge_judgment"]   = j["judgment"]
            result["judge_reasoning"]  = j["reasoning"]

        return idx, result

    _workers = 1 if exact_match_only else workers
    prefix = f"[{progress_prefix}] " if progress_prefix else ""

    with ThreadPoolExecutor(max_workers=_workers) as executor:
        futures = {executor.submit(judge_one, i, r): i for i, r in enumerate(pending)}
        for future in as_completed(futures):
            idx, result = future.result()
            judged_pending[idx] = result
            completed    += 1
            total_so_far += 1
            correct      += int(result.get("judge_is_correct", False))

            acc    = correct / total_so_far
            status = "V" if result.get("judge_is_correct") else "X"
            src    = result.get("source", "?")
            print(f"{prefix}[{completed:4d}/{total}] {status}  acc={acc:.1%}  [{src}]  "
                  f"{result.get('question','')[:50]}")

            if verbose:
                print(f"  Pred      : {result.get('prediction','')[:100]}")
                print(f"  Reference : {normalize_reference(result.get('answer',''))[:80]}")
                print(f"  Reasoning : {result.get('judge_reasoning','')[:100]}")

    return done + judged_pending


# Main pipeline

def main():
    load_env()

    parser = argparse.ArgumentParser(description="Judge agent eval results, supports pass@N")
    # Input: single file OR directory
    parser.add_argument("--input", "-i",
                        default=None,
                        help="Single agent output JSONL (single-rollout mode)")
    parser.add_argument("--input-dir", "-d",
                        default=os.getenv("OUTPUT_PATH"),
                        help="Directory containing iter1.jsonl…iterN.jsonl (pass@N mode)")
    parser.add_argument("--rollout-count", type=int,
                        default=int(os.getenv("COEVOKG_EVAL_ROLLOUT_COUNT", "0")),
                        help="Number of rollouts (used with --input-dir; 0=auto-detect)")
    parser.add_argument("--output-dir", "-o",
                        default=os.getenv("COEVOKG_EVAL_OUTPUT_DIR", "judge_outputs"),
                        help="Output directory (default: judge_outputs)")
    parser.add_argument("--judge-model",
                        default=os.getenv("COEVOKG_EVAL_JUDGE_MODEL"),
                        help="Judge model name; can also be set via COEVOKG_EVAL_JUDGE_MODEL")
    parser.add_argument("--judge-base-url",
                        default=os.getenv("COEVOKG_EVAL_BASE_URL"),
                        help="OpenAI-compatible judge API base URL; can also be set via COEVOKG_EVAL_BASE_URL")
    parser.add_argument("--judge-api-key",
                        default=os.getenv("COEVOKG_EVAL_API_KEY"),
                        help="Judge API key; can also be set via COEVOKG_EVAL_API_KEY")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--workers", type=int, default=50,
                        help="Concurrent judge threads (default: 50)")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--exact-match-only", action="store_true",
                        help="Skip LLM judge, only exact match")
    parser.add_argument("--resume", action="store_true",
                        help="Skip records that already have judge_is_correct")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    # Source model name for trajectory outputs.
    agent_model_name = os.getenv("MODEL_NAME") or os.getenv("MODEL") or ""

    # Select execution mode.
    pass_at_n_mode = bool(args.input_dir) or args.rollout_count > 1

    # Initialize the judge.
    judge: Optional[LLMJudge] = None
    judge_model_name = "exact_match"

    if not args.exact_match_only:
        api_key = (
            args.judge_api_key
            or os.getenv("COEVOKG_EVAL_API_KEY")
            or os.getenv("COEVOKG_API_KEY")
            or "EMPTY"
        )
        base_url = (
            args.judge_base_url
            or os.getenv("COEVOKG_EVAL_BASE_URL")
            or os.getenv("COEVOKG_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
        )
        model = (
            args.judge_model
            or os.getenv("COEVOKG_EVAL_JUDGE_MODEL")
            or os.getenv("COEVOKG_MODEL")
        )

        if not base_url or not model:
            print("Warning: judge endpoint or model is not set. Falling back to exact-match.")
            args.exact_match_only = True
        else:
            from openai import OpenAI
            client = OpenAI(api_key=api_key, base_url=base_url)
            judge = LLMJudge(client=client, model=model)
            judge_model_name = model
            print(f"Judge model : {model}")
            print(f"Judge URL   : {base_url}")
            print(f"Workers     : {args.workers}")

    eval_data_path = Path(os.getenv("COEVOKG_EVAL_DATA", os.getenv("EVAL_DATA_PATH", str(Path.cwd() / "eval_data.jsonl"))))

    # ══════════════════════════════════════════════════════════════════════════
    # pass@N mode.
    # ══════════════════════════════════════════════════════════════════════════
    if pass_at_n_mode:
        input_dir = args.input_dir
        if not input_dir:
            if not args.input:
                print("Error: pass@N mode requires --input-dir or --input (to infer directory).")
                print("  Example: python evaluation/judge_eval.py --input-dir outputs_eval")
                return
            # Infer the rollout directory from --input.
            input_dir = str(Path(args.input).parent)

        rc = args.rollout_count if args.rollout_count > 0 else None
        iter_files = detect_iter_files(input_dir, rc)
        if not iter_files:
            print(f"Error: no iter*.jsonl files found in {input_dir}")
            return

        n_rollouts = iter_files[-1][0]  # highest index found
        print(f"pass@N mode: found {len(iter_files)} rollout file(s), N={n_rollouts}")
        for ri, fp in iter_files:
            print(f"  Rollout {ri}: {fp}")

        # Load and judge each rollout.
        rollout_records: dict[int, list[dict]] = {}
        all_judged_flat: list[dict] = []

        for ri, fp in iter_files:
            print(f"\n── Rollout {ri} ─────────────────────────────────────────")
            records = load_jsonl(fp)
            backfill_source(records, eval_data_path)
            if args.max_samples:
                records = records[:args.max_samples]
            judged = judge_records(
                records,
                judge=judge,
                exact_match_only=args.exact_match_only,
                workers=args.workers,
                max_retries=args.retries,
                resume=args.resume,
                verbose=args.verbose,
                progress_prefix=f"iter{ri}",
            )
            # tag each record with its rollout index
            for r in judged:
                r["rollout_idx"] = ri
            rollout_records[ri] = judged
            all_judged_flat.extend(judged)

        # Aggregate correct_count by question.
        # unique questions (preserve insertion order from rollout 1)
        seen: dict[str, None] = {}
        for r in rollout_records.get(iter_files[0][0], []):
            seen[r.get("question", "")] = None
        questions = list(seen.keys())

        correct_counts: dict[str, int] = defaultdict(int)
        for records in rollout_records.values():
            for r in records:
                if r.get("judge_is_correct", False):
                    correct_counts[r.get("question", "")] += 1

        # Compute pass@k metrics.
        pn_metrics = calculate_pass_at_n_metrics(
            questions=questions,
            correct_counts=correct_counts,
            n_rollouts=n_rollouts,
            rollout_records=rollout_records,
        )
        # attach source→questions mapping for display
        src_qs: dict[str, list[str]] = defaultdict(list)
        for q in questions:
            for r in rollout_records.get(iter_files[0][0], []):
                if r.get("question") == q:
                    src_qs[r.get("source", "unknown")].append(q)
                    break
        pn_metrics["_source_questions"] = dict(src_qs)

        print_pass_at_n_summary(pn_metrics, judge_model=judge_model_name)
        save_results(all_judged_flat, pn_metrics, args.output_dir, suffix="_pass_at_n", model_name=agent_model_name)
        return

    # ══════════════════════════════════════════════════════════════════════════
    # Single-file mode.
    # ══════════════════════════════════════════════════════════════════════════
    input_path = args.input or "outputs_eval/iter1.jsonl"
    print(f"Loading: {input_path}")
    records = load_jsonl(input_path)
    print(f"Loaded {len(records)} records.")

    backfill_source(records, eval_data_path)

    if args.max_samples:
        records = records[:args.max_samples]
        print(f"Limited to {len(records)} records.")

    print(f"\nEvaluating {len(records)} records...")
    all_results = judge_records(
        records,
        judge=judge,
        exact_match_only=args.exact_match_only,
        workers=args.workers,
        max_retries=args.retries,
        resume=args.resume,
        verbose=args.verbose,
    )

    metrics = calculate_metrics(all_results)
    if not metrics:
        print("No results to evaluate.")
        return
    print_summary(metrics, judge_model=judge_model_name)
    save_results(all_results, metrics, args.output_dir, model_name=agent_model_name)


if __name__ == "__main__":
    main()
