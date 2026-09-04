"""
Question Quality Checker.

Core design change: one API call per chain (not per candidate).
All M format-valid candidates of a chain share the same chain context
and ground truth, so they are evaluated together in one batch request.
The API returns a JSON array; Python recomputes the final score.

API call budget:
  Before: Σ |format_valid_i|  calls  (~300 per step with M=4)
  After:  |chains_with_valid| calls  (~90 per step)  → ~3-4× reduction

no_leak is enforced programmatically (Python), not trusted from LLM arithmetic.
Python pre-check for exact substring leakage skips the API entirely.

Scoring formula (Python-side):
    no_leak  = 1 if model no_leak >= 0.5 else 0
    overall  = no_leak × (0.30×chain_faithful + 0.30×multi_hop
                         + 0.20×single_focus  + 0.20×clarity)
"""

import json
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

from openai import OpenAI

from coevokg.prompts.quality_check_prompts import (
    QUALITY_CHECK_SYSTEM_PROMPT,
    QUALITY_CHECK_USER_PROMPT,
    RAG_VERIFY_SYSTEM,
    RAG_VERIFY_USER,
    RULE_GENERATION_SYSTEM_PROMPT,
    RULE_GENERATION_USER_PROMPT,
)
from coevokg.utils.api_provider_pool import (
    BadRequestNoRetry,
    FatalProviderPoolExhausted,
    get_api_provider_pool,
)

logger = logging.getLogger(__name__)


# Concurrency across chains (not candidates — one call per chain)
_MAX_CHAIN_WORKERS = 40    # Match the total slots in COEVOKG_MODEL_SLOT_TOTALS.
_DEFAULT_TIMEOUT   = 60.0   # single-candidate baseline
_MAX_TIMEOUT       = 120.0  # Per-call timeout cap.
_SECS_PER_CANDIDATE = 30.0  # added per non-leaked candidate in the batch

_W_CHAIN_FAITHFUL = 0.30
_W_MULTI_HOP      = 0.30
_W_SINGLE_FOCUS   = 0.20
_W_CLARITY        = 0.20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _exact_leak(question: str, ground_truth: str) -> bool:
    """True if ground_truth appears verbatim (case-insensitive) in question."""
    if not question or not ground_truth:
        return False
    return ground_truth.lower().strip() in question.lower()


def _normalize_str(s: str) -> str:
    """Lowercase + strip punctuation/articles for loose comparison."""
    import string
    s = s.lower().strip()
    for ch in string.punctuation:
        s = s.replace(ch, " ")
    stop = {"a", "an", "the"}
    return " ".join(w for w in s.split() if w not in stop)


def _compute_overall(raw: dict) -> float:
    """Recompute overall from raw LLM scores. no_leak is a hard binary gate."""
    no_leak = 1 if float(raw.get("no_leak", 1.0)) >= 0.5 else 0
    cf  = max(0.0, min(1.0, float(raw.get("chain_faithful", 0.5))))
    mh  = max(0.0, min(1.0, float(raw.get("multi_hop",      0.5))))
    sf  = max(0.0, min(1.0, float(raw.get("single_focus",   0.5))))
    cl  = max(0.0, min(1.0, float(raw.get("clarity",        0.5))))
    return float(no_leak) * (
        _W_CHAIN_FAITHFUL * cf + _W_MULTI_HOP * mh
        + _W_SINGLE_FOCUS * sf + _W_CLARITY   * cl
    )


def _format_chain_with_intros(
    chain: Optional[List[str]],
    node_snippets: Optional[Dict[str, str]] = None,
    max_chars: int = 150,
) -> str:
    """Format chain as numbered hops with optional one-line descriptions."""
    if not chain:
        return "  (chain not provided)"
    lines = []
    for i, entity in enumerate(chain):
        line = f"  hop {i}: {entity}"
        if node_snippets and entity in node_snippets:
            snippet = node_snippets[entity].strip().replace("\n", " ")[:max_chars]
            line += f" — {snippet}"
        lines.append(line)
    return "\n".join(lines)


def _parse_array_response(content: str, expected_n: int) -> List[dict]:
    """Parse a JSON array from LLM response. Returns list of raw dicts.

    Falls back gracefully:
    - strips markdown fences
    - tries regex extraction if direct parse fails
    - pads with empty dicts if fewer items than expected
    """
    content = content.strip()
    if "```" in content:
        parts = content.split("```")
        content = parts[1] if len(parts) >= 3 else parts[-1]
        if content.lstrip().startswith("json"):
            content = content.lstrip()[4:]
        content = content.strip()

    # Try direct parse
    try:
        data = json.loads(content)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]  # single object wrapped incorrectly
    except json.JSONDecodeError:
        pass

    # Fallback: extract all {...} objects
    objects = re.findall(r"\{[^{}]+\}", content, re.DOTALL)
    if objects:
        parsed = []
        for obj in objects:
            try:
                parsed.append(json.loads(obj))
            except json.JSONDecodeError:
                continue
        if parsed:
            return parsed

    return []


class QuestionQualityChecker:
    """Evaluates question quality via LLM API calls (one call per chain group)."""

    _MAX_KEY_SWITCHES = 10

    def __init__(self, base_url: str, api_key: str, model: str,
                 timeout: float = _MAX_TIMEOUT,
                 extra_api_keys: Optional[List[str]] = None):
        self._clients = []
        self._current_idx = 0
        self._lock = threading.Lock()
        self.model = model
        self._pool = get_api_provider_pool("judge", base_url=base_url, timeout=timeout)

    def _call_with_fallback(self, fn: Callable[[OpenAI], Any]) -> Any:
        """Cycle through keys on failure; raise after > _MAX_KEY_SWITCHES switches."""
        return self._pool.call(lambda client, model: fn(client, model))
        n = len(self._clients)
        with self._lock:
            start = self._current_idx

        last_exc: Optional[Exception] = None
        switches = 0
        idx = start

        while switches <= self._MAX_KEY_SWITCHES:
            try:
                result = fn(self._clients[idx])
                with self._lock:
                    self._current_idx = idx
                return result
            except Exception as e:
                last_exc = e
                switches += 1
                next_idx = (idx + 1) % n
                logger.warning(
                    f"[quality_checker] key[{idx}] failed ({type(e).__name__}: {e}), "
                    f"switch #{switches}/{self._MAX_KEY_SWITCHES} → key[{next_idx}]"
                )
                idx = next_idx

        raise RuntimeError(
            f"All API keys exhausted after {switches} switches"
        ) from last_exc

    # ------------------------------------------------------------------
    # Core: one API call for all candidates of a single chain
    # ------------------------------------------------------------------

    def check_chain_candidates(
        self,
        questions: List[str],
        ground_truth: str,
        chain: Optional[List[str]] = None,
        node_snippets: Optional[Dict[str, str]] = None,
    ) -> Optional[List[float]]:
        """Evaluate all M candidates for one chain in a single API call.
        Returns None when the API is unavailable (timeout/error), so callers
        can distinguish "unknown quality" from "genuinely low score (0.0)".

        Args:
            questions:     All format-valid question candidates for this chain.
            ground_truth:  Shared correct answer entity (last chain node).
            chain:         Full knowledge chain shared by all candidates.
            node_snippets: {entity: description} for context.

        Returns list of overall scores, same length and order as questions.
        """
        if not questions:
            return []

        # Python fast-path: mark exact-leak candidates before API call
        leak_mask = [_exact_leak(q, ground_truth) for q in questions]
        non_leak_indices = [i for i, leaked in enumerate(leak_mask) if not leaked]

        # If all candidates leaked, skip API entirely
        if not non_leak_indices:
            logger.debug(f"All {len(questions)} candidates leaked (Python fast-path)")
            return [0.0] * len(questions)

        # Build prompt with only non-leaked questions
        questions_to_check = [questions[i] for i in non_leak_indices]
        n = len(questions_to_check)

        numbered = "\n\n".join(
            f"Question {i+1}:\n{q}" for i, q in enumerate(questions_to_check)
        )
        chain_with_intros = _format_chain_with_intros(chain, node_snippets)
        prompt = QUALITY_CHECK_USER_PROMPT.format(
            chain_with_intros=chain_with_intros,
            ground_truth=ground_truth,
            n_candidates=n,
            numbered_questions=numbered,
        )

        # Token budget and timeout scale with batch size.
        max_tokens = max(150, n * 90)
        timeout = min(_MAX_TIMEOUT,
                      max(_DEFAULT_TIMEOUT, n * _SECS_PER_CANDIDATE))

        try:
            response = self._call_with_fallback(lambda c, m: c.chat.completions.create(
                model=m,
                messages=[
                    {"role": "system", "content": QUALITY_CHECK_SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.1,
                max_tokens=max_tokens,
                timeout=timeout,
            ))
            raw_content = response.choices[0].message.content.strip()
            raw_list = _parse_array_response(raw_content, n)

            # Compute scores for non-leaked candidates
            api_scores: List[float] = []
            for j in range(n):
                raw = raw_list[j] if j < len(raw_list) else {}
                api_scores.append(_compute_overall(raw) if raw else 0.0)

            logger.debug(
                f"Chain quality check: {n} candidates → "
                f"scores={[round(s,3) for s in api_scores]}"
            )

        except BadRequestNoRetry as e:
            logger.warning(f"Chain quality check bad request (not retried): {e}")
            api_scores = None
        except FatalProviderPoolExhausted:
            raise
        except Exception as e:
            logger.warning(f"Chain quality check API failed: {e}")
            api_scores = None  # None means API unavailable, distinct from a true low-quality score (0.0).

        if api_scores is None:
            return None  # The caller skips filtering on None and uses the question directly.

        # Reassemble full-length result (leaked → 0.0, others → API score)
        scores = [0.0] * len(questions)
        for rank, orig_i in enumerate(non_leak_indices):
            scores[orig_i] = api_scores[rank] if rank < len(api_scores) else 0.0
        return scores

    # ------------------------------------------------------------------
    # Legacy single-question interface (kept for external callers)
    # ------------------------------------------------------------------

    def check_quality(
        self,
        question: str,
        ground_truth: str,
        chain: Optional[List[str]] = None,
        node_snippets: Optional[Dict[str, str]] = None,
    ) -> float:
        """Evaluate a single question. Delegates to check_chain_candidates."""
        results = self.check_chain_candidates(
            [question], ground_truth, chain=chain, node_snippets=node_snippets
        )
        return results[0] if results else 0.0

    # ------------------------------------------------------------------
    # Legacy batch interface (kept for external callers)
    # Used when candidates don't share a common chain context.
    # For the normal training path, prefer check_chain_candidates.
    # ------------------------------------------------------------------

    def batch_check(
        self,
        questions: List[str],
        ground_truths: List[str],
        chains: Optional[List[Optional[List[str]]]] = None,
        node_snippets_list: Optional[List[Optional[Dict[str, str]]]] = None,
    ) -> List[float]:
        """Evaluate a flat list where each item may have a different chain.

        Each item gets its own API call. Use check_chain_candidates instead
        when all items share the same chain (normal training path).
        """
        assert len(questions) == len(ground_truths)
        scores = [0.0] * len(questions)
        max_workers = min(_MAX_CHAIN_WORKERS, len(questions))

        def _job(i: int) -> float:
            chain = chains[i] if chains else None
            snippets = node_snippets_list[i] if node_snippets_list else None
            return self.check_quality(questions[i], ground_truths[i],
                                      chain=chain, node_snippets=snippets)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            fut_map = {executor.submit(_job, i): i for i in range(len(questions))}
            for fut in as_completed(fut_map):
                idx = fut_map[fut]
                try:
                    scores[idx] = fut.result()
                except Exception as e:
                    logger.warning(f"batch_check item {idx} failed: {e}")
        return scores

    # ------------------------------------------------------------------
    # RAG-verify: answer question from chain docs, check if correct
    # ------------------------------------------------------------------

    def rag_verify_candidates(
        self,
        questions: List[str],
        ground_truth: str,
        node_docs: Optional[Dict[str, str]] = None,
    ) -> List[float]:
        """Evaluate all M candidates for one chain via RAG: use chain documents
        as context, ask the model to answer, compare with ground_truth.

        Score per candidate:
          1.0 — model answered correctly from docs (chain supports question)
          0.3 — model answered but wrongly (chain exists, question misaligned)
          0.0 — model said "Cannot determine" (docs don't support question)
          0.0 — answer leaked (Python fast-path, no API call)

        Args:
            questions:    List of candidate question texts.
            ground_truth: The correct answer entity.
            node_docs:    {entity: text} from chain. If None, falls back to
                          dimension-based scoring.

        Returns list of float scores, same length and order as questions.
        """
        if not questions:
            return []

        # No docs available — fall back to dimension scoring
        if not node_docs:
            return self.check_chain_candidates(questions, ground_truth,
                                               node_snippets=None)

        # Build context from chain docs (max 400 chars per entity)
        context_parts = []
        for entity, text in node_docs.items():
            if text:
                snippet = text.strip().replace("\n", " ")[:400]
                context_parts.append(f"[{entity}]:\n{snippet}")
        context = "\n\n".join(context_parts) if context_parts else "(no documents)"

        # Token budget: 200 to accommodate reasoning models (kimi-k2.6 uses
        # thinking tokens internally before visible output; 40 was too small).
        # deepseek answers in <10 tokens, so 200 adds negligible cost for it.
        n = len(questions)
        timeout = min(_MAX_TIMEOUT, max(_DEFAULT_TIMEOUT, n * 15.0))

        scores = []
        for q in questions:
            # Python fast-path: exact leak
            if _exact_leak(q, ground_truth):
                scores.append(0.0)
                continue

            prompt = RAG_VERIFY_USER.format(context=context, question=q)
            try:
                resp = self._call_with_fallback(lambda c, m: c.chat.completions.create(
                    model=m,
                    messages=[
                        {"role": "system", "content": RAG_VERIFY_SYSTEM},
                        {"role": "user",   "content": prompt},
                    ],
                    temperature=0.0,
                    max_tokens=500,
                    timeout=timeout,
                ))
                if not resp or not resp.choices:
                    scores.append(0.0)
                    continue
                raw_content = resp.choices[0].message.content
                if not raw_content:
                    scores.append(0.0)
                    continue
                answer = raw_content.strip()

                if answer.lower().startswith("cannot determine") or not answer:
                    scores.append(0.0)
                elif _normalize_str(answer) == _normalize_str(ground_truth):
                    scores.append(1.0)
                else:
                    # Semantic near-match: small positive signal
                    gt_lower = ground_truth.lower()
                    ans_lower = answer.lower()
                    if gt_lower in ans_lower or ans_lower in gt_lower:
                        scores.append(0.8)
                    else:
                        scores.append(0.3)
            except BadRequestNoRetry as e:
                logger.warning(f"rag_verify bad request (not retried): {e}")
                scores.append(0.0)
            except FatalProviderPoolExhausted:
                raise
            except Exception as e:
                logger.warning(f"rag_verify failed: {e}")
                scores.append(0.0)

        logger.debug(f"RAG verify {n} candidates → scores={[round(s,2) for s in scores]}")
        return scores

    # ------------------------------------------------------------------
    # Rule generation (kept for API compatibility; unused in training)
    # ------------------------------------------------------------------

    def generate_rule(self, question: str, ground_truth: str,
                      success_rate: float, quality_score: float) -> str:
        prompt = RULE_GENERATION_USER_PROMPT.format(
            question=question, ground_truth=ground_truth,
            success_rate=success_rate, quality_score=quality_score,
        )
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": RULE_GENERATION_SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.3, max_tokens=120,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"Rule generation failed: {e}")
            return ""
