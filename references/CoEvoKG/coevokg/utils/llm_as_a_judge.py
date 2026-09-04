import json
import logging
import os
import random
import re
import threading
from typing import Any, Callable, List, Optional, Tuple

from openai import OpenAI

from coevokg.utils.env import get_coevokg_env
from coevokg.prompts.judge_prompts import (
    JUDGE_SYSTEM_PROMPT,
    JUDGE_EVALUATION_PROMPT,
    JUDGE_ANSWER_FROM_MATERIALS_PROMPT,
    JUDGE_WITH_CHAIN_EXTRACTION_PROMPT,
    WRITEBACK_CHAIN_EXTRACTION_PROMPT,
    JUDGE_AND_EXTRACT_PROMPT,
)
from coevokg.utils.api_provider_pool import (
    BadRequestNoRetry,
    FatalProviderPoolExhausted,
    get_api_provider_pool,
)

logger = logging.getLogger(__name__)


_MAX_KEY_SWITCHES = 10   # raise only after this many key-switches in one call


class SyncLLMAsAJudge:
    """Synchronous LLM-as-a-Judge with cycling API key pool.

    Keys have a daily quota. Strategy:
      - Try current key; on any failure switch to the next key and retry.
      - Cycle: key0 → key1 → key0 → key1 → …
      - Only raise an exception after switching keys > _MAX_KEY_SWITCHES times
        in a single call, which means all keys are genuinely exhausted.
      - On success: remember that key so the next call starts there.
    """

    def __init__(self, base_url: str, api_key: str, model: str,
                 extra_api_keys: Optional[List[str]] = None):
        _timeout = float(get_coevokg_env("JUDGE_TIMEOUT", "60.0"))
        self._clients = []
        self._current_idx = 0
        self._lock = threading.Lock()
        self.model = model
        self._pool = get_api_provider_pool("judge", base_url=base_url, timeout=_timeout)

    def _call_with_fallback(self, fn: Callable[[OpenAI], Any]) -> Any:
        """Cycle through keys on failure; raise after > _MAX_KEY_SWITCHES switches.

        Each failure switches to the next key (key0→key1→key0→…). Raises only
        when the switch count exceeds _MAX_KEY_SWITCHES, indicating both keys
        are exhausted. Resets the preferred key index on success.
        """
        return self._pool.call(lambda client, model: fn(client, model))
        n = len(self._clients)
        with self._lock:
            start = self._current_idx

        last_exc: Optional[Exception] = None
        switches = 0
        idx = start

        while switches <= _MAX_KEY_SWITCHES:
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
                    f"[judge] key[{idx}] failed ({type(e).__name__}: {e}), "
                    f"switch #{switches}/{_MAX_KEY_SWITCHES} → key[{next_idx}]"
                )
                idx = next_idx

        raise RuntimeError(
            f"All API keys exhausted after {switches} switches"
        ) from last_exc

    @staticmethod
    def _completion_content(response: Any) -> str:
        choices = getattr(response, "choices", None)
        if not choices:
            raise RuntimeError("empty chat completion choices")
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        if not content:
            raise RuntimeError("empty chat completion content")
        return str(content).strip()

    def _call_chat_content(self, fn: Callable[[OpenAI, str], Any]) -> str:
        return self._pool.call(lambda client, model: self._completion_content(fn(client, model)))

    @staticmethod
    def _strip_markdown_fence(content: str) -> str:
        text = content.strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
        return text.strip()

    @staticmethod
    def _parse_judge_chain_response(content: str) -> Tuple[bool, List[str]]:
        """Parse judge+chain output, accepting mildly malformed JSON.

        Lite judge models occasionally return almost-JSON such as:
          {"correct": false, "chain": ["A", B", "C"]}
        The final-answer judgment is still usable, and quoted chain entities are
        better than dropping the whole call to a plain judge fallback.
        """
        text = SyncLLMAsAJudge._strip_markdown_fence(content)
        try:
            parsed = json.loads(text)
            correct = bool(parsed.get("correct", False))
            chain = [str(e) for e in parsed.get("chain", []) if e]
            return correct, chain
        except Exception:
            pass

        correct_match = re.search(r'"?correct"?\s*:\s*(true|false)', text, flags=re.IGNORECASE)
        if correct_match:
            correct = correct_match.group(1).lower() == "true"
        else:
            lower = text.lower()
            if re.search(r"\bcorrect\b|\btrue\b", lower):
                correct = True
            elif re.search(r"\bwrong\b|\bincorrect\b|\bfalse\b", lower):
                correct = False
            else:
                raise ValueError(f"cannot parse correctness from judge response: {text[:200]}")

        chain: List[str] = []
        chain_match = re.search(r'"?chain"?\s*:\s*\[(.*?)\]', text, flags=re.DOTALL | re.IGNORECASE)
        if chain_match:
            chain_blob = chain_match.group(1)
            # Prefer properly quoted items. This skips malformed fragments
            # instead of failing the entire response.
            chain = [m.strip() for m in re.findall(r'"([^"]+)"', chain_blob) if m.strip()]
        return correct, chain

    def model_based_match(self, question: str, golden_answer: Any, model_answer: str) -> bool:
        """Model-based answer matching"""
        prompt = JUDGE_EVALUATION_PROMPT.format(
            question=question,
            model_answer=model_answer,
            golden_answer=golden_answer,
        )

        try:
            judgment = self._call_chat_content(lambda c, m: c.chat.completions.create(
                model=m,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                stream=False,
                temperature=0.1,
                top_p=1.0,
                max_tokens=int(get_coevokg_env("JUDGE_SIMPLE_MAX_TOKENS", "10")),
            ))
            logger.debug(f"LLM judgment result: {judgment}")

            # NB: substring "correct" also appears in "incorrect" — exclude the
            # negative forms explicitly so "Incorrect"/"Wrong" are not read as correct.
            j = judgment.strip().lower()
            return ("correct" in j) and ("incorrect" not in j) and ("wrong" not in j)

        except BadRequestNoRetry as e:
            logger.error(f"LLM judgment bad request (not retried): {e}")
            return False
        except FatalProviderPoolExhausted:
            raise
        except Exception as e:
            logger.error(f"LLM judgment error: {e}")
            return False

    def batch_model_based_match(self, evaluations: list[dict]) -> list[bool]:
        """
        Batch evaluation of answers

        Args:
            evaluations: List of dictionaries containing question, golden_answer, model_answer

        Returns:
            list[bool]: Judgment result for each answer
        """
        results = []
        for eval_data in evaluations:
            try:
                result = self.model_based_match(
                    question=eval_data["question"],
                    golden_answer=eval_data["golden_answer"],
                    model_answer=eval_data["model_answer"],
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Batch evaluation error: {e}")
                results.append(False)
        return results

    def model_based_match_with_chain(
        self, question: str, golden_answer: Any, model_answer: str,
        final_answer: Optional[str] = None,
    ) -> Tuple[bool, List[str]]:
        """
        Judge correctness AND extract the named-entity thought chain in one call.

        Correctness is judged on `final_answer` (the model's extracted <answer>);
        `model_answer` (the full reasoning) is used ONLY for entity extraction.
        If `final_answer` is None, falls back to using `model_answer` for both.

        Returns:
            (correct: bool, chain: list[str])
            On any error, falls back to model_based_match() and returns (result, []).
        """
        prompt = JUDGE_WITH_CHAIN_EXTRACTION_PROMPT.format(
            question=question,
            final_answer=final_answer if final_answer is not None else model_answer,
            reasoning=model_answer,
            golden_answer=golden_answer,
        )

        try:
            max_tokens = int(get_coevokg_env("JUDGE_CHAIN_MAX_TOKENS", "192"))
            content = self._call_chat_content(lambda c, m: c.chat.completions.create(
                model=m,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                stream=False,
                temperature=0.1,
                top_p=1.0,
                max_tokens=max_tokens,
            ))
            correct, chain = self._parse_judge_chain_response(content)
            logger.debug(f"Judge+chain: correct={correct}, chain={chain}")
            return correct, chain
        except BadRequestNoRetry as e:
            logger.warning(f"model_based_match_with_chain bad request (not retried): {e}")
            return False, []
        except FatalProviderPoolExhausted:
            raise
        except Exception as e:
            logger.warning(f"model_based_match_with_chain failed ({e}), falling back to plain judge")
            try:
                # Judge correctness on the final answer only. Prefer the explicit
                # final_answer; otherwise recover it from the reasoning's <answer> tag.
                if final_answer is not None:
                    fallback_answer = final_answer
                else:
                    import re as _re
                    _m = _re.search(r'<answer>(.*?)</answer>', model_answer, _re.DOTALL)
                    fallback_answer = _m.group(1).strip() if _m else model_answer[:500]
                correct = self.model_based_match(question, golden_answer, fallback_answer)
            except Exception:
                correct = False
            return correct, []

    def judge_and_extract(
        self, question: str, golden_answer: Any, final_answer: str,
        reasoning: str, titles: List[str],
    ) -> Tuple[Optional[bool], List[str], List[str]]:
        """Unified single call: judge correctness AND extract canonical chain + relations.

        Used by the reward stage so the SAME chain feeds the path-support process
        reward and (via passback) the KG write-back — no second extraction call.

        Returns (correct, chain, relations):
          - correct is True/False on success; ``None`` signals the LLM call/parse
            failed, so the caller should fall back to EM for correctness.
          - chain/relations are ([], []) when the LLM returns no usable chain.
            On success len(relations) == len(chain)-1 (best-effort; caller pads).
        """
        prompt = JUDGE_AND_EXTRACT_PROMPT.format(
            question=question,
            final_answer=final_answer,
            golden_answer=golden_answer,
            titles=json.dumps(titles, ensure_ascii=False),
            reasoning=reasoning,
        )
        try:
            max_tokens = int(get_coevokg_env("JUDGE_CHAIN_MAX_TOKENS", "384"))
            content = self._call_chat_content(lambda c, m: c.chat.completions.create(
                model=m,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                stream=False,
                temperature=0.1,
                top_p=1.0,
                max_tokens=max_tokens,
            ))
        except BadRequestNoRetry as e:
            logger.warning(f"judge_and_extract bad request (not retried): {e}")
            return None, [], []
        except FatalProviderPoolExhausted:
            raise
        except Exception as e:
            logger.warning(f"judge_and_extract failed ({e}); caller should EM-fallback")
            return None, [], []
        # Parse {"correct": bool, "chain": [...], "relations": [...]}
        try:
            text = content.strip()
            text = re.sub(r'^```(json)?|```$', '', text, flags=re.MULTILINE).strip()
            m = re.search(r'\{.*\}', text, re.DOTALL)
            if not m:
                return None, [], []
            obj = json.loads(m.group(0))
            if "correct" not in obj:
                return None, [], []
            correct = bool(obj.get("correct", False))
            chain = [str(e).strip() for e in (obj.get("chain") or []) if str(e).strip()]
            relations = [str(r).strip() for r in (obj.get("relations") or [])]
            return correct, chain, relations
        except Exception as e:
            logger.warning(f"judge_and_extract parse failed ({e}); caller should EM-fallback")
            return None, [], []

    def extract_chain_relations(
        self, question: str, final_answer: str, titles: List[str], reasoning: str,
    ) -> Tuple[List[str], List[str]]:
        """Extract a canonical-named evidence chain + per-edge relations for write-back.

        Correctness is already known at write-back time, so this only extracts the
        chain and relation labels. Returns (chain, relations) with
        len(relations) == len(chain)-1 on success; ([], []) on any error/empty.
        """
        prompt = WRITEBACK_CHAIN_EXTRACTION_PROMPT.format(
            question=question,
            final_answer=final_answer,
            titles=json.dumps(titles, ensure_ascii=False),
            reasoning=reasoning,
        )
        try:
            max_tokens = int(get_coevokg_env("JUDGE_CHAIN_MAX_TOKENS", "384"))
            content = self._call_chat_content(lambda c, m: c.chat.completions.create(
                model=m,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                stream=False,
                temperature=0.1,
                top_p=1.0,
                max_tokens=max_tokens,
            ))
        except FatalProviderPoolExhausted:
            raise
        except Exception as e:
            logger.warning(f"extract_chain_relations failed ({e})")
            return [], []
        # Parse {"chain": [...], "relations": [...]}
        try:
            text = content.strip()
            text = re.sub(r'^```(json)?|```$', '', text, flags=re.MULTILINE).strip()
            m = re.search(r'\{.*\}', text, re.DOTALL)
            if not m:
                return [], []
            obj = json.loads(m.group(0))
            chain = [str(e).strip() for e in (obj.get("chain") or []) if str(e).strip()]
            relations = [str(r).strip() for r in (obj.get("relations") or [])]
            return chain, relations
        except Exception as e:
            logger.warning(f"extract_chain_relations parse failed ({e})")
            return [], []

    def model_based_answer(self, materials, question) -> str:
        """
        Model-based answer generation
        """

        prompt = JUDGE_ANSWER_FROM_MATERIALS_PROMPT.format(
            materials=materials,
            question=question,
        )

        try:
            answer = self._call_chat_content(lambda c, m: c.chat.completions.create(
                model=m,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                stream=False,
                temperature=0.1,
                top_p=1.0,
                max_tokens=70,
            ))

            return answer

        except BadRequestNoRetry as e:
            logger.error(f"LLM answer bad request (not retried): {e}")
            return ""
        except FatalProviderPoolExhausted:
            raise
        except Exception as e:
            logger.error(f"LLM answer error: {e}")
            return ""

    def close(self):
        """Close all client connections."""
        for c in getattr(self, "_clients", []):
            try:
                c.close()
            except Exception:
                pass

    def __del__(self):
        """Auto close on destruction"""
        self.close()


# Singleton pattern management
_global_judge = None


def get_global_judge(base_url: str, api_key: str, model: str = "gpt-4") -> SyncLLMAsAJudge:
    """Get global judge instance (auto-reads COEVOKG_API_KEY_2 as fallback key)."""
    global _global_judge
    if _global_judge is None:
        extra = [k for k in [get_coevokg_env("API_KEY_2")] if k]
        _global_judge = SyncLLMAsAJudge(
            base_url=base_url, api_key=api_key, model=model,
            extra_api_keys=extra or None,
        )
        if extra:
            logger.info(f"LLM judge: {1 + len(extra)} API keys (primary + fallback on 429/5xx)")
    return _global_judge


def create_sync_llm_judge(base_url: str, api_key: str, model: str = "gpt-4") -> SyncLLMAsAJudge:
    """Create synchronous LLM judge instance"""
    return SyncLLMAsAJudge(base_url=base_url, api_key=api_key, model=model)


# Usage example
def example_usage():
    """Usage example"""
    import os

    base_url = get_coevokg_env("BASE_URL")
    model = get_coevokg_env("MODEL")
    judge = create_sync_llm_judge(base_url=base_url, api_key="your-api-key", model=model)

    # Single evaluation
    result = judge.model_based_match(
        question="What is the capital of France?", golden_answer="Paris", model_answer="Paris is the capital city of France"
    )
    print(f"Evaluation result: {result}")

    # Batch evaluation
    evaluations = [
        {"question": "What is 1+1?", "golden_answer": "2", "model_answer": "The answer is 2"},
        {"question": "How many moons does Earth have?", "golden_answer": "1", "model_answer": "Earth has one natural satellite, the Moon"},
    ]

    batch_results = judge.batch_model_based_match(evaluations)
    print(f"Batch evaluation results: {batch_results}")

    # Explicit close (optional)
    judge.close()


if __name__ == "__main__":
    example_usage()
