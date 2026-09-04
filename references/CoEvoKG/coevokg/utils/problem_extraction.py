"""
Problem extraction utilities for self-play training.

This module provides utilities to extract search-based problems from
proposer generation trajectories and format them for the solver phase.
"""

import json
import logging
import os
import random
import re
import string
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from coevokg.prompts.solver_prompts import (
    SOLVER_SYSTEM_PROMPT,
    SOLVER_USER_PROMPT,
    RSEARCH_SOLVER_SYSTEM_PROMPT,
    RSEARCH_SOLVER_USER_PROMPT,
)
from coevokg.reward.score.coevokg_score import em_check
from coevokg.utils.llm_as_a_judge import get_global_judge

logger = logging.getLogger(__name__)

# Enable debug logging for self-play
SELF_PLAY_DEBUG = os.environ.get("SELF_PLAY_DEBUG", "False").lower() == "true"
if SELF_PLAY_DEBUG:
    logger.setLevel(logging.DEBUG)


_LEAK_STOPWORDS = {
    "a", "an", "the", "of", "and", "or", "for", "to", "in", "on", "at",
    "by", "with", "from",
}


def _normalize_for_leak_check(text: str) -> str:
    text = (text or "").lower().strip()
    text = text.translate(str.maketrans({ch: " " for ch in string.punctuation}))
    return " ".join(text.split())


def _answer_leaked(text: str, ground_truth: str) -> bool:
    """Catch verbatim and reordered answer leakage in proposer questions."""
    text_norm = _normalize_for_leak_check(text)
    gt_norm = _normalize_for_leak_check(ground_truth)
    if not text_norm or not gt_norm:
        return False
    if gt_norm in text_norm:
        return True

    gt_tokens = [tok for tok in gt_norm.split() if tok]
    text_tokens = [tok for tok in text_norm.split() if tok]
    significant = [tok for tok in gt_tokens if tok not in _LEAK_STOPWORDS] or gt_tokens
    if len(significant) < 2:
        return False

    text_counter = Counter(text_tokens)
    return all(text_counter[tok] >= count for tok, count in Counter(significant).items())


def _entity_aliases_for_leak_check(entity: str) -> List[str]:
    """Return conservative aliases for detecting chain-entity leakage."""
    entity = (entity or "").strip()
    if not entity:
        return []

    aliases = [entity]
    no_paren = re.sub(r"\s*\([^)]*\)", "", entity).strip()
    if no_paren and no_paren != entity:
        aliases.append(no_paren)

    result = []
    seen = set()
    for alias in aliases:
        norm = _normalize_for_leak_check(alias)
        if not norm or norm in seen:
            continue
        tokens = [tok for tok in norm.split() if tok not in _LEAK_STOPWORDS]
        if len(tokens) >= 2 or len(norm) >= 4:
            result.append(alias)
            seen.add(norm)
    return result


def _chain_entity_leaked(text: str, chain_relations: List[str]) -> Optional[str]:
    """Return the first chain entity leaked in text, or None."""
    for entity in chain_relations or []:
        for alias in _entity_aliases_for_leak_check(str(entity)):
            if _answer_leaked(text, alias):
                return str(entity)
    return None


class ProblemExtractor:
    """
    Extracts search-based problems from proposer generation trajectories.

    This extractor specifically handles problems that require search capabilities,
    extracting questions from <answer></answer> tags and formatting them for solver phase.
    """

    def __init__(
        self,
        lang="zh",
        use_rag_filter=False,
        use_search_terms_filter=False,
        noisy_rag_materials=0,
        answer_pattern="answer",
        log_path=None,
    ):
        """Initialize the ProblemExtractor for search-based problems."""
        if answer_pattern == "question":
            self.answer_pattern = r"<question>\s*(.*?)\s*</question>"
        else:
            self.answer_pattern = r"<answer>\s*(.*?)\s*</answer>"

        self.information_pattern = r"<information>\s*(.*?)\s*</information>"

        self.llm_judge = get_global_judge(
            base_url=get_coevokg_env("BASE_URL"), api_key=get_coevokg_env("API_KEY", "dummy_api_key"), model=get_coevokg_env("MODEL")
        )

        self.use_rag_filter = use_rag_filter

        self.use_search_terms_filter = use_search_terms_filter

        self.noisy_rag_materials = noisy_rag_materials

        self.log_path = log_path
        if log_path:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)

        if lang == "R-Search":
            self.solver_system_prompt = RSEARCH_SOLVER_SYSTEM_PROMPT
            self.solver_user_prompt = RSEARCH_SOLVER_USER_PROMPT
        else:
            self.solver_system_prompt = SOLVER_SYSTEM_PROMPT
            self.solver_user_prompt = SOLVER_USER_PROMPT

    def extract_problems_from_trajectory_with_stats(
        self, input_text: str, output_text: str, metadata: Dict[str, Any] = None, batch_materials: List[str] = None
    ) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
        """
        Extract search-based problems from proposer generation trajectory with detailed statistics.

        Args:
            input_text: The input prompt used for proposer generation
            output_text: The proposer's generated output containing <answer></answer>
            metadata: Additional metadata about the generation
            batch_materials: List of materials from other trajectories in the batch

        Returns:
            Tuple of (extracted problems list, trajectory statistics dict)
        """
        extracted_problems = []
        metadata = metadata or {}

        traj_stats = {
            "answer_matches": 0,
            "format_error": 0,
            "valid_questions": 0,
            "successful_problems": 0,
        }

        if SELF_PLAY_DEBUG:
            logger.debug(f"Extracting problems from trajectory {metadata.get('trajectory_index', 'unknown')}")

        try:
            answer_matches = re.finditer(self.answer_pattern, output_text, re.DOTALL)

            match_list = list(answer_matches)
            traj_stats["answer_matches"] = len(match_list)

            if len(match_list) == 0:
                traj_stats["format_error"] += 1
                self._write_extraction_log("", False, "no_answer_tag", metadata)
                return extracted_problems, traj_stats

            if SELF_PLAY_DEBUG:
                logger.debug(f"Found {len(match_list)} <answer></answer> matches")

            valid_problems = []
            for match in match_list:
                question_text = match.group(1).strip()

                question_text = re.sub(r"<\\?/?answer\s*>", "", question_text)
                question_text = re.sub(r"</\s*$", "", question_text)
                question_text = re.sub(r"</answer\s*$", "", question_text)

                is_valid, reason = self._is_valid_search_question(question_text, output_text, metadata, batch_materials)
                self._write_extraction_log(question_text, is_valid, reason, metadata)

                if is_valid:
                    traj_stats["valid_questions"] += 1

                    formatted_problem = self._format_search_problem(question_text, input_text, output_text, metadata)
                    if formatted_problem:
                        valid_problems.append(formatted_problem)

            if valid_problems:
                # Randomly select one to preserve diversity when multiple valid questions exist
                extracted_problems.append(random.choice(valid_problems))
                traj_stats["successful_problems"] += 1

        except Exception as e:
            logger.warning(f"Error extracting search problems with stats: {e}")

        return extracted_problems, traj_stats

    def _extract_information_content(self, output_text: str) -> str:
        """Extract all content from <information></information> tags in the trajectory."""
        information_matches = re.findall(self.information_pattern, output_text, re.DOTALL)
        if information_matches:
            return "\n\n".join(match.strip() for match in information_matches)
        return None

    def _extract_search_terms(self, output_text: str) -> List[str]:
        """Extract search terms from <search></search> tags in the output text."""
        search_pattern = r"<search>\s*(.*?)\s*</search>"
        search_matches = re.findall(search_pattern, output_text, re.DOTALL)

        search_terms = []
        for match in search_matches:
            try:
                terms = json.loads(match)
                if isinstance(terms, list):
                    search_terms.extend([term.strip() for term in terms if term.strip()])
                else:
                    search_terms.append(match.strip())
            except (json.JSONDecodeError, ValueError):
                search_terms.append(match.strip())

        return search_terms

    def _validate_with_external_llm(
        self, question_text: str, materials: str, ground_truth: Any, assigned_noisy_materials: List[str] = None
    ) -> bool:
        """Use external LLM to validate if the question can be answered correctly with the materials."""
        if not materials or not ground_truth:
            if SELF_PLAY_DEBUG:
                logger.debug(f"Skipping external validation - no judge or no materials/ground_truth")
            return False

        try:
            final_materials = materials

            if self.noisy_rag_materials > 0 and assigned_noisy_materials:
                all_materials = [materials] + assigned_noisy_materials
                random.shuffle(all_materials)
                final_materials = "\n\n".join(all_materials)

                if SELF_PLAY_DEBUG:
                    logger.debug(
                        f"Added {len(assigned_noisy_materials)} pre-assigned noisy RAG materials to validation, material: {final_materials[:100]}..."
                    )

            llm_answer = self.llm_judge.model_based_answer(final_materials, question_text)

            if not llm_answer:
                if SELF_PLAY_DEBUG:
                    logger.debug(f"External LLM returned empty answer for question: {question_text[:100]}...")
                return False

            answer_match = re.search(r"Answer[:：]?\s*(.*)", llm_answer, re.IGNORECASE)
            if answer_match:
                llm_answer = answer_match.group(1).strip()

            is_consistent = (
                em_check(
                    prediction=llm_answer,
                    golden_answers=ground_truth,
                )
                == 1
            )

            if not is_consistent:
                is_consistent = self.llm_judge.model_based_match(
                    question=question_text, golden_answer=ground_truth, model_answer=llm_answer
                )

            if SELF_PLAY_DEBUG:
                logger.debug(f"External validation result for question '{question_text[:50]}...': {is_consistent}")
                logger.debug(f"  Ground truth: {ground_truth}")
                logger.debug(f"  LLM answer: {llm_answer[:100]}...")
                logger.debug(f"LLM Judge is consistent: {is_consistent}")

            return is_consistent

        except Exception as e:
            logger.warning(f"External LLM validation failed: {e}")
            return True

    def _write_extraction_log(self, question_text: str, passed: bool, reason: str, metadata: Dict = None):
        """Append one question validation result to the log file."""
        if not self.log_path:
            return
        import datetime
        entry = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "trajectory_index": (metadata or {}).get("trajectory_index", -1),
            "global_step": (metadata or {}).get("global_step", -1),
            "question": question_text,
            "passed": passed,
            "reason": reason,
        }
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write extraction log: {e}")

    def _is_valid_search_question(
        self,
        question_text: str,
        output_text: str = "",
        metadata: Dict = None,
        assigned_noisy_materials: List[str] = None,
    ) -> tuple:
        """Check if the extracted question is a valid search-based question.
        Returns (is_valid, reason_str). reason_str is empty string when valid.
        """
        if not question_text:
            return False, "empty_question"

        word_count = len(question_text.split(" "))
        char_count = len(question_text.strip())
        has_cjk = any('\u4e00' <= c <= '\u9fff' for c in question_text)
        if char_count < 10 or char_count > 1000 or (not has_cjk and word_count < 7):
            if SELF_PLAY_DEBUG:
                logger.debug(f"[Question Validation] question_text is invalid: {question_text}")
            return False, f"too_short_or_long: {word_count} words, {char_count} chars"

        reward_model = metadata.get("reward_model", {})
        ground_truth = None
        if isinstance(reward_model, dict):
            ground_truth_info = reward_model.get("ground_truth")
            if isinstance(ground_truth_info, dict):
                ground_truth = ground_truth_info.get("target")
            elif ground_truth_info is not None:
                ground_truth = ground_truth_info

        if _answer_leaked(question_text, ground_truth or ""):
            if SELF_PLAY_DEBUG:
                logger.debug(f"[Question Validation] ground_truth is in question_text, Invalid question")
            return False, f"answer_leaked: '{ground_truth}' in question"

        extra_info = metadata.get("extra_info", {}) if isinstance(metadata, dict) else {}
        search_terms = self._extract_search_terms(output_text)
        did_search = len(search_terms) > 0

        if did_search:
            # Proposer used search: need at least 1 query + returned information
            materials = self._extract_information_content(output_text)

            if not materials:
                if SELF_PLAY_DEBUG:
                    logger.debug(f"[Question Validation] materials is None-Invalid question")
                return False, "no_search_materials"
        else:
            # Proposer used in-prompt context (chain nodes) without searching: skip search checks
            materials = None

        if self.use_search_terms_filter:
            for search_term in search_terms:
                if _answer_leaked(search_term, ground_truth or ""):
                    if SELF_PLAY_DEBUG:
                        logger.debug(
                            f"[Question Validation] ground_truth is in search_term, Invalid question"
                        )
                    return False, f"answer_leaked_in_search_term: '{ground_truth}' in '{search_term}'"

        if self.llm_judge and self.use_rag_filter:
            ok = self._validate_with_external_llm(question_text, materials, ground_truth, assigned_noisy_materials)
            if not ok:
                return False, "rag_filter_rejected"
            return True, ""

        return True, ""

    def _format_search_problem(
        self, question_text: str, input_text: str, output_text: str, metadata: Dict
    ) -> Optional[Dict[str, Any]]:
        """Format an extracted search question into solver phase format."""
        try:
            if SELF_PLAY_DEBUG:
                logger.debug(f"Formatting search problem: {question_text[:100]}...")
                logger.debug(f"Metadata received: {metadata}")

            formatted_prompt = []
            if self.solver_system_prompt:
                formatted_prompt.append({"role": "system", "content": self.solver_system_prompt})

            formatted_prompt.append(
                {
                    "role": "user",
                    "content": self.solver_user_prompt.format(question_text),
                }
            )

            reward_model = metadata.get("reward_model", {"ground_truth": {"style": "rule"}})

            data_source = metadata.get("data_source", "self_generated")

            if SELF_PLAY_DEBUG:
                logger.debug(f"Using reward_model: {reward_model}")
                logger.debug(f"Using data_source: {data_source}")

            formatted_problem = {
                "data_source": data_source,
                "prompt": formatted_prompt,
                "ability": "fact-reasoning",
                "reward_model": reward_model,
                "extra_info": {
                    "question": question_text,
                    "need_tools_kwargs": True,
                    "split": "train",
                    "tools_kwargs": {
                        "search": {
                            "create_kwargs": {
                                "data_source": "self_generated",
                                "question": question_text,
                                "ground_truth": reward_model.get("ground_truth"),
                            }
                        }
                    },
                },
                "metadata": None,
                "extracted_question": question_text,
                "formatted_prompt": formatted_prompt,
                "problem_type": "search",
                "trajectory_index": metadata.get("trajectory_index", -1),
            }

            if SELF_PLAY_DEBUG:
                logger.debug(f"Successfully formatted problem: {formatted_problem['extracted_question']}")

            return formatted_problem

        except Exception as e:
            logger.warning(f"Error formatting search problem: {e}")
            if SELF_PLAY_DEBUG:
                logger.debug(f"Exception details: {e}", exc_info=True)
                logger.debug(f"Failed on question: {question_text}")
                logger.debug(f"Failed with metadata: {metadata}")
            return None


def _process_single_trajectory(
    trajectory_data: tuple[int, Dict[str, Any], ProblemExtractor, List[str]],
) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Process a single trajectory for problem extraction.

    Args:
        trajectory_data: Tuple of (trajectory_index, trajectory_dict, extractor, assigned_noisy_materials)

    Returns:
        Tuple of (extracted problems list, trajectory statistics dict)
    """
    i, trajectory, extractor, assigned_noisy_materials = trajectory_data
    problems = []

    traj_stats = {
        "answer_matches": 0,
        "valid_questions": 0,
        "successful_problems": 0,
        "format_error": 0,
    }

    try:
        input_text = trajectory.get("input", "")
        output_text = trajectory.get("output", "")
        metadata = trajectory.get("metadata", {}).copy()

        if SELF_PLAY_DEBUG:
            logger.debug(f"Processing trajectory {i}")

        problems, traj_stats = extractor.extract_problems_from_trajectory_with_stats(
            input_text, output_text, metadata, assigned_noisy_materials
        )

        if SELF_PLAY_DEBUG:
            logger.debug(f"Trajectory {i} yielded {len(problems)} problems")

    except Exception as e:
        logger.warning(f"Error processing trajectory {i}: {e}")
        if SELF_PLAY_DEBUG:
            logger.debug(f"Exception details for trajectory {i}: {e}", exc_info=True)

    return problems, traj_stats


def extract_problems_batch(
    trajectories: List[Dict[str, Any]], extractor: ProblemExtractor = None, max_workers: Optional[int] = None
) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Extract search-based problems from a batch of proposer generation trajectories using multi-threading.

    Args:
        trajectories: List of trajectory dictionaries with 'input', 'output', and optional 'metadata'
        extractor: ProblemExtractor instance (creates default if None)
        max_workers: Maximum number of threads to use (defaults to min(32, number of trajectories))

    Returns:
        Tuple of (extracted problems list, extraction statistics dict)
    """
    if extractor is None:
        extractor = ProblemExtractor()

    if not trajectories:
        return [], {
            "trajectories_count": 0,
            "answer_matches_count": 0,
            "valid_questions_count": 0,
            "successful_problems_count": 0,
            "format_error_count": 0,
        }

    if max_workers is None:
        max_workers = min(32, len(trajectories))

    all_problems = []

    stats = {
        "trajectories_count": len(trajectories),
        "answer_matches_count": 0,
        "valid_questions_count": 0,
        "successful_problems_count": 0,
        "format_error_count": 0,
    }

    if SELF_PLAY_DEBUG:
        logger.debug(f"Starting batch extraction from {len(trajectories)} trajectories using {max_workers} threads")

    trajectory_data = []
    if extractor.noisy_rag_materials > 0:
        all_materials = []
        for trajectory in trajectories:
            output_text = trajectory.get("output", "")
            materials = extractor._extract_information_content(output_text)
            if materials:
                all_materials.append(materials)

        if SELF_PLAY_DEBUG:
            logger.debug(
                f"Extracted {len(all_materials)} materials for noisy RAG from {len(trajectories)} trajectories"
            )

        for i, trajectory in enumerate(trajectories):
            current_materials = extractor._extract_information_content(trajectory.get("output", ""))

            available_materials = [mat for mat in all_materials if mat != current_materials]
            assigned_noisy_materials = []

            if available_materials and current_materials:
                num_to_select = min(extractor.noisy_rag_materials, len(available_materials))
                if num_to_select > 0:
                    assigned_noisy_materials = random.sample(available_materials, num_to_select)

                    if SELF_PLAY_DEBUG:
                        logger.debug(f"Trajectory {i}: assigned {num_to_select} noisy materials")

            trajectory_data.append((i, trajectory, extractor, assigned_noisy_materials))
    else:
        trajectory_data = [(i, trajectory, extractor, []) for i, trajectory in enumerate(trajectories)]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {executor.submit(_process_single_trajectory, data): data[0] for data in trajectory_data}

        for future in as_completed(future_to_index):
            trajectory_index = future_to_index[future]
            try:
                problems, traj_stats = future.result()

                all_problems.extend(problems)

                if traj_stats["answer_matches"] > 0:
                    stats["answer_matches_count"] += 1
                stats["valid_questions_count"] += traj_stats["valid_questions"]
                stats["successful_problems_count"] += traj_stats["successful_problems"]
                stats["format_error_count"] += traj_stats["format_error"]

            except Exception as e:
                logger.warning(f"Thread processing trajectory {trajectory_index} failed: {e}")
                if SELF_PLAY_DEBUG:
                    logger.debug(f"Thread exception details for trajectory {trajectory_index}: {e}", exc_info=True)

    if SELF_PLAY_DEBUG:
        logger.debug(f"Batch extraction complete: {len(all_problems)} total problems extracted")
        logger.debug(f"Extraction statistics: {stats}")

    logger.info(
        f"Extracted {len(all_problems)} search problems from {len(trajectories)} trajectories using {max_workers} threads"
    )
    return all_problems, stats
