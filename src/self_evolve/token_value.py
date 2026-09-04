from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np


RELATIONS = {"at", "from", "in", "into", "on", "onto", "to", "with"}
SPECIAL_COMMANDS = {"help", "inventory", "look"}


def semantic_word_roles(action: str) -> list[str]:
    """Assign ALFWorld command words to stable, human-readable roles."""
    words = action.split()
    if not words:
        return []
    verb = words[0].lower()
    roles = ["action_verb"]
    relation_index = next(
        (index for index, word in enumerate(words[1:], start=1) if word.lower() in RELATIONS),
        None,
    )
    for index, word in enumerate(words[1:], start=1):
        lower = word.lower()
        if lower in RELATIONS:
            roles.append("relation")
        elif re.fullmatch(r"\d+", lower):
            roles.append("index")
        elif verb in SPECIAL_COMMANDS:
            roles.append("special")
        elif verb == "go":
            roles.append("location")
        elif verb in {"open", "close"}:
            roles.append("receptacle")
        elif verb in {"clean", "cool", "heat"}:
            roles.append("appliance" if relation_index is not None and index > relation_index else "object")
        elif verb in {"put", "move", "take"}:
            roles.append("receptacle" if relation_index is not None and index > relation_index else "object")
        elif verb in {"examine", "use"}:
            roles.append("object")
        else:
            roles.append("argument")
    return roles


def semantic_token_roles(action: str, offsets: Sequence[Sequence[int]]) -> list[str]:
    """Map fast-tokenizer character offsets for ``" " + action`` to word roles."""
    text = " " + action
    words = list(re.finditer(r"\S+", text))
    word_roles = semantic_word_roles(action)
    if len(words) != len(word_roles):
        raise AssertionError("Word-role alignment failed")
    result: list[str] = []
    for start, end in offsets:
        overlaps = [
            (min(int(end), match.end()) - max(int(start), match.start()), role)
            for match, role in zip(words, word_roles, strict=True)
            if min(int(end), match.end()) > max(int(start), match.start())
        ]
        if overlaps:
            result.append(max(overlaps)[1])
        elif text[int(start) : int(end)].isspace():
            result.append("separator")
        else:
            result.append("format_or_special")
    return result


def build_prefix_trie(token_ids_by_action: Sequence[Sequence[int]]) -> dict[tuple[int, ...], dict[int, tuple[int, ...]]]:
    """Return every pre-token prefix and the actions reachable through each child."""
    mutable: dict[tuple[int, ...], dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for action_index, token_ids in enumerate(token_ids_by_action):
        if not token_ids:
            raise ValueError(f"Action {action_index} has no tokens")
        for depth, token_id in enumerate(token_ids):
            mutable[tuple(int(x) for x in token_ids[:depth])][int(token_id)].append(action_index)
    return {
        prefix: {token_id: tuple(indices) for token_id, indices in children.items()}
        for prefix, children in mutable.items()
    }


def role_score_contributions(
    token_logprob_deltas: Sequence[float], token_roles: Sequence[str]
) -> dict[str, float]:
    """Decompose a length-normalized action-score delta exactly by token role."""
    if len(token_logprob_deltas) != len(token_roles):
        raise ValueError("Token deltas and roles must have identical lengths")
    if not token_roles:
        raise ValueError("At least one token is required")
    scale = 1.0 / len(token_roles)
    result: dict[str, float] = defaultdict(float)
    for delta, role in zip(token_logprob_deltas, token_roles, strict=True):
        result[role] += float(delta) * scale
    return dict(sorted(result.items()))


def conditional_probabilities(probabilities: Sequence[float]) -> list[float]:
    total = math.fsum(float(value) for value in probabilities)
    if total <= 0.0:
        raise ValueError("Conditional probability mass must be positive")
    return [float(value) / total for value in probabilities]


def branch_value_summary(action_indices: Sequence[int], values: Sequence[float]) -> dict[str, Any]:
    selected = [float(values[index]) for index in action_indices]
    if not selected:
        raise ValueError("A trie branch must contain an action")
    return {
        "actions": len(selected),
        "mean": math.fsum(selected) / len(selected),
        "min": min(selected),
        "max": max(selected),
    }


def _softmax_array(scores: Sequence[float]) -> np.ndarray:
    array = np.asarray(scores, dtype=np.float64)
    if array.ndim != 1 or not len(array):
        raise ValueError("scores must be a non-empty vector")
    weights = np.exp(array - np.max(array))
    return weights / np.sum(weights)


def value_expectation_loss_coefficients(
    scores: Sequence[float], values: Sequence[float]
) -> np.ndarray:
    """d[-E_softmax(score)[V]] / d score."""
    probabilities = _softmax_array(scores)
    value_array = np.asarray(values, dtype=np.float64)
    if value_array.shape != probabilities.shape:
        raise ValueError("scores and values must have identical shapes")
    expected = float(np.dot(probabilities, value_array))
    return -probabilities * (value_array - expected)


def optimal_set_loss_coefficients(
    scores: Sequence[float], values: Sequence[float], atol: float = 1e-12
) -> np.ndarray:
    """Gradient of -log probability mass on the complete value-optimal set."""
    probabilities = _softmax_array(scores)
    value_array = np.asarray(values, dtype=np.float64)
    if value_array.shape != probabilities.shape:
        raise ValueError("scores and values must have identical shapes")
    optimal = np.isclose(value_array, np.max(value_array), rtol=0.0, atol=atol)
    optimal_mass = float(np.sum(probabilities[optimal]))
    result = probabilities.copy()
    result[optimal] -= probabilities[optimal] / optimal_mass
    return result


def one_hot_nll_coefficients(scores: Sequence[float], target_index: int) -> np.ndarray:
    """Gradient of candidate-set cross entropy; retained only as a control."""
    probabilities = _softmax_array(scores)
    if not 0 <= target_index < len(probabilities):
        raise IndexError("target index is outside the candidate set")
    probabilities[target_index] -= 1.0
    return probabilities
