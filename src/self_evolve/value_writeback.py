from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np


CONTROL_BUCKETS = (
    "same_task_same_verb",
    "same_task_different_verb",
    "different_task_same_verb",
    "different_task_different_verb",
)


def relationship_bucket(source: dict[str, Any], target: dict[str, Any]) -> str:
    same_task = source["task_type"] == target["task_type"]
    same_verb = source["action_verb"] == target["action_verb"]
    if same_task and same_verb:
        return "same_task_same_verb"
    if same_task:
        return "same_task_different_verb"
    if same_verb:
        return "different_task_same_verb"
    return "different_task_different_verb"


def _softmax(scores: Sequence[float]) -> np.ndarray:
    array = np.asarray(scores, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.all(np.isfinite(array)):
        raise ValueError("scores must be a finite, non-empty vector")
    weights = np.exp(array - np.max(array))
    return weights / np.sum(weights)


def candidate_distribution_metrics(
    baseline_scores: Sequence[float],
    updated_scores: Sequence[float],
    values: Sequence[float],
    expert_index: int,
    optimal_atol: float = 1e-12,
) -> dict[str, float | int]:
    baseline = np.asarray(baseline_scores, dtype=np.float64)
    updated = np.asarray(updated_scores, dtype=np.float64)
    value_array = np.asarray(values, dtype=np.float64)
    if baseline.shape != updated.shape or baseline.shape != value_array.shape:
        raise ValueError("score and value vectors must have identical shapes")
    if not 0 <= expert_index < len(baseline):
        raise IndexError("expert_index is outside the candidate set")
    p = _softmax(baseline)
    q = _softmax(updated)
    optimal = np.isclose(value_array, np.max(value_array), rtol=0.0, atol=optimal_atol)
    baseline_expected = float(np.dot(p, value_array))
    updated_expected = float(np.dot(q, value_array))
    baseline_optimal_mass = float(np.sum(p[optimal]))
    updated_optimal_mass = float(np.sum(q[optimal]))
    baseline_top = int(np.argmax(baseline))
    updated_top = int(np.argmax(updated))
    return {
        "candidate_count": len(baseline),
        "kl_baseline_to_updated": float(np.dot(p, np.log(p) - np.log(q))),
        "total_variation": 0.5 * float(np.sum(np.abs(p - q))),
        "max_absolute_score_delta": float(np.max(np.abs(updated - baseline))),
        "baseline_expected_value": baseline_expected,
        "updated_expected_value": updated_expected,
        "expected_value_delta": updated_expected - baseline_expected,
        "baseline_optimal_mass": baseline_optimal_mass,
        "updated_optimal_mass": updated_optimal_mass,
        "optimal_mass_delta": updated_optimal_mass - baseline_optimal_mass,
        "baseline_expert_probability": float(p[expert_index]),
        "updated_expert_probability": float(q[expert_index]),
        "expert_probability_delta": float(q[expert_index] - p[expert_index]),
        "baseline_top_index": baseline_top,
        "updated_top_index": updated_top,
        "baseline_top_value": float(value_array[baseline_top]),
        "updated_top_value": float(value_array[updated_top]),
        "top_value_delta": float(value_array[updated_top] - value_array[baseline_top]),
    }


def group_gradient_norm(parameters: Sequence[Any]) -> float:
    squares = []
    for parameter in parameters:
        if parameter.grad is None:
            raise ValueError("all group parameters must have gradients")
        gradient = parameter.grad.detach().float()
        squares.append(float(np.float64((gradient * gradient).sum().item())))
    return math.sqrt(max(math.fsum(squares), 0.0))
