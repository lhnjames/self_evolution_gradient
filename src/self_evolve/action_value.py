from __future__ import annotations

import math

import numpy as np


def discounted_success(won: bool, recovery_steps: int, gamma: float) -> float:
    """Binary task success discounted by steps from the forced action onward."""
    if not 0.0 < gamma <= 1.0:
        raise ValueError("gamma must be in (0, 1]")
    if recovery_steps < 1:
        raise ValueError("recovery_steps must include the forced action")
    return float(gamma ** (recovery_steps - 1)) if won else 0.0


def softmax(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("scores must be a non-empty one-dimensional vector")
    weights = np.exp(values - np.max(values))
    return weights / np.sum(weights)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    x = _average_ranks(np.asarray(left, dtype=np.float64))
    y = _average_ranks(np.asarray(right, dtype=np.float64))
    x -= np.mean(x)
    y -= np.mean(y)
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / denominator) if denominator > 1e-12 else float("nan")


def probability_value_metrics(
    base_scores: np.ndarray,
    seed_scores: np.ndarray,
    values: np.ndarray,
) -> dict[str, float | int]:
    """Measure whether checkpoint probability moves toward higher-value actions.

    Added/removed probability mass means are unique even though a pairwise flow
    decomposition between donor and receiver actions is not unique.
    """
    base_scores = np.asarray(base_scores, dtype=np.float64)
    seed_scores = np.asarray(seed_scores, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if base_scores.shape != seed_scores.shape or base_scores.shape != values.shape:
        raise ValueError("base scores, seed scores, and values must have the same shape")
    if base_scores.ndim != 1 or not len(base_scores):
        raise ValueError("inputs must be non-empty one-dimensional vectors")
    if not np.all(np.isfinite(values)):
        raise ValueError("values must be finite")

    base_p = softmax(base_scores)
    seed_p = softmax(seed_scores)
    delta = seed_p - base_p
    added = np.clip(delta, 0.0, None)
    removed = np.clip(-delta, 0.0, None)
    moved_mass = float(np.sum(added))
    added_value = float(np.dot(added, values) / moved_mass) if moved_mass > 1e-12 else math.nan
    removed_value = float(np.dot(removed, values) / moved_mass) if moved_mass > 1e-12 else math.nan
    optimum = float(np.max(values))
    optimal = np.isclose(values, optimum, rtol=0.0, atol=1e-12)
    base_top = int(np.argmax(base_scores))
    seed_top = int(np.argmax(seed_scores))
    return {
        "base_expected_value": float(np.dot(base_p, values)),
        "seed_expected_value": float(np.dot(seed_p, values)),
        "expected_value_delta": float(np.dot(delta, values)),
        "moved_probability_mass": moved_mass,
        "added_mass_mean_value": added_value,
        "removed_mass_mean_value": removed_value,
        "added_minus_removed_value": added_value - removed_value,
        "base_top_index": base_top,
        "seed_top_index": seed_top,
        "base_top_value": float(values[base_top]),
        "seed_top_value": float(values[seed_top]),
        "top_value_delta": float(values[seed_top] - values[base_top]),
        "optimal_value": optimum,
        "base_top_is_value_optimal": int(optimal[base_top]),
        "seed_top_is_value_optimal": int(optimal[seed_top]),
        "base_probability_on_value_optimal": float(np.sum(base_p[optimal])),
        "seed_probability_on_value_optimal": float(np.sum(seed_p[optimal])),
        "probability_on_value_optimal_delta": float(np.sum(delta[optimal])),
        "base_score_value_spearman": spearman(base_scores, values),
        "seed_score_value_spearman": spearman(seed_scores, values),
    }
