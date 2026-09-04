from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, Sequence

import numpy as np


def scope_label(value: float, epsilon: float = 0.0) -> str:
    """Classify an empirical value response into positive/neutral/negative scope."""
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    if value > epsilon:
        return "positive"
    if value < -epsilon:
        return "negative"
    return "neutral"


def scope_profile(values: Iterable[float], epsilon: float = 0.0) -> dict[str, float | int]:
    values_array = np.asarray(list(values), dtype=np.float64)
    if values_array.size == 0:
        raise ValueError("scope_profile requires at least one value")
    labels = [scope_label(float(value), epsilon) for value in values_array]
    positive = values_array[np.asarray([label == "positive" for label in labels])]
    negative = values_array[np.asarray([label == "negative" for label in labels])]
    positive_count = int(positive.size)
    negative_count = int(negative.size)
    active_count = positive_count + negative_count
    return {
        "count": int(values_array.size),
        "epsilon": float(epsilon),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "neutral_count": int(values_array.size - active_count),
        "positive_coverage": positive_count / float(values_array.size),
        "negative_coverage": negative_count / float(values_array.size),
        "scope_purity": positive_count / float(active_count) if active_count else float("nan"),
        "mean_value_delta": float(values_array.mean()),
        "mean_positive_strength": float(positive.mean()) if positive_count else 0.0,
        "mean_negative_strength": float((-negative).mean()) if negative_count else 0.0,
    }


def scope_transition(
    previous: Sequence[float], current: Sequence[float], epsilon: float = 0.0
) -> dict[str, float | int | dict[str, int]]:
    if len(previous) != len(current) or not previous:
        raise ValueError("previous/current must be non-empty and have equal length")
    before = [scope_label(float(value), epsilon) for value in previous]
    after = [scope_label(float(value), epsilon) for value in current]
    transitions = Counter(f"{left}->{right}" for left, right in zip(before, after, strict=True))
    before_positive = {i for i, label in enumerate(before) if label == "positive"}
    after_positive = {i for i, label in enumerate(after) if label == "positive"}
    union = before_positive | after_positive
    jaccard = len(before_positive & after_positive) / len(union) if union else 1.0
    changed = sum(left != right for left, right in zip(before, after, strict=True))
    return {
        "count": len(previous),
        "epsilon": float(epsilon),
        "changed_count": changed,
        "changed_fraction": changed / len(previous),
        "positive_scope_jaccard": jaccard,
        "transitions": dict(sorted(transitions.items())),
    }


def direction_cosine(
    left_weights: Sequence[float],
    right_weights: Sequence[float],
    source_cosine: Sequence[Sequence[float]],
) -> float:
    cosine = np.asarray(source_cosine, dtype=np.float64)
    left = np.asarray(left_weights, dtype=np.float64)
    right = np.asarray(right_weights, dtype=np.float64)
    if cosine.shape != (left.size, left.size) or right.size != left.size:
        raise ValueError("weight and cosine dimensions do not match")
    left_norm = math.sqrt(max(float(left @ cosine @ left), 0.0))
    right_norm = math.sqrt(max(float(right @ cosine @ right), 0.0))
    if min(left_norm, right_norm) <= 1e-20:
        raise ValueError("direction norm vanished")
    return float(np.clip((left @ cosine @ right) / (left_norm * right_norm), -1.0, 1.0))


def residual_novelty_from_gram(
    previous_gram: Sequence[Sequence[float]],
    cross_dots: Sequence[float],
    new_squared_norm: float,
    rcond: float = 1e-10,
) -> float:
    """Return ||g_new - P_span(previous) g_new|| / ||g_new|| using only Gram products."""
    gram = np.asarray(previous_gram, dtype=np.float64)
    cross = np.asarray(cross_dots, dtype=np.float64)
    if gram.shape != (cross.size, cross.size):
        raise ValueError("previous Gram and cross-dot dimensions do not match")
    if new_squared_norm <= 0 or rcond <= 0:
        raise ValueError("new_squared_norm and rcond must be positive")
    if cross.size == 0:
        return 1.0
    projection_squared_norm = float(cross @ np.linalg.pinv(gram, rcond=rcond) @ cross)
    residual_squared_norm = max(float(new_squared_norm) - projection_squared_norm, 0.0)
    return float(np.clip(math.sqrt(residual_squared_norm / new_squared_norm), 0.0, 1.0))
