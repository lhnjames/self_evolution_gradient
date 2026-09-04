from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MinimalTop1Repair:
    gold_index: int
    competitor_index: int
    gap: float
    delta: np.ndarray
    l2_norm: float


def minimal_top1_repair(scores: np.ndarray, gold_index: int) -> MinimalTop1Repair:
    """Return the minimum-L2 boundary edit against the highest non-gold action.

    The returned edit reaches a tie.  An arbitrarily small positive epsilon on
    the gold coordinate makes the gold action strictly top-1.
    """
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("scores must be a one-dimensional vector with at least two actions")
    if not 0 <= gold_index < len(values):
        raise IndexError("gold_index is outside the score vector")
    masked = values.copy()
    masked[gold_index] = -np.inf
    competitor = int(np.argmax(masked))
    gap = max(0.0, float(values[competitor] - values[gold_index]))
    delta = np.zeros_like(values)
    delta[gold_index] = gap / 2.0
    delta[competitor] = -gap / 2.0
    return MinimalTop1Repair(
        gold_index=gold_index,
        competitor_index=competitor,
        gap=gap,
        delta=delta,
        l2_norm=float(np.linalg.norm(delta)),
    )


def centered(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return array - array.mean()


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 1e-12 else float("nan")


def softmax(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    weights = np.exp(values - values.max())
    return weights / weights.sum()

