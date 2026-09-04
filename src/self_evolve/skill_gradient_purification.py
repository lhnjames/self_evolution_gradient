from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np


def cosine_consensus_weights(
    cosine_matrix: Sequence[Sequence[float]],
    *,
    keep_fraction: float = 2.0 / 3.0,
) -> np.ndarray:
    """Return robust non-negative weights for a bank of experience gradients.

    Each experience is ranked by its median cosine to the other experiences.
    The least coherent tail is discarded and the retained weights are shifted
    above the weakest retained score.  This makes the operation deterministic,
    scale-free, and independent of any held-out evaluation state.
    """
    matrix = np.asarray(cosine_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("cosine_matrix must be square")
    count = matrix.shape[0]
    if count < 2:
        raise ValueError("at least two gradients are required")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("cosine_matrix must be finite")
    if not 0.0 < keep_fraction <= 1.0:
        raise ValueError("keep_fraction must be in (0, 1]")

    off_diagonal = np.empty((count, count - 1), dtype=np.float64)
    for index in range(count):
        off_diagonal[index] = np.delete(matrix[index], index)
    coherence = np.median(off_diagonal, axis=1)
    keep_count = max(2, min(count, int(math.ceil(count * keep_fraction))))
    retained = np.argsort(coherence, kind="stable")[-keep_count:]
    floor = float(np.min(coherence[retained]))
    raw = np.maximum(coherence[retained] - floor, 0.0)
    # Preserve all retained experiences when coherence ties exactly.
    raw += max(float(np.ptp(coherence[retained])), 1e-6) / keep_count
    weights = np.zeros(count, dtype=np.float64)
    weights[retained] = raw / raw.sum()
    return weights


def cosine_matrix_from_dots(
    dots: Sequence[Sequence[float]], norms: Sequence[float]
) -> np.ndarray:
    matrix = np.asarray(dots, dtype=np.float64)
    norm_array = np.asarray(norms, dtype=np.float64)
    if matrix.shape != (len(norm_array), len(norm_array)):
        raise ValueError("dots and norms have inconsistent shapes")
    denominator = np.outer(norm_array, norm_array)
    if np.any(denominator <= 0.0) or not np.all(np.isfinite(denominator)):
        raise ValueError("gradient norms must be positive and finite")
    result = matrix / denominator
    np.fill_diagonal(result, 1.0)
    return np.clip(result, -1.0, 1.0)


def normalized_weighted_coefficients(
    weights: Sequence[float], norms: Sequence[float]
) -> np.ndarray:
    """Coefficients for a weighted mean of unit-normalized gradients."""
    weight_array = np.asarray(weights, dtype=np.float64)
    norm_array = np.asarray(norms, dtype=np.float64)
    if weight_array.shape != norm_array.shape or weight_array.ndim != 1:
        raise ValueError("weights and norms must be one-dimensional and aligned")
    if np.any(weight_array < 0.0) or not np.isclose(weight_array.sum(), 1.0):
        raise ValueError("weights must be non-negative and sum to one")
    if np.any(norm_array <= 0.0) or not np.all(np.isfinite(norm_array)):
        raise ValueError("norms must be positive and finite")
    return weight_array / norm_array


def route_from_action(action: str) -> str:
    action = action.strip()
    return action.split(maxsplit=1)[0] if action else ""


def validate_delta_bank_metadata(
    metadata: Mapping[str, object], expected_verbs: Sequence[str]
) -> None:
    verbs = metadata.get("verbs")
    if not isinstance(verbs, list) or verbs != list(expected_verbs):
        raise ValueError("delta-bank verbs do not match the requested routing verbs")
    if metadata.get("format") != "skill_conditioned_parameter_delta_bank_v1":
        raise ValueError("unsupported delta-bank format")
