from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class RepairBasis:
    """Orthonormal basis represented as coefficients over unit gradient atoms."""

    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    atom_coefficients: np.ndarray


def orthonormal_repair_basis(
    unit_atom_gram: Sequence[Sequence[float]],
    *,
    relative_tolerance: float = 1e-10,
) -> RepairBasis:
    gram = np.asarray(unit_atom_gram, dtype=np.float64)
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
        raise ValueError("unit_atom_gram must be square")
    if not np.all(np.isfinite(gram)):
        raise ValueError("unit_atom_gram must be finite")
    gram = 0.5 * (gram + gram.T)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    if not len(eigenvalues) or eigenvalues[0] <= 0:
        raise ValueError("unit_atom_gram has no positive direction")
    keep = eigenvalues > relative_tolerance * eigenvalues[0]
    eigenvalues = eigenvalues[keep]
    eigenvectors = eigenvectors[:, keep]
    atom_coefficients = eigenvectors / np.sqrt(eigenvalues)[None, :]
    return RepairBasis(eigenvalues, eigenvectors, atom_coefficients)


def target_coordinates(
    basis: RepairBasis,
    unit_atom_target_dots: Sequence[Sequence[float]],
) -> np.ndarray:
    dots = np.asarray(unit_atom_target_dots, dtype=np.float64)
    if dots.ndim != 2 or dots.shape[0] != basis.atom_coefficients.shape[0]:
        raise ValueError("unit_atom_target_dots has incompatible shape")
    return basis.atom_coefficients.T @ dots


def atom_weights_from_coordinates(
    basis: RepairBasis,
    coordinates: Sequence[float],
) -> np.ndarray:
    values = np.asarray(coordinates, dtype=np.float64)
    if values.ndim != 1 or len(values) > basis.atom_coefficients.shape[1]:
        raise ValueError("coordinates has incompatible shape")
    return basis.atom_coefficients[:, : len(values)] @ values


def sample_unit_directions(rank: int, count: int, seed: int) -> np.ndarray:
    if rank <= 0 or count <= 0:
        raise ValueError("rank and count must be positive")
    if rank == 1:
        return np.asarray([[1.0], [-1.0]], dtype=np.float64)
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(count, rank))
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    return values


def circular_component_count(mask: Sequence[bool]) -> int:
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 1 or not len(values):
        raise ValueError("mask must be a non-empty vector")
    if np.all(values):
        return 1
    if not np.any(values):
        return 0
    return int(np.sum(values & ~np.roll(values, 1)))


def pareto_mask(gain: Sequence[float], harm: Sequence[float]) -> np.ndarray:
    gains = np.asarray(gain, dtype=np.float64)
    harms = np.asarray(harm, dtype=np.float64)
    if gains.shape != harms.shape or gains.ndim != 1:
        raise ValueError("gain and harm must be equal-length vectors")
    order = np.lexsort((-gains, harms))
    result = np.zeros(len(gains), dtype=bool)
    best_gain = -np.inf
    for index in order:
        if gains[index] > best_gain + 1e-15:
            result[index] = True
            best_gain = gains[index]
    return result
