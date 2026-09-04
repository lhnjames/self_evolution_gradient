from __future__ import annotations

import math

import numpy as np
import pytest

from self_evolve.action_value import (
    discounted_success,
    probability_value_metrics,
    spearman,
)


def test_discounted_success_includes_forced_action_as_zero_discount_step() -> None:
    assert discounted_success(True, 1, 0.9) == 1.0
    assert discounted_success(True, 3, 0.9) == pytest.approx(0.81)
    assert discounted_success(False, 1, 0.9) == 0.0


def test_probability_shift_toward_high_value_is_positive() -> None:
    metrics = probability_value_metrics(
        base_scores=np.asarray([2.0, 0.0]),
        seed_scores=np.asarray([0.0, 2.0]),
        values=np.asarray([0.0, 1.0]),
    )
    assert metrics["expected_value_delta"] > 0
    assert metrics["added_mass_mean_value"] == pytest.approx(1.0)
    assert metrics["removed_mass_mean_value"] == pytest.approx(0.0)
    assert metrics["top_value_delta"] == 1.0


def test_expected_value_identity_matches_mass_decomposition() -> None:
    metrics = probability_value_metrics(
        base_scores=np.asarray([0.3, -0.4, 0.1]),
        seed_scores=np.asarray([-0.2, 0.6, -0.1]),
        values=np.asarray([0.2, 0.9, 0.5]),
    )
    reconstructed = metrics["moved_probability_mass"] * metrics["added_minus_removed_value"]
    assert metrics["expected_value_delta"] == pytest.approx(reconstructed)


def test_spearman_handles_ties_and_constants() -> None:
    assert spearman(np.asarray([1.0, 2.0, 3.0]), np.asarray([3.0, 2.0, 1.0])) == pytest.approx(-1.0)
    assert math.isnan(spearman(np.ones(3), np.arange(3.0)))
