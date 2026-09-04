import math

import pytest

from self_evolve.gradient_scope import (
    direction_cosine,
    residual_novelty_from_gram,
    scope_label,
    scope_profile,
    scope_transition,
)


def test_scope_profile_distinguishes_coverage_purity_and_strength():
    profile = scope_profile([0.3, 0.02, -0.2, 0.0], epsilon=0.01)
    assert profile["positive_coverage"] == 0.5
    assert profile["negative_coverage"] == 0.25
    assert profile["scope_purity"] == pytest.approx(2 / 3)
    assert profile["mean_positive_strength"] == pytest.approx(0.16)
    assert profile["mean_negative_strength"] == pytest.approx(0.2)


def test_scope_transition_reports_drift_and_positive_jaccard():
    transition = scope_transition([0.2, 0.0, -0.2], [0.3, 0.2, 0.0], epsilon=0.01)
    assert transition["changed_count"] == 2
    assert transition["transitions"] == {
        "negative->neutral": 1,
        "neutral->positive": 1,
        "positive->positive": 1,
    }
    assert transition["positive_scope_jaccard"] == 0.5


def test_scope_label_rejects_negative_epsilon():
    with pytest.raises(ValueError):
        scope_label(0.0, -0.1)


def test_direction_cosine_uses_gradient_gram_matrix():
    gram = [[1.0, 0.0], [0.0, 1.0]]
    assert direction_cosine([1, 0], [0, 1], gram) == 0.0
    assert direction_cosine([1, 0], [1, 1], gram) == pytest.approx(1 / math.sqrt(2))


def test_residual_novelty_from_gram_handles_redundant_and_orthogonal_updates():
    assert residual_novelty_from_gram([[1.0]], [1.0], 1.0) == pytest.approx(0.0)
    assert residual_novelty_from_gram([[1.0]], [0.0], 1.0) == pytest.approx(1.0)
    assert residual_novelty_from_gram([[1.0]], [0.6], 1.0) == pytest.approx(0.8)
