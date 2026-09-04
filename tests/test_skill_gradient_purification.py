import numpy as np
import pytest

from self_evolve.skill_gradient_purification import (
    cosine_consensus_weights,
    cosine_matrix_from_dots,
    normalized_weighted_coefficients,
    route_from_action,
)


def test_cosine_consensus_discards_incompatible_tail():
    matrix = np.asarray(
        [
            [1.0, 0.8, 0.7, -0.9],
            [0.8, 1.0, 0.6, -0.8],
            [0.7, 0.6, 1.0, -0.7],
            [-0.9, -0.8, -0.7, 1.0],
        ]
    )
    weights = cosine_consensus_weights(matrix, keep_fraction=0.75)
    assert weights.sum() == pytest.approx(1.0)
    assert np.count_nonzero(weights) == 3
    assert weights[3] == 0.0


def test_cosine_matrix_and_unit_gradient_coefficients():
    dots = [[4.0, 3.0], [3.0, 9.0]]
    matrix = cosine_matrix_from_dots(dots, [2.0, 3.0])
    np.testing.assert_allclose(matrix, [[1.0, 0.5], [0.5, 1.0]])
    coefficients = normalized_weighted_coefficients([0.25, 0.75], [2.0, 3.0])
    assert coefficients.tolist() == pytest.approx([0.125, 0.25])


def test_action_route_uses_executed_verb():
    assert route_from_action("open drawer 1") == "open"
    assert route_from_action("  move apple 1 to table 1 ") == "move"
    assert route_from_action("") == ""
