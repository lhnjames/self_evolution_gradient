from __future__ import annotations

import numpy as np
import pytest

from self_evolve.token_value import (
    branch_value_summary,
    build_prefix_trie,
    conditional_probabilities,
    one_hot_nll_coefficients,
    optimal_set_loss_coefficients,
    role_score_contributions,
    semantic_token_roles,
    semantic_word_roles,
    value_expectation_loss_coefficients,
)


def test_semantic_word_roles_cover_alfworld_grammar() -> None:
    assert semantic_word_roles("go to sinkbasin 1") == [
        "action_verb",
        "relation",
        "location",
        "index",
    ]
    assert semantic_word_roles("take apple 1 from countertop 2") == [
        "action_verb",
        "object",
        "index",
        "relation",
        "receptacle",
        "index",
    ]
    assert semantic_word_roles("heat apple 1 with microwave 1") == [
        "action_verb",
        "object",
        "index",
        "relation",
        "appliance",
        "index",
    ]


def test_semantic_token_roles_support_subwords() -> None:
    # Offsets refer to " go to sinkbasin 1" and split sinkbasin into two pieces.
    offsets = [(0, 3), (3, 6), (7, 11), (11, 16), (17, 18)]
    assert semantic_token_roles("go to sinkbasin 1", offsets) == [
        "action_verb",
        "relation",
        "location",
        "location",
        "index",
    ]


def test_semantic_token_roles_marks_standalone_space() -> None:
    assert semantic_token_roles("go to sinkbasin 1", [(16, 17)]) == ["separator"]


def test_prefix_trie_preserves_shared_branches() -> None:
    trie = build_prefix_trie([[10, 20, 30], [10, 20, 31], [11]])
    assert trie[()][10] == (0, 1)
    assert trie[()][11] == (2,)
    assert trie[(10,)][20] == (0, 1)
    assert trie[(10, 20)][30] == (0,)
    assert trie[(10, 20)][31] == (1,)


def test_role_contributions_reconstruct_normalized_score_delta() -> None:
    result = role_score_contributions([0.6, -0.2, 0.8], ["action_verb", "object", "object"])
    assert sum(result.values()) == pytest.approx((0.6 - 0.2 + 0.8) / 3)
    assert result["object"] == pytest.approx(0.2)


def test_probability_and_branch_helpers() -> None:
    assert conditional_probabilities([0.2, 0.3]) == pytest.approx([0.4, 0.6])
    assert branch_value_summary([0, 2], [0.1, 0.4, 0.9]) == {
        "actions": 2,
        "mean": 0.5,
        "min": 0.1,
        "max": 0.9,
    }


@pytest.mark.parametrize(
    ("coefficient_function", "loss_function"),
    [
        (
            value_expectation_loss_coefficients,
            lambda scores, values: -sum(
                probability * value
                for probability, value in zip(
                    np.exp(scores - max(scores)) / np.exp(scores - max(scores)).sum(),
                    values,
                    strict=True,
                )
            ),
        ),
        (
            optimal_set_loss_coefficients,
            lambda scores, values: -np.log(
                (
                    np.exp(scores - max(scores)) / np.exp(scores - max(scores)).sum()
                )[values == max(values)].sum()
            ),
        ),
    ],
)
def test_value_loss_coefficients_match_finite_difference(
    coefficient_function, loss_function
) -> None:
    scores = np.asarray([0.3, -0.2, 0.7])
    values = np.asarray([0.1, 0.9, 0.9])
    expected = coefficient_function(scores, values)
    actual = []
    epsilon = 1e-6
    for index in range(len(scores)):
        plus, minus = scores.copy(), scores.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        actual.append((loss_function(plus, values) - loss_function(minus, values)) / (2 * epsilon))
    assert actual == pytest.approx(expected, abs=1e-7)
    assert sum(expected) == pytest.approx(0.0)


def test_expert_nll_coefficients_are_a_control_distribution_gradient() -> None:
    coefficients = one_hot_nll_coefficients([0.0, 1.0, -1.0], 2)
    assert sum(coefficients) == pytest.approx(0.0)
    assert coefficients[2] < 0
