import numpy as np
import pytest

from self_evolve.value_writeback import (
    candidate_distribution_metrics,
    relationship_bucket,
)


@pytest.mark.parametrize(
    ("task", "verb", "expected"),
    [
        ("task_a", "go", "same_task_same_verb"),
        ("task_a", "take", "same_task_different_verb"),
        ("task_b", "go", "different_task_same_verb"),
        ("task_b", "take", "different_task_different_verb"),
    ],
)
def test_relationship_bucket(task, verb, expected):
    source = {"task_type": "task_a", "action_verb": "go"}
    assert relationship_bucket(source, {"task_type": task, "action_verb": verb}) == expected


def test_candidate_distribution_metrics_detect_value_improvement():
    result = candidate_distribution_metrics(
        baseline_scores=[0.0, 0.0],
        updated_scores=[-1.0, 1.0],
        values=[0.0, 1.0],
        expert_index=1,
    )
    assert result["expected_value_delta"] > 0
    assert result["optimal_mass_delta"] == pytest.approx(result["expected_value_delta"])
    assert result["expert_probability_delta"] > 0
    assert result["kl_baseline_to_updated"] > 0


def test_candidate_distribution_metrics_identity_is_zero():
    scores = np.array([-2.0, 0.5, 1.0])
    result = candidate_distribution_metrics(scores, scores, [0.1, 0.3, 0.2], 0)
    assert result["kl_baseline_to_updated"] == pytest.approx(0.0, abs=1e-15)
    assert result["total_variation"] == 0.0
    assert result["expected_value_delta"] == 0.0
