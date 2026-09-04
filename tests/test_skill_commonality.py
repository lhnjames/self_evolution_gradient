import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "analyze_skill_commonality.py"
SPEC = importlib.util.spec_from_file_location("analyze_skill_commonality", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_aligned_skill_direction_has_positive_first_order_gain():
    row = {
        "normalized_scores": {
            "plain": [0.0, 0.0],
            "evolved_skill": [-1.0, 1.0],
            "mismatched_skill": [1.0, -1.0],
        },
        "gold_index": 1,
        "action_verb": "take",
        "admissible_actions": ["go to shelf 1", "take mug 1 from shelf 1"],
    }
    MODULE.add_effects([row], {"go": 0, "take": 1})
    assert row["effect"]["first_order_gain"] > 0
    assert row["effect"]["alignment_cosine"] > 0
    assert row["effect"]["mismatch_first_order_gain"] < 0
    assert row["effect"]["gold_logp_gain"] > 0


def test_pairwise_commonality_excludes_same_episode_pairs():
    base = {
        "task_type": "pick",
        "action_verb": "take",
        "schema_shift": np.asarray([1.0, -1.0]),
    }
    rows = [
        {**base, "episode_key": "a"},
        {**base, "episode_key": "a"},
        {**base, "episode_key": "b"},
    ]
    result = MODULE.pairwise_commonality(rows, "schema_shift")
    assert result["same_task__same_stage"]["pairs"] == 2
    assert result["same_task__same_stage"]["mean_cosine"] == pytest.approx(1.0)
