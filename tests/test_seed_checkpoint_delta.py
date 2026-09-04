import math

import numpy as np

from self_evolve.seed_checkpoint_delta import minimal_top1_repair


def test_minimal_top1_repair_reaches_boundary_with_expected_norm() -> None:
    scores = np.asarray([2.1, 1.8, 0.5, 1.2])
    repair = minimal_top1_repair(scores, gold_index=1)
    assert repair.competitor_index == 0
    assert math.isclose(repair.gap, 0.3)
    assert np.allclose(repair.delta, [-0.15, 0.15, 0.0, 0.0])
    assert math.isclose(repair.l2_norm, 0.3 / math.sqrt(2.0))
    repaired = scores + repair.delta
    assert math.isclose(repaired[0], repaired[1])


def test_minimal_top1_repair_is_zero_when_gold_is_already_top1() -> None:
    repair = minimal_top1_repair(np.asarray([0.2, 1.0, -0.5]), gold_index=1)
    assert repair.gap == 0.0
    assert repair.l2_norm == 0.0
    assert np.allclose(repair.delta, 0.0)

