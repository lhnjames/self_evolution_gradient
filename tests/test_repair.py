import torch

from self_evolve.repair import repair_distribution


def test_verifier_can_repair_wrong_top1():
    logits = torch.tensor([2.0, 1.5, -1.0])
    scores = torch.tensor([0.0, 1.0, 0.0])
    mask = torch.tensor([True, True, True])
    result = repair_distribution(logits, scores, mask, beta=4.0)
    assert result.proposed_action == 0
    assert result.repaired_action == 1
    assert not result.accepted


def test_top_k_limits_which_actions_are_verified():
    logits = torch.tensor([3.0, 2.0, -10.0])
    scores = torch.tensor([0.0, 0.0, 1.0])
    mask = torch.tensor([True, True, True])
    result = repair_distribution(logits, scores, mask, beta=20.0, top_k=2)
    assert result.repaired_action != 2
    assert not result.verified_mask[2]

