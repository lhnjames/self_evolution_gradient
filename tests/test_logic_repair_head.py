import torch

from self_evolve.logic_repair_head import ConvexSkillGate, SignedLogicRepairHead, StageCalibrationHead


def test_repair_heads_support_variable_candidate_count():
    plain = torch.tensor([-2.0, -1.0, -3.0])
    skill = torch.tensor([-1.0, -2.0, -2.5])
    verbs = torch.tensor([1, 2, 1])
    lengths = torch.tensor([2.0, 3.0, 1.0])
    for head in (ConvexSkillGate(), StageCalibrationHead(3), SignedLogicRepairHead(3)):
        logits, alpha = head(plain, skill, verbs, lengths)
        assert logits.shape == (3,)
        assert alpha.ndim == 0
        assert torch.isfinite(logits).all()
