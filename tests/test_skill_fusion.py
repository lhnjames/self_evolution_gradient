import torch

from self_evolve.skill_evolution import categorical_kl, trust_region_fusion


def test_trust_region_respects_kl_and_improves_verified_action():
    plain_logits = torch.tensor([1.0, 0.5, -0.2, -1.0, -2.0, 0.0, 0.0, 0.0, 0.0])
    skill_logits = torch.tensor([0.5, 1.2, -0.3, -1.0, -2.0, 0.0, 0.0, 0.0, 0.0])
    values = torch.zeros(9)
    values[1] = 1.0
    mask = torch.zeros(9, dtype=torch.bool)
    mask[:5] = True
    q, diagnostics = trust_region_fusion(
        plain_logits, skill_logits, values, mask, temperature=1.0,
        skill_weight=1.0, verifier_weight=2.0, residual_clip=2.0,
        max_kl=0.1, require_positive_skill_gain=True,
    )
    p = torch.zeros(9)
    p[mask] = torch.softmax(plain_logits[mask], dim=-1)
    assert categorical_kl(q, p, mask).item() <= 0.10001
    assert q[1] > p[1]
    assert diagnostics["skill_gate"] == 1.0


def test_harmful_skill_is_gated_without_verifier_shift():
    plain_logits = torch.tensor([2.0, 0.0, -1.0, -2.0, -3.0, 0.0, 0.0, 0.0, 0.0])
    skill_logits = torch.tensor([0.0, 2.0, -1.0, -2.0, -3.0, 0.0, 0.0, 0.0, 0.0])
    values = torch.zeros(9)
    values[0] = 1.0
    mask = torch.zeros(9, dtype=torch.bool)
    mask[:5] = True
    q, diagnostics = trust_region_fusion(
        plain_logits, skill_logits, values, mask, temperature=1.0,
        skill_weight=1.0, verifier_weight=0.0, residual_clip=2.0,
        max_kl=0.1, require_positive_skill_gain=True,
    )
    p = torch.zeros(9)
    p[mask] = torch.softmax(plain_logits[mask], dim=-1)
    assert torch.allclose(q, p)
    assert diagnostics["skill_gate"] == 0.0
