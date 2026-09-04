from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RepairResult:
    proposal_prob: torch.Tensor
    repaired_prob: torch.Tensor
    proposed_action: int
    repaired_action: int
    accepted: bool
    verified_mask: torch.Tensor


def repair_distribution(
    logits: torch.Tensor,
    verifier_scores: torch.Tensor,
    action_mask: torch.Tensor,
    beta: float = 4.0,
    top_k: int = 0,
) -> RepairResult:
    """Reweight a proposal distribution with action-level verifier scores.

    q(a|s) is proportional to p(a|s) * exp(beta * V(s,a)). A positive
    ``top_k`` verifies only the top-k proposals, matching speculative-action use.
    ``top_k=0`` verifies every available action and is the controlled oracle mode.
    """

    if logits.ndim != 1:
        raise ValueError("repair_distribution expects one unbatched distribution")
    proposal_prob = torch.softmax(logits.masked_fill(~action_mask, -torch.inf), dim=-1)
    available = int(action_mask.sum().item())
    verified_mask = action_mask.clone()
    if top_k > 0 and top_k < available:
        indices = torch.topk(proposal_prob, k=top_k).indices
        verified_mask = torch.zeros_like(action_mask)
        verified_mask[indices] = True
    applied_scores = torch.where(verified_mask, verifier_scores, torch.zeros_like(verifier_scores))
    repaired_logits = torch.log(proposal_prob.clamp_min(1e-12)) + beta * applied_scores
    repaired_logits = repaired_logits.masked_fill(~action_mask, -torch.inf)
    repaired_prob = torch.softmax(repaired_logits, dim=-1)
    proposed = int(proposal_prob.argmax().item())
    repaired = int(repaired_prob.argmax().item())
    return RepairResult(
        proposal_prob=proposal_prob,
        repaired_prob=repaired_prob,
        proposed_action=proposed,
        repaired_action=repaired,
        accepted=proposed == repaired,
        verified_mask=verified_mask,
    )

