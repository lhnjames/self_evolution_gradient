from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class DistributionRepairHead(nn.Module):
    """A tiny residual head conditioned on hidden state and the base distribution.

    The frozen random projection makes the trainable parameter count independent of
    the LLM hidden width. With 9 actions and rank 8, only 162 scalars are trained.
    """

    def __init__(self, hidden_size: int, action_count: int, rank: int = 8, seed: int = 0):
        super().__init__()
        generator = torch.Generator(device="cpu").manual_seed(seed)
        projection = torch.randn(rank, hidden_size, generator=generator) / math.sqrt(hidden_size)
        self.register_buffer("projection", projection)
        self.residual = nn.Linear(rank + action_count, action_count)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)
        self.action_count = action_count

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def forward(
        self,
        hidden: torch.Tensor,
        base_logits: torch.Tensor,
        action_mask: torch.Tensor,
        base_temperature: float = 1.0,
    ) -> torch.Tensor:
        if hidden.ndim == 1:
            hidden = hidden.unsqueeze(0)
            base_logits = base_logits.unsqueeze(0)
            action_mask = action_mask.unsqueeze(0)
        masked_base = (base_logits / base_temperature).masked_fill(~action_mask, -torch.inf)
        base_prob = torch.softmax(masked_base, dim=-1)
        normalized_hidden = F.layer_norm(hidden.float(), (hidden.shape[-1],))
        compressed_hidden = torch.tanh(F.linear(normalized_hidden, self.projection.float()))
        features = torch.cat([compressed_hidden, base_prob.float()], dim=-1)
        delta = self.residual(features)
        return (masked_base.float() + delta).masked_fill(~action_mask, -torch.inf)

