from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters


@dataclass
class ZOStepStats:
    baseline_loss: float
    mean_probe_loss: float
    gradient_norm: float


class ZerothOrderAdam:
    """SPSA/random-direction gradient estimates applied through Adam."""

    def __init__(
        self,
        module: nn.Module,
        learning_rate: float,
        sigma: float,
        directions: int,
        two_sided: bool = True,
    ):
        self.module = module
        self.parameters = [p for p in module.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(self.parameters, lr=learning_rate)
        self.sigma = sigma
        self.directions = directions
        self.two_sided = two_sided

    @torch.no_grad()
    def step(self, loss_fn: Callable[[], torch.Tensor]) -> ZOStepStats:
        theta = parameters_to_vector(self.parameters).detach().clone()
        gradient = torch.zeros_like(theta)
        probe_losses: list[float] = []
        baseline = float(loss_fn().item())
        for _ in range(self.directions):
            direction = torch.randn_like(theta)
            vector_to_parameters(theta + self.sigma * direction, self.parameters)
            plus = float(loss_fn().item())
            probe_losses.append(plus)
            if self.two_sided:
                vector_to_parameters(theta - self.sigma * direction, self.parameters)
                minus = float(loss_fn().item())
                probe_losses.append(minus)
                coefficient = (plus - minus) / (2.0 * self.sigma)
            else:
                coefficient = (plus - baseline) / self.sigma
            gradient.add_(coefficient * direction / self.directions)
        vector_to_parameters(theta, self.parameters)
        self.optimizer.zero_grad(set_to_none=True)
        cursor = 0
        for parameter in self.parameters:
            count = parameter.numel()
            parameter.grad = gradient[cursor : cursor + count].view_as(parameter).clone()
            cursor += count
        self.optimizer.step()
        return ZOStepStats(
            baseline_loss=baseline,
            mean_probe_loss=sum(probe_losses) / len(probe_losses),
            gradient_norm=float(gradient.norm().item()),
        )

