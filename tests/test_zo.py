import torch
from torch import nn

from self_evolve.zo import ZerothOrderAdam


def test_zero_order_optimizer_reduces_black_box_loss():
    torch.manual_seed(0)
    module = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        module.weight.zero_()
    optimizer = ZerothOrderAdam(
        module, learning_rate=0.1, sigma=0.05, directions=32, two_sided=True
    )

    def loss_fn():
        return (module.weight.reshape(()) - 2.0).pow(2)

    initial = float(loss_fn().detach())
    for _ in range(20):
        optimizer.step(loss_fn)
    assert float(loss_fn().detach()) < initial * 0.2
