import pytest
import torch
from torch import nn

from rulztrainer.lora import LoRALinear


@pytest.mark.parametrize("alpha", [4.0, 2.0])
def test_lora_forward_matches_explicit_scaled_update(alpha):
    torch.manual_seed(0)
    base = nn.Linear(8, 6)
    layer = LoRALinear(base, rank=4, alpha=alpha)
    with torch.no_grad():
        layer.lora_up.weight.normal_()
    x = torch.randn(3, 8)
    update = layer.lora_up(layer.lora_down(x))
    expected = base(x) + (update if alpha == 4.0 else update * (alpha / 4))
    assert layer.multiplier == 1.0
    assert torch.equal(layer(x), expected)
