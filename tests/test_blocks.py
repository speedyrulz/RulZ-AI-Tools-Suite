import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import lora as L  # noqa: E402
from rulztrainer.moss.model import MOSSImageTransformer  # noqa: E402

TINY = dict(dim=64, ffn_dim=128, num_heads=2, num_layers=6, text_dim=32, freq_dim=16, latent_channels=128)


def test_parse_and_format_blocks():
    assert L.parse_blocks("", 42) is None and L.parse_blocks("all", 42) is None
    assert L.parse_blocks("21-41", 42) == list(range(21, 42))
    assert L.parse_blocks("10-12, 40, 41-39", 42) == [10, 11, 12, 39, 40, 41]
    assert L.parse_blocks("even", 6) == [0, 2, 4] and L.parse_blocks("odd", 6) == [1, 3, 5]
    assert L.parse_blocks("50-60", 42) == [41 - 0] or L.parse_blocks("50-60", 42) == []  # clamped to range
    assert L.parse_blocks("none", 42) == []
    assert L.format_blocks([0, 1, 2, 5, 8, 9]) == "0-2,5,8-9" and L.format_blocks([]) == "none"


def test_inject_with_block_subset_and_backward_skips_early_blocks():
    torch.manual_seed(0)
    model = MOSSImageTransformer(**TINY).to(torch.bfloat16)
    targets = ["q", "k", "v", "o", "ffn_up_gate", "ffn_down"]
    inj = L.inject_lora(model, 4, 4, targets, blocks=[3, 4, 5])
    assert len(inj) == 3 * 6
    assert all(int(n.split(".")[1]) >= 3 for n in inj)
    # layers 0-2 untouched, so the backward never has to pass through them; verify the forward still works
    model.train()
    x = torch.randn(1, 128, 4, 4).to(torch.bfloat16)
    c = torch.randn(1, 5, 32).to(torch.bfloat16)
    out = model(x, torch.tensor([300.0]), c)
    out.float().pow(2).mean().backward()
    assert all(p.grad is not None for p in L.lora_parameters(inj))
    # export only contains the selected blocks
    sd = L.export_state_dict(inj)
    assert all(int(k.split(".")[2]) >= 3 for k in sd)
    # the full-model LoRA still loads into a subset injection (extra keys ignored) and vice versa
    model2 = MOSSImageTransformer(**TINY).to(torch.bfloat16)
    inj_all = L.inject_lora(model2, 4, 4, targets)
    assert len(inj_all) == 6 * 6
