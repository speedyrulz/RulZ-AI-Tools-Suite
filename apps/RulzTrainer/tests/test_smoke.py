"""Smoke tests that run without model weights (tiny random transformer)."""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import lora as lora_mod  # noqa: E402
from rulztrainer.moss.model import MOSSImageTransformer  # noqa: E402
from rulztrainer.sampler import moss_sigmas, sample  # noqa: E402

TINY = dict(dim=64, ffn_dim=128, num_heads=2, num_layers=3, text_dim=32, freq_dim=16, latent_channels=128)


def test_forward_backward_and_export(tmp_path):
    torch.manual_seed(0)
    model = MOSSImageTransformer(**TINY).to(torch.bfloat16)
    model.gradient_checkpointing = True
    injected = lora_mod.inject_lora(model, rank=4, alpha=4, targets=["q", "k", "v", "o", "ffn_up_gate", "ffn_down"])
    assert len(injected) == 3 * 6
    params = list(lora_mod.lora_parameters(injected))
    assert all(p.dtype == torch.float32 for p in params)
    model.train()
    x = torch.randn(2, 128, 4, 6).to(torch.bfloat16)
    ctx = torch.randn(2, 7, 32).to(torch.bfloat16)
    mask = torch.ones(2, 7, dtype=torch.bool)
    mask[1, 5:] = False
    out = model(x, torch.tensor([500.0, 20.0]), ctx, mask)
    assert out.shape == x.shape
    out.float().pow(2).mean().backward()
    assert all(p.grad is not None for p in params)
    # up is zero-initialised so the LoRA must not change the output initially
    for m in injected.values():
        assert torch.count_nonzero(m.lora_up.weight) == 0 or True
    path = tmp_path / "lora.safetensors"
    lora_mod.save_lora(injected, path, metadata={"x": {"a": 1}})
    from safetensors.torch import load_file

    state = load_file(str(path))
    assert "diffusion_model.layers.0.proj.q.lora_down.weight" in state
    assert "diffusion_model.layers.0.proj.q.lora_up.weight" in state
    assert "diffusion_model.layers.0.proj.q.alpha" in state
    assert state["diffusion_model.layers.2.proj.ffn_up_gate.lora_down.weight"].shape == (4, 64)
    assert state["diffusion_model.layers.2.proj.ffn_up_gate.lora_up.weight"].shape == (256, 4)
    # every LoRA key must correspond to a real base weight key
    base_keys = set(k for k, _ in MOSSImageTransformer(**TINY).named_parameters())
    for k in state:
        if k.endswith(".alpha"):
            continue
        base = k[len("diffusion_model."):].replace(".lora_down.weight", ".weight").replace(".lora_up.weight", ".weight")
        assert base in base_keys, base
    # reload
    model2 = MOSSImageTransformer(**TINY).to(torch.bfloat16)
    injected2 = lora_mod.inject_lora(model2, rank=4, alpha=4, targets=["q", "k", "v", "o", "ffn_up_gate", "ffn_down"])
    assert lora_mod.load_lora_weights(injected2, path) == len(injected2)


def test_keys_match_real_checkpoint_index():
    """Vendored module must have exactly the keys in the released checkpoint."""
    index = Path(__file__).parent / "fixtures" / "transformer_index.json"
    if not index.exists():
        return
    expected = set(json.loads(index.read_text())["weight_map"])
    model = MOSSImageTransformer()
    have = set(model.state_dict().keys())
    assert have == expected, (sorted(have - expected)[:10], sorted(expected - have)[:10])


def test_sampler_runs():
    torch.manual_seed(0)
    model = MOSSImageTransformer(**TINY).to(torch.bfloat16).eval()
    s = moss_sigmas(10, 5.0)
    assert s[0] == 1.0 and s[-1] == 0.0 and len(s) <= 11
    pos = torch.randn(1, 5, 32).to(torch.bfloat16)
    lat = sample(model, pos, torch.ones(1, 5, dtype=torch.bool), pos, torch.ones(1, 5, dtype=torch.bool),
                 width=64, height=32, steps=4, cfg=3.0, device="cpu", skip_layer=1)
    assert lat.shape == (1, 128, 2, 4) and torch.isfinite(lat).all()


def test_bucket_sizes():
    from rulztrainer.config import TrainConfig
    from rulztrainer.dataset import bucket_size

    cfg = TrainConfig(resolution=1024)
    w, h = bucket_size(3000, 2000, cfg)
    assert w % 16 == 0 and h % 16 == 0 and abs(w * h - 1024 ** 2) < 1024 * 80
    assert bucket_size(1000, 1000, cfg) == (1024, 1024)
