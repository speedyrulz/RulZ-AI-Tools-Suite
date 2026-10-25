"""Full fine-tune (finetune.py): components and windows, swapping a window in and out of a quantized model, deltas
against the original base, extraction to a LoRA that reproduces the delta, delta files, the CLI, and the trainer's
rotation / save cadence on a tiny stand-in model (CPU)."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import finetune as FT  # noqa: E402
from rulztrainer import lora as L  # noqa: E402
from rulztrainer import quant as Q  # noqa: E402
from rulztrainer.config import TrainConfig  # noqa: E402
from rulztrainer.trainer import Trainer, build_scheduler  # noqa: E402


class Block(nn.Module):
    def __init__(self, d, big):
        super().__init__()
        self.q = nn.Linear(d, d, bias=False)
        self.ffn = nn.Linear(d, big, bias=True)

    def forward(self, x):
        return x + self.ffn(self.q(x))[..., : x.shape[-1]]


class Tiny(nn.Module):
    def __init__(self, d=32, big=64, n=3):
        super().__init__()
        torch.manual_seed(0)
        self.text = nn.Linear(d, d)
        self.layers = nn.ModuleList(Block(d, big) for _ in range(n))

    def forward(self, x):
        x = self.text(x)
        for b in self.layers:
            x = b(x)
        return x


def _names(model):
    return [n for n, m in model.named_modules() if isinstance(m, (nn.Linear, Q.QuantLinear))]


def test_components_and_windows():
    assert FT.component_key("layers.7.proj.q") == ("layers.*.proj.q", 7)
    assert FT.component_key("text_embedder.blocks.1.qkv") == ("text_embedder.blocks.*.qkv", 1)
    assert FT.component_key("text_fusion.proj") == ("text_fusion.proj", None)
    slots = [FT.Slot(f"layers.{i}.q", "layers.*.q", i, 100, (10, 10), "bf16", None, torch.bfloat16) for i in range(4)]
    slots += [FT.Slot(f"layers.{i}.ffn", "layers.*.ffn", i, 300, (30, 10), "bf16", None, torch.bfloat16)
              for i in range(4)]
    slots += [FT.Slot("head", "head", None, 50, (5, 10), "bf16", None, torch.bfloat16)]
    # budget 1000 bytes at 1 byte per weight: q (400) fits whole; ffn (1200) is cut by blocks (3 + 1); head joins
    wins = FT.plan_windows(slots, 1000, 1)
    assert wins == [["layers.0.q", "layers.1.q", "layers.2.q", "layers.3.q"],
                    ["layers.0.ffn", "layers.1.ffn", "layers.2.ffn"], ["layers.3.ffn"], ["head"]]
    assert FT.describe_window(wins[1]) == "layers.*.ffn blocks 0-2"
    assert FT.describe_window(wins[0]) == "layers.*.q blocks 0-3"
    assert FT.describe_window(["head"]) == "head"
    # everything in one window when it all fits; a layer bigger than the budget is refused with advice
    assert len(FT.plan_windows(slots, 10**6, 1)) == 1
    with pytest.raises(ValueError, match="alone needs"):
        FT.plan_windows(slots, 200, 1)
    assert FT.plan_budget(10 * 2**30, 3.0, 0) == 7 * 2**30 and FT.plan_budget(10 * 2**30, 3.0, 2.5) == int(2.5 * 2**30)
    assert FT.default_reserve_gb(1.0) == 6.0 and FT.epoch_steps(10, 4, 1.0) == 3 and FT.epoch_steps(10, 4, 0) == 1


def _quantized_tiny():
    m = Tiny()
    Q.quantize_module_linears(m.layers, "int8", min_dim=32)  # the blocks' Linears quantized; text stays a Linear
    for n, mod in m.named_modules():
        if isinstance(mod, Q.QuantLinear):
            mod.int8_matmul = False  # the CPU path: dequantize per forward
    return m


def test_window_round_trip_and_deltas():
    model = _quantized_tiny()
    names = _names(model)
    ft = FT.FineTuner(model, names, "cpu")
    assert {s.fmt for s in ft.slots.values()} == {"int8", "bf16"}
    ft.plan(10**9, 10)
    assert len(ft.windows) == 1
    x = torch.randn(2, 32)
    with torch.no_grad():
        before = model(x)
    params = ft.activate(0)
    assert all(isinstance(FT._inner(model, n), FT.TunedLinear) for n in names) and len(params) == len(names)
    with torch.no_grad():
        same = model(x)
    assert torch.allclose(before, same, atol=1e-5)  # the fp32 window starts at the dequantized base: no change
    with torch.no_grad():
        for p in params:
            p.add_(0.01 * torch.randn_like(p))
    with torch.no_grad():
        trained = model(x)
    live_delta = ft.delta_of(ft.slots[names[0]]).clone()
    ft.deactivate()
    assert ft.cycles == 1 and all(not isinstance(FT._inner(model, n), FT.TunedLinear) for n in names)
    slot = ft.slots[names[0]]
    assert slot.delta is not None and slot.delta.dtype == torch.bfloat16
    assert torch.allclose(slot.delta.float(), live_delta, atol=2e-3)  # the delta's own bf16 precision
    with torch.no_grad():
        after = model(x)
    assert not torch.allclose(after, before, atol=1e-3)  # the trunk carries the update (re-quantized)
    assert torch.allclose(after, trained, atol=0.1)  # within re-quantization error
    # a second activation starts from base + delta, and the base is still the original
    params = ft.activate(0)
    w0 = params[0].detach()
    assert torch.allclose(w0, ft.base_weight(slot) + slot.delta.float(), atol=1e-6)
    ft.deactivate()
    assert torch.allclose(ft.slots[names[0]].delta.float(), slot.delta.float(), atol=2e-3)


def test_extraction_reproduces_the_delta(tmp_path):
    model = _quantized_tiny()
    names = _names(model)
    ft = FT.FineTuner(model, names, "cpu")
    ft.plan(10**9, 10)
    params = ft.activate(0)
    with torch.no_grad():
        u, v = torch.randn(params[0].shape[0], 2), torch.randn(2, params[0].shape[1])
        params[0].add_(u @ v * 0.05)  # a rank-2 update on the first layer only
    ft.deactivate()
    stubs = ft.extract(2)
    assert list(stubs) == [names[0]]  # untouched layers are left out
    parts = stubs[names[0]].adapter_params()
    delta = ft.slots[names[0]].delta.float()
    assert torch.allclose(parts["lora_up"] @ parts["lora_down"], delta, atol=5e-3)  # exact at rank 2
    down, up = FT.lowrank_factors(delta, 1)
    assert (up @ down - delta).norm() < delta.norm()  # rank 1: the best rank-1 approximation
    # saved as a LoRA in the trainer's own format and read back by its loader
    path = tmp_path / "x.safetensors"
    L.save_lora(stubs, path, metadata={"ss_network_dim": "2"}, fmt="comfy")
    state = __import__("safetensors.torch", fromlist=["load_file"]).load_file(str(path))
    assert f"diffusion_model.{names[0]}.lora_down.weight" in state and float(state[f"diffusion_model.{names[0]}.alpha"]) == 2.0
    # the delta file: saved, extracted by the CLI, and continued from
    dpath = tmp_path / "job_finetune.safetensors"
    ft.save_delta(dpath, {"lora_format": "comfy", "ss_network_dim": "2"})
    from rulztrainer.extract_finetune import main

    assert main([str(dpath), "--rank", "2", "4", "--device", "cpu"]) == 0
    assert (tmp_path / "job_r2.safetensors").is_file() and (tmp_path / "job_r4.safetensors").is_file()
    fresh = _quantized_tiny()
    ft2 = FT.FineTuner(fresh, names, "cpu")
    assert ft2.load_delta(dpath) == 1
    assert torch.allclose(ft2.slots[names[0]].delta.float(), delta, atol=1e-6)
    x = torch.randn(2, 32)
    with torch.no_grad():
        assert torch.allclose(fresh(x), model(x), atol=1e-5)  # the same re-quantized trunk


def test_scheduler_offset():
    cfg = TrainConfig(steps=100, warmup_steps=10, lr_scheduler="linear", learning_rate=1.0)
    opt = torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr=1.0)
    sched = build_scheduler(cfg, opt, offset=50)
    assert sched.get_last_lr()[0] == pytest.approx(1 - 40 / 90)  # step 50 of the run, after the warm-up


def test_finetune_validation():
    base = dict(finetune=True, learning_rate=1e-5)
    TrainConfig(**base).validate()
    with pytest.raises(ValueError, match="race"):
        TrainConfig(**base, race=True, race_configs=[{"name": "a"}, {"name": "b"}]).validate()
    with pytest.raises(ValueError, match="data parallel"):
        TrainConfig(**base, multi_gpu="data_parallel", devices="cuda:0,cuda:1").validate()
    with pytest.raises(ValueError, match="resume_from"):
        TrainConfig(**base, resume_from="x.safetensors").validate()
    with pytest.raises(ValueError, match="region_prior"):
        TrainConfig(**base, region_prior_weight=0.5).validate()
    with pytest.raises(ValueError, match="finetune_extract_ranks"):
        TrainConfig(**base, finetune_extract_ranks=[]).validate()


def test_trainer_rotation_and_save_cadence(tmp_path, monkeypatch):
    """The trainer's fine-tune helpers on a stand-in: windows rotate every N steps, saves wait for a cycle's end."""
    model = _quantized_tiny()
    names = _names(model)
    tr = Trainer.__new__(Trainer)
    tr.cfg = TrainConfig(job_name="ft", output_dir=str(tmp_path), finetune=True, optimizer="adamw", learning_rate=1e-5,
                         batch_size=1, gradient_accumulation=1, finetune_rotate_epochs=1.0, finetune_window_gb=0.0001,
                         finetune_extract_ranks=[2, 4], finetune_save_delta=True, lora_save_dtype="bf16")
    tr.job_dir = tmp_path
    tr.devices = ["cpu"]
    tr.arch = SimpleNamespace(modelspec="tiny", lora_format="comfy")
    tr.status = SimpleNamespace(update=lambda **k: None, write=lambda: None, data={"saved": []})
    tr.ft = FT.FineTuner(model, names, "cpu")
    tr.ft_rotated_at = -1
    items = [SimpleNamespace(repeats=1)] * 4  # 4 images per epoch, 1 per step: a window every 4 steps
    params, opt, sched = tr._finetune_setup(tr.cfg, items)
    n = len(tr.ft.windows)
    assert n >= 2 and tr.ft_rotate_steps == 4 and tr.ft_window == 0 and len(params) == len(tr.ft.windows[0])
    assert not tr._finetune_cycle_done(4) if n > 1 else True
    assert tr._finetune_cycle_done(4 * n) and not tr._finetune_cycle_done(4 * n - 1)
    with torch.no_grad():
        params[0].add_(0.01)
    params, opt, sched = tr._finetune_rotate(tr.cfg, 4)
    assert tr.ft_window == 1 and tr.ft.active == 1 and tr.ft_rotated_at == 4
    assert tr.ft.slots[tr.ft.windows[0][0]].delta is not None  # the first window's update is kept
    for _ in range(n - 1):
        params, opt, sched = tr._finetune_rotate(tr.cfg, 8)
    assert tr.ft.cycles == 1 and tr.ft_window == 0
    first = tr.save({}, 4 * n)
    assert first == tmp_path / f"ft_{4 * n:06d}.safetensors" and (tmp_path / f"ft_{4 * n:06d}_r4.safetensors").is_file()
    assert (tmp_path / "ft_finetune.safetensors").is_file()
    assert tr.save({}, 4 * n, final=True) == tmp_path / "ft.safetensors"
