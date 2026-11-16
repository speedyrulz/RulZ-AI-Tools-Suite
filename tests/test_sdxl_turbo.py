"""SDXL's DMD2 turbo LoRA: kohya names resolved through the model's `unet.` modules, Conv2d entries applied by forward
hooks while sampling, the LCM sampler picked while a turbo LoRA is attached, and the SDXL downloads."""
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import comfy_files as CF  # noqa: E402
from rulztrainer import lora as L  # noqa: E402
from rulztrainer.config import ARCH_SAMPLE_DEFAULTS, ARCH_TURBO, SDXL_SINGLE_FILE, TrainConfig  # noqa: E402
from rulztrainer.sdxl import sampler as S  # noqa: E402


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 6, 3, padding=1)
        self.shortcut = nn.Conv2d(4, 6, 1)
        self.to_q = nn.Linear(6, 6)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_blocks = nn.ModuleList([Block()])


class Facade(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = Net()


def _file(tmp_path, rank=2):
    g = torch.Generator().manual_seed(1)
    r = lambda *s: torch.randn(*s, generator=g)  # noqa: E731
    state = {
        "lora_unet_down_blocks_0_conv1.lora_down.weight": r(rank, 4, 3, 3),
        "lora_unet_down_blocks_0_conv1.lora_up.weight": r(6, rank, 1, 1),
        "lora_unet_down_blocks_0_conv1.alpha": torch.tensor(float(rank)),
        "lora_unet_down_blocks_0_shortcut.lora_down.weight": r(rank, 4, 1, 1),
        "lora_unet_down_blocks_0_shortcut.lora_up.weight": r(6, rank, 1, 1),
        "lora_unet_down_blocks_0_shortcut.alpha": torch.tensor(float(rank)),
        "lora_unet_down_blocks_0_to_q.lora_down.weight": r(rank, 6),
        "lora_unet_down_blocks_0_to_q.lora_up.weight": r(6, rank),
        "lora_unet_down_blocks_0_to_q.alpha": torch.tensor(float(rank)),
        "lora_unet_down_blocks_0_missing.lora_down.weight": r(rank, 6),
        "lora_unet_down_blocks_0_missing.lora_up.weight": r(6, rank),
    }
    path = tmp_path / "dmd2.safetensors"
    save_file(state, str(path))
    return path, state


def test_kohya_extra_lora_applies_conv_and_linear_entries(tmp_path):
    path, st = _file(tmp_path)
    model = Facade().eval()
    x = torch.randn(1, 4, 8, 8)
    base_conv = model.unet.down_blocks[0].conv1(x)
    base_short = model.unet.down_blocks[0].shortcut(x)
    n = L.attach_extra_lora(model, str(path), 0.5)
    assert n == 3  # two convs and a linear; the unknown name is missed
    blk = model.unet.down_blocks[0]
    assert torch.equal(blk.conv1(x), base_conv)  # extras start switched off

    def manual(conv_in, key, base, kernel):
        down, up = st[f"{key}.lora_down.weight"], st[f"{key}.lora_up.weight"]
        h = torch.nn.functional.conv2d(conv_in, down, padding=kernel // 2)
        return base + torch.nn.functional.conv2d(h, up) * (1.0 * 0.5)  # alpha / rank = 1, strength 0.5

    for _ in range(2):  # switching on and off twice is stable (no stacked hooks)
        L.set_extras_enabled(model, True)
        assert L.extras_active(model)
        assert torch.allclose(blk.conv1(x), manual(x, "lora_unet_down_blocks_0_conv1", base_conv, 3), atol=1e-5)
        assert torch.allclose(blk.shortcut(x), manual(x, "lora_unet_down_blocks_0_shortcut", base_short, 1), atol=1e-5)
        assert len(model.__dict__["_rz_conv_hooks"]) == 2
        L.set_extras_enabled(model, False)
        assert torch.equal(blk.conv1(x), base_conv) and torch.equal(blk.shortcut(x), base_short)
        assert not model.__dict__.get("_rz_conv_hooks") and not L.extras_active(model)
    # the linear entry went through the usual extras
    assert isinstance(blk.to_q, L.LoRALinear) and blk.to_q.extras


def test_two_files_add_up_and_wrong_conv_shapes_are_missed(tmp_path):
    path, _ = _file(tmp_path)
    model = Facade().eval()
    one = Facade()
    one.load_state_dict(model.state_dict())
    x = torch.randn(1, 4, 8, 8)
    base = model.unet.down_blocks[0].conv1(x)
    L.attach_extra_lora(model, str(path), 1.0)
    L.attach_extra_lora(model, str(path), 1.0)
    L.set_extras_enabled(model, True)
    L.attach_extra_lora(one, str(path), 1.0)
    L.set_extras_enabled(one, True)
    d_two = model.unet.down_blocks[0].conv1(x) - base
    d_one = one.unet.down_blocks[0].conv1(x) - base
    assert torch.allclose(d_two, 2 * d_one, atol=1e-4)
    bad = tmp_path / "bad.safetensors"
    save_file({"lora_unet_down_blocks_0_conv1.lora_down.weight": torch.zeros(2, 5, 3, 3),  # 5 in-channels: no fit
               "lora_unet_down_blocks_0_conv1.lora_up.weight": torch.zeros(6, 2, 1, 1)}, str(bad))
    assert L.attach_extra_lora(Facade(), str(bad), 1.0) == 0


def test_ambiguous_kohya_names_are_refused():
    m = nn.Module()
    m.unet = nn.Module()
    m.unet.a_b = nn.Linear(2, 2)
    m.unet.a = nn.Module()
    m.unet.a.b = nn.Linear(2, 2)
    with pytest.raises(ValueError, match="Ambiguous"):
        L._kohya_name_map(m)


# ---------------------------------------------------------------------------------------------------- sampler
class Exact(nn.Module):
    """Predicts the exact flow velocity (eps - x0) for a known x0 from the latents it is given: eps is read off the
    state, so the LCM jump lands on x0 whatever the noise is."""
    latent_channels = 4

    def __init__(self, x0):
        super().__init__()
        self.x0 = x0

    def forward(self, x, t, embeds, mask):
        sigma = float(t) / 1000.0
        x = x.float()
        eps = (x - (1 - sigma) * self.x0) / sigma
        return (eps - self.x0).to(torch.bfloat16)


def _run(model, **kw):
    e = torch.zeros(1, 3, 4)
    return S.sample(model, e, None, None, None, width=32, height=32, steps=kw.pop("steps", 4), cfg=1.0, seed=3,
                    device="cpu", **kw)


def test_lcm_lands_on_x0_and_differs_from_euler_per_step():
    x0 = torch.full((1, 4, 2, 2), 0.7)
    lcm = _run(Exact(x0), sampler="lcm")
    assert torch.allclose(lcm, x0, atol=2e-2)  # bf16 model input
    # Euler with the same exact model also converges, but the paths differ before the end: one step each
    one_lcm, one_euler = _run(Exact(x0), steps=1, sampler="lcm"), _run(Exact(x0), steps=1, sampler="euler")
    assert torch.allclose(one_lcm, x0, atol=2e-2) and torch.allclose(one_euler, x0, atol=2e-2)


def test_lcm_renoises_with_fresh_noise_from_the_seeded_generator():
    class Zero(nn.Module):  # velocity 0: x0 = latents, so the path is fixed by the re-noising alone
        latent_channels = 4

        def forward(self, x, t, embeds, mask):
            return torch.zeros_like(x)

    a, b = _run(Zero(), sampler="lcm"), _run(Zero(), sampler="lcm")
    assert torch.equal(a, b)  # seeded
    assert not torch.allclose(a, _run(Zero(), sampler="euler"))  # euler with zero velocity keeps the start noise


def test_sampler_default_follows_the_turbo_lora_and_others_are_refused():
    x0 = torch.full((1, 4, 2, 2), 0.7)
    plain, turbo = Exact(x0), Exact(x0)
    turbo.turbo_lora_attached = True
    assert torch.equal(_run(turbo), _run(Exact(x0), sampler="lcm"))
    assert torch.equal(_run(plain), _run(Exact(x0), sampler="euler"))
    with pytest.raises(ValueError, match="euler or lcm"):
        _run(plain, sampler="heun")


def test_sdxl_arch_passes_the_sampler_through():
    from rulztrainer.archs import arch_by_name

    x0 = torch.full((1, 4, 2, 2), 0.7)
    out = arch_by_name("sdxl").sample(Exact(x0), torch.zeros(1, 3, 4), None, None, None, width=32, height=32, steps=2,
                                      cfg=1.0, seed=1, device="cpu", skip_layer=22, sampler="lcm")
    assert torch.allclose(out, x0, atol=2e-2)


# ---------------------------------------------------------------------------------------------------- config
def test_sdxl_turbo_defaults_and_downloads():
    t = ARCH_TURBO["sdxl"]
    assert (t["repo"], t["file"], t["steps"], t["cfg"]) == ("tianweiy/DMD2", "dmd2_sdxl_4step_lora.safetensors", 8, 1.0)
    assert ARCH_SAMPLE_DEFAULTS["sdxl"] == (8, 1.0)
    assert SDXL_SINGLE_FILE == ("stabilityai/stable-diffusion-xl-base-1.0", "sd_xl_base_1.0_0.9vae.safetensors")
    plan = CF.planned_downloads(TrainConfig(model_arch="sdxl", model_source="files"))
    assert plan == [("transformer_file", *SDXL_SINGLE_FILE), ("turbo_lora", t["repo"], t["file"])]
    hf = CF.planned_downloads(TrainConfig(model_arch="sdxl", model_source="hf",
                                          model_path="stabilityai/stable-diffusion-xl-base-1.0"))
    assert hf == [("turbo_lora", t["repo"], t["file"])]
    TrainConfig(model_arch="sdxl", model_path="stabilityai/stable-diffusion-xl-base-1.0",
                turbo_lora="dmd2.safetensors").validate()
