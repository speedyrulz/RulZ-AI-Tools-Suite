"""SDXL in the trainer's registry, config, noise levels, loss weights and LoRA layer choice (CPU only). The family's
own package (rulztrainer.sdxl) is tested in test_sdxl.py; this covers what the trainer does around it."""
import dataclasses
import json
import math
import types

import pytest
import torch

from rulztrainer import archs
from rulztrainer.archs import Arch, arch_by_name, arch_from_transformer_file, detect_arch, get_arch
from rulztrainer.config import TrainConfig, family_named_in
from rulztrainer.trainer import loss_weight, sample_sigmas

SDXL_HF = "stabilityai/stable-diffusion-xl-base-1.0"


def sdxl_cfg(**kw) -> TrainConfig:
    kw.setdefault("model_arch", "sdxl")
    kw.setdefault("model_source", "hf")
    kw.setdefault("model_path", SDXL_HF)
    kw.setdefault("timestep_sampling", "reference")
    return TrainConfig(**kw)


def textbook(zero_terminal_snr: bool = False):
    """alphabar and sigma = b / (a + b) of the scaled-linear 1000-step DDPM schedule, computed here from scratch."""
    betas = torch.linspace(0.00085 ** 0.5, 0.012 ** 0.5, 1000, dtype=torch.float64) ** 2
    ab = torch.cumprod(1.0 - betas, dim=0)
    if zero_terminal_snr:  # Lin et al.: shift and scale sqrt(alphabar) so the last step is pure noise
        a = ab.sqrt()
        a0, aT = a[0].clone(), a[-1].clone()
        a = (a - aT) * a0 / (a0 - aT)
        ab = a ** 2
    a, b = ab.sqrt(), (1 - ab).sqrt()
    return ab, (b / (a + b))


# ----------------------------------------------------------------------------------------------------- registry
def test_registry_and_attributes():
    arch = arch_by_name("sdxl")
    assert isinstance(arch, archs.SdxlArch) and "sdxl" in archs.ARCH_NAMES
    assert (arch.name, arch.label, arch.modelspec) == ("sdxl", "SDXL", "stable-diffusion-xl-v1-base")
    assert arch.lora_format == "kohya" and arch.n_blocks == 70 and arch.size_multiple == 64
    assert arch.latent_tag == "_sdxlvae" and arch.empty_prompt == "" and arch.default_negative == ""
    assert (arch.default_sample_steps, arch.default_sample_cfg) == (30, 6.0)
    assert arch.default_targets == ("q", "k", "v", "o", "ffn_up_gate", "ffn_down", "proj")
    assert set(arch.target_groups) == set(arch.default_targets)
    assert arch.supports_single_files and arch.weighted_dop and arch.reference_exact_curriculum
    assert arch.reference_sigmas_exact and not arch.supports_kept_blocks and arch.supports_compile
    assert arch.max_text_tokens == 77 and not arch.linear_blocks
    assert get_arch(sdxl_cfg()) is arch
    assert arch.lora_metadata() == {"modelspec.architecture": "stable-diffusion-xl-v1-base/lora",
                                    "ss_base_model_version": "sdxl_base_v1-0"}
    facade = types.SimpleNamespace(layers=[object(), object()])
    assert arch.compile_blocks(facade) == facade.layers


def test_saved_lora_metadata_comes_from_the_arch():
    from rulztrainer.trainer import arch_metadata

    assert arch_metadata(arch_by_name("sdxl")) == arch_by_name("sdxl").lora_metadata()
    assert arch_metadata(arch_by_name("krea2")) == {"modelspec.architecture": "krea-2"}
    assert arch_metadata(types.SimpleNamespace(modelspec="x")) == {"modelspec.architecture": "x"}  # a test stub


def test_names_guess_the_family():
    for name in ("sd_xl_base_1.0.safetensors", "illustrious-xl-v1.safetensors", "ponyDiffusionV6XL.safetensors",
                 "noobaiXL_vpred.safetensors", "stabilityai/stable-diffusion-xl-base-1.0", "SDXL/unet"):
        assert family_named_in(name) == "sdxl", name
    assert family_named_in("krea/Krea-2-Raw") == "krea2"
    assert family_named_in("OpenMOSS-Team/MOSS-Image-Base") == "moss_image"
    assert family_named_in("ponytail_photos.safetensors") is None
    assert TrainConfig(model_arch="auto", model_source="hf", model_path=SDXL_HF).arch_name() == "sdxl"


def test_base_hooks_leave_the_other_families_alone():
    cfg = TrainConfig(model_arch="krea2", loss_weighting="min_snr")
    sigma = torch.linspace(0.02, 0.98, 25)
    for name in archs.ARCH_NAMES:
        if name == "sdxl":
            continue
        a = arch_by_name(name)
        assert a.sigma_bounds(cfg) == (0.0, 1.0) and a.init_noise_scale(0.93) == 1.0
        assert not a.weighted_dop and a.supports_kept_blocks and a.supports_compile and a.linear_blocks
        assert not a.reference_exact_curriculum and a.max_text_tokens is None
        for mode in ("none", "min_snr", "sigma_sqrt"):
            c = TrainConfig(loss_weighting=mode)
            assert torch.equal(a.loss_weight(c, sigma), loss_weight(c, sigma))
        assert a.lora_metadata() == {"modelspec.architecture": a.modelspec}


def test_exact_note_default_wording_is_the_step_distilled_one():
    h3 = arch_by_name("minimax_h3_image")
    note = h3.reference_exact_note(TrainConfig(timestep_sampling="uniform"))
    assert note[0][0] == "warning" and "one output head per sampling step" in note[0][1]
    assert "timestep_sampling: uniform" in note[0][1]
    ref = h3.reference_exact_note(TrainConfig(timestep_sampling="reference", timestep_shift=2.0, clean_share="40"))
    assert [lvl for lvl, _ in ref] == ["warning", "info"]
    assert arch_by_name("sdxl").reference_exact_note(sdxl_cfg())[-1][0] == "info"
    assert "SDXL" in arch_by_name("sdxl").reference_exact_note(sdxl_cfg(sigma_min=0.2))[0][1]


# ------------------------------------------------------------------------------------------------- detection
def test_detect_arch_reads_the_unet_folder(tmp_path):
    (tmp_path / "unet").mkdir()
    (tmp_path / "unet" / "config.json").write_text(json.dumps({"_class_name": "UNet2DConditionModel"}))
    cfg = TrainConfig(model_arch="auto", model_source="hf", model_path="some/folder")
    assert detect_arch(cfg, tmp_path).name == "sdxl"
    with pytest.raises(ValueError, match="holds a sdxl transformer"):
        detect_arch(TrainConfig(model_arch="krea2", model_source="hf", model_path="x"), tmp_path)
    (tmp_path / "unet" / "config.json").write_text(json.dumps({"_class_name": "Other"}))
    assert detect_arch(cfg, tmp_path).name == "moss_image"  # not recognized: the config's own guess


def write_header_file(path, keys):
    from safetensors.torch import save_file

    save_file({k: torch.zeros(1) for k in keys}, str(path))


def test_single_file_sniff(tmp_path):
    ldm = ["label_emb.0.0.weight", "input_blocks.0.0.weight", "middle_block.0.in_layers.0.weight"]
    for prefix in ("model.diffusion_model.", ""):
        f = tmp_path / f"net_{len(prefix)}.safetensors"
        write_header_file(f, [prefix + k for k in ldm] + ["first_stage_model.decoder.conv_in.weight"])
        assert arch_from_transformer_file(str(f)) == "sdxl"
    # SD 1.5's checkpoint has input_blocks but no label_emb: not SDXL
    f = tmp_path / "sd15.safetensors"
    write_header_file(f, ["model.diffusion_model.input_blocks.0.0.weight"])
    assert arch_from_transformer_file(str(f)) is None
    cfg = TrainConfig(model_arch="auto", model_source="files", transformer_file=str(tmp_path / "net_0.safetensors"))
    assert get_arch(cfg).name == "sdxl" and cfg.family() == "sdxl"  # no hint in the name


# --------------------------------------------------------------------------------------------------- config
def test_fresh_sdxl_config_validates_and_clamps_the_text_length():
    cfg = sdxl_cfg()
    assert cfg.max_token_length == 1024
    cfg.validate()
    assert cfg.max_token_length == 77
    assert cfg.checkpoint_keep_blocks == -1 and cfg.compile_model == "auto" and cfg.transformer_quant == "auto"
    assert cfg.training_adapter == cfg.turbo_lora == cfg.sample_lora == "auto"
    assert cfg.resolved_lora_targets() == list(arch_by_name("sdxl").default_targets)
    sdxl_cfg(timestep_sampling="reference", loss_weighting="min_snr", isolation=True).validate()
    sdxl_cfg(optimizer="keel", compile_model=False, transformer_quant="none", checkpoint_keep_blocks=0).validate()
    for quant in ("int8", "int8_convrot"):  # the UNet's transformer-block linears quantize (sdxl/loading.py)
        sdxl_cfg(transformer_quant=quant).validate()


def test_prediction_settings_are_checked():
    for key, bad in (("sdxl_prediction", "x0"), ("sdxl_zero_terminal_snr", "yes")):
        with pytest.raises(ValueError, match=key):
            TrainConfig(**{key: bad}).validate()
    sdxl_cfg(sdxl_prediction="v_prediction", sdxl_zero_terminal_snr=True).validate()
    assert TrainConfig.from_dict({"sdxl_zero_terminal_snr": "true"}).sdxl_zero_terminal_snr is True


REFUSALS = [
    ({"block_swap": 4}, "block_swap"),
    ({"multi_gpu": "model_split"}, "model_split"),
    ({"activation_offload": True}, "activation_offload"),
    ({"attn_localize": True, "trigger_word": "sks"}, "attn_localize"),
    ({"id_align": True}, "id_align"),
    ({"body_align": True}, "body_align"),
    ({"optimizer": "keel", "keel_face_gain": 0.5}, "keel_face_gain"),
    ({"optimizer": "keel_fb"}, "keel_fb"),
    ({"optimizer": "keel3_fb"}, "keel3_fb"),
    ({"optimizer": "hull", "hull_rho": 0.01}, "hull"),
    ({"training_maps": True}, "training_maps"),
    ({"transformer_quant": "nvfp4"}, "transformer_quant"),
    ({"train_frames": 4}, "train_frames"),
    ({"loss_type": "band"}, "loss_type"),
    ({"loss_weighting": "sigma_sqrt"}, "sigma_sqrt"),
    ({"finetune": True}, "finetune"),
    ({"sample_lora": "x.safetensors"}, "sample_lora"),
    ({"training_adapter": "x.safetensors"}, "training_adapter"),
    ({"checkpoint_keep_blocks": 3}, "checkpoint_keep_blocks"),
    ({"network_type": "lokr"}, "network_type"),
    ({"dora": True}, "dora"),
    ({"train_text_embedder": True}, "train_text_embedder"),
]


@pytest.mark.parametrize("override,setting", REFUSALS, ids=[s for _, s in REFUSALS])
def test_sdxl_refusals_name_the_setting(override, setting):
    with pytest.raises(ValueError, match=setting):
        sdxl_cfg(**override).validate()
    # the same setting is not an SDXL refusal for the families it fits (where it is not refused otherwise)
    if setting in ("sigma_sqrt", "loss_type", "network_type", "dora"):
        TrainConfig(model_arch="krea2", **override).validate()


def test_a_file_named_nothing_is_still_refused_by_what_it_holds(tmp_path):
    f = tmp_path / "model.safetensors"
    write_header_file(f, ["label_emb.0.0.weight", "input_blocks.0.0.weight"])
    cfg = TrainConfig(model_arch="auto", model_source="files", transformer_file=str(f), block_swap=2)
    with pytest.raises(ValueError, match="block_swap"):
        cfg.validate()


# ------------------------------------------------------------------------------------------------ downloads
def test_single_file_downloads_for_sdxl(tmp_path):
    from rulztrainer.comfy_files import planned_downloads, resolve_downloads

    f = tmp_path / "sd_xl_base_1.0.safetensors"
    f.write_bytes(b"")
    cfg = TrainConfig(model_arch="sdxl", model_source="files", transformer_file=str(f))
    turbo = ("turbo_lora", "tianweiy/DMD2", "dmd2_sdxl_4step_lora.safetensors")
    # one all-in-one file: only the DMD2 turbo LoRA is fetched, VAE / text encoders = the same file
    assert planned_downloads(cfg) == [turbo]
    assert planned_downloads(TrainConfig(model_arch="sdxl", model_source="files", transformer_file=str(f),
                                         turbo_lora="")) == []
    # no file set: the base 1.0 checkpoint with the 0.9 VAE is downloaded
    assert planned_downloads(TrainConfig(model_arch="sdxl", model_source="files")) == [
        ("transformer_file", "stabilityai/stable-diffusion-xl-base-1.0", "sd_xl_base_1.0_0.9vae.safetensors"), turbo]
    with pytest.raises(ValueError, match="PAN 2|pan2"):
        planned_downloads(TrainConfig(model_arch="pan2", model_source="files"))
    sep = TrainConfig(model_arch="sdxl", model_source="files", transformer_file=str(f), vae_file="vae.safetensors")
    assert planned_downloads(sep) == [turbo]
    out = resolve_downloads(dataclasses.replace(cfg, turbo_lora=""))
    assert (out.turbo_lora, out.sample_lora, out.training_adapter) == ("", "", "")
    assert out.vae_file == "" and out.text_encoder_file == ""


def test_runplan_builds_the_family_on_meta():
    from rulztrainer.runplan import FAMILY_MODEL

    assert FAMILY_MODEL["sdxl"] == ("sdxl", "SdxlUNet")


# -------------------------------------------------------------------------------------------- noise levels
def test_sigma_bounds_follow_the_prediction_settings():
    arch = arch_by_name("sdxl")
    _, s = textbook()
    lo, hi = arch.sigma_bounds(sdxl_cfg())
    assert lo == pytest.approx(float(s[0]), abs=1e-6) and hi == pytest.approx(float(s[999]), abs=1e-6)
    assert 0.02 < lo < 0.04 and 0.92 < hi < 0.95
    zlo, zhi = arch.sigma_bounds(sdxl_cfg(sdxl_prediction="v_prediction", sdxl_zero_terminal_snr=True))
    assert zhi == 1.0 and zlo == pytest.approx(float(textbook(True)[1][0]), abs=1e-6)
    assert arch.sigma_bounds(None) == (0.0, 1.0)


class _Model:
    prediction_type, zero_terminal_snr = "v_prediction", True


def test_bind_model_overrides_auto():
    arch = arch_by_name("sdxl")
    cfg = sdxl_cfg(model_path="stabilityai/sdxl-bound-test")
    assert arch.prediction(cfg) == ("epsilon", False)  # auto before a model is loaded
    arch.bind_model(cfg, _Model())
    assert arch.prediction(cfg) == ("v_prediction", True)
    assert arch.sigma_bounds(cfg)[1] == 1.0
    assert arch.prediction(sdxl_cfg(model_path="stabilityai/sdxl-other")) == ("epsilon", False)


@pytest.mark.parametrize("mode", ["uniform", "logit_normal", "sigmoid", "reference", "identity", "content", "logsnr"])
def test_sdxl_sigmas_stay_inside_the_schedule(mode):
    arch = arch_by_name("sdxl")
    cfg = sdxl_cfg(timestep_sampling=mode, timestep_shift=1.0)
    lo, hi = arch.sigma_bounds(cfg)
    for shift in (0.0, -3.0, 3.0):
        s = sample_sigmas(cfg, 4000, torch.Generator().manual_seed(3), logit_shift=shift, tokens=4096, arch=arch)
        assert float(s.min()) >= lo - 1e-7 and float(s.max()) <= hi + 1e-7, (mode, shift)
        assert s.dtype == torch.float32 and bool(torch.isfinite(s).all())
    z = sdxl_cfg(timestep_sampling=mode, sdxl_prediction="v_prediction", sdxl_zero_terminal_snr=True)
    s = sample_sigmas(z, 4000, torch.Generator().manual_seed(4), tokens=4096, arch=arch)
    assert float(s.max()) <= 1.0 and float(s.min()) >= arch.sigma_bounds(z)[0] - 1e-7


def test_non_exact_modes_remap_not_clamp():
    arch = arch_by_name("sdxl")
    cfg = sdxl_cfg(timestep_sampling="uniform")
    lo, hi = arch.sigma_bounds(cfg)
    s = sample_sigmas(cfg, 20000, torch.Generator().manual_seed(5), arch=arch)
    # a clamp would pile a share of the draws on the two ends; a remap spreads them over the whole range
    assert float((s == lo).float().mean()) < 0.01 and float((s == hi).float().mean()) < 0.01
    assert float(s.min()) < lo + 0.01 and float(s.max()) > hi - 0.01
    assert torch.allclose(s * 1000, (s * 1000).round(), atol=1e-3) or float(s.min()) == pytest.approx(lo, abs=2e-3)


def test_reference_path_returns_exact_schedule_values():
    arch = arch_by_name("sdxl")
    _, table = textbook()
    cfg = sdxl_cfg()
    s = sample_sigmas(cfg, 3000, torch.Generator().manual_seed(6), tokens=4096, arch=arch)
    uniq = torch.unique(s).double()
    want = table.double()
    # every value is s(t) for an integer t: in the table, not on the k/1000 grid
    nearest = (want[None, :] - uniq[:, None]).abs().min(dim=1).values
    assert float(nearest.max()) < 1e-6
    assert not torch.allclose(s * 1000, (s * 1000).round(), atol=1e-3)
    assert len(uniq) > 900  # 3000 uniform draws over 1000 timesteps


def test_reference_path_honours_the_curriculum():
    arch = arch_by_name("sdxl")
    _, table = textbook()
    n = 6000

    def draws(**kw):
        cfg = sdxl_cfg(**{k: v for k, v in kw.items() if k != "shift"})
        return sample_sigmas(cfg, n, torch.Generator().manual_seed(7), logit_shift=kw.get("shift", 0.0), tokens=4096,
                             arch=arch)

    base = draws()
    gen = torch.Generator().manual_seed(7)
    u = torch.rand(n, generator=gen)
    t = (u * 1000).floor().clamp(0, 999).long()
    assert torch.allclose(base.double(), table[t], atol=1e-6)  # the plain draw is s(floor(1000 u)) exactly
    noisier = draws(shift=1.5)  # logit shift up: noisier steps (clean_share and the curriculum use it)
    cleaner = draws(shift=-1.5)
    assert float(noisier.median()) > float(base.median()) > float(cleaner.median())
    # timestep_shift: u -> ts u / (1 + (ts - 1) u), computed here
    ts = 3.0
    shifted = draws(timestep_shift=ts)
    u2 = ts * u / (1 + (ts - 1) * u)
    assert torch.allclose(shifted.double(), table[(u2 * 1000).floor().clamp(0, 999).long()], atol=1e-6)
    # a logit shift of exactly the textbook value
    from rulztrainer.trainer import _shift_logit

    u3 = _shift_logit(u, 0.7)
    assert torch.allclose(draws(shift=0.7).double(), table[(u3 * 1000).floor().clamp(0, 999).long()], atol=1e-6)


def test_reference_path_takes_the_strata_draw():
    from rulztrainer.trainer import make_strata

    cfg = sdxl_cfg(timestep_buckets=4)
    strata = make_strata(cfg)
    assert strata is not None  # SDXL's reference draw is bucketed, unlike the step-distilled families'
    s = sample_sigmas(cfg, 8, torch.Generator().manual_seed(8), tokens=4096, arch=arch_by_name("sdxl"), strata=strata)
    assert s.shape == (8,)
    assert make_strata(TrainConfig(model_arch="krea2", timestep_sampling="reference", timestep_buckets=4)) is None


def old_sample_sigmas(cfg, n, gen, logit_shift):
    """trainer.sample_sigmas as it was before the arch bounds (logit_normal and uniform modes only)."""
    if cfg.timestep_sampling == "uniform":
        s = torch.rand(n, generator=gen)
        if logit_shift:
            s = s.clamp(1e-4, 1 - 1e-4)
            s = torch.sigmoid(torch.log(s / (1 - s)) + logit_shift)
    else:
        s = torch.sigmoid(torch.randn(n, generator=gen) * cfg.logit_std + cfg.logit_mean + logit_shift)
    if cfg.timestep_shift != 1.0:
        s = cfg.timestep_shift * s / (1 + (cfg.timestep_shift - 1) * s)
    if cfg.sigma_min > 0.0 or cfg.sigma_max < 1.0:
        s = cfg.sigma_min + (cfg.sigma_max - cfg.sigma_min) * s
    return (s * 1000).round().clamp(1, 1000) / 1000


@pytest.mark.parametrize("family", ["krea2", "moss_image", "qwen_image21", "fibo2"])
@pytest.mark.parametrize("mode", ["logit_normal", "uniform"])
def test_other_families_sample_the_same_sigmas_as_before(family, mode):
    cfg = TrainConfig(model_arch=family, timestep_sampling=mode, timestep_shift=2.0, sigma_min=0.05, logit_std=1.2)
    for shift in (0.0, 0.8):
        got = sample_sigmas(cfg, 500, torch.Generator().manual_seed(11), shift, tokens=4096)
        want = old_sample_sigmas(cfg, 500, torch.Generator().manual_seed(11), shift)
        assert torch.equal(got, want)
        again = sample_sigmas(cfg, 500, torch.Generator().manual_seed(11), shift, tokens=4096, arch=arch_by_name(family))
        assert torch.equal(again, want)


# --------------------------------------------------------------------------------------------- loss weight
def sigma_at(t):
    _, table = textbook()
    return table[torch.as_tensor(t)].float()


@pytest.mark.parametrize("t", [100, 500, 999])
def test_epsilon_loss_weights_are_the_textbook_ones(t):
    arch = arch_by_name("sdxl")
    ab, _ = textbook()
    s = sigma_at([t])
    snr = float(ab[t] / (1 - ab[t]))
    flow = (1 - float(s)) ** 2  # the facade's velocity error is the epsilon error / (1 - s)
    none = arch.loss_weight(sdxl_cfg(loss_weighting="none"), s)
    assert float(none) == pytest.approx(flow, rel=1e-5)
    mins = arch.loss_weight(sdxl_cfg(loss_weighting="min_snr"), s)
    assert float(mins) == pytest.approx(min(snr, 5.0) / snr * flow, rel=1e-4)
    # epsilon MSE = flow MSE x (1 - s)^2: an elementwise check with a made-up prediction error
    eps_err = torch.randn(4, 16, 3, 3, dtype=torch.float64)
    vel_err = eps_err / (1 - s.double())
    assert torch.allclose((vel_err ** 2).mean() * none.double(), (eps_err ** 2).mean(), rtol=1e-4)


@pytest.mark.parametrize("t", [100, 500, 999])
def test_v_prediction_loss_weights_are_the_textbook_ones(t):
    arch = arch_by_name("sdxl")
    ab, _ = textbook()
    s = sigma_at([t])
    snr = float(ab[t] / (1 - ab[t]))
    flow = (1 - float(s)) ** 2 + float(s) ** 2
    cfg = sdxl_cfg(sdxl_prediction="v_prediction")
    assert float(arch.loss_weight(cfg, s)) == pytest.approx(flow, rel=1e-5)
    mins = arch.loss_weight(sdxl_cfg(sdxl_prediction="v_prediction", loss_weighting="min_snr"), s)
    assert float(mins) == pytest.approx(min(snr, 5.0) / (snr + 1) * flow, rel=1e-4)


def test_sdxl_loss_weight_shapes_and_refusal():
    arch = arch_by_name("sdxl")
    s = sigma_at(torch.arange(0, 1000, 50))
    w = arch.loss_weight(sdxl_cfg(loss_weighting="min_snr"), s)
    assert w.shape == s.shape and bool((w > 0).all()) and bool(torch.isfinite(w).all())
    assert arch.loss_weight(sdxl_cfg(), s.bfloat16()).dtype == torch.float32
    with pytest.raises(ValueError, match="sigma_sqrt"):
        arch.loss_weight(sdxl_cfg(loss_weighting="sigma_sqrt"), s)


def test_init_noise_scale():
    assert arch_by_name("krea2").init_noise_scale(0.5) == 1.0
    arch = arch_by_name("sdxl")
    for s0 in (0.5, 0.9362, 1.0):
        assert arch.init_noise_scale(s0) == pytest.approx(math.sqrt((1 - s0) ** 2 + s0 ** 2), rel=1e-6)


def test_reward_group_starts_from_the_scaled_noise():
    from rulztrainer import reward

    class Zero(torch.nn.Module):
        def forward(self, x, t, e, m, **kw):
            return torch.zeros_like(x)

    shape = (16, 40, 40)
    sig = torch.tensor([0.9, 0.0])
    plain, _ = reward.sample_group(Zero(), sig, None, None, None, None, cfg_scale=1.0, noise_level=0.0, seeds=[1, 2],
                                   shape=shape, device="cpu")
    scaled, _ = reward.sample_group(Zero(), sig, None, None, None, None, cfg_scale=1.0, noise_level=0.0, seeds=[1, 2],
                                    shape=shape, device="cpu", init_scale=0.5)
    assert torch.allclose(scaled[0], plain[0] * 0.5)
    assert float(plain[0].std()) == pytest.approx(1.0, abs=0.05)


# ------------------------------------------------------------------------------------------------------ DOP
class _Net(torch.nn.Module):
    """A stand-in model whose output depends on one trainable scale and on sigma."""

    def __init__(self):
        super().__init__()
        self.k = torch.nn.Parameter(torch.tensor(1.0))
        self.mult = 1.0

    def forward(self, x, t, emb, mask):
        return x.float() * (self.k * self.mult) + 0.01 * t.view(-1, 1, 1, 1)


def run_dop(arch, cfg, sigma):
    from rulztrainer.trainer import Trainer

    net = _Net()
    inj = {"m": net}  # set_multiplier sets .multiplier on every value
    net.multiplier = 1.0
    lora_off_mult = {}

    class Wrapped(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = net

        def forward(self, x, t, e, m):
            self.net.mult = 0.0 if net.multiplier == 0.0 else 1.0
            return self.net(x, t, e, m)

    model = Wrapped()
    me = types.SimpleNamespace(arch=arch, dop_values=[])
    gen = torch.Generator().manual_seed(0)
    latents = torch.randn(3, 16, 4, 4, generator=gen)
    noise = torch.randn(2, 16, 4, 4, generator=gen)
    net.k.data.fill_(1.7)
    dop = ([0, 2], torch.zeros(2, 5, 8), torch.ones(2, 5, dtype=torch.bool), sigma, noise)
    Trainer._dop_pass(me, model, inj, "cpu", latents, 3, dop, 1, cfg)
    return me.dop_values[0], net.k.grad.clone(), noise, latents


def test_dop_loss_is_unweighted_for_the_flow_families_and_weighted_for_sdxl():
    sigma = torch.tensor([0.2, 0.9])
    cfg = TrainConfig(model_arch="krea2", isolation=True, dop_weight=2.0, loss_weighting="min_snr")
    val, grad, noise, latents = run_dop(arch_by_name("krea2"), cfg, sigma)
    # by hand, the way the flow families always did it: a plain MSE against the LoRA-off prediction
    net = _Net()
    s = sigma.view(-1, 1, 1, 1)
    noisy = ((1 - s) * latents[[0, 2]] + s * noise).to(torch.bfloat16)
    t = sigma * 1000.0
    target = (noisy.float() * 0.0 + 0.01 * t.view(-1, 1, 1, 1))
    net.k.data.fill_(1.7)
    want = torch.nn.functional.mse_loss(net(noisy, t, None, None).float(), target)
    assert val == pytest.approx(float(want.detach()), rel=1e-5)
    scfg = sdxl_cfg(isolation=True, dop_weight=2.0, loss_weighting="none")
    sval, sgrad, _, _ = run_dop(arch_by_name("sdxl"), scfg, sigma)
    per = (net(noisy, t, None, None).float() - target).pow(2).mean(dim=(1, 2, 3))
    w = arch_by_name("sdxl").loss_weight(scfg, sigma)
    assert sval == pytest.approx(float((per * w).mean()), rel=1e-5)
    assert sval != pytest.approx(val, rel=1e-3)


# ----------------------------------------------------------------------------------------------- LoRA names
@pytest.fixture(scope="module")
def facade():
    pytest.importorskip("diffusers")
    model_mod = pytest.importorskip("rulztrainer.sdxl.model")
    with torch.device("meta"):
        return model_mod.SdxlUNet()


def block_name(model, i):
    target = model.layers[i]
    return next(n for n, m in model.named_modules() if m is target)


def test_lora_module_names_on_the_full_facade(facade):
    arch = arch_by_name("sdxl")
    assert len(facade.layers) == 70 == arch.n_blocks
    counts = {g: len(arch.lora_module_names(facade, [g], None)) for g in arch.target_groups}
    assert counts == {"q": 140, "k": 140, "v": 140, "o": 140, "ffn_up_gate": 70, "ffn_down": 70, "proj": 22}
    allnames = arch.lora_module_names(facade, arch.default_targets, None)
    assert len(allnames) == 722 == len(set(allnames))
    assert all(n.startswith("unet.") for n in allnames)
    assert "unet.down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k" in allnames
    assert "unet.mid_block.attentions.0.proj_out" in allnames
    assert arch.lora_module_names(facade, ["bogus"], None) == []


def test_lora_block_selection_follows_the_compute_order(facade):
    arch = arch_by_name("sdxl")
    for i in (0, 5, 69):
        names = arch.lora_module_names(facade, ["q", "ffn_down"], [i])
        head = block_name(facade, i) + "."
        assert len(names) == 3 and all(n.startswith(head) for n in names), (i, names)
    both = arch.lora_module_names(facade, ["q"], [0, 69])
    assert len(both) == 4
    # proj_in / proj_out go with the first block of their transformer: block 0 holds the first transformer's
    names = arch.lora_module_names(facade, ["proj"], [0])
    assert len(names) == 2 and all(n.rpartition(".")[2] in ("proj_in", "proj_out") for n in names)
    parent = names[0].rpartition(".")[0]
    assert block_name(facade, 0) == f"{parent}.transformer_blocks.0"
    assert arch.lora_module_names(facade, ["proj"], [1]) == []  # a transformer's second block has no projections
    assert arch.lora_module_names(facade, ["q"], []) == []


def test_runplan_counts_the_adapter_on_meta(facade):
    from rulztrainer import runplan

    shapes = runplan.target_shapes("sdxl", tuple(arch_by_name("sdxl").default_targets), "", False)
    assert len(shapes) == 722 and all(len(s) == 2 for s in shapes)


# ------------------------------------------------------------------------------------- review follow-ups
def test_epsilon_with_zero_terminal_snr_refused():
    with pytest.raises(ValueError, match="sdxl_zero_terminal_snr"):
        sdxl_cfg(sdxl_prediction="epsilon", sdxl_zero_terminal_snr=True).validate()
    sdxl_cfg(sdxl_prediction="v_prediction", sdxl_zero_terminal_snr=True).validate()
    arch = arch_by_name("sdxl")
    model = types.SimpleNamespace(prediction_type="epsilon", zero_terminal_snr=True)
    with pytest.raises(ValueError, match="zero terminal SNR"):
        arch.bind_model(sdxl_cfg(model_path="x/eps-ztsnr"), model)


def test_default_timestep_sampling_becomes_reference_for_sdxl():
    cfg = sdxl_cfg(timestep_sampling="logit_normal")
    cfg.validate()
    assert cfg.timestep_sampling == "reference"
    cfg = sdxl_cfg(timestep_sampling="sigmoid")
    cfg.validate()
    assert cfg.timestep_sampling == "sigmoid"


def test_sdxl_found_late_is_validated(monkeypatch):
    """A folder whose name names no family is guessed as another family by validate(); once detect_arch finds SDXL,
    the trainer runs SDXL's refusals (the source must call validate_sdxl right after detection)."""
    import inspect

    from rulztrainer import trainer

    src = inspect.getsource(trainer.Trainer._run)
    detect = src.index("self.arch = detect_arch(cfg, self.model_dir)")
    assert src.index("cfg.validate_sdxl()") > detect
    cfg = sdxl_cfg(block_swap=4)
    with pytest.raises(ValueError, match="block_swap"):
        cfg.validate_sdxl()


def test_plan_download_sharded_fp16():
    from rulztrainer.sdxl import loading as SL

    files = ["unet/config.json", "unet/diffusion_pytorch_model.safetensors.index.fp16.json",
             "unet/diffusion_pytorch_model.fp16-00001-of-00002.safetensors",
             "unet/diffusion_pytorch_model.fp16-00002-of-00002.safetensors",
             "unet/diffusion_pytorch_model.safetensors.index.json",
             "unet/diffusion_pytorch_model-00001-of-00002.safetensors",
             "unet/diffusion_pytorch_model-00002-of-00002.safetensors"]
    unet = [f for f in SL.plan_download(files) if f.startswith("unet/") and f != "unet/config.json"]
    assert sorted(unet) == sorted(f for f in files if "fp16" in f)
