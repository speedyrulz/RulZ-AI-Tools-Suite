"""SDXL support without model weights: the flow-matching facade around a tiny UNet2DConditionModel with the SDXL
structure (epsilon and v prediction, zero terminal SNR), the schedule maths, Euler on the facade against diffusers'
DDIMScheduler, the loss weights that turn velocity error back into epsilon / v error, patchify, the text path with
tiny CLIP models and a tiny tokenizer, loading from a diffusers folder, the all-in-one checkpoint converters (round
trip through reverse-named tiny state dicts), the bundled configs on the meta device, and the sampler."""
import json
import logging
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

diffusers = pytest.importorskip("diffusers")
transformers = pytest.importorskip("transformers")
safetensors_torch = pytest.importorskip("safetensors.torch")

from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel  # noqa: E402
from diffusers.models.attention import BasicTransformerBlock  # noqa: E402
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer  # noqa: E402

from rulztrainer.sdxl import loading as SL  # noqa: E402
from rulztrainer.sdxl import model as M  # noqa: E402
from rulztrainer.sdxl import sampler as SS  # noqa: E402
from rulztrainer.sdxl import single_file as SF  # noqa: E402

POOLED, TEXT_D = 16, 24
TINY_UNET = dict(
    sample_size=8, in_channels=4, out_channels=4, layers_per_block=2, block_out_channels=(16, 32, 64),
    down_block_types=("DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"),
    up_block_types=("CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"),
    attention_head_dim=(2, 4, 8), transformer_layers_per_block=(1, 2, 3), cross_attention_dim=TEXT_D,
    use_linear_projection=True, addition_embed_type="text_time", addition_time_embed_dim=8,
    projection_class_embeddings_input_dim=POOLED + 6 * 8, norm_num_groups=8, mid_block_type="UNetMidBlock2DCrossAttn",
    flip_sin_to_cos=True, freq_shift=0, upcast_attention=None, resnet_time_scale_shift="default")
TINY_VAE = dict(in_channels=3, out_channels=3, latent_channels=4, block_out_channels=(8, 8, 16, 16),
                down_block_types=("DownEncoderBlock2D",) * 4, up_block_types=("UpDecoderBlock2D",) * 4,
                layers_per_block=1, norm_num_groups=4, scaling_factor=0.13025, sample_size=64)
CLIP_L = dict(hidden_size=16, intermediate_size=32, num_hidden_layers=3, num_attention_heads=2, vocab_size=80,
              max_position_embeddings=77, projection_dim=16, hidden_act="quick_gelu", eos_token_id=2, bos_token_id=0,
              pad_token_id=1)
CLIP_G = dict(CLIP_L, hidden_size=24, intermediate_size=48, num_attention_heads=2, projection_dim=POOLED,
              hidden_act="gelu")
LAT = 4  # latent cells per side: 64 px images


def tiny_unet(seed=0, dtype=torch.float32):
    torch.manual_seed(seed)
    unet = UNet2DConditionModel(**TINY_UNET).to(dtype).eval()
    # a trained UNet's last layers are not zero: give the output a real signal
    for p in unet.parameters():
        if p.abs().max() == 0:
            p.data.normal_(0, 0.1)
    return unet


def facade(prediction="epsilon", ztsnr=False, seed=0):
    return M.SdxlUNet(tiny_unet(seed), prediction, ztsnr).eval()


def embeds(batch=2, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(batch, M.TEXT_ROWS, TEXT_D, generator=g)


@pytest.mark.parametrize("fmt", ["int8", "int8_convrot"])
def test_sdxl_transformer_linear_quant_and_lora(fmt, tmp_path):
    from rulztrainer import lora, quant as Q

    ref = M.SdxlUNet(tiny_unet(dtype=torch.bfloat16)).requires_grad_(False)
    model = M.SdxlUNet(tiny_unet(dtype=torch.bfloat16)).requires_grad_(False)
    eligible = {n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)
                and "transformer_blocks." in n and Q.can_quantize(m.weight, fmt, 16)}
    assert eligible
    assert Q.quantize_module_linears(model, fmt, min_dim=16,
                                     name_filter=lambda n: "transformer_blocks." in n) == len(eligible)
    assert {n for n, m in model.named_modules() if isinstance(m, Q.QuantLinear)} == eligible
    x = torch.randn(1, 16, LAT, LAT).bfloat16()
    cond = embeds(1).bfloat16()
    with torch.no_grad():
        expected, actual = ref(x, 400.0, cond), model(x, 400.0, cond)
    assert torch.allclose(actual, expected, atol=0.03, rtol=0.05)

    names = sorted(eligible)
    a = lora.inject_lora(ref, 2, 2, [], names=names)
    b = lora.inject_lora(model, 2, 2, [], names=names)
    lora.save_lora(a, tmp_path / "ref.safetensors", fmt="kohya")
    lora.save_lora(b, tmp_path / "quant.safetensors", fmt="kohya")
    assert set(safetensors_torch.load_file(tmp_path / "ref.safetensors")) == set(
        safetensors_torch.load_file(tmp_path / "quant.safetensors"))
    model(x, 400.0, cond).square().mean().backward()
    assert all(m.lora_up.weight.grad is not None for m in b.values())


def test_sdxl_quant_loader_rejects_nvfp4_before_loading():
    with pytest.raises(ValueError, match="SDXL.*nvfp4"):
        SL.load_any_transformer("missing", "cpu", "nvfp4")


# ---------------------------------------------------------------- schedule


@pytest.mark.parametrize("ztsnr", [False, True])
def test_sigma_timestep_round_trip_all_integers(ztsnr):
    t = torch.arange(1000)
    s = M.sigma_of_timestep(t, ztsnr)
    assert s.dtype == torch.float32
    assert torch.all(s[1:] > s[:-1])  # noise increases with t
    back = M.timestep_of_sigma(s, ztsnr)
    assert back.dtype == torch.float32
    assert (back - t.float()).abs().max() < 1e-2  # the facade snaps within 1e-2 of an integer
    assert torch.equal(back.round(), t.float())


def test_schedule_table_matches_ddpm_and_ztsnr_ends_at_zero():
    ac = M.alphas_cumprod()
    betas = torch.linspace(0.00085**0.5, 0.012**0.5, 1000, dtype=torch.float64) ** 2
    assert ac.dtype == torch.float64 and torch.allclose(ac, torch.cumprod(1 - betas, 0))
    z = M.alphas_cumprod(True)
    assert float(z[-1]) == pytest.approx(0.0, abs=1e-12)
    assert float(z[0]) == pytest.approx(float(ac[0]), rel=1e-9)  # the first value is kept
    # diffusers' own rescale
    ref = DDIMScheduler(rescale_betas_zero_snr=True, beta_schedule="scaled_linear", beta_start=0.00085,
                        beta_end=0.012).alphas_cumprod.double()
    assert torch.allclose(z, ref, atol=1e-6)


def test_fractional_timestep_interpolates_log_alphabar():
    ac = M.alphas_cumprod()
    s = M.sigma_of_timestep(torch.tensor([10.5], dtype=torch.float64))
    a2 = float(ac[10]) ** 0.5 * float(ac[11]) ** 0.5  # alphabar at 10.5: the log-midpoint (geometric mean)
    a, b = math.sqrt(a2), math.sqrt(1 - a2)
    assert float(s) == pytest.approx(b / (a + b), abs=1e-6)


def test_bounds_and_init_scale():
    lo, hi = M.sigma_bounds("epsilon", False)
    assert lo == pytest.approx(0.02834, abs=1e-4) and hi == pytest.approx(0.93596, abs=1e-4)
    lo2, hi2 = M.sigma_bounds("v_prediction", True)
    assert hi2 == 1.0 and lo2 == pytest.approx(lo, abs=1e-3)
    assert M.init_noise_scale(0.0) == 1.0
    s0 = 0.9
    assert M.init_noise_scale(s0) == pytest.approx(math.sqrt(0.01 + 0.81))
    k, a, b = M.flow_to_vp(torch.tensor([s0]))
    assert float(1 / k) == pytest.approx(M.init_noise_scale(s0), rel=1e-6)
    assert float(a**2 + b**2) == pytest.approx(1.0, rel=1e-6)


def test_flow_to_vp_round_trip():
    g = torch.Generator().manual_seed(3)
    x0, eps = torch.randn(2, 4, 8, 8, generator=g), torch.randn(2, 4, 8, 8, generator=g)
    s = torch.tensor([0.2, 0.8])
    k, a, b = (v.view(-1, 1, 1, 1) for v in M.flow_to_vp(s))
    s4 = s.view(-1, 1, 1, 1)
    x_flow = (1 - s4) * x0 + s4 * eps
    x_t = k * x_flow
    assert torch.allclose(x_t, a * x0 + b * eps, atol=1e-6)
    assert torch.allclose(x_t / k, x_flow, atol=1e-6)
    # from x_t and the true eps, x0 is recovered the way the facade does it
    assert torch.allclose((x_t - b * eps) / a, x0, atol=1e-5)


# ---------------------------------------------------------------- the facade


def _spy_unet(model):
    seen = {}

    def hook(module, args, kwargs):
        seen["sample"] = args[0] if args else kwargs["sample"]
        seen["t"] = args[1] if len(args) > 1 else kwargs["timestep"]
        seen["cond"] = kwargs.get("added_cond_kwargs")

    model.unet.register_forward_pre_hook(hook, with_kwargs=True)
    return seen


@pytest.mark.parametrize("prediction", ["epsilon", "v_prediction"])
def test_facade_equals_direct_unet_call(prediction):
    m = facade(prediction)
    g = torch.Generator().manual_seed(5)
    x0 = torch.randn(2, 4, 8, 8, generator=g)
    eps = torch.randn(2, 4, 8, 8, generator=g)
    s = torch.tensor([0.37, 0.61])
    s4 = s.view(-1, 1, 1, 1)
    x_flow = (1 - s4) * x0 + s4 * eps
    emb = embeds()
    seen = _spy_unet(m)
    got = m(M.patchify(x_flow), s * 1000, emb)
    assert got.dtype == torch.float32 and got.shape == (2, 16, LAT, LAT)

    k, a, b = (v.view(-1, 1, 1, 1) for v in M.flow_to_vp(s))
    x_t = a * x0 + b * eps
    t = M.timestep_of_sigma(s)
    assert (t - t.round()).abs().min() > 1e-2  # a genuinely continuous timestep
    pooled = emb[:, 77, :POOLED]
    ids = torch.tensor([[64.0, 64.0, 0.0, 0.0, 64.0, 64.0]]).expand(2, 6)
    with torch.no_grad():
        out = m.unet(x_t, t, encoder_hidden_states=emb[:, :77],
                     added_cond_kwargs={"text_embeds": pooled, "time_ids": ids}).sample
    if prediction == "epsilon":
        e_hat, x0_hat = out, (x_t - b * out) / a
    else:
        e_hat, x0_hat = b * x_t + a * out, a * x_t - b * out
    want = M.patchify(e_hat - x0_hat)
    assert torch.allclose(got, want, atol=1e-4, rtol=1e-4)
    # what the UNet received: fp32 timestep, VP input, pooled row, [H, W, 0, 0, H, W]
    assert seen["t"].dtype == torch.float32 and torch.allclose(seen["t"], t, atol=1e-3)
    assert torch.allclose(seen["sample"], x_t, atol=1e-5)
    assert torch.equal(seen["cond"]["text_embeds"], pooled)
    assert torch.equal(seen["cond"]["time_ids"], ids)


def test_integer_timestep_is_fed_as_integer_and_scalar_t_works():
    m = facade()
    seen = _spy_unet(m)
    s = float(M.sigma_of_timestep(500))
    lat = torch.randn(1, 16, LAT, LAT)
    m(lat, torch.tensor(s * 1000.0), embeds(1))
    assert seen["t"].dtype == torch.float32 and seen["t"].tolist() == [500.0]
    m(lat, s * 1000.0, embeds(1))  # a python float
    assert seen["t"].tolist() == [500.0]


@pytest.mark.parametrize("prediction", ["epsilon", "v_prediction"])
def test_perfect_unet_gives_exact_velocity_and_loss_weights_match_native_loss(prediction):
    """A UNet that returns the true target gives eps - x0; with a wrong one the velocity error times the arch's
    weight is exactly the native epsilon (or v) error: (1-s)^2 for epsilon, (1-s)^2 + s^2 for v."""
    m = facade(prediction)
    g = torch.Generator().manual_seed(8)
    x0, eps = torch.randn(3, 4, 8, 8, generator=g), torch.randn(3, 4, 8, 8, generator=g)
    delta = torch.randn(3, 4, 8, 8, generator=g)
    s = torch.tensor([0.15, 0.5, 0.9])
    s4 = s.view(-1, 1, 1, 1)
    x_flow = (1 - s4) * x0 + s4 * eps
    k, a, b = (v.view(-1, 1, 1, 1) for v in M.flow_to_vp(s))
    target = eps if prediction == "epsilon" else a * eps - b * x0  # what the UNet should output

    state = {"extra": torch.zeros_like(delta)}
    m.unet.forward = lambda *args, **kw: (target + state["extra"],)
    exact = m(M.patchify(x_flow), s * 1000, embeds(3))
    assert torch.allclose(exact, M.patchify(eps - x0), atol=2e-4, rtol=1e-4)

    state["extra"] = delta
    wrong = m(M.patchify(x_flow), s * 1000, embeds(3))
    v_err = (wrong - M.patchify(eps - x0)).pow(2).mean(dim=(1, 2, 3))
    weight = (1 - s) ** 2 if prediction == "epsilon" else (1 - s) ** 2 + s**2
    native = delta.pow(2).mean(dim=(1, 2, 3))
    assert torch.allclose(v_err * weight, native, rtol=2e-3)


def test_facade_zero_terminal_snr_v_prediction_at_sigma_one_is_finite_and_exact():
    m = facade("v_prediction", ztsnr=True)
    g = torch.Generator().manual_seed(9)
    x0, eps = torch.randn(1, 4, 8, 8, generator=g), torch.randn(1, 4, 8, 8, generator=g)
    s = torch.tensor([1.0])
    k, a, b = (v.view(-1, 1, 1, 1) for v in M.flow_to_vp(s))
    assert float(a) == 0.0
    target = a * eps - b * x0
    m.unet.forward = lambda *args, **kw: (target,)
    seen = _spy_unet(m)
    out = m(M.patchify(eps), s * 1000, embeds(1))  # x_flow = eps at s = 1
    assert torch.isfinite(out).all()
    assert torch.allclose(out, M.patchify(eps - x0), atol=1e-5)
    assert seen["t"].tolist() == [999.0]


def test_compile_blocks_are_the_facades_basic_transformer_blocks():
    from diffusers.models.attention import BasicTransformerBlock
    from rulztrainer.archs import SdxlArch

    m = facade()
    blocks = SdxlArch().compile_blocks(m)
    assert blocks and blocks == list(m.layers)
    assert all(isinstance(b, BasicTransformerBlock) for b in blocks)


def test_facade_input_checks_and_skip_layers():
    m = facade()
    lat = torch.randn(1, 16, LAT, LAT)
    with pytest.raises(ValueError):
        m(lat, 500.0, embeds(1)[:, :77])  # 78 rows are required
    with pytest.raises(ValueError):
        m(lat, 500.0, embeds(1), skip_layers=[3])
    with pytest.raises(ValueError):
        m(torch.randn(1, 64, LAT, LAT), 500.0, embeds(1))
    assert m(lat, 500.0, embeds(1), skip_layers=None).shape == lat.shape
    assert m.latent_channels == 16 and m.dtype == torch.float32 and m.device.type == "cpu"


def test_facade_backward_reaches_the_unet_and_bf16_unet_runs():
    m = facade()
    lat = torch.randn(2, 16, LAT, LAT)
    out = m(lat, torch.tensor([300.0, 700.0]), embeds())
    out.pow(2).mean().backward()
    grads = [p.grad for p in m.unet.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    mb = M.SdxlUNet(tiny_unet(dtype=torch.bfloat16), "epsilon")
    out = mb(lat.bfloat16(), torch.tensor([300.0, 700.0]), embeds().bfloat16())
    assert out.dtype == torch.float32 and torch.isfinite(out).all() and mb.dtype == torch.bfloat16


def test_layers_order_stability_and_checkpointing_flag_on_tiny():
    m = facade()
    n_down = 2 * 2 + 2 * 3  # blocks 1, 2: two attentions each, with 2 and 3 transformer layers
    n_mid = 3
    n_up = 3 * 3 + 3 * 2  # up_blocks 0 / 1 reverse the per-block counts: 3 layers then 2 layers, 3 attentions each
    assert len(m.layers) == n_down + n_mid + n_up
    assert m.layers is m.layers and all(isinstance(b, BasicTransformerBlock) for b in m.layers)
    assert not any(k.startswith("layers") for k in m.state_dict())
    assert all(k.startswith("unet.") for k in m.state_dict())
    ids = [id(b) for b in m.layers]
    m.to(torch.float64)
    assert [id(b) for b in m.layers] == ids
    assert m.layers[0] is m.unet.down_blocks[1].attentions[0].transformer_blocks[0]
    assert m.layers[-1] is m.unet.up_blocks[1].attentions[-1].transformer_blocks[-1]
    assert [m.unet.conv_norm_out, m.unet.conv_out] == m.split_tail_modules()
    assert m.gradient_checkpointing is False
    m.gradient_checkpointing = True
    assert m.gradient_checkpointing is True and m.unet.is_gradient_checkpointing
    m.gradient_checkpointing = False
    assert not m.unet.is_gradient_checkpointing


def test_full_sdxl_config_on_meta_has_70_layers_and_stable_ids():
    with torch.device("meta"):
        m = M.SdxlUNet()
    assert len(m.layers) == 70
    ids = [id(b) for b in m.layers]
    assert [id(b) for b in m.layers] == ids
    assert m.layers is m.layers
    n = sum(p.numel() for p in m.unet.parameters())
    assert 2.55e9 < n < 2.60e9  # SDXL base: 2.567 B
    assert (m.text_dim, m.pooled_dim, m.latent_channels) == (2048, 1280, 16)
    m.gradient_checkpointing = True
    assert m.unet.is_gradient_checkpointing
    names = [n for n, _ in m.named_modules() if n.endswith("attn1.to_q")]
    assert names[0] == "unet.down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q"
    assert len(names) == 70
    assert m.unet.config.addition_time_embed_dim == 256


# ---------------------------------------------------------------- Euler on the facade == DDIM


@pytest.mark.parametrize("prediction", ["epsilon", "v_prediction"])
@pytest.mark.parametrize("steps", [5, 10])  # divisors of 1000: DDIM steps back by 1000 // steps exactly
def test_euler_on_facade_equals_ddim(prediction, steps):
    m = facade(prediction, seed=2)
    emb = embeds(1)
    g = torch.Generator().manual_seed(11)
    noise = torch.randn(1, 4, 8, 8, generator=g)

    ddim = DDIMScheduler(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                         clip_sample=False, set_alpha_to_one=True, steps_offset=0, timestep_spacing="trailing",
                         prediction_type=prediction)
    ddim.set_timesteps(steps)
    assert ddim.timesteps.tolist() == SS.trailing_timesteps(steps).tolist()
    ids = torch.tensor([[64.0, 64.0, 0.0, 0.0, 64.0, 64.0]])
    x = noise.clone()
    for t in ddim.timesteps:
        with torch.no_grad():
            out = m.unet(x, t.float().reshape(1), encoder_hidden_states=emb[:, :77],
                         added_cond_kwargs={"text_embeds": emb[:, 77, :POOLED], "time_ids": ids}).sample
        x = ddim.step(out, t, x).prev_sample

    sigmas = SS.sdxl_sigmas(steps, prediction)
    assert sigmas.shape == (steps + 1,) and sigmas[-1] == 0 and torch.all(sigmas[1:] < sigmas[:-1])
    lat = M.patchify(noise * M.init_noise_scale(float(sigmas[0])))
    with torch.no_grad():
        for i in range(steps):
            vel = m(lat, sigmas[i] * 1000.0, emb)
            lat = lat + (sigmas[i + 1] - sigmas[i]) * vel
    assert torch.allclose(M.unpatchify(lat), x, atol=1e-4, rtol=1e-4)


def test_sigmas_zero_terminal_snr_start_at_one():
    s = SS.sdxl_sigmas(10, "v_prediction", True)
    assert float(s[0]) == pytest.approx(1.0, abs=1e-6) and float(s[-1]) == 0.0
    s = SS.sdxl_sigmas(10, "epsilon", False)
    assert float(s[0]) == pytest.approx(0.93596, abs=1e-4)


def test_sample_runs_deterministically_with_cfg_and_refuses_skip_layer():
    m = facade()
    pos, neg = embeds(1, 1), embeds(1, 2)
    mask = torch.ones(1, M.TEXT_ROWS, dtype=torch.bool)
    seen = []
    kw = dict(width=64, height=64, steps=3, cfg=4.0, seed=7, device="cpu", progress=lambda i, n: seen.append((i, n)))
    a = SS.sample(m, pos, mask, neg, mask, **kw)
    b = SS.sample(m, pos, mask, neg, mask, **kw)
    assert a.shape == (1, 16, LAT, LAT) and a.dtype == torch.float32 and torch.isfinite(a).all()
    assert torch.equal(a, b) and seen[-1] == (3, 3)
    off = SS.sample(m, pos, mask, None, None, **dict(kw, cfg=1.0))
    assert not torch.allclose(a, off)
    with pytest.raises(ValueError):
        SS.sample(m, pos, mask, neg, mask, **dict(kw, skip_layer=22))
    for ok in (-1, None):
        SS.sample(m, pos, mask, neg, mask, **dict(kw, steps=1, skip_layer=ok))


# ---------------------------------------------------------------- patchify, VAE, text


def test_patchify_is_the_exact_inverse_and_orders_channel_dy_dx():
    x = torch.randn(2, 4, 8, 12)
    p = M.patchify(x)
    assert p.shape == (2, 16, 4, 6)
    assert torch.equal(M.unpatchify(p), x)
    lat16 = torch.randn(2, 16, 3, 5)
    assert torch.equal(M.patchify(M.unpatchify(lat16)), lat16)
    # feature index = channel * 4 + dy * 2 + dx
    assert p[0, 1 * 4 + 1 * 2 + 0, 2, 3] == x[0, 1, 2 * 2 + 1, 3 * 2 + 0]


def tiny_vae(seed=0):
    torch.manual_seed(seed)
    return AutoencoderKL(**TINY_VAE).eval().requires_grad_(False)


def test_vae_encode_decode_shapes_scaling_and_grad_copy():
    vae = tiny_vae()
    img = torch.rand(2, 3, 64, 64) * 2 - 1
    lat = SL.encode_images(vae, img)
    assert lat.shape == (2, 16, LAT, LAT) and lat.dtype == torch.float32
    mean = vae.encode(img).latent_dist.mode() * 0.13025
    assert torch.allclose(M.unpatchify(lat), mean, atol=1e-6)  # the posterior mean, scaled
    dec = SL.decode_latents(vae, lat)
    assert dec.shape == img.shape and dec.min() >= -1 and dec.max() <= 1
    raw = vae.decode(mean / 0.13025).sample.clamp(-1, 1)
    assert torch.allclose(dec, raw, atol=1e-5)
    lat_g = lat.clone().requires_grad_(True)
    out = SL.decode_latents_grad(vae, lat_g)
    out.mean().backward()
    assert lat_g.grad is not None and torch.isfinite(lat_g.grad).all()
    gv = SL.grad_vae(vae)
    assert next(gv.parameters()).dtype == torch.bfloat16 and gv.is_gradient_checkpointing
    assert SL.decode_latents_grad(gv, lat.bfloat16()).shape == img.shape


def test_pad_embeds_shape_mask_and_row_check():
    rows = [torch.randn(78, 2048), torch.randn(78, 2048)]
    e, mask = SL.pad_embeds(rows)
    assert e.shape == (2, 78, 2048) and mask.shape == (2, 78) and mask.dtype == torch.bool and mask.all()
    assert torch.equal(e[1], rows[1])
    with pytest.raises(ValueError):
        SL.pad_embeds([torch.randn(77, 2048)])


def make_tokenizer_dir(root: Path, pad_g="!") -> Path:
    """Two tiny CLIP tokenizer folders (letters + words), bigG padding with '!' (id 0) like the SDXL repo."""
    words = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "o", "l", "p", "r", "t", "x", "y", "photo", "cat"]
    vocab = {"!": 0, "<|startoftext|>": 1, "<|endoftext|>": 2}
    for w in words:
        vocab[w + "</w>"] = len(vocab)
    for w in words:
        vocab.setdefault(w, len(vocab))
    for name, pad in (("tokenizer", "<|endoftext|>"), ("tokenizer_2", pad_g)):
        d = root / name
        d.mkdir(parents=True)
        (d / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
        (d / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
        (d / "special_tokens_map.json").write_text(json.dumps(
            {"bos_token": "<|startoftext|>", "eos_token": "<|endoftext|>", "pad_token": pad,
             "unk_token": "<|endoftext|>"}), encoding="utf-8")
        (d / "tokenizer_config.json").write_text(json.dumps(
            {"tokenizer_class": "CLIPTokenizer", "model_max_length": 77, "bos_token": "<|startoftext|>",
             "eos_token": "<|endoftext|>", "pad_token": pad, "unk_token": "<|endoftext|>"}), encoding="utf-8")
    return root


def tiny_text_encoders(seed=0):
    torch.manual_seed(seed)
    cl = CLIPTextModel(CLIPTextConfig(**CLIP_L)).eval()
    cg = CLIPTextModelWithProjection(CLIPTextConfig(**CLIP_G)).eval()
    return SL.SdxlTextEncoders(cl, cg).requires_grad_(False)


@pytest.fixture()
def tok(tmp_path):
    root = make_tokenizer_dir(tmp_path / "tok")
    try:
        return SL.SdxlTokenizers.from_dir(root)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"cannot build a tiny CLIP tokenizer here: {e}")


def test_tokenizer_windows_and_padding(tok):
    ids_l, ids_g = tok(["a b", "c"])
    assert ids_l.shape == ids_g.shape == (2, 77)
    assert ids_l[0, 0] == 1 and ids_l[0, 3] == 2 and ids_l[0, 4] == 2  # BOS a b EOS, CLIP-L pads with EOS
    assert ids_g[0, 3] == 2 and ids_g[0, 4] == 0 and ids_g[0, 76] == 0  # bigG pads with "!" (id 0)
    empty_l, empty_g = tok([""])
    assert empty_l[0, 0] == 1 and empty_l[0, 1] == 2 and empty_g[0, 1] == 2 and empty_g[0, 2] == 0


def test_encode_prompt_layout_rows_and_empty_prompt(tok):
    te = tiny_text_encoders()
    e = SL.encode_prompt(te, tok, "a b o a c", 77)
    assert e.shape == (78, 24 + 16) or e.shape == (78, 40)  # tiny dims: CLIP-L 16 + bigG 24 = 40 wide
    assert e.dtype == torch.bfloat16
    # recompute by hand: penultimate hidden states of both, pooled = bigG text_embeds in the last row
    ids_l, ids_g = tok(["a b o a c"])
    with torch.no_grad():
        ol = te.clip_l(input_ids=ids_l, output_hidden_states=True)
        og = te.clip_g(input_ids=ids_g, output_hidden_states=True)
    hidden = torch.cat([ol.hidden_states[-2], og.hidden_states[-2]], dim=-1)[0]
    assert torch.allclose(e[:77].float(), hidden, atol=2e-2, rtol=2e-2)
    assert torch.allclose(e[77, :POOLED].float(), og.text_embeds[0], atol=2e-2, rtol=2e-2)
    assert (e[77, POOLED:] == 0).all()
    assert not torch.allclose(e[:77].float(), og.hidden_states[-1][0].new_zeros(77, 40))
    # a batch is the same as single prompts, and the empty prompt is ENCODED (not zeros)
    batch = SL.encode_prompts(te, tok, ["a b o a c", ""], 77)
    assert len(batch) == 2 and torch.allclose(batch[0].float(), e.float(), atol=1e-2)
    empty = SL.encode_prompt(te, tok, "", 77)
    assert empty.abs().sum() > 0 and not torch.allclose(empty.float(), batch[0].float())
    # final layer norm is NOT applied: hidden_states[-2] differs from last_hidden_state
    assert not torch.allclose(ol.hidden_states[-2], ol.last_hidden_state)


def test_long_caption_is_cut_with_one_warning(tok, caplog):
    te = tiny_text_encoders()
    long = " ".join(["c"] * 120)
    with caplog.at_level(logging.WARNING, logger="rulztrainer"):
        a = SL.encode_prompt(te, tok, long, 77)
        SL.encode_prompt(te, tok, long, 77)
    warnings = [r for r in caplog.records if "77-token" in r.getMessage()]
    assert len(warnings) == 1
    cut = SL.encode_prompt(te, tok, " ".join(["c"] * 75), 77)
    assert torch.allclose(a.float(), cut.float(), atol=1e-5)  # 75 tokens + EOS is exactly the window


def test_prompt_tokens(tok):
    text, offsets, dropped, length = SL.prompt_tokens(tok, "a b c", 77)
    assert text == "a b c" and dropped == 0 and length == len(offsets) == 5  # BOS a b c EOS
    assert offsets == [(0, 0), (0, 1), (2, 3), (4, 5), (0, 0)]
    long = " ".join(["c"] * 120)
    assert SL.prompt_tokens(tok, long, 77)[3] == 77


def test_composite_text_encoder_is_one_module():
    te = tiny_text_encoders()
    n = sum(p.numel() for p in te.parameters())
    assert n == sum(p.numel() for p in te.clip_l.parameters()) + sum(p.numel() for p in te.clip_g.parameters())
    assert te.input_device.type == "cpu"
    te.to(torch.bfloat16)
    assert all(p.dtype == torch.bfloat16 for p in te.parameters())


# ---------------------------------------------------------------- the diffusers folder


def save_tiny_repo(root: Path, variant=None, scheduler=None) -> Path:
    unet = tiny_unet()
    unet.save_pretrained(root / "unet", variant=variant)
    tiny_vae().save_pretrained(root / "vae", variant=variant)
    torch.manual_seed(0)
    CLIPTextModel(CLIPTextConfig(**CLIP_L)).save_pretrained(root / "text_encoder", variant=variant)
    CLIPTextModelWithProjection(CLIPTextConfig(**CLIP_G)).save_pretrained(root / "text_encoder_2", variant=variant)
    if scheduler is not None:
        (root / "scheduler").mkdir()
        (root / "scheduler" / "scheduler_config.json").write_text(json.dumps(scheduler), encoding="utf-8")
    return root


def cfg_for(tmp, **kw):
    base = dict(model_source="hf", model_path="", models_dir=str(tmp / "models"), device="cpu", transformer_file="",
                text_encoder_file="", vae_file="", tokenizer_path="", text_encoder_device="cpu",
                multi_gpu="off", transformer_quant="none")
    base.update(kw)
    return SimpleNamespace(**base)


def test_get_transformer_from_folder_accepts_extra_args_and_reads_scheduler(tmp_path):
    root = save_tiny_repo(tmp_path / "repo", scheduler={"prediction_type": "v_prediction",
                                                        "rescale_betas_zero_snr": True})
    assert SL.is_sdxl_dir(root)
    cfg = cfg_for(tmp_path, model_path=str(root))
    assert SL.resolve_for_config(cfg) == root.resolve()
    model = SL.get_transformer(cfg, root, device="cpu", model_split=True, quantize_staged_on="cuda:0", staging=1)
    assert isinstance(model, M.SdxlUNet) and model.prediction_type == "v_prediction" and model.zero_terminal_snr
    assert model.dtype == torch.bfloat16 and not any(p.requires_grad for p in model.parameters())
    assert not model.training
    # the loaded weights are the saved ones (bf16 rounding)
    ref = tiny_unet()
    assert torch.allclose(model.unet.conv_in.weight.float(), ref.conv_in.weight.bfloat16().float())
    out = model(torch.randn(1, 16, LAT, LAT).bfloat16(), 400.0, embeds(1).bfloat16())
    assert out.dtype == torch.float32 and out.shape == (1, 16, LAT, LAT)
    vae = SL.get_vae(cfg, root)
    assert next(vae.parameters()).dtype == torch.float32
    anyload = SL.load_any_transformer(str(root), "cpu")
    assert anyload.prediction_type == "v_prediction"


def test_fp16_variant_files_are_found(tmp_path):
    root = save_tiny_repo(tmp_path / "repo16", variant="fp16")
    assert (root / "unet" / "diffusion_pytorch_model.fp16.safetensors").is_file()
    model = SL.load_unet_dir(root, "cpu")
    assert model.dtype == torch.bfloat16
    vae = SL.load_vae_dir(root, "cpu")
    assert next(vae.parameters()).dtype == torch.float32
    clip_l, clip_g = SL.load_text_models_dir(root, "cpu")
    assert clip_l.dtype == torch.bfloat16 and isinstance(clip_g, CLIPTextModelWithProjection)
    assert SL.component_ready(root) is False  # no tokenizers in this folder yet
    make_tokenizer_dir(root)
    assert SL.component_ready(root) is True


def test_resolve_prediction_rules(tmp_path):
    def cfg(pred="auto", z=False):
        return SimpleNamespace(sdxl_prediction=pred, sdxl_zero_terminal_snr=z)

    d = tmp_path / "m"
    (d / "scheduler").mkdir(parents=True)
    sched = d / "scheduler" / "scheduler_config.json"
    sched.write_text(json.dumps({"prediction_type": "v_prediction", "rescale_betas_zero_snr": True}))
    assert SL.resolve_prediction(cfg(), d) == ("v_prediction", True)
    assert SL.resolve_prediction(cfg("epsilon"), d) == ("epsilon", True)  # explicit type; the schedule still read
    sched.write_text(json.dumps({"prediction_type": "epsilon"}))
    assert SL.resolve_prediction(cfg(), d) == ("epsilon", False)
    assert SL.resolve_prediction(cfg("v_prediction", True), d) == ("v_prediction", True)
    sched.write_text(json.dumps({"prediction_type": "sample"}))
    with pytest.raises(ValueError):
        SL.resolve_prediction(cfg(), d)
    assert SL.resolve_prediction(cfg("epsilon"), d)[0] == "epsilon"
    sched.unlink()
    assert SL.resolve_prediction(cfg(), d) == ("epsilon", False)
    assert SL.resolve_prediction(SimpleNamespace(), None) == ("epsilon", False)  # fields not added yet: defaults
    assert SL.resolve_prediction(cfg(), tmp_path / "missing.safetensors") == ("epsilon", False)
    with pytest.raises(ValueError):
        SL.resolve_prediction(cfg("bogus"), d)
    f = tmp_path / "vp.safetensors"
    safetensors_torch.save_file({"v_pred": torch.zeros(1), "ztsnr": torch.zeros(1),
                                 "model.diffusion_model.label_emb.0.0.weight": torch.zeros(2, 2)}, str(f))
    assert SL.resolve_prediction(cfg(), f) == ("v_prediction", True)


def test_plan_download_picks_one_precision_and_no_all_in_one():
    files = ["model_index.json", "sd_xl_base_1.0.safetensors", "sd_xl_base_1.0_0.9vae.safetensors",
             "sd_xl_offset_example-lora_1.0.safetensors", "README.md", ".gitattributes",
             "unet/config.json", "unet/diffusion_pytorch_model.safetensors", "unet/diffusion_pytorch_model.fp16.safetensors",
             "unet/diffusion_pytorch_model.bin", "unet/diffusion_pytorch_model.fp16.bin", "unet/model.onnx",
             "vae/config.json", "vae/diffusion_pytorch_model.safetensors", "vae/diffusion_pytorch_model.fp16.safetensors",
             "vae_1_0/config.json", "vae_1_0/diffusion_pytorch_model.fp16.safetensors",
             "text_encoder/config.json", "text_encoder/model.safetensors", "text_encoder/model.fp16.safetensors",
             "text_encoder/openvino_model.xml", "text_encoder_2/config.json", "text_encoder_2/model.safetensors",
             "text_encoder_2/model.fp16.safetensors", "tokenizer/vocab.json", "tokenizer/merges.txt",
             "tokenizer/tokenizer_config.json", "tokenizer/special_tokens_map.json", "tokenizer_2/vocab.json",
             "tokenizer_2/merges.txt", "tokenizer_2/tokenizer_config.json", "tokenizer_2/special_tokens_map.json",
             "scheduler/scheduler_config.json"]
    plan = SL.plan_download(files)
    assert "unet/diffusion_pytorch_model.fp16.safetensors" in plan and "unet/diffusion_pytorch_model.safetensors" not in plan
    assert "vae/diffusion_pytorch_model.safetensors" in plan and "vae/diffusion_pytorch_model.fp16.safetensors" not in plan
    assert "text_encoder/model.fp16.safetensors" in plan and "text_encoder_2/model.fp16.safetensors" in plan
    assert "text_encoder/model.safetensors" not in plan
    assert not any(f.startswith("sd_xl") or f.endswith((".bin", ".onnx", ".xml")) or f.startswith("vae_1_0") for f in plan)
    for need in ("model_index.json", "unet/config.json", "vae/config.json", "text_encoder/config.json",
                 "text_encoder_2/config.json", "tokenizer/vocab.json", "tokenizer_2/merges.txt",
                 "scheduler/scheduler_config.json"):
        assert need in plan
    # a repo that only has plain weights (a finetune) gets those
    plain_only = [f for f in files if ".fp16." not in f]
    plan2 = SL.plan_download(plain_only)
    assert "unet/diffusion_pytorch_model.safetensors" in plan2 and "text_encoder/model.safetensors" in plan2
    # sharded
    sharded = ["unet/config.json", "unet/diffusion_pytorch_model-00001-of-00002.safetensors",
               "unet/diffusion_pytorch_model-00002-of-00002.safetensors",
               "unet/diffusion_pytorch_model.safetensors.index.json"]
    plan3 = SL.plan_download(sharded)
    assert all(f in plan3 for f in sharded)


def test_tokenizers_dir_resolution(tmp_path):
    root = make_tokenizer_dir(tmp_path / "repo")
    cfg = cfg_for(tmp_path)
    assert SL.resolve_tokenizers_dir(cfg, root) == root
    cfg2 = cfg_for(tmp_path, tokenizer_path=str(root))
    assert SL.resolve_tokenizers_dir(cfg2, None) == root
    with pytest.raises(FileNotFoundError):
        SL.resolve_tokenizers_dir(cfg_for(tmp_path, tokenizer_path=str(tmp_path / "nope")), None)
    models = tmp_path / "models"
    make_tokenizer_dir(models / "stabilityai__stable-diffusion-xl-base-1.0")
    assert SL.resolve_tokenizers_dir(cfg, None).name == "stabilityai__stable-diffusion-xl-base-1.0"


def test_resolve_for_config_files_mode(tmp_path):
    cfg = cfg_for(tmp_path, model_source="files", transformer_file="x.safetensors")
    assert SL.resolve_for_config(cfg) is None
    assert SL._checkpoint(cfg, "vae_file") == "x.safetensors"
    assert SL._checkpoint(cfg_for(tmp_path, vae_file="v.safetensors"), "vae_file") == ""  # hf mode ignores files
    cfg3 = cfg_for(tmp_path, model_source="files", transformer_file="x.safetensors", vae_file="vae.safetensors")
    assert SL._checkpoint(cfg3, "vae_file") == "vae.safetensors"


# ---------------------------------------------------------------- the all-in-one checkpoint


RESNET = (("norm1", "in_layers.0"), ("conv1", "in_layers.2"), ("time_emb_proj", "emb_layers.1"),
          ("norm2", "out_layers.0"), ("conv2", "out_layers.3"), ("conv_shortcut", "skip_connection"))


def unet_to_ldm(sd: dict, layers_per_block: int) -> dict:
    """diffusers UNet state dict -> the LDM / A1111 naming of an SDXL checkpoint (reverse of diffusers' converter)."""
    L, out = layers_per_block, {}
    top = {"time_embedding.linear_1": "time_embed.0", "time_embedding.linear_2": "time_embed.2",
           "add_embedding.linear_1": "label_emb.0.0", "add_embedding.linear_2": "label_emb.0.2",
           "conv_in": "input_blocks.0.0", "conv_norm_out": "out.0", "conv_out": "out.2"}
    for k, v in sd.items():
        mod, _, leaf = k.rpartition(".")
        if mod in top:
            out[f"{top[mod]}.{leaf}"] = v
            continue
        p = mod.split(".")
        if p[0] == "down_blocks":
            i, kind, j = int(p[1]), p[2], int(p[3]) if p[2] != "downsamplers" else 0
            if kind == "downsamplers":
                n, rest = 1 + i * (L + 1) + L, ["0", "op"]
            else:
                n = 1 + i * (L + 1) + j
                rest = (["0"] + [dict(RESNET)[p[4]] if p[4] in dict(RESNET) else p[4]] if kind == "resnets"
                        else ["1"] + p[4:])
            if kind == "resnets":
                rest = ["0", dict(RESNET)[p[4]]]
            out[".".join(["input_blocks", str(n), *rest, leaf])] = v
        elif p[0] == "mid_block":
            kind, j = p[1], int(p[2])
            if kind == "resnets":
                out[".".join(["middle_block", str(0 if j == 0 else 2), dict(RESNET)[p[3]], leaf])] = v
            else:
                out[".".join(["middle_block", "1", *p[3:], leaf])] = v
        elif p[0] == "up_blocks":
            i, kind = int(p[1]), p[2]
            if kind == "upsamplers":
                n = i * (L + 1) + L
                has_attn = any(x.startswith(f"up_blocks.{i}.attentions") for x in sd)
                out[f"output_blocks.{n}.{2 if has_attn else 1}.conv.{leaf}"] = v
            elif kind == "resnets":
                n = i * (L + 1) + int(p[3])
                out[".".join(["output_blocks", str(n), "0", dict(RESNET)[p[4]], leaf])] = v
            else:
                n = i * (L + 1) + int(p[3])
                out[".".join(["output_blocks", str(n), "1", *p[4:], leaf])] = v
        else:
            raise AssertionError(f"unmapped key {k}")
    return out


def vae_to_ldm(sd: dict, n_blocks: int) -> dict:
    out = {}
    rn = {"conv_shortcut": "nin_shortcut"}
    for k, v in sd.items():
        mod, _, leaf = k.rpartition(".")
        p = mod.split(".")
        if p[0] in ("quant_conv", "post_quant_conv"):
            out[k] = v
        elif p[1] in ("conv_in", "conv_out"):
            out[k] = v
        elif p[1] == "conv_norm_out":
            out[f"{p[0]}.norm_out.{leaf}"] = v
        elif p[1] == "down_blocks":
            i = int(p[2])
            if p[3] == "resnets":
                out[f"encoder.down.{i}.block.{p[4]}.{rn.get(p[5], p[5])}.{leaf}"] = v
            else:
                out[f"encoder.down.{i}.downsample.conv.{leaf}"] = v
        elif p[1] == "up_blocks":
            i = n_blocks - 1 - int(p[2])
            if p[3] == "resnets":
                out[f"decoder.up.{i}.block.{p[4]}.{rn.get(p[5], p[5])}.{leaf}"] = v
            else:
                out[f"decoder.up.{i}.upsample.conv.{leaf}"] = v
        elif p[1] == "mid_block":
            if p[2] == "resnets":
                out[f"{p[0]}.mid.block_{int(p[3]) + 1}.{rn.get(p[4], p[4])}.{leaf}"] = v
            else:
                name = {"group_norm": "norm", "to_q": "q", "to_k": "k", "to_v": "v", "to_out": "proj_out"}[p[4]]
                if p[4] == "to_out":
                    assert p[5] == "0"
                if leaf == "weight" and v.ndim == 2:
                    v = v[:, :, None, None]
                out[f"{p[0]}.mid.attn_1.{name}.{leaf}"] = v
        else:
            raise AssertionError(f"unmapped key {k}")
    return out


def clipg_to_openclip(sd: dict, prefix: str) -> dict:
    out = {}
    pending = {}
    for k, v in sd.items():
        if k.endswith("position_ids"):
            continue
        if k == "text_projection.weight":
            out[prefix + "text_projection"] = v.T.contiguous()
        elif k == "text_model.embeddings.position_embedding.weight":
            out[prefix + "positional_embedding"] = v
        elif k == "text_model.embeddings.token_embedding.weight":
            out[prefix + "token_embedding.weight"] = v
        elif k.startswith("text_model.final_layer_norm."):
            out[prefix + "ln_final." + k.split(".")[-1]] = v
        else:
            rest = k[len("text_model.encoder.layers."):]
            n, _, tail = rest.partition(".")
            base = f"{prefix}transformer.resblocks.{n}."
            if tail.startswith("self_attn.") and tail.split(".")[1] in ("q_proj", "k_proj", "v_proj"):
                _, which, leaf = tail.split(".")
                pending.setdefault((n, leaf), {})[which] = v
            else:
                tail = (tail.replace("layer_norm1", "ln_1").replace("layer_norm2", "ln_2")
                        .replace("mlp.fc1", "mlp.c_fc").replace("mlp.fc2", "mlp.c_proj").replace("self_attn", "attn"))
                out[base + tail] = v
    for (n, leaf), parts in pending.items():
        out[f"{prefix}transformer.resblocks.{n}.attn.in_proj_{leaf}"] = torch.cat(
            [parts["q_proj"], parts["k_proj"], parts["v_proj"]], dim=0)
    return out


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory):
    d = tmp_path_factory.mktemp("ckpt")
    unet, vae = tiny_unet(3), tiny_vae(3)
    torch.manual_seed(3)
    cl = CLIPTextModel(CLIPTextConfig(**CLIP_L)).eval()
    cg = CLIPTextModelWithProjection(CLIPTextConfig(**CLIP_G)).eval()
    ldm = {"model.diffusion_model." + k: v.contiguous() for k, v in unet_to_ldm(unet.state_dict(), 2).items()}
    ldm.update({"first_stage_model." + k: v.contiguous() for k, v in vae_to_ldm(vae.state_dict(), 4).items()})
    # real checkpoints hold CLIP-L as transformer.text_model.* (the layout of transformers 4), whatever layout the
    # installed transformers gives CLIPTextModel (5.x: flat)
    ldm.update({"conditioner.embedders.0.transformer." + ("" if k.startswith("text_model.") else "text_model.") + k:
                v.contiguous() for k, v in cl.state_dict().items() if not k.endswith("position_ids")})
    ldm.update({k: v.contiguous() for k, v in clipg_to_openclip(cg.state_dict(),
                                                                "conditioner.embedders.1.model.").items()})
    full = d / "all.safetensors"
    safetensors_torch.save_file(ldm, str(full))
    vp = d / "vp.safetensors"
    safetensors_torch.save_file(dict(ldm, v_pred=torch.zeros(1), ztsnr=torch.zeros(1)), str(vp),
                                metadata={"modelspec.prediction_type": "v"})
    bare = d / "bare.safetensors"
    safetensors_torch.save_file({k[len("model.diffusion_model."):]: v for k, v in ldm.items()
                                 if k.startswith("model.diffusion_model.")}, str(bare))
    vae_only = d / "vae_only.safetensors"
    safetensors_torch.save_file({k[len("first_stage_model."):]: v for k, v in ldm.items()
                                 if k.startswith("first_stage_model.")}, str(vae_only))
    return SimpleNamespace(full=full, vp=vp, bare=bare, vae_only=vae_only, unet=unet, vae=vae, cl=cl, cg=cg, dir=d)


def test_is_sdxl_file_and_file_prediction_on_synthetic_headers(tmp_path):
    ok = tmp_path / "ok.safetensors"
    safetensors_torch.save_file({"model.diffusion_model.label_emb.0.0.weight": torch.zeros(2, 2),
                                 "model.diffusion_model.input_blocks.0.0.weight": torch.zeros(1)}, str(ok))
    sd15 = tmp_path / "sd15.safetensors"
    safetensors_torch.save_file({"model.diffusion_model.input_blocks.0.0.weight": torch.zeros(1)}, str(sd15))
    bare = tmp_path / "bare.safetensors"
    safetensors_torch.save_file({"label_emb.0.0.weight": torch.zeros(2, 2), "input_blocks.0.0.weight": torch.zeros(1)},
                                str(bare))
    junk = tmp_path / "junk.safetensors"
    junk.write_bytes(b"not a safetensors file")
    assert SF.is_sdxl_file(ok) and SF.is_sdxl_file(bare)
    assert not SF.is_sdxl_file(sd15) and not SF.is_sdxl_file(junk) and not SF.is_sdxl_file(tmp_path / "none.bin")
    assert SF.file_prediction(ok) == (None, None)
    v = tmp_path / "v.safetensors"
    safetensors_torch.save_file({"v_pred": torch.zeros(1), "ztsnr": torch.zeros(1), "x": torch.zeros(1)}, str(v))
    assert SF.file_prediction(v) == ("v_prediction", True)
    e = tmp_path / "e.safetensors"
    safetensors_torch.save_file({"x": torch.zeros(1)}, str(e), metadata={"modelspec.prediction_type": "epsilon"})
    assert SF.file_prediction(e) == ("epsilon", None)


def test_single_file_unet_round_trip_matches_the_original(tiny_checkpoint):
    cfg = dict(tiny_checkpoint.unet.config)
    assert SF.is_sdxl_file(tiny_checkpoint.full)
    for path in (tiny_checkpoint.full, tiny_checkpoint.bare):
        unet = SF.load_unet_file(path, "cpu", torch.float32, config=cfg)
        assert not any(p.requires_grad for p in unet.parameters()) and not unet.training
        for (k1, v1), (k2, v2) in zip(sorted(unet.state_dict().items()),
                                      sorted(tiny_checkpoint.unet.state_dict().items())):
            assert k1 == k2 and torch.equal(v1, v2), k1
    # and through the facade it gives the same velocity as the original model
    new = M.SdxlUNet(SF.load_unet_file(tiny_checkpoint.full, "cpu", torch.float32, config=cfg))
    old = M.SdxlUNet(tiny_checkpoint.unet)
    lat, emb = torch.randn(2, 16, LAT, LAT), embeds()
    t = torch.tensor([250.0, 640.0])
    assert torch.allclose(new(lat, t, emb), old(lat, t, emb), atol=1e-5)


def test_single_file_vae_round_trip(tiny_checkpoint):
    for path in (tiny_checkpoint.full, tiny_checkpoint.vae_only):
        vae = SF.load_vae_file(path, "cpu", config=dict(tiny_checkpoint.vae.config))
        assert next(vae.parameters()).dtype == torch.float32
        for k, v in tiny_checkpoint.vae.state_dict().items():
            assert torch.allclose(vae.state_dict()[k], v), k
    img = torch.rand(1, 3, 64, 64) * 2 - 1
    vae = SF.load_vae_file(tiny_checkpoint.full, "cpu", config=dict(tiny_checkpoint.vae.config))
    assert torch.allclose(SL.encode_images(vae, img), SL.encode_images(tiny_checkpoint.vae, img), atol=1e-6)


def test_single_file_text_encoders_round_trip(tiny_checkpoint):
    clip_l, clip_g = SF.load_text_encoders_file(tiny_checkpoint.full, "cpu", torch.float32,
                                                configs=(CLIP_L, CLIP_G))
    assert isinstance(clip_l, CLIPTextModel) and isinstance(clip_g, CLIPTextModelWithProjection)
    ids = torch.randint(3, 80, (2, 77))
    with torch.no_grad():
        ref_l = tiny_checkpoint.cl(input_ids=ids, output_hidden_states=True)
        got_l = clip_l(input_ids=ids, output_hidden_states=True)
        ref_g = tiny_checkpoint.cg(input_ids=ids, output_hidden_states=True)
        got_g = clip_g(input_ids=ids, output_hidden_states=True)
    assert torch.allclose(ref_l.hidden_states[-2], got_l.hidden_states[-2], atol=1e-6)
    assert torch.allclose(ref_g.hidden_states[-2], got_g.hidden_states[-2], atol=1e-6)
    assert torch.allclose(ref_g.text_embeds, got_g.text_embeds, atol=1e-6)  # the projection survived the transpose


def test_get_functions_in_files_mode_use_the_all_in_one_file(tiny_checkpoint, tmp_path, monkeypatch):
    # bundled configs are the full SDXL ones: point the loaders at the tiny ones for this test
    unet_cfg, vae_cfg = dict(tiny_checkpoint.unet.config), dict(tiny_checkpoint.vae.config)
    real_unet, real_vae, real_te = SF.load_unet_file, SF.load_vae_file, SF.load_text_encoders_file
    monkeypatch.setattr(SF, "load_unet_file", lambda p, d, dt=torch.bfloat16, config=None: real_unet(p, d, dt, unet_cfg))
    monkeypatch.setattr(SF, "load_vae_file", lambda p, d, config=None: real_vae(p, d, vae_cfg))
    monkeypatch.setattr(SF, "load_text_encoders_file",
                        lambda p, d, dt=torch.bfloat16, configs=None: real_te(p, d, dt, (CLIP_L, CLIP_G)))
    make_tokenizer_dir(tmp_path / "tok")
    cfg = cfg_for(tmp_path, model_source="files", transformer_file=str(tiny_checkpoint.vp), tokenizer_path=str(tmp_path / "tok"))
    assert SL.resolve_for_config(cfg) is None
    model = SL.get_transformer(cfg, None, device="cpu", model_split=False)
    assert model.prediction_type == "v_prediction" and model.zero_terminal_snr and model.dtype == torch.bfloat16
    cfg_auto_eps = cfg_for(tmp_path, model_source="files", transformer_file=str(tiny_checkpoint.vp), sdxl_prediction="epsilon")
    assert SL.get_transformer(cfg_auto_eps, None, "cpu").prediction_type == "epsilon"
    vae = SL.get_vae(cfg, None)
    assert next(vae.parameters()).dtype == torch.float32
    te, tk = SL.get_text_encoder(cfg, None)
    assert isinstance(te, SL.SdxlTextEncoders) and isinstance(tk, SL.SdxlTokenizers)
    row = SL.encode_prompt(te, tk, "a c", 77)
    assert row.shape == (78, 40)
    with pytest.raises(FileNotFoundError):
        SL.get_transformer(cfg_for(tmp_path, model_source="files", transformer_file=str(tmp_path / "no.safetensors")), None, "cpu")


# ---------------------------------------------------------------- bundled configs on the meta device


def test_bundled_configs_build_the_real_sdxl_components_on_meta():
    unet_cfg = json.loads((Path(SF.HERE) / "unet_config.json").read_text(encoding="utf-8"))
    assert unet_cfg == M.unet_config()
    with torch.device("meta"):
        unet = UNet2DConditionModel.from_config(unet_cfg)
        vae = AutoencoderKL.from_config(SF._config("vae_config.json"))
        clip_l = CLIPTextModel(CLIPTextConfig.from_dict(SF._config("clip_l_config.json")))
        clip_g = CLIPTextModelWithProjection(CLIPTextConfig.from_dict(SF._config("clip_g_config.json")))
    count = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    assert 2.55e9 < count(unet) < 2.60e9
    assert 83e6 < count(vae) < 84.5e6 and vae.config.scaling_factor == 0.13025
    assert 122e6 < count(clip_l) < 124e6  # CLIP ViT-L/14 text tower: 123.06 M
    assert 690e6 < count(clip_g) < 700e6  # OpenCLIP bigG text tower + projection: 694.7 M
    assert clip_g.config.hidden_size == 1280 and clip_l.config.hidden_size == 768
    # the all-in-one converter keys cover the full SDXL UNet (names map one to one)
    from diffusers.loaders.single_file_utils import convert_ldm_unet_checkpoint
    fake = {"model.diffusion_model." + k: torch.empty(0, device="meta") for k in
            unet_to_ldm({k: v for k, v in unet.state_dict().items()}, 2)}
    conv = convert_ldm_unet_checkpoint(fake, unet_cfg)
    assert set(conv) == set(unet.state_dict())
