"""MiniMax H3 support without model weights: the ported transformer against diffusers' MiniMaxH3Transformer3DModel
(tiny random config, fp32), fed the packed one-frame layout diffusers' own pipeline builds
(MiniMaxH3PrepareLayoutStep.build_packed_sequence, MiniMaxH3SetTimestepsStep.build_row_timesteps), per sample with
padded text; the pruned AdaLN table path; the layouts (diffusers <-> original, Comfy-Org's single files, MiniMaxAI's
interleaved qkv) against diffusers' own converters; LoRA naming and the export layout the ecosystem loads; the training
adapter and the turbo LoRA in their published key layouts; the sampler schedule against MiniMaxH3Scheduler; the
VAE layouts; architecture detection, caches and the 32 px buckets."""
import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import archs  # noqa: E402
from rulztrainer import lora as L  # noqa: E402
from rulztrainer.config import TrainConfig  # noqa: E402
from rulztrainer.h3 import convert as C  # noqa: E402
from rulztrainer.h3.model import (AUDIO_TAG, TEXT_TAG, VIDEO_TAG, MiniMaxH3Transformer, audio_sigma,  # noqa: E402
                                  patchify, unpatchify)

diffusers = pytest.importorskip("diffusers")

# a tiny H3: the attention is wider than the residual stream (as the real 56 x 128 > 5376), 2 x 3 x 4 rotary
# channels of 32 per head, the real 24 video / 32 audio latent channels and the 256-wide sinusoidal time input
TINY_DIFFUSERS = dict(num_attention_heads=2, attention_head_dim=32, hidden_size=48, num_layers=2, num_refiner_layers=1,
                      ffn_dim=64, in_channels=24, audio_in_channels=32, patch_size=(1, 2, 2), text_dim=40,
                      freq_dim=256, time_embed_hidden_dim=56, time_embed_dim=24, rope_freq_dim=4, rope_theta=10000.0)
TINY = C.diffusers_config_to_port(TINY_DIFFUSERS)


def pair(seed=0):
    """Our port and diffusers' model with the same random weights, fp32."""
    torch.manual_seed(seed)
    ref = diffusers.MiniMaxH3Transformer3DModel(**TINY_DIFFUSERS).float().eval()
    with torch.no_grad():
        for p in ref.parameters():
            p.normal_(0, 0.1)
    ours = MiniMaxH3Transformer(**TINY).float().eval()
    missing, unexpected = ours.load_state_dict(C.diffusers_to_port(dict(ref.state_dict())), strict=True)
    assert not missing and not unexpected
    return ours, ref


def ref_call(ref, x, sigma, enc, audio_rows, times=None):
    """diffusers' forward for one sample on the pipeline's own one-frame t2va layout: [text | 2 x 2 audio rows |
    one latent frame], the row timesteps of its plan (video t = 1 - sigma for text and video, the shifted audio time
    for the audio rows; or times = (t_video, t_audio)). Returns the video velocity [1, 24, h, w] and the audio rows'
    [1, 4, 32], data-ward."""
    from diffusers.modular_pipelines.minimax_h3.before_denoise import (MiniMaxH3PrepareLayoutStep,
                                                                       MiniMaxH3SetTimestepsStep,
                                                                       patchify_video_latents)

    _, c, h, w = x.shape
    n_text = enc.shape[1]
    pos, tags, vid_idx, aud_idx, txt_idx, n_cv, n_ca = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
        torch.full((n_text,), TEXT_TAG), num_latent_frames=1, latent_height=h, latent_width=w, num_audio_latents=2,
        patch_size=(1, 2, 2), audio_channels=2, audio_tag=AUDIO_TAG, video_tag=VIDEO_TAG)
    if times is not None:
        t_v, t_a = (float(t) for t in times)
    else:
        t_v = 1.0 - float(sigma)
        t_a = 1.0 - float(audio_sigma(torch.tensor(float(sigma))))
    ts, idx = MiniMaxH3SetTimestepsStep.build_row_timesteps(vid_idx, aud_idx, n_cv, n_ca, n_text, t_v, t_a,
                                                            max(t_v, 0.999), 1.0)
    rows = patchify_video_latents(x.unsqueeze(2), (1, 2, 2)).unsqueeze(0)
    out = ref(hidden_states=rows, audio_hidden_states=audio_rows, encoder_hidden_states=enc, timestep=ts,
              timestep_indices=idx, token_tags=tags, position_ids=pos.float(), video_indices=vid_idx,
              audio_indices=aud_idx, text_indices=txt_idx, return_dict=False)
    return unpatchify(out[0], c, h, w), out[1]


def test_layout_helpers_match_the_pipeline():
    from diffusers.modular_pipelines.minimax_h3.before_denoise import (MiniMaxH3PrepareLayoutStep,
                                                                       patchify_video_latents)

    ours = MiniMaxH3Transformer(**TINY)
    for h, w, n_text in ((4, 6, 7), (6, 4, 3), (8, 8, 1)):
        pos = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
            torch.ones(n_text, dtype=torch.long), 1, h, w, 2, (1, 2, 2), 2, AUDIO_TAG, VIDEO_TAG)[0]
        got = ours.positions(torch.tensor([n_text]), n_text, h, w)[0]
        assert torch.equal(got, pos)  # the same float64 grid, bit for bit
        x = torch.randn(2, 24, h, w)
        rows = patchify_video_latents(x.unsqueeze(2), (1, 2, 2)).reshape(2, -1, 96)
        assert torch.equal(patchify(x), rows) and torch.equal(unpatchify(rows, 24, h, w), x)


def test_port_matches_diffusers_per_sample_with_padding():
    ours, ref = pair()
    x = torch.randn(2, 24, 4, 6)
    sigma = torch.tensor([0.93, 0.31])
    lens = (9, 5)
    enc = torch.randn(2, max(lens), 40)
    mask = torch.zeros(2, max(lens), dtype=torch.bool)
    for i, n in enumerate(lens):
        mask[i, :n] = True
    audio = torch.randn(2, 4, 32)
    with torch.no_grad():
        got, got_audio = ours(x, sigma * 1000.0, enc, mask, audio_states=audio, return_audio=True)
        for i, n in enumerate(lens):  # each sample as the reference computes it alone, without the padding
            want, want_audio = ref_call(ref, x[i:i + 1], sigma[i], enc[i:i + 1, :n], audio[i:i + 1])
            assert torch.allclose(got[i:i + 1], -want, atol=2e-5, rtol=1e-4)  # noise-ward = -(data-ward)
            assert torch.allclose(got_audio[i:i + 1], -want_audio, atol=2e-5, rtol=1e-4)
        # the mask matters (padding seen as text changes the shorter sample), and an all-true mask is no mask
        unmasked = ours(x, sigma * 1000.0, enc, None, audio_states=audio)
        assert (unmasked[1] - got[1]).abs().max() > 1e-4
        assert torch.equal(ours(x[:1], 700.0, enc[:1], mask[:1], audio_states=audio[:1]),
                           ours(x[:1], 700.0, enc[:1], None, audio_states=audio[:1]))


def test_audio_placeholder_is_the_noised_silence():
    ours, ref = pair(1)
    silence = torch.randn(4, 32)
    ours.audio_silence = silence
    x, enc = torch.randn(1, 24, 4, 4), torch.randn(1, 5, 40)
    sigma = torch.tensor([0.8])
    s_a = audio_sigma(sigma)
    assert abs(float(s_a) - 3 * (0.8 / (12 - 11 * 0.8)) / (1 + 2 * (0.8 / (12 - 11 * 0.8)))) < 1e-6
    seed = ours.audio_noise_seed
    with torch.no_grad():
        got = ours(x, sigma * 1000.0, enc, None)
        noise = torch.randn(1, 4, 32, generator=torch.Generator().manual_seed(seed))
        rows = (1 - s_a) * silence + s_a * noise
        want, _ = ref_call(ref, x, sigma[0], enc, rows)
        assert torch.allclose(got, -want, atol=2e-5, rtol=1e-4)
        assert ours.audio_noise_seed == seed + 1  # a fresh draw per forward
        assert not torch.equal(ours(x, sigma * 1000.0, enc, None), got)
    # the shifts agree with the released schedulers at every grid point (one base time, two shifts)
    from diffusers import MiniMaxH3Scheduler

    v, a = MiniMaxH3Scheduler(shift=12.0), MiniMaxH3Scheduler(shift=3.0)
    v.set_timesteps(31)
    a.set_timesteps(31)
    assert torch.allclose(audio_sigma(v.sigmas), a.sigmas, atol=1e-5)


def test_bf16_gradients_checkpointing_and_layers():
    ours, _ = pair(2)
    m = ours.to(torch.bfloat16)
    m.gradient_checkpointing = True
    m.train()
    for p in m.parameters():
        p.requires_grad_(True)
    x = torch.randn(1, 24, 4, 6).to(torch.bfloat16)
    enc = torch.randn(1, 4, 40).to(torch.bfloat16)
    out = m(x, torch.tensor([500.0]), enc, None)
    assert out.shape == x.shape and out.dtype == torch.float32 and torch.isfinite(out).all()
    out.pow(2).mean().backward()
    assert all(p.grad is not None for p in m.blocks.parameters())
    with torch.no_grad():
        skipped = m(x, 500.0, enc, None, skip_layers=[1])
    assert skipped.shape == out.shape
    assert m.layers is m.blocks and len(m.layers) == 2 and m.latent_channels == 24
    assert m.split_tail_modules() == [m.final_layer]
    with pytest.raises(ValueError):
        m(torch.randn(1, 24, 3, 4), 500.0, enc, None)  # an odd latent grid is not a 32 px image


def test_model_split_hand_off_matches():
    from rulztrainer.multigpu import split_model

    ours, _ = pair(3)
    x, enc, audio = torch.randn(1, 24, 2, 2), torch.randn(1, 4, 40), torch.randn(1, 4, 32)
    with torch.no_grad():
        want = ours(x, 300.0, enc, None, audio_states=audio)
        split_model(ours, ["cpu", "cpu"])  # same device twice: exercises the hand-off path
        assert torch.equal(ours(x, 300.0, enc, None, audio_states=audio), want)


def pruned_pair(seed=4):
    """A pruned port whose curve table is the full model's SiLU'd time-embedding curve (rank = t_dim, no truncation):
    the pruned path then computes exactly what the full one does, up to the table's linear interpolation."""
    full, ref = pair(seed)
    grid = 1025
    with torch.no_grad():
        t = torch.linspace(0.0, 1.0, grid)
        curve = torch.nn.functional.silu(full.time_embedding(t))  # [grid, t_dim]
    cfg = dict(TINY, adaln_curve_grid=grid)
    pruned = MiniMaxH3Transformer(**cfg).float().eval()
    state = {k: v for k, v in full.state_dict().items() if not k.startswith("time_embedder.")}
    state["adaln_t_table"] = curve
    assert not any(k.startswith("time_embedder") for k in pruned.state_dict())
    pruned.load_state_dict(state, strict=True)
    return full, pruned, ref


def test_pruned_adaln_table_path():
    full, pruned, _ = pruned_pair()
    x, enc, audio = torch.randn(1, 24, 4, 4), torch.randn(1, 6, 40), torch.randn(1, 4, 32)
    with torch.no_grad():
        for sigma in (1.0, 0.937, 0.5, 0.0):  # on and between grid points, and the ends
            got = pruned(x, sigma * 1000.0, enc, None, audio_states=audio)
            want = full(x, sigma * 1000.0, enc, None, audio_states=audio)
            assert torch.allclose(got, want, atol=1e-4, rtol=1e-4)
        assert pruned.pruned and pruned.config["adaln_curve_grid"] == 1025


def test_layouts_round_trip_through_diffusers_own_converter():
    from diffusers.loaders.single_file_utils import convert_minimax_h3_transformer_checkpoint_to_diffusers as convert

    ours, ref = pair(5)
    want = {k: v.clone() for k, v in ref.state_dict().items()}
    port = {k: v.clone() for k, v in ours.state_dict().items()}
    back = C.port_to_diffusers(port)
    assert set(back) == set(want) and all(torch.equal(back[k], want[k]) for k in want)
    config = dict(TINY_DIFFUSERS)
    # Comfy-Org's single files: the port's names, [q; k; v] stacked, with diffusion_model. and rope.inv_freq
    comfy = {f"diffusion_model.{k}": v.clone() for k, v in port.items()}
    comfy["diffusion_model.rope.inv_freq"] = torch.zeros(4)
    got = convert({k.removeprefix("diffusion_model."): v.clone() for k, v in comfy.items()}, config=config,
                  qkv_layout="stacked")
    assert set(got) == set(want) and all(torch.equal(got[k], want[k]) for k in want)
    assert C.is_h3_state(k.removeprefix("diffusion_model.") for k in comfy)
    loaded = C.original_to_port(dict(comfy), interleaved=False, heads=2, head_dim=32)
    assert loaded.pop("rope.inv_freq").shape == (4,)  # kept for the config, dropped by the loader
    assert set(loaded) == set(port) and all(torch.equal(loaded[k], port[k]) for k in port)
    # MiniMaxAI's own shards: per-head interleaved qkv rows
    inter = {k: (C.interleave_qkv(v, 2, 32) if k.endswith("qkv_proj.weight") else v.clone()) for k, v in port.items()}
    got = convert({k: v.clone() for k, v in inter.items()}, config=config, qkv_layout="interleaved")
    assert all(torch.equal(got[k], want[k]) for k in want)
    loaded = C.original_to_port(inter, interleaved=True, heads=2, head_dim=32)
    assert all(torch.equal(loaded[k], port[k]) for k in port)
    cfg = C.config_from_state(port)
    assert all(cfg[k] == TINY[k] for k in cfg), (cfg, TINY)
    assert C.is_diffusers_state(want) and not C.is_diffusers_state(port)


# ---------------------------------------------------------------- the released files, by their headers

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "minimax_h3_headers.json").read_text(encoding="utf-8"))
H3 = archs.arch_by_name("minimax_h3")


def meta_state(entries: dict) -> dict:
    return {k: torch.empty(shape, device="meta") for k, (_, shape) in entries.items()}


def _block0_or_top(k: str) -> bool:
    import re

    m = re.search(r"(?:^|\.)blocks\.(\d+)\.", k)
    return m is None or m.group(1) == "0"


def released_model(cfg: dict):
    with torch.device("meta"):
        return MiniMaxH3Transformer.from_config(cfg)


def test_released_dit_headers_fit_the_port():
    """Comfy-Org's pruned int8 and full bf16 files (names, dtypes and shapes from their headers): the port's names and
    shapes, the config ComfyUI's model_detection reads, and the fp32 islands."""
    islands = {"video_patch_proj.weight", "video_patch_proj.bias", "audio_patch_proj.weight", "audio_patch_proj.bias",
               "final_layer.video_out.weight", "final_layer.video_out.bias", "final_layer.audio_out.weight",
               "final_layer.audio_out.bias", "rope.inv_freq"}
    for name, pruned in (("dit_pruned_int8_convrot", True), ("dit_bf16", False)):
        entries = {k: v for k, v in FIXTURE[name].items() if not k.endswith((".comfy_quant", ".weight_scale"))}
        f32 = {k for k, (dt, _) in entries.items() if dt == "F32"}
        assert f32 == (islands | ({"adaln_t_table"} if pruned else {
            "time_embedder.proj_in.weight", "time_embedder.proj_in.bias", "time_embedder.proj_out.weight",
            "time_embedder.proj_out.bias"})), name
        if pruned:
            assert {dt for k, (dt, _) in entries.items() if ".adaln_proj.linear." in k} == {"F16"}
            assert entries["blocks.0.attn.qkv_proj.weight"][0] == "I8"
        state = C.original_to_port(meta_state(entries), interleaved=False, heads=0, head_dim=0)
        cfg = C.config_from_state(state)
        assert state.pop("rope.inv_freq").shape == (16,) and cfg["rope_inv_freq_len"] == 16
        assert (cfg["hidden_size"], cfg["num_layers"], cfg["num_attention_heads"], cfg["attention_head_dim"],
                cfg["ffn_hidden_size"], cfg["text_dim"], cfg["token_refiner_num_layers"]) == (
            5376, 50, 56, 128, 14336, 5120, 2)
        assert cfg.get("adaln_curve_grid") == (1025 if pruned else None) and cfg["time_embed_dim"] == (
            8 if pruned else 2688)
        want = released_model(cfg).state_dict()
        assert set(state) <= set(want), sorted(set(state) - set(want))[:5]
        assert all(tuple(state[k].shape) == tuple(want[k].shape) for k in state)
        assert {k for k in want if _block0_or_top(k)} <= set(state), sorted(
            k for k in want if _block0_or_top(k) and k not in state)[:5]
    meta = json.loads(FIXTURE["dit_bf16_metadata"]["config"])["transformer"]  # the file's own config: the defaults
    defaults = MiniMaxH3Transformer.__init__.__defaults__
    names = MiniMaxH3Transformer.__init__.__code__.co_varnames[1:1 + len(defaults)]
    assert all(meta[k] == (list(v) if isinstance(v, tuple) else v) for k, v in zip(names, defaults) if k in meta)
    assert meta["image_model"] == "minimax_h3"


def test_published_loras_fit_the_port_names():
    """lightx2v's turbo LoRA (ComfyUI layout, block-diagonal fused qkv, .alpha) and the circlestone / ostris training
    adapters (diffusion_model.*.lora_A/B, no alpha): every module they name is a Linear of the port, shapes matching."""
    cfg = C.config_from_state(C.original_to_port(meta_state(
        {k: v for k, v in FIXTURE["dit_bf16"].items()}), interleaved=False, heads=0, head_dim=0))
    model = released_model(cfg)
    for name in ("turbo_8step_lora", "adapter_circlestone", "adapter_ostris_v2"):
        entries = FIXTURE[name]
        stems = {k.rsplit(".lora_", 1)[0] for k in entries if ".lora_" in k}
        assert stems and all(s.startswith("diffusion_model.") for s in stems)
        for stem in stems:
            mod = model.get_submodule(L._module_name(stem))
            a, b = entries[f"{stem}.lora_A.weight"][1], entries[f"{stem}.lora_B.weight"][1]
            assert a[1] == mod.in_features and b[0] == mod.out_features and a[0] == b[1], stem
        assert any(".token_refiner." in s for s in stems)
        has_alpha = any(k.endswith(".alpha") for k in entries)
        assert has_alpha == (name == "turbo_8step_lora")
    meta = FIXTURE["turbo_8step_lora_metadata"]
    assert meta["qkv_fusion"].startswith("block diagonal B") and "ComfyUI [gate;value]" in meta["swi_glu_mapping"]


def test_released_text_encoder_headers_build():
    from transformers import Qwen3VLTextConfig, Qwen3VLTextModel

    from rulztrainer.h3 import single_file as HSF

    with torch.device("meta"):
        te = Qwen3VLTextModel(Qwen3VLTextConfig(**HSF.TEXT_CONFIG))
    want = te.state_dict()
    for name in ("te_int8_convrot", "te_nvfp4_awq"):
        keys = [k for k in FIXTURE[name] if not k.endswith((".comfy_quant", ".weight_scale", ".weight_scale_2",
                                                              ".pre_quant_scale"))]
        stripped = {k.removeprefix("model."): v for k, v in FIXTURE[name].items() if k in keys}
        assert all(k in want for k in stripped), [k for k in stripped if k not in want][:3]
        assert "norm.weight" not in stripped  # cut at layer 50, no final norm (the loader supplies a bypassed one)
    awq = {k for k in FIXTURE["te_nvfp4_awq"] if k.endswith(".pre_quant_scale")}
    assert awq == {"model.layers.0.self_attn.o_proj.pre_quant_scale", "model.layers.0.mlp.down_proj.pre_quant_scale"}
    assert json.loads(FIXTURE["te_int8_convrot_metadata"]["minimax_h3_te"])["num_hidden_layers"] == 50
    assert len(want) and HSF.TEXT_CONFIG["num_hidden_layers"] == 50


def test_released_video_vae_converts_to_diffusers():
    from rulztrainer.h3 import single_file as HSF

    state = HSF.vae_original_to_diffusers(meta_state(FIXTURE["video_vae"]))
    with torch.device("meta"):
        vae = HSF.vae_class()(**HSF.VAE_CONFIG)
    want = vae.state_dict()
    assert set(state) == set(want), (sorted(set(want) - set(state))[:5], sorted(set(state) - set(want))[:5])
    assert all(tuple(state[k].shape) == tuple(want[k].shape) for k in want)


def test_released_audio_vae_encoder_converts_to_diffusers():
    from rulztrainer.h3 import single_file as HSF

    want = FIXTURE["audio_vae_diffusers"]
    state = HSF.audio_original_to_diffusers(meta_state(FIXTURE["audio_vae_comfy"]), want)
    assert set(state) == set(want) - {"latents_mean", "latents_std"}
    assert all(tuple(state[k].shape) == tuple(want[k][1]) for k in state)


# ---------------------------------------------------------------- VAEs on tiny random configs

TINY_VAE = dict(block_out_channels=(8, 8, 8, 8, 8, 16), layers_per_block=1, norm_num_groups=4, decoder_num_layers=1,
                decoder_num_attention_heads=2, decoder_attention_head_dim=16)


def tiny_vae(seed=0):
    from rulztrainer.h3 import single_file as HSF

    torch.manual_seed(seed)
    vae = HSF.vae_class()(**{**HSF.VAE_CONFIG, **TINY_VAE}).float().eval()
    with torch.no_grad():
        for p in vae.parameters():
            p.normal_(0, 0.05)
    return vae


def _rms(x, w=None, eps=1e-5):
    return torch.nn.functional.rms_norm(x, (x.shape[-1],), w, eps)


def test_vae_layout_conversion_means_the_original_math():
    """The original ViT decoder block (MiniMaxAI's vae_vit / attention / base_module code): to_qkv rows per head
    [q k v], ff.w1 = [gate; value]; diffusers' block on the converted weights computes the same."""
    from rulztrainer.h3 import single_file as HSF

    vae = tiny_vae(1)
    sd = {k: v.clone() for k, v in vae.state_dict().items()}
    orig = HSF.vae_diffusers_to_original(sd, heads=2)
    back = HSF.vae_original_to_diffusers(dict(orig), heads=2)
    assert set(back) == set(sd) and all(torch.equal(back[k], sd[k]) for k in sd)
    block = vae.decoder.transformer_blocks[0]
    p = "decoder.transformer_blocks.0."
    x = torch.randn(1, 7, 32)
    with torch.no_grad():
        want = block(x, None)
        h = _rms(x, orig[p + "norm1.weight"])
        qkv = h @ orig[p + "attn.to_qkv.weight"].T + orig[p + "attn.to_qkv.bias"]
        q, k, v = qkv.view(1, 7, 2, 48).chunk(3, dim=-1)
        q, k = _rms(q), _rms(k)
        a = torch.nn.functional.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        a = a.transpose(1, 2).flatten(2) @ orig[p + "attn.to_out.weight"].T + orig[p + "attn.to_out.bias"]
        y = x + a * orig[p + "scale1"]
        g, hid = (_rms(y, orig[p + "norm2.weight"]) @ orig[p + "ff.w1.weight"].T + orig[p + "ff.w1.bias"]).chunk(2, -1)
        got = y + ((torch.nn.functional.silu(g) * hid) @ orig[p + "ff.w2.weight"].T + orig[p + "ff.w2.bias"]) * orig[
            p + "scale2"]
    assert torch.allclose(got, want, atol=1e-5)


def test_vae_file_loads_and_one_frame_round_trip(tmp_path, monkeypatch):
    from safetensors.torch import save_file

    from rulztrainer.h3 import loading as HL
    from rulztrainer.h3 import single_file as HSF

    vae = tiny_vae(2)
    orig = HSF.vae_diffusers_to_original({k: v.clone() for k, v in vae.state_dict().items()}, heads=2)
    orig["latents_mean"] = torch.tensor(HSF.VAE_CONFIG["latents_mean"])
    orig["latents_std"] = torch.tensor(HSF.VAE_CONFIG["latents_std"])
    orig["decoder.mask_token"] = torch.zeros(1, 1, 32)
    path = tmp_path / "minimax_h3_video_vae_fp16.safetensors"
    save_file({k: v.to(torch.float16).contiguous() for k, v in orig.items()}, str(path))
    monkeypatch.setattr(HSF, "VAE_CONFIG", {**HSF.VAE_CONFIG, **TINY_VAE})
    loaded = HSF.load_vae_file(path, "cpu", dtype=torch.float32)
    img = torch.rand(1, 3, 64, 96) * 2 - 1
    with torch.no_grad():
        lat = HL.encode_images(loaded, img)
        assert lat.shape == (1, 24, 4, 6) and torch.isfinite(lat).all()
        # a one-frame video through the spatial encoder alone: the posterior mode, normalized
        x = (((img + 1) / 2 - torch.tensor(HL.IMAGENET_MEAN).view(1, 3, 1, 1)) /
             torch.tensor(HL.IMAGENET_STD).view(1, 3, 1, 1)).unsqueeze(2)
        z = loaded.encode(x).latent_dist.mode()[:, :, 0]
        mean, std = HL._stats(loaded, z)
        assert torch.allclose(lat, (z - mean) / std)
        out = HL.decode_latents(loaded, lat)
        assert out.shape == img.shape and out.min() >= -1 and out.max() <= 1
        # the still path: the latent as a 5-latent clip, the first real frame (after the 3-frame pre-padding)
        zz = (lat * std + mean).unsqueeze(2).repeat(1, 1, 5, 1, 1)
        frame = loaded._decode_clip(zz)[:, :, 3]
        pm, ps = HL._pixel_norm(frame)
        assert torch.allclose(out, ((frame * ps + pm) * 2 - 1).clamp(-1, 1), atol=1e-5)
    sol = tmp_path / "minimax_h3_sol_vae_fp16.safetensors"
    save_file({"blocks.0.scale1": torch.zeros(4), "x_embedder.weight": torch.zeros(4, 24)}, str(sol))
    with pytest.raises(ValueError):
        HSF.load_vae_file(sol, "cpu")


TINY_AUDIO = dict(encoder_dim=4, encoder_rates=(2, 2), latent_dim=16, latent_channels=8, num_attention_heads=2,
                  decoder_dim=8, decoder_rates=(2, 2), decoder_kernel_sizes=(4, 4), resblock_kernel_sizes=(3,),
                  resblock_dilation_sizes=((1,),))


def test_audio_vae_file_and_silence_placeholder(tmp_path):
    """Comfy-Org's audio VAE layout (weight norms folded into plain weights) loads into diffusers' encoder and encodes
    as the original does; the silence placeholder is the normalized posterior mode of zeros, channel-major."""
    from diffusers import AutoencoderKLMiniMaxH3Audio
    from safetensors.torch import save_file

    from rulztrainer.h3 import loading as HL
    from rulztrainer.h3 import single_file as HSF

    torch.manual_seed(3)
    ref = AutoencoderKLMiniMaxH3Audio(**TINY_AUDIO).eval()
    sd = ref.state_dict()
    folded = {}
    for k, v in sd.items():
        if k.endswith(".weight_v"):
            stem = k[: -len("_v")]
            g = sd[f"{stem}_g"]
            folded[stem] = g * v / torch.linalg.vector_norm(v, dim=tuple(range(1, v.ndim)), keepdim=True)
        elif not k.endswith(".weight_g"):
            folded[k] = v
    mean, std = torch.randn(8), torch.rand(8) + 0.5
    path = tmp_path / "minimax_h3_audio_vae_fp32.safetensors"
    save_file({**{k: v.contiguous() for k, v in folded.items()}, "latents_mean": mean, "latents_std": std}, str(path))
    enc, m2, s2 = HSF.load_audio_encoder_file(path, config=TINY_AUDIO)
    wav = torch.randn(2, 1, 40)
    with torch.no_grad():
        assert torch.allclose(enc.encode(wav).latent_dist.mode(), ref.encode(wav).latent_dist.mode(), atol=1e-5)
        silence = HL.silence_latent(enc, m2, s2)
        z = ref.encode(torch.zeros(2, 1, 2 * ref.hop_length)).latent_dist.mode().transpose(1, 2)
        want = ((z - mean.view(1, 1, -1)) / std.view(1, 1, -1)).reshape(4, 8)
    assert silence.shape == (4, 8) and torch.allclose(silence, want, atol=1e-5)
    assert torch.equal(m2, mean) and torch.equal(s2, std)


# ---------------------------------------------------------------- single files and load-time AdaLN pruning


def comfy_file(tmp_path, model, name, adaln_dtype=torch.float32):
    """The port's state as a Comfy-Org file: diffusion_model., rope.inv_freq, the AdaLN in `adaln_dtype`."""
    from safetensors.torch import save_file

    from rulztrainer.h3.model import inv_freq

    sd = {f"diffusion_model.{k}": (v.to(adaln_dtype) if ".adaln_proj.linear." in k else v).contiguous()
          for k, v in model.state_dict().items()}
    sd["diffusion_model.rope.inv_freq"] = inv_freq(TINY["rope_inv_freq_len"])
    path = tmp_path / name
    save_file(sd, str(path), metadata={"config": json.dumps({"transformer": {"image_model": "minimax_h3"}})})
    return path


def test_single_files_load_pruned_and_full(tmp_path, monkeypatch):
    from rulztrainer.h3 import loading as HL
    from rulztrainer.h3 import single_file as HSF

    full, pruned, _ = pruned_pair(6)
    x, enc, audio = torch.randn(1, 24, 4, 4), torch.randn(1, 6, 40), torch.randn(1, 4, 32)
    path = comfy_file(tmp_path, pruned, "minimax_h3_fl2va_pruned_bf16.safetensors", torch.float16)
    assert HSF.is_h3_file(path) and archs.arch_from_transformer_file(str(path)) == "minimax_h3"
    loaded = HSF.load_transformer_file(path, "cpu", quant="none", dtype=torch.float32)
    assert loaded.pruned and loaded.final_layer.adaln_proj.linear.weight.dtype == torch.float32
    with torch.no_grad():
        want = pruned(x, 900.0, enc, None, audio_states=audio)
        assert torch.allclose(loaded(x, 900.0, enc, None, audio_states=audio), want, atol=1e-3)
    # a full file is pruned on load: exact with a basis as wide as the embedding, close with rank 8
    path = comfy_file(tmp_path, full, "minimax_h3_fl2va_bf16.safetensors")
    with torch.no_grad():
        want = full(x, 600.0, enc, None, audio_states=audio)
        monkeypatch.setattr(HL, "PRUNE_RANK", TINY["time_embed_dim"])
        exact = HSF.load_transformer_file(path, "cpu", quant="none", dtype=torch.float32)
        assert exact.adaln_rank == TINY["time_embed_dim"] and exact.blocks[0].adaln_proj.linear.in_features == 25
        assert torch.allclose(exact(x, 600.0, enc, None, audio_states=audio), want, atol=1e-4)
        monkeypatch.setattr(HL, "PRUNE_RANK", 8)
        approx = HSF.load_transformer_file(path, "cpu", quant="none", dtype=torch.float32)
        err = (approx(x, 600.0, enc, None, audio_states=audio) - want).abs().max() / want.abs().max()
        assert err < 0.05, float(err)
        monkeypatch.setenv("RULZ_H3_FULL_ADALN", "1")
        kept = HSF.load_transformer_file(path, "cpu", quant="none", dtype=torch.float32)
        assert kept.adaln_rank is None and torch.allclose(kept(x, 600.0, enc, None, audio_states=audio), want,
                                                          atol=1e-5)


# ---------------------------------------------------------------- LoRA: names, export, adapters, turbo


def test_lora_names_export_and_the_ecosystem_loaders(tmp_path):
    from diffusers.loaders.lora_conversion_utils import _convert_non_diffusers_minimax_h3_lora_to_diffusers as conv
    from safetensors.torch import load_file

    ours, ref = pair(7)
    default = H3.lora_module_names(ours, H3.default_targets, None)
    assert sorted(default) == sorted(f"blocks.{i}.{m}" for i in range(2)
                                     for m in ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"))
    assert H3.lora_module_names(ours, ["q"], [1]) == ["blocks.1.attn.qkv_proj"]  # q, k or v: the fused layer
    assert H3.lora_module_names(ours, ["k", "v"], [1]) == ["blocks.1.attn.qkv_proj"]
    refiner = H3.lora_module_names(ours, ["o"], [0], include_text_embedder=True)
    assert "token_refiner.blocks.0.attn.out_proj" in refiner and "token_refiner.blocks.0.mlp.fc1" in refiner
    assert H3.param_group_of("token_refiner.blocks.0.mlp.fc1") == "text"
    assert H3.param_group_of("blocks.3.attn.qkv_proj") == "attn" and H3.param_group_of("blocks.3.mlp.fc2") == "mlp"
    names = default + ["token_refiner.blocks.0.mlp.fc2"]
    inj = L.inject_lora(ours, 4, 8.0, ["q"], names=names)  # alpha 8, rank 4: scale 2, folded into the export
    with torch.no_grad():
        for mod in inj.values():
            mod.lora_up.weight.normal_(0, 0.1)
    path = tmp_path / "h3_lora.safetensors"
    L.save_lora(inj, path, dtype=torch.float32, fmt=H3.lora_format, metadata={"ss_network_dim": "4",
                                                                              "ss_network_alpha": "8"})
    state = load_file(str(path))
    assert set(state) == {f"diffusion_model.{n}.lora_{p}.weight" for n in names for p in "AB"}  # no alpha keys
    x, enc, audio = torch.randn(1, 24, 4, 6), torch.randn(1, 5, 40), torch.randn(1, 4, 32)
    with torch.no_grad():
        got = ours(x, 400.0, enc, None, audio_states=audio)
        # diffusers' own H3 LoRA loader conversion, merged into its model at scale 1 (alpha = rank)
        converted = conv(dict(state))
        for k, a in converted.items():
            if k.endswith(".lora_A.weight"):
                b = converted[k.replace("lora_A", "lora_B")]
                ref.get_submodule(k.removeprefix("transformer.").removesuffix(".lora_A.weight")).weight += b @ a
        want, _ = ref_call(ref, x, torch.tensor(0.4), enc, audio)
        assert torch.allclose(got, -want, atol=1e-4)
    # back into the trainer's wrappers (resume): the same prediction, the module's own alpha kept
    fresh, _ = pair(7)
    inj2 = L.inject_lora(fresh, 4, 8.0, ["q"], names=names)
    assert L.load_lora_weights(inj2, path) == len(names)
    assert all(m.alpha == 8.0 for m in inj2.values())
    with torch.no_grad():
        assert torch.allclose(fresh(x, 400.0, enc, None, audio_states=audio), got, atol=1e-5)


def test_training_adapter_and_turbo_lora_in_their_published_layouts(tmp_path):
    from safetensors.torch import save_file

    ours, _ = pair(8)
    plain, _ = pair(8)
    x, enc, audio = torch.randn(1, 24, 4, 4), torch.randn(1, 5, 40), torch.randn(1, 4, 32)
    # a training adapter the way circlestone / ostris ship it: diffusion_model.*.lora_A/B, no alpha, refiner too
    a1, b1 = torch.randn(3, 48) * 0.2, torch.randn(192, 3) * 0.2
    a2, b2 = torch.randn(3, 48) * 0.2, torch.randn(128, 3) * 0.2
    adapter = tmp_path / "minimax_h3_image_training_adapter.safetensors"
    save_file({"diffusion_model.blocks.1.attn.qkv_proj.lora_A.weight": a1,
               "diffusion_model.blocks.1.attn.qkv_proj.lora_B.weight": b1,
               "diffusion_model.token_refiner.blocks.0.mlp.fc1.lora_A.weight": a2,
               "diffusion_model.token_refiner.blocks.0.mlp.fc1.lora_B.weight": b2}, str(adapter))
    with torch.no_grad():
        before = ours(x, 500.0, enc, None, audio_states=audio)
    L.inject_lora(ours, 2, 2.0, ["q"], names=["blocks.0.attn.qkv_proj"])
    assert L.attach_training_adapter(ours, str(adapter), rename=H3.lora_module_name) == 2
    with torch.no_grad():
        training = ours(x, 500.0, enc, None, audio_states=audio)
        L.set_extras_enabled(ours, True)
        sampling = ours(x, 500.0, enc, None, audio_states=audio)
        L.set_extras_enabled(ours, False)
        plain.blocks[1].attn.qkv_proj.weight += b1 @ a1
        plain.token_refiner.blocks[0].mlp.fc1.weight += b2 @ a2
        merged = plain(x, 500.0, enc, None, audio_states=audio)
    assert torch.equal(sampling, before) and torch.allclose(training, merged, atol=1e-5)
    # the turbo LoRA in ComfyUI's layout: block-diagonal fused qkv (rank 3 r), alpha multiplied by 3
    turbo_model, _ = pair(9)
    plain, _ = pair(9)
    r = 2
    down = torch.randn(3 * r, 48) * 0.2
    up = torch.block_diag(*(torch.randn(64, r) * 0.2 for _ in range(3)))
    turbo = tmp_path / "turbo.safetensors"
    save_file({"diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight": down,
               "diffusion_model.blocks.0.attn.qkv_proj.lora_B.weight": up,
               "diffusion_model.blocks.0.attn.qkv_proj.alpha": torch.tensor(3 * 1.0)}, str(turbo))
    assert L.attach_extra_lora(turbo_model, str(turbo), 1.0, rename=H3.lora_module_name) == 1
    L.set_extras_enabled(turbo_model, True)
    with torch.no_grad():
        plain.blocks[0].attn.qkv_proj.weight += (3.0 / (3 * r)) * up @ down
        assert torch.allclose(turbo_model(x, 800.0, enc, None, audio_states=audio),
                              plain(x, 800.0, enc, None, audio_states=audio), atol=1e-5)
    # musubi-tuner's flattened names
    assert H3.lora_module_name("blocks_12_attn_qkv_proj") == "blocks.12.attn.qkv_proj"
    assert H3.lora_module_name("token_refiner_blocks_1_mlp_fc2") == "token_refiner.blocks.1.mlp.fc2"
    assert L._module_name("lora_unet_blocks_3_mlp_fc1") == "blocks_3_mlp_fc1"


# ---------------------------------------------------------------- sampler and timesteps


def test_sampler_matches_the_released_scheduler_pair():
    from diffusers import MiniMaxH3Scheduler

    from rulztrainer.h3.sampler import h3_sigmas, sample

    for steps in (8, 30, 49):
        sched = MiniMaxH3Scheduler(shift=12.0)
        sched.set_timesteps(steps + 1)  # the scheduler counts grid points
        assert torch.allclose(h3_sigmas(steps), sched.sigmas, atol=1e-7) and len(h3_sigmas(steps)) == steps + 1
    turbo = h3_sigmas(4)  # lightx2v's Turbo card: q = [1, 0.75, 0.5, 0.25] -> video [1, 0.9730, 0.9231, 0.8000] -> 0
    assert torch.allclose(turbo, torch.tensor([1.0, 0.9730, 0.9231, 0.8000, 0.0]), atol=1e-4)
    ours, ref = pair(10)
    enc = torch.randn(1, 5, 40)
    mask = torch.ones(1, 5, dtype=torch.bool)
    lat = sample(ours, enc, mask, None, None, width=64, height=96, steps=4, seed=3, device="cpu")
    gen = torch.Generator("cpu").manual_seed(3)
    x, audio = torch.randn((1, 24, 6, 4), generator=gen), torch.randn((1, 4, 32), generator=gen)
    sv, sa = MiniMaxH3Scheduler(shift=12.0), MiniMaxH3Scheduler(shift=3.0)
    sv.set_timesteps(5)
    sa.set_timesteps(5)
    with torch.no_grad():
        for tv, ta in zip(sv.timesteps, sa.timesteps):
            v, a = ref_call(ref, x.to(torch.bfloat16).float(), None, enc, audio, times=(tv, ta))
            x = sv.step(v, tv, x).prev_sample
            audio = sa.step(a, ta, audio).prev_sample
    assert lat.shape == (1, 24, 6, 4) and torch.allclose(lat, x, atol=2e-4), float((lat - x).abs().max())
    assert H3.sample_sigmas(ours, 8, 1024, 1024, 5.0).shape == (9,)


def test_reference_timesteps_are_one_base_time_two_shifts():
    s = H3.reference_sigmas(20000, torch.Generator().manual_seed(0), 1024)
    expect_mean = 12 / 11 * (1 - torch.log(torch.tensor(12.0)).item() / 11)  # E[12u / (1 + 11u)], u ~ U(0, 1)
    assert s.min() >= 0 and s.max() <= 1 and abs(float(s.mean()) - expect_mean) < 0.01
    assert abs(float(s.median()) - 12 * 0.5 / (1 + 11 * 0.5)) < 0.01
    assert H3.size_multiple == 32 and H3.latent_tag == "_h3vae" and H3.block_token_px == 16


# ---------------------------------------------------------------- detection, caches, buckets, downloads


def test_detection_caches_buckets_and_downloads(tmp_path):
    from PIL import Image

    from rulztrainer import dataset as D
    from rulztrainer.comfy_files import planned_downloads
    from rulztrainer.config import ARCH_TRAINING_ADAPTER, ARCH_TURBO, MODEL_PRESETS
    from rulztrainer.h3 import loading as HL

    for path in ("MiniMaxAI/MiniMax-H3", "MiniMaxAI/MiniMax-H3#ref2va",
                 "D:/models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"):
        assert TrainConfig(model_path=path).arch_name() == "minimax_h3"
    assert MODEL_PRESETS["MiniMax-H3-FL2VA"] == "MiniMaxAI/MiniMax-H3"
    assert HL.split_model_path(MODEL_PRESETS["MiniMax-H3-Ref2VA"]) == ("MiniMaxAI/MiniMax-H3", True)
    for cls in ("MiniMaxH3Transformer3DModel", "MiniMaxH3DiTModel"):
        mdir = tmp_path / cls / "transformer"
        mdir.mkdir(parents=True)
        (mdir / "config.json").write_text(json.dumps({"_class_name": cls}))
        assert archs.detect_arch(TrainConfig(model_path=str(mdir.parent)), mdir.parent).name == "minimax_h3"
        assert HL.is_h3_dir(mdir.parent)
    data = tmp_path / "data"
    data.mkdir()
    Image.new("RGB", (900, 600)).save(data / "a.png")
    cfg = TrainConfig(model_path="MiniMaxAI/MiniMax-H3", dataset_dir=str(data), resolutions=[1.0])
    item = D.scan_dataset(cfg)[0]
    assert item.width % 32 == 0 and item.height % 32 == 0 and item.latent_path.name.endswith("_h3vae.safetensors")
    assert D.text_cache_path(cfg, "p").name.endswith("_minimax_h3.safetensors")
    plan = {f: (r, n) for f, r, n in planned_downloads(TrainConfig(model_path="MiniMaxAI/MiniMax-H3",
                                                                   model_file_quant="pruned_int8_convrot"))}
    assert plan["transformer_file"] == ("Comfy-Org/MiniMax-H3",
                                        "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors")
    assert plan["text_encoder_file"][1] == "text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
    assert plan["vae_file"][1] == "vae/minimax_h3_video_vae_fp16.safetensors"
    assert plan["turbo_lora"] == (ARCH_TURBO["minimax_h3"]["repo"], ARCH_TURBO["minimax_h3"]["file"])
    assert plan["training_adapter"] == ("circlestone-labs/MiniMax-H3-Image-Training-Adapter",
                                        "minimax_h3_image_training_adapter.safetensors")
    assert ARCH_TRAINING_ADAPTER["minimax_h3"]["repo"].startswith("circlestone-labs/")
    ref = {f: n for f, _, n in planned_downloads(TrainConfig(model_path="MiniMaxAI/MiniMax-H3#ref2va",
                                                             text_encoder_file_quant="nvfp4_awq",
                                                             model_file_quant="pruned_int8_convrot"))}
    assert ref["transformer_file"].endswith("minimax_h3_ref2va_pruned_int8_convrot.safetensors")
    assert ref["text_encoder_file"].endswith("qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors")
    allow = HL._allow_patterns(False)
    assert "transformer/*" in allow and "transformer_ref/*" not in allow and \
        "text_encoder/model-00012-of-00014.safetensors" not in allow


def test_auto_quant_sizes_for_the_gpus(monkeypatch):
    from rulztrainer.h3 import loading as HL

    sizes = {0: 15.9, 1: 15.9, 2: 80.0}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(HL, "_gpu_gb", lambda d: sizes[torch.device(d).index or 0])
    pruned_bf16 = 40.2e9 / 2**30
    assert HL.resolve_quant("auto", "cuda:0", pruned_bf16) == "nvfp4"
    assert HL.resolve_quant("auto", ["cuda:0", "cuda:1"], pruned_bf16) == "int8_convrot"
    assert HL.resolve_quant("auto", "cuda:2", pruned_bf16) == "none"
    assert HL.resolve_quant("auto", "cpu", pruned_bf16) == "nvfp4"  # block swap: sized for the smallest GPU
    assert HL.resolve_quant("auto", "cuda:0", pruned_bf16, quantized=True) == "auto"  # keep the file's formats
    assert HL.resolve_quant("int8_convrot", "cuda:0", pruned_bf16) == "int8_convrot"
