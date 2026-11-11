"""MiniMax-H3-Image support without model weights: the port without its audio stream against diffusers'
MiniMaxH3ImageTransformer2DModel.forward (written out here on the H3 blocks of the installed diffusers, fed the layout
of its build_image_packed_sequence: a padded batch against each sample alone); the per-step (PDD) output heads in their
three layouts and the head a training or a sampling forward uses; the released files' headers
(tests/fixtures/minimax_h3_image_headers.json: names, dtypes and shapes only) against the port; ComfyUI's W4A8 / W6A8
weights decoded to int8 ConvRot; the image VAE's 2D decoder against diffusers' MiniMaxH3ImageViTDecoder2d.forward
(written out likewise); single files and both folder layouts; the sampler; detection, downloads, caches, the wizard."""
import json
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import archs  # noqa: E402
from rulztrainer import lora as L  # noqa: E402
from rulztrainer import quant as Q  # noqa: E402
from rulztrainer.config import TrainConfig, family_named_in  # noqa: E402
from rulztrainer.h3 import convert as C  # noqa: E402
from rulztrainer.h3 import model as M  # noqa: E402
from rulztrainer.h3.model import (H3_IMAGE_PDD_SIGMAS, TEXT_TAG, VIDEO_TAG, MiniMaxH3Transformer,  # noqa: E402
                                  inv_freq, pdd_head_weights, pdd_span, unpatchify)

diffusers = pytest.importorskip("diffusers")

from rulztrainer.h3 import loading as HL  # noqa: E402
from rulztrainer.h3 import single_file as HSF  # noqa: E402
from rulztrainer.h3image import loading as IL  # noqa: E402
from rulztrainer.h3image import vae as V  # noqa: E402
from rulztrainer.h3image.sampler import h3image_sigmas, sample  # noqa: E402

HEADERS = json.loads((Path(__file__).parent / "fixtures" / "minimax_h3_image_headers.json").read_text("utf-8"))
ARCH = archs.arch_by_name("minimax_h3_image")
TINY_D = dict(num_attention_heads=2, attention_head_dim=32, hidden_size=48, num_layers=2, num_refiner_layers=1,
              ffn_dim=64, in_channels=24, audio_in_channels=32, patch_size=(1, 2, 2), text_dim=40, freq_dim=256,
              time_embed_hidden_dim=56, time_embed_dim=24, rope_freq_dim=4, rope_theta=10000.0)
HEADS = 3
GRID = [1.0, 0.8, 0.45, 0.0]  # the tiny model's own 3-interval sigma grid
TINY = {**C.diffusers_config_to_port({**TINY_D, "num_pdd_heads": HEADS}, audio=False), "pdd_sigmas": GRID}


def diffusers_state(seed=0):
    """(reference modules, its per-step heads, the state of a MiniMaxH3ImageTransformer2DModel with those weights):
    diffusers' H3 model without its audio projections and output layers, plus pdd_proj_out.0..2."""
    torch.manual_seed(seed)
    ref = diffusers.MiniMaxH3Transformer3DModel(**TINY_D).float().eval()
    heads = nn.ModuleList([nn.Linear(48, 96) for _ in range(HEADS)]).float().eval()
    with torch.no_grad():
        for p in [*ref.parameters(), *heads.parameters()]:
            p.normal_(0, 0.1)
    state = {k: v.clone() for k, v in ref.state_dict().items()
             if not k.startswith(("audio_proj_in.", "audio_proj_out.", "proj_out."))}
    state.update({f"pdd_proj_out.{k}": v.clone() for k, v in heads.state_dict().items()})
    return ref, heads, state


def pair(seed=0):
    """Our port and the reference with the same random weights, fp32."""
    ref, heads, state = diffusers_state(seed)
    ours = MiniMaxH3Transformer(**TINY).float().eval()
    missing, unexpected = ours.load_state_dict(C.fold_diffusers_pdd_heads(C.diffusers_to_port(state)), strict=True)
    assert not missing and not unexpected
    return ours, ref, heads


def ref_image(ref, heads, x, sigma, enc, step):
    """diffusers' MiniMaxH3ImageTransformer2DModel.forward for one text-to-image sample, on the layout of its
    build_image_packed_sequence: [text | image], text token j at (j, 0, 0), the image one frame at t = the text
    length, every row at the one time 1 - sigma, tags text 1 / image 0, the output head of `step`. Returns the
    data-ward velocity [1, 24, h, w]."""
    from diffusers.modular_pipelines.minimax_h3.before_denoise import _frame_position_grid, patchify_video_latents

    n_text = enc.shape[1]
    h, w = x.shape[-2:]
    grid, _ = _frame_position_grid(h, w, 2, 2)
    n_img = grid.shape[0]
    pos = torch.zeros(n_text + n_img, 3, dtype=torch.float64)
    pos[:n_text, 0] = torch.arange(n_text, dtype=torch.float64)
    pos[n_text:, 0] = float(n_text)
    pos[n_text:, 1:] = grid
    tags = torch.cat([torch.full((n_text,), TEXT_TAG), torch.full((n_img,), VIDEO_TAG)])
    t_idx = torch.zeros(n_text + n_img, dtype=torch.long)
    rows = patchify_video_latents(x[:, :, None], (1, 2, 2))[None]
    rotary = ref.rope(pos)
    hidden = torch.cat([ref.token_refiner(ref.context_embedder(enc)), ref.proj_in(rows)], dim=1)
    temb = ref.time_embedder(ref.time_proj(torch.tensor([1.0 - float(sigma)], dtype=torch.float32)))
    for block in ref.transformer_blocks:
        hidden = block(hidden, temb, t_idx * 3 + tags, rotary)
    out = heads[step](ref.norm_out(hidden, temb, t_idx))[:, n_text:]
    return unpatchify(out, 24, h, w)


# ---------------------------------------------------------------- the transformer


def test_port_matches_the_reference_on_the_packed_image_layout():
    ours, ref, heads = pair(0)
    assert not ours.has_audio and ours.audio_frames == 0 and ours.pdd_steps == HEADS
    assert not hasattr(ours, "audio_patch_proj") and ours.final_layer.audio_out is None
    assert tuple(ours.final_layer.video_out.weight.shape) == (HEADS * 96, 48)
    x = torch.randn(2, 24, 4, 6)
    enc = torch.randn(2, 7, 40)
    mask = torch.ones(2, 7, dtype=torch.bool)
    mask[1, 5:] = False  # the second caption is 5 tokens, padded to 7
    with torch.no_grad():
        for step in range(HEADS):  # a sampling step of the grid: that step's head
            sigma = torch.full((2,), GRID[step])
            got = ours(x, sigma * 1000.0, enc, mask, sigma_next=GRID[step + 1])
            for i, n in enumerate((7, 5)):
                want = ref_image(ref, heads, x[i:i + 1], sigma[i], enc[i:i + 1, :n], step)
                assert torch.allclose(got[i:i + 1], -want, atol=2e-5, rtol=1e-4), (step, i)
        # a training forward: each sample through the head of the interval that holds its sigma
        sigma = torch.tensor([0.9, 0.3])
        got = ours(x, sigma * 1000.0, enc, mask)
        for i, (n, step) in enumerate(((7, 0), (5, 2))):
            want = ref_image(ref, heads, x[i:i + 1], sigma[i], enc[i:i + 1, :n], step)
            assert torch.allclose(got[i:i + 1], -want, atol=2e-5, rtol=1e-4), i
        # an unpadded batch needs no mask
        assert torch.equal(ours(x[:1], 700.0, enc[:1], mask[:1]), ours(x[:1], 700.0, enc[:1], None))
        with pytest.raises(ValueError, match="no audio stream"):
            ours(x, 500.0, enc, mask, return_audio=True)
        with pytest.raises(ValueError, match="no audio stream"):
            ours(x, 500.0, enc, mask, audio_states=torch.zeros(2, 4, 32))


def test_trainer_hooks_work_without_the_audio_rows():
    ours, _, _ = pair(1)
    x, enc = torch.randn(1, 24, 4, 4), torch.randn(1, 6, 40)
    assert ours.positions(torch.tensor([6]), 6, 4, 4).shape == (1, 6 + 4, 3)
    assert len(ours.layers) == 2 and ours.split_tail_modules() == [ours.final_layer]
    with torch.no_grad():
        want = ours(x, 300.0, enc, None)
        assert torch.equal(ours(x, 300.0, enc, None, skip_layers=()), want)
        assert not torch.equal(ours(x, 300.0, enc, None, skip_layers=(1,)), want)
        odd = ours(torch.randn(1, 24, 3, 5), 300.0, enc, None)  # a probe size that is not whole 32 px
    assert odd.shape == (1, 24, 3, 5) and torch.isfinite(odd).all()
    # gradient checkpointing and a LoRA on the fused attention layer train through the stacked heads
    names = ARCH.lora_module_names(ours, ARCH.default_targets, None)
    assert sorted(names) == sorted(f"blocks.{i}.{m}" for i in range(2)
                                   for m in ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"))
    inj = L.inject_lora(ours, 2, 2.0, ["q"], names=names)
    ours.gradient_checkpointing = True
    ours.train()
    for mod in inj.values():
        nn.init.normal_(mod.lora_up.weight, std=0.05)
    out = ours(x, torch.tensor([600.0]), enc, None)
    out.pow(2).mean().backward()
    grads = [m.lora_down.weight.grad for m in inj.values()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads) and any(g.abs().sum() > 0 for g in grads)


# ---------------------------------------------------------------- the per-step output heads


def test_head_layouts_convert_into_each_other():
    torch.manual_seed(2)
    w, b = torch.randn(HEADS, 96, 48), torch.randn(HEADS, 96)  # full heads: pdd_fused_heads / pdd_proj_out
    sw, sb = C.stack_pdd_heads(w, b)
    assert sw.shape == (HEADS * 96, 48) and sb.shape == (HEADS * 96,)
    assert torch.equal(sw[:96], w[0]) and torch.allclose(sw[96:192], w[1] - w[0])  # block 0 full, then offsets
    w2, b2 = C.unstack_pdd_heads(sw, sb, HEADS)
    assert torch.allclose(w2, w, atol=1e-6) and torch.allclose(b2, b, atol=1e-6)
    state = {f"pdd_proj_out.{k}.{p}": t[k] for k in range(HEADS) for p, t in (("weight", w), ("bias", b))}
    state["other"] = torch.zeros(1)
    folded = C.fold_diffusers_pdd_heads(dict(state))
    assert set(folded) == {"other", "final_layer.video_out.weight", "final_layer.video_out.bias"}
    assert torch.equal(folded["final_layer.video_out.weight"], sw)
    assert C.fold_diffusers_pdd_heads({"other": 1}) == {"other": 1}
    del state["pdd_proj_out.1.bias"]
    with pytest.raises(ValueError, match="incomplete"):
        C.fold_diffusers_pdd_heads(state)


def test_which_heads_a_forward_uses():
    grid = list(H3_IMAGE_PDD_SIGMAS)
    assert len(grid) == 7 and grid[0] == 1.0 and grid[-1] == 0.0 and grid[3] == 0.75
    # training: the interval that holds sigma; a grid point opens its own interval, also through an fp32 round trip
    assert [pdd_span(grid, s)[0] for s in (1.0, 0.98, 0.9, 0.8, 0.6, 0.3, 0.1, 0.0)] == [0, 0, 1, 2, 3, 4, 5, 5]
    for k in range(6):
        fp32 = float(torch.tensor(grid[k] * 1000.0, dtype=torch.float32) / 1000.0)
        assert pdd_span(grid, grid[k]) == (k, k + 1) and pdd_span(grid, fp32) == (k, k + 1)
    # sampling: nearest grid points (ComfyUI's rule), at least one interval
    assert pdd_span(grid, grid[2], grid[3]) == (2, 3) and pdd_span(grid, 1.0, 0.0) == (0, 6)
    assert pdd_span(grid, grid[1], grid[4]) == (1, 4) and pdd_span(grid, 0.0, 0.0) == (5, 6)
    assert pdd_head_weights(grid, 0, 1).tolist() == [1, 0, 0, 0, 0, 0]  # the full head alone
    assert pdd_head_weights(grid, 4, 5).tolist() == [1, 0, 0, 0, 1, 0]  # the full head plus its own offset
    w = pdd_head_weights(grid, 1, 4)  # a step over three intervals: their offsets by step size
    dt = [grid[k] - grid[k + 1] for k in (1, 2, 3)]
    assert w.tolist() == pytest.approx([1, dt[0] / sum(dt), dt[1] / sum(dt), dt[2] / sum(dt), 0, 0])
    w = pdd_head_weights(grid, 0, 2)  # interval 0 has no offset: its share of the step stays on the full head
    assert w.tolist() == pytest.approx([1, (grid[1] - grid[2]) / (grid[0] - grid[2]), 0, 0, 0, 0])
    uniform = M.pdd_grid(4, 12.0)  # ComfyUI's grid for heads distilled on the shifted uniform base grid
    assert uniform == pytest.approx([1.0, 36 / 37, 12 / 13, 0.8, 0.0])


def test_a_step_over_several_intervals_takes_their_weighted_mean():
    ours, ref, heads = pair(3)
    x, enc = torch.randn(1, 24, 4, 4), torch.randn(1, 5, 40)
    with torch.no_grad():
        per_head = [-ref_image(ref, heads, x, 1.0, enc, k) for k in range(HEADS)]
        got = ours(x, 1000.0, enc, None, sigma_next=0.0)  # one step down the whole grid
        dt = [GRID[k] - GRID[k + 1] for k in range(HEADS)]
        want = sum(d * v for d, v in zip(dt, per_head)) / sum(dt)
        assert torch.allclose(got, want, atol=2e-5, rtol=1e-4)
        two = ours(x, 1000.0, enc, None, sigma_next=GRID[2])
        want = (dt[0] * per_head[0] + dt[1] * per_head[1]) / (dt[0] + dt[1])
        assert torch.allclose(two, want, atol=2e-5, rtol=1e-4)
    weights = ours.pdd_weights(torch.tensor([1.0, 0.5, 0.2]))
    assert weights.tolist() == [[1, 0, 0], [1, 1, 0], [1, 0, 1]]
    assert MiniMaxH3Transformer(**{**TINY, "pdd_steps": 1, "pdd_sigmas": None}).pdd_weights(torch.tensor([0.5])) is None
    with pytest.raises(ValueError, match="decreasing"):
        MiniMaxH3Transformer(**{**TINY, "pdd_sigmas": [1.0, 0.5, 0.6, 0.0]})
    with pytest.raises(ValueError, match="FastH3"):  # a video + audio checkpoint with stacked heads
        MiniMaxH3Transformer(**{**TINY, "audio_latents_dim": 32})
    six = MiniMaxH3Transformer(**{**TINY, "pdd_steps": 6, "pdd_sigmas": None})
    assert tuple(six.pdd_sigmas) == H3_IMAGE_PDD_SIGMAS  # 6 heads without a grid: the released one


# ---------------------------------------------------------------- the released files, by their headers

QUANT_SIDE = (".comfy_quant", ".weight_scale", *HSF.GROUPED_PARTS)
DTYPES = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16, "I8": torch.int8, "U8": torch.uint8,
          "F8_E4M3": torch.uint8}


def meta_state(entries: dict) -> dict:
    return {k: torch.empty(shape, dtype=DTYPES[dt], device="meta") for k, (dt, shape) in entries.items()}


def test_released_dit_headers_fit_the_port():
    """Comfy-Org's three diffusion models: the port's names and shapes, the config ComfyUI's model_detection reads
    (no audio stream, 6 stacked heads, the pruned AdaLN table), the fp32 islands and the heads' sigma grid."""
    meta = json.loads(HEADERS["dit_metadata"]["config"])["transformer"]
    assert meta["pdd_steps"] == 6 and tuple(meta["pdd_sigmas"]) == H3_IMAGE_PDD_SIGMAS
    islands = {"video_patch_proj.weight", "video_patch_proj.bias", "final_layer.video_out.weight",
               "final_layer.video_out.bias", "rope.inv_freq", "adaln_t_table"}
    for name, weight in (("dit_bf16", "BF16"), ("dit_int8", "I8"), ("dit_w4a8", "I8")):
        entries = {k: v for k, v in HEADERS[name].items() if not k.endswith(QUANT_SIDE)}
        assert {k for k, (dt, _) in entries.items() if dt == "F32"} == islands, name
        assert {dt for k, (dt, _) in entries.items() if k.endswith(".adaln_proj.linear.weight")} == {"F16"}
        assert entries["blocks.0.attn.qkv_proj.weight"][0] == weight
        assert not any(k.startswith("audio_") or ".audio_out." in k or k.startswith("time_embedder.") for k in entries)
        assert sum(1 for k in entries if k.endswith("comfy_attention.config")) == 50  # dropped by original_to_port
        assert C.is_h3_image_state(entries) and not C.is_h3_state(entries)
        state = meta_state(entries)
        if name == "dit_w4a8":  # the packed 4-bit rows: the weight's own shape is in its scales
            for k in [k for k in state if k.endswith(".weight") and f"{k}_s_rel" in HEADERS[name]]:
                rows, groups = HEADERS[name][f"{k}_s_rel"][1]
                assert tuple(state[k].shape) == (rows, groups * 16 // 2)
                assert HEADERS[name][f"{k}_s_channel"][1] == [rows] and HEADERS[name][f"{k}_codebook"][1] == [16]
                state[k] = torch.empty(rows, groups * 16, device="meta")
        state = C.original_to_port(state, interleaved=False, heads=0, head_dim=0)
        cfg = C.config_from_state(state)
        assert state.pop("rope.inv_freq").shape == (16,)
        assert (cfg["hidden_size"], cfg["num_layers"], cfg["num_attention_heads"], cfg["attention_head_dim"],
                cfg["ffn_hidden_size"], cfg["text_dim"], cfg["token_refiner_num_layers"], cfg["latents_dim"]) == (
            5376, 50, 56, 128, 14336, 5120, 2, 24)
        assert cfg["audio_latents_dim"] is None and cfg["pdd_steps"] == 6
        assert (cfg["adaln_curve_grid"], cfg["time_embed_dim"]) == (1025, 8)
        with torch.device("meta"):
            model = MiniMaxH3Transformer.from_config(cfg)
        assert tuple(model.pdd_sigmas) == H3_IMAGE_PDD_SIGMAS
        want = {k: tuple(v.shape) for k, v in model.state_dict().items()}
        assert {k: tuple(v.shape) for k, v in state.items()} == want, name
    assert len([k for k in HEADERS["dit_w4a8"] if k.endswith(".weight_s_rel")]) == 200  # the 50 blocks' 4 Linears


def test_released_training_adapter_fits_the_port_names():
    """ostris' minimax_h3_image_training_adapter_v1 (what ai-toolkit trains the model with): every layer it names is
    a Linear of the port with those shapes, in the layout attach_training_adapter reads."""
    state = C.original_to_port(meta_state({k: v for k, v in HEADERS["dit_bf16"].items()}), interleaved=False,
                               heads=0, head_dim=0)
    with torch.device("meta"):
        model = MiniMaxH3Transformer.from_config(C.config_from_state(state))
    modules = dict(model.named_modules())
    adapter = HEADERS["adapter"]
    names = {k.split(".lora_")[0][len("diffusion_model."):] for k in adapter}
    assert len(names) == 208 and len(adapter) == 416  # 50 blocks + 2 token-refiner blocks, 4 Linears each
    assert sum(n.startswith("token_refiner.") for n in names) == 8 and not any("final_layer" in n for n in names)
    for name in names:
        m = modules[name]
        down = tuple(adapter[f"diffusion_model.{name}.lora_A.weight"][1])
        up = tuple(adapter[f"diffusion_model.{name}.lora_B.weight"][1])
        assert down == (16, m.in_features) and up == (m.out_features, 16), name
        assert ARCH.lora_module_name(name) == name
    assert all(".alpha" not in k for k in adapter)  # alpha = rank: the adapter applies at scale 1


def test_released_folder_shards_map_onto_the_port():
    """MiniMaxAI's transformer/ (original names, full AdaLN, its own single output head) and transformer_diffusers/
    (pdd_proj_out.0-5), by the first and last shard of each, and pdd_fused_heads.safetensors."""
    full = dict(TINY, hidden_size=5376, num_layers=50, token_refiner_num_layers=2, num_attention_heads=56,
                attention_head_dim=128, ffn_hidden_size=14336, text_dim=5120, time_embed_hidden_size=5376,
                time_embed_dim=2688, rope_inv_freq_len=16, pdd_steps=6, pdd_sigmas=None)
    with torch.device("meta"):
        want = {k: tuple(v.shape) for k, v in MiniMaxH3Transformer(**full).state_dict().items()}
    fused = HEADERS["pdd_fused_heads"]
    assert fused == {"video_out.weight": ["F32", [6, 96, 5376]], "video_out.bias": ["F32", [6, 96]]}
    seen = set()
    for name in ("original_first_shard", "original_last_shard"):
        state = C.original_to_port(meta_state(HEADERS[name]), interleaved=False, heads=56, head_dim=128)
        state.pop("rope.inv_freq", None)
        for k, v in state.items():
            if k.startswith("final_layer.video_out."):  # the undistilled head: replaced by the fused ones
                assert tuple(v.shape) in ((96, 5376), (96,))
                continue
            assert want[k] == tuple(v.shape), k
            seen.add(k)
    assert {"video_patch_proj.weight", "condition_proj.weight", "time_embedder.proj_in.weight",
            "final_layer.adaln_proj.linear.weight", "token_refiner.final_norm.weight"} <= seen
    assert not any("audio" in k for k in HEADERS["original_first_shard"])
    heads = {}
    for name in ("diffusers_first_shard", "diffusers_last_shard"):
        entries = HEADERS[name]
        state = meta_state(entries)
        qkv = {k for k in state if ".attn.to_" in k and k.endswith(("to_q.weight", "to_k.weight", "to_v.weight"))}
        whole = {k.rsplit(".to_", 1)[0] for k in qkv}
        assert all(all(f"{s}.to_{p}.weight" in state for p in "qkv") for s in whole)
        state = C.diffusers_to_port(state)
        for k, v in state.items():
            if C.PDD_HEAD.match(k):
                heads[k] = v
                continue
            assert want[k] == tuple(v.shape), k
    assert set(heads) == {f"pdd_proj_out.{i}.{p}" for i in range(6) for p in ("weight", "bias")}
    assert all(tuple(heads[f"pdd_proj_out.{i}.weight"].shape) == (96, 5376) for i in range(6))


def test_released_vae_headers_build():
    comfy, diff = HEADERS["vae_comfy"], HEADERS["vae_diffusers"]
    assert {dt for dt, _ in comfy.values()} == {"F16"} and "latents_mean" not in comfy
    assert comfy["post_quant_conv.weight"][1] == [24, 24, 1, 1] and comfy["quant_conv.weight"][1] == [48, 48, 1, 1, 1]
    assert comfy["decoder.proj_out.weight"][1] == [3 * 16 * 16, 1024] and "decoder.mask_token" in comfy
    state = meta_state(comfy)
    assert IL._original_heads(state, "vae") == 16
    state = HSF.vae_original_to_diffusers(state, heads=16)
    assert {k: tuple(v.shape) for k, v in state.items()} == {k: tuple(s) for k, (_, s) in diff.items()}
    decoder = V.decoder_config_from_state(state)
    assert decoder == V.DECODER_CONFIG  # 24 blocks, 16 heads of 64, 4 register tokens, ffn x4
    with torch.device("meta"):
        vae = HSF.vae_class()(**{k: v for k, v in HSF.VAE_CONFIG.items() if not k.startswith("decoder_")},
                              decoder_num_layers=1)
        vae.decoder = V.decoder_class()(rope_dim=32, rope_theta=100.0, in_channels=24, out_channels=3, patch_size=16,
                                        num_layers=24, num_attention_heads=16, attention_head_dim=64,
                                        num_register_tokens=4, ffn_mult=4, norm_eps=1e-5)
        vae.post_quant_conv = nn.Conv2d(24, 24, 1)
    assert {k: tuple(v.shape) for k, v in vae.state_dict().items()} == {k: tuple(v.shape) for k, v in state.items()}
    # the released configs' latent statistics are H3's own
    assert len(HSF.VAE_CONFIG["latents_mean"]) == 24 and HSF.VAE_CONFIG["latents_mean"][0] == 0.858090341091156


# ---------------------------------------------------------------- ComfyUI's W4A8 / W6A8 weights


def pack_codes(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """comfy-kitchen's _pack_codes: K / 2 bytes of nibbles (the even column low), then at 6 bits K / 4 bytes of the
    top 2 bits (column c in byte c // 4 at bit 2 * (c % 4))."""
    codes = codes.to(torch.int32)
    low = ((codes[:, 0::2] & 0xF) | ((codes[:, 1::2] & 0xF) << 4)).to(torch.uint8)
    if bits == 4:
        return low.view(torch.int8).contiguous()
    high = torch.zeros(codes.shape[0], codes.shape[1] // 4, dtype=torch.int32)
    for j in range(4):
        high |= ((codes[:, j::4] >> 4) & 0x3) << (2 * j)
    return torch.cat([low, high.to(torch.uint8)], dim=1).view(torch.int8).contiguous()


def test_grouped_int_weights_decode_to_int8_convrot():
    g = torch.Generator().manual_seed(4)
    n, k = 6, 512
    s_rel = (torch.rand(n, k // 16, generator=g) * 100 + 20).to(torch.float8_e4m3fn)
    s_channel = torch.rand(n, generator=g) * 0.01 + 0.001
    codebook = torch.sort(torch.randn(16, generator=g).clamp(-1, 1))[0]
    codes = torch.randint(0, 16, (n, k), generator=g)
    packed = pack_codes(codes, 4)
    assert packed.shape == (n, k // 2) and torch.equal(Q.unpack_grouped_codes(packed, k, 4).long(), codes)
    want = (codebook[codes].view(n, k // 16, 16) * s_rel.float()[:, :, None]).view(n, k).round().clamp(-127, 127)
    got = Q.decode_grouped_int8(packed, s_rel, codebook, rows=4)  # in row chunks
    assert got.dtype == torch.int8 and torch.equal(got.float(), want)
    uniform = Q.decode_grouped_int8(packed, s_rel, None)  # no codebook: the uniform code - 8
    assert torch.equal(uniform.float(), ((codes - 8).float().view(n, k // 16, 16)
                                         * s_rel.float()[:, :, None]).view(n, k).round().clamp(-127, 127))
    qw = Q.quant_weight_from_grouped({"format": "asym_w4a8_int8"}, packed, s_rel, s_channel, codebook)
    assert qw.fmt == "int8_convrot" and qw.group == 256 and qw.orig_shape == (n, k) and qw.scale.shape == (n, 1)
    # ComfyUI's dequantize: the int8 grid times the row scale, rotated back (comfy-kitchen's
    # dequantize_w4a8_int8_weight)
    assert torch.allclose(qw.dequantize(torch.float32), Q.rotate_weight(want * s_channel[:, None], 256), atol=1e-6)
    codes6 = torch.randint(1, 64, (n, k), generator=g)
    packed6 = pack_codes(codes6, 6)
    assert packed6.shape == (n, 3 * k // 4) and torch.equal(Q.unpack_grouped_codes(packed6, k, 6).long(), codes6)
    got6 = Q.quant_weight_from_grouped({"format": "w6a8_int8", "params": {"group_size": 16}}, packed6, s_rel,
                                       s_channel)
    want6 = ((codes6 - 32).float().view(n, k // 16, 16) * s_rel.float()[:, :, None]).view(n, k).round().clamp(-127, 127)
    assert torch.equal(got6.qdata.float(), want6)
    # groups of 32 columns, as Comfy-Org's H3 w6a8 files have them (their spec: group_size 32, convrot_groupsize 256)
    s32 = (torch.rand(n, k // 32, generator=g) * 3 + 1).to(torch.float8_e4m3fn)
    got32 = Q.quant_weight_from_grouped({"format": "w6a8_int8", "group_size": 32, "convrot": True,
                                         "convrot_groupsize": 256}, packed6, s32, s_channel)
    want32 = ((codes6 - 32).float().view(n, k // 32, 32) * s32.float()[:, :, None]).view(n, k).round().clamp(-127, 127)
    assert got32.orig_shape == (n, k) and torch.equal(got32.qdata.float(), want32)
    with pytest.raises(ValueError, match="no codebook"):
        Q.decode_grouped_int8(packed6, s_rel, codebook)
    with pytest.raises(ValueError, match="does not match"):
        Q.decode_grouped_int8(packed[:, :-1].contiguous(), s_rel, codebook)
    with pytest.raises(ValueError, match="only read for MiniMax H3"):
        Q.quant_weight_from_comfy({"format": "asym_w4a8_int8"}, packed, s_rel, None)


def comfy_file(tmp_path, model, name, metadata=True, quant=None):
    """The port's state as a Comfy-Org file: diffusion_model., rope.inv_freq, a comfy_attention.config per block, the
    heads' grid in the "config" metadata. quant "w4a8": the blocks' attention output as ComfyUI's 4-bit weight."""
    from safetensors.torch import save_file

    sd = {f"diffusion_model.{k}": v.contiguous() for k, v in model.state_dict().items()}
    sd["diffusion_model.rope.inv_freq"] = inv_freq(TINY["rope_inv_freq_len"])
    packed = {}
    for i in range(len(model.blocks)):
        sd[f"diffusion_model.blocks.{i}.attn.comfy_attention.config"] = torch.zeros(37, dtype=torch.uint8)
        if quant == "w4a8":
            key = f"diffusion_model.blocks.{i}.attn.out_proj"
            w = sd.pop(f"{key}.weight")  # [48, 64]: K = 64, rotation group 64 here
            g = torch.Generator().manual_seed(i)
            codes = torch.randint(0, 16, w.shape, generator=g)
            codebook = torch.linspace(-1, 1, 16)
            s_rel = torch.full((w.shape[0], w.shape[1] // 16), 100.0).to(torch.float8_e4m3fn)
            s_channel = torch.full((w.shape[0],), 0.002)
            sd[f"{key}.weight"] = pack_codes(codes, 4)
            sd[f"{key}.weight_s_rel"], sd[f"{key}.weight_s_channel"] = s_rel, s_channel
            sd[f"{key}.weight_codebook"] = codebook
            spec = json.dumps({"format": "asym_w4a8_int8", "convrot_groupsize": 64})
            sd[f"{key}.comfy_quant"] = torch.tensor(list(spec.encode()), dtype=torch.uint8)
            grid = (codebook[codes] * float(s_rel.float()[0, 0])).round().clamp(-127, 127)  # fp8: 100 is stored as 96
            packed[f"blocks.{i}.attn.out_proj"] = Q.rotate_weight(grid * 0.002, 64)
    meta = {"config": json.dumps({"transformer": {"pdd_steps": HEADS, "pdd_sigmas": GRID}})} if metadata else None
    path = tmp_path / name
    save_file(sd, str(path), metadata=meta)
    return path, packed


# ---------------------------------------------------------------- single files and folders


def full_model(seed=5):
    """A tiny image model with full AdaLN (a time embedder), as MiniMaxAI's folders hold it."""
    ours, _, _ = pair(seed)
    return ours


def test_single_file_loads_and_is_told_from_h3(tmp_path, monkeypatch):
    monkeypatch.setenv("RULZ_H3_FULL_ADALN", "1")  # (pruning on load is H3's own, tested there)
    model = full_model()
    x, enc = torch.randn(1, 24, 4, 4), torch.randn(1, 6, 40)
    path, _ = comfy_file(tmp_path, model, "minimax_h3_image_pruned_bf16.safetensors")
    assert HSF.is_h3_image_file(path) and not HSF.is_h3_file(path)
    assert archs.arch_from_transformer_file(str(path)) == "minimax_h3_image"
    loaded = IL.load_transformer_file(path, "cpu", quant="none", dtype=torch.float32)
    assert loaded.pdd_steps == HEADS and loaded.pdd_sigmas == GRID and not loaded.has_audio
    with torch.no_grad():
        for sigma, nxt in ((1.0, 0.8), (0.8, 0.45), (0.6, None)):
            want = model(x, sigma * 1000.0, enc, None, sigma_next=nxt)
            assert torch.allclose(loaded(x, sigma * 1000.0, enc, None, sigma_next=nxt), want, atol=1e-5)
    assert torch.allclose(ARCH.load_any_transformer(str(path), "cpu", "none")(x, 500.0, enc, None).float(),
                          model(x, 500.0, enc, None), atol=0.05, rtol=0.05)  # bf16
    with pytest.raises(ValueError, match="MiniMax-H3-Image diffusion model"):  # H3's own loader names the model
        HSF.load_transformer_file(path, "cpu")
    # an H3 video + audio file is refused, by name
    torch.manual_seed(0)
    h3 = MiniMaxH3Transformer(**C.diffusers_config_to_port(TINY_D)).float()
    from safetensors.torch import save_file

    video = tmp_path / "minimax_h3_fl2va_bf16.safetensors"
    save_file({k: v.contiguous() for k, v in h3.state_dict().items()}, str(video))
    with pytest.raises(ValueError, match="video \\+ audio"):
        IL.load_transformer_file(video, "cpu")
    with pytest.raises(ValueError, match="not a MiniMax-H3-Image"):
        save_file({"x": torch.zeros(1)}, str(tmp_path / "other.safetensors"))
        IL.load_transformer_file(tmp_path / "other.safetensors", "cpu")


def test_six_heads_without_metadata_take_the_released_grid(tmp_path, monkeypatch):
    monkeypatch.setenv("RULZ_H3_FULL_ADALN", "1")
    torch.manual_seed(6)
    model = MiniMaxH3Transformer(**{**TINY, "pdd_steps": 6, "pdd_sigmas": None}).float().eval()
    path, _ = comfy_file(tmp_path, model, "requantized_by_someone.safetensors", metadata=False)
    loaded = IL.load_transformer_file(path, "cpu", quant="none", dtype=torch.float32)
    assert loaded.pdd_steps == 6 and tuple(loaded.pdd_sigmas) == H3_IMAGE_PDD_SIGMAS


def test_w4a8_file_loads_as_int8_convrot(tmp_path, monkeypatch):
    monkeypatch.setenv("RULZ_H3_FULL_ADALN", "1")
    model = full_model(7)
    path, packed = comfy_file(tmp_path, model, "minimax_h3_image_pruned_w4a8.safetensors", quant="w4a8")
    assert HSF.is_quantized_file(path)
    state = dict(HSF.iter_state(path))
    qw = state["diffusion_model.blocks.0.attn.out_proj.weight"]
    assert isinstance(qw, Q.QuantWeight) and qw.fmt == "int8_convrot" and qw.group == 64 and qw.orig_shape == (48, 64)
    assert not any(k.endswith(HSF.GROUPED_PARTS) for k in state)
    loaded = IL.load_transformer_file(path, "cpu", quant="auto", dtype=torch.float32)  # keeps the file's format
    lin = loaded.blocks[0].attn.out_proj
    assert isinstance(lin, Q.QuantLinear) and lin.qw.fmt == "int8_convrot"
    for name, want in packed.items():
        got = loaded.get_submodule(name).dequantized(torch.float32)
        assert torch.allclose(got, want, atol=1e-6), name
    x, enc = torch.randn(1, 24, 4, 4), torch.randn(1, 6, 40)
    with torch.no_grad():
        for name, want in packed.items():
            model.get_submodule(name).weight.copy_(want)
        assert torch.allclose(loaded(x, 700.0, enc, None).float(), model(x, 700.0, enc, None), atol=2e-2, rtol=2e-2)


def _shards(tdir, state, stem):
    from safetensors.torch import save_file

    tdir.mkdir(parents=True)
    save_file({k: v.contiguous() for k, v in state.items()}, str(tdir / f"{stem}.safetensors"))


def test_a_bf16_folder_pruned_on_load_runs_in_bf16(tmp_path):
    """MiniMaxAI's folder keeps the AdaLN in bf16 and the shard reader reads it so: pruning on load makes each AdaLN
    weight fp32, and its bias must follow (a bf16 bias beside it failed the first sample with "BFloat16 and Float")."""
    _, _, dstate = diffusers_state(8)
    model = MiniMaxH3Transformer(**TINY).float().eval()
    model.load_state_dict(C.fold_diffusers_pdd_heads(C.diffusers_to_port({k: v.clone() for k, v in dstate.items()})))
    orig = tmp_path / "MiniMaxAI__MiniMax-H3-Image"
    port = {k: v.clone() for k, v in model.state_dict().items()}
    fw, fb = C.unstack_pdd_heads(port.pop("final_layer.video_out.weight"), port.pop("final_layer.video_out.bias"), HEADS)
    for k in [k for k in port if k.endswith("qkv_proj.weight")]:
        port[k] = C.interleave_qkv(port[k], 2, 32)
    port["final_layer.video_out.weight"], port["final_layer.video_out.bias"] = torch.randn(96, 48), torch.randn(96)
    _shards(orig / "transformer", {k: v.bfloat16() for k, v in port.items()} | {"rope.inv_freq": inv_freq(4)}, "model")
    (orig / "transformer" / "config.json").write_text(json.dumps({
        "_class_name": "MiniMaxH3DiTModel", "audio_enabled": False, "audio_latents_dim": 32,
        **{k: v for k, v in TINY.items() if k not in ("audio_latents_dim", "pdd_steps", "pdd_sigmas", "rope_theta")}}))
    from safetensors.torch import save_file

    save_file({"video_out.weight": fw.contiguous(), "video_out.bias": fb.contiguous()}, str(orig / IL.PDD_FILE))
    (orig / IL.PDD_CONFIG).write_text(json.dumps({"num_inference_steps": HEADS, "sigmas": GRID}))
    m = IL.load_transformer(orig, "cpu", quant="none", dtype=torch.bfloat16)
    assert m.adaln_rank is not None  # pruned on load
    for proj in [b.adaln_proj.linear for b in m.blocks] + [m.final_layer.adaln_proj.linear]:
        assert proj.weight.dtype == proj.bias.dtype == torch.float32
    x, enc = torch.randn(1, 24, 4, 6), torch.randn(1, 5, 40)
    with torch.no_grad():
        got = m(x.bfloat16(), 450.0, enc.bfloat16(), None, sigma_next=0.0).float()
        want = model(x, 450.0, enc, None, sigma_next=0.0)
    assert torch.isfinite(got).all() and (got - want).abs().max() / want.abs().max() < 0.15


def test_both_folder_layouts_load_the_same_model(tmp_path, monkeypatch):
    from safetensors.torch import save_file

    monkeypatch.setenv("RULZ_H3_FULL_ADALN", "1")
    ref, heads, dstate = diffusers_state(8)
    model = MiniMaxH3Transformer(**TINY).float().eval()
    model.load_state_dict(C.fold_diffusers_pdd_heads(C.diffusers_to_port({k: v.clone() for k, v in dstate.items()})))
    x, enc = torch.randn(1, 24, 4, 6), torch.randn(1, 5, 40)
    # MiniMaxAI's own layout: transformer/ (interleaved qkv, a single head that is none of the six),
    # pdd_fused_heads.safetensors, pdd_config.json
    orig = tmp_path / "MiniMaxAI__MiniMax-H3-Image"
    port = {k: v.clone() for k, v in model.state_dict().items()}
    fw, fb = C.unstack_pdd_heads(port.pop("final_layer.video_out.weight"), port.pop("final_layer.video_out.bias"), HEADS)
    for k in [k for k in port if k.endswith("qkv_proj.weight")]:
        port[k] = C.interleave_qkv(port[k], 2, 32)
    port["final_layer.video_out.weight"], port["final_layer.video_out.bias"] = torch.randn(96, 48), torch.randn(96)
    port["rope.inv_freq"] = inv_freq(4)
    _shards(orig / "transformer", port, "model")
    (orig / "transformer" / "config.json").write_text(json.dumps({
        "_class_name": "MiniMaxH3DiTModel", "audio_enabled": False, "audio_latents_dim": 32,
        **{k: v for k, v in TINY.items() if k not in ("audio_latents_dim", "pdd_steps", "pdd_sigmas", "rope_theta")}}))
    assert not IL.is_h3_image_dir(tmp_path)
    with pytest.raises(FileNotFoundError, match="per-step output heads"):
        IL.load_transformer(orig, "cpu")
    save_file({"video_out.weight": fw.contiguous(), "video_out.bias": fb.contiguous()}, str(orig / IL.PDD_FILE))
    (orig / IL.PDD_CONFIG).write_text(json.dumps({"num_inference_steps": HEADS, "sigmas": GRID}))
    assert IL.is_h3_image_dir(orig) and IL.find_transformer(orig)[1] == "original" and IL.folder_sigmas(orig) == GRID
    assert archs.detect_arch(TrainConfig(model_path=str(orig)), orig).name == "minimax_h3_image"
    assert IL.resolve_model_dir(str(orig)) == orig.resolve()
    a = IL.load_transformer(orig, "cpu", quant="none", dtype=torch.float32)
    # diffusers' layout: transformer_diffusers/ alone, the grid in modular_model_index.json
    diff = tmp_path / "diffusers_only"
    _shards(diff / "transformer_diffusers", dstate, "diffusion_pytorch_model")
    (diff / "transformer_diffusers" / "config.json").write_text(json.dumps({
        "_class_name": "MiniMaxH3ImageTransformer2DModel", "num_pdd_heads": HEADS,
        **{k: v for k, v in TINY_D.items() if k != "audio_in_channels"}}))
    (diff / "modular_model_index.json").write_text(json.dumps({"sample_sigmas": GRID}))
    assert IL.find_transformer(diff)[1] == "diffusers" and IL.folder_sigmas(diff) == GRID
    assert archs.detect_arch(TrainConfig(model_path=str(diff)), diff).name == "minimax_h3_image"
    b = IL.load_transformer(diff, "cpu", quant="none", dtype=torch.float32)
    with torch.no_grad():
        for sigma, nxt in ((1.0, 0.8), (0.45, 0.0), (0.7, None)):
            want = model(x, sigma * 1000.0, enc, None, sigma_next=nxt)
            assert torch.allclose(a(x, sigma * 1000.0, enc, None, sigma_next=nxt), want, atol=1e-5)
            assert torch.allclose(b(x, sigma * 1000.0, enc, None, sigma_next=nxt), want, atol=1e-5)
        step = ref_image(ref, heads, x, 0.45, enc, 2)
        assert torch.allclose(a(x, 450.0, enc, None, sigma_next=0.0), -step, atol=2e-5, rtol=1e-4)
    assert IL.folder_sigmas(tmp_path) == list(H3_IMAGE_PDD_SIGMAS)  # neither file: the released grid
    # an H3 video folder (the same transformer class, with its audio stream) is not an image folder
    h3 = tmp_path / "h3" / "transformer"
    h3.mkdir(parents=True)
    (h3 / "config.json").write_text(json.dumps({"_class_name": "MiniMaxH3DiTModel"}))
    assert not IL.is_h3_image_dir(h3.parent)
    assert archs.detect_arch(TrainConfig(model_path=str(h3.parent)), h3.parent).name == "minimax_h3"


# ---------------------------------------------------------------- VAE

TINY_VAE = dict(block_out_channels=(8, 8, 8, 8, 8, 16), layers_per_block=1, norm_num_groups=4)
TINY_DECODER = {**V.DECODER_CONFIG, "decoder_num_layers": 2, "decoder_num_attention_heads": 2,
                "decoder_attention_head_dim": 16}


def tiny_vae_state(seed=0):
    """(a random tiny image VAE's diffusers-layout fp32 state, its config)."""
    torch.manual_seed(seed)
    config = {**{k: v for k, v in HSF.VAE_CONFIG.items() if not k.startswith("decoder_")}, **TINY_VAE, **TINY_DECODER}
    host = {k: v for k, v in config.items() if not k.startswith("decoder_")}
    vae = HSF.vae_class()(**host, decoder_num_layers=1, decoder_num_attention_heads=2, decoder_attention_head_dim=16)
    vae.decoder = V.decoder_class()(rope_dim=8, rope_theta=100.0, in_channels=24, out_channels=3, patch_size=16,
                                    num_layers=2, num_attention_heads=2, attention_head_dim=16, num_register_tokens=4,
                                    ffn_mult=4, norm_eps=1e-5)
    vae.post_quant_conv = nn.Conv2d(24, 24, 1)
    with torch.no_grad():
        for p in vae.parameters():
            p.normal_(0, 0.05)
    return {k: v.clone().float() for k, v in vae.state_dict().items()}, config


def reference_decoder(dec, rope, z):
    """diffusers' MiniMaxH3ImageViTDecoder2d.forward (fp32) on diffusers' own transformer blocks: every latent cell a
    token at its (h, w) position in [-1, 1), the register tokens and one zero token at position 0, each token decoded
    to a 16 x 16 pixel block."""
    b, c, h, w = z.shape
    x = dec.proj_in(z.permute(0, 2, 3, 1).reshape(b, h * w, c))
    n = x.shape[1]
    x = torch.cat([x, dec.register_tokens.expand(b, -1, -1), torch.zeros_like(x[:, :1])], dim=1)
    grids = [2.0 * (torch.arange(0.5, s, dtype=torch.float32) / s) - 1.0 for s in (h, w)]
    pos = torch.stack(torch.meshgrid(*grids, indexing="ij"), dim=-1).flatten(0, 1).unsqueeze(0).expand(b, -1, -1)
    pos = torch.cat([pos, pos.new_zeros(b, dec.num_register_tokens + 1, 2)], dim=1)
    rot = rope(pos)
    for block in dec.transformer_blocks:
        x = block(x, rot)
    x = dec.proj_out(dec.norm_out(x))[:, :n]
    return x.view(b, h, w, 3, 16, 16).permute(0, 3, 1, 4, 2, 5).reshape(b, 3, h * 16, w * 16)


def test_rope_is_diffusers_two_axis_embedding():
    from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import MiniMaxH3VideoRotaryPosEmbed

    pos = torch.rand(2, 5, 2) * 2 - 1
    cos, sin = V.Rope2d(32, 100.0)(pos)
    want_cos, want_sin = MiniMaxH3VideoRotaryPosEmbed(32, theta=100.0, num_axes=2)(pos)
    assert cos.shape == (2, 5, 1, 32) and torch.equal(cos, want_cos) and torch.equal(sin, want_sin)
    with pytest.raises(ValueError):
        V.Rope2d(30)


def test_vae_file_and_folders_decode_as_diffusers(tmp_path, monkeypatch):
    from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import MiniMaxH3VideoRotaryPosEmbed
    from safetensors.torch import save_file

    state, config = tiny_vae_state(2)
    monkeypatch.setattr(HSF, "VAE_CONFIG", {**HSF.VAE_CONFIG, **TINY_VAE})
    monkeypatch.setattr(V, "DECODER_CONFIG", TINY_DECODER)
    original = HSF.vae_diffusers_to_original({k: v.clone() for k, v in state.items()}, heads=2)
    original["decoder.mask_token"] = torch.zeros(1, 1, 32)  # in the released file, unused by the 2D decoder
    path = tmp_path / "minimax_h3_image_vae_fp16.safetensors"
    save_file({k: v.contiguous() for k, v in original.items()}, str(path))
    monkeypatch.setattr(IL, "_original_heads", lambda state, name, head_dim=64: 2)
    loaded = IL.load_vae_file(path, "cpu", dtype=torch.float32)
    assert isinstance(loaded.post_quant_conv, nn.Conv2d) and loaded.image_decoder_config == TINY_DECODER
    img = torch.rand(1, 3, 64, 96) * 2 - 1
    with torch.no_grad():
        lat = ARCH.encode_images(loaded, img)
        assert lat.shape == (1, 24, 4, 6) and torch.isfinite(lat).all()
        out = ARCH.decode_latents(loaded, lat)
        mean, std = HL._stats(loaded, lat)
        rope = MiniMaxH3VideoRotaryPosEmbed(8, theta=100.0, num_axes=2)
        frame = reference_decoder(loaded.decoder, rope, loaded.post_quant_conv(lat * std + mean))
        pm, ps = HL._pixel_norm(frame)
        want = ((frame * ps + pm) * 2 - 1).clamp(-1, 1)
    assert out.shape == img.shape and torch.allclose(out, want, atol=1e-5), float((out - want).abs().max())
    # taller than one 256 px tile: two tiles (rows 0-255 and 64-319 here), blended over their overlap as released
    with torch.no_grad():
        big = torch.randn(1, 24, 20, 16)
        tiled = ARCH.decode_latents_grad(loaded, big)
        ys, lens, overlaps = loaded._split_tiles(320, 256, 64)
        assert (ys, lens, overlaps) == ([0, 64], [256, 256], [192])
        z = loaded.post_quant_conv(big * std + mean)
        top, bottom = (reference_decoder(loaded.decoder, rope, z[..., y // 16: y // 16 + 16, :]) for y in ys)
        ramp = (torch.arange(192) / 192).view(1, 1, 192, 1)
        blend = top[..., 64:, :] * (1 - ramp) + bottom[..., :192, :] * ramp
        frame = torch.cat([top[..., :64, :], blend, bottom[..., 192:, :]], dim=-2)
        pm, ps = HL._pixel_norm(frame)
        assert tiled.shape == (1, 3, 320, 256) and torch.allclose(tiled, (frame * ps + pm) * 2 - 1, atol=1e-5)
        loaded.use_tiling = False  # one pass over the whole latent
        whole = reference_decoder(loaded.decoder, rope, z)
        assert torch.allclose(ARCH.decode_latents_grad(loaded, big), (whole * ps + pm) * 2 - 1, atol=1e-5)
        loaded.use_tiling = True
    # bf16 (fp32 encoder, norms, scales and proj_out): runs, and stays near fp32
    half = IL.load_vae_file(path, "cpu", dtype=torch.bfloat16)
    assert half.decoder.proj_out.weight.dtype == torch.float32 and half.decoder.proj_in.weight.dtype == torch.bfloat16
    assert half.encoder.conv_in.weight.dtype == torch.float32 and half.post_quant_conv.weight.dtype == torch.bfloat16
    with torch.no_grad():
        assert (ARCH.decode_latents(half, lat) - out).abs().mean() < 2e-2
    lat_g = lat.clone().requires_grad_(True)  # decoding with gradients (the identity loss) reaches the latents
    ARCH.decode_latents_grad(loaded, lat_g).mean().backward()
    assert lat_g.grad is not None and torch.isfinite(lat_g.grad).all()
    # the Hugging Face folder: vae/ in the original layout (MiniMaxH3ImageVAE), vae_diffusers/ in diffusers'
    stats = {"latents_mean": HSF.VAE_CONFIG["latents_mean"], "latents_std": HSF.VAE_CONFIG["latents_std"]}
    hf = tmp_path / "hf"
    (hf / "vae").mkdir(parents=True)
    save_file({k: v.contiguous() for k, v in original.items()}, str(hf / "vae" / "diffusion_pytorch_model.safetensors"))
    (hf / "vae" / "config.json").write_text(json.dumps({
        "_class_name": "MiniMaxH3ImageVAE", **stats,
        "vit_decoder_kwargs": {"dim_head": 16, "heads": 2, "num_layers": 2, "rope_dim_ratio": 0.5, "rope_theta": 100.0}}))
    only = tmp_path / "diffusers_only"
    (only / "vae_diffusers").mkdir(parents=True)
    save_file({k: v.contiguous() for k, v in state.items()},
              str(only / "vae_diffusers" / "diffusion_pytorch_model.safetensors"))
    (only / "vae_diffusers" / "config.json").write_text(json.dumps({
        "_class_name": "AutoencoderKLMiniMaxH3Image", **stats, **TINY_DECODER}))
    monkeypatch.undo()
    monkeypatch.setattr(HSF, "VAE_CONFIG", {**HSF.VAE_CONFIG, **TINY_VAE})
    with torch.no_grad():
        for folder in (hf, only):
            vae = IL.load_vae(folder, "cpu", dtype=torch.float32)
            assert torch.allclose(ARCH.decode_latents(vae, lat), out, atol=1e-5), folder.name
    with pytest.raises(FileNotFoundError):
        IL.load_vae(tmp_path, "cpu")


def test_other_h3_vaes_are_refused(tmp_path, monkeypatch):
    from safetensors.torch import save_file

    monkeypatch.setattr(HSF, "VAE_CONFIG", {**HSF.VAE_CONFIG, **TINY_VAE})
    video = HSF.vae_class()(**{**HSF.VAE_CONFIG, "decoder_num_layers": 1, "decoder_num_attention_heads": 2,
                               "decoder_attention_head_dim": 16}).float()
    orig = HSF.vae_diffusers_to_original({k: v.clone() for k, v in video.state_dict().items()}, heads=2)
    path = tmp_path / "minimax_h3_video_vae_fp16.safetensors"
    save_file({k: v.contiguous() for k, v in orig.items()}, str(path))
    monkeypatch.setattr(IL, "_original_heads", lambda state, name, head_dim=64: 2)
    with pytest.raises(ValueError, match="video VAE"):
        IL.load_vae_file(path, "cpu")
    save_file({"x": torch.zeros(1)}, str(tmp_path / "other.safetensors"))
    monkeypatch.undo()
    with pytest.raises(ValueError, match="not a MiniMax-H3-Image VAE"):
        IL.load_vae_file(tmp_path / "other.safetensors", "cpu")


# ---------------------------------------------------------------- text encoder, sampler, timesteps


def test_text_features_are_minimax_h3s():
    assert IL.encode_prompts is HL.encode_prompts and IL.prompt_tokens is HL.prompt_tokens
    assert IL.pad_embeds is HL.pad_embeds and ARCH._loading is IL
    assert "tokenizer/*" in IL.ALLOW_PATTERNS and "transformer/*" in IL.ALLOW_PATTERNS
    assert IL.PDD_FILE in IL.ALLOW_PATTERNS and IL.PDD_CONFIG in IL.ALLOW_PATTERNS
    assert "text_encoder/model-00011-of-00014.safetensors" in IL.ALLOW_PATTERNS  # layers 0-49: the first 11 shards
    assert "text_encoder/model-00012-of-00014.safetensors" not in IL.ALLOW_PATTERNS
    assert not any(p.startswith(("transformer_diffusers", "vae_diffusers", "processor")) for p in IL.ALLOW_PATTERNS)


def test_sampler_steps_the_released_grid_through_its_heads():
    assert h3image_sigmas(6).tolist() == pytest.approx(list(H3_IMAGE_PDD_SIGMAS))
    assert ARCH.sample_sigmas(None, 6, 2048, 2048, 5.0).tolist() == pytest.approx(list(H3_IMAGE_PDD_SIGMAS))
    twelve = h3image_sigmas(12)  # another count: the grid read piecewise-linearly in its index (ComfyUI)
    assert twelve.shape == (13,) and twelve[::2].tolist() == pytest.approx(list(H3_IMAGE_PDD_SIGMAS))
    assert float(twelve[1]) == pytest.approx((1.0 + H3_IMAGE_PDD_SIGMAS[1]) / 2)
    three = h3image_sigmas(3)
    assert three.tolist() == pytest.approx([1.0, H3_IMAGE_PDD_SIGMAS[2], H3_IMAGE_PDD_SIGMAS[4], 0.0])
    assert h3image_sigmas(3, GRID).tolist() == pytest.approx(GRID)
    ours, ref, heads = pair(9)
    enc, mask = torch.randn(1, 5, 40), torch.ones(1, 5, dtype=torch.bool)
    lat = sample(ours, enc, mask, None, None, width=64, height=96, steps=HEADS, seed=3, device="cpu")
    x = torch.randn((1, 24, 6, 4), generator=torch.Generator("cpu").manual_seed(3))
    with torch.no_grad():
        for step, (a, b) in enumerate(zip(GRID[:-1], GRID[1:])):  # the released loop: head `step`, an Euler step
            v = ref_image(ref, heads, x.to(torch.bfloat16).float(), a, enc, step)
            x = x + (b - a) * -v
    assert lat.shape == (1, 24, 6, 4) and torch.allclose(lat, x, atol=2e-4), float((lat - x).abs().max())
    assert ARCH.sample_sigmas(ours, HEADS, 64, 64, 5.0).tolist() == pytest.approx(GRID)  # a model's own grid
    assert ARCH.default_sample_steps == 6 and ARCH.default_sample_cfg == 1.0 and ARCH.distilled_sample_model("")
    calls = []
    lat2 = sample(ours, enc, mask, enc, mask, width=64, height=64, steps=2, cfg=2.0, seed=1, device="cpu",
                  progress=lambda i, n: calls.append((i, n)))
    assert calls == [(1, 2), (2, 2)] and lat2.shape == (1, 24, 4, 4) and torch.isfinite(lat2).all()


def test_reference_timesteps_are_the_six_sampling_sigmas_each_through_its_own_head():
    """ai-toolkit's rule (fixed_train_sigmas): only the sigmas the model samples at, evenly, used as drawn."""
    from rulztrainer.trainer import make_strata, sample_sigmas

    grid = torch.tensor(H3_IMAGE_PDD_SIGMAS, dtype=torch.float32)
    s = ARCH.reference_sigmas(60000, torch.Generator().manual_seed(0), 4096)
    assert ARCH.reference_sigmas_exact and not archs.arch_by_name("minimax_h3").reference_sigmas_exact
    assert sorted(set(s.tolist()), reverse=True) == grid[:-1].tolist()  # never 0, never between two grid points
    share = [float((s == g).float().mean()) for g in grid[:-1]]
    assert all(abs(x - 1 / 6) < 0.01 for x in share), share
    # the trainer leaves them alone: no shift, no range, no rounding to the 1000-step grid, no buckets
    cfg = TrainConfig(model_path="MiniMaxAI/MiniMax-H3-Image", timestep_sampling="reference", timestep_shift=3.0,
                      sigma_min=0.2, sigma_max=0.9, timestep_buckets=8)
    assert make_strata(cfg) is None
    drawn = sample_sigmas(cfg, 4000, torch.Generator().manual_seed(1), logit_shift=-1.5, tokens=4096)
    assert set(drawn.tolist()) == set(grid[:-1].tolist())
    # each one reaches the head of its own sampling step, through the trainer's t = sigma * 1000 and back
    for k, g in enumerate(grid[:-1]):
        back = float((g * 1000.0) / 1000.0)
        assert pdd_span(H3_IMAGE_PDD_SIGMAS, back) == (k, k + 1)
    # rounded to the 1000-step grid three of them would use the head before: why they are kept exact
    rounded = [pdd_span(H3_IMAGE_PDD_SIGMAS, round(float(g) * 1000) / 1000)[0] for g in grid[:-1]]
    assert rounded == [0, 0, 1, 3, 3, 5]
    six = MiniMaxH3Transformer(**{**TINY, "pdd_steps": 6, "pdd_sigmas": None})
    weights = six.pdd_weights(grid[:-1])
    assert weights[:, 0].tolist() == [1.0] * 6  # block 0 is the full head, block k >= 1 head k's offset from it
    assert weights[:, 1:].tolist() == torch.eye(6)[:, 1:].tolist()
    # the other samplers still draw anywhere, on the 1000-step grid
    free = sample_sigmas(TrainConfig(model_path="MiniMaxAI/MiniMax-H3-Image"), 2000, torch.Generator().manual_seed(2))
    assert len(set(free.tolist())) > 100
    assert ARCH.size_multiple == 32 and ARCH.latent_tag == "_h3ivae" and ARCH.block_token_px == 16
    assert ARCH.lora_format == "comfy_ab" and ARCH.n_blocks == 50 and ARCH.modelspec == "minimax-h3-image"


def test_lora_export_is_h3s_layout(tmp_path):
    from safetensors.torch import load_file

    ours, _, _ = pair(10)
    names = ARCH.lora_module_names(ours, ARCH.default_targets, [0], include_text_embedder=True)
    assert "token_refiner.blocks.0.mlp.fc1" in names and "blocks.0.attn.qkv_proj" in names
    assert not any("final_layer" in n for n in names)  # the output heads are never a LoRA target
    inj = L.inject_lora(ours, 4, 4.0, ["q"], names=names)
    path = tmp_path / "h3_image_lora.safetensors"
    L.save_lora(inj, path, dtype=torch.float32, fmt=ARCH.lora_format)
    assert set(load_file(str(path))) == {f"diffusion_model.{n}.lora_{p}.weight" for n in names for p in "AB"}
    assert ARCH.lora_module_name("blocks_12_attn_qkv_proj") == "blocks.12.attn.qkv_proj"


# ---------------------------------------------------------------- detection, downloads, caches, UI


def test_detection_downloads_and_caches(tmp_path):
    from PIL import Image

    from rulztrainer import dataset as D
    from rulztrainer.comfy_files import planned_downloads
    from rulztrainer.config import ARCH_TRAINING_ADAPTER, ARCH_TURBO, MODEL_PRESETS

    assert MODEL_PRESETS["MiniMax-H3-Image"] == "MiniMaxAI/MiniMax-H3-Image"
    for path in ("MiniMaxAI/MiniMax-H3-Image", "Comfy-Org/MiniMax-H3-Image",
                 "D:/models/minimax_h3_image_pruned_int8_convrot.safetensors",
                 "/workspace/models/MiniMaxAI__MiniMax-H3-Image"):
        assert TrainConfig(model_path=path).arch_name() == "minimax_h3_image", path
    # its neighbours keep their own families
    assert TrainConfig(model_path="MiniMaxAI/MiniMax-H3").arch_name() == "minimax_h3"
    assert TrainConfig(model_path="PrunaAI/P-H3-Image").arch_name() == "p_h3_image"
    assert family_named_in("minimax_h3_fl2va_pruned_int8_convrot.safetensors") == "minimax_h3"
    # circlestone-labs' image training adapter is H3's own, whatever its name
    assert family_named_in("circlestone-labs/MiniMax-H3-Image-Training-Adapter") == "minimax_h3"
    assert family_named_in("minimax_h3_image_training_adapter.safetensors") == "minimax_h3"
    # ostris' adapter for MiniMax-H3-Image carries a version, and is this model's
    assert family_named_in("minimax_h3_image_training_adapter_v1.safetensors") == "minimax_h3_image"
    assert family_named_in("ostris/minimax_h3_training_adapter") == "minimax_h3"
    TrainConfig(model_path="MiniMaxAI/MiniMax-H3-Image",
                training_adapter="D:/loras/minimax_h3_image_training_adapter_v1.safetensors").validate()
    with pytest.raises(ValueError, match="another model's files"):
        TrainConfig(model_path="MiniMaxAI/MiniMax-H3",
                    training_adapter="D:/loras/minimax_h3_image_training_adapter_v1.safetensors").validate()
    TrainConfig(model_path="MiniMaxAI/MiniMax-H3",
                training_adapter="D:/loras/minimax_h3_image_training_adapter.safetensors").validate()
    with pytest.raises(ValueError, match="another model's files"):
        TrainConfig(model_path="MiniMaxAI/MiniMax-H3-Image",
                    training_adapter="D:/loras/minimax_h3_image_training_adapter.safetensors").validate()
    with pytest.raises(ValueError, match="another model's files"):
        TrainConfig(model_path="MiniMaxAI/MiniMax-H3",
                    sample_model_path="D:/models/minimax_h3_image_pruned_bf16.safetensors").validate()
    assert "minimax_h3_image" in archs.ARCH_NAMES and archs.get_arch(
        TrainConfig(model_path="MiniMaxAI/MiniMax-H3-Image")) is ARCH
    plan = {f: (r, n) for f, r, n in planned_downloads(TrainConfig(model_path="MiniMaxAI/MiniMax-H3-Image"))}
    assert plan == {
        "transformer_file": ("Comfy-Org/MiniMax-H3-Image",
                             "diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors"),
        # MiniMax H3's own text encoder file (the same bytes in both repos): one download serves both models
        "text_encoder_file": ("Comfy-Org/MiniMax-H3", "text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors"),
        "vae_file": ("Comfy-Org/MiniMax-H3-Image", "vae/minimax_h3_image_vae_fp16.safetensors"),
        # ostris' adapter, what ai-toolkit trains the model with
        "training_adapter": ("ostris/minimax_h3_training_adapter", "minimax_h3_image_training_adapter_v1.safetensors")}
    assert "minimax_h3_image" not in ARCH_TURBO
    assert ARCH_TRAINING_ADAPTER["minimax_h3_image"]["file"] in HEADERS["source"]  # the file the header test reads
    assert "training_adapter" not in {f for f, _, _ in planned_downloads(TrainConfig(
        model_path="MiniMaxAI/MiniMax-H3-Image", training_adapter=""))}
    other = {f: n for f, _, n in planned_downloads(TrainConfig(
        model_path="MiniMaxAI/MiniMax-H3-Image", model_file_quant="w4a8", text_encoder_file_quant="nvfp4_awq"))}
    assert other["transformer_file"].endswith("minimax_h3_image_pruned_w4a8.safetensors")
    assert other["text_encoder_file"].endswith("qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors")
    bf16 = {f: n for f, _, n in planned_downloads(TrainConfig(model_path="MiniMaxAI/MiniMax-H3-Image",
                                                              model_file_quant="bf16"))}
    assert bf16["transformer_file"].endswith("minimax_h3_image_pruned_bf16.safetensors")
    with pytest.raises(ValueError, match="model_file_quant"):
        planned_downloads(TrainConfig(model_path="MiniMaxAI/MiniMax-H3-Image", model_file_quant="nvfp4"))
    data = tmp_path / "data"
    data.mkdir()
    Image.new("RGB", (900, 600)).save(data / "a.png")
    cfg = TrainConfig(model_path="MiniMaxAI/MiniMax-H3-Image", dataset_dir=str(data), resolutions=[1.0])
    item = D.scan_dataset(cfg)[0]
    assert item.width % 32 == 0 and item.height % 32 == 0 and item.latent_path.name.endswith("_h3ivae.safetensors")
    assert D.text_cache_path(cfg, "p").name.endswith("_minimax_h3_image.safetensors")


def test_wizard_recommends_the_h3_image_setup():
    import webui

    cfg = TrainConfig(model_path="MiniMaxAI/MiniMax-H3-Image")
    two = webui.wizard_recommendations(cfg, "character", [{"total": 16311}, {"total": 16380}], None)
    assert two["multi_gpu"][0] == "model_split" and two["block_swap"][0] == 0
    assert two["model_file_quant"][0] == "int8_convrot" and two["text_encoder_file_quant"][0] == "int8_convrot"
    assert two["adapter_mode"][0] == "auto" and "ostris" in two["adapter_mode"][1]
    assert two["timestep_sampling"][0] == "reference" and two["resolutions"][0] == [1.0]
    assert (two["lora_rank"][0], two["lora_alpha"][0], two["learning_rate"][0]) == (16, 16.0, 1e-4)
    assert (two["sample_steps"][0], two["sample_cfg"][0]) == (6, 1.0)
    assert (two["probe_steps"][0], two["probe_cfg"][0]) == (6, 1.0)
    one = webui.wizard_recommendations(cfg, "style", [{"total": 16311}], None)
    assert one["multi_gpu"][0] == "off" and 20 <= one["block_swap"][0] <= 30 and one["lora_rank"][0] == 32
    assert one["text_encoder_file_quant"][0] == "nvfp4_awq"
    assert webui.N_BLOCKS["minimax_h3_image"] == 50 and "minimax_h3_image" in webui.EXTRA_TARGETS["token_refiner"][0]
    assert set(webui.file_quant_options("minimax_h3_image", "model")) == {"int8_convrot", "bf16", "w4a8"}
    assert set(webui.file_quant_options("minimax_h3_image", "text")) == {"int8_convrot", "nvfp4_awq", "bf16"}
    assert webui.preset_arch("MiniMax-H3-Image") == "minimax_h3_image"
    assert webui.MODEL_NOTES["MiniMax-H3-Image"][0] == "Experimental"
    assert webui.block_swap_plan(TrainConfig(model_path="MiniMaxAI/MiniMax-H3-Image", model_source="files",
                                             model_file_quant="w4a8"), 16.0, "auto")[0] > 0  # int8 in memory
    assert "## MiniMax-H3-Image" in webui.HELP


def test_clean_share_puts_its_share_of_steps_on_the_last_level_and_keeps_the_six():
    """clean_share on MiniMax-H3-Image (Fizgig's presets: 40): that share of the steps at the last sampling level
    (0.232, below 0.5), the rest evenly over the other five, solved by clean_share's own bisection; the levels stay
    exact. Without clean_share the curriculum's shift moves nothing (ai-toolkit's even draw)."""
    from rulztrainer.clean_share import CleanShare, offset_for
    from rulztrainer.trainer import sample_sigmas

    grid = torch.tensor(H3_IMAGE_PDD_SIGMAS[:-1], dtype=torch.float32)
    assert archs.clean_end_weights(grid, 0.0).tolist() == pytest.approx([1.0 / 6] * 6)
    cfg = TrainConfig(model_path="MiniMaxAI/MiniMax-H3-Image", timestep_sampling="reference", clean_share="40")
    off, got = offset_for(cfg, 40.0, 4096, ARCH)
    assert got == pytest.approx(40.0, abs=0.5) and off < 0  # more clean steps: a negative shift, as on a curve
    s = sample_sigmas(cfg, 60000, torch.Generator().manual_seed(3), off, tokens=4096, arch=ARCH)
    assert set(s.tolist()) == set(grid.tolist())
    share = [float((s == g).float().mean()) for g in grid]
    assert share[-1] == pytest.approx(0.40, abs=0.01)
    assert all(x == pytest.approx(0.12, abs=0.01) for x in share[:-1]), share
    # a schedule works as on any model: 11% (below even) then 40%
    cs = CleanShare(TrainConfig(model_path="MiniMaxAI/MiniMax-H3-Image", timestep_sampling="reference",
                                clean_share="11:2:40"), ARCH)
    low, high = cs.offset(0, 100, 4096), cs.offset(250, 100, 4096)
    assert low > 0 > high
    # no clean_share: a shift (the likeness curriculum's) leaves the even draw alone
    plain = TrainConfig(model_path="MiniMaxAI/MiniMax-H3-Image", timestep_sampling="reference")
    a = sample_sigmas(plain, 500, torch.Generator().manual_seed(4), -2.0, tokens=4096, arch=ARCH)
    b = ARCH.reference_sigmas(500, torch.Generator().manual_seed(4), 4096)
    assert torch.equal(a, b)
