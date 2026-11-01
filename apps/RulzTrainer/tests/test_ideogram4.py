"""Ideogram 4 support without model weights: the ported transformer against diffusers' Ideogram4Transformer2DModel
(tiny random config; padded batches per sample, the image-only "unconditional" sequence), the release's fp8 and nf4
formats and the fused attention layout on synthetic tensors with the real key names, LoRA naming and the export
ComfyUI / diffusers read, the sampler schedule against Ideogram's own, the JSON caption wrapper against the caption
verifier's rules, and architecture detection, caches and buckets."""
import json
import os
import re
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import archs  # noqa: E402
from rulztrainer import dataset as D  # noqa: E402
from rulztrainer import lora as L  # noqa: E402
from rulztrainer import quant as Q  # noqa: E402
from rulztrainer.config import TrainConfig  # noqa: E402
from rulztrainer.ideogram4 import loading as IL  # noqa: E402
from rulztrainer.ideogram4 import prompts as P  # noqa: E402
from rulztrainer.ideogram4 import sampler as S  # noqa: E402
from rulztrainer.ideogram4 import single_file as ISF  # noqa: E402
from rulztrainer.ideogram4.model import Ideogram4Transformer  # noqa: E402

diffusers = pytest.importorskip("diffusers")
from safetensors.torch import load_file, save_file  # noqa: E402

IDEO = archs.arch_by_name("ideogram4")
# every key of the released checkpoint (unsloth/ideogram-4-fp8 transformer/, the same names in Comfy-Org's
# ideogram4_int8_convrot / _fp8_scaled files), block index as N, read from the files' safetensors headers
RELEASE_KEYS = {
    "adaln_proj.bias", "adaln_proj.weight", "embed_image_indicator.weight", "final_layer.adaln_modulation.bias",
    "final_layer.adaln_modulation.weight", "final_layer.linear.bias", "final_layer.linear.weight", "input_proj.bias",
    "input_proj.weight", "layers.N.adaln_modulation.bias", "layers.N.adaln_modulation.weight",
    "layers.N.attention.norm_k.weight", "layers.N.attention.norm_q.weight", "layers.N.attention.o.weight",
    "layers.N.attention.qkv.weight", "layers.N.attention_norm1.weight", "layers.N.attention_norm2.weight",
    "layers.N.feed_forward.w1.weight", "layers.N.feed_forward.w2.weight", "layers.N.feed_forward.w3.weight",
    "layers.N.ffn_norm1.weight", "layers.N.ffn_norm2.weight", "llm_cond_norm.weight", "llm_cond_proj.bias",
    "llm_cond_proj.weight", "t_embedding.mlp_in.bias", "t_embedding.mlp_in.weight", "t_embedding.mlp_out.bias",
    "t_embedding.mlp_out.weight"}
# the release's fp8 Linear weights, each with a float32 per-row "<name>.weight_scale" [out] next to it
RELEASE_FP8 = {"adaln_proj", "final_layer.adaln_modulation", "final_layer.linear", "input_proj",
               "layers.N.adaln_modulation", "layers.N.attention.o", "layers.N.attention.qkv", "layers.N.feed_forward.w1",
               "layers.N.feed_forward.w2", "layers.N.feed_forward.w3", "llm_cond_proj", "t_embedding.mlp_in",
               "t_embedding.mlp_out"}
NF4_MAP = (-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635,
           -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725,
           0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0)

# the release's RoPE sections (24, 20, 20 of 128 frequencies) scaled to head_dim 64, as single files infer them
TINY = dict(in_channels=16, num_layers=2, attention_head_dim=64, num_attention_heads=2, intermediate_size=96,
            adaln_dim=32, llm_features_dim=40, rope_theta=5_000_000, mrope_section=(6, 5, 5), norm_eps=1e-5)


def ref_model(seed=0):
    """Random weights under which the attention, the text and the positions all move the output (norms near 1,
    sizeable gates): with this init a q/k swap in one block changes the output by about its own size, a mirrored
    image by a sixth of it."""
    torch.manual_seed(seed)
    ref = diffusers.Ideogram4Transformer2DModel(**TINY).float().eval()
    with torch.no_grad():
        for n, p in ref.named_parameters():
            if n.endswith(("norm1.weight", "norm2.weight", "norm_q.weight", "norm_k.weight", "llm_cond_norm.weight")):
                p.normal_(1.0, 0.2)
            else:
                p.normal_(0, 0.5 if "adaln_modulation" in n else 0.05)
    return ref


def pair(seed=0):
    """Our port and diffusers' model with the same random weights, fp32 (diffusers' to_q / to_k / to_v fused)."""
    ref = ref_model(seed)
    ours = Ideogram4Transformer(**TINY).float().eval()
    state = ISF.normalize_dit_state({k: v.clone() for k, v in ref.state_dict().items()})
    missing, unexpected = ours.load_state_dict(state, strict=True)
    assert not missing and not unexpected
    return ours, ref


def ref_call(ref, x, sigma, feats):
    """diffusers' forward on the pipeline's packed layout ([left padding][text][image], segment ids, indicators) for
    a batch: x [B, C, h, w], sigma [B], feats a list of [n_i, D] (n_i = 0: the image-only unconditional sequence).
    Returns the image tokens' output in the trainer's layout, in Ideogram's convention (clean - noise)."""
    b, c, h, w = x.shape
    lens = [f.shape[0] for f in feats]
    n = max(lens)
    pos, seg, ind = diffusers.Ideogram4Pipeline._prepare_ids(lens, h, w, n, x.device)
    hidden = torch.cat([torch.zeros(b, n, c), x.flatten(2).transpose(1, 2)], dim=1)
    enc = torch.zeros(b, n + h * w, feats[0].shape[-1])
    for i, f in enumerate(feats):
        enc[i, n - f.shape[0]:n] = f
    t = 1.0 - (sigma.float() * 1000.0) / 1000.0  # what the port computes from sigma * 1000
    out = ref(hidden_states=hidden, timestep=t, encoder_hidden_states=enc, position_ids=pos, segment_ids=seg,
              indicator=ind, return_dict=False)[0]
    return out[:, n:].transpose(1, 2).reshape(b, c, h, w)


def test_port_matches_diffusers_per_sample_with_padding():
    ours, ref = pair()
    x = torch.randn(2, 16, 3, 5)
    sigma = torch.tensor([0.7, 0.25])
    lens = (9, 4)
    enc = torch.randn(2, max(lens), 40)
    mask = torch.zeros(2, max(lens), dtype=torch.bool)
    for i, n in enumerate(lens):
        mask[i, :n] = True
        enc[i, n:] = 0.0  # pad_embeds zero-fills
    with torch.no_grad():
        got = ours(x, sigma * 1000.0, enc, mask)
        # the trainer's convention is noise - clean: the port negates Ideogram's clean - noise
        for i, n in enumerate(lens):  # each sample alone, as the reference pipeline computes it
            want = -ref_call(ref, x[i:i + 1], sigma[i:i + 1], [enc[i, :n]])
            assert torch.allclose(got[i:i + 1], want, atol=1e-5, rtol=1e-4), (got[i:i + 1] - want).abs().max()
        # and the reference's own left-padded batch gives the same
        want = -ref_call(ref, x, sigma, [enc[i, :n] for i, n in enumerate(lens)])
        assert torch.allclose(got, want, atol=1e-5, rtol=1e-4)
        # an all-true mask is no mask
        assert torch.equal(ours(x[:1], 700.0, enc[:1], torch.ones(1, 9, dtype=torch.bool)),
                           ours(x[:1], 700.0, enc[:1], None))
        # the comparison is sensitive: other text, or q and k swapped in one block, moves the output far more than
        # the tolerance
        assert (ours(x, sigma * 1000.0, torch.randn_like(enc) * mask[..., None], mask) - got).abs().mean() > 0.05
        w = ours.layers[1].attention.qkv.weight
        d = w.shape[0] // 3
        w.data = torch.cat([w[d:2 * d], w[:d], w[2 * d:]])
        assert (ours(x, sigma * 1000.0, enc, mask) - got).abs().mean() > 0.05


def test_image_only_sequence_is_the_unconditional_branch():
    ours, ref = pair(1)
    x = torch.randn(1, 16, 4, 4)
    with torch.no_grad():
        got = ours(x, torch.tensor([450.0]), torch.zeros(1, 0, 40), torch.zeros(1, 0, dtype=torch.bool))
        want = -ref_call(ref, x, torch.tensor([0.45]), [torch.zeros(0, 40)])
    assert torch.allclose(got, want, atol=1e-5, rtol=1e-4)


def test_bf16_gradients_checkpointing_and_layers():
    ours, _ = pair(2)
    m = ours.to(torch.bfloat16)
    m.gradient_checkpointing = True
    m.train()
    for p in m.parameters():
        p.requires_grad_(True)
    x = torch.randn(1, 16, 3, 5).to(torch.bfloat16)
    enc = torch.randn(1, 4, 40).to(torch.bfloat16)
    out = m(x, torch.tensor([500.0]), enc, None)
    assert out.shape == x.shape and out.dtype == torch.bfloat16 and torch.isfinite(out.float()).all()
    out.float().pow(2).mean().backward()
    assert all(p.grad is not None for p in m.layers.parameters())
    assert m.llm_cond_proj.weight.grad is not None and m.input_proj.weight.grad is not None
    with torch.no_grad():
        skipped = m(x, 500.0, enc, None, skip_layers=[1])
    assert not torch.equal(skipped, out.detach())
    assert len(m.layers) == 2 and m.latent_channels == 16 and m.text_dim == 40
    assert m.split_tail_modules() == [m.final_layer] and IDEO.compile_blocks(m) == list(m.layers)


def test_model_split_hand_off_matches():
    from rulztrainer.multigpu import split_model

    ours, _ = pair(3)
    x, enc = torch.randn(1, 16, 2, 2), torch.randn(1, 4, 40)
    with torch.no_grad():
        want = ours(x, 300.0, enc, None)
        split_model(ours, ["cpu", "cpu"])  # same device twice: exercises the hand-off path
        assert torch.equal(ours(x, 300.0, enc, None), want)


# ---------------------------------------------------------------- the release's formats and layouts


def _pattern(key: str) -> str:
    return re.sub(r"^layers\.\d+\.", "layers.N.", key)


def fp8_rows(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Ideogram's quantize_weight_to_fp8: e4m3 with a float32 per-row scale [out] (row max -> 448)."""
    scale = w.float().abs().amax(dim=1).clamp(min=1e-12) / 448.0
    return (w.float() / scale[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn), scale


def release_state(model: Ideogram4Transformer) -> tuple[dict, dict]:
    """(the model's weights in the release's fp8 layout, the dequantized weights the loader must give back)."""
    fp8, plain = {}, {}
    for k, v in model.state_dict().items():
        base = k[: -len(".weight")]
        if k.endswith(".weight") and isinstance(model.get_submodule(base), torch.nn.Linear):
            q, s = fp8_rows(v)
            fp8[k], fp8[f"{k}_scale"] = q, s
            plain[k] = q.float() * s[:, None]
        else:
            fp8[k], plain[k] = v.clone(), v.clone()
    return fp8, plain


def test_port_has_the_release_checkpoint_names():
    ours, _ = pair(4)
    assert {_pattern(k) for k in ours.state_dict()} == RELEASE_KEYS
    linears = {_pattern(n) for n, m in ours.named_modules() if isinstance(m, torch.nn.Linear)}
    assert linears == RELEASE_FP8  # every Linear is fp8 in the release; nothing else is
    assert ISF.is_ideogram4_state(ours.state_dict())


def test_release_fp8_folder_and_file_load_with_per_row_scales(tmp_path):
    ours, ref = pair(5)
    fp8, plain = release_state(ours)
    assert fp8["layers.0.attention.o.weight"].shape[0] == fp8["layers.0.attention.o.weight"].shape[1]  # square:
    # a per-row scale applied per column (broadcasting [out, in] * [out]) would pass silently there
    want = Ideogram4Transformer(**TINY).float().eval()
    want.load_state_dict(plain)
    mdir = tmp_path / "ideogram-4-fp8"
    (mdir / "transformer").mkdir(parents=True)
    (mdir / "transformer" / "config.json").write_text(json.dumps({"_class_name": "Ideogram4Transformer2DModel",
                                                                   **TINY}))
    save_file({k: v.contiguous() for k, v in fp8.items()}, str(mdir / "transformer" / "diffusion_pytorch_model.safetensors"))
    x, enc = torch.randn(1, 16, 3, 3), torch.randn(1, 5, 40)
    for loaded in (IL.load_transformer(mdir, "cpu", quant="none", dtype=torch.float32),
                   IL.load_any_transformer(str(mdir), "cpu", quant="none"),
                   ISF.load_transformer_file(mdir / "transformer" / "diffusion_pytorch_model.safetensors", "cpu",
                                             quant="none", dtype=torch.float32)):
        got = loaded.state_dict()
        for k, v in want.state_dict().items():  # load_any_transformer loads in bf16: its rounding
            tol = 1e-6 if got[k].dtype == torch.float32 else 2e-2 * float(v.abs().max())
            assert torch.allclose(got[k].float(), v, atol=tol), k
    loaded = IL.load_transformer(mdir, "cpu", quant="none", dtype=torch.float32)
    assert loaded.config["num_layers"] == 2 and loaded.config["llm_features_dim"] == 40
    with torch.no_grad():
        assert torch.allclose(loaded(x, 500.0, enc, None), want(x, 500.0, enc, None), atol=1e-6)
    assert archs.detect_arch(TrainConfig(model_path=str(mdir)), mdir).name == "ideogram4"
    # the release's per-row scale shapes, and ComfyUI's per-tensor one
    w = torch.randn(6, 4)
    q, s = fp8_rows(w)
    assert torch.allclose(ISF.dequantize_fp8(q, s), q.float() * s[:, None])
    assert torch.allclose(ISF.dequantize_fp8(q, s[:, None]), q.float() * s[:, None])
    assert torch.allclose(ISF.dequantize_fp8(q, torch.tensor(0.5)), q.float() * 0.5)
    with pytest.raises(ValueError, match="fp8 scale"):
        ISF.dequantize_fp8(q, torch.ones(4))
    with pytest.raises(ValueError, match="without their weight"):
        ISF.dequantize_release({"a.weight_scale": torch.ones(2)})


def nf4_pack(w: torch.Tensor, blocksize: int = 64):
    """A bitsandbytes nf4 weight as diffusers saves it (test oracle): codes of w / blockwise absmax in the NF4 code
    book, two per byte, high nibble first. Returns (packed, absmax, quant_map, quant_state blob, dequantized)."""
    qmap = torch.tensor(NF4_MAP)
    flat = w.float().reshape(-1)
    n = flat.numel()
    pad = (-n) % blocksize
    blocks = torch.cat([flat, flat.new_zeros(pad)]).view(-1, blocksize)
    absmax = blocks.abs().amax(1).clamp(min=1e-12)
    codes = ((blocks / absmax[:, None])[..., None] - qmap).abs().argmin(-1).reshape(-1)
    deq = (qmap[codes].view(-1, blocksize) * absmax[:, None]).reshape(-1)[:n].reshape(w.shape)
    codes = codes[:n]
    if codes.numel() % 2:
        codes = torch.cat([codes, codes.new_zeros(1)])
    packed = ((codes[0::2] << 4) | codes[1::2]).to(torch.uint8).reshape(-1, 1)
    meta = {"quant_type": "nf4", "absmax": "absmax", "blocksize": blocksize, "quant_map": "quant_map",
            "dtype": "bfloat16", "shape": list(w.shape)}
    blob = torch.tensor(list(json.dumps(meta).encode()), dtype=torch.uint8)
    return packed, absmax, qmap, blob, deq


def test_diffusers_nf4_layout_dequantizes_and_fuses_qkv(tmp_path):
    """diffusers' names (to_q / to_k / to_v / to_out.0) in bitsandbytes nf4, as ideogram-ai/ideogram-4-nf4 ships."""
    ref = ref_model(6)
    state, want = {}, {}
    for k, v in ref.state_dict().items():
        mod = ref.get_submodule(k[: -len(".weight")]) if k.endswith(".weight") else None
        if isinstance(mod, torch.nn.Linear):
            packed, absmax, qmap, blob, deq = nf4_pack(v)
            state[k], state[f"{k}.absmax"], state[f"{k}.quant_map"] = packed, absmax, qmap
            state[f"{k}.quant_state.bitsandbytes__nf4"] = blob
            want[k] = deq
        else:
            state[k] = want[k] = v.clone()
    path = tmp_path / "diffusion_pytorch_model.safetensors"
    save_file({k: v.contiguous() for k, v in state.items()}, str(path))
    read = ISF.read_state(path, torch.float32)
    assert set(read) == set(want)
    for k in want:
        assert torch.allclose(read[k], want[k], atol=1e-6), k
    loaded = ISF.load_transformer_file(path, "cpu", quant="none", dtype=torch.float32)
    sd = loaded.state_dict()
    for i in range(2):  # [q; k; v] by rows: Ideogram's order
        qkv = torch.cat([want[f"layers.{i}.attention.to_{n}.weight"] for n in "qkv"])
        assert torch.equal(sd[f"layers.{i}.attention.qkv.weight"], qkv)
        assert torch.equal(sd[f"layers.{i}.attention.o.weight"], want[f"layers.{i}.attention.to_out.0.weight"])
    with pytest.raises(ValueError, match="without their to_q"):
        ISF.fuse_diffusers_attention({"layers.0.attention.to_k.weight": torch.zeros(2, 2)})


def test_comfy_files_int8_and_scaled_fp8_load(tmp_path):
    """Comfy-Org's int8_convrot (blocks as int8_tensorwise, the rest bf16) and fp8_scaled (per-tensor scale) layouts."""
    ours, _ = pair(7)
    sd = ours.state_dict()
    int8, fp8, want8, wantf = {}, {}, {}, {}
    for k, v in sd.items():
        base = k[: -len(".weight")]
        lin = k.endswith(".weight") and isinstance(ours.get_submodule(base), torch.nn.Linear)
        int8[k] = fp8[k] = want8[k] = wantf[k] = v
        if lin and k.startswith("layers."):
            qw = Q.quantize_weight(v, "int8")
            int8[k], int8[f"{base}.weight_scale"] = qw.qdata, qw.scale
            int8[f"{base}.comfy_quant"] = torch.tensor(list(json.dumps(
                {"format": "int8_tensorwise", "convrot": False}).encode()), dtype=torch.uint8)
            want8[k] = qw.dequantize(torch.float32)
        if lin:
            s = v.abs().max() / 448.0
            fp8[k], fp8[f"{base}.weight_scale"] = (v / s).to(torch.float8_e4m3fn), s.reshape(())
            fp8[f"{base}.comfy_quant"] = torch.tensor(list(json.dumps({"format": "float8_e4m3fn"}).encode()),
                                                      dtype=torch.uint8)
            wantf[k] = fp8[k].float() * s
    p8, pf = tmp_path / "ideogram4_int8_convrot.safetensors", tmp_path / "ideogram4_fp8_scaled.safetensors"
    save_file({k: v.contiguous() for k, v in int8.items()}, str(p8))
    save_file({k: v.contiguous() for k, v in fp8.items()}, str(pf))
    a = ISF.load_transformer_file(p8, "cpu", quant="auto", dtype=torch.float32)  # the file's int8 is kept
    assert sum(isinstance(m, Q.QuantLinear) for m in a.modules()) == 2 * 6  # qkv o w1 w2 w3 adaln per block
    Q.configure_int8(a, False)  # weight-only: the int8 file's weights exactly, no activation rounding
    b = ISF.load_transformer_file(pf, "cpu", quant="none", dtype=torch.float32)
    x, enc = torch.randn(1, 16, 2, 3), torch.randn(1, 4, 40)
    with torch.no_grad():
        for got, state in ((a, want8), (b, wantf)):
            want = Ideogram4Transformer(**TINY).float().eval()
            want.load_state_dict(state)
            assert torch.allclose(got(x, 400.0, enc, None).float(), want(x, 400.0, enc, None), atol=1e-4)
    for p in (p8, pf):
        assert ISF.is_ideogram4_file(p) and archs.arch_from_transformer_file(str(p)) == "ideogram4"
        assert archs.guess_arch_name(TrainConfig(model_source="files", transformer_file=str(p))) == "ideogram4"


# ---------------------------------------------------------------- LoRA


def test_lora_names_export_and_what_comfyui_and_diffusers_read(tmp_path):
    ours, ref = pair(8)
    default = IDEO.lora_module_names(ours, IDEO.default_targets, None)
    assert len(default) == 2 * 5  # the fused qkv, o, w1, w3, w2 per block
    assert IDEO.lora_module_names(ours, ["q"], [1]) == ["layers.1.attention.qkv"]  # q, k, v: one fused layer
    everything = IDEO.lora_module_names(ours, list(IDEO.target_groups), None)
    linears = [n for n, m in ours.named_modules() if isinstance(m, torch.nn.Linear)]
    assert sorted(everything) == sorted(linears)  # all groups together = every Linear
    assert IDEO.lora_module_names(ours, ["o"], None, include_text_embedder=True) == [
        "llm_cond_proj", "layers.0.attention.o", "layers.1.attention.o"]
    assert IDEO.param_group_of("layers.0.attention.qkv") == "attn" and IDEO.param_group_of("llm_cond_proj") == "text"
    assert IDEO.param_group_of("layers.0.feed_forward.w2") == "mlp"
    assert L.backprop_blocks(ours, {n: ours.get_submodule(n) for n in default}, [1]) == [1]
    names = IDEO.lora_module_names(ours, ["q", "o", "ffn_up_gate", "ffn_down", "adaln"], None)
    inj = L.inject_lora(ours, 4, 8.0, ["q"], names=names)
    with torch.no_grad():
        for mod in inj.values():
            mod.lora_up.weight.normal_(0, 0.1)
    path = tmp_path / "ideogram4_lora.safetensors"
    L.save_lora(inj, path, dtype=torch.float32, fmt=IDEO.lora_format)
    state = load_file(str(path))
    assert "diffusion_model.layers.0.attention.qkv.lora_down.weight" in state
    assert "diffusion_model.layers.1.adaln_modulation.lora_up.weight" in state
    # ComfyUI (no Ideogram 4 mapping in comfy/lora.py) reads "diffusion_model.<module>": every stem must be a weight of
    # its model, which has the release's names
    stems = {k.rsplit(".", 2)[0] for k in state if k.endswith(".lora_down.weight")}
    assert {_pattern(s[len("diffusion_model."):] + ".weight") for s in stems} <= RELEASE_KEYS
    # it loads back here
    again, _ = pair(8)
    inj2 = L.inject_lora(again, 4, 8.0, ["q"], names=names)
    assert L.load_lora_weights(inj2, path) == len(names)
    x, enc = torch.randn(1, 16, 2, 3), torch.randn(1, 5, 40)
    with torch.no_grad():
        mine = ours(x, 400.0, enc, None)
        assert torch.allclose(again(x, 400.0, enc, None), mine, atol=1e-6)
    # diffusers' Ideogram 4 loader converts this layout (qkv split into to_q / to_k / to_v, alpha folded): merged into
    # its model that way it predicts what our LoRA model predicts
    from diffusers.loaders.lora_conversion_utils import _convert_non_diffusers_ideogram4_lora_to_diffusers

    converted = _convert_non_diffusers_ideogram4_lora_to_diffusers(dict(state))
    assert "transformer.layers.0.attention.to_k.lora_A.weight" in converted
    with torch.no_grad():
        for k in [k for k in converted if k.endswith(".lora_A.weight")]:
            stem = k[len("transformer."): -len(".lora_A.weight")]
            ref.get_submodule(stem).weight += converted[stem.join(("transformer.", ".lora_B.weight"))] @ converted[k]
        want = -ref_call(ref, x, torch.tensor([0.4]), [enc[0]])
    assert torch.allclose(mine, want, atol=1e-4, rtol=1e-3)


def test_sampling_loras_in_the_ai_toolkit_and_diffusers_layouts(tmp_path):
    from safetensors.torch import save_file as save

    # ai-toolkit (ostris' TurboTime and most community LoRAs): diffusion_model.<release name>.lora_A / B, no alpha
    ours, ref = pair(9)
    a1, b1 = torch.randn(3, 128) * 0.1, torch.randn(3 * 128, 3) * 0.1
    a2, b2 = torch.randn(3, 32) * 0.1, torch.randn(4 * 128, 3) * 0.1
    p1 = tmp_path / "aitk.safetensors"
    save({"diffusion_model.layers.0.attention.qkv.lora_A.weight": a1,
          "diffusion_model.layers.0.attention.qkv.lora_B.weight": b1,
          "diffusion_model.layers.1.adaln_modulation.lora_A.weight": a2,
          "diffusion_model.layers.1.adaln_modulation.lora_B.weight": b2}, str(p1))
    assert L.attach_extra_lora(ours, str(p1), 1.0, rename=IDEO.lora_module_name, convert=IDEO.extra_lora_converter) == 2
    L.set_extras_enabled(ours, True)
    x, enc = torch.randn(1, 16, 2, 2), torch.randn(1, 3, 40)
    with torch.no_grad():
        got = ours(x, 700.0, enc, None)
        att = ref.layers[0].attention
        delta = b1 @ a1
        att.to_q.weight += delta[:128]
        att.to_k.weight += delta[128:256]
        att.to_v.weight += delta[256:]
        ref.layers[1].adaln_modulation.weight += b2 @ a2
        assert torch.allclose(got, -ref_call(ref, x, torch.tensor([0.7]), [enc[0]]), atol=1e-4)
    # diffusers' training layout: transformer.<diffusers name>, alpha in the PEFT metadata; q / k / v land in their
    # thirds of the fused qkv, to_out.0 on o, each with its own scale
    ours, ref = pair(10)
    state = {}
    parts = {}
    for n in ("to_q", "to_k", "to_v", "to_out.0"):
        a, b = torch.randn(2, 128) * 0.1, torch.randn(128, 2) * 0.1
        parts[n] = (a, b)
        state[f"transformer.layers.1.attention.{n}.lora_A.weight"] = a
        state[f"transformer.layers.1.attention.{n}.lora_B.weight"] = b
    meta = {"lora_adapter_metadata": json.dumps({"transformer.lora_alpha": 4.0, "transformer.r": 2})}
    p2 = tmp_path / "diffusers_layout.safetensors"
    save(state, str(p2), metadata=meta)
    assert L.attach_extra_lora(ours, str(p2), 1.0, rename=IDEO.lora_module_name, convert=IDEO.extra_lora_converter) == 2
    L.set_extras_enabled(ours, True)
    with torch.no_grad():
        got = ours(x, 300.0, enc, None)
        for n, (a, b) in parts.items():
            ref.layers[1].attention.get_submodule(n).weight += (4.0 / 2) * b @ a
        assert torch.allclose(got, -ref_call(ref, x, torch.tensor([0.3]), [enc[0]]), atol=1e-4)
    assert IDEO.lora_module_name("conditional_transformer.layers.3.feed_forward.w1") == "layers.3.feed_forward.w1"
    assert IDEO.lora_module_name("layers.3.attention.to_out.0") == "layers.3.attention.o"


# ---------------------------------------------------------------- sampler


def test_sampler_schedule_matches_ideograms_and_diffusers():
    from diffusers.pipelines.ideogram4.pipeline_ideogram4 import _logit_normal_sigmas, _resolution_aware_mu

    for steps, (h, w), (mu, std) in ((20, (1024, 1024), (0.0, 1.75)), (48, (2048, 2048), (0.0, 1.5)),
                                     (12, (768, 1344), (0.5, 1.75))):
        ours = S.ideogram4_sigmas(steps, h, w, mu, std)
        sched = diffusers.FlowMatchEulerDiscreteScheduler(shift=1.0, use_dynamic_shifting=False)
        sched.set_timesteps(sigmas=_logit_normal_sigmas(steps, _resolution_aware_mu(h, w, mu), std=std).tolist())
        assert len(ours) == steps + 1 and torch.allclose(ours, sched.sigmas.float(), atol=1e-6)
        # Ideogram's own LogitNormalSchedule (scheduler.py) gives the same steps in model time t = 1 - sigma
        u = torch.linspace(0, 1, steps + 1, dtype=torch.float64)
        t = (1 - torch.special.expit(S.resolution_mean(mu, h, w) + std * torch.special.ndtri(u))).clamp(
            1 / (1 + torch.e ** 9), 1 / (1 + torch.e ** -7.5))
        assert torch.allclose(ours[:-1], (1 - t).flip(0)[:-1].float(), atol=1e-6)
        assert ours[0] < 1.0 and ours[-1] == 0.0 and bool((ours[1:] < ours[:-1]).all())
    assert S.schedule_params(48) == (0.0, 1.5) and S.schedule_params(20) == (0.0, 1.75)
    assert S.schedule_params(12) == (0.5, 1.75) and S.schedule_params(30) == (0.0, 1.75)
    assert S.schedule_params(8) == (0.5, 1.75) and S.schedule_params(8, distilled=True) == S.TURBO_SCHEDULE
    # Ideogram's presets: 45 steps at 7 then 3 at 3; 18 + 2; 11 + 1. No guidance stays off.
    assert S.guidance_schedule(48, 7.0) == [7.0] * 45 + [3.0] * 3
    assert S.guidance_schedule(20, 7.0) == [7.0] * 18 + [3.0] * 2
    assert S.guidance_schedule(12, 7.0) == [7.0] * 11 + [3.0]
    assert S.guidance_schedule(8, 1.0) == [1.0] * 8 and S.guidance_schedule(10, 2.5) == [2.5] * 10
    r = IDEO.reference_sigmas(20000, torch.Generator().manual_seed(0), 4096)
    assert 0.0 < float(r.min()) and float(r.max()) < 1.0
    assert abs(float(torch.logit(r.double()).mean()) - S.resolution_mean(0.5, 1024, 1024)) < 0.05


def test_sample_runs_cfg_against_the_text_free_pass():
    ours, _ = pair(11)
    calls = []

    class Spy(torch.nn.Module):
        latent_channels = 16

        def forward(self, x, t, e, m):
            calls.append(e.shape[1])
            return ours(x.float(), t, e, m)

    pos_e, pos_m = IL.pad_embeds([torch.randn(5, 40)])
    neg_e, neg_m = IL.pad_embeds([torch.zeros(0, 40)])
    assert neg_e.shape == (1, 0, 40) and neg_m.shape == (1, 0)
    lat = S.sample(Spy(), pos_e, pos_m, neg_e, neg_m, width=64, height=48, steps=4, cfg=7.0, seed=1, device="cpu")
    assert lat.shape == (1, 16, 3, 4) and torch.isfinite(lat).all()
    assert calls == [5, 0] * 4  # every step: the prompt, then the image tokens alone
    calls.clear()
    S.sample(Spy(), pos_e, pos_m, neg_e, neg_m, width=64, height=48, steps=3, cfg=1.0, seed=1, device="cpu")
    assert calls == [5] * 3


def _alignment_grads(checkpointing: bool):
    """Attention localization and identity alignment through the port's taps (rulztrainer.alignment)."""
    from types import SimpleNamespace

    from rulztrainer import alignment as A

    ours, _ = pair(12)
    ours.gradient_checkpointing = checkpointing
    ours.train()
    taps = A.Taps([0], [1])
    ours._rz_taps = taps
    A.attention_module(ours.layers[1])._rz_tap = A.AttnTap(taps, 1)
    aligner = A.IdAligner(A.model_dim(ours))
    prior = torch.ones(2, 1, 4, 6, dtype=torch.bool)
    prior[:, :, 1:4, 1:4] = False
    regions = SimpleNamespace(prior=prior, text_paths=["0", "1"], meta=[
        SimpleNamespace(prior=prior[b], face_cells=(1, 1, 3, 3), has_person=True) for b in range(2)])
    torch.manual_seed(1)
    x, enc = torch.randn(2, 16, 4, 6), torch.randn(2, 7, 40)
    mask = torch.ones(2, 7, dtype=torch.bool)
    mask[1, 5:] = False
    targets = [torch.nn.functional.normalize(torch.randn(A.FACE_EMB_DIM), dim=0) for _ in range(2)]
    taps.begin([[0, 1], [2]])
    out = ours(x, torch.tensor([600.0, 200.0]), enc, mask)
    taps.stop()
    loc, n_loc = A.localization_loss(taps, regions)
    ali, n_ali = aligner.loss(taps, 0, regions, targets)
    assert n_loc == 2 and n_ali == 2 and 0 < loc.item() < 1 and 0 < ali.item() < 2
    (out.float().pow(2).mean() + 0.5 * loc + 0.5 * ali).backward()
    taps.end()
    return [p.grad.clone() for p in list(ours.parameters()) + list(aligner.parameters()) if p.grad is not None]


def test_alignment_taps_backpropagate_the_same_with_checkpointing():
    plain, ckpt = _alignment_grads(False), _alignment_grads(True)
    assert len(plain) == len(ckpt) > 0
    for a, b in zip(plain, ckpt):
        assert torch.allclose(a, b, atol=1e-6, rtol=1e-4)


# ---------------------------------------------------------------- captions


def test_captions_wrap_plain_text_and_place_the_trigger():
    plain = P.add_trigger("a woman on a beach", "zxc")
    assert json.loads(plain) == {"high_level_description": "a woman on a beach", "compositional_deconstruction": {
        "background": "", "elements": [{"type": "obj", "desc": "zxc"}]}}
    assert P.verify(json.loads(plain)) == [] and P.verify(json.loads(P.EMPTY_CAPTION)) == []
    assert P.verify(P.wrap_plain("x")) == [] and P.add_trigger("", "") == P.EMPTY_CAPTION
    full = {"high_level_description": "A man reading.", "style_description": {
        "aesthetics": "calm", "lighting": "soft", "photo": "50mm", "medium": "photograph",
        "color_palette": ["#1B1B2F", "#F5F5F5"]}, "compositional_deconstruction": {"background": "A library.",
        "elements": [{"type": "text", "bbox": [10, 10, 50, 90], "text": "EXIT", "desc": "a sign"},
                     {"type": "obj", "bbox": [100, 200, 900, 800], "desc": "A man in a grey coat."}]}}
    assert P.verify(full) == []
    raw = json.dumps(full, indent=2)  # a hand-written file, pretty-printed
    cleaned = P.clean_prompt(raw)
    assert cleaned == json.dumps(full, separators=(",", ":"), ensure_ascii=False) and "\n" not in cleaned
    withtrig = json.loads(P.add_trigger(cleaned, "zxc"))
    assert withtrig["compositional_deconstruction"]["elements"][1]["desc"] == "zxc, A man in a grey coat."
    assert list(withtrig["compositional_deconstruction"]["elements"][1]) == ["type", "bbox", "desc"]
    assert P.verify(withtrig) == [] and P.add_trigger(P.add_trigger(cleaned, "zxc"), "zxc") == P.add_trigger(cleaned, "zxc")
    texts_only = {"compositional_deconstruction": {"background": "", "elements": [{"type": "text", "text": "A",
                                                                                    "desc": "a"}]}}
    out = json.loads(P.add_trigger(json.dumps(texts_only), "zxc"))
    assert out["compositional_deconstruction"]["elements"][0] == {"type": "obj", "desc": "zxc"}
    # what the text encoder reads: compact JSON always, the empty prompt empty (the unconditional branch)
    assert P.as_caption("") == "" and P.as_caption("   ") == ""
    assert P.as_caption(raw) == cleaned and P.as_caption(cleaned) == cleaned
    assert json.loads(P.as_caption("photo of zxc"))["high_level_description"] == "photo of zxc"
    crop = json.loads(P.as_caption("close-up portrait photo of " + cleaned))  # the crop augmentation's template
    assert crop["high_level_description"] == "close-up portrait photo of: A man reading." and P.verify(crop) == []
    assert list(crop) == ["high_level_description", "style_description", "compositional_deconstruction"]
    # the verifier's rules flag what Ideogram's would
    assert any("must exist" in w for w in P.verify({"high_level_description": "x"}))
    assert any("key order" in w for w in P.verify({"compositional_deconstruction": {"elements": [], "background": ""}}))
    assert any("exactly one" in w for w in P.verify({**full, "style_description": {**full["style_description"],
                                                                                    "art_style": "x"}}))
    assert any("hex" in w for w in P.verify({"style_description": {"aesthetics": "", "lighting": "", "photo": "",
                                                                   "medium": "", "color_palette": ["#fff"]},
                                             "compositional_deconstruction": {"background": "", "elements": []}}))
    assert any("bbox" in w for w in P.verify({"compositional_deconstruction": {"background": "", "elements": [
        {"type": "obj", "bbox": [5, 0, 1, 10], "desc": "x"}]}}))
    assert P.check_ascii_escapes('{"a":"\\u00e9"}') and not P.check_ascii_escapes('{"a":"é"}')


def test_dataset_captions_caches_buckets_and_downloads(tmp_path):
    from PIL import Image

    from rulztrainer.comfy_files import planned_downloads

    for path in ("ideogram-ai/ideogram-4-fp8", "unsloth/ideogram-4-fp8", "C:/m/ideogram4_int8_convrot.safetensors"):
        assert TrainConfig(model_path=path).arch_name() == "ideogram4"
    data = tmp_path / "data"
    data.mkdir()
    Image.new("RGB", (900, 600)).save(data / "a.png")
    (data / "a.txt").write_text("a red bicycle", encoding="utf-8")
    Image.new("RGB", (600, 900)).save(data / "b.png")
    (data / "b.txt").write_text("ignored: the .json wins", encoding="utf-8")
    (data / "b.json").write_text(json.dumps({"high_level_description": "A bicycle.", "compositional_deconstruction": {
        "background": "A street.", "elements": [{"type": "obj", "desc": "A bicycle."}]}}, indent=1), encoding="utf-8")
    cfg = TrainConfig(model_path="ideogram-ai/ideogram-4-fp8", dataset_dir=str(data), resolutions=[0.5],
                      trigger_word="zxc", cache_dir=str(tmp_path / "c"))
    items = {it.image.name: it for it in D.scan_dataset(cfg)}
    a, b = items["a.png"], items["b.png"]
    assert a.width % 16 == 0 and a.height % 16 == 0 and abs(a.width * a.height - 0.5 * 2**20) < 0.05 * 2**20
    assert a.latent_path.name.endswith("_ideo4.safetensors")  # not MOSS / FIBO's Flux2 latents
    assert json.loads(a.caption)["compositional_deconstruction"]["elements"] == [{"type": "obj", "desc": "zxc"}]
    assert json.loads(b.caption)["compositional_deconstruction"]["elements"][0]["desc"] == "zxc, A bicycle."
    assert D.text_cache_path(cfg, "p").name.endswith("_ideogram4.safetensors")
    assert D.empty_text_path(cfg) == D.text_cache_path(cfg, P.EMPTY_CAPTION)
    assert json.loads(D.dropout_prompt(cfg))["compositional_deconstruction"]["elements"] == [
        {"type": "obj", "desc": "zxc"}]
    assert IDEO.default_negative == "" and IDEO.size_multiple == 16
    plan = {f: (r, n) for f, r, n in planned_downloads(cfg)}
    assert plan["transformer_file"] == ("Comfy-Org/Ideogram-4", "diffusion_models/ideogram4_int8_convrot.safetensors")
    assert plan["text_encoder_file"] == ("Comfy-Org/Qwen-Image-2.1", "text_encoders/qwen3vl_8b_int8_convrot.safetensors")
    assert plan["vae_file"] == ("Comfy-Org/Ideogram-4", "vae/flux2-vae.safetensors")
    assert plan["turbo_lora"] == ("ostris/ideogram_4_turbotime_lora", "ideogram_4_turbotime_v1.safetensors")
    assert "training_adapter" not in plan
    fp8 = {f: (r, n) for f, r, n in planned_downloads(TrainConfig(model_path="ideogram-ai/ideogram-4-fp8",
                                                                  model_file_quant="nvfp4_mixed",
                                                                  text_encoder_file_quant="fp8_scaled"))}
    assert fp8["transformer_file"][1] == "diffusion_models/ideogram4_nvfp4_mixed.safetensors"
    assert fp8["text_encoder_file"] == ("Comfy-Org/Ideogram-4", "text_encoders/qwen3vl_8b_fp8_scaled.safetensors")


def test_auto_quant_counts_the_gpus_the_model_runs_on(monkeypatch):
    sizes = {0: 16, 1: 12, 2: 32}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(IL, "_gpu_gb", lambda d: sizes[torch.device(d).index or 0] - 0.1)
    # 16 GB cards report about 15.9 GB, 32 GB ones 31.x
    assert IL.resolve_quant("auto", "cuda:0") == "int8_convrot" and IL.resolve_quant("auto", "cuda:1") == "nvfp4"
    assert IL.resolve_quant("auto", "cuda:2") == "none" and IL.resolve_quant("auto", ["cuda:0", "cuda:0"]) == "none"
    assert IL.resolve_quant("auto", ["cuda:1", "cuda:1"]) == "int8_convrot"
    assert IL.resolve_quant("nvfp4", "cuda:0") == "nvfp4"


# ---------------------------------------------------------------- latents and text features


def test_latent_layout_and_normalization():
    z = torch.randn(2, 32, 6, 8)
    p = IL.patchify(z)
    assert p.shape == (2, 128, 3, 4) and torch.equal(IL.unpatchify(p), z)
    for ph, pw, c, i, j in ((0, 0, 0, 0, 0), (1, 0, 5, 2, 1), (0, 1, 31, 1, 3), (1, 1, 17, 2, 2)):
        assert p[1, (ph * 2 + pw) * 32 + c, i, j] == z[1, c, 2 * i + ph, 2 * j + pw]  # Ideogram's token order
    # the pipeline's own unpacking of a token sequence gives back the same image latents
    tokens = p.flatten(2).transpose(1, 2)
    ref = tokens.view(2, 3, 4, 2, 2, 32).permute(0, 5, 1, 3, 2, 4).reshape(2, 32, 6, 8)
    assert torch.equal(ref, z)

    class FakeVAE:
        dtype, device = torch.float32, torch.device("cpu")

        class _Dist:
            def __init__(self, m):
                self._m = m

            def mode(self):
                return self._m

        def encode(self, x):
            return type("Out", (), {"latent_dist": self._Dist(x[:, :1].repeat(1, 32, 1, 1)[:, :, ::8, ::8] * 3)})()

        def decode(self, z, return_dict=False):
            return (z,)

    img = torch.rand(1, 3, 64, 48) * 2 - 1
    lat = IL.encode_images(FakeVAE(), img)
    shift = torch.tensor(IL.LATENT_SHIFT).view(1, -1, 1, 1)
    scale = torch.tensor(IL.LATENT_SCALE).view(1, -1, 1, 1)
    raw = img[:, :1].repeat(1, 32, 1, 1)[:, :, ::8, ::8] * 3
    assert lat.shape == (1, 128, 4, 3) and torch.allclose(lat, (IL.patchify(raw) - shift) / scale)
    assert torch.allclose(IL._decode(FakeVAE(), lat), raw, atol=1e-5)
    assert len(IL.LATENT_SHIFT) == len(IL.LATENT_SCALE) == 128


def _tokenizer_dir():
    for p in (os.environ.get("RULZ_QWEN3VL_TOKENIZER", ""), "models/tokenizer__Qwen-Image-2.1"):
        if p and (Path(p) / "tokenizer_config.json").is_file():
            return p
    return None


@pytest.mark.skipif(_tokenizer_dir() is None, reason="no Qwen3-VL tokenizer folder (set RULZ_QWEN3VL_TOKENIZER)")
def test_text_features_match_diffusers_pipeline_encode():
    """encode_prompt against diffusers' Ideogram4Pipeline.encode_prompt on a tiny random Qwen3-VL text model: the chat
    turn, the 13 tapped layers (outputs of layers 0, 3, ..., 33, 35) and their interleaved stacking."""
    from types import SimpleNamespace

    from transformers import AutoTokenizer, Qwen3VLTextConfig, Qwen3VLTextModel

    tok = AutoTokenizer.from_pretrained(_tokenizer_dir())
    torch.manual_seed(0)
    cfg = Qwen3VLTextConfig(vocab_size=len(tok) + 64, hidden_size=16, intermediate_size=32, num_hidden_layers=36,
                            num_attention_heads=2, num_key_value_heads=1, head_dim=8,
                            rope_parameters={"rope_type": "default", "rope_theta": 5000000.0, "mrope_section": [2, 1, 1],
                                             "mrope_interleaved": True})
    model = Qwen3VLTextModel(cfg).eval()
    prompt = P.add_trigger("a red bicycle in the rain", "zxc")
    fake = SimpleNamespace(tokenizer=tok, text_encoder=SimpleNamespace(language_model=model, device=torch.device("cpu")),
                           _prepare_ids=diffusers.Ideogram4Pipeline._prepare_ids,
                           _get_text_encoder_hidden_states=diffusers.Ideogram4Pipeline._get_text_encoder_hidden_states)
    with torch.no_grad():
        want, _, _, ind = diffusers.Ideogram4Pipeline.encode_prompt(fake, prompt, 1, 1, 256, torch.device("cpu"))
        got = IL.encode_prompt(model, tok, prompt, 1024)
    n = int((ind[0] == 3).sum())
    assert got.shape == (n, 16 * 13)
    assert torch.allclose(got.float(), want[0, 256 - n:256].to(torch.bfloat16).float(), atol=1e-2, rtol=1e-2)
    text, offsets, drop, length = IL.prompt_tokens(tok, prompt)
    assert drop == 0 and length == n and "zxc" in text
    assert IL.encode_prompt(model, tok, "", 1024).shape == (0, IL.TEXT_DIM) and IL.prompt_tokens(tok, "")[3] == 0
