"""Load Ideogram 4 weights from single files and from the checkpoint's own formats.

Diffusion model files (state-dict names of ideogram4/model.py, optionally under diffusion_model. /
model.diffusion_model. / conditional_transformer.):
  - Ideogram's release layout (ideogram-ai/ideogram-4-fp8 and its unsloth mirror, transformer/
    diffusion_pytorch_model.safetensors): fused attention.qkv / attention.o; every Linear weight float8_e4m3fn with a
    float32 per-row scale `<name>.weight_scale` [out] (weight-only fp8, w = fp8 * scale[:, None]); norms, biases and
    the indicator embedding bf16. There is no bf16 release: these fp8 weights are the source.
  - Comfy-Org/Ideogram-4 diffusion_models/ideogram4_{int8_convrot, fp8_scaled, nvfp4_mixed}.safetensors: the same
    names. int8_convrot holds the 34 blocks as ComfyUI int8_tensorwise with ConvRot (groups of 256) and everything
    else bf16; fp8_scaled is float8_e4m3fn with a per-tensor scale (1.0 in the file); nvfp4_mixed nvfp4 blocks and fp8
    elsewhere. Measured against the release's fp8 on 16 rows each: int8_convrot is within 0.9% (the trainer's own
    int8_convrot of the release: 0.9%), fp8_scaled 3.7-3.9% on the block layers and 19% on input_proj (its scale of 1
    leaves the small weights in fp8's subnormal range), so int8_convrot is the file to train from.
  - diffusers' layout (attention.to_q / to_k / to_v / to_out.0; ideogram-ai/ideogram-4-nf4 and its unsloth mirror
    store it as bitsandbytes nf4): q / k / v are fused into qkv on load (fuse_diffusers_attention), nf4 is
    dequantized (dequantize_nf4).
A plain diffusers from_pretrained of the release layout would leave the attention randomly initialized (its fused
names are not diffusers'): here every key is mapped and anything left over is an error.

Formats the trainer does not quantize to (fp8, nf4) are dequantized to bf16 on load and quantized again for the GPU
(transformer_quant: auto picks by memory, see loading.resolve_quant); ComfyUI int8 / nvfp4 files are kept as they are
with "auto". The unconditional model (diffusion_models/ideogram4_unconditional_*, unconditional_transformer/) has the
same layout and is not trained.

The text encoder is Qwen3-VL-8B-Instruct, the same weights as Qwen-Image 2.1's (Comfy-Org/Qwen-Image-2.1
text_encoders/qwen3vl_8b_*: identical norms and embeddings, projections within fp8's error), read by
qwen21.single_file; the release's own fp8 text encoder (per-row scales like the DiT) is read here. The VAE is the Flux2
VAE (moss.single_file.load_vae_file)."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

from .. import quant as Q
from ..moss import single_file as sf
from .model import Ideogram4Transformer

log = logging.getLogger("rulztrainer")

DIT_PREFIXES = ("model.diffusion_model.", "diffusion_model.", "conditional_transformer.", "transformer.")
MARKERS = ("embed_image_indicator.weight", "llm_cond_proj.weight")  # ComfyUI detects the family by the first
FLOAT8 = sf.FLOAT8
NF4_SUFFIXES = (".absmax", ".quant_map", ".nested_absmax", ".nested_quant_map")


def is_ideogram4_state(keys) -> bool:
    keys = set(keys)
    return all(m in keys for m in MARKERS)


def is_ideogram4_file(path: str | Path) -> bool:
    """A cheap look at a safetensors header: does this file hold an Ideogram 4 diffusion model?"""
    from safetensors import safe_open

    try:
        with safe_open(str(path), "pt") as f:
            keys = list(f.keys())
    except Exception:  # noqa: BLE001
        return False
    return any(is_ideogram4_state(k[len(p):] for k in keys if k.startswith(p)) for p in ("", *DIT_PREFIXES))


# ---------------------------------------------------------------- dequantization


def dequantize_fp8(weight: torch.Tensor, scale: torch.Tensor | None) -> torch.Tensor:
    """fp8 weight x its scale in fp32: a per-tensor scale (a scalar), a per-row one ([out] as Ideogram's release
    stores it, or [out, 1])."""
    w = weight.to(torch.float32)
    if scale is None:
        return w
    s = scale.to(torch.float32)
    if s.numel() == 1:
        return w * s.reshape(())
    if s.ndim == 1 or (s.ndim == 2 and s.shape[1] == 1):
        if s.shape[0] != w.shape[0]:
            raise ValueError(f"fp8 scale of {tuple(s.shape)} for a weight of {tuple(w.shape)}")
        return w * s.reshape(-1, *([1] * (w.ndim - 1)))
    raise ValueError(f"Unsupported fp8 scale shape {tuple(s.shape)} for a weight of {tuple(w.shape)}")


def dequantize_nf4(packed: torch.Tensor, absmax: torch.Tensor, quant_map: torch.Tensor, quant_state: dict) -> torch.Tensor:
    """A bitsandbytes 4-bit weight (as diffusers saves it: two codes per byte, the high nibble first; one absmax per
    block of `blocksize` values; the 16-entry code book) in fp32, in its original shape."""
    if quant_state.get("nested_blocksize") is not None or "nested_absmax" in quant_state:
        raise ValueError("Nested (double-quantized) bitsandbytes 4-bit weights are not supported")
    flat = packed.reshape(-1).to(torch.uint8)
    codes = torch.stack([flat >> 4, flat & 0x0F], dim=1).reshape(-1).long()
    blocksize = int(quant_state["blocksize"])
    shape = tuple(int(s) for s in quant_state["shape"])
    n = 1
    for s in shape:
        n *= s
    vals = quant_map.to(torch.float32)[codes][:n]
    pad = (-n) % blocksize
    if pad:
        vals = torch.cat([vals, vals.new_zeros(pad)])
    vals = vals.view(-1, blocksize) * absmax.to(torch.float32).reshape(-1, 1)
    return vals.reshape(-1)[:n].reshape(shape)


def _is_sibling(k: str) -> bool:
    """A quantization tensor that belongs to a weight (read with it)."""
    return k.endswith(".weight_scale") or ".quant_state." in k or k.endswith(NF4_SUFFIXES)


def _release_entries(keys, get, dtype):
    """(key, plain `dtype` tensor) for every weight of Ideogram's release formats, reading each with its siblings
    through get(key): fp8 with a per-row / per-tensor `<name>.weight_scale`, bitsandbytes nf4 (`<name>.weight` uint8
    + `.absmax` / `.quant_map` / `.quant_state.bitsandbytes__nf4`). Other tensors are cast; ComfyUI QuantWeights pass
    through."""
    keys = list(keys)
    have = set(keys)
    used = set()
    for k in keys:
        if _is_sibling(k):
            continue
        v = get(k)
        if isinstance(v, Q.QuantWeight):
            yield k, v
            continue
        qs_key = next((s for s in (f"{k}.quant_state.bitsandbytes__nf4", f"{k}.quant_state.bitsandbytes__fp4")
                       if s in have), None)
        if qs_key is not None:
            meta = json.loads(bytes(get(qs_key).to(torch.uint8).tolist()).decode("utf-8"))
            if meta.get("quant_type", "nf4") != "nf4":
                raise ValueError(f"{k}: unsupported bitsandbytes quant type {meta.get('quant_type')!r}")
            used |= {qs_key, f"{k}.absmax", f"{k}.quant_map"}
            yield k, dequantize_nf4(v, get(f"{k}.absmax"), get(f"{k}.quant_map"), meta).to(dtype)
            continue
        scale = None
        if k.endswith(".weight") and f"{k}_scale" in have:
            scale = get(f"{k}_scale")
            used.add(f"{k}_scale")
        if v.dtype in FLOAT8 or scale is not None:
            v = dequantize_fp8(v, scale)
        yield k, (v.to(dtype) if v.is_floating_point() else v)
    leftover = [k for k in keys if _is_sibling(k) and k not in used]
    if leftover:
        raise ValueError(f"Quantization tensors without their weight: {leftover[:5]}")


def dequantize_release(state: dict, dtype=torch.bfloat16) -> dict:
    """Ideogram's release formats in a state dict (consumed: each tensor goes as its plain copy comes) as plain
    `dtype` tensors (see _release_entries)."""
    return dict(_release_entries(list(state), state.pop, dtype))


def read_state(path: str | Path, dtype=torch.bfloat16) -> dict:
    """A safetensors file in any of the formats above: ComfyUI's (with comfy_quant entries) through
    moss.single_file.read_state, Ideogram's release fp8 / diffusers nf4 / plain floats one tensor at a time (only the
    bf16 copies stay in memory, about 2 bytes per weight)."""
    from safetensors import safe_open

    with safe_open(str(path), "pt") as f:
        keys = list(f.keys())
        if not any(k.endswith(".comfy_quant") for k in keys):
            return dict(_release_entries(keys, f.get_tensor, dtype))
    return sf.read_state(path, dtype)


# ---------------------------------------------------------------- layouts


def fuse_diffusers_attention(state: dict) -> dict:
    """diffusers' attention names -> the release's: to_q / to_k / to_v stacked by rows into qkv ([q; k; v], the
    order Ideogram's attention splits it in), to_out.0 -> o."""
    out = {}
    for k in list(state):
        if k not in state:
            continue
        if k.endswith(".attention.to_q.weight"):
            stem = k[: -len(".to_q.weight")]
            parts = [state.pop(f"{stem}.to_{n}.weight") for n in "qkv"]
            if any(isinstance(p, Q.QuantWeight) for p in parts):
                parts = [p.dequantize(torch.float32) if isinstance(p, Q.QuantWeight) else p for p in parts]
            out[f"{stem}.qkv.weight"] = torch.cat(parts, dim=0)
        elif ".attention.to_k." in k or ".attention.to_v." in k:
            continue  # taken with to_q
        elif ".attention.to_out.0." in k:
            out[k.replace(".attention.to_out.0.", ".attention.o.")] = state.pop(k)
        else:
            out[k] = state.pop(k)
    left = [k for k in state if ".attention.to_" in k]
    if left:
        raise ValueError(f"diffusers attention weights without their to_q: {left[:5]}")
    return out


def normalize_dit_state(state: dict) -> dict:
    """A diffusion-model state dict (any prefix, either attention layout) in the port's names, without ComfyUI's
    attention config entries."""
    state = sf.strip_prefix(state, DIT_PREFIXES)
    state = {k: v for k, v in state.items() if not k.endswith("comfy_attention.config")}
    if any(".attention.to_q." in k for k in state):
        state = fuse_diffusers_attention(state)
    state.pop("rotary_emb.inv_freq", None)
    return state


def _count(state: dict, pattern: str) -> int:
    n = 0
    while pattern.format(n) in state:
        n += 1
    return n


def transformer_config_from_state(state: dict) -> dict:
    """Ideogram4Transformer config from a (port-named, maybe quantized) state dict; the head size and RoPE are the
    released model's (they leave no trace in the shapes)."""
    def shape(k):
        v = state[k]
        return tuple(v.orig_shape) if isinstance(v, Q.QuantWeight) else tuple(v.shape)

    head_dim = shape("layers.0.attention.norm_q.weight")[0]
    dim = shape("input_proj.weight")[0]
    return dict(in_channels=shape("input_proj.weight")[1], num_layers=_count(state, "layers.{}.attention.qkv.weight"),
                attention_head_dim=head_dim, num_attention_heads=dim // head_dim,
                intermediate_size=shape("layers.0.feed_forward.w1.weight")[0], adaln_dim=shape("adaln_proj.weight")[0],
                llm_features_dim=shape("llm_cond_proj.weight")[1], rope_theta=5_000_000, mrope_section=[24, 20, 20],
                norm_eps=1e-5)


def build_transformer(state: dict, device, quant: str, dtype=torch.bfloat16, quant_devices=None,
                      source: str = "") -> Ideogram4Transformer:
    """The port from a normalized state dict (consumed). quant "auto" keeps a ComfyUI int8 / nvfp4 file's format and
    sizes an unquantized one (bf16, or fp8 / nf4 dequantized on load) for the memory it will run in."""
    from .loading import QUANT_FILTER, resolve_quant

    if not is_ideogram4_state(state):
        raise ValueError(f"{source or 'state dict'}: not an Ideogram 4 diffusion model (no embed_image_indicator / "
                         "llm_cond_proj)")
    config = transformer_config_from_state(state)
    if quant == "auto" and not any(isinstance(v, Q.QuantWeight) for v in state.values()):
        quant = resolve_quant("auto", quant_devices or device)
    with torch.device("meta"):
        model = Ideogram4Transformer.from_config(config)
    sf.install_state(model, state, quant, dtype, quant_filter=QUANT_FILTER, **sf.quant_target(device, quant_devices))
    model = model.to(device).eval().requires_grad_(False)
    log.info("Loaded Ideogram 4 transformer from %s: %d layers, dim %d, quant=%s", source, config["num_layers"],
             config["attention_head_dim"] * config["num_attention_heads"], quant)
    return model


def load_transformer_file(path: str | Path, device, quant: str = "auto", dtype=torch.bfloat16,
                          quant_devices=None) -> Ideogram4Transformer:
    state = normalize_dit_state(read_state(path, dtype))
    if "unconditional" in Path(path).name.lower():
        log.warning("%s is Ideogram 4's unconditional model (the guidance branch without text): LoRAs train on the "
                    "conditional model (ideogram4_int8_convrot / _fp8_scaled / _nvfp4_mixed)", Path(path).name)
    return build_transformer(state, device, quant, dtype, quant_devices, Path(path).name)


# ---------------------------------------------------------------- LoRA layouts


_QKV = {"to_q": 0, "to_k": 1, "to_v": 2}


def lora_module_name(name: str) -> str:
    """Another trainer's Ideogram 4 module name -> the port's: diffusers' attention.to_q / to_k / to_v (one third
    each of the fused qkv, see lora_to_fused_rows) and to_out.0 -> attention.qkv / attention.o; the
    conditional_transformer. prefix some trainers write is dropped."""
    if name.startswith("conditional_transformer."):
        name = name[len("conditional_transformer."):]
    stem, _, last = name.rpartition(".")
    if last in _QKV and stem.endswith(".attention"):
        return f"{stem}.qkv"
    if name.endswith(".attention.to_out.0"):
        return name[: -len("to_out.0")] + "o"
    return name


def lora_to_fused_rows(state: dict) -> dict:
    """A LoRA in diffusers' Ideogram 4 layout (layers.N.attention.to_q / to_k / to_v, as diffusers' training scripts
    save it) made to fit the fused qkv: each part's up matrix is placed in its third of the fused rows (zeros in the
    other two), names, ranks and alphas unchanged, so lora_module_name maps all three onto the one qkv layer and each
    keeps its own scale (an alpha of its own, or the file's PEFT metadata). Other entries pass unchanged."""
    out = dict(state)
    for k, v in state.items():
        if not k.endswith((".lora_B.weight", ".lora_up.weight")):
            continue
        stem = k.rsplit(".", 2)[0]
        attn, _, part = stem.rpartition(".")
        if part not in _QKV or not attn.endswith(".attention"):
            continue
        rows = v.shape[0]
        up = v.new_zeros(3 * rows, v.shape[1])
        up[_QKV[part] * rows:(_QKV[part] + 1) * rows] = v
        out[k] = up
    return out


# ---------------------------------------------------------------- text encoder (Qwen3-VL-8B, text only)


def load_text_encoder_file(path: str | Path, device: str = "auto", quant: str = "auto"):
    """Qwen3-VL-8B from a single file: ComfyUI's (qwen21.single_file reads it) or the release's per-row fp8 one
    (unsloth/ideogram-4-fp8's ideogram-4-text_encoder-FP8.safetensors or text_encoder/model.safetensors)."""
    from safetensors import safe_open

    with safe_open(str(path), "pt") as f:
        release = any(k.endswith(".weight_scale") for k in f.keys()) and not any(
            k.endswith(".comfy_quant") for k in f.keys())
    if not release:
        from ..qwen21 import single_file as qsf

        return qsf.load_text_encoder_file(path, device, quant)
    return _build_text_encoder(read_state(path, torch.bfloat16), device, quant, Path(path).name)


def load_hf_text_encoder(te_dir: Path, device: str = "auto", quant: str = "auto"):
    """The release's text_encoder/ folder (Qwen3VLModel, fp8 with per-row scales, "ideogram_fp8_weight_only")."""
    from ..moss.loading import _read_sharded

    state = dequantize_release(_read_sharded(Path(te_dir), "model"), torch.bfloat16)
    return _build_text_encoder(state, device, quant, Path(te_dir).parent.name)


def _build_text_encoder(state: dict, device, quant: str, source: str):
    from ..fibo.loading import place_text_encoder
    from ..krea2.single_file import build_text_model
    from ..qwen21 import single_file as qsf

    model = build_text_model(state, "none" if quant == "auto" else quant, "cpu", config=qsf.text_config(),
                             work_device=sf.quant_target(device).get("work_device"))
    Q.configure_int8(model, qsf.TE_INT8_MATMUL)
    log.info("Loaded Qwen3-VL-8B text encoder from %s (%d layers, fp8 release weights dequantized)", source,
             model.config.num_hidden_layers)
    return place_text_encoder(model, device)
