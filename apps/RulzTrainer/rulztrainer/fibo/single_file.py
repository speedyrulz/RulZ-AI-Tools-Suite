"""FIBO 2.0 from ComfyUI's single files (Comfy-Org/Fibo-2.0): the diffusion model in ComfyUI's layout (bf16 or
int8_convrot); the text encoder is the Qwen3-VL-4B file Krea 2 uses too (byte-identical in both repos) and the VAE the
Flux2 file MOSS-Image uses.

ComfyUI's layout differs from Bria's diffusers one (rulztrainer.fibo.model) in names and fusion:

    attention.qkv                         -> attention.to_q / to_k / to_v (rows, in that order)
    cross_attention.to_kv, perceiver *.to_kv -> to_k / to_v
    attention.out, *.to_out               -> to_out.0 ; attention.q_norm / k_norm -> norm_q / norm_k
    feed_forward.w13, perceiver ffn.w13   -> w1 / w3, the feed-forward width zero-padded from 9536 to 9728 (w1 is rows
                                             0..9535, w3 rows 9728..19263); w2 carries 192 zero input columns

The padding is cut on load, so the model, and every LoRA trained on it, has Bria's shapes (a LoRA from single files
loads in diffusers and the other way round). An int8 w2 is dequantized, cut and quantized again: its Hadamard rotation
groups of 256 columns do not survive a cut at 9536, so it comes back as plain per-row int8. Checked against Bria's
diffusers weights: every split part matches with cosine 1.00, the padding is exactly zero."""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import torch

from .. import quant as Q
from ..moss import single_file as sf
from .model import FiboTransformer

log = logging.getLogger("rulztrainer")
HERE = Path(__file__).parent
CONFIG = HERE / "fibo2_transformer_config.json"


def _shape(w) -> tuple[int, int]:
    return tuple(w.orig_shape) if isinstance(w, Q.QuantWeight) else tuple(w.shape)


def _rows(w, a: int, b: int):
    """Rows a..b of a weight (out_features), quantized or not."""
    if isinstance(w, Q.QuantWeight):
        if w.fmt == "nvfp4":
            return Q.quantize_weight(w.dequantize(torch.float32)[a:b].contiguous(), "nvfp4")
        return Q.QuantWeight(w.fmt, w.qdata[a:b], w.scale[a:b], None, (b - a, w.orig_shape[1]), w.group)
    return w[a:b]


def _cols(w, n: int):
    """The first n input columns of a weight; a quantized one is dequantized, cut and quantized again."""
    if isinstance(w, Q.QuantWeight):
        return Q.quantize_weight(w.dequantize(torch.float32)[:, :n].contiguous(), w.fmt, w.group)
    return w[:, :n]


def ffn_hidden(config: dict) -> int:
    """The feed-forward width of Bria's model (rulztrainer.fibo.model): (dim * 8 // 3) rounded down to 64."""
    return (int(config["dim"]) * 8 // 3) // 64 * 64


def to_diffusers_layout(state: dict, hidden: int) -> dict:
    """ComfyUI's FIBO 2.0 keys and fused weights to Bria's diffusers layout (see the module docstring)."""
    out = {}
    for k, v in state.items():
        if k.endswith("comfy_attention.config"):
            continue
        if k.endswith(".attention.qkv.weight"):
            stem, n = k[: -len("qkv.weight")], _shape(v)[0] // 3
            for i, name in enumerate(("to_q", "to_k", "to_v")):
                out[f"{stem}{name}.weight"] = _rows(v, i * n, (i + 1) * n)
        elif k.endswith(".to_kv.weight"):
            stem, n = k[: -len("to_kv.weight")], _shape(v)[0] // 2
            out[stem + "to_k.weight"], out[stem + "to_v.weight"] = _rows(v, 0, n), _rows(v, n, 2 * n)
        elif k.endswith(".w13.weight"):
            stem, half = k[: -len("w13.weight")], _shape(v)[0] // 2
            out[stem + "w1.weight"], out[stem + "w3.weight"] = _rows(v, 0, hidden), _rows(v, half, half + hidden)
        elif k.endswith(".w2.weight") and _shape(v)[1] != hidden:
            out[k] = _cols(v, hidden)
        else:
            k2 = re.sub(r"\.attention\.out\.", ".attention.to_out.0.", k)
            k2 = re.sub(r"\.attention\.(q|k)_norm\.", r".attention.norm_\1.", k2)
            k2 = re.sub(r"\.(cross_attention|self_attention)\.to_out\.", r".\1.to_out.0.", k2)
            out[k2] = v
    return out


def is_fibo_file_state(names) -> bool:
    return "perceiver.latent_queries" in names and any(n.endswith("attention.qkv.weight") for n in names)


def load_transformer_file(path: str | Path, device, quant: str = "auto", dtype=torch.bfloat16) -> FiboTransformer:
    """The diffusion model from a ComfyUI FIBO 2.0 file. quant "auto" keeps a quantized file's format and quantizes an
    unquantized one for the GPU it runs on, as the diffusers path does."""
    from .loading import QUANT_FILTER, resolve_quant

    state = sf.strip_prefix(sf.read_state(path, dtype), ("model.diffusion_model.", "diffusion_model."))
    if not is_fibo_file_state(state):
        raise ValueError(f"{Path(path).name} does not look like a FIBO 2.0 diffusion model")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    state = to_diffusers_layout(state, ffn_hidden(config))
    quantized = any(isinstance(v, Q.QuantWeight) for v in state.values())
    fmt = quant if (quant != "auto" or quantized) else resolve_quant("auto", device)
    with torch.device("meta"):
        model = FiboTransformer.from_config(config)
    sf.install_state(model, state, fmt, dtype, quant_filter=QUANT_FILTER)
    del state
    model = model.to(device).eval().requires_grad_(False)
    log.info("Loaded FIBO 2.0 transformer from %s: %d layers (+%d refiner), dim %d, quant=%s", Path(path).name,
             config["n_layers"], config["n_refiner_layers"], config["dim"], fmt)
    return model


_LORA_SUFFIXES = (".lora_down.weight", ".lora_up.weight", ".lora_A.weight", ".lora_B.weight", ".alpha", ".diff_b",
                  ".diff")


def _rename(stem: str) -> str:
    stem = re.sub(r"\.attention\.out$", ".attention.to_out.0", stem)
    stem = re.sub(r"\.attention\.(q|k)_norm$", r".attention.norm_\1", stem)
    return re.sub(r"\.(cross_attention|self_attention)\.to_out$", r".\1.to_out.0", stem)


def lora_to_diffusers_layout(state: dict, hidden: int | None = None) -> dict:
    """A LoRA in ComfyUI's FIBO 2.0 layout (Comfy-Org's Turbo LoRA) in Bria's diffusers layout: fused qkv / to_kv / w13
    entries split per part (the shared lora_down and alpha copied, lora_up rows and diff_b split; w13's zero padding
    left out), w2's lora_down cut to the real feed-forward width, the renamed modules renamed. A diffusers-layout file
    passes through unchanged."""
    hidden = hidden or ffn_hidden(json.loads(CONFIG.read_text(encoding="utf-8")))
    out = {}
    for k, v in state.items():
        suf = next((s for s in _LORA_SUFFIXES if k.endswith(s)), None)
        if suf is None:
            out[k] = v
            continue
        stem = k[: -len(suf)]
        up_like = suf in (".lora_up.weight", ".lora_B.weight", ".diff_b")
        if stem.endswith(".attention.qkv"):
            n = v.shape[0] // 3 if up_like else None
            for i, part in enumerate(("to_q", "to_k", "to_v")):
                out[f"{stem[: -len('qkv')]}{part}{suf}"] = v[i * n:(i + 1) * n] if up_like else v
        elif stem.endswith(".to_kv"):
            n = v.shape[0] // 2 if up_like else None
            for i, part in enumerate(("to_k", "to_v")):
                out[f"{stem[: -len('to_kv')]}{part}{suf}"] = v[i * n:(i + 1) * n] if up_like else v
        elif stem.endswith(".w13"):
            half = v.shape[0] // 2 if up_like else None
            for part, start in (("w1", 0), ("w3", half)):
                out[f"{stem[: -len('w13')]}{part}{suf}"] = v[start:start + hidden] if up_like else v
        elif stem.endswith(".w2") and suf in (".lora_down.weight", ".lora_A.weight") and v.shape[1] != hidden:
            out[k] = v[:, :hidden]
        else:
            out[_rename(stem) + suf] = v
    return out
