"""Z-Image transformer single files: ComfyUI's z_image_*_bf16.safetensors (fused attention.qkv, attention.out, q_norm /
k_norm, x_embedder, final_layer) and the diffusers folder's names (the facade's own, zimage/model.py), converted both
ways, plus detection of what a file is.

ComfyUI's NextDiT -> diffusers renaming (comfy.utils z_image_to_diffusers, and diffusers' own conversion script):
  layers.N.attention.qkv.weight [3 * dim, dim]  <->  attention.to_q / to_k / to_v (chunk(3, dim 0), q then k then v)
  attention.out                                  <->  attention.to_out.0
  attention.q_norm / k_norm                      <->  attention.norm_q / norm_k
  x_embedder.*                                   <->  all_x_embedder.2-1.*
  final_layer.*                                  <->  all_final_layer.2-1.*
Everything else (feed_forward, adaLN_modulation, *_norm1/2, refiners, cap_embedder, t_embedder, the pad tokens) has the
same name in both. The ComfyUI file's keys may carry `model.diffusion_model.` or `diffusion_model.`.
"""
from __future__ import annotations

import re
from pathlib import Path

import torch

DIT_PREFIXES = ("model.diffusion_model.", "diffusion_model.")
CAP_FEAT_DIM = 2560  # Qwen3-4B's hidden size; Lumina2 (the same skeleton) reads Gemma 2 2B, 2304

_QKV = re.compile(r"^(?P<head>.*\.attention)\.qkv\.(?P<kind>weight|bias)$")
_TO_QKV = re.compile(r"^(?P<head>.*\.attention)\.to_(?P<which>[qkv])\.(?P<kind>weight|bias)$")
_RENAMES = (  # (comfy, facade) as key fragments, applied to the key as a whole segment
    (re.compile(r"^(.*\.attention)\.out\.(weight|bias)$"), r"\1.to_out.0.\2"),
    (re.compile(r"^(.*\.attention)\.q_norm\.weight$"), r"\1.norm_q.weight"),
    (re.compile(r"^(.*\.attention)\.k_norm\.weight$"), r"\1.norm_k.weight"),
    (re.compile(r"^x_embedder\.(weight|bias)$"), r"all_x_embedder.2-1.\1"),
    (re.compile(r"^final_layer\.(.+)$"), r"all_final_layer.2-1.\1"),
)
_BACK = (
    (re.compile(r"^(.*\.attention)\.to_out\.0\.(weight|bias)$"), r"\1.out.\2"),
    (re.compile(r"^(.*\.attention)\.norm_q\.weight$"), r"\1.q_norm.weight"),
    (re.compile(r"^(.*\.attention)\.norm_k\.weight$"), r"\1.k_norm.weight"),
    (re.compile(r"^all_x_embedder\.2-1\.(weight|bias)$"), r"x_embedder.\1"),
    (re.compile(r"^all_final_layer\.2-1\.(.+)$"), r"final_layer.\1"),
)


def strip_prefix(key: str) -> str:
    for p in DIT_PREFIXES:
        if key.startswith(p):
            return key[len(p):]
    return key


def comfy_to_facade(state: dict) -> dict:
    """A ComfyUI-named state dict (either prefix) -> the facade's (diffusers') names. Diffusers-named keys pass
    through unchanged, so a mixed or diffusers state is fine. The q / k / v of a fused qkv are views of its tensor."""
    out: dict = {}
    for key, value in state.items():
        key = strip_prefix(key)
        m = _QKV.match(key)
        if m:
            q, k, v = value.chunk(3, dim=0)
            for name, part in zip("qkv", (q, k, v)):
                out[f"{m['head']}.to_{name}.{m['kind']}"] = part
            continue
        for pat, rep in _RENAMES:
            if pat.match(key):
                key = pat.sub(rep, key)
                break
        out[key] = value
    return out


def facade_to_comfy(state: dict) -> dict:
    """The inverse of comfy_to_facade, without a prefix (a caller that writes a ComfyUI file adds
    `model.diffusion_model.` or `diffusion_model.`). q / k / v are fused into attention.qkv (a copy)."""
    out: dict = {}
    parts: dict[tuple[str, str], dict[str, torch.Tensor]] = {}
    for key, value in state.items():
        key = strip_prefix(key)
        m = _TO_QKV.match(key)
        if m:
            parts.setdefault((m["head"], m["kind"]), {})[m["which"]] = value
            continue
        for pat, rep in _BACK:
            if pat.match(key):
                key = pat.sub(rep, key)
                break
        out[key] = value
    for (head, kind), got in parts.items():
        missing = [w for w in "qkv" if w not in got]
        if missing:
            raise KeyError(f"{head}: to_{', to_'.join(missing)} missing, cannot fuse attention.qkv.{kind}")
        out[f"{head}.qkv.{kind}"] = torch.cat([got["q"], got["k"], got["v"]], dim=0)
    return out


def zimage_kind(keys_and_shapes: dict) -> str | None:
    """"zimage" for a Z-Image (Base / Turbo / De-Turbo) transformer, "zimage_omni" for Omni / Edit (it has a SigLIP
    branch: refused by the callers), None for anything else. Either key naming; shapes are tuples or torch.Size."""
    shapes = {strip_prefix(k): tuple(v) for k, v in keys_and_shapes.items()}
    if any("siglip_embedder" in k for k in shapes):
        return "zimage_omni"
    if "cap_pad_token" not in shapes or not any(k.startswith("context_refiner.") for k in shapes):
        return None
    cap = shapes.get("cap_embedder.1.weight")
    if cap is None or len(cap) != 2 or cap[1] != CAP_FEAT_DIM:
        return None
    return "zimage"


def read_shapes(path: str | Path) -> dict[str, tuple]:
    """Tensor names and shapes of a .safetensors file, without reading the weights."""
    from safetensors import safe_open
    with safe_open(str(path), framework="pt", device="cpu") as f:
        return {k: tuple(f.get_slice(k).get_shape()) for k in f.keys()}


def check_keys(state: dict, expected, strict: bool = True) -> tuple[list[str], list[str]]:
    """(missing, unexpected) facade keys of `state` against `expected` (an iterable of names). With strict, any
    difference raises a ValueError that lists them."""
    have, want = set(state), set(expected)
    missing, unexpected = sorted(want - have), sorted(have - want)
    if strict and (missing or unexpected):
        def show(names):
            return ", ".join(names[:12]) + (f" ... ({len(names)} in all)" if len(names) > 12 else "")
        raise ValueError("Z-Image transformer keys do not match: "
                         f"missing {len(missing)}" + (f" [{show(missing)}]" if missing else "") +
                         f", unexpected {len(unexpected)}" + (f" [{show(unexpected)}]" if unexpected else ""))
    return missing, unexpected


def load_state(path: str | Path, expected_keys=None, strict: bool = True, device="cpu",
               dtype: torch.dtype | None = None) -> dict:
    """Read a .safetensors transformer file (ComfyUI's single file or the diffusers folder's, either naming) into the
    facade's key names. `expected_keys` (e.g. model.state_dict().keys()) enables the missing / unexpected check, which
    raises with the lists when `strict`. An Omni / Edit file is refused by name."""
    from safetensors.torch import load_file
    state = load_file(str(path), device=str(device))
    kind = zimage_kind({k: v.shape for k, v in state.items()})
    if kind == "zimage_omni":
        raise ValueError(f"{Path(path).name} is a Z-Image Omni / Edit file (it has a SigLIP branch); only Z-Image "
                         "Base and Turbo text-to-image models are supported")
    state = comfy_to_facade(state)
    if dtype is not None:
        state = {k: (v.to(dtype) if v.is_floating_point() else v) for k, v in state.items()}
    if expected_keys is not None:
        check_keys(state, expected_keys, strict)
    return state
