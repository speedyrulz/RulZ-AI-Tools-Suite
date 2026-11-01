"""Load FLUX 3 Klein components from single safetensors files: Black Forest Labs' own layout (the HF repos' root
model.safetensors, and black-forest-labs/flux-3-image-klein-comfy: diffusion_models, text_encoders, vae).

The DiT file carries BFL's module names, which the port keeps (flux3/model.py), so most keys load as they are:

    emb_in.image / txt_in / time_in.in_layer, out_layer / ref_time_in.* (the distilled flow map only)
    early_stream_modulations.{image,txt}.lin / single_stream_modulations.{image,txt}.lin
    txt_mode_blocks.N / content_mode_blocks.image.N / single_blocks.N:
        q_proj, k_proj, v_proj, attn_out, mlp_in, mlp_out, norm.query_norm.scale, norm.key_norm.scale
    final_layer.image.linear / final_layer.image.adaLN_modulation.1 ([shift, scale])

ComfyUI's int8-convrot files fuse each block's q / k / v into one `qkv` (rows [q; k; v], one scale per row), split
here (an int8 weight splits by rows as it is); their `comfy_attention.config` entries are dropped. The bf16 and nvfp4
files keep q / k / v apart. The reference-image mode (content_mode_blocks.image_cond, emb_in.image_cond,
*_modulations.image_cond) and the training-only repa_projectors are left out (see model.py). A file holding
ref_time_in is the distilled flow-map model (16 steps, samples only).

The diffusers layout (the repos' transformer/ folder, Flux3Transformer2DModel) is renamed to BFL's on load: the
inverse of diffusers' convert_flux3_transformer_checkpoint_to_diffusers, with norm_out's [scale, shift] swapped back.
The text encoder is the same Qwen3-VL-4B file as Krea 2's and FIBO 2.0's (`model.language_model.*` + the vision
tower, which is dropped) and the VAE file is the Flux2 VAE in BFL's original layout (MOSS-Image / FIBO 2.0 load it).
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch

from .. import quant as Q
from ..moss import single_file as sf
from .model import Flux3KleinTransformer

log = logging.getLogger("rulztrainer")

DIT_PREFIXES = ("model.diffusion_model.", "diffusion_model.")
MARKER = "early_stream_modulations.image.lin.weight"  # diffusers' own detection key for the family (BFL layout)
DIFFUSERS_MARKER = "early_stream_modulations.image.linear.weight"
FLOW_MAP_MARKER = "ref_time_in.in_layer.weight"
UNUSED = ("repa_projectors.",)  # training-only heads in some exports
UNUSED_PARTS = (".image_cond.",)  # the reference-image mode (editing), not built

# diffusers -> BFL names (the inverse of diffusers' convert_flux3_transformer_checkpoint_to_diffusers)
_EXACT = {
    "timestep_embedder.linear_1": "time_in.in_layer", "timestep_embedder.linear_2": "time_in.out_layer",
    "target_timestep_embedder.linear_1": "ref_time_in.in_layer",
    "target_timestep_embedder.linear_2": "ref_time_in.out_layer",
    "proj_out": "final_layer.image.linear", "norm_out.linear": "final_layer.image.adaLN_modulation.1",
}
_PREFIX = (("x_embedders.", "emb_in."), ("context_embedder", "txt_in"), ("mode_blocks.", "content_mode_blocks."),
           ("single_transformer_blocks.", "single_blocks."))
_IN_BLOCK = (("attn.to_q", "q_proj"), ("attn.to_k", "k_proj"), ("attn.to_v", "v_proj"), ("attn.to_out.0", "attn_out"),
             ("attn.norm_q", "norm.query_norm"), ("attn.norm_k", "norm.key_norm"), ("ff.linear_in", "mlp_in"),
             ("ff.linear_out", "mlp_out"))


def diffusers_module_to_bfl(name: str) -> str:
    """A diffusers module name (Flux3Transformer2DModel) -> BFL's (the port's): for state keys and LoRA files."""
    if name in _EXACT:
        return _EXACT[name]
    for old, new in _PREFIX:
        if name.startswith(old):
            name = new + name[len(old):]
            break
    if name.endswith(".linear") and "stream_modulations." in name:
        return name[: -len(".linear")] + ".lin"
    for old, new in _IN_BLOCK:
        if name.endswith("." + old):
            return name[: -len(old)] + new
    return name


def bfl_module_to_diffusers(name: str) -> str:
    """BFL's module name -> diffusers' (the names diffusers' LoRA loader lands on)."""
    for old, new in _EXACT.items():
        if name == new:
            return old
    for old, new in _PREFIX:
        if name.startswith(new):
            name = old + name[len(new):]
            break
    if name.endswith(".lin") and "stream_modulations." in name:
        return name[: -len(".lin")] + ".linear"
    for old, new in _IN_BLOCK:
        if name.endswith("." + new):
            return name[: -len(new)] + old
    return name


def _swap_halves(v):
    if isinstance(v, Q.QuantWeight):
        v = v.dequantize(torch.float32)
    a, b = v.chunk(2, dim=0)
    return torch.cat([b, a], dim=0)


def diffusers_to_bfl(state: dict) -> dict:
    """A diffusers-layout DiT state dict in BFL's names (norm_out's [scale, shift] back to [shift, scale])."""
    out = {}
    for k, v in state.items():
        stem, _, param = k.rpartition(".")
        nk = diffusers_module_to_bfl(stem)
        if nk.endswith(("query_norm", "key_norm")) and param == "weight":
            param = "scale"
        if nk == "final_layer.image.adaLN_modulation.1":
            v = _swap_halves(v)
        out[f"{nk}.{param}"] = v
    return out


def unfuse_qkv(state: dict) -> dict:
    """ComfyUI's fused <block>.qkv ([q; k; v] by rows) -> q_proj / k_proj / v_proj."""
    out = {}
    for k, v in state.items():
        if k.endswith(".qkv.weight"):
            stem = k[: -len("qkv.weight")]
            parts = v.split_rows(3) if isinstance(v, Q.QuantWeight) else v.chunk(3, 0)
            for n, p in zip(("q_proj", "k_proj", "v_proj"), parts):
                out[f"{stem}{n}.weight"] = p
        else:
            out[k] = v
    return out


def to_model_layout(state: dict) -> dict:
    """A DiT state dict (BFL's, ComfyUI's or diffusers' layout, prefixes stripped) in the port's names, without the
    parts the port does not build."""
    if DIFFUSERS_MARKER in state:
        state = diffusers_to_bfl(state)
    elif MARKER not in state:
        raise ValueError("not a FLUX 3 Klein diffusion model (no early_stream_modulations.image)")
    state = {k: v for k, v in state.items()
             if not k.startswith(UNUSED) and not any(p in k for p in UNUSED_PARTS)
             and not k.endswith("comfy_attention.config") and not k.startswith("emb_in.image_cond")
             and not k.endswith("modulations.image_cond.lin.weight")}
    return unfuse_qkv(state)


def _count(state: dict, pattern: str) -> int:
    n = 0
    while pattern.format(n) in state:
        n += 1
    return n


def transformer_config_from_state(state: dict) -> dict:
    """Flux3KleinTransformer config from a port-layout (maybe quantized) state dict. The RoPE layout (4 axes of a
    quarter head each, theta 10000, image positions scaled by 0.5) is the checkpoint's config.json."""
    def shape(k):
        v = state[k]
        return tuple(v.orig_shape) if isinstance(v, Q.QuantWeight) else tuple(v.shape)

    head_dim = shape("single_blocks.0.norm.query_norm.scale")[0]
    dim, in_channels = shape("emb_in.image.weight")
    return dict(in_channels=in_channels, num_layers=_count(state, "txt_mode_blocks.{}.q_proj.weight"),
                num_single_layers=_count(state, "single_blocks.{}.q_proj.weight"), attention_head_dim=head_dim,
                num_attention_heads=dim // head_dim, joint_attention_dim=shape("txt_in.weight")[1],
                timestep_channels=shape("time_in.in_layer.weight")[1],
                mlp_ratio=shape("single_blocks.0.mlp_out.weight")[1] / dim, axes_dims_rope=[head_dim // 4] * 4,
                rope_theta=10000, eps=1e-6, flow_map=FLOW_MAP_MARKER in state, image_rope_scale=0.5)


def norm_scales_fp32(state: dict) -> dict:
    """The q / k RMSNorm scales in fp32, as diffusers keeps them (_keep_in_fp32_modules)."""
    return {k: (v.float() if isinstance(v, torch.Tensor) and k.endswith("_norm.scale") else v) for k, v in state.items()}


def is_flux3_state(keys) -> bool:
    keys = set(keys)
    return MARKER in keys or DIFFUSERS_MARKER in keys


def is_flux3_file(path: str | Path) -> bool:
    """A cheap look at a safetensors header: does this file hold a FLUX 3 Klein diffusion model?"""
    from safetensors import safe_open

    try:
        with safe_open(str(path), "pt") as f:
            keys = list(f.keys())
    except Exception:  # noqa: BLE001
        return False
    return any(is_flux3_state(k[len(p):] for k in keys if k.startswith(p)) for p in ("", *DIT_PREFIXES))


def load_transformer_file(path: str | Path, device, quant: str = "auto", dtype=torch.bfloat16,
                          quant_devices=None) -> Flux3KleinTransformer:
    """quant "auto" keeps a quantized file's format; an unquantized (bf16) file is quantized for the memory it will
    run in, as the HF loader does (quant_devices: the GPUs of a model split)."""
    from .loading import QUANT_FILTER, resolve_quant

    state = sf.read_state(path, dtype)
    state = sf.strip_prefix(state, DIT_PREFIXES)
    try:
        state = to_model_layout(state)
    except ValueError as e:
        raise ValueError(f"{Path(path).name}: {e}") from e
    config = transformer_config_from_state(state)
    if quant == "auto":
        fmts = {v.fmt for v in state.values() if isinstance(v, Q.QuantWeight)}
        if not fmts:
            quant = resolve_quant("auto", quant_devices or device)
    state = norm_scales_fp32(state)
    with torch.device("meta"):
        model = Flux3KleinTransformer.from_config(config)
    sf.install_state(model, state, quant, dtype, quant_filter=QUANT_FILTER, **sf.quant_target(device, quant_devices))
    del state
    model = model.to(device).eval().requires_grad_(False)
    log.info("Loaded FLUX 3 Klein transformer from %s: %d + %d per-mode and %d joint blocks, dim %d, %s, quant=%s",
             Path(path).name, config["num_layers"], config["num_layers"], config["num_single_layers"],
             config["attention_head_dim"] * config["num_attention_heads"],
             "distilled (flow map, 16 steps)" if model.flow_map else "base", quant)
    return model


# ---------------------------------------------------------------- text encoder (Qwen3-VL-4B) and tokenizer

TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.json")


def load_text_encoder_file(path: str | Path, device: str = "auto", quant: str = "auto"):
    """The Qwen3-VL-4B file (text only): the one Krea 2 and FIBO 2.0 load."""
    from ..krea2.single_file import load_text_encoder_file as krea_te

    return krea_te(path, device, quant)


def load_vae_file(path: str | Path, device):
    """The Flux2 VAE (flux2-vae.safetensors / ae.safetensors, BFL's layout): the one FIBO 2.0 loads."""
    return sf.load_vae_file(path, device)


def resolve_tokenizer(tokenizer_path: str, models_dir: str = "models") -> Path:
    """The Qwen3-VL tokenizer folder: the given one, a downloaded Klein repo's text_encoder/ (its tokenizer files are
    Qwen3-VL-4B-Instruct's, byte for byte), else the one Krea 2 / FIBO 2.0 use (Qwen/Qwen3-VL-4B-Instruct's files,
    downloaded once, a few MB; not gated)."""
    from ..krea2.single_file import resolve_tokenizer as krea_tokenizer

    if tokenizer_path:
        p = Path(tokenizer_path).expanduser()
        for cand in (p, p / "text_encoder", p / "tokenizer"):
            if (cand / "tokenizer_config.json").is_file():
                return cand
        raise FileNotFoundError(f"No tokenizer_config.json under {p}")
    for cand in sorted(Path(models_dir).glob("black-forest-labs__flux-3-image-klein*/text_encoder/"
                                             "tokenizer_config.json")):
        return cand.parent
    return krea_tokenizer("", models_dir)
