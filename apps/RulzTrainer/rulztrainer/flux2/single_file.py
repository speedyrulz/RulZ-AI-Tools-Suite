"""Load FLUX.2 Klein 9B components from single safetensors files: BFL's flux-2-klein-*.safetensors (bf16, and the fp8
repos, whose fp8 weights carry a weight_scale per Linear, with an input_scale the bf16 forward has no use for), Comfy-Org's
Qwen3-8B text encoder and the Flux2 VAE.

The DiT file carries the module names the port keeps (flux2/model.py). The diffusers layout (the repos' transformer/
folder, Flux2Transformer2DModel) is fused back into them on load: the inverse of diffusers'
convert_flux2_transformer_checkpoint_to_diffusers. Klein's Base and distilled files have the same keys, so the
distilled model is known by its file name alone.

The text encoder is plain Qwen3-8B (Qwen3ForCausalLM's `model.*`): built as a text-only Qwen3Model, the LM head left out.
"""
from __future__ import annotations

import inspect
import logging
import re
from pathlib import Path

import torch

from .. import quant as Q
from ..moss import single_file as sf
from .model import Flux2KleinTransformer

log = logging.getLogger("rulztrainer")

DIT_PREFIXES = ("model.diffusion_model.", "diffusion_model.")
MARKER = "double_stream_modulation_img.lin.weight"  # Flux 1 has no such key; FLUX.2 dev has it too (see check_klein9b)
DIFFUSERS_MARKER = "double_stream_modulation_img.linear.weight"
TEXT_DIM_9B = 12288  # Qwen3-8B's three stacked layers; FLUX.2 dev reads Mistral (15360), Klein 4B Qwen3-4B (7680)

# Qwen/Qwen3-8B's config.json (the text model's part)
TEXT_CONFIG = dict(vocab_size=151936, hidden_size=4096, intermediate_size=12288, num_hidden_layers=36,
                   num_attention_heads=32, num_key_value_heads=8, head_dim=128, hidden_act="silu",
                   max_position_embeddings=40960, max_window_layers=36, rms_norm_eps=1e-6, attention_bias=False,
                   attention_dropout=0.0, use_sliding_window=False, sliding_window=None, tie_word_embeddings=False,
                   bos_token_id=151643, eos_token_id=151645, rope_parameters={"rope_type": "default",
                                                                              "rope_theta": 1000000.0})
TOKENIZER_REPO = "Qwen/Qwen3-8B"
TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]

_EXACT = {"x_embedder": "img_in", "context_embedder": "txt_in",
          "time_guidance_embed.timestep_embedder.linear_1": "time_in.in_layer",
          "time_guidance_embed.timestep_embedder.linear_2": "time_in.out_layer",
          "double_stream_modulation_img.linear": "double_stream_modulation_img.lin",
          "double_stream_modulation_txt.linear": "double_stream_modulation_txt.lin",
          "single_stream_modulation.linear": "single_stream_modulation.lin", "proj_out": "final_layer.linear",
          "norm_out.linear": "final_layer.adaLN_modulation.1"}
_DOUBLE = {"attn.norm_q": "img_attn.norm.query_norm", "attn.norm_k": "img_attn.norm.key_norm",
           "attn.to_out.0": "img_attn.proj", "ff.linear_in": "img_mlp.0", "ff.linear_out": "img_mlp.2",
           "attn.norm_added_q": "txt_attn.norm.query_norm", "attn.norm_added_k": "txt_attn.norm.key_norm",
           "attn.to_add_out": "txt_attn.proj", "ff_context.linear_in": "txt_mlp.0", "ff_context.linear_out": "txt_mlp.2"}
_SINGLE = {"attn.to_qkv_mlp_proj": "linear1", "attn.to_out": "linear2", "attn.norm_q": "norm.query_norm",
           "attn.norm_k": "norm.key_norm"}
_FUSE = (("attn.to_q", "attn.to_k", "attn.to_v", "img_attn.qkv"),
         ("attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj", "txt_attn.qkv"))


def diffusers_to_bfl(state: dict) -> dict:
    """A diffusers-layout DiT state dict in BFL's names: q / k / v fused by rows, norm_out's [scale, shift] swapped back."""
    out, pending = {}, {}
    for key, v in state.items():
        stem, _, param = key.rpartition(".")
        if stem in _EXACT:
            if stem == "norm_out.linear":
                a, b = v.chunk(2, dim=0)
                v = torch.cat([b, a], dim=0)
            out[f"{_EXACT[stem]}.{param}"] = v
            continue
        m = re.fullmatch(r"(transformer_blocks|single_transformer_blocks)\.(\d+)\.(.+)", stem)
        if m is None:
            raise ValueError(f"not a FLUX.2 transformer key: {key}")
        kind, n, rest = m.groups()
        if kind == "transformer_blocks" and rest in {x for f in _FUSE for x in f[:3]}:
            pending[(n, rest)] = v
            continue
        table, block = (_DOUBLE, "double_blocks") if kind == "transformer_blocks" else (_SINGLE, "single_blocks")
        if rest not in table:
            raise ValueError(f"not a FLUX.2 transformer key: {key}")
        name = table[rest]
        out[f"{block}.{n}.{name}.{'scale' if name.endswith('_norm') else param}"] = v
    for n in sorted({n for n, _ in pending}, key=int):
        for q, k, v, name in _FUSE:
            out[f"double_blocks.{n}.{name}.weight"] = torch.cat([pending[(n, q)], pending[(n, k)], pending[(n, v)]])
    return out


def to_model_layout(state: dict) -> dict:
    """A DiT state dict (BFL's or diffusers' layout, prefixes stripped) in the port's names."""
    if DIFFUSERS_MARKER in state:
        state = diffusers_to_bfl(state)
    elif MARKER not in state:
        raise ValueError("not a FLUX.2 Klein diffusion model (no double_stream_modulation_img)")
    state = {k: v for k, v in state.items() if not k.endswith(".input_scale")}  # the fp8 repos' activation scales
    return norm_scales_fp32(check_klein9b(state))


def norm_scales_fp32(state: dict) -> dict:
    """The q / k RMSNorm scales in fp32, as flux3/model.py's RMSNorm computes them."""
    return {k: (v.float() if isinstance(v, torch.Tensor) and k.endswith("_norm.scale") else v) for k, v in state.items()}


def _shape(v) -> tuple:
    return tuple(v.orig_shape) if isinstance(v, Q.QuantWeight) else tuple(v.shape)


def check_klein9b(state: dict) -> dict:
    """Only Klein 9B trains here: FLUX.2 dev and Klein 4B need another text encoder than this family's."""
    text_dim = _shape(state["txt_in.weight"])[1]
    if text_dim != TEXT_DIM_9B or "guidance_in.in_layer.weight" in state:
        raise ValueError(f"this is not FLUX.2 Klein 9B (text features of {text_dim}, 9B has {TEXT_DIM_9B}): FLUX.2 dev "
                         "and Klein 4B are not supported")
    return state


def _count(state: dict, pattern: str) -> int:
    n = 0
    while pattern.format(n) in state:
        n += 1
    return n


def transformer_config_from_state(state: dict) -> dict:
    """Flux2KleinTransformer config from a port-layout (maybe quantized) state dict."""
    head_dim = _shape(state["single_blocks.0.norm.query_norm.scale"])[0]
    dim, in_channels = _shape(state["img_in.weight"])
    hidden = (_shape(state["single_blocks.0.linear1.weight"])[0] - 3 * dim) // 2
    return dict(in_channels=in_channels, context_in_dim=_shape(state["txt_in.weight"])[1], hidden_size=dim,
                num_heads=dim // head_dim, depth=_count(state, "double_blocks.{}.img_attn.qkv.weight"),
                depth_single_blocks=_count(state, "single_blocks.{}.linear1.weight"), axes_dim=[head_dim // 4] * 4,
                mlp_ratio=hidden / dim, timestep_channels=_shape(state["time_in.in_layer.weight"])[1])


def is_distilled_name(name: str) -> bool:
    """The 4-step distilled Klein by its file or folder name (flux-2-klein-9b*): not flux-2-klein-base-9b*."""
    stem = (name or "").replace("\\", "/").lower().rstrip("/").rsplit("/", 1)[-1]
    return "klein" in stem and "base" not in stem


def is_flux2_state(keys) -> bool:
    keys = set(keys)
    return MARKER in keys or DIFFUSERS_MARKER in keys


def load_transformer_file(path: str | Path, device, quant: str = "auto", dtype=torch.bfloat16,
                          quant_devices=None) -> Flux2KleinTransformer:
    """quant "auto" keeps a quantized file's format; a float file (bf16, or fp8 read up to bf16) is quantized for the
    memory it will run in (quant_devices: the GPUs of a model split)."""
    from .loading import QUANT_FILTER, resolve_quant

    state = sf.strip_prefix(sf.read_state(path, dtype), DIT_PREFIXES)
    try:
        state = to_model_layout(state)
    except ValueError as e:
        raise ValueError(f"{Path(path).name}: {e}") from e
    config = transformer_config_from_state(state)
    if quant == "auto" and not any(isinstance(v, Q.QuantWeight) for v in state.values()):
        quant = resolve_quant("auto", quant_devices or device)
    with torch.device("meta"):
        model = Flux2KleinTransformer.from_config(config)
    sf.install_state(model, state, quant, dtype, quant_filter=QUANT_FILTER, **sf.quant_target(device, quant_devices))
    del state
    model = model.to(device).eval().requires_grad_(False)
    model.is_distilled = is_distilled_name(Path(path).name)
    log.info("Loaded FLUX.2 Klein transformer from %s: %d double and %d single blocks, dim %d, %s, quant=%s",
             Path(path).name, config["depth"], config["depth_single_blocks"], config["hidden_size"],
             "distilled (4 steps)" if model.is_distilled else "base", quant)
    return model


# ---------------------------------------------------------------- text encoder (Qwen3-8B) and tokenizer


def build_text_model(state: dict, quant: str = "none", work_device=None):
    """Text-only Qwen3-8B from an HF-style state dict, which is consumed. work_device: where to quantize."""
    from transformers import Qwen3Config, Qwen3Model

    cfg = Qwen3Config(**TEXT_CONFIG)
    state = {k: v for k, v in state.items() if not k.startswith("lm_head.")}
    state = sf.strip_prefix(state, ("model.",))
    with torch.device("meta"):
        model = Qwen3Model(cfg)
    sf.install_state(model, state, quant, torch.bfloat16, ignore_missing=("inv_freq",), work_device=work_device)
    rot_cls = type(model.rotary_emb)  # its buffers were created on the meta device: build it again for real
    kwargs = {"device": "cpu"} if "device" in inspect.signature(rot_cls.__init__).parameters else {}
    model.rotary_emb = rot_cls(config=cfg, **kwargs)
    for name, p in model.named_parameters():
        if p.device.type == "meta":
            raise ValueError(f"Text encoder parameter {name} was not loaded")
    return model.eval().requires_grad_(False)


def load_text_encoder_file(path: str | Path, device: str = "auto", quant: str = "auto"):
    """Comfy-Org's qwen_3_8b*.safetensors (bf16, fp8mixed read up to bf16)."""
    from ..fibo.loading import place_text_encoder
    from .loading import resolve_text_quant

    fmt = resolve_text_quant(quant)
    state = sf.read_state(path, torch.bfloat16)
    if not any(k.endswith("layers.0.self_attn.q_proj.weight") for k in state):
        raise ValueError(f"{Path(path).name} is not a Qwen3 text encoder")
    model = build_text_model(state, fmt, sf.quant_target(device).get("work_device"))
    del state
    return place_text_encoder(model, device)


def resolve_tokenizer(tokenizer_path: str, models_dir: str = "models") -> Path:
    """The Qwen3-8B tokenizer folder: the given one, a downloaded Klein repo's tokenizer/, else Qwen/Qwen3-8B's files
    (downloaded once, a few MB; not gated)."""
    if tokenizer_path:
        p = Path(tokenizer_path).expanduser()
        for cand in (p, p / "tokenizer"):
            if (cand / "tokenizer_config.json").is_file():
                return cand
        raise FileNotFoundError(f"No tokenizer_config.json under {p}")
    for cand in sorted(Path(models_dir).glob("black-forest-labs__FLUX.2-klein*/tokenizer/tokenizer_config.json")):
        return cand.parent
    local = Path(models_dir) / "tokenizer__Qwen3-8B"
    if not (local / "tokenizer_config.json").is_file():
        from huggingface_hub import snapshot_download

        log.info("Downloading the Qwen3-8B tokenizer to %s", local)
        snapshot_download(TOKENIZER_REPO, local_dir=str(local), allow_patterns=TOKENIZER_FILES)
    return local
