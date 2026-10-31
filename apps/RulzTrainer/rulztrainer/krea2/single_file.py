"""Load Krea 2 components from single ComfyUI-style safetensors files (models/diffusion_models, text_encoders, vae).

The DiT file uses ComfyUI's module names (comfy/ldm/krea2); our port follows diffusers' Krea2Transformer2DModel, so
the keys are renamed on load:

    first                      img_in                 blocks.N                  transformer_blocks.N
    tmlp.0 / tmlp.2            time_embed.linear_1/2  attn.wq / wk / wv         attn.to_q / to_k / to_v
    tproj.1                    time_mod_proj          attn.gate / wo            attn.to_gate / to_out.0
    txtfusion.*                text_fusion.*          attn.qknorm.qnorm/knorm   attn.norm_q / norm_k
    txtmlp.0 / 1 / 3           txt_in.norm/linear_1/2 prenorm / postnorm        norm1 / norm2
    last.norm/linear           final_layer.norm/linear mlp                      ff
    last.modulation.lin [2,d]  final_layer.scale_shift_table   mod.lin [6d]     scale_shift_table [6, d]

Both sides store the RMSNorm scale zero-centred (the multiplier is 1 + scale), so values copy over. Weights may be
bf16, scaled fp8 (dequantized by read_state), ComfyUI int8 / int8_convrot or nvfp4 (kept with transformer_quant
"auto", or converted). A file name containing "turbo" marks the distilled model (its sampler shift is fixed).

The text encoder file is Qwen3-VL-4B as ComfyUI saves it (`model.layers.*` with `model.visual.*`) or in the HF layout
(`language_model.*`); the vision tower is dropped and the language model is built as a text-only Qwen3VLTextModel,
which gives the same hidden states for text-only prompts. The VAE file is the Qwen-Image (Wan 2.1) VAE in its
original layout (`encoder.downsamples`, `decoder.upsamples`, `conv1` / `conv2`), converted to diffusers'
AutoencoderKLQwenImage with diffusers' own converter; its latent statistics are constants of the model."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

from .. import quant as Q
from ..moss import single_file as sf
from .model import NORM_NAMES, Krea2Transformer

log = logging.getLogger("rulztrainer")
HERE = Path(__file__).resolve().parent

DIT_PREFIXES = ("model.diffusion_model.", "diffusion_model.")
# ComfyUI -> diffusers names inside a transformer block or a text-fusion block (order matters: attn.gate before mlp)
_BLOCK_RENAMES = (
    (".attn.wq.", ".attn.to_q."), (".attn.wk.", ".attn.to_k."), (".attn.wv.", ".attn.to_v."),
    (".attn.gate.", ".attn.to_gate."), (".attn.wo.", ".attn.to_out.0."),
    (".attn.qknorm.qnorm.scale", ".attn.norm_q.weight"), (".attn.qknorm.knorm.scale", ".attn.norm_k.weight"),
    (".prenorm.scale", ".norm1.weight"), (".postnorm.scale", ".norm2.weight"), (".mlp.", ".ff."),
    (".mod.lin", ".scale_shift_table"),
)
_TOP_RENAMES = (("first.", "img_in."), ("tmlp.0.", "time_embed.linear_1."), ("tmlp.2.", "time_embed.linear_2."),
                ("tproj.1.", "time_mod_proj."), ("txtmlp.1.", "txt_in.linear_1."), ("txtmlp.3.", "txt_in.linear_2."),
                ("last.linear.", "final_layer.linear."))
_EXACT_RENAMES = {"txtmlp.0.scale": "txt_in.norm.weight", "last.norm.scale": "final_layer.norm.weight",
                  "last.modulation.lin": "final_layer.scale_shift_table"}
COMFY_MARKER, DIFFUSERS_MARKER = "txtfusion.projector.weight", "text_fusion.projector.weight"


def comfy_key_to_diffusers(key: str) -> str:
    if key in _EXACT_RENAMES:
        return _EXACT_RENAMES[key]
    for old, new in _TOP_RENAMES:
        if key.startswith(old):
            return new + key[len(old):]
    if key.startswith("blocks."):
        key = "transformer_blocks." + key[len("blocks."):]
    elif key.startswith("txtfusion."):
        key = "text_fusion." + key[len("txtfusion."):]
    for old, new in _BLOCK_RENAMES:
        key = key.replace(old, new)
    return key


def to_diffusers_layout(state: dict) -> dict:
    """A DiT state dict in our (diffusers) key layout, from ComfyUI's or an already-diffusers one."""
    if DIFFUSERS_MARKER in state:
        return dict(state)
    if COMFY_MARKER not in state:
        raise ValueError("not a Krea 2 diffusion model (no txtfusion.projector / text_fusion.projector)")
    out = {}
    for k, v in state.items():
        nk = comfy_key_to_diffusers(k)
        if nk.endswith("scale_shift_table") and isinstance(v, torch.Tensor) and v.ndim == 1:
            v = v.view(6, -1)  # a block's shared modulation table, stored flat
        out[nk] = v
    return out


def _count(state: dict, pattern: str) -> int:
    n = 0
    while pattern.format(n) in state:
        n += 1
    return n


def transformer_config_from_state(state: dict) -> dict:
    """Krea2Transformer config from a diffusers-layout state dict (shapes of the stored, maybe quantized, weights)."""
    def shape(k):
        v = state[k]
        return tuple(v.orig_shape) if isinstance(v, Q.QuantWeight) else tuple(v.shape)

    head_dim = shape("transformer_blocks.0.attn.norm_q.weight")[0]  # the QK-norm runs per head
    text_head = shape("text_fusion.layerwise_blocks.0.attn.norm_q.weight")[0]
    unit = head_dim // 16  # ComfyUI derives the RoPE axes from the head size: [head - 12u, 6u, 6u]
    return dict(
        in_channels=shape("img_in.weight")[1],
        num_layers=_count(state, "transformer_blocks.{}.attn.to_q.weight"), attention_head_dim=head_dim,
        num_attention_heads=shape("transformer_blocks.0.attn.to_q.weight")[0] // head_dim,
        num_key_value_heads=shape("transformer_blocks.0.attn.to_k.weight")[0] // head_dim,
        intermediate_size=shape("transformer_blocks.0.ff.gate.weight")[0],
        timestep_embed_dim=shape("time_embed.linear_1.weight")[1],
        text_hidden_dim=shape("text_fusion.layerwise_blocks.0.norm1.weight")[0],
        num_text_layers=shape("text_fusion.projector.weight")[1],
        text_num_attention_heads=shape("text_fusion.layerwise_blocks.0.attn.to_q.weight")[0] // text_head,
        text_num_key_value_heads=shape("text_fusion.layerwise_blocks.0.attn.to_k.weight")[0] // text_head,
        text_intermediate_size=shape("text_fusion.layerwise_blocks.0.ff.gate.weight")[0],
        num_layerwise_text_blocks=_count(state, "text_fusion.layerwise_blocks.{}.attn.to_q.weight"),
        num_refiner_text_blocks=_count(state, "text_fusion.refiner_blocks.{}.attn.to_q.weight"),
        axes_dims_rope=[head_dim - 12 * unit, 6 * unit, 6 * unit], rope_theta=1000.0, norm_eps=1e-5,
    )


def is_krea2_file(path: str | Path) -> bool:
    """A cheap look at a safetensors header: does this file hold a Krea 2 diffusion model?"""
    from safetensors import safe_open

    try:
        with safe_open(str(path), "pt") as f:
            keys = set(f.keys())
    except Exception:  # noqa: BLE001
        return False
    return any(p + m in keys for p in ("", *DIT_PREFIXES) for m in (COMFY_MARKER, DIFFUSERS_MARKER))


def load_transformer_file(path: str | Path, device, quant: str = "auto", dtype=torch.bfloat16,
                          quant_devices=None) -> Krea2Transformer:
    """quant "auto" keeps a quantized file's format; an unquantized (bf16 / fp8) file is quantized for the memory it
    will run in, as the HF loader does (quant_devices: the GPUs of a model split)."""
    from .loading import QUANT_FILTER, resolve_quant

    state = sf.read_state(path, dtype)
    state = sf.strip_prefix(state, DIT_PREFIXES)
    state = {k: v for k, v in state.items() if not k.endswith("comfy_attention.config")}
    try:
        state = to_diffusers_layout(state)
    except ValueError as e:
        raise ValueError(f"{Path(path).name}: {e}") from e
    config = transformer_config_from_state(state)
    if quant == "auto":
        fmts = {v.fmt for v in state.values() if isinstance(v, Q.QuantWeight)}
        fit = resolve_quant("auto", quant_devices or device)  # what the memory the model runs in allows
        if not fmts:
            quant = fit
        elif fit == "nvfp4" and fmts <= {"int8", "int8_convrot"}:  # an int8 file on one 16 GB card: no room to train
            log.info("Krea 2 transformer: the file's int8 layers are converted to nvfp4 for this memory")
            quant = "nvfp4"
    state = {k: (v.float() if isinstance(v, torch.Tensor) and k.split(".")[-2] in NORM_NAMES else v)
             for k, v in state.items()}  # RMSNorm scales stay fp32, as diffusers keeps them
    with torch.device("meta"):
        model = Krea2Transformer.from_config(config)
    sf.install_state(model, state, quant, dtype, quant_filter=QUANT_FILTER, **sf.quant_target(device, quant_devices))
    del state
    model.is_distilled = "turbo" in Path(path).name.lower()
    model = model.to(device).eval().requires_grad_(False)
    log.info("Loaded Krea 2 transformer from %s: %d layers, dim %d, %s, quant=%s", Path(path).name,
             config["num_layers"], config["attention_head_dim"] * config["num_attention_heads"],
             "distilled (turbo, by its file name)" if model.is_distilled else "base", quant)
    return model


# ---------------------------------------------------------------- VAE (Qwen-Image / Wan 2.1)


def load_vae_file(path: str | Path, device):
    from diffusers import AutoencoderKLQwenImage
    from diffusers.loaders.single_file_utils import convert_wan_vae_to_diffusers

    state = sf.read_state(path, torch.float32)
    if any(isinstance(v, Q.QuantWeight) for v in state.values()):
        raise ValueError("Quantized VAE files are not supported; use the bf16 / fp32 Qwen-Image VAE")
    state = sf.strip_prefix(state, ("vae.", "first_stage_model."))
    if "decoder.middle.0.residual.0.gamma" in state:  # the original (ComfyUI / Wan) layout
        state = convert_wan_vae_to_diffusers(state)
    if "decoder.mid_block.resnets.0.norm1.gamma" not in state or "encoder.conv_in.weight" not in state:
        raise ValueError(f"{Path(path).name} is not a Qwen-Image (Wan 2.1) VAE")
    config = json.loads((HERE / "qwen_image_vae_config.json").read_text(encoding="utf-8"))
    config["z_dim"] = state["decoder.conv_in.weight"].shape[1]
    config["base_dim"] = state["encoder.conv_in.weight"].shape[0]
    vae = sf.assign_on_meta(lambda: AutoencoderKLQwenImage(**config), state, Path(path).name)
    return vae.to(device, torch.float32).eval().requires_grad_(False)


# ---------------------------------------------------------------- text encoder (Qwen3-VL-4B, text only)

TE_PREFIXES = ("model.language_model.", "language_model.", "text_encoders.qwen3vl_4b.transformer.model.", "model.")
TE_DROP = ("visual.", "model.visual.", "lm_head.")
TOKENIZER_REPO = "Qwen/Qwen3-VL-4B-Instruct"  # the same Qwen2 tokenizer as Krea's gated repo ships
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.json")


def build_text_model(state: dict, quant: str = "auto", device="cpu", config: dict | None = None, work_device=None):
    """Text-only Qwen3-VL from an HF- or ComfyUI-style (possibly quantized) state dict, which is consumed.
    work_device: where to quantize (moss.single_file.quant_target; the model is built on the CPU)."""
    from transformers import Qwen3VLTextConfig, Qwen3VLTextModel

    cfg = Qwen3VLTextConfig(**(config or json.loads((HERE / "qwen3vl_4b_text_config.json").read_text("utf-8"))))
    state = sf.take_state(state, TE_DROP)
    state = sf.strip_prefix(state, TE_PREFIXES)
    with torch.device("meta"):
        model = Qwen3VLTextModel(cfg)
    sf.install_state(model, state, quant, torch.bfloat16, ignore_missing=("inv_freq",), work_device=work_device)
    import inspect

    rot_cls = type(model.rotary_emb)  # its buffers were created on the meta device: build it again for real
    kwargs = {"device": "cpu"} if "device" in inspect.signature(rot_cls.__init__).parameters else {}
    model.rotary_emb = rot_cls(config=cfg, **kwargs)
    for name, p in model.named_parameters():
        if p.device.type == "meta":
            raise ValueError(f"Text encoder parameter {name} was not loaded")
    return model.to(device).eval().requires_grad_(False)


def load_text_encoder_file(path: str | Path, device: str = "auto", quant: str = "auto"):
    from ..fibo.loading import place_text_encoder

    state = sf.read_state(path, torch.bfloat16)
    model = build_text_model(state, quant, "cpu", work_device=sf.quant_target(device).get("work_device"))
    n_q = sum(1 for m in model.modules() if isinstance(m, Q.QuantLinear))
    log.info("Loaded Qwen3-VL text encoder from %s (%d layers%s)", Path(path).name, model.config.num_hidden_layers,
             f", {n_q} quantized Linear layers" if n_q else "")
    del state
    return place_text_encoder(model, device)


def hf_text_config(te_dir: Path) -> dict:
    """The language model's config from an HF text_encoder/ folder (a Qwen3-VL config.json's text_config), in the
    names this transformers reads."""
    config = json.loads((Path(te_dir) / "config.json").read_text(encoding="utf-8"))
    text_cfg = dict(config.get("text_config", config))
    rope = text_cfg.pop("rope_scaling", None)
    if rope is not None and "rope_parameters" not in text_cfg:  # transformers 4.x names in a 5.x config
        text_cfg["rope_parameters"] = {**rope, "rope_theta": text_cfg.pop("rope_theta", 5000000)}
    text_cfg.pop("dtype", None)
    return text_cfg


def load_hf_text_model(te_dir: Path, quant: str = "none", device="auto"):
    """The text-only Qwen3-VL language model from an HF text_encoder/ folder (Qwen3VLModel or
    Qwen3VLForConditionalGeneration weights): the vision tower and the LM head are never loaded. Built on the CPU;
    `device` only picks where a quantization runs."""
    from ..moss.loading import _read_sharded

    state = _read_sharded(Path(te_dir), "model")
    return build_text_model(state, quant, "cpu", config=hf_text_config(te_dir),
                            work_device=sf.quant_target(device).get("work_device"))


def resolve_tokenizer(tokenizer_path: str, models_dir: str = "models") -> Path:
    """The Qwen3-VL tokenizer folder: the given one, a downloaded Krea 2 repo's, or Qwen3-VL-4B's (downloaded once)."""
    if tokenizer_path:
        p = Path(tokenizer_path).expanduser()
        for cand in (p, p / "tokenizer"):
            if (cand / "tokenizer_config.json").is_file():
                return cand
        raise FileNotFoundError(f"No tokenizer_config.json under {p}")
    for cand in sorted(Path(models_dir).glob("krea__Krea-2-*/tokenizer/tokenizer_config.json")):
        return cand.parent
    local = Path(models_dir) / "tokenizer__Qwen3-VL-4B"
    if not (local / "tokenizer_config.json").is_file():
        from huggingface_hub import snapshot_download

        log.info("Downloading the Qwen3-VL-4B tokenizer to %s", local)
        snapshot_download(TOKENIZER_REPO, local_dir=str(local), allow_patterns=list(TOKENIZER_FILES))
    return local
