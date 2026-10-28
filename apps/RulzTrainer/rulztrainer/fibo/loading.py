"""Load FIBO 2.0 components from the Hugging Face / diffusers layout (briaai/Fibo-2.0-dev, -edit-turbo) and encode.

    transformer/   config.json (BriaFibo2Transformer2DModel) + sharded diffusion_pytorch_model*.safetensors
    text_encoder/  Qwen3-VL-4B-Instruct (Qwen3VLForConditionalGeneration), model.safetensors
    processor/     Qwen3VLProcessor (tokenizer)
    vae/           AutoencoderKLFlux2 (the same VAE and latent normalization as MOSS-Image)

Text features: the pipeline's _get_qwen_prompt_embeds, text only. Each prompt gives [6, S, 7680]: the Perceiver's
bundle and one per injection block, each three Qwen3-VL hidden layers scaled to unit RMS per token and concatenated.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from torch import nn

from .. import quant as Q
from ..config import TrainConfig
from ..moss import loading as moss_loading
from ..moss import single_file as sf
from .model import FiboTransformer
from .prompts import PROMPT_TEMPLATE, TEXT_ENCODER_OUT_LAYERS, clean_prompt

log = logging.getLogger("rulztrainer")

TRANSFORMER_CLASSES = ("BriaFibo2Transformer2DModel",)
# quantize the block projections (attention, feed-forward, cross-attention), the noise refiner and the Perceiver;
# keep the embedders, the timestep MLP, the adaLN modulations and the final layer in bf16
QUANT_FILTER = lambda name: name.startswith(("layers.", "noise_refiner.", "perceiver."))  # noqa: E731
AUTO_INT8_BELOW_GB = 22  # "auto" quantizes the bf16 (16 GB) transformer to int8 on GPUs with less memory


def is_fibo_dir(model_dir: Path) -> bool:
    cfg = model_dir / "transformer" / "config.json"
    if not cfg.is_file():
        return False
    try:
        return json.loads(cfg.read_text(encoding="utf-8")).get("_class_name") in TRANSFORMER_CLASSES
    except (OSError, ValueError):
        return False


def resolve_quant(quant: str, device) -> str:
    if quant != "auto":
        return quant
    dev = torch.device(device)
    if dev.type == "cuda" and torch.cuda.is_available():
        total = torch.cuda.get_device_properties(dev.index or 0).total_memory / 2**30
        if total < AUTO_INT8_BELOW_GB:
            log.info("FIBO 2.0 transformer: auto quantization -> int8 (W8A8) for a %.0f GB GPU", total)
            return "int8"
    return "none"


def load_transformer(model_dir: Path, device, quant: str = "auto", dtype=torch.bfloat16) -> FiboTransformer:
    config = json.loads((model_dir / "transformer" / "config.json").read_text(encoding="utf-8"))
    if config.get("_class_name") not in TRANSFORMER_CLASSES:
        raise ValueError(f"Not a FIBO 2.0 transformer: {config.get('_class_name')!r}")
    quant = resolve_quant(quant, device)
    with torch.device("meta"):
        model = FiboTransformer.from_config(config)
    state = moss_loading._read_sharded(model_dir / "transformer", "diffusion_pytorch_model")
    state = {k: v.to(dtype) for k, v in state.items()}
    sf.install_state(model, state, quant, dtype, quant_filter=QUANT_FILTER)
    del state
    model = model.to(device).eval().requires_grad_(False)
    log.info("Loaded FIBO 2.0 transformer from %s: %d layers (+%d refiner), dim %d, injection blocks %s, quant=%s",
             model_dir.name, config["n_layers"], config["n_refiner_layers"], config["dim"],
             config["injection_layer_ids"], quant)
    return model


def load_text_encoder(model_dir: Path, device: str = "auto", quant: str = "auto"):
    """Qwen3-VL (Qwen3VLForConditionalGeneration) + its processor. quant int8/int8_convrot/nvfp4 quantizes the language
    model's Linear layers (W8A8) so the 8.9 GB encoder takes about half the memory."""
    from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

    processor = Qwen3VLProcessor.from_pretrained(str(model_dir / "processor"))
    model = Qwen3VLForConditionalGeneration.from_pretrained(str(model_dir / "text_encoder"), dtype=torch.bfloat16)
    model.eval().requires_grad_(False)
    fmt = "none" if quant == "auto" else quant
    if fmt != "none":
        lm = getattr(getattr(model, "model", model), "language_model", None) or model
        n = Q.quantize_module_linears(lm, fmt, min_dim=1024)
        log.info("Text encoder: %d Linear layers quantized to %s", n, fmt)
    model = place_text_encoder(model, device)
    return model, processor


def place_text_encoder(model: nn.Module, device: str):
    if device != "auto":
        return model.to(device)
    size = sum(p.numel() * p.element_size() for p in model.parameters()) + sum(
        b.numel() * b.element_size() for b in model.buffers())
    if torch.cuda.is_available():
        best = None
        for i in range(torch.cuda.device_count()):
            free, _ = torch.cuda.mem_get_info(i)
            if free > size * 1.1 + 1.5 * 2**30 and (best is None or free > best[0]):
                best = (free, i)
        if best is not None:
            log.info("Text encoder (%.1f GB) placed on cuda:%d", size / 2**30, best[1])
            return model.to(f"cuda:{best[1]}")
    from accelerate import dispatch_model, infer_auto_device_map

    dmap = infer_auto_device_map(model, dtype=torch.bfloat16,
                                 no_split_module_classes=["Qwen3VLTextDecoderLayer", "Qwen3VLVisionBlock"])
    log.info("Text encoder device map: %s", sorted({str(v) for v in dmap.values()}))
    return dispatch_model(model, device_map=dmap)


def _input_device(model: nn.Module) -> torch.device:
    for m in model.modules():
        if isinstance(m, nn.Embedding) and m.weight.device.type != "meta":
            return m.weight.device
    return next(model.parameters()).device


@torch.no_grad()
def encode_prompt(text_encoder, processor, prompt: str, max_length: int = 4096) -> torch.Tensor:
    """Text features for one prompt: [6, S, 7680] bf16 on the CPU (the six bundles, see the module docstring)."""
    text = PROMPT_TEMPLATE.format(clean_prompt(prompt))
    extra = {"images": None} if hasattr(processor, "image_processor") else {}  # the processor, or a plain tokenizer
    inputs = processor(text=[text], padding="longest", max_length=max_length, truncation=True, return_tensors="pt",
                       **extra)
    device = _input_device(text_encoder)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    hidden = text_encoder(**inputs, output_hidden_states=True).hidden_states
    dtype = torch.bfloat16
    normalized = {}
    for layer in sorted({layer for layers in TEXT_ENCODER_OUT_LAYERS for layer in layers}):
        states = hidden[layer].float()
        normalized[layer] = (states * torch.rsqrt(states.pow(2).mean(dim=-1, keepdim=True) + 1e-6)).to(dtype)
    bundles = [torch.cat([normalized[layer] for layer in layers], dim=-1) for layers in TEXT_ENCODER_OUT_LAYERS]
    return torch.stack(bundles, dim=1)[0].cpu()  # [6, S, 7680]


def get_tokenizer(cfg: TrainConfig, model_dir: Path | None):
    from transformers import AutoTokenizer

    if moss_loading._use_file(cfg, "text_encoder_file"):
        from ..krea2 import single_file as ksf

        return AutoTokenizer.from_pretrained(str(ksf.resolve_tokenizer(cfg.tokenizer_path, cfg.models_dir)))
    return AutoTokenizer.from_pretrained(str(Path(model_dir) / "processor"))


def prompt_tokens(tokenizer, prompt: str) -> tuple[str, list, int, int]:
    """encode_prompt's text (the chat template around the cleaned prompt) with its token offsets; the cache keeps
    every token."""
    tok = getattr(tokenizer, "tokenizer", tokenizer)  # a processor, or the tokenizer itself
    text = PROMPT_TEMPLATE.format(clean_prompt(prompt))
    offsets = tok(text, return_offsets_mapping=True)["offset_mapping"]
    return text, [tuple(o) for o in offsets], 0, len(offsets)


def pad_embeds(rows: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """[6, S_i, 7680] rows -> [B, 6, S, 7680] and a [B, S] bool mask of the real tokens."""
    length = max(r.shape[1] for r in rows)
    embeds = rows[0].new_zeros(len(rows), rows[0].shape[0], length, rows[0].shape[-1])
    mask = torch.zeros(len(rows), length, dtype=torch.bool)
    for i, r in enumerate(rows):
        embeds[i, :, : r.shape[1]] = r
        mask[i, : r.shape[1]] = True
    return embeds, mask


# ---------------------------------------------------------------- config-driven entry points


def resolve_for_config(cfg: TrainConfig) -> Path | None:
    """Download / locate the HF model dir when any component comes from it (single files cover the rest)."""
    if cfg.model_source == "files" and all(getattr(cfg, f) for f in ("transformer_file", "text_encoder_file", "vae_file")):
        return None
    return moss_loading.resolve_model_dir(cfg.model_path, cfg.models_dir)


def get_transformer(cfg: TrainConfig, model_dir: Path | None, device: str | None = None) -> FiboTransformer:
    if moss_loading._use_file(cfg, "transformer_file"):
        from . import single_file as fsf

        moss_loading._check_files(cfg, "transformer_file")
        return fsf.load_transformer_file(cfg.transformer_file, device or cfg.device, cfg.transformer_quant)
    return load_transformer(model_dir, device or cfg.device, cfg.transformer_quant)


def get_vae(cfg: TrainConfig, model_dir: Path | None):
    if cfg.model_source == "files" and cfg.vae_file:  # the Flux2 VAE file is shared with MOSS-Image
        return sf.load_vae_file(cfg.vae_file, cfg.device)
    return moss_loading.load_vae(model_dir, cfg.device)


def get_text_encoder(cfg: TrainConfig, model_dir: Path | None):
    if moss_loading._use_file(cfg, "text_encoder_file"):  # Comfy-Org's Qwen3-VL-4B file, the same one Krea 2 uses
        from transformers import AutoTokenizer

        from ..krea2 import single_file as ksf

        moss_loading._check_files(cfg, "text_encoder_file")
        tokenizer = AutoTokenizer.from_pretrained(str(ksf.resolve_tokenizer(cfg.tokenizer_path, cfg.models_dir)))
        model = ksf.load_text_encoder_file(cfg.text_encoder_file, cfg.text_encoder_device, cfg.text_encoder_quant)
        return model, tokenizer
    return load_text_encoder(model_dir, cfg.text_encoder_device, cfg.text_encoder_quant)


def load_any_transformer(path: str, device, quant: str = "auto", models_dir: str = "models") -> FiboTransformer:
    p = Path(path).expanduser()
    if p.is_file():
        from . import single_file as fsf

        return fsf.load_transformer_file(p, device, quant)
    if p.is_dir() and (p / "transformer" / "config.json").is_file():
        return load_transformer(p, device, quant)
    return load_transformer(moss_loading.resolve_model_dir(path, models_dir), device, quant)
