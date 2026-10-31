"""Load Krea 2 components from the Hugging Face / diffusers layout (krea/Krea-2-Raw, Krea-2-Turbo) and encode;
single ComfyUI-style files go through krea2.single_file (model_source "files").

    model_index.json  Krea2Pipeline: text_encoder_select_layers, is_distilled (Turbo)
    transformer/      config.json (Krea2Transformer2DModel) + sharded diffusion_pytorch_model*.safetensors
    text_encoder/     Qwen3-VL-4B (Qwen3VLModel), model.safetensors
    tokenizer/        Qwen2Tokenizer
    vae/              AutoencoderKLQwenImage (16 channels, 8x; per-channel latents_mean / latents_std)

Text features: the pipeline's get_text_hidden_states. The prompt goes into a fixed system/user template, the encoder
runs with positions that count only real tokens, and twelve hidden layers are stacked per token; the 34 template
tokens in front are dropped. The pipeline pads every prompt to 512 tokens in the middle of the template
([prompt | pad | suffix]) and masks the padding; padding is never attended to and text tokens carry no position in
the DiT, so the cache keeps only the real tokens ([S, 12, 2560], S = prompt tokens + the 5 suffix tokens) and gives
the same result as the padded layout.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

from .. import quant as Q
from ..config import TrainConfig
from ..fibo.loading import place_text_encoder
from ..moss import loading as moss_loading
from ..moss import single_file as sf
from .model import NORM_NAMES, Krea2Transformer

log = logging.getLogger("rulztrainer")

TRANSFORMER_CLASSES = ("Krea2Transformer2DModel",)
# quantize the 28 main blocks; the text fusion stage, projections, timestep MLP and final layer stay in bf16
QUANT_FILTER = lambda name: name.startswith("transformer_blocks.")  # noqa: E731
# "auto" by the GPU memory the model will live in (all GPUs of a model split): bf16 is 25 GB, int8 about 13.5 GB
# (14.2 GB with the VAE: a 16 GB card has no room left to train at 1 MP), nvfp4 about 8 GB
AUTO_BF16_FROM_GB, AUTO_INT8_FROM_GB = 40, 22

PROMPT_PREFIX = ("<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, "
                 "spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n")
PROMPT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
PREFIX_TOKENS = 34  # the system prefix, dropped from the encoder output
SUFFIX_TOKENS = 5
MAX_TEXT_TOKENS = 512  # the pipeline's fixed text length (prompt + suffix)
SELECT_LAYERS = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)  # model_index.json text_encoder_select_layers


def is_krea2_dir(model_dir: Path) -> bool:
    cfg = model_dir / "transformer" / "config.json"
    if not cfg.is_file():
        return False
    try:
        return json.loads(cfg.read_text(encoding="utf-8")).get("_class_name") in TRANSFORMER_CLASSES
    except (OSError, ValueError):
        return False


def pipeline_config(model_dir: Path) -> dict:
    try:
        return json.loads((model_dir / "model_index.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _gpu_gb(device) -> float:
    dev = torch.device(device)
    return torch.cuda.get_device_properties(dev.index or 0).total_memory / 2**30


def resolve_quant(quant: str, devices) -> str:
    """`auto`: bf16 with 40 GB+, int8 from 22 GB, else nvfp4, counting the memory of every GPU the model is split
    over. A model loaded on the CPU (a sampling model) is sized for the smallest GPU, where it will run."""
    if quant != "auto" or not torch.cuda.is_available():
        return "none" if quant == "auto" else quant
    devices = [devices] if isinstance(devices, (str, torch.device)) else list(devices)
    gpus = [d for d in devices if torch.device(d).type == "cuda"]
    if gpus:
        total, where = sum(_gpu_gb(d) for d in gpus), "+".join(str(d) for d in gpus)
    else:
        total = min(_gpu_gb(f"cuda:{i}") for i in range(torch.cuda.device_count()))
        where = "the smallest GPU"
    fmt = "none" if total >= AUTO_BF16_FROM_GB else "int8" if total >= AUTO_INT8_FROM_GB else "nvfp4"
    if fmt != "none":
        log.info("Krea 2 transformer: auto quantization -> %s for %.0f GB (%s)", fmt, total, where)
    return fmt


def load_transformer(model_dir: Path, device, quant: str = "auto", dtype=torch.bfloat16,
                     quant_devices=None) -> Krea2Transformer:
    """quant_devices: the GPUs the model will run on, for quant "auto" (default: `device`)."""
    config = json.loads((model_dir / "transformer" / "config.json").read_text(encoding="utf-8"))
    if config.get("_class_name") not in TRANSFORMER_CLASSES:
        raise ValueError(f"Not a Krea 2 transformer: {config.get('_class_name')!r}")
    quant = resolve_quant(quant, quant_devices or device)
    with torch.device("meta"):
        model = Krea2Transformer.from_config(config)
    state = moss_loading._read_sharded(model_dir / "transformer", "diffusion_pytorch_model")
    # like diffusers with torch_dtype=bf16: everything in bf16 except the RMSNorm scales, which stay fp32
    state = {k: v.float() if k.split(".")[-2] in NORM_NAMES else v.to(dtype) for k, v in state.items()}
    sf.install_state(model, state, quant, dtype, quant_filter=QUANT_FILTER, **sf.quant_target(device, quant_devices))
    del state
    model.is_distilled = bool(pipeline_config(model_dir).get("is_distilled", False))
    model = model.to(device).eval().requires_grad_(False)
    log.info("Loaded Krea 2 transformer from %s: %d layers, dim %d, %s, quant=%s", model_dir.name,
             config["num_layers"], config["attention_head_dim"] * config["num_attention_heads"],
             "distilled (turbo)" if model.is_distilled else "base", quant)
    return model


def load_vae(model_dir: Path, device):
    from diffusers import AutoencoderKLQwenImage

    vae = AutoencoderKLQwenImage.from_pretrained(str(model_dir / "vae"), torch_dtype=torch.float32)
    return vae.to(device).eval().requires_grad_(False)


def load_text_encoder(model_dir: Path, device: str = "auto", quant: str = "auto"):
    """Qwen3-VL's language model (text only: the folder's Qwen3VLModel without its vision tower, which text prompts
    never use, 0.8 GB) + tokenizer. quant int8/int8_convrot/nvfp4 quantizes its Linear layers (W8A8) so the 8.9 GB
    encoder takes about half the memory."""
    from transformers import AutoTokenizer

    from .single_file import load_hf_text_model

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir / "tokenizer"))
    fmt = "none" if quant == "auto" else quant
    model = load_hf_text_model(model_dir / "text_encoder", fmt, device)
    if fmt != "none":
        n = sum(1 for m in model.modules() if isinstance(m, Q.QuantLinear))
        log.info("Text encoder: %d Linear layers quantized to %s", n, fmt)
    model = place_text_encoder(model, device)
    return model, tokenizer


def _input_device(model) -> torch.device:
    emb = model.get_input_embeddings()
    return emb.weight.device


@torch.no_grad()
def encode_prompt(text_encoder, tokenizer, prompt: str, max_length: int = MAX_TEXT_TOKENS) -> torch.Tensor:
    """Text features for one prompt: [S, 12, 2560] bf16 on the CPU (see the module docstring)."""
    length = min(int(max_length), MAX_TEXT_TOKENS)
    ids = tokenizer([PROMPT_PREFIX + prompt], truncation=True, max_length=length + PREFIX_TOKENS - SUFFIX_TOKENS,
                    return_tensors="pt").input_ids
    suffix = tokenizer([PROMPT_SUFFIX], return_tensors="pt").input_ids
    input_ids = torch.cat([ids, suffix], dim=1).to(_input_device(text_encoder))
    n = input_ids.shape[1]
    position_ids = torch.arange(n, device=input_ids.device).view(1, 1, n).expand(3, 1, n)
    out = text_encoder(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), position_ids=position_ids,
                       output_hidden_states=True)
    hidden = torch.stack([out.hidden_states[i] for i in SELECT_LAYERS], dim=2)
    return hidden[0, PREFIX_TOKENS:].to(torch.bfloat16).cpu()


def get_tokenizer(cfg: TrainConfig, model_dir: Path | None):
    from transformers import AutoTokenizer

    if moss_loading._use_file(cfg, "text_encoder_file"):
        from . import single_file as ksf

        return AutoTokenizer.from_pretrained(str(ksf.resolve_tokenizer(cfg.tokenizer_path, cfg.models_dir)))
    return AutoTokenizer.from_pretrained(str(_need_dir(cfg, model_dir, "tokenizer") / "tokenizer"))


def prompt_tokens(tokenizer, prompt: str) -> tuple[str, list, int, int]:
    """encode_prompt's text (the system prefix + the prompt; the suffix follows) with its token offsets; the cache
    drops the PREFIX_TOKENS prefix tokens and keeps the SUFFIX_TOKENS suffix ones."""
    text = PROMPT_PREFIX + prompt
    offsets = tokenizer(text, return_offsets_mapping=True)["offset_mapping"]
    return text, [tuple(o) for o in offsets], PREFIX_TOKENS, len(offsets) - PREFIX_TOKENS + SUFFIX_TOKENS


def pad_embeds(rows: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """[S_i, 12, 2560] rows -> [B, S, 12, 2560] and a [B, S] bool mask of the real tokens."""
    length = max(r.shape[0] for r in rows)
    embeds = rows[0].new_zeros(len(rows), length, *rows[0].shape[1:])
    mask = torch.zeros(len(rows), length, dtype=torch.bool)
    for i, r in enumerate(rows):
        embeds[i, : r.shape[0]] = r
        mask[i, : r.shape[0]] = True
    return embeds, mask


# ---------------------------------------------------------------- VAE (Qwen-Image), the trainer's latent layout


def _stats(vae, like: torch.Tensor):
    shape = (1, vae.config.z_dim, 1, 1)
    mean = torch.tensor(vae.config.latents_mean).view(shape).to(like)
    std = torch.tensor(vae.config.latents_std).view(shape).to(like)
    return mean, std


def encode_images(vae, images: torch.Tensor) -> torch.Tensor:
    """images: [B,3,H,W] in [-1,1] (H,W multiples of 16). Returns normalized, patchified [B,64,H/16,W/16]; the 64
    features are (channel, dy, dx), the order Krea2Pipeline packs them in."""
    images = images.to(vae.device, vae.dtype).unsqueeze(2)  # a one-frame video
    z = vae.encode(images).latent_dist.mode()[:, :, 0]
    mean, std = _stats(vae, z)
    z = (z - mean) / std
    b, c, h, w = z.shape
    return z.view(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(b, c * 4, h // 2, w // 2)


def _decode(vae, latents: torch.Tensor) -> torch.Tensor:
    z = latents.to(vae.device, torch.float32)  # the latent statistics apply in fp32 (a bf16 VAE gets the result)
    b, c, h, w = z.shape
    z = z.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c // 4, 2 * h, 2 * w)
    mean, std = _stats(vae, z)
    z = (z * std + mean).to(vae.dtype)
    return vae.decode(z.unsqueeze(2), return_dict=False)[0][:, :, 0]


@torch.no_grad()
def decode_latents(vae, latents: torch.Tensor) -> torch.Tensor:
    """Inverse of encode_images. Returns images [B,3,H,W] in [-1,1].

    Decoded in 512 px tiles (384 px apart, the overlaps blended): this VAE normalizes per pixel and its convolutions
    see a few pixels around each tile, so the result is the full decode's to 0.06 of an 8-bit level on average
    (PSNR 68 dB, measured at 1024 x 1536), with a peak of +2.0 GB instead of +7.9 GB at 1 MP and +11.8 GB at 1.5 MP.
    The tiles run on a bf16 copy of the VAE (moss.loading.decode_vae): the fp32 tiles to 0.18-0.24 of an 8-bit level
    on average (PSNR 53-55 dB, measured at 1024 x 1024, 832 x 1248 and 1248 x 1248), with a peak of +1.3 GB instead
    of +2.0 GB (the 240 MB copy included) and 25-35% faster. Encoding (the training latents) is never tiled and stays
    fp32."""
    dec = moss_loading.decode_vae(vae)
    prev = dec.use_tiling
    dec.enable_tiling(tile_sample_min_height=512, tile_sample_min_width=512, tile_sample_stride_height=384,
                      tile_sample_stride_width=384)
    try:
        return _decode(dec, latents).float().clamp(-1, 1)
    finally:
        dec.use_tiling = prev


def decode_latents_grad(vae, latents: torch.Tensor) -> torch.Tensor:
    """decode_latents with gradients to `latents` (identity loss on a face crop), not clamped."""
    return _decode(vae, latents).float()


def grad_vae(vae):
    """A bf16 copy of the VAE for decode_latents_grad (this VAE has no gradient checkpointing)."""
    return moss_loading.bf16_copy(vae)


# ---------------------------------------------------------------- config-driven entry points


def _need_dir(cfg: TrainConfig, model_dir: Path | None, what: str) -> Path:
    if model_dir is None:
        raise ValueError(f"No {what} file is set and no Hugging Face model to take it from: set the file (model "
                         f"source 'files') or the krea/Krea-2-Raw repo id / a diffusers folder")
    return model_dir


def resolve_for_config(cfg: TrainConfig) -> Path | None:
    """Download / locate the HF model dir when any component comes from it (single files cover the rest)."""
    if cfg.model_source == "files" and all(getattr(cfg, f) for f in ("transformer_file", "text_encoder_file", "vae_file")):
        return None
    return moss_loading.resolve_model_dir(cfg.model_path, cfg.models_dir)


def get_transformer(cfg: TrainConfig, model_dir: Path | None, device: str | None = None) -> Krea2Transformer:
    from . import single_file as ksf

    use_file = moss_loading._use_file(cfg, "transformer_file")
    if use_file:
        moss_loading._check_files(cfg, "transformer_file")
    else:
        model_dir = _need_dir(cfg, model_dir, "diffusion model")
    if cfg.multi_gpu == "model_split" and device is None:
        # the trainer splits it right after loading: stay on the CPU (25 GB of bf16 fit no single 16 GB card) and
        # size "auto" for all the split GPUs
        from ..multigpu import parse_devices

        gpus = parse_devices(cfg.devices, cfg.device)
        if use_file:
            return ksf.load_transformer_file(cfg.transformer_file, "cpu", cfg.transformer_quant, quant_devices=gpus)
        return load_transformer(model_dir, "cpu", cfg.transformer_quant, quant_devices=gpus)
    if use_file:
        return ksf.load_transformer_file(cfg.transformer_file, device or cfg.device, cfg.transformer_quant)
    return load_transformer(model_dir, device or cfg.device, cfg.transformer_quant)


def get_vae(cfg: TrainConfig, model_dir: Path | None):
    if moss_loading._use_file(cfg, "vae_file"):
        from . import single_file as ksf

        moss_loading._check_files(cfg, "vae_file")
        return ksf.load_vae_file(cfg.vae_file, cfg.device)
    return load_vae(_need_dir(cfg, model_dir, "VAE"), cfg.device)


def get_text_encoder(cfg: TrainConfig, model_dir: Path | None):
    if moss_loading._use_file(cfg, "text_encoder_file"):
        from transformers import AutoTokenizer

        from . import single_file as ksf

        moss_loading._check_files(cfg, "text_encoder_file")
        tokenizer = AutoTokenizer.from_pretrained(str(ksf.resolve_tokenizer(cfg.tokenizer_path, cfg.models_dir)))
        model = ksf.load_text_encoder_file(cfg.text_encoder_file, cfg.text_encoder_device, cfg.text_encoder_quant)
        return model, tokenizer
    return load_text_encoder(_need_dir(cfg, model_dir, "text encoder"), cfg.text_encoder_device,
                             cfg.text_encoder_quant)


def load_any_transformer(path: str, device, quant: str = "auto", models_dir: str = "models") -> Krea2Transformer:
    """Load a transformer from a single .safetensors file, a diffusers directory, or a HF repo id."""
    p = Path(path).expanduser()
    if p.is_file():
        from . import single_file as ksf

        return ksf.load_transformer_file(p, device, quant)
    if p.is_dir() and (p / "transformer" / "config.json").is_file():
        return load_transformer(p, device, quant)
    return load_transformer(moss_loading.resolve_model_dir(path, models_dir), device, quant)
