"""Load Qwen-Image 2.1 components from the Hugging Face / diffusers layout (Qwen/Qwen-Image-2.1, -Turbo) and encode;
single ComfyUI-style files (Comfy-Org/Qwen-Image-2.1, the default) go through qwen21.single_file.

    model_index.json  QwenImage21Pipeline; the Turbo checkpoint lists its own sample_sigmas
    transformer/      config.json (QwenImage21Transformer2DModel) + sharded diffusion_pytorch_model*.safetensors
    text_encoder/     Qwen3-VL-8B (Qwen3VLForConditionalGeneration), sharded model*.safetensors
    processor/        Qwen3VLProcessor (its tokenizer is the one used here)
    vae/              AutoencoderKLQwenImage21 (RGBA in, 64 channels, 16x; per-channel latents_mean / latents_std)

Text features: the pipeline's _get_qwen_prompt_embeds for a text-to-image prompt. The prompt goes into a raw
system/user template string, Qwen3-VL runs over it and the last decoder layer's output BEFORE the final RMSNorm is
taken (the reference pipeline hooks the norm out: transformers 5 returns the normed state otherwise); the system
turn's tokens in front are dropped. The cache keeps [S, 4096] bf16, S = the user turn's tokens (prompt included) and
the assistant header. An empty prompt is encoded as " " (Qwen has no BOS token to read).

Images: RGB training pictures get an opaque alpha channel (the VAE is RGBA), and decode drops the alpha again."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

from .. import quant as Q
from ..config import TrainConfig
from ..moss import loading as moss_loading
from ..moss import single_file as sf
from .model import QwenImage21Transformer

log = logging.getLogger("rulztrainer")

TRANSFORMER_CLASSES = ("QwenImage21Transformer2DModel",)
# quantize the 32 blocks; the projections, timestep MLP, shared modulation and final layer stay in bf16 (as ComfyUI's
# int8_convrot file does)
QUANT_FILTER = lambda name: name.startswith("transformer_blocks.")  # noqa: E731
# "auto" by the GPU memory the model will live in (all GPUs of a model split): bf16 is 14.2 GB, int8 7.3 GB, nvfp4
# about 4 GB (Fizgig: bf16 from 24 GB, int8 on 12-16 GB cards)
AUTO_BF16_FROM_GB, AUTO_INT8_FROM_GB = 24, 11

SYSTEM_PROMPT = "Comprehend and analyze the provided prompt."
PROMPT_PREFIX = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n"
PROMPT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
SYSTEM_TURN = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"  # the tokens dropped from the encoder output


def is_qwen21_dir(model_dir: Path) -> bool:
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
    """`auto`: bf16 with 24 GB+, int8 from 11 GB, else nvfp4, counting the memory of every GPU the model is split
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
    fmt = "none" if total >= AUTO_BF16_FROM_GB else "int8_convrot" if total >= AUTO_INT8_FROM_GB else "nvfp4"
    if fmt != "none":
        log.info("Qwen-Image 2.1 transformer: auto quantization -> %s for %.0f GB (%s)", fmt, total, where)
    return fmt


def load_transformer(model_dir: Path, device, quant: str = "auto", dtype=torch.bfloat16,
                     quant_devices=None) -> QwenImage21Transformer:
    """quant_devices: the GPUs the model will run on, for quant "auto" (default: `device`)."""
    config = json.loads((model_dir / "transformer" / "config.json").read_text(encoding="utf-8"))
    if config.get("_class_name") not in TRANSFORMER_CLASSES:
        raise ValueError(f"Not a Qwen-Image 2.1 transformer: {config.get('_class_name')!r}")
    if config.get("patch_size", 1) != 1:
        raise ValueError("Qwen-Image 2.1 reads its latents unpatched (patch_size 1)")
    quant = resolve_quant(quant, quant_devices or device)
    with torch.device("meta"):
        model = QwenImage21Transformer.from_config(config)
    state = moss_loading._read_sharded(model_dir / "transformer", "diffusion_pytorch_model")
    state = {k: v.to(dtype) for k, v in state.items()}
    from .single_file import unfuse_mlp

    state = unfuse_mlp(state)
    sf.install_state(model, state, quant, dtype, quant_filter=QUANT_FILTER, **sf.quant_target(device, quant_devices))
    del state
    model.is_distilled = bool(pipeline_config(model_dir).get("sample_sigmas"))
    model = model.to(device).eval().requires_grad_(False)
    log.info("Loaded Qwen-Image 2.1 transformer from %s: %d layers, dim %d, %s, quant=%s", model_dir.name,
             config["num_layers"], config["attention_head_dim"] * config["num_attention_heads"],
             "distilled (turbo)" if model.is_distilled else "base", quant)
    return model


def load_vae(model_dir: Path, device):
    from .single_file import vae_class

    vae = vae_class().from_pretrained(str(model_dir / "vae"), torch_dtype=torch.bfloat16)
    return vae.to(device).eval().requires_grad_(False)


def load_text_encoder(model_dir: Path, device: str = "auto", quant: str = "auto"):
    """Qwen3-VL-8B's language model (text only) + tokenizer. quant int8 / int8_convrot / nvfp4 quantizes its Linear
    layers so the 16 GB encoder takes about half."""
    from transformers import AutoTokenizer

    from ..fibo.loading import place_text_encoder
    from ..krea2.single_file import load_hf_text_model
    from .single_file import TE_INT8_MATMUL

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir / "processor"))
    model = load_hf_text_model(model_dir / "text_encoder", "none" if quant == "auto" else quant, device)
    Q.configure_int8(model, TE_INT8_MATMUL)
    return place_text_encoder(model, device), tokenizer


def _input_device(model) -> torch.device:
    return model.get_input_embeddings().weight.device


def system_tokens(tokenizer) -> int:
    """The system turn's token count (the pipeline derives it from the tokenized system message too)."""
    return len(tokenizer(SYSTEM_TURN, add_special_tokens=False).input_ids)


@torch.no_grad()
def encode_prompts(text_encoder, tokenizer, prompts: list[str], max_length: int = 1024) -> list[torch.Tensor]:
    """Text features for each prompt: [S_i, 4096] bf16 on the CPU (see the module docstring), in one forward. A
    prompt is cut to max_length tokens of user turn (the assistant header always follows)."""
    drop = system_tokens(tokenizer)
    suffix = tokenizer([PROMPT_SUFFIX], add_special_tokens=False, return_tensors="pt").input_ids[0]
    rows = [torch.cat([tokenizer([PROMPT_PREFIX + (prompt or " ")], add_special_tokens=False, truncation=True,
                                 max_length=drop + max(8, int(max_length) - suffix.numel()),
                                 return_tensors="pt").input_ids[0], suffix]) for prompt in prompts]
    ids, mask = moss_loading.right_pad(rows)
    device = _input_device(text_encoder)
    b, n = ids.shape
    position_ids = torch.arange(n, device=device).view(1, 1, n).expand(3, b, n)
    norm = getattr(text_encoder, "norm", None) or text_encoder.language_model.norm
    handle = norm.register_forward_hook(lambda module, args, output: args[0])  # the state before the final norm
    try:
        out = text_encoder(input_ids=ids.to(device), attention_mask=mask.to(device), position_ids=position_ids)
    finally:
        handle.remove()
    hidden = out.last_hidden_state.to(torch.bfloat16).cpu()
    return [hidden[i, drop: r.numel()].clone() for i, r in enumerate(rows)]


def encode_prompt(text_encoder, tokenizer, prompt: str, max_length: int = 1024) -> torch.Tensor:
    """Text features for one prompt: [S, 4096] bf16 on the CPU (see the module docstring). The prompt is cut to
    max_length tokens of user turn (the assistant header always follows)."""
    return encode_prompts(text_encoder, tokenizer, [prompt], max_length)[0]


def get_tokenizer(cfg: TrainConfig, model_dir: Path | None):
    from transformers import AutoTokenizer

    if moss_loading._use_file(cfg, "text_encoder_file"):
        from . import single_file as qsf

        return AutoTokenizer.from_pretrained(str(qsf.resolve_tokenizer(cfg.tokenizer_path, cfg.models_dir)))
    return AutoTokenizer.from_pretrained(str(_need_dir(cfg, model_dir, "tokenizer") / "processor"))


def prompt_tokens(tokenizer, prompt: str) -> tuple[str, list, int, int]:
    """encode_prompt's text (the template up to and including the prompt; the suffix follows) with its token
    offsets; the cache drops the system turn's tokens and keeps the suffix ones."""
    text = PROMPT_PREFIX + (prompt or " ")
    offsets = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]
    drop = system_tokens(tokenizer)
    n_suffix = len(tokenizer(PROMPT_SUFFIX, add_special_tokens=False).input_ids)
    return text, [tuple(o) for o in offsets], drop, len(offsets) - drop + n_suffix


def pad_embeds(rows: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """[S_i, 4096] rows -> [B, S, 4096] (right-padded with zeros) and a [B, S] bool mask of the real tokens."""
    length = max(r.shape[0] for r in rows)
    embeds = rows[0].new_zeros(len(rows), length, rows[0].shape[-1])
    mask = torch.zeros(len(rows), length, dtype=torch.bool)
    for i, r in enumerate(rows):
        embeds[i, : r.shape[0]] = r
        mask[i, : r.shape[0]] = True
    return embeds, mask


# ---------------------------------------------------------------- VAE (Qwen-Image 2.1, RGBA), the trainer's layout


def _stats(vae, like: torch.Tensor):
    shape = (1, vae.config.z_dim, 1, 1)
    mean = torch.tensor(vae.config.latents_mean).view(shape).to(like)
    std = torch.tensor(vae.config.latents_std).view(shape).to(like)
    return mean, std


def encode_images(vae, images: torch.Tensor) -> torch.Tensor:
    """images: [B,3,H,W] in [-1,1] (H,W multiples of 16). Returns normalized latents [B,64,H/16,W/16] (the model reads
    them unpatched). An opaque alpha channel is added: the VAE is RGBA."""
    images = images.to(vae.device, vae.dtype)
    images = torch.cat([images, torch.ones_like(images[:, :1])], dim=1).unsqueeze(2)  # a one-frame RGBA video
    z = vae.encode(images).latent_dist.mode()[:, :, 0].float()
    mean, std = _stats(vae, z)
    return (z - mean) / std


def _decode(vae, latents: torch.Tensor) -> torch.Tensor:
    z = latents.to(vae.device, torch.float32)
    mean, std = _stats(vae, z)
    z = (z * std + mean).to(vae.dtype)
    return vae.decode(z.unsqueeze(2), return_dict=False)[0][:, :3, 0]  # RGB: the alpha channel is dropped


TILE_ABOVE_PX = 512 * 512  # decodes above this many pixels go through the VAE's tiles


@torch.no_grad()
def decode_latents(vae, latents: torch.Tensor) -> torch.Tensor:
    """Inverse of encode_images. Returns images [B,3,H,W] in [-1,1]. Pictures above 512 x 512 decode in 512 px tiles
    (384 px apart, the overlaps blended): this VAE normalizes per pixel, so the tiles give the full decode to 0.3 of an
    8-bit level on average (PSNR 54 dB, measured at 1024 x 1024), with a peak of +1.7 GB instead of +6.6 GB, which
    beside the model on a 16 GB card ran out of memory. Encoding (the training latents) is never tiled."""
    pixels = latents.shape[-2] * latents.shape[-1] * 256
    tiled = pixels > TILE_ABOVE_PX
    prev = getattr(vae, "use_tiling", False)
    if tiled:
        vae.enable_tiling(tile_sample_min_height=512, tile_sample_min_width=512, tile_sample_stride_height=384,
                          tile_sample_stride_width=384)
    try:
        return _decode(vae, latents).float().clamp(-1, 1)
    finally:
        vae.use_tiling = prev


def decode_latents_grad(vae, latents: torch.Tensor) -> torch.Tensor:
    """decode_latents with gradients to `latents` (identity loss on a face crop), not clamped."""
    return _decode(vae, latents).float()


def grad_vae(vae):
    """The VAE decode_latents_grad runs on: this one is already bf16 and frozen, so itself."""
    return vae


# ---------------------------------------------------------------- config-driven entry points


def _need_dir(cfg: TrainConfig, model_dir: Path | None, what: str) -> Path:
    if model_dir is None:
        raise ValueError(f"No {what} file is set and no Hugging Face model to take it from: set the file (model "
                         f"source 'files') or the Qwen/Qwen-Image-2.1 repo id / a diffusers folder")
    return model_dir


def resolve_for_config(cfg: TrainConfig) -> Path | None:
    """Download / locate the HF model dir when any component comes from it (single files cover the rest)."""
    if cfg.model_source == "files" and all(getattr(cfg, f) for f in ("transformer_file", "text_encoder_file", "vae_file")):
        return None
    return moss_loading.resolve_model_dir(cfg.model_path, cfg.models_dir)


def get_transformer(cfg: TrainConfig, model_dir: Path | None, device: str | None = None) -> QwenImage21Transformer:
    from . import single_file as qsf

    use_file = moss_loading._use_file(cfg, "transformer_file")
    if use_file:
        moss_loading._check_files(cfg, "transformer_file")
    else:
        model_dir = _need_dir(cfg, model_dir, "diffusion model")
    if cfg.multi_gpu == "model_split" and device is None:
        # the trainer splits it right after loading: stay on the CPU and size "auto" for all the split GPUs
        from ..multigpu import parse_devices

        gpus = parse_devices(cfg.devices, cfg.device)
        if use_file:
            return qsf.load_transformer_file(cfg.transformer_file, "cpu", cfg.transformer_quant, quant_devices=gpus)
        return load_transformer(model_dir, "cpu", cfg.transformer_quant, quant_devices=gpus)
    if use_file:
        return qsf.load_transformer_file(cfg.transformer_file, device or cfg.device, cfg.transformer_quant)
    return load_transformer(model_dir, device or cfg.device, cfg.transformer_quant)


def get_vae(cfg: TrainConfig, model_dir: Path | None):
    if moss_loading._use_file(cfg, "vae_file"):
        from . import single_file as qsf

        moss_loading._check_files(cfg, "vae_file")
        return qsf.load_vae_file(cfg.vae_file, cfg.device)
    return load_vae(_need_dir(cfg, model_dir, "VAE"), cfg.device)


def get_text_encoder(cfg: TrainConfig, model_dir: Path | None):
    if moss_loading._use_file(cfg, "text_encoder_file"):
        from transformers import AutoTokenizer

        from . import single_file as qsf

        moss_loading._check_files(cfg, "text_encoder_file")
        tokenizer = AutoTokenizer.from_pretrained(str(qsf.resolve_tokenizer(cfg.tokenizer_path, cfg.models_dir)))
        model = qsf.load_text_encoder_file(cfg.text_encoder_file, cfg.text_encoder_device, cfg.text_encoder_quant)
        return model, tokenizer
    return load_text_encoder(_need_dir(cfg, model_dir, "text encoder"), cfg.text_encoder_device,
                             cfg.text_encoder_quant)


def load_any_transformer(path: str, device, quant: str = "auto", models_dir: str = "models") -> QwenImage21Transformer:
    """Load a transformer from a single .safetensors file, a diffusers directory, or a HF repo id."""
    p = Path(path).expanduser()
    if p.is_file():
        from . import single_file as qsf

        return qsf.load_transformer_file(p, device, quant)
    if p.is_dir() and (p / "transformer" / "config.json").is_file():
        return load_transformer(p, device, quant)
    return load_transformer(moss_loading.resolve_model_dir(path, models_dir), device, quant)
