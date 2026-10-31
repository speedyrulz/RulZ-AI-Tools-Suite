"""Download and load MOSS-Image components (HF/diffusers layout or single ComfyUI files); encode helpers.

HF/diffusers model directory layout:
    transformer/   config.json + diffusion_pytorch_model*.safetensors (+ index)
    text_encoder/  Qwen3.5-9B (model_type qwen3_5), sharded safetensors
    tokenizer/
    vae/           AutoencoderKLFlux2
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from safetensors.torch import load_file

from ..config import TrainConfig
from . import single_file as sf
from .model import MOSSImageTransformer

log = logging.getLogger("rulztrainer")

# also skipped: Krea 2's sample images and its single-file copies of the transformer (26 GB each, the diffusers
# transformer/ folder holds the same weights)
IGNORE_PATTERNS = ["assets/*", "*.webp", "*.svg", "*.md", "scheduler/*.py", "transformer/*.py", "images/*",
                   "raw.safetensors", "turbo.safetensors"]


def resolve_model_dir(model_path: str, models_dir: str = "models", download: bool = True) -> Path:
    """Return a local directory for a HF repo id or local path, downloading if needed."""
    p = Path(model_path).expanduser()
    if p.is_dir():
        return p.resolve()
    local = Path(models_dir) / model_path.replace("/", "__")
    if (local / "transformer" / "config.json").is_file():
        return local.resolve()
    if not download:
        raise FileNotFoundError(f"Model not found locally: {model_path} (looked in {local})")
    from huggingface_hub import snapshot_download

    log.info("Downloading %s to %s (about 30 GB)...", model_path, local)
    snapshot_download(model_path, local_dir=str(local), ignore_patterns=IGNORE_PATTERNS)
    return local.resolve()


def _read_sharded(directory: Path, stem: str, device="cpu") -> dict[str, torch.Tensor]:
    index = directory / f"{stem}.safetensors.index.json"
    if not index.exists():
        return load_file(str(directory / f"{stem}.safetensors"), device=device)
    mapping = json.loads(index.read_text())["weight_map"]
    state = {}
    for filename in sorted(set(mapping.values())):
        state.update(load_file(str(directory / filename), device=device))
    return state


# ---------------------------------------------------------------- HF-layout loaders


def load_transformer(model_dir: Path, device: str | torch.device, dtype=torch.bfloat16,
                     quant: str = "none", quant_devices=None) -> MOSSImageTransformer:
    """quant_devices: the GPUs of a model split, when the model is staged on the CPU (they quantize it)."""
    config = json.loads((model_dir / "transformer" / "config.json").read_text())
    if config.get("_class_name") not in ("MOSImageTransformer", "MOSSImageTransformer"):
        raise ValueError(f"Unsupported transformer class {config.get('_class_name')!r}")
    state = _read_sharded(model_dir / "transformer", "diffusion_pytorch_model")
    state = {k: v.to(dtype) for k, v in state.items()}
    model = sf.transformer_from_hf_state(state, config, device, quant, dtype, quant_devices=quant_devices)
    del state
    log.info("Loaded transformer: %d layers, dim %d, quant=%s", config["num_layers"], config["dim"], quant)
    return model


def load_vae(model_dir: Path, device):
    from diffusers import AutoencoderKLFlux2

    vae = AutoencoderKLFlux2.from_pretrained(str(model_dir / "vae"), torch_dtype=torch.float32)
    return vae.to(device).eval().requires_grad_(False)


def load_text_encoder(model_dir: Path, device: str = "auto", quant: str = "auto"):
    """Qwen3.5 text encoder (text-only tower) + tokenizer from an HF-layout directory."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir / "tokenizer"))
    state = _read_sharded(model_dir / "text_encoder", "model")
    model = sf.build_text_model(state, "none" if quant == "auto" else quant, "cpu",
                                work_device=sf.quant_target(device).get("work_device"))
    del state
    return sf.place_text_encoder(model, device), tokenizer


# ---------------------------------------------------------------- config-driven dispatch


def _check_files(cfg: TrainConfig, *names: str) -> None:
    for n in names:
        p = getattr(cfg, n)
        if not p or not Path(p).is_file():
            raise FileNotFoundError(f"{n} is not set or does not exist: {p!r}")


def _use_file(cfg: TrainConfig, field: str) -> bool:
    """In files mode each component comes from its file; an empty field falls back to the HF repo."""
    return cfg.model_source == "files" and bool(getattr(cfg, field))


def get_transformer(cfg: TrainConfig, model_dir: Path | None, device: str | None = None) -> MOSSImageTransformer:
    use_file = _use_file(cfg, "transformer_file")
    if use_file:
        _check_files(cfg, "transformer_file")
    if cfg.multi_gpu == "model_split" and device is None:
        # the trainer splits it right after loading: stay on the CPU (loading it whole onto the first GPU first ran
        # out of memory on a pair of 16 GB cards), quantizing on the split's GPUs, as Krea 2 does
        from ..multigpu import parse_devices

        gpus = parse_devices(cfg.devices, cfg.device)
        if use_file:
            return sf.load_transformer_file(cfg.transformer_file, "cpu", cfg.transformer_quant, quant_devices=gpus)
        return load_transformer(model_dir, "cpu", quant=cfg.transformer_quant, quant_devices=gpus)
    device = device or cfg.device
    if use_file:
        return sf.load_transformer_file(cfg.transformer_file, device, cfg.transformer_quant)
    return load_transformer(model_dir, device, quant=cfg.transformer_quant)


def get_vae(cfg: TrainConfig, model_dir: Path | None):
    if _use_file(cfg, "vae_file"):
        _check_files(cfg, "vae_file")
        return sf.load_vae_file(cfg.vae_file, cfg.device)
    return load_vae(model_dir, cfg.device)


def check_tokenizer(tokenizer) -> None:
    """MOSS-Image's text encoder reads Qwen3.5 token ids: refuse a smaller vocabulary (another model's tokenizer)."""
    if len(tokenizer) < sf.MOSS_TOKENIZER_MIN_VOCAB:
        raise ValueError(f"The tokenizer has {len(tokenizer)} tokens: it is not MOSS-Image's Qwen3.5 tokenizer "
                         f"(248k). Point tokenizer_path at MOSS-Image's tokenizer folder, or leave it empty to download it")


def get_text_encoder(cfg: TrainConfig, model_dir: Path | None):
    if _use_file(cfg, "text_encoder_file"):
        from transformers import AutoTokenizer

        _check_files(cfg, "text_encoder_file")
        tok_dir = sf.resolve_tokenizer(cfg.tokenizer_path, cfg.models_dir)
        tokenizer = AutoTokenizer.from_pretrained(str(tok_dir))
        check_tokenizer(tokenizer)
        return sf.load_text_encoder_file(cfg.text_encoder_file, cfg.text_encoder_device, cfg.text_encoder_quant), tokenizer
    return load_text_encoder(model_dir, cfg.text_encoder_device, cfg.text_encoder_quant)


def load_any_transformer(path: str, device, quant: str = "auto", models_dir: str = "models") -> MOSSImageTransformer:
    """Load a transformer from a single .safetensors file, a diffusers directory, or a HF repo id."""
    p = Path(path).expanduser()
    if p.is_file():
        return sf.load_transformer_file(p, device, quant)
    if p.is_dir() and (p / "transformer" / "config.json").is_file():
        return load_transformer(p, device, quant=quant)
    return load_transformer(resolve_model_dir(path, models_dir), device, quant=quant)


def resolve_for_config(cfg: TrainConfig) -> Path | None:
    """Download/locate the HF model dir when any component comes from it."""
    if cfg.model_source == "files" and all(getattr(cfg, f) for f in ("transformer_file", "text_encoder_file", "vae_file")):
        return None
    return resolve_model_dir(cfg.model_path, cfg.models_dir)


# ---------------------------------------------------------------- encoding


@torch.no_grad()
def encode_images(vae, images: torch.Tensor) -> torch.Tensor:
    """images: [B,3,H,W] in [-1,1] (H,W multiples of 16). Returns normalized [B,128,H/16,W/16]."""
    images = images.to(vae.device, vae.dtype)
    posterior = vae.encode(images).latent_dist
    z = posterior.mode()
    b, c, h, w = z.shape
    z = z.view(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(b, c * 4, h // 2, w // 2)
    mean = vae.bn.running_mean.view(1, -1, 1, 1).to(z)
    std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(z)
    return (z - mean) / std


def _decode(vae, latents: torch.Tensor) -> torch.Tensor:
    z = latents.to(vae.device, torch.float32)  # the latent statistics apply in fp32 (a bf16 VAE gets the result)
    mean = vae.bn.running_mean.view(1, -1, 1, 1).to(z)
    std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1).to(z) + vae.config.batch_norm_eps)
    z = (z * std + mean).to(vae.dtype)
    b, c, h, w = z.shape
    z = z.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c // 4, 2 * h, 2 * w)
    return vae.decode(z, return_dict=False)[0]


def decode_vae(vae):
    """The VAE a sample / probe decode runs on: a bf16 copy of an fp32 VAE on a CUDA device (built for the decode
    and dropped after it), else the VAE itself. Encoding (the training latents) keeps the fp32 VAE. bf16 autocast was
    tried first: GroupNorm stays fp32 under it, and so do most activations, so it saved only 10% of the peak."""
    if vae.dtype == torch.float32 and vae.device.type == "cuda":
        return bf16_copy(vae)
    return vae


@torch.no_grad()
def decode_latents(vae, latents: torch.Tensor) -> torch.Tensor:
    """Inverse of encode_images. Returns images [B,3,H,W] in [-1,1].

    Decoded whole (never tiled: this VAE's GroupNorm and attention see the whole image, tiles shifted tones by 4-7
    levels) on a bf16 copy of the VAE (decode_vae): the fp32 decode to 0.16 of an 8-bit level on average (PSNR 56 dB,
    measured at 1024 x 1024, 832 x 1248 and 1248 x 1248), with a peak of +2.55 GB instead of +3.76 GB at 1 MP and
    +3.7 GB instead of +5.6 GB at 1.5 MP (the 160 MB copy included), and 35-40% faster."""
    return _decode(decode_vae(vae), latents).float().clamp(-1, 1)


def decode_latents_grad(vae, latents: torch.Tensor) -> torch.Tensor:
    """decode_latents with gradients to `latents` (identity loss on a face crop). Not clamped: a clamp would zero
    the gradient of every saturated pixel. The identity loss passes its own bf16 copy of the VAE with gradient
    checkpointing on (`grad_vae`), which keeps a face crop's forward + backward around 1 GB."""
    return _decode(vae, latents).float()


def bf16_copy(vae):
    """A frozen bf16 copy of a VAE, built on the meta device and filled with bf16 casts of the weights one tensor at
    a time (a deepcopy first made a second fp32 VAE on the GPU, about 0.5 GB more for a moment). Buffers keep their
    dtype: the Flux2 VAE's latent statistics (bn.running_mean / running_var) stay fp32."""
    import copy

    with torch.device("meta"):
        out = type(vae).from_config(vae.config)
    params = {k for k, _ in vae.named_parameters()}
    state = {k: (v.to(torch.bfloat16) if k in params and v.is_floating_point() else v)
             for k, v in vae.state_dict().items()}
    out.load_state_dict(state, strict=True, assign=True)
    if any(t.device.type == "meta" for t in (*out.parameters(), *out.buffers())):  # a buffer outside the state dict
        out = copy.deepcopy(vae).to(torch.bfloat16)
    return out.eval().requires_grad_(False)


def grad_vae(vae):
    """A bf16 copy of the VAE with gradient checkpointing, for decode_latents_grad (the shared fp32 VAE that
    samples and probes use stays untouched)."""
    out = bf16_copy(vae)
    out.enable_gradient_checkpointing()
    return out


def _te_input_device(text_encoder) -> torch.device:
    emb = getattr(text_encoder, "embed_tokens", None)
    if emb is not None and hasattr(emb, "weight") and emb.weight.device.type != "meta":
        return emb.weight.device
    return next(text_encoder.parameters()).device


@torch.no_grad()
def encode_prompt(text_encoder, tokenizer, prompt: str, max_length: int = 4096) -> torch.Tensor:
    """Returns last_hidden_state [S, 4096] (bf16) for one prompt, using the chat template."""
    formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
    inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=max_length)
    device = _te_input_device(text_encoder)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = text_encoder(**inputs, use_cache=False, return_dict=True)
    return out.last_hidden_state[0].to(torch.bfloat16).cpu()


def get_tokenizer(cfg: TrainConfig, model_dir: Path | None):
    from transformers import AutoTokenizer

    if _use_file(cfg, "text_encoder_file"):
        tokenizer = AutoTokenizer.from_pretrained(str(sf.resolve_tokenizer(cfg.tokenizer_path, cfg.models_dir)))
    else:
        tokenizer = AutoTokenizer.from_pretrained(str(Path(model_dir) / "tokenizer"))
    check_tokenizer(tokenizer)
    return tokenizer


def prompt_tokens(tokenizer, prompt: str) -> tuple[str, list, int, int]:
    """encode_prompt's text (the chat template) with its token offsets; the cache keeps every token."""
    formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
    offsets = tokenizer(formatted, return_offsets_mapping=True, truncation=True, max_length=4096)["offset_mapping"]
    return formatted, [tuple(o) for o in offsets], 0, len(offsets)


def pad_embeds(rows: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    length = max(r.shape[0] for r in rows)
    embeds = rows[0].new_zeros(len(rows), length, rows[0].shape[-1])
    mask = torch.zeros(len(rows), length, dtype=torch.bool)
    for i, r in enumerate(rows):
        embeds[i, : r.shape[0]] = r
        mask[i, : r.shape[0]] = True
    return embeds, mask
