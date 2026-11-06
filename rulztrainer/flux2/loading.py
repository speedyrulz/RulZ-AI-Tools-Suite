"""Load FLUX.2 Klein 9B components from a Hugging Face repo or folder and encode prompts; single files (the default)
go through flux2.single_file.

    black-forest-labs/FLUX.2-klein-base-9B   the model to train on: flux-2-klein-base-9b.safetensors (BFL's layout, which
                                             the port keeps), text_encoder/ (Qwen3-8B), tokenizer/, vae/ (Flux2 VAE),
                                             transformer/ (the same weights in diffusers' layout: not downloaded)
    black-forest-labs/FLUX.2-klein-9B        the 4-step distilled model, for samples
    black-forest-labs/FLUX.2-klein-*-fp8     the same two as fp8 files (8.9 GB)

The repos are gated (FLUX Non-Commercial License, approved automatically). Comfy-Org/Flux-2-Klein-9B holds the text
encoder and VAE as single files and is not.

Text features, where ComfyUI, diffusers and ai-toolkit agree: the prompt goes through Qwen3's chat template as one user
turn, thinking off ("<|im_start|>user\\n{prompt}<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n"), is
right-padded with the pad token to 512 tokens (cut at 512) and runs with the padding masked inside Qwen3. Hidden layers
9, 18, 27 are stacked per token: [512, 3 * 4096 = 12288]. Klein attends to the padding rows, so the cache keeps all 512.
Fizgig also runs these layers through Qwen3's final RMSNorm; ComfyUI (layer_norm_hidden_state=False), diffusers and
ai-toolkit do not, and ComfyUI is where the LoRAs are used, so neither does this.
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch

from .. import quant as Q
from ..config import TrainConfig
from ..fibo.loading import place_text_encoder
from ..flux3.loading import _gpu_gb, gated_error, is_gated_error, pad_embeds  # noqa: F401 (pad_embeds: the same batch)
from ..moss import loading as moss_loading
from ..moss import single_file as sf
from . import single_file as fsf
from .model import Flux2KleinTransformer

log = logging.getLogger("rulztrainer")

TRANSFORMER_CLASSES = ("Flux2Transformer2DModel",)
# quantize the blocks; the projections, timestep MLP, modulations and final layer stay in bf16
QUANT_FILTER = lambda name: name.startswith(("double_blocks.", "single_blocks."))  # noqa: E731
# "auto" by the GPU memory the model will live in (all GPUs of a model split). Fizgig measured a rank-32 run on the
# full model (a 5090, 3 Oct 2026, gradient checkpointing, adamw8bit): int8 12.6 GB at 0.25 MP, 17.1 GB at 1 MP; 4-bit
# 7.9 / 9.9 GB. Its bf16 is 17 GB of weights alone. This trainer's nvfp4 is not the same code as Fizgig's NF4.
AUTO_BF16_FROM_GB, AUTO_INT8_FROM_GB = 30, 18
AUTO_TEXT_FULL_FROM_GB = 19.5  # Qwen3-8B in bf16 is 15 GB: Fizgig halves it below this much VRAM

PROMPT_PREFIX = "<|im_start|>user\n"
PROMPT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
SELECT_LAYERS = (9, 18, 27)
MAX_TOKENS = 512
ROOT_FILES = "flux-2-klein*.safetensors"
HF_PATTERNS = ["*.json", ROOT_FILES, "text_encoder/*", "tokenizer/*", "vae/*", "LICENSE.md"]
SAMPLE_PATTERNS = ["*.json", ROOT_FILES, "LICENSE.md"]


def resolve_quant(quant: str, devices) -> str:
    """`auto`: bf16 with 30 GB+, int8 from 18 GB, else nvfp4, counting the memory of every GPU the model is split over.
    A model loaded on the CPU (a sampling model) is sized for the smallest GPU, where it will run."""
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
        log.info("FLUX.2 Klein transformer: auto quantization -> %s for %.0f GB (%s)", fmt, total, where)
    return fmt


def resolve_text_quant(quant: str) -> str:
    """`auto`: bf16 on a GPU with room for it, else int8 ConvRot (about half)."""
    if quant != "auto":
        return quant
    if not torch.cuda.is_available():
        return "none"
    largest = max(_gpu_gb(f"cuda:{i}") for i in range(torch.cuda.device_count()))
    return "none" if largest >= AUTO_TEXT_FULL_FROM_GB else "int8_convrot"


def is_flux2_dir(model_dir: Path) -> bool:
    return any(model_dir.glob(ROOT_FILES)) or (model_dir / "transformer" / "config.json").is_file()


def root_file(model_dir: Path) -> Path | None:
    """The repo's BFL-layout file (which the port reads as it is), the Base one first."""
    files = sorted(model_dir.glob(ROOT_FILES), key=lambda p: ("base" not in p.name, p.name))
    return files[0] if files else None


def resolve_model_dir(model_path: str, models_dir: str = "models", patterns: list[str] | None = None) -> Path:
    """A local folder for a repo id or path, downloading what the trainer reads (about 32 GB; not the diffusers copy of
    the transformer). patterns: fewer (a sampling model needs its root file alone, 17 GB)."""
    p = Path(model_path).expanduser()
    if p.is_dir():
        return p.resolve()
    local = Path(models_dir) / model_path.replace("/", "__")
    want_rest = patterns is None
    if root_file(local) is not None and (not want_rest or (local / "text_encoder" / "config.json").is_file()):
        return local.resolve()
    from huggingface_hub import snapshot_download

    log.info("Downloading %s to %s (%s)...", model_path, local, "the diffusion model, about 17 GB" if patterns else
             "diffusion model, text encoder and VAE, about 32 GB")
    try:
        snapshot_download(model_path, local_dir=str(local), allow_patterns=patterns or HF_PATTERNS)
    except Exception as e:  # noqa: BLE001
        if is_gated_error(e):
            raise gated_error(model_path, e) from e
        raise
    return local.resolve()


def load_transformer(model_dir: Path, device, quant: str = "auto", dtype=torch.bfloat16,
                     quant_devices=None) -> Flux2KleinTransformer:
    """From the folder's root file, else its diffusers transformer/. quant_devices: the GPUs the model will run on."""
    root = root_file(model_dir)
    if root is not None:
        return fsf.load_transformer_file(root, device, quant, dtype, quant_devices)
    state = moss_loading._read_sharded(model_dir / "transformer", "diffusion_pytorch_model")
    state = fsf.to_model_layout({k: v.to(dtype) for k, v in state.items()})
    config = fsf.transformer_config_from_state(state)
    quant = resolve_quant(quant, quant_devices or device)
    with torch.device("meta"):
        model = Flux2KleinTransformer.from_config(config)
    sf.install_state(model, state, quant, dtype, quant_filter=QUANT_FILTER, **sf.quant_target(device, quant_devices))
    del state
    model = model.to(device).eval().requires_grad_(False)
    model.is_distilled = fsf.is_distilled_name(model_dir.name)
    log.info("Loaded FLUX.2 Klein transformer from %s: %d double and %d single blocks, quant=%s", model_dir.name,
             config["depth"], config["depth_single_blocks"], quant)
    return model


def load_text_encoder(model_dir: Path, device: str = "auto", quant: str = "auto"):
    """Qwen3-8B's text model (no LM head) + its tokenizer from text_encoder/ and tokenizer/."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir / "tokenizer"))
    fmt = resolve_text_quant(quant)
    state = moss_loading._read_sharded(model_dir / "text_encoder", "model")
    model = fsf.build_text_model(state, fmt, sf.quant_target(device).get("work_device"))
    if fmt != "none":
        n = sum(1 for m in model.modules() if isinstance(m, Q.QuantLinear))
        log.info("Text encoder: %d Linear layers quantized to %s", n, fmt)
    return place_text_encoder(model, device), tokenizer


def _input_device(model) -> torch.device:
    return model.get_input_embeddings().weight.device


def chat_text(prompt: str) -> str:
    """Qwen3's chat template for one user prompt with the generation prompt, thinking off."""
    return PROMPT_PREFIX + (prompt or "") + PROMPT_SUFFIX


def tokenize(tokenizer, prompt: str, max_length: int = MAX_TOKENS):
    """(input_ids [1, 512], attention_mask [1, 512]): cut at max_length (512 at most), always padded to 512."""
    tok = getattr(tokenizer, "tokenizer", tokenizer)
    ids = tok(chat_text(prompt), truncation=True, max_length=min(int(max_length), MAX_TOKENS)).input_ids
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    input_ids = torch.full((1, MAX_TOKENS), int(pad), dtype=torch.long)
    input_ids[0, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    attention_mask = torch.zeros(1, MAX_TOKENS, dtype=torch.long)
    attention_mask[0, : len(ids)] = 1
    return input_ids, attention_mask


@torch.no_grad()
def encode_prompts(text_encoder, tokenizer, prompts: list[str], max_length: int = MAX_TOKENS) -> list[torch.Tensor]:
    """Text features for each prompt: [512, 12288] bf16 on the CPU, in one forward."""
    parts = [tokenize(tokenizer, p, max_length) for p in prompts]
    device = _input_device(text_encoder)
    out = text_encoder(input_ids=torch.cat([i for i, _ in parts]).to(device),
                       attention_mask=torch.cat([m for _, m in parts]).to(device), output_hidden_states=True,
                       use_cache=False)
    hidden = torch.stack([out.hidden_states[i].to(torch.bfloat16).cpu() for i in SELECT_LAYERS], dim=2)  # [B, 512, 3, D]
    return [row.reshape(row.shape[0], -1).clone() for row in hidden]


def encode_prompt(text_encoder, tokenizer, prompt: str, max_length: int = MAX_TOKENS) -> torch.Tensor:
    return encode_prompts(text_encoder, tokenizer, [prompt], max_length)[0]


def get_tokenizer(cfg: TrainConfig, model_dir: Path | None):
    from transformers import AutoTokenizer

    if moss_loading._use_file(cfg, "text_encoder_file") or model_dir is None:
        return AutoTokenizer.from_pretrained(str(fsf.resolve_tokenizer(cfg.tokenizer_path, cfg.models_dir)))
    return AutoTokenizer.from_pretrained(str(model_dir / "tokenizer"))


def prompt_tokens(tokenizer, prompt: str) -> tuple[str, list, int, int]:
    """encode_prompt's text (the whole chat template) with its token offsets; the cache keeps all 512 rows."""
    tok = getattr(tokenizer, "tokenizer", tokenizer)
    text = chat_text(prompt)
    offsets = tok(text, return_offsets_mapping=True)["offset_mapping"]
    return text, [tuple(o) for o in offsets], 0, MAX_TOKENS


# ---------------------------------------------------------------- config-driven entry points


def _need_dir(cfg: TrainConfig, model_dir: Path | None, what: str) -> Path:
    if model_dir is None:
        raise ValueError(f"No {what} file is set and no Hugging Face model to take it from: set the file (model source "
                         "'files') or the black-forest-labs/FLUX.2-klein-base-9B repo id / a folder")
    return model_dir


def resolve_for_config(cfg: TrainConfig) -> Path | None:
    """Download / locate the HF model dir when any component comes from it (single files cover the rest)."""
    if cfg.model_source == "files" and all(getattr(cfg, f) for f in ("transformer_file", "text_encoder_file", "vae_file")):
        return None
    return resolve_model_dir(cfg.model_path, cfg.models_dir)


def _trainable(model: Flux2KleinTransformer, source: str) -> Flux2KleinTransformer:
    """Base and the distilled file have the same keys, so this goes by the name and only warns."""
    if model.is_distilled:
        log.warning("%s is named as the 4-step distilled FLUX.2 Klein. Train on the Base model "
                    "(flux-2-klein-base-9b) and set the distilled one as the sampling model; if this file is Base, "
                    "put \"base\" in its name.", source)
    return model


def get_transformer(cfg: TrainConfig, model_dir: Path | None, device: str | None = None) -> Flux2KleinTransformer:
    use_file = moss_loading._use_file(cfg, "transformer_file")
    if use_file:
        moss_loading._check_files(cfg, "transformer_file")
        source = Path(cfg.transformer_file).name
    else:
        model_dir = _need_dir(cfg, model_dir, "diffusion model")
        source = model_dir.name
    if cfg.multi_gpu == "model_split" and device is None:
        # the trainer splits it right after loading: stay on the CPU and size "auto" for all the split GPUs
        from ..multigpu import parse_devices

        gpus = parse_devices(cfg.devices, cfg.device)
        if use_file:
            return _trainable(fsf.load_transformer_file(cfg.transformer_file, "cpu", cfg.transformer_quant,
                                                        quant_devices=gpus), source)
        return _trainable(load_transformer(model_dir, "cpu", cfg.transformer_quant, quant_devices=gpus), source)
    if use_file:
        return _trainable(fsf.load_transformer_file(cfg.transformer_file, device or cfg.device,
                                                    cfg.transformer_quant), source)
    return _trainable(load_transformer(model_dir, device or cfg.device, cfg.transformer_quant), source)


def get_vae(cfg: TrainConfig, model_dir: Path | None):
    if moss_loading._use_file(cfg, "vae_file"):  # the Flux2 VAE file MOSS-Image and FIBO 2.0 load
        moss_loading._check_files(cfg, "vae_file")
        return sf.load_vae_file(cfg.vae_file, cfg.device)
    return moss_loading.load_vae(_need_dir(cfg, model_dir, "VAE"), cfg.device)


def get_text_encoder(cfg: TrainConfig, model_dir: Path | None):
    if moss_loading._use_file(cfg, "text_encoder_file"):
        from transformers import AutoTokenizer

        moss_loading._check_files(cfg, "text_encoder_file")
        tokenizer = AutoTokenizer.from_pretrained(str(fsf.resolve_tokenizer(cfg.tokenizer_path, cfg.models_dir)))
        model = fsf.load_text_encoder_file(cfg.text_encoder_file, cfg.text_encoder_device, cfg.text_encoder_quant)
        return model, tokenizer
    return load_text_encoder(_need_dir(cfg, model_dir, "text encoder"), cfg.text_encoder_device,
                             cfg.text_encoder_quant)


def load_any_transformer(path: str, device, quant: str = "auto", models_dir: str = "models") -> Flux2KleinTransformer:
    """Load a transformer (Base or the distilled sampling model) from a single .safetensors file, a folder, or a HF
    repo id (its root file alone is downloaded)."""
    p = Path(path).expanduser()
    if p.is_file():
        return fsf.load_transformer_file(p, device, quant)
    if p.is_dir():
        return load_transformer(p, device, quant)
    return load_transformer(resolve_model_dir(path, models_dir, SAMPLE_PATTERNS), device, quant)
