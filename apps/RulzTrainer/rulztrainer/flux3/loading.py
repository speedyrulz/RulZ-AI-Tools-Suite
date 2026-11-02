"""Load FLUX 3 Klein components from the Hugging Face / diffusers layout (black-forest-labs/flux-3-image-klein-base,
the one to train on, and flux-3-image-klein, the 16-step distilled model for samples) and encode; single files
(black-forest-labs/flux-3-image-klein-comfy, the default) go through flux3.single_file.

    modular_model_index.json  Flux3ModularPipeline (Base) / Flux3AutoBlocks; config.json BFL's recipe (steps,
                              guidance, alpha = the scheduler shift, image_rope_scale)
    transformer/              Flux3Transformer2DModel, 2 shards (diffusers names, renamed to BFL's on load)
    text_encoder/             Qwen3-VL-4B-Instruct (Qwen3VLForConditionalGeneration) with its tokenizer files
    vae/                      AutoencoderKLFlux2 (the MOSS-Image / FIBO 2.0 VAE and latent normalization)
    model.safetensors, ae.safetensors, variants/   BFL's single files: not downloaded here (the folders hold the
                              same weights; the -comfy repo has the quantized files)

All Black Forest Labs repos are gated (FLUX Kommunity License, approved automatically): accept it once on the model
page with the Hugging Face account whose token the trainer uses.

Text features: diffusers' Flux3TextEncoderStep. The prompt goes through Qwen3-VL's chat template as one user turn
with the generation prompt ("<|im_start|>user\\n{prompt}<|im_end|>\\n<|im_start|>assistant\\n", no system turn),
is right-padded with the pad token to a multiple of 80 tokens (at most max_length) and runs with the padding masked
inside Qwen3-VL and the padding's position ids set to 1 (the training environment's). Hidden layers 4, 8, ..., 32
are stacked per token along the channels: [S, 8 * 2560 = 20480]. FLUX 3 attends to the padding rows (they are part of
the recipe), so the cache keeps the whole padded window; only a batch's extra padding is masked in the DiT. There is
no pooled embedding. The trainer does not run BFL's prompt upsampler (flux-3-image-klein-pup): captions should be
full scene captions (framing, subject, background, lighting).
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import torch

from .. import quant as Q
from ..config import TrainConfig
from ..fibo.loading import place_text_encoder
from ..moss import loading as moss_loading
from ..moss import single_file as sf
from .model import Flux3KleinTransformer

log = logging.getLogger("rulztrainer")

TRANSFORMER_CLASSES = ("Flux3Transformer2DModel",)
# quantize the per-mode and joint blocks; the projections, timestep MLP, modulations and final layer stay in bf16 (as
# the -comfy int8-convrot file does)
QUANT_FILTER = lambda name: name.startswith(("txt_mode_blocks.", "content_mode_blocks.", "single_blocks."))  # noqa: E731
# "auto" by the GPU memory the model will live in (all GPUs of a model split). Without the reference-image mode the
# port holds 4.87B parameters (the -comfy files' headers): 9.07 GiB in bf16, 4.76 int8, 2.84 nvfp4. A rank-32 LoRA
# step on the default targets at 1 MP (gradient checkpointing, AdamW) peaked at 12.2 / 7.8 / 6.0 GiB reserved on an
# RTX 5060 Ti (the real shapes with random weights, not a real-weights run). A "16 GB" card reports 15.9 GiB.
AUTO_BF16_FROM_GB, AUTO_INT8_FROM_GB = 15, 9.5

PROMPT_PREFIX = "<|im_start|>user\n"
PROMPT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
SELECT_LAYERS = (4, 8, 12, 16, 20, 24, 28, 32)  # config.json text_encoder.layers
PAD_MULTIPLE = 80  # config.json text_encoder.pad_multiple
MAX_TOKENS = 8192  # the pipeline's max_sequence_length (the trainer's max_token_length caps it)
HF_PATTERNS = ["*.json", "transformer/*", "text_encoder/*", "vae/*", "LICENSE.md", "NOTICE.md"]
GATE_HELP = ("{what} is gated on Hugging Face{license}: open https://huggingface.co/{repo}, accept the license with "
             "your Hugging Face account, and make sure the trainer uses that account's token (hf auth login, or the "
             "HF_TOKEN environment variable)")


def is_flux3_dir(model_dir: Path) -> bool:
    cfg = model_dir / "transformer" / "config.json"
    if not cfg.is_file():
        return False
    try:
        return json.loads(cfg.read_text(encoding="utf-8")).get("_class_name") in TRANSFORMER_CLASSES
    except (OSError, ValueError):
        return False


def bfl_config(model_dir: Path) -> dict:
    """The repo's root config.json (BFL's recipe: dit, text_encoder, sampler)."""
    try:
        return json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _gpu_gb(device) -> float:
    dev = torch.device(device)
    return torch.cuda.get_device_properties(dev.index or 0).total_memory / 2**30


def resolve_quant(quant: str, devices) -> str:
    """`auto`: bf16 with 15 GB+, int8 from 9.5 GB, else nvfp4, counting the memory of every GPU the model is split
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
        log.info("FLUX 3 Klein transformer: auto quantization -> %s for %.0f GB (%s)", fmt, total, where)
    return fmt


def gated_error(repo: str, err: Exception) -> Exception:
    """A PermissionError that says which license to accept where (the download's own error is only a 401 / 403)."""
    bfl = repo.lower().startswith("black-forest-labs/")
    text = GATE_HELP.format(what="Black Forest Labs' FLUX 3 Klein" if bfl else repo, repo=repo,
                            license=" (FLUX Kommunity License, approved automatically)" if bfl else "")
    return PermissionError(f"{text} ({type(err).__name__})")


def is_gated_error(err: Exception) -> bool:
    try:
        from huggingface_hub.errors import GatedRepoError
    except ImportError:  # older huggingface_hub
        from huggingface_hub.utils import GatedRepoError
    if isinstance(err, GatedRepoError):
        return True
    status = getattr(getattr(err, "response", None), "status_code", None)
    return status in (401, 403)


def resolve_model_dir(model_path: str, models_dir: str = "models", patterns: list[str] | None = None) -> Path:
    """A local diffusers folder for a repo id or path, downloading the folders the trainer reads (transformer/,
    text_encoder/, vae/ and the configs, about 20 GB; not BFL's root single files or variants/). patterns: fewer
    (a sampling model needs its transformer/ alone, 11 GB)."""
    p = Path(model_path).expanduser()
    if p.is_dir():
        return p.resolve()
    local = Path(models_dir) / model_path.replace("/", "__")
    if (local / "transformer" / "config.json").is_file():
        return local.resolve()
    from huggingface_hub import snapshot_download

    log.info("Downloading %s to %s (%s)...", model_path, local, "the transformer, about 11 GB" if patterns else
             "transformer, text encoder and VAE, about 20 GB")
    try:
        snapshot_download(model_path, local_dir=str(local), allow_patterns=patterns or HF_PATTERNS)
    except Exception as e:  # noqa: BLE001
        if is_gated_error(e):
            raise gated_error(model_path, e) from e
        raise
    return local.resolve()


def load_transformer(model_dir: Path, device, quant: str = "auto", dtype=torch.bfloat16,
                     quant_devices=None) -> Flux3KleinTransformer:
    """quant_devices: the GPUs the model will run on, for quant "auto" (default: `device`)."""
    from .single_file import norm_scales_fp32, to_model_layout

    config = json.loads((model_dir / "transformer" / "config.json").read_text(encoding="utf-8"))
    if config.get("_class_name") not in TRANSFORMER_CLASSES:
        raise ValueError(f"Not a FLUX 3 transformer: {config.get('_class_name')!r}")
    config["image_rope_scale"] = bfl_config(model_dir).get("dit", {}).get("image_rope_scale", 0.5)
    quant = resolve_quant(quant, quant_devices or device)
    with torch.device("meta"):
        model = Flux3KleinTransformer.from_config(config)
    state = moss_loading._read_sharded(model_dir / "transformer", "diffusion_pytorch_model")
    state = norm_scales_fp32(to_model_layout({k: v.to(dtype) for k, v in state.items()}))
    sf.install_state(model, state, quant, dtype, quant_filter=QUANT_FILTER, **sf.quant_target(device, quant_devices))
    del state
    model = model.to(device).eval().requires_grad_(False)
    log.info("Loaded FLUX 3 Klein transformer from %s: %d + %d per-mode and %d joint blocks, %s, quant=%s",
             model_dir.name, config["num_layers"], config["num_layers"], config["num_single_layers"],
             "distilled (flow map, 16 steps)" if model.flow_map else "base", quant)
    return model


def load_text_encoder(model_dir: Path, device: str = "auto", quant: str = "auto"):
    """Qwen3-VL-4B's language model (text only: no vision tower, no LM head) + its tokenizer from text_encoder/.
    quant int8 / int8_convrot / nvfp4 quantizes its Linear layers so the 8.9 GB encoder takes about half."""
    from transformers import AutoTokenizer

    from ..krea2.single_file import load_hf_text_model

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir / "text_encoder"))
    fmt = "none" if quant == "auto" else quant
    model = load_hf_text_model(model_dir / "text_encoder", fmt, device)
    if fmt != "none":
        n = sum(1 for m in model.modules() if isinstance(m, Q.QuantLinear))
        log.info("Text encoder: %d Linear layers quantized to %s", n, fmt)
    return place_text_encoder(model, device), tokenizer


def _input_device(model) -> torch.device:
    return model.get_input_embeddings().weight.device


def chat_text(prompt: str) -> str:
    """The chat template's text for one user prompt with the generation prompt (Qwen3-VL-4B-Instruct's template)."""
    return PROMPT_PREFIX + (prompt or "") + PROMPT_SUFFIX


def window_length(real_tokens: int, max_length: int = MAX_TOKENS) -> int:
    """The text window: the prompt's tokens rounded up to the pad multiple, at most max_length."""
    return min(math.ceil(real_tokens / PAD_MULTIPLE) * PAD_MULTIPLE, int(max_length))


def tokenize(tokenizer, prompt: str, max_length: int = MAX_TOKENS):
    """(input_ids [1, W], attention_mask [1, W], position_ids [1, W]) as the pipeline forms them."""
    tok = getattr(tokenizer, "tokenizer", tokenizer)  # a processor, or the tokenizer itself
    max_length = min(int(max_length), MAX_TOKENS)
    ids = tok(chat_text(prompt), truncation=True, max_length=max_length).input_ids
    width = window_length(len(ids), max_length)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    input_ids = torch.full((1, width), int(pad), dtype=torch.long)
    input_ids[0, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    attention_mask = torch.zeros(1, width, dtype=torch.long)
    attention_mask[0, : len(ids)] = 1
    position_ids = torch.arange(width)[None].masked_fill(attention_mask == 0, 1)
    return input_ids, attention_mask, position_ids


@torch.no_grad()
def encode_prompts(text_encoder, tokenizer, prompts: list[str], max_length: int = MAX_TOKENS) -> list[torch.Tensor]:
    """Text features for each prompt: [W_i, 20480] bf16 on the CPU, W_i its padded window (see the module docstring),
    in one forward. Each row keeps its own window; the batch pads the shorter ones further, with the same masked
    padding, which the causal encoder's earlier positions never see."""
    parts = [tokenize(tokenizer, p, max_length) for p in prompts]
    widths = [ids.shape[1] for ids, _, _ in parts]
    width = max(widths)

    def pad(t, value):
        return torch.cat([t, t.new_full((1, width - t.shape[1]), value)], dim=1)

    tok = getattr(tokenizer, "tokenizer", tokenizer)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    input_ids = torch.cat([pad(ids, int(pad_id)) for ids, _, _ in parts])
    attention_mask = torch.cat([pad(m, 0) for _, m, _ in parts])
    position_ids = torch.cat([pad(p, 1) for _, _, p in parts])
    device = _input_device(text_encoder)
    out = text_encoder(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device),
                       position_ids=position_ids.to(device), output_hidden_states=True, use_cache=False)
    hidden = torch.stack([out.hidden_states[i].to(torch.bfloat16).cpu() for i in SELECT_LAYERS], dim=2)  # [B,W,8,D]
    return [hidden[i, :w].reshape(w, -1).clone() for i, w in enumerate(widths)]


def encode_prompt(text_encoder, tokenizer, prompt: str, max_length: int = MAX_TOKENS) -> torch.Tensor:
    """Text features for one prompt: [W, 20480] bf16 on the CPU, W the padded window (see the module docstring)."""
    return encode_prompts(text_encoder, tokenizer, [prompt], max_length)[0]


def get_tokenizer(cfg: TrainConfig, model_dir: Path | None):
    from transformers import AutoTokenizer

    if moss_loading._use_file(cfg, "text_encoder_file"):
        from . import single_file as fsf

        return AutoTokenizer.from_pretrained(str(fsf.resolve_tokenizer(cfg.tokenizer_path, cfg.models_dir)))
    return AutoTokenizer.from_pretrained(str(_need_dir(cfg, model_dir, "tokenizer") / "text_encoder"))


def prompt_tokens(tokenizer, prompt: str) -> tuple[str, list, int, int]:
    """encode_prompt's text (the whole chat template) with its token offsets; the cache keeps every token and the
    padding up to the window (at the default max_token_length, 1024)."""
    tok = getattr(tokenizer, "tokenizer", tokenizer)
    text = chat_text(prompt)
    offsets = tok(text, return_offsets_mapping=True)["offset_mapping"]
    return text, [tuple(o) for o in offsets], 0, window_length(len(offsets), 1024)


def pad_embeds(rows: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """[S_i, 20480] rows -> [B, S, 20480] (right-padded with zeros) and a [B, S] bool mask of each sample's rows (its
    whole window: the model attends to the window's padding, not to the batch's)."""
    length = max(r.shape[0] for r in rows)
    embeds = rows[0].new_zeros(len(rows), length, rows[0].shape[-1])
    mask = torch.zeros(len(rows), length, dtype=torch.bool)
    for i, r in enumerate(rows):
        embeds[i, : r.shape[0]] = r
        mask[i, : r.shape[0]] = True
    return embeds, mask


# ---------------------------------------------------------------- config-driven entry points


def _need_dir(cfg: TrainConfig, model_dir: Path | None, what: str) -> Path:
    if model_dir is None:
        raise ValueError(f"No {what} file is set and no Hugging Face model to take it from: set the file (model "
                         f"source 'files') or the black-forest-labs/flux-3-image-klein-base repo id / a diffusers "
                         f"folder")
    return model_dir


def resolve_for_config(cfg: TrainConfig) -> Path | None:
    """Download / locate the HF model dir when any component comes from it (single files cover the rest)."""
    if cfg.model_source == "files" and all(getattr(cfg, f) for f in ("transformer_file", "text_encoder_file", "vae_file")):
        return None
    return resolve_model_dir(cfg.model_path, cfg.models_dir)


def _trainable(model: Flux3KleinTransformer, source: str) -> Flux3KleinTransformer:
    if model.flow_map:
        raise ValueError(f"{source} is the distilled FLUX 3 Klein (a 16-step flow map): it samples, it does not "
                         "train. Train on flux-3-image-klein-base (its LoRAs run on the distilled model too, at 16 "
                         "steps, CFG 1, LoRA strength about 0.75) and set the distilled model as the sampling model")
    return model


def get_transformer(cfg: TrainConfig, model_dir: Path | None, device: str | None = None) -> Flux3KleinTransformer:
    from . import single_file as fsf

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
    if moss_loading._use_file(cfg, "text_encoder_file"):  # the Qwen3-VL-4B file Krea 2 and FIBO 2.0 load
        from transformers import AutoTokenizer

        from . import single_file as fsf

        moss_loading._check_files(cfg, "text_encoder_file")
        tokenizer = AutoTokenizer.from_pretrained(str(fsf.resolve_tokenizer(cfg.tokenizer_path, cfg.models_dir)))
        model = fsf.load_text_encoder_file(cfg.text_encoder_file, cfg.text_encoder_device, cfg.text_encoder_quant)
        return model, tokenizer
    return load_text_encoder(_need_dir(cfg, model_dir, "text encoder"), cfg.text_encoder_device,
                             cfg.text_encoder_quant)


def load_any_transformer(path: str, device, quant: str = "auto", models_dir: str = "models") -> Flux3KleinTransformer:
    """Load a transformer (Base or the distilled sampling model) from a single .safetensors file, a diffusers
    directory, or a HF repo id."""
    p = Path(path).expanduser()
    if p.is_file():
        from . import single_file as fsf

        return fsf.load_transformer_file(p, device, quant)
    if p.is_dir() and (p / "transformer" / "config.json").is_file():
        return load_transformer(p, device, quant)
    return load_transformer(resolve_model_dir(path, models_dir, ["*.json", "transformer/*", "LICENSE.md"]), device,
                            quant)
