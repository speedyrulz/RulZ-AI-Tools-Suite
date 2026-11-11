"""Locate and load PAN 2's components (IFM/PAN2's diffusers folder or single files); encode and decode helpers.

HF/diffusers model directory layout (IFM/PAN2, about 81 GB):
    transformer/   config.json + 14 diffusion_pytorch_model-*.safetensors shards (bf16, some layers fp32)
    text_encoder/  Qwen3.5-9B's language model (transformers Qwen3_5TextModel), sharded
    tokenizer/
    vae/           AutoencoderKLWan (Wan 2.2, 48 channels, 16x)

Single files (model_source "files"): the files rulztrainer/../scripts/pan2/convert_pan2.py makes from that folder
(PAN2_int8_convrot*.safetensors, PAN2_nvfp4.safetensors, PAN2_qwen3_5_*.safetensors, PAN2_vae_fp16.safetensors), the
ones the ComfyUI build with PAN 2 support loads. Nobody publishes them, so none is downloaded: set all three.

Stills are one-frame videos, as IFM's trainer feeds images (docs/data.md: "encode images with the same Wan 2.2 VAE
as one-frame videos, same normalization"; "images always train as t2v"): latents [B, 48, H/16, W/16], normalized with
the VAE config's per-channel mean and std.

Text: the caption goes verbatim into the user turn of the model's chat template and Qwen3.5-9B reads the whole
prompt; the features are its hidden state two layers before the last, without the final norm, from the first token
after the user-turn header on (the caption, the end of the turn and the opened assistant turn with its think block),
at most 1536 tokens. That is IFM's QwenTextEncoder and ComfyUI's PAN 2 text encoder."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

from ..config import TrainConfig
from ..downloads import component_ready
from ..moss import loading as moss_loading
from ..moss import single_file as msf
from . import single_file as sf
from .model import MAX_TEXT_TOKENS, PAN2Transformer

log = logging.getLogger("rulztrainer")

REPO = "IFM/PAN2"
SYSTEM_PROMPT = "You are a helpful assistant. Generate the video with the following instructions."
USER_HEADER = "<|im_start|>user\n"
TEMPLATE = ("<|im_start|>system\n" + SYSTEM_PROMPT + "<|im_end|>\n" + USER_HEADER
            + "{}<|im_end|>\n<|im_start|>assistant\n<think>\n")
NEGATIVE_FILE = Path(__file__).with_name("negative_prompt.json")
TOKENIZER_MIN_VOCAB = msf.MOSS_TOKENIZER_MIN_VOCAB


def default_negative(seconds: float = 0.0) -> str:
    """The structured negative prompt PAN 2 was released with (IFM's inference/prompts/negative_prompt.json), every
    time range set to the clip's length: a still is 0 seconds."""
    span = f"0 to {seconds:g}"

    def fill(node):
        if isinstance(node, dict):
            return {k: span if k == "time_range_sec" else fill(v) for k, v in node.items()}
        if isinstance(node, list):
            return [fill(v) for v in node]
        return node

    return json.dumps(fill(json.loads(NEGATIVE_FILE.read_text(encoding="utf-8"))), ensure_ascii=False,
                      separators=(",", ":"))


# ---------------------------------------------------------------- locating


def resolve_model_dir(model_path: str, models_dir: str = "models") -> Path:
    p = Path(model_path).expanduser()
    if p.is_dir():
        return p.resolve()
    local = Path(models_dir) / model_path.replace("/", "__")
    if component_ready(local / "transformer") and component_ready(local / "text_encoder"):
        return local.resolve()  # (an interrupted download is resumed below)
    from huggingface_hub import snapshot_download

    log.info("Downloading %s to %s (about 81 GB: transformer 63, text encoder 15, VAE 2.7)...", model_path, local)
    snapshot_download(model_path, local_dir=str(local),
                      allow_patterns=["transformer/*", "text_encoder/*", "tokenizer/*", "vae/*", "model_index.json"])
    return local.resolve()


def _use_file(cfg: TrainConfig, field: str) -> bool:
    return cfg.model_source == "files" and bool(getattr(cfg, field))


def _check_file(cfg: TrainConfig, name: str) -> None:
    p = getattr(cfg, name)
    if not p or not Path(p).is_file():
        raise FileNotFoundError(f"{name} is not set or does not exist: {p!r}")


def resolve_for_config(cfg: TrainConfig) -> Path | None:
    """The diffusers folder, when any component comes from it."""
    if cfg.model_source == "files" and all(getattr(cfg, f) for f in ("transformer_file", "text_encoder_file", "vae_file")):
        return None
    return resolve_model_dir(cfg.model_path or REPO, cfg.models_dir)


def _need_dir(model_dir: Path | None, what: str) -> Path:
    if model_dir is None:
        raise FileNotFoundError(f"PAN 2: no {what} file is set and no model folder was resolved")
    return Path(model_dir)


# ---------------------------------------------------------------- transformer


def quant_norms(cfg: TrainConfig) -> bool:
    """Quantize the blocks' fp32 norm1.linear layers along with the rest when a quantization is asked for (12 GB of
    fp32 otherwise: the layers PAN2_int8_convrot.safetensors keeps and PAN2_int8_convrot_full.safetensors does not)."""
    return bool(getattr(cfg, "pan2_quant_norms", True))


def load_transformer(model_dir: Path, device, quant: str = "none", quant_devices=None,
                     norms: bool = True) -> PAN2Transformer:
    return sf.load_transformer_dir(Path(model_dir) / "transformer", device, quant, quant_devices=quant_devices,
                                   quant_norms=norms)


def get_transformer(cfg: TrainConfig, model_dir: Path | None, device: str | None = None) -> PAN2Transformer:
    use_file = _use_file(cfg, "transformer_file")
    if use_file:
        _check_file(cfg, "transformer_file")
    gpus = None
    if cfg.multi_gpu == "model_split" and device is None:
        from ..multigpu import parse_devices

        device, gpus = "cpu", parse_devices(cfg.devices, cfg.device)  # split right after loading: staged on the CPU
    device = device or cfg.device
    if gpus is None and torch.device(device).type == "cpu" and torch.device(cfg.device).type == "cuda":
        # staged in RAM (block swap): quantize on the training GPU, shard by shard, and bring the result back. On the
        # CPU the 29B weights were not done after 4.7 minutes on 24 cores (NVFP4, measured)
        gpus = [cfg.device]
    if use_file:
        return sf.load_transformer_file(cfg.transformer_file, device, cfg.transformer_quant, quant_devices=gpus,
                                        quant_norms=quant_norms(cfg))
    return load_transformer(_need_dir(model_dir, "diffusion model"), device, cfg.transformer_quant, gpus, quant_norms(cfg))


def load_any_transformer(path: str, device, quant: str = "auto", models_dir: str = "models") -> PAN2Transformer:
    """A transformer from a single .safetensors file, a diffusers directory, or a HF repo id."""
    p = Path(path).expanduser()
    if p.is_file():
        return sf.load_transformer_file(p, device, quant)
    if p.is_dir() and (p / "config.json").is_file() and not (p / "transformer").is_dir():
        return sf.load_transformer_dir(p, device, quant)
    return load_transformer(p if p.is_dir() else resolve_model_dir(path, models_dir), device, quant)


# ---------------------------------------------------------------- VAE


def load_vae(model_dir: Path, device):
    from safetensors.torch import load_file

    vdir = Path(model_dir) / "vae"
    config = json.loads((vdir / "config.json").read_text(encoding="utf-8"))
    return sf.vae_from_state(load_file(str(vdir / "diffusion_pytorch_model.safetensors")), config, device, str(vdir))


def get_vae(cfg: TrainConfig, model_dir: Path | None):
    if _use_file(cfg, "vae_file"):
        _check_file(cfg, "vae_file")
        return sf.load_vae_file(cfg.vae_file, cfg.device)
    return load_vae(_need_dir(model_dir, "VAE"), cfg.device)


def _stats(vae, like: torch.Tensor):
    mean = torch.tensor(vae.config.latents_mean, device=like.device, dtype=torch.float32).view(1, -1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=like.device, dtype=torch.float32).view(1, -1, 1, 1)
    return mean, std


def _vae_dtype(vae) -> torch.dtype:
    return next(vae.parameters()).dtype


@torch.no_grad()
def encode_images(vae, images: torch.Tensor) -> torch.Tensor:
    """images: [B,3,H,W] in [-1,1] (H, W multiples of 32). Returns normalized latents [B, 48, H/16, W/16]: each image
    a one-frame video, its posterior mode."""
    x = images.to(vae.device, _vae_dtype(vae)).unsqueeze(2)  # [B, 3, 1, H, W]
    z = vae.encode(x).latent_dist.mode()[:, :, 0].float()
    mean, std = _stats(vae, z)
    return (z - mean) / std


def _decode(vae, latents: torch.Tensor) -> torch.Tensor:
    z = latents.to(vae.device, torch.float32)
    mean, std = _stats(vae, z)
    z = (z * std + mean).unsqueeze(2).to(_vae_dtype(vae))
    return vae.decode(z, return_dict=False)[0][:, :, 0].float()


@torch.no_grad()
def decode_latents(vae, latents: torch.Tensor) -> torch.Tensor:
    """Inverse of encode_images. Returns images [B,3,H,W] in [-1,1]."""
    return _decode(vae, latents).clamp(-1, 1)


def decode_latents_grad(vae, latents: torch.Tensor) -> torch.Tensor:
    """decode_latents with gradients to `latents` (identity loss on a face crop), not clamped."""
    return _decode(vae, latents)


def grad_vae(vae):
    """The VAE decode_latents_grad runs on: the frozen fp32 VAE itself (AutoencoderKLWan has no gradient
    checkpointing)."""
    return vae


# ---------------------------------------------------------------- text encoder


def check_tokenizer(tokenizer) -> None:
    if len(tokenizer) < TOKENIZER_MIN_VOCAB:
        raise ValueError(f"The tokenizer has {len(tokenizer)} tokens: it is not PAN 2's Qwen3.5 tokenizer (248k). "
                         "Point tokenizer_path at PAN 2's tokenizer folder, or leave it empty to download it")


def resolve_tokenizer(tokenizer_path: str, models_dir: str = "models") -> Path:
    """A local Qwen3.5 tokenizer folder: tokenizer_path, else PAN 2's (found under models_dir or downloaded, a few
    MB), else MOSS-Image's, the same Qwen3.5-9B vocabulary (the chat template is written out here, not read from
    the tokenizer)."""
    if tokenizer_path:
        return msf.resolve_tokenizer(tokenizer_path, models_dir)
    for pattern in ("IFM__PAN2*/tokenizer/tokenizer_config.json", "tokenizer__PAN2/tokenizer/tokenizer_config.json"):
        for cand in sorted(Path(models_dir).glob(pattern)):
            return cand.parent
    local = Path(models_dir) / "tokenizer__PAN2"
    try:
        from huggingface_hub import snapshot_download

        log.info("Downloading PAN 2's tokenizer to %s", local)
        snapshot_download(REPO, local_dir=str(local), allow_patterns=["tokenizer/*"])
        if (local / "tokenizer" / "tokenizer_config.json").is_file():
            return local / "tokenizer"
    except Exception as e:  # noqa: BLE001  (offline, or the repository is not reachable)
        log.warning("PAN 2's tokenizer could not be downloaded (%s): using MOSS-Image's Qwen3.5 tokenizer", e)
    return msf.resolve_tokenizer("", models_dir)


def _tokenizer_from(cfg: TrainConfig, model_dir: Path | None):
    from transformers import AutoTokenizer

    if _use_file(cfg, "text_encoder_file") or model_dir is None or not (Path(model_dir) / "tokenizer").is_dir():
        tokenizer = AutoTokenizer.from_pretrained(str(resolve_tokenizer(cfg.tokenizer_path, cfg.models_dir)))
    else:
        tokenizer = AutoTokenizer.from_pretrained(str(Path(model_dir) / "tokenizer"))
    check_tokenizer(tokenizer)
    return tokenizer


def get_tokenizer(cfg: TrainConfig, model_dir: Path | None):
    return _tokenizer_from(cfg, model_dir)


def load_text_encoder(model_dir: Path, device: str = "auto", quant: str = "auto"):
    state = moss_loading._read_sharded(Path(model_dir) / "text_encoder", "model")
    model = sf.build_text_model(state, "none" if quant == "auto" else quant, "cpu",
                                work_device=msf.quant_target(device).get("work_device"))
    del state
    return msf.place_text_encoder(model, device)


def get_text_encoder(cfg: TrainConfig, model_dir: Path | None):
    tokenizer = _tokenizer_from(cfg, model_dir)
    if _use_file(cfg, "text_encoder_file"):
        _check_file(cfg, "text_encoder_file")
        return sf.load_text_encoder_file(cfg.text_encoder_file, cfg.text_encoder_device, cfg.text_encoder_quant), tokenizer
    return load_text_encoder(_need_dir(model_dir, "text encoder"), cfg.text_encoder_device, cfg.text_encoder_quant), tokenizer


def structured(prompt: str, seconds: float = 0.0) -> str:
    """A plain caption inside PAN 2's structured ("world state") caption format; a caption that is already JSON, and
    the empty prompt, as they are.

    PAN 2 was trained on structured captions alone. Measured on the released weights (one RTX 5090, NVFP4, 50 steps
    at CFG 2.5): a plain sentence gives an unreadable picture with the structured negative, with an empty negative
    and without guidance; the same sentence under `init_frame_visual_description` alone is no better, with a style,
    one subject and the quality block it is recognisable but torn, and in this skeleton (those plus one event that
    repeats the description under a static camera) it is a clean picture."""
    text = (prompt or "").strip()
    if not text or text.startswith("{"):
        return text
    span = f"0 to {seconds:g}"
    return json.dumps({
        "video_style": "Photorealistic, sharp detail", "lighting": "", "perspective_view": "",
        "subjects_lifecycle": [{"subject_id": "<|Subject1|>", "display_name": "subject", "subject_attribute": ""}],
        "init_frame_visual_description": text,
        "events": [{"event_index": 0, "time_range_sec": span, "event_description": text,
                    "camera_motion": [{"time_range_sec": span, "camera_motion": ["static"]}],
                    "contained_actions": []}],
        "quality": {"video_quality": "High clarity", "text_logos_overlays": None, "artifacts": None},
    }, ensure_ascii=False, separators=(",", ":"))


def format_prompt(prompt: str) -> str:
    """The templated prompt Qwen reads: the caption in the structured format (`structured`), in the reference's chat
    template. The chat template trims the user turn, so an empty prompt (caption dropout, the reference's single
    space) is an empty turn."""
    return TEMPLATE.format(structured(prompt))


def _find(ids: list[int], sub: list[int]) -> int:
    for pos in range(len(ids) - len(sub) + 1):
        if ids[pos:pos + len(sub)] == sub:
            return pos
    raise ValueError(f"The user-turn header {USER_HEADER!r} was not found in the templated prompt")


def tokenize(tokenizer, prompt: str, max_length: int = MAX_TEXT_TOKENS) -> tuple[torch.Tensor, int]:
    """(token ids of the templated prompt, cut so that at most max_length follow the user-turn header; the index of
    the first of those)."""
    ids = tokenizer(format_prompt(prompt), add_special_tokens=False)["input_ids"]
    header = tokenizer(USER_HEADER, add_special_tokens=False)["input_ids"]
    start = _find(ids, header) + len(header)
    keep = max(1, min(int(max_length), MAX_TEXT_TOKENS))
    return torch.tensor(ids[: start + keep], dtype=torch.long), start


@torch.no_grad()
def encode_prompts(text_encoder, tokenizer, prompts: list[str], max_length: int = MAX_TEXT_TOKENS) -> list[torch.Tensor]:
    """Text features [S_i, 4096] (bf16, CPU) for each prompt, in one forward (the encoder is causal: a row's real
    tokens never see the padding after them)."""
    rows, starts = zip(*(tokenize(tokenizer, p, max_length) for p in prompts))
    ids, mask = moss_loading.right_pad(list(rows))
    device = moss_loading._te_input_device(text_encoder)
    out = text_encoder(input_ids=ids.to(device), attention_mask=mask.to(device), use_cache=False, return_dict=True)
    hidden = out.last_hidden_state.to(torch.bfloat16).cpu()
    return [hidden[i, s: r.numel()].clone() for i, (r, s) in enumerate(zip(rows, starts))]


def encode_prompt(text_encoder, tokenizer, prompt: str, max_length: int = MAX_TEXT_TOKENS) -> torch.Tensor:
    return encode_prompts(text_encoder, tokenizer, [prompt], max_length)[0]


def prompt_tokens(tokenizer, prompt: str) -> tuple[str, list, int, int]:
    """encode_prompt's text with its token offsets, the leading tokens the cache drops (the system turn and the user
    header) and the cached length."""
    text = format_prompt(prompt)
    offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    rows, start = tokenize(tokenizer, prompt)
    return text, [tuple(o) for o in offsets[: rows.numel()]], start, rows.numel() - start


pad_embeds = moss_loading.pad_embeds
