"""Load MiniMax-H3-Image components and encode prompts. Single ComfyUI-style files (Comfy-Org/MiniMax-H3-Image, the
default) or the Hugging Face release (MiniMaxAI/MiniMax-H3-Image):

    diffusion_models/minimax_h3_image_pruned_*   H3's transformer without the audio stream, pruned AdaLN (the [1025, 8]
                                                  curve table of Comfy-Org's H3 files), 6 output heads stacked in
                                                  final_layer.video_out; int8_convrot, bf16, or w4a8 (ComfyUI's
                                                  asym_w4a8_int8, decoded on read to the int8 ConvRot weight it stands
                                                  for: 8 bits a weight in memory)
    text_encoders/qwen3vl_32b_minimax_h3_*       MiniMax H3's own text encoder files (h3/single_file.py)
    vae/minimax_h3_image_vae_fp16                the image VAE in MiniMax's original key layout (h3image/vae.py)

    transformer/               MiniMaxAI's original layout (MiniMaxH3DiTModel, audio_enabled false, per-head interleaved
                               qkv, full AdaLN: pruned on load); its own final_layer.video_out is not used:
    pdd_fused_heads.safetensors  the 6 per-step output heads replace it, on the sigmas of pdd_config.json
    transformer_diffusers/     diffusers' MiniMaxH3ImageTransformer2DModel (pdd_proj_out.0-5), read when a folder has
                               no transformer/ (or a diffusers-saved transformer/ of that class)
    text_encoder/, tokenizer/  Qwen3-VL-32B and H3's tokenizer, the same files as MiniMaxAI/MiniMax-H3's
    vae/, vae_diffusers/       the VAE in the original layout (MiniMaxH3ImageVAE) or diffusers'
Only what the trainer reads is downloaded (transformer/, the heads, the text encoder's first 11 shards, the tokenizer
and vae/).

Text features are MiniMax H3's (diffusers' MiniMaxH3ImageTextEncoderStep for a text-to-image request: the prompt
verbatim, no chat template, no special tokens, the state after decoder layer 50), so h3/loading.py encodes them and a
caption cache made for one could serve the other.

The transformer is rulztrainer.h3.model with `audio_latents_dim=None` and its 6 PDD heads (see its docstring): one
sequence [text | image], one time for both. Training uses the head of the grid interval that holds each sigma.

Images encode as H3's (the same encoder weights: ImageNet-normalized, the posterior mode, normalized by latents_mean /
latents_std); a latent decodes straight to an image. Edit training (reference images) is not supported."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

from .. import quant as Q
from ..config import TrainConfig
from ..downloads import component_ready
from ..h3 import convert as C
from ..h3 import loading as HL
from ..h3 import single_file as hsf
from ..h3.model import H3_IMAGE_PDD_SIGMAS, MiniMaxH3Transformer
from ..moss import loading as moss_loading
from ..moss import single_file as sf

log = logging.getLogger("rulztrainer")

REPO = "MiniMaxAI/MiniMax-H3-Image"
COMFY_REPO = "Comfy-Org/MiniMax-H3-Image"
PDD_SIGMAS = H3_IMAGE_PDD_SIGMAS
PDD_FILE = "pdd_fused_heads.safetensors"
PDD_CONFIG = "pdd_config.json"
ORIGINAL_CLASSES = ("MiniMaxH3DiTModel",)  # transformer/: MiniMaxAI's own layout
DIFFUSERS_CLASSES = ("MiniMaxH3ImageTransformer2DModel",)  # transformer_diffusers/
VAE_DIFFUSERS_CLASSES = ("AutoencoderKLMiniMaxH3Image",)
VAE_FILE = "minimax_h3_image_vae_fp16.safetensors"
# what the trainer reads of the repository (not transformer_diffusers/, vae_diffusers/, the processor, the examples or
# the text encoder's last 3 shards: layers 50-63 and the vision tower)
ALLOW_PATTERNS = ["model_index.json", "modular_model_index.json", PDD_CONFIG, PDD_FILE, "transformer/*", "tokenizer/*",
                  "vae/*", "text_encoder/config.json", "text_encoder/model.safetensors.index.json",
                  *[f"text_encoder/model-{i:05d}-of-00014.safetensors" for i in range(1, 12)]]


def _json(path: Path) -> dict:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


# ---------------------------------------------------------------- transformer


def find_transformer(model_dir: Path) -> tuple[Path, str, dict] | None:
    """(folder, "original" | "diffusers", its config) of a MiniMax-H3-Image transformer under model_dir: transformer/
    first, then transformer_diffusers/. None: the folder holds none."""
    for sub in ("transformer", "transformer_diffusers"):
        tdir = Path(model_dir) / sub
        config = _json(tdir / "config.json")
        cls = config.get("_class_name")
        if cls in DIFFUSERS_CLASSES:
            return tdir, "diffusers", config
        if cls in ORIGINAL_CLASSES and (config.get("audio_enabled") is False or (Path(model_dir) / PDD_FILE).is_file()):
            return tdir, "original", config
    return None


def is_h3_image_dir(model_dir: Path) -> bool:
    return find_transformer(Path(model_dir)) is not None


def folder_sigmas(model_dir: Path) -> list[float]:
    """The heads' sigma grid of a folder: pdd_config.json, else diffusers' sample_sigmas, else the released grid."""
    sigmas = _json(Path(model_dir) / PDD_CONFIG).get("sigmas") \
        or _json(Path(model_dir) / "modular_model_index.json").get("sample_sigmas")
    return [float(s) for s in (sigmas or PDD_SIGMAS)]


def _with_fused_heads(reader, heads_file: Path):
    """An original-layout reader with transformer/'s own output head replaced by the per-step heads of
    pdd_fused_heads.safetensors, stacked as the port holds them."""
    def wrapped(want):
        for k, v in reader(want):
            if not k.startswith("final_layer.video_out."):
                yield k, v
        if want("final_layer.video_out.weight"):
            from safetensors import safe_open

            with safe_open(str(heads_file), "pt") as f:
                weight, bias = f.get_tensor("video_out.weight"), f.get_tensor("video_out.bias")
            if weight.ndim != 3 or bias.shape != weight.shape[:2]:
                raise ValueError(f"{heads_file.name}: expected video_out.weight [heads, out, hidden] and video_out.bias "
                                 f"[heads, out], got {tuple(weight.shape)} and {tuple(bias.shape)}")
            w, b = C.stack_pdd_heads(weight, bias)
            yield "final_layer.video_out.weight", w
            yield "final_layer.video_out.bias", b
    return wrapped


def _with_folded_heads(reader):
    """A diffusers-layout reader with pdd_proj_out.0..n-1 folded into the stacked final_layer.video_out."""
    def wrapped(want):
        heads = {}
        for k, v in reader(want):
            if C.PDD_HEAD.match(k):
                heads[k] = v
            else:
                yield k, v
        yield from C.fold_diffusers_pdd_heads(heads).items()
    return wrapped


def load_transformer(model_dir: Path, device, quant: str = "auto", dtype=torch.bfloat16,
                     quant_devices=None) -> MiniMaxH3Transformer:
    """The transformer of a downloaded MiniMaxAI/MiniMax-H3-Image (or a diffusers folder of it)."""
    found = find_transformer(Path(model_dir))
    if found is None:
        raise ValueError(f"{model_dir} holds no MiniMax-H3-Image transformer (transformer/ or transformer_diffusers/)")
    tdir, layout, config = found
    sigmas = folder_sigmas(Path(model_dir))
    if layout == "diffusers":
        reader, size = HL._shard_reader(tdir, "diffusion_pytorch_model", dtype, C.diffusers_to_port)
        reader = _with_folded_heads(reader)
        overrides = C.diffusers_config_to_port(config, audio=False)
    else:
        heads_file = Path(model_dir) / PDD_FILE
        if not heads_file.is_file():
            raise FileNotFoundError(f"{heads_file} is missing: MiniMax-H3-Image's per-step output heads, without "
                                    "which transformer/ is not the released 6-step model")
        heads, head_dim = config["num_attention_heads"], config["attention_head_dim"]
        reader, size = HL._shard_reader(tdir, "model", dtype, lambda s: C.original_to_port(
            s, interleaved=True, heads=heads, head_dim=head_dim))
        reader = _with_fused_heads(reader, heads_file)
        overrides = {k: config[k] for k in MiniMaxH3Transformer.CONFIG_KEYS if k in config}
    overrides.pop("adaln_curve_grid", None)
    overrides.update(audio_latents_dim=None, pdd_steps=len(sigmas) - 1, pdd_sigmas=sigmas)
    return HL.build_transformer(reader, device, quant, dtype, quant_devices,
                                source=f"{Path(model_dir).name}/{tdir.name}", quantized=False, source_bytes=size,
                                config_overrides=overrides)


def load_transformer_file(path, device, quant: str = "auto", dtype=torch.bfloat16,
                          quant_devices=None) -> MiniMaxH3Transformer:
    """A Comfy-Org-layout file. The heads' sigma grid is the file's "config" metadata (transformer.pdd_sigmas), else
    the released one."""
    path = Path(path)
    if not hsf.is_h3_image_file(path):
        if hsf.is_h3_file(path):
            raise ValueError(f"{path.name} is a MiniMax H3 video + audio diffusion model (it has an audio stream): "
                             "pick MiniMax-H3-FL2VA or P-H3-Image for it")
        raise ValueError(f"{path.name}: not a MiniMax-H3-Image diffusion model (no video_patch_proj / condition_proj)")
    overrides = {}
    try:
        meta = json.loads(hsf.file_metadata(path).get("config") or "{}").get("transformer") or {}
    except (ValueError, AttributeError):
        meta = {}
    if meta.get("pdd_sigmas"):
        overrides["pdd_sigmas"] = [float(s) for s in meta["pdd_sigmas"]]
    return HL.build_transformer(hsf.file_reader(path, dtype), device, quant, dtype, quant_devices, source=path.name,
                                quantized=hsf.is_quantized_file(path), source_bytes=path.stat().st_size,
                                config_overrides=overrides)


def resolve_model_dir(model_path: str, models_dir: str = "models") -> Path:
    path = (model_path or REPO).strip()
    p = Path(path).expanduser()
    if p.is_dir():
        return p.resolve()
    local = Path(models_dir) / path.replace("/", "__")
    if component_ready(local / "transformer") and (local / PDD_FILE).is_file() \
            and component_ready(local / "text_encoder"):  # (an interrupted download: resumed)
        return local.resolve()
    from huggingface_hub import snapshot_download

    log.info("Downloading %s (about 66 GB of transformer, 54 GB of text encoder shards, 2.3 GB of VAE) to %s", path,
             local)
    snapshot_download(path, local_dir=str(local), allow_patterns=ALLOW_PATTERNS)
    return local.resolve()


def resolve_for_config(cfg: TrainConfig) -> Path | None:
    if cfg.model_source == "files" and all(getattr(cfg, f) for f in ("transformer_file", "text_encoder_file", "vae_file")):
        return None
    return resolve_model_dir(cfg.model_path, cfg.models_dir)


def _need_dir(model_dir: Path | None, what: str) -> Path:
    if model_dir is None:
        raise ValueError(f"No {what} file is set and no Hugging Face model to take it from: set the file (model "
                         f"source 'files') or the {REPO} repo id / a downloaded folder of it")
    return model_dir


def get_transformer(cfg: TrainConfig, model_dir: Path | None, device: str | None = None) -> MiniMaxH3Transformer:
    use_file = moss_loading._use_file(cfg, "transformer_file")
    if use_file:
        moss_loading._check_files(cfg, "transformer_file")
    else:
        model_dir = _need_dir(model_dir, "diffusion model")
    quant_devices = None
    if cfg.multi_gpu == "model_split" and device is None:
        # the trainer splits it right after loading: stay on the CPU and size "auto" for all the split GPUs
        from ..multigpu import parse_devices

        quant_devices, device = parse_devices(cfg.devices, cfg.device), "cpu"
    device = device or cfg.device
    if use_file:
        return load_transformer_file(cfg.transformer_file, device, cfg.transformer_quant, quant_devices=quant_devices)
    return load_transformer(model_dir, device, cfg.transformer_quant, quant_devices=quant_devices)


def load_any_transformer(path: str, device, quant: str = "auto", models_dir: str = "models") -> MiniMaxH3Transformer:
    """A sampling model: a single .safetensors file, a downloaded folder or a repo id."""
    p = Path(path).expanduser()
    if p.is_file():
        return load_transformer_file(p, device, quant)
    return load_transformer(p.resolve() if p.is_dir() else resolve_model_dir(path, models_dir), device, quant)


# ---------------------------------------------------------------- text encoder (MiniMax H3's: Qwen3-VL-32B, layer 50)


def _tokenizer(cfg: TrainConfig, model_dir: Path | None, use_file: bool):
    from transformers import AutoTokenizer

    if use_file or model_dir is None:
        tok = AutoTokenizer.from_pretrained(str(hsf.resolve_tokenizer(cfg.tokenizer_path, cfg.models_dir, REPO)))
    else:
        tok = AutoTokenizer.from_pretrained(str(Path(model_dir) / "tokenizer"))
    hsf.check_tokenizer(tok)
    return tok


def get_tokenizer(cfg: TrainConfig, model_dir: Path | None):
    return _tokenizer(cfg, model_dir, moss_loading._use_file(cfg, "text_encoder_file"))


def get_text_encoder(cfg: TrainConfig, model_dir: Path | None):
    if moss_loading._use_file(cfg, "text_encoder_file"):
        moss_loading._check_files(cfg, "text_encoder_file")
        tokenizer = _tokenizer(cfg, None, True)
        model = hsf.load_text_encoder_file(cfg.text_encoder_file, cfg.text_encoder_device, cfg.text_encoder_quant)
        return model, tokenizer
    return HL.load_text_encoder(_need_dir(model_dir, "text encoder"), cfg.text_encoder_device, cfg.text_encoder_quant)


encode_prompts = HL.encode_prompts
encode_prompt = HL.encode_prompt
prompt_tokens = HL.prompt_tokens
pad_embeds = HL.pad_embeds


# ---------------------------------------------------------------- VAE


def _vae_config(decoder: dict, stats: dict | None = None) -> dict:
    """H3's encoder config and latent statistics (the image VAE's are the same numbers) with the image decoder's."""
    config = {k: v for k, v in hsf.VAE_CONFIG.items() if not k.startswith("decoder_")}
    for key in ("latents_mean", "latents_std"):
        if stats and stats.get(key) is not None:
            config[key] = [float(x) for x in stats[key]]
    config.update(decoder)
    return config


def _check_image_vae(state: dict, name: str) -> None:
    if "decoder.proj_in.weight" not in state or "encoder.conv_in.weight" not in state:
        raise ValueError(f"{name} is not a MiniMax-H3-Image VAE")
    pq = state.get("post_quant_conv.weight")
    if pq is None or pq.ndim != 4:
        raise ValueError(f"{name} is a video VAE (MiniMax H3's, or P-H3-Image's): MiniMax-H3-Image needs its own image "
                         f"VAE, vae/{VAE_FILE}")


def vae_from_state(state: dict, name: str, decoder_config: dict | None = None, stats: dict | None = None,
                   dtype=torch.bfloat16):
    """The VAE from a diffusers-layout fp32 state (consumed)."""
    from .vae import build_vae, decoder_config_from_state

    _check_image_vae(state, name)
    return build_vae(state, _vae_config(decoder_config_from_state(state, decoder_config), stats), dtype)


def load_vae_file(path, device, dtype=torch.bfloat16):
    """Comfy-Org's image VAE file (fp16), in MiniMax's original key layout; it carries no latent statistics (H3's
    are the released config's)."""
    state = hsf.read_state(path, torch.float32)
    if any(isinstance(v, Q.QuantWeight) for v in state.values()):
        raise ValueError(f"{Path(path).name}: quantized VAE files are not supported; use vae/{VAE_FILE}")
    state = sf.strip_prefix(state, ("vae.", "first_stage_model."))
    stats = {k: state.pop(k).float() for k in ("latents_mean", "latents_std") if k in state}
    heads = _original_heads(state, Path(path).name)
    state = hsf.vae_original_to_diffusers(state, heads=heads)
    return vae_from_state(state, Path(path).name, None, stats, dtype).to(device).eval().requires_grad_(False)


def _original_heads(state: dict, name: str, head_dim: int = 64) -> int:
    """The decoder's head count of an original-layout state (its fused to_qkv rows are per head)."""
    w = state.get("decoder.x_embedder.weight")
    if w is None:
        raise ValueError(f"{name} is not a MiniMax-H3-Image VAE")
    return w.shape[0] // head_dim


def load_vae(model_dir: Path, device, dtype=torch.bfloat16):
    """vae/ of a downloaded MiniMaxAI/MiniMax-H3-Image (the original layout, config MiniMaxH3ImageVAE) or a diffusers
    folder's (AutoencoderKLMiniMaxH3Image; the repository's vae_diffusers/)."""
    from safetensors import safe_open

    for sub in ("vae", "vae_diffusers"):
        vdir = Path(model_dir) / sub
        file = vdir / "diffusion_pytorch_model.safetensors"
        if not file.is_file():
            continue
        cfg = _json(vdir / "config.json")
        with safe_open(str(file), "pt") as f:
            state = {k: f.get_tensor(k).float() for k in f.keys()}
        stats = {k: cfg[k] for k in ("latents_mean", "latents_std") if k in cfg}
        if cfg.get("_class_name") in VAE_DIFFUSERS_CLASSES or "decoder.proj_in.weight" in state:
            decoder = {k: v for k, v in cfg.items() if k.startswith("decoder_")}
        else:
            kw = cfg.get("vit_decoder_kwargs") or {}
            decoder = {"decoder_attention_head_dim": kw.get("dim_head", 64),
                       "decoder_rope_theta": kw.get("rope_theta", 100.0),
                       "decoder_rope_dim_ratio": kw.get("rope_dim_ratio", 0.5)}
            heads = _original_heads(state, f"{vdir.name}/", int(decoder["decoder_attention_head_dim"]))
            state = hsf.vae_original_to_diffusers(state, heads=heads)
        vae = vae_from_state(state, f"{Path(model_dir).name}/{sub}", decoder, stats, dtype)
        return vae.to(device).eval().requires_grad_(False)
    raise FileNotFoundError(f"{model_dir} holds no vae/ or vae_diffusers/ with a diffusion_pytorch_model.safetensors")


def get_vae(cfg: TrainConfig, model_dir: Path | None):
    if moss_loading._use_file(cfg, "vae_file"):
        moss_loading._check_files(cfg, "vae_file")
        return load_vae_file(cfg.vae_file, cfg.device)
    return load_vae(_need_dir(model_dir, "VAE"), cfg.device)


encode_images = HL.encode_images


def _decode(vae, latents: torch.Tensor) -> torch.Tensor:
    from .vae import decode_image

    z = latents.to(vae.device, torch.float32)
    mean, std = HL._stats(vae, z)
    z = (z * std + mean).to(next(vae.post_quant_conv.parameters()).dtype)
    x = decode_image(vae, z)
    pm, ps = HL._pixel_norm(x)
    return (x.float() * ps + pm) * 2.0 - 1.0  # ImageNet space -> [0, 1] -> [-1, 1]


@torch.no_grad()
def decode_latents(vae, latents: torch.Tensor) -> torch.Tensor:
    """Normalized latents [B, 24, H/16, W/16] -> images [B, 3, H, W] in [-1, 1]."""
    return _decode(vae, latents).clamp(-1, 1)


def decode_latents_grad(vae, latents: torch.Tensor) -> torch.Tensor:
    return _decode(vae, latents)


def grad_vae(vae):
    return vae
