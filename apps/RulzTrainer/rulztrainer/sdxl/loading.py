"""Load SDXL components from a diffusers folder (stabilityai/stable-diffusion-xl-base-1.0 layout) or from one
all-in-one checkpoint (sdxl.single_file, model_source "files"), and encode.

    unet/             config.json + diffusion_pytorch_model[.fp16].safetensors   (UNet2DConditionModel, 2.6B)
    vae/              AutoencoderKL (4 channels, 8x, scaling 0.13025; fp16 overflows, so it runs in fp32)
    text_encoder/     CLIP-L  (CLIPTextModel, 768)
    text_encoder_2/   OpenCLIP bigG (CLIPTextModelWithProjection, 1280)
    tokenizer/, tokenizer_2/, scheduler/

Text features: the SDXL pipeline's. Both encoders see the same 77-token window (BOS + 75 + EOS, padded to 77 with
each tokenizer's own pad token: CLIP-L pads with EOS, bigG with "!" = id 0); the penultimate hidden states
(hidden_states[-2], no final layer norm) are concatenated into [77, 2048] and bigG's projected pooled embedding
(text_embeds of the same window) goes in row 77, zero-padded to 2048: one cached tensor [78, 2048] per prompt. The
empty prompt is encoded like any other (as ComfyUI and A1111 encode a negative ""). Captions longer than 75 tokens
are cut, with one warning per tokenizer object.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from ..config import TrainConfig
from . import single_file as sf
from .model import POOLED_ROW, TEXT_DIM, TEXT_ROWS, SdxlUNet, patchify, unpatchify

log = logging.getLogger("rulztrainer")

MAX_TOKENS = 77
SCALING_FACTOR = 0.13025
REPO = "stabilityai/stable-diffusion-xl-base-1.0"
COMPONENTS = ("unet", "vae", "text_encoder", "text_encoder_2")
TOKENIZER_DIRS = ("tokenizer", "tokenizer_2")
WEIGHT_STEM = {"unet": "diffusion_pytorch_model", "vae": "diffusion_pytorch_model", "text_encoder": "model",
               "text_encoder_2": "model"}
# which precision to read from a folder that has both: the UNet and the text encoders run in bf16 (an fp16 file holds
# everything bf16 does, at half the size of the fp32 one); the VAE runs in fp32, so its fp32 file is the better source
PREFER_FP16 = {"unet": True, "vae": False, "text_encoder": True, "text_encoder_2": True}
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".ckpt", ".pt", ".msgpack", ".h5", ".onnx", ".onnx_data", ".xml", ".pb")


# ---------------------------------------------------------------- folder layout


def is_sdxl_dir(model_dir: Path) -> bool:
    cfg = Path(model_dir) / "unet" / "config.json"
    try:
        return json.loads(cfg.read_text(encoding="utf-8")).get("_class_name") == "UNet2DConditionModel"
    except (OSError, ValueError):
        return False


def _is_weight(name: str) -> bool:
    return name.endswith(_WEIGHT_SUFFIXES) or ".safetensors.index" in name or ".bin.index" in name


def _weight_files(files: list[str], component: str) -> list[str]:
    return [f for f in files if f.startswith(component + "/") and f.count("/") == 1 and _is_weight(f)
            and f.endswith((".safetensors", ".json"))]


def plan_download(files: list[str]) -> list[str]:
    """The files of a repo listing to fetch: every config / tokenizer / scheduler file of the SDXL layout, plus ONE
    precision of each component's safetensors (shards and their index included), never the all-in-one checkpoints
    in the repo root or both copies of a component."""
    keep = [f for f in files if f == "model_index.json"]
    for d in (*COMPONENTS, *TOKENIZER_DIRS, "scheduler"):
        keep += [f for f in files if f.startswith(d + "/") and not _is_weight(f)]
    for comp in COMPONENTS:
        cands = _weight_files(files, comp)
        fp16 = [f for f in cands if ".fp16." in f or ".fp16-" in f]  # (sharded: model.fp16-00001-of-00002.safetensors)
        plain = [f for f in cands if f not in fp16 and ".ema" not in f and ".non_ema" not in f]
        order = (fp16, plain) if PREFER_FP16[comp] else (plain, fp16)
        keep += next((group for group in order if group), [])
    return sorted(set(keep))


def _variant(directory: Path, comp: str) -> str | None:
    """The `variant` argument for from_pretrained: None for plain weights, "fp16" when only those exist or are
    preferred (and exist). Raises when the component has no safetensors."""
    stem = WEIGHT_STEM[comp]
    has_plain = (directory / f"{stem}.safetensors").is_file() or (directory / f"{stem}.safetensors.index.json").is_file()
    has_fp16 = ((directory / f"{stem}.fp16.safetensors").is_file()
                or (directory / f"{stem}.safetensors.index.fp16.json").is_file())
    if not (has_plain or has_fp16):
        raise FileNotFoundError(f"No {stem}[.fp16].safetensors in {directory}")
    if has_plain and has_fp16:
        return "fp16" if PREFER_FP16[comp] else None
    return "fp16" if has_fp16 else None


def component_ready(model_dir: Path) -> bool:
    for comp in COMPONENTS:
        d = Path(model_dir) / comp
        try:
            _variant(d, comp)
        except FileNotFoundError:
            return False
        if not (d / "config.json").is_file():
            return False
    return all((Path(model_dir) / t / "tokenizer_config.json").is_file() for t in TOKENIZER_DIRS)


def resolve_model_dir(model_path: str, models_dir: str = "models", download: bool = True) -> Path:
    """A local directory for a HF repo id or local path, downloading what SDXL needs (one precision per component, no
    all-in-one checkpoint) if it is not there yet."""
    p = Path(model_path).expanduser()
    if p.is_dir():
        return p.resolve()
    local = Path(models_dir) / model_path.replace("/", "__")
    if component_ready(local):
        return local.resolve()
    if not download:
        raise FileNotFoundError(f"Model not found locally: {model_path} (looked in {local})")
    from huggingface_hub import HfApi, snapshot_download

    files = plan_download(HfApi().list_repo_files(model_path))
    if not any(f.startswith("unet/") and f.endswith((".safetensors", ".safetensors.index.json")) for f in files):
        raise FileNotFoundError(f"{model_path} is not an SDXL repo in the diffusers layout (no unet/*.safetensors)")
    log.info("Downloading %s to %s (%d files)...", model_path, local, len(files))
    snapshot_download(model_path, local_dir=str(local), allow_patterns=files)
    return local.resolve()


def resolve_tokenizers_dir(cfg: TrainConfig, model_dir: Path | None = None) -> Path:
    """A folder holding tokenizer/ and tokenizer_2/: the SDXL folder in use, the configured tokenizer_path (that
    folder or a diffusers folder), an SDXL repo already in models_dir, or the base repo's two tokenizer folders
    (downloaded once, a few MB)."""
    def has(d: Path) -> bool:
        return all((d / t / "tokenizer_config.json").is_file() for t in TOKENIZER_DIRS)

    if model_dir is not None and has(Path(model_dir)):
        return Path(model_dir)
    tp = getattr(cfg, "tokenizer_path", "") or ""
    if tp:
        p = Path(tp).expanduser()
        if has(p):
            return p
        raise FileNotFoundError(f"No tokenizer/ and tokenizer_2/ with a tokenizer_config.json under {p}")
    models = Path(cfg.models_dir)
    for cand in sorted(models.glob("*stable-diffusion-xl*")) if models.is_dir() else ():
        if has(cand):
            return cand
    local = models / "tokenizer__sdxl"
    if not has(local):
        from huggingface_hub import snapshot_download

        log.info("Downloading the SDXL tokenizers to %s", local)
        snapshot_download(REPO, local_dir=str(local), allow_patterns=[f"{t}/*" for t in TOKENIZER_DIRS])
    return local


# ---------------------------------------------------------------- prediction type


def resolve_prediction(cfg, path) -> tuple[str, bool]:
    """(prediction_type, zero_terminal_snr). cfg.sdxl_prediction "auto" reads scheduler/scheduler_config.json of a
    diffusers folder (prediction_type, rescale_betas_zero_snr), or the `v_pred` / `ztsnr` marker tensors of a single
    file, and falls back to epsilon; "epsilon" / "v_prediction" are taken as given. cfg.sdxl_zero_terminal_snr True
    turns zero terminal SNR on; detection (when the file or scheduler says so) turns it on too."""
    want = str(getattr(cfg, "sdxl_prediction", "auto") or "auto")
    if want not in ("auto", "epsilon", "v_prediction"):
        raise ValueError(f"sdxl_prediction must be auto, epsilon or v_prediction, not {want!r}")
    pred, ztsnr = None, None
    p = Path(path) if path else None
    if p is not None:
        if p.is_dir():
            try:
                sched = json.loads((p / "scheduler" / "scheduler_config.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                sched = {}
            pred = sched.get("prediction_type")
            ztsnr = bool(sched["rescale_betas_zero_snr"]) if "rescale_betas_zero_snr" in sched else None
        elif p.is_file():
            try:
                pred, ztsnr = sf.file_prediction(p)
            except Exception:  # noqa: BLE001 - not a safetensors file: epsilon
                pred, ztsnr = None, None
    if pred not in (None, "epsilon", "v_prediction"):
        if want == "auto":
            raise ValueError(f"The model's scheduler uses prediction_type {pred!r}; SDXL training supports epsilon "
                             f"and v_prediction (set sdxl_prediction to pick one)")
        pred = None
    result = (pred or "epsilon") if want == "auto" else want
    return result, bool(getattr(cfg, "sdxl_zero_terminal_snr", False)) or bool(ztsnr)


# ---------------------------------------------------------------- text: tokenizers and encoders


class SdxlTokenizers:
    """CLIP-L's and bigG's tokenizers as one object (a window of 77 tokens each)."""

    def __init__(self, tok_l, tok_g):
        self.tok_l, self.tok_g = tok_l, tok_g
        self._warned = False

    @classmethod
    def from_dir(cls, root: Path) -> "SdxlTokenizers":
        from transformers import CLIPTokenizer

        root = Path(root)
        return cls(CLIPTokenizer.from_pretrained(str(root / "tokenizer")),
                   CLIPTokenizer.from_pretrained(str(root / "tokenizer_2")))

    def _warn_cut(self, prompts: list[str]) -> None:
        if self._warned:
            return
        lengths = self.tok_l(prompts, truncation=False)["input_ids"]
        if any(len(i) > MAX_TOKENS for i in lengths):
            self._warned = True
            log.warning("A caption is longer than SDXL's 77-token window (75 words/pieces): it is cut there. This "
                        "is warned once per run; long captions lose their tail.")

    def __call__(self, prompts: list[str], max_length: int = MAX_TOKENS):
        """(ids_l, ids_g), each [B, 77] long: BOS, the prompt cut to 75 tokens, EOS, then the tokenizer's pad."""
        self._warn_cut(prompts)  # max_length is accepted for the shared signature: the window is fixed at 77
        kw = dict(padding="max_length", truncation=True, max_length=MAX_TOKENS, return_tensors="pt")
        return self.tok_l(prompts, **kw)["input_ids"], self.tok_g(prompts, **kw)["input_ids"]


class SdxlTextEncoders(nn.Module):
    """CLIP-L + bigG as one module (the trainer moves, unloads and measures the text encoder as a whole)."""

    def __init__(self, clip_l, clip_g):
        super().__init__()
        self.clip_l, self.clip_g = clip_l, clip_g

    @property
    def input_device(self) -> torch.device:
        for p in self.parameters():
            if p.device.type != "meta":
                return p.device
        return torch.device("cpu")

    def forward(self, ids_l: torch.Tensor, ids_g: torch.Tensor):
        """-> (hidden [B, 77, 768 + 1280], pooled [B, 1280]): penultimate hidden states and bigG's text_embeds."""
        out_l = self.clip_l(input_ids=ids_l, output_hidden_states=True)
        out_g = self.clip_g(input_ids=ids_g, output_hidden_states=True)
        hidden = torch.cat([out_l.hidden_states[-2], out_g.hidden_states[-2]], dim=-1)
        return hidden, out_g.text_embeds


@torch.no_grad()
def encode_prompts(text_encoder, tokenizer, prompts: list[str], max_length: int = MAX_TOKENS) -> list[torch.Tensor]:
    """Text features for each prompt: [78, 2048] bf16 on the CPU (rows 0..76 hidden states, row 77 the pooled
    embedding), in one forward. Empty prompts are encoded like any other."""
    ids_l, ids_g = tokenizer(list(prompts), max_length)
    dev = text_encoder.input_device
    hidden, pooled = text_encoder(ids_l.to(dev), ids_g.to(dev))
    dim = hidden.shape[-1]
    if pooled.shape[-1] > dim:
        raise ValueError("the pooled embedding is wider than the hidden states")
    row = F.pad(pooled, (0, dim - pooled.shape[-1])).unsqueeze(1)
    rows = torch.cat([hidden, row], dim=1).to(torch.bfloat16).cpu()
    return [rows[i].clone() for i in range(rows.shape[0])]


def encode_prompt(text_encoder, tokenizer, prompt: str, max_length: int = MAX_TOKENS) -> torch.Tensor:
    return encode_prompts(text_encoder, tokenizer, [prompt], max_length)[0]


def prompt_tokens(tokenizer, prompt: str, max_length: int = MAX_TOKENS) -> tuple[str, list, int, int]:
    """(text, token offsets, leading tokens the cache drops, cached length). The cache keeps every row (BOS is row 0),
    so nothing is dropped; the offsets are CLIP-L's for the 77-token window as encode_prompt cuts it (BOS and EOS
    have the empty span (0, 0)), and the cached length is the number of real tokens, EOS included."""
    enc = tokenizer.tok_l(prompt, return_offsets_mapping=True, truncation=True, max_length=MAX_TOKENS)
    offsets = [tuple(o) for o in enc["offset_mapping"]]
    return prompt, offsets, 0, len(offsets)


def pad_embeds(rows: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """[78, 2048] rows -> [B, 78, 2048] and an all-True [B, 78] mask (every row is the full window)."""
    for r in rows:
        if r.shape[0] != TEXT_ROWS:
            raise ValueError(f"an SDXL text row must have {TEXT_ROWS} rows, got {tuple(r.shape)}")
    embeds = torch.stack(list(rows))
    return embeds, torch.ones(embeds.shape[:2], dtype=torch.bool)


# ---------------------------------------------------------------- VAE, the trainer's latent layout


def encode_images(vae, images: torch.Tensor) -> torch.Tensor:
    """images [B,3,H,W] in [-1,1] (H, W multiples of 16) -> scaled posterior mean, patchified [B,16,H/16,W/16]."""
    z = vae.encode(images.to(vae.device, vae.dtype)).latent_dist.mode()
    return patchify(z * vae.config.scaling_factor)


def _decode(vae, latents: torch.Tensor) -> torch.Tensor:
    z = unpatchify(latents.to(vae.device, torch.float32)) / vae.config.scaling_factor
    return vae.decode(z.to(vae.dtype), return_dict=False)[0]


@torch.no_grad()
def decode_latents(vae, latents: torch.Tensor) -> torch.Tensor:
    """Inverse of encode_images: images [B,3,H,W] in [-1,1] (clamped)."""
    return _decode(vae, latents).float().clamp(-1, 1)


def decode_latents_grad(vae, latents: torch.Tensor) -> torch.Tensor:
    """decode_latents with gradients to `latents` (identity loss on a face crop), not clamped."""
    return _decode(vae, latents).float()


def grad_vae(vae):
    """A bf16 copy of the VAE with gradient checkpointing, for decode_latents_grad."""
    from ..moss import loading as moss_loading

    out = moss_loading.bf16_copy(vae)
    out.enable_gradient_checkpointing()
    return out


# ---------------------------------------------------------------- loaders


def _validate_quant(quant):
    if _norm_quant(quant) not in (None, "none", "auto", "int8", "int8_convrot"):
        raise ValueError(f"SDXL UNet quantization {quant!r} is unsupported; use none, auto, int8 or int8_convrot")


def _norm_quant(quant):
    """The quant name in lower case (the run-start check takes INT8 as int8 too)."""
    return quant.lower() if isinstance(quant, str) else quant


def _quantize_unet(unet, quant, device):
    _validate_quant(quant)
    quant = _norm_quant(quant)
    if quant in (None, "none", "auto"):
        return unet
    from .. import quant as Q
    from ..moss.single_file import quant_target

    target = quant_target(device)
    Q.quantize_module_linears(unet, quant, min_dim=512, log=log,
                              name_filter=lambda name: "transformer_blocks." in name,
                              work_device=target.get("work_device"))
    return unet


def load_unet_dir(model_dir: Path, device, prediction_type: str = "epsilon", zero_terminal_snr: bool = False,
                  dtype=torch.bfloat16, quant: str = "auto") -> SdxlUNet:
    _validate_quant(quant)
    from diffusers import UNet2DConditionModel

    d = Path(model_dir) / "unet"
    variant = _variant(d, "unet")
    unet = UNet2DConditionModel.from_pretrained(str(d), torch_dtype=dtype, variant=variant)
    model = SdxlUNet(_quantize_unet(unet, quant, device), prediction_type, zero_terminal_snr)
    return model.to(device).eval().requires_grad_(False)


def load_unet_file(path, device, prediction_type="epsilon", zero_terminal_snr=False, quant="auto"):
    _validate_quant(quant)
    unet = sf.load_unet_file(path, "cpu")
    model = SdxlUNet(_quantize_unet(unet, quant, device), prediction_type, zero_terminal_snr)
    return model.to(device).eval().requires_grad_(False)


def load_vae_dir(model_dir: Path, device):
    from diffusers import AutoencoderKL

    d = Path(model_dir) / "vae"
    vae = AutoencoderKL.from_pretrained(str(d), torch_dtype=torch.float32, variant=_variant(d, "vae"))
    return vae.to(device).eval().requires_grad_(False)


def load_text_models_dir(model_dir: Path, device, dtype=torch.bfloat16):
    from transformers import CLIPTextModel, CLIPTextModelWithProjection

    d1, d2 = Path(model_dir) / "text_encoder", Path(model_dir) / "text_encoder_2"
    clip_l = CLIPTextModel.from_pretrained(str(d1), torch_dtype=dtype, variant=_variant(d1, "text_encoder"))
    clip_g = CLIPTextModelWithProjection.from_pretrained(str(d2), torch_dtype=dtype,
                                                         variant=_variant(d2, "text_encoder_2"))
    return clip_l.eval().requires_grad_(False), clip_g.eval().requires_grad_(False)


def _place_text(model: SdxlTextEncoders, device: str) -> SdxlTextEncoders:
    if device != "auto":
        return model.to(device)
    size = sum(p.numel() * p.element_size() for p in model.parameters())
    if torch.cuda.is_available():
        best = None
        for i in range(torch.cuda.device_count()):
            free, _ = torch.cuda.mem_get_info(i)
            if free > size * 1.1 + 1.5 * 2**30 and (best is None or free > best[0]):
                best = (free, i)
        if best is not None:
            log.info("Text encoders (%.1f GB) placed on cuda:%d", size / 2**30, best[1])
            return model.to(f"cuda:{best[1]}")
    log.info("Text encoders (%.1f GB) stay on the CPU", size / 2**30)
    return model


# ---------------------------------------------------------------- config-driven entry points


def _files_mode(cfg: TrainConfig) -> bool:
    return cfg.model_source == "files"


def _checkpoint(cfg: TrainConfig, field: str) -> str:
    """The file a component comes from in files mode: its own field, else the all-in-one transformer_file; '' = the
    HF folder."""
    if not _files_mode(cfg):
        return ""
    return getattr(cfg, field, "") or getattr(cfg, "transformer_file", "") or ""


def _need_dir(model_dir: Path | None, what: str) -> Path:
    if model_dir is None:
        raise ValueError(f"No {what} file is set and no Hugging Face model to take it from: set the checkpoint "
                         f"(model source 'files') or the {REPO} repo id / a diffusers folder")
    return Path(model_dir)


def resolve_for_config(cfg: TrainConfig) -> Path | None:
    """Download / locate the HF folder when any component comes from it (a single all-in-one checkpoint covers
    everything; the tokenizers are fetched separately by get_tokenizer)."""
    if _files_mode(cfg) and cfg.transformer_file:
        return None
    return resolve_model_dir(cfg.model_path, cfg.models_dir)


def get_transformer(cfg: TrainConfig, model_dir: Path | None, device: str | None = None,
                    quant: str = "auto", **ignored) -> SdxlUNet:
    """Load a frozen UNet, optionally quantizing its transformer-block linears."""
    _validate_quant(quant)
    dev = device or cfg.device
    path = _checkpoint(cfg, "transformer_file")
    if path:
        if not Path(path).is_file():
            raise FileNotFoundError(f"transformer_file does not exist: {path!r}")
        pred, ztsnr = resolve_prediction(cfg, path)
        model = load_unet_file(path, dev, pred, ztsnr, quant)
    else:
        d = _need_dir(model_dir, "checkpoint")
        pred, ztsnr = resolve_prediction(cfg, d)
        model = load_unet_dir(d, dev, pred, ztsnr, quant=quant)
    log.info("Loaded SDXL UNet: %d transformer blocks, %s prediction%s", len(model.layers), pred,
             ", zero terminal SNR" if ztsnr else "")
    return model


def get_vae(cfg: TrainConfig, model_dir: Path | None):
    path = _checkpoint(cfg, "vae_file")
    if path:
        if not Path(path).is_file():
            raise FileNotFoundError(f"vae_file does not exist: {path!r}")
        return sf.load_vae_file(path, cfg.device)
    return load_vae_dir(_need_dir(model_dir, "VAE"), cfg.device)


def get_tokenizer(cfg: TrainConfig, model_dir: Path | None) -> SdxlTokenizers:
    return SdxlTokenizers.from_dir(resolve_tokenizers_dir(cfg, model_dir))


def get_text_encoder(cfg: TrainConfig, model_dir: Path | None):
    path = _checkpoint(cfg, "text_encoder_file")
    if path:
        if not Path(path).is_file():
            raise FileNotFoundError(f"text_encoder_file does not exist: {path!r}")
        clip_l, clip_g = sf.load_text_encoders_file(path, "cpu")
    else:
        clip_l, clip_g = load_text_models_dir(_need_dir(model_dir, "text encoder"), "cpu")
    model = _place_text(SdxlTextEncoders(clip_l, clip_g), cfg.text_encoder_device)
    return model, get_tokenizer(cfg, model_dir)


def load_any_transformer(path: str, device, quant: str = "auto", models_dir: str = "models") -> SdxlUNet:
    """A UNet from a single file, diffusers folder, or HF repo id."""
    _validate_quant(quant)
    class _Auto:
        sdxl_prediction, sdxl_zero_terminal_snr = "auto", False

    p = Path(path).expanduser()
    if p.is_file():
        pred, ztsnr = resolve_prediction(_Auto, p)
        return load_unet_file(p, device, pred, ztsnr, quant)
    d = p if p.is_dir() else resolve_model_dir(path, models_dir)
    pred, ztsnr = resolve_prediction(_Auto, d)
    return load_unet_dir(d, device, pred, ztsnr, quant=quant)
