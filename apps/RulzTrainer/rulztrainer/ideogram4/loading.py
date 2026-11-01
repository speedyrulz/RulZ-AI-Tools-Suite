"""Load Ideogram 4 components from the Hugging Face layout (ideogram-ai/ideogram-4-fp8, gated; the unsloth/
ideogram-4-fp8 mirror; the nf4 repos) and encode; single ComfyUI-style files (Comfy-Org/Ideogram-4, the default) go
through ideogram4.single_file.

    model_index.json             Ideogram4Pipeline
    transformer/                 Ideogram4Transformer2DModel, weight-only fp8 (per-row scales), fused attention
    unconditional_transformer/   the guidance branch's model (same size; not trained, not downloaded here)
    text_encoder/                Qwen3-VL-8B-Instruct (Qwen3VLModel), weight-only fp8
    tokenizer/                   its tokenizer (the same as Qwen-Image 2.1's)
    vae/                         AutoencoderKLFlux2

Text features (the pipelines' encode_prompt): the prompt (as_caption: JSON, compact) in Qwen3-VL's chat turn
"<|im_start|>user\\n{prompt}<|im_end|>\\n<|im_start|>assistant\\n" (the chat template's output for a user text turn,
no system turn), Qwen3-VL-8B's text model over it, and the outputs of its decoder layers 0, 3, ..., 33 and 35 (after
the layer, before the final norm for the last) side by side per token with the layer index running fastest: feature
d * 13 + i is channel d of the i-th tapped layer, as the pipelines stack them. The cache keeps every token, [S, 53248]
bf16: 104 KiB per token, so a 400-token JSON caption takes about 42 MB. The empty prompt has no tokens ([0, 53248]):
the unconditional branch.

Latents: the Flux2 VAE's mean latent [B, 32, H/8, W/8], cut in 2x2 patches into 128 channels ordered (patch row,
patch column, channel) with the channel fastest, then normalized with Ideogram's own per-channel shift and scale
(LATENT_SHIFT / LATENT_SCALE, Ideogram's latent_norm.py: what its pipeline decodes with and ai-toolkit trains
with). These are not the VAE's batch-norm statistics MOSS-Image and FIBO 2.0 use (up to 0.4 apart in shift, scales
0.87-1.04 of the bn std), and the channel order differs too, so the latents get their own cache files (latent_tag).
diffusers' Ideogram4Pipeline decodes with the VAE's bn statistics instead, and ComfyUI's Flux2 latent format uses
them as well."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

from ..config import TrainConfig
from ..moss import loading as moss_loading
from . import prompts
from . import single_file as isf
from .model import Ideogram4Transformer

log = logging.getLogger("rulztrainer")

TRANSFORMER_CLASSES = ("Ideogram4Transformer2DModel",)
# quantize the 34 blocks; the text / image / timestep projections and the final layer stay bf16, as Comfy-Org's
# int8_convrot file has them
QUANT_FILTER = lambda name: name.startswith("layers.")  # noqa: E731
# "auto" by the GPU memory the model will live in (all GPUs of a model split): bf16 is about 18.6 GB (9.29 B weights,
# the release's fp8 file is 9.29 GB at a byte each), int8_convrot 9.6 GB (Comfy-Org's file: 9.58 GB) and nvfp4 about
# 5.5 GB (Comfy-Org's nvfp4_mixed: 5.49 GB). bf16 on 32 GB cards (30 GB or more), int8 on 16 GB cards (15 GB or more),
# nvfp4 below: estimates from the file sizes plus training activations, not measured runs
AUTO_BF16_FROM_GB, AUTO_INT8_FROM_GB = 30, 15

# skipped from a Hugging Face snapshot: the unconditional model (9.29 GB, the guidance branch: not trained), the
# unsloth mirror's second copy of the text encoder (8.77 GB) and the model card's pictures
IGNORE_PATTERNS = ["assets/*", "*.md", "*.png", "*.jpg", "*.webp", "*.svg", "unconditional_transformer/*",
                   "ideogram-4-text_encoder-FP8.safetensors"]

QWEN3_VL_ACTIVATION_LAYERS = (0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35)
TEXT_DIM = 4096 * len(QWEN3_VL_ACTIVATION_LAYERS)  # 53248
MAX_TEXT_TOKENS = 2048  # the pipelines refuse longer prompts
PROMPT_PREFIX = "<|im_start|>user\n"
PROMPT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"

# Ideogram's latent normalization (src/ideogram4/latent_norm.py, Apache-2.0), in the (patch row, patch column,
# channel) order of the 128 patchified channels
LATENT_SHIFT = (
    0.01984364, 0.10149707, 0.29689495, 0.27188619, -0.21445648, -0.15979549, 0.05021099, -0.15083604, -0.15360136,
    -0.20131799, 0.01922352, 0.0622626, 0.10140969, -0.06739428, 0.3758261, -0.233712, 0.35164491, -0.02590912,
    -0.0271935, -0.10833897, -0.1476848, -0.01130957, -0.2298372, 0.23526423, -0.10893522, 0.11957631, 0.04047799,
    0.3134589, -0.17225064, -0.18646109, -0.34691978, -0.03571246, 0.02583857, 0.10190072, 0.28402294, 0.26952152,
    -0.21634675, -0.17938656, 0.04358909, -0.15007621, -0.1548502, -0.18971131, 0.02710861, 0.05609494, 0.10697846,
    -0.06854968, 0.38167698, -0.24269937, 0.35705471, -0.03063305, -0.02946109, -0.11244286, -0.14336038, -0.01362137,
    -0.21863696, 0.23228983, -0.11739769, 0.11693044, 0.02563311, 0.31356594, -0.17420591, -0.19006285, -0.34905377,
    -0.04025005, 0.01924137, 0.07652984, 0.2995608, 0.2628057, -0.22011674, -0.12715361, 0.04879879, -0.14075719,
    -0.15935895, -0.2123584, 0.01974813, 0.05523547, 0.10011992, -0.06428964, 0.37781868, -0.21491644, 0.34254215,
    -0.03153528, -0.0310082, -0.10761415, -0.14730405, -0.02475182, -0.2285588, 0.2515081, -0.10445128, 0.12446,
    0.07062869, 0.30880162, -0.18016875, -0.18869164, -0.34533499, -0.0129177, 0.02578168, 0.07993659, 0.28642181,
    0.26038408, -0.22459419, -0.14820155, 0.04059549, -0.14043529, -0.16111187, -0.2020305, 0.02602069, 0.04852717,
    0.10432153, -0.06309942, 0.38402443, -0.22397003, 0.34814481, -0.03774432, -0.03381438, -0.11245691, -0.14128767,
    -0.02853208, -0.21752016, 0.24872463, -0.11399775, 0.1222687, 0.05620835, 0.309178, -0.18065738, -0.19401479,
    -0.34495114, -0.01760592,
)
LATENT_SCALE = (
    1.63933691, 1.70204478, 1.73642566, 1.90004803, 1.6675316, 1.69059584, 1.56853198, 1.62314944, 1.89106626,
    1.58086668, 1.60822129, 1.60962993, 1.63322129, 1.56074359, 1.73419528, 1.7919265, 1.64040632, 1.66802808,
    1.60390303, 1.75480492, 1.63187587, 1.64334594, 1.61722884, 1.60146046, 1.63459219, 1.55291476, 1.68771497,
    1.68415657, 1.78966054, 1.66631641, 1.65626686, 1.65976433, 1.63487607, 1.69513249, 1.72933756, 1.91310663,
    1.67035057, 1.72286863, 1.56719251, 1.61934825, 1.88628859, 1.56911539, 1.59455129, 1.60829869, 1.62470611,
    1.56052853, 1.73677003, 1.77563606, 1.63732541, 1.66370527, 1.59508952, 1.75153949, 1.63029275, 1.64517667,
    1.61659342, 1.59722044, 1.64103121, 1.5408531, 1.68610394, 1.67772755, 1.78998563, 1.66621713, 1.65458955,
    1.66041308, 1.64710857, 1.68163503, 1.74000294, 1.92784786, 1.67411194, 1.67395548, 1.57406532, 1.62199356,
    1.87618195, 1.5584375, 1.57438785, 1.61711053, 1.63094305, 1.55644029, 1.73124302, 1.80666627, 1.6463621,
    1.65932006, 1.60816188, 1.75682671, 1.64695873, 1.63121722, 1.61380832, 1.60478651, 1.63396035, 1.53505068,
    1.65534289, 1.67132281, 1.80317197, 1.6767314, 1.65700938, 1.68426259, 1.65339716, 1.67540638, 1.73298504,
    1.94067348, 1.67893609, 1.70635117, 1.5730906, 1.61928553, 1.87148809, 1.56244866, 1.56697152, 1.61584394,
    1.62759496, 1.55480378, 1.73484107, 1.79055143, 1.64688773, 1.66121492, 1.60135887, 1.75254572, 1.64798332,
    1.62989921, 1.61381592, 1.60792883, 1.63939668, 1.53075757, 1.65371318, 1.66801185, 1.80029087, 1.67591476,
    1.65655173, 1.68533454,
)


def is_ideogram4_dir(model_dir: Path) -> bool:
    cfg = Path(model_dir) / "transformer" / "config.json"
    if not cfg.is_file():
        return False
    try:
        return json.loads(cfg.read_text(encoding="utf-8")).get("_class_name") in TRANSFORMER_CLASSES
    except (OSError, ValueError):
        return False


def _gpu_gb(device) -> float:
    dev = torch.device(device)
    return torch.cuda.get_device_properties(dev.index or 0).total_memory / 2**30


def resolve_quant(quant: str, devices) -> str:
    """`auto`: bf16 from 30 GB (a 32 GB card), int8_convrot from 15 GB (a 16 GB one), else nvfp4, counting the memory
    of every GPU the model is split over. A model loaded on the CPU (a sampling model) is sized for the smallest GPU,
    where it will run."""
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
        log.info("Ideogram 4 transformer: auto quantization -> %s for %.0f GB (%s)", fmt, total, where)
    return fmt


def resolve_model_dir(model_path: str, models_dir: str = "models", download: bool = True) -> Path:
    """A local directory for a HF repo id or local path, downloading what training needs (about 18.2 GB of the
    fp8 repos: transformer 9.29 GB, text encoder 8.78 GB, VAE 0.17 GB; not the unconditional model)."""
    p = Path(model_path).expanduser()
    if p.is_dir():
        return p.resolve()
    local = Path(models_dir) / model_path.replace("/", "__")
    if (local / "transformer" / "config.json").is_file():
        return local.resolve()
    if not download:
        raise FileNotFoundError(f"Model not found locally: {model_path} (looked in {local})")
    from huggingface_hub import snapshot_download

    log.info("Downloading %s to %s (transformer, text encoder, VAE; about 18 GB for the fp8 repos)...", model_path,
             local)
    snapshot_download(model_path, local_dir=str(local), ignore_patterns=IGNORE_PATTERNS)
    return local.resolve()


def _component_files(directory: Path, stem: str) -> list[Path]:
    index = directory / f"{stem}.safetensors.index.json"
    if index.is_file():
        mapping = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        return [directory / f for f in sorted(set(mapping.values()))]
    return [directory / f"{stem}.safetensors"]


def load_transformer(model_dir: Path, device, quant: str = "auto", dtype=torch.bfloat16,
                     quant_devices=None) -> Ideogram4Transformer:
    """quant_devices: the GPUs the model will run on, for quant "auto" (default: `device`)."""
    config = json.loads((Path(model_dir) / "transformer" / "config.json").read_text(encoding="utf-8"))
    if config.get("_class_name") not in TRANSFORMER_CLASSES:
        raise ValueError(f"Not an Ideogram 4 transformer: {config.get('_class_name')!r}")
    state = {}
    for f in _component_files(Path(model_dir) / "transformer", "diffusion_pytorch_model"):
        state.update(isf.read_state(f, dtype))
    return isf.build_transformer(isf.normalize_dit_state(state), device, quant, dtype, quant_devices,
                                 Path(model_dir).name, config=config)


def load_vae(model_dir: Path, device):
    from diffusers import AutoencoderKLFlux2

    vae = AutoencoderKLFlux2.from_pretrained(str(Path(model_dir) / "vae"), torch_dtype=torch.float32)
    return vae.to(device).eval().requires_grad_(False)


def load_text_encoder(model_dir: Path, device: str = "auto", quant: str = "auto"):
    """Qwen3-VL-8B's language model (text only) + tokenizer from the repo's text_encoder/ and tokenizer/. The fp8 (or
    bitsandbytes nf4) weights are dequantized; quant int8 / int8_convrot / nvfp4 quantizes them again."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(Path(model_dir) / "tokenizer"))
    return isf.load_hf_text_encoder(Path(model_dir) / "text_encoder", device, quant), tokenizer


# ---------------------------------------------------------------- text


def _input_device(model) -> torch.device:
    return model.get_input_embeddings().weight.device


def _layers(text_encoder):
    layers = getattr(text_encoder, "layers", None)
    if layers is None:
        layers = text_encoder.language_model.layers
    return layers


def _prompt_ids(tokenizer, text: str, max_length: int) -> torch.Tensor:
    limit = max(16, min(int(max_length), MAX_TEXT_TOKENS))
    suffix = tokenizer([PROMPT_SUFFIX], add_special_tokens=False, return_tensors="pt").input_ids
    head = tokenizer([PROMPT_PREFIX + text], add_special_tokens=False, truncation=True,
                     max_length=limit - suffix.shape[1], return_tensors="pt").input_ids
    return torch.cat([head, suffix], dim=1)


@torch.no_grad()
def encode_prompt(text_encoder, tokenizer, prompt: str, max_length: int = 1024) -> torch.Tensor:
    """Text features for one prompt: [S, 53248] bf16 on the CPU (see the module docstring); [0, 53248] for the empty
    prompt. The prompt is cut to max_length tokens (at most 2048, the pipelines' limit) with the chat turn's end kept."""
    text = prompts.as_caption(prompt)
    if not text:
        return torch.zeros(0, TEXT_DIM, dtype=torch.bfloat16)
    input_ids = _prompt_ids(tokenizer, text, max_length).to(_input_device(text_encoder))
    n = input_ids.shape[1]
    position_ids = torch.arange(n, device=input_ids.device).view(1, 1, n).expand(3, 1, n)
    layers = _layers(text_encoder)
    taps: dict[int, torch.Tensor] = {}

    def hook(i):
        def fn(module, args, output):
            taps[i] = (output[0] if isinstance(output, (tuple, list)) else output).detach()
        return fn

    handles = [layers[i].register_forward_hook(hook(i)) for i in QWEN3_VL_ACTIVATION_LAYERS]
    try:
        text_encoder(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), position_ids=position_ids)
    finally:
        for h in handles:
            h.remove()
    feats = torch.stack([taps[i][0].to("cpu", torch.float32) for i in QWEN3_VL_ACTIVATION_LAYERS], dim=-1)
    return feats.reshape(n, -1).to(torch.bfloat16)  # [S, 4096, 13] -> [S, 53248], the layer index fastest


def get_tokenizer(cfg: TrainConfig, model_dir: Path | None):
    from transformers import AutoTokenizer

    if moss_loading._use_file(cfg, "text_encoder_file"):
        from ..qwen21 import single_file as qsf

        return AutoTokenizer.from_pretrained(str(qsf.resolve_tokenizer(cfg.tokenizer_path, cfg.models_dir)))
    return AutoTokenizer.from_pretrained(str(_need_dir(cfg, model_dir, "tokenizer") / "tokenizer"))


def prompt_tokens(tokenizer, prompt: str) -> tuple[str, list, int, int]:
    """encode_prompt's text (the chat turn up to and including the caption; its end follows) with its token offsets;
    the cache keeps every token."""
    caption = prompts.as_caption(prompt)
    if not caption:
        return "", [], 0, 0
    text = PROMPT_PREFIX + caption
    offsets = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]
    n_suffix = len(tokenizer(PROMPT_SUFFIX, add_special_tokens=False).input_ids)
    return text, [tuple(o) for o in offsets], 0, len(offsets) + n_suffix


def pad_embeds(rows: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """[S_i, 53248] rows -> [B, S, 53248] (right-padded with zeros) and a [B, S] bool mask of the real tokens. Rows
    may be empty (the unconditional prompt)."""
    length = max(r.shape[0] for r in rows)
    embeds = rows[0].new_zeros(len(rows), length, rows[0].shape[-1])
    mask = torch.zeros(len(rows), length, dtype=torch.bool)
    for i, r in enumerate(rows):
        embeds[i, : r.shape[0]] = r
        mask[i, : r.shape[0]] = True
    return embeds, mask


# ---------------------------------------------------------------- VAE (Flux2), Ideogram's latent layout


def _norm(like: torch.Tensor):
    shift = torch.tensor(LATENT_SHIFT, dtype=torch.float32, device=like.device).view(1, -1, 1, 1)
    scale = torch.tensor(LATENT_SCALE, dtype=torch.float32, device=like.device).view(1, -1, 1, 1)
    return shift, scale


def patchify(z: torch.Tensor) -> torch.Tensor:
    """[B, 32, H/8, W/8] -> [B, 128, H/16, W/16], channels (patch row, patch column, channel), the channel fastest."""
    b, c, h, w = z.shape
    return z.view(b, c, h // 2, 2, w // 2, 2).permute(0, 3, 5, 1, 2, 4).reshape(b, 4 * c, h // 2, w // 2)


def unpatchify(z: torch.Tensor) -> torch.Tensor:
    b, c, h, w = z.shape
    return z.view(b, 2, 2, c // 4, h, w).permute(0, 3, 4, 1, 5, 2).reshape(b, c // 4, 2 * h, 2 * w)


@torch.no_grad()
def encode_images(vae, images: torch.Tensor) -> torch.Tensor:
    """images: [B,3,H,W] in [-1,1] (H,W multiples of 16). Returns normalized latents [B,128,H/16,W/16] (fp32)."""
    z = vae.encode(images.to(vae.device, vae.dtype)).latent_dist.mode().float()
    shift, scale = _norm(z)
    return (patchify(z) - shift) / scale


def _decode(vae, latents: torch.Tensor) -> torch.Tensor:
    z = latents.to(vae.device, torch.float32)  # the normalization applies in fp32 (a bf16 VAE gets the result)
    shift, scale = _norm(z)
    return vae.decode(unpatchify(z * scale + shift).to(vae.dtype), return_dict=False)[0]


@torch.no_grad()
def decode_latents(vae, latents: torch.Tensor) -> torch.Tensor:
    """Inverse of encode_images. Returns images [B,3,H,W] in [-1,1], decoded whole on a bf16 copy of the VAE as
    MOSS-Image's (the same VAE; moss.loading.decode_latents measured the copy at 0.16 of an 8-bit level)."""
    return _decode(moss_loading.decode_vae(vae), latents).float().clamp(-1, 1)


def decode_latents_grad(vae, latents: torch.Tensor) -> torch.Tensor:
    """decode_latents with gradients to `latents` (identity loss on a face crop), not clamped; `vae` is grad_vae's."""
    return _decode(vae, latents).float()


def grad_vae(vae):
    return moss_loading.grad_vae(vae)


# ---------------------------------------------------------------- config-driven entry points


def _need_dir(cfg: TrainConfig, model_dir: Path | None, what: str) -> Path:
    if model_dir is None:
        raise ValueError(f"No {what} file is set and no Hugging Face model to take it from: set the file (model "
                         f"source 'files') or the ideogram-ai/ideogram-4-fp8 repo id / a diffusers folder")
    return Path(model_dir)


def resolve_for_config(cfg: TrainConfig) -> Path | None:
    """Download / locate the HF model dir when any component comes from it (single files cover the rest)."""
    if cfg.model_source == "files" and all(getattr(cfg, f) for f in ("transformer_file", "text_encoder_file", "vae_file")):
        return None
    return resolve_model_dir(cfg.model_path, cfg.models_dir)


def get_transformer(cfg: TrainConfig, model_dir: Path | None, device: str | None = None) -> Ideogram4Transformer:
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
            return isf.load_transformer_file(cfg.transformer_file, "cpu", cfg.transformer_quant, quant_devices=gpus)
        return load_transformer(model_dir, "cpu", cfg.transformer_quant, quant_devices=gpus)
    if use_file:
        return isf.load_transformer_file(cfg.transformer_file, device or cfg.device, cfg.transformer_quant)
    return load_transformer(model_dir, device or cfg.device, cfg.transformer_quant)


def get_vae(cfg: TrainConfig, model_dir: Path | None):
    if moss_loading._use_file(cfg, "vae_file"):
        from ..moss import single_file as msf

        moss_loading._check_files(cfg, "vae_file")
        return msf.load_vae_file(cfg.vae_file, cfg.device)
    return load_vae(_need_dir(cfg, model_dir, "VAE"), cfg.device)


def get_text_encoder(cfg: TrainConfig, model_dir: Path | None):
    if moss_loading._use_file(cfg, "text_encoder_file"):
        from transformers import AutoTokenizer

        from ..qwen21 import single_file as qsf

        moss_loading._check_files(cfg, "text_encoder_file")
        tokenizer = AutoTokenizer.from_pretrained(str(qsf.resolve_tokenizer(cfg.tokenizer_path, cfg.models_dir)))
        model = isf.load_text_encoder_file(cfg.text_encoder_file, cfg.text_encoder_device, cfg.text_encoder_quant)
        return model, tokenizer
    return load_text_encoder(_need_dir(cfg, model_dir, "text encoder"), cfg.text_encoder_device,
                             cfg.text_encoder_quant)


def load_any_transformer(path: str, device, quant: str = "auto", models_dir: str = "models") -> Ideogram4Transformer:
    """Load a transformer from a single .safetensors file, a diffusers directory, or a HF repo id."""
    p = Path(path).expanduser()
    if p.is_file():
        return isf.load_transformer_file(p, device, quant)
    if p.is_dir() and (p / "transformer" / "config.json").is_file():
        return load_transformer(p, device, quant)
    return load_transformer(resolve_model_dir(path, models_dir), device, quant)
