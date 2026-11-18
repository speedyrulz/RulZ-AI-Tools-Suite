"""Load Z-Image (Tongyi-MAI Z-Image Base and Z-Image-Turbo) components from the Hugging Face / diffusers layout or from
ComfyUI single files (Comfy-Org/z_image_turbo, Comfy-Org/z_image; the default), and encode.

    transformer/   config.json (ZImageTransformer2DModel) + diffusion_pytorch_model*.safetensors (diffusers' names)
    text_encoder/  Qwen3-4B (Qwen3Model), sharded model*.safetensors
    tokenizer/     Qwen3's tokenizer
    vae/           AutoencoderKL, the FLUX.1 VAE (16 channels, 8x; scaling 0.3611, shift 0.1159)

Text features, where Tongyi's pipeline, ComfyUI, ai-toolkit, OneTrainer and Fizgig agree: the prompt goes through
Qwen3's chat template as one user turn with the generation prompt and thinking ON (enable_thinking=True), is
tokenized to at most 512 tokens, and the text encoder's `hidden_states[-2]` is kept, before any final norm: the
second last of the 36 layers' outputs, which is the last of 35 built layers (the encoder is built to layer 35 with the
final norm an identity, flux2.single_file.build_text_model(keep=35)). Only the real tokens are kept ([L, 2560]); the
transformer pads each caption itself (zimage.model).

Latents: the FLUX.1 VAE's posterior mean, (z - 0.1159) * 0.3611, then the 2x2 patchify of zimage.model (channel index
c * 4 + ph * 2 + pw): [B, 64, H/16, W/16]. The FLUX.2 VAE has the same file name (ae.safetensors in some repos) and 32
latent channels: it is refused by the shape of decoder.conv_in.weight.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

from .. import quant as Q
from ..config import ZIMAGE_VARIANT_DATA, TrainConfig, resolve_zimage_variant
from ..downloads import component_ready
from ..fibo.loading import place_text_encoder
from ..flux2 import single_file as fsf
from ..flux3.loading import pad_embeds  # noqa: F401 (the same batch: zero-padded rows and a bool mask)
from ..moss import loading as moss_loading
from ..moss import single_file as sf
from . import model as M
from . import single_file as zsf
from .model import ZImageTransformer

log = logging.getLogger("rulztrainer")

TRANSFORMER_CLASSES = ("ZImageTransformer2DModel",)
# quantize the 30 main layers' projections; the pad tokens, embedders, final layer, norms and the refiners stay bf16
QUANT_FILTER = lambda name: name.startswith("layers.") and not M.matches(name, M.FULL_PRECISION)  # noqa: E731
# "auto" by the GPU memory the model will live in (all GPUs of a model split). The DiT is 12.3 GB in bf16 and 6.2 GB in
# int8; Fizgig measured a rank-16 run at 1 MP (gradient checkpointing, preview included): bf16 17.2 GB, int8 12.3 GB,
# nf4 10.1 GB. So bf16 from 22 GB, int8 from 10 GB, else nvfp4 (estimates scaled from those peaks, not measured here)
AUTO_BF16_FROM_GB, AUTO_INT8_FROM_GB = 22, 10
FLUX1_VAE_WIDTH, FLUX1_VAE_CHANNELS = 512, 16  # decoder.conv_in.weight of the FLUX.1 VAE: (512, 16, 3, 3)
VAE_SHIFT, VAE_SCALE = 0.1159, 0.3611
MAX_TOKENS = 512
TE_KEEP = 35  # layers built: the last one is hidden_states[-2] of the 36-layer model (amendment 1)
TE_REPO = "Tongyi-MAI/Z-Image-Turbo"
HF_PATTERNS = ["*.json", "transformer/*", "text_encoder/*", "tokenizer/*", "vae/*", "LICENSE*"]
SAMPLE_PATTERNS = ["*.json", "transformer/*"]


def resolve_quant(quant: str, devices) -> str:
    """`auto`: bf16 with 22 GB+, int8 from 10 GB, else nvfp4, counting the memory of every GPU the model is split over.
    A model loaded on the CPU (a sampling model) is sized for the smallest GPU, where it will run."""
    if quant != "auto" or not torch.cuda.is_available():
        return "none" if quant == "auto" else quant
    from ..flux3.loading import _gpu_gb

    devices = [devices] if isinstance(devices, (str, torch.device)) else list(devices)
    gpus = [d for d in devices if torch.device(d).type == "cuda"]
    if gpus:
        total, where = sum(_gpu_gb(d) for d in gpus), "+".join(str(d) for d in gpus)
    else:
        total = min(_gpu_gb(f"cuda:{i}") for i in range(torch.cuda.device_count()))
        where = "the smallest GPU"
    fmt = "none" if total >= AUTO_BF16_FROM_GB else "int8_convrot" if total >= AUTO_INT8_FROM_GB else "nvfp4"
    if fmt != "none":
        log.info("Z-Image transformer: auto quantization -> %s for %.0f GB (%s)", fmt, total, where)
    return fmt


def resolve_text_quant(quant: str) -> str:
    from ..flux2.loading import resolve_text_quant as resolve

    return resolve(quant, "4b")  # Qwen3-4B: the same encoder as Klein 4B's (bf16 from 12 GB, else int8)


# ---------------------------------------------------------------- transformer


def is_zimage_dir(model_dir: Path) -> bool:
    cfg = Path(model_dir) / "transformer" / "config.json"
    try:
        return json.loads(cfg.read_text(encoding="utf-8")).get("_class_name") in TRANSFORMER_CLASSES
    except (OSError, ValueError):
        return False


def _shape(v) -> tuple:
    return tuple(v.orig_shape if isinstance(v, Q.QuantWeight) else v.shape)


def _count(state: dict, pattern: str) -> int:
    n = 0
    while pattern.format(n) in state:
        n += 1
    return n


def config_from_state(state: dict) -> dict:
    """ZImageTransformer options from a facade-named (maybe quantized) state dict; the released models' values for
    the rest (rope, patch size, ...)."""
    dim = _shape(state["all_x_embedder.2-1.weight"])[0]
    head_dim = _shape(state["layers.0.attention.norm_q.weight"])[0]
    heads = dim // head_dim
    return dict(dim=dim, n_heads=heads, n_kv_heads=heads, n_layers=_count(state, "layers.{}.attention.to_q.weight"),
                n_refiner_layers=_count(state, "noise_refiner.{}.attention.to_q.weight"),
                cap_feat_dim=_shape(state["cap_embedder.1.weight"])[1])


def _split_quant_qkv(state: dict) -> None:
    """A quantized ComfyUI file's fused attention.qkv, split into to_q / to_k / to_v (comfy_to_facade chunks plain
    tensors only)."""
    for k in [k for k, v in state.items() if isinstance(v, Q.QuantWeight) and k.endswith(".attention.qkv.weight")]:
        fused, stem = state.pop(k), k[: -len("qkv.weight")]
        for n, part in zip("qkv", fused.split_rows(3)):
            state[f"{stem}to_{n}.weight"] = part


def to_model_layout(state: dict, name: str = "") -> dict:
    """A ComfyUI-named or diffusers-named state dict (either prefix) -> the facade's names. Omni / Edit files and files
    that are not Z-Image are refused."""
    shapes = {k: _shape(v) for k, v in state.items()}
    kind = zsf.zimage_kind(shapes)
    prefix = f"{name}: " if name else ""
    if kind == "zimage_omni":
        raise ValueError(f"{prefix}this is a Z-Image Omni / Edit file (it has a SigLIP branch); only Z-Image Base and "
                         "Turbo text-to-image models are supported")
    if kind is None:
        raise ValueError(f"{prefix}not a Z-Image diffusion model (needs cap_pad_token, context_refiner.* and a "
                         f"cap_embedder reading {zsf.CAP_FEAT_DIM} text features)")
    _split_quant_qkv(state)
    return zsf.comfy_to_facade(state)


def build_transformer(state: dict, device, quant: str = "auto", dtype=torch.bfloat16, quant_devices=None,
                      variant: str = "base", name: str = "", overrides: dict | None = None) -> ZImageTransformer:
    """A frozen ZImageTransformer from a facade-named state dict (consumed). variant: "turbo" or "base" (sets
    is_distilled and the sampling shift). overrides: options for a model that is not the released size (tests)."""
    config = {**config_from_state(state), **(overrides or {})}
    with torch.device("meta"):
        model = ZImageTransformer(**config)
    zsf.check_keys(state, model.state_dict().keys())  # strict: a ValueError that lists what differs
    if quant == "auto" and not any(isinstance(v, Q.QuantWeight) for v in state.values()):
        quant = resolve_quant("auto", quant_devices or device)
    sf.install_state(model, state, quant, dtype, quant_filter=QUANT_FILTER, **sf.quant_target(device, quant_devices))
    model = model.to(device).eval().requires_grad_(False)
    data = ZIMAGE_VARIANT_DATA[variant]
    model.is_distilled = variant == "turbo"
    model.zimage_shift = data["shift"]
    log.info("Loaded Z-Image transformer from %s: %d layers, dim %d, %s, quant=%s", name or "state", config["n_layers"],
             config["dim"], "Turbo (distilled)" if model.is_distilled else "Base", quant)
    return model


def load_transformer_file(path: str | Path, device, quant: str = "auto", dtype=torch.bfloat16, quant_devices=None,
                          variant: str | None = None) -> ZImageTransformer:
    """ComfyUI's z_image*_bf16.safetensors (or an int8_convrot / nvfp4 one, or a diffusers-named file). variant: None
    reads the file's name (turbo and not de-turbo)."""
    name = Path(path).name
    state = to_model_layout(sf.read_state(path, dtype), name)
    return build_transformer(state, device, quant, dtype, quant_devices,
                             variant or resolve_zimage_variant("auto", name), name)


def load_transformer(model_dir: Path, device, quant: str = "auto", dtype=torch.bfloat16, quant_devices=None,
                     variant: str | None = None) -> ZImageTransformer:
    """From the folder's diffusers transformer/ (its names are the facade's)."""
    config = json.loads((model_dir / "transformer" / "config.json").read_text(encoding="utf-8"))
    if config.get("_class_name") not in TRANSFORMER_CLASSES:
        raise ValueError(f"Not a Z-Image transformer: {config.get('_class_name')!r}")
    state = moss_loading._read_sharded(model_dir / "transformer", "diffusion_pytorch_model")
    state = to_model_layout({k: (v.to(dtype) if v.is_floating_point() else v) for k, v in state.items()},
                            model_dir.name)
    return build_transformer(state, device, quant, dtype, quant_devices,
                             variant or resolve_zimage_variant("auto", model_dir.name), model_dir.name)


# ---------------------------------------------------------------- text encoder (Qwen3-4B) and tokenizer


def build_text_encoder(state: dict, quant: str = "none", work_device=None):
    """Qwen3-4B's text model cut at layer 35 without the final norm (hidden_states[-1] is the full model's [-2])."""
    return fsf.build_text_model(state, quant, work_device, fsf.TEXT_CONFIG_4B, keep=TE_KEEP)


def load_text_encoder_file(path: str | Path, device: str = "auto", quant: str = "auto"):
    """Comfy-Org's qwen_3_4b.safetensors (or the fp8_mixed file, read up to bf16)."""
    fmt = resolve_text_quant(quant)
    state = sf.read_state(path, torch.bfloat16)
    if not any(k.endswith("layers.0.self_attn.q_proj.weight") for k in state):
        raise ValueError(f"{Path(path).name} is not a Qwen3 text encoder")
    embed = state.get("model.embed_tokens.weight", state.get("embed_tokens.weight"))
    if embed is not None and int(embed.shape[-1]) != int(fsf.TEXT_CONFIG_4B["hidden_size"]):
        raise ValueError(f"{Path(path).name} is not Qwen3-4B (width {int(embed.shape[-1])}; Z-Image reads 2560)")
    model = build_text_encoder(state, fmt, sf.quant_target(device).get("work_device"))
    del state
    return place_text_encoder(model, device)


def resolve_tokenizer(tokenizer_path: str, models_dir: str = "models") -> Path:
    """The Qwen3 tokenizer folder: the given one, a downloaded Z-Image repo's tokenizer/, Tongyi-MAI/Z-Image-Turbo's
    tokenizer/ (a few MB, downloaded once), else what Klein 4B's single-file runs use (Qwen/Qwen3-4B's)."""
    if tokenizer_path:
        return fsf.resolve_tokenizer(tokenizer_path, models_dir, "4b")
    for cand in sorted(Path(models_dir).glob("Tongyi-MAI__Z-Image*/tokenizer/tokenizer_config.json")):
        return cand.parent
    local = Path(models_dir) / f"tokenizer__{TE_REPO.split('/')[-1]}"
    if not (local / "tokenizer" / "tokenizer_config.json").is_file():
        try:
            from huggingface_hub import snapshot_download

            log.info("Downloading the Z-Image tokenizer to %s", local)
            snapshot_download(TE_REPO, local_dir=str(local), allow_patterns=["tokenizer/*"])
        except Exception as e:  # noqa: BLE001
            log.warning("Z-Image tokenizer download failed (%s: %s); using Qwen/Qwen3-4B's", type(e).__name__, e)
            return fsf.resolve_tokenizer("", models_dir, "4b")
    return local / "tokenizer"


def load_text_encoder(model_dir: Path, device: str = "auto", quant: str = "auto"):
    """Qwen3-4B's text model + tokenizer from text_encoder/ and tokenizer/."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir / "tokenizer"))
    fmt = resolve_text_quant(quant)
    state = moss_loading._read_sharded(model_dir / "text_encoder", "model")
    model = build_text_encoder(state, fmt, sf.quant_target(device).get("work_device"))
    if fmt != "none":
        n = sum(1 for m in model.modules() if isinstance(m, Q.QuantLinear))
        log.info("Text encoder: %d Linear layers quantized to %s", n, fmt)
    return place_text_encoder(model, device), tokenizer


def _input_device(model) -> torch.device:
    return model.get_input_embeddings().weight.device


def chat_text(tokenizer, prompt: str) -> str:
    """Qwen3's chat template for one user prompt with the generation prompt and thinking on (what Tongyi's pipeline,
    ComfyUI and ai-toolkit feed the encoder)."""
    tok = getattr(tokenizer, "tokenizer", tokenizer)
    return tok.apply_chat_template([{"role": "user", "content": prompt or ""}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=True)


def tokenize(tokenizer, prompt: str, max_length: int = MAX_TOKENS) -> list[int]:
    """The templated prompt's token ids, cut at max_length (512 at most)."""
    tok = getattr(tokenizer, "tokenizer", tokenizer)
    return list(tok(chat_text(tokenizer, prompt), truncation=True, max_length=min(int(max_length), MAX_TOKENS),
                    add_special_tokens=False).input_ids)


@torch.no_grad()
def encode_prompts(text_encoder, tokenizer, prompts: list[str], max_length: int = MAX_TOKENS) -> list[torch.Tensor]:
    """Text features for each prompt: [L, 2560] bf16 on the CPU, L its real tokens (at most 512), in one forward
    (right-padded to 512 and masked, so a row's tokens do not depend on the others)."""
    tok = getattr(tokenizer, "tokenizer", tokenizer)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    rows = [tokenize(tokenizer, p, max_length) for p in prompts]
    n = max(len(r) for r in rows)
    ids = torch.full((len(rows), n), int(pad), dtype=torch.long)
    mask = torch.zeros(len(rows), n, dtype=torch.long)
    for i, r in enumerate(rows):
        ids[i, : len(r)] = torch.tensor(r, dtype=torch.long)
        mask[i, : len(r)] = 1
    device = _input_device(text_encoder)
    out = text_encoder(input_ids=ids.to(device), attention_mask=mask.to(device), output_hidden_states=True,
                       use_cache=False)
    hidden = out.hidden_states[-1].to(torch.bfloat16).cpu()  # the 35th layer's raw output (see the module docstring)
    return [hidden[i, : len(r)].clone() for i, r in enumerate(rows)]


def encode_prompt(text_encoder, tokenizer, prompt: str, max_length: int = MAX_TOKENS) -> torch.Tensor:
    return encode_prompts(text_encoder, tokenizer, [prompt], max_length)[0]


def get_tokenizer(cfg: TrainConfig, model_dir: Path | None):
    from transformers import AutoTokenizer

    if moss_loading._use_file(cfg, "text_encoder_file") or model_dir is None:
        return AutoTokenizer.from_pretrained(str(resolve_tokenizer(cfg.tokenizer_path, cfg.models_dir)))
    return AutoTokenizer.from_pretrained(str(model_dir / "tokenizer"))


def prompt_tokens(tokenizer, prompt: str, max_length: int = MAX_TOKENS) -> tuple[str, list, int, int]:
    """encode_prompt's text (the whole chat template) with its token offsets, cut where encode_prompt cuts; the cache
    keeps exactly these tokens."""
    tok = getattr(tokenizer, "tokenizer", tokenizer)
    text = chat_text(tokenizer, prompt)
    offsets = tok(text, return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]
    offsets = offsets[: min(int(max_length), MAX_TOKENS)]
    return text, [tuple(o) for o in offsets], 0, len(offsets)


# ---------------------------------------------------------------- VAE (FLUX.1), the trainer's layout


def check_flux1_vae_shape(shape, width: int = FLUX1_VAE_WIDTH, source: str = "this VAE") -> None:
    """Refuse a VAE whose decoder.conv_in.weight is not the FLUX.1 VAE's (512, 16, 3, 3): the FLUX.2 VAE (32 latent
    channels) and others share the file name ae.safetensors."""
    shape = tuple(shape)
    if shape != (width, FLUX1_VAE_CHANNELS, 3, 3):
        hint = " (that is the FLUX.2 VAE, 32 latent channels)" if len(shape) == 4 and shape[1] == 32 else ""
        raise ValueError(f"{source} is not the FLUX.1 VAE Z-Image uses: decoder.conv_in.weight is {shape}, expected "
                         f"{(width, FLUX1_VAE_CHANNELS, 3, 3)}{hint}")


def load_vae_file(path: str | Path, device, config: dict | None = None):
    """The FLUX.1 VAE (BFL's / ComfyUI's ae.safetensors, or a diffusers-layout file) as an fp32 AutoencoderKL, frozen.
    config replaces the bundled one (tests build a tiny VAE)."""
    from diffusers import AutoencoderKL

    name = Path(path).name
    state = sf.strip_prefix(sf.read_state(path, torch.float32), ("vae.", "first_stage_model."))
    if any(isinstance(v, Q.QuantWeight) for v in state.values()):
        raise ValueError("Quantized VAE files are not supported; use the fp32 / bf16 FLUX.1 VAE")
    config = config or json.loads((Path(__file__).resolve().parent / "vae_config.json").read_text(encoding="utf-8"))
    conv = state.get("decoder.conv_in.weight")
    if conv is None:
        raise ValueError(f"{name} is not a VAE (no decoder.conv_in.weight)")
    check_flux1_vae_shape(conv.shape, int(config["block_out_channels"][-1]), name)
    if "decoder.up.0.block.0.conv1.weight" in state:  # the original BFL / ComfyUI key layout (ae.safetensors)
        state = sf.vae_original_to_diffusers(state)
    if "decoder.up_blocks.0.resnets.0.conv1.weight" not in state:
        raise ValueError(f"{name}: unrecognized VAE key layout (neither diffusers nor original)")
    vae = sf.assign_on_meta(lambda: AutoencoderKL.from_config(config), state, name)
    return vae.to(device, torch.float32).eval().requires_grad_(False)


def load_vae(model_dir: Path, device):
    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(str(model_dir / "vae"), torch_dtype=torch.float32)
    check_flux1_vae_shape(vae.decoder.conv_in.weight.shape, source=f"{model_dir.name}/vae")
    return vae.to(device).eval().requires_grad_(False)


def _stats(vae) -> tuple[float, float]:
    shift = getattr(vae.config, "shift_factor", None)
    scale = getattr(vae.config, "scaling_factor", None)
    return (VAE_SHIFT if shift is None else float(shift)), (VAE_SCALE if scale is None else float(scale))


@torch.no_grad()
def encode_images(vae, images: torch.Tensor) -> torch.Tensor:
    """images [B,3,H,W] in [-1,1] (H, W multiples of 16) -> (posterior mean - 0.1159) * 0.3611, patchified
    [B,64,H/16,W/16] (fp32)."""
    z = vae.encode(images.to(vae.device, vae.dtype)).latent_dist.mode().float()
    shift, scale = _stats(vae)
    return M.patchify((z - shift) * scale)


def _decode(vae, latents: torch.Tensor) -> torch.Tensor:
    shift, scale = _stats(vae)
    z = M.unpatchify(latents.to(vae.device, torch.float32)) / scale + shift  # the statistics apply in fp32
    return vae.decode(z.to(vae.dtype), return_dict=False)[0]


@torch.no_grad()
def decode_latents(vae, latents: torch.Tensor) -> torch.Tensor:
    """Inverse of encode_images: images [B,3,H,W] in [-1,1] (clamped), decoded in fp32."""
    return _decode(vae, latents).float().clamp(-1, 1)


def decode_latents_grad(vae, latents: torch.Tensor) -> torch.Tensor:
    """decode_latents with gradients to `latents` (identity loss on a face crop), not clamped."""
    return _decode(vae, latents).float()


def grad_vae(vae):
    """A bf16 copy of the VAE with gradient checkpointing, for decode_latents_grad."""
    out = moss_loading.bf16_copy(vae)
    out.enable_gradient_checkpointing()
    return out


# ---------------------------------------------------------------- download / config-driven entry points


def resolve_model_dir(model_path: str, models_dir: str = "models", patterns: list[str] | None = None) -> Path:
    """A local folder for a repo id or path, downloading what the trainer reads (about 21 GB; patterns: fewer, a
    sampling model needs its transformer alone, 12 GB)."""
    p = Path(model_path).expanduser()
    if p.is_dir():
        return p.resolve()
    local = Path(models_dir) / model_path.replace("/", "__")
    if component_ready(local / "transformer") and (patterns is not None or (
            component_ready(local / "text_encoder") and component_ready(local / "vae"))):
        return local.resolve()
    from huggingface_hub import snapshot_download

    log.info("Downloading %s to %s (%s)...", model_path, local, "the transformer, about 12 GB" if patterns else
             "transformer, text encoder and VAE, about 21 GB")
    snapshot_download(model_path, local_dir=str(local), allow_patterns=patterns or HF_PATTERNS)
    return local.resolve()


def _need_dir(cfg: TrainConfig, model_dir: Path | None, what: str) -> Path:
    if model_dir is None:
        raise ValueError(f"No {what} file is set and no Hugging Face model to take it from: set the file (model "
                         "source 'files') or the Tongyi-MAI/Z-Image-Turbo / Tongyi-MAI/Z-Image repo id / a folder")
    return model_dir


def resolve_for_config(cfg: TrainConfig) -> Path | None:
    """Download / locate the HF model dir when any component comes from it (single files cover the rest)."""
    if cfg.model_source == "files" and all(getattr(cfg, f) for f in ("transformer_file", "text_encoder_file", "vae_file")):
        return None
    return resolve_model_dir(cfg.model_path, cfg.models_dir)


def get_transformer(cfg: TrainConfig, model_dir: Path | None, device: str | None = None) -> ZImageTransformer:
    use_file = moss_loading._use_file(cfg, "transformer_file")
    if use_file:
        moss_loading._check_files(cfg, "transformer_file")
    else:
        model_dir = _need_dir(cfg, model_dir, "diffusion model")
    variant = cfg.zimage_variant_resolved()
    log.info("Z-Image variant: %s (zimage_variant %s)", variant, cfg.zimage_variant)
    if cfg.multi_gpu == "model_split" and device is None:
        # the trainer splits it right after loading: stay on the CPU and size "auto" for all the split GPUs
        from ..multigpu import parse_devices

        gpus = parse_devices(cfg.devices, cfg.device)
        if use_file:
            return load_transformer_file(cfg.transformer_file, "cpu", cfg.transformer_quant, quant_devices=gpus,
                                         variant=variant)
        return load_transformer(model_dir, "cpu", cfg.transformer_quant, quant_devices=gpus, variant=variant)
    with sf.quantize_staged_on(sf.staging_gpu(cfg, device)):  # block swap: quantized on the GPU, kept in RAM
        if use_file:
            return load_transformer_file(cfg.transformer_file, device or cfg.device, cfg.transformer_quant,
                                         variant=variant)
        return load_transformer(model_dir, device or cfg.device, cfg.transformer_quant, variant=variant)


def get_vae(cfg: TrainConfig, model_dir: Path | None):
    if moss_loading._use_file(cfg, "vae_file"):
        moss_loading._check_files(cfg, "vae_file")
        return load_vae_file(cfg.vae_file, cfg.device)
    return load_vae(_need_dir(cfg, model_dir, "VAE"), cfg.device)


def get_text_encoder(cfg: TrainConfig, model_dir: Path | None):
    if moss_loading._use_file(cfg, "text_encoder_file"):
        from transformers import AutoTokenizer

        moss_loading._check_files(cfg, "text_encoder_file")
        tokenizer = AutoTokenizer.from_pretrained(str(resolve_tokenizer(cfg.tokenizer_path, cfg.models_dir)))
        return load_text_encoder_file(cfg.text_encoder_file, cfg.text_encoder_device, cfg.text_encoder_quant), tokenizer
    return load_text_encoder(_need_dir(cfg, model_dir, "text encoder"), cfg.text_encoder_device,
                             cfg.text_encoder_quant)


def load_any_transformer(path: str, device, quant: str = "auto", models_dir: str = "models") -> ZImageTransformer:
    """Load a transformer (Base or Turbo, as the sampling model) from a single .safetensors file, a folder, or a HF repo
    id (its transformer/ alone is downloaded). Turbo or Base by its name (de-turbo is Base)."""
    p = Path(path).expanduser()
    if p.is_file():
        return load_transformer_file(p, device, quant)
    if p.is_dir():
        return load_transformer(p, device, quant)
    return load_transformer(resolve_model_dir(path, models_dir, SAMPLE_PATTERNS), device, quant,
                            variant=resolve_zimage_variant("auto", path))
