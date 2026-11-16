"""Load SDXL from one all-in-one checkpoint (the A1111 / ComfyUI / kohya layout) with diffusers' own converters.

    model.diffusion_model.*                 the UNet (LDM names: input_blocks, middle_block, output_blocks, label_emb)
    conditioner.embedders.0.transformer.*   CLIP-L, already in Hugging Face's CLIPTextModel names
    conditioner.embedders.1.model.*         OpenCLIP bigG (resblocks, in_proj_weight, text_projection)
    first_stage_model.*                     the VAE

The converters come from diffusers.loaders.single_file_utils (convert_ldm_unet_checkpoint,
convert_ldm_vae_checkpoint, convert_ldm_clip_checkpoint, convert_open_clip_checkpoint); the model configs are the
JSON files bundled in this package, so nothing is fetched from the network. Only the keys a component needs are read
from the file. A bare UNet or VAE file (no `model.diffusion_model.` / `first_stage_model.` prefix) loads too.

Prediction type: a checkpoint that carries the (empty) `v_pred` tensor is a v-prediction model, and `ztsnr` marks zero
terminal SNR (the A1111 / ComfyUI convention); a `modelspec.prediction_type` metadata entry is read as well."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import torch

log = logging.getLogger("rulztrainer")
HERE = Path(__file__).resolve().parent

UNET_PREFIX = "model.diffusion_model."
CLIP_L_PREFIX = "conditioner.embedders.0.transformer."
CLIP_G_PREFIX = "conditioner.embedders.1.model."
VAE_PREFIX = "first_stage_model."
MARKER = "label_emb.0.0.weight"


def _config(name: str) -> dict:
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def _keys(path: str | Path) -> list[str]:
    from safetensors import safe_open

    with safe_open(str(path), "pt") as f:
        return list(f.keys())


def is_sdxl_file(path: str | Path) -> bool:
    """An SDXL UNet in the file: label_emb.0.0.weight (the pooled-text / size embedding SD 1.x/2.x lack) beside
    input_blocks, under model.diffusion_model. or bare. False for anything unreadable."""
    try:
        keys = _keys(path)
    except Exception:  # noqa: BLE001 - not a safetensors file
        return False
    ks = set(keys)
    if UNET_PREFIX + MARKER not in ks and MARKER not in ks:
        return False
    return any(k.startswith((UNET_PREFIX + "input_blocks.", "input_blocks.")) for k in keys)


def file_prediction(path: str | Path) -> tuple[str | None, bool | None]:
    """(prediction_type | None, zero_terminal_snr | None) from the `v_pred` / `ztsnr` marker tensors (or the
    modelspec metadata). None means the file does not say."""
    from safetensors import safe_open

    pred, ztsnr = None, None
    with safe_open(str(path), "pt") as f:
        keys = set(f.keys())
        meta = f.metadata() or {}
    if "v_pred" in keys:
        pred = "v_prediction"
    if "ztsnr" in keys:
        ztsnr = True
    spec = str(meta.get("modelspec.prediction_type", "")).lower()
    if pred is None and spec in ("v", "v_prediction"):
        pred = "v_prediction"
    elif pred is None and spec in ("epsilon", "eps"):
        pred = "epsilon"
    return pred, ztsnr


def _read(path: str | Path, prefixes: tuple[str, ...], dtype=None) -> dict[str, torch.Tensor]:
    """The tensors whose key starts with one of the prefixes (loaded one by one), cast to `dtype` if given."""
    from safetensors import safe_open

    out = {}
    with safe_open(str(path), "pt") as f:
        for k in f.keys():
            if k.startswith(prefixes):
                t = f.get_tensor(k)
                out[k] = t.to(dtype) if dtype is not None and t.is_floating_point() else t
    return out


def _fill(model, state: dict, what: str, allow_missing: tuple[str, ...] = ()):
    """load_state_dict(strict) for a model built on the meta device, refusing leftovers on meta."""
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    missing = [k for k in missing if not k.endswith(allow_missing)] if allow_missing else list(missing)
    if missing or unexpected:
        raise ValueError(f"{what}: the file does not match (missing {missing[:6]}{'...' if len(missing) > 6 else ''}, "
                         f"unexpected {list(unexpected)[:6]}{'...' if len(unexpected) > 6 else ''})")
    return model


def load_unet_file(path: str | Path, device, dtype=torch.bfloat16, config: dict | None = None):
    """The UNet of an all-in-one (or bare) SDXL checkpoint as a diffusers UNet2DConditionModel, frozen. `config`
    replaces the bundled SDXL config (tests build a tiny one)."""
    from diffusers import UNet2DConditionModel
    from diffusers.loaders.single_file_utils import convert_ldm_unet_checkpoint

    keys = _keys(path)
    bare = not any(k.startswith(UNET_PREFIX) for k in keys)
    state = _read(path, ("input_blocks.", "middle_block.", "output_blocks.", "time_embed.", "label_emb.", "out.")
                  if bare else (UNET_PREFIX,), dtype)
    if bare:
        state = {UNET_PREFIX + k: v for k, v in state.items()}
    if UNET_PREFIX + MARKER not in state:
        raise ValueError(f"{Path(path).name} is not an SDXL checkpoint (no {MARKER})")
    config = config or _config("unet_config.json")
    converted = convert_ldm_unet_checkpoint(state, config)
    del state
    with torch.device("meta"):
        unet = UNet2DConditionModel.from_config(config)
    _fill(unet, converted, "SDXL UNet")
    return unet.to(device).eval().requires_grad_(False)


def load_vae_file(path: str | Path, device, config: dict | None = None):
    """The VAE of an all-in-one checkpoint (first_stage_model.*), a bare LDM VAE file, or a diffusers-layout VAE
    file, as an fp32 AutoencoderKL, frozen."""
    from diffusers import AutoencoderKL
    from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint

    keys = _keys(path)
    config = config or _config("vae_config.json")
    if any(k.startswith(VAE_PREFIX) for k in keys):
        state = _read(path, (VAE_PREFIX,), torch.float32)
    elif any(k.startswith("encoder.down_blocks.") for k in keys):
        state = _read(path, ("encoder.", "decoder.", "quant_conv.", "post_quant_conv."), torch.float32)
        converted = state
    else:
        state = _read(path, ("encoder.", "decoder.", "quant_conv.", "post_quant_conv."), torch.float32)
    if not state:
        raise ValueError(f"{Path(path).name} holds no VAE (no {VAE_PREFIX}* or encoder./decoder. keys)")
    if not any(k.startswith("encoder.down_blocks.") for k in state):
        converted = convert_ldm_vae_checkpoint(state, config)
    del state
    with torch.device("meta"):
        vae = AutoencoderKL.from_config(config)
    _fill(vae, converted, "SDXL VAE")
    return vae.to(device).eval().requires_grad_(False)


def _build_clip(cls, config: dict, state: dict, dtype, what: str):
    """A CLIP text model with real (non-meta) buffers, loaded from a converted state dict."""
    from transformers import CLIPTextConfig

    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        model = cls(CLIPTextConfig.from_dict(config))
    finally:
        torch.set_default_dtype(prev)
    # transformers 5 keeps CLIPTextModel flat (embeddings.*, encoder.*) but CLIPTextModelWithProjection under
    # text_model.*; diffusers' converters always write text_model.* (as real checkpoints hold CLIP-L): match the model
    own = {k.split(".", 1)[0] for k in model.state_dict()}
    if "text_model" not in own:
        state = {k.removeprefix("text_model."): v for k, v in state.items()}
    elif not any(k.startswith("text_model.") for k in state):
        state = {(k if k.split(".", 1)[0] in own else "text_model." + k): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if not k.endswith("position_ids")]
    unexpected = [k for k in unexpected if not k.endswith("position_ids")]
    if missing or unexpected:
        raise ValueError(f"{what}: the file does not match (missing {missing[:6]}, unexpected {unexpected[:6]})")
    return model.eval().requires_grad_(False)


def load_text_encoders_file(path: str | Path, device, dtype=torch.bfloat16, configs: tuple | None = None):
    """(CLIPTextModel, CLIPTextModelWithProjection): CLIP-L and OpenCLIP bigG of an all-in-one checkpoint, frozen."""
    from diffusers.loaders.single_file_utils import convert_ldm_clip_checkpoint, convert_open_clip_checkpoint
    from transformers import CLIPTextModel, CLIPTextModelWithProjection

    raw_l = _read(path, (CLIP_L_PREFIX,), dtype)
    raw_g = _read(path, (CLIP_G_PREFIX,), dtype)
    if not raw_l or not raw_g:
        raise ValueError(f"{Path(path).name} holds no SDXL text encoders (no {CLIP_L_PREFIX}* / {CLIP_G_PREFIX}*)")
    conf_l, conf_g = configs or (_config("clip_l_config.json"), _config("clip_g_config.json"))
    clip_l = _build_clip(CLIPTextModel, conf_l, convert_ldm_clip_checkpoint(raw_l), dtype, "CLIP-L")
    holder = SimpleNamespace(config=SimpleNamespace(hidden_size=conf_g["hidden_size"]))  # only read if the file lacks
    state_g = convert_open_clip_checkpoint(holder, raw_g, prefix=CLIP_G_PREFIX)  # its text_projection
    clip_g = _build_clip(CLIPTextModelWithProjection, conf_g, state_g, dtype, "OpenCLIP bigG")
    return clip_l.to(device), clip_g.to(device)
