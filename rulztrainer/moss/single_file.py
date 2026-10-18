"""Load MOSS-Image components from single ComfyUI-style safetensors files, with optional quantization.

Weight formats accepted: bf16 / fp16 / fp32, fp8 (upcast), ComfyUI int8_tensorwise (with or without
ConvRot) and ComfyUI nvfp4. A requested quant format ("auto" keeps what the file has, "none" dequantizes,
or int8 / int8_convrot / nvfp4) is applied when the model is built.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn

from .. import quant as Q
from .model import MOSSImageTransformer

log = logging.getLogger("rulztrainer")
HERE = Path(__file__).resolve().parent
FLOAT8 = {getattr(torch, n) for n in ("float8_e4m3fn", "float8_e5m2") if hasattr(torch, n)}


def read_state(path: str | Path, dtype=torch.bfloat16) -> dict[str, torch.Tensor | Q.QuantWeight]:
    """Read a safetensors file. Quantized ComfyUI layers come back as QuantWeight; floats as `dtype` on CPU."""
    state: dict = {}
    quant_meta: dict[str, dict] = {}
    with safe_open(str(path), "pt") as f:
        keys = list(f.keys())
        for k in keys:
            if k.endswith(".comfy_quant"):
                quant_meta[k[: -len(".comfy_quant")]] = Q.parse_comfy_quant(f.get_tensor(k))
        for k in keys:
            if k.endswith((".comfy_quant", ".weight_scale", ".weight_scale_2")):
                continue
            t = f.get_tensor(k)
            base = k[: -len(".weight")] if k.endswith(".weight") else None
            if base is not None and base in quant_meta:
                s2 = f.get_tensor(f"{base}.weight_scale_2") if f"{base}.weight_scale_2" in keys else None
                state[k] = Q.quant_weight_from_comfy(quant_meta[base], t, f.get_tensor(f"{base}.weight_scale"), s2)
                continue
            if t.dtype in FLOAT8:
                t = t.to(torch.float32)
            if t.is_floating_point():
                t = t.to(dtype)
            state[k] = t
    return state


def strip_prefix(state: dict, prefixes: tuple[str, ...]) -> dict:
    for p in prefixes:
        if any(k.startswith(p) for k in state):
            return {k[len(p):] if k.startswith(p) else k: v for k, v in state.items()}
    return state


def install_state(model: nn.Module, state: dict, quant: str, dtype=torch.bfloat16, min_dim: int = 1024,
                  ignore_missing: tuple[str, ...] = (), quant_filter=None) -> None:
    """Load `state` into a meta-device model, replacing Linear layers with QuantLinear where appropriate.

    quant: "auto" keeps file formats (bf16 stays bf16), "none" dequantizes everything,
    int8 / int8_convrot / nvfp4 converts every eligible Linear (passing quant_filter) to that format.
    """
    state = dict(state)
    n_quant = 0
    handled: set[str] = set()
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        wkey, bkey = f"{name}.weight", f"{name}.bias"
        if wkey not in state:
            continue
        w = state.pop(wkey)
        b = state.pop(bkey, None)
        handled.add(name)
        target = None
        if isinstance(w, Q.QuantWeight):
            target = w.fmt if quant == "auto" else quant
            if target != w.fmt:
                w = w.dequantize(torch.float32) if target == "none" else Q.quantize_weight(w.dequantize(torch.float32), target)
        elif quant not in ("auto", "none") and Q.can_quantize(w, quant, min_dim)                 and (quant_filter is None or quant_filter(name)):
            w = Q.quantize_weight(w, quant)
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        if isinstance(w, Q.QuantWeight):
            setattr(parent, child, Q.QuantLinear(w, b, dtype))
            n_quant += 1
        else:
            lin = nn.Linear(w.shape[1], w.shape[0], bias=b is not None, device="meta", dtype=dtype)
            lin.weight = nn.Parameter(w.to(dtype), requires_grad=False)
            if b is not None:
                lin.bias = nn.Parameter(b.to(dtype), requires_grad=False)
            setattr(parent, child, lin)
    # anything quantized that is not a Linear weight (e.g. an int8 embedding table) is dequantized to `dtype`
    for k, v in list(state.items()):
        if isinstance(v, Q.QuantWeight):
            state[k] = v.dequantize(dtype)
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    missing = [k for k in missing if k.rpartition(".")[0] not in handled and not k.endswith(ignore_missing)]
    if missing or unexpected:
        raise ValueError(f"State mismatch. missing={missing[:5]} unexpected={unexpected[:5]}")
    if n_quant:
        log.info("Installed %d quantized Linear layers (%s)", n_quant, quant)


# ---------------------------------------------------------------- transformer

# Quantize only the main block projections (q/k/v/o/ffn), as ComfyUI's repacked files do; keep the
# embedders, time/modulation MLPs and the two text-projection blocks in bf16.
DIT_QUANT_FILTER = lambda name: name.startswith("layers.")  # noqa: E731


def transformer_config_from_state(state: dict) -> dict:
    def shape(k):
        v = state[k]
        return v.orig_shape if isinstance(v, Q.QuantWeight) else tuple(v.shape)

    dim = shape("shared_block_modulation.weight")[0] // 6
    return dict(
        dim=dim, ffn_dim=shape("layers.0.proj.ffn_up_gate.weight")[0] // 2, num_heads=dim // 128,
        num_layers=max(int(k.split(".")[1]) for k in state if k.startswith("layers.")) + 1,
        text_dim=shape("text_embedder.proj.weight")[1], freq_dim=shape("time_embedding.0.weight")[1],
        latent_channels=shape("final_linear_video.weight")[0], rope_base=10000.0,
    )


def unfuse_qkv(state: dict) -> None:
    for k in [k for k in state if k.endswith(".proj.qkv.weight")]:
        v = state.pop(k)
        stem = k[: -len("qkv.weight")]
        parts = v.split_rows(3) if isinstance(v, Q.QuantWeight) else v.chunk(3, 0)
        for n, p in zip("qkv", parts):
            state[stem + f"{n}.weight"] = p


def load_transformer_file(path: str | Path, device, quant: str = "auto", dtype=torch.bfloat16) -> MOSSImageTransformer:
    state = read_state(path, dtype)
    state = strip_prefix(state, ("model.diffusion_model.", "diffusion_model."))
    state = {k: v for k, v in state.items() if not k.endswith("comfy_attention.config")}
    if "shared_block_modulation.weight" not in state or "layers.0.proj.ffn_up_gate.weight" not in state:
        raise ValueError(f"{Path(path).name} does not look like a MOSS-Image diffusion model")
    unfuse_qkv(state)
    state.pop("rope.bands", None)
    config = transformer_config_from_state(state)
    with torch.device("meta"):
        model = MOSSImageTransformer.from_config(config)
    install_state(model, state, quant, dtype, ignore_missing=("rope.bands",), quant_filter=DIT_QUANT_FILTER)
    model.rope.bands = model.rope._default_bands(torch.device("cpu"))
    model = model.to(device).eval().requires_grad_(False)
    log.info("Loaded transformer from %s: %d layers, dim %d, quant=%s", Path(path).name, config["num_layers"],
             config["dim"], quant)
    return model


def transformer_from_hf_state(state: dict[str, torch.Tensor], config: dict, device, quant: str = "none",
                              dtype=torch.bfloat16) -> MOSSImageTransformer:
    """Build from diffusers-layout weights (already read), optionally quantizing on load."""
    with torch.device("meta"):
        model = MOSSImageTransformer.from_config(config)
    state = dict(state)
    state.pop("rope.bands", None)
    install_state(model, state, "none" if quant == "auto" else quant, dtype, ignore_missing=("rope.bands",),
                  quant_filter=DIT_QUANT_FILTER)
    model.rope.bands = model.rope._default_bands(torch.device("cpu"))
    return model.to(device).eval().requires_grad_(False)


# ---------------------------------------------------------------- VAE


def load_vae_file(path: str | Path, device):
    from diffusers import AutoencoderKLFlux2

    state = read_state(path, torch.float32)
    if any(isinstance(v, Q.QuantWeight) for v in state.values()):
        raise ValueError("Quantized VAE files are not supported; use the fp32/bf16 Flux2 VAE")
    state = strip_prefix(state, ("vae.", "first_stage_model."))
    if "encoder.conv_in.weight" not in state or "bn.running_mean" not in state:
        raise ValueError(f"{Path(path).name} is not a Flux2 VAE in diffusers key format")
    config = json.loads((HERE / "flux2_vae_config.json").read_text())
    enc = [state[f"encoder.down_blocks.{i}.resnets.0.conv1.weight"].shape[0] for i in range(8)
           if f"encoder.down_blocks.{i}.resnets.0.conv1.weight" in state]
    dec = [state[f"decoder.up_blocks.{i}.resnets.0.conv1.weight"].shape[0] for i in range(8)
           if f"decoder.up_blocks.{i}.resnets.0.conv1.weight" in state]
    if enc:
        config["block_out_channels"] = enc
        config["down_block_types"] = ["DownEncoderBlock2D"] * len(enc)
    if dec:
        config["decoder_block_out_channels"] = list(reversed(dec))
        config["up_block_types"] = ["UpDecoderBlock2D"] * len(dec)
    config["latent_channels"] = state["decoder.conv_in.weight"].shape[1]
    import inspect

    accepted = set(inspect.signature(AutoencoderKLFlux2.__init__).parameters)
    config = {k: v for k, v in config.items() if k in accepted}  # diffusers versions differ in accepted keys
    vae = AutoencoderKLFlux2(**config)
    missing, unexpected = vae.load_state_dict(state, strict=False)
    missing = [k for k in missing if "num_batches_tracked" not in k]
    if missing or unexpected:
        raise ValueError(f"VAE state mismatch. missing={missing[:5]} unexpected={unexpected[:5]}")
    return vae.to(device, torch.float32).eval().requires_grad_(False)


# ---------------------------------------------------------------- text encoder


TE_PREFIXES = ("model.language_model.", "language_model.", "text_encoders.qwen35_9b.transformer.model.", "model.")
TE_DROP = ("lm_head.", "visual.", "model.visual.", "mtp.", "model.mtp.", "language_model.mtp.",
           "model.language_model.mtp.")


def build_text_model(state: dict, quant: str = "auto", device="cpu"):
    """Text-only Qwen3.5-9B from an HF-style (possibly prefixed, possibly quantized) state dict."""
    from transformers import Qwen3_5TextConfig, Qwen3_5TextModel

    cfg = Qwen3_5TextConfig(**json.loads((HERE / "qwen35_9b_text_config.json").read_text()))
    state = {k: v for k, v in state.items() if not k.startswith(TE_DROP)}
    state = strip_prefix(state, TE_PREFIXES)
    state = {k: v for k, v in state.items() if not k.startswith("mtp.")}
    with torch.device("meta"):
        model = Qwen3_5TextModel(cfg)
    install_state(model, state, quant, torch.bfloat16, ignore_missing=("inv_freq",))
    # rotary buffers were created on the meta device; rebuild them for real (constructor signature varies by version)
    import inspect

    rot_cls = type(model.rotary_emb)
    kwargs = {"device": "cpu"} if "device" in inspect.signature(rot_cls.__init__).parameters else {}
    model.rotary_emb = rot_cls(config=cfg, **kwargs)
    for name, p in model.named_parameters():
        if p.device.type == "meta":
            raise ValueError(f"Text encoder parameter {name} was not loaded")
    return model.to(device).eval().requires_grad_(False)


def load_text_encoder_file(path: str | Path, device: str = "auto", quant: str = "auto"):
    state = read_state(path, torch.bfloat16)
    model = build_text_model(state, quant, "cpu")
    del state
    return place_text_encoder(model, device)


def place_text_encoder(model, device: str):
    if device == "auto":
        from accelerate import dispatch_model, infer_auto_device_map

        dmap = infer_auto_device_map(model, dtype=torch.bfloat16, no_split_module_classes=["Qwen3_5DecoderLayer"])
        log.info("Text encoder device map: %s", sorted({str(v) for v in dmap.values()}))
        return dispatch_model(model, device_map=dmap)
    return model.to(device)


def resolve_tokenizer(tokenizer_path: str, models_dir: str = "models") -> Path:
    """Local tokenizer dir; downloads the MOSS-Image tokenizer folder (small) if none is given."""
    if tokenizer_path:
        p = Path(tokenizer_path).expanduser()
        if (p / "tokenizer_config.json").is_file():
            return p
        if (p / "tokenizer" / "tokenizer_config.json").is_file():
            return p / "tokenizer"
        raise FileNotFoundError(f"No tokenizer_config.json under {p}")
    for cand in Path(models_dir).glob("*/tokenizer/tokenizer_config.json"):
        return cand.parent
    from huggingface_hub import snapshot_download

    local = Path(models_dir) / "tokenizer__MOSS-Image"
    log.info("Downloading MOSS-Image tokenizer to %s", local)
    snapshot_download("OpenMOSS-Team/MOSS-Image-Base", local_dir=str(local), allow_patterns=["tokenizer/*"])
    return local / "tokenizer"
