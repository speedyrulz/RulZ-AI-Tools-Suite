"""Load Qwen-Image 2.1 components from single ComfyUI-style safetensors files (Comfy-Org/Qwen-Image-2.1:
diffusion_models, text_encoders, vae).

The DiT file uses diffusers' module names except for the feed-forward, whose two input projections ComfyUI fuses into
img_mlp.gate_up = [gate_layer; proj] (rows). The port keeps them apart (qwen21/model.py), so gate_up is split on
load; an int8 / int8_convrot weight splits by rows as it is (one scale per row), nvfp4 is requantized per half. Weights
may be bf16, ComfyUI int8 / int8_convrot or nvfp4 (kept with transformer_quant "auto", or converted). A file name
containing "turbo" marks the distilled model (Qwen-Image-2.1-Turbo: its own fixed 8-step schedule).

The text encoder file is Qwen3-VL-8B as ComfyUI saves it (`model.layers.*`, `model.embed_tokens`, `lm_head`; int8
files quantize the embedding too, which is dequantized) or in the HF layout (`model.language_model.*`); the vision
tower and the LM head are dropped and the language model is built as a text-only Qwen3VLTextModel, which gives the
same hidden states for text-only prompts. The VAE file is ComfyUI's (the original 3D-convolution layout:
encoder.downsamples / decoder.upsamples, residual.N, conv1 / conv2), converted to diffusers' AutoencoderKLQwenImage21
(2D convolutions: the 3D kernels have a time extent of 1 and are squeezed); its latent statistics are constants of the
model (qwen21_vae_config.json)."""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import torch

from .. import quant as Q
from ..moss import single_file as sf
from .model import QwenImage21Transformer

log = logging.getLogger("rulztrainer")
HERE = Path(__file__).resolve().parent

DIT_PREFIXES = ("model.diffusion_model.", "diffusion_model.")
MARKERS = ("txt_in.text_norm.weight", "modulation.1.weight")  # ComfyUI's own detection keys for the family


def is_qwen21_state(keys) -> bool:
    keys = set(keys)
    return all(m in keys for m in MARKERS)


def unfuse_mlp(state: dict) -> dict:
    """img_mlp.gate_up ([gate_layer; proj] by rows) -> img_mlp.gate_layer + img_mlp.proj."""
    out = {}
    for k, v in state.items():
        if k.endswith(".img_mlp.gate_up.weight"):
            stem = k[: -len(".gate_up.weight")]
            gate, proj = v.split_rows(2) if isinstance(v, Q.QuantWeight) else v.chunk(2, 0)
            out[f"{stem}.gate_layer.weight"], out[f"{stem}.proj.weight"] = gate, proj
        else:
            out[k] = v
    return out


def _count(state: dict, pattern: str) -> int:
    n = 0
    while pattern.format(n) in state:
        n += 1
    return n


def transformer_config_from_state(state: dict) -> dict:
    """QwenImage21Transformer config from a (diffusers-layout, maybe quantized) state dict."""
    def shape(k):
        v = state[k]
        return tuple(v.orig_shape) if isinstance(v, Q.QuantWeight) else tuple(v.shape)

    head_dim = shape("transformer_blocks.0.attn.norm_q.weight")[0]
    dim = shape("img_in.weight")[0]
    unit = head_dim // 16  # 128 -> (16, 56, 56), the checkpoint's axes
    return dict(in_channels=shape("img_in.weight")[1], out_channels=shape("proj_out.weight")[0],
                num_layers=_count(state, "transformer_blocks.{}.attn.to_q.weight"), attention_head_dim=head_dim,
                num_attention_heads=dim // head_dim, context_in_dim=shape("txt_in.in_layer.weight")[1],
                mlp_ratio=shape("transformer_blocks.0.img_mlp.gate_layer.weight")[0] // dim,
                axes_dims_rope=[2 * unit, 7 * unit, 7 * unit], eps=1e-6, causal_condition=True)


def is_qwen21_file(path: str | Path) -> bool:
    """A cheap look at a safetensors header: does this file hold a Qwen-Image 2.1 diffusion model?"""
    from safetensors import safe_open

    try:
        with safe_open(str(path), "pt") as f:
            keys = list(f.keys())
    except Exception:  # noqa: BLE001
        return False
    return any(is_qwen21_state(k[len(p):] for k in keys if k.startswith(p)) for p in ("", *DIT_PREFIXES))


def load_transformer_file(path: str | Path, device, quant: str = "auto", dtype=torch.bfloat16,
                          quant_devices=None) -> QwenImage21Transformer:
    """quant "auto" keeps a quantized file's format; an unquantized (bf16) file is quantized for the memory it will
    run in, as the HF loader does (quant_devices: the GPUs of a model split)."""
    from .loading import QUANT_FILTER, resolve_quant

    state = sf.read_state(path, dtype)
    state = sf.strip_prefix(state, DIT_PREFIXES)
    state = {k: v for k, v in state.items() if not k.endswith("comfy_attention.config")}
    if not is_qwen21_state(state):
        raise ValueError(f"{Path(path).name}: not a Qwen-Image 2.1 diffusion model (no txt_in.text_norm / "
                         "modulation.1)")
    state = unfuse_mlp(state)
    config = transformer_config_from_state(state)
    if quant == "auto":
        fmts = {v.fmt for v in state.values() if isinstance(v, Q.QuantWeight)}
        if not fmts:
            quant = resolve_quant("auto", quant_devices or device)
    with torch.device("meta"):
        model = QwenImage21Transformer.from_config(config)
    sf.install_state(model, state, quant, dtype, quant_filter=QUANT_FILTER, **sf.quant_target(device, quant_devices))
    del state
    model.is_distilled = "turbo" in Path(path).name.lower()
    model = model.to(device).eval().requires_grad_(False)
    log.info("Loaded Qwen-Image 2.1 transformer from %s: %d layers, dim %d, %s, quant=%s", Path(path).name,
             config["num_layers"], config["attention_head_dim"] * config["num_attention_heads"],
             "distilled (turbo, by its file name)" if model.is_distilled else "base", quant)
    return model


# ---------------------------------------------------------------- VAE


_VAE_BLOCK = {"0": "norm1", "2": "conv1", "3": "norm2", "6": "conv2"}  # the residual Sequential's parts


def vae_comfy_to_diffusers(state: dict) -> dict:
    """ComfyUI's Qwen-Image 2.1 VAE keys -> diffusers' AutoencoderKLQwenImage21 (shapes are fitted on load)."""
    out = {}
    for k, v in state.items():
        nk = k
        nk = re.sub(r"^conv1\.", "quant_conv.", nk)
        nk = re.sub(r"^conv2\.", "post_quant_conv.", nk)
        nk = re.sub(r"^(encoder|decoder)\.conv1\.", r"\1.conv_in.", nk)
        nk = re.sub(r"^(encoder|decoder)\.head\.0\.", r"\1.norm_out.", nk)
        nk = re.sub(r"^(encoder|decoder)\.head\.2\.", r"\1.conv_out.", nk)
        nk = re.sub(r"^(encoder|decoder)\.middle\.1\.", r"\1.mid_block.attentions.0.", nk)
        nk = re.sub(r"^(encoder|decoder)\.middle\.([02])\.",
                    lambda m: f"{m.group(1)}.mid_block.resnets.{int(m.group(2)) // 2}.", nk)
        nk = re.sub(r"^encoder\.downsamples\.(\d+)\.downsamples\.(\d+)\.(resample|time_conv)",
                    r"encoder.down_blocks.\1.downsampler.\3", nk)
        nk = re.sub(r"^encoder\.downsamples\.(\d+)\.downsamples\.(\d+)\.", r"encoder.down_blocks.\1.resnets.\2.", nk)
        nk = re.sub(r"^decoder\.upsamples\.(\d+)\.upsamples\.(\d+)\.(resample|time_conv)",
                    r"decoder.up_blocks.\1.upsampler.\3", nk)
        nk = re.sub(r"^decoder\.upsamples\.(\d+)\.upsamples\.(\d+)\.", r"decoder.up_blocks.\1.resnets.\2.", nk)
        nk = re.sub(r"\.residual\.(\d)\.", lambda m: f".{_VAE_BLOCK[m.group(1)]}.", nk)
        nk = nk.replace(".shortcut.", ".conv_shortcut.")
        out[nk] = v
    return out


def vae_class():
    """diffusers' AutoencoderKLQwenImage21 (diffusers 0.41 and later)."""
    try:
        from diffusers import AutoencoderKLQwenImage21
    except ImportError as e:
        raise ImportError("Qwen-Image 2.1's VAE needs diffusers 0.41 or later: run the installer again (or pip "
                          "install -U diffusers)") from e
    return AutoencoderKLQwenImage21


def vae_config() -> dict:
    return json.loads((HERE / "qwen21_vae_config.json").read_text(encoding="utf-8"))


def load_vae_file(path: str | Path, device, dtype=torch.bfloat16):
    AutoencoderKLQwenImage21 = vae_class()
    state = sf.read_state(path, torch.float32)
    if any(isinstance(v, Q.QuantWeight) for v in state.values()):
        raise ValueError("Quantized VAE files are not supported; use the bf16 Qwen-Image 2.1 VAE")
    state = sf.strip_prefix(state, ("vae.", "first_stage_model."))
    if "decoder.middle.0.residual.0.gamma" in state:  # ComfyUI's layout
        state = vae_comfy_to_diffusers(state)
    if "decoder.mid_block.resnets.0.norm1.gamma" not in state or "encoder.conv_in.weight" not in state:
        raise ValueError(f"{Path(path).name} is not a Qwen-Image 2.1 VAE")
    with torch.device("meta"):
        vae = AutoencoderKLQwenImage21(**vae_config())
    want = vae.state_dict()
    if state.get("encoder.conv_in.weight") is not None and \
            state["encoder.conv_in.weight"].shape[1] != want["encoder.conv_in.weight"].shape[1]:
        raise ValueError(f"{Path(path).name} is not a Qwen-Image 2.1 VAE (it reads "
                         f"{state['encoder.conv_in.weight'].shape[1]} image channels, not 4)")
    fitted = {}
    for k, v in state.items():
        if k in want and tuple(v.shape) != tuple(want[k].shape) and v.numel() == want[k].numel():
            v = v.reshape(want[k].shape)  # a 3D kernel with a time extent of 1 -> the 2D one
        fitted[k] = v.to(dtype)
    missing, unexpected = vae.load_state_dict(fitted, strict=False, assign=True)
    if missing or unexpected:
        raise ValueError(f"VAE state mismatch. missing={missing[:5]} unexpected={unexpected[:5]}")
    return vae.to(device).eval().requires_grad_(False)


# ---------------------------------------------------------------- text encoder (Qwen3-VL-8B, text only)

# the encoder's int8 layers multiply with the weight dequantized (weight-only), not W8A8: the int8 activations moved the
# cached features 5-8% (cosine 0.998) against weight-only on the same file, at about the same caching time
TE_INT8_MATMUL = False
TOKENIZER_REPO = "Qwen/Qwen-Image-2.1"  # its processor/ files (Fizgig: they differ from Qwen3-VL-4B's copy)
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "added_tokens.json",
                   "special_tokens_map.json", "chat_template.jinja")


def text_config() -> dict:
    return json.loads((HERE / "qwen3vl_8b_text_config.json").read_text(encoding="utf-8"))


def load_text_encoder_file(path: str | Path, device: str = "auto", quant: str = "auto"):
    from ..fibo.loading import place_text_encoder
    from ..krea2.single_file import build_text_model

    state = sf.read_state(path, torch.bfloat16)
    model = build_text_model(state, quant, "cpu", config=text_config(),
                             work_device=sf.quant_target(device).get("work_device"))
    Q.configure_int8(model, TE_INT8_MATMUL)
    n_q = sum(1 for m in model.modules() if isinstance(m, Q.QuantLinear))
    log.info("Loaded Qwen3-VL-8B text encoder from %s (%d layers%s)", Path(path).name,
             model.config.num_hidden_layers, f", {n_q} quantized Linear layers" if n_q else "")
    del state
    return place_text_encoder(model, device)


def resolve_tokenizer(tokenizer_path: str, models_dir: str = "models") -> Path:
    """The Qwen-Image 2.1 tokenizer folder: the given one, a downloaded Qwen-Image 2.1 repo's processor/, or the
    official repo's processor files (downloaded once, a few MB)."""
    if tokenizer_path:
        p = Path(tokenizer_path).expanduser()
        for cand in (p, p / "processor", p / "tokenizer"):
            if (cand / "tokenizer_config.json").is_file():
                return cand
        raise FileNotFoundError(f"No tokenizer_config.json under {p}")
    for cand in sorted(Path(models_dir).glob("Qwen__Qwen-Image-2.1*/processor/tokenizer_config.json")):
        return cand.parent
    local = Path(models_dir) / "tokenizer__Qwen-Image-2.1"
    if not (local / "tokenizer_config.json").is_file():
        from huggingface_hub import snapshot_download

        log.info("Downloading the Qwen-Image 2.1 tokenizer to %s", local)
        tmp = Path(snapshot_download(TOKENIZER_REPO, allow_patterns=[f"processor/{f}" for f in TOKENIZER_FILES]))
        local.mkdir(parents=True, exist_ok=True)
        for f in TOKENIZER_FILES:
            if (tmp / "processor" / f).is_file():
                (local / f).write_bytes((tmp / "processor" / f).read_bytes())
    return local
