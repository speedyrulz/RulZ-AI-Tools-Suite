"""Load MiniMax H3 components from single ComfyUI-style safetensors files (Comfy-Org/MiniMax-H3: diffusion_models,
text_encoders, vae).

Diffusion model: the original module names (h3/convert.py), [q; k; v] stacked; bf16, ComfyUI int8_tensorwise with
ConvRot (int8_convrot, group 256; group 64 on the full files' AdaLN), or the scaled fp8 file (dequantized on read). The
fp32 islands of the checkpoint (patch projections, time embedder, output heads, rope.inv_freq) stay fp32 and the
pruned files' F16 AdaLN projections are read as fp32, as ComfyUI keeps them. The w6a8 files use a format this trainer
cannot read. Full-AdaLN files (no adaln_t_table) are pruned on load to a rank-8 basis (h3/loading.prune_adaln, as
musubi-tuner's --prune_adaln), unless RULZ_H3_FULL_ADALN=1: about 13B of the 33B parameters live in the AdaLN
projections.

Text encoder: Qwen3-VL-32B as Comfy-Org repackaged it, the language model truncated to its first 50 decoder layers with
no final norm and no LM head (the files' "minimax_h3_te" metadata: "unnormalized_hidden_after_layer_50"), so the last
layer's output is H3's conditioning; the vision tower is dropped. bf16, int8_convrot (the embedding bf16), or NVFP4 with
AWQ (cybermotaz/Qwen3-VL-32B-Instruct-NVFP4 converted: per-16 fp8 block scales, an fp32 per-tensor scale, the embedding
int8 per row, which is dequantized; o_proj and down_proj carry a pre_quant_scale multiplied into their inputs; the
comfy_quant spec asks for full-precision matmuls, so the weights are dequantized per forward, no FP4 activations).

VAE: Comfy-Org's video VAE file is the original MiniMax layout (the same keys as MiniMaxAI/MiniMax-H3's
FL2VA/video_vae/source/model.safetensors) plus latents_mean / latents_std; it is converted to diffusers'
AutoencoderKLMiniMaxH3 (encoder.down.N.block.M -> down_blocks.N.resnets.M, nin_shortcut -> conv_shortcut, the ViT
decoder's per-head interleaved to_qkv split into to_q / to_k / to_v, ff.w1's [gate; value] into SwiGLU's [value; gate],
x_embedder -> proj_in). The Sol VAE (minimax_h3_sol_vae_*: a 24-layer distilled whole-clip decoder) and the int8 VAE
are not supported; for a still the stock fp16 decoder is what ComfyUI's Sol path matches anyway. The audio VAE file is
the original DAC / BigVGAN layout with the weight norms folded into plain weights; only its encoder is needed (the
silence placeholder, h3/loading.silence_latent), converted to diffusers' weight_g / weight_v."""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn

from .. import quant as Q
from ..moss import single_file as sf
from . import convert as C

log = logging.getLogger("rulztrainer")
FLOAT8 = {getattr(torch, n) for n in ("float8_e4m3fn", "float8_e5m2") if hasattr(torch, n)}
# quantize the 50 blocks' Linears (attention, MLP, AdaLN), as Comfy-Org's int8 files do; the token refiner, the
# projections, the time embedder and the final layer stay as they are
QUANT_FILTER = lambda name: name.startswith("blocks.")  # noqa: E731
COMFY_REPO = "Comfy-Org/MiniMax-H3"
AUDIO_VAE_FILE = "vae/minimax_h3_audio_vae_fp32.safetensors"
TOKENIZER_REPO = "MiniMaxAI/MiniMax-H3"  # its tokenizer/ (Qwen2 + H3's added tokens; upstream Qwen3-VL's lacks them)
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")
H3_TOKENS = ("<d>", "</d>", "<|cutoff|>", "<|lyrics_start|>")  # added by H3's tokenizer (ComfyUI: fixed ids 151669+)


def iter_state(path: str | Path, dtype=torch.bfloat16, keep_fp32=lambda key: False, want=lambda key: True):
    """(key, tensor | QuantWeight) of a safetensors file, one at a time: ComfyUI-quantized Linears as QuantWeight,
    scaled fp8 dequantized, F32 tensors kept fp32 (the checkpoint's fp32 islands), other floats as `dtype` (or fp32
    where keep_fp32(key)). want: which keys to read."""
    with safe_open(str(path), "pt") as f:
        keys = set(f.keys())
        quant_meta = {k[: -len(".comfy_quant")]: Q.parse_comfy_quant(f.get_tensor(k))
                      for k in keys if k.endswith(".comfy_quant")}
        for k in sorted(keys):
            if k.endswith((".comfy_quant", ".weight_scale", ".weight_scale_2", ".input_scale")) or not want(k):
                continue
            t = f.get_tensor(k)
            base = k[: -len(".weight")] if k.endswith(".weight") else None
            fmt = quant_meta.get(base, {}).get("format") if base is not None else None
            if fmt is not None and not str(fmt).startswith("float8"):
                s2 = f.get_tensor(f"{base}.weight_scale_2") if f"{base}.weight_scale_2" in keys else None
                yield k, Q.quant_weight_from_comfy(quant_meta[base], t, f.get_tensor(f"{base}.weight_scale"), s2)
                continue
            if base is not None and f"{base}.weight_scale" in keys:  # scaled fp8: per-tensor or per-row scale
                t = t.to(torch.float32) * f.get_tensor(f"{base}.weight_scale").to(torch.float32)
            if t.dtype in FLOAT8:
                t = t.to(torch.float32)
            if t.is_floating_point():
                t = t.to(torch.float32 if (t.dtype == torch.float32 or keep_fp32(k)) else dtype)
            yield k, t


def read_state(path: str | Path, dtype=torch.bfloat16, keep_fp32=lambda key: False) -> dict:
    """iter_state as a dict."""
    return dict(iter_state(path, dtype, keep_fp32))


def is_quantized_file(path: str | Path) -> bool:
    try:
        with safe_open(str(path), "pt") as f:
            return any(k.endswith(".comfy_quant") for k in f.keys())
    except Exception:  # noqa: BLE001
        return False


def file_metadata(path: str | Path) -> dict:
    try:
        with safe_open(str(path), "pt") as f:
            return f.metadata() or {}
    except Exception:  # noqa: BLE001
        return {}


def is_h3_file(path: str | Path) -> bool:
    """A cheap look at a safetensors header: does this file hold a MiniMax H3 diffusion model?"""
    try:
        with safe_open(str(path), "pt") as f:
            keys = list(f.keys())
    except Exception:  # noqa: BLE001
        return False
    return any(C.is_h3_state(k[len(p):] for k in keys if k.startswith(p)) for p in ("", *C.DIT_PREFIXES))


def load_transformer_file(path: str | Path, device, quant: str = "auto", dtype=torch.bfloat16, quant_devices=None):
    """quant "auto" keeps a quantized file's formats; a bf16 file is quantized for the memory it will run in
    (loading.resolve_quant), layer by layer as it is read."""
    from . import loading as L

    if not is_h3_file(path):
        raise ValueError(f"{Path(path).name}: not a MiniMax H3 diffusion model (no video_patch_proj / "
                         "audio_patch_proj / condition_proj)")
    keep = lambda k: ".adaln_proj.linear." in k  # noqa: E731 - a pruned file's F16 AdaLN, kept fp32 as ComfyUI does

    def reader(want):
        for k, v in iter_state(path, dtype, keep, want):
            yield from C.original_to_port({k: v}, interleaved=False, heads=0, head_dim=0).items()  # [q; k; v]

    return L.build_transformer(reader, device, quant, dtype, quant_devices, source=Path(path).name,
                               quantized=is_quantized_file(path), source_bytes=Path(path).stat().st_size)


# ---------------------------------------------------------------- video VAE


def vae_original_to_diffusers(state: dict, heads: int = 32) -> dict:
    """The original / Comfy-Org video VAE keys -> diffusers' AutoencoderKLMiniMaxH3 (latents_mean / _std and the unused
    mask_token dropped: they are config / not part of the model)."""
    out = {}
    for k in list(state):
        v = state.pop(k)
        if k in ("latents_mean", "latents_std", "decoder.mask_token"):
            continue
        nk = re.sub(r"^encoder\.down\.(\d+)\.block\.(\d+)\.", r"encoder.down_blocks.\1.resnets.\2.", k)
        nk = re.sub(r"^encoder\.down\.(\d+)\.downsample\.", r"encoder.down_blocks.\1.downsamplers.0.", nk)
        nk = nk.replace(".nin_shortcut.", ".conv_shortcut.")
        nk = re.sub(r"^decoder\.x_embedder\.", "decoder.proj_in.", nk)
        nk = nk.replace(".attn.to_out.", ".attn.to_out.0.")
        nk = nk.replace(".ff.w2.", ".ff.net.2.")
        m = re.match(r"^(decoder\.transformer_blocks\.\d+\.attn)\.to_qkv\.(weight|bias)$", nk)
        if m:  # [head: q k v] rows (the original Attention: qkv.view(B, L, heads, 3 * dh).chunk(3, -1))
            stem, kind = m.groups()
            dh = v.shape[0] // (3 * heads)
            parts = v.reshape(heads, 3, dh, *v.shape[1:])
            for i, name in enumerate(("to_q", "to_k", "to_v")):
                out[f"{stem}.{name}.{kind}"] = parts[:, i].reshape(heads * dh, *v.shape[1:]).contiguous()
            continue
        if ".ff.w1." in nk:  # the original FFN: gate, hidden = w1(x).chunk(2); silu(gate) * hidden
            nk = nk.replace(".ff.w1.", ".ff.net.0.proj.")
            v = C.swap_halves(v)
        out[nk] = v
    return out


def vae_diffusers_to_original(state: dict, heads: int = 32) -> dict:
    """The inverse of vae_original_to_diffusers (tests)."""
    out, qkv = {}, {}
    for k, v in state.items():
        m = re.match(r"^(decoder\.transformer_blocks\.\d+\.attn)\.to_([qkv])\.(weight|bias)$", k)
        if m:
            qkv.setdefault((m.group(1), m.group(3)), {})[m.group(2)] = v
            continue
        nk = re.sub(r"^encoder\.down_blocks\.(\d+)\.resnets\.(\d+)\.", r"encoder.down.\1.block.\2.", k)
        nk = re.sub(r"^encoder\.down_blocks\.(\d+)\.downsamplers\.0\.", r"encoder.down.\1.downsample.", nk)
        nk = nk.replace(".conv_shortcut.", ".nin_shortcut.")
        nk = re.sub(r"^decoder\.proj_in\.", "decoder.x_embedder.", nk)
        nk = nk.replace(".attn.to_out.0.", ".attn.to_out.").replace(".ff.net.2.", ".ff.w2.")
        if ".ff.net.0.proj." in nk:
            nk = nk.replace(".ff.net.0.proj.", ".ff.w1.")
            v = C.swap_halves(v)
        out[nk] = v
    for (stem, kind), parts in qkv.items():
        q = parts["q"]
        dh = q.shape[0] // heads
        stacked = torch.stack([p.reshape(heads, dh, *p.shape[1:]) for p in (parts["q"], parts["k"], parts["v"])], 1)
        out[f"{stem}.to_qkv.{kind}"] = stacked.reshape(3 * heads * dh, *q.shape[1:])
    return out


def vae_class():
    try:
        from diffusers import AutoencoderKLMiniMaxH3
    except ImportError as e:
        raise ImportError("MiniMax H3's VAE needs diffusers 0.40 or later: run the installer again (or pip install -U "
                          "diffusers)") from e
    return AutoencoderKLMiniMaxH3


# the released video VAE's config (MiniMaxAI/MiniMax-H3 vae/config.json, read from the Hub; Comfy-Org's file carries
# the same latents_mean / latents_std in its metadata and as tensors)
VAE_CONFIG = {
    "in_channels": 3, "out_channels": 3, "latent_channels": 24, "block_out_channels": [128, 256, 256, 512, 512, 1024],
    "layers_per_block": 2, "spatial_downsample_factors": [2, 2, 2, 2, 1, 1],
    "temporal_downsample_factors": [1, 2, 2, 1, 1, 1], "norm_num_groups": 32, "norm_eps": 1e-06,
    "spatial_padding_mode": "reflect", "decoder_num_layers": 36, "decoder_num_attention_heads": 32,
    "decoder_attention_head_dim": 64, "decoder_num_register_tokens": 4, "decoder_ffn_mult": 4,
    "decoder_rope_theta": 100.0, "decoder_rope_dim_ratio": 0.75, "decoder_norm_eps": 1e-05, "clip_length": 17,
    "token_drop": 3,
    "latents_mean": [0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075,
                     -0.2727581858634949, -1.3675414323806763, -0.2553254961967468, -0.26907554268836975,
                     -0.5376840829849243, -0.0464097298681736, 0.6657370328903198, 0.19690127670764923,
                     -0.5460608005523682, -0.4035342037677765, -0.23683024942874908, 0.25928452610969543,
                     -0.30133944749832153, 0.211341992020607, -1.1206848621368408, 0.3581933379173279,
                     -0.04225143790245056, 0.2604829967021942, 0.22864092886447906, 0.7056031823158264],
    "latents_std": [1.2223774194717407, 1.2767263650894165, 1.6831774711608887, 1.7549455165863037,
                    1.5636216402053833, 2.194143533706665, 0.9653137922286987, 1.0569885969161987, 0.841948926448822,
                    0.7729952931404114, 1.8955937623977661, 0.946841835975647, 0.7996809482574463,
                    0.44988900423049927, 0.7197399735450745, 0.6936293244361877, 2.961095094680786,
                    2.7694199085235596, 3.0496184825897217, 2.1088054180145264, 3.276226282119751, 3.1627357006073,
                    2.2816812992095947, 2.6127843856811523],
}
# diffusers' _keep_in_fp32_modules of AutoencoderKLMiniMaxH3 (the rest follows the compute dtype)
VAE_FP32 = ("encoder", "quant_conv", "norm1", "norm2", "norm_out", "scale1", "scale2")


def vae_param_dtype(name: str, dtype) -> torch.dtype:
    parts = name.split(".")
    return torch.float32 if any(p in VAE_FP32 for p in parts) else dtype


def assign_vae(state: dict, config: dict, dtype=torch.bfloat16):
    """A diffusers-layout state into a freshly built AutoencoderKLMiniMaxH3 (meta, assigned), with diffusers'
    fp32 / compute-dtype split."""
    cls = vae_class()
    with torch.device("meta"):
        vae = cls(**config)
    want = vae.state_dict()
    missing = sorted(set(want) - set(state))
    unexpected = sorted(set(state) - set(want))
    if missing or unexpected:
        raise ValueError(f"MiniMax H3 VAE state mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    fitted = {}
    for k, v in state.items():
        if tuple(v.shape) != tuple(want[k].shape):
            raise ValueError(f"MiniMax H3 VAE {k}: {tuple(v.shape)} where {tuple(want[k].shape)} is expected")
        fitted[k] = v.to(vae_param_dtype(k, dtype))
    vae.load_state_dict(fitted, strict=True, assign=True)
    rope = getattr(vae.decoder, "rope", None)  # its inverse frequencies are computed, not stored: built again for real
    if rope is not None:
        cfg = vae.config
        vae.decoder.rope = type(rope)(int(cfg.decoder_attention_head_dim * cfg.decoder_rope_dim_ratio),
                                      theta=cfg.decoder_rope_theta)
    for name, b in list(vae.named_buffers()):
        if b.device.type == "meta":
            raise ValueError(f"MiniMax H3 VAE buffer {name} was not built")
    return vae


def load_vae_file(path: str | Path, device, dtype=torch.bfloat16):
    state = read_state(path, torch.float32)
    if any(isinstance(v, Q.QuantWeight) for v in state.values()):
        raise ValueError(f"{Path(path).name}: quantized VAE files are not supported; use "
                         "vae/minimax_h3_video_vae_fp16.safetensors")
    state = sf.strip_prefix(state, ("vae.", "first_stage_model."))
    if any(k.startswith("blocks.") or k.startswith("decoder.blocks.") for k in state):
        raise ValueError(f"{Path(path).name} is the Sol VAE (a distilled whole-clip decoder), which this trainer does "
                         "not read: use vae/minimax_h3_video_vae_fp16.safetensors (for a still image ComfyUI's Sol "
                         "path decodes as the stock decoder does)")
    if "decoder.transformer_blocks.0.attn.to_qkv.weight" not in state or "encoder.conv_in.weight" not in state:
        raise ValueError(f"{Path(path).name} is not a MiniMax H3 video VAE")
    config = dict(VAE_CONFIG)
    for key in ("latents_mean", "latents_std"):
        if key in state:
            config[key] = [float(x) for x in state[key].float()]
    state = vae_original_to_diffusers(state, heads=config["decoder_num_attention_heads"])
    vae = assign_vae(state, config, dtype)
    return vae.to(device).eval().requires_grad_(False)


# ---------------------------------------------------------------- audio VAE (encoder only: the silence latent)


def audio_original_to_diffusers(state: dict, want_keys) -> dict:
    """Comfy-Org's audio VAE (weight norms folded into plain weights) -> diffusers' AutoencoderKLMiniMaxH3Audio names:
    a weight-normed layer's `weight` becomes weight_v = weight and weight_g = its norm over every dim but the first
    (torch's weight_norm, dim=0), which gives back the same weight. Only the keys in `want_keys` are kept."""
    want_keys = set(want_keys)
    out = {}
    for k, v in state.items():
        if k in want_keys:
            out[k] = v
            continue
        if k.endswith(".weight") and f"{k}_v" in want_keys:
            out[f"{k}_v"] = v
            out[f"{k}_g"] = torch.linalg.vector_norm(v.float(), dim=tuple(range(1, v.ndim)), keepdim=True).to(v.dtype)
    return out


def load_audio_encoder_file(path: str | Path, config: dict | None = None):
    """diffusers' AutoencoderKLMiniMaxH3Audio with its encoder, pre_block and latent projections from Comfy-Org's file
    (fp32, CPU; the decoder is left unbuilt), and the latents_mean / latents_std it normalizes with."""
    try:
        from diffusers import AutoencoderKLMiniMaxH3Audio
    except ImportError as e:
        raise ImportError("MiniMax H3's audio VAE needs diffusers 0.40 or later") from e
    state = read_state(path, torch.float32)
    meta = file_metadata(path).get("minimax_h3_audio_vae")
    mean = state.pop("latents_mean", None)
    std = state.pop("latents_std", None)
    if (mean is None or std is None) and meta:
        info = json.loads(meta)
        mean, std = torch.tensor(info["latents_mean"]), torch.tensor(info["latents_std"])
    if mean is None or std is None:
        raise ValueError(f"{Path(path).name}: no audio latents_mean / latents_std")
    with torch.device("meta"):
        model = AutoencoderKLMiniMaxH3Audio(**(config or {}))
    enc_keys = [k for k in model.state_dict() if k.startswith(("encoder.", "pre_block.", "mean_proj.", "logs_proj."))]
    enc = audio_original_to_diffusers(state, enc_keys)
    missing = sorted(set(enc_keys) - set(enc))
    if missing:
        raise ValueError(f"{Path(path).name} is not a MiniMax H3 audio VAE (missing {missing[:3]})")
    model.load_state_dict(enc, strict=False, assign=True)
    return model.eval().requires_grad_(False), mean.float(), std.float()


# ---------------------------------------------------------------- text encoder (Qwen3-VL-32B, 50 layers, text only)

TE_PREFIXES = ("model.language_model.", "language_model.", "text_encoders.qwen3vl_32b.transformer.model.", "model.")
TE_DROP = ("visual.", "model.visual.", "lm_head.")
TE_LAYERS = 50  # H3 conditions on hidden_states[50]: the output of decoder layer 50, before any final norm
# Qwen3-VL-32B's language model (MiniMaxAI/MiniMax-H3 text_encoder/config.json, text_config), cut to 50 layers
TEXT_CONFIG = {
    "attention_bias": False, "attention_dropout": 0.0, "bos_token_id": 151643, "eos_token_id": 151645,
    "head_dim": 128, "hidden_act": "silu", "hidden_size": 5120, "initializer_range": 0.02,
    "intermediate_size": 25600, "max_position_embeddings": 262144, "model_type": "qwen3_vl_text",
    "num_attention_heads": 64, "num_hidden_layers": TE_LAYERS, "num_key_value_heads": 8, "pad_token_id": None,
    "rms_norm_eps": 1e-06,
    "rope_parameters": {"mrope_interleaved": True, "mrope_section": [24, 20, 20], "rope_theta": 5000000,
                        "rope_type": "default"},
    "tie_word_embeddings": False, "use_cache": False, "vocab_size": 151936,
}
# the encoder's quantized layers multiply with the weight dequantized (weight-only, as the AWQ file's
# full_precision_matrix_mult asks and as the other families' text encoders do)
TE_INT8_MATMUL = False


class PreScaled(nn.Module):
    """AWQ's pre_quant_scale: the layer sees its input times a per-channel scale."""

    def __init__(self, inner: nn.Module, scale: torch.Tensor):
        super().__init__()
        self.inner = inner
        self.register_buffer("pre_quant_scale", scale)
        self.in_features = getattr(inner, "in_features", scale.numel())
        self.out_features = getattr(inner, "out_features", None)

    def forward(self, x):
        return self.inner(x * self.pre_quant_scale.to(x.dtype))


def _layer_index(key: str) -> int | None:
    m = re.match(r"^layers\.(\d+)\.", key)
    return int(m.group(1)) if m else None


def build_text_model(state: dict, quant: str = "auto", device="cpu", config: dict | None = None, work_device=None):
    """The text-only Qwen3-VL language model, 50 layers, from a ComfyUI- or HF-style (possibly quantized, possibly
    AWQ) state dict, which is consumed."""
    from transformers import Qwen3VLTextConfig, Qwen3VLTextModel

    cfg_dict = dict(config or TEXT_CONFIG)
    cfg_dict["num_hidden_layers"] = TE_LAYERS
    cfg_dict["use_cache"] = False
    cfg = Qwen3VLTextConfig(**cfg_dict)
    state = sf.take_state(state, TE_DROP)
    state = sf.strip_prefix(state, TE_PREFIXES)
    for k in list(state):  # layers past the 50th (an HF folder holds all 64) and the final norm are never run
        i = _layer_index(k)
        if (i is not None and i >= TE_LAYERS) or k == "norm.weight":
            state.pop(k)
    prescale = {k[: -len(".pre_quant_scale")]: state.pop(k) for k in list(state) if k.endswith(".pre_quant_scale")}
    state["norm.weight"] = torch.ones(cfg.hidden_size, dtype=torch.bfloat16)  # bypassed (encode_prompt's hook)
    with torch.device("meta"):
        model = Qwen3VLTextModel(cfg)
    sf.install_state(model, state, quant, torch.bfloat16, ignore_missing=("inv_freq",), work_device=work_device)
    for name, scale in prescale.items():
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name)
        setattr(parent, child, PreScaled(getattr(parent, child), scale.to(torch.bfloat16)))
    import inspect

    rot_cls = type(model.rotary_emb)  # its buffers were created on the meta device: build it again for real
    kwargs = {"device": "cpu"} if "device" in inspect.signature(rot_cls.__init__).parameters else {}
    model.rotary_emb = rot_cls(config=cfg, **kwargs)
    for name, p in model.named_parameters():
        if p.device.type == "meta":
            raise ValueError(f"Text encoder parameter {name} was not loaded")
    Q.configure_int8(model, TE_INT8_MATMUL)
    Q.configure_nvfp4(model, False)
    if prescale:
        log.info("Text encoder: %d AWQ pre-quant scales applied to their layers' inputs", len(prescale))
    return model.to(device).eval().requires_grad_(False)


def load_text_encoder_file(path: str | Path, device: str = "auto", quant: str = "auto"):
    from ..fibo.loading import place_text_encoder

    meta = file_metadata(path).get("minimax_h3_te")
    if meta:
        info = json.loads(meta)
        if int(info.get("num_hidden_layers", TE_LAYERS)) < TE_LAYERS:
            raise ValueError(f"{Path(path).name} holds fewer than {TE_LAYERS} layers")
    state = sf.read_state(path, torch.bfloat16)
    model = build_text_model(state, quant, "cpu", work_device=sf.quant_target(device).get("work_device"))
    n_q = sum(1 for m in model.modules() if isinstance(m, Q.QuantLinear))
    log.info("Loaded Qwen3-VL-32B text encoder (%d of 64 layers) from %s%s", model.config.num_hidden_layers,
             Path(path).name, f", {n_q} quantized Linear layers" if n_q else "")
    del state
    return place_text_encoder(model, device)


def check_tokenizer(tokenizer) -> None:
    vocab = tokenizer.get_vocab()
    if not all(t in vocab for t in H3_TOKENS):
        raise ValueError("The tokenizer lacks MiniMax H3's added tokens (<d>, </d>, <|cutoff|>, <|lyrics_start|>): use "
                         "MiniMaxAI/MiniMax-H3's tokenizer/ folder (upstream Qwen3-VL-32B files are not the same)")


def resolve_tokenizer(tokenizer_path: str, models_dir: str = "models") -> Path:
    """H3's tokenizer folder: the given one, a downloaded MiniMaxAI/MiniMax-H3's tokenizer/, or its four tokenizer
    files downloaded once (about 11 MB)."""
    if tokenizer_path:
        p = Path(tokenizer_path).expanduser()
        for cand in (p, p / "tokenizer"):
            if (cand / "tokenizer_config.json").is_file():
                return cand
        raise FileNotFoundError(f"No tokenizer_config.json under {p}")
    for cand in sorted(Path(models_dir).glob("MiniMaxAI__MiniMax-H3*/tokenizer/tokenizer_config.json")):
        return cand.parent
    local = Path(models_dir) / "tokenizer__MiniMax-H3"
    if not all((local / f).is_file() for f in TOKENIZER_FILES):
        from huggingface_hub import hf_hub_download

        log.info("Downloading the MiniMax H3 tokenizer to %s", local)
        local.mkdir(parents=True, exist_ok=True)
        for f in TOKENIZER_FILES:
            src = Path(hf_hub_download(TOKENIZER_REPO, f"tokenizer/{f}"))
            (local / f).write_bytes(src.read_bytes())
    return local


def full_adaln_wanted() -> bool:
    return os.environ.get("RULZ_H3_FULL_ADALN", "").strip() in ("1", "true", "yes")
