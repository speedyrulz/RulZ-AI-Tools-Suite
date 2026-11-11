"""Build PAN 2's transformer, text encoder and VAE from single files or from the shards of the diffusers folder.

The transformer is installed piece by piece (`install`): a single file is one piece, the diffusers folder one piece
per shard, so the 58 GB of bf16 weights are never in memory at once when they are quantized on load.

fp32 islands: the checkpoint stores the time embedder, the output norm's projection, every block's norm1.linear and
every block's text modulation table in fp32, and the reference runs them in fp32. They stay fp32 here (`FP32_MODULES`) unless a file holds them quantized
(the files with those layers quantized too: PAN2_int8_convrot_full, PAN2_nvfp4), in which case they run in the
compute dtype like any other quantized layer. A requested quantization (`quant`) covers the blocks and the text
refiner (`QUANT_FILTER`); with `quant_norms` it also covers the blocks' norm1.linear (24576 x 4096 each, 12 GB of
fp32 in all), which is what makes the model fit a 24 GB card.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn

from .. import quant as Q
from ..moss import single_file as msf
from .model import PAN2Transformer

log = logging.getLogger("rulztrainer")
MOSS_HERE = Path(msf.__file__).resolve().parent
TE_SKIP_LAYERS = 2  # text_encoder_hidden_state_skip_layer: the state two layers before the last, not normed
PREFIXES = ("model.diffusion_model.", "diffusion_model.")


def is_fp32_module(name: str) -> bool:
    return name.startswith(("time_embedder.", "norm_out.")) or name.endswith(".norm1.linear")


def is_fp32_key(key: str) -> bool:
    """A tensor the model keeps in fp32: a parameter of an fp32 module, or a block's text modulation table."""
    return key.endswith(".scale_shift_table_context") or is_fp32_module(key.rpartition(".")[0])


def quant_filter(name: str, quant_norms: bool = True) -> bool:
    """Layers a requested quantization converts: the blocks' and the text refiner's projections (what the released
    single files quantize), and with quant_norms the blocks' norm1.linear."""
    if not name.startswith(("transformer_blocks.", "context_refiner.")):
        return False
    return quant_norms or not name.endswith(".norm1.linear")


def looks_like_pan2(keys) -> bool:
    return "context_type_embedding" in keys and "transformer_blocks.0.attn.add_q_proj.weight" in keys


def config_from_state(state: dict) -> dict:
    """The constructor arguments a state dict implies (a file without the config in its metadata)."""
    def shape(k):
        v = state[k]
        return tuple(v.orig_shape) if isinstance(v, Q.QuantWeight) else tuple(v.shape)

    dim, in_channels = shape("x_embedder.weight")[:2]
    head_dim = shape("transformer_blocks.0.attn.norm_q.weight")[0]

    def count(prefix):
        return max(int(k.split(".")[1]) for k in state if k.startswith(prefix)) + 1

    return dict(patch_size=list(shape("x_embedder.weight")[2:]), in_channels=in_channels,
                out_channels=shape("proj_out.weight")[0] // 4, num_attention_heads=dim // head_dim,
                attention_head_dim=head_dim, num_layers=count("transformer_blocks."),
                num_refiner_layers=count("context_refiner."),
                mlp_ratio=shape("transformer_blocks.0.ff.net.0.proj.weight")[0] / dim,
                context_mlp_ratio=shape("transformer_blocks.0.ff_context.net.0.proj.weight")[0] / dim,
                refiner_mlp_ratio=shape("context_refiner.0.ff.net.2.weight")[1] / dim,
                text_embed_dim=shape("context_embedder.1.weight")[1])


def read_state(path: str | Path, dtype=torch.bfloat16) -> dict:
    """moss.single_file.read_state, with the tensors the file stores in fp32 read back in fp32 where the model keeps
    them so (the fp32 islands and the per-row scales are untouched by the bf16 cast)."""
    state = msf.read_state(path, dtype)
    with safe_open(str(path), "pt") as f:
        for k in list(state):
            name = k.split("diffusion_model.", 1)[-1]
            if isinstance(state[k], torch.Tensor) and is_fp32_key(name) and f.get_slice(k).get_dtype() == "F32":
                state[k] = f.get_tensor(k)
    return state


def install(model: nn.Module, state: dict, quant: str, dtype=torch.bfloat16, quant_norms: bool = True,
            min_dim: int = 1024, work_device=None, keep_on_work_device: bool = False, carry: dict | None = None) -> int:
    """Put one piece of the weights into a meta-device model (`state` is consumed) and return the number of layers
    now quantized from it. quant: "auto" keeps what the piece holds, "none" dequantizes, int8 / int8_convrot / nvfp4
    converts every eligible Linear (quant_filter). Plain fp32 islands keep fp32.

    carry: the half of a layer a shard holds without the other (its weight in one shard, its bias in the next) waits
    here for the piece that brings the rest; without it a layer missing its bias is an error."""
    if carry:
        state.update(carry)
        carry.clear()
    work = torch.device(work_device) if work_device is not None else None
    if work is not None and work.type != "cuda":
        work = None

    def on_work(w):
        if work is None:
            return w
        return msf._qw_to(w, work) if isinstance(w, Q.QuantWeight) else w.to(work)

    def from_work(w):
        if work is None or keep_on_work_device:
            return w
        return msf._qw_to(w, "cpu") if isinstance(w, Q.QuantWeight) else w.to("cpu")

    n_quant = 0
    for name, module in list(model.named_modules()):
        wkey, bkey = f"{name}.weight", f"{name}.bias"
        if not isinstance(module, nn.Linear) or (wkey not in state and bkey not in state):
            continue
        if (wkey in state) != (bkey in state) and module.bias is not None:  # half a layer
            if carry is None:
                raise ValueError(f"Not PAN 2 transformer weights: {name} has no {'bias' if wkey in state else 'weight'}")
            for k in (wkey, bkey):
                if k in state:
                    carry[k] = state.pop(k)
            continue
        if wkey not in state:
            continue
        w, b = state.pop(wkey), state.pop(bkey, None)
        if isinstance(w, Q.QuantWeight):
            target = w.fmt if quant == "auto" else quant
            if target != w.fmt:
                w = on_work(w)
                w = w.dequantize(torch.float32) if target == "none" else Q.quantize_weight(w.dequantize(torch.float32), target)
                w = from_work(w.to(dtype) if target == "none" else w)
        elif quant not in ("auto", "none") and Q.can_quantize(w, quant, min_dim) and quant_filter(name, quant_norms):
            w = from_work(Q.quantize_weight(on_work(w), quant))
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        if isinstance(w, Q.QuantWeight):
            setattr(parent, child, Q.QuantLinear(w, b, dtype))
            n_quant += 1
            continue
        keep = torch.float32 if is_fp32_module(name) and w.dtype == torch.float32 else dtype
        lin = nn.Linear(w.shape[1], w.shape[0], bias=b is not None, device="meta", dtype=keep)
        lin.weight = nn.Parameter(w.to(keep), requires_grad=False)
        if b is not None:
            lin.bias = nn.Parameter(b.to(keep), requires_grad=False)
        setattr(parent, child, lin)
    for k, v in list(state.items()):  # anything quantized that is not a Linear weight
        if isinstance(v, Q.QuantWeight):
            state[k] = v.dequantize(dtype)
    _, unexpected = model.load_state_dict(state, strict=False, assign=True)
    if unexpected:
        raise ValueError(f"Not PAN 2 transformer weights: unexpected {unexpected[:5]}")
    state.clear()
    return n_quant


def finish(model: PAN2Transformer, device, source: str, quant: str, n_quant: int) -> PAN2Transformer:
    for module in model.modules():  # layers quantized or rebuilt on load take the place of the meta ones
        if isinstance(module, nn.Linear) and module.weight.device.type != "meta" and module.bias is not None \
                and module.bias.device.type == "meta":
            raise ValueError(f"{source}: a PAN 2 layer came without its bias")
    left = [k for k, t in (*model.named_parameters(), *model.named_buffers()) if t.device.type == "meta"]
    if left:
        raise ValueError(f"{source}: PAN 2 weights not found: {left[:5]}")
    model = model.to(device).eval().requires_grad_(False)
    log.info("Loaded PAN 2 transformer from %s: %d blocks, dim %d, %d quantized layers (quant=%s)", source,
             len(model.transformer_blocks), model.hidden_size, n_quant, quant)
    return model


def file_config(path: str | Path) -> dict | None:
    """The transformer config a converted file keeps in its metadata (pan2-convert writes it)."""
    with safe_open(str(path), "pt") as f:
        raw = (f.metadata() or {}).get("config")
    try:
        cfg = json.loads(raw) if raw else None
    except ValueError:
        return None
    return cfg if isinstance(cfg, dict) and "num_layers" in cfg else None


def load_transformer_file(path: str | Path, device, quant: str = "auto", dtype=torch.bfloat16, quant_devices=None,
                          quant_norms: bool = True, min_dim: int = 1024) -> PAN2Transformer:
    """quant_devices: the GPUs of a model split, when the model is staged on the CPU (they quantize it)."""
    state = msf.strip_prefix(read_state(path, dtype), PREFIXES)
    if not looks_like_pan2(state):
        raise ValueError(f"{Path(path).name} does not look like a PAN 2 diffusion model")
    config = file_config(path) or config_from_state(state)
    with torch.device("meta"):
        model = PAN2Transformer.from_config(config)
    n = install(model, state, quant, dtype, quant_norms, min_dim, **msf.quant_target(device, quant_devices))
    return finish(model, device, Path(path).name, quant, n)


def load_transformer_dir(tdir: Path, device, quant: str = "none", dtype=torch.bfloat16, quant_devices=None,
                         quant_norms: bool = True) -> PAN2Transformer:
    """The diffusers transformer/ folder, one shard at a time."""
    config = json.loads((tdir / "config.json").read_text(encoding="utf-8"))
    if config.get("_class_name") not in ("PAN2Transformer3DModel", None):
        raise ValueError(f"Unsupported transformer class {config.get('_class_name')!r}")
    shards = sorted(tdir.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No .safetensors files in {tdir}")
    with torch.device("meta"):
        model = PAN2Transformer.from_config(config)
    target, n, carry = msf.quant_target(device, quant_devices), 0, {}
    for shard in shards:
        n += install(model, read_state(shard, dtype), "none" if quant == "auto" else quant, dtype, quant_norms,
                     carry=carry, **target)
    if carry:
        raise ValueError(f"{tdir}: PAN 2 weights not found: the other half of {sorted(carry)[:5]}")
    return finish(model, device, str(tdir), quant, n)


# ---------------------------------------------------------------- VAE (Wan 2.2, diffusers' AutoencoderKLWan)


def vae_from_state(state: dict, config: dict, device, name: str):
    import inspect

    from diffusers import AutoencoderKLWan

    accepted = set(inspect.signature(AutoencoderKLWan.__init__).parameters)
    config = {k: v for k, v in config.items() if k in accepted}  # diffusers versions differ in accepted keys
    if config.get("z_dim", 48) != 48:
        raise ValueError(f"{name} is not the Wan 2.2 VAE PAN 2 uses (48 latent channels)")
    vae = msf.assign_on_meta(lambda: AutoencoderKLWan(**config), {k: v.to(torch.float32) for k, v in state.items()}, name)
    return vae.to(device, torch.float32).eval().requires_grad_(False)


def load_vae_file(path: str | Path, device):
    """A Wan 2.2 VAE in the diffusers layout with its config in the metadata (PAN2_vae_fp16.safetensors)."""
    with safe_open(str(path), "pt") as f:
        raw = (f.metadata() or {}).get("config")
        state = {k: f.get_tensor(k) for k in f.keys()}
    state = msf.strip_prefix(state, ("vae.", "first_stage_model."))
    if "encoder.conv_in.weight" not in state or state.get("decoder.conv_in.weight", torch.zeros(1, 1)).shape[1] != 48:
        raise ValueError(f"{Path(path).name} is not a Wan 2.2 VAE in the diffusers layout (the file pan2-convert makes)")
    config = json.loads(raw) if raw else json.loads((Path(__file__).parent / "wan22_vae_config.json").read_text())
    return vae_from_state(state, config, device, Path(path).name)


# ---------------------------------------------------------------- text encoder (Qwen3.5-9B, 30 of 32 layers)


def build_text_model(state: dict, quant: str = "auto", device="cpu", work_device=None):
    """Text-only Qwen3.5-9B without its last TE_SKIP_LAYERS layers and without the final norm, from an HF-style
    (possibly prefixed, possibly quantized) state dict, which is consumed: its last_hidden_state is the full model's
    hidden_states[-3], what PAN 2 conditions on (IFM's load_qwen_text_encoder)."""
    import inspect

    from transformers import Qwen3_5TextConfig, Qwen3_5TextModel

    raw = json.loads((MOSS_HERE / "qwen35_9b_text_config.json").read_text())
    keep = raw["num_hidden_layers"] - TE_SKIP_LAYERS
    raw["num_hidden_layers"], raw["layer_types"] = keep, raw["layer_types"][:keep]
    cfg = Qwen3_5TextConfig(**raw)
    state = msf.strip_prefix(msf.take_state(state, msf.TE_DROP), msf.TE_PREFIXES)
    for k in list(state):
        if k.startswith(("mtp.", "norm.")) or (k.startswith("layers.") and int(k.split(".")[1]) >= keep):
            del state[k]
    if "layers.0.linear_attn.A_log" not in state:
        raise ValueError("Not a Qwen3.5 text encoder (PAN 2 uses Qwen3.5-9B)")
    with torch.device("meta"):
        model = Qwen3_5TextModel(cfg)
        model.norm = nn.Identity()
    msf.install_state(model, state, quant, torch.bfloat16, ignore_missing=("inv_freq",), work_device=work_device)
    rot_cls = type(model.rotary_emb)
    kwargs = {"device": "cpu"} if "device" in inspect.signature(rot_cls.__init__).parameters else {}
    model.rotary_emb = rot_cls(config=cfg, **kwargs)
    for name, p in model.named_parameters():
        if p.device.type == "meta":
            raise ValueError(f"Text encoder parameter {name} was not loaded")
    return model.to(device).eval().requires_grad_(False)


def load_text_encoder_file(path: str | Path, device: str = "auto", quant: str = "auto"):
    state = msf.read_state(path, torch.bfloat16)
    model = build_text_model(state, quant, "cpu", work_device=msf.quant_target(device).get("work_device"))
    del state
    return msf.place_text_encoder(model, device)
