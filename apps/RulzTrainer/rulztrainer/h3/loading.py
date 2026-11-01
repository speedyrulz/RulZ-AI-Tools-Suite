"""Load MiniMax H3 components from the Hugging Face / diffusers layout (MiniMaxAI/MiniMax-H3) and encode; single
ComfyUI-style files (Comfy-Org/MiniMax-H3, the default and the realistic route on 16 GB cards) go through
h3.single_file.

    model_index.json       MiniMaxH3ModularPipeline (one repository, both task checkpoints)
    transformer/           FL2VA (t2va / fl2va): MiniMaxH3Transformer3DModel, 14 bf16 shards, 66 GB
    transformer_ref/       Ref2VA: the same layout (model_path "MiniMaxAI/MiniMax-H3#ref2va" picks it)
    text_encoder/          Qwen3-VL-32B (Qwen3VLForConditionalGeneration), 14 shards; layers 0-49 are read
    tokenizer/             Qwen2TokenizerFast with H3's added tokens
    vae/                   AutoencoderKLMiniMaxH3 (f16t4d24: 16x spatial, 4x temporal, 24 channels; fp32, 10.4 GB)
    audio_vae/             AutoencoderKLMiniMaxH3Audio (32 kHz, 40 latents/s, 32 channels)
FL2VA/ and Ref2VA/ hold the original-layout copies (per-head interleaved qkv); a local copy of either loads too.
Only the files the trainer reads are downloaded (not the 498 GB repository).

Text features: diffusers' MiniMaxH3TextEncoderStep for a t2va prompt: the prompt verbatim, no chat template and no
special tokens (an empty prompt is a single space, musubi-tuner's unconditional probe), through Qwen3-VL's language
model; the state after decoder layer 50 (hidden_states[50], before any norm) is the conditioning, [S, 5120] bf16. The
50-layer model is all that is built (the Comfy-Org files hold no more), and its final norm is hooked out.

Images: a training picture is a one-frame video. The VAE takes ImageNet-normalized RGB in [0, 1]; a single frame goes
through the spatial encoder alone (diffusers' _encode, ComfyUI's encode: no temporal padding, one latent frame), its
posterior mode normalized by latents_mean / latents_std -> [B, 24, H/16, W/16]. A single latent decodes badly on its
own (ComfyUI: "a still latent is the first latent of a clip"): it is repeated into one 5-latent clip, decoded (spatially
tiled, as released) and its first real frame (index 3, after the decoder's 3-frame pre-padding) kept, ComfyUI's still
path (musubi-tuner duplicates it into 2 latents instead).

Audio: the silence placeholder of the one-frame layout is the audio VAE's posterior mode of 1600 zero samples (2 latent
frames, both channels), normalized, as musubi-tuner encodes it (its docs: the encoding of a zero waveform is not an
all-zero latent). It is computed once on the CPU from the audio VAE (Comfy-Org's 0.6 GB file, or audio_vae/), cached in
models_dir and set on the transformer; the audio VAE is never kept in memory."""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

import torch
import torch.nn.functional as F

from .. import quant as Q
from ..config import TrainConfig
from ..moss import loading as moss_loading
from ..moss import single_file as sf
from . import convert as C
from . import single_file as hsf
from .model import AUDIO_CHANNELS, ONE_FRAME_AUDIO_LATENTS, MiniMaxH3Transformer, timestep_embedding

log = logging.getLogger("rulztrainer")

REPO = "MiniMaxAI/MiniMax-H3"
REF_SUFFIX = "#ref2va"  # model_path suffix: the Ref2VA checkpoint (transformer_ref/, Comfy-Org's *_ref2va_* files)
TRANSFORMER_CLASSES = ("MiniMaxH3Transformer3DModel",)  # diffusers
ORIGINAL_CLASSES = ("MiniMaxH3DiTModel",)  # MiniMaxAI's FL2VA/ and Ref2VA/ folders
PRUNE_RANK, PRUNE_GRID = 8, 4096  # musubi-tuner's --prune_adaln: rank 8 (the published pruned files'), 4096 points
SILENCE_FILE = "minimax_h3_silence_latent.safetensors"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def split_model_path(model_path: str) -> tuple[str, bool]:
    """(repo id or folder, Ref2VA?) of a model_path."""
    p = (model_path or REPO).strip()
    if p.lower().endswith(REF_SUFFIX):
        return p[: -len(REF_SUFFIX)], True
    return p, False


def is_h3_dir(model_dir: Path) -> bool:
    for sub in ("transformer", "transformer_ref"):
        cfg = Path(model_dir) / sub / "config.json"
        try:
            if json.loads(cfg.read_text(encoding="utf-8")).get("_class_name") in TRANSFORMER_CLASSES + ORIGINAL_CLASSES:
                return True
        except (OSError, ValueError):
            continue
    return False


def _gpu_gb(device) -> float:
    dev = torch.device(device)
    return torch.cuda.get_device_properties(dev.index or 0).total_memory / 2**30


def resolve_quant(quant: str, devices, bf16_gb: float, quantized: bool = False) -> str:
    """`auto`: a quantized file keeps its formats; a bf16 source runs in bf16 where the GPUs it runs on hold it with a
    quarter to spare (plus 4 GB), else int8_convrot where that holds half of it, else nvfp4 (a quarter). A model on
    the CPU (block swap, a sampling model) is sized for the smallest GPU. Estimates, not measurements."""
    if quant != "auto":
        return quant
    if quantized:
        return "auto"
    if not torch.cuda.is_available():
        return "none"
    devices = [devices] if isinstance(devices, (str, torch.device)) else list(devices)
    gpus = [d for d in devices if torch.device(d).type == "cuda"]
    if gpus:
        total, where = sum(_gpu_gb(d) for d in gpus), "+".join(str(d) for d in gpus)
    else:
        total = min(_gpu_gb(f"cuda:{i}") for i in range(torch.cuda.device_count()))
        where = "the smallest GPU"
    fmt = ("none" if total >= bf16_gb * 1.25 + 4 else "int8_convrot" if total >= bf16_gb * 0.625 + 4 else "nvfp4")
    if fmt != "none":
        log.info("MiniMax H3 transformer: auto quantization -> %s for %.0f GB (%s; the bf16 weights are %.0f GB)", fmt,
                 total, where, bf16_gb)
    return fmt


# ---------------------------------------------------------------- AdaLN pruning on load


def adaln_basis(time_state: dict, freq_dim: int, rank: int = PRUNE_RANK, grid: int = PRUNE_GRID):
    """(basis [t_dim, rank], mean [t_dim]): a mean-centred rank-`rank` SVD basis of the SiLU'd time-embedding curve over
    `grid` model times in [0, 1], fp32 (musubi-tuner's compute_pruned_adaln_basis)."""
    w1, b1 = time_state["time_embedder.proj_in.weight"].float(), time_state["time_embedder.proj_in.bias"].float()
    w2, b2 = time_state["time_embedder.proj_out.weight"].float(), time_state["time_embedder.proj_out.bias"].float()
    emb = timestep_embedding(torch.linspace(0.0, 1.0, grid), freq_dim)
    curve = F.silu(F.linear(F.silu(F.linear(emb, w1, b1)), w2, b2))
    mean = curve.mean(dim=0)
    _, _, vh = torch.linalg.svd(curve - mean, full_matrices=False)
    return vh[:rank].T.contiguous(), mean


def prune_weight(w, projection: torch.Tensor, work=None) -> torch.Tensor:
    """An AdaLN projection W [out, t_dim] -> W @ [basis | mean] [out, rank + 1], fp32 on the CPU."""
    dev = work if work is not None else "cpu"
    if isinstance(w, Q.QuantWeight):
        w = sf._qw_to(w, dev).dequantize(torch.float32)
    return (w.to(dev, torch.float32) @ projection.to(dev)).cpu()


# ---------------------------------------------------------------- transformer


def build_transformer(reader, device, quant: str = "auto", dtype=torch.bfloat16, quant_devices=None, source: str = "",
                      quantized: bool = False, source_bytes: int = 0,
                      config_overrides: dict | None = None) -> MiniMaxH3Transformer:
    """The port from a reader of port-named (key, value) pairs (reader(want) yields the keys `want` accepts, by the
    source's own key names): read once for the time embedder when a full-AdaLN source is pruned on load, then once
    more for everything, each block Linear quantized as it arrives when the target is a quant format (the bf16 model
    never sits in RAM whole). quant "auto": loading.resolve_quant."""
    target = sf.quant_target(device, quant_devices)
    work = target.get("work_device")
    quant = resolve_quant(quant, quant_devices or device, source_bytes / 2**30, quantized)
    stream_fmt = quant if quant in Q.FORMATS and quant != "none" else None

    time_state = {k: v for k, v in reader(lambda key: "time_embedder." in key)}
    prune = bool(time_state) and not hsf.full_adaln_wanted()
    projection = basis = mean = None
    if prune:
        freq_dim = time_state["time_embedder.proj_in.weight"].shape[1]
        basis, mean = adaln_basis(time_state, freq_dim, PRUNE_RANK, PRUNE_GRID)
        projection = torch.cat([basis, mean[:, None]], dim=1)

    state = {}
    for k, v in reader(lambda key: True):
        if prune and k.endswith(".adaln_proj.linear.weight"):
            v = prune_weight(v, projection, work)
        elif (stream_fmt and hsf.QUANT_FILTER(k) and k.endswith(".weight") and isinstance(v, torch.Tensor)
              and v.dtype != torch.float32 and Q.can_quantize(v, stream_fmt)):
            on = v.to(work) if work is not None else v
            qw = Q.quantize_weight(on, stream_fmt)
            v = qw if work is None or target.get("keep_on_work_device") else sf._qw_to(qw, "cpu")
        state[k] = v
    if prune:
        state["adaln_basis"], state["adaln_mean"] = basis, mean
    if not C.is_h3_state(state):
        raise ValueError(f"{source}: not a MiniMax H3 transformer")
    config = {**C.config_from_state(state), **(config_overrides or {})}
    state.pop("rope.inv_freq", None)  # recomputed from rope_inv_freq_len / rope_theta
    if prune:
        config["adaln_rank"] = PRUNE_RANK
    with torch.device("meta"):
        model = MiniMaxH3Transformer.from_config(config)
    install(model, state, quant, dtype, **target)
    model = model.to(device).eval().requires_grad_(False)
    adaln = ("pruned file (curve table)" if model.pruned else
             f"AdaLN pruned on load (rank {PRUNE_RANK})" if prune else "full AdaLN")
    log.info("Loaded MiniMax H3 transformer from %s: %d blocks, dim %d, %s, quant=%s", source, config["num_layers"],
             config["hidden_size"], adaln, quant)
    return model


def install(model: MiniMaxH3Transformer, state: dict, quant: str, dtype=torch.bfloat16, work_device=None,
            keep_on_work_device: bool = False) -> None:
    """install_state for the blocks, the token refiner and the condition projection in `dtype` (quantized by
    QUANT_FILTER), then the fp32 tensors assigned as they are: the patch projections, the time embedder, the output
    heads and the pruned AdaLN (the checkpoint's fp32 islands)."""
    islands = {k: state.pop(k) for k in list(state)
               if isinstance(state[k], torch.Tensor) and state[k].dtype == torch.float32}
    sf.install_state(model, state, quant, dtype, quant_filter=hsf.QUANT_FILTER, ignore_missing=tuple(islands),
                     work_device=work_device, keep_on_work_device=keep_on_work_device)
    missing, unexpected = model.load_state_dict(islands, strict=False, assign=True)
    if unexpected:
        raise ValueError(f"MiniMax H3 state mismatch: unexpected={unexpected[:5]}")
    for name, t in [*model.named_parameters(), *model.named_buffers()]:
        if t.device.type == "meta":
            raise ValueError(f"MiniMax H3 transformer tensor {name} was not loaded")


def _shard_reader(tdir: Path, stem: str, dtype, rename):
    """reader(want) over a sharded HF folder: bf16 except F32 tensors, renamed to the port's keys by rename(dict)."""
    from safetensors import safe_open

    index = tdir / f"{stem}.safetensors.index.json"
    files = sorted(set(json.loads(index.read_text())["weight_map"].values())) if index.is_file() \
        else [f"{stem}.safetensors"]

    def reader(want):
        pending = {}
        for name in files:
            with safe_open(str(tdir / name), "pt") as f:
                for k in f.keys():
                    if not want(k):
                        continue
                    t = f.get_tensor(k)
                    if t.is_floating_point() and t.dtype != torch.float32:
                        t = t.to(dtype)
                    m = re.match(r"^(.*\.attn)\.to_([qkv])\.weight$", k)
                    if m:  # diffusers' split q / k / v: fused once all three are read
                        group = pending.setdefault(m.group(1), {})
                        group[k] = t
                        if len(group) == 3:
                            yield from rename(pending.pop(m.group(1))).items()
                        continue
                    yield from rename({k: t}).items()
        if pending:
            raise ValueError(f"{tdir}: incomplete q / k / v for {sorted(pending)[:3]}")
    return reader, sum((tdir / n).stat().st_size for n in files)


def load_transformer(model_dir: Path, device, quant: str = "auto", dtype=torch.bfloat16, quant_devices=None,
                     ref: bool = False) -> MiniMaxH3Transformer:
    """The FL2VA (or Ref2VA) transformer of a downloaded MiniMaxAI/MiniMax-H3 (transformer/ or transformer_ref/), or
    of a folder in MiniMaxAI's original layout (FL2VA/transformer, per-head interleaved qkv)."""
    sub = "transformer_ref" if ref and (Path(model_dir) / "transformer_ref").is_dir() else "transformer"
    tdir = Path(model_dir) / sub
    config = json.loads((tdir / "config.json").read_text(encoding="utf-8"))
    cls = config.get("_class_name")
    if cls in TRANSFORMER_CLASSES:
        reader, size = _shard_reader(tdir, "diffusion_pytorch_model", dtype, C.diffusers_to_port)
        overrides = C.diffusers_config_to_port(config)
    elif cls in ORIGINAL_CLASSES:
        heads, head_dim = config["num_attention_heads"], config["attention_head_dim"]
        reader, size = _shard_reader(tdir, "model", dtype, lambda s: C.original_to_port(
            s, interleaved=True, heads=heads, head_dim=head_dim))
        overrides = {k: config[k] for k in MiniMaxH3Transformer.CONFIG_KEYS if k in config}
    else:
        raise ValueError(f"Not a MiniMax H3 transformer: {cls!r}")
    overrides.pop("adaln_curve_grid", None)
    return build_transformer(reader, device, quant, dtype, quant_devices, source=f"{Path(model_dir).name}/{sub}",
                             quantized=False, source_bytes=size, config_overrides=overrides)


# ---------------------------------------------------------------- the silence placeholder


@torch.no_grad()
def silence_latent(encoder, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """[4, 32]: the audio posterior mode of a zero waveform of 2 latent frames per channel (both stereo channels as
    the batch, as the pipeline encodes them), normalized, channel-major (left f0, f1, right f0, f1)."""
    hop = int(getattr(encoder, "hop_length", 800))
    wav = torch.zeros(AUDIO_CHANNELS, 1, ONE_FRAME_AUDIO_LATENTS * hop)
    post = encoder.encode(wav).latent_dist
    z = post.mode().float().transpose(1, 2)  # [2, frames, 32]
    if z.shape[1] != ONE_FRAME_AUDIO_LATENTS:
        raise ValueError(f"the audio VAE gave {z.shape[1]} latent frames for {wav.shape[-1]} samples")
    z = (z - mean.view(1, 1, -1)) / std.view(1, 1, -1)
    return z.reshape(-1, z.shape[-1]).contiguous()


def _file_id(path: Path) -> str:
    st = path.stat()
    return f"{path.name}:{st.st_size}:{int(st.st_mtime)}"


def audio_vae_source(cfg: TrainConfig, model_dir: Path | None) -> Path:
    """Where the audio VAE comes from: the HF folder's audio_vae/, Comfy-Org's file next to the video VAE file, or
    Comfy-Org's file downloaded (0.6 GB, the Hugging Face cache)."""
    if model_dir is not None and (Path(model_dir) / "audio_vae" / "config.json").is_file() and \
            not moss_loading._use_file(cfg, "vae_file"):
        return Path(model_dir) / "audio_vae"
    if cfg.vae_file:
        near = Path(cfg.vae_file).expanduser().parent / Path(hsf.AUDIO_VAE_FILE).name
        if near.is_file():
            return near
    from ..downloads import ensure_hf_file

    return ensure_hf_file(hsf.COMFY_REPO, hsf.AUDIO_VAE_FILE, Path(hsf.AUDIO_VAE_FILE).name)


def get_silence(cfg: TrainConfig, model_dir: Path | None) -> torch.Tensor:
    """The silence placeholder, from the cache in models_dir or computed once from the audio VAE (CPU, then freed)."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    src = audio_vae_source(cfg, model_dir)
    ident = _file_id(src / "diffusion_pytorch_model.safetensors" if src.is_dir() else src)
    cache = Path(cfg.models_dir) / SILENCE_FILE
    if cache.is_file():
        try:
            with safe_open(str(cache), "pt") as f:
                if (f.metadata() or {}).get("source") == ident:
                    return f.get_tensor("silence")
        except Exception:  # noqa: BLE001 - recompute
            pass
    if src.is_dir():
        from diffusers import AutoencoderKLMiniMaxH3Audio

        enc = AutoencoderKLMiniMaxH3Audio.from_pretrained(str(src), torch_dtype=torch.float32).eval()
        mean, std = torch.tensor(enc.config.latents_mean), torch.tensor(enc.config.latents_std)
    else:
        enc, mean, std = hsf.load_audio_encoder_file(src)
    silence = silence_latent(enc, mean, std)
    del enc
    cache.parent.mkdir(parents=True, exist_ok=True)
    save_file({"silence": silence}, str(cache), metadata={"source": ident})
    log.info("MiniMax H3 audio placeholder: the silence latent of %s (cached in %s)", src.name, cache)
    return silence


def attach_silence(model: MiniMaxH3Transformer, cfg: TrainConfig, model_dir: Path | None) -> MiniMaxH3Transformer:
    try:
        model.audio_silence = get_silence(cfg, model_dir)
    except Exception as e:  # noqa: BLE001 - the model still runs on a zero placeholder
        log.warning("MiniMax H3: the audio VAE could not give the silence latent (%s); the audio placeholder is zeros",
                    e)
    return model


# ---------------------------------------------------------------- VAE (video), the trainer's layout


def load_vae(model_dir: Path, device, dtype=torch.bfloat16):
    vae = hsf.vae_class().from_pretrained(str(Path(model_dir) / "vae"), torch_dtype=dtype)
    return vae.to(device).eval().requires_grad_(False)


def _stats(vae, like: torch.Tensor):
    shape = (1, vae.config.latent_channels, 1, 1)
    mean = torch.tensor(vae.config.latents_mean).view(shape).to(like)
    std = torch.tensor(vae.config.latents_std).view(shape).to(like)
    return mean, std


def _pixel_norm(like: torch.Tensor):
    mean = torch.tensor(IMAGENET_MEAN, device=like.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=like.device).view(1, 3, 1, 1)
    return mean, std


def encode_images(vae, images: torch.Tensor) -> torch.Tensor:
    """images: [B,3,H,W] in [-1,1] (H, W multiples of 32). Returns normalized latents [B, 24, H/16, W/16]: each image
    a one-frame video, ImageNet-normalized, its posterior mode."""
    images = images.to(vae.device, torch.float32)
    pm, ps = _pixel_norm(images)
    x = (((images + 1.0) / 2.0 - pm) / ps).unsqueeze(2)  # [B, 3, 1, H, W]
    z = vae.encode(x).latent_dist.mode()[:, :, 0].float()
    mean, std = _stats(vae, z)
    return (z - mean) / std


STILL_REPEAT = 5  # one clip of tokens_chunk_size latents (ComfyUI's still path)


def _decode(vae, latents: torch.Tensor) -> torch.Tensor:
    z = latents.to(vae.device, torch.float32)
    mean, std = _stats(vae, z)
    z = (z * std + mean).unsqueeze(2).repeat(1, 1, int(getattr(vae, "tokens_chunk_size", STILL_REPEAT)), 1, 1)
    z = z.to(next(vae.post_quant_conv.parameters()).dtype)
    if hasattr(vae, "_decode_clip"):  # one clip, spatially tiled as released; frame 3 = the first real frame
        frames = vae._decode_clip(z)
        x = frames[:, :, int(getattr(vae, "frame_pre_padding", 3))]
    else:
        x = vae.decode(z, return_dict=False)[0][:, :, 0]
    pm, ps = _pixel_norm(x)
    return (x.float() * ps + pm) * 2.0 - 1.0  # ImageNet space -> [0, 1] -> [-1, 1]


@torch.no_grad()
def decode_latents(vae, latents: torch.Tensor) -> torch.Tensor:
    """Inverse of encode_images. Returns images [B,3,H,W] in [-1,1] (see the module docstring for the still path)."""
    return _decode(vae, latents).clamp(-1, 1)


def decode_latents_grad(vae, latents: torch.Tensor) -> torch.Tensor:
    """decode_latents with gradients to `latents` (identity loss on a face crop), not clamped."""
    return _decode(vae, latents)


def grad_vae(vae):
    """The VAE decode_latents_grad runs on: its decoder is already in the compute dtype and frozen, so itself."""
    return vae


# ---------------------------------------------------------------- text encoder (Qwen3-VL-32B, layers 0-49)


def _te_reader(te_dir: Path) -> dict:
    """The language model's first 50 layers and the embedding from an HF text_encoder/ folder: only the shards that
    hold them are opened and only their keys read."""
    from safetensors import safe_open

    index = json.loads((te_dir / "model.safetensors.index.json").read_text())["weight_map"]
    keep = {}
    for k, f in index.items():
        m = re.search(r"language_model\.layers\.(\d+)\.", k)
        if (m and int(m.group(1)) < hsf.TE_LAYERS) or k.endswith("language_model.embed_tokens.weight"):
            keep.setdefault(f, []).append(k)
    state = {}
    for f, keys in sorted(keep.items()):
        with safe_open(str(te_dir / f), "pt") as h:
            for k in keys:
                state[k] = h.get_tensor(k).to(torch.bfloat16)
    return state


def load_text_encoder(model_dir: Path, device: str = "auto", quant: str = "auto"):
    """Qwen3-VL-32B's language model (50 layers, text only) + H3's tokenizer. quant int8 / int8_convrot / nvfp4
    quantizes its Linear layers (the 50 bf16 layers are about 49 GB)."""
    from transformers import AutoTokenizer

    from ..fibo.loading import place_text_encoder
    from ..krea2.single_file import hf_text_config

    tokenizer = AutoTokenizer.from_pretrained(str(Path(model_dir) / "tokenizer"))
    hsf.check_tokenizer(tokenizer)
    te_dir = Path(model_dir) / "text_encoder"
    model = hsf.build_text_model(_te_reader(te_dir), "none" if quant == "auto" else quant, "cpu",
                                 config=hf_text_config(te_dir), work_device=sf.quant_target(device).get("work_device"))
    return place_text_encoder(model, device), tokenizer


def _input_device(model) -> torch.device:
    return model.get_input_embeddings().weight.device


@torch.no_grad()
def encode_prompt(text_encoder, tokenizer, prompt: str, max_length: int = 1024) -> torch.Tensor:
    """Text features for one prompt: [S, 5120] bf16 on the CPU (see the module docstring)."""
    ids = tokenizer(prompt or " ", add_special_tokens=False, truncation=True, max_length=max(8, int(max_length)),
                    return_tensors="pt").input_ids.to(_input_device(text_encoder))
    n = ids.shape[1]
    position_ids = torch.arange(n, device=ids.device).view(1, 1, n).expand(3, 1, n)  # text only: t = h = w
    norm = getattr(text_encoder, "norm", None) or text_encoder.language_model.norm
    handle = norm.register_forward_hook(lambda module, args, output: args[0])  # the state after layer 50, unnormed
    try:
        out = text_encoder(input_ids=ids, attention_mask=torch.ones_like(ids), position_ids=position_ids)
    finally:
        handle.remove()
    return out.last_hidden_state[0].to(torch.bfloat16).cpu()


def get_tokenizer(cfg: TrainConfig, model_dir: Path | None):
    from transformers import AutoTokenizer

    if moss_loading._use_file(cfg, "text_encoder_file") or model_dir is None:
        tok = AutoTokenizer.from_pretrained(str(hsf.resolve_tokenizer(cfg.tokenizer_path, cfg.models_dir)))
    else:
        tok = AutoTokenizer.from_pretrained(str(Path(model_dir) / "tokenizer"))
    hsf.check_tokenizer(tok)
    return tok


def prompt_tokens(tokenizer, prompt: str) -> tuple[str, list, int, int]:
    """encode_prompt's text (the prompt itself) with its token offsets; nothing is dropped from the cache."""
    text = prompt or " "
    offsets = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]
    return text, [tuple(o) for o in offsets], 0, len(offsets)


def pad_embeds(rows: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """[S_i, 5120] rows -> [B, S, 5120] (right-padded with zeros) and a [B, S] bool mask of the real tokens."""
    length = max(r.shape[0] for r in rows)
    embeds = rows[0].new_zeros(len(rows), length, rows[0].shape[-1])
    mask = torch.zeros(len(rows), length, dtype=torch.bool)
    for i, r in enumerate(rows):
        embeds[i, : r.shape[0]] = r
        mask[i, : r.shape[0]] = True
    return embeds, mask


# ---------------------------------------------------------------- config-driven entry points


# what the trainer reads of the 498 GB repository (the other transformer, the original-layout folders and the text
# encoder's last 3 shards, layers 50-63 and the vision tower, are left out)
def _allow_patterns(ref: bool) -> list[str]:
    tdir = "transformer_ref" if ref else "transformer"
    return ["model_index.json", f"{tdir}/*", "tokenizer/*", "vae/*", "audio_vae/*", "text_encoder/config.json",
            "text_encoder/model.safetensors.index.json", *[f"text_encoder/model-{i:05d}-of-00014.safetensors"
                                                             for i in range(1, 12)]]


def resolve_model_dir(model_path: str, models_dir: str = "models") -> Path:
    path, ref = split_model_path(model_path)
    p = Path(path).expanduser()
    if p.is_dir():
        return p.resolve()
    local = Path(models_dir) / path.replace("/", "__")
    tdir = local / ("transformer_ref" if ref else "transformer")
    if (tdir / "config.json").is_file() and (local / "text_encoder" / "config.json").is_file():
        return local.resolve()
    from huggingface_hub import snapshot_download

    log.info("Downloading %s (%s: about 66 GB of transformer, 52 GB of text encoder layers, 11 GB of VAEs) to %s",
             path, "Ref2VA" if ref else "FL2VA", local)
    snapshot_download(path, local_dir=str(local), allow_patterns=_allow_patterns(ref))
    return local.resolve()


def _need_dir(cfg: TrainConfig, model_dir: Path | None, what: str) -> Path:
    if model_dir is None:
        raise ValueError(f"No {what} file is set and no Hugging Face model to take it from: set the file (model "
                         f"source 'files') or the MiniMaxAI/MiniMax-H3 repo id / a diffusers folder")
    return model_dir


def resolve_for_config(cfg: TrainConfig) -> Path | None:
    """Download / locate the HF model dir when any component comes from it (single files cover the rest)."""
    if cfg.model_source == "files" and all(getattr(cfg, f) for f in ("transformer_file", "text_encoder_file", "vae_file")):
        return None
    return resolve_model_dir(cfg.model_path, cfg.models_dir)


def get_transformer(cfg: TrainConfig, model_dir: Path | None, device: str | None = None) -> MiniMaxH3Transformer:
    use_file = moss_loading._use_file(cfg, "transformer_file")
    if use_file:
        moss_loading._check_files(cfg, "transformer_file")
    else:
        model_dir = _need_dir(cfg, model_dir, "diffusion model")
    ref = split_model_path(cfg.model_path)[1]
    quant_devices = None
    if cfg.multi_gpu == "model_split" and device is None:
        # the trainer splits it right after loading: stay on the CPU and size "auto" for all the split GPUs
        from ..multigpu import parse_devices

        quant_devices, device = parse_devices(cfg.devices, cfg.device), "cpu"
    device = device or cfg.device
    if use_file:
        model = hsf.load_transformer_file(cfg.transformer_file, device, cfg.transformer_quant,
                                          quant_devices=quant_devices)
    else:
        model = load_transformer(model_dir, device, cfg.transformer_quant, quant_devices=quant_devices, ref=ref)
    return attach_silence(model, cfg, model_dir)


def get_vae(cfg: TrainConfig, model_dir: Path | None):
    if moss_loading._use_file(cfg, "vae_file"):
        moss_loading._check_files(cfg, "vae_file")
        return hsf.load_vae_file(cfg.vae_file, cfg.device)
    return load_vae(_need_dir(cfg, model_dir, "VAE"), cfg.device)


def get_text_encoder(cfg: TrainConfig, model_dir: Path | None):
    if moss_loading._use_file(cfg, "text_encoder_file"):
        from transformers import AutoTokenizer

        moss_loading._check_files(cfg, "text_encoder_file")
        tokenizer = AutoTokenizer.from_pretrained(str(hsf.resolve_tokenizer(cfg.tokenizer_path, cfg.models_dir)))
        hsf.check_tokenizer(tokenizer)
        model = hsf.load_text_encoder_file(cfg.text_encoder_file, cfg.text_encoder_device, cfg.text_encoder_quant)
        return model, tokenizer
    return load_text_encoder(_need_dir(cfg, model_dir, "text encoder"), cfg.text_encoder_device,
                             cfg.text_encoder_quant)


def load_any_transformer(path: str, device, quant: str = "auto", models_dir: str = "models") -> MiniMaxH3Transformer:
    """Load a transformer from a single .safetensors file, a diffusers directory, or a HF repo id (a sampling model);
    the silence placeholder comes from the cache the training model left in models_dir."""
    p = Path(path).expanduser()
    if p.is_file():
        model = hsf.load_transformer_file(p, device, quant)
    else:
        repo, ref = split_model_path(path)
        d = Path(repo).expanduser()
        model_dir = d if d.is_dir() else resolve_model_dir(path, models_dir)
        model = load_transformer(model_dir, device, quant, ref=ref)
    cache = Path(models_dir) / SILENCE_FILE
    if cache.is_file():
        from safetensors.torch import load_file

        model.audio_silence = load_file(str(cache))["silence"]
    return model
