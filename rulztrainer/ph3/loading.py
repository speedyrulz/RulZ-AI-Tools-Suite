"""Load P-H3-Image components and encode prompts. Single ComfyUI-style files (PrunaAI/P-H3-Image-Comfy, the default)
or the diffusers release (PrunaAI/P-H3-Image, a ModularPipeline):

    diffusion_models/p-h3-image_{int8_convrot,bf16}   H3's transformer, pruned (a [4097, 16] AdaLN curve table) with
                                                       Pruna's LoRA merged in: loads through h3.single_file as it is
    text_encoders/p-h3-image_text_encoder_{...}       Qwen3-VL-4B (HF key names, vision tower included) + the text
                                                       connector (text_connector.*, fp32)
    vae/p-h3-image_vae_{bf16,int8_convrot}             H3's video VAE encoder + a single-frame decoder (ph3/vae.py)
    transformer/, text_encoder/, text_connector/,      the diffusers layout (the full-AdaLN transformer is pruned on
    tokenizer/, vae/                                   load to rank 16; the 19 GB prompt upsampler is never fetched)

Text features: diffusers' PH3ImageTextEncoderStep for a text-to-image prompt: the prompt verbatim, no chat template and
no special tokens (an empty prompt is a single space, as H3's), through Qwen3-VL-4B's language model with every hidden
state kept; the connector concatenates hidden_states[13, 19, 25, 31, 36] (the last is the final normed state),
standardizes, projects to 5120, rescales, and replaces token 0 by its sink embedding -> [S, 5120] bf16, the width of
H3's own conditioning.

The transformer runs H3's one-frame layout with two differences, set by configure(): the audio rows follow the video
sigma (both of P-H3's schedulers sample the same sigmas: no shift-3 remap), and the audio placeholder is noised zeros
(P-H3 ships no audio VAE; ai-toolkit trains it on zeros too).

Images encode as H3's (ImageNet-normalized, the posterior mode, normalized by latents_mean / latents_std); a latent
decodes as one frame."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from torch import nn

from .. import quant as Q
from ..config import TrainConfig
from ..h3 import loading as HL
from ..h3 import single_file as hsf
from ..moss import loading as moss_loading
from ..moss import single_file as sf

log = logging.getLogger("rulztrainer")

REPO = "PrunaAI/P-H3-Image"
COMFY_REPO = "PrunaAI/P-H3-Image-Comfy"
HIDDEN_STATE_LAYERS = (13, 19, 25, 31, 36)  # hidden_states indices the connector reads (36: after the final norm)
TEXT_DIM = 5120
PRUNE_RANK = 16  # a full-AdaLN source is pruned to P-H3's own table width
# the checkpoint's sampling grid (modular_model_index.json sample_sigmas): 8 steps, no guidance
SAMPLE_SIGMAS = (1.0, 0.984753, 0.967905, 0.951688, 0.925372, 0.881923, 0.810083, 0.528718, 0.0)
CONNECTOR_PREFIX = "text_connector."
CONNECTOR_FILE = "text_connector/diffusion_pytorch_model.safetensors"
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
SINGLE_FRAME_PROJ_OUT = 3 * 16 * 16  # decoder.proj_out rows of the single-frame decoder (H3's video decoder: 4x)
ALLOW_PATTERNS = ["modular_model_index.json", "transformer/*", "text_encoder/*", "text_connector/*", "tokenizer/*",
                  "vae/*"]


# ---------------------------------------------------------------- transformer


def configure(model):
    """H3's transformer as P-H3-Image runs it (see the module docstring)."""
    model.shift_audio = model.shift_video  # audio sigma = video sigma
    model.audio_silence = torch.zeros(model.audio_frames * 2, model.audio_channels)
    model.is_distilled = True
    return model


def is_ph3_dir(model_dir: Path) -> bool:
    """A diffusers P-H3-Image folder (its pipeline index names PH3Image, or it has the text connector)."""
    d = Path(model_dir)
    try:
        cls = json.loads((d / "modular_model_index.json").read_text(encoding="utf-8")).get("_class_name", "")
        if str(cls).startswith("PH3Image"):
            return True
    except (OSError, ValueError):
        pass
    return (d / "text_connector" / "config.json").is_file()


def resolve_model_dir(model_path: str, models_dir: str = "models") -> Path:
    path = (model_path or REPO).strip()
    p = Path(path).expanduser()
    if p.is_dir():
        return p.resolve()
    local = Path(models_dir) / path.replace("/", "__")
    if (local / "transformer" / "config.json").is_file() and (local / "text_connector" / "config.json").is_file():
        return local.resolve()
    from huggingface_hub import snapshot_download

    log.info("Downloading %s (about 66 GB of transformer, 9 GB of text encoder, 5.6 GB of VAE; not the prompt "
             "upsampler) to %s", path, local)
    snapshot_download(path, local_dir=str(local), allow_patterns=ALLOW_PATTERNS)
    return local.resolve()


def resolve_for_config(cfg: TrainConfig) -> Path | None:
    if cfg.model_source == "files" and all(getattr(cfg, f) for f in ("transformer_file", "text_encoder_file", "vae_file")):
        return None
    return resolve_model_dir(cfg.model_path, cfg.models_dir)


def _need_dir(model_dir: Path | None, what: str) -> Path:
    if model_dir is None:
        raise ValueError(f"No {what} file is set and no Hugging Face model to take it from: set the file (model "
                         f"source 'files') or the {REPO} repo id / a diffusers folder")
    return model_dir


def load_transformer_file(path, device, quant: str = "auto", quant_devices=None):
    return configure(hsf.load_transformer_file(path, device, quant, quant_devices=quant_devices))


def load_transformer(model_dir: Path, device, quant: str = "auto", quant_devices=None):
    return configure(HL.load_transformer(model_dir, device, quant, quant_devices=quant_devices,
                                         prune_rank=PRUNE_RANK))


def get_transformer(cfg: TrainConfig, model_dir: Path | None, device: str | None = None):
    use_file = moss_loading._use_file(cfg, "transformer_file")
    if use_file:
        moss_loading._check_files(cfg, "transformer_file")
    else:
        model_dir = _need_dir(model_dir, "diffusion model")
    quant_devices = None
    if cfg.multi_gpu == "model_split" and device is None:
        from ..multigpu import parse_devices

        quant_devices, device = parse_devices(cfg.devices, cfg.device), "cpu"
    device = device or cfg.device
    if use_file:
        return load_transformer_file(cfg.transformer_file, device, cfg.transformer_quant, quant_devices)
    return load_transformer(model_dir, device, cfg.transformer_quant, quant_devices)


def load_any_transformer(path: str, device, quant: str = "auto", models_dir: str = "models"):
    """A sampling model: a single .safetensors file, a diffusers folder or a repo id."""
    p = Path(path).expanduser()
    if p.is_file():
        return load_transformer_file(p, device, quant)
    return load_transformer(p.resolve() if p.is_dir() else resolve_model_dir(path, models_dir), device, quant)


# ---------------------------------------------------------------- text encoder (Qwen3-VL-4B + the text connector)


class TextConnector(nn.Module):
    """diffusers' PH3ImageTextConnector: hidden states -> [B, S, 5120], computed in fp32."""

    def __init__(self, layers=HIDDEN_STATE_LAYERS, text_hidden_size: int = 2560, out_channels: int = TEXT_DIM):
        super().__init__()
        self.layers = tuple(int(i) for i in layers)
        width = len(self.layers) * text_hidden_size
        self.input_mean = nn.Parameter(torch.zeros(width))
        self.input_std = nn.Parameter(torch.ones(width))
        self.proj = nn.Linear(width, out_channels, bias=False)
        self.output_mean = nn.Parameter(torch.zeros(out_channels))
        self.output_std = nn.Parameter(torch.ones(out_channels))
        self.sink_embedding = nn.Parameter(torch.zeros(out_channels))

    def forward(self, hidden_states) -> torch.Tensor:
        dev = self.proj.weight.device
        x = torch.cat([hidden_states[i].to(dev, torch.float32) for i in self.layers], dim=-1)
        x = (x - self.input_mean.float()) / self.input_std.float()
        x = nn.functional.linear(x, self.proj.weight.float()) * self.output_std.float() + self.output_mean.float()
        sink = self.sink_embedding.float().expand(x.shape[0], 1, -1)
        return torch.cat([sink, x[:, 1:]], dim=1)


CONNECTOR_KEYS = ("input_mean", "input_std", "proj.weight", "output_mean", "output_std", "sink_embedding")


def build_connector(state: dict) -> TextConnector:
    """The connector from its six fp32 tensors (prefix already stripped)."""
    missing = [k for k in CONNECTOR_KEYS if k not in state]
    if missing:
        raise ValueError(f"P-H3-Image text connector: missing {missing}")
    width = state["input_mean"].shape[0]
    out = state["output_mean"].shape[0]
    hidden = width // len(HIDDEN_STATE_LAYERS)
    if hidden * len(HIDDEN_STATE_LAYERS) != width or tuple(state["proj.weight"].shape) != (out, width):
        raise ValueError(f"P-H3-Image text connector: unexpected shapes (input {width}, proj "
                         f"{tuple(state['proj.weight'].shape)})")
    with torch.device("meta"):
        conn = TextConnector(HIDDEN_STATE_LAYERS, hidden, out)
    conn.load_state_dict({k: state[k].float() for k in CONNECTOR_KEYS}, strict=True, assign=True)
    return conn.eval().requires_grad_(False)


def _read_connector_file(path: Path, prefix: str = "") -> dict:
    from safetensors import safe_open

    with safe_open(str(path), "pt") as f:
        return {k[len(prefix):]: f.get_tensor(k).float() for k in f.keys() if k.startswith(prefix)}


def connector_from_repo() -> TextConnector:
    """The released connector (0.26 GB, fp32), for a text encoder file without one."""
    from ..downloads import ensure_hf_file

    path = ensure_hf_file(REPO, CONNECTOR_FILE, "p-h3-image_text_connector.safetensors")
    return build_connector(_read_connector_file(Path(path)))


def attach_connector(model, connector: TextConnector):
    expect = len(connector.layers) * model.config.hidden_size
    if connector.input_mean.shape[0] != expect:
        raise ValueError(f"The text connector reads {connector.input_mean.shape[0]} channels, the text encoder gives "
                         f"{expect}: use P-H3-Image's own Qwen3-VL-4B")
    if model.config.num_hidden_layers < max(connector.layers):
        raise ValueError(f"P-H3-Image reads hidden state {max(connector.layers)}; the text encoder has "
                         f"{model.config.num_hidden_layers} layers")
    model.text_connector = connector
    return model


def load_text_encoder_file(path, device: str = "auto", quant: str = "auto"):
    """Qwen3-VL-4B's language model (text only) from a P-H3-Image text encoder file, with its connector (the file's
    text_connector.* tensors, kept fp32; downloaded when the file has none)."""
    from ..fibo.loading import place_text_encoder
    from ..krea2.single_file import build_text_model

    path = Path(path)
    connector_state = _read_connector_file(path, CONNECTOR_PREFIX)
    state = sf.read_state(path, torch.bfloat16)
    for k in [k for k in state if k.startswith(CONNECTOR_PREFIX)]:
        state.pop(k)
    model = build_text_model(state, "none" if quant == "auto" else quant, "cpu",
                             work_device=sf.quant_target(device).get("work_device"))
    del state
    connector = build_connector(connector_state) if connector_state else connector_from_repo()
    attach_connector(model, connector)
    n_q = sum(1 for m in model.modules() if isinstance(m, Q.QuantLinear))
    log.info("Loaded P-H3-Image text encoder (Qwen3-VL-4B, %d layers, + text connector) from %s%s",
             model.config.num_hidden_layers, path.name, f", {n_q} quantized Linear layers" if n_q else "")
    return place_text_encoder(model, device)


def load_text_encoder(model_dir: Path, device: str = "auto", quant: str = "auto"):
    """From a diffusers folder: text_encoder/ (the vision tower and LM head are never loaded), text_connector/,
    tokenizer/."""
    from transformers import AutoTokenizer

    from ..fibo.loading import place_text_encoder
    from ..krea2.single_file import load_hf_text_model

    tokenizer = AutoTokenizer.from_pretrained(str(Path(model_dir) / "tokenizer"))
    model = load_hf_text_model(Path(model_dir) / "text_encoder", "none" if quant == "auto" else quant, device)
    cfile = Path(model_dir) / CONNECTOR_FILE
    attach_connector(model, build_connector(_read_connector_file(cfile)) if cfile.is_file() else connector_from_repo())
    return place_text_encoder(model, device), tokenizer


def resolve_tokenizer(tokenizer_path: str, models_dir: str = "models") -> Path:
    """P-H3-Image's tokenizer folder: the given one, a downloaded PrunaAI/P-H3-Image's tokenizer/, or its files
    downloaded once (about 11 MB)."""
    if tokenizer_path:
        p = Path(tokenizer_path).expanduser()
        for cand in (p, p / "tokenizer"):
            if (cand / "tokenizer_config.json").is_file():
                return cand
        raise FileNotFoundError(f"No tokenizer_config.json under {p}")
    for cand in sorted(Path(models_dir).glob("PrunaAI__P-H3-Image*/tokenizer/tokenizer_config.json")):
        return cand.parent
    local = Path(models_dir) / "tokenizer__P-H3-Image"
    if not all((local / f).is_file() for f in TOKENIZER_FILES):
        from huggingface_hub import hf_hub_download

        log.info("Downloading the P-H3-Image tokenizer to %s", local)
        local.mkdir(parents=True, exist_ok=True)
        for f in TOKENIZER_FILES:
            src = Path(hf_hub_download(REPO, f"tokenizer/{f}"))
            (local / f).write_bytes(src.read_bytes())
    return local


def get_tokenizer(cfg: TrainConfig, model_dir: Path | None):
    from transformers import AutoTokenizer

    if moss_loading._use_file(cfg, "text_encoder_file") or model_dir is None:
        return AutoTokenizer.from_pretrained(str(resolve_tokenizer(cfg.tokenizer_path, cfg.models_dir)))
    return AutoTokenizer.from_pretrained(str(Path(model_dir) / "tokenizer"))


def get_text_encoder(cfg: TrainConfig, model_dir: Path | None):
    if moss_loading._use_file(cfg, "text_encoder_file"):
        moss_loading._check_files(cfg, "text_encoder_file")
        tokenizer = get_tokenizer(cfg, None)
        return load_text_encoder_file(cfg.text_encoder_file, cfg.text_encoder_device, cfg.text_encoder_quant), tokenizer
    return load_text_encoder(_need_dir(model_dir, "text encoder"), cfg.text_encoder_device, cfg.text_encoder_quant)


def _input_device(model) -> torch.device:
    return model.get_input_embeddings().weight.device


@torch.no_grad()
def encode_prompts(text_encoder, tokenizer, prompts: list[str], max_length: int = 1024) -> list[torch.Tensor]:
    """Text features for each prompt: [S_i, 5120] bf16 on the CPU (see the module docstring), in one forward
    (right-padded: the causal encoder's real tokens never see the padding)."""
    rows = [tokenizer(prompt or " ", add_special_tokens=False, truncation=True, max_length=max(8, int(max_length)),
                      return_tensors="pt").input_ids[0] for prompt in prompts]
    ids, mask = moss_loading.right_pad(rows)
    device = _input_device(text_encoder)
    b, n = ids.shape
    position_ids = torch.arange(n, device=device).view(1, 1, n).expand(3, b, n)  # text only: t = h = w
    out = text_encoder(input_ids=ids.to(device), attention_mask=mask.to(device), position_ids=position_ids,
                       output_hidden_states=True)
    cond = text_encoder.text_connector(out.hidden_states).to(torch.bfloat16).cpu()
    return [cond[i, : r.numel()].clone() for i, r in enumerate(rows)]


def encode_prompt(text_encoder, tokenizer, prompt: str, max_length: int = 1024) -> torch.Tensor:
    return encode_prompts(text_encoder, tokenizer, [prompt], max_length)[0]


prompt_tokens = HL.prompt_tokens
pad_embeds = HL.pad_embeds


# ---------------------------------------------------------------- VAE


def _vae_config(stats: dict, num_layers: int) -> dict:
    config = dict(hsf.VAE_CONFIG)
    config["decoder_num_layers"] = num_layers
    for key in ("latents_mean", "latents_std"):
        if key in stats:
            config[key] = [float(x) for x in stats[key]]
    return config


def load_vae_file(path, device, dtype=torch.bfloat16):
    """P-H3-Image's VAE file (bf16, or int8 ConvRot dequantized on read), in MiniMax's original key layout."""
    from .vae import build_vae

    state = {k: (v.dequantize(torch.float32) if isinstance(v, Q.QuantWeight) else v)
             for k, v in hsf.read_state(path, torch.float32).items()}
    state = sf.strip_prefix(state, ("vae.", "first_stage_model."))
    if "decoder.transformer_blocks.0.attn.to_qkv.weight" not in state or "encoder.conv_in.weight" not in state:
        raise ValueError(f"{Path(path).name} is not a P-H3-Image VAE")
    proj = state.get("decoder.proj_out.weight")
    if proj is not None and proj.shape[0] != SINGLE_FRAME_PROJ_OUT:
        raise ValueError(f"{Path(path).name} is a video VAE (MiniMax H3's): P-H3-Image needs its own single-frame "
                         "VAE, vae/p-h3-image_vae_bf16.safetensors")
    stats = {k: state.pop(k).float() for k in ("latents_mean", "latents_std") if k in state}
    state.pop("decoder.mask_token", None)
    layers = sum(1 for k in state if k.startswith("decoder.transformer_blocks.") and k.endswith(".scale1"))
    config = _vae_config(stats, layers)
    state = hsf.vae_original_to_diffusers(state, heads=config["decoder_num_attention_heads"])
    vae = build_vae(state, config, dtype)
    return vae.to(device).eval().requires_grad_(False)


def load_vae(model_dir: Path, device, dtype=torch.bfloat16):
    """The diffusers vae/ folder (AutoencoderKLPH3Image: H3's diffusers key names)."""
    from safetensors import safe_open

    from .vae import build_vae

    vdir = Path(model_dir) / "vae"
    cfg = json.loads((vdir / "config.json").read_text(encoding="utf-8"))
    with safe_open(str(vdir / "diffusion_pytorch_model.safetensors"), "pt") as f:
        state = {k: f.get_tensor(k).float() for k in f.keys()}
    layers = sum(1 for k in state if k.startswith("decoder.transformer_blocks.") and k.endswith(".scale1"))
    config = _vae_config({k: torch.tensor(cfg[k]) for k in ("latents_mean", "latents_std") if k in cfg}, layers)
    for k in ("decoder_num_attention_heads", "decoder_attention_head_dim", "decoder_num_register_tokens",
              "decoder_ffn_mult", "decoder_rope_theta", "decoder_rope_dim_ratio", "decoder_norm_eps"):
        if k in cfg:
            config[k] = cfg[k]
    return build_vae(state, config, dtype).to(device).eval().requires_grad_(False)


def get_vae(cfg: TrainConfig, model_dir: Path | None):
    if moss_loading._use_file(cfg, "vae_file"):
        moss_loading._check_files(cfg, "vae_file")
        return load_vae_file(cfg.vae_file, cfg.device)
    return load_vae(_need_dir(model_dir, "VAE"), cfg.device)


encode_images = HL.encode_images


def _decode(vae, latents: torch.Tensor) -> torch.Tensor:
    z = latents.to(vae.device, torch.float32)
    mean, std = HL._stats(vae, z)
    z = (z * std + mean).unsqueeze(2).to(next(vae.post_quant_conv.parameters()).dtype)
    x = vae._decode_clip(z)[:, :, 0]  # one latent frame -> one image (spatially tiled, as released)
    pm, ps = HL._pixel_norm(x)
    return (x.float() * ps + pm) * 2.0 - 1.0


@torch.no_grad()
def decode_latents(vae, latents: torch.Tensor) -> torch.Tensor:
    """Normalized latents [B, 24, H/16, W/16] -> images [B, 3, H, W] in [-1, 1]."""
    return _decode(vae, latents).clamp(-1, 1)


def decode_latents_grad(vae, latents: torch.Tensor) -> torch.Tensor:
    return _decode(vae, latents)


def grad_vae(vae):
    return vae
