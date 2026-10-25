"""Model architectures the trainer can train: MOSS-Image, Bria FIBO 2.0 and Krea 2.

Everything the training loop, sampler, caches and LoRA export need to know about a model family lives in an `Arch`.
All families share the training loss (flow matching, velocity target), the 16-pixel latent cell (so region maps and
the likeness code are the same) and the model call convention: `model(latents, sigma * 1000, embeds, mask)` with
latents in a normalized 2x2-patchified layout [B, C, H/16, W/16]. MOSS-Image and FIBO 2.0 share the Flux2 VAE
(C = 128); Krea 2 uses the Qwen-Image VAE (C = 64) and caches its latents under their own names (`latent_tag`).

    get_arch(cfg)              the Arch for a config (model_arch, or guessed from the model path)
    detect_arch(cfg, model_dir) the Arch from a resolved model directory's transformer/config.json
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Iterable

import torch
from torch import nn

from .config import ARCH_DEFAULT_TARGETS, ARCH_SAMPLE_DEFAULTS, TrainConfig

log = logging.getLogger("rulztrainer")

ARCH_NAMES = ("moss_image", "fibo2", "krea2")


def _match_suffix(name: str, suffixes: Iterable[str]) -> bool:
    return any(name == s or name.endswith("." + s) for s in suffixes)


class Arch:
    name = ""
    label = ""
    modelspec = ""  # modelspec.architecture metadata in saved LoRAs
    lora_format = "comfy"  # comfy (diffusion_model.<module>.lora_down/up) | peft (transformer.<module>.lora_A/B)
    n_blocks = 0  # blocks of the reference checkpoint (the trainer uses len(model.layers))
    target_groups: dict[str, str] = {}  # lora_targets name -> description (for the UI)
    default_targets: tuple[str, ...] = ()
    supports_single_files = False
    empty_prompt = ""  # what caption dropout trains on
    default_negative = ""  # negative prompt for samples/probes when none is given
    default_sample_steps = 20
    default_sample_cfg = 3.0
    block_token_px = 16  # pixels per latent cell (an 8x VAE, 2x2 patches)
    latent_tag = ""  # latent cache file suffix: families with another VAE must not reuse each other's latents

    # ---- loading / encoding (see rulztrainer.moss.loading / rulztrainer.fibo.loading)
    def resolve_for_config(self, cfg: TrainConfig) -> Path | None: ...
    def get_transformer(self, cfg: TrainConfig, model_dir: Path | None, device: str | None = None): ...
    def get_vae(self, cfg: TrainConfig, model_dir: Path | None): ...
    def get_text_encoder(self, cfg: TrainConfig, model_dir: Path | None): ...
    def load_any_transformer(self, path: str, device, quant: str = "auto", models_dir: str = "models"): ...
    def encode_prompt(self, text_encoder, tokenizer, prompt: str, max_length: int) -> torch.Tensor: ...
    def pad_embeds(self, rows: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]: ...

    def encode_images(self, vae, images: torch.Tensor) -> torch.Tensor:
        from .moss.loading import encode_images

        return encode_images(vae, images)

    def decode_latents(self, vae, latents: torch.Tensor) -> torch.Tensor:
        from .moss.loading import decode_latents

        return decode_latents(vae, latents)

    def decode_latents_grad(self, vae, latents: torch.Tensor) -> torch.Tensor:
        from .moss.loading import decode_latents_grad

        return decode_latents_grad(vae, latents)

    def grad_vae(self, vae):
        """The VAE copy decode_latents_grad runs on (bf16, gradient checkpointing)."""
        from .moss.loading import grad_vae

        return grad_vae(vae)

    def sample(self, model, pos_e, pos_m, neg_e, neg_m, **kw) -> torch.Tensor: ...

    # ---- prompts
    def clean_prompt(self, prompt: str) -> str:
        return prompt

    def add_trigger(self, caption: str, trigger: str) -> str:
        if not trigger:
            return caption
        return f"{trigger}, {caption}" if caption else trigger

    def caption_file(self, image: Path, cfg: TrainConfig) -> Path:
        return image.with_suffix(cfg.caption_ext)

    # ---- LoRA
    def lora_module_names(self, model: nn.Module, targets: Iterable[str], blocks: Iterable[int] | None,
                          include_text_embedder: bool = False) -> list[str]: ...

    def compile_blocks(self, model: nn.Module) -> list[nn.Module]:
        return list(model.layers)

    text_prefixes: tuple[str, ...] = ()  # LoRA modules on the text side (automagic_groups: by_type)

    def param_group_of(self, name: str) -> str:
        """The layer type of a LoRA module: text / attn / mlp / other (automagic_groups: by_type gives each type its
        own adaptive learning rate)."""
        if name.startswith(self.text_prefixes):
            return "text"
        parts = name.split(".")
        if {"attn", "attention", "cross_attention"} & set(parts) or parts[-1] in ("q", "k", "v", "o"):
            return "attn"
        if {"ff", "feed_forward", "mlp"} & set(parts) or parts[-1].startswith("ffn"):
            return "mlp"
        return "other"

    # ---- timesteps
    def reference_sigmas(self, n: int, gen: torch.Generator, tokens: int) -> torch.Tensor:
        """The family's own training noise-level distribution (timestep_sampling: reference)."""
        raise NotImplementedError(f"{self.name} has no reference timestep recipe")


class MossArch(Arch):
    name = "moss_image"
    label = "MOSS-Image"
    modelspec = "moss-image"
    lora_format = "comfy"
    n_blocks = 42
    target_groups = {"q": "attention query", "k": "attention key", "v": "attention value", "o": "attention out",
                     "ffn_up_gate": "feed-forward up/gate", "ffn_down": "feed-forward down"}
    default_targets = ARCH_DEFAULT_TARGETS["moss_image"]
    text_prefixes = ("text_embedder.",)
    supports_single_files = True
    default_sample_steps, default_sample_cfg = ARCH_SAMPLE_DEFAULTS["moss_image"]

    def __init__(self):
        from .moss import loading

        self._loading = loading

    def resolve_for_config(self, cfg):
        return self._loading.resolve_for_config(cfg)

    def get_transformer(self, cfg, model_dir, device=None):
        return self._loading.get_transformer(cfg, model_dir, device)

    def get_vae(self, cfg, model_dir):
        return self._loading.get_vae(cfg, model_dir)

    def get_text_encoder(self, cfg, model_dir):
        return self._loading.get_text_encoder(cfg, model_dir)

    def load_any_transformer(self, path, device, quant="auto", models_dir="models"):
        return self._loading.load_any_transformer(path, device, quant, models_dir)

    def encode_prompt(self, text_encoder, tokenizer, prompt, max_length):
        return self._loading.encode_prompt(text_encoder, tokenizer, prompt, max_length)

    def pad_embeds(self, rows):
        return self._loading.pad_embeds(rows)

    def sample(self, model, pos_e, pos_m, neg_e, neg_m, **kw):
        from .sampler import sample

        return sample(model, pos_e, pos_m, neg_e, neg_m, **kw)

    def lora_module_names(self, model, targets, blocks, include_text_embedder=False):
        from .lora import _is_linear_like, _target_matches

        targets = set(targets)
        block_set = None if blocks is None else {int(b) for b in blocks}
        text_targets = {"qkv", "proj", "ffn_gate", "ffn_up", "ffn_down"}
        names = []
        for name, module in model.named_modules():
            if not _is_linear_like(module):
                continue
            if name.startswith("layers."):
                if not _target_matches(name, targets):
                    continue
                if block_set is not None and int(name.split(".")[1]) not in block_set:
                    continue
            elif name.startswith("text_embedder."):
                if not include_text_embedder or not _target_matches(name, text_targets):
                    continue
            else:
                continue
            names.append(name)
        return names

    def compile_blocks(self, model):
        return [*model.layers, *model.text_embedder.blocks]

    def reference_sigmas(self, n, gen, tokens):
        # MOSS-Image report eq. 4: S = 1 - sigmoid(Z), Z ~ N(-1.09, 0.95^2), truncated to S <= 0.95 and rescaled;
        # with probability 0.01 uniform over the grid. About 11-13% of steps below sigma 0.5.
        z = torch.randn(n * 4, generator=gen) * 0.95 - 1.09
        s = torch.sigmoid(-z)
        s = s[s <= 0.95][:n] / 0.95
        while s.numel() < n:  # pathological draw: top up
            z = torch.randn(n * 4, generator=gen) * 0.95 - 1.09
            s = torch.cat([s, torch.sigmoid(-z)[:n]])[:n]
        u = torch.rand(n, generator=gen)
        return torch.where(u < 0.01, torch.rand(n, generator=gen), s)


class FiboArch(Arch):
    name = "fibo2"
    label = "FIBO 2.0 (Bria)"
    modelspec = "bria-fibo2"
    lora_format = "peft"
    n_blocks = 44
    # groups map onto the module names inside each block; the extra groups cover what Bria's own trainer also adapts
    GROUPS = {
        "q": ("attention.to_q",), "k": ("attention.to_k",), "v": ("attention.to_v",), "o": ("attention.to_out.0",),
        "ffn_up_gate": ("feed_forward.w1", "feed_forward.w3"), "ffn_down": ("feed_forward.w2",),
        "cross_attn": ("cross_attention.to_q", "cross_attention.to_k", "cross_attention.to_v",
                       "cross_attention.to_out.0"),
        "adaln": ("adaLN_modulation.0",),
    }
    target_groups = {"q": "attention query", "k": "attention key", "v": "attention value", "o": "attention out",
                     "ffn_up_gate": "feed-forward w1/w3", "ffn_down": "feed-forward w2",
                     "cross_attn": "text cross-attention (5 injection blocks)", "adaln": "block modulation",
                     "refiner": "noise refiner blocks", "perceiver": "gist-token resampler",
                     "embed": "patch/timestep embedders and final layer"}
    default_targets = ARCH_DEFAULT_TARGETS["fibo2"]
    text_prefixes = ("perceiver.",)
    supports_single_files = False
    default_sample_steps, default_sample_cfg = ARCH_SAMPLE_DEFAULTS["fibo2"]

    def __init__(self):
        from .fibo import loading, prompts

        self._loading, self._prompts = loading, prompts
        self.empty_prompt = prompts.NULL_CAPTION
        self.default_negative = prompts.NULL_CAPTION

    def resolve_for_config(self, cfg):
        return self._loading.resolve_for_config(cfg)

    def get_transformer(self, cfg, model_dir, device=None):
        return self._loading.get_transformer(cfg, model_dir, device)

    def get_vae(self, cfg, model_dir):
        return self._loading.get_vae(cfg, model_dir)

    def get_text_encoder(self, cfg, model_dir):
        return self._loading.get_text_encoder(cfg, model_dir)

    def load_any_transformer(self, path, device, quant="auto", models_dir="models"):
        return self._loading.load_any_transformer(path, device, quant, models_dir)

    def encode_prompt(self, text_encoder, tokenizer, prompt, max_length):
        return self._loading.encode_prompt(text_encoder, tokenizer, prompt, max_length)

    def pad_embeds(self, rows):
        return self._loading.pad_embeds(rows)

    def sample(self, model, pos_e, pos_m, neg_e, neg_m, **kw):
        from .fibo.sampler import sample

        return sample(model, pos_e, pos_m, neg_e, neg_m, **kw)

    def clean_prompt(self, prompt):
        return self._prompts.clean_prompt(prompt)

    def add_trigger(self, caption, trigger):
        return self._prompts.add_trigger(caption, trigger)

    def caption_file(self, image, cfg):
        j = image.with_suffix(".json")  # a structured (VGL) caption wins over the plain-sentence file
        return j if j.is_file() else image.with_suffix(cfg.caption_ext)

    def lora_module_names(self, model, targets, blocks, include_text_embedder=False):
        from .lora import _is_linear_like

        targets = set(targets)
        block_set = None if blocks is None else {int(b) for b in blocks}
        in_block = [s for t in targets for s in self.GROUPS.get(t, ())]
        names = []
        for name, module in model.named_modules():
            if not _is_linear_like(module):
                continue
            if name.startswith("layers."):
                if block_set is not None and int(name.split(".")[1]) not in block_set:
                    continue
                if _match_suffix(name, in_block):
                    names.append(name)
            elif name.startswith("noise_refiner."):
                if "refiner" in targets:
                    names.append(name)
            elif name.startswith("perceiver."):
                if "perceiver" in targets:
                    names.append(name)
            elif "embed" in targets and name.split(".")[0] in ("x_embedder", "t_embedder", "final_layer"):
                names.append(name)
        return names

    def compile_blocks(self, model):
        return [*model.noise_refiner, *model.layers]

    def reference_sigmas(self, n, gen, tokens):
        # Bria's trainer: sigma ~ sigmoid(N(shift, 1)) with probability 0.9, else uniform;
        # shift = 0.5 + 0.65 * (tokens - 256) / 3840 (the image's token count, as the sampler's time shift)
        shift = 0.5 + (1.15 - 0.5) * (tokens - 256) / (4096 - 256)
        s = torch.sigmoid(torch.randn(n, generator=gen) + shift)
        u = torch.rand(n, generator=gen)
        return torch.where(u < 0.1, torch.rand(n, generator=gen), s)


class Krea2Arch(Arch):
    name = "krea2"
    label = "Krea 2"
    modelspec = "krea-2"
    lora_format = "peft"  # the diffusers / PEFT layout Krea's own Krea-2 LoRAs use (transformer.<module>.lora_A/B)
    n_blocks = 28
    GROUPS = {
        "q": ("attn.to_q",), "k": ("attn.to_k",), "v": ("attn.to_v",), "o": ("attn.to_out.0",),
        "ffn_up_gate": ("ff.gate", "ff.up"), "ffn_down": ("ff.down",), "attn_gate": ("attn.to_gate",),
    }
    EMBED = ("img_in", "txt_in", "time_embed", "time_mod_proj", "final_layer")
    target_groups = {"q": "attention query", "k": "attention key", "v": "attention value", "o": "attention out",
                     "ffn_up_gate": "feed-forward gate/up", "ffn_down": "feed-forward down",
                     "attn_gate": "attention output gate", "text_fusion": "text fusion blocks",
                     "embed": "image/text/timestep projections and final layer"}
    default_targets = ARCH_DEFAULT_TARGETS["krea2"]  # + attention gate and text fusion (see config.py)
    text_prefixes = ("text_fusion.",)
    supports_single_files = True  # ComfyUI's Krea 2 diffusion model, Qwen3-VL-4B text encoder and Qwen-Image VAE files
    default_sample_steps, default_sample_cfg = ARCH_SAMPLE_DEFAULTS["krea2"]
    latent_tag = "_qwenvae"

    @staticmethod
    def lora_module_name(name: str) -> str:
        """ComfyUI's Krea 2 module names (blocks.N.attn.wq, tproj.1, txtfusion...) -> ours, for LoRA files trained
        on ComfyUI's layout (turbo / distill LoRAs and others made for ComfyUI)."""
        from .krea2.single_file import comfy_key_to_diffusers

        return comfy_key_to_diffusers(name + ".weight")[: -len(".weight")]

    def __init__(self):
        from .krea2 import loading

        self._loading = loading

    def resolve_for_config(self, cfg):
        return self._loading.resolve_for_config(cfg)

    def get_transformer(self, cfg, model_dir, device=None):
        return self._loading.get_transformer(cfg, model_dir, device)

    def get_vae(self, cfg, model_dir):
        return self._loading.get_vae(cfg, model_dir)

    def get_text_encoder(self, cfg, model_dir):
        return self._loading.get_text_encoder(cfg, model_dir)

    def load_any_transformer(self, path, device, quant="auto", models_dir="models"):
        return self._loading.load_any_transformer(path, device, quant, models_dir)

    def encode_prompt(self, text_encoder, tokenizer, prompt, max_length):
        return self._loading.encode_prompt(text_encoder, tokenizer, prompt, max_length)

    def pad_embeds(self, rows):
        return self._loading.pad_embeds(rows)

    def encode_images(self, vae, images):
        return self._loading.encode_images(vae, images)

    def decode_latents(self, vae, latents):
        return self._loading.decode_latents(vae, latents)

    def decode_latents_grad(self, vae, latents):
        return self._loading.decode_latents_grad(vae, latents)

    def grad_vae(self, vae):
        return self._loading.grad_vae(vae)

    def sample(self, model, pos_e, pos_m, neg_e, neg_m, **kw):
        from .krea2.sampler import sample

        return sample(model, pos_e, pos_m, neg_e, neg_m, **kw)

    def lora_module_names(self, model, targets, blocks, include_text_embedder=False):
        from .lora import _is_linear_like

        targets = set(targets)
        block_set = None if blocks is None else {int(b) for b in blocks}
        in_block = [s for t in targets for s in self.GROUPS.get(t, ())]
        names = []
        for name, module in model.named_modules():
            if not _is_linear_like(module):
                continue
            if name.startswith("transformer_blocks."):
                if block_set is not None and int(name.split(".")[1]) not in block_set:
                    continue
                if _match_suffix(name, in_block):
                    names.append(name)
            elif name.startswith("text_fusion."):
                if "text_fusion" in targets:
                    names.append(name)
            elif "embed" in targets and name.split(".")[0] in self.EMBED:
                names.append(name)
        return names

    def compile_blocks(self, model):
        return list(model.transformer_blocks)

    def reference_sigmas(self, n, gen, tokens):
        # Krea has not published its training noise distribution. This is the usual recipe for a model sampled with
        # a resolution-dependent shift: logit-normal sigmas shifted by the same mu as the sampler (0.5 at 256 tokens
        # to 1.15 at 6400).
        from .krea2.sampler import calculate_shift, time_shift

        s = torch.sigmoid(torch.randn(n, generator=gen))
        return time_shift(calculate_shift(tokens), s).clamp(0.0, 1.0)


_ARCHS = {"moss_image": MossArch, "fibo2": FiboArch, "krea2": Krea2Arch}
_cache: dict[str, Arch] = {}


def arch_by_name(name: str) -> Arch:
    if name not in _ARCHS:
        raise ValueError(f"Unknown model_arch {name!r}; choose one of {list(_ARCHS)}")
    if name not in _cache:
        _cache[name] = _ARCHS[name]()
    return _cache[name]


_FILE_ARCH_CACHE: dict[tuple, str | None] = {}


def arch_from_transformer_file(path: str) -> str | None:
    """The family a single diffusion-model file holds, read from its safetensors header (None: unknown, or the file
    cannot be read). Cached per path, size and modification time."""
    p = Path(path).expanduser()
    try:
        st = p.stat()
    except OSError:
        return None
    key = (str(p), st.st_size, st.st_mtime_ns)
    if key not in _FILE_ARCH_CACHE:
        found = None
        try:
            from safetensors import safe_open

            with safe_open(str(p), "pt") as f:
                names = {k.split("diffusion_model.", 1)[-1] for k in f.keys()}
            if "txtfusion.projector.weight" in names or "text_fusion.projector.weight" in names:
                found = "krea2"
            elif "shared_block_modulation.weight" in names:
                found = "moss_image"
        except Exception:  # noqa: BLE001
            found = None
        _FILE_ARCH_CACHE[key] = found
    return _FILE_ARCH_CACHE[key]


def guess_arch_name(cfg: TrainConfig) -> str:
    """The architecture a config refers to: model_arch when set, else the transformer file's own family (single
    files), else guessed from the model path or file names."""
    if cfg.model_arch and cfg.model_arch != "auto":
        return cfg.model_arch
    if cfg.model_source == "files" and cfg.transformer_file:
        found = arch_from_transformer_file(cfg.transformer_file)
        if found:
            return found
    return cfg.arch_name()


def get_arch(cfg: TrainConfig) -> Arch:
    return arch_by_name(guess_arch_name(cfg))


def detect_arch(cfg: TrainConfig, model_dir: Path | None) -> Arch:
    """The Arch for a resolved model directory (its transformer/config.json), falling back to the config's."""
    guessed = guess_arch_name(cfg)
    if model_dir is not None:
        try:
            cls = json.loads((Path(model_dir) / "transformer" / "config.json").read_text(encoding="utf-8")).get("_class_name")
        except (OSError, ValueError):
            cls = None
        found = {"BriaFibo2Transformer2DModel": "fibo2", "Krea2Transformer2DModel": "krea2",
                 "MOSImageTransformer": "moss_image", "MOSSImageTransformer": "moss_image"}.get(cls)
        if found and found != guessed:
            if cfg.model_arch not in ("", "auto"):
                raise ValueError(f"model_arch is {cfg.model_arch!r} but {model_dir} holds a {found} transformer")
            log.info("Model architecture detected: %s", found)
            guessed = found
    if cfg.model_source == "files" and cfg.transformer_file:
        found = arch_from_transformer_file(cfg.transformer_file)
        if found and found != guessed:  # guess_arch_name already follows the file unless model_arch is explicit
            raise ValueError(f"model_arch is {cfg.model_arch!r} but {cfg.transformer_file} holds a {found} model")
    return arch_by_name(guessed)
