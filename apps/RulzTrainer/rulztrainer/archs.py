"""Model architectures the trainer can train: MOSS-Image, Bria FIBO 2.0, Krea 2, Qwen-Image 2.1 and the others below.

Everything the training loop, sampler, caches and LoRA export need to know about a model family lives in an `Arch`.
All families share the training loss (flow matching, velocity target), the 16-pixel latent cell (so region maps and
the likeness code are the same) and the model call convention: `model(latents, sigma * 1000, embeds, mask)` with
latents in a normalized 2x2-patchified layout [B, C, H/16, W/16]. MOSS-Image and FIBO 2.0 share the Flux2 VAE
(C = 128); Krea 2 uses the Qwen-Image VAE (C = 64) and caches its latents under their own names (`latent_tag`).
Qwen-Image 2.1 has its own RGBA VAE, 16x with 64 channels, read unpatched: the same [B, 64, H/16, W/16] layout, on a
32 px grid (`size_multiple`: its image tokens come in 2x2 groups).
FLUX 3 Klein and FLUX.2 Klein 9B use the Flux2 VAE too and share MOSS-Image's / FIBO 2.0's latents (C = 128, 16 px grid).

    get_arch(cfg)              the Arch for a config (model_arch, or guessed from the model path)
    detect_arch(cfg, model_dir) the Arch from a resolved model directory's transformer/config.json
"""
from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from typing import Iterable

import torch
from torch import nn

from .config import ARCH_DEFAULT_TARGETS, ARCH_FULL_SAMPLE, TrainConfig

log = logging.getLogger("rulztrainer")

ARCH_NAMES = ("moss_image", "fibo2", "krea2", "qwen_image21", "flux3_klein")
ARCH_NAMES += ("flux2_klein",)
ARCH_NAMES += ("ideogram4",)
ARCH_NAMES += ("minimax_h3",)
ARCH_NAMES += ("p_h3_image",)
ARCH_NAMES += ("minimax_h3_image",)
ARCH_NAMES += ("pan2",)


def _match_suffix(name: str, suffixes: Iterable[str]) -> bool:
    return any(name == s or name.endswith("." + s) for s in suffixes)


def contains_word(text: str, word: str) -> bool:
    """Does `text` hold `word` as a whole word ("ana" is not in "banana")? No word character may touch it on either
    side: lookarounds rather than word boundaries, which would miss a trigger that starts or ends with punctuation
    ("<sks>")."""
    return bool(word) and re.search(rf"(?<!\w){re.escape(word)}(?!\w)", text) is not None


class Arch:
    name = ""
    label = ""
    modelspec = ""  # modelspec.architecture metadata in saved LoRAs
    lora_format = "comfy"  # comfy (diffusion_model.<module>.lora_down/up) | peft (transformer.<module>.lora_A/B)
    n_blocks = 0  # blocks of the reference checkpoint (the trainer uses len(model.layers))
    target_groups: dict[str, str] = {}  # lora_targets name -> description (for the UI)
    default_targets: tuple[str, ...] = ()
    supports_single_files = False
    supports_video = False  # trains on video clips (train_frames) and renders clips (sample_frames / probe_frames)
    video_fps = 24
    empty_prompt = ""  # what caption dropout trains on
    default_negative = ""  # negative prompt for samples/probes when none is given
    default_sample_steps = 20
    default_sample_cfg = 3.0
    block_token_px = 16  # pixels per latent cell (an 8x VAE, 2x2 patches)
    latent_tag = ""  # latent cache file suffix: families with another VAE must not reuse each other's latents
    size_multiple = 16  # training buckets are multiples of this many pixels

    # ---- loading / encoding (see rulztrainer.moss.loading / rulztrainer.fibo.loading)
    def resolve_for_config(self, cfg: TrainConfig) -> Path | None: ...
    def get_transformer(self, cfg: TrainConfig, model_dir: Path | None, device: str | None = None): ...
    def get_vae(self, cfg: TrainConfig, model_dir: Path | None): ...
    def get_text_encoder(self, cfg: TrainConfig, model_dir: Path | None): ...
    def load_any_transformer(self, path: str, device, quant: str = "auto", models_dir: str = "models"): ...
    def encode_prompt(self, text_encoder, tokenizer, prompt: str, max_length: int) -> torch.Tensor: ...

    def encode_prompts(self, text_encoder, tokenizer, prompts: list[str], max_length: int) -> list[torch.Tensor]:
        """encode_prompt for several prompts in one text-encoder forward (right-padded; each row cut back to its own
        length), when the family's loading module has a batched encoder; one prompt at a time otherwise."""
        batched = getattr(getattr(self, "_loading", None), "encode_prompts", None)
        if batched is None:
            return [self.encode_prompt(text_encoder, tokenizer, p, max_length) for p in prompts]
        return batched(text_encoder, tokenizer, prompts, max_length)

    def pad_embeds(self, rows: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]: ...

    def get_tokenizer(self, cfg: TrainConfig, model_dir: Path | None):
        """The text encoder's tokenizer alone (a few MB), for finding the trigger word's tokens."""
        return self._loading.get_tokenizer(cfg, model_dir)

    def prompt_tokens(self, tokenizer, prompt: str, max_length: int | None = None) -> tuple[str, list, int, int]:
        """(text, token offsets, leading tokens the cache drops, cached length) of a prompt as encode_prompt forms it
        at max_length (the run's max_token_length; None: the config's default), so a word's tokens can be found in
        the cached features: a long caption is cut where its cache was cut, and its positions stay inside it."""
        return self._loading.prompt_tokens(tokenizer, prompt,
                                           TrainConfig.max_token_length if max_length is None else int(max_length))

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

    # ---- video (supports_video: clips to train on, clip samples and probes)
    def snap_frames(self, frames: int) -> int:
        """The clip length the VAE takes for a request (stills: 1)."""
        return 1

    def latent_frames(self, frames: int) -> int:
        """Latent frames of a clip of `frames` pixel frames (as snap_frames gives them)."""
        return 1

    def encode_video(self, vae, frames: torch.Tensor) -> torch.Tensor:
        """frames [B, 3, F, H, W] in [-1, 1] -> normalized latents [B, C, T, H/16, W/16]."""
        raise NotImplementedError(f"{self.label} does not train on video clips")

    def decode_video(self, vae, latents: torch.Tensor) -> torch.Tensor:
        """A clip's latents [1, C, T, h, w] -> its frames [F, 3, H, W] in [-1, 1]."""
        raise NotImplementedError(f"{self.label} does not render clips")

    def render_prompt(self, prompt: str, frames: int = 1) -> str:
        """A prompt as a render (or a training clip) of `frames` frames reads it."""
        return prompt

    def render_negative(self, frames: int = 1) -> str:
        return self.default_negative

    def distilled_sample_model(self, path: str) -> bool:
        """Does sample_model_path name a distilled (few-step, no guidance) model? By name: a turbo model."""
        return "turbo" in (path or "").lower()

    def sample_sigmas(self, model, steps: int, height: int, width: int, shift: float) -> torch.Tensor:
        """The sampler's noise schedule (ending in 0) for an image of height x width pixels."""
        from .sampler import moss_sigmas

        return moss_sigmas(steps, shift)

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
        if {"t_embedder", "time_embed", "time_text_embed", "time_embedding"} & set(parts):
            return "other"  # the timestep path (FIBO's t_embedder.mlp is not a feed-forward layer)
        if {"attn", "attention", "cross_attention"} & set(parts) or parts[-1] in ("q", "k", "v", "o"):
            return "attn"
        if {"ff", "feed_forward", "mlp", "img_mlp"} & set(parts) or parts[-1].startswith("ffn"):
            return "mlp"
        return "other"

    # ---- timesteps
    # True: reference_sigmas returns the exact sigmas to train on (a step-distilled model's own sampling sigmas), so
    # the trainer uses them as drawn: no logit shift, timestep_shift, sigma range or 1000-step rounding
    reference_sigmas_exact = False

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
    default_sample_steps, default_sample_cfg = ARCH_FULL_SAMPLE["moss_image"]  # full quality, without a turbo LoRA

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
        while s.numel() < n:  # pathological draw: top up, truncated and rescaled the same way
            z = torch.randn(n * 4, generator=gen) * 0.95 - 1.09
            more = torch.sigmoid(-z)
            s = torch.cat([s, more[more <= 0.95] / 0.95])[:n]
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
    supports_single_files = True  # ComfyUI's FIBO 2.0 diffusion model, Qwen3-VL-4B text encoder and Flux2 VAE files
    default_sample_steps, default_sample_cfg = ARCH_FULL_SAMPLE["fibo2"]

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

    def sample_sigmas(self, model, steps, height, width, shift):
        from .fibo.sampler import fibo_sigmas

        return fibo_sigmas(steps, (height // 16) * (width // 16))

    def clean_prompt(self, prompt):
        return self._prompts.clean_prompt(prompt)

    def add_trigger(self, caption, trigger):
        return self._prompts.add_trigger(caption, trigger)

    @staticmethod
    def extra_lora_converter(state: dict) -> dict:
        """A sampling-only LoRA in ComfyUI's fused FIBO layout (Comfy-Org's Turbo LoRA) in the model's own layout."""
        from .fibo.single_file import lora_to_diffusers_layout

        return lora_to_diffusers_layout(state)

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
    text_prefixes = ("text_fusion.", "txt_in.")  # the text projection is text-side too
    supports_single_files = True  # ComfyUI's Krea 2 diffusion model, Qwen3-VL-4B text encoder and Qwen-Image VAE files
    default_sample_steps, default_sample_cfg = ARCH_FULL_SAMPLE["krea2"]
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

    def sample_sigmas(self, model, steps, height, width, shift):
        from .krea2.sampler import krea2_sigmas, uses_distilled_schedule

        return krea2_sigmas(steps, (height // 16) * (width // 16), uses_distilled_schedule(model))

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


class QwenImage21Arch(Arch):
    name = "qwen_image21"
    label = "Qwen-Image 2.1"
    modelspec = "qwen-image-2.1"
    # diffusers / PEFT names (transformer.<module>.lora_A/B): ComfyUI maps them, gate_layer / proj onto the halves of
    # its fused gate_up (only for bare or transformer.-prefixed keys), and diffusers loads them as they are
    lora_format = "peft"
    n_blocks = 32
    GROUPS = {
        "q": ("attn.to_q",), "k": ("attn.to_k",), "v": ("attn.to_v",), "o": ("attn.to_out.0",),
        "ffn_up_gate": ("img_mlp.gate_layer", "img_mlp.proj"), "ffn_down": ("img_mlp.out",),
    }
    EMBED = ("img_in", "time_text_embed", "modulation", "norm_out", "proj_out")
    target_groups = {"q": "attention query", "k": "attention key", "v": "attention value", "o": "attention out",
                     "ffn_up_gate": "feed-forward gate/up", "ffn_down": "feed-forward down",
                     "text_in": "text projection (the encoder's features into the model)",
                     "embed": "image / timestep projections, the shared modulation and the final layer"}
    default_targets = ARCH_DEFAULT_TARGETS["qwen_image21"]
    text_prefixes = ("txt_in.",)
    supports_single_files = True  # Comfy-Org's Qwen-Image 2.1 diffusion model, Qwen3-VL-8B text encoder and VAE
    default_sample_steps, default_sample_cfg = ARCH_FULL_SAMPLE["qwen_image21"]
    latent_tag = "_qi21vae"
    size_multiple = 32

    def __init__(self):
        from .qwen21 import loading

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
        from .qwen21.sampler import sample

        return sample(model, pos_e, pos_m, neg_e, neg_m, **kw)

    def sample_sigmas(self, model, steps, height, width, shift):
        from .qwen21.sampler import qwen21_sigmas, uses_distilled_schedule

        return qwen21_sigmas(steps, (height // 16) * (width // 16), uses_distilled_schedule(model))

    @staticmethod
    def extra_lora_converter(state: dict) -> dict:
        """A LoRA trained on ComfyUI's fused feed-forward (img_mlp.gate_up) in the model's own layout: its up factor
        split into the gate_layer and proj halves, each with the shared down factor."""
        out = {}
        for k, v in state.items():
            if ".img_mlp.gate_up." not in k:
                out[k] = v
                continue
            for up, down in ((".lora_up.weight", ".lora_down.weight"), (".lora_B.weight", ".lora_A.weight")):
                if k.endswith(".img_mlp.gate_up" + up):
                    stem = k[: -len(".gate_up" + up)]
                    for half, w in zip(("gate_layer", "proj"), v.chunk(2, 0)):
                        out[f"{stem}.{half}{up}"] = w.contiguous()
                        out[f"{stem}.{half}{down}"] = state[f"{stem}.gate_up{down}"]
                        if f"{stem}.gate_up.alpha" in state:
                            out[f"{stem}.{half}.alpha"] = state[f"{stem}.gate_up.alpha"]
        return out

    def lora_module_names(self, model, targets, blocks, include_text_embedder=False):
        from .lora import _is_linear_like

        targets = set(targets)
        if include_text_embedder:
            targets.add("text_in")
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
            elif name.startswith("txt_in."):
                if "text_in" in targets:
                    names.append(name)
            elif "embed" in targets and name.split(".")[0] in self.EMBED:
                names.append(name)
        return names

    def compile_blocks(self, model):
        return list(model.transformer_blocks)

    def reference_sigmas(self, n, gen, tokens):
        # Qwen has not published its training noise distribution. Fizgig trains it with the usual recipe for a model
        # sampled with a resolution-dependent shift: logit-normal sigmas shifted by the sampler's mu (0.5 at 256
        # tokens to 0.9 at 8192), without the sampler's terminal stretch.
        from .qwen21.sampler import calculate_shift, time_shift

        s = torch.sigmoid(torch.randn(n, generator=gen))
        return time_shift(calculate_shift(tokens), s).clamp(0.0, 1.0)


class Flux3KleinArch(Arch):
    name = "flux3_klein"
    label = "FLUX 3 Klein"
    modelspec = "flux-3-image-klein"
    # Black Forest Labs' module names under diffusion_model. (the port keeps them), lora_A / lora_B with alpha folded
    # in: ComfyUI's loader maps them natively (and onto the fused qkv of its int8 files), and diffusers'
    # Flux3LoraLoaderMixin converts them, which refuses .alpha keys
    lora_format = "folded"
    n_blocks = 28
    GROUPS = {"q": ("q_proj",), "k": ("k_proj",), "v": ("v_proj",), "o": ("attn_out",), "ffn_up_gate": ("mlp_in",),
              "ffn_down": ("mlp_out",)}
    EMBED = ("emb_in", "time_in", "final_layer")
    target_groups = {"q": "attention query", "k": "attention key", "v": "attention value", "o": "attention out",
                     "ffn_up_gate": "feed-forward in (SwiGLU gate/up)", "ffn_down": "feed-forward out",
                     "mode_blocks": "the 5 text and 5 image per-mode blocks before the joint ones (the groups above)",
                     "modulation": "the per-mode shift / scale / gate projections of the timestep",
                     "text_in": "text projection (the encoder's 8 stacked layers into the model)",
                     "embed": "image / timestep embedders and the final layer"}
    default_targets = ARCH_DEFAULT_TARGETS["flux3_klein"]  # + the per-mode blocks (see config.py)
    text_prefixes = ("txt_mode_blocks.", "txt_in")
    supports_single_files = True  # black-forest-labs/flux-3-image-klein-comfy: diffusion model, Qwen3-VL-4B, Flux2 VAE
    default_sample_steps, default_sample_cfg = ARCH_FULL_SAMPLE["flux3_klein"]
    # latent_tag "": the Flux2 VAE, encoded and normalized as MOSS-Image and FIBO 2.0 do, so their latents are shared

    def __init__(self):
        from .flux3 import loading

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
        from .flux3.sampler import sample

        return sample(model, pos_e, pos_m, neg_e, neg_m, **kw)

    def sample_sigmas(self, model, steps, height, width, shift):
        from .flux3.sampler import flux3_sigmas

        return flux3_sigmas(steps)

    def distilled_sample_model(self, path: str) -> bool:
        """The 16-step distilled Klein (flux-3-image-klein, flux3-image-klein-<quant>.safetensors): no "base"."""
        name = (path or "").replace("\\", "/").lower().rstrip("/").rsplit("/", 1)[-1]
        return "klein" in name and "base" not in name

    @staticmethod
    def lora_module_name(name: str) -> str:
        """diffusers' Flux3 module names (transformer.single_transformer_blocks.N.attn.to_q ...) -> ours, for LoRA
        files trained in diffusers' layout."""
        from .flux3.single_file import diffusers_module_to_bfl

        return diffusers_module_to_bfl(name)

    def lora_module_names(self, model, targets, blocks, include_text_embedder=False):
        from .lora import _is_linear_like

        targets = set(targets)
        if include_text_embedder:
            targets.add("text_in")
        block_set = None if blocks is None else {int(b) for b in blocks}
        in_block = [s for t in targets for s in self.GROUPS.get(t, ())]
        names = []
        for name, module in model.named_modules():
            if not _is_linear_like(module):
                continue
            if name.startswith("single_blocks."):
                if block_set is not None and int(name.split(".")[1]) not in block_set:
                    continue
                if _match_suffix(name, in_block):
                    names.append(name)
            elif name.startswith(("txt_mode_blocks.", "content_mode_blocks.")):
                if "mode_blocks" in targets and _match_suffix(name, in_block):
                    names.append(name)
            elif name.startswith(("early_stream_modulations.", "single_stream_modulations.")):
                if "modulation" in targets:
                    names.append(name)
            elif name == "txt_in":
                if "text_in" in targets:
                    names.append(name)
            elif "embed" in targets and name.split(".")[0] in self.EMBED:
                names.append(name)
        return names

    def param_group_of(self, name):
        if name.startswith(self.text_prefixes):
            return "text"
        last = name.rsplit(".", 1)[-1]
        if last in ("q_proj", "k_proj", "v_proj", "attn_out"):
            return "attn"
        if last in ("mlp_in", "mlp_out"):
            return "mlp"
        return "other"

    def compile_blocks(self, model):
        return [*model.mode_blocks, *model.single_blocks]

    def reference_sigmas(self, n, gen, tokens):
        # Black Forest Labs has not published Klein's training noise distribution. This is the usual recipe for a
        # model sampled with a constant shift: logit-normal sigmas shifted by the sampler's own 6.93.
        from .flux3.sampler import SHIFT

        s = torch.sigmoid(torch.randn(n, generator=gen))
        return (SHIFT * s / (1 + (SHIFT - 1) * s)).clamp(0.0, 1.0)


_ARCHS = {"moss_image": MossArch, "fibo2": FiboArch, "krea2": Krea2Arch, "qwen_image21": QwenImage21Arch,
          "flux3_klein": Flux3KleinArch}


class Flux2KleinArch(Arch):
    name = "flux2_klein"
    label = "FLUX.2 Klein 9B"
    modelspec = "Flux.2-klein-9b"
    # BFL's module names under diffusion_model. (the port keeps them), lora_A / lora_B with alpha folded in: what
    # ai-toolkit saves for Klein, what ComfyUI's generic diffusion_model. keys read, what diffusers converts
    lora_format = "folded"
    n_blocks = 32  # 8 double + 24 single, one list in `model.layers` (double_blocks.N = N, single_blocks.N = 8 + N)
    # q, k and v share one fused Linear, and a single block fuses its attention and MLP projections into linear1 and
    # linear2 (as in the checkpoint, ai-toolkit and Fizgig): any group that names a part trains the fused layer
    GROUPS = {"q": ("img_attn.qkv", "txt_attn.qkv", "linear1"), "k": ("img_attn.qkv", "txt_attn.qkv", "linear1"),
              "v": ("img_attn.qkv", "txt_attn.qkv", "linear1"), "o": ("img_attn.proj", "txt_attn.proj", "linear2"),
              "ffn_up_gate": ("img_mlp.0", "txt_mlp.0", "linear1"), "ffn_down": ("img_mlp.2", "txt_mlp.2", "linear2")}
    EMBED = ("img_in", "time_in", "final_layer")
    MODULATION = ("double_stream_modulation_img", "double_stream_modulation_txt", "single_stream_modulation")
    target_groups = {"q": "attention query (double blocks: the fused qkv; single blocks: the fused linear1)",
                     "k": "attention key (the same fused layers as q)", "v": "attention value (the same fused layers)",
                     "o": "attention out (double blocks: proj; single blocks: the fused linear2)",
                     "ffn_up_gate": "feed-forward in (double blocks: *_mlp.0; single blocks: the fused linear1)",
                     "ffn_down": "feed-forward out (double blocks: *_mlp.2; single blocks: the fused linear2)",
                     "modulation": "the shift / scale / gate projections of the timestep",
                     "text_in": "text projection (the encoder's 3 stacked layers into the model)",
                     "embed": "image / timestep embedders and the final layer"}
    default_targets = ARCH_DEFAULT_TARGETS["flux2_klein"]
    text_prefixes = ("txt_in",)
    supports_single_files = True  # BFL's fp8 / bf16 files; Comfy-Org's Qwen3-8B and Flux2 VAE
    default_sample_steps, default_sample_cfg = ARCH_FULL_SAMPLE["flux2_klein"]
    # latent_tag "": the Flux2 VAE, encoded and normalized as MOSS-Image and FIBO 2.0 do, so their latents are shared

    def __init__(self):
        from .flux2 import loading

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
        from .flux2.sampler import sample

        return sample(model, pos_e, pos_m, neg_e, neg_m, **kw)

    def sample_sigmas(self, model, steps, height, width, shift):
        from .flux2.sampler import model_sigmas

        return model_sigmas(model, steps, max(1, height // 16) * max(1, width // 16))

    def distilled_sample_model(self, path: str) -> bool:
        """The 4-step distilled Klein (FLUX.2-klein-9B, flux-2-klein-9b*.safetensors): no "base"."""
        from .flux2.single_file import is_distilled_name

        return is_distilled_name(path)

    @staticmethod
    def lora_module_name(name: str) -> str:
        """kohya's flattened names (double_blocks_0_img_attn_qkv, the lora_unet_ prefix already stripped) -> ours."""
        import re

        for pat, rep in ((r"^double_blocks_(\d+)_(img|txt)_attn_(qkv|proj)$", r"double_blocks.\1.\2_attn.\3"),
                         (r"^double_blocks_(\d+)_(img|txt)_mlp_(\d+)$", r"double_blocks.\1.\2_mlp.\3"),
                         (r"^single_blocks_(\d+)_(linear[12])$", r"single_blocks.\1.\2")):
            if re.match(pat, name):
                return re.sub(pat, rep, name)
        return name

    def lora_module_names(self, model, targets, blocks, include_text_embedder=False):
        from .lora import _is_linear_like

        targets = set(targets)
        if include_text_embedder:
            targets.add("text_in")
        block_set = None if blocks is None else {int(b) for b in blocks}
        in_block = sorted({s for t in targets for s in self.GROUPS.get(t, ())})
        n_double = len(model.double_blocks)
        names = []
        for name, module in model.named_modules():
            if not _is_linear_like(module):
                continue
            head = name.split(".")[0]
            if head in ("double_blocks", "single_blocks"):
                block = int(name.split(".")[1]) + (n_double if head == "single_blocks" else 0)
                if (block_set is None or block in block_set) and _match_suffix(name, in_block):
                    names.append(name)
            elif head in self.MODULATION:
                if "modulation" in targets:
                    names.append(name)
            elif name == "txt_in":
                if "text_in" in targets:
                    names.append(name)
            elif "embed" in targets and head in self.EMBED:
                names.append(name)
        return names

    def param_group_of(self, name):
        parts = name.split(".")
        if name.startswith(self.text_prefixes) or {"txt_attn", "txt_mlp"} & set(parts):
            return "text"
        if "img_mlp" in parts or parts[-1] == "linear2":
            return "mlp"  # linear2 ends both streams of a single block; most of its width is the MLP's
        if "img_attn" in parts or parts[-1] == "linear1":
            return "attn"
        return "other"

    def reference_sigmas(self, n, gen, tokens):
        # flux2_shift, as Fizgig trains Klein: logit-normal sigmas, shifted by exp(mu), mu linear in the latent token
        # count from 0.5 at 256 tokens to 1.15 at 4096
        slope = (1.15 - 0.5) / (4096 - 256)
        shift = math.exp(slope * tokens + 0.5 - slope * 256)
        s = torch.sigmoid(torch.randn(n, generator=gen))
        return (shift * s / (1 + (shift - 1) * s)).clamp(0.0, 1.0)


_ARCHS["flux2_klein"] = Flux2KleinArch


class Ideogram4Arch(Arch):
    name = "ideogram4"
    label = "Ideogram 4"
    modelspec = "ideogram-4"
    # diffusion_model.<module>.lora_down/up + alpha in the checkpoint's own names (fused attention.qkv): ComfyUI has no
    # Ideogram 4 LoRA key mapping, so it reads only its own module names (the generic diffusion_model. keys), and
    # diffusers' Ideogram 4 loader converts this layout itself (qkv split into to_q / to_k / to_v); ai-toolkit saves
    # the same. A PEFT file with diffusers' names would not load in ComfyUI.
    lora_format = "comfy"
    n_blocks = 34
    # q, k and v share one fused Linear (as in the checkpoint, ComfyUI and ai-toolkit): any of them trains all three
    GROUPS = {
        "q": ("attention.qkv",), "k": ("attention.qkv",), "v": ("attention.qkv",), "o": ("attention.o",),
        "ffn_up_gate": ("feed_forward.w1", "feed_forward.w3"), "ffn_down": ("feed_forward.w2",),
        "adaln": ("adaln_modulation",),
    }
    EMBED = ("input_proj", "t_embedding", "adaln_proj", "final_layer")
    target_groups = {"q": "attention query (q, k and v are one fused layer: any of the three trains it)",
                     "k": "attention key (the fused qkv layer)", "v": "attention value (the fused qkv layer)",
                     "o": "attention out", "ffn_up_gate": "feed-forward w1/w3", "ffn_down": "feed-forward w2",
                     "adaln": "per-block modulation Linear (ai-toolkit's Ideogram 4 LoRAs train it too)",
                     "text_in": "text projection (the 13 Qwen3-VL layers into the model)",
                     "embed": "image / timestep projections and the final layer"}
    default_targets = ARCH_DEFAULT_TARGETS["ideogram4"]
    text_prefixes = ("llm_cond_proj",)
    supports_single_files = True  # Comfy-Org's Ideogram 4 diffusion models, Qwen3-VL-8B text encoder and Flux2 VAE
    default_sample_steps, default_sample_cfg = ARCH_FULL_SAMPLE["ideogram4"]
    latent_tag = "_ideo4"  # the Flux2 VAE, but Ideogram's own normalization and channel order (ideogram4/loading.py)

    def __init__(self):
        from .ideogram4 import loading, prompts

        self._loading, self._prompts = loading, prompts
        self.empty_prompt = prompts.EMPTY_CAPTION  # caption dropout: the smallest valid caption, empty
        self.default_negative = ""  # no text tokens: the pipeline's image-only unconditional branch

    @staticmethod
    def lora_module_name(name: str) -> str:
        """diffusers' Ideogram 4 names (to_q / to_k / to_v / to_out.0) and ai-toolkit's conditional_transformer.
        prefix -> the port's, for LoRA files made elsewhere."""
        from .ideogram4.single_file import lora_module_name

        return lora_module_name(name)

    @staticmethod
    def extra_lora_converter(state: dict) -> dict:
        """A diffusers-layout LoRA's q / k / v parts placed in their thirds of the fused qkv rows."""
        from .ideogram4.single_file import lora_to_fused_rows

        return lora_to_fused_rows(state)

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
        from .ideogram4.sampler import sample

        return sample(model, pos_e, pos_m, neg_e, neg_m, **kw)

    def sample_sigmas(self, model, steps, height, width, shift):
        from .ideogram4.sampler import sample_sigmas

        return sample_sigmas(model, steps, height, width)

    def clean_prompt(self, prompt):
        return self._prompts.clean_prompt(prompt)

    def add_trigger(self, caption, trigger):
        return self._prompts.add_trigger(caption, trigger)

    def caption_file(self, image, cfg):
        j = image.with_suffix(".json")  # a structured JSON caption wins over the plain caption file
        return j if j.is_file() else image.with_suffix(cfg.caption_ext)

    def lora_module_names(self, model, targets, blocks, include_text_embedder=False):
        from .lora import _is_linear_like

        targets = set(targets)
        if include_text_embedder:
            targets.add("text_in")
        block_set = None if blocks is None else {int(b) for b in blocks}
        in_block = sorted({s for t in targets for s in self.GROUPS.get(t, ())})
        names = []
        for name, module in model.named_modules():
            if not _is_linear_like(module):
                continue
            if name.startswith("layers."):
                if block_set is not None and int(name.split(".")[1]) not in block_set:
                    continue
                if _match_suffix(name, in_block):
                    names.append(name)
            elif name == "llm_cond_proj":
                if "text_in" in targets:
                    names.append(name)
            elif "embed" in targets and name.split(".")[0] in self.EMBED:
                names.append(name)
        return names

    def compile_blocks(self, model):
        return list(model.layers)

    def reference_sigmas(self, n, gen, tokens):
        # Ideogram has not published its training noise distribution. Its documented time distribution is the
        # logit-normal of the pipeline with the resolution-aware mean (docs/pipeline.md; the reference pipeline's
        # defaults mu 0.5, std 1): sigma = expit(0.5 + 0.5 ln(pixels / 512^2) + N(0, 1)), clamped to the schedule's
        # logSNR range. tokens are 16 x 16 px cells.
        from .ideogram4.sampler import LOGSNR_MAX, LOGSNR_MIN, resolution_mean

        mean = resolution_mean(0.5, 16 * int(tokens), 16)
        s = torch.sigmoid(torch.randn(n, generator=gen) + mean)
        t_min, t_max = 1.0 / (1.0 + math.exp(0.5 * LOGSNR_MAX)), 1.0 / (1.0 + math.exp(0.5 * LOGSNR_MIN))
        return s.clamp(1.0 - t_max, 1.0 - t_min)


_ARCHS["ideogram4"] = Ideogram4Arch


class MiniMaxH3Arch(Arch):
    """MiniMax H3, the 33B video + audio model, trained on stills as one-frame videos (rulztrainer.h3): a 16x VAE with
    24 channels read unpatched [B, 24, H/16, W/16] on a 32 px grid (the model patchifies 2x2), its own latents."""
    name = "minimax_h3"
    label = "MiniMax H3"
    modelspec = "minimax-h3"
    # diffusion_model.<module>.lora_A/B without alpha (alpha = rank): what ai-toolkit and diffusion-pipe write for H3
    # and musubi-tuner, ComfyUI's loader and diffusers' MiniMaxH3LoraLoaderMixin read
    lora_format = "comfy_ab"
    n_blocks = 50
    # H3 fuses q / k / v into one attn.qkv_proj: q, k or v selects it whole (one LoRA on the fused layer, the layout
    # every published H3 LoRA and training adapter uses); fc1 holds gate and value
    GROUPS = {"q": ("attn.qkv_proj",), "k": ("attn.qkv_proj",), "v": ("attn.qkv_proj",), "o": ("attn.out_proj",),
              "ffn_up_gate": ("mlp.fc1",), "ffn_down": ("mlp.fc2",)}
    target_groups = {"q": "attention q/k/v (fused in one qkv_proj: any of q, k, v trains it whole)",
                     "k": "attention q/k/v (fused: see q)", "v": "attention q/k/v (fused: see q)",
                     "o": "attention out", "ffn_up_gate": "feed-forward fc1 (gate and value)",
                     "ffn_down": "feed-forward fc2",
                     "token_refiner": "the 2 token-refiner blocks over the text (musubi-tuner leaves them out)"}
    default_targets = ARCH_DEFAULT_TARGETS["minimax_h3"]
    text_prefixes = ("token_refiner.",)
    supports_single_files = True  # Comfy-Org's diffusion models, Qwen3-VL-32B text encoders and VAEs
    default_sample_steps, default_sample_cfg = ARCH_FULL_SAMPLE["minimax_h3"]
    latent_tag = "_h3vae"
    size_multiple = 32
    supports_video = True  # clips of 17n + 5 frames at 24 fps (5n + 2 latent frames), with the silent audio stream

    def __init__(self):
        from .h3 import loading

        self._loading = loading

    def snap_frames(self, frames):
        from .h3.sampler import snap_frames

        return snap_frames(frames)

    def latent_frames(self, frames):
        from .h3.model import video_latent_frames

        return video_latent_frames(frames)

    def encode_video(self, vae, frames):
        return self._loading.encode_video(vae, frames)

    def decode_video(self, vae, latents):
        return self._loading.decode_video(vae, latents)

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
        from .h3.sampler import sample

        return sample(model, pos_e, pos_m, neg_e, neg_m, **kw)

    def sample_sigmas(self, model, steps, height, width, shift):
        from .h3.sampler import h3_sigmas

        return h3_sigmas(steps, getattr(model, "shift_video", 12.0))

    @staticmethod
    def lora_module_name(name: str) -> str:
        """musubi-tuner's flattened names (lora_unet_blocks_0_attn_qkv_proj, the prefix already stripped) -> ours."""
        import re

        for pat, rep in ((r"^blocks_(\d+)_attn_(qkv|out)_proj$", r"blocks.\1.attn.\2_proj"),
                         (r"^blocks_(\d+)_mlp_fc([12])$", r"blocks.\1.mlp.fc\2"),
                         (r"^blocks_(\d+)_adaln_proj_linear$", r"blocks.\1.adaln_proj.linear"),
                         (r"^token_refiner_blocks_(\d+)_attn_(qkv|out)_proj$", r"token_refiner.blocks.\1.attn.\2_proj"),
                         (r"^token_refiner_blocks_(\d+)_mlp_fc([12])$", r"token_refiner.blocks.\1.mlp.fc\2")):
            if re.match(pat, name):
                return re.sub(pat, rep, name)
        return name

    def lora_module_names(self, model, targets, blocks, include_text_embedder=False):
        from .lora import _is_linear_like

        targets = set(targets)
        if include_text_embedder:
            targets.add("token_refiner")
        block_set = None if blocks is None else {int(b) for b in blocks}
        in_block = sorted({s for t in targets for s in self.GROUPS.get(t, ())})
        refiner = ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")
        names = []
        for name, module in model.named_modules():
            if not _is_linear_like(module):
                continue
            if name.startswith("blocks."):
                if block_set is not None and int(name.split(".")[1]) not in block_set:
                    continue
                if _match_suffix(name, in_block):
                    names.append(name)
            elif name.startswith("token_refiner.blocks."):
                if "token_refiner" in targets and _match_suffix(name, refiner):
                    names.append(name)
        return names

    def compile_blocks(self, model):
        return list(model.blocks)

    def reference_sigmas(self, n, gen, tokens):
        # H3 (musubi-tuner's trainer, ai-toolkit's, after the released schedulers): one uniform base time per item;
        # the video sigma is its shift-12 image (the audio sigma, shift 3, follows inside the model)
        from .h3.model import SHIFT_VIDEO, shift_sigma

        return shift_sigma(torch.rand(n, generator=gen), SHIFT_VIDEO).clamp(0.0, 1.0)


_ARCHS["minimax_h3"] = MiniMaxH3Arch


class PH3ImageArch(MiniMaxH3Arch):
    """P-H3-Image (PrunaAI): MiniMax H3 fine-tuned into an 8-step, guidance-free text-to-image model (rulztrainer.ph3).
    H3's transformer, LoRA names and one-frame layout; its own Qwen3-VL-4B text encoder + connector, single-frame VAE
    decoder and 8-sigma grid. Its latents are cached apart from H3's (the VAE file is its own)."""
    name = "p_h3_image"
    label = "P-H3-Image"
    modelspec = "p-h3-image"
    default_targets = ARCH_DEFAULT_TARGETS["p_h3_image"]
    default_sample_steps, default_sample_cfg = ARCH_FULL_SAMPLE["p_h3_image"]
    latent_tag = "_ph3vae"
    supports_video = False  # an image model: H3's clip methods stay unused (the trainer reads supports_video)

    def __init__(self):
        from .ph3 import loading

        self._loading = loading

    def sample(self, model, pos_e, pos_m, neg_e, neg_m, **kw):
        from .ph3.sampler import sample

        return sample(model, pos_e, pos_m, neg_e, neg_m, **kw)

    def sample_sigmas(self, model, steps, height, width, shift):
        from .ph3.sampler import ph3_sigmas

        return ph3_sigmas(steps)

    def distilled_sample_model(self, path: str) -> bool:
        return True  # the released model is the 8-step distilled one: no turbo LoRA on top


_ARCHS["p_h3_image"] = PH3ImageArch


def clean_end_weights(grid: torch.Tensor, logit_shift: float, clean_sigma: float = 0.5) -> torch.Tensor:
    """Draw weights for fixed noise levels moved by a logit shift: the levels below clean_sigma share
    sigmoid(logit(their count / all) - shift) evenly, the rest share the remainder evenly. Shift 0: every level alike;
    a larger shift: fewer clean steps, as it gives a curve. All levels at one end: every level alike."""
    clean = grid < clean_sigma
    k, m = int(clean.sum()), grid.numel()
    if k in (0, m):
        return torch.ones(m)
    p = 1.0 / (1.0 + math.exp(-(math.log(k / (m - k)) - float(logit_shift))))
    return torch.where(clean, torch.full((m,), p / k), torch.full((m,), (1.0 - p) / (m - k)))


class MiniMaxH3ImageArch(MiniMaxH3Arch):
    """MiniMax-H3-Image (MiniMaxAI): H3's transformer without the audio stream, distilled to 6 sampling steps with one
    output head per step and no guidance (rulztrainer.h3image). H3's LoRA names, text encoder and latent layout; its
    own image VAE (H3's encoder, a 2D ViT decoder), cached apart from H3's (the VAE file is its own)."""
    name = "minimax_h3_image"
    label = "MiniMax-H3-Image"
    modelspec = "minimax-h3-image"
    default_targets = ARCH_DEFAULT_TARGETS["minimax_h3_image"]
    default_sample_steps, default_sample_cfg = ARCH_FULL_SAMPLE["minimax_h3_image"]
    latent_tag = "_h3ivae"
    supports_video = False  # no audio stream, no temporal decoder: stills only, unlike MiniMax H3

    def __init__(self):
        from .h3image import loading

        self._loading = loading

    def sample(self, model, pos_e, pos_m, neg_e, neg_m, **kw):
        from .h3image.sampler import sample

        return sample(model, pos_e, pos_m, neg_e, neg_m, **kw)

    def sample_sigmas(self, model, steps, height, width, shift):
        from .h3image.sampler import h3image_sigmas

        return h3image_sigmas(steps, getattr(model, "pdd_sigmas", None))

    def distilled_sample_model(self, path: str) -> bool:
        return True  # the released model is the 6-step distilled one: no turbo LoRA on top

    reference_sigmas_exact = True

    def reference_sigmas(self, n, gen, tokens, logit_shift: float = 0.0):
        # ai-toolkit's rule for the model (its scheduler's fixed_train_sigmas, commit 5b1670a), which its training
        # adapter is made with: only the 6 sigmas the model samples at, picked evenly, each through its own output
        # head. A head was distilled at its one sigma, so no step lands between them. Exact values: rounded to the
        # 1000-step grid, 0.968, 0.888 and 0.533 would fall above their grid points, into the head before.
        # logit_shift (only with clean_share, trainer.sample_sigmas): moves steps between the clean end and the rest
        # the way it moves a curve, with the levels kept. The levels below sigma 0.5 (the last, 0.232) take a share
        # sigmoid(logit(their even share) - shift) of the steps, split evenly, the others the rest evenly: 0 is the
        # even draw, and clean_share's bisection finds the shift for a share (Fizgig's MiniMax-H3-Image presets: 40%)
        from .h3image.loading import PDD_SIGMAS

        grid = torch.tensor(PDD_SIGMAS[:-1], dtype=torch.float32)
        if not logit_shift:
            return grid[torch.randint(0, grid.numel(), (n,), generator=gen)]
        return grid[torch.multinomial(clean_end_weights(grid, logit_shift), n, replacement=True, generator=gen)]


_ARCHS["minimax_h3_image"] = MiniMaxH3ImageArch


class Pan2Arch(Arch):
    """PAN 2 (IFM), the 29B video model, trained on stills as one-frame videos (rulztrainer.pan2), which is how IFM's
    own trainer feeds images: the Wan 2.2 VAE's 16x latents with 48 channels read unpatched [B, 48, H/16, W/16] on a
    32 px grid (the model patchifies 2x2), its own latents; Qwen3.5-9B's hidden state two layers before the last."""
    name = "pan2"
    label = "PAN 2"
    modelspec = "pan2"
    # diffusion_model.<module>.lora_down/up + alpha under the checkpoint's own (diffusers) module names: what the
    # ComfyUI build with PAN 2 support maps for every weight of the model
    lora_format = "comfy"
    n_blocks = 44
    # the video stream's and the text stream's projections of each joint block go together (to_q with add_q_proj, ...)
    GROUPS = {"q": ("attn.to_q", "attn.add_q_proj"), "k": ("attn.to_k", "attn.add_k_proj"),
              "v": ("attn.to_v", "attn.add_v_proj"), "o": ("attn.to_out.0", "attn.to_add_out"),
              "ffn_up_gate": ("ff.net.0.proj", "ff_context.net.0.proj"), "ffn_down": ("ff.net.2", "ff_context.net.2")}
    target_groups = {"q": "attention q (video and text projections)", "k": "attention k (video and text)",
                     "v": "attention v (video and text)", "o": "attention out (video and text)",
                     "ffn_up_gate": "feed-forward in (video and text)", "ffn_down": "feed-forward out (video and text)",
                     "token_refiner": "the 4 text-refiner blocks over the caption"}
    default_targets = ARCH_DEFAULT_TARGETS["pan2"]
    text_prefixes = ("context_refiner.",)
    supports_single_files = True  # the files scripts/pan2/convert_pan2.py makes (nobody publishes them)
    empty_prompt = ""  # the reference's empty prompt template (cfg_drop_prob)
    default_sample_steps, default_sample_cfg = ARCH_FULL_SAMPLE["pan2"]
    latent_tag = "_wan22vae"
    size_multiple = 32
    supports_video = True

    def __init__(self):
        from .pan2 import loading

        self._loading = loading
        self.default_negative = loading.default_negative(0.0)  # the released structured negative, for a still

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
        from .pan2.sampler import sample

        return sample(model, pos_e, pos_m, neg_e, neg_m, **kw)

    def snap_frames(self, frames):
        from .pan2.sampler import snap_frames

        return snap_frames(frames)

    def latent_frames(self, frames):
        from .pan2.sampler import latent_frames

        return latent_frames(frames)

    def encode_video(self, vae, frames):
        return self._loading.encode_video(vae, frames)

    def render_prompt(self, prompt, frames=1):
        """A clip's prompt carries the clip's length in its time ranges, so a plain one is wrapped here, ahead of
        the text cache; a still's is wrapped when it is encoded (0 seconds), and JSON stays as it is."""
        if frames <= 1:
            return prompt
        return self._loading.structured(prompt, seconds=(frames - 1) / self.video_fps)

    def render_negative(self, frames=1):
        if frames <= 1:
            return self.default_negative
        return self._loading.default_negative((frames - 1) / self.video_fps)

    def decode_video(self, vae, latents):
        return self._loading.decode_video(vae, latents)

    def sample_sigmas(self, model, steps, height, width, shift):
        from .pan2.sampler import pan2_sigmas

        return pan2_sigmas(steps, getattr(model, "shift", 9.0))

    def add_trigger(self, caption, trigger):
        """A plain caption gets the trigger in front. A structured (JSON) caption stays valid JSON: the trigger goes
        in front of its first subject's display name; without a subject, in front of its first-frame description,
        else of its first event's description; with neither, it becomes the display name of a subject of its own
        (the skeleton's <|Subject1|>). A caption that names the trigger as a word already is left alone.

        Only a caption that does not parse as a JSON object gets the trigger in front: "trigger, {..." then reads as
        plain text and is wrapped (structured), since there is no field to put it in."""
        if not trigger or contains_word(caption or "", trigger):
            return caption
        text = (caption or "").strip()
        if text.startswith("{"):
            try:
                data = json.loads(text)
            except ValueError:
                data = None
            if isinstance(data, dict) and self._trigger_into(data, trigger):
                return json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        return super().add_trigger(caption, trigger)

    @staticmethod
    def _trigger_into(data: dict, trigger: str) -> bool:
        """Put the trigger into a structured caption (add_trigger's order); False when it has no place for it."""
        def front(text: str) -> str:
            return f"{trigger}, {text}" if text.strip() else trigger

        subjects = data.get("subjects_lifecycle")
        if isinstance(subjects, list) and subjects and isinstance(subjects[0], dict):
            name = subjects[0].get("display_name")
            subjects[0]["display_name"] = f"{trigger} {name if isinstance(name, str) else ''}".strip()
            return True
        if isinstance(data.get("init_frame_visual_description"), str):
            data["init_frame_visual_description"] = front(data["init_frame_visual_description"])
            return True
        events = data.get("events")
        first = events[0] if isinstance(events, list) and events else None
        if isinstance(first, dict) and isinstance(first.get("event_description"), str):
            first["event_description"] = front(first["event_description"])
            return True
        if subjects is None or subjects == []:
            data["subjects_lifecycle"] = [{"subject_id": "<|Subject1|>", "display_name": trigger,
                                           "subject_attribute": ""}]
            return True
        return False

    def lora_module_names(self, model, targets, blocks, include_text_embedder=False):
        from .lora import _is_linear_like

        targets = set(targets)
        if include_text_embedder:
            targets.add("token_refiner")
        block_set = None if blocks is None else {int(b) for b in blocks}
        in_block = sorted({s for t in targets for s in self.GROUPS.get(t, ())})
        refiner = ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "ff.net.0.proj", "ff.net.2")
        # the last block's text stream is never read again: its text queries, its attention's text output and its
        # text feed-forward only write to the text tokens, which reach no output, so a LoRA there would get no
        # gradient (its text keys and values still feed the video tokens)
        last = f"transformer_blocks.{len(model.transformer_blocks) - 1}."
        dead = tuple(last + m for m in ("attn.add_q_proj", "attn.to_add_out", "ff_context.net.0.proj",
                                        "ff_context.net.2"))
        names = []
        for name, module in model.named_modules():
            if not _is_linear_like(module):
                continue
            if name.startswith("transformer_blocks."):
                if block_set is not None and int(name.split(".")[1]) not in block_set:
                    continue
                if _match_suffix(name, in_block) and name not in dead:
                    names.append(name)
            elif name.startswith("context_refiner."):
                if "token_refiner" in targets and _match_suffix(name, refiner):
                    names.append(name)
        return names

    def compile_blocks(self, model):
        return list(model.transformer_blocks)

    def reference_sigmas(self, n, gen, tokens):
        # IFM's fine-tuning recipe (config/sft_29B_720p.yaml): a uniform draw, kept 1e-5 inside (0, 1), through the
        # fixed shift 9 (pretraining drew a logit-normal instead, under the same shift)
        from .pan2.model import SHIFT, shift_sigma

        u = torch.rand(n, generator=gen) * (1.0 - 2e-5) + 1e-5
        return shift_sigma(u, SHIFT)


_ARCHS["pan2"] = Pan2Arch
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
            elif "txt_in.text_norm.weight" in names and "modulation.1.weight" in names:
                found = "qwen_image21"
            elif "shared_block_modulation.weight" in names:
                found = "moss_image"
            elif "perceiver.latent_queries" in names:
                found = "fibo2"
            elif "early_stream_modulations.image.lin.weight" in names or \
                    "early_stream_modulations.image.linear.weight" in names:
                found = "flux3_klein"
            elif "double_stream_modulation_img.lin.weight" in names or \
                    "double_stream_modulation_img.linear.weight" in names:
                found = "flux2_klein"
            elif "embed_image_indicator.weight" in names and "llm_cond_proj.weight" in names:
                found = "ideogram4"
            elif "video_patch_proj.weight" in names and "audio_patch_proj.weight" in names:
                found = "p_h3_image" if _is_ph3_dit(p) else "minimax_h3"
            elif "video_patch_proj.weight" in names and "condition_proj.weight" in names:
                found = "minimax_h3_image"  # H3's layout without the audio stream (ComfyUI's detection rule)
            elif "context_type_embedding" in names and "transformer_blocks.0.attn.add_q_proj.weight" in names:
                found = "pan2"
        except Exception:  # noqa: BLE001
            found = None
        _FILE_ARCH_CACHE[key] = found
    return _FILE_ARCH_CACHE[key]


PH3_ADALN_TABLE = (4097, 16)  # P-H3-Image's pruned AdaLN curve table (Comfy-Org's H3 files: [1025, 8])


def _is_ph3_dit(path: Path) -> bool:
    """An H3-layout diffusion model that is P-H3-Image's: named so, or with its [4097, 16] AdaLN table."""
    from safetensors import safe_open

    from .config import family_named_in

    if family_named_in(path.name) == "p_h3_image":
        return True
    with safe_open(str(path), "pt") as f:
        for k in f.keys():
            if k.split("diffusion_model.", 1)[-1] == "adaln_t_table":
                return tuple(f.get_slice(k).get_shape()) == PH3_ADALN_TABLE
    return False


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
                 "MOSImageTransformer": "moss_image", "MOSSImageTransformer": "moss_image",
                 "QwenImage21Transformer2DModel": "qwen_image21",
                 "Flux3Transformer2DModel": "flux3_klein", "Flux2Transformer2DModel": "flux2_klein",
                 "Ideogram4Transformer2DModel": "ideogram4",
                 "MiniMaxH3Transformer3DModel": "minimax_h3", "MiniMaxH3DiTModel": "minimax_h3",
                 "MiniMaxH3ImageTransformer2DModel": "minimax_h3_image",
                 "PAN2Transformer3DModel": "pan2"}.get(cls)
        if found in (None, "minimax_h3"):
            from .h3image.loading import is_h3_image_dir

            if is_h3_image_dir(Path(model_dir)):
                # MiniMaxAI's transformer/ is H3's own class (MiniMaxH3DiTModel) with audio_enabled false; a folder
                # with only transformer_diffusers/ has no transformer/config.json at all
                found = "minimax_h3_image"
        if found == "minimax_h3":
            from .ph3.loading import is_ph3_dir

            if is_ph3_dir(Path(model_dir)):
                found = "p_h3_image"  # the same transformer class: the pipeline index tells them apart
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
