"""MiniMax H3 omni-transformer (H3-Omni-Transformer, 33B) for one-frame (image) LoRA training, in RulzTrainer's
conventions.

Ported from diffusers' transformer_minimax_h3.py (MiniMaxH3Transformer3DModel, Copyright 2025 The MiniMax Team and The
HuggingFace Team, Apache-2.0), with the packed layout of diffusers' modular pipeline
(modular_pipelines/minimax_h3/before_denoise.py: build_packed_sequence, the timestep plan) and ComfyUI's
comfy/ldm/minimax/model.py (the pruned AdaLN curve table, the video -> audio sigma mapping) as references. What is
modeled on evidence and how it is checked is in tests/test_h3.py; nothing here was run on the real weights.

Module names are the original checkpoint's (MiniMaxAI/MiniMax-H3 FL2VA/transformer, Comfy-Org's single files):
blocks.N.attn.qkv_proj ([q; k; v] stacked by rows, the layout of Comfy-Org's files), attn.q_norm / k_norm,
attn.out_proj, mlp.fc1 ([gate; value]), mlp.fc2, adaln_proj.linear, norm1 / norm2; token_refiner.blocks.N (the same
attention / MLP without AdaLN or RoPE); video_patch_proj, audio_patch_proj, condition_proj, time_embedder.proj_in /
proj_out (or adaln_t_table in a pruned file) and final_layer.{norm, adaln_proj.linear, video_out, audio_out}. A LoRA on
these names is the layout the ecosystem trains and loads (diffusion_model.blocks.N.attn.qkv_proj.lora_A/B: ai-toolkit,
diffusion-pipe, musubi-tuner, ComfyUI's loader, diffusers' MiniMaxH3LoraLoaderMixin), and the training adapters fit it
as they are.

The forward takes what the trainer gives every family: `hidden_states` the normalized H3 video-VAE latents of ONE frame
[B, 24, H/16, W/16] (the model patchifies 1x2x2, so H and W are multiples of 32 px), `timestep` sigma * 1000, the text
features [B, S, 5120] (Qwen3-VL-32B's hidden state after decoder layer 50, h3/loading.py) and a [B, S] mask of the real
tokens. It returns the velocity in the trainer's convention (noise - x0): H3's heads predict the data-ward velocity
(x0 - noise; diffusers' MiniMaxH3Scheduler: x0 = x_t + sigma * v), so the output is negated, as ComfyUI does.

One packed sequence per sample, [text | target audio | target video], full self-attention, no cross-attention:
  - the one-frame layout of musubi-tuner's one-frame mode (docs/minimax_h3_1f.md) and ComfyUI's EmptyMiniMaxH3LatentAV:
    one target video latent frame plus the two audio latent frames per stereo channel the joint layout requires
    (round(1 pixel frame * 40 / 24) = 2), channel-major, 4 audio rows;
  - RoPE over (t, h, w): text token j at (j, 0, 0); the media clock starts at the sample's real text length L: the
    video frame at t = L with the area-normalized (h, w) grid (diffusers' _frame_position_grid), audio frame f at
    t = L + f, h = 0, w pinned to the two ends of the width grid (left channel, right channel). Positions are per
    sample, so a padded batch gives each sample what it gets alone; padded text tokens are masked out as keys;
  - timesteps: H3 runs one base time per item with two shifts, 12 for video and 3 for audio (the released schedulers,
    ComfyUI's MiniMaxH3SigmaShift, musubi's training). The trainer draws the video sigma; the audio sigma follows in
    closed form (base = s_v / (12 - 11 s_v), s_a = 3 base / (1 + 2 base)), as ComfyUI's model does. The model reads
    t = 1 - sigma (1 = clean); text rows run at the video time (the pipeline's row plan);
  - AdaLN: each block has one projection giving 6 vectors per (timestep, modality) pair, modality tags 0 video, 1 text,
    2 audio (the pipeline's tags; a T2VA presentation is all text). Pruned files (Comfy-Org's *_pruned_*) replace the
    time embedder by an fp32 [1025, 8] table of the SiLU'd time-embedding curve over t, interpolated linearly, feeding
    8-wide projections without the SiLU (ComfyUI's use_adaln_curves path; kept in fp32 as ComfyUI keeps them);
  - the audio stream: an image has no sound. Without `audio_states` the 4 audio rows are the silence placeholder
    (`audio_silence`, the audio VAE's encoding of a zero waveform, set by the loader) noised to the audio sigma with
    fresh noise, as musubi's one-frame training does (--video_only: the audio rows are context, never supervised). The
    trainer's loss reads the video output alone. The sampler (h3/sampler.py) passes its own audio rows and steps them
    with the audio prediction (`return_audio`), as joint generation does.

Gradient checkpointing (torch.utils.checkpoint), the model-split device hand-off and the alignment taps work like the
other ports; `skip_layers` skips blocks (kept for the shared sampler signature); QuantLinear / LoRALinear replace
nn.Linear transparently. The fp32 islands of the checkpoint (patch projections, time embedder, output heads, the pruned
AdaLN) stay fp32 when the loader keeps them so; the blocks run in the condition projection's dtype (bf16)."""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..attention import sdpa
from ..compiling import thread_prepare

VIDEO_TAG, TEXT_TAG, AUDIO_TAG = 0, 1, 2  # the pipeline's modality tags (they index the AdaLN table)
MODALITIES = 3
SHIFT_VIDEO, SHIFT_AUDIO = 12.0, 3.0  # the released schedulers' shifts (scheduler/ and audio_scheduler/ configs)
AUDIO_CHANNELS = 2  # stereo, packed channel-major
ONE_FRAME_AUDIO_LATENTS = 2  # audio latent frames per channel for a one-frame target (musubi, ComfyUI)
ROPE_SPATIAL_SCALE = 32.0
PATCH = 2  # the transformer's 1x2x2 patch (latents per token edge)


def shift_sigma(base, shift: float):
    """The schedulers' exponential shift: s * b / (1 + (s - 1) * b)."""
    return shift * base / (1.0 + (shift - 1.0) * base)


def time_shift_sigma(sigma, from_shift: float, to_shift: float):
    """A sigma on one shift's curve to the same base time on another shift's (ComfyUI's time_shift_sigma)."""
    base = sigma / (from_shift + sigma * (1.0 - from_shift))
    return shift_sigma(base, to_shift)


def audio_sigma(sigma_video, shift_video: float = SHIFT_VIDEO, shift_audio: float = SHIFT_AUDIO):
    return time_shift_sigma(sigma_video, shift_video, shift_audio)


def _wdtype(m: nn.Module) -> torch.dtype:
    d = getattr(m, "compute_dtype", None)
    return d if d is not None else m.weight.dtype


def _module_device(m: nn.Module) -> torch.device | None:
    p = next(m.parameters(), None)
    if p is not None:
        return p.device
    b = next(m.buffers(), None)
    return b.device if b is not None else None


class RMSNorm(nn.Module):
    """torch.nn.RMSNorm (what diffusers' port uses): F.rms_norm with the stored weight."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim, self.eps = dim, eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (self.dim,), self.weight.to(x.dtype), self.eps)


def inv_freq(freq_dim: int, theta: float = 10000.0, device=None) -> torch.Tensor:
    """diffusers' MiniMaxH3RotaryPosEmbed.inv_freq (the checkpoint's rope.inv_freq): theta^(-k / freq_dim)."""
    return 1.0 / (theta ** (torch.arange(0, 2 * freq_dim, 2, dtype=torch.float32, device=device) / (2 * freq_dim)))


def rope_tables(pos: torch.Tensor, freqs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """cos / sin [B, L, 6 * F] for positions [B, L, 3] (float64), as diffusers computes them: positions to fp32, one
    inv_freq for the three axes, (t, h, w) concatenated, then the whole block twice (rotate-half convention)."""
    angles = pos.to(torch.float32).unsqueeze(-1) * freqs.view(1, 1, 1, -1)  # [B, L, 3, F]
    angles = torch.cat(angles.unbind(dim=2), dim=-1)
    angles = torch.cat((angles, angles), dim=-1)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x [B, L, heads, D]: the leading cos.shape[-1] channels of every head rotate (rotate_half), the rest pass."""
    r = cos.shape[-1]
    rot, rest = x[..., :r], x[..., r:]
    c, s = cos.to(x.dtype)[:, :, None, :], sin.to(x.dtype)[:, :, None, :]
    x1, x2 = rot.chunk(2, dim=-1)
    rot = rot * c + torch.cat((-x2, x1), dim=-1) * s
    return torch.cat((rot, rest), dim=-1)


def _axis(dim: int, sqrt_area: float) -> torch.Tensor:
    """One aspect-normalized spatial axis (diffusers' _spatial_position_grid): dim // 2 coordinates over the unit
    interval centred on 0.5, times 32, the right end excluded; numpy's linspace in float64 as the reference does."""
    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    return torch.from_numpy(np.linspace(left, left + ratio, dim // PATCH, endpoint=False) * ROPE_SPATIAL_SCALE)


def frame_grid(height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The (h, w) rotary coordinates [N, 2] of one latent frame of height x width latent cells, and its width axis."""
    sqrt_area = float(np.sqrt(height * width))
    hs, ws = _axis(height, sqrt_area), _axis(width, sqrt_area)
    gh, gw = torch.meshgrid(hs, ws, indexing="ij")
    return torch.stack([gh.reshape(-1), gw.reshape(-1)], dim=-1), ws


def patchify(latents: torch.Tensor) -> torch.Tensor:
    """[B, C, h, w] (one frame) -> [B, h/2 * w/2, C * 4] rows, row-major, channels (c, ph, pw) as the pipeline packs."""
    b, c, h, w = latents.shape
    x = latents.reshape(b, c, h // PATCH, PATCH, w // PATCH, PATCH).permute(0, 2, 4, 1, 3, 5)
    return x.reshape(b, (h // PATCH) * (w // PATCH), c * PATCH * PATCH)


def unpatchify(rows: torch.Tensor, channels: int, h: int, w: int) -> torch.Tensor:
    b = rows.shape[0]
    x = rows.reshape(b, h // PATCH, w // PATCH, channels, PATCH, PATCH).permute(0, 3, 1, 4, 2, 5)
    return x.reshape(b, channels, h, w)


class Attention(nn.Module):
    """Self-attention: fused [q; k; v] projection, per-head q/k RMSNorm, partial RoPE (blocks only), padding mask."""

    def __init__(self, hidden: int, heads: int, head_dim: int, eps: float = 1e-5):
        super().__init__()
        self.heads, self.head_dim = heads, head_dim
        self.inner = heads * head_dim  # larger than hidden in H3 (56 x 128 = 7168 > 5376)
        self.qkv_proj = nn.Linear(hidden, 3 * self.inner, bias=False)
        self.q_norm = RMSNorm(head_dim, eps)
        self.k_norm = RMSNorm(head_dim, eps)
        self.out_proj = nn.Linear(self.inner, hidden, bias=False)

    def forward(self, x: torch.Tensor, cos=None, sin=None, key_mask: torch.Tensor | None = None) -> torch.Tensor:
        """key_mask: [B, 1, 1, L] bool of the keys every query may see (None: all)."""
        q, k, v = self.qkv_proj(x).split(self.inner, dim=-1)
        q = self.q_norm(q.unflatten(-1, (self.heads, self.head_dim)))
        k = self.k_norm(k.unflatten(-1, (self.heads, self.head_dim)))
        v = v.unflatten(-1, (self.heads, self.head_dim))
        if cos is not None:
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        q, k, v = q.transpose(1, 2), k.to(q.dtype).transpose(1, 2), v.to(q.dtype).transpose(1, 2)
        tap = getattr(self, "_rz_tap", None)  # rulztrainer.alignment: the trigger's attention (training-only loss)
        if tap is not None:
            tap(q, k)
        out = sdpa(q, k, v, attn_mask=key_mask)
        return self.out_proj(out.transpose(1, 2).flatten(2).to(x.dtype))


class MLP(nn.Module):
    """fc2(silu(gate) * value), fc1 = [gate; value] by rows (the checkpoint's order; diffusers' SwiGLU reads the
    halves swapped, see h3/loading.py)."""

    def __init__(self, hidden: int, ffn: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden, 2 * ffn, bias=False)
        self.fc2 = nn.Linear(ffn, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(F.silu(gate) * value)


class RefinerBlock(nn.Module):
    """Plain pre-norm block over the text alone: no AdaLN, no RoPE."""

    def __init__(self, hidden: int, heads: int, head_dim: int, ffn: int, eps: float, qk_eps: float):
        super().__init__()
        self.norm1 = RMSNorm(hidden, eps)
        self.attn = Attention(hidden, heads, head_dim, qk_eps)
        self.norm2 = RMSNorm(hidden, eps)
        self.mlp = MLP(hidden, ffn)

    def forward(self, x: torch.Tensor, key_mask=None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), key_mask=key_mask)
        return x + self.mlp(self.norm2(x))


class TokenRefiner(nn.Module):
    def __init__(self, num_layers: int, hidden: int, heads: int, head_dim: int, ffn: int, eps: float, qk_eps: float,
                 final_eps: float):
        super().__init__()
        self.blocks = nn.ModuleList([RefinerBlock(hidden, heads, head_dim, ffn, eps, qk_eps)
                                     for _ in range(num_layers)])
        self.final_norm = RMSNorm(hidden, final_eps)

    def forward(self, x: torch.Tensor, key_mask=None) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, key_mask)
        return self.final_norm(x)


class AdalnProj(nn.Module):
    """One projection of the time embedding to expand x hidden values per modality (block: 6 x 3, final layer: 2 x 1).
    The full model applies SiLU to the fp32 embedding and casts to the projection's dtype after (diffusers); a pruned
    file's curve table has the SiLU baked in."""

    def __init__(self, t_dim: int, hidden: int, expand: int, modalities: int, apply_silu: bool = True):
        super().__init__()
        self.expand, self.modalities, self.hidden, self.apply_silu = expand, modalities, hidden, apply_silu
        self.linear = nn.Linear(t_dim, expand * hidden * modalities, bias=True)

    def forward(self, temb: torch.Tensor) -> torch.Tensor:
        """[M, t_dim] -> [M, modalities, expand, hidden]."""
        x = F.silu(temb) if self.apply_silu else temb
        out = self.linear(x.to(_wdtype(self.linear)))
        return out.unflatten(-1, (self.modalities, self.expand, self.hidden))


def _modulate(h: torch.Tensor, bounds, mods, k_scale: int, k_shift: int) -> torch.Tensor:
    """h * (1 + scale) + shift per segment; mods: per segment [B, 6, hidden]."""
    parts = []
    for (a, b), m in zip(bounds, mods):
        scale, shift = m[:, k_scale, None].to(h.dtype), m[:, k_shift, None].to(h.dtype)
        parts.append(h[:, a:b] * (1.0 + scale) + shift)
    return torch.cat(parts, dim=1)


def _gated(x: torch.Tensor, y: torch.Tensor, bounds, mods, k_gate: int) -> torch.Tensor:
    return x + torch.cat([y[:, a:b] * m[:, k_gate, None].to(y.dtype) for (a, b), m in zip(bounds, mods)], dim=1)


class DiTBlock(nn.Module):
    """Pre-norm self-attention and feed-forward over the packed sequence, each AdaLN-modulated per segment
    (text: video time + text tag, audio: audio time + audio tag, video: video time + video tag)."""

    def __init__(self, hidden: int, heads: int, head_dim: int, ffn: int, t_dim: int, eps: float, qk_eps: float,
                 apply_silu: bool = True):
        super().__init__()
        self.hidden = hidden
        self.norm1 = RMSNorm(hidden, eps)
        self.attn = Attention(hidden, heads, head_dim, qk_eps)
        self.norm2 = RMSNorm(hidden, eps)
        self.mlp = MLP(hidden, ffn)
        self.adaln_proj = AdalnProj(t_dim, hidden, 6, MODALITIES, apply_silu)

    def forward(self, x, temb, cos, sin, key_mask, n_text: int, n_audio: int):
        """temb [2B, t_dim]: the video time's embeddings, then the audio time's."""
        batch = x.shape[0]
        mod = self.adaln_proj(temb)  # [2B, 3, 6, hidden]: shift, scale, gate (attention), shift, scale, gate (MLP)
        mv, ma = mod[:batch], mod[batch:]
        mods = (mv[:, TEXT_TAG], ma[:, AUDIO_TAG], mv[:, VIDEO_TAG])
        bounds = ((0, n_text), (n_text, n_text + n_audio), (n_text + n_audio, x.shape[1]))
        h = _modulate(self.norm1(x), bounds, mods, 1, 0)
        x = _gated(x, self.attn(h, cos, sin, key_mask), bounds, mods, 2)
        h = _modulate(self.norm2(x), bounds, mods, 4, 3)
        return _gated(x, self.mlp(h), bounds, mods, 5)


class FinalLayer(nn.Module):
    """RMSNorm, shift / scale from the time embedding (no modality axis), and the two fp32 heads."""

    def __init__(self, hidden: int, t_dim: int, video_dim: int, audio_dim: int, eps: float, apply_silu: bool = True):
        super().__init__()
        self.norm = RMSNorm(hidden, eps)
        self.adaln_proj = AdalnProj(t_dim, hidden, 2, 1, apply_silu)
        self.video_out = nn.Linear(hidden, video_dim, bias=True)
        self.audio_out = nn.Linear(hidden, audio_dim, bias=True)

    def forward(self, x, temb, n_text: int, n_audio: int):
        batch = x.shape[0]
        mod = self.adaln_proj(temb)[:, 0]  # [2B, 2, hidden]: shift, scale
        a0, v0 = n_text, n_text + n_audio

        def head(rows, m, out):
            h = self.norm(rows)
            h = h * (1.0 + m[:, 1, None].to(h.dtype)) + m[:, 0, None].to(h.dtype)
            return out(h.to(_wdtype(out)))

        return head(x[:, v0:], mod[:batch], self.video_out), head(x[:, a0:v0], mod[batch:], self.audio_out)


class MiniMaxH3Transformer(nn.Module):
    """MiniMaxH3Transformer3DModel for one-frame targets. hidden_states: [B, 24, H/16, W/16] normalized latents."""

    # the original checkpoint's config names (FL2VA/transformer/config.json, the Comfy-Org files' "config" metadata)
    CONFIG_KEYS = ("hidden_size", "num_layers", "token_refiner_num_layers", "num_attention_heads",
                   "attention_head_dim", "ffn_hidden_size", "latents_dim", "audio_latents_dim", "patch_size",
                   "text_dim", "timestep_input_dim", "time_embed_hidden_size", "time_embed_dim", "rope_inv_freq_len",
                   "rope_theta", "norm_eps", "qk_norm_eps", "final_norm_eps", "adaln_curve_grid")

    def __init__(self, hidden_size=5376, num_layers=50, token_refiner_num_layers=2, num_attention_heads=56,
                 attention_head_dim=128, ffn_hidden_size=14336, latents_dim=24, audio_latents_dim=32,
                 patch_size=(1, 2, 2), text_dim=5120, timestep_input_dim=256, time_embed_hidden_size=5376,
                 time_embed_dim=2688, rope_inv_freq_len=16, rope_theta=10000.0, norm_eps=1e-5, qk_norm_eps=1e-5,
                 final_norm_eps=1e-5, adaln_curve_grid=None):
        super().__init__()
        if tuple(patch_size) != (1, PATCH, PATCH):
            raise ValueError(f"MiniMax H3 patchifies 1x2x2, got {tuple(patch_size)}")
        if 6 * rope_inv_freq_len > attention_head_dim:
            raise ValueError("the rotary channels (6 x rope_inv_freq_len) exceed the head size")
        self.config = dict(hidden_size=hidden_size, num_layers=num_layers,
                           token_refiner_num_layers=token_refiner_num_layers, num_attention_heads=num_attention_heads,
                           attention_head_dim=attention_head_dim, ffn_hidden_size=ffn_hidden_size,
                           latents_dim=latents_dim, audio_latents_dim=audio_latents_dim, patch_size=list(patch_size),
                           text_dim=text_dim, timestep_input_dim=timestep_input_dim,
                           time_embed_hidden_size=time_embed_hidden_size, time_embed_dim=time_embed_dim,
                           rope_inv_freq_len=rope_inv_freq_len, rope_theta=rope_theta, norm_eps=norm_eps,
                           qk_norm_eps=qk_norm_eps, final_norm_eps=final_norm_eps, adaln_curve_grid=adaln_curve_grid)
        self.latent_channels = latents_dim
        self.audio_channels = audio_latents_dim
        self.text_dim = text_dim
        self.hidden_size = hidden_size
        self.freq_dim = timestep_input_dim
        self.rope_freq_dim, self.rope_theta = rope_inv_freq_len, rope_theta
        self.pruned = adaln_curve_grid is not None
        self.audio_frames = ONE_FRAME_AUDIO_LATENTS
        self.shift_video, self.shift_audio = SHIFT_VIDEO, SHIFT_AUDIO
        self.gradient_checkpointing = False
        self.is_distilled = True  # the released checkpoints are CFG-distilled (no guidance: sample at CFG 1)
        # the audio VAE's encoding of 1600 zero samples, normalized: [4, 32] rows (left f0, f1, right f0, f1). Set by
        # the loader (h3/loading.py); None: zeros (logged once)
        self.audio_silence: torch.Tensor | None = None
        self.audio_noise_seed = 0  # the placeholder's noise: a fresh draw per forward, reproducible per run
        apply_silu = not self.pruned
        t_dim = time_embed_dim
        video_dim = latents_dim * PATCH * PATCH
        self.video_patch_proj = nn.Linear(video_dim, hidden_size, bias=True)
        self.audio_patch_proj = nn.Linear(audio_latents_dim, hidden_size, bias=True)
        self.condition_proj = nn.Linear(text_dim, hidden_size, bias=True)
        if self.pruned:
            self.register_buffer("adaln_t_table", torch.zeros(adaln_curve_grid, time_embed_dim))
            self.time_embedder = None
        else:
            self.time_embedder = nn.Module()
            self.time_embedder.proj_in = nn.Linear(timestep_input_dim, time_embed_hidden_size, bias=True)
            self.time_embedder.proj_out = nn.Linear(time_embed_hidden_size, time_embed_dim, bias=True)
        self.token_refiner = TokenRefiner(token_refiner_num_layers, hidden_size, num_attention_heads,
                                          attention_head_dim, ffn_hidden_size, norm_eps, qk_norm_eps, final_norm_eps)
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_attention_heads, attention_head_dim, ffn_hidden_size, t_dim, norm_eps,
                     qk_norm_eps, apply_silu) for _ in range(num_layers)])
        self.final_layer = FinalLayer(hidden_size, t_dim, video_dim, audio_latents_dim, final_norm_eps, apply_silu)

    @property
    def layers(self) -> nn.ModuleList:
        """The main blocks under the name the trainer uses (block selection, block swap, model split)."""
        return self.blocks

    def split_tail_modules(self) -> list[nn.Module]:
        """Modules that run after the last block (they follow it in a model split)."""
        return [self.final_layer]

    @classmethod
    def from_config(cls, config: dict) -> "MiniMaxH3Transformer":
        return cls(**{k: config[k] for k in cls.CONFIG_KEYS if k in config})

    def _run(self, block, *args):
        thread_prepare()  # dynamo's limits in this thread (autograd recompute, data-parallel workers)
        if (self.gradient_checkpointing and self.training and torch.is_grad_enabled()
                and not getattr(block, "keep_activations", False)):  # see rulztrainer.checkpointing
            return checkpoint(block, *args, use_reentrant=False)
        return block(*args)

    # ---- time
    def time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """[M] model times t = 1 - sigma in [0, 1] -> [M, t_dim]: the sinusoidal embedding (cos first, unscaled t, as
        diffusers' Timesteps(flip_sin_to_cos=True)) through the fp32 time MLP, or the pruned curve table, linearly
        interpolated over 1025 points of t (ComfyUI)."""
        t = t.to(torch.float32)
        if self.pruned:
            table = self.adaln_t_table.to(device=t.device, dtype=torch.float32)
            pos = t.clamp(0.0, 1.0) * (table.shape[0] - 1)
            i0 = pos.floor().long().clamp(max=table.shape[0] - 2)
            return torch.lerp(table[i0], table[i0 + 1], (pos - i0.to(pos.dtype)).unsqueeze(1))
        half = self.freq_dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
        args = t[:, None] * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        te = self.time_embedder
        return te.proj_out(F.silu(te.proj_in(emb.to(_wdtype(te.proj_in)))))

    # ---- layout
    def positions(self, text_lens: torch.Tensor, text_len: int, height: int, width: int) -> torch.Tensor:
        """[B, S + 4 + N, 3] float64 RoPE positions of [text | audio | video] (see the module docstring)."""
        batch = text_lens.shape[0]
        frames = self.audio_frames
        n_audio = frames * AUDIO_CHANNELS
        grid, w_axis = frame_grid(height, width)
        n_video = grid.shape[0]
        pos = torch.zeros(batch, text_len + n_audio + n_video, 3, dtype=torch.float64)
        pos[:, :text_len, 0] = torch.arange(text_len, dtype=torch.float64)
        cursor = text_lens.detach().to("cpu", torch.float64).view(batch, 1)
        a0, v0 = text_len, text_len + n_audio
        pos[:, a0:v0, 0] = cursor + torch.arange(frames, dtype=torch.float64).repeat(AUDIO_CHANNELS)[None]
        pos[:, a0:v0, 2] = torch.cat([torch.full((frames,), float(w_axis[0]), dtype=torch.float64),
                                      torch.full((frames,), float(w_axis[-1]), dtype=torch.float64)])[None]
        pos[:, v0:, 0] = cursor
        pos[:, v0:, 1:] = grid[None]
        return pos

    def audio_placeholder(self, sigma_a: torch.Tensor, batch: int, device, noise: torch.Tensor | None = None):
        """[B, 4, 32] fp32: the silence latent noised to each sample's audio sigma."""
        n_audio = self.audio_frames * AUDIO_CHANNELS
        silence = self.audio_silence
        if silence is None:
            if not getattr(self, "_warned_silence", False):
                import logging

                logging.getLogger("rulztrainer").warning(
                    "MiniMax H3: no silence latent was set; the audio placeholder is zeros (load the audio VAE)")
                self._warned_silence = True
            silence = torch.zeros(n_audio, self.audio_channels)
        silence = silence.to(device, torch.float32).reshape(1, n_audio, self.audio_channels)
        if noise is None:
            gen = torch.Generator().manual_seed(int(self.audio_noise_seed))
            self.audio_noise_seed = int(self.audio_noise_seed) + 1
            noise = torch.randn(batch, n_audio, self.audio_channels, generator=gen)
        s = sigma_a.to(device, torch.float32).view(-1, 1, 1)
        return (1.0 - s) * silence + s * noise.to(device, torch.float32)

    def forward(self, hidden_states: torch.Tensor, timestep, encoder_hidden_states: torch.Tensor,
                encoder_attention_mask: torch.Tensor | None = None, skip_layers=None,
                audio_states: torch.Tensor | None = None, return_audio: bool = False):
        """timestep: the video noise level in 0..1000 (sigma * 1000). audio_states: [B, 4, 32] audio rows at the
        matching audio sigma (None: the noised silence placeholder). Returns the noise-ward velocity [B, 24, h, w]
        (fp32), and with return_audio also the audio rows' [B, 4, 32]."""
        skip_layers = frozenset(skip_layers or ())
        if hidden_states.ndim != 4 or hidden_states.shape[1] != self.latent_channels:
            raise ValueError(f"hidden_states must be one frame of latents [B, {self.latent_channels}, H/16, W/16]")
        if encoder_hidden_states.ndim != 3 or encoder_hidden_states.shape[2] != self.text_dim:
            raise ValueError(f"encoder_hidden_states must be [B, S, {self.text_dim}]")
        batch, _, height, width = hidden_states.shape
        if height % PATCH or width % PATCH:
            raise ValueError("MiniMax H3 latents need an even height and width (images in multiples of 32 px)")
        device = hidden_states.device
        dtype = _wdtype(self.condition_proj)
        text_len = encoder_hidden_states.shape[1]
        n_audio = self.audio_frames * AUDIO_CHANNELS

        sigma_v = torch.as_tensor(timestep, device=device, dtype=torch.float32).reshape(-1).expand(batch) / 1000.0
        sigma_a = audio_sigma(sigma_v, self.shift_video, self.shift_audio)
        temb = self.time_embedding(torch.cat([1.0 - sigma_v, 1.0 - sigma_a]))  # [2B, t_dim]: video, then audio
        if audio_states is None:
            audio_states = self.audio_placeholder(sigma_a, batch, device)
        if audio_states.shape != (batch, n_audio, self.audio_channels):
            raise ValueError(f"audio_states must be [B, {n_audio}, {self.audio_channels}]")

        if encoder_attention_mask is not None:
            valid = encoder_attention_mask.to(device=device, dtype=torch.bool)
            text_lens = valid.sum(dim=1)
            padded = not bool(valid.all())
        else:
            valid, padded = None, False
            text_lens = torch.full((batch,), text_len, device=device, dtype=torch.long)
        n_video = (height // PATCH) * (width // PATCH)
        key_mask = txt_mask = None
        if padded:
            key_mask = torch.cat([valid, valid.new_ones(batch, n_audio + n_video)], dim=1)[:, None, None, :]
            txt_mask = valid[:, None, None, :]
        pos = self.positions(text_lens, text_len, height, width).to(device)
        cos, sin = rope_tables(pos, inv_freq(self.rope_freq_dim, self.rope_theta, device))

        txt = self.token_refiner(self.condition_proj(encoder_hidden_states.to(device, dtype)), txt_mask)
        vid = self.video_patch_proj(patchify(hidden_states).to(_wdtype(self.video_patch_proj))).to(txt.dtype)
        aud = self.audio_patch_proj(audio_states.to(device, _wdtype(self.audio_patch_proj))).to(txt.dtype)
        x = torch.cat([txt, aud, vid], dim=1)

        split = getattr(self, "split_devices", None)  # blocks may live on several devices (model split)
        taps = getattr(self, "_rz_taps", None)  # rulztrainer.alignment (training-only losses)
        if taps is not None and taps.armed:
            taps.set_layout(0, (text_len + n_audio, n_video), (0, text_len), valid)
        for i, block in enumerate(self.blocks):
            if i in skip_layers:
                continue
            if split:
                bdev = _module_device(block)
                if bdev is not None and x.device != bdev:
                    x, temb, cos, sin = x.to(bdev), temb.to(bdev), cos.to(bdev), sin.to(bdev)
                    key_mask = key_mask.to(bdev) if key_mask is not None else None
            x = self._run(block, x, temb, cos, sin, key_mask, text_len, n_audio)
            if taps is not None and taps.armed:
                taps.record_hidden(i, x[:, text_len + n_audio:])

        if split:
            fdev = _module_device(self.final_layer)
            if fdev is not None and x.device != fdev:
                x, temb = x.to(fdev), temb.to(fdev)
        video, audio = self.final_layer(x, temb, text_len, n_audio)
        out = -unpatchify(video.float(), self.latent_channels, height, width).to(device)
        if return_audio:
            return out, -audio.float().to(device)
        return out
