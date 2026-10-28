"""Qwen-Image 2.1 DiT (QwenImage21Transformer2DModel) for training, in RulzTrainer's conventions.

Ported from diffusers' transformer_qwenimage21.py (Copyright 2026 Qwen-Image Team and The HuggingFace Team,
Apache-2.0) with the diffusers mixins removed, so that:
  - the state-dict keys are the Hugging Face checkpoint's (Qwen/Qwen-Image-2.1 transformer/): the feed-forward keeps
    its two input projections apart (img_mlp.gate_layer / img_mlp.proj, which ComfyUI's single file fuses into
    gate_up and its LoRA loader maps back onto the halves), so a LoRA exported with these names loads in diffusers
    and in ComfyUI, and Fizgig's training adapter fits it;
  - the forward takes what the trainer and sampler give the other families: `hidden_states` as the normalized
    Qwen-Image 2.1 VAE latents [B, 64, H/16, W/16] (the VAE is 16x with 64 channels and the model reads them
    unpatched), `timestep` in 0..1000 units, and the text features as [B, S, 4096] (Qwen3-VL-8B's last layer before
    its final norm, see qwen21/loading.py) with a [B, S] mask of the real tokens;
  - gradient checkpointing (torch.utils.checkpoint) and the model-split device hand-off work like the other ports;
    `skip_layers` skips blocks (kept for the shared sampler signature);
  - QuantLinear / LoRALinear replace nn.Linear modules transparently (only their dtype handling is consulted).

Architecture: 32 single-stream blocks over [text, image] with q/k RMSNorm, 3-axis RoPE and a SwiGLU feed-forward;
one modulation Linear shared by every block (no per-block tables), tanh-gated residuals, and a scale-only final norm.
Two things set 2.1 apart and are kept exactly:
  - block-causal attention: text tokens see the text before them (causal), image tokens see everything. Padded text
    tokens sit after the real ones, so the causal text pass never reaches them and the image pass masks them out;
  - causal_condition: text tokens are modulated with t = 0, only the image tokens with the sampled timestep.
RoPE positions: text token j sits at (j, j, j); the image's frame axis is the number of real text tokens and its
height / width axes a grid centred on zero. Positions are per sample, so a padded batch gives each sample exactly
what it gets alone (the reference pipeline counts the padding instead). The edit path (reference images placed in
the text's vision slots) is not ported: the trainer trains text-to-image LoRAs.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..attention import sdpa
from ..compiling import thread_prepare

ROPE_THETA = 10000


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
    """diffusers' RMSNorm: normalized in fp32, cast to the weight's dtype, times the (stored, not zero-centred) weight."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim, self.eps = dim, eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.rms_norm(x.float(), (self.dim,), eps=self.eps)
        if self.weight.dtype in (torch.float16, torch.bfloat16):
            h = h.to(self.weight.dtype)
        return (h * self.weight).to(x.dtype)


class ZeroCenterRMSNorm(nn.Module):
    """RMSNorm whose stored weight is zero-centred: the multiplier is 1 + weight, computed in fp32."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim, self.eps = dim, eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x.float(), (self.dim,), weight=self.weight.float() + 1.0, eps=self.eps).to(x.dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x [B, L, H, D]; cos/sin [B, L, D] with interleaved (real, imaginary) pairs (diffusers' complex RoPE)."""
    x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
    rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
    cos, sin = cos[:, :, None, :], sin[:, :, None, :]
    return (x.float() * cos + rotated.float() * sin).to(x.dtype)


def rope_tables(pos: torch.Tensor, axes_dims, theta: float = ROPE_THETA) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin [B, L, sum(axes_dims)] for integer positions [B, L, 3], each pair's angle repeated (interleaved). The
    angles are computed in fp32 as diffusers' rope_params does."""
    cos_out, sin_out = [], []
    for i, dim in enumerate(axes_dims):
        freqs = 1.0 / torch.pow(theta, torch.arange(0, dim, 2, device=pos.device).to(torch.float32).div(dim))
        angles = pos[..., i].to(torch.float32)[..., None] * freqs
        cos_out.append(angles.cos().repeat_interleave(2, dim=-1))
        sin_out.append(angles.sin().repeat_interleave(2, dim=-1))
    return torch.cat(cos_out, dim=-1), torch.cat(sin_out, dim=-1)


class Attention(nn.Module):
    """Self-attention over [text, image]: q/k RMSNorm per head, RoPE, block-causal (see the module docstring)."""

    def __init__(self, dim: int, heads: int, head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.heads, self.head_dim = heads, head_dim
        inner = heads * head_dim
        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_k = nn.Linear(dim, inner, bias=False)
        self.to_v = nn.Linear(dim, inner, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(inner, dim, bias=False)])
        self.norm_q = RMSNorm(head_dim, eps)
        self.norm_k = RMSNorm(head_dim, eps)

    def forward(self, x: torch.Tensor, cos, sin, text_len: int, key_mask: torch.Tensor | None) -> torch.Tensor:
        """key_mask: [B, 1, 1, L] bool of the keys image tokens may see (None: all of them)."""
        q = self.to_q(x).unflatten(-1, (self.heads, self.head_dim))
        k = self.to_k(x).unflatten(-1, (self.heads, self.head_dim))
        v = self.to_v(x).unflatten(-1, (self.heads, self.head_dim))
        q = self.norm_q(q).to(v.dtype)
        k = self.norm_k(k).to(v.dtype)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        tap = getattr(self, "_rz_tap", None)  # rulztrainer.alignment: the trigger's attention (training-only loss)
        if tap is not None:
            tap(q, k)
        # text: causal over the text (padding comes last, so a real token never sees it); image: every real token
        txt = sdpa(q[:, :, :text_len], k[:, :, :text_len], v[:, :, :text_len], is_causal=True)
        img = sdpa(q[:, :, text_len:], k, v, attn_mask=key_mask)
        out = torch.cat([txt, img], dim=2).transpose(1, 2).flatten(2, 3)
        return self.to_out[0](out)


class SwiGLU(nn.Module):
    """out(silu(gate_layer(x)) * proj(x)); ComfyUI's file stores gate_layer and proj fused as gate_up = [gate; proj]."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_layer = nn.Linear(dim, hidden_dim, bias=False)
        self.proj = nn.Linear(dim, hidden_dim, bias=False)
        self.out = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(F.silu(self.gate_layer(x)) * self.proj(x))


def _modulate(h: torch.Tensor, text_len: int, txt_scale, img_scale) -> torch.Tensor:
    """h * (1 + scale), the text tokens with their t = 0 row, the image tokens with their sample's."""
    return torch.cat([h[:, :text_len] * (1 + txt_scale), h[:, text_len:] * (1 + img_scale)], dim=1)


def _gated(y: torch.Tensor, text_len: int, txt_gate, img_gate) -> torch.Tensor:
    return torch.cat([y[:, :text_len] * txt_gate.tanh(), y[:, text_len:] * img_gate.tanh()], dim=1)


class TransformerBlock(nn.Module):
    """Single-stream block; the modulation comes from the model (shared by every block)."""

    def __init__(self, dim: int, heads: int, head_dim: int, mlp_ratio: int = 3, eps: float = 1e-6):
        super().__init__()
        self.img_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.attn = Attention(dim, heads, head_dim, eps)
        self.img_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.img_mlp = SwiGLU(dim, dim * mlp_ratio)

    def forward(self, x, txt_mod, img_mod, cos, sin, text_len: int, key_mask=None):
        """txt_mod [1, 1, 4 * dim] (t = 0), img_mod [B, 1, 4 * dim]: [scale1, gate1, scale2, gate2]."""
        ts1, tg1, ts2, tg2 = txt_mod.chunk(4, dim=-1)
        is1, ig1, is2, ig2 = img_mod.chunk(4, dim=-1)
        h = _modulate(self.img_norm1(x), text_len, ts1, is1)
        x = x + _gated(self.attn(h, cos, sin, text_len, key_mask), text_len, tg1, ig1)
        h = _modulate(self.img_norm2(x), text_len, ts2, is2)
        return x + _gated(self.img_mlp(h), text_len, tg2, ig2)


class TimestepEmbedding(nn.Module):
    """diffusers' TimestepEmbedding(256, dim, sample_proj_bias=False): linear_1 -> SiLU -> linear_2, no biases."""

    def __init__(self, in_channels: int, dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, dim, bias=False)
        self.linear_2 = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_2(F.silu(self.linear_1(x)))


class TimestepProjEmbeddings(nn.Module):
    """Sinusoidal embedding of 1000 * t (cos first, 256 channels), in the timestep's dtype, then the MLP."""

    def __init__(self, dim: int, embed_dim: int = 256):
        super().__init__()
        self.embed_dim = embed_dim
        self.timestep_embedder = TimestepEmbedding(embed_dim, dim)

    def forward(self, t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        half = self.embed_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
        args = (1000.0 * t.float())[:, None] * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1).to(t.dtype)
        return self.timestep_embedder(emb.to(dtype))


class TextProjection(nn.Module):
    def __init__(self, context_dim: int, dim: int, eps: float = 1e-6):
        super().__init__()
        self.text_norm = ZeroCenterRMSNorm(context_dim, eps)
        self.in_layer = nn.Linear(context_dim, dim, bias=False)
        self.out_layer = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_layer(F.gelu(self.in_layer(self.text_norm(x)), approximate="tanh"))


class FinalNorm(nn.Module):
    """Scale-only adaptive LayerNorm (no shift)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim, eps, elementwise_affine=False, bias=False)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        scale = self.linear(F.silu(temb).to(x.dtype))
        return self.norm(x) * (1 + scale[:, None])


class QwenImage21Transformer(nn.Module):
    """QwenImage21Transformer2DModel. hidden_states: [B, 64, H/16, W/16] normalized Qwen-Image 2.1 VAE latents."""

    CONFIG_KEYS = ("in_channels", "out_channels", "num_layers", "attention_head_dim", "num_attention_heads",
                   "context_in_dim", "mlp_ratio", "axes_dims_rope", "eps", "causal_condition")

    def __init__(self, in_channels=64, out_channels=64, num_layers=32, attention_head_dim=128, num_attention_heads=32,
                 context_in_dim=4096, mlp_ratio=3, axes_dims_rope=(16, 56, 56), eps=1e-6, causal_condition=True):
        super().__init__()
        dim = attention_head_dim * num_attention_heads
        if sum(axes_dims_rope) != attention_head_dim:
            raise ValueError(f"sum(axes_dims_rope)={sum(axes_dims_rope)} must equal attention_head_dim")
        out_channels = out_channels or in_channels
        self.config = dict(in_channels=in_channels, out_channels=out_channels, num_layers=num_layers,
                           attention_head_dim=attention_head_dim, num_attention_heads=num_attention_heads,
                           context_in_dim=context_in_dim, mlp_ratio=mlp_ratio, axes_dims_rope=list(axes_dims_rope),
                           eps=eps, causal_condition=causal_condition)
        self.latent_channels = in_channels
        self.text_dim = context_in_dim
        self.axes_dims = tuple(axes_dims_rope)
        self.causal_condition = causal_condition
        self.gradient_checkpointing = False
        self.is_distilled = False  # set by the loader (Qwen-Image-2.1-Turbo: its own fixed 8-step schedule)
        self.time_text_embed = TimestepProjEmbeddings(dim)
        self.txt_in = TextProjection(context_in_dim, dim, eps)
        self.img_in = nn.Linear(in_channels, dim, bias=False)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 4 * dim, bias=False))
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(dim, num_attention_heads, attention_head_dim, mlp_ratio, eps) for _ in range(num_layers)])
        self.norm_out = FinalNorm(dim, eps)
        self.proj_out = nn.Linear(dim, out_channels, bias=False)

    @property
    def layers(self) -> nn.ModuleList:
        """The main blocks under the name the trainer uses (block selection, model split)."""
        return self.transformer_blocks

    def split_tail_modules(self) -> list[nn.Module]:
        """Modules that run after the last block (they follow it in a model split)."""
        return [self.norm_out, self.proj_out]

    @classmethod
    def from_config(cls, config: dict) -> "QwenImage21Transformer":
        return cls(**{k: config[k] for k in cls.CONFIG_KEYS if k in config})

    def _run(self, block, *args):
        thread_prepare()  # dynamo's limits in this thread (autograd recompute, data-parallel workers)
        if (self.gradient_checkpointing and self.training and torch.is_grad_enabled()
                and not getattr(block, "keep_activations", False)):  # see rulztrainer.checkpointing
            return checkpoint(block, *args, use_reentrant=False)
        return block(*args)

    @staticmethod
    def positions(text_lens: torch.Tensor, text_len: int, height: int, width: int) -> torch.Tensor:
        """[B, text_len + H*W, 3] RoPE positions: text token j at (j, j, j); the image at frame = the sample's real
        text length, its rows and columns centred on zero (rows -(H - H//2) .. H//2 - 1)."""
        device = text_lens.device
        batch = text_lens.shape[0]
        txt = torch.arange(text_len, device=device)[None, :, None].expand(batch, text_len, 3)
        rows = torch.arange(-(height - height // 2), height // 2, device=device)
        cols = torch.arange(-(width - width // 2), width // 2, device=device)
        grid = torch.stack(torch.meshgrid(rows, cols, indexing="ij"), dim=-1).reshape(1, height * width, 2)
        frame = text_lens.view(batch, 1, 1).expand(batch, height * width, 1)
        img = torch.cat([frame, grid.expand(batch, -1, -1)], dim=-1)
        return torch.cat([txt, img], dim=1)

    def forward(self, hidden_states: torch.Tensor, timestep, encoder_hidden_states: torch.Tensor,
                encoder_attention_mask: torch.Tensor | None = None, skip_layers=None) -> torch.Tensor:
        """timestep: noise level in 0..1000 (sigma * 1000). Returns the predicted velocity in the input's layout."""
        skip_layers = frozenset(skip_layers or ())
        if hidden_states.ndim != 4 or hidden_states.shape[1] != self.latent_channels:
            raise ValueError(f"hidden_states must be [B, {self.latent_channels}, H/16, W/16]")
        if encoder_hidden_states.ndim != 3 or encoder_hidden_states.shape[2] != self.text_dim:
            raise ValueError(f"encoder_hidden_states must be [B, S, {self.text_dim}]")
        batch, _, height, width = hidden_states.shape
        device = hidden_states.device
        dtype = _wdtype(self.img_in)
        text_len = encoder_hidden_states.shape[1]

        # the pipeline hands the model t / 1000 in bf16; the embedding multiplies by 1000 again in fp32
        sigma = (torch.as_tensor(timestep, device=device, dtype=torch.float32).reshape(-1).expand(batch) / 1000.0)
        sigma = sigma.to(dtype)
        if self.causal_condition:  # one extra row at t = 0 for the text tokens
            sigma = torch.cat([sigma, sigma.new_zeros(1)])
        temb = self.time_text_embed(sigma, dtype)
        mod = self.modulation(temb)
        img_mod = mod[:batch, None]
        txt_mod = mod[-1:, None] if self.causal_condition else img_mod

        if encoder_attention_mask is not None:
            valid = encoder_attention_mask.to(device=device, dtype=torch.bool)
            text_lens = valid.sum(dim=1)
            key_mask = None if bool(valid.all()) else (
                torch.cat([valid, valid.new_ones(batch, height * width)], dim=1)[:, None, None, :])
        else:
            valid, key_mask = None, None
            text_lens = torch.full((batch,), text_len, device=device, dtype=torch.long)
        cos, sin = rope_tables(self.positions(text_lens, text_len, height, width), self.axes_dims)

        enc = self.txt_in(encoder_hidden_states.to(device, dtype))
        x = self.img_in(hidden_states.flatten(2).transpose(1, 2).to(dtype))
        x = torch.cat([enc, x], dim=1)

        split = getattr(self, "split_devices", None)  # blocks may live on several devices (model split)
        taps = getattr(self, "_rz_taps", None)  # rulztrainer.alignment (training-only losses)
        if taps is not None and taps.armed:
            taps.set_layout(0, (text_len, height * width), (0, text_len), valid)
        for i, block in enumerate(self.transformer_blocks):
            if i in skip_layers:
                continue
            if split:
                bdev = _module_device(block)
                if bdev is not None and x.device != bdev:
                    x, txt_mod, img_mod = x.to(bdev), txt_mod.to(bdev), img_mod.to(bdev)
                    cos, sin = cos.to(bdev), sin.to(bdev)
                    key_mask = key_mask.to(bdev) if key_mask is not None else None
            x = self._run(block, x, txt_mod, img_mod, cos, sin, text_len, key_mask)
            if taps is not None and taps.armed:
                taps.record_hidden(i, x[:, text_len:])

        if split:
            fdev = _module_device(self.norm_out)
            if fdev is not None and x.device != fdev:
                x = x.to(fdev)
        out = self.proj_out(self.norm_out(x[:, text_len:], temb[:batch].to(x.device)))
        return out.transpose(1, 2).reshape(batch, -1, height, width).to(device)
