"""Krea 2 DiT (Krea2Transformer2DModel) for training, in RulzTrainer's conventions.

Ported from diffusers' transformer_krea2.py (Copyright 2026 Krea AI and The HuggingFace Team, Apache-2.0) with the
diffusers mixins removed, so that:
  - the state-dict keys are exactly the Hugging Face checkpoint's (krea/Krea-2-Raw transformer/), and a LoRA exported
    with the same module names loads into diffusers' model and matches Krea's own published LoRAs;
  - the forward takes what the trainer and sampler give the other families: `hidden_states` as the normalized
    Qwen-Image VAE latents in their 2x2-patchified form [B, 64, H/16, W/16] (the dataset cache), `timestep` in
    0..1000 units, and the text features as [B, S, 12, 2560] (twelve Qwen3-VL hidden layers per token, see
    krea2/loading.py) with a [B, S] mask. The 64 patch features are ordered (channel, dy, dx), as the pipeline packs
    them, so the image tokens are the same tensor;
  - gradient checkpointing (torch.utils.checkpoint) and the model-split device hand-off work like the MOSS model;
    `skip_layers` skips main blocks (kept for the shared sampler signature);
  - QuantLinear / LoRALinear replace nn.Linear modules transparently (only their dtype handling is consulted).

Architecture: a text-fusion stage (2 blocks attending across the 12 tapped encoder layers of each token, a linear
projector collapsing that axis, 2 refiner blocks across the tokens), text and image projections, 28 single-stream
blocks over [text, image] with grouped-query attention, q/k RMSNorm, 3-axis RoPE (text tokens at the origin) and a
sigmoid output gate, one timestep modulation shared by every block plus a learned table per block, and a final
adaptive norm + projection on the image tokens. All RMSNorms scale by (1 + weight) and run in fp32.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

NORM_NAMES = ("norm", "norm1", "norm2", "norm_q", "norm_k")  # modules kept in fp32 (diffusers' _keep_in_fp32_modules)


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
    """Zero-centred RMSNorm: the multiplier is 1 + weight, computed in fp32, result in the input's dtype."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim, self.eps = dim, eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x.float(), (self.dim,), weight=self.weight.float() + 1.0, eps=self.eps).to(x.dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x [B, L, H, D]; cos/sin [L, D] with interleaved (real, imaginary) pairs (diffusers' apply_rotary_emb)."""
    x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
    rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
    cos, sin = cos[None, :, None, :], sin[None, :, None, :]
    return (x.float() * cos + rotated.float() * sin).to(x.dtype)


def rope_tables(ids: torch.Tensor, axes_dims, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin [L, sum(axes_dims)] for position ids [L, 3] (diffusers' get_1d_rotary_pos_embed per axis)."""
    dtype = torch.float32 if ids.device.type == "mps" else torch.float64
    cos_out, sin_out = [], []
    for i, dim in enumerate(axes_dims):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=dtype, device=ids.device)[: dim // 2] / dim))
        angles = torch.outer(ids[:, i].to(dtype), freqs)
        cos_out.append(angles.cos().repeat_interleave(2, dim=1).float())
        sin_out.append(angles.sin().repeat_interleave(2, dim=1).float())
    return torch.cat(cos_out, dim=-1), torch.cat(sin_out, dim=-1)


class Attention(nn.Module):
    """Self-attention: grouped-query projections, q/k RMSNorm, optional RoPE, sigmoid output gate."""

    def __init__(self, dim: int, heads: int, kv_heads: int, eps: float = 1e-5):
        super().__init__()
        self.heads, self.kv_heads, self.head_dim = heads, kv_heads, dim // heads
        self.to_q = nn.Linear(dim, self.head_dim * heads, bias=False)
        self.to_k = nn.Linear(dim, self.head_dim * kv_heads, bias=False)
        self.to_v = nn.Linear(dim, self.head_dim * kv_heads, bias=False)
        self.to_gate = nn.Linear(dim, dim, bias=False)
        self.norm_q = RMSNorm(self.head_dim, eps)
        self.norm_k = RMSNorm(self.head_dim, eps)
        self.to_out = nn.ModuleList([nn.Linear(dim, dim, bias=False)])

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, rope=None) -> torch.Tensor:
        q = self.norm_q(self.to_q(x).unflatten(-1, (self.heads, self.head_dim)))
        k = self.norm_k(self.to_k(x).unflatten(-1, (self.kv_heads, self.head_dim)))
        v = self.to_v(x).unflatten(-1, (self.kv_heads, self.head_dim))
        gate = self.to_gate(x)
        if rope is not None:
            q, k = apply_rope(q, *rope), apply_rope(k, *rope)
        if self.heads != self.kv_heads:  # repeated heads: every SDPA kernel accepts a mask then (see diffusers)
            k = k.repeat_interleave(self.heads // self.kv_heads, dim=2)
            v = v.repeat_interleave(self.heads // self.kv_heads, dim=2)
        out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask)
        out = out.transpose(1, 2).flatten(2, 3) * torch.sigmoid(gate)
        return self.to_out[0](out)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate = nn.Linear(dim, hidden_dim, bias=False)
        self.up = nn.Linear(dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class TextFusionBlock(nn.Module):
    """Pre-norm block without RoPE or time modulation (the text fusion stage)."""

    def __init__(self, dim: int, heads: int, kv_heads: int, intermediate_size: int, eps: float):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(dim, eps), RMSNorm(dim, eps)
        self.attn = Attention(dim, heads, kv_heads, eps)
        self.ff = SwiGLU(dim, intermediate_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask)
        return x + self.ff(self.norm2(x))


class TextFusion(nn.Module):
    """[B, S, layers, dim] tapped encoder states -> [B, S, dim]: attention across the layer axis of each token, a
    linear projector collapsing it, then attention across the tokens."""

    def __init__(self, num_text_layers, dim, heads, kv_heads, intermediate_size, num_layerwise, num_refiner, eps):
        super().__init__()
        self.layerwise_blocks = nn.ModuleList(
            [TextFusionBlock(dim, heads, kv_heads, intermediate_size, eps) for _ in range(num_layerwise)])
        self.projector = nn.Linear(num_text_layers, 1, bias=False)
        self.refiner_blocks = nn.ModuleList(
            [TextFusionBlock(dim, heads, kv_heads, intermediate_size, eps) for _ in range(num_refiner)])

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, run=None) -> torch.Tensor:
        """run(block, *args): the transformer's block runner, so these blocks are checkpointed like the main ones
        (with a LoRA here they would otherwise keep every activation: several MB per text token)."""
        run = run or (lambda block, *args: block(*args))
        batch, seq, layers, dim = x.shape
        h = x.reshape(batch * seq, layers, dim)
        for block in self.layerwise_blocks:
            h = run(block, h.contiguous())
        h = self.projector(h.reshape(batch, seq, layers, dim).permute(0, 1, 3, 2)).squeeze(-1)
        for block in self.refiner_blocks:
            h = run(block, h, mask)
        return h


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, intermediate_size: int, heads: int, kv_heads: int, eps: float):
        super().__init__()
        self.scale_shift_table = nn.Parameter(torch.zeros(6, dim))
        self.norm1, self.norm2 = RMSNorm(dim, eps), RMSNorm(dim, eps)
        self.attn = Attention(dim, heads, kv_heads, eps)
        self.ff = SwiGLU(dim, intermediate_size)

    def forward(self, x, temb_mod, cos, sin, mask=None):
        # temb_mod [B, 1, 6 * dim] is shared by every block; each block adds its own table
        prescale, preshift, pregate, postscale, postshift, postgate = (
            temb_mod.unflatten(-1, (6, -1)) + self.scale_shift_table).unbind(-2)
        x = x + pregate * self.attn((1.0 + prescale) * self.norm1(x) + preshift, mask, (cos, sin))
        return x + postgate * self.ff((1.0 + postscale) * self.norm2(x) + postshift)


class TimestepEmbedding(nn.Module):
    """Sinusoidal embedding of the flow time (cos first, input x 1000), then a two-layer MLP; output [B, 1, dim]."""

    def __init__(self, embed_dim: int, dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.linear_1 = nn.Linear(embed_dim, dim, bias=True)
        self.linear_2 = nn.Linear(dim, dim, bias=True)

    def forward(self, t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        half = self.embed_dim // 2
        freqs = torch.exp(-math.log(1e4) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
        args = (t.float() * 1e3)[:, None, None] * freqs
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1).to(dtype)
        return self.linear_2(F.gelu(self.linear_1(emb), approximate="tanh"))


class TextProjection(nn.Module):
    def __init__(self, text_dim: int, dim: int, eps: float):
        super().__init__()
        self.norm = RMSNorm(text_dim, eps)
        self.linear_1 = nn.Linear(text_dim, dim, bias=True)
        self.linear_2 = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_2(F.gelu(self.linear_1(self.norm(x)), approximate="tanh"))


class FinalLayer(nn.Module):
    def __init__(self, dim: int, out_channels: int, eps: float):
        super().__init__()
        self.scale_shift_table = nn.Parameter(torch.zeros(2, dim))
        self.norm = RMSNorm(dim, eps)
        self.linear = nn.Linear(dim, out_channels, bias=True)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        scale, shift = (temb + self.scale_shift_table).chunk(2, dim=1)
        return self.linear((1.0 + scale) * self.norm(x) + shift)


class Krea2Transformer(nn.Module):
    """Krea2Transformer2DModel. hidden_states: [B, 64, H/16, W/16] normalized, patchified Qwen-Image VAE latents."""

    CONFIG_KEYS = ("in_channels", "num_layers", "attention_head_dim", "num_attention_heads", "num_key_value_heads",
                   "intermediate_size", "timestep_embed_dim", "text_hidden_dim", "num_text_layers",
                   "text_num_attention_heads", "text_num_key_value_heads", "text_intermediate_size",
                   "num_layerwise_text_blocks", "num_refiner_text_blocks", "axes_dims_rope", "rope_theta", "norm_eps")

    def __init__(self, in_channels=64, num_layers=28, attention_head_dim=128, num_attention_heads=48,
                 num_key_value_heads=12, intermediate_size=16384, timestep_embed_dim=256, text_hidden_dim=2560,
                 num_text_layers=12, text_num_attention_heads=20, text_num_key_value_heads=20,
                 text_intermediate_size=6912, num_layerwise_text_blocks=2, num_refiner_text_blocks=2,
                 axes_dims_rope=(32, 48, 48), rope_theta=1000.0, norm_eps=1e-5):
        super().__init__()
        dim = attention_head_dim * num_attention_heads
        if sum(axes_dims_rope) != attention_head_dim:
            raise ValueError(f"sum(axes_dims_rope)={sum(axes_dims_rope)} must equal attention_head_dim")
        self.config = dict(in_channels=in_channels, num_layers=num_layers, attention_head_dim=attention_head_dim,
                           num_attention_heads=num_attention_heads, num_key_value_heads=num_key_value_heads,
                           intermediate_size=intermediate_size, timestep_embed_dim=timestep_embed_dim,
                           text_hidden_dim=text_hidden_dim, num_text_layers=num_text_layers,
                           text_num_attention_heads=text_num_attention_heads,
                           text_num_key_value_heads=text_num_key_value_heads,
                           text_intermediate_size=text_intermediate_size,
                           num_layerwise_text_blocks=num_layerwise_text_blocks,
                           num_refiner_text_blocks=num_refiner_text_blocks, axes_dims_rope=list(axes_dims_rope),
                           rope_theta=rope_theta, norm_eps=norm_eps)
        self.latent_channels = in_channels  # 64: 16 VAE channels x 2x2, the trainer's (patchified) layout
        self.num_text_layers, self.text_dim = num_text_layers, text_hidden_dim
        self.axes_dims, self.rope_theta = tuple(axes_dims_rope), rope_theta
        self.gradient_checkpointing = False
        self.is_distilled = False  # set by the loader from model_index.json (the turbo checkpoint's fixed shift)
        self.img_in = nn.Linear(in_channels, dim, bias=True)
        self.time_embed = TimestepEmbedding(timestep_embed_dim, dim)
        self.time_mod_proj = nn.Linear(dim, 6 * dim, bias=True)
        self.text_fusion = TextFusion(num_text_layers, text_hidden_dim, text_num_attention_heads,
                                      text_num_key_value_heads, text_intermediate_size, num_layerwise_text_blocks,
                                      num_refiner_text_blocks, norm_eps)
        self.txt_in = TextProjection(text_hidden_dim, dim, norm_eps)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(dim, intermediate_size, num_attention_heads, num_key_value_heads, norm_eps)
            for _ in range(num_layers)])
        self.final_layer = FinalLayer(dim, in_channels, norm_eps)

    @property
    def layers(self) -> nn.ModuleList:
        """The main blocks under the name the trainer uses (block selection, model split)."""
        return self.transformer_blocks

    def split_tail_modules(self) -> list[nn.Module]:
        """Modules that run after the last block (they follow it in a model split)."""
        return [self.final_layer]

    @classmethod
    def from_config(cls, config: dict) -> "Krea2Transformer":
        return cls(**{k: config[k] for k in cls.CONFIG_KEYS if k in config})

    def _run(self, block, *args):
        if (self.gradient_checkpointing and self.training and torch.is_grad_enabled()
                and not getattr(block, "keep_activations", False)):  # see rulztrainer.checkpointing
            return checkpoint(block, *args, use_reentrant=False)
        return block(*args)

    @staticmethod
    def position_ids(text_len: int, height: int, width: int, device) -> torch.Tensor:
        """[text_len + H*W, 3] RoPE coordinates: text tokens at the origin, image tokens at (0, h, w)."""
        image = torch.zeros(height, width, 3, device=device)
        image[..., 1] = torch.arange(height, device=device)[:, None]
        image[..., 2] = torch.arange(width, device=device)[None, :]
        return torch.cat([torch.zeros(text_len, 3, device=device), image.reshape(-1, 3)])

    def forward(self, hidden_states: torch.Tensor, timestep, encoder_hidden_states: torch.Tensor,
                encoder_attention_mask: torch.Tensor | None = None, skip_layers=None) -> torch.Tensor:
        """timestep: noise level in 0..1000 (sigma * 1000). Returns the predicted velocity in the input's layout."""
        skip_layers = frozenset(skip_layers or ())
        if hidden_states.ndim != 4 or hidden_states.shape[1] != self.latent_channels:
            raise ValueError(f"hidden_states must be [B, {self.latent_channels}, H/16, W/16]")
        if encoder_hidden_states.ndim != 4 or encoder_hidden_states.shape[2:] != (self.num_text_layers, self.text_dim):
            raise ValueError(f"encoder_hidden_states must be [B, S, {self.num_text_layers}, {self.text_dim}]")
        batch, _, height, width = hidden_states.shape
        device = hidden_states.device
        dtype = _wdtype(self.img_in)
        text_len = encoder_hidden_states.shape[1]

        # the pipeline passes t / 1000 in the latents' dtype; the embedding multiplies it by 1000 again in fp32
        sigma = (torch.as_tensor(timestep, device=device, dtype=torch.float32).reshape(-1).expand(batch) / 1000.0)
        temb = self.time_embed(sigma.to(dtype), dtype)
        temb_mod = self.time_mod_proj(F.gelu(temb, approximate="tanh"))

        mask = text_mask = None
        if encoder_attention_mask is not None and not bool(encoder_attention_mask.all()):
            # key padding: padded text tokens are never attended to (with nothing padded, no mask at all)
            m = encoder_attention_mask.to(device=device, dtype=torch.bool)
            text_mask = m[:, None, None, :]
            mask = torch.cat([m, m.new_ones(batch, height * width)], dim=1)[:, None, None, :]
        enc = self.txt_in(self.text_fusion(encoder_hidden_states.to(device, dtype), text_mask, run=self._run))
        x = self.img_in(hidden_states.flatten(2).transpose(1, 2).to(dtype))
        x = torch.cat([enc, x], dim=1)
        cos, sin = rope_tables(self.position_ids(text_len, height, width, device), self.axes_dims, self.rope_theta)

        split = getattr(self, "split_devices", None)  # blocks may live on several devices (model split)
        for i, block in enumerate(self.transformer_blocks):
            if i in skip_layers:
                continue
            if split:
                bdev = _module_device(block)
                if bdev is not None and x.device != bdev:
                    x, temb_mod, cos, sin = x.to(bdev), temb_mod.to(bdev), cos.to(bdev), sin.to(bdev)
                    mask = mask.to(bdev) if mask is not None else None
            x = self._run(block, x, temb_mod, cos, sin, mask)

        if split:
            fdev = _module_device(self.final_layer)
            if fdev is not None and x.device != fdev:
                x = x.to(fdev)
        out = self.final_layer(x[:, text_len:], temb.to(x.device))
        return out.transpose(1, 2).reshape(batch, self.latent_channels, height, width).to(device)
