"""Ideogram 4 DiT (Ideogram4Transformer2DModel) for training, in RulzTrainer's conventions.

Ported from diffusers' transformer_ideogram4.py and Ideogram's reference modeling_ideogram4.py (Copyright 2026
Ideogram AI and The HuggingFace Team, Apache-2.0) with the diffusers mixins removed, so that:
  - the state-dict keys are the original checkpoint's (ideogram-ai/ideogram-4-fp8 transformer/, ComfyUI's
    diffusion_models/ideogram4_*): the attention keeps its fused projections, layers.N.attention.qkv ([q; k; v] by
    rows) and layers.N.attention.o, as Ideogram's own code, ComfyUI and ai-toolkit do. A LoRA exported with these
    names (diffusion_model.layers.N.attention.qkv...) is what ComfyUI loads for this model and what diffusers'
    Ideogram 4 loader converts (it splits qkv into to_q / to_k / to_v itself). diffusers' own layout (to_q, to_k,
    to_v, to_out.0) is fused on load (single_file.fuse_diffusers_attention);
  - the forward takes what the trainer and sampler give the other families: `hidden_states` as normalized latents
    [B, 128, H/16, W/16] (Flux2 VAE, 2x2 patches, channels in Ideogram's (patch row, patch column, channel) order, see
    ideogram4/loading.py), `timestep` as sigma * 1000 (1000 = noise) and the text features [B, S, 53248] (13
    Qwen3-VL-8B layers side by side) with a [B, S] mask of the real tokens. It returns the velocity in the trainer's
    convention (noise - clean): Ideogram's model runs on t = 1 - sigma (0 = noise, 1 = clean) and predicts
    clean - noise, so the time is flipped going in and the output negated coming out (ComfyUI and ai-toolkit do the
    same);
  - gradient checkpointing, the model-split device hand-off and the alignment taps work like the other ports;
    `skip_layers` skips blocks (kept for the shared sampler signature);
  - QuantLinear / LoRALinear replace nn.Linear modules transparently (only their dtype handling is consulted).

Architecture: 34 single-stream blocks over one [text, image] sequence. Each block: RMSNorm, scale (1 + s) from a
per-block AdaLN Linear on a shared 512-wide conditioning vector, self-attention with q/k RMSNorm and 3-axis
interleaved MRoPE (theta 5e6, sections 24/20/20 over head_dim 256), a post-norm on the attention output and a
tanh-gated residual; the same around a SwiGLU feed-forward (w2(silu(w1 x) * w3 x)). The text features pass an
RMSNorm over all 53248 channels and one Linear (llm_cond_proj) into the model; image tokens come in through
input_proj; a learned embedding marks image (1) and text (0) tokens. The final layer is a scale-only adaptive
LayerNorm and a Linear.

Attention: every real token of a sample sees every other one (text and image, both ways); padding is masked out
(the reference packs [left padding, text, image] with segment ids). RoPE positions: text token j at (j, j, j)
counted from the sample's own first token; image token (h, w) at (65536, 65536 + h, 65536 + w), independent of the
text length. Here the text is right-padded and masked by key, so a padded batch gives each sample exactly what it
gets alone. A sample without text tokens (S = 0) is the reference's image-only "unconditional" sequence.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..attention import sdpa
from ..compiling import thread_prepare

ROPE_THETA = 5_000_000
IMAGE_POSITION_OFFSET = 65536  # image positions start here, clear of every text position


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
    """diffusers' RMSNorm: normalized in fp32, cast to the weight's dtype, times the weight."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim, self.eps = dim, eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.rms_norm(x.float(), (self.dim,), eps=self.eps)
        if self.weight.dtype in (torch.float16, torch.bfloat16):
            h = h.to(self.weight.dtype)
        return (h * self.weight).to(x.dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def rope_tables(pos: torch.Tensor, head_dim: int, sections=(24, 20, 20), theta: float = ROPE_THETA):
    """cos / sin [B, L, head_dim] for integer positions [B, L, 3] (t, h, w): Qwen3-VL's interleaved MRoPE as the
    reference computes it in fp32 (frequency i belongs to h when i % 3 == 1 and i < 3 * sections[1], to w when
    i % 3 == 2 and i < 3 * sections[2], else to t), the half-split (rotate_half) layout."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=pos.device) / head_dim))
    angles = pos.to(torch.float32)[..., None] * inv_freq  # [B, L, 3, head_dim / 2]
    freqs = angles[..., 0, :].clone()
    for axis in (1, 2):
        idx = torch.arange(axis, sections[axis] * 3, 3, device=pos.device)
        freqs[..., idx] = angles[..., axis, :][..., idx]
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


class Attention(nn.Module):
    """Self-attention over [text, image]: fused qkv, q/k RMSNorm per head, MRoPE, key padding mask."""

    def __init__(self, dim: int, heads: int, eps: float = 1e-5):
        super().__init__()
        self.heads, self.head_dim = heads, dim // heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.norm_q = RMSNorm(self.head_dim, eps)
        self.norm_k = RMSNorm(self.head_dim, eps)
        self.o = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor, cos, sin, key_mask: torch.Tensor | None, tap=None) -> torch.Tensor:
        """key_mask: [B, 1, 1, L] bool of the keys every token may see (None: all of them)."""
        b, n, _ = x.shape
        q, k, v = self.qkv(x).view(b, n, 3, self.heads, self.head_dim).unbind(2)
        q = self.norm_q(q)
        k = self.norm_k(k)
        c, s = cos[:, :, None, :], sin[:, :, None, :]
        q = q * c + _rotate_half(q) * s
        k = k * c + _rotate_half(k) * s
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if tap is not None:  # rulztrainer.alignment: the trigger's attention (training-only loss)
            tap(q, k)
        out = sdpa(q, k, v, attn_mask=key_mask)
        return self.o(out.transpose(1, 2).flatten(2, 3))


class FeedForward(nn.Module):
    """SwiGLU: w2(silu(w1 x) * w3 x)."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, intermediate_size: int, heads: int, norm_eps: float, adaln_dim: int):
        super().__init__()
        self.attention = Attention(dim, heads, eps=1e-5)
        self.feed_forward = FeedForward(dim, intermediate_size)
        self.attention_norm1 = RMSNorm(dim, norm_eps)
        self.ffn_norm1 = RMSNorm(dim, norm_eps)
        self.attention_norm2 = RMSNorm(dim, norm_eps)
        self.ffn_norm2 = RMSNorm(dim, norm_eps)
        self.adaln_modulation = nn.Linear(adaln_dim, 4 * dim, bias=True)

    def forward(self, x, adaln_input, cos, sin, key_mask=None):
        """adaln_input [B, 1, adaln_dim]: [scale_msa, gate_msa, scale_mlp, gate_mlp] come from it."""
        scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaln_modulation(adaln_input).chunk(4, dim=-1)
        attn = self.attention(self.attention_norm1(x) * (1.0 + scale_msa), cos, sin, key_mask,
                              getattr(self, "_rz_tap", None))
        x = x + torch.tanh(gate_msa) * self.attention_norm2(attn)
        ff = self.feed_forward(self.ffn_norm1(x) * (1.0 + scale_mlp))
        return x + torch.tanh(gate_mlp) * self.ffn_norm2(ff)


def sinusoidal_embedding(t: torch.Tensor, dim: int, scale: float = 1e4) -> torch.Tensor:
    """The reference's embedding: sin first, then cos, frequencies exp(-i log(scale) / (half - 1))."""
    t = t.to(torch.float32)
    half = dim // 2
    freq = torch.exp(torch.arange(half, dtype=torch.float32, device=t.device) * -(math.log(scale) / (half - 1)))
    emb = t.unsqueeze(-1) * freq
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    return F.pad(emb, (0, 1)) if dim % 2 == 1 else emb


class EmbedScalar(nn.Module):
    """The timestep embedding: sinusoidal of 1e4 * t (t in 0..1), then Linear -> SiLU -> Linear."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp_in = nn.Linear(dim, dim, bias=True)
        self.mlp_out = nn.Linear(dim, dim, bias=True)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        emb = sinusoidal_embedding(1e4 * t.to(torch.float32), self.dim).to(t.dtype)
        return self.mlp_out(F.silu(self.mlp_in(emb)))


class FinalLayer(nn.Module):
    def __init__(self, dim: int, out_channels: int, adaln_dim: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        self.linear = nn.Linear(dim, out_channels, bias=True)
        self.adaln_modulation = nn.Linear(adaln_dim, dim, bias=True)

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        scale = 1.0 + self.adaln_modulation(F.silu(conditioning))
        return self.linear(self.norm_final(x) * scale)


class Ideogram4Transformer(nn.Module):
    """Ideogram4Transformer2DModel. hidden_states: [B, 128, H/16, W/16] normalized Ideogram 4 latents."""

    CONFIG_KEYS = ("in_channels", "num_layers", "attention_head_dim", "num_attention_heads", "intermediate_size",
                   "adaln_dim", "llm_features_dim", "rope_theta", "mrope_section", "norm_eps")

    def __init__(self, in_channels=128, num_layers=34, attention_head_dim=256, num_attention_heads=18,
                 intermediate_size=12288, adaln_dim=512, llm_features_dim=53248, rope_theta=ROPE_THETA,
                 mrope_section=(24, 20, 20), norm_eps=1e-5):
        super().__init__()
        dim = attention_head_dim * num_attention_heads
        self.config = dict(in_channels=in_channels, num_layers=num_layers, attention_head_dim=attention_head_dim,
                           num_attention_heads=num_attention_heads, intermediate_size=intermediate_size,
                           adaln_dim=adaln_dim, llm_features_dim=llm_features_dim, rope_theta=rope_theta,
                           mrope_section=list(mrope_section), norm_eps=norm_eps)
        self.latent_channels = in_channels
        self.text_dim = llm_features_dim
        self.hidden_size = dim
        self.head_dim = attention_head_dim
        self.rope_theta = rope_theta
        self.mrope_section = tuple(mrope_section)
        self.gradient_checkpointing = False
        self.is_distilled = False  # no distilled Ideogram 4 checkpoint exists (the turbo LoRA marks its own schedule)
        self.input_proj = nn.Linear(in_channels, dim, bias=True)
        self.llm_cond_norm = RMSNorm(llm_features_dim, eps=1e-6)
        self.llm_cond_proj = nn.Linear(llm_features_dim, dim, bias=True)
        self.t_embedding = EmbedScalar(dim)
        self.adaln_proj = nn.Linear(dim, adaln_dim, bias=True)
        self.embed_image_indicator = nn.Embedding(2, dim)
        self.layers = nn.ModuleList([TransformerBlock(dim, intermediate_size, num_attention_heads, norm_eps, adaln_dim)
                                     for _ in range(num_layers)])
        self.final_layer = FinalLayer(dim, in_channels, adaln_dim)

    def split_tail_modules(self) -> list[nn.Module]:
        """Modules that run after the last block (they follow it in a model split)."""
        return [self.final_layer]

    @classmethod
    def from_config(cls, config: dict) -> "Ideogram4Transformer":
        return cls(**{k: config[k] for k in cls.CONFIG_KEYS if k in config})

    def _run(self, block, *args):
        thread_prepare()  # dynamo's limits in this thread (autograd recompute, data-parallel workers)
        if (self.gradient_checkpointing and self.training and torch.is_grad_enabled()
                and not getattr(block, "keep_activations", False)):  # see rulztrainer.checkpointing
            return checkpoint(block, *args, use_reentrant=False)
        return block(*args)

    @staticmethod
    def positions(text_len: int, height: int, width: int, device) -> torch.Tensor:
        """[1, text_len + H*W, 3] RoPE positions, the same for every sample: text token j at (j, j, j) (padding
        after a sample's text is masked, so its positions never matter), image cell (h, w) at
        (65536, 65536 + h, 65536 + w)."""
        txt = torch.arange(text_len, device=device)[:, None].expand(text_len, 3)
        rows = torch.arange(height, device=device)[:, None].expand(height, width).reshape(-1)
        cols = torch.arange(width, device=device)[None, :].expand(height, width).reshape(-1)
        img = torch.stack([torch.zeros_like(rows), rows, cols], dim=-1) + IMAGE_POSITION_OFFSET
        return torch.cat([txt, img], dim=0)[None]

    def forward(self, hidden_states: torch.Tensor, timestep, encoder_hidden_states: torch.Tensor,
                encoder_attention_mask: torch.Tensor | None = None, skip_layers=None) -> torch.Tensor:
        """timestep: noise level in 0..1000 (sigma * 1000). Returns the velocity (noise - clean) in the input's layout."""
        skip_layers = frozenset(skip_layers or ())
        if hidden_states.ndim != 4 or hidden_states.shape[1] != self.latent_channels:
            raise ValueError(f"hidden_states must be [B, {self.latent_channels}, H/16, W/16]")
        if encoder_hidden_states.ndim != 3 or encoder_hidden_states.shape[2] != self.text_dim:
            raise ValueError(f"encoder_hidden_states must be [B, S, {self.text_dim}]")
        batch, _, height, width = hidden_states.shape
        device = hidden_states.device
        dtype = _wdtype(self.input_proj)
        text_len = encoder_hidden_states.shape[1]

        # the model's own time runs the other way (0 = noise, 1 = clean); the pipelines hand it in the model's dtype
        sigma = torch.as_tensor(timestep, device=device, dtype=torch.float32).reshape(-1).expand(batch) / 1000.0
        t = (1.0 - sigma).to(dtype)
        adaln_input = F.silu(self.adaln_proj(self.t_embedding(t)))[:, None]  # [B, 1, adaln_dim]

        if encoder_attention_mask is not None and text_len:
            valid = encoder_attention_mask.to(device=device, dtype=torch.bool)
            key_mask = None if bool(valid.all()) else (
                torch.cat([valid, valid.new_ones(batch, height * width)], dim=1)[:, None, None, :])
        else:
            valid, key_mask = None, None
        cos, sin = rope_tables(self.positions(text_len, height, width, device), self.head_dim, self.mrope_section,
                               self.rope_theta)
        cos, sin = cos.to(dtype), sin.to(dtype)

        indicator = self.embed_image_indicator.weight.to(device, dtype)
        img = self.input_proj(hidden_states.flatten(2).transpose(1, 2).to(dtype)) + indicator[1]
        if text_len:
            enc = self.llm_cond_proj(self.llm_cond_norm(encoder_hidden_states.to(device, dtype))) + indicator[0]
            x = torch.cat([enc, img], dim=1)
        else:
            x = img

        split = getattr(self, "split_devices", None)  # blocks may live on several devices (model split)
        taps = getattr(self, "_rz_taps", None)  # rulztrainer.alignment (training-only losses)
        if taps is not None and taps.armed:
            taps.set_layout(0, (text_len, height * width), (0, text_len), valid)
        for i, block in enumerate(self.layers):
            if i in skip_layers:
                continue
            if split:
                bdev = _module_device(block)
                if bdev is not None and x.device != bdev:
                    x, adaln_input = x.to(bdev), adaln_input.to(bdev)
                    cos, sin = cos.to(bdev), sin.to(bdev)
                    key_mask = key_mask.to(bdev) if key_mask is not None else None
            x = self._run(block, x, adaln_input, cos, sin, key_mask)
            if taps is not None and taps.armed:
                taps.record_hidden(i, x[:, text_len:])

        if split:
            fdev = _module_device(self.final_layer)
            if fdev is not None and x.device != fdev:
                x, adaln_input = x.to(fdev), adaln_input.to(fdev)
        out = self.final_layer(x[:, text_len:], adaln_input)  # clean - noise, Ideogram's convention
        return (-out).transpose(1, 2).reshape(batch, -1, height, width).to(device)
