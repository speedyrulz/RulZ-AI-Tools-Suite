"""Bria FIBO 2.0 DiT (BriaFibo2Transformer2DModel) for training, in RulzTrainer's conventions.

Ported from diffusers' transformer_bria_fibo2.py (Copyright 2026 Bria AI and The HuggingFace Team, Apache-2.0)
with the diffusers mixins removed, so that:
  - the state-dict keys are exactly the Hugging Face checkpoint's (briaai/Fibo-2.0-dev transformer/), and a LoRA
    exported with the same module names loads into diffusers' model;
  - the forward takes what the trainer and sampler give MOSS-Image: `hidden_states` as the normalized Flux2 latents
    in their 2x2-patchified form [B, 128, H/16, W/16] (the dataset cache), `timestep` in 0..1000 units, and the text
    features as [B, 6, S, 7680] (the six Qwen3-VL bundles, see fibo/loading.py). The 128 patch features are ordered
    (channel, dy, dx) in both the Flux2 cache and the model's own patchify, so the tokens are the same tensor;
  - gradient checkpointing (torch.utils.checkpoint) and the model-split device hand-off work like the MOSS model;
    `skip_layers` skips main blocks (unused by FIBO's sampler, kept for the shared sampler signature);
  - QuantLinear / LoRALinear replace nn.Linear modules transparently (only their dtype handling is consulted).

Architecture: x_embedder, timestep MLP, 2 noise-refiner blocks over the image tokens, a 2-layer Perceiver resampler
turning the first text bundle into 256-512 "gist" tokens, 44 single-stream blocks of which 5 (injection blocks) also
cross-attend to their own text bundle, and a final adaLN layer on the image tokens. Context (edit) images are extra
clean tokens on their own RoPE planes.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

ADALN_EMBED_DIM = 256


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
    """diffusers.models.normalization.RMSNorm with elementwise affine (fp32 statistics, result in the weight's dtype)."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        var = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        if self.weight.dtype in (torch.float16, torch.bfloat16):
            x = x.to(self.weight.dtype)
        x = x * self.weight
        return x.to(in_dtype) if self.weight.dtype == torch.float32 else x


class TimestepEmbedder(nn.Module):
    def __init__(self, out_size: int, mid_size: int | None = None, frequency_embedding_size: int = 256):
        super().__init__()
        mid_size = mid_size or out_size
        self.mlp = nn.Sequential(nn.Linear(frequency_embedding_size, mid_size, bias=True), nn.SiLU(),
                                 nn.Linear(mid_size, out_size, bias=True))
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq.to(_wdtype(self.mlp[0])))


def select_per_token(value_noisy: torch.Tensor, value_clean: torch.Tensor, noise_mask: torch.Tensor,
                     seq_len: int) -> torch.Tensor:
    mask = noise_mask.unsqueeze(-1)
    return torch.where(mask == 1, value_noisy.unsqueeze(1).expand(-1, seq_len, -1),
                       value_clean.unsqueeze(1).expand(-1, seq_len, -1))


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(min(hidden_size, ADALN_EMBED_DIM), hidden_size,
                                                                    bias=True))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        scale = (1.0 + self.adaLN_modulation(c)).unsqueeze(1)
        x = self.norm_final(x.float()).to(scale.dtype) * scale
        return self.linear(x.to(_wdtype(self.linear)))


class RopeEmbedder:
    def __init__(self, theta: float = 256.0, axes_dims=(32, 48, 48), axes_lens=(2048, 512, 512)):
        self.theta, self.axes_dims, self.axes_lens = theta, list(axes_dims), list(axes_lens)
        self.freqs_cis: list[torch.Tensor] | None = None

    def _precompute(self) -> list[torch.Tensor]:
        out = []
        for d, e in zip(self.axes_dims, self.axes_lens):
            freqs = 1.0 / (self.theta ** (torch.arange(0, d, 2, dtype=torch.float64) / d))
            pos = torch.arange(e, dtype=torch.float64)
            freqs = torch.outer(pos, freqs).float()
            out.append(torch.polar(torch.ones_like(freqs), freqs).to(torch.complex64))
        return out

    def __call__(self, ids: torch.Tensor) -> torch.Tensor:
        if self.freqs_cis is None or self.freqs_cis[0].device != ids.device:
            self.freqs_cis = [f.to(ids.device) for f in (self.freqs_cis or self._precompute())]
        return torch.cat([self.freqs_cis[i][ids[:, i]] for i in range(len(self.axes_dims))], dim=-1)


def compute_gist_count(text_lengths: list[int], min_gist: int, max_gist: int, gist_step: int, min_text_len: int,
                       max_text_len: int) -> list[int]:
    span = max_text_len - min_text_len
    counts = []
    for n in text_lengths:
        ratio = min(1.0, max(0.0, (float(n) - min_text_len) / span))
        raw = min_gist + ratio * (max_gist - min_gist)
        quantized = int(math.ceil(raw / gist_step)) * gist_step
        counts.append(max(min_gist, min(max_gist, quantized)))
    return counts


def _apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(xc * freqs_cis.unsqueeze(2)).flatten(3).type_as(x)


class Attention(nn.Module):
    """Self-attention (with RoPE when `freqs_cis` is given) or cross-attention to `encoder_hidden_states`."""

    def __init__(self, query_dim: int, heads: int, dim_head: int, kv_dim: int | None = None, eps: float = 1e-5):
        super().__init__()
        self.heads = heads
        kv_dim = kv_dim or query_dim
        self.to_q = nn.Linear(query_dim, heads * dim_head, bias=False)
        self.to_k = nn.Linear(kv_dim, heads * dim_head, bias=False)
        self.to_v = nn.Linear(kv_dim, heads * dim_head, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(heads * dim_head, query_dim, bias=False)])
        self.norm_q = RMSNorm(dim_head, eps=eps)
        self.norm_k = RMSNorm(dim_head, eps=eps)

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor | None = None,
                attention_mask: torch.Tensor | None = None, freqs_cis: torch.Tensor | None = None) -> torch.Tensor:
        q = self.to_q(hidden_states)
        kv_in = hidden_states if encoder_hidden_states is None else encoder_hidden_states.to(hidden_states.dtype)
        k, v = self.to_k(kv_in), self.to_v(kv_in)
        q = self.norm_q(q.unflatten(-1, (self.heads, -1)))
        k = self.norm_k(k.unflatten(-1, (self.heads, -1)))
        v = v.unflatten(-1, (self.heads, -1))
        if freqs_cis is not None:
            q, k = _apply_rope(q, freqs_cis), _apply_rope(k, freqs_cis)
        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = attention_mask[:, None, None, :]
        out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                             attn_mask=attention_mask)
        return self.to_out[0](out.transpose(1, 2).flatten(2))


class TransformerBlock(nn.Module):
    """Z-Image's block; with `cross_attention_dim` an injection block that also reads a text bundle."""

    def __init__(self, dim: int, n_heads: int, norm_eps: float, cross_attention_dim: int | None = None):
        super().__init__()
        self.attention = Attention(dim, n_heads, dim // n_heads, eps=norm_eps)
        self.feed_forward = FeedForward(dim, (dim * 8 // 3) // 64 * 64)
        self.attention_norm1 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = RMSNorm(dim, eps=norm_eps)
        self.attention_norm2 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = RMSNorm(dim, eps=norm_eps)
        self.adaLN_modulation = nn.Sequential(nn.Linear(min(dim, ADALN_EMBED_DIM), 4 * dim, bias=True))
        if cross_attention_dim is not None:
            self.cross_attn_norm1 = RMSNorm(dim, eps=norm_eps)
            self.cross_attention = Attention(dim, n_heads, dim // n_heads, kv_dim=cross_attention_dim, eps=norm_eps)
            self.cross_attn_norm2 = RMSNorm(dim, eps=norm_eps)
            self.cross_attn_gate = nn.Parameter(torch.zeros(1))
        else:
            self.cross_attention = None

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None, freqs_cis: torch.Tensor,
                noise_mask: torch.Tensor | None, adaln_noisy: torch.Tensor, adaln_clean: torch.Tensor,
                encoder_hidden_states: torch.Tensor | None = None,
                encoder_attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        seq_len = x.shape[1]
        if noise_mask is not None:
            mod_noisy = self.adaLN_modulation(adaln_noisy)
            mod_clean = self.adaLN_modulation(adaln_clean)
            s_msa_n, g_msa_n, s_mlp_n, g_mlp_n = mod_noisy.chunk(4, dim=1)
            s_msa_c, g_msa_c, s_mlp_c, g_mlp_c = mod_clean.chunk(4, dim=1)
            scale_msa = select_per_token(1.0 + s_msa_n, 1.0 + s_msa_c, noise_mask, seq_len)
            scale_mlp = select_per_token(1.0 + s_mlp_n, 1.0 + s_mlp_c, noise_mask, seq_len)
            gate_msa = select_per_token(g_msa_n.tanh(), g_msa_c.tanh(), noise_mask, seq_len)
            gate_mlp = select_per_token(g_mlp_n.tanh(), g_mlp_c.tanh(), noise_mask, seq_len)
        else:
            mod = self.adaLN_modulation(adaln_noisy)
            scale_msa, gate_msa, scale_mlp, gate_mlp = mod.unsqueeze(1).chunk(4, dim=2)
            gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
            scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp
        attn_out = self.attention(self.attention_norm1(x) * scale_msa, attention_mask=attn_mask, freqs_cis=freqs_cis)
        x = x + gate_msa * self.attention_norm2(attn_out)
        if self.cross_attention is not None:
            cross_out = self.cross_attention(self.cross_attn_norm1(x), encoder_hidden_states=encoder_hidden_states,
                                             attention_mask=encoder_attention_mask)
            x = x + self.cross_attn_gate.tanh().to(x.dtype) * self.cross_attn_norm2(cross_out)
        return x + gate_mlp * self.ffn_norm2(self.feed_forward(self.ffn_norm1(x) * scale_mlp))


class PerceiverLayer(nn.Module):
    def __init__(self, dim: int, n_heads: int, cross_attention_dim: int, norm_eps: float):
        super().__init__()
        self.cross_attn_norm = RMSNorm(dim, eps=norm_eps)
        self.cross_attention = Attention(dim, n_heads, dim // n_heads, kv_dim=cross_attention_dim, eps=norm_eps)
        self.self_attn_norm = RMSNorm(dim, eps=norm_eps)
        self.self_attention = Attention(dim, n_heads, dim // n_heads, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn = FeedForward(dim, (dim * 8 // 3) // 64 * 64)

    def forward(self, latents, encoder_hidden_states, encoder_attention_mask=None, attention_mask=None):
        latents = latents + self.cross_attention(self.cross_attn_norm(latents), encoder_hidden_states=encoder_hidden_states,
                                                 attention_mask=encoder_attention_mask)
        latents = latents + self.self_attention(self.self_attn_norm(latents), attention_mask=attention_mask)
        return latents + self.ffn(self.ffn_norm(latents))


class PerceiverResampler(nn.Module):
    """Squeezes the first text bundle into `num_gist_tokens` gist tokens."""

    def __init__(self, dim, n_heads, cross_attention_dim, num_queries, num_layers, norm_eps):
        super().__init__()
        self.latent_queries = nn.Parameter(torch.randn(num_queries, dim) * 0.02)
        self.layers = nn.ModuleList([PerceiverLayer(dim, n_heads, cross_attention_dim, norm_eps) for _ in range(num_layers)])
        self.norm_out = RMSNorm(dim, eps=norm_eps)

    def forward(self, encoder_hidden_states, num_gist_tokens, encoder_attention_mask=None, attention_mask=None):
        b = encoder_hidden_states.shape[0]
        latents = self.latent_queries[:num_gist_tokens].unsqueeze(0).expand(b, -1, -1).to(encoder_hidden_states.dtype)
        for layer in self.layers:
            latents = layer(latents, encoder_hidden_states, encoder_attention_mask, attention_mask)
        return self.norm_out(latents)


class FiboTransformer(nn.Module):
    """BriaFibo2Transformer2DModel. hidden_states: [B, 128, H/16, W/16] normalized, patchified Flux2 latents."""

    CONFIG_KEYS = ("in_channels", "patch_size", "dim", "n_layers", "n_refiner_layers", "n_heads", "norm_eps",
                   "cap_feat_dim", "injection_layer_ids", "perceiver_num_layers", "min_num_gist_tokens",
                   "max_num_gist_tokens", "gist_step", "gist_min_text_len", "gist_max_text_len", "rope_theta",
                   "t_scale", "axes_dims", "axes_lens")

    def __init__(self, in_channels=32, patch_size=2, dim=3584, n_layers=44, n_refiner_layers=2, n_heads=28,
                 norm_eps=1e-5, cap_feat_dim=7680, injection_layer_ids=(6, 14, 22, 30, 38), perceiver_num_layers=2,
                 min_num_gist_tokens=256, max_num_gist_tokens=512, gist_step=64, gist_min_text_len=256,
                 gist_max_text_len=2048, rope_theta=256.0, t_scale=1000.0, axes_dims=(32, 48, 48),
                 axes_lens=(2048, 512, 512)):
        super().__init__()
        self.config = dict(in_channels=in_channels, patch_size=patch_size, dim=dim, n_layers=n_layers,
                           n_refiner_layers=n_refiner_layers, n_heads=n_heads, norm_eps=norm_eps,
                           cap_feat_dim=cap_feat_dim, injection_layer_ids=list(injection_layer_ids),
                           perceiver_num_layers=perceiver_num_layers, min_num_gist_tokens=min_num_gist_tokens,
                           max_num_gist_tokens=max_num_gist_tokens, gist_step=gist_step,
                           gist_min_text_len=gist_min_text_len, gist_max_text_len=gist_max_text_len,
                           rope_theta=rope_theta, t_scale=t_scale, axes_dims=list(axes_dims), axes_lens=list(axes_lens))
        self.latent_channels = patch_size * patch_size * in_channels  # 128: the trainer's (patchified) layout
        self.text_dim = cap_feat_dim
        self.injection_layer_ids = tuple(injection_layer_ids)
        self.gradient_checkpointing = False
        self.x_embedder = nn.Linear(self.latent_channels, dim, bias=True)
        self.t_embedder = TimestepEmbedder(min(dim, ADALN_EMBED_DIM), mid_size=1024)
        self.rope_embedder = RopeEmbedder(theta=rope_theta, axes_dims=axes_dims, axes_lens=axes_lens)
        self.noise_refiner = nn.ModuleList([TransformerBlock(dim, n_heads, norm_eps) for _ in range(n_refiner_layers)])
        self.perceiver = PerceiverResampler(dim, n_heads, cap_feat_dim, max_num_gist_tokens, perceiver_num_layers, norm_eps)
        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads, norm_eps, cross_attention_dim=cap_feat_dim if i in self.injection_layer_ids else None)
            for i in range(n_layers)])
        self.final_layer = FinalLayer(dim, self.latent_channels)

    def split_tail_modules(self) -> list[nn.Module]:
        """Modules that run after the last block (they follow it in a model split)."""
        return [self.final_layer]

    @classmethod
    def from_config(cls, config: dict) -> "FiboTransformer":
        return cls(**{k: config[k] for k in cls.CONFIG_KEYS if k in config})

    @staticmethod
    def _grid(size, start, device) -> torch.Tensor:
        axes = [torch.arange(x0, x0 + span, dtype=torch.int32, device=device) for x0, span in zip(start, size)]
        return torch.stack(torch.meshgrid(axes, indexing="ij"), dim=-1).flatten(0, 2)

    def _run(self, block, *args):
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(block, *args, use_reentrant=False)
        return block(*args)

    def forward(self, hidden_states: torch.Tensor, timestep, encoder_hidden_states: torch.Tensor,
                encoder_attention_mask: torch.Tensor | None = None, skip_layers=None,
                context_latents: list[torch.Tensor] | None = None) -> torch.Tensor:
        """timestep: noise level in 0..1000 (sigma * 1000, as the MOSS model takes it). Returns the predicted velocity
        in the input's layout. context_latents: edit images in the same [B, 128, h, w] layout (clean tokens)."""
        skip_layers = frozenset(skip_layers or ())
        if hidden_states.ndim != 4 or hidden_states.shape[1] != self.latent_channels:
            raise ValueError(f"hidden_states must be [B, {self.latent_channels}, H/16, W/16]")
        if encoder_hidden_states.ndim != 4 or encoder_hidden_states.shape[1] != 1 + len(self.injection_layer_ids):
            raise ValueError("encoder_hidden_states must be [B, 6, S, cap_feat_dim] (the Qwen3-VL bundles)")
        batch, _, height, width = hidden_states.shape
        device = hidden_states.device
        cfg = self.config
        dtype = _wdtype(self.x_embedder)
        if encoder_attention_mask is not None and bool(encoder_attention_mask.all()):
            encoder_attention_mask = None  # nothing padded: attention runs without a mask
        enc = encoder_hidden_states.to(dtype)

        # 1. timestep embeddings: image tokens get the real noise level, gist and context tokens t = 0
        times = torch.as_tensor(timestep, device=device, dtype=torch.float32).reshape(-1).expand(batch)
        adaln_noisy = self.t_embedder(times)
        adaln_clean = self.t_embedder(torch.zeros_like(times))

        # 2. tokens: the 128 patch features are already in (channel, dy, dx) order, as the model's patchify orders them
        image_plane = cfg["max_num_gist_tokens"] + 1
        tokens = [hidden_states.flatten(2).transpose(1, 2)]
        ids = [self._grid((1, height, width), (image_plane, 0, 0), device)]
        for k, ctx in enumerate(context_latents or []):
            tokens.append(ctx.flatten(2).transpose(1, 2))
            ids.append(self._grid((1, ctx.shape[2], ctx.shape[3]), (image_plane + k + 1, 0, 0), device))
        num_image_tokens = height * width
        x = self.x_embedder(torch.cat(tokens, dim=1).to(dtype))
        image_freqs = self.rope_embedder(torch.cat(ids)).unsqueeze(0)

        # 3. noise refiner (with context images only the image tokens are noisy)
        refiner_mask = None
        if context_latents:
            refiner_mask = torch.zeros(batch, x.shape[1], dtype=torch.long, device=device)
            refiner_mask[:, :num_image_tokens] = 1
        for block in self.noise_refiner:
            x = self._run(block, x, None, image_freqs, refiner_mask, adaln_noisy, adaln_clean)

        # 4. gist tokens from the first bundle
        if encoder_attention_mask is not None:
            text_lengths = encoder_attention_mask.sum(dim=1).tolist()
        else:
            text_lengths = [enc.shape[2]] * batch
        gist_counts = compute_gist_count(text_lengths, cfg["min_num_gist_tokens"], cfg["max_num_gist_tokens"],
                                         cfg["gist_step"], cfg["gist_min_text_len"], cfg["gist_max_text_len"])
        num_gist = max(gist_counts)
        gist_mask = None
        if min(gist_counts) < num_gist:
            gist_mask = torch.arange(num_gist, device=device) < torch.tensor(gist_counts, device=device)[:, None]
        gist = self.perceiver(enc[:, 0], num_gist, encoder_attention_mask, gist_mask)
        gist_freqs = self.rope_embedder(self._grid((num_gist, 1, 1), (1, 0, 0), device)).unsqueeze(0)

        # 5. one stream: gist first, then image, then context tokens (gist last when the batch pads gist tokens)
        num_context = x.shape[1] - num_image_tokens
        attention_mask = None
        if gist_mask is None:
            image_start = num_gist
            x = torch.cat([gist.to(x.dtype), x], dim=1)
            freqs_cis = torch.cat([gist_freqs, image_freqs], dim=1)
        else:
            image_start = 0
            x = torch.cat([x, gist.to(x.dtype)], dim=1)
            freqs_cis = torch.cat([image_freqs, gist_freqs], dim=1)
            attention_mask = torch.cat([torch.ones(batch, num_image_tokens + num_context, dtype=torch.bool,
                                                   device=device), gist_mask], dim=1)
        noise_mask = torch.zeros(batch, x.shape[1], dtype=torch.long, device=device)
        noise_mask[:, image_start:image_start + num_image_tokens] = 1

        # 6. main blocks; injection blocks read their own bundle
        bundle_index = {lid: i + 1 for i, lid in enumerate(self.injection_layer_ids)}
        split = getattr(self, "split_devices", None)  # blocks may live on several devices (model split)
        for lid, block in enumerate(self.layers):
            if lid in skip_layers:
                continue
            text = enc[:, bundle_index[lid]] if lid in bundle_index else None
            if split:
                ldev = _module_device(block)
                if ldev is not None and x.device != ldev:
                    x, freqs_cis, noise_mask = x.to(ldev), freqs_cis.to(ldev), noise_mask.to(ldev)
                    adaln_noisy, adaln_clean = adaln_noisy.to(ldev), adaln_clean.to(ldev)
                    attention_mask = attention_mask.to(ldev) if attention_mask is not None else None
                    enc = enc.to(ldev)
                    text = enc[:, bundle_index[lid]] if lid in bundle_index else None
                    if encoder_attention_mask is not None:
                        encoder_attention_mask = encoder_attention_mask.to(ldev)
            x = self._run(block, x, attention_mask, freqs_cis, noise_mask, adaln_noisy, adaln_clean, text,
                          encoder_attention_mask)

        # 7. final layer on the image tokens, back to the input layout
        if split:
            fdev = self.final_layer.linear.weight.device
            if x.device != fdev:
                x, adaln_noisy = x.to(fdev), adaln_noisy.to(fdev)
        out = self.final_layer(x[:, image_start:image_start + num_image_tokens], adaln_noisy)
        return out.transpose(1, 2).reshape(batch, self.latent_channels, height, width).to(device)
