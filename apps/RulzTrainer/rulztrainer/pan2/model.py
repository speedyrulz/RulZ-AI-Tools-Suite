"""PAN 2 (IFM / MBZUAI), the 29B video model, for one-frame targets: stills trained as one-frame videos.

Ported from diffusers' PAN2Transformer3DModel as ComfyUI's comfy/ldm/pan2/model.py reads it (the parameter names are
the checkpoint's, so the diffusers folder, the single files made from it and every LoRA key line up), with IFM's
trainer (github.com/ifm-ai/PAN2, Apache-2.0) as the reference for how the model is trained.

Call convention (the trainer's): `model(latents, sigma * 1000, embeds, mask)` with normalized Wan 2.2 latents
[B, 48, H/16, W/16] (the model patchifies 1x2x2, so H and W are multiples of 32 px), the text features [B, S, 4096]
(Qwen3.5-9B's hidden state two layers before the last, pan2/loading.py) and a [B, S] mask of the real tokens. It
returns the velocity noise - x0, which is what PAN 2 predicts (pan/flow_matching.py), [B, 48, H/16, W/16] in fp32.

  - the input has 97 channels: 48 noisy + 48 condition latents + 1 mask of the given frames. A still trains as text
    to video (IFM's loader: "images always train as t2v"), where the 49 extra channels are zero, so the patch
    embedding reads the first 48 input channels of its weight and the rest never enter;
  - one joint sequence [video | text] per sample: each stream has its own projections and feed-forward network, the
    video stream is modulated by the timestep and the text stream by a learned table; only the video tokens are
    rotated (RoPE over frame, row, column; a still is frame 0);
  - padded text tokens are masked out as keys, in the text refiner and in the joint attention (IFM's packed loader
    never pads: `mask_text_padding`), so a sample in a padded batch gets what it gets alone;
  - as in the reference, the time embedding, the norms with their shift and scale and the gated residuals run in
    fp32; the checkpoint keeps the time embedder and the norm projections in fp32 and the loader leaves them so
    (unless a file stores them quantized).

Gradient checkpointing, the model-split device hand-off, `skip_layers` and the alignment taps work as in the other
ports (the taps see each 2x2 token on its 4 latent cells, the grid of the region maps); an odd latent grid (a probe
size that is not whole 32 px) is padded circularly and cropped back, as ComfyUI pads to the patch size.
QuantLinear / LoRALinear replace nn.Linear transparently."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from ..attention import sdpa
from ..checkpointing import checkpointed
from ..compiling import thread_prepare

PATCH = 2
SHIFT = 9.0  # the released model's training and sampling shift (config/sft_29B_720p.yaml: train_timestep_shift)
MAX_TEXT_TOKENS = 1536  # packed_caption_segment_limit


def shift_sigma(sigma: torch.Tensor, shift: float = SHIFT) -> torch.Tensor:
    """s * f / (1 + (s - 1) * f): the reference's fixed time shift (pan/flow_matching.shift_timesteps)."""
    return shift * sigma / (1.0 + (shift - 1.0) * sigma)


def timestep_embedding(t: torch.Tensor, dim: int = 256, max_period: float = 10000.0) -> torch.Tensor:
    """diffusers' Timesteps(flip_sin_to_cos=True, downscale_freq_shift=0): [cos, sin] of t in 0..1000."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def rope_tables(sizes, axes_dim, theta: float, temporal_scale: float, device):
    """(cos, sin), each [T * H * W, sum(axes_dim)] fp32; only the frame positions are rescaled."""
    grids = torch.meshgrid(*[torch.arange(s, device=device, dtype=torch.float32) for s in sizes], indexing="ij")
    cos, sin = [], []
    for i, dim in enumerate(axes_dim):
        pos = grids[i].reshape(-1)
        if i == 0:
            pos = pos * temporal_scale
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device)[: dim // 2] / dim))
        freqs = torch.outer(pos, freqs)
        cos.append(freqs.cos().repeat_interleave(2, dim=1))
        sin.append(freqs.sin().repeat_interleave(2, dim=1))
    return torch.cat(cos, dim=1), torch.cat(sin, dim=1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x [B, S, heads, head_dim], rotated in fp32 as the reference does."""
    x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
    rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
    return (x.float() * cos[None, :, None, :] + rotated.float() * sin[None, :, None, :]).to(x.dtype)


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
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.shape[-1],), self.weight.to(x.dtype), self.eps)


class FeedForward(nn.Module):
    """diffusers' FeedForward names: net.0.proj, the activation, net.2. GELU(tanh) in the blocks, SwiGLU (the
    projection twice as wide, value * silu(gate)) in the text refiner."""

    def __init__(self, dim: int, inner: int, swiglu: bool = False, bias: bool = True):
        super().__init__()
        self.swiglu = swiglu
        proj = nn.Module()
        proj.proj = nn.Linear(dim, inner * 2 if swiglu else inner, bias=bias)
        self.net = nn.ModuleList([proj, nn.Identity(), nn.Linear(inner, dim, bias=bias)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net[0].proj(x)
        if self.swiglu:
            x, gate = x.chunk(2, dim=-1)
            x = x * F.silu(gate)
        else:
            x = F.gelu(x, approximate="tanh")
        return self.net[2](x)


class Attention(nn.Module):
    """Self-attention over the text (the refiner), or joint attention over [video | text] with the text's own
    projections (`joint`). Per-head RMSNorm on q and k."""

    def __init__(self, dim: int, heads: int, head_dim: int, eps: float, bias: bool = True, joint: bool = False):
        super().__init__()
        self.heads, self.head_dim = heads, head_dim
        self.to_q = nn.Linear(dim, dim, bias=bias)
        self.to_k = nn.Linear(dim, dim, bias=bias)
        self.to_v = nn.Linear(dim, dim, bias=bias)
        self.norm_q = RMSNorm(head_dim, eps)
        self.norm_k = RMSNorm(head_dim, eps)
        self.to_out = nn.ModuleList([nn.Linear(dim, dim, bias=bias)])
        if joint:
            self.add_q_proj = nn.Linear(dim, dim, bias=bias)
            self.add_k_proj = nn.Linear(dim, dim, bias=bias)
            self.add_v_proj = nn.Linear(dim, dim, bias=bias)
            self.norm_added_q = RMSNorm(head_dim, eps)
            self.norm_added_k = RMSNorm(head_dim, eps)
            self.to_add_out = nn.Linear(dim, dim, bias=bias)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        return x.unflatten(-1, (self.heads, self.head_dim))

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None, cos=None, sin=None,
                key_mask: torch.Tensor | None = None):
        """key_mask: [B, 1, 1, L] bool of the keys every query may see (None: all). Returns the video rows, and with
        `context` also the text rows."""
        q, k, v = self.norm_q(self._heads(self.to_q(x))), self.norm_k(self._heads(self.to_k(x))), self._heads(self.to_v(x))
        if cos is not None:  # only the video stream is rotated
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        n = x.shape[1]
        if context is not None:
            q = torch.cat([q, self.norm_added_q(self._heads(self.add_q_proj(context))).to(q.dtype)], dim=1)
            k = torch.cat([k, self.norm_added_k(self._heads(self.add_k_proj(context))).to(k.dtype)], dim=1)
            v = torch.cat([v, self._heads(self.add_v_proj(context)).to(v.dtype)], dim=1)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        tap = getattr(self, "_rz_tap", None)  # rulztrainer.alignment: the trigger's attention (training-only loss)
        if tap is not None and context is not None:
            grid = getattr(self, "rz_grid", None)  # (token rows, token columns): each 2x2 token's query stands for
            if grid is not None:  # its 4 latent cells, the grid the region maps are on
                hp, wp = grid
                cells = q[:, :, :n].unflatten(2, (hp, wp)).repeat_interleave(PATCH, 2).repeat_interleave(PATCH, 3)
                tap(torch.cat([cells.flatten(2, 3), q[:, :, n:]], dim=2), k)
            else:
                tap(q, k)
        out = sdpa(q, k, v, attn_mask=key_mask).transpose(1, 2).flatten(2).to(x.dtype)
        if context is None:
            return self.to_out[0](out)
        return self.to_out[0](out[:, :n]), self.to_add_out(out[:, n:])


class TokenRefinerBlock(nn.Module):
    """Plain pre-norm block over the text alone: no modulation, no RoPE, no biases."""

    def __init__(self, dim: int, heads: int, head_dim: int, mlp_ratio: float, eps: float):
        super().__init__()
        self.norm1 = RMSNorm(dim, eps)
        self.attn = Attention(dim, heads, head_dim, eps, bias=False)
        self.norm2 = RMSNorm(dim, eps)
        self.ff = FeedForward(dim, int(dim * mlp_ratio), swiglu=True, bias=False)

    def forward(self, x: torch.Tensor, key_mask=None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), key_mask=key_mask)
        return x + self.ff(self.norm2(x))


class AdaLayerNorm(nn.Module):
    """The `linear` of diffusers' AdaLayerNormZero / AdaLayerNormContinuous: silu(temb) -> `chunks` vectors, fp32."""

    def __init__(self, dim: int, chunks: int):
        super().__init__()
        self.linear = nn.Linear(dim, chunks * dim)

    def forward(self, temb: torch.Tensor) -> torch.Tensor:
        return self.linear(F.silu(temb).to(_wdtype(self.linear))).float()


def modulated_norm(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return (F.layer_norm(x.float(), x.shape[-1:], None, None, eps) * (1 + scale) + shift).to(x.dtype)


def gated_residual(x: torch.Tensor, y: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return (x.float() + y.float() * gate).to(x.dtype)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int, mlp_ratio: float, context_mlp_ratio: float,
                 eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.norm1 = AdaLayerNorm(dim, 6)
        self.scale_shift_table_context = nn.Parameter(torch.zeros(6 * dim))
        self.attn = Attention(dim, heads, head_dim, eps, joint=True)
        self.ff = FeedForward(dim, int(dim * mlp_ratio))
        self.ff_context = FeedForward(dim, int(dim * context_mlp_ratio))

    def forward(self, x, context, temb, cos, sin, key_mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(temb).unsqueeze(1).chunk(6, dim=-1)
        table = self.scale_shift_table_context.to(device=x.device, dtype=torch.float32)
        c_shift_msa, c_scale_msa, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = table.chunk(6, dim=-1)

        attn_x, attn_c = self.attn(modulated_norm(x, scale_msa, shift_msa, self.eps),
                                   modulated_norm(context, c_scale_msa, c_shift_msa, self.eps), cos, sin, key_mask)
        x = gated_residual(x, attn_x, gate_msa)
        context = gated_residual(context, attn_c, c_gate_msa)
        x = gated_residual(x, self.ff(modulated_norm(x, scale_mlp, shift_mlp, self.eps)), gate_mlp)
        context = gated_residual(context, self.ff_context(modulated_norm(context, c_scale_mlp, c_shift_mlp, self.eps)),
                                 c_gate_mlp)
        return x, context


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, dim)
        self.linear_2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_1(x.to(_wdtype(self.linear_1)))
        return self.linear_2(F.silu(x).to(_wdtype(self.linear_2))).float()


class PAN2Transformer(nn.Module):
    """PAN2Transformer3DModel for one-frame targets. hidden_states: [B, 48, H/16, W/16] normalized latents."""

    CONFIG_KEYS = ("patch_size", "in_channels", "out_channels", "num_attention_heads", "attention_head_dim",
                   "num_layers", "num_refiner_layers", "mlp_ratio", "context_mlp_ratio", "refiner_mlp_ratio",
                   "text_embed_dim", "rope_axes_dim", "rope_theta", "rope_temporal_scale")

    def __init__(self, patch_size=(1, 2, 2), in_channels=97, out_channels=48, num_attention_heads=32,
                 attention_head_dim=128, num_layers=44, num_refiner_layers=4, mlp_ratio=8.0, context_mlp_ratio=4.0,
                 refiner_mlp_ratio=4.0, text_embed_dim=4096, rope_axes_dim=(16, 56, 56), rope_theta=256.0,
                 rope_temporal_scale=16 / 24):
        super().__init__()
        if tuple(patch_size) != (1, PATCH, PATCH):
            raise ValueError(f"PAN 2 patchifies 1x2x2, got {tuple(patch_size)}")
        if sum(rope_axes_dim) != attention_head_dim:
            raise ValueError("the rotary axes must add up to the head size")
        if in_channels < out_channels:
            raise ValueError("in_channels holds the noisy latents first: it cannot be smaller than out_channels")
        dim = num_attention_heads * attention_head_dim
        self.config = dict(patch_size=list(patch_size), in_channels=in_channels, out_channels=out_channels,
                           num_attention_heads=num_attention_heads, attention_head_dim=attention_head_dim,
                           num_layers=num_layers, num_refiner_layers=num_refiner_layers, mlp_ratio=mlp_ratio,
                           context_mlp_ratio=context_mlp_ratio, refiner_mlp_ratio=refiner_mlp_ratio,
                           text_embed_dim=text_embed_dim, rope_axes_dim=list(rope_axes_dim), rope_theta=rope_theta,
                           rope_temporal_scale=rope_temporal_scale)
        self.latent_channels = out_channels
        self.text_dim = text_embed_dim
        self.hidden_size = dim
        self.rope_axes_dim, self.rope_theta = tuple(rope_axes_dim), rope_theta
        self.rope_temporal_scale = rope_temporal_scale
        self.shift = SHIFT
        self.gradient_checkpointing = False
        self.x_embedder = nn.Conv3d(in_channels, dim, kernel_size=(1, PATCH, PATCH), stride=(1, PATCH, PATCH))
        self.time_embedder = TimestepEmbedding(256, dim)
        self.context_embedder = nn.Sequential(RMSNorm(text_embed_dim, 1e-5), nn.Linear(text_embed_dim, dim))
        self.context_refiner = nn.ModuleList([
            TokenRefinerBlock(dim, num_attention_heads, attention_head_dim, refiner_mlp_ratio, 1e-5)
            for _ in range(num_refiner_layers)])
        self.context_type_embedding = nn.Parameter(torch.zeros(dim))
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(dim, num_attention_heads, attention_head_dim, mlp_ratio, context_mlp_ratio)
            for _ in range(num_layers)])
        self.norm_out = AdaLayerNorm(dim, 2)
        self.proj_out = nn.Linear(dim, PATCH * PATCH * out_channels)

    @property
    def layers(self) -> nn.ModuleList:
        """The main blocks under the name the trainer uses (block selection, block swap, model split)."""
        return self.transformer_blocks

    def split_tail_modules(self) -> list[nn.Module]:
        """Modules that run after the last block (they follow it in a model split)."""
        return [self.norm_out, self.proj_out]

    @classmethod
    def from_config(cls, config: dict) -> "PAN2Transformer":
        return cls(**{k: config[k] for k in cls.CONFIG_KEYS if k in config})

    def _run(self, block, *args):
        thread_prepare()  # dynamo's limits in this thread (autograd recompute, data-parallel workers)
        if (self.gradient_checkpointing and self.training and torch.is_grad_enabled()
                and not getattr(block, "keep_activations", False)):  # see rulztrainer.checkpointing
            return checkpointed(self, block, *args)
        return block(*args)

    def embed_patches(self, latents: torch.Tensor) -> torch.Tensor:
        """[B, 48, h, w] or a clip [B, 48, T, h, w] -> [B, T * h/2 * w/2, dim]: the patch embedding on the noisy
        channels (see the module docstring: the condition latents and their mask are zero for text-to-video, so
        their weights add nothing)."""
        w = self.x_embedder.weight
        if latents.ndim == 4:
            latents = latents.unsqueeze(2)
        x = F.conv3d(latents.to(w.dtype), w[:, : self.latent_channels], self.x_embedder.bias,
                     stride=self.x_embedder.stride)
        return x.flatten(2).transpose(1, 2)

    def forward(self, hidden_states: torch.Tensor, timestep, encoder_hidden_states: torch.Tensor,
                encoder_attention_mask: torch.Tensor | None = None, skip_layers=None) -> torch.Tensor:
        """timestep: the noise level in 0..1000 (sigma * 1000). Returns the velocity noise - x0, [B, 48, h, w] fp32.

        A clip [B, 48, T, h, w] (T latent frames, sample and probe renders) comes back in that shape. The training
        losses' taps read stills only."""
        skip_layers = frozenset(skip_layers or ())
        clip = hidden_states.ndim == 5
        if hidden_states.ndim not in (4, 5) or hidden_states.shape[1] != self.latent_channels:
            raise ValueError(f"hidden_states must be latents [B, {self.latent_channels}, H/16, W/16] or a clip "
                             f"[B, {self.latent_channels}, T, H/16, W/16]")
        n_frames = hidden_states.shape[2] if clip else 1
        if clip:  # frames ride in the batch axis for the padding below, and come back after it
            hidden_states = hidden_states.transpose(1, 2).flatten(0, 1)
        if encoder_hidden_states.ndim != 3 or encoder_hidden_states.shape[2] != self.text_dim:
            raise ValueError(f"encoder_hidden_states must be [B, S, {self.text_dim}]")
        out_h, out_w = hidden_states.shape[-2:]
        batch = hidden_states.shape[0] // n_frames
        if out_h % PATCH or out_w % PATCH:  # not a 32 px image (a probe size): padded to whole tokens, cropped after
            hidden_states = F.pad(hidden_states, (0, out_w % PATCH, 0, out_h % PATCH), mode="circular")
        height, width = hidden_states.shape[-2:]
        hp, wp = height // PATCH, width // PATCH
        device = hidden_states.device
        dtype = _wdtype(self.context_embedder[1])
        text_len = encoder_hidden_states.shape[1]
        n_video = n_frames * hp * wp
        if clip:
            hidden_states = hidden_states.unflatten(0, (batch, n_frames)).transpose(1, 2)  # [B, 48, T, h, w]

        t = torch.as_tensor(timestep, device=device, dtype=torch.float32).reshape(-1).expand(batch)
        temb = self.time_embedder(timestep_embedding(t))  # fp32

        valid = key_mask = txt_mask = None
        if encoder_attention_mask is not None:
            valid = encoder_attention_mask.to(device=device, dtype=torch.bool)
            if not bool(valid.all()):
                txt_mask = valid[:, None, None, :]
                key_mask = torch.cat([valid.new_ones(batch, n_video), valid], dim=1)[:, None, None, :]
        cos, sin = rope_tables((n_frames, hp, wp), self.rope_axes_dim, self.rope_theta, self.rope_temporal_scale, device)

        x = self.embed_patches(hidden_states).to(dtype)
        context = self.context_embedder[1](self.context_embedder[0](encoder_hidden_states.to(device, dtype)))
        for block in self.context_refiner:
            context = block(context, txt_mask)
        context = context + self.context_type_embedding.to(device=device, dtype=context.dtype)

        split = getattr(self, "split_devices", None)  # blocks may live on several devices (model split)
        taps = getattr(self, "_rz_taps", None)  # rulztrainer.alignment (training-only losses), on the latent cells
        if clip:
            taps = None
        if taps is not None:
            for blk in self.transformer_blocks:
                blk.attn.rz_grid = (hp, wp)
        if taps is not None and taps.armed:
            taps.set_layout(0, (0, height * width), (n_video, text_len), valid)

        def cells(rows):  # [B, N, D] token states -> [B, h * w, D], each token's state on its 4 cells
            return rows.unflatten(1, (hp, wp)).repeat_interleave(PATCH, 1).repeat_interleave(PATCH, 2).flatten(1, 2)

        for i, block in enumerate(self.transformer_blocks):
            if i in skip_layers:
                continue
            if split:
                bdev = _module_device(block)
                if bdev is not None and x.device != bdev:
                    x, context, temb, cos, sin = (v.to(bdev) for v in (x, context, temb, cos, sin))
                    key_mask = key_mask.to(bdev) if key_mask is not None else None
            x, context = self._run(block, x, context, temb, cos, sin, key_mask)
            if taps is not None and taps.armed and taps.recording and i in taps.hidden_layers:
                taps.record_hidden(i, cells(x))

        if split:
            fdev = _module_device(self.proj_out)
            if fdev is not None and x.device != fdev:
                x, temb = x.to(fdev), temb.to(fdev)
        scale, shift = self.norm_out(temb).unsqueeze(1).chunk(2, dim=-1)
        x = self.proj_out(modulated_norm(x, scale, shift).to(_wdtype(self.proj_out))).float()
        # [B, N, 2 * 2 * C] with the channel first inside a token, as the reference unpatchifies
        if clip:
            x = x.reshape(batch, n_frames, hp, wp, self.latent_channels, PATCH, PATCH).permute(0, 4, 1, 2, 5, 3, 6)
            return x.reshape(batch, self.latent_channels, n_frames, height, width)[..., :out_h, :out_w].to(device)
        x = x.reshape(batch, hp, wp, self.latent_channels, PATCH, PATCH).permute(0, 3, 1, 4, 2, 5)
        return x.reshape(batch, self.latent_channels, height, width)[:, :, :out_h, :out_w].to(device)
