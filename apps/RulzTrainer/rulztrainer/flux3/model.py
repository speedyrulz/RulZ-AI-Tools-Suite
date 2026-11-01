"""FLUX 3 Image Klein DiT (Flux3Transformer2DModel) for training, in RulzTrainer's conventions.

Ported from diffusers' transformer_flux3.py (Copyright 2026 Black Forest Labs and The HuggingFace Team, Apache-2.0)
with the diffusers mixins removed, so that:
  - the state-dict keys are Black Forest Labs' own (the root model.safetensors of the HF repos, which ComfyUI's
    flux3-image-klein-*-bf16 files are byte for byte): content_mode_blocks / txt_mode_blocks / single_blocks with
    q_proj, k_proj, v_proj, attn_out, mlp_in, mlp_out and norm.query_norm / key_norm, emb_in, txt_in, time_in,
    early/single_stream_modulations.*.lin and final_layer.image. A LoRA exported with these names under
    "diffusion_model." is what ComfyUI's loader maps natively (and onto the fused qkv of its int8 files), and what
    diffusers' Flux3LoraLoaderMixin converts (_convert_non_diffusers_flux3_lora_to_diffusers). The diffusers folder
    (transformer/) is renamed on load (flux3/single_file.py);
  - the forward takes what the trainer and sampler give the other families: `hidden_states` as the normalized,
    2x2-packed Flux2 VAE latents [B, 128, H/16, W/16] (the MOSS-Image / FIBO 2.0 layout), `timestep` in 0..1000 units
    (sigma * 1000) and the text features as [B, S, 20480] (eight Qwen3-VL-4B hidden layers stacked per token, see
    flux3/loading.py) with a [B, S] mask of the rows that belong to each sample;
  - gradient checkpointing (torch.utils.checkpoint) and the model-split device hand-off work like the other ports;
    `skip_layers` skips joint blocks (kept for the shared sampler signature);
  - QuantLinear / LoRALinear replace nn.Linear modules transparently (only their dtype handling is consulted).

Architecture (5.5B, 3072 wide, 24 heads of 128): three token modes. The text context and the noisy image each pass
through their own 5 per-mode blocks (self-attention within the mode only), then 28 joint blocks run over [text,
image]. Every block is a parallel block (attention and a SwiGLU feed-forward from the same LayerNorm-ed, modulated
input, one gated residual), with q/k RMSNorm per head in fp32 and 4-axis RoPE (t, h, w, l; 32 dims each, theta
10000) applied in fp32. The modulation is per mode: the image tokens by the flow timestep, the text by t = 0, each
with its own shift / scale / gate projection for the per-mode and for the joint blocks. RoPE positions: text token j
at (0, 0, 0, j); image cell (row, col) at (0, row / 2, col / 2, 0) (the model's image_rope_scale 0.5). Positions do
not depend on the text length, so a padded batch only needs its padding masked out of the keys to give each sample
exactly what it gets alone. A flow-map checkpoint (flow_map, the 16-step distilled Klein) adds `ref_time_in`, the
embedding of the timestep each step jumps to, to every mode's timestep embedding.

Not ported: the reference-image mode (image_cond: editing with references), whose 5 blocks never run for
text-to-image; its weights are left out on load (about 1.3 GB in bf16), the trainer trains text-to-image LoRAs.
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
IMAGE_ROPE_SCALE = 0.5  # the checkpoint's image H/W position scale (config.json dit.image_rope_scale)


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
    """torch's RMSNorm in fp32 (diffusers keeps norm_q / norm_k in fp32); BFL stores the multiplier as `scale`.
    Returns fp32: RoPE runs in fp32 before the single cast back."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim, self.eps = dim, eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x.float(), (self.dim,), weight=self.scale.float(), eps=self.eps)


class QKNorm(nn.Module):
    def __init__(self, head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.query_norm = RMSNorm(head_dim, eps)
        self.key_norm = RMSNorm(head_dim, eps)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x [B, L, H, D] (fp32); cos/sin [L, D] with interleaved (real, imaginary) pairs (diffusers' apply_rotary_emb,
    use_real_unbind_dim -1)."""
    x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
    rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
    cos, sin = cos[None, :, None, :], sin[None, :, None, :]
    return (x.float() * cos + rotated.float() * sin).to(x.dtype)


def rope_tables(ids: torch.Tensor, axes_dims, theta: float = ROPE_THETA) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin [L, sum(axes_dims)] for positions ids [L, n_axes], each pair's angle repeated (interleaved). The angles
    are computed in float64 as diffusers' Flux3PosEmbed does, then cast to fp32."""
    cos_out, sin_out = [], []
    pos = ids.to(torch.float64)
    for i, dim in enumerate(axes_dims):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float64, device=ids.device) / dim))
        angles = pos[:, i, None] * freqs[None]
        cos_out.append(angles.cos().repeat_interleave(2, dim=1).float())
        sin_out.append(angles.sin().repeat_interleave(2, dim=1).float())
    return torch.cat(cos_out, dim=-1), torch.cat(sin_out, dim=-1)


def text_ids(length: int, device=None) -> torch.Tensor:
    """[L, 4] positions (t, h, w, l) of the text tokens: text advances along l."""
    ids = torch.zeros(length, 4, device=device)
    ids[:, 3] = torch.arange(length, device=device, dtype=torch.float32)
    return ids


def image_ids(height: int, width: int, scale: float = IMAGE_ROPE_SCALE, device=None) -> torch.Tensor:
    """[H*W, 4] positions of a latent grid, rows first (the pipeline's flatten(2) order): (0, row * s, col * s, 0)."""
    rows = torch.arange(height, device=device, dtype=torch.float32)
    cols = torch.arange(width, device=device, dtype=torch.float32)
    grid = torch.stack(torch.meshgrid(rows, cols, indexing="ij"), dim=-1).reshape(-1, 2) * scale
    ids = torch.zeros(height * width, 4, device=device)
    ids[:, 1:3] = grid
    return ids


class Block(nn.Module):
    """A parallel block: one LayerNorm-ed, modulated input for attention and the SwiGLU feed-forward. The per-mode
    blocks (ModeBlock) attend within one mode, the joint blocks (SingleBlock) over [text, image]."""

    def __init__(self, dim: int, heads: int, head_dim: int, mlp_ratio: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.heads, self.head_dim = heads, head_dim
        inner = heads * head_dim
        hidden = int(dim * mlp_ratio)
        self.pre_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)  # no weights (not in the checkpoint)
        self.q_proj = nn.Linear(dim, inner, bias=False)
        self.k_proj = nn.Linear(dim, inner, bias=False)
        self.v_proj = nn.Linear(dim, inner, bias=False)
        self.norm = QKNorm(head_dim, eps)
        self.attn_out = nn.Linear(inner, dim, bias=False)
        self.mlp_in = nn.Linear(dim, 2 * hidden, bias=False)  # [gate; up] fused, SwiGLU halves
        self.mlp_out = nn.Linear(hidden, dim, bias=False)

    def attention(self, h: torch.Tensor, cos, sin, key_mask: torch.Tensor | None) -> torch.Tensor:
        """key_mask: [B, 1, 1, L] bool of the keys every query may see (None: all of them)."""
        v = self.v_proj(h).unflatten(-1, (self.heads, self.head_dim))
        q = self.norm.query_norm(self.q_proj(h).unflatten(-1, (self.heads, self.head_dim)))
        k = self.norm.key_norm(self.k_proj(h).unflatten(-1, (self.heads, self.head_dim)))
        q, k = apply_rope(q, cos, sin).to(v.dtype), apply_rope(k, cos, sin).to(v.dtype)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        tap = getattr(self, "_rz_tap", None)  # rulztrainer.alignment: the trigger's attention (training-only loss)
        if tap is not None:
            tap(q, k)
        out = sdpa(q, k, v, attn_mask=key_mask)
        return self.attn_out(out.transpose(1, 2).flatten(2, 3).to(v.dtype))

    def mlp(self, h: torch.Tensor) -> torch.Tensor:
        x = self.mlp_in(h)
        half = x.shape[-1] // 2
        return self.mlp_out(F.silu(x[..., :half]) * x[..., half:])

    def modulated(self, x: torch.Tensor, mod: torch.Tensor):
        shift, scale, gate = mod.chunk(3, dim=-1)
        return (1 + scale) * self.pre_norm(x) + shift, gate


class ModeBlock(Block):
    def forward(self, x, mod, cos, sin, key_mask=None):
        """x [B, L, dim] of one mode; mod [B, 1, 3 * dim] = [shift, scale, gate]."""
        h, gate = self.modulated(x, mod)
        return x + gate * (self.attention(h, cos, sin, key_mask) + self.mlp(h))


class SingleBlock(Block):
    def forward(self, img, txt, img_mod, txt_mod, cos, sin, key_mask=None):
        """Joint over [txt, img] (cos / sin in that order); returns (txt, img)."""
        hi, gi = self.modulated(img, img_mod)
        ht, gt = self.modulated(txt, txt_mod)
        h = torch.cat([ht, hi], dim=1)
        out = self.attention(h, cos, sin, key_mask) + self.mlp(h)
        n = txt.shape[1]
        return txt + gt * out[:, :n], img + gi * out[:, n:]


class TimestepEmbedding(nn.Module):
    """diffusers' TimestepEmbedding(256, dim, sample_proj_bias=False) under BFL's names: in_layer -> SiLU -> out_layer."""

    def __init__(self, in_channels: int, dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_channels, dim, bias=False)
        self.out_layer = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_layer(F.silu(self.in_layer(x)))


def timestep_proj(t: torch.Tensor, channels: int = 256) -> torch.Tensor:
    """diffusers' Timesteps(channels, flip_sin_to_cos=True, downscale_freq_shift=0): [cos, sin] of t * freqs, fp32."""
    half = channels // 2
    exponent = -math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    args = t[:, None].float() * torch.exp(exponent)[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class Modulation(nn.Module):
    """One mode's shift / scale / gate projection of the (already SiLU-activated) timestep embedding."""

    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Linear(dim, 3 * dim, bias=False)

    def forward(self, temb_act: torch.Tensor) -> torch.Tensor:
        return self.lin(temb_act)


class FinalLayer(nn.Module):
    """LayerNorm modulated by the image timestep embedding, then the projection to the latent channels. BFL's
    adaLN_modulation gives [shift, scale] (diffusers' norm_out stores them swapped)."""

    def __init__(self, dim: int, out_channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim, bias=False))
        self.linear = nn.Linear(dim, out_channels, bias=False)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(temb).chunk(2, dim=-1)
        return self.linear(self.norm(x) * (1 + scale) + shift)


class Flux3KleinTransformer(nn.Module):
    """Flux3Transformer2DModel (text-to-image). hidden_states: [B, 128, H/16, W/16] normalized Flux2 VAE latents."""

    CONFIG_KEYS = ("in_channels", "num_layers", "num_single_layers", "attention_head_dim", "num_attention_heads",
                   "joint_attention_dim", "timestep_channels", "mlp_ratio", "axes_dims_rope", "rope_theta", "eps",
                   "flow_map", "image_rope_scale")

    def __init__(self, in_channels=128, num_layers=5, num_single_layers=28, attention_head_dim=128,
                 num_attention_heads=24, joint_attention_dim=20480, timestep_channels=256, mlp_ratio=3.0,
                 axes_dims_rope=(32, 32, 32, 32), rope_theta=ROPE_THETA, eps=1e-6, flow_map=False,
                 image_rope_scale=IMAGE_ROPE_SCALE):
        super().__init__()
        dim = attention_head_dim * num_attention_heads
        if sum(axes_dims_rope) != attention_head_dim:
            raise ValueError(f"sum(axes_dims_rope)={sum(axes_dims_rope)} must equal attention_head_dim")
        self.config = dict(in_channels=in_channels, num_layers=num_layers, num_single_layers=num_single_layers,
                           attention_head_dim=attention_head_dim, num_attention_heads=num_attention_heads,
                           joint_attention_dim=joint_attention_dim, timestep_channels=timestep_channels,
                           mlp_ratio=mlp_ratio, axes_dims_rope=list(axes_dims_rope), rope_theta=rope_theta, eps=eps,
                           flow_map=bool(flow_map), image_rope_scale=image_rope_scale)
        self.latent_channels = in_channels
        self.text_dim = joint_attention_dim
        self.axes_dims = tuple(axes_dims_rope)
        self.rope_theta = rope_theta
        self.image_rope_scale = image_rope_scale
        self.timestep_channels = timestep_channels
        self.flow_map = bool(flow_map)
        self.gradient_checkpointing = False
        self.is_distilled = self.flow_map  # the 16-step flow-map Klein samples with its own steps (flux3/sampler.py)
        block = dict(dim=dim, heads=num_attention_heads, head_dim=attention_head_dim, mlp_ratio=mlp_ratio, eps=eps)
        self.time_in = TimestepEmbedding(timestep_channels, dim)
        self.ref_time_in = TimestepEmbedding(timestep_channels, dim) if flow_map else None
        self.early_stream_modulations = nn.ModuleDict({m: Modulation(dim) for m in ("image", "txt")})
        self.single_stream_modulations = nn.ModuleDict({m: Modulation(dim) for m in ("image", "txt")})
        self.emb_in = nn.ModuleDict({"image": nn.Linear(in_channels, dim, bias=False)})
        self.txt_in = nn.Linear(joint_attention_dim, dim, bias=False)
        self.content_mode_blocks = nn.ModuleDict({"image": nn.ModuleList(ModeBlock(**block)
                                                                          for _ in range(num_layers))})
        self.txt_mode_blocks = nn.ModuleList(ModeBlock(**block) for _ in range(num_layers))
        self.single_blocks = nn.ModuleList(SingleBlock(**block) for _ in range(num_single_layers))
        self.final_layer = nn.ModuleDict({"image": FinalLayer(dim, in_channels, eps)})

    @property
    def layers(self) -> nn.ModuleList:
        """The joint blocks under the name the trainer uses (block selection, block swap, model split, alignment
        taps); the per-mode blocks run before them, like Krea 2's text fusion."""
        return self.single_blocks

    @property
    def mode_blocks(self) -> list[nn.Module]:
        """The per-mode blocks, text first, in the order they run."""
        return [*self.txt_mode_blocks, *self.content_mode_blocks["image"]]

    def split_tail_modules(self) -> list[nn.Module]:
        """Modules that run after the last joint block (they follow it in a model split)."""
        return [self.final_layer]

    @classmethod
    def from_config(cls, config: dict) -> "Flux3KleinTransformer":
        return cls(**{k: config[k] for k in cls.CONFIG_KEYS if k in config})

    def _run(self, block, *args):
        thread_prepare()  # dynamo's limits in this thread (autograd recompute, data-parallel workers)
        if (self.gradient_checkpointing and self.training and torch.is_grad_enabled()
                and not getattr(block, "keep_activations", False)):  # see rulztrainer.checkpointing
            return checkpoint(block, *args, use_reentrant=False)
        return block(*args)

    def _temb(self, t: torch.Tensor, dtype, embedder: TimestepEmbedding) -> torch.Tensor:
        # the pipeline hands the model t / 1000 in the model's dtype, which multiplies it by 1000 in that dtype again
        return embedder(timestep_proj(t.to(dtype) * 1000, self.timestep_channels).to(dtype))

    def forward(self, hidden_states: torch.Tensor, timestep, encoder_hidden_states: torch.Tensor,
                encoder_attention_mask: torch.Tensor | None = None, skip_layers=None,
                target_timestep=None) -> torch.Tensor:
        """timestep: noise level in 0..1000 (sigma * 1000); target_timestep (flow-map models only, the same units):
        the noise level this step jumps to. Returns the predicted velocity in the input's layout."""
        skip_layers = frozenset(skip_layers or ())
        if hidden_states.ndim != 4 or hidden_states.shape[1] != self.latent_channels:
            raise ValueError(f"hidden_states must be [B, {self.latent_channels}, H/16, W/16]")
        if encoder_hidden_states.ndim != 3 or encoder_hidden_states.shape[2] != self.text_dim:
            raise ValueError(f"encoder_hidden_states must be [B, S, {self.text_dim}]")
        if (target_timestep is not None) != self.flow_map:
            raise ValueError("target_timestep is needed by, and only by, a flow-map (distilled) FLUX 3 Klein; the "
                             "distilled model samples only: train on flux-3-image-klein-base")
        batch, _, height, width = hidden_states.shape
        device = hidden_states.device
        dtype = _wdtype(self.emb_in["image"])
        text_len = encoder_hidden_states.shape[1]

        def level(t):
            return torch.as_tensor(t, device=device, dtype=torch.float32).reshape(-1).expand(batch) / 1000.0

        img_temb = self._temb(level(timestep), dtype, self.time_in)[:, None]
        zero_temb = self._temb(torch.zeros(batch, device=device), dtype, self.time_in)[:, None]
        if self.flow_map:
            target = self._temb(level(target_timestep), dtype, self.ref_time_in)[:, None]
            img_temb, zero_temb = img_temb + target, zero_temb + target
        img_act, zero_act = F.silu(img_temb), F.silu(zero_temb)
        txt_early, txt_single = self.early_stream_modulations["txt"](zero_act), \
            self.single_stream_modulations["txt"](zero_act)
        img_early, img_single = self.early_stream_modulations["image"](img_act), \
            self.single_stream_modulations["image"](img_act)

        txt_cos, txt_sin = rope_tables(text_ids(text_len, device), self.axes_dims, self.rope_theta)
        img_cos, img_sin = rope_tables(image_ids(height, width, self.image_rope_scale, device), self.axes_dims,
                                       self.rope_theta)
        cos, sin = torch.cat([txt_cos, img_cos]), torch.cat([txt_sin, img_sin])

        valid, txt_mask, joint_mask = None, None, None
        if encoder_attention_mask is not None:
            valid = encoder_attention_mask.to(device=device, dtype=torch.bool)
            if not bool(valid.all()):  # the batch's padding rows: no sample's real query sees them
                txt_mask = valid[:, None, None, :]
                joint_mask = torch.cat([valid, valid.new_ones(batch, height * width)], dim=1)[:, None, None, :]

        txt = self.txt_in(encoder_hidden_states.to(device, dtype))
        img = self.emb_in["image"](hidden_states.flatten(2).transpose(1, 2).to(dtype))
        for block in self.txt_mode_blocks:
            txt = self._run(block, txt, txt_early, txt_cos, txt_sin, txt_mask)
        for block in self.content_mode_blocks["image"]:
            img = self._run(block, img, img_early, img_cos, img_sin, None)

        split = getattr(self, "split_devices", None)  # joint blocks may live on several devices (model split)
        taps = getattr(self, "_rz_taps", None)  # rulztrainer.alignment (training-only losses)
        if taps is not None and taps.armed:
            taps.set_layout(0, (text_len, height * width), (0, text_len), valid)
        for i, block in enumerate(self.single_blocks):
            if i in skip_layers:
                continue
            if split:
                bdev = _module_device(block)
                if bdev is not None and img.device != bdev:
                    img, txt, img_single, txt_single = img.to(bdev), txt.to(bdev), img_single.to(bdev), \
                        txt_single.to(bdev)
                    cos, sin = cos.to(bdev), sin.to(bdev)
                    joint_mask = joint_mask.to(bdev) if joint_mask is not None else None
            txt, img = self._run(block, img, txt, img_single, txt_single, cos, sin, joint_mask)
            if taps is not None and taps.armed:
                taps.record_hidden(i, img)

        if split:
            fdev = _module_device(self.final_layer)
            if fdev is not None and img.device != fdev:
                img = img.to(fdev)
        out = self.final_layer["image"](img, img_temb.to(img.device))
        return out.transpose(1, 2).reshape(batch, -1, height, width).to(device)
