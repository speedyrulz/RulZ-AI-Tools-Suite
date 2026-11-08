"""FLUX.2 Klein 9B DiT for training, in RulzTrainer's conventions.

A port of Black Forest Labs' Flux2 transformer (flux2 repo, Apache-2.0; ai-toolkit and Fizgig carry the same model),
written so that:
  - the state-dict keys are BFL's own, the ones of the root flux-2-klein-*.safetensors files: double_blocks with the
    fused img_attn / txt_attn.qkv, proj and img_mlp / txt_mlp.0, .2, single_blocks with the fused linear1 ([q; k; v;
    MLP in]) and linear2 (attention and MLP out), img_in, txt_in, time_in, double_stream_modulation_img / _txt,
    single_stream_modulation, final_layer. A LoRA exported with these names under "diffusion_model." is what ai-toolkit
    saves, what ComfyUI loads and what diffusers' Flux2 LoRA loader converts. The diffusers folder (transformer/) is
    renamed and fused on load (flux2/single_file.py);
  - the forward takes what the trainer and sampler give the other families: `hidden_states` as the normalized,
    2x2-packed Flux2 VAE latents [B, 128, H/16, W/16], `timestep` in 0..1000 units (sigma * 1000), the text features
    as [B, 512, 12288] and a [B, S] mask (the model attends to every row, padding included, as BFL's does: the mask
    only tells the alignment taps which rows belong to a sample);
  - `layers` holds all 32 blocks in the order they run, double then single, so block selection, block swap, the model
    split and the alignment taps count one list: double 0-7 are blocks 0-7, single 0-23 are blocks 8-31. The list is
    not registered, so the blocks keep their BFL names;
  - gradient checkpointing and the model-split hand-off work like the other ports; `skip_layers` skips blocks (kept
    for the shared sampler signature);
  - QuantLinear / LoRALinear replace nn.Linear modules transparently.

Architecture (9B, 4096 wide, 32 heads of 128): 8 double blocks, each stream with its own attention projections and
SwiGLU MLP around one joint attention, then 24 single blocks over [text, image]. One timestep embedding drives three
modulations. Klein has no guidance embedding: Base is steered by CFG alone. RoPE: 4 axes (t, h, w, l) of 32 dims,
theta 2000; text token j at (0, 0, 0, j), image cell (row, col) at (0, row, col, 0).

Not ported: reference images (editing), which ride after the image tokens at time offsets 10, 20, ... Text-to-image
LoRAs do not need them, and the same weights serve both, so a LoRA trained here still loads for editing.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..attention import sdpa
from ..checkpointing import checkpointed
from ..compiling import thread_prepare
from ..flux3.model import (QKNorm, TimestepEmbedding, _module_device, _wdtype, apply_rope, image_ids, rope_tables,
                           text_ids, timestep_proj)

ROPE_THETA = 2000
EPS = 1e-6


def _rotate(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, norm: QKNorm, cos, sin):
    """Heads [B, L, H, D]: q / k RMS-normalized and rotated, in v's dtype."""
    q, k = norm.query_norm(q), norm.key_norm(k)
    return apply_rope(q, cos, sin).to(v.dtype), apply_rope(k, cos, sin).to(v.dtype)


class SiLUActivation(nn.Module):
    """SwiGLU: the input halves are gate and value."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = x.chunk(2, dim=-1)
        return F.silu(gate) * value


class SelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.norm = QKNorm(dim // heads, EPS)
        self.proj = nn.Linear(dim, dim, bias=False)


def _mlp(dim: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(dim, 2 * hidden, bias=False), SiLUActivation(), nn.Linear(hidden, dim, bias=False))


class DoubleBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 3.0):
        super().__init__()
        self.heads = heads
        hidden = int(dim * mlp_ratio)
        for side in ("img", "txt"):
            setattr(self, f"{side}_norm1", nn.LayerNorm(dim, elementwise_affine=False, eps=EPS))
            setattr(self, f"{side}_attn", SelfAttention(dim, heads))
            setattr(self, f"{side}_norm2", nn.LayerNorm(dim, elementwise_affine=False, eps=EPS))
            setattr(self, f"{side}_mlp", _mlp(dim, hidden))

    def _qkv(self, x, shift, scale, norm1, attn, cos, sin):
        h = (1 + scale) * norm1(x) + shift
        q, k, v = attn.qkv(h).unflatten(-1, (3, self.heads, -1)).unbind(2)
        q, k = _rotate(q, k, v, attn.norm, cos, sin)
        return q, k, v

    def forward(self, img, txt, img_mod, txt_mod, cos, sin):
        """img [B, I, dim], txt [B, T, dim]; mods [B, 1, 6 * dim]: attention then MLP (shift, scale, gate); cos / sin
        over [txt, img]."""
        i_s1, i_a1, i_g1, i_s2, i_a2, i_g2 = img_mod.chunk(6, dim=-1)
        t_s1, t_a1, t_g1, t_s2, t_a2, t_g2 = txt_mod.chunk(6, dim=-1)
        n = txt.shape[1]
        ti = self._qkv(txt, t_s1, t_a1, self.txt_norm1, self.txt_attn, cos[:n], sin[:n])
        ii = self._qkv(img, i_s1, i_a1, self.img_norm1, self.img_attn, cos[n:], sin[n:])
        q, k, v = (torch.cat([t, i], dim=1).transpose(1, 2) for t, i in zip(ti, ii))
        tap = getattr(self, "_rz_tap", None)  # rulztrainer.alignment: the trigger's attention (training-only loss)
        if tap is not None:
            tap(q, k)
        out = sdpa(q, k, v).transpose(1, 2).flatten(2, 3).to(img.dtype)
        img = img + i_g1 * self.img_attn.proj(out[:, n:])
        img = img + i_g2 * self.img_mlp((1 + i_a2) * self.img_norm2(img) + i_s2)
        txt = txt + t_g1 * self.txt_attn.proj(out[:, :n])
        txt = txt + t_g2 * self.txt_mlp((1 + t_a2) * self.txt_norm2(txt) + t_s2)
        return img, txt


class SingleBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 3.0):
        super().__init__()
        self.dim, self.heads = dim, heads
        self.hidden = int(dim * mlp_ratio)
        self.pre_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=EPS)
        self.linear1 = nn.Linear(dim, 3 * dim + 2 * self.hidden, bias=False)
        self.norm = QKNorm(dim // heads, EPS)
        self.linear2 = nn.Linear(dim + self.hidden, dim, bias=False)
        self.mlp_act = SiLUActivation()

    def forward(self, x, mod, cos, sin):
        """x [B, T + I, dim]; mod [B, 1, 3 * dim]."""
        shift, scale, gate = mod.chunk(3, dim=-1)
        h = (1 + scale) * self.pre_norm(x) + shift
        qkv, mlp = self.linear1(h).split([3 * self.dim, 2 * self.hidden], dim=-1)
        q, k, v = qkv.unflatten(-1, (3, self.heads, -1)).unbind(2)
        q, k = _rotate(q, k, v, self.norm, cos, sin)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        tap = getattr(self, "_rz_tap", None)
        if tap is not None:
            tap(q, k)
        out = sdpa(q, k, v).transpose(1, 2).flatten(2, 3).to(x.dtype)
        return x + gate * self.linear2(torch.cat([out, self.mlp_act(mlp)], dim=-1))


class Modulation(nn.Module):
    def __init__(self, dim: int, sets: int):
        super().__init__()
        self.lin = nn.Linear(dim, sets * dim, bias=False)

    def forward(self, vec: torch.Tensor) -> torch.Tensor:
        return self.lin(F.silu(vec))


class LastLayer(nn.Module):
    def __init__(self, dim: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(dim, elementwise_affine=False, eps=EPS)
        self.linear = nn.Linear(dim, out_channels, bias=False)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim, bias=False))

    def forward(self, x: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=-1)
        return self.linear((1 + scale) * self.norm_final(x) + shift)


class Flux2KleinTransformer(nn.Module):
    """FLUX.2 Klein (text-to-image). hidden_states: [B, 128, H/16, W/16] normalized Flux2 VAE latents."""

    CONFIG_KEYS = ("in_channels", "context_in_dim", "hidden_size", "num_heads", "depth", "depth_single_blocks",
                   "axes_dim", "theta", "mlp_ratio", "timestep_channels")

    def __init__(self, in_channels=128, context_in_dim=12288, hidden_size=4096, num_heads=32, depth=8,
                 depth_single_blocks=24, axes_dim=(32, 32, 32, 32), theta=ROPE_THETA, mlp_ratio=3.0,
                 timestep_channels=256):
        super().__init__()
        if hidden_size % num_heads or sum(axes_dim) != hidden_size // num_heads:
            raise ValueError(f"axes_dim {axes_dim} must add up to hidden_size / num_heads")
        self.config = dict(in_channels=in_channels, context_in_dim=context_in_dim, hidden_size=hidden_size,
                           num_heads=num_heads, depth=depth, depth_single_blocks=depth_single_blocks,
                           axes_dim=list(axes_dim), theta=theta, mlp_ratio=mlp_ratio,
                           timestep_channels=timestep_channels)
        self.latent_channels = in_channels
        self.text_dim = context_in_dim
        self.axes_dims = tuple(axes_dim)
        self.rope_theta = theta
        self.timestep_channels = timestep_channels
        self.gradient_checkpointing = False
        self.is_distilled = False  # the 4-step Klein: the loaders set it by file name (its keys are Base's)
        self.img_in = nn.Linear(in_channels, hidden_size, bias=False)
        self.time_in = TimestepEmbedding(timestep_channels, hidden_size)
        self.txt_in = nn.Linear(context_in_dim, hidden_size, bias=False)
        self.double_blocks = nn.ModuleList(DoubleBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth))
        self.single_blocks = nn.ModuleList(SingleBlock(hidden_size, num_heads, mlp_ratio)
                                           for _ in range(depth_single_blocks))
        self.double_stream_modulation_img = Modulation(hidden_size, 6)
        self.double_stream_modulation_txt = Modulation(hidden_size, 6)
        self.single_stream_modulation = Modulation(hidden_size, 3)
        self.final_layer = LastLayer(hidden_size, in_channels)
        # not a registered child (the blocks keep their BFL names); deepcopy keeps it on the copy's blocks
        object.__setattr__(self, "_chain", nn.ModuleList([*self.double_blocks, *self.single_blocks]))

    @property
    def layers(self) -> nn.ModuleList:
        """Every block in the order they run: double then single."""
        return self._chain

    def split_block_lists(self) -> list[nn.Module]:
        """The registered lists `layers` chains (a model split places their blocks one by one)."""
        return [self.double_blocks, self.single_blocks]

    def split_tail_modules(self) -> list[nn.Module]:
        """Modules that run after the last block (they follow it in a model split)."""
        return [self.final_layer]

    @classmethod
    def from_config(cls, config: dict) -> "Flux2KleinTransformer":
        return cls(**{k: config[k] for k in cls.CONFIG_KEYS if k in config})

    def _run(self, block, *args):
        thread_prepare()  # dynamo's limits in this thread (autograd recompute, data-parallel workers)
        if (self.gradient_checkpointing and self.training and torch.is_grad_enabled()
                and not getattr(block, "keep_activations", False)):  # see rulztrainer.checkpointing
            return checkpointed(self, block, *args)
        return block(*args)

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

        # the pipeline hands the model t / 1000 in the model's dtype, which multiplies it by 1000 in that dtype again
        t = torch.as_tensor(timestep, device=device, dtype=torch.float32).reshape(-1).expand(batch) / 1000.0
        vec = self.time_in(timestep_proj(t.to(dtype) * 1000, self.timestep_channels).to(dtype))[:, None]
        img_mod = self.double_stream_modulation_img(vec)
        txt_mod = self.double_stream_modulation_txt(vec)
        single_mod = self.single_stream_modulation(vec)

        txt_cos, txt_sin = rope_tables(text_ids(text_len, device), self.axes_dims, self.rope_theta)
        img_cos, img_sin = rope_tables(image_ids(height, width, 1.0, device), self.axes_dims, self.rope_theta)
        cos, sin = torch.cat([txt_cos, img_cos]), torch.cat([txt_sin, img_sin])

        img = self.img_in(hidden_states.flatten(2).transpose(1, 2).to(dtype))
        txt = self.txt_in(encoder_hidden_states.to(device, dtype))

        split = getattr(self, "split_devices", None)  # blocks may live on several devices (model split)
        taps = getattr(self, "_rz_taps", None)  # rulztrainer.alignment (training-only losses)
        if taps is not None and taps.armed:
            valid = None if encoder_attention_mask is None else encoder_attention_mask.to(device, dtype=torch.bool)
            taps.set_layout(0, (text_len, height * width), (0, text_len), valid)

        def to_block(block, *tensors):
            bdev = _module_device(block) if split else None
            if bdev is None or tensors[0].device == bdev:
                return tensors
            return tuple(t.to(bdev) for t in tensors)

        n_double = len(self.double_blocks)
        for i, block in enumerate(self.double_blocks):
            if i in skip_layers:
                continue
            img, txt, img_mod, txt_mod, cos, sin = to_block(block, img, txt, img_mod, txt_mod, cos, sin)
            img, txt = self._run(block, img, txt, img_mod, txt_mod, cos, sin)
            if taps is not None and taps.armed:
                taps.record_hidden(i, img)
        x = torch.cat([txt, img], dim=1)
        for i, block in enumerate(self.single_blocks):
            if i + n_double in skip_layers:
                continue
            x, single_mod, cos, sin = to_block(block, x, single_mod, cos, sin)
            x = self._run(block, x, single_mod, cos, sin)
            if taps is not None and taps.armed:
                taps.record_hidden(i + n_double, x[:, text_len:])

        img = x[:, text_len:]
        if split:
            fdev = _module_device(self.final_layer)
            if fdev is not None and img.device != fdev:
                img = img.to(fdev)
        out = self.final_layer(img, vec.to(img.device))
        return out.transpose(1, 2).reshape(batch, -1, height, width).to(device)
