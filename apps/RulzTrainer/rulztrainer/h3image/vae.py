"""MiniMax-H3-Image's VAE: MiniMax H3's video VAE encoder run over one frame, with an image-only 2D ViT decoder
(diffusers' AutoencoderKLMiniMaxH3Image, Copyright 2026 The MiniMax and HuggingFace Teams, Apache-2.0; ComfyUI reads
the same file).

The encoder and quant_conv are H3's, weight for weight (Comfy-Org's image and video VAE files were compared on the
first and last encoder convolutions, a ResNet convolution and quant_conv: identical), and so are the latent statistics.
The decoder is its own: 24 blocks of H3's decoder block at width 1024 (16 heads of 64), a rotary embedding over (h, w)
only (half of each head, 8 frequencies per axis, base 100; H3's video decoder rotates three axes), post_quant_conv a
2D convolution, no mask token, and each latent cell decoded to one 16 x 16 pixel block. A latent therefore decodes
straight to an image.

It is built from the H3 classes diffusers already ships (0.40 and later) with the decoder swapped, as P-H3-Image's is
(ph3/vae.py), so no newer diffusers is needed. The decoder runs its blocks itself (the block's own math) so the
residual stream stays in the compute dtype with fp32 norms and scales. Decoding is tiled as released (256 px tiles,
overlaps of at least 64 px, blended), by diffusers' own _split_tiles / _stitch_tiles."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..ph3.vae import SPATIAL_PATCH, param_dtype

# the released decoder (MiniMaxAI/MiniMax-H3-Image vae/config.json vit_decoder_kwargs, vae_diffusers/config.json)
DECODER_CONFIG = {"decoder_num_layers": 24, "decoder_num_attention_heads": 16, "decoder_attention_head_dim": 64,
                  "decoder_num_register_tokens": 4, "decoder_ffn_mult": 4, "decoder_rope_theta": 100.0,
                  "decoder_rope_dim_ratio": 0.5, "decoder_norm_eps": 1e-05}
_BUILD_ROPE_RATIO = 0.75  # H3's own ratio: what its 3-axis decoder is constructed with before the rope is replaced

_DECODER_CLS = None


class Rope2d(nn.Module):
    """diffusers' MiniMaxH3ImageRotaryPosEmbed with num_axes 2: positions [B, L, 2] in [-1, 1) -> cos / sin
    [B, L, 1, dim], the (h, w) angles concatenated and then duplicated (rotate-half)."""

    def __init__(self, dim: int, theta: float = 100.0):
        super().__init__()
        if dim % 4:
            raise ValueError(f"the rotary size {dim} must be a multiple of 4 (2 axes)")
        self.register_buffer("inv_freq", 1.0 / theta ** torch.arange(0, 1, 4 / dim, dtype=torch.float32),
                             persistent=False)

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        angles = 2.0 * math.pi * position_ids[:, :, :, None] * self.inv_freq.to(position_ids.device)[None, None, None]
        angles = angles.flatten(2, 3).tile(2).unsqueeze(2)
        return angles.cos(), angles.sin()


def decoder_class():
    """MiniMaxH3ImageViTDecoder2d on top of diffusers' MiniMaxH3VideoViTDecoder3d (built once, on first use)."""
    global _DECODER_CLS
    if _DECODER_CLS is not None:
        return _DECODER_CLS
    try:
        from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import MiniMaxH3VideoViTDecoder3d
    except ImportError as e:
        raise ImportError("MiniMax-H3-Image's VAE needs diffusers 0.41 or later: run the installer again (or pip "
                          "install -U diffusers)") from e

    class H3ImageViTDecoder2d(MiniMaxH3VideoViTDecoder3d):
        """Latents [B, C, h, w] -> an image [B, 3, 16h, 16w]."""

        def __init__(self, rope_dim: int, rope_theta: float, **kwargs):
            kwargs["patch_size_t"] = 1
            kwargs["rope_dim_ratio"] = _BUILD_ROPE_RATIO
            kwargs["rope_theta"] = rope_theta
            super().__init__(**kwargs)
            self.rope = Rope2d(rope_dim, rope_theta)

        @staticmethod
        def _block(block, x, rotary):
            dtype = block.attn.to_q.weight.dtype
            h = block.norm1(x.float()).to(dtype)
            x = x + (block.attn(h, rotary) * block.scale1).to(x.dtype)
            h = block.norm2(x.float()).to(dtype)
            return x + (block.ff(h) * block.scale2).to(x.dtype)

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            batch, channels, height, width = hidden_states.shape
            dtype = self.proj_in.weight.dtype
            x = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
            x = self.proj_in(x.to(dtype))
            n = x.shape[1]
            registers = self.register_tokens.expand(batch, -1, -1).to(x.dtype)
            x = torch.cat([x, registers, torch.zeros_like(x[:, :1])], dim=1)
            device = x.device
            grids = [2.0 * (torch.arange(0.5, size, dtype=torch.float32, device=device) / size) - 1.0
                     for size in (height, width)]
            pos = torch.stack(torch.meshgrid(*grids, indexing="ij"), dim=-1).flatten(0, 1)
            pos = pos.unsqueeze(0).expand(batch, -1, -1)
            pos = torch.cat([pos, pos.new_zeros(batch, self.num_register_tokens + 1, 2)], dim=1)
            rotary = self.rope(pos)
            for block in self.transformer_blocks:
                if self.gradient_checkpointing and torch.is_grad_enabled():
                    x = checkpoint(self._block, block, x, rotary, use_reentrant=False)
                else:
                    x = self._block(block, x, rotary)
            x = self.norm_out(x.float()).to(self.proj_out.weight.dtype)
            x = self.proj_out(x)[:, :n]
            p = self.patch_size
            x = x.view(batch, height, width, self.out_channels, p, p).permute(0, 3, 1, 4, 2, 5)
            return x.reshape(batch, self.out_channels, height * p, width * p)

    _DECODER_CLS = H3ImageViTDecoder2d
    return _DECODER_CLS


def decoder_config_from_state(state: dict, config: dict | None = None) -> dict:
    """The decoder's shape from a diffusers-layout state (its width, depth, register tokens, feed-forward ratio); the
    head size, rotary settings and epsilon are config (the released DECODER_CONFIG unless `config` names them)."""
    cfg = {**DECODER_CONFIG, **{k: v for k, v in (config or {}).items() if k in DECODER_CONFIG}}
    width = state["decoder.proj_in.weight"].shape[0]
    head_dim = int(cfg["decoder_attention_head_dim"])
    if width % head_dim:
        raise ValueError(f"the decoder is {width} wide, not a multiple of its head size {head_dim}")
    cfg["decoder_num_attention_heads"] = width // head_dim
    cfg["decoder_num_layers"] = sum(1 for k in state if k.startswith("decoder.transformer_blocks.")
                                    and k.endswith(".scale1"))
    cfg["decoder_num_register_tokens"] = state["decoder.register_tokens"].shape[1]
    cfg["decoder_ffn_mult"] = state["decoder.transformer_blocks.0.ff.net.2.weight"].shape[1] // width
    return cfg


def build_vae(state: dict, config: dict, dtype=torch.bfloat16):
    """A diffusers-layout state (encoder, quant_conv, the 2D post_quant_conv and the decoder under H3's diffusers key
    names) into the MiniMax-H3-Image VAE: diffusers' AutoencoderKLMiniMaxH3 with the decoder and post_quant_conv
    swapped, built on meta and assigned. config: H3's encoder config and latent statistics plus decoder_* (see
    decoder_config_from_state)."""
    from ..h3.single_file import vae_class

    cfg = dict(config)
    host = {k: v for k, v in cfg.items() if not k.startswith("decoder_")}
    with torch.device("meta"):
        # the host's own video decoder is a throwaway: one block of it is built, then replaced
        vae = vae_class()(**host, decoder_num_layers=1, decoder_num_attention_heads=cfg["decoder_num_attention_heads"],
                          decoder_attention_head_dim=cfg["decoder_attention_head_dim"])
        vae.decoder = decoder_class()(
            rope_dim=int(cfg["decoder_attention_head_dim"] * cfg["decoder_rope_dim_ratio"]),
            rope_theta=cfg["decoder_rope_theta"], in_channels=cfg["latent_channels"],
            out_channels=cfg.get("out_channels", 3), patch_size=SPATIAL_PATCH, num_layers=cfg["decoder_num_layers"],
            num_attention_heads=cfg["decoder_num_attention_heads"],
            attention_head_dim=cfg["decoder_attention_head_dim"],
            num_register_tokens=cfg["decoder_num_register_tokens"], ffn_mult=cfg["decoder_ffn_mult"],
            norm_eps=cfg["decoder_norm_eps"])
        vae.post_quant_conv = nn.Conv2d(cfg["latent_channels"], cfg["latent_channels"], kernel_size=1)
    want = vae.state_dict()
    missing, unexpected = sorted(set(want) - set(state)), sorted(set(state) - set(want))
    if missing or unexpected:
        raise ValueError(f"MiniMax-H3-Image VAE state mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    fitted = {}
    for k, v in state.items():
        if tuple(v.shape) != tuple(want[k].shape):
            raise ValueError(f"MiniMax-H3-Image VAE {k}: {tuple(v.shape)} where {tuple(want[k].shape)} is expected")
        fitted[k] = v.to(param_dtype(k, dtype))
    vae.load_state_dict(fitted, strict=True, assign=True)
    # the rotary frequencies are computed, not stored: built again off the meta device
    vae.decoder.rope = Rope2d(int(cfg["decoder_attention_head_dim"] * cfg["decoder_rope_dim_ratio"]),
                              cfg["decoder_rope_theta"])
    for name, b in list(vae.named_buffers()):
        if b.device.type == "meta":
            raise ValueError(f"MiniMax-H3-Image VAE buffer {name} was not built")
    vae.image_decoder_config = {k: cfg[k] for k in DECODER_CONFIG}
    return vae


def decode_image(vae, z: torch.Tensor) -> torch.Tensor:
    """Denormalized latents [B, 24, h, w] -> ImageNet-space images [B, 3, 16h, 16w]: diffusers'
    AutoencoderKLMiniMaxH3Image._decode, tiled when the VAE's tiling is on (as released)."""
    def one(tile):
        return vae.decoder(vae.post_quant_conv(tile))

    if not getattr(vae, "use_tiling", True):
        return one(z)
    ratio = vae.spatial_compression_ratio
    height, width = z.shape[-2] * ratio, z.shape[-1] * ratio
    ys, y_len, y_over = vae._split_tiles(height, vae.tile_sample_min_height, vae.tile_sample_min_overlap_height)
    xs, x_len, x_over = vae._split_tiles(width, vae.tile_sample_min_width, vae.tile_sample_min_overlap_width)
    rows = [[one(z[..., i // ratio: i // ratio + il // ratio, j // ratio: j // ratio + jl // ratio])
             for j, jl in zip(xs, x_len)] for i, il in zip(ys, y_len)]
    return vae._stitch_tiles(rows, y_over, x_over)
