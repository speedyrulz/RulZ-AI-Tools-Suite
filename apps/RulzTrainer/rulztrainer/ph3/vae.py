"""P-H3-Image's VAE: MiniMax H3's video VAE encoder with a single-frame ViT decoder (diffusers'
AutoencoderKLPH3Image, Copyright 2026 Pruna AI and The HuggingFace Team, Apache-2.0).

The encoder, the latent statistics and every decoder block are H3's. The decoder differs in three ways: each latent
cell decodes to one 16 x 16 pixel block of ONE frame (patch_size_t 1, so proj_out is 3 x 16 x 16 = 768 wide, where
H3's is 3 x 4 x 16 x 16), every token sits at one fixed rotary time coordinate (FRAME_POSITION), and there is no mask
token. A still therefore decodes directly: no repeat into a 5-latent clip, no pre-padding frames to drop.

It is built from the H3 classes diffusers already ships (0.40 and later), with the decoder swapped, so no newer
diffusers is needed. The decoder runs its blocks itself (the same math as the block's forward) so the residual stream
stays in the compute dtype with fp32 norms and scales, on every diffusers version (diffusers 0.40's H3 block mixes the
two and fails in bf16)."""
from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint

# the rotary time coordinate shared by every token of the decoded frame (diffusers' FRAME_POSITION)
FRAME_POSITION = 2.0 * (0.5 / 7) - 1.0
SPATIAL_PATCH = 16  # pixels per latent cell edge
# kept in fp32 (diffusers' _keep_in_fp32_modules of AutoencoderKLPH3Image: encoder, quant_conv, proj_out) plus the
# norms and residual scales, which the blocks apply in fp32
FP32_TOP = ("encoder", "quant_conv")
FP32_PARTS = ("norm1", "norm2", "norm_out", "scale1", "scale2")

_DECODER_CLS = None


def decoder_class():
    """PH3ImageViTDecoder on top of diffusers' MiniMaxH3VideoViTDecoder3d (built once, on first use)."""
    global _DECODER_CLS
    if _DECODER_CLS is not None:
        return _DECODER_CLS
    try:
        from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import MiniMaxH3VideoViTDecoder3d
    except ImportError as e:
        raise ImportError("P-H3-Image's VAE needs diffusers 0.40 or later: run the installer again (or pip install -U "
                          "diffusers)") from e

    class PH3ImageViTDecoder(MiniMaxH3VideoViTDecoder3d):
        """One latent frame [B, C, 1, h, w] -> one image [B, 3, 1, 16h, 16w]."""

        def __init__(self, **kwargs):
            kwargs["patch_size_t"] = 1
            super().__init__(**kwargs)

        @staticmethod
        def _block(block, x, rotary):
            dtype = block.attn.to_q.weight.dtype
            h = block.norm1(x.float()).to(dtype)
            x = x + (block.attn(h, rotary) * block.scale1).to(x.dtype)
            h = block.norm2(x.float()).to(dtype)
            return x + (block.ff(h) * block.scale2).to(x.dtype)

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            batch, channels, frames, height, width = hidden_states.shape
            if frames != 1:
                raise ValueError(f"P-H3-Image decodes single latent frames, got {frames}")
            dtype = self.proj_in.weight.dtype
            x = hidden_states.permute(0, 2, 3, 4, 1).reshape(batch, height * width, channels)
            x = self.proj_in(x.to(dtype))
            n = x.shape[1]
            registers = self.register_tokens.expand(batch, -1, -1).to(x.dtype)
            x = torch.cat([x, registers, torch.zeros_like(x[:, :1])], dim=1)
            device = x.device
            grids = [torch.full((1,), FRAME_POSITION, device=device)] + [
                2.0 * (torch.arange(0.5, size, dtype=torch.float32, device=device) / size) - 1.0
                for size in (height, width)]
            pos = torch.stack(torch.meshgrid(*grids, indexing="ij"), dim=-1).flatten(0, 2)
            pos = pos.unsqueeze(0).expand(batch, -1, -1)
            pos = torch.cat([pos, pos.new_zeros(batch, self.num_register_tokens + 1, 3)], dim=1)
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
            return x.reshape(batch, self.out_channels, 1, height * p, width * p)

    _DECODER_CLS = PH3ImageViTDecoder
    return _DECODER_CLS


def param_dtype(name: str, dtype) -> torch.dtype:
    parts = name.split(".")
    if parts[0] in FP32_TOP or name.startswith("decoder.proj_out.") or any(p in FP32_PARTS for p in parts):
        return torch.float32
    return dtype


def build_vae(state: dict, config: dict, dtype=torch.bfloat16):
    """A diffusers-layout state (H3's key names, single-frame decoder) into the P-H3 VAE: diffusers'
    AutoencoderKLMiniMaxH3 with the decoder swapped, built on meta and assigned."""
    from ..h3.single_file import vae_class

    cfg = dict(config)
    cfg["decoder_num_layers"] = int(cfg.get("decoder_num_layers", 36))
    with torch.device("meta"):
        vae = vae_class()(**cfg)
        vae.decoder = decoder_class()(
            in_channels=cfg["latent_channels"], out_channels=cfg.get("out_channels", 3), patch_size=SPATIAL_PATCH,
            num_layers=cfg["decoder_num_layers"], num_attention_heads=cfg["decoder_num_attention_heads"],
            attention_head_dim=cfg["decoder_attention_head_dim"],
            num_register_tokens=cfg["decoder_num_register_tokens"], ffn_mult=cfg["decoder_ffn_mult"],
            rope_theta=cfg["decoder_rope_theta"], rope_dim_ratio=cfg["decoder_rope_dim_ratio"],
            norm_eps=cfg["decoder_norm_eps"])
    want = vae.state_dict()
    missing, unexpected = sorted(set(want) - set(state)), sorted(set(state) - set(want))
    if missing or unexpected:
        raise ValueError(f"P-H3-Image VAE state mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    fitted = {}
    for k, v in state.items():
        if tuple(v.shape) != tuple(want[k].shape):
            raise ValueError(f"P-H3-Image VAE {k}: {tuple(v.shape)} where {tuple(want[k].shape)} is expected")
        fitted[k] = v.to(param_dtype(k, dtype))
    vae.load_state_dict(fitted, strict=True, assign=True)
    rope = vae.decoder.rope  # its inverse frequencies are computed, not stored: built again for real
    vae.decoder.rope = type(rope)(int(cfg["decoder_attention_head_dim"] * cfg["decoder_rope_dim_ratio"]),
                                  theta=cfg["decoder_rope_theta"])
    for name, b in list(vae.named_buffers()):
        if b.device.type == "meta":
            raise ValueError(f"P-H3-Image VAE buffer {name} was not built")
    return vae
