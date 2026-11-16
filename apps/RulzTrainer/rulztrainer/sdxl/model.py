"""SDXL's UNet behind the trainer's flow-matching convention.

Every other family is a flow-matching transformer: the trainer noises with `x_s = (1-s) x0 + s eps`, asks the model for
the velocity `eps - x0` at `t1000 = 1000 s`, and builds loss, region maps, identity loss, CFG and the Euler sampler on
that. SDXL is a DDPM UNet (epsilon, or v prediction in some finetunes, on a 1000-step scaled-linear schedule). The
facade converts at the boundary, in fp32, so nothing in the trainer needs a second path:

    flow -> variance preserving: with a = (1-s) k, b = s k, k = 1 / sqrt((1-s)^2 + s^2) (so a^2 + b^2 = 1 and
    b / a = s / (1-s)), the VP input is x_t = a x0 + b eps = k x_flow, and alphabar = a^2.
    alphabar -> a continuous DDPM timestep: log-interpolation over the schedule's 1000 alphabar values (an integer
    timestep, which is what the trainer's reference draw gives, is fed as exactly that integer).
    epsilon model: x0 = (x_t - b eps) / a.    v model: x0 = a x_t - b v, eps = b x_t + a v.
    The velocity is eps - x0, re-patchified.

Euler in flow space on this facade is DDIM, and classifier-free guidance on the velocity equals guidance on epsilon
(the two terms' weights sum to 1).

Latents are the trainer's 2x2-patchified layout [B, 16, H/16, W/16]; the 16 features are (channel, dy, dx), the order
every other family's patchify uses. The text input is [B, 78, D]: rows 0..76 are the two encoders' penultimate hidden
states (CLIP-L || bigG, D = 2048), row 77 is bigG's pooled (projected) text embedding, zero-padded to D.
"""
from __future__ import annotations

import functools
import json
import math
from pathlib import Path

import torch
from torch import nn

NUM_TRAIN_TIMESTEPS = 1000
BETA_START, BETA_END = 0.00085, 0.012
POOLED_ROW, TEXT_ROWS, TEXT_DIM = 77, 78, 2048
HERE = Path(__file__).resolve().parent
UNET_CONFIG_FILE = HERE / "unet_config.json"
SNAP = 1e-2  # a continuous timestep this close to an integer is fed as that integer


def unet_config() -> dict:
    return json.loads(UNET_CONFIG_FILE.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- the schedule


def _table(zero_terminal_snr: bool) -> torch.Tensor:
    betas = torch.linspace(math.sqrt(BETA_START), math.sqrt(BETA_END), NUM_TRAIN_TIMESTEPS, dtype=torch.float64) ** 2
    ac = torch.cumprod(1.0 - betas, dim=0)
    if zero_terminal_snr:  # Lin et al. 2023, rescale_zero_terminal_snr: the last alphabar becomes 0
        root = ac.sqrt()
        first, last = root[0].clone(), root[-1].clone()
        root = (root - last) * (first / (first - last))
        ac = root**2
    return ac


@functools.lru_cache(maxsize=2)
def _cached_table(zero_terminal_snr: bool) -> torch.Tensor:
    return _table(bool(zero_terminal_snr))


def alphas_cumprod(zero_terminal_snr: bool = False) -> torch.Tensor:
    """alphabar for timesteps 0..999, float64 (a fresh copy)."""
    return _cached_table(bool(zero_terminal_snr)).clone()


def _interp_alphabar(t: torch.Tensor, ac: torch.Tensor) -> torch.Tensor:
    """alphabar at (fractional) timesteps t: geometric between the two table entries, linear where one is 0."""
    t = t.clamp(0, NUM_TRAIN_TIMESTEPS - 1)
    lo = t.floor().long()
    hi = (lo + 1).clamp(max=NUM_TRAIN_TIMESTEPS - 1)
    frac = t - lo.to(t.dtype)
    a_lo, a_hi = ac[lo], ac[hi]
    positive = (a_lo > 0) & (a_hi > 0)
    geo = torch.exp(torch.log(a_lo.clamp_min(1e-300)) * (1 - frac) + torch.log(a_hi.clamp_min(1e-300)) * frac)
    lin = a_lo * (1 - frac) + a_hi * frac
    return torch.where(positive, geo, lin)


def sigma_of_timestep(t, zero_terminal_snr: bool = False) -> torch.Tensor:
    """The flow noise level s = b / (a + b) of DDPM timestep(s) t (a = sqrt(alphabar), b = sqrt(1 - alphabar)), fp32.
    t may be fractional (log-interpolated alphabar)."""
    t = torch.as_tensor(t)
    ac = _cached_table(bool(zero_terminal_snr)).to(t.device)
    t64 = t.to(torch.float64)
    alpha = _interp_alphabar(t64, ac)
    a, b = alpha.sqrt(), (1.0 - alpha).clamp_min(0).sqrt()
    return (b / (a + b)).to(torch.float32)


def timestep_of_sigma(s, zero_terminal_snr: bool = False) -> torch.Tensor:
    """Continuous inverse of sigma_of_timestep, fp32, clamped to [0, 999] (a sigma outside the schedule's range maps
    to the nearest end)."""
    s = torch.as_tensor(s)
    ac = _cached_table(bool(zero_terminal_snr)).to(s.device)
    s64 = s.to(torch.float64)
    one_minus = 1.0 - s64
    x = one_minus**2 / (one_minus**2 + s64**2)  # alphabar = a^2 for b / a = s / (1 - s)
    asc = ac.flip(0)  # ascending
    n = NUM_TRAIN_TIMESTEPS
    idx = torch.searchsorted(asc.contiguous(), x.contiguous()).clamp(1, n - 1)
    lo, hi = asc[idx - 1], asc[idx]
    positive = lo > 0
    geo_frac = (x.clamp_min(1e-300).log() - lo.clamp_min(1e-300).log()) / (hi.log() - lo.clamp_min(1e-300).log())
    lin_frac = (x - lo) / (hi - lo)
    frac = torch.where(positive, geo_frac, lin_frac).clamp(0, 1)
    t_asc = (idx - 1).to(torch.float64) + frac
    return ((n - 1) - t_asc).to(torch.float32)


def sigma_bounds(prediction_type: str = "epsilon", zero_terminal_snr: bool = False) -> tuple[float, float]:
    """(s(t=0), s(t=999)): the noise levels the model's schedule covers; the top is 1.0 with zero terminal SNR."""
    lo = float(sigma_of_timestep(0, zero_terminal_snr))
    hi = 1.0 if zero_terminal_snr else float(sigma_of_timestep(NUM_TRAIN_TIMESTEPS - 1, False))
    return lo, hi


def init_noise_scale(sigma0: float) -> float:
    """The flow-unit scale of the VP start noise at sigma0: x_flow = x_t / k = sqrt((1-s0)^2 + s0^2) x_t."""
    return math.sqrt((1.0 - sigma0) ** 2 + sigma0**2)


def flow_to_vp(s: torch.Tensor):
    """(k, a, b) for noise levels s: x_t = k x_flow = a x0 + b eps, a^2 + b^2 = 1."""
    k = torch.rsqrt((1.0 - s) ** 2 + s**2)
    return k, (1.0 - s) * k, s * k


# ---------------------------------------------------------------- patchify


def patchify(lat: torch.Tensor) -> torch.Tensor:
    """[B, C, H, W] -> [B, 4C, H/2, W/2], the features ordered (channel, dy, dx)."""
    b, c, h, w = lat.shape
    return lat.reshape(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(b, c * 4, h // 2, w // 2)


def unpatchify(lat: torch.Tensor) -> torch.Tensor:
    """Exact inverse of patchify: [B, 4C, h, w] -> [B, C, 2h, 2w]."""
    b, c, h, w = lat.shape
    return lat.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c // 4, 2 * h, 2 * w)


# ---------------------------------------------------------------- the facade


def _collect_layers(unet) -> list[nn.Module]:
    """The BasicTransformerBlocks in compute order: down blocks, mid block, up blocks (70 for SDXL base)."""
    blocks = []
    for stage in (*unet.down_blocks, unet.mid_block, *unet.up_blocks):
        for attn in getattr(stage, "attentions", None) or ():
            blocks.extend(getattr(attn, "transformer_blocks", ()))
    return blocks


class SdxlUNet(nn.Module):
    def __init__(self, unet=None, prediction_type: str = "epsilon", zero_terminal_snr: bool = False, config=None):
        super().__init__()
        if prediction_type not in ("epsilon", "v_prediction"):
            raise ValueError(f"prediction_type must be 'epsilon' or 'v_prediction', not {prediction_type!r}")
        if unet is None:
            from diffusers import UNet2DConditionModel

            unet = UNet2DConditionModel.from_config(dict(config) if config is not None else unet_config())
        self.unet = unet
        self.prediction_type = prediction_type
        self.zero_terminal_snr = bool(zero_terminal_snr)
        cfg = unet.config
        self.vae_channels = int(cfg.in_channels)
        self.text_dim = int(cfg.cross_attention_dim)
        self.pooled_dim = int(cfg.projection_class_embeddings_input_dim) - 6 * int(cfg.addition_time_embed_dim)
        self.latent_channels = self.vae_channels * 4  # 16: 4 VAE channels x 2x2, the trainer's patchified layout
        # a plain list: not registered, so state dicts and module walks stay the UNet's; the same object every call
        self.layers = _collect_layers(unet)

    # ---- trainer-facing properties
    @property
    def gradient_checkpointing(self) -> bool:
        return bool(getattr(self.unet, "is_gradient_checkpointing", False))

    @gradient_checkpointing.setter
    def gradient_checkpointing(self, value: bool) -> None:
        if value:
            self.unet.enable_gradient_checkpointing()
        else:
            self.unet.disable_gradient_checkpointing()

    def _first_param(self):
        for p in self.unet.parameters():
            return p
        return torch.empty(0)

    @property
    def dtype(self) -> torch.dtype:
        return self._first_param().dtype

    @property
    def device(self) -> torch.device:
        return self._first_param().device

    def split_tail_modules(self) -> list[nn.Module]:
        """Modules that run after the last block."""
        return [self.unet.conv_norm_out, self.unet.conv_out]

    # ---- conversion
    def _timesteps(self, s: torch.Tensor) -> torch.Tensor:
        t = timestep_of_sigma(s, self.zero_terminal_snr)
        near = (t - t.round()).abs() < SNAP
        return torch.where(near, t.round(), t)

    def forward(self, latents: torch.Tensor, t1000, embeds: torch.Tensor, mask=None, **kw) -> torch.Tensor:
        """latents [B, 16, H/16, W/16] (flow-noised), t1000 (scalar or [B], 1000 s), embeds [B, 78, D] ->
        the flow velocity eps - x0, patchified, fp32."""
        skip = kw.pop("skip_layers", None)
        if skip:
            raise ValueError("SDXL cannot skip layers (skip_layer is not supported)")
        if latents.ndim != 4 or latents.shape[1] != self.latent_channels:
            raise ValueError(f"latents must be [B, {self.latent_channels}, H/16, W/16]")
        if embeds.ndim != 3 or embeds.shape[1] != TEXT_ROWS:
            raise ValueError(f"embeds must be [B, {TEXT_ROWS}, D] (77 text rows + the pooled row), got "
                             f"{tuple(embeds.shape)}")
        batch = latents.shape[0]
        dev, dt = latents.device, self.dtype
        x_flow = unpatchify(latents.float())
        s = torch.as_tensor(t1000, device=dev).float().reshape(-1) / 1000.0
        s = s.expand(batch).contiguous()
        k, a, b = flow_to_vp(s)
        k4, a4, b4 = (v.view(-1, 1, 1, 1) for v in (k, a, b))
        x_t = k4 * x_flow
        t = self._timesteps(s).to(dev)  # fp32 throughout
        pooled = embeds[:, POOLED_ROW, : self.pooled_dim].to(dt)
        h, w = x_flow.shape[-2] * 8, x_flow.shape[-1] * 8
        ids = torch.tensor([h, w, 0, 0, h, w], dtype=torch.float32, device=dev).expand(batch, 6)
        out = self.unet(x_t.to(dt), t, encoder_hidden_states=embeds[:, :POOLED_ROW, : self.text_dim].to(dt),
                        added_cond_kwargs={"text_embeds": pooled, "time_ids": ids}, return_dict=False)[0].float()
        if self.prediction_type == "epsilon":
            eps = out
            x0 = (x_t - b4 * eps) / a4
        else:
            x0 = a4 * x_t - b4 * out
            eps = b4 * x_t + a4 * out
        return patchify(eps - x0)
