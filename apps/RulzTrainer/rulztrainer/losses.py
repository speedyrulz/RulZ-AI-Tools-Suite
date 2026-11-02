"""The diffusion loss per latent element: plain MSE, or band loss (ported from Ostris's AI-Toolkit, experimental).

Band loss splits the residual (pred - target) into Laplacian frequency bands and weights each band by how
recoverable it is at the sample's noise level: snr_k / (1 + snr_k), with snr_k = snr(sigma) x the band's power in the
clean latents / its power in the noise (both measured over the batch). Near sigma 0 every weight is 1 and this is the
plain MSE; at high sigma the fine bands the noise has buried drop out instead of adding gradient noise. The filter is
linear, so it applies to the velocity residual as it is. It never weighs more than MSE: the loss (and the logged
value) runs lower, most at high noise.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

LOSS_TYPES = ("mse", "band")
BAND_LEVELS = 3  # Laplacian levels (fine to coarse) plus the remaining low-pass


def _lows(x: torch.Tensor, levels: int) -> list[torch.Tensor]:
    """Gaussian-pyramid low-passes of (N, C, H, W) at the input size: lows[0] = x, lows[k] = x averaged over
    2^k x 2^k cells and resampled back bilinearly."""
    lows = [x]
    cur = x
    for _ in range(levels):
        cur = F.avg_pool2d(cur, 2, ceil_mode=True)
        lows.append(F.interpolate(cur, size=x.shape[-2:], mode="bilinear", align_corners=False))
    return lows


def bands(x: torch.Tensor, levels: int = BAND_LEVELS) -> list[torch.Tensor]:
    """Laplacian bands of (N, C, H, W) at the input size, fine to coarse; they sum back to x."""
    lows = _lows(x, levels)
    return [lows[k] - lows[k + 1] for k in range(levels)] + [lows[levels]]


def band_weights(latents: torch.Tensor, noise: torch.Tensor, sigma: torch.Tensor,
                 levels: int = BAND_LEVELS) -> torch.Tensor:
    """(N, levels + 1) weights snr_k / (1 + snr_k) per sample and band (flow matching: snr = ((1 - s) / s)^2)."""
    with torch.no_grad():
        p_clean = torch.stack([b.pow(2).mean() for b in bands(latents.float(), levels)])  # over the batch: a prior,
        p_noise = torch.stack([b.pow(2).mean() for b in bands(noise.float(), levels)]).clamp_min(1e-12)  # not per image
        s = sigma.float().flatten().clamp(1e-4, 1 - 1e-4)
        snr = ((1 - s) / s) ** 2
        snr_k = snr.view(-1, 1) * (p_clean / p_noise).view(1, -1)
        return snr_k / (1 + snr_k)


def band_loss(pred: torch.Tensor, target: torch.Tensor, latents: torch.Tensor, noise: torch.Tensor,
              sigma: torch.Tensor, levels: int = BAND_LEVELS) -> torch.Tensor:
    """Squared band-weighted residual, pred's (N, C, H, W) shape (see the module docstring)."""
    pred, target = pred.float(), target.float()
    if pred.dim() != 4:
        return (pred - target).pow(2)
    w = band_weights(latents.to(pred.device), noise.to(pred.device), sigma.to(pred.device), levels)
    w = w.t().reshape(levels + 1, -1, 1, 1, 1)
    filtered = sum(w[k] * b for k, b in enumerate(bands(pred - target, levels)))
    return filtered.pow(2)


def elementwise_loss(loss_type: str, pred: torch.Tensor, target: torch.Tensor, latents: torch.Tensor,
                     noise: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """The training loss per latent element for loss_type ("mse" or "band")."""
    if loss_type == "mse":
        return (pred - target).pow(2)
    if loss_type == "band":
        return band_loss(pred, target, latents, noise, sigma)
    raise ValueError(f"Unknown loss_type {loss_type!r}: one of {LOSS_TYPES}")
