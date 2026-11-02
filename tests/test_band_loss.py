"""Band loss (losses.py): plain MSE near sigma 0, buried fine bands dropped at high sigma, never above MSE."""
import pytest
import torch

from rulztrainer import losses as L
from rulztrainer.config import TrainConfig


def _batch(n=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    # smooth "clean latents" (power in the coarse bands) and white noise (power everywhere, mostly fine)
    lat = torch.nn.functional.interpolate(torch.randn(n, 16, 8, 8, generator=g), size=(32, 32), mode="bilinear")
    noise = torch.randn(n, 16, 32, 32, generator=g)
    pred = torch.randn(n, 16, 32, 32, generator=g)
    target = noise - lat
    return pred, target, lat, noise


def test_bands_sum_back():
    x = torch.randn(2, 4, 20, 28)
    assert torch.allclose(sum(L.bands(x)), x, atol=1e-5)  # odd-sized pools (ceil_mode) too


def test_band_loss_is_mse_at_low_noise():
    pred, target, lat, noise = _batch()
    sigma = torch.full((4,), 1e-4)
    got = L.band_loss(pred, target, lat, noise, sigma)
    assert torch.allclose(got, (pred - target).pow(2), rtol=1e-3, atol=1e-4)


def test_band_loss_drops_fine_bands_at_high_noise():
    pred, target, lat, noise = _batch()
    w = L.band_weights(lat, noise, torch.tensor([0.05, 0.5, 0.9, 0.99]))
    assert w.shape == (4, L.BAND_LEVELS + 1)
    assert torch.all(w[:, :-1] <= w[:, 1:] + 1e-6)  # coarse bands (smooth latents) survive longer than fine ones
    assert torch.all(w[:-1] >= w[1:])  # every band fades as sigma rises
    assert w[-1, 0] < 0.01 and w[0, -1] > 0.99
    mse = (pred - target).pow(2).mean(dim=(1, 2, 3))
    band = L.band_loss(pred, target, lat, noise, torch.tensor([0.05, 0.5, 0.9, 0.99])).mean(dim=(1, 2, 3))
    assert torch.all(band <= mse + 1e-5) and band[-1] < 0.2 * mse[-1]


def test_band_loss_trains_like_a_loss():
    pred, target, lat, noise = _batch()
    pred.requires_grad_(True)
    L.elementwise_loss("band", pred, target, lat, noise, torch.full((4,), 0.3)).mean().backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all() and pred.grad.abs().sum() > 0


def test_loss_type_validated():
    TrainConfig(loss_type="band").validate()
    with pytest.raises(ValueError):
        TrainConfig(loss_type="huber").validate()
    with pytest.raises(ValueError):
        L.elementwise_loss("huber", *_batch(), torch.full((4,), 0.3))
