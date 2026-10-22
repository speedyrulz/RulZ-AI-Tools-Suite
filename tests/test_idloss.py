"""Identity loss (phase 5): x0 recovery, face crops, differentiable alignment, plumbing and the controller veto.
CPU only: tiny transformer, stub decoder and embedder."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import idloss as I  # noqa: E402
from rulztrainer import likeness as L  # noqa: E402
from rulztrainer import lora as lora_mod  # noqa: E402
from rulztrainer import regions as R  # noqa: E402
from rulztrainer.config import TrainConfig  # noqa: E402
from rulztrainer.moss.model import MOSSImageTransformer  # noqa: E402

TINY = dict(dim=64, ffn_dim=128, num_heads=2, num_layers=2, text_dim=32, freq_dim=16, latent_channels=128)
# a frontal face inside cells (2, 2)-(6, 6) of an 8 x 8 latent (128 x 128 px)
KPS = [[48.0, 56.0], [80.0, 56.0], [64.0, 72.0], [52.0, 86.0], [76.0, 86.0]]


def _meta(face_cells=(2, 2, 6, 6), kps=KPS):
    return R.RegionMap(torch.ones(1, 8, 8), torch.zeros(1, 8, 8, dtype=torch.bool), face_cells, kps, True)


def test_x0_recovery():
    torch.manual_seed(0)
    x0, noise = torch.randn(3, 128, 4, 6), torch.randn(3, 128, 4, 6)
    sigma = torch.tensor([0.1, 0.35, 0.9])
    s = sigma.view(-1, 1, 1, 1)
    noisy = (1 - s) * x0 + s * noise
    pred = noise - x0  # the flow-matching target
    assert torch.allclose(I.predicted_x0(noisy, pred, sigma), x0, atol=1e-5)


def test_face_crop_clipped_at_latent_border():
    assert I.face_crop_cells((2, 2, 6, 6), 2, 8, 8) == (0, 0, 8, 8)
    assert I.face_crop_cells((0, 1, 3, 5), 2, 6, 8) == (0, 0, 5, 6)
    assert I.face_crop_cells((6, 3, 8, 6), 1, 6, 8) == (5, 2, 8, 6)

    seen = []

    def decode(z):
        seen.append(tuple(z.shape))
        return torch.zeros(1, 3, z.shape[-2] * 16, z.shape[-1] * 16) + z.mean()

    emb = I.Embedder(nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 4)), 32)
    loss = I.IdentityLoss(emb, torch.randn(2, 4), decode, margin_cells=2, min_face_cells=2)
    noisy, pred = torch.randn(1, 128, 6, 8), torch.randn(1, 128, 6, 8)
    kps = [[x - 16, y] for x, y in KPS]
    val, n = loss(noisy, pred, torch.tensor([0.2]), [_meta((0, 1, 3, 5), kps)])
    assert n == 1 and torch.isfinite(val) and seen == [(1, 128, 6, 5)]  # rows 0-6 (clipped), cols 0-5


def _paint(kps, w, h):
    img = np.zeros((h, w, 3), np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    for k, (x, y) in enumerate(kps):
        img[..., k % 3] += np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 3.0 ** 2))
    return img


def test_alignment_theta_matches_cv2_warp():
    cv2 = pytest.importorskip("cv2")
    # a rotated, scaled face in a non-square image
    ang = np.deg2rad(20)
    rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    kps = (np.array(KPS) - 64) @ rot.T * 1.4 + np.array([110.0, 70.0])
    img = _paint(kps, 200, 150)
    M = I.estimate_norm(kps, 112)
    try:  # the same transform as insightface's estimate_norm (skimage SimilarityTransform)
        from insightface.utils.face_align import estimate_norm

        assert np.allclose(M, estimate_norm(kps.astype(np.float32), 112), atol=1e-3)
    except ImportError:
        pass
    ref = cv2.warpAffine(img, M, (112, 112), borderValue=0.0)
    theta = I.align_theta(M, 200, 150, 112)
    out = I.warp(torch.from_numpy(img).permute(2, 0, 1)[None], theta[None], 112)[0].permute(1, 2, 0).numpy()
    assert np.abs(out - ref).max() < 0.02
    # each painted landmark lands where M sends it, close to the ArcFace template (a least-squares fit)
    dst = np.hstack([kps, np.ones((5, 1))]) @ M.T
    assert np.abs(dst - I.ARCFACE_DST).max() < 4
    for k, (x, y) in enumerate(dst):
        assert out[int(round(y)), int(round(x)), k % 3] > 0.8


def test_flip_mirrors_landmark_order():
    rm = _meta()
    fl = R.flip_region_map(rm, 128)
    assert fl.kps[0] == [128 - KPS[1][0], KPS[1][1]] and fl.kps[3] == [128 - KPS[4][0], KPS[4][1]]
    assert fl.kps[0][0] < fl.kps[1][0]  # left eye still on the image left
    # the aligned flip is an upright face: similarity fit to the template with a small residual
    M = I.estimate_norm(fl.kps)
    pts = np.hstack([np.array(fl.kps), np.ones((5, 1))]) @ M.T
    assert np.abs(pts - I.ARCFACE_DST).max() < 6


def test_eligibility_and_every():
    emb = I.Embedder(nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 4)), 32)
    loss = I.IdentityLoss(emb, torch.randn(1, 4), None, sigma_max=0.35, every=2, min_face_cells=4)
    assert not loss.eligible(0.5, _meta())
    assert not loss.eligible(0.2, _meta((2, 2, 5, 6)))  # 3 cells wide
    assert not loss.eligible(0.2, _meta(None, None))
    assert loss.eligible(0.2, _meta())
    metas = [_meta(), _meta(), _meta((2, 2, 3, 3)), _meta()]
    assert loss.select([0.1, 0.1, 0.1, 0.1], metas) == [0, 3]  # 3 eligible: the 1st and the 3rd
    assert I.effective_weight(2.0) == 0.5 and I.effective_weight(0.1) == 0.1 and I.effective_weight(-1) == 0.0


class _Decoder(nn.Module):
    """Stand-in for the VAE: 1x1 conv to RGB and 16x nearest upsampling, differentiable."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(128, 3, 1)

    def forward(self, z):
        return torch.tanh(nn.functional.interpolate(self.conv(z), scale_factor=16, mode="nearest"))


def _tiny_step(embed_net, ref):
    torch.manual_seed(0)
    model = MOSSImageTransformer(**TINY)
    injected = lora_mod.inject_lora(model, rank=4, alpha=4, targets=["q", "k", "v", "o", "ffn_up_gate", "ffn_down"])
    for m in injected.values():  # non-zero up so every LoRA weight gets a gradient
        nn.init.normal_(m.lora_up.weight, std=0.01)
    dec = _Decoder().requires_grad_(False)
    emb = I.Embedder(embed_net, 112)
    loss_fn = I.IdentityLoss(emb, ref[None], dec, sigma_max=0.35, margin_cells=2, min_face_cells=4)
    latents, noise = torch.randn(2, 128, 8, 8), torch.randn(2, 128, 8, 8)
    sigma = torch.tensor([0.2, 0.9])  # the second sample is too noisy
    s = sigma.view(-1, 1, 1, 1)
    noisy = (1 - s) * latents + s * noise
    pred = model(noisy, sigma * 1000.0, torch.randn(2, 5, 32), torch.ones(2, 5, dtype=torch.bool)).float()
    val, n = loss_fn(noisy, pred, sigma, [_meta(), _meta()])
    return val, n, list(lora_mod.lora_parameters(injected))


def test_plumbing_lora_grads_finite():
    net = nn.Sequential(nn.Conv2d(3, 8, 3, stride=4), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                        nn.Linear(8, 16))
    val, n, params = _tiny_step(net, torch.nn.functional.normalize(torch.randn(16), dim=0))
    assert n == 1 and 0.0 <= float(val) <= 2.0
    (0.2 * val).backward()
    grads = [p.grad for p in params]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert sum(float(g.abs().sum()) for g in grads) > 0


def test_loss_zero_when_embedder_returns_ref():
    ref = torch.nn.functional.normalize(torch.randn(16), dim=0)

    class Constant(nn.Module):
        def forward(self, x):
            return ref[None].expand(x.shape[0], -1) + 0.0 * x.mean()

    val, n, _ = _tiny_step(Constant(), ref)
    assert n == 1 and abs(float(val)) < 1e-6


def _cfg(**kw):
    return TrainConfig(probe_every=10, probe_ema=0.0, likeness_noise_band=1.0, likeness_track_best=True, **kw)


def test_controller_id_loss_veto():
    ctrl = L.LikenessController(_cfg(id_loss_weight=0.1))
    vetoes = [ctrl.update(10 * k, f, None)[1].get("id_loss_veto") for k, f in enumerate([60, 70, 67.5, 70, 66, 65, 64])]
    # 67.5 and 66 are each a single probe more than 2 bands below the best (70): no veto; 66, 65 in a row: veto once
    assert vetoes == [None, None, None, None, None, True, None]
    assert ctrl.state()["id_loss_vetoed"]
    off = L.LikenessController(_cfg())  # id loss off: never vetoes
    assert not any(off.update(10 * k, f, None)[1].get("id_loss_veto") for k, f in enumerate([70, 60, 50, 40]))


def test_config_validates_backend():
    with pytest.raises(ValueError):
        TrainConfig(id_loss_backend="vgg").validate()
    assert not TrainConfig().wants_regions() and TrainConfig(id_loss_weight=0.1).wants_regions()
