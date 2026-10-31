"""AI JPEG artifact removal (the "Best quality" mode).

Uses FBCNN - "Towards Flexible Blind JPEG Artifacts Removal", Jiang et al., ICCV 2021
(https://github.com/jiaxi-jiang/FBCNN, Apache-2.0). The network below is a trimmed rebuild of
their models/network_fbcnn.py with identical layer names, so their released weights load as-is.
FBCNN takes a JPEG quality factor as a control: the lower it is, the more it cleans up; the
strength slider maps onto it.

PyTorch and the ~290 MB weights file are optional extras, set up from the app on first use.
"""
from __future__ import annotations

import gc
import hashlib
import importlib.util
import sys
import threading
from pathlib import Path

import numpy as np

from . import config

MODEL_URL = "https://github.com/jiaxi-jiang/FBCNN/releases/download/v1.0/fbcnn_color.pth"
MODEL_FILE = "fbcnn_color.pth"
MODEL_SIZE = 287755111
MODEL_SHA256 = "8b0e4ef23d59cf7ac934a342cb31a17619e4fa4a0b3374a9d78c5174312387e8"
TILE, OVERLAP = 768, 32  # both multiples of 8
MIN_TILE = 128
# Peak GPU memory per tile pixel in half precision (measured ~900-1000: ~510 MB for a 768 tile).
BYTES_PER_PIXEL = 1024

READY, NO_TORCH, NO_MODEL = "ready", "no_torch", "no_model"


def model_path() -> Path:
    return config.models_dir() / MODEL_FILE


def torch_installed() -> bool:
    return importlib.util.find_spec("torch") is not None


def model_ok(path: Path | None = None) -> bool:
    path = path or model_path()
    return path.is_file() and path.stat().st_size == MODEL_SIZE


def status() -> str:
    if not torch_installed():
        return NO_TORCH
    if not model_ok():
        return NO_MODEL
    return READY


def verify_download(path: Path) -> bool:
    if not model_ok(path):
        return False
    if not MODEL_SHA256:
        return True
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest() == MODEL_SHA256


def device_name() -> str:
    m = _load()
    import torch

    if m[1].type == "cuda":
        return torch.cuda.get_device_name(m[1])
    return "CPU (slow - no supported NVIDIA GPU found)"


# ---- network ---------------------------------------------------------------------
def _build():
    import torch
    from torch import nn

    def res_conv(nc):
        # In place is safe: the ReLU's input is the first conv's own output, used nowhere else.
        return nn.Sequential(nn.Conv2d(nc, nc, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv2d(nc, nc, 3, 1, 1))

    class ResBlock(nn.Module):
        def __init__(self, nc):
            super().__init__()
            self.res = res_conv(nc)

        def forward(self, x):
            return x + self.res(x)

    class QFAttention(nn.Module):
        def __init__(self, nc):
            super().__init__()
            self.res = res_conv(nc)

        def forward(self, x, gamma, beta):
            return x + gamma[..., None, None] * self.res(x) + beta[..., None, None]

    class FBCNN(nn.Module):
        def __init__(self, nc=(64, 128, 256, 512), nb=4):
            super().__init__()
            self.nb = nb
            self.m_head = nn.Conv2d(3, nc[0], 3, 1, 1)
            self.m_down1 = nn.Sequential(*[ResBlock(nc[0]) for _ in range(nb)], nn.Conv2d(nc[0], nc[1], 2, 2))
            self.m_down2 = nn.Sequential(*[ResBlock(nc[1]) for _ in range(nb)], nn.Conv2d(nc[1], nc[2], 2, 2))
            self.m_down3 = nn.Sequential(*[ResBlock(nc[2]) for _ in range(nb)], nn.Conv2d(nc[2], nc[3], 2, 2))
            self.m_body_encoder = nn.Sequential(*[ResBlock(nc[3]) for _ in range(nb)])
            self.m_body_decoder = nn.Sequential(*[ResBlock(nc[3]) for _ in range(nb)])
            self.m_up3 = nn.ModuleList([nn.ConvTranspose2d(nc[3], nc[2], 2, 2), *[QFAttention(nc[2]) for _ in range(nb)]])
            self.m_up2 = nn.ModuleList([nn.ConvTranspose2d(nc[2], nc[1], 2, 2), *[QFAttention(nc[1]) for _ in range(nb)]])
            self.m_up1 = nn.ModuleList([nn.ConvTranspose2d(nc[1], nc[0], 2, 2), *[QFAttention(nc[0]) for _ in range(nb)]])
            self.m_tail = nn.Conv2d(nc[0], 3, 3, 1, 1)
            self.qf_pred = nn.Sequential(
                *[ResBlock(nc[3]) for _ in range(nb)], nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
                nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 1), nn.Sigmoid(),
            )
            self.qf_embed = nn.Sequential(
                nn.Linear(1, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(),
            )
            for i, n in ((3, nc[2]), (2, nc[1]), (1, nc[0])):
                setattr(self, f"to_gamma_{i}", nn.Sequential(nn.Linear(512, n), nn.Sigmoid()))
                setattr(self, f"to_beta_{i}", nn.Sequential(nn.Linear(512, n), nn.Tanh()))

        def forward(self, x, qf):
            # x: (1,3,H,W) RGB 0..1 with H, W multiples of 8; qf: (1,1) quality 0..1
            x1 = self.m_head(x)
            x2 = self.m_down1(x1)
            x3 = self.m_down2(x2)
            x4 = self.m_down3(x3)
            x = self.m_body_decoder(self.m_body_encoder(x4))
            emb = self.qf_embed(qf)
            for level, skip in ((3, x4), (2, x3), (1, x2)):
                up = getattr(self, f"m_up{level}")
                gamma = getattr(self, f"to_gamma_{level}")(emb)
                beta = getattr(self, f"to_beta_{level}")(emb)
                x = up[0](x + skip)
                for blk in up[1:]:
                    x = blk(x, gamma, beta)
            return self.m_tail(x + x1)

    return torch, FBCNN


_model = None
_lock = threading.Lock()
_release_pending = False


def _load():
    global _model
    if _model is None:
        torch, FBCNN = _build()
        dev = torch.device("cpu")
        if torch.cuda.is_available():  # newest architecture first, then most memory
            best = max(range(torch.cuda.device_count()), key=lambda i: (
                torch.cuda.get_device_capability(i), torch.cuda.get_device_properties(i).total_memory))
            dev = torch.device(f"cuda:{best}")
        net = FBCNN()
        state = torch.load(model_path(), map_location="cpu", weights_only=True)
        net.load_state_dict(state)
        del state
        if dev.type == "cuda":
            net.half()  # before moving, so only the half-size copy is ever uploaded
        net.eval().to(dev)
        if dev.type == "cuda":
            net.to(memory_format=torch.channels_last)  # ~1.3x faster half-precision convs (measured)
        _model = (net, dev)
    return _model


def _drop() -> None:
    """Forget the model and hand its GPU memory back (call with _lock held)."""
    global _model, _release_pending
    _model, _release_pending = None, False
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.empty_cache()


def release() -> None:
    """Unload the model to free its (GPU) memory; the next use loads it again. Never waits: while a picture
    is being cleaned up, the model is dropped as soon as that finishes."""
    global _release_pending
    if not _lock.acquire(blocking=False):
        _release_pending = True
        return
    try:
        _drop()
    finally:
        _lock.release()


def _tile_size(torch, dev) -> int:
    """Biggest tile (up to TILE) whose activations fit in the GPU memory that is free now."""
    if dev.type != "cuda":
        return TILE
    free = torch.cuda.mem_get_info(dev)[0]
    side = int((free * 0.5 / BYTES_PER_PIXEL) ** 0.5) // 64 * 64
    return max(MIN_TILE, min(TILE, side))


def _run(rgb: np.ndarray, damage: float, tile: int | None = None) -> np.ndarray:
    import torch

    net, dev = _load()
    dtype = torch.float16 if dev.type == "cuda" else torch.float32
    qf = torch.tensor([[damage]], device=dev, dtype=dtype)
    h, w = rgb.shape[:2]
    out = np.empty_like(rgb)
    tile = tile or _tile_size(torch, dev)
    step = tile - 2 * OVERLAP
    for y0 in range(0, h, step):
        for x0 in range(0, w, step):
            # tile with margin, snapped to the 8x8 grid so JPEG blocks line up
            ty0, tx0 = max(0, y0 - OVERLAP) // 8 * 8, max(0, x0 - OVERLAP) // 8 * 8
            ty1, tx1 = min(h, y0 + step + OVERLAP), min(w, x0 + step + OVERLAP)
            t_ = rgb[ty0:ty1, tx0:tx1]
            th, tw = t_.shape[:2]
            ph, pw = -th % 8, -tw % 8
            if ph or pw:
                t_ = np.pad(t_, ((0, ph), (0, pw), (0, 0)), mode="edge")
            t = torch.from_numpy(np.ascontiguousarray(t_.transpose(2, 0, 1)))[None].to(dev, dtype)
            if dev.type == "cuda":
                t = t.contiguous(memory_format=torch.channels_last)
            res = net(t, qf)[0, :, :th, :tw].float().clamp_(0, 1).cpu().numpy().transpose(1, 2, 0)
            cy1, cx1 = min(h, y0 + step), min(w, x0 + step)
            out[y0:cy1, x0:cx1] = res[y0 - ty0:cy1 - ty0, x0 - tx0:cx1 - tx0]
    return out


def remove_artifacts(rgb: np.ndarray, strength: float) -> np.ndarray:
    """rgb: float32 HxWx3 in 0..1. Returns the cleaned image, same shape."""
    import torch

    # FBCNN control = 1 - JPEG quality: higher cleans harder. 100 treats it like a quality-10 JPEG.
    damage = 0.9 * min(max(strength, 0.0), 100.0) / 100.0
    with _lock, torch.inference_mode():
        try:
            tile = None
            while True:
                try:
                    return _run(rgb, damage, tile)
                except torch.cuda.OutOfMemoryError:
                    if _model is None:  # not even the model fits
                        raise
                    # something else took the memory meanwhile: try again with smaller tiles
                    tile = (tile or _tile_size(torch, _model[1])) // 2 // 8 * 8
                    if tile < MIN_TILE:
                        raise
                torch.cuda.empty_cache()  # out here the failed attempt's tensors are gone
        finally:
            if _release_pending:
                _drop()


def warm_up() -> None:
    """Load the model and run a tiny image so the first real preview doesn't pay the start-up cost."""
    try:
        remove_artifacts(np.zeros((64, 64, 3), np.float32), 50)
    except Exception:  # noqa: BLE001 - the real call reports problems
        pass
