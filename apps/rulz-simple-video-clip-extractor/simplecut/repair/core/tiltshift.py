"""Miniature (tilt-shift) effect: progressive bokeh blur outside a sharp band.

Copied unchanged from RulZ Image Distortion Repair (package viewpointer).
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .params import TiltShiftParams

BLUR_SHAPES = {
    "circular": "Circular",
    "circular_soft": "Circular soft",
    "circular_sharp": "Circular sharp",
    "blades6": "6 Blades",
    "blades8": "8 Blades",
    "blades9": "9 Blades",
}


def bokeh_kernel(shape: str, radius: float) -> np.ndarray:
    r = max(1.0, radius)
    n = int(math.ceil(r)) * 2 + 1
    c = n // 2
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32) - c
    d = np.sqrt(xx * xx + yy * yy)
    if shape.startswith("blades"):
        blades = int(shape[len("blades"):])
        k = np.zeros((n, n), np.float32)
        pts = [(c + r * math.cos(2 * math.pi * i / blades + math.pi / 2),
                c + r * math.sin(2 * math.pi * i / blades + math.pi / 2)) for i in range(blades)]
        cv2.fillPoly(k, [np.round(np.array(pts) * 16).astype(np.int32)], 1.0, cv2.LINE_AA, shift=4)
    elif shape == "circular_soft":
        k = np.clip(1.0 - d / r, 0, 1) ** 0.8
    elif shape == "circular_sharp":
        k = np.clip(r + 0.5 - d, 0, 1) * (0.6 + 0.4 * np.clip(d / r, 0, 1) ** 2)
    else:
        k = np.clip(r + 0.5 - d, 0, 1)
    s = k.sum()
    return k / s if s > 0 else k


def band_weights(shape_hw, p: TiltShiftParams) -> np.ndarray:
    """0 inside the sharp band, rising to 1 at the transition lines and beyond."""
    h, w = shape_hw
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = p.center[0] * w, p.center[1] * h
    a = math.radians(p.angle)
    # signed distance perpendicular to the band (y-down, positive = below the centre line)
    nx, ny = math.sin(a), math.cos(a)
    d = ((xx - cx) * nx + (yy - cy) * ny) / h
    st, sb = p.sharp_top, p.sharp_bottom
    tt, tb = max(p.trans_top, st + 1e-3), max(p.trans_bottom, sb + 1e-3)
    up = np.clip((-d - st) / (tt - st), 0, 1)
    down = np.clip((d - sb) / (tb - sb), 0, 1)
    wgt = np.where(d < 0, up, down)
    return wgt * wgt * (3 - 2 * wgt)   # smoothstep


def apply_tiltshift(img: np.ndarray, p: TiltShiftParams) -> np.ndarray:
    """img: float32 RGB(A) in [0, 1]. Returns a new image."""
    h, w = img.shape[:2]
    rmax = p.intensity / 100.0 * 0.025 * math.hypot(w, h)
    if rmax < 0.75:
        return img
    rgb = np.ascontiguousarray(np.clip(img[..., :3], 0, 1))
    lin = cv2.pow(rgb, 2.2)                       # blur in linear light for brighter bokeh
    levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    wgt = band_weights((h, w), p)
    pos = wgt * (len(levels) - 1)
    out = np.zeros_like(lin)
    for i, t in enumerate(levels):
        hat = np.maximum(0.0, 1.0 - np.abs(pos - i))
        if not hat.any():
            continue
        if t == 0:
            layer = lin
        else:
            r = rmax * t
            ds = int(min(4, max(1, r // 6)))      # large kernels run on a downscaled copy
            small = lin if ds == 1 else cv2.resize(lin, (max(1, w // ds), max(1, h // ds)), interpolation=cv2.INTER_AREA)
            small = cv2.filter2D(small, -1, bokeh_kernel(p.blur_shape, r / ds), borderType=cv2.BORDER_REFLECT)
            layer = small if ds == 1 else cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        out += layer * hat[..., None]
    res = img.copy()
    res[..., :3] = cv2.pow(np.clip(out, 0, 1), 1 / 2.2)
    return res
