"""Miniature (tilt-shift) effect: progressive bokeh blur outside a sharp band.

Copied unchanged from RulZ Image Distortion Repair (package viewpointer), plus TiltShiftPlan
at the end (Clip Extractor addition: the same effect for a stream of video frames).
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


# ------------------------------------------------------------------ Clip Extractor addition
LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)   # the blur levels apply_tiltshift blends


class TiltShiftPlan:
    """apply_tiltshift for many uint8 RGB frames of one size (video), same result to within a
    level. What depends only on the settings and the frame size (the band weights, each blur
    level's blend weight and kernel) is worked out once instead of per frame, the gamma curves
    are lookup tables, and each frame needs two full-size buffers instead of a dozen."""

    def __init__(self, shape_hw: tuple[int, int], p: TiltShiftParams):
        h, w = shape_hw
        self.shape = (int(h), int(w))
        rmax = p.intensity / 100.0 * 0.025 * math.hypot(w, h)
        self.levels: list[tuple[np.ndarray, tuple[int, np.ndarray] | None]] = []
        if rmax < 0.75:
            return  # too little blur to show (apply_tiltshift returns the frame unchanged too)
        pos = band_weights((h, w), p) * (len(LEVELS) - 1)
        for i, t in enumerate(LEVELS):
            hat = np.maximum(0.0, 1.0 - np.abs(pos - i)).astype(np.float32)
            if not hat.any():
                continue
            blur = None
            if t:
                r = rmax * t
                ds = int(min(4, max(1, r // 6)))
                blur = (ds, bokeh_kernel(p.blur_shape, r / ds))
            self.levels.append((hat[..., None], blur))
        self._to_lin = ((np.arange(256) / 255.0) ** 2.2).astype(np.float32)
        # linear light (16-bit steps) back to 8-bit gamma, rounded like apply_tiltshift's caller
        self._from_lin = np.floor((np.arange(65536) / 65535.0) ** (1 / 2.2) * 255.0 + 0.5).astype(np.uint8)

    @property
    def active(self) -> bool:
        return bool(self.levels)

    def apply(self, rgb: np.ndarray) -> np.ndarray:
        """uint8 RGB frame of `shape` -> a new uint8 frame with the effect (or rgb itself)."""
        if not self.levels:
            return rgb
        h, w = self.shape
        if rgb.shape[:2] != (h, w):
            raise ValueError(f"frame size {rgb.shape[1]}x{rgb.shape[0]} does not match the plan's {w}x{h}")
        lin = cv2.LUT(np.ascontiguousarray(rgb[..., :3]), self._to_lin)   # blur in linear light
        # Buffers are per call, not kept: an export and a frame PNG can share one plan across threads.
        acc = np.empty((h, w, 3), np.float32)
        tmp = np.empty((h, w, 3), np.float32)
        first = True
        for hat, blur in self.levels:
            if blur is None:
                layer = lin
            else:
                ds, k = blur
                small = lin if ds == 1 else cv2.resize(lin, (max(1, w // ds), max(1, h // ds)),
                                                       interpolation=cv2.INTER_AREA)
                small = cv2.filter2D(small, -1, k, borderType=cv2.BORDER_REFLECT)
                layer = small if ds == 1 else cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
            if first:
                np.multiply(layer, hat, out=acc)
                first = False
            else:
                np.multiply(layer, hat, out=tmp)
                acc += tmp
        np.clip(acc, 0.0, 1.0, out=acc)
        acc *= 65535.0
        acc += 0.5
        return self._from_lin[acc.astype(np.uint16)]
