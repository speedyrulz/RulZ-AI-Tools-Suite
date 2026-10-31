"""Crop box from a mask: a NumPy port of the RulZ Mask Intelligent Crop ComfyUI node (ComfyUI-RulZnodes).

The box is the mask's bounding box, extended on every side by a percentage of the final crop, the mask
box or the whole image, then grown (never shrunk) to stay within an aspect limit, then snapped so width
and height are multiples of `divisible_by`. It always stays inside the image and always contains the mask.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

ASPECT_LIMITS = {
    "off": None,
    "1:1": 1.0,
    "4:5 / 5:4": 5.0 / 4.0,
    "3:4 / 4:3": 4.0 / 3.0,
    "2:3 / 3:2": 3.0 / 2.0,
    "9:16 / 16:9": 16.0 / 9.0,
}
PADDING_REFERENCES = ("crop", "mask", "image")
PADDING_MODES = ("uniform", "per_edge")

# Largest combined share of an axis the two "crop"-referenced borders may take (the formula diverges at 100%).
_MAX_CROP_SHARE = 0.9


@dataclass
class CropSettings:
    """The node's settings, with its defaults."""

    padding_mode: str = "uniform"
    padding_percent: float = 10.0
    padding_top: float = 10.0
    padding_bottom: float = 10.0
    padding_left: float = 10.0
    padding_right: float = 10.0
    padding_reference: str = "crop"
    aspect_limit: str = "2:3 / 3:2"
    divisible_by: int = 1
    mask_threshold: float = 0.0

    def padding(self) -> float | tuple[float, float, float, float]:
        if self.padding_mode == "per_edge":
            return (self.padding_top, self.padding_bottom, self.padding_left, self.padding_right)
        return self.padding_percent

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CropSettings":
        """Unknown keys are ignored and bad values fall back to the default, one field at a time."""
        out = cls()
        for k, default in asdict(out).items():
            if k not in d:
                continue
            try:
                v = type(default)(d[k])
            except (TypeError, ValueError):
                continue
            if k == "padding_mode" and v not in PADDING_MODES:
                continue
            if k == "padding_reference" and v not in PADDING_REFERENCES:
                continue
            if k == "aspect_limit" and v not in ASPECT_LIMITS:
                continue
            setattr(out, k, v)
        return out


def _place_span(lo: int, hi: int, limit: int, new_size: int, must_lo: int, must_hi: int) -> tuple[int, int]:
    """Re-centre [lo, hi) at length new_size inside [0, limit), still containing [must_lo, must_hi)."""
    new_size = max(must_hi - must_lo, min(new_size, limit))
    center = (lo + hi) / 2.0
    new_lo = int(round(center - new_size / 2.0))
    new_lo = max(0, min(new_lo, limit - new_size))
    new_hi = new_lo + new_size
    # Keep the masked region inside (only reachable when the span had to shrink for divisible_by).
    if new_lo > must_lo:
        new_lo = must_lo
        new_hi = new_lo + new_size
    if new_hi < must_hi:
        new_hi = must_hi
        new_lo = new_hi - new_size
    return max(0, new_lo), min(limit, new_hi)


def _fit_span(lo: int, hi: int, limit: int, multiple: int, must_lo: int, must_hi: int) -> tuple[int, int]:
    """Make the span's length a multiple of `multiple`: grown when it fits, else the largest fitting multiple
    that still covers the mask, else the whole axis."""
    size = hi - lo
    if multiple > 1:
        grown = math.ceil(size / multiple) * multiple
        if grown <= limit:
            new_size = grown
        else:
            shrunk = (limit // multiple) * multiple
            new_size = shrunk if shrunk >= must_hi - must_lo else limit
    else:
        new_size = size
    return _place_span(lo, hi, limit, new_size, must_lo, must_hi)


def _limit_aspect(x0, x1, y0, y1, width, height, max_ratio, mx0, mx1, my0, my1):
    """Grow the box so long side / short side <= max_ratio: too tall is widened, too wide is heightened."""
    if max_ratio is None:
        return x0, x1, y0, y1
    w, h = x1 - x0, y1 - y0
    if w <= 0 or h <= 0:
        return x0, x1, y0, y1
    if h / w > max_ratio:
        x0, x1 = _place_span(x0, x1, width, int(math.ceil(h / max_ratio)), mx0, mx1)
    elif w / h > max_ratio:
        y0, y1 = _place_span(y0, y1, height, int(math.ceil(w / max_ratio)), my0, my1)
    return x0, x1, y0, y1


def _padding(percent_lo: float, percent_hi: float, reference: str, frame_size: int, mask_size: int) -> tuple[int, int]:
    """Pixels added before and after the mask on one axis."""
    p_lo, p_hi = percent_lo / 100.0, percent_hi / 100.0
    if reference == "image":
        pad_lo, pad_hi = frame_size * p_lo, frame_size * p_hi
    elif reference == "mask":
        pad_lo, pad_hi = mask_size * p_lo, mask_size * p_hi
    else:
        # Each side is its share of the final crop: crop = mask / (1 - p_lo - p_hi), capped below 100%.
        total = p_lo + p_hi
        if total > _MAX_CROP_SHARE:
            p_lo, p_hi = _MAX_CROP_SHARE * (p_lo / total), _MAX_CROP_SHARE * (p_hi / total)
        denom = 1.0 - (p_lo + p_hi)
        pad_lo, pad_hi = mask_size * p_lo / denom, mask_size * p_hi / denom
    return int(math.floor(pad_lo + 0.5)), int(math.floor(pad_hi + 0.5))


def compute_crop_box(mask: np.ndarray, padding_percent, divisible_by: int, threshold: float,
                     frame_height: int | None = None, frame_width: int | None = None,
                     padding_reference: str = "crop", max_aspect_ratio: float | None = None):
    """
    mask: HxW (or BxHxW, cropped by the union) at any resolution; frame_height/width: the image being
    cropped (defaults to the mask size). The mask box is scaled to frame pixels before padding.

    Returns (x0, y0, x1, y1), a half-open box in frame pixels, or None when no pixel is above threshold.
    """
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = mask.max(axis=0)
    mask_height, mask_width = mask.shape
    height = frame_height if frame_height is not None else mask_height
    width = frame_width if frame_width is not None else mask_width

    active = mask > threshold
    rows = np.flatnonzero(active.any(axis=1))
    cols = np.flatnonzero(active.any(axis=0))
    if rows.size == 0:
        return None
    my0, my1 = int(rows[0]), int(rows[-1]) + 1
    mx0, mx1 = int(cols[0]), int(cols[-1]) + 1

    if mask_height != height or mask_width != width:  # outward rounding
        sy, sx = height / mask_height, width / mask_width
        my0, my1 = int(math.floor(my0 * sy)), min(height, int(math.ceil(my1 * sy)))
        mx0, mx1 = int(math.floor(mx0 * sx)), min(width, int(math.ceil(mx1 * sx)))

    if isinstance(padding_percent, (int, float)):
        top = bottom = left = right = padding_percent
    else:
        top, bottom, left, right = padding_percent
    pad_left, pad_right = _padding(left, right, padding_reference, width, mx1 - mx0)
    pad_top, pad_bottom = _padding(top, bottom, padding_reference, height, my1 - my0)

    x0, x1 = max(0, mx0 - pad_left), min(width, mx1 + pad_right)
    y0, y1 = max(0, my0 - pad_top), min(height, my1 + pad_bottom)
    x0, x1, y0, y1 = _limit_aspect(x0, x1, y0, y1, width, height, max_aspect_ratio, mx0, mx1, my0, my1)
    x0, x1 = _fit_span(x0, x1, width, divisible_by, mx0, mx1)
    y0, y1 = _fit_span(y0, y1, height, divisible_by, my0, my1)
    return x0, y0, x1, y1


def crop_box(mask: np.ndarray, s: CropSettings, frame_height: int, frame_width: int):
    """compute_crop_box with the settings from the Auto crop screen."""
    return compute_crop_box(mask, s.padding(), max(1, s.divisible_by), s.mask_threshold, frame_height,
                            frame_width, s.padding_reference, ASPECT_LIMITS.get(s.aspect_limit))
