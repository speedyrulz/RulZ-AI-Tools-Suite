"""Image clean-up filters and the pipeline that runs them.

Everything works in CIE Lab: L is lightness (detail, sharpness), a/b are color. Order:
1. JPEG artifact removal - blocking, ringing and blocky color. "Balanced" is a fast classic
   filter; "Best quality" is an AI model run on the GPU (ai_jpeg.py).
1b. Shadow lift in painted areas (shadows.py), before the noise steps so they clean what it reveals.
2. Color noise removal - smooths a/b only (see remove-color-noise notes below).
3. Detail recovery - thresholded sharpening of L to win back crispness lost in step 1.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

BALANCED, BEST = "balanced", "best"

# name: (label, default)
ADJUSTMENTS = {
    "color": ("Color noise removal", 0),
    "jpeg": ("JPEG artifact removal", 0),
    "detail": ("Detail recovery", 0),
}


# ---- helpers -----------------------------------------------------------------
def _box(x: np.ndarray, r: int) -> np.ndarray:
    return cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1), borderType=cv2.BORDER_REFLECT)


def _guided_coeffs(guide: np.ndarray, src: np.ndarray, r: int, eps: float):
    """Linear coefficients (a, b) of a guided filter: output = a * guide + b."""
    mean_i = _box(guide, r)
    mean_p = _box(src, r)
    var_i = _box(guide * guide, r) - mean_i * mean_i
    cov_ip = _box(guide * src, r) - mean_i * mean_p
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    return _box(a, r), _box(b, r)


def _smooth_chroma(lab: np.ndarray, radius: int, eps: float, mix: float) -> None:
    """Edge-aware smoothing of a/b guided by L, in place (fast guided filter)."""
    lum = lab[:, :, 0] / 100.0
    h, w = lum.shape
    step = max(1, radius // 4)
    if step > 1:
        size = (max(1, w // step), max(1, h // step))
        lum_s = cv2.resize(lum, size, interpolation=cv2.INTER_AREA)
        chroma_s = cv2.resize(lab[:, :, 1:], size, interpolation=cv2.INTER_AREA)
    else:
        lum_s, chroma_s = lum, lab[:, :, 1:]
    r_s = max(1, radius // step)
    for c in range(2):
        a, b = _guided_coeffs(lum_s, np.ascontiguousarray(chroma_s[:, :, c]), r_s, eps)
        if step > 1:
            a = cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)
            b = cv2.resize(b, (w, h), interpolation=cv2.INTER_LINEAR)
        lab[:, :, 1 + c] += mix * (a * lum + b - lab[:, :, 1 + c])


# ---- the three adjustments (all in place on a float32 Lab image) ------------------
def _color_noise(lab: np.ndarray, strength: float) -> None:
    s = strength / 100.0
    if s > 0.15:  # knocks out isolated colored specks before smoothing
        lab[:, :, 1:] = cv2.medianBlur(np.ascontiguousarray(lab[:, :, 1:]), 5 if s > 0.6 else 3)
    h, w = lab.shape[:2]
    # The filter size scales with the image, so a small preview and the full image look alike.
    radius = max(1, round(max(h, w) * (0.002 + 0.010 * s)))
    _smooth_chroma(lab, radius, 0.0002 + 0.004 * s * s, min(1.0, 0.3 + s))


def _jpeg_balanced(lab: np.ndarray, strength: float) -> None:
    """Fast: edge-preserving smoothing of lightness, plus color smoothing guided by lightness
    (JPEG stores color at half resolution, which leaves blocky color edges)."""
    # JPEG artifacts live on a fixed 8-pixel grid, so these sizes are in pixels, not relative.
    s = strength / 100.0
    lab[:, :, 0] = cv2.bilateralFilter(lab[:, :, 0] * 2.55, 7, 3.0 + 22.0 * s, 2.0 + 2.0 * s) / 2.55
    _smooth_chroma(lab, round(3 + 5 * s), 5e-5 + 9.5e-4 * s * s, 1.0)


def _jpeg_best(bgr: np.ndarray, strength: float) -> np.ndarray:
    """AI model (see ai_jpeg.py); bgr is float32 0..1."""
    from . import ai_jpeg

    if ai_jpeg.status() != ai_jpeg.READY:
        raise RuntimeError("The AI JPEG model isn't set up yet (Best quality mode)")
    return ai_jpeg.remove_artifacts(bgr[:, :, ::-1], strength)[:, :, ::-1]


def _detail(lab: np.ndarray, strength: float) -> None:
    """Two-scale unsharp mask on lightness; small differences (leftover noise) are not boosted."""
    d = strength / 100.0
    lum = lab[:, :, 0]
    out = lum.copy()
    for sigma, amount, thresh in ((0.8, 1.6, 0.6), (2.0, 0.7, 0.8)):
        hp = lum - cv2.GaussianBlur(lum, (0, 0), sigma)
        hp = np.sign(hp) * np.maximum(np.abs(hp) - thresh, 0.0)
        out += d * amount * hp
    lab[:, :, 0] = np.clip(out, 0.0, 100.0)


# ---- pipeline -------------------------------------------------------------------
def _to_float(img: np.ndarray) -> tuple[np.ndarray, float]:
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0, 255.0
    if img.dtype == np.uint16:
        return img.astype(np.float32) / 65535.0, 65535.0
    return img.astype(np.float32), 1.0


def is_noop(settings: dict) -> bool:
    return not any(settings.get(k, 0) > 0 for k in ADJUSTMENTS)


def process(img: np.ndarray, settings: dict, mode: str = BALANCED, full_size: bool = True,
            shadow=None) -> np.ndarray:
    """Return a cleaned copy of a BGR/BGRA/gray image (8-bit, 16-bit or float).

    settings maps adjustment name -> strength 0..100. full_size=False marks a shrunken preview:
    JPEG blocks and fine detail don't survive shrinking, so those two steps are skipped there.
    shadow is the picture's shadows.ShadowEdit (painted areas to lift), or None.
    """
    shadow = shadow if shadow is not None and shadow.active else None
    if img is None or (is_noop(settings) and shadow is None):
        return img
    color = settings.get("color", 0)
    jpeg = settings.get("jpeg", 0) if full_size else 0
    detail = settings.get("detail", 0) if full_size else 0
    if not (color or jpeg or detail or shadow):
        return img
    return _finish(_prepare(img, jpeg, mode, shadow), color, detail)


@dataclass
class Stage:
    """A picture after JPEG clean-up and shadow lift (the slow part), in Lab, ready for the rest."""

    lab: np.ndarray
    gray: bool
    alpha: np.ndarray | None
    scale: float
    dtype: np.dtype
    shape: tuple


def _prepare(img: np.ndarray, jpeg: int, mode: str, shadow) -> Stage:
    gray = img.ndim == 2 or img.shape[2] == 1
    alpha = img[:, :, 3:] if not gray and img.shape[2] == 4 else None
    src = img if gray else img[:, :, :3]
    f, scale = _to_float(src)
    if gray:
        f = cv2.cvtColor(f.reshape(f.shape[:2]), cv2.COLOR_GRAY2BGR)
    if jpeg and mode == BEST:
        f = np.ascontiguousarray(_jpeg_best(f, jpeg))
    if jpeg and mode != BEST and shadow:  # lift the cleaned picture, not its JPEG blocks
        lab = cv2.cvtColor(f, cv2.COLOR_BGR2Lab)
        _jpeg_balanced(lab, jpeg)
        f, jpeg = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR), 0
    if shadow:
        from .shadows import lift_shadows

        f = lift_shadows(f, shadow)
    lab = cv2.cvtColor(f, cv2.COLOR_BGR2Lab)
    if jpeg and mode != BEST:
        _jpeg_balanced(lab, jpeg)
    return Stage(lab, gray, alpha, scale, img.dtype, img.shape)


def _finish(stage: Stage, color: int, detail: int, keep: bool = False) -> np.ndarray:
    """The rest of the pipeline; keep=True leaves stage.lab untouched so it can be used again."""
    lab = stage.lab.copy() if keep and (color or detail) else stage.lab
    if color and not stage.gray:  # grayscale has no color noise
        _color_noise(lab, color)
    if detail:
        _detail(lab, detail)

    out = np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2BGR), 0.0, 1.0)
    if stage.gray:
        out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).reshape(stage.shape)
    out = np.round(out * stage.scale).astype(stage.dtype) if stage.scale != 1.0 else out.astype(stage.dtype)
    if stage.alpha is not None:
        out = np.concatenate([out, stage.alpha], axis=2)
    return out


def process_cached(img: np.ndarray, settings: dict, mode: str = BALANCED, shadow=None, cache=None):
    """process() for one picture edited again and again: when only color noise or detail changed, the JPEG
    clean-up and shadow lift are reused from `cache` (what the last call returned). Returns (result, cache)."""
    shadow = shadow if shadow is not None and shadow.active else None
    if img is None or (is_noop(settings) and shadow is None):
        return img, cache
    jpeg = settings.get("jpeg", 0)
    key = (jpeg, mode if jpeg else BALANCED, shadow)
    stage = cache[1] if cache is not None and cache[0] == key else _prepare(img, jpeg, mode, shadow)
    return _finish(stage, settings.get("color", 0), settings.get("detail", 0), keep=True), (key, stage)


# Pixels kept around a crop when only it is cleaned up: more than every filter's reach that doesn't scale
# with the picture (and than ai_jpeg's tile overlap), so the crop comes out as if the whole picture was done.
REGION_MARGIN = 64


def process_crop(img: np.ndarray, box: tuple[int, int, int, int], settings: dict, mode: str = BALANCED,
                 shadow=None) -> np.ndarray:
    """process(img)[box], but when it gives the same picture only the box plus a margin is cleaned up.

    Color noise removal (its filter size follows the picture size) and shadow areas (measured over the
    whole picture) need the whole picture.
    """
    x0, y0, x1, y1 = box
    shadow = shadow if shadow is not None and shadow.active else None
    if shadow is not None or settings.get("color", 0) > 0:
        return np.ascontiguousarray(process(img, settings, mode, shadow=shadow)[y0:y1, x0:x1])
    h, w = img.shape[:2]
    # snapped to the 8-px grid so JPEG blocks stay where the filters expect them
    mx0, my0 = max(0, x0 - REGION_MARGIN) // 8 * 8, max(0, y0 - REGION_MARGIN) // 8 * 8
    mx1, my1 = min(w, x1 + REGION_MARGIN), min(h, y1 + REGION_MARGIN)
    out = process(np.ascontiguousarray(img[my0:my1, mx0:mx1]), settings, mode)
    return np.ascontiguousarray(out[y0 - my0:y1 - my0, x0 - mx0:x1 - mx0])
