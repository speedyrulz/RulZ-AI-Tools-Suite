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
    r = max(1, radius // step)
    # _guided_coeffs for each color channel, with the guide's own statistics worked out once for both
    mean_i = _box(lum_s, r)
    denom = _box(lum_s * lum_s, r) - mean_i * mean_i + eps
    for c in range(2):
        src = np.ascontiguousarray(chroma_s[:, :, c])
        mean_p = _box(src, r)
        a = (_box(lum_s * src, r) - mean_i * mean_p) / denom
        b = mean_p - a * mean_i
        a, b = _box(a, r), _box(b, r)
        if step > 1:
            a = cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)
            b = cv2.resize(b, (w, h), interpolation=cv2.INTER_LINEAR)
        ch = np.ascontiguousarray(lab[:, :, 1 + c])
        a *= lum  # in place: a full-size picture's worth of temporaries less per step
        a += b
        a -= ch
        a *= np.float32(mix)
        ch += a
        lab[:, :, 1 + c] = ch


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
    lum = np.ascontiguousarray(lab[:, :, 0])
    out = lum.copy()
    for sigma, amount, thresh in ((0.8, 1.6, 0.6), (2.0, 0.7, 0.8)):
        hp = cv2.GaussianBlur(lum, (0, 0), sigma)
        np.subtract(lum, hp, out=hp)
        hp -= np.clip(hp, -thresh, thresh)  # = sign(hp) * max(|hp| - thresh, 0), with fewer passes
        out += np.float32(d * amount) * hp
    lab[:, :, 0] = np.clip(out, 0.0, 100.0)


# ---- pipeline -------------------------------------------------------------------
def _scale(dtype) -> float:
    return 255.0 if dtype == np.uint8 else 65535.0 if dtype == np.uint16 else 1.0


def _to_float(img: np.ndarray) -> tuple[np.ndarray, float]:
    scale = _scale(img.dtype)
    return (img.astype(np.float32) / scale if scale != 1.0 else img.astype(np.float32)), scale


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
    return _finish(_prepare(img, jpeg, mode, shadow)[0], color, detail)


@dataclass
class Stage:
    """A picture after JPEG clean-up and shadow lift (the slow part), in Lab, ready for the rest."""

    lab: np.ndarray
    gray: bool
    alpha: np.ndarray | None
    scale: float
    dtype: np.dtype
    shape: tuple


def _float_bgr(img: np.ndarray, gray: bool) -> np.ndarray:
    f, _ = _to_float(img if gray else img[:, :, :3])
    return cv2.cvtColor(f.reshape(f.shape[:2]), cv2.COLOR_GRAY2BGR) if gray else f


def _prepare(img: np.ndarray, jpeg: int, mode: str, shadow, cleaned: np.ndarray | None = None,
             lighting=None, keep_lighting: bool = False):
    """(Stage, cleaned, lighting): the picture after JPEG clean-up and shadow lift, plus the picture after JPEG
    clean-up alone when it was made in BGR on the way (Best quality, or Balanced before a shadow lift), else
    None. Pass that back as `cleaned` for the same picture, jpeg and mode to skip the JPEG step.

    lighting: the shadows.Lighting of the picture going into the shadow lift (the slow part of a lift, which
    doesn't depend on the painting), or None. With keep_lighting, it's measured here the first time a lift
    needs it (a one-shot save leaves it to the lift, which skips it when nothing is painted). It's returned
    either way, to pass back like `cleaned`."""
    gray = img.ndim == 2 or img.shape[2] == 1
    alpha = img[:, :, 3:] if not gray and img.shape[2] == 4 else None
    if jpeg and (mode == BEST or shadow):  # lift the cleaned picture, not its JPEG blocks
        if cleaned is None:
            f = _float_bgr(img, gray)
            if mode == BEST:
                cleaned = np.ascontiguousarray(_jpeg_best(f, jpeg))
            else:
                lab = cv2.cvtColor(f, cv2.COLOR_BGR2Lab)
                _jpeg_balanced(lab, jpeg)
                cleaned = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
        f, jpeg = cleaned, 0
    else:
        f, cleaned = _float_bgr(img, gray), None
    if shadow:
        from .shadows import lift_shadows, measure

        if keep_lighting and lighting is None:
            lighting = measure(f)
        f = lift_shadows(f, shadow, lighting)  # a new picture: `cleaned` is never changed
    lab = cv2.cvtColor(f, cv2.COLOR_BGR2Lab)
    if jpeg:  # Balanced without shadow areas: straight in Lab
        _jpeg_balanced(lab, jpeg)
    return Stage(lab, gray, alpha, _scale(img.dtype), img.dtype, img.shape), cleaned, lighting


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
    """process() for one picture edited again and again, reusing `cache` (what the last call returned):
    when only color noise or detail changed, the JPEG clean-up and shadow lift are reused; when only the
    shadow areas changed, the JPEG clean-up and the shadow lift's measurement of the picture are, so a brush
    stroke doesn't wait for them again.
    Returns (result, cache); the cache is (key, Stage, picture after JPEG clean-up or None, shadows.Lighting
    of the picture going into the shadow lift or None)."""
    shadow = shadow if shadow is not None and shadow.active else None
    if img is None or (is_noop(settings) and shadow is None):
        return img, cache
    jpeg = settings.get("jpeg", 0)
    key = (jpeg, mode if jpeg else BALANCED, shadow)
    if cache is not None and cache[0] == key:
        _, stage, cleaned, lighting = cache
    else:
        # Same JPEG step: the same picture goes into the shadow lift (the cleaned one, or the picture itself).
        reuse, lighting = cache[2:] if cache is not None and cache[0][:2] == key[:2] else (None, None)
        stage, cleaned, lighting = _prepare(img, jpeg, mode, shadow, reuse, lighting, keep_lighting=True)
        cleaned = reuse if cleaned is None else cleaned  # kept for when shadow areas come back (Undo)
    return (_finish(stage, settings.get("color", 0), settings.get("detail", 0), keep=True),
            (key, stage, cleaned, lighting))


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
