"""Shadow lift for painted areas.

A picture is lighting x surface. The lighting is estimated with an edge-aware smoothing of log-luminance
that removes texture but keeps shadow edges sharp; what's left over is the surface detail. Inside
the painted area the lighting is raised toward the light level just around it (filled in smoothly from
the unpainted pixels). The shadow's depth is measured once for the painted area and undone by a
multiplication in linear light, the way a real light would brighten the spot, so texture and local
contrast keep their ratios instead of being flattened, and parts of the painted area that are already as
bright as their surroundings aren't touched. The tint shadows usually
carry (blue from the sky) is matched to the surroundings too, and the painted area's edge is feathered
so the change blends in.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .filters import _guided_coeffs

DEFAULT_STRENGTH, DEFAULT_BLEND = 70, 40
MASK_MAX_SIDE = 2048  # the painted area is drawn at this size or smaller, then scaled up
FIELD_MAX_SIDE = 256  # the surrounding light level is filled in at this size
MAX_LIFT = 2.5  # natural log: at most ~12x brighter
GAMMA = 2.2


@dataclass(frozen=True)
class Stroke:
    """One brush stroke. Points are fractions of the picture's width and height; the radius is a
    fraction of its longer side, so a stroke fits the thumbnail, the preview and the full picture alike."""

    points: tuple[tuple[float, float], ...]
    radius: float
    erase: bool = False


@dataclass(frozen=True)
class ShadowEdit:
    strokes: tuple[Stroke, ...] = field(default_factory=tuple)
    strength: int = DEFAULT_STRENGTH  # 0..100
    blend: int = DEFAULT_BLEND  # 0..100, how soft the painted area's edge is

    @property
    def active(self) -> bool:
        return self.strength > 0 and any(not s.erase for s in self.strokes)

    def to_dict(self) -> dict:
        return {"strength": self.strength, "blend": self.blend,
                "strokes": [{"p": [list(p) for p in s.points], "r": s.radius, "e": s.erase} for s in self.strokes]}

    @classmethod
    def from_dict(cls, d: dict) -> "ShadowEdit":
        strokes = tuple(Stroke(tuple((float(x), float(y)) for x, y in s["p"]), float(s["r"]), bool(s.get("e")))
                        for s in d.get("strokes", []) if s.get("p"))
        return cls(strokes, int(d.get("strength", DEFAULT_STRENGTH)), int(d.get("blend", DEFAULT_BLEND)))


def _draw(mask: np.ndarray, strokes) -> None:
    h, w = mask.shape
    side = max(h, w)
    for s in strokes:
        r = max(1, round(s.radius * side))
        pts = np.array([[x * w, y * h] for x, y in s.points], np.float32)
        val = 0 if s.erase else 255
        shift = 4  # sub-pixel precision for smooth strokes at small sizes
        ipts = np.round(pts * (1 << shift)).astype(np.int32)
        if len(ipts) == 1:
            cv2.circle(mask, tuple(ipts[0]), r << shift, val, -1, cv2.LINE_AA, shift)
        else:
            cv2.polylines(mask, [ipts], False, val, 2 * r, cv2.LINE_AA, shift)


def render_mask(edit: ShadowEdit, h: int, w: int, feather: bool = True) -> np.ndarray:
    """The painted area as a float32 0..1 mask of h x w, with its edge feathered by `blend`."""
    f = min(1.0, MASK_MAX_SIDE / max(h, w))
    sh, sw = max(1, round(h * f)), max(1, round(w * f))
    small = np.zeros((sh, sw), np.uint8)
    _draw(small, edit.strokes)
    m = small.astype(np.float32) / 255.0
    if feather and edit.blend > 0:
        sigma = max(sh, sw) * 0.025 * (edit.blend / 100.0) ** 1.5
        if sigma >= 0.3:
            m = cv2.GaussianBlur(m, (0, 0), sigma)
    if (sh, sw) != (h, w):
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(m, 0.0, 1.0)


def _fill(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Smoothly fill where weights are 0 from where they're 1 (push-pull), keeping known values."""
    levels = [(values * weights, weights)]
    while min(levels[-1][1].shape) > 2:
        vw, wt = levels[-1]
        size = (max(1, wt.shape[1] // 2), max(1, wt.shape[0] // 2))
        levels.append((cv2.resize(vw, size, interpolation=cv2.INTER_AREA),
                       cv2.resize(wt, size, interpolation=cv2.INTER_AREA)))
    vw, wt = levels[-1]
    total = float(wt.sum())
    est = np.full(wt.shape, float(vw.sum()) / total if total > 1e-6 else 0.0, np.float32)
    for vw, wt in reversed(levels[:-1]):
        up = cv2.resize(est, (wt.shape[1], wt.shape[0]), interpolation=cv2.INTER_LINEAR)
        known = np.divide(vw, wt, out=np.zeros_like(vw), where=wt > 1e-6)
        a = np.clip(wt * 4.0, 0.0, 1.0)  # trust the level's own value where it has enough support
        est = a * known + (1.0 - a) * up
    return est


def _lighting(log_y: np.ndarray) -> np.ndarray:
    """The lighting, without the surface detail: a rolling-guidance smoothing of log-luminance. It starts
    from a blur, which has no texture, and each pass guides the original by it, so large steps such as
    shadow edges come back sharp while small-scale texture never does."""
    h, w = log_y.shape
    radius = max(2, round(max(h, w) * 0.012))
    step = max(1, radius // 4)
    size = (max(1, w // step), max(1, h // step))
    small = cv2.resize(log_y, size, interpolation=cv2.INTER_AREA) if step > 1 else log_y
    r = max(1, radius // step)
    guide = cv2.GaussianBlur(small, (0, 0), r / 2.0)
    for _ in range(4):
        a, b = _guided_coeffs(guide, small, r, 0.01)
        guide = a * guide + b
    # Bring it to full size following the picture's own edges: where the smoothed lighting has an edge,
    # the output takes the sharp full-size edge; where the picture only has texture, it stays flat.
    a, b = _guided_coeffs(small, guide, 1, 0.01)  # a small window, so the edge band stays thin
    if step > 1:
        a = cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)
        b = cv2.resize(b, (w, h), interpolation=cv2.INTER_LINEAR)
    return a * log_y + b


def lift_shadows(bgr: np.ndarray, edit: ShadowEdit) -> np.ndarray:
    """bgr: float32 HxWx3 in 0..1 (gamma-encoded). Returns the picture with the painted shadows lifted."""
    if not edit.active:
        return bgr
    h, w = bgr.shape[:2]
    s = edit.strength / 100.0
    m = render_mask(edit, h, w)
    if not m.any():
        return bgr

    lin = np.power(np.clip(bgr, 0.0, 1.0), GAMMA)
    y = lin @ np.array([0.0722, 0.7152, 0.2126], np.float32)
    log_y = np.log(y + 1e-4)
    light = _lighting(log_y)

    # The light level around the painted area, filled in across it.
    f = min(1.0, FIELD_MAX_SIDE / max(h, w))
    fs = (max(1, round(w * f)), max(1, round(h * f)))
    light_s = cv2.resize(light, fs, interpolation=cv2.INTER_AREA)
    m_s = cv2.resize(m, fs, interpolation=cv2.INTER_AREA)
    outside = (m_s < 0.02).astype(np.float32)
    if outside.sum() < 1:  # painted everywhere: lift toward the brighter parts of the picture
        outside = (light_s >= np.percentile(light_s, 70)).astype(np.float32)
    around = _fill(light_s, outside)
    around = cv2.resize(cv2.GaussianBlur(around, (0, 0), 1.5), (w, h), interpolation=cv2.INTER_LINEAR)

    # A shadow dims everything under it by about the same factor, whatever the surfaces are, so lift by one
    # depth measured for the whole painted area, not by each pixel's own difference from the surroundings: a
    # dark tyre isn't pushed up to a bright car's level, and a bright object in the shadow gets the full lift.
    # The depth is the median over painted pixels clearly darker than their surroundings, so lit bits painted
    # over don't water it down. With hardly any of those, there's no shadow to lift.
    gap = around - light
    painted_px = m > 0.5
    dark = gap[painted_px & (gap > 0.3)]
    if dark.size < max(16, 0.1 * painted_px.sum()):
        return bgr
    depth = float(np.clip(np.median(dark), 0.0, MAX_LIFT))
    # How much each pixel is in the shadow. Near the edge of the painting, where the shadow's soft fringe and
    # any lit bits painted over are, it's the pixel's own share of the depth. Deeper inside, everything counts
    # as shadow unless its neighborhood is plainly lit (a brush stroke running on past the shadow); decided
    # per neighborhood so a bright object in the shadow doesn't turn blotchy.
    radius = max((st.radius for st in edit.strokes if not st.erase), default=0.0) * max(fs)
    band = max(1.0, 0.35 * radius)
    inside = cv2.distanceTransform((m_s > 0.5).astype(np.uint8), cv2.DIST_L2, 3)
    deep = np.clip((inside - band) / band, 0.0, 1.0)
    deep = cv2.resize(cv2.GaussianBlur(deep, (0, 0), 1.0), (w, h), interpolation=cv2.INTER_LINEAR)
    part = np.clip(gap / depth, 0.0, 1.0)
    part_s = cv2.GaussianBlur(cv2.resize(part, fs, interpolation=cv2.INTER_AREA), (0, 0), 3.0)
    not_lit = cv2.resize(np.clip((part_s - 0.1) / 0.25, 0.0, 1.0), (w, h), interpolation=cv2.INTER_LINEAR)
    share = np.maximum(part, deep * not_lit)
    lift = depth * share * (s * m)
    out = lin * np.exp(lift)[..., None]

    # Match the shadow's tint (often blue from the sky) to the surroundings. It's measured in a band on
    # each side of the edge (inside the shadow, and outside the painting), where the surface is usually
    # the same material, with medians so a brightly
    # colored object in the shadow doesn't skew it; then applied in proportion to each pixel's lift.
    lin_s = cv2.resize(lin, fs, interpolation=cv2.INTER_AREA)
    y_s = cv2.resize(y, fs, interpolation=cv2.INTER_AREA)
    tint = np.log(lin_s + 1e-4) - np.log(y_s + 1e-4)[..., None]
    lift_s = cv2.resize(lift, fs, interpolation=cv2.INTER_AREA)
    k = np.ones((2 * max(2, round(max(fs) * 0.06)) + 1,) * 2, np.uint8)
    painted = (m_s > 0.02).astype(np.uint8)
    ring = (cv2.dilate(painted, k) > 0) & (outside > 0)
    shaded = (lift_s > 0.5 * s * depth).astype(np.uint8)
    shaded = (shaded > 0) & (cv2.erode(shaded, k) == 0)  # the shadow's inner edge band
    if shaded.sum() >= 8 and ring.sum() >= 8:
        cast = np.median(tint[shaded], axis=0) - np.median(tint[ring], axis=0)
        cast = np.clip(cast - cast @ np.array([0.0722, 0.7152, 0.2126], np.float32), -0.5, 0.5)
        amount = np.clip(lift / max(s * depth, 1e-3), 0.0, 1.0)
        out *= np.exp(-cast[None, None, :] * amount[..., None])

    # Roll off highlights that the lift pushed past white, keeping their color.
    peak = out.max(axis=2)
    soft = np.where(peak > 0.9, 0.9 + 0.1 * (1.0 - np.exp(-(peak - 0.9) / 0.1)), peak)
    k = np.divide(soft, peak, out=np.ones_like(peak), where=peak > 1e-6)
    k = 1.0 + (k - 1.0) * np.clip(lift / 0.1, 0.0, 1.0)  # only where something was lifted
    out *= k[..., None]
    return np.power(np.clip(out, 0.0, 1.0), 1.0 / GAMMA).astype(np.float32)
