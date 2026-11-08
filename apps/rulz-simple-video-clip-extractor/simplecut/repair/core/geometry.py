"""Geometric correction stages.

Every stage works in *frame coordinates* ``(u, v)`` in [0, 1]² of its input image and exposes
``inverse(u, v) -> (u_src, v_src)`` — for each output position, where to sample its input.
Stages are chained and evaluated back-to-front, so a full render resamples the source once.

Internally the radial/projective maths uses *centred coordinates* scaled by the half-diagonal
(corners at radius 1), which keeps everything isotropic and resolution independent.

Copied unchanged from RulZ Image Distortion Repair (package viewpointer).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .params import Corrections, PerspectiveParams

# Stage order. The crop stage is applied by the renderer (it selects the output window).
STAGES = ("distortion", "volume", "perspective", "horizon", "warp")
DEFAULT_FOCAL_35MM = 24.0
HALF_DIAG_35MM = math.hypot(36.0, 24.0) / 2.0


@dataclass
class Frame:
    """Aspect information shared by all stages (all stages keep the frame size)."""
    width: int
    height: int

    @property
    def half_x(self) -> float:
        return self.width / math.hypot(self.width, self.height)

    @property
    def half_y(self) -> float:
        return self.height / math.hypot(self.width, self.height)

    @property
    def aspect(self) -> float:
        return self.width / self.height

    def to_centered(self, u, v):
        return (np.asarray(u) - 0.5) * 2 * self.half_x, (np.asarray(v) - 0.5) * 2 * self.half_y

    def from_centered(self, x, y):
        return np.asarray(x) / (2 * self.half_x) + 0.5, np.asarray(y) / (2 * self.half_y) + 0.5


# --------------------------------------------------------------------------- homographies

def apply_h(h: np.ndarray, x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = h[2, 0] * x + h[2, 1] * y + h[2, 2]
    w = np.where(np.abs(w) < 1e-9, 1e-9, w)
    return (h[0, 0] * x + h[0, 1] * y + h[0, 2]) / w, (h[1, 0] * x + h[1, 1] * y + h[1, 2]) / w


def homography_from_points(src, dst) -> np.ndarray:
    """Exact homography from four point pairs (DLT)."""
    a = []
    for (x, y), (xp, yp) in zip(src, dst):
        a.append([x, y, 1, 0, 0, 0, -xp * x, -xp * y, -xp])
        a.append([0, 0, 0, x, y, 1, -yp * x, -yp * y, -yp])
    _, _, vt = np.linalg.svd(np.asarray(a, dtype=np.float64))
    h = vt[-1].reshape(3, 3)
    return h / h[2, 2]


def rotation_homography(pitch_deg: float, yaw_deg: float, roll_deg: float, focal: float) -> np.ndarray:
    """Synthetic camera rotation K·R·K⁻¹ in centred coordinates."""
    p, y, r = (math.radians(a) for a in (pitch_deg, yaw_deg, roll_deg))
    rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    k = np.diag([focal, focal, 1.0])
    return k @ rz @ ry @ rx @ np.linalg.inv(k)


def line_intersection(l1, l2):
    (x1, y1), (x2, y2) = l1
    (x3, y3), (x4, y4) = l2
    d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(d) < 1e-12:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / d
    return x1 + t * (x2 - x1), y1 + t * (y2 - y1)


def order_quad(pts):
    """Order four points TL, TR, BR, BL (image coordinates, y down)."""
    pts = np.asarray(pts, dtype=np.float64)
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    pts = pts[np.argsort(ang)]          # starts at the top-left-ish corner going clockwise
    start = int(np.argmin(pts.sum(axis=1)))
    return np.roll(pts, -start, axis=0)


def quad_to_rect_targets(quad):
    tl, tr, br, bl = quad
    w = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    h = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
    c = quad.mean(axis=0)
    return np.array([[c[0] - w / 2, c[1] - h / 2], [c[0] + w / 2, c[1] - h / 2],
                     [c[0] + w / 2, c[1] + h / 2], [c[0] - w / 2, c[1] + h / 2]])


def guide_homography(p: PerspectiveParams, frame: Frame) -> np.ndarray | None:
    """Homography (centred coords) that straightens the user's lines/points, or None."""
    def c(pt):
        x, y = frame.to_centered(pt[0], pt[1])
        return np.array([float(x), float(y)])

    lines = [(c(a), c(b)) for a, b in p.lines if a is not None and b is not None] \
        if p.mode != "rect" else []
    if p.mode == "verticals" and len(lines) >= 2:
        src, dst = [], []
        for a, b in lines[:2]:
            xm = (a[0] + b[0]) / 2
            src += [a, b]
            dst += [(xm, a[1]), (xm, b[1])]
        return homography_from_points(src, dst)
    if p.mode == "horizontals" and len(lines) >= 2:
        src, dst = [], []
        for a, b in lines[:2]:
            ym = (a[1] + b[1]) / 2
            src += [a, b]
            dst += [(a[0], ym), (b[0], ym)]
        return homography_from_points(src, dst)
    if p.mode == "rect" and len(p.lines) >= 4:
        quad = order_quad([c(pt) for pt in p.lines[:4]])
        return homography_from_points(quad, quad_to_rect_targets(quad))
    if p.mode == "eight" and len(lines) >= 4:
        v1, v2, h1, h2 = lines[:4]
        corners = [line_intersection(v, h) for v, h in ((v1, h1), (v2, h1), (v2, h2), (v1, h2))]
        if any(pt is None for pt in corners):
            return None
        quad = order_quad(corners)
        return homography_from_points(quad, quad_to_rect_targets(quad))
    return None


def blend_homography(h: np.ndarray, t: float, frame: Frame) -> np.ndarray:
    """Interpolate between identity (t=0) and ``h`` (t=1) through the frame corners."""
    if abs(t - 1.0) < 1e-6:
        return h
    corners = np.array([[-frame.half_x, -frame.half_y], [frame.half_x, -frame.half_y],
                        [frame.half_x, frame.half_y], [-frame.half_x, frame.half_y]])
    hx, hy = apply_h(h, corners[:, 0], corners[:, 1])
    moved = np.stack([hx, hy], axis=1)
    return homography_from_points(corners, corners + t * (moved - corners))


def fit_homography(h: np.ndarray, frame: Frame) -> np.ndarray:
    """Scale/translate the result so the corrected image's bounding box fits the frame."""
    corners = np.array([[-frame.half_x, -frame.half_y], [frame.half_x, -frame.half_y],
                        [frame.half_x, frame.half_y], [-frame.half_x, frame.half_y]])
    hx, hy = apply_h(h, corners[:, 0], corners[:, 1])
    if not (np.all(np.isfinite(hx)) and np.all(np.isfinite(hy))):
        return h
    bw, bh = hx.max() - hx.min(), hy.max() - hy.min()
    if bw <= 1e-9 or bh <= 1e-9:
        return h
    s = min(2 * frame.half_x / bw, 2 * frame.half_y / bh)
    cx, cy = (hx.max() + hx.min()) / 2, (hy.max() + hy.min()) / 2
    t = np.array([[s, 0, -s * cx], [0, s, -s * cy], [0, 0, 1]])
    return t @ h


def perspective_homography(p: PerspectiveParams, frame: Frame, focal: float) -> np.ndarray:
    h = np.eye(3)
    if p.mode == "auto" and p.auto_angles:
        h = rotation_homography(*p.auto_angles, focal)
    elif p.mode != "auto":
        g = guide_homography(p, frame)
        if g is not None:
            h = g
    h = blend_homography(h, p.intensity / 100.0, frame)
    if p.up_down or p.left_right:
        h = rotation_homography(p.up_down / 125.0 * 30.0, p.left_right / 125.0 * 30.0, 0.0, focal) @ h
    if p.hv_ratio:
        h = np.diag([1.0, 2.0 ** (p.hv_ratio / 100.0 * 0.5), 1.0]) @ h
    if np.allclose(h, np.eye(3)):
        return h
    return fit_homography(h, frame)


# --------------------------------------------------------------------------- stages

class Stage:
    def inverse(self, u, v):
        raise NotImplementedError


class IdentityStage(Stage):
    def inverse(self, u, v):
        return u, v


class DistortionStage(Stage):
    def __init__(self, kind: str, intensity: int, frame: Frame):
        self.kind, self.k, self.frame = kind, intensity / 100.0, frame

    def inverse(self, u, v):
        x, y = self.frame.to_centered(u, v)
        r2 = x * x + y * y
        if self.kind == "barrel":
            s = 1.0 - 0.3 * self.k * r2
        elif self.kind == "pincushion":
            s = (1.0 + 0.3 * self.k * r2) / (1.0 + 0.3 * self.k * self.frame.half_y ** 2)
        else:  # fisheye → rectilinear, equidistant model
            f = 0.75
            r = np.sqrt(r2)
            rs = f * np.arctan(r / f)
            s = np.where(r > 1e-9, 1.0 + self.k * (rs / np.maximum(r, 1e-9) - 1.0), 1.0)
            s = s / (1.0 + self.k * (f * math.atan(self.frame.half_y / f) / self.frame.half_y - 1.0))
        return self.frame.from_centered(x * s, y * s)


class VolumeStage(Stage):
    """Wide-angle "volume deformation" correction: cylindrical (H/V) or stereographic."""

    def __init__(self, mode: str, h: int, v: int, d: int, frame: Frame, focal: float):
        self.mode, self.sh, self.sv, self.sd = mode, h / 100.0, v / 100.0, d / 100.0
        self.frame, self.f = frame, focal

    def inverse(self, u, v):
        x, y = self.frame.to_centered(u, v)
        f = self.f
        lim = math.pi / 2 * 0.98
        if self.mode == "hv":
            if self.sv:
                phi = np.clip(y / f, -lim, lim)
                ys, xs = f * np.tan(phi), x / np.cos(phi)
                x, y = x + self.sv * (xs - x), y + self.sv * (ys - y)
            if self.sh:
                th = np.clip(x / f, -lim, lim)
                xs, ys = f * np.tan(th), y / np.cos(th)
                x, y = x + self.sh * (xs - x), y + self.sh * (ys - y)
        else:
            r = np.sqrt(x * x + y * y)
            th = np.clip(2 * np.arctan(r / (2 * f)), 0, lim)
            rs = f * np.tan(th)
            s = np.where(r > 1e-9, rs / np.maximum(r, 1e-9), 1.0)
            s = 1.0 + self.sd * (s - 1.0)
            x, y = x * s, y * s
        return self.frame.from_centered(x, y)


class HomographyStage(Stage):
    def __init__(self, h: np.ndarray, frame: Frame):
        self.hinv, self.frame = np.linalg.inv(h), frame

    def inverse(self, u, v):
        x, y = self.frame.to_centered(u, v)
        x, y = apply_h(self.hinv, x, y)
        return self.frame.from_centered(x, y)


def rotation_matrix(angle_deg: float) -> np.ndarray:
    """Visual counter-clockwise rotation in y-down centred coordinates."""
    a = math.radians(angle_deg)
    return np.array([[math.cos(a), math.sin(a), 0], [-math.sin(a), math.cos(a), 0], [0, 0, 1]])


class WarpStage(Stage):
    """Mesh warp: grid point P moves to P + D(P); inverse solved by fixed-point iteration."""

    FIELD = 513

    def __init__(self, disp: np.ndarray):
        self.d = np.asarray(disp, dtype=np.float64)  # (rows+1, cols+1, 2)
        # Pre-sample a dense displacement field so per-pixel lookups go through cv2.remap.
        g = np.linspace(0.0, 1.0, self.FIELD)
        gu, gv = np.meshgrid(g, g)
        du, dv = self.grid_sample(gu, gv)
        self.field = np.dstack([du, dv]).astype(np.float32)

    def sample(self, u, v):
        n = self.FIELD - 1
        mx = (np.asarray(u, dtype=np.float32) * n)
        my = (np.asarray(v, dtype=np.float32) * n)
        shape = mx.shape
        mx, my = mx.reshape(-1, mx.shape[-1] if mx.ndim > 1 else mx.size), my.reshape(mx.shape)
        f = cv2.remap(self.field, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return f[..., 0].reshape(shape).astype(np.float64), f[..., 1].reshape(shape).astype(np.float64)

    def grid_sample(self, u, v):
        rows, cols = self.d.shape[0] - 1, self.d.shape[1] - 1
        gu = np.clip(u, 0, 1) * cols
        gv = np.clip(v, 0, 1) * rows
        i0 = np.clip(np.floor(gv).astype(np.int64), 0, rows - 1)
        j0 = np.clip(np.floor(gu).astype(np.int64), 0, cols - 1)
        fy, fx = gv - i0, gu - j0
        d = self.d
        out = []
        for k in range(2):
            a = d[i0, j0, k] * (1 - fx) + d[i0, j0 + 1, k] * fx
            b = d[i0 + 1, j0, k] * (1 - fx) + d[i0 + 1, j0 + 1, k] * fx
            out.append(a * (1 - fy) + b * fy)
        return out[0], out[1]

    def inverse(self, u, v):
        pu, pv = np.array(u, dtype=np.float64), np.array(v, dtype=np.float64)
        for _ in range(8):
            du, dv = self.sample(pu, pv)
            pu, pv = u - du, v - dv
        return pu, pv


def focal_from_35mm(f35: float | None) -> float:
    return (f35 or DEFAULT_FOCAL_35MM) / HALF_DIAG_35MM


def build_stages(c: Corrections, frame: Frame, focal: float) -> dict[str, Stage]:
    st: dict[str, Stage] = {name: IdentityStage() for name in STAGES}
    if c.distortion.enabled and c.distortion.intensity:
        st["distortion"] = DistortionStage(c.distortion.kind, c.distortion.intensity, frame)
    vp = c.volume
    if vp.enabled and ((vp.mode == "hv" and (vp.horizontal or vp.vertical)) or (vp.mode == "diagonal" and vp.diagonal)):
        st["volume"] = VolumeStage(vp.mode, vp.horizontal, vp.vertical, vp.diagonal, frame, focal)
    if c.perspective.enabled:
        h = perspective_homography(c.perspective, frame, focal)
        if not np.allclose(h, np.eye(3)):
            st["perspective"] = HomographyStage(h, frame)
    if c.horizon.enabled and c.horizon.angle:
        st["horizon"] = HomographyStage(rotation_matrix(c.horizon.angle), frame)
    if c.warp.enabled and c.warp.disp:
        d = np.asarray(c.warp.disp, dtype=np.float64)
        if np.any(d):
            st["warp"] = WarpStage(d)
    return st


def chain_inverse(stages: dict[str, Stage], u, v, upto: str | None = None):
    """Map final-frame coords back to source coords through stages before ``upto`` (exclusive).

    ``upto=None`` uses every stage.
    """
    names = list(STAGES)
    if upto is not None:
        names = names[: names.index(upto)] if upto in names else names
    for name in reversed(names):
        u, v = stages[name].inverse(u, v)
    return u, v
