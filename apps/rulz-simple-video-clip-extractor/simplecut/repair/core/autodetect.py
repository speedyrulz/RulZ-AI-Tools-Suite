"""Automatic perspective/horizon estimation and automatic crop.

From RulZ Image Distortion Repair. The one change: the perspective fit uses the small
Nelder-Mead below instead of scipy.optimize.minimize, so the app doesn't need SciPy.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .geometry import Frame, apply_h, rotation_homography


def nelder_mead(fun, x0, xatol: float = 1e-4, fatol: float = 1e-4, maxiter: int = 200):
    """Minimise fun from x0; returns (x, f). The same algorithm, start simplex and stopping
    rule as scipy.optimize.minimize(method="Nelder-Mead") with default (non-adaptive) options."""
    rho, chi, psi, sigma = 1.0, 2.0, 0.5, 0.5
    x0 = np.asarray(x0, dtype=np.float64).ravel()
    n = len(x0)
    sim = np.empty((n + 1, n))
    sim[0] = x0
    for k in range(n):
        y = x0.copy()
        y[k] = 1.05 * y[k] if y[k] != 0 else 0.00025
        sim[k + 1] = y
    fsim = np.array([fun(s) for s in sim], dtype=np.float64)
    order = np.argsort(fsim)
    sim, fsim = sim[order], fsim[order]
    for _ in range(maxiter - 1):
        if np.max(np.abs(sim[1:] - sim[0])) <= xatol and np.max(np.abs(fsim[0] - fsim[1:])) <= fatol:
            break
        xbar = sim[:-1].sum(axis=0) / n
        xr = (1 + rho) * xbar - rho * sim[-1]
        fxr = fun(xr)
        shrink = False
        if fxr < fsim[0]:
            xe = (1 + rho * chi) * xbar - rho * chi * sim[-1]
            fxe = fun(xe)
            sim[-1], fsim[-1] = (xe, fxe) if fxe < fxr else (xr, fxr)
        elif fxr < fsim[-2]:
            sim[-1], fsim[-1] = xr, fxr
        elif fxr < fsim[-1]:
            xc = (1 + psi * rho) * xbar - psi * rho * sim[-1]
            fxc = fun(xc)
            if fxc <= fxr:
                sim[-1], fsim[-1] = xc, fxc
            else:
                shrink = True
        else:
            xcc = (1 - psi) * xbar + psi * sim[-1]
            fxcc = fun(xcc)
            if fxcc < fsim[-1]:
                sim[-1], fsim[-1] = xcc, fxcc
            else:
                shrink = True
        if shrink:
            for j in range(1, n + 1):
                sim[j] = sim[0] + sigma * (sim[j] - sim[0])
                fsim[j] = fun(sim[j])
        order = np.argsort(fsim)
        sim, fsim = sim[order], fsim[order]
    return sim[0], float(fsim[0])


def detect_segments(rgb: np.ndarray, frame: Frame, min_len_frac: float = 0.04):
    """Return line segments as an (N, 4) array of centred coordinates plus their lengths."""
    gray = cv2.cvtColor(rgb[..., :3], cv2.COLOR_RGB2GRAY)
    if gray.dtype != np.uint8:
        gray = cv2.convertScaleAbs(gray, alpha=255.0 / max(1.0, float(gray.max())))
    h, w = gray.shape
    min_len = min_len_frac * max(w, h)
    segs = None
    if hasattr(cv2, "createLineSegmentDetector"):
        try:
            lsd = cv2.createLineSegmentDetector()
            found = lsd.detect(gray)[0]
            if found is not None:
                segs = found.reshape(-1, 4)
        except cv2.error:
            segs = None
    if segs is None:
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
        found = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=60,
                                minLineLength=int(min_len), maxLineGap=int(min_len / 4))
        segs = found.reshape(-1, 4).astype(np.float64) if found is not None else np.zeros((0, 4))
    segs = segs.astype(np.float64)
    lengths = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    keep = lengths >= min_len
    segs, lengths = segs[keep], lengths[keep]
    x1, y1 = frame.to_centered((segs[:, 0] + 0.5) / w, (segs[:, 1] + 0.5) / h)
    x2, y2 = frame.to_centered((segs[:, 2] + 0.5) / w, (segs[:, 3] + 0.5) / h)
    return np.stack([x1, y1, x2, y2], axis=1), lengths


def _angles(segs):
    """Angle of each segment, folded into [0, 180)."""
    a = np.degrees(np.arctan2(segs[:, 3] - segs[:, 1], segs[:, 2] - segs[:, 0]))
    return np.mod(a, 180.0)


def classify(segs, tol=25.0):
    a = _angles(segs)
    vert = np.abs(a - 90.0) < tol
    horiz = (a < tol) | (a > 180.0 - tol)
    return vert, horiz


def auto_perspective(rgb: np.ndarray, frame: Frame, focal: float, mode: str = "both"):
    """Estimate [pitch, yaw, roll] (degrees) that makes verticals/horizontals straight.

    Returns None when not enough structure is found.
    """
    segs, lengths = detect_segments(rgb, frame)
    if len(segs) < 2:
        return None
    vert, horiz = classify(segs)
    use_v = mode in ("both", "verticals")
    use_h = mode in ("both", "horizontals")
    if (use_v and vert.sum() < 2) and (use_h and horiz.sum() < 2):
        return None
    if mode == "verticals" and vert.sum() < 2:
        return None
    if mode == "horizontals" and horiz.sum() < 2:
        return None
    wv = lengths[vert] / lengths.max()
    wh = lengths[horiz] / lengths.max()
    sv, sh = segs[vert], segs[horiz]
    clip = math.radians(8.0) ** 2

    def deviation(seg, want_vertical):
        dx = seg[:, 2] - seg[:, 0]
        dy = seg[:, 3] - seg[:, 1]
        a = np.arctan2(np.abs(dx), np.abs(dy)) if want_vertical else np.arctan2(np.abs(dy), np.abs(dx))
        return np.minimum(a * a, clip)

    def transform(seg, h):
        x1, y1 = apply_h(h, seg[:, 0], seg[:, 1])
        x2, y2 = apply_h(h, seg[:, 2], seg[:, 3])
        return np.stack([x1, y1, x2, y2], axis=1)

    def cost(params):
        pitch, yaw, roll = params
        h = rotation_homography(pitch, yaw, roll, focal)
        c = 0.0
        if use_v and len(sv):
            c += float(np.sum(wv * deviation(transform(sv, h), True)))
        if use_h and len(sh):
            c += float(np.sum(wh * deviation(transform(sh, h), False)))
        return c + 1e-5 * float(np.dot(params, params))

    free = {"both": (0, 1, 2), "verticals": (0, 2), "horizontals": (1, 2)}[mode]

    def wrapped(sub):
        p = np.zeros(3)
        p[list(free)] = sub
        return cost(p)

    best = None
    for start in (0.0, 8.0, -8.0):
        x0 = np.full(len(free), 0.0)
        x0[0] = start
        res = nelder_mead(wrapped, x0, xatol=0.02, fatol=1e-7, maxiter=600)
        if best is None or res[1] < best[1]:
            best = res
    p = np.zeros(3)
    p[list(free)] = best[0]
    if np.any(np.abs(p) > 40):
        return None
    return [float(a) for a in p]


def auto_horizon(rgb: np.ndarray, frame: Frame):
    """Estimate the rotation (degrees, CCW positive) that levels the image, or None."""
    segs, lengths = detect_segments(rgb, frame)
    if len(segs) == 0:
        return None
    a = _angles(segs)
    dev_h = np.where(a > 90, a - 180, a)            # deviation of near-horizontals
    dev_v = a - 90.0                                 # deviation of near-verticals
    pick_h = np.abs(dev_h) < 15
    pick_v = np.abs(dev_v) < 15
    devs = np.concatenate([dev_h[pick_h], dev_v[pick_v]])
    w = np.concatenate([lengths[pick_h], lengths[pick_v] * 0.6])
    if len(devs) < 2:
        return None
    order = np.argsort(devs)
    cw = np.cumsum(w[order])
    median = float(devs[order][np.searchsorted(cw, cw[-1] / 2)])
    # Segment angles are measured in y-down space: a positive deviation is a clockwise tilt,
    # which needs a counter-clockwise (positive) correction.
    return round(median, 2)


# --------------------------------------------------------------------------- auto crop

def _fits(invalid_integral, rw, rh):
    """Boolean map of top-left positions where an rw x rh rectangle contains no invalid cell."""
    ii = invalid_integral
    h, w = ii.shape[0] - 1, ii.shape[1] - 1
    if rw > w or rh > h or rw < 1 or rh < 1:
        return None
    s = ii[rh:, rw:] - ii[:-rh, rw:] - ii[rh:, :-rw] + ii[:-rh, :-rw]
    return s == 0


def largest_rect(valid: np.ndarray, ratio: float | None):
    """Largest axis-aligned rectangle inside ``valid`` (h x w bool).

    ``ratio`` is width/height in *cells*; None = unconstrained. Returns (x, y, w, h) in cells.
    """
    h, w = valid.shape
    if valid.all():
        if ratio is None:
            return 0, 0, w, h
    ii = np.zeros((h + 1, w + 1), dtype=np.int64)
    ii[1:, 1:] = np.cumsum(np.cumsum(~valid, axis=0), axis=1)
    cy, cx = h / 2, w / 2

    def best_pos(ok, rw, rh):
        ys, xs = np.nonzero(ok)
        d = (xs + rw / 2 - cx) ** 2 + (ys + rh / 2 - cy) ** 2
        k = int(np.argmin(d))
        return int(xs[k]), int(ys[k])

    if ratio is not None:
        lo, hi = 0.0, min(h, w / ratio)
        best = None
        for _ in range(24):
            mid = (lo + hi) / 2
            rh = max(1, int(mid))
            rw = max(1, int(round(mid * ratio)))
            ok = _fits(ii, rw, rh)
            if ok is not None and ok.any():
                lo = mid
                best = (*best_pos(ok, rw, rh), rw, rh)
            else:
                hi = mid
        return best or (0, 0, w, h)

    # Unconstrained: maximal rectangle via histogram stacks.
    heights = np.zeros(w, dtype=np.int64)
    best_area, best = 0, (0, 0, w, h)
    for row in range(h):
        heights = np.where(valid[row], heights + 1, 0)
        stack = []
        for col in range(w + 1):
            cur = heights[col] if col < w else 0
            start = col
            while stack and stack[-1][1] >= cur:
                s, hh = stack.pop()
                area = hh * (col - s)
                if area > best_area:
                    best_area, best = area, (s, row - hh + 1, col - s, hh)
                start = s
            stack.append((start, cur))
    return best


def aspect_value(aspect: str, custom: float, frame: Frame) -> float | None:
    if aspect == "unconstrained":
        return None
    if aspect == "preserve":
        return frame.aspect
    if aspect == "custom":
        return custom
    try:
        a, b = aspect.split("/")
        return float(a) / float(b)
    except ValueError:
        return frame.aspect


def auto_crop_rect(valid: np.ndarray, ratio_px: float | None, frame: Frame):
    """Normalised [x, y, w, h] of the largest valid rectangle with the given pixel aspect."""
    h, w = valid.shape
    ratio_cells = None
    if ratio_px is not None:
        # cells are (frame.width / w) x (frame.height / h) pixels
        ratio_cells = ratio_px * (frame.height / h) / (frame.width / w)
        # an image in portrait keeps the requested orientation-agnostic ratio
        if (ratio_px > 1) != (frame.aspect > 1) and abs(ratio_px - 1) > 1e-6:
            ratio_cells = (1 / ratio_px) * (frame.height / h) / (frame.width / w)
    x, y, rw, rh = largest_rect(valid, ratio_cells)
    return [x / w, y / h, rw / w, rh / h]
