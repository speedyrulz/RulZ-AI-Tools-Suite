"""Rendering of a corrected image at preview or full resolution.

From RulZ Image Distortion Repair. Changes: the image is a plain StillImage (a video frame)
instead of a file loaded with its EXIF data, and the validity mask / automatic crop are
module functions, so the export pipeline computes exactly the crop the window showed.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .autodetect import aspect_value, auto_crop_rect
from .geometry import Frame, build_stages, chain_inverse, focal_from_35mm
from .params import Corrections, Orientation
from .tiltshift import apply_tiltshift

PREVIEW_MAX = 1800


@dataclass
class StillImage:
    """The picture being corrected: RGB(A) uint8/uint16 pixels plus what the maths needs."""
    pixels: np.ndarray
    has_alpha: bool = False
    focal_35mm: float | None = None     # video has no EXIF: the 24 mm default is used
    path: str = ""


def validity_mask_for(c: Corrections, frame: Frame, focal: float, cells: int = 220,
                      upto: str | None = None) -> np.ndarray:
    """Cells of the corrected frame that sample inside the source (False = empty area)."""
    stages = build_stages(c, frame, focal)
    if frame.width >= frame.height:
        w, h = cells, max(2, round(cells / frame.aspect))
    else:
        h, w = cells, max(2, round(cells * frame.aspect))
    u, v = np.meshgrid((np.arange(w) + 0.5) / w, (np.arange(h) + 0.5) / h)
    su, sv = chain_inverse(stages, u, v, upto)
    eps = 0.5 / max(frame.width, frame.height)
    return (su >= -eps) & (su <= 1 + eps) & (sv >= -eps) & (sv <= 1 + eps)


def crop_rect_for(c: Corrections, frame: Frame, focal: float) -> list:
    """Normalised [x, y, w, h] the Crop module keeps (the whole frame when it is off)."""
    cp = c.crop
    if not cp.enabled:
        return [0.0, 0.0, 1.0, 1.0]
    if not cp.auto and cp.rect:
        return list(cp.rect)
    valid = validity_mask_for(c, frame, focal)
    if valid.all() and cp.aspect in ("preserve", "unconstrained"):
        return [0.0, 0.0, 1.0, 1.0]
    # the mask is sampled at cell centres; erode one cell so edges never show empty slivers
    valid = cv2.erode(valid.astype(np.uint8), np.ones((3, 3), np.uint8), borderType=cv2.BORDER_REPLICATE) > 0
    return auto_crop_rect(valid, aspect_value(cp.aspect, cp.custom_ratio, frame), frame)


def orient(px: np.ndarray, o: Orientation) -> np.ndarray:
    if o.rot90 % 4:
        px = np.rot90(px, o.rot90 % 4)
    if o.flip_h:
        px = px[:, ::-1]
    if o.flip_v:
        px = px[::-1]
    return np.ascontiguousarray(px)


@dataclass
class RenderResult:
    rgba: np.ndarray            # uint8 (preview) or uint8/uint16 (full) RGBA, alpha = valid
    crop: list                  # normalised crop rect inside the stage frame
    frame_size: tuple[int, int]  # full-resolution frame (oriented) size
    upto: str | None


class Renderer:
    """Holds one image and renders it through the correction pipeline."""

    def __init__(self, image: StillImage):
        self.image = image
        self.focal = focal_from_35mm(image.focal_35mm)
        self._orient_key = None
        self._full = None
        self._preview = None

    # -- sources -----------------------------------------------------------------------
    def sources(self, o: Orientation):
        key = (o.rot90 % 4, o.flip_h, o.flip_v)
        if key != self._orient_key:
            self._full = orient(self.image.pixels, o)
            h, w = self._full.shape[:2]
            s = min(1.0, PREVIEW_MAX / max(w, h))
            prev = self._full
            if prev.dtype == np.uint16:
                prev = (prev >> 8).astype(np.uint8)
            if s < 1.0:
                prev = cv2.resize(prev, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
            self._preview = np.ascontiguousarray(prev)
            self._orient_key = key
        return self._full, self._preview

    def frame(self, o: Orientation) -> Frame:
        full, _ = self.sources(o)
        return Frame(full.shape[1], full.shape[0])

    # -- crop ---------------------------------------------------------------------------
    def validity_mask(self, c: Corrections, cells: int = 220, upto: str | None = None) -> np.ndarray:
        return validity_mask_for(c, self.frame(c.orientation), self.focal, cells, upto)

    def crop_rect(self, c: Corrections) -> list:
        return crop_rect_for(c, self.frame(c.orientation), self.focal)

    # -- render -------------------------------------------------------------------------
    def render(self, c: Corrections, upto: str | None = None, full: bool = False,
               max_side: int | None = None, apply_crop: bool = True, apply_effects: bool = True) -> RenderResult:
        """Render the image.

        ``upto`` names the first stage *not* applied (a geometry stage name, "crop" or
        "tiltshift"); None renders everything.
        """
        full_px, prev_px = self.sources(c.orientation)
        src = full_px if full else prev_px
        frame = Frame(full_px.shape[1], full_px.shape[0])
        stages = build_stages(c, frame, self.focal)
        geo_upto = upto if upto in ("distortion", "volume", "perspective", "horizon", "warp") else None
        do_crop = apply_crop and upto not in ("distortion", "volume", "perspective", "horizon", "warp", "crop")
        crop = self.crop_rect(c) if do_crop else [0.0, 0.0, 1.0, 1.0]

        sh, sw = src.shape[:2]
        out_w = max(1, round(sw * crop[2]))
        out_h = max(1, round(sh * crop[3]))
        if max_side and max(out_w, out_h) > max_side:
            s = max_side / max(out_w, out_h)
            out_w, out_h = max(1, round(out_w * s)), max(1, round(out_h * s))

        interp = cv2.INTER_CUBIC if full else cv2.INTER_LINEAR
        out = np.zeros((out_h, out_w, 4), dtype=src.dtype)
        maxval = 65535 if src.dtype == np.uint16 else 255
        strip = 512
        xs = crop[0] + (np.arange(out_w) + 0.5) / out_w * crop[2]
        for y0 in range(0, out_h, strip):
            y1 = min(out_h, y0 + strip)
            ys = crop[1] + (np.arange(y0, y1) + 0.5) / out_h * crop[3]
            u, v = np.meshgrid(xs, ys)
            su, sv = chain_inverse(stages, u, v, geo_upto)
            mx = (su * sw - 0.5).astype(np.float32)
            my = (sv * sh - 0.5).astype(np.float32)
            px = cv2.remap(src, mx, my, interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            valid = (mx >= -0.5) & (mx <= sw - 0.5) & (my >= -0.5) & (my <= sh - 0.5)
            out[y0:y1, :, :3] = px[..., :3]
            src_alpha = px[..., 3] if src.shape[2] == 4 else maxval
            out[y0:y1, :, 3] = np.where(valid, src_alpha, 0)

        if apply_effects and upto is None and c.tiltshift.enabled:
            f = out.astype(np.float32) / maxval
            f = apply_tiltshift(f, c.tiltshift)
            out = np.clip(f * maxval + 0.5, 0, maxval).astype(src.dtype)
        return RenderResult(out, crop, (frame.width, frame.height), upto)

    def render_full(self, c: Corrections) -> np.ndarray:
        """Full-resolution output: RGBA when the source has transparency, RGB otherwise."""
        res = self.render(c, full=True)
        return np.ascontiguousarray(res.rgba if self.image.has_alpha else res.rgba[..., :3])

    def render_full_rgb(self, c: Corrections) -> np.ndarray:
        res = self.render(c, full=True)
        return np.ascontiguousarray(res.rgba[..., :3])

    def set_pixels(self, pixels: np.ndarray):
        """Swap in a new source picture of the same kind (e.g. the current video frame)."""
        self.image.pixels = pixels
        self._orient_key = None

    def mapper(self, c: Corrections, src_size: tuple[int, int], max_side: int | None = None,
               full: bool = False) -> "FrameMapper":
        """Cached remap tables for applying ``c`` to many frames of ``src_size`` (oriented-before) pixels."""
        frame = self.frame(c.orientation)
        return FrameMapper(c, frame, self.focal, self.crop_rect(c), src_size, max_side,
                           cv2.INTER_CUBIC if full else cv2.INTER_LINEAR)


class FrameMapper:
    """Applies one fixed set of corrections to a stream of same-sized RGB frames.

    The geometric chain is evaluated once into remap tables, so each frame costs one
    ``cv2.remap`` (plus the tilt-shift blur when that effect is on).
    """

    def __init__(self, c: Corrections, frame: Frame, focal: float, crop: list, src_size: tuple[int, int],
                 max_side: int | None = None, interp: int = cv2.INTER_LINEAR):
        self.orientation = c.orientation
        self.crop = list(crop)
        self.tiltshift = c.tiltshift if c.tiltshift.enabled else None
        self.interp = interp
        w, h = src_size
        if self.orientation.rot90 % 2:
            w, h = h, w                       # oriented frame size
        self.src_size = (w, h)
        out_w = max(1, round(w * crop[2]))
        out_h = max(1, round(h * crop[3]))
        if max_side and max(out_w, out_h) > max_side:
            s = max_side / max(out_w, out_h)
            out_w, out_h = max(1, round(out_w * s)), max(1, round(out_h * s))
        self.out_size = (out_w, out_h)
        stages = build_stages(c, frame, focal)
        # pure resize (no geometry, no crop): skip the remap entirely
        self.plain = all(type(s).__name__ == "IdentityStage" for s in stages.values()) and crop == [0.0, 0.0, 1.0, 1.0]
        self.mx = self.my = self.invalid = None
        if self.plain:
            return
        # Clip Extractor change: the tables are built in strips of rows, so a 4K frame needs
        # tens of MB of temporaries instead of most of a GB.
        xs = crop[0] + (np.arange(out_w) + 0.5) / out_w * crop[2]
        ys = crop[1] + (np.arange(out_h) + 0.5) / out_h * crop[3]
        self.mx = np.empty((out_h, out_w), np.float32)
        self.my = np.empty((out_h, out_w), np.float32)
        for y0 in range(0, out_h, 256):
            u, v = np.meshgrid(xs, ys[y0:y0 + 256])
            su, sv = chain_inverse(stages, u, v)
            self.mx[y0:y0 + 256] = su * w - 0.5
            self.my[y0:y0 + 256] = sv * h - 0.5
        self.invalid = ~((self.mx >= -0.5) & (self.mx <= w - 0.5) & (self.my >= -0.5) & (self.my <= h - 0.5))
        if not self.invalid.any():
            self.invalid = None

    def apply(self, rgb: np.ndarray, effects: bool = True) -> np.ndarray:
        """RGB uint8 frame (not yet oriented) → corrected RGB uint8 frame."""
        px = orient(rgb, self.orientation)
        if px.shape[1] != self.src_size[0] or px.shape[0] != self.src_size[1]:
            raise ValueError(f"frame size {px.shape[1]}x{px.shape[0]} does not match mapper {self.src_size}")
        if self.plain:
            out = px if (px.shape[1], px.shape[0]) == self.out_size else \
                cv2.resize(px, self.out_size, interpolation=cv2.INTER_AREA)
        else:
            out = cv2.remap(px, self.mx, self.my, self.interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            if self.invalid is not None:
                out[self.invalid] = 0
        if effects and self.tiltshift is not None:
            f = apply_tiltshift(out.astype(np.float32) / 255.0, self.tiltshift)
            out = np.clip(f * 255.0 + 0.5, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(out)
