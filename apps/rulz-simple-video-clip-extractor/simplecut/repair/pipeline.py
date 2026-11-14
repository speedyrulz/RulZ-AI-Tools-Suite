"""Saved distortion-repair adjustments, bound to one video and applied to its frames.

Orientation model. The window edits the frame as it was on screen when it opened: the source's
own rotation flag, then the app's rotation and flip (the *base* orientation). The adjustments
are stored in that frame's normalised coordinates, together with the base orientation. When the
user rotates or flips afterwards, the corrected picture is turned as a whole:

    screen = D o corrections o base,   with D = current o base^-1

so perspective lines, the reshape grid and the crop stay on the content they were drawn on.
D is one of the eight 90-degree rotations/mirrors, usually the identity.
"""
from __future__ import annotations

import json
import math
import threading
from collections import OrderedDict
from dataclasses import dataclass, field, fields, is_dataclass

import cv2
import numpy as np

from .core.geometry import Frame, IdentityStage, build_stages, focal_from_35mm
from .core.params import Corrections, Orientation
from .core.render import FrameMapper, crop_rect_for
from .core.tiltshift import BLUR_SHAPES

PRESET_KIND = "rulz-distortion-repair"
PRESET_VERSION = 1
MAPPER_CACHE = 4


# ------------------------------------------------------------------ orientation
def orient(px: np.ndarray, rot_cw: int, flip: bool) -> np.ndarray:
    """Turn px clockwise by rot_cw degrees (a multiple of 90), then mirror it left-right.
    The same transform the main window applies to its preview (QTransform.rotate, then mirrored)."""
    k = (int(rot_cw) // 90) % 4
    if k:
        px = np.rot90(px, -k)
    if flip:
        px = px[:, ::-1]
    return px


def delta_orientation(base_rot: int, base_flip: bool, rot: int, flip: bool) -> tuple[int, bool]:
    """(rotation, flip) D with orient(orient(x, base), D) == orient(x, current) for every x."""
    probe = np.arange(6).reshape(2, 3)
    want = orient(probe, rot, flip)
    have = orient(probe, base_rot, base_flip)
    for r in (0, 90, 180, 270):
        for f in (False, True):
            got = orient(have, r, f)
            if got.shape == want.shape and np.array_equal(got, want):
                return r, f
    raise AssertionError("unreachable: the 8 orientations form a group")


def oriented_size(w: int, h: int, rot_cw: int) -> tuple[int, int]:
    return (h, w) if (rot_cw // 90) % 2 else (w, h)


# ------------------------------------------------------------------ settings
@dataclass
class RepairSettings:
    """What the window saves: the corrections and the orientation they were drawn in."""
    corrections: Corrections = field(default_factory=Corrections)
    base_rotation: int = 0       # the app's clockwise rotation (on top of the source flag) at the time
    base_flip: bool = False

    def __post_init__(self):
        self.corrections = self.corrections.copy()
        self.corrections.orientation = Orientation()   # orientation lives in base_* instead


MODULE_TITLES = (("distortion", "Distortion"), ("volume", "Volume deformation"),
                 ("perspective", "Perspective"), ("warp", "Reshape"), ("horizon", "Horizon"),
                 ("crop", "Crop"), ("tiltshift", "Miniature effect"))


def describe(c: Corrections) -> str:
    """One line listing the corrections that are switched on, e.g. for a tooltip."""
    parts = []
    if c.distortion.enabled and c.distortion.intensity:
        parts.append(f"Distortion ({c.distortion.kind} {c.distortion.intensity})")
    v = c.volume
    if v.enabled:
        parts.append("Volume deformation (" + (f"H {v.horizontal} / V {v.vertical}" if v.mode == "hv"
                                               else f"diagonal {v.diagonal}") + ")")
    p = c.perspective
    if p.enabled:
        how = "auto" if p.mode == "auto" else {"verticals": "2 verticals", "horizontals": "2 horizontals",
                                               "rect": "rectangle", "eight": "8 points"}.get(p.mode, p.mode)
        parts.append(f"Perspective ({how})")
    if c.warp.enabled and c.warp.disp and np.any(np.asarray(c.warp.disp, dtype=np.float64)):
        parts.append(f"Reshape ({c.warp.rows}x{c.warp.cols} grid)")
    if c.horizon.enabled and c.horizon.angle:
        parts.append(f"Horizon ({c.horizon.angle:+.2f}°)")
    if c.crop.enabled and (not c.crop.auto or c.crop.aspect not in ("preserve", "unconstrained")):
        parts.append("Crop (" + ("manual" if not c.crop.auto else c.crop.aspect) + ")")
    if c.tiltshift.enabled:
        parts.append(f"Miniature effect (blur {c.tiltshift.intensity})")
    return ", ".join(parts) if parts else "no corrections"


# ------------------------------------------------------------------ presets (JSON)
def to_json(c: Corrections) -> str:
    c = c.copy()
    c.orientation = Orientation()
    return json.dumps({"kind": PRESET_KIND, "version": PRESET_VERSION, "corrections": c.to_dict()}, indent=1)


def _sanitize(obj, data: dict):
    """Copy the fields of `data` into dataclass `obj` that have a usable type; skip the rest."""
    for f in fields(obj):
        if f.name not in data:
            continue
        cur, val = getattr(obj, f.name), data[f.name]
        if is_dataclass(cur):
            if isinstance(val, dict):
                _sanitize(cur, val)
        elif isinstance(cur, bool):
            if isinstance(val, bool):
                setattr(obj, f.name, val)
        elif isinstance(cur, int):
            if isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(val):
                setattr(obj, f.name, int(round(val)))
        elif isinstance(cur, float):
            if isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(val):
                setattr(obj, f.name, float(val))
        elif isinstance(cur, str):
            if isinstance(val, str):
                setattr(obj, f.name, val)
        elif isinstance(cur, list) or cur is None:
            if val is None or isinstance(val, list):
                setattr(obj, f.name, val)


def from_json(text: str) -> Corrections:
    """Corrections from a preset file's text. Raises ValueError when it isn't one."""
    try:
        d = json.loads(text)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ValueError(f"not a settings file ({e})") from e
    if not isinstance(d, dict) or d.get("kind") != PRESET_KIND or not isinstance(d.get("corrections"), dict):
        raise ValueError("not a distortion repair settings file")
    c = Corrections()
    _sanitize(c, d["corrections"])
    c.orientation = Orientation()
    if c.crop.rect is not None and not _usable_rect(c.crop.rect):
        raise ValueError("the settings could not be used: the crop rectangle is empty or outside the picture")
    # The miniature effect reads these per frame, and only when its blur is big enough to show,
    # which the small test render below may not reach.
    ts = c.tiltshift
    if not (isinstance(ts.center, list) and len(ts.center) == 2
            and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                    for v in ts.center)):
        raise ValueError("the settings could not be used: the miniature effect's centre is not a point")
    if ts.blur_shape not in BLUR_SHAPES:
        ts.blur_shape = "circular"
    try:  # anything still unusable (e.g. a malformed point list) is caught here, not mid-export
        # Render a small frame too, so settings that are only read per frame get used once.
        RepairPipeline(c, 64, 48).apply(np.zeros((48, 64, 3), np.uint8), full=True)
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"the settings could not be used: {e}") from e
    return c


def _usable_rect(rect) -> bool:
    """A manual crop [x, y, w, h] in normalised coordinates: four finite numbers, a size above
    zero, inside the frame (the window keeps it there, with 2 % as the smallest side)."""
    if not isinstance(rect, (list, tuple)) or len(rect) != 4:
        return False
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in rect):
        return False
    x, y, w, h = rect
    tol = 1e-6
    return w > 1e-3 and h > 1e-3 and x >= -tol and y >= -tol and x + w <= 1 + tol and y + h <= 1 + tol


# ------------------------------------------------------------------ pipeline
class RepairPipeline:
    """Corrections bound to one full-resolution frame size (the base-oriented frame).

    The automatic crop is worked out once here, so the preview, frame PNGs and exports all
    keep the same window of the picture. Remap tables are cached per input size.
    """

    def __init__(self, corrections: Corrections, frame_w: int, frame_h: int):
        c = corrections.copy()
        c.orientation = Orientation()
        self.c = c
        self.frame_size = (int(frame_w), int(frame_h))
        self.frame = Frame(int(frame_w), int(frame_h))
        self.focal = focal_from_35mm(None)
        self.crop = crop_rect_for(c, self.frame, self.focal)
        stages = build_stages(c, self.frame, self.focal)
        # True when the result would equal the input (e.g. modules switched on with nothing set)
        self.changes_nothing = (all(isinstance(s, IdentityStage) for s in stages.values())
                                and self.crop == [0.0, 0.0, 1.0, 1.0]
                                and not (c.tiltshift.enabled and c.tiltshift.intensity > 0))
        # full-resolution output, even on both sides (yuv420p video needs that)
        w = max(2, int(round(frame_w * self.crop[2])) & ~1)
        h = max(2, int(round(frame_h * self.crop[3])) & ~1)
        self.out_size = (w, h)
        self._mappers: OrderedDict[tuple, FrameMapper] = OrderedDict()
        self._lock = threading.Lock()

    def mapper(self, w: int, h: int, full: bool) -> FrameMapper:
        key = (int(w), int(h), bool(full))
        with self._lock:
            m = self._mappers.get(key)
            if m is not None:
                self._mappers.move_to_end(key)
                return m
        m = FrameMapper(self.c, self.frame, self.focal, self.crop, (int(w), int(h)),
                        interp=cv2.INTER_CUBIC if full else cv2.INTER_LINEAR)
        with self._lock:
            self._mappers[key] = m
            while len(self._mappers) > MAPPER_CACHE:
                self._mappers.popitem(last=False)
        return m

    def apply(self, rgb: np.ndarray, full: bool = False) -> np.ndarray:
        """Correct one base-oriented RGB uint8 frame (any size with the frame's aspect).

        full: cubic interpolation, and the result is exactly out_size (export quality)."""
        h, w = rgb.shape[:2]
        out = self.mapper(w, h, full).apply(rgb)
        if full:
            out = fit_size(out, self.out_size)
        return out


def fit_size(px: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """The mapper's frame trimmed to `size` (the even-sided output). It is never smaller, except
    for a crop under 2 pixels: then it is stretched, so a raw-video pipe still gets every byte
    of every frame it was promised."""
    ow, oh = size
    if px.shape[1] < ow or px.shape[0] < oh:
        px = cv2.resize(px, (max(ow, px.shape[1]), max(oh, px.shape[0])), interpolation=cv2.INTER_NEAREST)
    return np.ascontiguousarray(px[:oh, :ow])


@dataclass
class RepairState:
    """Saved adjustments on the open video: settings + the pipeline for its frame size."""
    settings: RepairSettings
    pipeline: RepairPipeline

    @classmethod
    def build(cls, settings: RepairSettings, coded_w: int, coded_h: int, source_rotation: int) -> "RepairState":
        base_total = (source_rotation + settings.base_rotation) % 360
        fw, fh = oriented_size(coded_w, coded_h, base_total)
        return cls(settings, RepairPipeline(settings.corrections, fw, fh))

    def delta(self, rotation: int, flip: bool) -> tuple[int, bool]:
        return delta_orientation(self.settings.base_rotation, self.settings.base_flip, rotation, flip)

    def output_size(self, rotation: int, flip: bool) -> tuple[int, int]:
        """Full-resolution size of the corrected frame as shown with the current rotation/flip."""
        d_rot, _ = self.delta(rotation, flip)
        return oriented_size(*self.pipeline.out_size, d_rot)

    def screen_frame(self, raw_rgb: np.ndarray, source_rotation: int, rotation: int, flip: bool,
                     full: bool = False) -> np.ndarray:
        """A decoded (unrotated) RGB frame -> the corrected frame as it appears on screen."""
        s = self.settings
        base = orient(raw_rgb, (source_rotation + s.base_rotation) % 360, s.base_flip)
        out = self.pipeline.apply(np.ascontiguousarray(base), full=full)
        d_rot, d_flip = self.delta(rotation, flip)
        return np.ascontiguousarray(orient(out, d_rot, d_flip))
