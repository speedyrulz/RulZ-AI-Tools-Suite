"""Correction parameters for every module, plus (de)serialisation.

Coordinates stored here are *normalised frame coordinates* in [0, 1] x [0, 1] of the input of
the stage that owns them (see ``geometry.STAGES``), so they are independent of preview/full
resolution.

Copied unchanged from RulZ Image Distortion Repair (package viewpointer).
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Optional

Point = tuple[float, float]
Line = tuple[Point, Point]


@dataclass
class Orientation:
    rot90: int = 0          # number of 90° counter-clockwise turns
    flip_h: bool = False
    flip_v: bool = False


@dataclass
class DistortionParams:
    enabled: bool = False
    kind: str = "barrel"    # barrel | pincushion | fisheye
    intensity: int = 50     # 0..100


@dataclass
class VolumeParams:
    enabled: bool = False
    mode: str = "hv"        # hv | diagonal
    horizontal: int = 100   # 0..200
    vertical: int = 0       # 0..200
    diagonal: int = 150     # 0..200


@dataclass
class PerspectiveParams:
    enabled: bool = False
    mode: str = "verticals"         # auto | verticals | horizontals | rect | eight
    auto_mode: str = "both"         # both | verticals | horizontals
    intensity: int = 100            # 0..100
    up_down: int = 0                # -125..125
    left_right: int = 0             # -125..125
    hv_ratio: int = 0               # -100..100
    lines: list = field(default_factory=list)   # lines/points for the current mode
    auto_angles: Optional[list] = None          # [pitch, yaw, roll] found by auto mode


@dataclass
class HorizonParams:
    enabled: bool = False
    angle: float = 0.0              # degrees, -90..90, positive = counter-clockwise


@dataclass
class WarpParams:
    enabled: bool = False
    rows: int = 4
    cols: int = 4
    # (rows+1) x (cols+1) x 2 displacement in frame units; empty = identity
    disp: list = field(default_factory=list)


@dataclass
class CropParams:
    enabled: bool = True
    auto: bool = True
    aspect: str = "preserve"        # unconstrained | preserve | W/H | custom
    custom_ratio: float = 1.0
    constrain: bool = False
    rect: Optional[list] = None     # [x, y, w, h] normalised, manual crop only


@dataclass
class TiltShiftParams:
    enabled: bool = False
    center: list = field(default_factory=lambda: [0.5, 0.5])
    angle: float = 0.0              # degrees
    sharp_top: float = 0.10         # distance centre → sharp line (frame-height units)
    sharp_bottom: float = 0.10
    trans_top: float = 0.25         # distance centre → transition (dashed) line
    trans_bottom: float = 0.25
    intensity: int = 50             # 0..100
    blur_shape: str = "circular"
    sym_position: bool = True
    sym_blur: bool = True


@dataclass
class Guides:
    vertical: list = field(default_factory=list)    # x positions, normalised
    horizontal: list = field(default_factory=list)  # y positions, normalised


@dataclass
class Corrections:
    orientation: Orientation = field(default_factory=Orientation)
    distortion: DistortionParams = field(default_factory=DistortionParams)
    volume: VolumeParams = field(default_factory=VolumeParams)
    perspective: PerspectiveParams = field(default_factory=PerspectiveParams)
    horizon: HorizonParams = field(default_factory=HorizonParams)
    warp: WarpParams = field(default_factory=WarpParams)
    crop: CropParams = field(default_factory=CropParams)
    tiltshift: TiltShiftParams = field(default_factory=TiltShiftParams)

    def copy(self) -> "Corrections":
        return copy.deepcopy(self)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Corrections":
        c = cls()
        for name, sub in d.items():
            if hasattr(c, name) and isinstance(sub, dict):
                obj = getattr(c, name)
                for k, v in sub.items():
                    if hasattr(obj, k):
                        setattr(obj, k, v)
        return c

    def is_identity(self) -> bool:
        return self.to_dict() == Corrections().to_dict()


MODULES = ("distortion", "volume", "perspective", "warp", "horizon", "crop", "tiltshift")
