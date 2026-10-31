"""Probe a video with ffprobe: stream info + list of keyframe timestamps."""
from __future__ import annotations

import json
import shutil
from bisect import bisect_left, bisect_right
import subprocess
from dataclasses import dataclass, field
from fractions import Fraction

CREATE_NO_WINDOW = 0x08000000  # keep console windows from popping up on Windows


def _run(args: list[str]) -> str:
    r = subprocess.run(args, capture_output=True, text=True, creationflags=CREATE_NO_WINDOW,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0:
        # ffprobe's own message ("Invalid data found when processing input", ...) says what's wrong
        msg = (r.stderr or "").strip() or f"exit code {r.returncode}"
        raise RuntimeError(f"ffprobe failed: {msg}")
    return r.stdout


def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


@dataclass
class VideoInfo:
    path: str
    width: int = 0
    height: int = 0
    fps: float = 30.0
    duration: float = 0.0          # seconds
    start_time: float = 0.0        # first video pts in seconds
    fmt_start_time: float = 0.0    # container start time; ffmpeg input -ss is relative to this
    codec: str = ""
    container: str = ""
    has_audio: bool = False
    audio_codec: str = ""
    rotation: int = 0              # clockwise degrees the source asks players to apply
    keyframes: list[float] = field(default_factory=list)  # sorted pts seconds
    keyframe_dts: dict[float, float] = field(default_factory=dict)  # pts -> dts seconds

    @property
    def frame_duration(self) -> float:
        return 1.0 / self.fps if self.fps > 0 else 1 / 30

    @property
    def end_time(self) -> float:
        return self.start_time + self.duration

    # --- keyframe helpers (binary search: `keyframes` is sorted) ------------
    def is_keyframe_time(self, t: float) -> bool:
        return self.keyframe_at(t) is not None

    def keyframe_at(self, t: float) -> float | None:
        """The first keyframe within half a frame of t."""
        tol = self.frame_duration / 2
        kfs = self.keyframes
        i = bisect_left(kfs, t - tol)
        if i < len(kfs) and kfs[i] <= t + tol:
            return kfs[i]
        return None

    def next_keyframe(self, t: float) -> float | None:
        tol = self.frame_duration / 2
        i = bisect_right(self.keyframes, t + tol)
        return self.keyframes[i] if i < len(self.keyframes) else None

    def prev_keyframe(self, t: float) -> float | None:
        tol = self.frame_duration / 2
        i = bisect_left(self.keyframes, t - tol)
        return self.keyframes[i - 1] if i > 0 else None


def probe(path: str) -> VideoInfo:
    info = VideoInfo(path=path)

    fmt = json.loads(_run([
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", path,
    ]))
    info.container = fmt.get("format", {}).get("format_name", "")
    info.fmt_start_time = float(fmt.get("format", {}).get("start_time", 0) or 0)
    info.duration = float(fmt.get("format", {}).get("duration", 0) or 0)

    vstream = None
    for s in fmt.get("streams", []):
        if s.get("codec_type") == "video" and vstream is None:
            vstream = s
        elif s.get("codec_type") == "audio" and not info.has_audio:
            info.has_audio = True
            info.audio_codec = s.get("codec_name", "")
    if vstream is None:
        raise ValueError("No video stream found")

    info.width = int(vstream.get("width", 0))
    info.height = int(vstream.get("height", 0))
    info.codec = vstream.get("codec_name", "")
    rate = vstream.get("avg_frame_rate") or vstream.get("r_frame_rate") or "30/1"
    try:
        fr = Fraction(rate)
        info.fps = float(fr) if fr > 0 else 30.0
    except (ValueError, ZeroDivisionError):
        info.fps = 30.0
    if vstream.get("duration"):
        info.duration = float(vstream["duration"])
    info.start_time = float(vstream.get("start_time", 0) or 0)
    for sd in vstream.get("side_data_list", []) or []:
        if "rotation" in sd:
            # ffprobe reports counter-clockwise-positive; we keep clockwise.
            info.rotation = int(round(-float(sd["rotation"]))) % 360

    # Keyframe timestamps: walk the packets of the first video stream.
    out = _run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "packet=pts_time,dts_time,flags",
        "-of", "csv=p=0", path,
    ])
    kfs: dict[float, float] = {}
    for line in out.splitlines():
        parts = line.strip().split(",")
        if len(parts) < 3:
            continue
        pts, dts, flags = parts[0], parts[1], parts[2]
        if "K" in flags and pts not in ("", "N/A"):
            p = float(pts)
            kfs[p] = float(dts) if dts not in ("", "N/A") else p
    info.keyframes = sorted(kfs)
    info.keyframe_dts = kfs
    return info
