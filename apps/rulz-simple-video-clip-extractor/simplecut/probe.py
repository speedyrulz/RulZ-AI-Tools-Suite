"""Probe a video with ffprobe: stream info + list of keyframe timestamps."""
from __future__ import annotations

import json
import shutil
from bisect import bisect_left, bisect_right
import subprocess
from dataclasses import dataclass, field
from fractions import Fraction

CREATE_NO_WINDOW = 0x08000000  # keep console windows from popping up on Windows


def _run(args: list[str], on_start=None) -> str:
    """Run ffprobe and return its output. on_start(proc) receives the process, so another thread
    can kill a long scan; a killed scan raises RuntimeError like any failure."""
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            creationflags=CREATE_NO_WINDOW, encoding="utf-8", errors="replace")
    if on_start is not None:
        on_start(proc)
    out, err = proc.communicate()
    if proc.returncode != 0:
        # ffprobe's own message ("Invalid data found when processing input", ...) says what's wrong
        msg = (err or "").strip() or f"exit code {proc.returncode}"
        raise RuntimeError(f"ffprobe failed: {msg}")
    return out


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
    has_b_frames: int = 0          # decoder reorder delay (frames); 0 = no B-frames
    rate: str = ""                 # exact frame rate as ffprobe gives it, e.g. "30000/1001"
    pix_fmt: str = ""
    color_space: str = ""          # colour tags ("" when the file has none), ffmpeg names
    color_range: str = ""
    color_primaries: str = ""
    color_transfer: str = ""
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
    """Stream info and keyframes. The app calls the two halves separately so the video can be
    shown before the (slower) keyframe scan finishes."""
    info = probe_streams(path)
    info.keyframes, info.keyframe_dts = scan_keyframes(path)
    return info


def probe_streams(path: str) -> VideoInfo:
    """Container and stream info (fast: reads the header only)."""
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
    info.has_b_frames = int(vstream.get("has_b_frames", 0) or 0)
    rate = vstream.get("avg_frame_rate") or vstream.get("r_frame_rate") or "30/1"
    try:
        fr = Fraction(rate)
        info.fps = float(fr) if fr > 0 else 30.0
        info.rate = f"{fr.numerator}/{fr.denominator}" if fr > 0 else ""
    except (ValueError, ZeroDivisionError):
        info.fps = 30.0
    info.pix_fmt = vstream.get("pix_fmt", "") or ""
    for key in ("color_space", "color_range", "color_primaries", "color_transfer"):
        v = vstream.get(key, "") or ""
        setattr(info, key, "" if v in ("unknown", "reserved", "unspecified") else v)
    if vstream.get("duration"):
        info.duration = float(vstream["duration"])
    info.start_time = float(vstream.get("start_time", 0) or 0)
    for sd in vstream.get("side_data_list", []) or []:
        if "rotation" in sd:
            # ffprobe reports counter-clockwise-positive; we keep clockwise.
            info.rotation = int(round(-float(sd["rotation"]))) % 360
    return info


def scan_keyframes(path: str, on_start=None) -> tuple[list[float], dict[float, float]]:
    """Keyframe timestamps: walk the packets of the first video stream (reads the whole file).

    Returns the sorted keyframe pts and a pts -> dts map. -fflags +genpts fills in pts that the
    container doesn't store (AVI with B-frames), the same way the decoder labels those frames;
    files that do store pts give the same list either way.
    """
    out = _run([
        "ffprobe", "-v", "error", "-fflags", "+genpts", "-select_streams", "v:0",
        "-show_entries", "packet=pts_time,dts_time,flags",
        "-of", "csv=p=0", path,
    ], on_start)
    kfs: dict[float, float] = {}
    for line in out.splitlines():
        parts = line.strip().split(",")
        if len(parts) < 3:
            continue
        pts, dts, flags = parts[0], parts[1], parts[2]
        if "K" not in flags:
            continue
        has_pts, has_dts = pts not in ("", "N/A"), dts not in ("", "N/A")
        if not (has_pts or has_dts):
            continue
        p = float(pts) if has_pts else float(dts)
        kfs[p] = float(dts) if has_dts else p
    return sorted(kfs), kfs
