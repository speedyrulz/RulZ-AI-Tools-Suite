"""Build and run ffmpeg export commands (lossless stream copy or NVENC re-encode)."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from fractions import Fraction

from PySide6.QtCore import QObject, QProcess, Signal

from .probe import VideoInfo

NVENC_CODECS = {
    "H.264 (h264_nvenc)": "h264_nvenc",
    "HEVC / H.265 (hevc_nvenc)": "hevc_nvenc",
    "AV1 (av1_nvenc)": "av1_nvenc",
}

ROTATE_FILTERS = {90: "transpose=1", 180: "transpose=1,transpose=1", 270: "transpose=2"}

# Label -> exact ffmpeg rate. NTSC rates are stored as fractions, not rounded decimals.
FPS_PRESETS = {
    "Same as source": None,
    "23.976 (24000/1001)": "24000/1001",
    "24": "24",
    "25": "25",
    "29.97 (30000/1001)": "30000/1001",
    "30": "30",
    "50": "50",
    "59.94 (60000/1001)": "60000/1001",
    "60": "60",
    "120": "120",
}


def parse_fps(text: str) -> str | None:
    """Turn a preset label or a typed value like '48', '23.976' or '24000/1001' into an ffmpeg rate."""
    text = text.strip()
    if not text or text in FPS_PRESETS:
        return FPS_PRESETS.get(text)
    token = text.split()[0]
    if "/" in token:
        num, den = token.split("/", 1)
        if num.strip().isdigit() and den.strip().isdigit() and int(den) > 0:
            return f"{int(num)}/{int(den)}"
        raise ValueError(f"Bad frame rate: {text}")
    try:
        v = float(token)
    except ValueError as e:
        raise ValueError(f"Bad frame rate: {text}") from e
    if v <= 0:
        raise ValueError(f"Bad frame rate: {text}")
    return f"{v:g}"


COPYABLE_AUDIO_IN_MP4 = ("aac", "mp3", "ac3", "eac3", "opus", "alac")


@dataclass
class ExportSettings:
    """Export options. The field defaults are THE default export settings: the
    export dialog starts from them and Quick export uses them as-is."""
    mode: str = "lossless"          # "lossless" | "nvenc"
    codec: str = "h264_nvenc"
    cq: int = 18                    # constant-quality value for NVENC (lower = better)
    preset: str = "p5"
    audio: str = "copy"             # "copy" | "aac" | "none"
    rotation: int = 0               # extra clockwise rotation chosen in the app (0/90/180/270)
    hflip: bool = False             # mirror horizontally (applied after rotation)
    fps: str | None = "24"          # output frame rate for re-encode, e.g. "30" or "24000/1001"; None = source
    crop: tuple[int, int, int, int] | None = None   # (x, y, w, h) in the rotated frame
    out_size: tuple[int, int] | None = None         # scale the (cropped) video to (w, h)
    lut_path: str | None = None                     # 1D .cube colour correction (skin-tone match)


def gop_frames(info: VideoInfo, s: ExportSettings) -> int:
    """One keyframe per second of output, so exported clips step and cut cleanly later.

    NVENC's default is one keyframe every 250 frames, which makes frame stepping
    slow and leaves almost no lossless cut points.
    """
    fps = info.fps
    if s.fps:
        try:
            fps = float(Fraction(s.fps))
        except (ValueError, ZeroDivisionError):
            pass
    return max(1, round(fps))


def build_command(info: VideoInfo, start: float, end: float,
                  out_path: str, s: ExportSettings, hwaccel: bool = False) -> list[str]:
    """hwaccel: decode the source on the GPU (NVDEC) for re-encodes. Frames are copied back to
    system memory, so every filter below works unchanged."""
    duration = max(end - start, 0.0)
    if s.mode == "lossless":
        # ffmpeg stops stream-copying when a packet's *decode* timestamp reaches
        # the cut point; with B-frames the Out keyframe's dts is earlier than its
        # pts, so cut at the keyframe's dts to end exactly before it.
        k = info.keyframe_at(end)
        if k is not None and end < info.end_time - 1e-4:
            duration = max(info.keyframe_dts.get(k, k) - start, 0.0)
    cmd = ["ffmpeg", "-y", "-hide_banner", "-nostats", "-progress", "pipe:1"]
    total_rot = (info.rotation + s.rotation) % 360
    if s.mode == "lossless" and s.rotation:
        # Pixels are copied untouched; store the rotation in the container's
        # display matrix (negative = clockwise), replacing the source's flag.
        cmd += ["-display_rotation", str(-total_rot)]
    # ffmpeg's input -ss counts from the container's start time, while our
    # marks are absolute pts, so subtract it.
    fmt0 = info.fmt_start_time
    if s.mode == "lossless":
        # start is on a keyframe (or the file start): the input seek lands exactly there
        cmd += ["-ss", f"{start - fmt0:.6f}", "-i", info.path, "-t", f"{duration:.6f}"]
    else:
        # Seek the input only to the keyframe at/before the start (fast), then trim
        # precisely on the output side. Trimming on the output side also drops
        # stream-copied audio packets before the start; an input-only seek would
        # keep every audio packet from the keyframe onward and hide them with an
        # edit list, which some players (MPC-HC, Explorer) report as extra length.
        k = info.keyframe_at(start)
        if k is None:
            k = info.prev_keyframe(start)
        if k is None:
            k = info.start_time
        k = min(k, start)
        if hwaccel:
            cmd += ["-hwaccel", "cuda"]
        cmd += ["-ss", f"{max(k - fmt0, 0.0):.6f}", "-i", info.path,
                "-ss", f"{max(start - k, 0.0):.6f}", "-t", f"{duration:.6f}"]

    if s.mode == "lossless":
        cmd += ["-map", "0", "-c", "copy", "-avoid_negative_ts", "make_zero"]
        if out_path.lower().endswith(".mp4"):
            cmd += ["-movflags", "+faststart"]
    else:
        cmd += ["-map", "0:v:0", "-c:v", s.codec, "-preset", s.preset,
                "-rc", "vbr", "-cq", str(s.cq), "-b:v", "0", "-pix_fmt", "yuv420p",
                "-g", str(gop_frames(info, s))]
        if s.codec == "hevc_nvenc":
            cmd += ["-tag:v", "hvc1"]
        # ffmpeg auto-applies the source's own rotation flag when re-encoding,
        # so only the app-chosen rotation needs a filter.
        filters = []
        if s.rotation:
            filters.append(ROTATE_FILTERS[s.rotation])
        if s.hflip:
            filters.append("hflip")
        if s.lut_path:
            from .colormatch import ffmpeg_lut_filter
            filters.append(ffmpeg_lut_filter(s.lut_path))
        if s.crop:
            x, y, w, h = s.crop
            filters.append(f"crop={w}:{h}:{x}:{y}")
        if s.out_size:
            w, h = s.out_size
            filters.append(f"scale={w}:{h}:flags=lanczos")
        if s.fps:
            filters.append(f"fps={s.fps}")
        if filters:
            cmd += ["-vf", ",".join(filters)]
        if s.audio == "none" or not info.has_audio:
            cmd += ["-an"]
        else:
            cmd += ["-map", "0:a:0?"]
            if s.audio == "copy" and info.audio_codec in COPYABLE_AUDIO_IN_MP4:
                cmd += ["-c:a", "copy"]
            else:
                cmd += ["-c:a", "aac", "-b:a", "192k"]
        cmd += ["-movflags", "+faststart"]
    cmd.append(out_path)
    return cmd


def build_commands(info: VideoInfo, start: float, end: float,
                   out_path: str, s: ExportSettings) -> list[list[str]]:
    """Commands to try in order: re-encodes first with GPU decoding (2-4x faster on 4K), then,
    if that fails (no CUDA device, a format NVDEC can't take), the same without it."""
    if s.mode == "lossless":
        return [build_command(info, start, end, out_path, s)]
    return [build_command(info, start, end, out_path, s, hwaccel=True),
            build_command(info, start, end, out_path, s)]


def command_text(cmd: list[str]) -> str:
    return " ".join(f'"{c}"' if " " in c else c for c in cmd)


class ExportJob(QObject):
    """Runs ffmpeg asynchronously via QProcess and reports progress in [0,1].

    `cmd` is one command or a list of them: when one fails (and wasn't cancelled) the next is
    tried; `finished` reports the last attempt."""
    progress = Signal(float)
    finished = Signal(bool, str)   # ok, stderr tail

    def __init__(self, cmd: list[str] | list[list[str]], duration: float, parent=None):
        super().__init__(parent)
        self.cmds = [cmd] if cmd and isinstance(cmd[0], str) else list(cmd)
        self.cmd = self.cmds[0]
        self.duration = max(duration, 1e-6)
        self.proc = QProcess(self)
        self.proc.setProcessChannelMode(QProcess.SeparateChannels)
        self.proc.readyReadStandardOutput.connect(self._on_stdout)
        self.proc.readyReadStandardError.connect(self._on_stderr)
        self.proc.finished.connect(self._on_finished)
        self.proc.errorOccurred.connect(self._on_error)
        self._err_tail: list[str] = []
        self._attempt = 0
        self.cancelled = False
        self.done = False

    @property
    def cmd_text(self) -> str:
        """The command line of the current (or last) attempt."""
        return command_text(self.cmd)

    def start(self):
        self.cmd = self.cmds[self._attempt]
        self._err_tail = []
        self.proc.start(self.cmd[0], self.cmd[1:])

    def cancel(self):
        self.cancelled = True
        if self.proc.state() != QProcess.NotRunning:
            self.proc.kill()

    def wait(self, msecs: int = 5000) -> bool:
        """Block until the process has exited (its `finished` is emitted before this returns)."""
        if self.proc.state() == QProcess.NotRunning:
            return True
        return self.proc.waitForFinished(msecs)

    def _on_error(self, err):
        # A process that never started emits no `finished`; report it here or the caller waits forever.
        if err == QProcess.FailedToStart:
            self._err_tail.append(f"Could not start {self.cmd[0]}: {self.proc.errorString()}")
            self._end(False)

    def _on_stdout(self):
        data = bytes(self.proc.readAllStandardOutput()).decode("utf-8", "replace")
        for m in re.finditer(r"out_time_us=(\d+)", data):
            self.progress.emit(min(int(m.group(1)) / 1e6 / self.duration, 1.0))
        if "progress=end" in data:
            self.progress.emit(1.0)

    def _on_stderr(self):
        data = bytes(self.proc.readAllStandardError()).decode("utf-8", "replace")
        self._err_tail.extend(data.splitlines())
        self._err_tail = self._err_tail[-40:]

    def _on_finished(self, code: int, status):
        self._end(code == 0 and status == QProcess.NormalExit)

    def _end(self, ok: bool):
        if self.done:
            return
        if not ok and not self.cancelled and self._attempt + 1 < len(self.cmds):
            self._attempt += 1
            self.progress.emit(0.0)
            self.start()  # e.g. GPU decoding unavailable: same export, decoded on the CPU
            self._err_tail.append("(the GPU-decoding attempt failed; this log is from the retry that decodes on the CPU)")
            return
        self.done = True
        self.finished.emit(ok, "\n".join(self._err_tail))


def default_output_path(info: VideoInfo, start: float, end: float, mode: str,
                        out_dir: str | None = None, cropped: bool = False,
                        whole: bool = False) -> str:
    """Default clip path: <out_dir or source folder>/<source name>_clip_<in>-<out>.<ext>
    (or _clip_full for the whole video)."""
    base, ext = os.path.splitext(info.path)
    if out_dir:
        base = os.path.join(out_dir, os.path.basename(base))
    tag = "full" if whole else f"{start:.2f}-{end:.2f}".replace(".", "_")
    if cropped:
        tag += "_crop"
    if mode == "lossless":
        # keep the source container so any codec can be copied as-is
        return f"{base}_clip_{tag}{ext or '.mkv'}"
    return f"{base}_clip_{tag}.mp4"


def unique_path(path: str) -> str:
    """Append _1, _2, ... if the file already exists."""
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    n = 1
    while os.path.exists(f"{base}_{n}{ext}"):
        n += 1
    return f"{base}_{n}{ext}"
