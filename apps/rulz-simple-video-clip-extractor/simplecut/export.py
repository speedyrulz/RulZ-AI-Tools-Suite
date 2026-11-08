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
    # Distortion repair (simplecut.repair.export.RepairExport), re-encode only. With it the export
    # runs through Python (make_export_job), and crop is in the corrected frame.
    repair: object | None = None


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


# Demuxers that seek by decode timestamp (no AVFMT_SEEK_TO_PTS). For them ffmpeg aims an input
# -ss 3/23 s early whenever the video has B-frames, which lands on the keyframe before In.
DTS_SEEKING_FORMATS = ("matroska", "flv", "avi")
DTS_SEEK_HEURISTIC = 3 / 23


def lossless_seek_nudge(info: VideoInfo) -> float:
    """Seconds to add to a lossless input -ss (and take off -t) so ffmpeg's own early aim
    lands on the In keyframe again. 1 ms extra keeps it from falling just short of it."""
    if info.has_b_frames and any(f in info.container.split(",") for f in DTS_SEEKING_FORMATS):
        return DTS_SEEK_HEURISTIC + 0.001
    return 0.0


def seek_keyframe(info: VideoInfo, start: float) -> float:
    """Where a re-encode seeks the input: the keyframe at or before the clip start."""
    k = info.keyframe_at(start)
    if k is None:
        k = info.prev_keyframe(start)
    if k is None:
        k = info.start_time
    return min(k, start)


def build_command(info: VideoInfo, start: float, end: float,
                  out_path: str, s: ExportSettings, hwaccel: bool = False) -> list[str]:
    """hwaccel: decode the source on the GPU (NVDEC) for re-encodes. Frames are copied back to
    system memory, so every filter below works unchanged.

    Paths are made absolute: with a colour LUT ffmpeg runs in the LUT's folder (command_cwd)."""
    src = os.path.abspath(info.path)
    out_path = os.path.abspath(out_path)
    duration = max(end - start, 0.0)
    if s.mode == "lossless":
        # ffmpeg stops stream-copying when a packet's *decode* timestamp reaches
        # the cut point; with B-frames the Out keyframe's dts is earlier than its
        # pts, so cut at the keyframe's dts to end exactly before it.
        k = info.keyframe_at(end)
        if k is not None and end < info.end_time - 1e-4:
            cut = info.keyframe_dts.get(k, k)
            if "avi" in info.container.split(","):
                # AVI stores no pts; ffmpeg's running dts there is a frame later than the
                # index says, so the keyframe's own dts still lets it through. Half a frame
                # earlier ends before it and keeps the frame before.
                cut -= info.frame_duration / 2
            duration = max(cut - start, 0.0)
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
        ss = start - fmt0
        if start <= info.start_time + 1e-4:
            # From the first frame: no seek at all. (An FLV seek to its start can land on the
            # second keyframe.) Without -ss the output clock starts at the file start, so -t
            # grows by the same amount.
            cmd += ["-i", src, "-t", f"{duration + max(ss, 0.0):.6f}"]
        else:
            nudge = lossless_seek_nudge(info)
            cmd += ["-ss", f"{ss + nudge:.6f}", "-i", src, "-t", f"{duration - nudge:.6f}"]
    else:
        # Seek the input only to the keyframe at/before the start (fast), then trim
        # precisely on the output side. Trimming on the output side also drops
        # stream-copied audio packets before the start; an input-only seek would
        # keep every audio packet from the keyframe onward and hide them with an
        # edit list, which some players (MPC-HC, Explorer) report as extra length.
        k = seek_keyframe(info, start)
        if hwaccel:
            cmd += ["-hwaccel", "cuda"]
        cmd += ["-ss", f"{max(k - fmt0, 0.0):.6f}", "-i", src,
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
        filters = video_filters(info, s)
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


def video_filters(info: VideoInfo, s: ExportSettings) -> list[str]:
    """The re-encode filter chain. Cheap steps go first so the costly ones see fewer pixels:
    crop, and dropping frames when the rate goes down, run before the colour LUT (an RGB
    round trip per pixel) and the lanczos scale. Every step works on single frames, so the
    order doesn't change the pixels (4K60 cropped to 1080x1920 at 24 fps with a LUT: 4.3 s of
    filtering became 0.9 s)."""
    # ffmpeg auto-applies the source's own rotation flag when re-encoding,
    # so only the app-chosen rotation needs a filter.
    filters = []
    if s.rotation:
        filters.append(ROTATE_FILTERS[s.rotation])
    if s.hflip:
        filters.append("hflip")
    if s.crop:
        x, y, w, h = s.crop
        filters.append(f"crop={w}:{h}:{x}:{y}")
    fps_early = False
    if s.fps:
        try:
            fps_early = float(Fraction(s.fps)) < info.fps - 1e-3
        except (ValueError, ZeroDivisionError):
            pass
        if fps_early:
            filters.append(f"fps={s.fps}")
    if s.lut_path:
        # (Untagged HD is left alone: tagging it BT.709 here with setparams made ffmpeg convert
        # back to YUV with a different matrix than the tag says, shifting greens by ~20 levels.)
        from .colormatch import ffmpeg_lut_filter
        filters.append(ffmpeg_lut_filter(s.lut_path))
    if s.out_size:
        w, h = s.out_size
        filters.append(f"scale={w}:{h}:flags=lanczos")
    if s.fps and not fps_early:
        filters.append(f"fps={s.fps}")
    return filters


def build_commands(info: VideoInfo, start: float, end: float,
                   out_path: str, s: ExportSettings) -> list[list[str]]:
    """Commands to try in order: re-encodes first with GPU decoding (2-4x faster on 4K), then,
    if that fails (no CUDA device, a format NVDEC can't take), the same without it."""
    if s.mode == "lossless":
        return [build_command(info, start, end, out_path, s)]
    return [build_command(info, start, end, out_path, s, hwaccel=True),
            build_command(info, start, end, out_path, s)]


def command_cwd(s: ExportSettings) -> str | None:
    """Folder ffmpeg has to run in: the colour LUT is passed by file name only."""
    if s.mode != "lossless" and s.lut_path:
        return os.path.dirname(os.path.abspath(s.lut_path))
    return None


def command_text(cmd: list[str]) -> str:
    return " ".join(f'"{c}"' if " " in c else c for c in cmd)


# ------------------------------------------------------------- piped export (distortion repair)
# Distortion repair is a per-pixel remap (plus an optional blur) that ffmpeg's filters can't do,
# so those exports run two ffmpeg processes with Python in between:
#   decoder  seek and trim exactly like the NVENC command, the base rotation/flip, the frame
#            rate, then raw RGB on stdout
#   Python   the corrections and any later rotation/flip on every frame (repair/export.py)
#   encoder  raw RGB on stdin plus the source's audio -> crop, colour LUT, scale -> NVENC MP4

# ffprobe colour_space tag -> swscale matrix name
SWS_MATRIX = {"bt709": "bt709", "smpte170m": "bt601", "bt470bg": "bt601", "bt2020nc": "bt2020",
              "bt2020c": "bt2020", "smpte240m": "smpte240m", "fcc": "fcc"}


def colour_matrix(info: VideoInfo) -> str:
    """The YUV<->RGB matrix for the RGB round trip: the file's own tag, else what players
    assume for untagged video (BT.709 from 720 lines up). The preview uses the same rule."""
    return SWS_MATRIX.get(info.color_space) or ("bt709" if info.height >= 720 else "bt601")


def full_range(info: VideoInfo) -> bool:
    return info.color_range == "pc" or info.pix_fmt.startswith("yuvj")


def output_rate(info: VideoInfo, s: ExportSettings) -> str:
    """Exact frame rate of the exported clip. 'Same as source' becomes the source's average
    rate here: raw frames carry no timestamps, so variable-rate video comes out constant-rate."""
    if s.fps:
        return s.fps
    if info.rate:
        return info.rate
    fr = Fraction(info.fps).limit_denominator(1001)
    return f"{fr.numerator}/{fr.denominator}"


def colour_tags(info: VideoInfo) -> list[str]:
    """Tag the output like the source. Untagged in, untagged out (as the plain NVENC export)."""
    if info.color_space not in SWS_MATRIX:
        return []
    tags = ["-colorspace", info.color_space, "-color_range", "tv"]
    if info.color_primaries:
        tags += ["-color_primaries", info.color_primaries]
    if info.color_transfer:
        tags += ["-color_trc", info.color_transfer]
    return tags


def encoder_filters(s: ExportSettings, matrix: str) -> list[str]:
    """The piped encoder's chain on the corrected RGB frames: crop, LUT, scale, back to YUV."""
    filters = []
    if s.crop:
        x, y, w, h = s.crop
        filters.append(f"crop={w}:{h}:{x}:{y}")
    if s.lut_path:
        from .colormatch import ffmpeg_lut_filter
        filters.append(ffmpeg_lut_filter(s.lut_path))
    # accurate_rnd: swscale's fast rounding darkens an RGB round trip by up to ~1.6 levels (green)
    to_yuv = f"out_color_matrix={matrix}:out_range=limited"
    if s.out_size:
        w, h = s.out_size
        filters.append(f"scale={w}:{h}:flags=lanczos+accurate_rnd+full_chroma_int:{to_yuv}")
    else:
        filters.append(f"scale=flags=accurate_rnd+full_chroma_int:{to_yuv}")
    filters.append("format=yuv420p")
    return filters


def build_piped_commands(info: VideoInfo, start: float, end: float, out_path: str, s: ExportSettings,
                         base_rotation: int, base_flip: bool, in_size: tuple[int, int],
                         out_size: tuple[int, int]) -> list[tuple[list[str], list[str]]]:
    """(decoder, encoder) command pairs to try in order: GPU decoding first, then without.

    base_rotation/base_flip: the orientation the repair adjustments were made in (the decoder
    applies it; Python applies the corrections and anything rotated or flipped since).
    in_size: the decoder's frame size; out_size: the corrected frames Python writes."""
    src = os.path.abspath(info.path)
    out_path = os.path.abspath(out_path)
    duration = max(end - start, 0.0)
    k = seek_keyframe(info, start)
    fmt0 = info.fmt_start_time
    offset = max(start - k, 0.0)
    rate = output_rate(info, s)
    matrix = colour_matrix(info)

    dec_filters = []
    if base_rotation:
        dec_filters.append(ROTATE_FILTERS[base_rotation])
    if base_flip:
        dec_filters.append("hflip")
    dec_filters.append(f"fps={rate}")
    # The explicit size makes the raw frames exactly what Python expects to read.
    dec_filters.append(f"scale={in_size[0]}:{in_size[1]}:flags=accurate_rnd+full_chroma_int"
                       f":in_color_matrix={matrix}:in_range={'full' if full_range(info) else 'limited'}"
                       f":out_range=full")
    dec_filters.append("format=rgb24")

    def decoder(hwaccel: bool) -> list[str]:
        cmd = ["ffmpeg", "-hide_banner", "-nostats", "-loglevel", "error"]
        if hwaccel:
            cmd += ["-hwaccel", "cuda"]
        cmd += ["-ss", f"{max(k - fmt0, 0.0):.6f}", "-i", src, "-ss", f"{offset:.6f}", "-t", f"{duration:.6f}",
                "-map", "0:v:0", "-vf", ",".join(dec_filters), "-fps_mode", "passthrough",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
        return cmd

    w, h = in_size
    ow, oh = out_size
    if (w, h) == (0, 0):
        raise ValueError("unknown frame size")
    audio = s.audio != "none" and info.has_audio
    enc = ["ffmpeg", "-y", "-hide_banner", "-nostats", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{ow}x{oh}", "-framerate", rate]
    if audio:
        # The source's audio is seeked to the decoder's keyframe and trimmed on the output side,
        # like build_command does (so copied audio holds only the clip). The video already starts
        # exactly at the clip start, so it is shifted to begin where that trim begins; the extra
        # half millisecond keeps its first frame from rounding to just before the trim point.
        enc += ["-itsoffset", f"{offset + 0.0005:.6f}"]
    enc += ["-i", "pipe:0"]
    if audio:
        enc += ["-ss", f"{max(k - fmt0, 0.0):.6f}", "-i", src, "-ss", f"{offset:.6f}", "-t", f"{duration:.6f}"]
    enc += ["-map", "0:v:0", "-c:v", s.codec, "-preset", s.preset,
            "-rc", "vbr", "-cq", str(s.cq), "-b:v", "0", "-pix_fmt", "yuv420p",
            "-g", str(gop_frames(info, s))]
    if s.codec == "hevc_nvenc":
        enc += ["-tag:v", "hvc1"]
    enc += ["-vf", ",".join(encoder_filters(s, matrix))]
    enc += colour_tags(info)
    if audio:
        enc += ["-map", "1:a:0?"]
        if s.audio == "copy" and info.audio_codec in COPYABLE_AUDIO_IN_MP4:
            enc += ["-c:a", "copy"]
        else:
            enc += ["-c:a", "aac", "-b:a", "192k"]
    else:
        enc += ["-an"]
    enc += ["-movflags", "+faststart", out_path]
    return [(decoder(True), enc), (decoder(False), enc)]


def make_export_job(info: VideoInfo, start: float, end: float, out_path: str, s: ExportSettings,
                    parent=None):
    """The job for one export: ffmpeg alone, or the piped job when distortion repair is on."""
    if s.mode != "lossless" and s.repair is not None:
        from .repair.export import PipedExportJob
        return PipedExportJob(info, start, end, out_path, s, parent)
    return ExportJob(build_commands(info, start, end, out_path, s), end - start, parent, cwd=command_cwd(s))


class ExportJob(QObject):
    """Runs ffmpeg asynchronously via QProcess and reports progress in [0,1].

    `cmd` is one command or a list of them: when one fails (and wasn't cancelled) the next is
    tried; `finished` reports the last attempt."""
    progress = Signal(float)
    finished = Signal(bool, str)   # ok, stderr tail

    def __init__(self, cmd: list[str] | list[list[str]], duration: float, parent=None,
                 cwd: str | None = None):
        super().__init__(parent)
        self.cmds = [cmd] if cmd and isinstance(cmd[0], str) else list(cmd)
        self.cmd = self.cmds[0]
        self.duration = max(duration, 1e-6)
        self.proc = QProcess(self)
        if cwd:
            self.proc.setWorkingDirectory(cwd)  # see command_cwd()
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
                        whole: bool = False, repaired: bool = False) -> str:
    """Default clip path: <out_dir or source folder>/<source name>_clip_<in>-<out>.<ext>
    (or _clip_full for the whole video), plus _repair / _crop when those are on."""
    base, ext = os.path.splitext(info.path)
    if out_dir:
        base = os.path.join(out_dir, os.path.basename(base))
    tag = "full" if whole else f"{start:.2f}-{end:.2f}".replace(".", "_")
    if repaired:
        tag += "_repair"
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
