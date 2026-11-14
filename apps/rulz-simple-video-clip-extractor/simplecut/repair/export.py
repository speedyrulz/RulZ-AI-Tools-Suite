"""Export with distortion repair: ffmpeg decodes, Python corrects every frame, ffmpeg encodes.

See export.build_piped_commands for the two command lines. PipedExportJob has the interface of
export.ExportJob (progress, finished, start, cancel, wait, cmd_text, cancelled, done), so the
export dialog and Quick export drive either one the same way.
"""
from __future__ import annotations

import subprocess
import tempfile
import threading
import traceback
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from PySide6.QtCore import QObject, Qt, Signal

from ..export import ExportSettings, build_piped_commands, command_cwd, command_text, output_rate
from ..probe import VideoInfo
from .pipeline import RepairState, fit_size, orient, oriented_size

NO_WINDOW = 0x08000000  # CREATE_NO_WINDOW: no console window per ffmpeg on Windows


@dataclass
class RepairExport:
    """The repair part of one export: saved adjustments plus the orientation now on screen."""
    state: RepairState
    rotation: int          # the app's current rotation/flip; anything since the base is applied after
    flip: bool
    source_rotation: int
    coded_size: tuple[int, int]

    def __post_init__(self):
        self._mapper = None
        self._delta = self.state.delta(self.rotation, self.flip)

    @property
    def base_rotation(self) -> int:
        return self.state.settings.base_rotation

    @property
    def base_flip(self) -> bool:
        return self.state.settings.base_flip

    @property
    def in_size(self) -> tuple[int, int]:
        """Frames the decoder writes: the source flag (ffmpeg's autorotate) plus the base rotation."""
        return oriented_size(*self.coded_size, (self.source_rotation + self.base_rotation) % 360)

    @property
    def out_size(self) -> tuple[int, int]:
        return self.state.output_size(self.rotation, self.flip)

    def process(self, rgb: np.ndarray) -> np.ndarray:
        """One decoded frame (in_size, base orientation) -> the corrected frame (out_size)."""
        p = self.state.pipeline
        if self._mapper is None:  # held for the whole export, whatever the preview evicts
            self._mapper = p.mapper(rgb.shape[1], rgb.shape[0], True)
        out = fit_size(self._mapper.apply(rgb), p.out_size)
        return np.ascontiguousarray(orient(out, *self._delta))


class PipedExportJob(QObject):
    progress = Signal(float)
    finished = Signal(bool, str)   # ok, log tail
    _progress_q = Signal(float)    # worker thread -> this object's thread
    _ended_q = Signal(bool, str)

    def __init__(self, info: VideoInfo, start: float, end: float, out_path: str, s: ExportSettings,
                 parent=None):
        super().__init__(parent)
        rx: RepairExport = s.repair
        self.repair = rx
        self.attempts = build_piped_commands(info, start, end, out_path, s, rx.base_rotation, rx.base_flip,
                                             rx.in_size, rx.out_size)
        self.cmd = self.attempts[0]
        self.cwd = command_cwd(s)
        try:
            rate = float(Fraction(output_rate(info, s)))
        except (ValueError, ZeroDivisionError):
            rate = info.fps
        self.expected = max(1, round((end - start) * rate))
        self.cancelled = False
        self.done = False
        self.warning = ""  # (ExportJob interface; nothing to report here)
        self.frames = 0
        self._cancel = threading.Event()
        self._procs: list[subprocess.Popen] = []   # the running attempt's decoder and encoder
        self._procs_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._result: tuple[bool, str] | None = None
        self._progress_q.connect(self._on_progress, Qt.QueuedConnection)
        self._ended_q.connect(self._end, Qt.QueuedConnection)

    @property
    def cmd_text(self) -> str:
        dec, enc = self.cmd
        return f"Decoder:\n{command_text(dec)}\n\nEncoder:\n{command_text(enc)}"

    def start(self):
        self._thread = threading.Thread(target=self._run, name="repair-export", daemon=True)
        self._thread.start()

    def cancel(self):
        """Stop now. The worker may be blocked reading a frame (a decoder that is still seeking
        through a long file sends nothing for minutes) or writing one, where it can't see the
        flag; killing both processes ends those reads and writes at once."""
        self.cancelled = True
        self._cancel.set()
        with self._procs_lock:
            procs = list(self._procs)
        for p in procs:
            if p.poll() is None:
                try:
                    p.kill()
                except OSError:
                    pass

    def wait(self, msecs: int = 30000) -> bool:
        """Block until the worker has stopped; `finished` is emitted before this returns."""
        t = self._thread
        if t is not None:
            t.join(msecs / 1000)
            if t.is_alive():
                return False
        if self._result is not None and not self.done:
            self._end(*self._result)
        return True

    # ---------------------------------------------------------------- worker thread
    def _run(self):
        notes = []
        ok, log = False, ""
        try:
            for i, (dec, enc) in enumerate(self.attempts):
                self.cmd = (dec, enc)
                ok, frames, decoder_failed, log = self._attempt(dec, enc)
                if ok or self._cancel.is_set():
                    break
                # Retry without GPU decoding only when the decoder gave up before the first frame
                # (no CUDA device, a format NVDEC can't take); other failures would just repeat.
                if not (decoder_failed and frames == 0) or i + 1 == len(self.attempts):
                    break
                notes = ["(the GPU-decoding attempt failed; this log is from the retry that decodes on the CPU)"]
                self._progress_q.emit(0.0)
        except Exception:  # noqa: BLE001 - reported in the export's error box
            ok, log = False, traceback.format_exc()
        self._result = (ok and not self._cancel.is_set(), "\n".join([log] + notes).strip())
        self._ended_q.emit(*self._result)

    def _attempt(self, dec_cmd: list[str], enc_cmd: list[str]) -> tuple[bool, int, bool, str]:
        """One decode -> correct -> encode run. Returns (ok, frames, decoder_failed, log)."""
        w, h = self.repair.in_size
        nbytes = w * h * 3
        dec_err, enc_err = tempfile.TemporaryFile(), tempfile.TemporaryFile()
        dec = enc = None
        frames, error = 0, ""
        last_report = -1.0
        try:
            dec = subprocess.Popen(dec_cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=dec_err,
                                   creationflags=NO_WINDOW)
            with self._procs_lock:
                self._procs = [dec]
            enc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=enc_err,
                                   cwd=self.cwd, creationflags=NO_WINDOW)
            with self._procs_lock:
                self._procs = [dec, enc]
            # (a cancel() that came before the list was set is seen by the loop condition)
            prev_in = prev_out = None
            while not self._cancel.is_set():
                buf = dec.stdout.read(nbytes)
                if not buf or self._cancel.is_set():
                    break
                if len(buf) < nbytes:
                    error = f"the decoder stopped in the middle of a frame ({len(buf)} of {nbytes} bytes)"
                    break
                if buf == prev_in:      # a frame the rate change duplicated: no second correction
                    out = prev_out
                else:
                    out = self.repair.process(np.frombuffer(buf, np.uint8).reshape(h, w, 3))
                    prev_in, prev_out = buf, out
                try:
                    enc.stdin.write(out.data)
                except (OSError, ValueError):  # (ValueError: a pipe closed by a cancel)
                    if not self._cancel.is_set():
                        error = "the encoder stopped accepting frames"
                    break
                frames += 1
                self.frames = frames
                f = min(frames / self.expected, 1.0)
                if f - last_report >= 0.002:
                    last_report = f
                    self._progress_q.emit(f)
        except Exception:  # noqa: BLE001
            error = traceback.format_exc()
        finally:
            stop = bool(error) or self._cancel.is_set()
            for p in (dec, enc):
                if p is not None and stop and p.poll() is None:
                    p.kill()
            if enc is not None:
                try:
                    enc.stdin.close()
                except OSError:
                    pass
            # (still listed while they finish: a cancel during the encoder's final flush kills it)
            dec_code = dec.wait() if dec is not None else -1
            enc_code = enc.wait() if enc is not None else -1
            with self._procs_lock:
                self._procs = []
            if dec is not None:
                dec.stdout.close()
            logs = []
            for name, f in (("decoder", dec_err), ("encoder", enc_err)):
                f.seek(0)
                text = f.read().decode("utf-8", "replace").strip()
                f.close()
                if text:
                    logs.append(f"[{name}]\n" + "\n".join(text.splitlines()[-40:]))
        if error:
            logs.append(error)
        if not error and not self._cancel.is_set() and frames == 0 and dec_code == 0:
            logs.append("the decoder produced no frames for this range")
        ok = not error and not self._cancel.is_set() and frames > 0 and dec_code == 0 and enc_code == 0
        decoder_failed = dec_code != 0 and not self._cancel.is_set()
        return ok, frames, decoder_failed, "\n\n".join(logs)

    # ---------------------------------------------------------------- main thread
    def _on_progress(self, f: float):
        if not self.done:
            self.progress.emit(f)

    def _end(self, ok: bool, log: str):
        if self.done:
            return
        self.done = True
        if ok:
            self.progress.emit(1.0)
        self.finished.emit(ok, log)
