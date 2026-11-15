"""Export with distortion repair: ffmpeg decodes, Python corrects every frame, ffmpeg encodes.

See export.build_piped_commands for the two command lines. PipedExportJob has the interface of
export.ExportJob (progress, finished, start, cancel, wait, cmd_text, cancelled, done), so the
export dialog and Quick export drive either one the same way.
"""
from __future__ import annotations

import os
import queue
import subprocess
import tempfile
import threading
import traceback
from dataclasses import dataclass
from fractions import Fraction

import cv2
import numpy as np
from PySide6.QtCore import QObject, Qt, Signal

from ..export import ExportSettings, build_piped_commands, command_cwd, command_text, output_rate
from ..probe import VideoInfo
from .pipeline import RepairState, fit_size, orient, oriented_size

NO_WINDOW = 0x08000000  # CREATE_NO_WINDOW: no console window per ffmpeg on Windows
# Windows gives an anonymous pipe a 4 KB buffer, so a 4K frame (25 MB) took thousands of small
# reads and writes and the pipes alone held exports to ~20 frames a second; 1 MB measured best.
PIPE_BYTES = 1 << 20
QUEUE_FRAMES = 2   # frames waiting between the stages (each 25 MB at 4K)


def _pipe(size: int = PIPE_BYTES) -> tuple[int, int]:
    """(read fd, write fd) of an anonymous pipe with a big buffer."""
    if os.name == "nt":
        import _winapi
        import msvcrt
        r, w = _winapi.CreatePipe(None, size)
        try:
            rfd = msvcrt.open_osfhandle(r, os.O_RDONLY)
        except BaseException:
            _winapi.CloseHandle(r)
            _winapi.CloseHandle(w)
            raise
        try:
            return rfd, msvcrt.open_osfhandle(w, 0)
        except BaseException:
            os.close(rfd)
            _winapi.CloseHandle(w)
            raise
    return os.pipe()


def _fdopen(fd: int, mode: str):
    try:
        return open(fd, mode, buffering=0)
    except BaseException:
        os.close(fd)
        raise


def _read_into(f, buf: bytearray) -> int:
    """Fill buf from an unbuffered pipe; fewer bytes only at the end of the stream."""
    mv = memoryview(buf)
    got = 0
    while got < len(mv):
        n = f.readinto(mv[got:])
        if not n:
            break
        got += n
    return got


def _write_all(f, px: np.ndarray):
    mv = memoryview(px).cast("B")
    sent = 0
    while sent < len(mv):
        n = f.write(mv[sent:])
        if n is None:
            raise OSError("the pipe would block")
        sent += n


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
        self._yuv = None
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

    def yuv_ok(self) -> bool:
        """Whether frames can be corrected as yuv420p planes: not with the miniature effect (it
        blurs in linear RGB) or a later rotate/flip, and only with even sizes and a crop that
        keeps at least 2 pixels each way (smaller ones are stretched, see fit_size)."""
        p = self.state.pipeline
        w, h = self.in_size
        ow, oh = self.out_size
        fw, fh = p.frame_size
        return (not p.c.tiltshift.enabled and self._delta == (0, False)
                and not (w % 2 or h % 2 or ow % 2 or oh % 2)
                and round(fw * p.crop[2]) >= ow and round(fh * p.crop[3]) >= oh)

    def process_yuv(self, buf) -> np.ndarray:
        """One decoded yuv420p frame (in_size, base orientation) -> the corrected yuv420p frame
        (out_size) as one flat array. Luma is remapped at full size and each chroma plane at half
        size with its own tables, so nothing is converted to RGB and back."""
        w, h = self.in_size
        ow, oh = self.out_size
        src = np.frombuffer(buf, np.uint8)
        if self._yuv is None:  # (built on the export's worker thread, once)
            m = self.state.pipeline.mapper(w, h, True)
            if m.plain:   # nothing moves (the sizes are equal, see yuv_ok)
                self._yuv = ()
            else:
                cx, cy = m.chroma_tables((ow // 2, oh // 2))
                self._yuv = (np.ascontiguousarray(m.mx[:oh, :ow]), np.ascontiguousarray(m.my[:oh, :ow]), cx, cy)
        if not self._yuv:
            return src
        mx, my, cx, cy = self._yuv
        out = np.empty(ow * oh * 3 // 2, np.uint8)
        n, nc, onc = w * h, (w // 2) * (h // 2), (ow // 2) * (oh // 2)
        planes = ((src[:n].reshape(h, w), out[:ow * oh].reshape(oh, ow), mx, my, 16),
                  (src[n:n + nc].reshape(h // 2, w // 2), out[ow * oh:ow * oh + onc].reshape(oh // 2, ow // 2), cx, cy, 128),
                  (src[n + nc:].reshape(h // 2, w // 2), out[ow * oh + onc:].reshape(oh // 2, ow // 2), cx, cy, 128))
        for plane, dst, tx, ty, black in planes:
            cv2.remap(plane, tx, ty, cv2.INTER_CUBIC, dst=dst, borderMode=cv2.BORDER_CONSTANT, borderValue=black)
        return out


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
        # 4:2:0 planes straight through when the corrections allow it and the encoder's crop is
        # on whole chroma samples (its crop filter would round an odd offset on yuv420p)
        self.yuv = rx.yuv_ok() and not (s.crop and any(v % 2 for v in s.crop))
        self.attempts = build_piped_commands(info, start, end, out_path, s, rx.base_rotation, rx.base_flip,
                                             rx.in_size, rx.out_size, self.yuv)
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
        """One decode -> correct -> encode run. Returns (ok, frames, decoder_failed, log).

        Three stages overlap: a reader thread takes frames from the decoder, this thread corrects
        them (cv2 releases the GIL), and a writer thread feeds the encoder, so a frame costs the
        slowest stage instead of all three added up."""
        w, h = self.repair.in_size
        nbytes = w * h * 3 // 2 if self.yuv else w * h * 3
        dec_err, enc_err = tempfile.TemporaryFile(), tempfile.TemporaryFile()
        dec = enc = None
        dec_out = enc_in = None
        errors: list[str] = []          # the first one is reported
        halt = threading.Event()        # this attempt is over (an error, or the end of the frames)
        inq: queue.Queue = queue.Queue(QUEUE_FRAMES)
        outq: queue.Queue = queue.Queue(QUEUE_FRAMES)
        threads: list[threading.Thread] = []
        last_report = [-1.0]
        dec_killed = [False]   # stopped by us (another stage failed), so its exit code says nothing

        def stopping() -> bool:
            return halt.is_set() or self._cancel.is_set()

        def put(q: queue.Queue, item) -> bool:
            while not stopping():
                try:
                    q.put(item, timeout=0.1)
                    return True
                except queue.Full:
                    pass
            return False

        def get(q: queue.Queue):
            while True:
                try:
                    return q.get(timeout=0.1)
                except queue.Empty:
                    if stopping():
                        return None

        def fail(msg: str):
            errors.append(msg)
            halt.set()

        def read_frames():
            try:
                while not stopping():
                    buf = bytearray(nbytes)
                    got = _read_into(dec_out, buf)
                    if got == 0:
                        break
                    if got < nbytes:
                        fail(f"the decoder stopped in the middle of a frame ({got} of {nbytes} bytes)")
                        break
                    if not put(inq, buf):
                        break
            except Exception:  # noqa: BLE001
                if not self._cancel.is_set():
                    fail(traceback.format_exc())
            finally:
                put(inq, None)

        def write_frames():
            try:
                while True:
                    out = get(outq)
                    if out is None:
                        break
                    try:
                        _write_all(enc_in, out)
                    except (OSError, ValueError):  # (ValueError: a pipe closed by a cancel)
                        if not self._cancel.is_set():
                            fail("the encoder stopped accepting frames")
                        break
                    self.frames += 1
                    f = min(self.frames / self.expected, 1.0)
                    if f - last_report[0] >= 0.002:
                        last_report[0] = f
                        self._progress_q.emit(f)
            except Exception:  # noqa: BLE001
                fail(traceback.format_exc())

        self.frames = 0
        try:
            r, wr = _pipe()
            try:
                dec = subprocess.Popen(dec_cmd, stdin=subprocess.DEVNULL, stdout=wr, stderr=dec_err,
                                       creationflags=NO_WINDOW)
            except BaseException:
                os.close(r)
                raise
            finally:
                os.close(wr)  # the decoder holds its own copy; ours would keep the pipe from ending
            dec_out = _fdopen(r, "rb")
            with self._procs_lock:
                self._procs = [dec]
            r, wr = _pipe()
            try:
                enc = subprocess.Popen(enc_cmd, stdin=r, stdout=subprocess.DEVNULL, stderr=enc_err,
                                       cwd=self.cwd, creationflags=NO_WINDOW)
            except BaseException:
                os.close(wr)
                raise
            finally:
                os.close(r)
            enc_in = _fdopen(wr, "wb")
            with self._procs_lock:
                self._procs = [dec, enc]
            # (a cancel() that came before the list was set is seen by stopping())
            threads = [threading.Thread(target=read_frames, name="repair-read", daemon=True),
                       threading.Thread(target=write_frames, name="repair-write", daemon=True)]
            for t in threads:
                t.start()
            prev_in = prev_out = None
            while True:
                buf = get(inq)
                if buf is None or stopping():
                    break
                if buf == prev_in:      # a frame the rate change duplicated: no second correction
                    out = prev_out
                else:
                    out = (self.repair.process_yuv(buf) if self.yuv
                           else self.repair.process(np.frombuffer(buf, np.uint8).reshape(h, w, 3)))
                    prev_in, prev_out = buf, out
                if not put(outq, out):
                    break
            put(outq, None)
        except Exception:  # noqa: BLE001
            fail(traceback.format_exc())
        finally:
            def kill_both():
                for p in (dec, enc):
                    if p is not None and p.poll() is None:
                        p.kill()
                        if p is dec:
                            dec_killed[0] = True

            if errors or self._cancel.is_set():
                halt.set()
                kill_both()    # ends a read or write the helper threads are blocked in
            for t in threads:
                t.join()
            if errors or self._cancel.is_set():
                kill_both()
            if enc_in is not None:
                try:
                    enc_in.close()
                except OSError:
                    pass
            # (still listed while they finish: a cancel during the encoder's final flush kills it)
            dec_code = dec.wait() if dec is not None else -1
            enc_code = enc.wait() if enc is not None else -1
            with self._procs_lock:
                self._procs = []
            if dec_out is not None:
                dec_out.close()
            logs = []
            for name, f in (("decoder", dec_err), ("encoder", enc_err)):
                f.seek(0)
                text = f.read().decode("utf-8", "replace").strip()
                f.close()
                if text:
                    logs.append(f"[{name}]\n" + "\n".join(text.splitlines()[-40:]))
        frames = self.frames
        error = errors[0] if errors else ""
        if error:
            logs.append(error)
        if not error and not self._cancel.is_set() and frames == 0 and dec_code == 0:
            logs.append("the decoder produced no frames for this range")
        ok = not error and not self._cancel.is_set() and frames > 0 and dec_code == 0 and enc_code == 0
        # (only the decoder's own failure is worth a retry without GPU decoding: when the encoder
        # gave up first, killing the decoder must not look like one and hide the encoder's error)
        decoder_failed = dec_code != 0 and not self._cancel.is_set() and not dec_killed[0]
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
