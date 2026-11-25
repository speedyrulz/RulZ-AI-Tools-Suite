"""Live list of NVIDIA GPUs, read from the Windows ``nvidia-smi``.

The Windows tool is used (not the one inside WSL) so that looking at GPUs never
starts WSL. Cards are identified by UUID, which is the same on both sides; the
launch script turns UUIDs into WSL's own device numbers at start time.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass

from PySide6.QtCore import QObject, QProcess, QTimer, Signal

_QUERY = "index,uuid,name,pci.bus_id,memory.used,memory.total,utilization.gpu,temperature.gpu"


@dataclass(frozen=True)
class GpuInfo:
    index: int            # nvidia-smi index (PCI bus order)
    uuid: str
    name: str
    bus_id: str = ""
    memory_used: int = 0  # MiB
    memory_total: int = 0  # MiB
    utilization: int = 0  # percent
    temperature: int = 0  # degrees C

    @property
    def short_name(self) -> str:
        name = self.name
        for prefix in ("NVIDIA GeForce ", "NVIDIA "):
            if name.startswith(prefix):
                return name[len(prefix):]
        return name

    @property
    def memory_text(self) -> str:
        return f"{self.memory_used / 1024:.1f} / {self.memory_total / 1024:.1f} GB"


def _to_int(text: str) -> int:
    try:
        return int(float(text))
    except ValueError:
        return 0   # "[N/A]" and similar


def parse_gpu_line(line: str) -> GpuInfo | None:
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 8 or not parts[1].startswith("GPU-"):
        return None
    try:
        index = int(parts[0])
    except ValueError:
        return None
    return GpuInfo(index=index, uuid=parts[1], name=parts[2], bus_id=parts[3],
                   memory_used=_to_int(parts[4]), memory_total=_to_int(parts[5]),
                   utilization=_to_int(parts[6]), temperature=_to_int(parts[7]))


def find_nvidia_smi() -> str:
    found = shutil.which("nvidia-smi")
    if found:
        return found
    candidate = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "nvidia-smi.exe")
    return candidate if os.path.exists(candidate) else ""


class GpuMonitor(QObject):
    """Keeps one ``nvidia-smi --loop`` process running and reports each update."""
    updated = Signal()

    def __init__(self, job=None, interval_seconds: int = 2, parent: QObject | None = None):
        super().__init__(parent)
        self._job = job
        self._interval = interval_seconds
        self._exe = find_nvidia_smi()
        self._gpus: dict[str, GpuInfo] = {}
        self._buffer = b""
        self._proc: QProcess | None = None
        self._wanted = False
        self._restart = QTimer(self)
        self._restart.setSingleShot(True)
        self._restart.timeout.connect(self._start_process)

    @property
    def available(self) -> bool:
        return bool(self._exe)

    def gpus(self) -> list[GpuInfo]:
        return sorted(self._gpus.values(), key=lambda g: g.index)

    def index_by_uuid(self) -> dict[str, int]:
        return {g.uuid: g.index for g in self._gpus.values()}

    def start(self):
        self._wanted = True
        if self._proc is None:
            self._start_process()

    def stop(self):
        self._wanted = False
        self._restart.stop()
        if self._proc is not None:
            proc, self._proc = self._proc, None
            proc.readyReadStandardOutput.disconnect(self._on_output)
            proc.finished.disconnect(self._on_finished)
            proc.kill()
            proc.waitForFinished(1000)
            proc.deleteLater()

    def _start_process(self):
        if not self._wanted or not self._exe or self._proc is not None:
            return
        proc = QProcess(self)
        proc.readyReadStandardOutput.connect(self._on_output)
        proc.finished.connect(self._on_finished)
        proc.errorOccurred.connect(self._on_error)
        self._proc = proc
        self._buffer = b""
        proc.start(self._exe, [f"--query-gpu={_QUERY}", "--format=csv,noheader,nounits",
                               f"--loop={self._interval}"])
        if self._job is not None and proc.waitForStarted(3000):
            self._job.assign(proc.processId())

    def _on_output(self):
        if self._proc is None:
            return
        self._buffer += self._proc.readAllStandardOutput().data()
        *lines, self._buffer = self._buffer.split(b"\n")
        changed = False
        for raw in lines:
            info = parse_gpu_line(raw.decode("utf-8", errors="replace"))
            if info is not None and self._gpus.get(info.uuid) != info:
                self._gpus[info.uuid] = info
                changed = True
        if changed:
            self.updated.emit()

    def _on_error(self, error):
        if error == QProcess.ProcessError.FailedToStart:
            self._on_finished()

    def _on_finished(self, *_):
        if self._proc is not None:
            self._proc.deleteLater()
            self._proc = None
        if self._wanted:
            self._restart.start(5000)
