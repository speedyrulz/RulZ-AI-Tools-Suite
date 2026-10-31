"""One-time setup for Best quality mode: installs PyTorch and downloads the FBCNN model."""
from __future__ import annotations

import importlib
import sys
import urllib.request
from pathlib import Path

from PySide6.QtCore import QProcess, QThread, Signal
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QLabel, QPlainTextEdit, QProgressBar, QVBoxLayout,
)

from .. import ai_jpeg, config


class _Download(QThread):
    progress = Signal(int, int)  # bytes done, total
    failed = Signal(str)
    done = Signal()

    def __init__(self):
        super().__init__()
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        dest = ai_jpeg.model_path()
        part = dest.with_suffix(".part")
        try:
            with urllib.request.urlopen(ai_jpeg.MODEL_URL, timeout=30) as r, open(part, "wb") as f:
                total = int(r.headers.get("Content-Length") or ai_jpeg.MODEL_SIZE)
                got = 0
                while chunk := r.read(1 << 20):
                    if self._cancel:
                        raise RuntimeError("Cancelled")
                    f.write(chunk)
                    got += len(chunk)
                    self.progress.emit(got, total)
            if not ai_jpeg.verify_download(part):
                raise RuntimeError("The downloaded file didn't match the expected model (size/checksum)")
            part.replace(dest)
            self.done.emit()
        except Exception as e:  # noqa: BLE001
            part.unlink(missing_ok=True)
            self.failed.emit(str(e))


def _python_exe() -> str:
    exe = Path(sys.executable)
    console = exe.with_name("python.exe")
    return str(console if exe.name.lower() == "pythonw.exe" and console.exists() else exe)


class AISetupDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Set up Best quality mode")
        self.setMinimumWidth(520)
        self._proc: QProcess | None = None
        self._dl: _Download | None = None

        need_torch = not ai_jpeg.torch_installed()
        need_model = not ai_jpeg.model_ok()
        steps = []
        if need_torch:
            steps.append("• PyTorch with NVIDIA GPU support (about 3 GB download)")
        if need_model:
            steps.append("• The FBCNN JPEG model (288 MB) from github.com/jiaxi-jiang/FBCNN")
        intro = QLabel(
            "Best quality mode removes JPEG damage with an AI model (FBCNN, Apache-2.0) that runs on "
            "your graphics card. It needs a one-time setup:\n\n" + "\n".join(steps) +
            "\n\nFiles are stored on this PC; nothing is uploaded."
        )
        intro.setWordWrap(True)
        self.status = QLabel(objectName="Muted")
        self.bar = QProgressBar()
        self.bar.hide()
        self.log = QPlainTextEdit(readOnly=True)
        self.log.setMaximumBlockCount(400)
        self.log.setMinimumHeight(140)
        self.log.hide()
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.button(QDialogButtonBox.Ok).setText("Set up")
        self.buttons.button(QDialogButtonBox.Ok).setProperty("primary", True)
        self.buttons.accepted.connect(self._start)
        self.buttons.rejected.connect(self._cancel)

        lay = QVBoxLayout(self)
        lay.setSpacing(12)
        lay.addWidget(intro)
        lay.addWidget(self.status)
        lay.addWidget(self.bar)
        lay.addWidget(self.log)
        lay.addWidget(self.buttons)

    # ---- steps -------------------------------------------------------------
    def _start(self):
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(False)
        self._next()

    def _next(self):
        importlib.invalidate_caches()
        if not ai_jpeg.torch_installed():
            if getattr(sys, "frozen", False):
                self._fail("PyTorch can't be added to the packaged app. Run it from source to use Best quality.")
                return
            self._install_torch()
        elif not ai_jpeg.model_ok():
            self._download_model()
        else:
            self.accept()

    def _install_torch(self):
        self.status.setText("Installing PyTorch… (this can take several minutes)")
        self.bar.setRange(0, 0)
        self.bar.show()
        self.log.show()
        req = config.resource_path("requirements-ai.txt")
        self._proc = QProcess(self)
        self._proc.setProcessChannelMode(QProcess.MergedChannels)
        self._proc.readyReadStandardOutput.connect(
            lambda: self.log.appendPlainText(bytes(self._proc.readAllStandardOutput()).decode(errors="replace").rstrip())
        )
        self._proc.finished.connect(self._torch_done)
        self._proc.start(_python_exe(), ["-m", "pip", "install", "--progress-bar", "off", "-r", str(req)])

    def _torch_done(self, code, _status):
        self._proc = None
        if code != 0:
            self._fail("Installing PyTorch failed - see the messages above.")
            return
        self._next()

    def _download_model(self):
        self.status.setText("Downloading the FBCNN model…")
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.bar.show()
        self._dl = _Download()
        self._dl.progress.connect(self._dl_progress)
        self._dl.failed.connect(lambda msg: self._fail(f"Download failed: {msg}"))
        self._dl.done.connect(self._next)
        self._dl.start()

    def _dl_progress(self, got: int, total: int):
        self.bar.setValue(int(got * 100 / max(1, total)))
        self.status.setText(f"Downloading the FBCNN model… {got / 1e6:.0f} / {total / 1e6:.0f} MB")

    def _fail(self, msg: str):
        self.status.setText(msg)
        self.bar.hide()
        ok = self.buttons.button(QDialogButtonBox.Ok)
        ok.setText("Try again")
        ok.setEnabled(True)

    def _cancel(self):
        if self._proc is not None:
            self._proc.kill()
            self._proc.waitForFinished(3000)
        if self._dl is not None and self._dl.isRunning():
            self._dl.cancel()
            self._dl.wait()
        self.reject()


def ensure_ready(parent) -> bool:
    """True when Best quality can run; otherwise offers the setup and returns its outcome."""
    if ai_jpeg.status() == ai_jpeg.READY:
        return True
    return AISetupDialog(parent).exec() == QDialog.Accepted and ai_jpeg.status() == ai_jpeg.READY
