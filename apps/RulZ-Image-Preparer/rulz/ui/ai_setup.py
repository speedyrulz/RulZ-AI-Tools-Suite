"""One-time setup for the AI features: installs Python packages, then downloads model files.

Best quality mode needs PyTorch and the FBCNN model; Auto crop needs PyTorch, Transformers and SAM 3.1.
"""
from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import sys
import urllib.request
from pathlib import Path

from PySide6.QtCore import QProcess, QThread, Signal
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QLabel, QPlainTextEdit, QProgressBar, QVBoxLayout,
)

from .. import ai_jpeg, ai_mask, config


class _Download(QThread):
    """Downloads (url, destination, size, sha256 check) files one after another."""

    progress = Signal(int, int)  # bytes done, total
    failed = Signal(str)
    done = Signal()

    def __init__(self, files: list[tuple[str, Path, int, callable]]):
        super().__init__()
        self.files = files
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        total = sum(size for _, _, size, _ in self.files)
        got = 0
        part = None
        try:
            for url, dest, size, check in self.files:
                part = dest.with_name(dest.name + ".part")
                with urllib.request.urlopen(url, timeout=30) as r, open(part, "wb") as f:
                    while chunk := r.read(1 << 20):
                        if self._cancel:
                            raise RuntimeError("Cancelled")
                        f.write(chunk)
                        got += len(chunk)
                        self.progress.emit(got, total)
                if not check(part):
                    raise RuntimeError(f"{dest.name} didn't match the expected file (size/checksum)")
                part.replace(dest)
                part = None
            self.done.emit()
        except Exception as e:  # noqa: BLE001
            if part is not None:
                part.unlink(missing_ok=True)
            self.failed.emit(str(e))


def _python_exe() -> str:
    exe = Path(sys.executable)
    console = exe.with_name("python.exe")
    return str(console if exe.name.lower() == "pythonw.exe" and console.exists() else exe)


def _torch_pin() -> list[str]:
    """The installed PyTorch as a requirement, so adding torchvision doesn't swap PyTorch for another build."""
    try:
        return [f"torch=={importlib.metadata.version('torch')}"]
    except importlib.metadata.PackageNotFoundError:
        return []


class _SetupDialog(QDialog):
    """Runs pip for whatever packages are missing, then downloads whatever model files are missing."""

    title = ""
    intro = ""
    packages_label = ""
    model_label = ""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.title)
        self.setMinimumWidth(540)
        self._proc: QProcess | None = None
        self._dl: _Download | None = None
        self._pip: list[list[str]] = []

        intro = QLabel(self.intro + "\n\n" + "\n".join(f"• {s}" for s in self.steps()) +
                       "\n\nFiles are stored on this PC; nothing is uploaded.")
        intro.setWordWrap(True)
        self.status = QLabel(objectName="Muted")
        self.status.setWordWrap(True)
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

    # ---- what each feature needs ------------------------------------------
    def steps(self) -> list[str]:
        raise NotImplementedError

    def packages_needed(self) -> bool:
        raise NotImplementedError

    def requirements(self) -> list[list[str]]:
        """pip install runs (their arguments) that add the missing packages, in order."""
        raise NotImplementedError

    def downloads(self) -> list[tuple[str, Path, int, callable]]:
        raise NotImplementedError

    # ---- steps -------------------------------------------------------------
    def _start(self):
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(False)
        self._next()

    def _next(self):
        importlib.invalidate_caches()
        if self.packages_needed():
            if getattr(sys, "frozen", False):
                self._fail("These packages can't be added to the packaged app. Run it from source to use this.")
                return
            self._pip = self.requirements()
            self._install()
        elif self.downloads():
            self._download()
        else:
            self.accept()

    def _install(self):
        self.status.setText(f"Installing {self.packages_label}… (this can take several minutes)")
        self.bar.setRange(0, 0)
        self.bar.show()
        self.log.show()
        self._proc = QProcess(self)
        self._proc.setProcessChannelMode(QProcess.MergedChannels)
        self._proc.readyReadStandardOutput.connect(
            lambda: self.log.appendPlainText(bytes(self._proc.readAllStandardOutput()).decode(errors="replace").rstrip())
        )
        self._proc.finished.connect(self._install_done)
        self._proc.start(_python_exe(), ["-m", "pip", "install", "--progress-bar", "off", *self._pip.pop(0)])

    def _install_done(self, code, _status):
        self._proc = None
        if code != 0:
            self._fail(f"Installing {self.packages_label} failed - see the messages above.")
            return
        if self._pip:
            self._install()
            return
        importlib.invalidate_caches()
        if self.packages_needed():  # pip succeeded but the packages still don't import
            self._fail(f"{self.packages_label} installed but can't be loaded. Restart the app and try again.")
            return
        self._next()

    def _download(self):
        self.status.setText(f"Downloading {self.model_label}…")
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.bar.show()
        self._dl = _Download(self.downloads())
        self._dl.progress.connect(self._dl_progress)
        self._dl.failed.connect(lambda msg: self._fail(f"Download failed: {msg}"))
        self._dl.done.connect(self._next)
        self._dl.start()

    def _dl_progress(self, got: int, total: int):
        self.bar.setValue(int(got * 100 / max(1, total)))
        self.status.setText(f"Downloading {self.model_label}… {got / 1e6:.0f} / {total / 1e6:.0f} MB")

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


class AISetupDialog(_SetupDialog):
    title = "Set up Best quality mode"
    intro = ("Best quality mode removes JPEG damage with an AI model (FBCNN, Apache-2.0) that runs on "
             "your graphics card. It needs a one-time setup:")
    packages_label = "PyTorch"
    model_label = "the FBCNN model"

    def steps(self):
        out = []
        if not ai_jpeg.torch_installed():
            out.append("PyTorch with NVIDIA GPU support (about 3 GB download)")
        if not ai_jpeg.model_ok():
            out.append("The FBCNN JPEG model (288 MB) from github.com/jiaxi-jiang/FBCNN")
        return out

    def packages_needed(self):
        return not ai_jpeg.torch_installed()

    def requirements(self):
        return [["-r", str(config.resource_path("requirements-ai.txt"))]]

    def downloads(self):
        if ai_jpeg.model_ok():
            return []
        return [(ai_jpeg.MODEL_URL, ai_jpeg.model_path(), ai_jpeg.MODEL_SIZE, ai_jpeg.verify_download)]


class CropSetupDialog(_SetupDialog):
    title = "Set up Auto crop"
    intro = ("Auto crop finds what you describe with SAM 3.1 (Meta's Segment Anything 3.1, SAM License), "
             "an AI model that runs on your graphics card. It needs a one-time setup:")
    packages_label = "the AI packages"
    model_label = "SAM 3.1"

    def steps(self):
        out = []
        if not ai_mask.torch_installed():
            out.append("PyTorch with NVIDIA GPU support (about 3 GB download)")
        if not ai_mask.libs_installed():
            out.append("Transformers and TorchVision, which run SAM 3.1 (about 100 MB)")
        missing = ai_mask.missing_files()
        if missing:
            mb = sum(size for _, size, _ in missing) / 1e6
            out.append(f"The SAM 3.1 model ({mb:,.0f} MB) from huggingface.co/{ai_mask.REPO}")
        return out

    def packages_needed(self):
        return ai_mask.status() in (ai_mask.NO_TORCH, ai_mask.NO_LIBS)

    def requirements(self):
        runs = []
        if not (ai_mask.torch_installed() and importlib.util.find_spec("torchvision")):
            runs.append(["-r", str(config.resource_path("requirements-ai.txt")), *_torch_pin(), "torchvision"])
        runs.append(["-r", str(config.resource_path("requirements-crop.txt"))])
        return runs

    def downloads(self):
        d = ai_mask.model_dir()
        return [(ai_mask.file_url(name), d / name, size, lambda p, s=size, h=sha: ai_mask.verify_download(p, s, h))
                for name, size, sha in ai_mask.missing_files()]


def ensure_ready(parent) -> bool:
    """True when Best quality can run; otherwise offers the setup and returns its outcome."""
    if ai_jpeg.status() == ai_jpeg.READY:
        return True
    return AISetupDialog(parent).exec() == QDialog.Accepted and ai_jpeg.status() == ai_jpeg.READY


def ensure_crop_ready(parent) -> bool:
    """True when Auto crop can run; otherwise offers the setup and returns its outcome."""
    if ai_mask.status() == ai_mask.READY:
        return True
    return CropSetupDialog(parent).exec() == QDialog.Accepted and ai_mask.status() == ai_mask.READY
