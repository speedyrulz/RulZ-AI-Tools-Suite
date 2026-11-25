"""Helpers for talking to WSL through ``wsl.exe``."""
from __future__ import annotations

import re

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, Signal

WSL_EXE = "wsl.exe"


def base_args(distro: str = "", user: str = "") -> list[str]:
    args: list[str] = []
    if distro.strip():
        args += ["-d", distro.strip()]
    if user.strip():
        args += ["-u", user.strip()]
    return args


def bash_stdin_args(distro: str = "", user: str = "") -> list[str]:
    """Arguments that make bash read its script from stdin. ``--exec`` skips the
    login shell, so no quoting of the script through a second shell is needed."""
    return base_args(distro, user) + ["--exec", "bash", "-s"]


_BROKEN_PATH_ENTRY = re.compile(r"^[A-Za-z]:\s")   # for example "C: \"


def clean_windows_path(path: str) -> str:
    """Drop PATH entries that wsl.exe cannot translate.

    WSL hands the Windows PATH on to Linux and prints a warning such as
    "wsl: Failed to translate 'C: \\'" for every malformed entry, on every start.
    Stray quotes are removed and entries that are empty or broken are left out.
    Only the copy given to wsl.exe is changed, never the system setting.
    """
    kept = []
    for entry in path.split(";"):
        entry = entry.strip().replace('"', "").strip()
        if entry and not _BROKEN_PATH_ENTRY.match(entry):
            kept.append(entry)
    return ";".join(kept)


def process_environment() -> QProcessEnvironment:
    env = QProcessEnvironment.systemEnvironment()
    env.insert("WSL_UTF8", "1")   # wsl.exe's own messages as UTF-8 instead of UTF-16
    for name in ("PATH", "Path"):
        if env.contains(name):
            env.insert(name, clean_windows_path(env.value(name)))
            break
    return env


def decode(data: bytes) -> str:
    """Decode output from wsl.exe. Its own messages can still be UTF-16 on old versions."""
    if data.count(b"\x00") > len(data) // 4:
        return data.decode("utf-16-le", errors="replace").replace("\x00", "")
    return data.decode("utf-8", errors="replace").replace("\x00", "")


class Command(QObject):
    """Run a short command and report its output once. Deletes itself when done.

    ``finished`` gives (exit code, stdout, stderr). The exit code is -1 when the
    program could not be started, crashed, or took longer than the timeout.
    """
    finished = Signal(int, str, str)

    def __init__(self, program: str, args: list[str], stdin_text: str | None = None,
                 timeout_ms: int = 20000, parent: QObject | None = None):
        super().__init__(parent)
        self._done = False
        self._stdin = stdin_text
        self._proc = QProcess(self)
        self._proc.setProcessEnvironment(process_environment())
        self._proc.started.connect(self._on_started)
        self._proc.finished.connect(self._on_finished)
        self._proc.errorOccurred.connect(self._on_error)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._on_timeout)
        self._timer.start(timeout_ms)
        # Started from the event loop, so the caller can connect to "finished"
        # first. A program that cannot be started reports that at once.
        QTimer.singleShot(0, self, lambda: self._proc.start(program, args))

    def _on_started(self):
        if self._stdin is not None:
            self._proc.write(self._stdin.encode("utf-8"))
        self._proc.closeWriteChannel()

    def _finish(self, code: int):
        if self._done:
            return
        self._done = True
        self._timer.stop()
        out = decode(self._proc.readAllStandardOutput().data())
        err = decode(self._proc.readAllStandardError().data())
        self.finished.emit(code, out, err)
        self.deleteLater()

    def _on_finished(self, code, status):
        self._finish(code if status == QProcess.ExitStatus.NormalExit else -1)

    def _on_error(self, error):
        if error == QProcess.ProcessError.FailedToStart:
            self._finish(-1)

    def _on_timeout(self):
        if self._done:
            return
        self._proc.kill()
        self._finish(-1)


def run_bash(script: str, distro: str = "", user: str = "", timeout_ms: int = 20000,
             parent: QObject | None = None) -> Command:
    """Run a bash script inside WSL (sent over stdin)."""
    return Command(WSL_EXE, bash_stdin_args(distro, user), script, timeout_ms, parent)


def list_distros(parent: QObject | None = None) -> Command:
    """``wsl -l -q``: one installed distro name per line."""
    return Command(WSL_EXE, ["-l", "-q"], None, 10000, parent)


def parse_distros(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]
