"""Application start-up."""
from __future__ import annotations

import argparse
import hashlib
import sys
import traceback
from pathlib import Path

from PySide6.QtCore import QLockFile, QTimer
from PySide6.QtNetwork import QLocalServer, QLocalSocket
from PySide6.QtWidgets import QApplication, QMessageBox

from . import APP_NAME, __version__
from .model import ConfigStore, default_config_dir
from .winjob import JobObject


def _server_name(config_path: Path) -> str:
    digest = hashlib.sha1(str(config_path).lower().encode("utf-8")).hexdigest()[:12]
    return f"rulz-comfyui-launcher-{digest}"


def _already_running(name: str) -> bool:
    """Ask a launcher that is already open (for the same config file) to show itself."""
    socket = QLocalSocket()
    socket.connectToServer(name)
    if not socket.waitForConnected(1500):
        return False
    socket.write(b"show")
    socket.waitForBytesWritten(300)
    socket.disconnectFromServer()
    return True


def _install_crash_log(folder: Path) -> None:
    def hook(kind, value, trace):
        text = "".join(traceback.format_exception(kind, value, trace))
        try:
            folder.mkdir(parents=True, exist_ok=True)
            with open(folder / "error.log", "a", encoding="utf-8") as handle:
                handle.write(text + "\n")
        except OSError:
            pass
        if QApplication.instance() is not None:
            QMessageBox.critical(None, APP_NAME, "Something went wrong:\n\n" + text[-1500:])

    sys.excepthook = hook


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog=APP_NAME, description="Start ComfyUI instances that run in WSL.")
    parser.add_argument("--config", help="path of the config file (default: %%APPDATA%%\\RulZ ComfyUI Launcher\\config.json)")
    parser.add_argument("--version", action="version", version=__version__)
    args, qt_args = parser.parse_known_args(sys.argv[1:] if argv is None else argv)

    config_path = Path(args.config).resolve() if args.config else default_config_dir() / "config.json"

    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("RulZ.ComfyUILauncher")
        except (OSError, AttributeError):
            pass

    app = QApplication([sys.argv[0], *qt_args])
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(__version__)
    app.setQuitOnLastWindowClosed(False)

    # One launcher per config file. The lock file decides who is first (a lock
    # left behind by a crash is detected and taken over); the local socket is
    # only used to ask the first launcher to show its window.
    server_name = _server_name(config_path)
    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    lock = QLockFile(str(config_path.with_name(config_path.name + ".lock")))
    lock.setStaleLockTime(0)
    if not lock.tryLock(200):
        if not _already_running(server_name):
            QMessageBox.information(None, APP_NAME, f"{APP_NAME} is already running.")
        return 0

    _install_crash_log(config_path.parent)

    from .ui import theme
    from .ui.main_window import MainWindow

    store = ConfigStore(config_path)
    store.load()
    theme.apply_theme(app, store.settings.theme)
    app.setWindowIcon(theme.app_icon())
    job = JobObject()
    window = MainWindow(store, job)

    server = QLocalServer(app)
    QLocalServer.removeServer(server_name)
    server.listen(server_name)

    def on_connection():
        connection = server.nextPendingConnection()
        if connection is not None:
            connection.disconnectFromServer()
            connection.deleteLater()
        window.show_and_raise()

    server.newConnection.connect(on_connection)
    window.show()
    QTimer.singleShot(0, window.raise_)
    code = app.exec()
    lock.unlock()
    return code
