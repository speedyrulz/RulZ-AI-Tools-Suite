"""Application entry point."""
from __future__ import annotations

import os
import sys

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")


def _install_busy_handler() -> None:
    """A UI edit that can't get the library's write lock in time shows a message instead of a traceback."""
    import sqlite3

    default = sys.excepthook
    showing = False

    def hook(exc_type, exc, tb):
        nonlocal showing
        if issubclass(exc_type, sqlite3.OperationalError) and "locked" in str(exc) and not showing:
            from PySide6.QtWidgets import QApplication

            from .ui import dialogs

            showing = True
            try:
                dialogs.info(QApplication.activeWindow(), "Library is busy",
                             "RulZ is saving scan results right now, so this change wasn't saved. "
                             "Please try again in a moment.")
            finally:
                showing = False
            return
        default(exc_type, exc, tb)

    sys.excepthook = hook


def main() -> int:
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import QApplication

    from . import config
    from .ui import theme
    from .ui.main_window import MainWindow

    if sys.platform == "win32":
        try:  # own taskbar identity (icon/grouping) instead of python.exe's
            import ctypes

            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("RulZ.ImageOrganizer")
        except Exception:
            pass
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    app.setApplicationName(config.APP_NAME)
    app.setOrganizationName("RulZ")
    icon_file = config.resource_path("assets", "icon.ico")
    if icon_file.exists():
        app.setWindowIcon(QIcon(str(icon_file)))
    # Dark by default; RULZ_THEME=light|dark forces one. The first launch asks (and saves the answer).
    theme.apply(app)
    if theme.env_override() is None and theme.saved_choice() is None:
        from .ui.dialogs import choose_theme_first_run

        choose_theme_first_run()
    _install_busy_handler()
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
