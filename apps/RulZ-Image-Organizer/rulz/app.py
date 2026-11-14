"""Application entry point."""
from __future__ import annotations

import os
import sys

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")


def _install_busy_handler() -> None:
    """A UI edit that can't get the library's write lock in time, or that refers to something a scan or
    regroup just changed, shows a message instead of a traceback (nobody sees those in the windowed app)."""
    import sqlite3

    from .db import PersonGone

    default = sys.excepthook
    showing = False

    def message(exc_type, exc) -> tuple[str, str] | None:
        if issubclass(exc_type, sqlite3.OperationalError) and "locked" in str(exc):
            return ("Library is busy", "RulZ is saving scan results right now, so this change wasn't saved. "
                    "Please try again in a moment.")
        if issubclass(exc_type, PersonGone) or (issubclass(exc_type, sqlite3.IntegrityError)
                                                and "FOREIGN KEY" in str(exc)):
            return ("Library changed", "That person or face was just changed by a scan or regroup, so this "
                    "change wasn't saved. The page has been updated; please try again.")
        return None

    def hook(exc_type, exc, tb):
        nonlocal showing
        text = message(exc_type, exc)
        if text is not None and not showing:
            from PySide6.QtWidgets import QApplication

            from .ui import dialogs

            showing = True
            try:
                win = QApplication.activeWindow()
                dialogs.info(win, *text)
                main = next((w for w in QApplication.topLevelWidgets()
                             if hasattr(w, "data_changed") and w.isVisible()), None)
                if main is not None and text[0] == "Library changed":
                    main.data_changed()
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
