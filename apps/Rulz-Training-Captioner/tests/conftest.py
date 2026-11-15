"""Shared test setup. Runs before the test modules import anything from the app."""
import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "minimal")  # no window needed; "offscreen" can hang on Windows
# settings and prompts go to a throwaway folder, never the user's %APPDATA%\RulzCaptioner
os.environ["RULZ_CAPTIONER_DIR"] = tempfile.mkdtemp(prefix="rulz-captioner-tests-")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

APP = QApplication.instance() or QApplication([])  # widgets (window, dialog) need a QApplication


@pytest.fixture
def boxes(monkeypatch):
    """Message boxes record what they would show instead of waiting for a click."""
    shown = []
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda parent, title, text, *a: shown.append(title)))
    monkeypatch.setattr(QMessageBox, "question",
                        staticmethod(lambda parent, title, text, *a: shown.append(title) or QMessageBox.Yes))
    return shown


@pytest.fixture
def win(boxes):
    """The main window, never shown. Imported here, after the settings folder above is in place."""
    from captioner import main_window
    w = main_window.MainWindow()
    yield w
    w.servers.stop()
    w.deleteLater()
