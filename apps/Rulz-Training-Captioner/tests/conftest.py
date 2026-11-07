"""Shared test setup. Runs before the test modules import anything from the app."""
import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "minimal")  # no window needed; "offscreen" can hang on Windows
# settings and prompts go to a throwaway folder, never the user's %APPDATA%\RulzCaptioner
os.environ["RULZ_CAPTIONER_DIR"] = tempfile.mkdtemp(prefix="rulz-captioner-tests-")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtWidgets import QApplication  # noqa: E402

APP = QApplication.instance() or QApplication([])  # widgets (window, dialog) need a QApplication
