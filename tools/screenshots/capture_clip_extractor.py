"""Screenshots of RulZ Clip Extractor (simplest-video-edit) using the repo's own sample videos.

Run with the app's own venv; Pillow is not needed:

    "C:\\simplest-video-edit\\.venv\\Scripts\\python.exe" -B capture_clip_extractor.py ^
        --config R:\\ce-config --out OUT_DIR

* The app keeps its preferences in QSettings("SimpleCut", "SimpleCut") = the registry. The harness swaps
  QSettings in the app's modules for an .ini file inside --config, so the registry is neither read nor written
  (and the screenshots show default preferences).
* Dialogs are modal (exec()), so a QTimer fires inside the dialog's event loop, captures it with PrintWindow
  (wincap.py), composites it over QWidget.grab() of the main window at its real offset, then cancels it.
* 01-crop-and-fixed-length.png is docs\\screenshot.png from the repo, copied as-is.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time

REPO = r"C:\simplest-video-edit"
HERE = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--size", default="1600x940")
args = ap.parse_args()
os.makedirs(args.config, exist_ok=True)
os.makedirs(args.out, exist_ok=True)
sys.dont_write_bytecode = True
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from PySide6.QtCore import QEventLoop, QPoint, QSettings, QTimer  # noqa: E402
from PySide6.QtGui import QColor, QPainter  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialog  # noqa: E402

INI = os.path.join(args.config, "SimpleCut.ini")


class TempSettings(QSettings):
    def __init__(self, *_a, **_k):
        super().__init__(INI, QSettings.IniFormat)


import simplecut.defaults as defaults_mod  # noqa: E402
import simplecut.colormatch as colormatch_mod  # noqa: E402
import simplecut.app as app_mod  # noqa: E402

for m in (defaults_mod, colormatch_mod, app_mod):
    m.QSettings = TempSettings
assert os.path.normcase(os.path.normpath(defaults_mod.settings().fileName())) == os.path.normcase(INI)

from simplecut import theme  # noqa: E402

SAMPLES = os.path.join(REPO, "samples")
shutil.copyfile(os.path.join(REPO, "docs", "screenshot.png"), os.path.join(args.out, "01-crop-and-fixed-length.png"))

app = QApplication(sys.argv)
theme.apply(app)
import wincap  # noqa: E402


def pump(seconds: float):
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents(QEventLoop.AllEvents, 50)
        time.sleep(0.01)


def save(img, name):
    path = os.path.join(args.out, name)
    img.save(path)
    print("saved", path, flush=True)


def composite(main, dlg):
    base = main.grab().toImage()
    hwnd = int(dlg.winId())
    w, h, bgra = wincap.capture(hwnd, "frame")
    shot = wincap.to_qimage(w, h, bgra)
    l, t, _, _ = wincap.frame_rect(hwnd)
    origin = main.mapToGlobal(QPoint(0, 0))
    x, y = l - origin.x(), t - origin.y()
    p = QPainter(base)
    p.fillRect(base.rect(), QColor(0, 0, 0, 90))
    for i in range(14, 0, -1):
        p.fillRect(x - i + 6, y - i + 10, w + 2 * i - 12, h + 2 * i - 12, QColor(0, 0, 0, 10))
    p.drawImage(x, y, shot)
    p.setPen(QColor(70, 70, 76))
    p.drawRect(x, y, w - 1, h - 1)
    p.end()
    return base


def modal_dialog():
    dlg = QApplication.activeModalWidget()
    if dlg is None:
        dlg = next((w for w in QApplication.topLevelWidgets() if isinstance(w, QDialog) and w.isVisible()), None)
    return dlg


def capture_modal(trigger, name, prep=None, delay_ms=1500):
    """Run trigger() (which exec()s a dialog); capture the dialog from inside its event loop."""
    def grab():
        dlg = modal_dialog()
        if dlg is None:
            print("no dialog for", name)
            return
        if prep:
            prep(dlg)
        pump(1.0)
        save(composite(win, dlg), name)
        dlg.reject()
    QTimer.singleShot(delay_ms, grab)
    trigger()
    pump(0.3)


def open_video(path, wait=4.0):
    win.open_file(path)
    t0 = time.time()
    while (win.info is None or os.path.normcase(win.info.path) != os.path.normcase(path)) and time.time() - t0 < 30:
        pump(0.2)
    pump(wait)


win = app_mod.MainWindow()
w, h = (int(v) for v in args.size.split("x"))
win.resize(w, h)
win.move(20, 20)
win.show()
pump(1.0)

# --- In/Out on keyframes -> the Export dialog offers a lossless stream copy
open_video(os.path.join(SAMPLES, "test_pattern_10s.mp4"))
win.seek_to(2.0)
pump(0.5)
win.set_in()
win.seek_to(6.0)
pump(0.5)
win.set_out()
win.seek_to(4.2)
pump(1.0)
capture_modal(win.export_clip, "02-export-dialog.png")

# --- Preferences
capture_modal(win.edit_preferences, "04-preferences.png")

# --- skin-tone colour match with the repo's synthetic sample (a drawn cartoon face, not a person)
TempSettings().setValue("ref_photo", os.path.join(SAMPLES, "skin_reference.png"))
open_video(os.path.join(SAMPLES, "skin_cast.mp4"))
win.seek_to(1.0)
pump(1.0)
capture_modal(win.match_color, "03-colour-match.png", delay_ms=2500)

win.stop()
win.close()
pump(0.5)
app.quit()
