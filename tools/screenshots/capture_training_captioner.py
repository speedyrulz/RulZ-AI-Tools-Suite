"""Screenshots of Rulz Training Captioner with a generated demo dataset.

Run with the app's own venv:

    "C:\\coding\\Rulz Training Captioner\\.venv\\Scripts\\python.exe" -B capture_training_captioner.py ^
        --appdata R:\\tc-appdata --folder R:\\Dataset\\landscapes --out OUT_DIR

* APPDATA is pointed at --appdata before the app's config module is imported, so settings.json and
  prompts.json are created there instead of %APPDATA%\\RulzCaptioner.
* Images are added with MainWindow._add() (what drag-and-drop calls). No llama-server is started.
* The main window is grabbed with QWidget.grab(); the Settings dialog is captured with PrintWindow
  (wincap.py) and composited over the main window at its real on-screen offset.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

REPO = r"C:\coding\Rulz Training Captioner"
HERE = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()
ap.add_argument("--appdata", required=True)
ap.add_argument("--folder", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--size", default="1600x960")
args = ap.parse_args()

os.makedirs(args.appdata, exist_ok=True)
os.makedirs(args.out, exist_ok=True)
os.environ["APPDATA"] = args.appdata
sys.dont_write_bytecode = True
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
sys.argv = [os.path.join(REPO, "main.py")]  # the app looks for its bundled llama.cpp next to argv[0]

from PySide6.QtCore import QEventLoop, QPoint, Qt  # noqa: E402
from PySide6.QtGui import QColor, QPainter  # noqa: E402
from PySide6.QtWidgets import QApplication, QSlider  # noqa: E402

from captioner import config, theme  # noqa: E402
from captioner.main_window import MainWindow  # noqa: E402
from captioner.settings_dialog import SettingsDialog  # noqa: E402
import wincap  # noqa: E402

assert str(config.APP_DIR).startswith(args.appdata), config.APP_DIR

app = QApplication(sys.argv)
app.setApplicationName("Rulz Training Captioner")
theme.apply(app)
app.setWindowIcon(theme.icon())


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
    """Main window grab with a PrintWindow capture of a dialog pasted where it really is."""
    base = main.grab().toImage()
    hwnd = int(dlg.winId())
    w, h, bgra = wincap.capture(hwnd, "frame")
    shot = wincap.to_qimage(w, h, bgra)
    l, t, _, _ = wincap.frame_rect(hwnd)
    origin = main.mapToGlobal(QPoint(0, 0))
    x, y = l - origin.x(), t - origin.y()
    p = QPainter(base)
    p.fillRect(base.rect(), QColor(0, 0, 0, 90))  # the dim a modal dialog gives the eye
    for i in range(14, 0, -1):  # soft shadow
        p.fillRect(x - i + 6, y - i + 10, w + 2 * i - 12, h + 2 * i - 12, QColor(0, 0, 0, 10))
    p.drawImage(x, y, shot)
    p.setPen(QColor(70, 70, 76))
    p.drawRect(x, y, w - 1, h - 1)
    p.end()
    return base


win = MainWindow()
w, h = (int(v) for v in args.size.split("x"))
win.resize(w, h)
win.move(30, 20)
win.show()
pump(0.5)
win._add([args.folder])
pump(3.0)  # thumbnails load on a thread pool

# select one card so the selection highlight shows
idx = win.proxy.index(1, 0)
win.grid.setCurrentIndex(idx)
win.statusBar().clearMessage()  # show the image/caption counts
pump(0.5)
save(win.grab().toImage(), "01-caption-grid.png")

# bigger thumbnails and the inline caption editor open on one image
for slider in win.findChildren(QSlider):  # the toolbar's Thumbnails slider
    slider.setValue(224)  # 5 columns at 1600 px
pump(0.5)
idx = win.proxy.index(2, 0)
win.grid.setCurrentIndex(idx)
win.grid.edit(idx)
pump(0.8)
save(win.grab().toImage(), "02-edit-caption.png")
win.grid.closePersistentEditor(idx)
win.grid.setFocus()
pump(0.3)

dlg = SettingsDialog(win.current_settings(), win)
dlg.show()
geo = win.geometry()
dlg.move(geo.x() + (geo.width() - dlg.width()) // 2, geo.y() + (geo.height() - dlg.height()) // 2 - 20)
pump(1.0)
save(composite(win, dlg), "03-settings.png")
dlg.reject()
pump(0.3)

win.close()  # closeEvent saves settings.json -> inside --appdata
pump(0.3)
app.quit()
