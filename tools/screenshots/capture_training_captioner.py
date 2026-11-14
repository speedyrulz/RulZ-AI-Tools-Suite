"""Screenshots of Rulz Training Captioner with a generated demo dataset.

Run with the app's own venv (PySide6):

    "%LOCALAPPDATA%\\Programs\\RulZ AI Tools\\Rulz Training Captioner\\.venv\\Scripts\\python.exe" -B ^
        capture_training_captioner.py --appdata R:\\tc-config --folder R:\\Dataset\\landscapes ^
        --out R:\\out\\training-captioner --readme R:\\out-readme\\training-captioner ^
        --app-dir "R:\\Rulz Training Captioner"

* The app runs from apps/Rulz-Training-Captioner in this repository.
* RULZ_CAPTIONER_DIR (and APPDATA) point at --appdata before the app's config module is imported, so
  settings.json and prompts.json are created there instead of %APPDATA%\\RulzCaptioner. Use a fresh folder.
* --app-dir stands in for the install folder: an empty llama.cpp\\llama-server.exe placeholder is made there and
  argv[0] points into it, so Settings shows the bundled-server path the way an install does (R:\\...). Without it
  the field shows whatever llama-server is on PATH, or nothing. No llama-server is started.
* Images are added with MainWindow._add() (what drag-and-drop calls).
* Qt renders at 100 % (QT_SCALE_FACTOR = 96 / system DPI), so the images have the same pixel size whatever the
  display scaling is.
* The main window is grabbed with QWidget.grab(); the Settings dialog is captured with PrintWindow
  (wincap.py) and composited over the main window at its real on-screen offset.
* --out gets the catalog set (01-03), --readme the app README's docs/screenshots set (01-05).
"""
from __future__ import annotations

import argparse
import ctypes
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", "apps", "Rulz-Training-Captioner"))

ap = argparse.ArgumentParser()
ap.add_argument("--appdata", required=True)
ap.add_argument("--folder", required=True)
ap.add_argument("--out", required=True, help="catalog set")
ap.add_argument("--readme", default="", help="README set (docs/screenshots); skipped if empty")
ap.add_argument("--app-dir", default="", help="stand-in install folder for the llama-server path in Settings")
ap.add_argument("--size", default="1600x960")
args = ap.parse_args()

for d in (args.appdata, args.out, args.readme):
    if d:
        os.makedirs(d, exist_ok=True)
os.environ["APPDATA"] = args.appdata
os.environ["RULZ_CAPTIONER_DIR"] = os.path.join(args.appdata, "RulzCaptioner")
sys.dont_write_bytecode = True
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
if args.app_dir:
    os.makedirs(os.path.join(args.app_dir, "llama.cpp"), exist_ok=True)
    open(os.path.join(args.app_dir, "llama.cpp", "llama-server.exe"), "ab").close()  # empty placeholder
    sys.argv = [os.path.join(args.app_dir, "main.py")]
else:
    sys.argv = [os.path.join(REPO, "main.py")]  # the app looks for its bundled llama.cpp next to argv[0]


def system_dpi() -> int:
    u = ctypes.windll.user32
    u.SetThreadDpiAwarenessContext.restype = ctypes.c_void_p
    u.SetThreadDpiAwarenessContext.argtypes = [ctypes.c_void_p]
    old = u.SetThreadDpiAwarenessContext(ctypes.c_void_p(-4))  # per-monitor v2, this thread only
    try:
        return u.GetDpiForSystem()
    finally:
        u.SetThreadDpiAwarenessContext(old)


if system_dpi() != 96:
    os.environ.setdefault("QT_SCALE_FACTOR", f"{96 / system_dpi():.6f}")

from PySide6.QtCore import QEventLoop, QPoint  # noqa: E402
from PySide6.QtGui import QColor, QPainter  # noqa: E402
from PySide6.QtWidgets import QApplication, QSlider  # noqa: E402

from captioner import config, theme  # noqa: E402
from captioner.look_dialog import ChooseLookDialog  # noqa: E402
from captioner.main_window import MainWindow  # noqa: E402
from captioner.settings_dialog import SettingsDialog  # noqa: E402
import wincap  # noqa: E402

assert os.path.normcase(str(config.APP_DIR)).startswith(os.path.normcase(args.appdata)), config.APP_DIR
assert not config.SETTINGS_FILE.exists(), "use a fresh --appdata folder"

app = QApplication(sys.argv)
app.setApplicationName("Rulz Training Captioner")
theme.apply(app, theme.DARK)
app.setWindowIcon(theme.icon())


def pump(seconds: float):
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents(QEventLoop.AllEvents, 50)
        time.sleep(0.01)


def save(img, shots):
    for folder, name in shots:
        if not folder:
            continue
        path = os.path.join(folder, name)
        img.save(path)
        print("saved", path, img.width(), "x", img.height(), flush=True)


def grab_frame(widget):
    """PrintWindow capture with the title bar drawn active (WM_NCACTIVATE; the focus isn't taken)."""
    hwnd = int(widget.winId())
    ctypes.windll.user32.SendMessageW(ctypes.c_void_p(hwnd), 0x0086, 1, 0)
    pump(0.3)
    w, h, bgra = wincap.capture(hwnd, "frame")
    return wincap.to_qimage(w, h, bgra)


def composite(main, dlg):
    """Main window grab with a PrintWindow capture of a dialog pasted where it really is."""
    base = main.grab().toImage()
    hwnd = int(dlg.winId())
    shot = grab_frame(dlg)
    w, h = shot.width(), shot.height()
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


def settings_shot(shots):
    dlg = SettingsDialog(win.current_settings(), win)
    dlg.show()
    geo = win.geometry()
    dlg.move(geo.x() + (geo.width() - dlg.width()) // 2, geo.y() + (geo.height() - dlg.height()) // 2 - 20)
    pump(1.0)
    save(composite(win, dlg), shots)
    dlg.reject()
    dlg.deleteLater()
    pump(0.3)


def set_light(on: bool):
    win.a_light.setChecked(on)  # View > Light mode
    pump(1.0)


# ---- the first-run "Choose your look" dialog, on its own
if args.readme:
    look = ChooseLookDialog()
    look.show()
    pump(1.0)
    save(grab_frame(look), [(args.readme, "03-choose-your-look.png")])
    look.reject()  # = Dark
    look.deleteLater()
    theme.apply(app, theme.DARK)
    pump(0.3)

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
save(win.grab().toImage(), [(args.out, "01-caption-grid.png"), (args.readme, "01-main-dark.png")])
if args.readme:
    set_light(True)
    win.statusBar().clearMessage()
    save(win.grab().toImage(), [(args.readme, "02-main-light.png")])
    set_light(False)

# bigger thumbnails and the inline caption editor open on one image
for slider in win.findChildren(QSlider):  # the toolbar's Thumbnails slider
    slider.setValue(224)  # 5 columns at 1600 px
pump(0.5)
idx = win.proxy.index(2, 0)
win.grid.setCurrentIndex(idx)
win.grid.edit(idx)
pump(0.8)
save(win.grab().toImage(), [(args.out, "02-edit-caption.png"), (args.readme, "04-edit-caption.png")])
win.grid.closePersistentEditor(idx)
win.grid.setFocus()
pump(0.3)

settings_shot([(args.out, "03-settings.png")])
if args.readme:
    set_light(True)
    settings_shot([(args.readme, "05-settings-light.png")])
    set_light(False)

win.close()  # closeEvent saves settings.json -> inside --appdata
pump(0.3)
app.quit()
