"""Screenshots of RulZ Image Organizer with a throw-away library of generated demo photos.

Run with a venv that has PySide6, OpenCV, numpy and Pillow (pillow-heif is optional; the app runs without it).
The Image Preparer venv has them:

    "%LOCALAPPDATA%\\Programs\\RulZ AI Tools\\RulZ Image Preparer\\.venv\\Scripts\\python.exe" -B ^
        capture_image_organizer.py --data R:\\io-data --folder "R:\\Photos\\Camera Roll" ^
        --folder "R:\\Photos\\Wallpapers" --out R:\\out\\image-organizer ^
        --readme R:\\out-readme\\image-organizer --viewer-file IMG_4272.jpg

* The app runs from apps/RulZ-Image-Organizer in this repository.
* RULZ_DATA_DIR points the library database, thumbnails and the app's settings.ini (theme, window geometry) at
  --data (never %LOCALAPPDATA%); with it set, the app doesn't read its old registry settings either. Use a fresh
  folder. RULZ_THEME=dark skips the first-run "Choose your look" prompt.
* The folders are added to the fresh library, a normal scan runs, then the window is grabbed with
  QWidget.grab() (client area only, nothing else on the desktop can leak in).
* Qt renders at 100 % (QT_SCALE_FACTOR = 96 / system DPI), so the images have the same pixel size whatever the
  display scaling is.
* --out gets the catalog set (01-04), --readme the app README's docs/screenshots set (01-05).
"""
from __future__ import annotations

import argparse
import ctypes
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", "apps", "RulZ-Image-Organizer"))

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--folder", action="append", required=True)
ap.add_argument("--out", required=True, help="catalog set")
ap.add_argument("--readme", default="", help="README set (docs/screenshots); skipped if empty")
ap.add_argument("--size", default="1600x960")
ap.add_argument("--viewer-index", type=int, default=5)
ap.add_argument("--viewer-file", default="", help="file name of the photo to open in the viewer")
ap.add_argument("--theme", default="dark")
args = ap.parse_args()

for d in (args.data, args.out, args.readme):
    if d:
        os.makedirs(d, exist_ok=True)
assert not os.path.exists(os.path.join(args.data, "settings.ini")), "use a fresh --data folder"
os.environ["RULZ_DATA_DIR"] = args.data
os.environ["RULZ_THEME"] = args.theme
sys.dont_write_bytecode = True
sys.path.insert(0, REPO)


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

from PySide6.QtCore import QEventLoop, Qt  # noqa: E402
from PySide6.QtGui import QIcon  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from rulz import config  # noqa: E402
from rulz.db import Database  # noqa: E402
from rulz.ui import theme  # noqa: E402
import rulz.ui.main_window as mw  # noqa: E402

assert os.path.normcase(str(config.data_dir())) == os.path.normcase(os.path.abspath(args.data)), config.data_dir()

QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
app = QApplication(sys.argv)
app.setApplicationName(config.APP_NAME)
app.setOrganizationName("RulZ")
icon_file = config.resource_path("assets", "icon.ico")
if icon_file.exists():
    app.setWindowIcon(QIcon(str(icon_file)))
dark = {"dark": True, "light": False}.get(args.theme, True)
theme.apply(app, dark)


def pump(seconds: float):
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents(QEventLoop.AllEvents, 50)
        time.sleep(0.01)


def save(widget, shots):
    img = widget.grab()
    for folder, name in shots:
        if folder:
            path = os.path.join(folder, name)
            img.save(path)
            print("saved", path, img.width(), "x", img.height(), flush=True)


# ---- "Choose your look", with the Light card picked (it previews the light theme)
if args.readme:
    from rulz.ui.dialogs import ThemeChoiceDialog

    look = ThemeChoiceDialog()
    look.light_card.setChecked(True)
    look.show()
    pump(1.0)
    save(look, [(args.readme, "05-choose-your-look.png")])
    look.reject()
    look.deleteLater()
    theme.apply(app, dark)
    pump(0.3)

db = Database()
if not db.folders():
    for f in args.folder:
        err = db.add_folder(f)
        if err:
            print("add_folder:", err)
db.close()

win = mw.MainWindow()
w, h = (int(v) for v in args.size.split("x"))
win.resize(w, h)
win.move(40, 20)
win.show()

# let the startup scan run to completion (it starts 300 ms after the window opens)
pump(1.5)
t0 = time.time()
while (win.scanner.running or win.scanner.queued) and time.time() - t0 < 900:
    pump(0.5)
print(f"scan finished in {time.time() - t0:.0f}s: {win.scanner.last_message}", flush=True)
pump(2.0)

win.navigate(win.PHOTOS)
pump(3.0)
save(win, [(args.out, "01-photos-timeline.png"), (args.readme, "01-photos-dark.png")])
if args.readme:
    win.set_theme(not dark)
    win.navigate(win.PHOTOS)
    pump(3.0)
    save(win, [(args.readme, "02-photos-light.png")])
    win.set_theme(dark)
    win.navigate(win.PHOTOS)
    pump(2.0)

win.navigate(win.PHOTOS)
pump(1.0)
ids = win.photos._photo_ids()
if ids:
    index = min(args.viewer_index, len(ids) - 1)
    if args.viewer_file:
        by_name = {os.path.basename(p.path).lower(): p.id for p in win.db.photos()}
        pid = by_name.get(args.viewer_file.lower())
        if pid in ids:
            index = ids.index(pid)
    win.open_viewer(ids, index)
    pump(2.5)
    save(win, [(args.out, "02-photo-viewer.png"), (args.readme, "03-photo-viewer.png")])
    win.viewer.close_viewer()
    pump(0.5)

win.photos.filter_btn.setChecked(True)
pump(2.0)
save(win, [(args.out, "03-search-filters.png"), (args.readme, "04-search-filters.png")])
win.photos.filter_btn.setChecked(False)

win.navigate(win.LIBRARY)
pump(1.5)
save(win, [(args.out, "04-library.png")])

win.scanner.stop()
win.scanner.wait(8000)
win.close()
pump(0.5)
app.quit()
