"""Screenshots of RulZ Image Organizer with a throw-away library of generated demo photos.

Run with the app's own venv (it has PySide6, OpenCV, Pillow):

    "C:\\coding\\RulZ Image Organizer\\.venv\\Scripts\\python.exe" -B capture_image_organizer.py ^
        --data R:\\io-data --folder "R:\\Photos\\Camera Roll" --folder "R:\\Photos\\Wallpapers" --out OUT_DIR

* RULZ_DATA_DIR points the library database/thumbnails at --data (never %LOCALAPPDATA%).
* QSettings (window geometry) is redirected to an .ini inside --data, so the registry is not touched.
* The folders are added to the fresh library, a normal scan runs, then the window is grabbed with
  QWidget.grab() (client area only, nothing else on the desktop can leak in).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

REPO = r"C:\coding\RulZ Image Organizer"

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--folder", action="append", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--size", default="1600x960")
ap.add_argument("--viewer-index", type=int, default=5)
ap.add_argument("--viewer-file", default="", help="file name of the photo to open in the viewer")
ap.add_argument("--theme", default="dark")
args = ap.parse_args()

os.makedirs(args.data, exist_ok=True)
os.makedirs(args.out, exist_ok=True)
os.environ["RULZ_DATA_DIR"] = args.data
os.environ["RULZ_THEME"] = args.theme
sys.dont_write_bytecode = True
sys.path.insert(0, REPO)

from PySide6.QtCore import QEventLoop, QSettings, Qt  # noqa: E402
from PySide6.QtGui import QIcon  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

INI = os.path.join(args.data, "qsettings.ini")


class TempSettings(QSettings):
    def __init__(self, *_a, **_k):
        super().__init__(INI, QSettings.IniFormat)


from rulz import config  # noqa: E402
from rulz.db import Database  # noqa: E402
from rulz.ui import theme  # noqa: E402
import rulz.ui.main_window as mw  # noqa: E402

mw.QSettings = TempSettings

QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
app = QApplication(sys.argv)
app.setApplicationName(config.APP_NAME)
app.setOrganizationName("RulZ")
icon_file = config.resource_path("assets", "icon.ico")
if icon_file.exists():
    app.setWindowIcon(QIcon(str(icon_file)))
theme.apply(app, {"dark": True, "light": False}.get(args.theme))


def pump(seconds: float):
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents(QEventLoop.AllEvents, 50)
        time.sleep(0.01)


def save(widget, name):
    path = os.path.join(args.out, name)
    widget.grab().save(path)
    print("saved", path, flush=True)


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
save(win, "01-photos-timeline.png")

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
    save(win, "02-photo-viewer.png")
    win.viewer.close_viewer()
    pump(0.5)

win.photos.filter_btn.setChecked(True)
pump(2.0)
save(win, "03-search-filters.png")
win.photos.filter_btn.setChecked(False)

win.navigate(win.LIBRARY)
pump(1.5)
save(win, "04-library.png")

win.scanner.stop()
win.scanner.wait(8000)
win.close()
pump(0.5)
app.quit()
