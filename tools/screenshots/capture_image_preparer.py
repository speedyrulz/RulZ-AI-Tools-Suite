"""Screenshots of RulZ Image Preparer with generated demo images.

Run with the app's own venv (PySide6, OpenCV and numpy; no installs):

    "C:\\coding\\RulZ AI Tools\\RulZ Image Preparer\\.venv\\Scripts\\python.exe" -B capture_image_preparer.py ^
        --images R:\\Dataset\\preparer --config R:\\ip-config --out R:\\out\\image-preparer

* The demo pictures in --images (make_demo_images.py) are damaged into --config\\photos so there is something to
  clean up: speckled color noise, then saved as low-quality JPEGs. The originals are not changed.
* The app's QSettings (registry) are swapped for an .ini in --config, and LOCALAPPDATA points there too, so default
  settings are shown and nothing of the real install is read or written. Best quality (AI) mode isn't used.
* The app runs from apps/RulZ-Image-Preparer in this repository; only the main window is grabbed (QWidget.grab()).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
APP = HERE.parent.parent / "apps" / "RulZ-Image-Preparer"

ap = argparse.ArgumentParser()
ap.add_argument("--images", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--size", default="1600x960")
ap.add_argument("--closeup", default="Hillside", help="file name start of the picture opened for 02 and 03")
args = ap.parse_args()

config_dir, out = Path(args.config), Path(args.out)
photos = config_dir / "photos"
photos.mkdir(parents=True, exist_ok=True)
out.mkdir(parents=True, exist_ok=True)
os.environ["LOCALAPPDATA"] = str(config_dir)
sys.dont_write_bytecode = True
sys.path.insert(0, str(APP))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

# ---- demo damage: chroma noise like a high-ISO photo, then heavy JPEG compression ----
rng = np.random.default_rng(5)
sources = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
for i, src in enumerate(sources):
    img = cv2.imdecode(np.fromfile(str(src), np.uint8), cv2.IMREAD_COLOR)
    lab = cv2.cvtColor(img.astype(np.float32) / 255, cv2.COLOR_BGR2Lab)
    noise = rng.normal(0, 9 + 3 * (i % 3), lab[:, :, 1:].shape).astype(np.float32)
    lab[:, :, 1:] += cv2.GaussianBlur(noise, (0, 0), 1.4) * 2.6  # blotchy, not per-pixel
    lab[:, :, 0] += rng.normal(0, 1.5, lab[:, :, 0].shape).astype(np.float32)
    noisy = (np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2BGR), 0, 1) * 255 + 0.5).astype(np.uint8)
    quality = (12, 22, 35)[i % 3]
    cv2.imencode(".jpg", noisy, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tofile(str(photos / f"{src.stem}.jpg"))

from PySide6.QtCore import QEventLoop, QSettings  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from rulz import config  # noqa: E402

ini = QSettings(str(config_dir / "settings.ini"), QSettings.IniFormat)
ini.clear()
config.settings = lambda: ini

from rulz.ui import theme  # noqa: E402
from rulz.ui.compare_view import SIDE, SLIDER  # noqa: E402
from rulz.ui.main_window import MainWindow  # noqa: E402

app = QApplication(sys.argv)
app.setApplicationName(config.APP_NAME)
theme.apply(app)


def pump(seconds: float):
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents(QEventLoop.AllEvents, 50)
        time.sleep(0.01)


def save(name: str):
    pump(0.3)
    path = out / name
    win.grab().save(str(path))
    print("saved", path, flush=True)


def wait_editor(limit: float = 60):
    ed = win.editor
    end = time.time() + limit
    while time.time() < end and (ed._running or ed._full is None or ed._shown_key != ed._wanted()):
        pump(0.05)


w, h = map(int, args.size.split("x"))
win = MainWindow()
win.resize(w, h)
win.show()
win.add_paths([str(photos)])
pump(3)
for key, value in (("color", 65), ("jpeg", 55), ("detail", 20)):
    win.global_rows[key].slider.setValue(value)
win.items[1].custom = {"color": 85, "jpeg": 70}  # one image with its own values: the red Custom badge
win._on_item_changed(win.items[1])
pump(2)
save("01-grid.png")



def show(zoom: float, fx: float, fy: float):
    """Zoom to a point given as a fraction of the image size."""
    from PySide6.QtCore import QPointF

    ed.view.set_zoom(zoom)
    ed.view._center = QPointF(ed._full.shape[1] * fx, ed._full.shape[0] * fy)
    ed.view.update()


# the close-ups use a colorful, detailed picture: blotches and blocks show best there
row = next(i for i, it in enumerate(win.items) if it.path.stem.startswith(args.closeup))
win._open(row)
wait_editor()
ed = win.editor
ed._set_mode(SIDE, save=False)
show(2.0, 0.42, 0.55)
save("02-side-by-side.png")

ed._set_mode(SLIDER, save=False)
show(1.25, 0.5, 0.5)
save("03-slider.png")

win._show_library()
win.close()
