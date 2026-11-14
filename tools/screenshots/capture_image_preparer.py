"""Screenshots of RulZ Image Preparer with generated demo images.

Run with the app's own venv (PySide6, OpenCV and numpy; no installs):

    "%LOCALAPPDATA%\\Programs\\RulZ AI Tools\\RulZ Image Preparer\\.venv\\Scripts\\python.exe" -B capture_image_preparer.py ^
        --images R:\\Dataset\\preparer --config R:\\ip-config --out R:\\out\\image-preparer

* The demo pictures in --images (make_demo_images.py) are damaged into --config\\photos so there is something to
  clean up: speckled color noise, then saved as low-quality JPEGs. The originals are not changed.
* The app's QSettings (registry) are swapped for an .ini in --config, and LOCALAPPDATA points there too, so default
  settings are shown and nothing of the real install is read or written. Best quality (AI) mode isn't used and
  Auto crop's Preview is never pressed, so no model is loaded.
* --auto-crop "the mountain" opens the Auto crop screen over the picture list (without the PyTorch/SAM setup
  check), types the text, captures it from a timer inside its event loop with PrintWindow (wincap.py), composited
  over the main window at its real position, and closes it.
* The --shadow picture also gets a soft, bluish cloud shadow before the damage; 05 paints over it, soft edge
  included, as the wiki says to (Shadow areas, strokes added the way a brush stroke adds them).
* The app runs from apps/RulZ-Image-Preparer in this repository; only the main window is grabbed (QWidget.grab()).
* Qt renders at 100 % (QT_SCALE_FACTOR = 96 / system DPI), so the images have the same pixel size whatever the
  display scaling is.
"""
from __future__ import annotations

import argparse
import ctypes
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
ap.add_argument("--shadow", default="Desert", help="file name start of the picture given a cloud shadow for 05")
ap.add_argument("--auto-crop", default="", metavar="TEXT",
                help="also capture 04-auto-crop.png: the Auto crop screen with TEXT typed in, before Preview")
ap.add_argument("--crop-picture", default="Alpine Haze", help="file name start of the picture Auto crop shows")
args = ap.parse_args()

config_dir, out = Path(args.config), Path(args.out)
photos = config_dir / "photos"
photos.mkdir(parents=True, exist_ok=True)
out.mkdir(parents=True, exist_ok=True)
os.environ["LOCALAPPDATA"] = str(config_dir)
sys.dont_write_bytecode = True
sys.path.insert(0, str(APP))


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

import cv2  # noqa: E402
import numpy as np  # noqa: E402

# ---- a cloud shadow for the Shadow areas shot: an ellipse (centre and radii as fractions of width/height,
# angle in degrees), soft edge, bluish
SHADOW = (0.66, 0.60, 0.13, 0.06, -6)


def cloud_shadow(img):
    h, w = img.shape[:2]
    cx, cy, rx, ry, angle = SHADOW
    mask = np.zeros((h, w), np.float32)
    cv2.ellipse(mask, (int(cx * w), int(cy * h)), (int(rx * w), int(ry * h)), angle, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), 0.006 * w)[..., None]
    shade = np.array([0.66, 0.52, 0.46], np.float32)  # B, G, R: darker, with a blue cast
    return (img.astype(np.float32) * (1 - mask + mask * shade) + 0.5).astype(np.uint8)


# ---- demo damage: chroma noise like a high-ISO photo, then heavy JPEG compression ----
rng = np.random.default_rng(5)
sources = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
for i, src in enumerate(sources):
    img = cv2.imdecode(np.fromfile(str(src), np.uint8), cv2.IMREAD_COLOR)
    shadowed = bool(args.shadow) and src.stem.startswith(args.shadow)
    if shadowed:
        img = cloud_shadow(img)
    lab = cv2.cvtColor(img.astype(np.float32) / 255, cv2.COLOR_BGR2Lab)
    # The shadow picture is damaged lightly: in its deep shadow, heavy noise and JPEG blocks come up about
    # five times with the lift (clipped noise there also drains the color), which hides what 05 is showing.
    noise = rng.normal(0, 4 if shadowed else 9 + 3 * (i % 3), lab[:, :, 1:].shape).astype(np.float32)
    lab[:, :, 1:] += cv2.GaussianBlur(noise, (0, 0), 1.4) * 2.6  # blotchy, not per-pixel
    lab[:, :, 0] += rng.normal(0, 1.5, lab[:, :, 0].shape).astype(np.float32)
    noisy = (np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2BGR), 0, 1) * 255 + 0.5).astype(np.uint8)
    quality = 75 if shadowed else (12, 22, 35)[i % 3]
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

# ---- Shadow areas: brush strokes over the cloud shadow, then the whole picture side by side
if args.shadow:
    import math

    from rulz.shadows import Stroke

    row = next(i for i, it in enumerate(win.items) if it.path.stem.startswith(args.shadow))
    win._open(row)
    wait_editor()
    ed._set_mode(SIDE, save=False)
    brush = 30
    ed.brush.setValue(brush)
    ed.show_areas.setChecked(False)  # the red tint would hide the shadow on the original
    # 100 is as bright as the light filled in from around the shadow, which on this dune (darker toward the
    # bottom) comes out a little too bright; 85 is what the wiki's tip would settle on.
    ed.shadow_rows["strength"].slider.setValue(85)
    full_h, full_w = ed._full.shape[:2]
    side = max(full_h, full_w)
    radius = 0.003 + 0.197 * (brush / 100.0) ** 2  # the Brush slider's size, as the editor computes it
    cx, cy, rx, ry, angle = SHADOW
    a = math.radians(angle)
    for v in (-0.75, -0.25, 0.25, 0.75):  # rows that cover the ellipse and run a little past its soft edge
        half = rx * full_w * math.sqrt(1 - v * v) + 0.4 * radius * side
        pts = []
        for t in np.linspace(-1, 1, 24):
            x, y = half * t, v * ry * full_h
            pts.append(((cx * full_w + x * math.cos(a) - y * math.sin(a)) / full_w,
                        (cy * full_h + x * math.sin(a) + y * math.cos(a)) / full_h))
        ed._on_stroke(Stroke(tuple(pts), radius))
        pump(0.2)
    wait_editor(120)
    ed.view.fit()
    pump(1.0)
    save("05-shadow-areas.png")

win._show_library()
pump(1.0)

# ---- Auto crop: the screen with a description typed in, before Preview. It is opened the way the toolbar
# button opens it, minus ensure_crop_ready() (the PyTorch/SAM setup offer) and ai_mask.warm_up_task(), so SAM
# is never loaded; nothing in the screen touches the model until Preview or Crop is pressed.
if args.auto_crop:
    from PySide6.QtCore import QPoint, QTimer
    from PySide6.QtGui import QColor, QPainter

    import wincap
    from rulz.ui.auto_crop import AutoCropDialog

    def grab_frame(widget):
        """PrintWindow capture with the title bar drawn active (WM_NCACTIVATE; the focus isn't taken)."""
        hwnd = int(widget.winId())
        ctypes.windll.user32.SendMessageW(ctypes.c_void_p(hwnd), 0x0086, 1, 0)
        pump(0.3)
        cw, ch, bgra = wincap.capture(hwnd, "frame")
        return wincap.to_qimage(cw, ch, bgra)

    def composite(main, dlg):
        """Main window grab with the dialog (title bar included) pasted where it really is, the window dimmed."""
        base = main.grab().toImage()
        shot = grab_frame(dlg)
        dw, dh = shot.width(), shot.height()
        left, top, _, _ = wincap.frame_rect(int(dlg.winId()))
        origin = main.mapToGlobal(QPoint(0, 0))
        x, y = left - origin.x(), top - origin.y()
        p = QPainter(base)
        p.fillRect(base.rect(), QColor(0, 0, 0, 90))
        for i in range(14, 0, -1):
            p.fillRect(x - i + 6, y - i + 10, dw + 2 * i - 12, dh + 2 * i - 12, QColor(0, 0, 0, 10))
        p.drawImage(x, y, shot)
        p.setPen(QColor(70, 70, 76))
        p.drawRect(x, y, dw - 1, dh - 1)
        p.end()
        return base

    def grab_auto_crop():
        dlg = next(w for w in QApplication.topLevelWidgets() if isinstance(w, AutoCropDialog) and w.isVisible())
        row = next((i for i, it in enumerate(dlg.items) if it.path.stem.startswith(args.crop_picture)), 0)
        dlg.strip.setCurrentRow(row)
        dlg.prompt.setText(args.auto_crop)
        assert dlg._finder is None and not ai_mask.loaded(), "SAM must not be loaded"
        pump(1.0)
        path = out / "04-auto-crop.png"
        composite(win, dlg).save(str(path))
        print("saved", path, flush=True)
        dlg.reject()

    from rulz import ai_mask

    QTimer.singleShot(1500, grab_auto_crop)
    crop_dlg = AutoCropDialog(list(win.items), win.globals, win.mode, win, library=win.items)
    crop_dlg.exec()
    crop_dlg.deleteLater()
    pump(0.3)
    assert not ai_mask.loaded()

win.close()
