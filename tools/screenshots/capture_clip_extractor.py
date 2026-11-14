"""Screenshots of RulZ Simple Video Clip Extractor using the repo's own sample videos and a drawn facade video.

Run with a venv that has PySide6, av and numpy; Pillow is not needed:

    "C:\\simplest-video-edit\\.venv\\Scripts\\python.exe" -B capture_clip_extractor.py ^
        --videos R:\\Videos --config R:\\ce-config --out R:\\out\\clip-extractor ^
        --readme R:\\out-readme\\clip-extractor --cv2-path "<a Python 3.12 site-packages with OpenCV>"

* The app runs from apps/rulz-simple-video-clip-extractor in this repository. Its preferences live in
  QSettings("SimpleCut", "SimpleCut") = the registry; SIMPLECUT_SETTINGS_INI points them at an .ini in --config
  (use a fresh folder), so the registry is neither read nor written and default preferences are shown.
* The repo's samples (test_pattern_10s.mp4, skin_cast.mp4 + skin_reference.png, a drawn cartoon face) are copied
  to --videos, so the paths the dialogs show read R:\\... . facade.mp4 (a drawn building seen from below, leaning
  inwards) is made there too, for the Distortion repair window.
* The Distortion repair window needs OpenCV. If the venv has none, --cv2-path appends a site-packages folder
  that has it (same Python version) to sys.path; anything the venv has itself still wins.
* Qt renders at 100 % (QT_SCALE_FACTOR = 96 / system DPI), so the images have the same pixel size whatever the
  display scaling is.
* Dialogs are modal (exec()), so a QTimer fires inside the dialog's event loop, captures it with PrintWindow
  (wincap.py), composites it over QWidget.grab() of the main window at its real offset, then cancels it.
  Nothing is exported.
* --out gets the catalog set (01-05), --readme the app README's docs/screenshots set (main-dark, main-light,
  choose-your-look, export-dialog, colour-match, distortion-repair).
"""
from __future__ import annotations

import argparse
import ctypes
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", "apps", "rulz-simple-video-clip-extractor"))

ap = argparse.ArgumentParser()
ap.add_argument("--videos", required=True, help="workspace folder for the sample videos")
ap.add_argument("--config", required=True)
ap.add_argument("--out", required=True, help="catalog set")
ap.add_argument("--readme", default="", help="README set (docs/screenshots); skipped if empty")
ap.add_argument("--cv2-path", default="", help="site-packages folder with OpenCV, if the venv has none")
ap.add_argument("--size", default="1612x940", help="main window size for the catalog set")
ap.add_argument("--readme-size", default="1612x900", help="main window size for the README set")
args = ap.parse_args()
for d in (args.videos, args.config, args.out, args.readme):
    if d:
        os.makedirs(d, exist_ok=True)
sys.dont_write_bytecode = True
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
if args.cv2_path:
    sys.path.append(args.cv2_path)

INI = os.path.join(args.config, "SimpleCut.ini")
os.environ["SIMPLECUT_SETTINGS_INI"] = INI


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

import numpy as np  # noqa: E402

# ---- the sample videos, copied so the dialogs show R:\ paths
for name in ("test_pattern_10s.mp4", "skin_cast.mp4", "skin_reference.png"):
    shutil.copyfile(os.path.join(REPO, "samples", name), os.path.join(args.videos, name))


def make_facade_video(path: str, w: int = 1280, h: int = 720, seconds: float = 4.0, fps: int = 30):
    """A drawn building facade seen from below: its sides lean inwards (keystone), sky around it."""
    import av

    ss = 2  # supersampling
    W, H = w * ss, h * ss
    # facade plane in metres (30 x 20), mapped to a trapezoid narrower at the top
    src = np.array([[0, 0], [30, 0], [30, 20], [0, 20]], float)
    dst = np.array([[0.13 * W, 0.30 * H], [0.87 * W, 0.30 * H], [1.02 * W, 1.16 * H], [-0.02 * W, 1.16 * H]])
    a = []
    for (x, y), (u, v) in zip(src, dst):
        a.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        a.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    hm = np.linalg.svd(np.array(a))[2][-1].reshape(3, 3)
    inv = np.linalg.inv(hm)
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float64)
    den = inv[2, 0] * xs + inv[2, 1] * ys + inv[2, 2]
    X = (inv[0, 0] * xs + inv[0, 1] * ys + inv[0, 2]) / den
    Y = (inv[1, 0] * xs + inv[1, 1] * ys + inv[1, 2]) / den
    t = ys / H
    img = np.empty((H, W, 3))
    img[..., 0] = 128 + 60 * t
    img[..., 1] = 186 + 34 * t
    img[..., 2] = 232 + 10 * t
    on = (X >= 0) & (X <= 30) & (Y >= 0) & (Y <= 20)
    wall = np.array([176.0, 132, 112])
    img[on] = wall
    img[on & (Y < 0.9)] = [128, 92, 80]                                   # parapet
    pitch_x, pitch_y = 30 / 11, 3.4
    cx, cy = np.mod(X, pitch_x), np.mod(Y - 1.6, pitch_y)
    win = on & (Y > 1.6) & (cx > 0.55) & (cx < 2.17) & (cy < 2.0)
    frame = win & ((cx < 0.67) | (cx > 2.05) | (cy < 0.12) | (cy > 1.88) | (np.abs(cx - 1.36) < 0.05))
    img[win] = [44, 54, 66]
    img[frame] = [236, 236, 230]
    ledge = on & (Y > 1.6) & (cy > 2.0) & (cy < 2.18) & (cx > 0.45) & (cx < 2.27)
    img[ledge] = [206, 170, 152]
    img = img.reshape(h, ss, w, ss, 3).mean(axis=(1, 3))
    rng = np.random.default_rng(3)
    img += rng.normal(0, 1.6, img.shape)
    rgb = np.clip(img + 0.5, 0, 255).astype(np.uint8)
    with av.open(path, "w") as out:
        st = out.add_stream("libx264", rate=fps)
        st.width, st.height, st.pix_fmt = w, h, "yuv420p"
        st.options = {"crf": "16", "g": str(fps)}
        for _ in range(int(seconds * fps)):
            for p in st.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")):
                out.mux(p)
        for p in st.encode():
            out.mux(p)


FACADE = os.path.join(args.videos, "facade.mp4")
make_facade_video(FACADE)

from PySide6.QtCore import QEventLoop, QPoint, QRectF, QTimer  # noqa: E402
from PySide6.QtGui import QColor, QPainter  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialog  # noqa: E402

import simplecut.app as app_mod  # noqa: E402
import simplecut.defaults as defaults_mod  # noqa: E402
from simplecut import theme  # noqa: E402

assert os.path.normcase(os.path.normpath(defaults_mod.settings().fileName())) == os.path.normcase(INI)
assert not defaults_mod.stored_theme(), "use a fresh --config folder"
if app_mod.REPAIR_UNAVAILABLE:
    print("warning:", app_mod.REPAIR_UNAVAILABLE, flush=True)

app = QApplication(sys.argv)
theme.apply(app)
import wincap  # noqa: E402


def pump(seconds: float):
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents(QEventLoop.AllEvents, 50)
        time.sleep(0.01)


def save(img, folder, name):
    if not folder:
        return
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
    base = main.grab().toImage()
    hwnd = int(dlg.winId())
    shot = grab_frame(dlg)
    w, h = shot.width(), shot.height()
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


def capture_modal(trigger, shots, prep=None, delay_ms=1500, alone=False):
    """Run trigger() (which exec()s a dialog); capture the dialog from inside its event loop.

    shots: [(folder, name)]. alone: save the dialog's own client area instead of a composite."""
    def grab():
        dlg = modal_dialog()
        if dlg is None:
            print("no dialog for", shots, flush=True)
            return
        if prep:
            prep(dlg)
        pump(1.0)
        img = dlg.grab().toImage() if alone else composite(win, dlg)
        for folder, name in shots:
            save(img, folder, name)
        if hasattr(dlg, "close_without_saving"):
            dlg.done(QDialog.Rejected)
        else:
            dlg.reject()
    QTimer.singleShot(delay_ms, grab)
    trigger()
    pump(0.3)


def open_video(path, wait=4.0):
    win.open_file(path)
    t0 = time.time()
    while (win.info is None or os.path.normcase(win.info.path) != os.path.normcase(path)) and time.time() - t0 < 30:
        pump(0.2)
    t0 = time.time()
    while not win.info.keyframes and time.time() - t0 < 30:  # lossless cuts wait for the keyframe scan
        pump(0.2)
    pump(wait)


def resize(size: str):
    w, h = (int(v) for v in size.split("x"))
    win.resize(w, h)
    pump(0.8)


# ---- "Choose your look": the first-run prompt, on its own
look = defaults_mod.ChooseLookDialog(app)
look.show()
pump(1.2)
save(grab_frame(look), args.readme, "choose-your-look.png")
look.reject()   # = Dark
look.deleteLater()
pump(0.3)
defaults_mod.save_theme(theme.DARK)

win = app_mod.MainWindow()
win.move(20, 20)
resize(args.size)
win.show()
pump(1.0)

# ---- a 16:9 crop and a 2.5 s fixed-length selection on the test pattern
TEST = os.path.join(args.videos, "test_pattern_10s.mp4")
open_video(TEST)
win.b_crop.setChecked(True)
win.aspect_combo.setCurrentText("16:9")
win.view.crop = QRectF(240, 100, 800, 450)
win.view.update()
win.view.cropChanged.emit()
win.len_sec.setValue(2.5)
win.seek_to(3.0)
pump(0.6)
win.b_fixed.setChecked(True)
pump(0.3)
win.seek_to(3.1)
pump(1.0)
save(win.grab().toImage(), args.out, "01-crop-and-fixed-length.png")
if args.readme:
    resize(args.readme_size)
    save(win.grab().toImage(), args.readme, "main-dark.png")
    win.set_theme(theme.LIGHT)
    pump(1.0)
    save(win.grab().toImage(), args.readme, "main-light.png")
    win.set_theme(theme.DARK)
    pump(0.8)
win.b_fixed.setChecked(False)
win.b_crop.setChecked(False)
win.aspect_combo.setCurrentText("Free")
pump(0.5)

# ---- In/Out on keyframes -> the Export dialog offers a lossless stream copy
win.seek_to(2.0)
pump(0.5)
win.set_in()
win.seek_to(6.0)
pump(0.5)
win.set_out()
win.seek_to(4.2)
pump(1.0)
if args.readme:
    capture_modal(win.export_clip, [(args.readme, "export-dialog.png")])
    resize(args.size)
capture_modal(win.export_clip, [(args.out, "02-export-dialog.png")])

# ---- Preferences
capture_modal(win.edit_preferences, [(args.out, "04-preferences.png")])

# ---- skin-tone colour match with the repo's synthetic sample (a drawn cartoon face, not a person)
defaults_mod.settings().setValue("ref_photo", os.path.join(args.videos, "skin_reference.png"))
open_video(os.path.join(args.videos, "skin_cast.mp4"))
win.seek_to(1.0)
pump(1.0)
capture_modal(win.match_color, [(args.out, "03-colour-match.png")], delay_ms=2500)
if args.readme:
    resize(args.readme_size)
    capture_modal(win.match_color, [(args.readme, "colour-match.png")], delay_ms=2500)


# ---- Distortion repair: Auto perspective (verticals) on the leaning facade; Auto crop is on by default
def straighten(dlg):
    dlg.resize(1612, 900)
    pump(1.5)
    dlg.c.perspective.auto_mode = "verticals"   # "Verticals only": a building seen from below
    dlg.run_auto_perspective()
    t0 = time.time()
    while (dlg.busy or dlg.pending is not None or dlg.render_timer.isActive() or dlg.hq_timer.isActive()) \
            and time.time() - t0 < 30:
        pump(0.1)
    pump(1.5)
    dlg.fit()
    pump(0.8)


if not app_mod.REPAIR_UNAVAILABLE:
    import simplecut.repair.window as repair_window
    from PySide6.QtWidgets import QMessageBox

    class _NoBox(QMessageBox):  # an "Auto failed" box would block the capture: print it instead
        @staticmethod
        def information(_parent, title, text, *_a, **_k):
            print(f"[{title}] {text}", flush=True)

    repair_window.QMessageBox = _NoBox
    open_video(FACADE)
    win.seek_to(2.0)
    pump(1.0)
    capture_modal(win.open_repair, [(args.out, "05-distortion-repair.png"), (args.readme, "distortion-repair.png")],
                  prep=straighten, delay_ms=2500, alone=True)

win.stop()
win.close()
pump(0.5)
app.quit()
