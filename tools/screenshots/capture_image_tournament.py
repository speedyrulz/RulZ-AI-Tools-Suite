"""Screenshots of RulZ Image Tournament without touching the user's settings or saved tournament.

The app keeps settings.ini and tournament.txt (the resumable tournament) in %APPDATA%\\RulZ Image Tournament and
rewrites both on exit, and declining its resume prompt deletes the saved tournament. Since 1.2.0 it uses
<RULZ_TOURNAMENT_DATA>\\RulZ Image Tournament instead when that variable is set; this script sets it to --data.

    1. build_tournament_copy.ps1 -Out R:\\tt-build   (compiles the repo's sources outside the repo, like build.ps1)
    2. python capture_image_tournament.py --exe "R:\\tt-build\\RulZ Image Tournament.exe" --data R:\\tt-data ^
           --images R:\\Tournament --pair "Sea Glow 08.jpg" --out OUT_DIR
       (--images: a folder of generated images, e.g. make_demo_images.py R:\\Tournament --count 8 --seed 33
       --names titles; --pair replaces one of them with an original + "warm edit" pair for the first match)

The app's README set (docs/screenshots) comes from the same images without --pair, in a fresh --data folder:

    python capture_image_tournament.py --exe ... --data R:\\tt-data-readme --images R:\\Tournament ^
        --out OUT_DIR --choose-look 05-choose-your-look.png --light-shot 02-light.png ^
        --shots 01-dark.png,03-split-view.png,04-synced-zoom.png --zoom-at 0.2 0.45

--choose-look leaves the theme out of settings.ini, so the first-run "Choose your look" window opens; it is
captured and closed (= Dark). --light-shot ticks "Light mode" after the first shot, captures, and unticks it.

Only generated demo images (in a temp folder) are passed in. No Keep button / arrow key is ever sent, so no
file is moved. Windows are captured with PrintWindow (wincap.py); keys and clicks are posted to the app's own
windows. With display scaling above 100 %, the app runs DPI-unaware (__COMPAT_LAYER=DPIUNAWARE, this process
only), so it draws at 100 % and the images keep the old pixel size. Needs Pillow (the Image Preparer venv has it).
"""
from __future__ import annotations

import argparse
import ctypes
import glob
import os
import subprocess
import sys
import time
from ctypes import wintypes

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wincap  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--exe", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--images", required=True)
ap.add_argument("--pair", default="", help="make the first match this image vs. a colour-graded copy of it")
ap.add_argument("--out", required=True)
ap.add_argument("--size", default="1600x1000")
ap.add_argument("--theme", default="dark", choices=("dark", "light"))
ap.add_argument("--shots", default="01-side-by-side.png,02-split-view.png,03-synced-zoom.png",
                help="file names for the side-by-side, split and zoom shots (an empty name skips that shot)")
ap.add_argument("--light-shot", default="", help="also capture the first match with Light mode ticked")
ap.add_argument("--choose-look", default="", help="capture the first-run 'Choose your look' window as this file")
ap.add_argument("--zoom-at", type=float, nargs=2, default=(0.313, 0.508),
                help="where to wheel-zoom, as fractions of the compare view (default: the sun in the left image)")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
shot_names = (a.shots.split(",") + ["", "", ""])[:3]

user32 = ctypes.WinDLL("user32", use_last_error=True)
user32.SetThreadDpiAwarenessContext.restype = ctypes.c_void_p
user32.SetThreadDpiAwarenessContext.argtypes = [ctypes.c_void_p]
user32.SendMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]


def system_dpi() -> int:
    old = user32.SetThreadDpiAwarenessContext(ctypes.c_void_p(-4))  # per-monitor v2, this thread only
    try:
        return user32.GetDpiForSystem()
    finally:
        user32.SetThreadDpiAwarenessContext(old)


UNAWARE = system_dpi() != 96
if UNAWARE:
    # This process stays DPI-unaware too, so window rectangles and PrintWindow agree (96 DPI units). Only the
    # DWM frame bounds come back in physical pixels: convert them.
    _dwm_frame_rect = wincap.frame_rect

    def _logical_frame_rect(hwnd):
        l, t, r, b = wincap.window_rect(hwnd)
        old = user32.SetThreadDpiAwarenessContext(ctypes.c_void_p(-4))
        try:
            pl, pt, pr, pb = wincap.window_rect(hwnd)
            fl, ft, fr, fb = _dwm_frame_rect(hwnd)
        finally:
            user32.SetThreadDpiAwarenessContext(old)
        s = (r - l) / max(1, pr - pl)
        return (l + round((fl - pl) * s), t + round((ft - pt) * s), r - round((pr - fr) * s), b - round((pb - fb) * s))

    wincap.frame_rect = _logical_frame_rect

cfg = os.path.join(a.data, "RulZ Image Tournament")
os.makedirs(cfg, exist_ok=True)
assert not os.path.exists(os.path.join(cfg, "tournament.txt")), "use a fresh --data folder"
w, h = (int(v) for v in a.size.split("x"))
with open(os.path.join(cfg, "settings.ini"), "w", encoding="utf-8") as f:
    f.write(f"x=40\ny=16\nw={w}\nh={h}\nmaximized=False\nviewMode=SideBySide\n")
    if not a.choose_look:
        f.write(f"theme={a.theme}\n")

if a.pair:
    # A "which edit is better?" match: the image and a warmer, punchier grade of it. The app sorts files
    # naturally, so a "00 " prefix makes this pair the first match.
    from PIL import Image, ImageEnhance

    src = os.path.join(a.images, a.pair)
    stem = os.path.splitext(a.pair)[0]
    im = Image.open(src).convert("RGB")
    im.save(os.path.join(a.images, f"00 {stem}.jpg"), quality=92)
    r, g, b = im.split()
    warm = Image.merge("RGB", (r.point(lambda v: min(255, int(v * 1.07))), g, b.point(lambda v: int(v * 0.9))))
    warm = ImageEnhance.Contrast(ImageEnhance.Color(warm).enhance(1.3)).enhance(1.12)
    warm.save(os.path.join(a.images, f"00 {stem} (warm edit).jpg"), quality=92)
    os.remove(src)

files = sorted(glob.glob(os.path.join(a.images, "*.jpg")))

env = dict(os.environ, RULZ_TOURNAMENT_DATA=a.data)
if UNAWARE:
    env["__COMPAT_LAYER"] = "DPIUNAWARE"
proc = subprocess.Popen([a.exe, *files], env=env)
print("started pid", proc.pid, flush=True)


def save(hwnd, name, area="client"):
    cw, ch, bgra = wincap.capture(hwnd, area)
    path = os.path.join(a.out, name)
    wincap.save_png(path, cw, ch, bgra)
    print("saved", path, cw, "x", ch, flush=True)


def client_size(hwnd):
    l, t, r, b = wincap.client_rect(hwnd)
    return r - l, b - t


def no_message_box():
    if any(c == "#32770" for _, _, c in wincap.find_windows(proc.pid)):
        raise SystemExit("a message box appeared: " + repr(wincap.find_windows(proc.pid)))


def wait_window(pred, what, limit=20.0):
    t0 = time.time()
    while time.time() - t0 < limit:
        time.sleep(0.3)
        no_message_box()
        for hw, t, c in wincap.find_windows(proc.pid):
            if pred(hw, t, c):
                return hw
    raise SystemExit(what + " not found")


def child_with_text(parent, text):
    for hw, _cls, _rect in wincap.child_windows(parent):
        buf = ctypes.create_unicode_buffer(256)
        user32.GetWindowTextW(hw, buf, 256)
        if buf.value == text:
            return hw
    return None


try:
    if a.choose_look:
        # the first-run prompt opens before the main window; it's the small "RulZ Image Tournament" window
        look = wait_window(lambda hw, t, c: t.startswith("RulZ Image Tournament") and client_size(hw)[0] < 800,
                           "the Choose your look window")
        time.sleep(1.5)
        user32.SendMessageW(look, 0x0086, 1, 0)  # WM_NCACTIVATE: draw the title bar active (focus isn't taken)
        time.sleep(0.3)
        save(look, a.choose_look, "frame")
        wincap.user32.PostMessageW(look, 0x0010, 0, 0)  # WM_CLOSE = Dark, saved to --data

    main = wait_window(lambda hw, t, c: t.startswith("RulZ Image Tournament") and client_size(hw)[0] >= 800,
                       "the main window")
    time.sleep(3.0)  # images decode on background tasks
    no_message_box()
    if shot_names[0]:
        save(main, shot_names[0])

    if a.light_shot:
        light = child_with_text(main, "Light mode")
        if light is None:
            raise SystemExit("Light mode checkbox not found")
        user32.SendMessageW(light, 0x00F5, 0, 0)  # BM_CLICK: tick Light mode
        time.sleep(1.5)
        save(main, a.light_shot)
        user32.SendMessageW(light, 0x00F5, 0, 0)  # and back
        time.sleep(1.0)

    # the compare view is the biggest child window; keys posted to it reach the form's ProcessCmdKey
    kids = wincap.child_windows(main)
    view = max(kids, key=lambda k: (k[2][2] - k[2][0]) * (k[2][3] - k[2][1]))[0]
    if shot_names[1]:
        wincap.post_key(view, 0x56)  # V: side by side <-> split
        time.sleep(1.5)
        save(main, shot_names[1])
        wincap.post_key(view, 0x56)  # back to side by side
        time.sleep(1.0)

    if shot_names[2]:
        # wheel-zoom towards a spot in the left image: both images zoom together
        l, t, r, b = wincap.client_rect(view)
        x, y = l + int((r - l) * a.zoom_at[0]), t + int((b - t) * a.zoom_at[1])
        for _ in range(4):
            wincap.user32.PostMessageW(view, 0x020A, (120 << 16), ((y & 0xFFFF) << 16) | (x & 0xFFFF))  # WM_MOUSEWHEEL
            time.sleep(0.3)
        time.sleep(1.5)
        save(main, shot_names[2])
finally:
    if proc.poll() is None:
        for hw, _t, _c in wincap.find_windows(proc.pid):
            wincap.user32.PostMessageW(hw, 0x0010, 0, 0)  # WM_CLOSE (saves into --data only)
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()
    print("exit code", proc.returncode)
