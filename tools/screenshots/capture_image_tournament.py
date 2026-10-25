"""Screenshots of RulZ Image Tournament without touching the user's settings or saved tournament.

The real app keeps settings.ini and tournament.txt (the resumable tournament) in
%APPDATA%\\RulZ Image Tournament and rewrites both on exit, and declining its resume prompt deletes the saved
tournament. .NET's GetFolderPath ignores the APPDATA variable, so this script runs a *copy* of the app built
from the repo's sources outside the repo, with one line changed: Settings.Folder honours RULZ_TOURNAMENT_DATA.

    1. build (PowerShell, see README.md): copy src\\*.cs, app.manifest, assets\\app.ico, assets\\logo.png to a
       scratch folder, patch Support.cs, compile with the Framework csc.exe exactly like build.ps1.
    2. python capture_image_tournament.py --exe R:\\tt-build\\"RulZ Image Tournament.exe" --data R:\\tt-data ^
           --images R:\\Tournament --pair "Sea Glow 08.jpg" --out OUT_DIR
       (--images: a folder of generated images, e.g. make_demo_images.py R:\\Tournament --count 8 --seed 33
       --names titles; --pair replaces one of them with an original + "warm edit" pair for the first match)

Only generated demo images (in a temp folder) are passed in. No Keep button / arrow key is ever sent, so no
file is moved. The window is captured with PrintWindow (wincap.py); keys are posted with PostMessage.
Needs Pillow (use the Image Organizer venv).
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wincap  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--exe", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--images", required=True)
ap.add_argument("--pair", default="", help="make the first match this image vs. a colour-graded copy of it")
ap.add_argument("--out", required=True)
ap.add_argument("--size", default="1600x1000")
ap.add_argument("--zoom-at", type=float, nargs=2, default=(0.313, 0.508),
                help="where to wheel-zoom, as fractions of the compare view (default: the sun in the left image)")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

cfg = os.path.join(a.data, "RulZ Image Tournament")
os.makedirs(cfg, exist_ok=True)
assert not os.path.exists(os.path.join(cfg, "tournament.txt")), "use a fresh --data folder"
w, h = (int(v) for v in a.size.split("x"))
with open(os.path.join(cfg, "settings.ini"), "w", encoding="utf-8") as f:
    f.write(f"x=40\ny=16\nw={w}\nh={h}\nmaximized=False\nviewMode=SideBySide\n")

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
proc = subprocess.Popen([a.exe, *files], env=env)
print("started pid", proc.pid, flush=True)


def save(name):
    cw, ch, bgra = wincap.capture(main, "client")
    path = os.path.join(a.out, name)
    wincap.save_png(path, cw, ch, bgra)
    print("saved", path, cw, "x", ch, flush=True)


try:
    main = None
    t0 = time.time()
    while time.time() - t0 < 20 and main is None:
        time.sleep(0.3)
        wins = wincap.find_windows(proc.pid)
        if any(c == "#32770" for _, _, c in wins):
            raise SystemExit("a message box appeared: " + repr(wins))
        main = next((hw for hw, t, c in wins if t.startswith("RulZ Image Tournament")), None)
    if main is None:
        raise SystemExit("main window not found")
    time.sleep(3.0)  # images decode on background tasks
    if any(c == "#32770" for _, _, c in wincap.find_windows(proc.pid)):
        raise SystemExit("a message box appeared")
    save("01-side-by-side.png")

    # the compare view is the biggest child window; keys posted to it reach the form's ProcessCmdKey
    kids = wincap.child_windows(main)
    view = max(kids, key=lambda k: (k[2][2] - k[2][0]) * (k[2][3] - k[2][1]))[0]
    wincap.post_key(view, 0x56)  # V: side by side <-> split
    time.sleep(1.5)
    save("02-split-view.png")

    # back to side by side, then wheel-zoom towards the sun in the left image: both images zoom together
    wincap.post_key(view, 0x56)
    time.sleep(1.0)
    l, t, r, b = wincap.client_rect(view)
    x, y = l + int((r - l) * a.zoom_at[0]), t + int((b - t) * a.zoom_at[1])
    for _ in range(4):
        wincap.user32.PostMessageW(view, 0x020A, (120 << 16), ((y & 0xFFFF) << 16) | (x & 0xFFFF))  # WM_MOUSEWHEEL
        time.sleep(0.3)
    time.sleep(1.5)
    save("03-synced-zoom.png")
finally:
    if proc.poll() is None:
        for hw, _t, _c in wincap.find_windows(proc.pid):
            wincap.user32.PostMessageW(hw, 0x0010, 0, 0)  # WM_CLOSE (saves into --data only)
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()
    print("exit code", proc.returncode)
