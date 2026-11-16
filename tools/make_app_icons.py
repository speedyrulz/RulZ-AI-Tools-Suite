"""Draws one icon per suite app, all in the RulZ family, into a staging folder for review.

Usage:  python tools/make_app_icons.py [--out DIR]      draw the staging set
        python tools/make_app_icons.py --install [--only ID ...]
                                                  render straight into every app under apps/ (INSTALL
                                                  table below: same file names, sizes and ico frame sets
                                                  as the files they replace, Android mipmaps included)

Needs Pillow and Python Playwright with its Chromium (`python -m playwright install chromium`).
Glyphs are drawn from Lucide icon geometry (https://lucide.dev, ISC licence, copyright the
Lucide Contributors; the Feather parts are MIT, Cole Bemis). Each glyph is stroked in near-white
with one red accent part, rendered as SVG by Chromium at the exact size of each frame.

Writes, per app id:  <id>.png (256), <id>-512.png, <id>.ico (16 24 32 48 64 128 256 frames,
each rendered on its own) and one contact-sheet.png of every icon at 256/64/32/16 on dark and light.
Frames of 48px and below leave out the small R badge and use a larger glyph so they stay legible.
"""
import argparse
import sys
import tempfile
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    sys.exit("make_app_icons: Pillow is required (pip install pillow)")

ROOT = Path(__file__).resolve().parent.parent
MARK = ROOT / "src" / "renderer" / "src" / "assets" / "mark.png"
DEFAULT_OUT = Path(tempfile.gettempdir()) / "rulz-app-icons"

WHITE_HEX, RED_HEX = "#f2f2f2", "#ed000d"
TILE = (10, 10, 12, 255)
RED_BORDER = (237, 0, 13, 255)
ICO_SIZES = [16, 24, 32, 48, 64, 128, 256]
BADGE_MAX_SIZE = 64          # the R badge is drawn on frames larger than 48px

THIN = {"trainer-dumbbell", "llm-suite"}   # busy glyphs get a slightly thinner stroke
W, R = "w", "r"             # white / red part

# Lucide geometry (24x24 box). Each part is (svg element markup, colour).
def p(d, c):
    return (f'<path d="{d}"/>', c)

def circ(cx, cy, r, c, fill=False):
    return (f'<circle cx="{cx}" cy="{cy}" r="{r}"' + (' fill="currentColor"' if fill else "") + "/>", c)

def rect(x, y, w, h, rx, c, fill=False):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}"' + (' fill="currentColor"' if fill else "") + "/>", c)

def tf(parts, transform, sw):
    """Wraps parts in a transform and sets their stroke width (in the part's own units)."""
    return [(f'<g transform="{transform}">' + el.replace("/>", f' stroke-width="{sw}"/>', 1) + "</g>", c)
            for el, c in parts]

KNOCK = "#0a0a0c"   # tile colour, used to hide lines behind an overlapping part

SCAN = [p("M3 7V5a2 2 0 0 1 2-2h2", R), p("M17 3h2a2 2 0 0 1 2 2v2", R),
        p("M21 17v2a2 2 0 0 1-2 2h-2", R), p("M7 21H5a2 2 0 0 1-2-2v-2", R)]

# id -> (name, lucide icon, parts, glyph shift in 256px units (dx, dy))
PLATES = [
    p("M17.596 12.768a2 2 0 1 0 2.829-2.829l-1.768-1.767a2 2 0 0 0 2.828-2.829l-2.828-2.828a2 2 0 0 0-2.829 2.828l-1.767-1.768a2 2 0 1 0-2.829 2.829z", R),
    p("M5.343 21.485a2 2 0 1 0 2.829-2.828l1.767 1.768a2 2 0 1 0 2.829-2.829l-6.364-6.364a2 2 0 1 0-2.829 2.829l1.768 1.767a2 2 0 0 0-2.828 2.829z", R),
    p("m9.6 14.4 4.8-4.8", W)]

# the two candidates for the rulz-trainer icon (see trainer-options.png)
TRAINER_OPTIONS = {
    "trainer-dumbbell": ("RulZ AI Trainer (dumbbell, level)", "dumbbell",
                         tf(PLATES, "rotate(45 12 12) translate(0 0) translate(12 12) scale(0.95) translate(-12 -12)", 2.5), (0, 0)),
    "trainer-biceps": ("RulZ AI Trainer (biceps)", "biceps-flexed", [
        p("M12.409 13.017A5 5 0 0 1 22 15c0 3.866-4 7-9 7-4.077 0-8.153-.82-10.371-2.462-.426-.316-.631-.832-.62-1.362C2.118 12.723 2.627 2 10 2a3 3 0 0 1 3 3 2 2 0 0 1-2 2c-1.105 0-1.64-.444-2-1", W),
        p("M15 14a5 5 0 0 0-7.584 2", R), p("M9.964 6.825C8.019 7.977 9.5 13 8 15", R)], (0, 0)),
}
TRAINER_CHOICE = "trainer-dumbbell"

APPS = {
    "rulz-trainer": TRAINER_OPTIONS[TRAINER_CHOICE],
    "image-organizer": ("RulZ Image Organizer", "scan-face", SCAN + [
        p("M8 14s1.5 2 4 2 4-2 4-2", W), p("M9 9h.01", W), p("M15 9h.01", W)], (0, 0)),
    "image-tournament": ("RulZ Image Tournament", "trophy", [
        p("M6 9a6 6 0 0 0 12 0V3a1 1 0 0 0-1-1H7a1 1 0 0 0-1 1z", R),
        p("M6 9H4.5a1 1 0 0 1 0-5H6", W), p("M18 9h1.5a1 1 0 0 0 0-5H18", W),
        p("M10 14.66v1.626a2 2 0 0 1-.976 1.696A5 5 0 0 0 7 21.978", W),
        p("M14 14.66v1.626a2 2 0 0 0 .976 1.696A5 5 0 0 1 17 21.978", W), p("M4 22h16", W)], (0, 0)),
    "llm-suite": ("RulZ LLM Suite", "brain-circuit", [
        p("M12 5a3 3 0 1 0-5.997.125 4 4 0 0 0-2.526 5.77 4 4 0 0 0 .556 6.588A4 4 0 1 0 12 18Z", W),
        p("M9 13a4.5 4.5 0 0 0 3-4", W), p("M6.003 5.125A3 3 0 0 0 6.401 6.5", W),
        p("M3.477 10.896a4 4 0 0 1 .585-.396", W), p("M6 18a4 4 0 0 1-1.967-.516", W),
        p("M12 13h4", R), p("M12 18h6a2 2 0 0 1 2 2v1", R), p("M12 8h8", R), p("M16 8V5a2 2 0 0 1 2-2", R),
        circ(16, 13, 1.3, R, True), circ(18, 3, 1.3, R, True), circ(20, 21, 1.3, R, True), circ(20, 8, 1.3, R, True)], (0, 0)),
    "coding-agent": ("RulZ Coding Agent", "code-xml", [
        p("m18 16 4-4-4-4", W), p("m6 8-4 4 4 4", W), p("m14.5 4-5 16", R)], (0, 0)),
    "coding-agent-android": ("RulZ Coding Agent for Android", "smartphone", [
        rect(5, 2, 14, 20, 2, W),
        # a small code-xml centred in the phone, drawn at 0.5 scale around (12, 12)
        p("m15 14 2-2-2-2", W), p("m9 10-2 2 2 2", W), p("m13.25 8-2.5 8", R)], (0, 0)),
    "training-captioner": ("Rulz Training Captioner", "captions", [
        rect(3, 5, 18, 14, 2, W),
        p("M7 15h4", R), p("M15 15h2", R), p("M7 11h2", R), p("M13 11h4", R)], (0, 0)),
    "clip-extractor": ("RulZ Simple Video Clip Extractor", "scissors", [
        circ(6, 6, 3, W), circ(6, 18, 3, W),
        p("M8.12 8.12 12 12", R), p("M20 4 8.12 15.88", R), p("M14.8 14.8 20 20", R)], (0, 0)),
    "rulz-image-crawler": ("RulZ Image Crawler", "globe", tf([
        circ(12, 12, 10, W), p("M12 2a14.5 14.5 0 0 0 0 20 14.5 14.5 0 0 0 0-20", W), p("M2 12h20", W)],
        "translate(11.6 11.6) scale(0.88) translate(-13 -13)", 2.7) + tf([
        (f'<circle cx="11" cy="11" r="8" fill="{KNOCK}"/>', R), p("m21 21-4.34-4.34", R)],
        "translate(9.6 9.6) scale(0.55)", 4.4), (0, 0)),
    "image-preparer": ("RulZ Image Preparer", "wand-sparkles", [
        p("m21.64 3.64-1.28-1.28a1.21 1.21 0 0 0-1.72 0L2.36 18.64a1.21 1.21 0 0 0 0 1.72l1.28 1.28a1.2 1.2 0 0 0 1.72 0L21.64 5.36a1.2 1.2 0 0 0 0-1.72", W),
        p("m14 7 3 3", W), p("M5 6v4", R), p("M19 14v4", R), p("M10 2v2", R),
        p("M7 8H3", R), p("M21 16h-4", R), p("M11 3H9", R)], (0, 0)),
    "image-grid": ("RulZ Image Grid", "layout-grid", [
        rect(3, 3, 7, 7, 1, W), rect(3, 14, 7, 7, 1, W), rect(14, 14, 7, 7, 1, W),
        rect(14, 3, 7, 7, 1, R, True)], (0, 0)),
    "hf-manager": ("RulZ Huggingface Manager", "hard-drive-download", [
        rect(2, 14, 20, 8, 2, W), p("M6 18h.01", W), p("M10 18h.01", W),
        p("M12 2v8", R), p("m16 6-4 4-4-4", R)], (0, 0)),
}


ALL = dict(APPS, **TRAINER_OPTIONS)


def svg_for(parts, stroke):
    body = "".join(
        el.replace("/>", f' stroke="{RED_HEX if c == R else WHITE_HEX}" color="{RED_HEX if c == R else WHITE_HEX}"/>', 1)
        for el, c in parts)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
            f'stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round" '
            f'width="{{px}}" height="{{px}}">{body}</svg>')


class Renderer:
    def __init__(self):
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            sys.exit("make_app_icons: Python Playwright is required (pip install playwright)")
        self._pw = sync_playwright().start()
        try:
            self._browser = self._pw.chromium.launch()
        except Exception as e:
            self._pw.stop()
            sys.exit("make_app_icons: Chromium would not start (%s).\n"
                     "Install it with: python -m playwright install chromium" % str(e).splitlines()[0])
        self._page = self._browser.new_page(device_scale_factor=1)

    def glyph(self, parts, px, stroke):
        """Returns the glyph as an RGBA image px by px, transparent background."""
        html = ('<html><body style="margin:0;background:transparent">'
                + svg_for(parts, stroke).replace("{px}", str(px)) + "</body></html>")
        self._page.set_viewport_size({"width": px, "height": px})
        self._page.set_content(html)
        png = self._page.screenshot(omit_background=True, clip={"x": 0, "y": 0, "width": px, "height": px})
        import io
        return Image.open(io.BytesIO(png)).convert("RGBA")

    def close(self):
        self._browser.close()
        self._pw.stop()


def tile(size, round_=False):
    """The launcher tile at any size: black rounded square (or circle), red rim, 6/256 transparent inset."""
    ss = 4
    big = size * ss
    k = big / 256
    img = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    box = (6 * k, 6 * k, big - 7 * k, big - 7 * k)
    if round_:
        ImageDraw.Draw(img).ellipse(box, fill=TILE, outline=RED_BORDER, width=max(1, round(8 * k)))
    else:
        ImageDraw.Draw(img).rounded_rectangle(
            box, radius=46 * k, fill=TILE, outline=RED_BORDER, width=max(1, round(8 * k)))
    return img.resize((size, size), Image.LANCZOS)


def render_icon(r, app_id, size, mark, badge=None, round_=False):
    """badge=None draws the R badge above 48px; True/False forces it (False also gives the big small-frame glyph)."""
    _, _, parts, (dx, dy) = ALL[app_id]
    img = tile(size, round_)
    if badge is None:
        badge = size > 48
    small = not badge
    # small frames drop the badge, so the glyph can be larger and centred
    box = (166 if small else 140) / 256 * size
    stroke = (2.6 if size <= 32 else 2.4) - (0.3 if app_id in THIN else 0)
    if small:
        dx = dy = 0
    px = max(8, round(box))
    g = r.glyph(parts, px, stroke)
    cx, cy = size / 2 + dx / 256 * size, size / 2 + dy / 256 * size
    img.alpha_composite(g, (round(cx - px / 2), round(cy - px / 2)))
    if badge:
        bs = 44 / 256 * size
        sc = bs / max(mark.width, mark.height)
        m = mark.resize((max(1, round(mark.width * sc)), max(1, round(mark.height * sc))), Image.LANCZOS)
        off = (50 if round_ else 20) / 256 * size      # a round tile needs the badge further in
        right, bottom = size - off, size - off
        img.alpha_composite(m, (round(right - m.width), round(bottom - m.height)))
    return img


FG_SPAN = 0.60       # glyph + badge fill 60% of an adaptive foreground; the safe zone is 66/108 = 61%


def render_foreground(r, app_id, size, mark, badge=True):
    """Adaptive-icon foreground: glyph (and the R badge) on transparent, centred inside the 66/108 safe zone.
    The tile and rim come from the background layer, so they are not drawn here."""
    _, _, parts, _ = ALL[app_id]
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    ext = 178 if badge else 140                    # extent in 256px design units (glyph 58..198, badge to 236)
    u = FG_SPAN * size / ext
    org = (size - ext * u) / 2
    px = max(8, round(140 * u))
    img.alpha_composite(r.glyph(parts, px, 2.4 - (0.3 if app_id in THIN else 0)), (round(org), round(org)))
    if badge:
        bs = 44 * u
        sc = bs / max(mark.width, mark.height)
        m = mark.resize((max(1, round(mark.width * sc)), max(1, round(mark.height * sc))), Image.LANCZOS)
        img.alpha_composite(m, (round(org + ext * u - m.width), round(org + ext * u - m.height)))
    return img


APPS_DIR = ROOT / "apps"
ICO9 = [16, 20, 24, 32, 40, 48, 64, 128, 256]
RES = "rulz-coding-agent-android/android/app/src/main/res"
DENSITY = {"mdpi": 48, "hdpi": 72, "xhdpi": 96, "xxhdpi": 144, "xxxhdpi": 192}   # launcher px; foreground = x 2.25

# (app id, path under apps/, kind, size or frame list); kind: png | ico | favicon (no badge) | round | fg
INSTALL = [
    ("rulz-trainer", "RulzTrainer/assets/icon.png", "png", 256),
    ("image-organizer", "RulZ-Image-Organizer/assets/icon.png", "png", 256),
    ("image-organizer", "RulZ-Image-Organizer/assets/icon.ico", "ico", ICO9),
    ("image-tournament", "RulZ-Image-Tournament/assets/app.png", "png", 256),
    ("image-tournament", "RulZ-Image-Tournament/assets/app.ico", "ico", ICO_SIZES),
    ("llm-suite", "rulz-llm-suite/build/icon.png", "png", 1024),
    ("llm-suite", "rulz-llm-suite/build/icon.ico", "ico", ICO_SIZES),
    ("coding-agent", "rulz-coding-agent/build/icon.png", "png", 1024),
    ("coding-agent", "rulz-coding-agent/build/icon.ico", "ico", ICO_SIZES),
    ("training-captioner", "Rulz-Training-Captioner/assets/icon.png", "png", 350),
    ("training-captioner", "Rulz-Training-Captioner/assets/icon.ico", "ico", ICO_SIZES),
    ("clip-extractor", "rulz-simple-video-clip-extractor/simplecut/assets/icon256.png", "png", 256),
    ("clip-extractor", "rulz-simple-video-clip-extractor/simplecut/assets/icon.ico", "ico", ICO_SIZES),
    ("rulz-image-crawler", "rulz-image-crawler/assets/icon.png", "png", 350),
    ("rulz-image-crawler", "rulz-image-crawler/assets/icon.ico", "ico", ICO_SIZES),
    ("rulz-image-crawler", "rulz-image-crawler/crawler_app/static/icon.png", "favicon", 64),
    ("image-preparer", "RulZ-Image-Preparer/assets/icon.png", "png", 256),
    ("image-preparer", "RulZ-Image-Preparer/assets/icon.ico", "ico", ICO9),
    ("image-grid", "RulZ-Image-Grid/assets/app.png", "png", 256),
    ("image-grid", "RulZ-Image-Grid/assets/app.ico", "ico", ICO_SIZES),
    ("hf-manager", "RulZ-Huggingface-Manager/assets/icon.png", "png", 256),
    ("hf-manager", "RulZ-Huggingface-Manager/assets/icon.ico", "ico", ICO_SIZES),
    ("hf-manager", "RulZ-Huggingface-Manager/rulz_hf_manager/static/icon.png", "favicon", 64),
] + [("coding-agent-android", f"{RES}/mipmap-{d}/ic_launcher.png", "png", px) for d, px in DENSITY.items()] \
  + [("coding-agent-android", f"{RES}/mipmap-{d}/ic_launcher_round.png", "round", px) for d, px in DENSITY.items()] \
  + [("coding-agent-android", f"{RES}/mipmap-{d}/ic_launcher_foreground.png", "fg", round(px * 2.25))
     for d, px in DENSITY.items()]


def install(only=None):
    mark = Image.open(MARK).convert("RGBA")
    r = Renderer()
    n = 0
    try:
        for app_id, rel, kind, spec in INSTALL:
            if only and app_id not in only:
                continue
            dest = APPS_DIR / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            if kind == "png":
                render_icon(r, app_id, spec, mark).save(dest)
            elif kind == "favicon":
                render_icon(r, app_id, spec, mark, badge=False).save(dest)
            elif kind == "round":
                render_icon(r, app_id, spec, mark, round_=True).save(dest)
            elif kind == "fg":
                render_foreground(r, app_id, spec, mark).save(dest)
            else:
                frames = [render_icon(r, app_id, s, mark) for s in spec]
                frames[-1].save(dest, format="ICO", sizes=[(s, s) for s in spec], append_images=frames[:-1])
            n += 1
            print("wrote", rel)
    finally:
        r.close()
    print(f"Installed {n} files under {APPS_DIR}")


def contact_sheet(cache, out, ids=None, name="contact-sheet.png"):
    ids = ids or list(APPS)
    sizes = [256, 64, 32, 16]
    pad, label_h = 16, 22
    col_w = sum(sizes) + pad * (len(sizes) + 1)
    row_h = 256 + label_h + pad
    cols = 2  # dark | light
    sheet = Image.new("RGB", (col_w * cols, row_h * len(ids) + pad), (128, 128, 128))
    d = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
    for ci, bg in enumerate([(5, 5, 5), (255, 255, 255)]):
        d.rectangle((ci * col_w, 0, (ci + 1) * col_w, sheet.height), fill=bg)
        for ri, app_id in enumerate(ids):
            y0 = pad + ri * row_h
            d.text((ci * col_w + pad, y0), app_id, fill=(200, 200, 200) if ci == 0 else (30, 30, 30), font=font)
            x = ci * col_w + pad
            for s in sizes:
                ic = cache[(app_id, s)]
                sheet.paste(ic, (x, y0 + label_h + (256 - s)), ic)
                x += s + pad
    sheet.save(out / name)


def main():
    ap = argparse.ArgumentParser(description="Draw one icon per suite app into a staging folder.")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="staging folder (default: %(default)s)")
    ap.add_argument("--install", action="store_true",
                    help="render into the apps' own icon files (see INSTALL) instead of the staging folder")
    ap.add_argument("--only", nargs="+", metavar="ID", help="with --install: only these app ids")
    args = ap.parse_args()
    out = args.out
    if not MARK.exists():
        sys.exit(f"make_app_icons: missing {MARK} (run tools/make_logo.py)")
    if args.install:
        install(args.only)
        return
    out.mkdir(parents=True, exist_ok=True)
    mark = Image.open(MARK).convert("RGBA")
    r = Renderer()
    cache = {}
    try:
        for app_id in ALL:
            for s in sorted(set(ICO_SIZES + [64, 32, 16, 512])):
                cache[(app_id, s)] = render_icon(r, app_id, s, mark)
            if app_id in TRAINER_OPTIONS:
                continue
            cache[(app_id, 256)].save(out / f"{app_id}.png")
            cache[(app_id, 512)].save(out / f"{app_id}-512.png")
            frames = [cache[(app_id, s)] for s in ICO_SIZES]
            frames[-1].save(out / f"{app_id}.ico", format="ICO", sizes=[(s, s) for s in ICO_SIZES],
                            append_images=frames[:-1])
    finally:
        r.close()
    contact_sheet(cache, out)
    contact_sheet(cache, out, list(TRAINER_OPTIONS), "trainer-options.png")
    print(f"Wrote {len(APPS)} icons and contact-sheet.png to {out}")


if __name__ == "__main__":
    main()
