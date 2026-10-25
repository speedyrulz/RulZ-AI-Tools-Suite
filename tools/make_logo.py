"""Builds the launcher's logo and icon from assets/logo-source.webp (needs Pillow).

Writes:
  src/renderer/src/assets/logo.png        transparent wordmark for the dark UI (black lettering turned white)
  src/renderer/src/assets/logo-light.png  the original colors (black lettering), for the light UI
  src/renderer/src/assets/mark.png        the "R" mark alone, white and red
  src/renderer/src/assets/mark-light.png  the "R" mark in its original colors
  build/icon.ico, build/icon.png     the "R" mark on a black tile with a red border (the .exe and window icon)
"""
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
UI_ASSETS = ROOT / "src" / "renderer" / "src" / "assets"
BUILD = ROOT / "build"
WHITE = (242, 242, 242)
TILE = (10, 10, 12, 255)
RED_BORDER = (237, 0, 13, 255)


def unblend_from_white(img):
    """Turns a logo on a white background into a transparent one, keeping edge antialiasing."""
    out = Image.new("RGBA", img.size)
    src, dst = img.load(), out.load()
    for y in range(img.height):
        for x in range(img.width):
            r, g, b = src[x, y]
            a = 255 - min(r, g, b)
            if a < 8:
                dst[x, y] = (0, 0, 0, 0)
                continue
            un = lambda c: max(0, min(255, round((c - (255 - a)) * 255 / a)))
            dst[x, y] = (un(r), un(g), un(b), a)
    return out


def lighten_black(img):
    """Recolors the black parts white, leaving the red untouched."""
    out = img.copy()
    px = out.load()
    for y in range(out.height):
        for x in range(out.width):
            r, g, b, a = px[x, y]
            if a and r - max(g, b) < 80:
                px[x, y] = WHITE + (a,)
    return out


source = Image.open(ROOT / "assets" / "logo-source.webp").convert("RGB")
logo = unblend_from_white(source)
logo = logo.crop(logo.getbbox())
UI_ASSETS.mkdir(parents=True, exist_ok=True)
lighten_black(logo).save(UI_ASSETS / "logo.png")
logo.save(UI_ASSETS / "logo-light.png")

# The "R" mark is the part left of the first wide gap in the artwork.
mark = logo.crop((0, 0, 320, logo.height))
mark = mark.crop(mark.getbbox())
mark.save(UI_ASSETS / "mark-light.png")
mark = lighten_black(mark)
mark.save(UI_ASSETS / "mark.png")

SIZE = 256
tile = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
d = ImageDraw.Draw(tile)
d.rounded_rectangle((6, 6, SIZE - 7, SIZE - 7), radius=46, fill=TILE, outline=RED_BORDER, width=8)
scale = min(176 / mark.width, 176 / mark.height)
mark = mark.resize((round(mark.width * scale), round(mark.height * scale)), Image.LANCZOS)
tile.alpha_composite(mark, ((SIZE - mark.width) // 2, (SIZE - mark.height) // 2))

BUILD.mkdir(exist_ok=True)
tile.save(BUILD / "icon.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
tile.save(BUILD / "icon.png")
print("Wrote logo.png, logo-light.png, mark.png, mark-light.png, build/icon.ico, build/icon.png")
