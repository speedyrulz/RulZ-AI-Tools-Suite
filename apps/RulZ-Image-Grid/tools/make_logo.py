"""Builds the app's logo and icon from assets/logo-source.webp (needs Pillow).

Writes:
  assets/logo.png        transparent wordmark for dark backgrounds (black lettering turned white)
  assets/logo-light.png  transparent wordmark in the original colors, for light backgrounds
  assets/app.ico         the "R" mark on a black tile, used as the .exe and window icon
  assets/app.png         256px preview of the icon
"""
from pathlib import Path

from PIL import Image, ImageDraw

ASSETS = Path(__file__).resolve().parent.parent / "assets"
WHITE = (242, 242, 242)
TILE = (14, 14, 16, 255)
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


source = Image.open(ASSETS / "logo-source.webp").convert("RGB")
logo = unblend_from_white(source)
logo = logo.crop(logo.getbbox())
logo.save(ASSETS / "logo-light.png")
lighten_black(logo).save(ASSETS / "logo.png")

# The "R" mark is the part left of the first wide gap in the artwork.
mark = logo.crop((0, 0, 320, logo.height))
mark = lighten_black(mark.crop(mark.getbbox()))

SIZE = 256
tile = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
d = ImageDraw.Draw(tile)
d.rounded_rectangle((6, 6, SIZE - 7, SIZE - 7), radius=46, fill=TILE, outline=RED_BORDER, width=8)
scale = min(176 / mark.width, 176 / mark.height)
mark = mark.resize((round(mark.width * scale), round(mark.height * scale)), Image.LANCZOS)
tile.alpha_composite(mark, ((SIZE - mark.width) // 2, (SIZE - mark.height) // 2))

tile.save(ASSETS / "app.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
tile.save(ASSETS / "app.png")
print("Wrote logo.png, logo-light.png, app.ico, app.png")
