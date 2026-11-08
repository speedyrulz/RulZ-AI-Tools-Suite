"""Pictures as training and scoring read them: transparency laid on white.

PIL's convert("RGB") drops the alpha channel and keeps whatever colour the transparent pixels happen to store (often
black, sometimes left-over junk from an editor), so a cut-out PNG trained and was scored on a background nobody sees.
to_rgb() lays a picture with transparency on white instead, as a viewer or a captioner shows it (after Fizgig 9.0.3,
which does the same for every model but Qwen-Image 2.1's alpha training)."""
from __future__ import annotations

from pathlib import Path

from PIL import Image

ALPHA_MODES = ("RGBA", "LA", "PA", "RGBa", "La")
WHITE = (255, 255, 255, 255)


def has_alpha(im: Image.Image) -> bool:
    """True when the picture can hold transparent pixels: an alpha channel, or a transparent colour (a PNG's tRNS on a
    palette, grey or RGB picture)."""
    return im.mode in ALPHA_MODES or "transparency" in im.info


def to_rgb(im: Image.Image) -> Image.Image:
    """An RGB copy of the picture, any transparency laid on white (convert("RGB") for every other picture)."""
    if im.mode == "RGB" and "transparency" not in im.info:
        return im
    if not has_alpha(im):
        return im.convert("RGB")
    rgba = im.convert("RGBA")
    return Image.alpha_composite(Image.new("RGBA", rgba.size, WHITE), rgba).convert("RGB")


def file_has_alpha(path: str | Path) -> bool:
    """has_alpha() of a picture file, from its header (nothing is decoded)."""
    with Image.open(path) as im:
        return has_alpha(im)
