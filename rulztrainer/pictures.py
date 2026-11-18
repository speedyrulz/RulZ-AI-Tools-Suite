"""Pictures as training and scoring read them: transparency laid on white.

PIL's convert("RGB") drops the alpha channel and keeps whatever colour the transparent pixels happen to store (often
black, sometimes left-over junk from an editor), so a cut-out PNG trained and was scored on a background nobody sees.
to_rgb() lays a picture with transparency on white instead, as a viewer or a captioner shows it (after Fizgig 9.0.3,
which does the same for every model but Qwen-Image 2.1's alpha training).

It also turns a picture into the sRGB the models train on: a picture with an embedded colour profile that is not sRGB
(Display P3 from a phone, Adobe RGB, ProPhoto, a CMYK print profile) is converted with littleCMS, so it trains as
bright as it looks, and a 16-bit or 32-bit grey picture is scaled to 8 bits over its range instead of clipped to
white."""
from __future__ import annotations

import io
import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms

log = logging.getLogger("rulztrainer")

ALPHA_MODES = ("RGBA", "LA", "PA", "RGBa", "La")
WHITE = (255, 255, 255, 255)


def has_alpha(im: Image.Image) -> bool:
    """True when the picture can hold transparent pixels: an alpha channel, or a transparent colour (a PNG's tRNS on a
    palette, grey or RGB picture)."""
    return im.mode in ALPHA_MODES or "transparency" in im.info


PROFILE_MODES = {"RGB": "RGB ", "RGBA": "RGB ", "CMYK": "CMYK"}  # modes converted, and the colour space their profile has
GREY_16_MODES = ("I;16", "I;16L", "I;16B", "I;16N", "I")  # grey pictures deeper than 8 bits
_GREY_16_TYPES = {"I;16": "<u2", "I;16L": "<u2", "I;16B": ">u2", "I;16N": "=u2", "I": "=i4"}


@lru_cache(maxsize=16)
def _srgb_transform(icc: bytes, mode: str):
    """The littleCMS transform from an embedded profile to sRGB (RGB/RGBA/CMYK in, RGB/RGBA out); None for a picture
    that needs none: an sRGB profile, one of another colour space than the picture, or one that cannot be read."""
    try:
        src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
        if src.profile.xcolor_space != PROFILE_MODES[mode]:
            return None
        desc = (src.profile.profile_description or "").lower()
        if "srgb" in desc and not any(w in desc for w in ("linear", "scrgb")):  # (linear sRGB is not sRGB)
            return None
        out_mode = "RGBA" if mode == "RGBA" else "RGB"
        return ImageCms.buildTransform(src, ImageCms.createProfile("sRGB"), mode, out_mode,
                                       renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
                                       flags=ImageCms.Flags.BLACKPOINTCOMPENSATION)
    except Exception as e:  # noqa: BLE001
        log.debug("Colour profile not used: %s", e)
        return None


def _profile_transform(im: Image.Image):
    icc = im.info.get("icc_profile")
    if not icc or im.mode not in PROFILE_MODES:
        return None
    return _srgb_transform(bytes(icc), im.mode)


def colour_tag(im: Image.Image) -> str:
    """What turns this picture's pixels into the ones training sees, for its latent cache name: "" for a picture read
    as it always was (no profile or an sRGB one, ordinary grey), "|icc" for one converted to sRGB, "|grey16" for a grey
    one deeper than 8 bits. Reads the header only."""
    if im.mode in GREY_16_MODES:
        return "|grey16"
    return "|icc" if _profile_transform(im) is not None else ""


def file_colour_tag(path: str | Path) -> str:
    """colour_tag() of a picture file, from its header (nothing is decoded)."""
    with Image.open(path) as im:
        return colour_tag(im)


def _grey_to_8bit(im: Image.Image) -> Image.Image:
    """An 8-bit grey copy of a 16-bit grey picture (v / 257, rounded) or a 32-bit integer one (by its 16-bit range when
    it fits, else by the 31 bits a positive value has)."""
    v = np.frombuffer(im.tobytes(), dtype=_GREY_16_TYPES[im.mode]).reshape(im.size[1], im.size[0]).astype(np.int64)
    v = np.clip(v, 0, None)
    if im.mode == "I" and v.size and int(v.max()) > 65535:
        out = np.rint(v * (255.0 / 2147483647.0))
    else:
        out = np.rint(v / 257.0)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "L")


def _to_srgb(im: Image.Image) -> Image.Image:
    """The picture in sRGB when it carries another profile (its old profile dropped), else the picture itself; a
    profile that cannot be applied leaves it as it was."""
    tf = _profile_transform(im)
    if tf is None:
        return im
    try:
        out = ImageCms.applyTransform(im, tf)
    except Exception as e:  # noqa: BLE001
        log.debug("Colour profile not applied: %s", e)
        return im
    out.info = {k: v for k, v in im.info.items() if k != "icc_profile"}
    return out


def to_rgb(im: Image.Image) -> Image.Image:
    """An RGB copy of the picture in sRGB, any transparency laid on white (convert("RGB") for every other picture)."""
    if im.mode in GREY_16_MODES:
        im = _grey_to_8bit(im)
    else:
        im = _to_srgb(im)
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
