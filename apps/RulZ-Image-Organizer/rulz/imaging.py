"""Image decoding, EXIF dates, thumbnails and face crops."""
from __future__ import annotations

import hashlib
import os
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from . import config

try:  # HEIC/HEIF support (iPhone photos)
    import pillow_heif

    pillow_heif.register_heif_opener()
except Exception:  # pragma: no cover - optional
    pass

Image.MAX_IMAGE_PIXELS = 300_000_000

_EXIF_IFD = 0x8769
_DATETIME_ORIGINAL = 36867
_DATETIME_DIGITIZED = 36868
_DATETIME = 306
_ORIENTATION = 274


class LoadedImage:
    """An image decoded (and EXIF-rotated) at reduced size."""

    def __init__(self, image: Image.Image, full_size: tuple[int, int], taken_at: float | None):
        self.image = image
        self.full_size = full_size  # oriented size of the original file
        self.taken_at = taken_at

    def bgr(self) -> np.ndarray:
        return np.ascontiguousarray(np.asarray(self.image)[:, :, ::-1])


def load(path: str, max_side: int) -> LoadedImage:
    with Image.open(path) as im:
        w, h = im.size
        exif = im.getexif()
        orientation = exif.get(_ORIENTATION, 1)
        if orientation in (5, 6, 7, 8):
            w, h = h, w
        taken = _exif_date(exif)
        if getattr(im, "n_frames", 1) > 1:
            im.seek(0)
        if im.format in ("JPEG", "MPO"):  # MPO: JPEG with a second (3D / depth) frame, common from phones
            # Decode at a reduced scale – much faster for big camera files.
            im.draft("RGB", (max_side, max_side))
        im.load()
        im = ImageOps.exif_transpose(im)
        if im.mode != "RGB":
            if im.mode in _HIGH_DEPTH:
                im = _to_8bit(im)
            elif im.mode in ("RGBA", "LA", "P", "PA"):
                bg = Image.new("RGB", im.size, (255, 255, 255))
                rgba = im.convert("RGBA")
                bg.paste(rgba, mask=rgba.getchannel("A"))
                im = bg
            else:
                im = im.convert("RGB")
        if max(im.size) > max_side:
            im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        return LoadedImage(im, (w, h), taken)


_HIGH_DEPTH = ("I;16", "I;16L", "I;16B", "I;16N", "I", "F")


def _to_8bit(im: Image.Image) -> Image.Image:
    """16-bit / 32-bit int / float grayscale to RGB. A plain convert() clips everything above 255 to white."""
    a = np.asarray(im, dtype=np.float32)
    hi = float(a.max()) if a.size else 0.0
    if im.mode == "F" and hi <= 1.0:
        scale = 255.0           # float images are usually 0..1
    elif hi <= 255:
        scale = 1.0             # 8-bit values stored in a wider mode
    elif hi <= 65535 and im.mode != "F":
        scale = 255.0 / 65535   # 16-bit range
    else:
        scale = 255.0 / hi
    gray = np.clip(a * scale + 0.5, 0, 255).astype(np.uint8)
    return Image.fromarray(gray, "L").convert("RGB")


def _exif_date(exif) -> float | None:
    values = []
    try:
        sub = exif.get_ifd(_EXIF_IFD)
        values += [sub.get(_DATETIME_ORIGINAL), sub.get(_DATETIME_DIGITIZED)]
    except Exception:
        pass
    values.append(exif.get(_DATETIME))
    for v in values:
        if not v:
            continue
        try:
            s = str(v).strip().replace("\x00", "")[:19]
            dt = datetime.strptime(s, "%Y:%m:%d %H:%M:%S")
            if dt.year > 1900:
                return dt.timestamp()
        except (ValueError, OverflowError, OSError):
            continue
    return None


def save_thumbnail(img: Image.Image, dest: Path) -> None:
    w, h = img.size
    short = min(w, h)
    scale = min(1.0, config.THUMB_SHORT_SIDE / short)
    # Keep panoramas from becoming huge.
    scale = min(scale, 1440 / max(w, h)) if max(w, h) * scale > 1440 else scale
    thumb = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.LANCZOS) \
        if scale < 1 else img
    dest.parent.mkdir(parents=True, exist_ok=True)
    thumb.save(dest, "JPEG", quality=85, optimize=True)


def save_face_crop(img: Image.Image, box_norm, dest: Path) -> None:
    """Save a square crop around a face (box normalized to ``img``)."""
    save_crop(face_crop(img, box_norm), dest)


def face_crop(img: Image.Image, box_norm) -> Image.Image:
    """Square crop around a face (box normalized to ``img``), FACE_CROP_SIZE pixels."""
    W, H = img.size
    x, y, w, h = box_norm[0] * W, box_norm[1] * H, box_norm[2] * W, box_norm[3] * H
    cx, cy = x + w / 2, y + h / 2
    side = max(w, h) * 1.6
    left, top = cx - side / 2, cy - side / 2
    crop = img.crop((round(left), round(top), round(left + side), round(top + side)))
    # Areas outside the image are filled black by crop(); that's fine for edge faces.
    return crop.resize((config.FACE_CROP_SIZE, config.FACE_CROP_SIZE), Image.Resampling.LANCZOS)


def save_crop(crop: Image.Image, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    crop.save(dest, "JPEG", quality=88)


def fingerprint(path: str, size: int) -> str:
    """Cheap content fingerprint used to recognise moved/renamed files."""
    h = hashlib.blake2b(digest_size=12)
    h.update(str(size).encode())
    try:
        with open(path, "rb") as f:
            h.update(f.read(65536))
            if size > 131072:
                f.seek(-65536, os.SEEK_END)
                h.update(f.read(65536))
    except OSError:
        return ""
    return h.hexdigest()
