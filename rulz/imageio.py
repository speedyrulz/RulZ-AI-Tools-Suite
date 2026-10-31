"""Reading, writing and converting images (Unicode-safe paths, 8/16-bit, alpha)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PySide6.QtGui import QImage

IMAGE_EXTS = {".jpg", ".jpeg", ".jpe", ".jfif", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def is_image(path: str | Path) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTS


def load(path: str | Path) -> np.ndarray:
    """Load as BGR or BGRA, keeping 16-bit depth, with camera rotation (EXIF) applied."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Can't read {Path(path).name}")
    if img.ndim == 3 and img.shape[2] == 4:
        return img
    # IMREAD_UNCHANGED ignores EXIF orientation; this flag set applies it.
    rotated = cv2.imdecode(data, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    return rotated if rotated is not None else img


def shrink(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    f = max_side / max(h, w)
    if f >= 1:
        return img
    return cv2.resize(img, (max(1, round(w * f)), max(1, round(h * f))), interpolation=cv2.INTER_AREA)


def to_8bit(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    if img.dtype == np.uint16:
        return (img >> 8).astype(np.uint8)
    return np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)


def to_qimage(img: np.ndarray) -> QImage:
    img = np.ascontiguousarray(to_8bit(img))
    h, w = img.shape[:2]
    if img.ndim == 2:
        q = QImage(img.data, w, h, img.strides[0], QImage.Format_Grayscale8)
    elif img.shape[2] == 4:
        rgba = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA))
        q = QImage(rgba.data, w, h, rgba.strides[0], QImage.Format_RGBA8888)
    else:
        rgb = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        q = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888)
    return q.copy()  # detach from the numpy buffer


def save(path: str | Path, img: np.ndarray) -> None:
    """Write in the format given by the file extension."""
    ext = Path(path).suffix.lower()
    params: list[int] = []
    if ext in (".jpg", ".jpeg", ".jpe", ".jfif"):
        ext = ".jpg"
        img = to_8bit(img)
        if img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        params = [cv2.IMWRITE_JPEG_QUALITY, 97]
    elif ext == ".webp":
        img = to_8bit(img)
        params = [cv2.IMWRITE_WEBP_QUALITY, 101]  # >100 = lossless
    elif ext == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
    elif ext == ".bmp":
        img = to_8bit(img)
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise ValueError(f"Can't encode {Path(path).name}")
    buf.tofile(str(path))
