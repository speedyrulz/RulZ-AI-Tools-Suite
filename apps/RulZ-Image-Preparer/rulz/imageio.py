"""Reading, writing and converting images (Unicode-safe paths, 8/16-bit, alpha)."""
from __future__ import annotations

import struct
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtGui import QImage

IMAGE_EXTS = {".jpg", ".jpeg", ".jpe", ".jfif", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def is_image(path: str | Path) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTS


_REDUCED = {
    False: {2: cv2.IMREAD_REDUCED_COLOR_2, 4: cv2.IMREAD_REDUCED_COLOR_4, 8: cv2.IMREAD_REDUCED_COLOR_8},
    True: {2: cv2.IMREAD_REDUCED_GRAYSCALE_2, 4: cv2.IMREAD_REDUCED_GRAYSCALE_4, 8: cv2.IMREAD_REDUCED_GRAYSCALE_8},
}


def load(path: str | Path, max_side: int | None = None) -> np.ndarray:
    """Load as BGR or BGRA, keeping 16-bit depth, with camera rotation (EXIF) applied.

    The file is decoded once. max_side marks a thumbnail: a big JPEG is then decoded at 1/2, 1/4 or 1/8
    size (still at least max_side), which is several times faster; shrink() takes it the rest of the way.
    """
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        raise ValueError(f"Can't read {Path(path).name}: the file is empty")
    try:
        img = _decode(data, max_side)
    except cv2.error:  # OpenCV's own message is a wall of C++ details
        img = None
    if img is None:
        raise ValueError(f"Can't read {Path(path).name}")
    return img


def _decode(data: np.ndarray, max_side: int | None) -> np.ndarray | None:
    if data[:2].tobytes() == b"\xff\xd8":
        # JPEG has no alpha, so these flags lose nothing and apply the EXIF orientation.
        flags = cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH
        info = _jpeg_info(data) if max_side else None
        if info is not None:
            f = 8
            while f > 1 and max(info[0], info[1]) // f < max_side:
                f //= 2
            flags = _REDUCED[info[2] == 1].get(f, flags)
        return cv2.imdecode(data, flags)
    # IMREAD_UNCHANGED keeps alpha and 16-bit but ignores the EXIF orientation, which is applied below.
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    tiff = data[:2].tobytes() in (b"II", b"MM")
    if tiff and img.dtype == np.uint8 and img.ndim == 3 and img.shape[2] == 4 \
            and _ifd_value(data, 0, len(data), _EXTRA_SAMPLES) == _UNASSOCIATED_ALPHA:
        _unpremultiply(img)
    orientation = _orientation(data)
    if orientation > 1 and tiff:
        # OpenCV's TIFF reader may already have rotated it; these flags always do, exactly once.
        if img.ndim == 2 or img.shape[2] != 4:
            rotated = cv2.imdecode(data, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            img = rotated if rotated is not None else img
    elif orientation > 1:
        img = orient(img, orientation)
    return img


_EXTRA_SAMPLES, _UNASSOCIATED_ALPHA = 0x0152, 2


def _unpremultiply(img: np.ndarray) -> None:
    """Undo, in place, the alpha multiplication OpenCV's TIFF reader (libtiff's RGBA interface) applies to
    8-bit pictures stored with straight alpha, as Photoshop, GIMP and Pillow write them. Without this,
    see-through pixels come in darker and Save all writes them back that way."""
    a = img[:, :, 3:].astype(np.uint32)
    c = img[:, :, :3].astype(np.uint32)
    img[:, :, :3] = np.minimum(255, (c * 255 + a // 2) // np.maximum(a, 1))


def can_copy_unchanged(path: str | Path) -> bool:
    """True when load() gives this file's picture as stored, with nothing to bake in: no EXIF rotation and
    no CMYK colors. Such a file can be copied as is when nothing about it changes, instead of re-encoded."""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data[:2].tobytes() == b"\xff\xd8":
        info = _jpeg_info(data)
        return info is not None and info[2] in (1, 3) and _jpeg_orientation(data) == 1
    return _orientation(data) == 1


def orient(img: np.ndarray, orientation: int) -> np.ndarray:
    """Apply an EXIF orientation (1-8) the way OpenCV does."""
    if orientation >= 5:
        img = img.swapaxes(0, 1)
    flip = {2: 1, 3: -1, 4: 0, 6: 1, 7: -1, 8: 0}.get(orientation)
    if flip == 1:
        img = img[:, ::-1]
    elif flip == 0:
        img = img[::-1]
    elif flip == -1:
        img = img[::-1, ::-1]
    return np.ascontiguousarray(img)


def _ifd_value(buf, start: int, end: int, tag: int) -> int | None:
    """First value of a short-number tag in the first IFD of the TIFF structure (EXIF block or TIFF file)
    at buf[start:end], or None when it isn't there."""
    try:
        bo = {b"II": "<", b"MM": ">"}[bytes(buf[start:start + 2])]
        ifd = start + struct.unpack_from(bo + "I", buf, start + 4)[0]
        for i in range(struct.unpack_from(bo + "H", buf, ifd)[0]):
            e = ifd + 2 + 12 * i
            if e + 12 > end:
                break
            if struct.unpack_from(bo + "H", buf, e)[0] == tag:
                count = struct.unpack_from(bo + "I", buf, e + 4)[0]
                # up to two shorts sit in the entry itself; more are stored at an offset
                at = e + 8 if count <= 2 else start + struct.unpack_from(bo + "I", buf, e + 8)[0]
                return struct.unpack_from(bo + "H", buf, at)[0]
    except (KeyError, struct.error):
        pass
    return None


def _exif_orientation(buf, start: int, end: int) -> int:
    """Orientation tag of the TIFF structure (EXIF block or TIFF file) at buf[start:end], else 1."""
    v = _ifd_value(buf, start, end, 0x0112)
    return v if v is not None and 1 <= v <= 8 else 1


def _orientation(data: np.ndarray) -> int:
    """EXIF orientation of a PNG, WebP or TIFF file (1 when there is none)."""
    head, n = data[:12].tobytes(), len(data)
    try:
        if head[:8] == b"\x89PNG\r\n\x1a\n":
            pos = 8
            while pos + 8 <= n:
                size, kind = struct.unpack_from(">I4s", data, pos)
                if kind == b"eXIf":
                    return _exif_orientation(data, pos + 8, pos + 8 + size)
                if kind == b"IEND":
                    break
                pos += 12 + size
        elif head[:4] == b"RIFF" and head[8:12] == b"WEBP":
            pos = 12
            while pos + 8 <= n:
                kind, size = struct.unpack_from("<4sI", data, pos)
                if kind == b"EXIF":
                    s = pos + 8 + (6 if data[pos + 8:pos + 14].tobytes() == b"Exif\0\0" else 0)
                    return _exif_orientation(data, s, pos + 8 + size)
                pos += 8 + size + (size & 1)
        elif head[:2] in (b"II", b"MM"):
            return _exif_orientation(data, 0, n)
    except struct.error:
        pass
    return 1


def _jpeg_segments(data: np.ndarray):
    """(marker, payload start, payload end) of each segment of a JPEG's header, up to the start of the scan."""
    pos, n = 2, len(data)
    while pos + 4 <= n:
        if data[pos] != 0xFF:
            return
        marker = int(data[pos + 1])
        if marker == 0xFF:  # fill byte
            pos += 1
            continue
        if marker in (0x01, *range(0xD0, 0xD8)):  # markers without a length
            pos += 2
            continue
        size = struct.unpack_from(">H", data, pos + 2)[0]
        yield marker, pos + 4, pos + 2 + size
        if marker == 0xDA:
            return
        pos += 2 + size


def _jpeg_info(data: np.ndarray) -> tuple[int, int, int] | None:
    """(height, width, components) from a JPEG's frame header, or None."""
    try:
        for marker, start, _end in _jpeg_segments(data):
            if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                h, w = struct.unpack_from(">HH", data, start + 1)
                return (h, w, int(data[start + 5])) if h and w else None
            if marker == 0xDA:  # start of scan before any frame header
                return None
    except (struct.error, IndexError):
        pass
    return None


def _jpeg_orientation(data: np.ndarray) -> int:
    """EXIF orientation of a JPEG (1 when there is none); OpenCV applies it when decoding."""
    try:
        for marker, start, end in _jpeg_segments(data):
            if marker == 0xE1 and data[start:start + 6].tobytes() == b"Exif\0\0":
                return _exif_orientation(data, start + 6, end)
    except (struct.error, IndexError):
        pass
    return 1


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
        q = QImage(img.data, w, h, img.strides[0], QImage.Format_BGR888)
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
