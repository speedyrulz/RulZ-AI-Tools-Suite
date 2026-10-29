"""Image pipeline: decode -> skip tiny -> resize (never upscale) -> re-encode -> SHA-256.

The hash of the re-encoded bytes is the image's cache identity, so the same image at the
same MAX_EDGE is never re-run.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
from dataclasses import dataclass

import httpx
from PIL import Image, ImageOps, UnidentifiedImageError

try:  # optional: lets Pillow open HEIC/AVIF if the extension sends them
    import pillow_heif

    pillow_heif.register_heif_opener()
except Exception:  # pragma: no cover
    pass

BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36"
)


class ImageError(Exception):
    """The image could not be fetched or decoded. Fails only that image."""


class ImageTooSmall(Exception):
    def __init__(self, width: int, height: int):
        super().__init__(f"short edge {min(width, height)} px is below the minimum")
        self.width, self.height = width, height


@dataclass(frozen=True)
class PreparedImage:
    data: bytes
    mime: str
    sha256: str
    width: int
    height: int

    @property
    def data_url(self) -> str:
        return f"data:{self.mime};base64,{base64.b64encode(self.data).decode('ascii')}"


def decode_b64(data_b64: str) -> bytes:
    if data_b64.startswith("data:"):
        _, _, data_b64 = data_b64.partition(",")
    try:
        return base64.b64decode(data_b64, validate=False)
    except (binascii.Error, ValueError) as e:
        raise ImageError(f"invalid base64: {e}") from e


async def fetch_url(client: httpx.AsyncClient, url: str, timeout_s: float, max_bytes: int) -> bytes:
    if not url.lower().startswith(("http://", "https://")):
        raise ImageError("only http(s) URLs are supported")
    try:
        resp = await client.get(
            url,
            timeout=timeout_s,
            follow_redirects=True,
            headers={"User-Agent": BROWSER_UA, "Accept": "image/avif,image/webp,image/*,*/*;q=0.8"},
        )
    except httpx.HTTPError as e:
        raise ImageError(f"fetch failed: {type(e).__name__}: {e}") from e
    if resp.status_code >= 400:
        raise ImageError(f"fetch failed: HTTP {resp.status_code}")
    if len(resp.content) > max_bytes:
        raise ImageError("image too large")
    return resp.content


def _has_transparency(img: Image.Image) -> bool:
    if img.mode in ("RGBA", "LA", "PA"):
        alpha = img.getchannel("A")
        return alpha.getextrema()[0] < 255
    if img.mode == "P" and "transparency" in img.info:
        return _has_transparency(img.convert("RGBA"))
    return False


def prepare_image(raw: bytes, max_edge: int, min_short_edge: int, jpeg_quality: int = 90) -> PreparedImage:
    """CPU-bound; call via asyncio.to_thread."""
    try:
        img = Image.open(io.BytesIO(raw))
        img.seek(0)  # first frame of animated GIF/WebP
        img.load()
    except (UnidentifiedImageError, OSError, ValueError) as e:
        raise ImageError(f"cannot decode image: {e}") from e

    img = ImageOps.exif_transpose(img)
    w, h = img.size
    if min(w, h) < min_short_edge:
        raise ImageTooSmall(w, h)

    scale = max_edge / max(w, h)
    if scale < 1:
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    if _has_transparency(img):
        img = img.convert("RGBA")
        img.save(buf, format="PNG", optimize=False)
        mime = "image/png"
    else:
        img = img.convert("RGB")
        img.save(buf, format="JPEG", quality=jpeg_quality)
        mime = "image/jpeg"
    data = buf.getvalue()
    return PreparedImage(data=data, mime=mime, sha256=hashlib.sha256(data).hexdigest(), width=img.width, height=img.height)
