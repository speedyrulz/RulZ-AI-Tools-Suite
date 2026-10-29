"""Fetch image bytes through the run's browser context, so requests carry the site's cookies and referer."""

from __future__ import annotations

import base64
import binascii
import hashlib
import asyncio
import dataclasses
import io
import time
from dataclasses import dataclass
from typing import Callable
from urllib.parse import urlsplit

from PIL import Image

from playwright.async_api import BrowserContext
from playwright.async_api import Error as PlaywrightError

from .crawler import ImageCandidate
from .urls import sniff_image

try:
    import pillow_heif  # noqa: F401  (the engine registers it with Pillow if present)

    HEIF_OK = True
except Exception:  # pragma: no cover
    HEIF_OK = False


class FetchError(Exception):
    """The image could not be fetched or is not a type the engine can read. Fails only that image."""


class RateLimited(FetchError):
    def __init__(self, retry_after_s: float):
        super().__init__(f"fetch failed: HTTP 429 (rate limited for {retry_after_s:.0f}s)")
        self.retry_after_s = retry_after_s


class HostGate:
    """Per-host politeness for image downloads: at most `per_host` requests at once, and after a 429
    that host's guessed originals are skipped (the page's own copy is fetched instead) until its
    Retry-After has passed, instead of hammering a site that asked us to slow down."""

    def __init__(self, per_host: int = 2, on_limited: Callable[[str, float], None] | None = None):
        self.per_host = per_host
        self.on_limited = on_limited
        self._slots: dict[str, asyncio.Semaphore] = {}
        self._cool_until: dict[str, float] = {}

    def slot(self, host: str) -> asyncio.Semaphore:
        if host not in self._slots:
            self._slots[host] = asyncio.Semaphore(self.per_host)
        return self._slots[host]

    def cooling(self, host: str) -> bool:
        return self._cool_until.get(host, 0.0) > time.monotonic()

    def limited(self, host: str, retry_after_s: float) -> None:
        first = not self.cooling(host)
        self._cool_until[host] = time.monotonic() + retry_after_s
        if first and self.on_limited:
            self.on_limited(host, retry_after_s)


def _retry_after(value: str | None) -> float:
    try:
        return min(900.0, max(5.0, float(value)))
    except (TypeError, ValueError):
        return 60.0


@dataclass(frozen=True)
class FetchedImage:
    url: str
    page: str
    data: bytes  # original bytes, exactly as served
    ext: str
    mime: str
    sha256: str
    width: int | None = None
    height: int | None = None
    is_original: bool = False  # a guessed full-size original was fetched, not the displayed URL

    @property
    def megapixels(self) -> float | None:
        return self.width * self.height / 1e6 if self.width and self.height else None

    @property
    def b64(self) -> str:
        return base64.b64encode(self.data).decode("ascii")


def _decode_data_url(url: str) -> bytes:
    header, _, payload = url.partition(",")
    if ";base64" not in header.lower():
        raise FetchError("inline image is not base64")
    try:
        return base64.b64decode(payload, validate=False)
    except (binascii.Error, ValueError) as e:
        raise FetchError(f"invalid inline image: {e}") from e


def _dimensions(data: bytes) -> tuple[int | None, int | None]:
    """Width and height from the file header (no full decode)."""
    try:
        with Image.open(io.BytesIO(data)) as im:
            return im.size
    except Exception:  # noqa: BLE001 - unknown size just means no megapixel check
        return None, None


def check_image_bytes(url: str, page: str, data: bytes, max_bytes: int) -> FetchedImage:
    if len(data) > max_bytes:
        raise FetchError("image too large")
    kind = sniff_image(data, HEIF_OK)
    if kind is None:
        raise FetchError("unsupported image type (SVG, ICO or not an image)")
    ext, mime = kind
    w, h = _dimensions(data)
    return FetchedImage(url, page, data, ext, mime, hashlib.sha256(data).hexdigest(), w, h)


async def fetch_image(
    context: BrowserContext, cand: ImageCandidate, timeout_s: float, max_bytes: int, gate: HostGate | None = None
) -> FetchedImage:
    """Tries cand.url, then each fallback; the error reported is the last URL's (the displayed image)."""
    gate = gate or HostGate()
    err: FetchError | None = None
    for n, url in enumerate(cand.urls):
        guess = n < len(cand.fallbacks)  # anything but the last URL is a guessed original
        host = urlsplit(url).hostname or ""
        if guess and gate.cooling(host):
            continue
        try:
            async with gate.slot(host):
                img = await _fetch_one(context, url, cand.page, timeout_s, max_bytes)
        except RateLimited as e:
            gate.limited(host, e.retry_after_s)
            err = e
            continue
        except FetchError as e:
            err = e
            continue
        return dataclasses.replace(img, is_original=True) if guess else img
    assert err is not None
    raise err


async def _fetch_one(context: BrowserContext, url: str, page: str, timeout_s: float, max_bytes: int) -> FetchedImage:
    if url[:5].lower() == "data:":
        return check_image_bytes(url, page, _decode_data_url(url), max_bytes)
    try:
        resp = await context.request.get(
            url,
            headers={"Referer": page, "Accept": "image/avif,image/webp,image/*,*/*;q=0.8"},
            timeout=timeout_s * 1000,
            fail_on_status_code=False,
        )
    except PlaywrightError as e:
        raise FetchError(f"fetch failed: {str(e).splitlines()[0]}") from e
    try:
        if resp.status == 429:
            raise RateLimited(_retry_after(resp.headers.get("retry-after")))
        if resp.status >= 400:
            raise FetchError(f"fetch failed: HTTP {resp.status}")
        length = resp.headers.get("content-length")
        if length and length.isdigit() and int(length) > max_bytes:
            raise FetchError("image too large")
        ctype = resp.headers.get("content-type", "").lower()
        if "svg" in ctype or "html" in ctype:
            raise FetchError(f"unsupported image type ({ctype.split(';')[0]})")
        try:
            data = await resp.body()
        except PlaywrightError as e:
            raise FetchError(f"fetch failed: {str(e).splitlines()[0]}") from e
        return check_image_bytes(url, page, data, max_bytes)
    finally:
        await resp.dispose()
