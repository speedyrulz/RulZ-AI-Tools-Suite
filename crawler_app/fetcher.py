"""Fetch image bytes through the run's browser context, so requests carry the site's cookies and referer."""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import dataclasses
import email.utils
import hashlib
import io
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import AsyncIterator, Callable
from urllib.parse import urlsplit

from PIL import Image, features

from playwright.async_api import BrowserContext
from playwright.async_api import Error as PlaywrightError

from .crawler import ImageCandidate
from .urls import sniff_image

try:
    import pillow_heif  # noqa: F401  (the engine registers it with Pillow if present)

    HEIF_OK = True
except Exception:  # pragma: no cover
    HEIF_OK = False


def _pillow_reads_avif() -> bool:
    # Pillow 11.2+ reads AVIF by itself; an older one only warns that the feature is unknown.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return bool(features.check("avif"))
        except Exception:  # noqa: BLE001 - pragma: no cover
            return False


AVIF_OK = _pillow_reads_avif()
# Only ask servers for AVIF when the engine can read it: CDNs that pick the format from this header
# would otherwise send every image in a format that is then thrown away.
ACCEPT = ("image/avif," if AVIF_OK else "") + "image/webp,image/*,*/*;q=0.8"


class FetchError(Exception):
    """The image could not be fetched or is not a type the engine can read. Fails only that image."""


class RateLimited(FetchError):
    def __init__(self, retry_after_s: float):
        super().__init__(f"fetch failed: HTTP 429 (rate limited for {retry_after_s:.0f}s)")
        self.retry_after_s = retry_after_s


class HostGate:
    """Per-host politeness for image downloads: at most `per_host` requests at once. After a 429 on a
    guessed original, that host's guesses are skipped (the page's own copy is fetched instead) until
    its Retry-After has passed. After a 429 on a page's own copy, every download from that host waits
    that long and the copy is tried once more, instead of hammering a site that asked us to slow down.
    `on_limited(host, seconds, held)` is told once per cool-down (`held`: downloads are waiting)."""

    def __init__(self, per_host: int = 2, on_limited: Callable[[str, float, bool], None] | None = None):
        self.per_host = per_host
        self.on_limited = on_limited
        self._slots: dict[str, asyncio.Semaphore] = {}
        self._cool_until: dict[str, float] = {}  # guesses are skipped until then
        self._hold_until: dict[str, float] = {}  # every download waits until then

    def slot(self, host: str) -> asyncio.Semaphore:
        if host not in self._slots:
            self._slots[host] = asyncio.Semaphore(self.per_host)
        return self._slots[host]

    def cooling(self, host: str) -> bool:
        return self._cool_until.get(host, 0.0) > time.monotonic()

    def holding(self, host: str) -> bool:
        return self._hold_until.get(host, 0.0) > time.monotonic()

    def limited(self, host: str, retry_after_s: float, hold: bool = False) -> None:
        first = not (self.holding(host) if hold else self.cooling(host))
        until = time.monotonic() + retry_after_s
        self._cool_until[host] = max(self._cool_until.get(host, 0.0), until)
        if hold:
            self._hold_until[host] = max(self._hold_until.get(host, 0.0), until)
        if first and self.on_limited:
            self.on_limited(host, retry_after_s, hold)

    async def wait_hold(self, host: str) -> None:
        while (left := self._hold_until.get(host, 0.0) - time.monotonic()) > 0:
            await asyncio.sleep(left)

    @contextlib.asynccontextmanager
    async def slot_after_hold(self, host: str) -> AsyncIterator[None]:
        """A download slot for `host`, taken only when the host is not on hold. The hold is checked
        again once the slot is ours: requests queued behind a full host would otherwise all go out
        the moment a 429 started the hold, and each get another 429."""
        while True:
            await self.wait_hold(host)
            async with self.slot(host):
                if not self.holding(host):
                    yield
                    return


def _retry_after(value: str | None) -> float:
    """Seconds to wait from a Retry-After header: delay-seconds or an HTTP date, kept within 5 s - 15 min."""
    try:
        seconds = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        try:
            when = email.utils.parsedate_to_datetime(value)  # type: ignore[arg-type]
            seconds = (when - datetime.now(timezone.utc)).total_seconds()
        except (TypeError, ValueError, IndexError, OverflowError):
            return 60.0
    if seconds != seconds:  # NaN
        return 60.0
    return min(900.0, max(5.0, seconds))


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
    except Image.DecompressionBombError as e:
        # Pillow refuses to open it at all, so the engine couldn't check it either.
        limit = f" (over {2 * Image.MAX_IMAGE_PIXELS / 1e6:.0f} megapixels)" if Image.MAX_IMAGE_PIXELS else ""
        raise FetchError(f"image too large to check{limit}") from e
    except Exception:  # noqa: BLE001 - unknown size just means no megapixel check
        return None, None


def check_image_bytes(url: str, page: str, data: bytes, max_bytes: int) -> FetchedImage:
    """Type, size and hash of downloaded bytes. CPU-bound for big files: call it in a worker thread."""
    if len(data) > max_bytes:
        raise FetchError("image too large")
    kind = sniff_image(data, HEIF_OK, AVIF_OK)
    if kind is None:
        raise FetchError("unsupported image type (SVG, ICO or not an image)")
    ext, mime = kind
    w, h = _dimensions(data)
    return FetchedImage(url, page, data, ext, mime, hashlib.sha256(data).hexdigest(), w, h)


def check_data_url(url: str, page: str, max_bytes: int) -> FetchedImage:
    return check_image_bytes(url, page, _decode_data_url(url), max_bytes)


async def fetch_image(
    context: BrowserContext, cand: ImageCandidate, timeout_s: float, max_bytes: int, gate: HostGate | None = None
) -> FetchedImage:
    """Tries each of cand.urls in turn; the error reported is the last one's."""
    gate = gate or HostGate()
    err: FetchError | None = None
    for n, url in enumerate(cand.urls):
        guess = n < cand.guesses
        host = urlsplit(url).hostname or ""
        if guess and gate.cooling(host):
            continue
        for attempt in (1, 2):
            try:
                async with gate.slot_after_hold(host):
                    if guess and gate.cooling(host):  # a 429 arrived while this waited for a slot
                        break
                    img = await _fetch_one(context, url, cand.page, timeout_s, max_bytes)
            except RateLimited as e:
                gate.limited(host, e.retry_after_s, hold=not guess)
                err = e
                if guess or attempt == 2:
                    break
                continue  # the page's own copy: wait out Retry-After, then one more try
            except FetchError as e:
                err = e
                break
            return dataclasses.replace(img, is_original=True) if guess else img
    raise err or FetchError("no address to fetch")


async def _fetch_one(context: BrowserContext, url: str, page: str, timeout_s: float, max_bytes: int) -> FetchedImage:
    if url[:5].lower() == "data:":
        return await asyncio.to_thread(check_data_url, url, page, max_bytes)
    try:
        resp = await context.request.get(
            url,
            headers={"Referer": page, "Accept": ACCEPT},
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
    finally:
        try:
            await resp.dispose()
        except PlaywrightError:  # pragma: no cover - the context is already closing
            pass
    # Hashing and reading the header of a large file takes a while: keep it off the event loop.
    return await asyncio.to_thread(check_image_bytes, url, page, data, max_bytes)
