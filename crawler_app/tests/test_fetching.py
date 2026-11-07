"""srcset parsing, AVIF, backup copies, rate limits and robots.txt (no browser needed)."""

from __future__ import annotations

import asyncio
import io
import time

import pytest
from PIL import Image

from crawler_app import fetcher
from crawler_app.crawler import ImageCandidate, Robots, extract_candidates
from crawler_app.urls import best_from_srcsets, parse_srcset, sniff_image


# ---- srcset, AVIF, backups ------------------------------------------------------------------


def test_srcset_urls_may_contain_commas():
    cloudinary = (
        "https://res.cloudinary.com/x/image/upload/c_scale,w_400/pic.jpg 400w, "
        "https://res.cloudinary.com/x/image/upload/c_scale,w_800/pic.jpg 800w"
    )
    assert parse_srcset(cloudinary) == [
        ("https://res.cloudinary.com/x/image/upload/c_scale,w_400/pic.jpg", 400.0),
        ("https://res.cloudinary.com/x/image/upload/c_scale,w_800/pic.jpg", 800.0),
    ]
    assert best_from_srcsets([cloudinary]) == "https://res.cloudinary.com/x/image/upload/c_scale,w_800/pic.jpg"
    assert parse_srcset("data:image/gif;base64,R0lGOD 1x, big.jpg 2x") == [
        ("data:image/gif;base64,R0lGOD", 1000.0),
        ("big.jpg", 2000.0),
    ]
    assert parse_srcset("  a.jpg   640w  ,b.jpg 1.5x,") == [("a.jpg", 640.0), ("b.jpg", 1500.0)]
    assert parse_srcset("") == [] and parse_srcset(" , ") == []


def test_sniff_avif_needs_a_decoder_and_reads_compatible_brands():
    major_avif = b"\x00\x00\x00\x1cftypavif\x00\x00\x00\x00avifmif1miaf"
    mif1_major = b"\x00\x00\x00\x1cftypmif1\x00\x00\x00\x00mif1avifmiaf"  # AVIF with a generic major brand
    heic = b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00mif1heic"
    assert sniff_image(major_avif, avif_ok=True) == (".avif", "image/avif")
    assert sniff_image(mif1_major, avif_ok=True) == (".avif", "image/avif")
    assert sniff_image(mif1_major, heif_ok=True, avif_ok=False) is None  # not mistaken for HEIC
    assert sniff_image(major_avif) is None  # nothing can decode it
    assert sniff_image(heic, heif_ok=True) == (".heic", "image/heic")
    assert sniff_image(heic, avif_ok=True) is None


def test_avif_is_accepted_when_pillow_reads_it():
    if not fetcher.AVIF_OK:
        pytest.skip("this Pillow can't read AVIF")
    buf = io.BytesIO()
    Image.new("RGB", (320, 240), "red").save(buf, "AVIF")
    img = fetcher.check_image_bytes("https://cdn.example/x", "https://example/", buf.getvalue(), 10**7)
    assert (img.ext, img.mime, img.width, img.height) == (".avif", "image/avif", 320, 240)
    assert fetcher.ACCEPT.startswith("image/avif,")


def test_image_too_large_to_open_is_a_fetch_error(monkeypatch):
    buf = io.BytesIO()
    Image.new("L", (400, 300)).save(buf, "PNG")
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 50_000)  # 400x300 is now over Pillow's 2x limit
    with pytest.raises(fetcher.FetchError, match="too large to check"):
        fetcher.check_image_bytes("https://e.com/map.png", "https://e.com/", buf.getvalue(), 10**7)


def test_extract_candidates_keeps_the_shown_copy_as_a_backup():
    raw = {
        "base": "https://e.com/",
        "imgs": [
            # the largest srcset entry differs from what the browser shows: the shown copy is the backup
            {"src": "/s.jpg", "current": "https://e.com/m.jpg", "srcset": "/m.jpg 800w, /l.jpg 1600w",
             "nw": 800, "nh": 600},
            # lazy images sharing a 1x1 placeholder: both kept, the placeholder is never a backup
            {"src": "/blank.gif", "current": "https://e.com/blank.gif", "dsrc": "/a.jpg", "nw": 1, "nh": 1},
            {"src": "/blank.gif", "current": "https://e.com/blank.gif", "dsrc": "/b.jpg", "nw": 1, "nh": 1},
            # a shared, loaded "loading" picture is a backup, but doesn't make the next image a duplicate
            {"src": "/loading.png", "current": "https://e.com/loading.png", "dsrc": "/c.jpg", "nw": 300, "nh": 200},
            {"src": "/loading.png", "current": "https://e.com/loading.png", "dsrc": "/d.jpg", "nw": 300, "nh": 200},
        ],
    }
    imgs, _ = extract_candidates(raw, "https://e.com/")
    assert [c.url for c in imgs] == [f"https://e.com/{n}.jpg" for n in ("l", "a", "b", "c", "d")]
    assert imgs[0].fallbacks == ("https://e.com/m.jpg",) and imgs[0].ids == ("https://e.com/l.jpg",)
    assert imgs[1].fallbacks == () and imgs[2].fallbacks == ()
    assert imgs[3].fallbacks == ("https://e.com/loading.png",) and imgs[3].guesses == 0


def test_backup_is_fetched_when_the_picked_url_fails():
    jpeg = _jpeg_bytes()
    ctx = _Context(lambda url: _Resp(404) if "big" in url else _Resp(200, jpeg, {"content-type": "image/jpeg"}))
    cand = ImageCandidate("https://e.com/big.jpg", "https://e.com/", None, ("https://e.com/shown.jpg",))
    img = asyncio.run(fetcher.fetch_image(ctx, cand, 5, 10**7))
    assert img.url == "https://e.com/shown.jpg" and not img.is_original


# ---- rate limits and robots.txt -----------------------------------------------------------------


class _Resp:
    def __init__(self, status, body=b"", headers=None):
        self.status, self._body, self.headers = status, body, headers or {}
        self.ok = status < 400

    async def body(self):
        return self._body

    async def text(self):
        return self._body.decode()

    async def dispose(self):
        pass


class _Context:
    """A browser context whose request.get answers with `answer(url)`, recording when each URL was asked."""

    def __init__(self, answer):
        self.calls: list[tuple[str, float]] = []
        calls = self.calls

        class Request:
            async def get(self, url, **kw):
                calls.append((url, time.monotonic()))
                return answer(url)

        self.request = Request()


def _jpeg_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (200, 150), "red").save(buf, "JPEG")
    return buf.getvalue()


def test_rate_limited_page_copy_waits_and_tries_once_more(monkeypatch):
    monkeypatch.setattr(fetcher, "_retry_after", lambda value: 0.3)
    jpeg = _jpeg_bytes()
    refused: set[str] = set()

    def answer(url):
        if "always" in url or url not in refused:  # the first ask for each URL is refused
            refused.add(url)
            return _Resp(429)
        return _Resp(200, jpeg, {"content-type": "image/jpeg"})

    ctx = _Context(answer)
    notices = []
    gate = fetcher.HostGate(on_limited=lambda host, s, held: notices.append((host, s, held)))

    def cand(name):
        return ImageCandidate(f"https://s.com/{name}.jpg", "https://s.com/")

    async def go():
        first = asyncio.create_task(fetcher.fetch_image(ctx, cand("1"), 5, 10**7, gate))
        await asyncio.sleep(0.05)
        # started during the cool-down: it waits too instead of asking the site right away
        second = await fetcher.fetch_image(ctx, cand("2"), 5, 10**7, gate)
        return await first, second

    first, second = asyncio.run(go())
    assert first.url.endswith("/1.jpg") and second.url.endswith("/2.jpg")
    times: dict[str, list[float]] = {}
    for url, t in ctx.calls:
        times.setdefault(url, []).append(t)
    one, two = times["https://s.com/1.jpg"], times["https://s.com/2.jpg"]
    assert len(one) == 2 and one[1] - one[0] >= 0.25  # refused, waited, asked again
    assert two[0] - one[0] >= 0.25
    assert notices[0] == ("s.com", 0.3, True)

    # refused twice: the image fails with the rate-limit error after just one retry
    with pytest.raises(fetcher.RateLimited):
        asyncio.run(fetcher.fetch_image(ctx, cand("always"), 5, 10**7, gate))
    assert sum(1 for url, _ in ctx.calls if url.endswith("always.jpg")) == 2


def test_robots_server_error_means_the_site_is_closed():
    notes: list[str] = []
    closed = Robots(_Context(lambda url: _Resp(503)), 5, notes.append)
    assert not asyncio.run(closed.allowed("https://s.com/a")) and not asyncio.run(closed.allowed("https://s.com/b"))
    assert len(notes) == 1 and "503" in notes[0]  # told once per site
    assert asyncio.run(Robots(_Context(lambda url: _Resp(404)), 5).allowed("https://s.com/a"))  # no robots.txt
    rules = Robots(_Context(lambda url: _Resp(200, b"User-agent: *\nDisallow: /private\n")), 5)
    assert asyncio.run(rules.allowed("https://s.com/a")) and not asyncio.run(rules.allowed("https://s.com/private/x"))
