from __future__ import annotations

from crawler_app.crawler import extract_candidates
from crawler_app.urls import (
    best_from_srcsets,
    file_stem,
    looks_like_page,
    normalize_image_url,
    normalize_page_url,
    original_candidates,
    site_key,
    sniff_image,
)


def test_normalize_page_url_strips_fragment_tracking_and_default_port():
    u = normalize_page_url("HTTPS://Example.com:443/a/b?utm_source=x&id=5&fbclid=y#top")
    assert u == "https://example.com/a/b?id=5"


def test_normalize_page_url_resolves_relative_and_rejects_non_http():
    assert normalize_page_url("../c", "https://example.com/a/b/") == "https://example.com/a/c"
    assert normalize_page_url("mailto:me@example.com") is None
    assert normalize_page_url("javascript:void(0)") is None


def test_image_url_keeps_query_verbatim():
    u = normalize_image_url("/img.jpg?sig=a%2Fb&utm_source=x#f", "https://cdn.example.com/p/")
    assert u == "https://cdn.example.com/img.jpg?sig=a%2Fb&utm_source=x"


def test_site_key_treats_www_as_same_site():
    assert site_key("https://www.example.com/x") == site_key("http://example.com/") == "example.com"
    assert site_key("https://shop.example.com/") != "example.com"


def test_looks_like_page():
    assert looks_like_page("https://e.com/gallery/")
    assert looks_like_page("https://e.com/post.html")
    assert not looks_like_page("https://e.com/file.PDF")
    assert not looks_like_page("https://e.com/photo.jpg")


def test_best_from_srcsets_picks_largest():
    assert best_from_srcsets(["a.jpg 320w, b.jpg 1280w, c.jpg 640w"]) == "b.jpg"
    assert best_from_srcsets(["a.jpg, b.jpg 2x"]) == "b.jpg"
    assert best_from_srcsets(["s.webp 400w", "l.webp 1600w"]) == "l.webp"
    assert best_from_srcsets([]) is None


def test_sniff_image():
    assert sniff_image(b"\xff\xd8\xff\xe0rest") == (".jpg", "image/jpeg")
    assert sniff_image(b"\x89PNG\r\n\x1a\nrest") == (".png", "image/png")
    assert sniff_image(b"RIFF\x00\x00\x00\x00WEBPVP8 ") == (".webp", "image/webp")
    assert sniff_image(b"<svg xmlns='http://www.w3.org/2000/svg'>") is None
    assert sniff_image(b"\x00\x00\x01\x00ico") is None


def test_file_stem_is_safe():
    assert file_stem("https://e.com/a/My%20Photo%20(1).JPG?x=1") == "my-photo-1"
    assert file_stem("https://e.com/../../etc/passwd") == "passwd"
    assert file_stem("https://e.com/") == "image"
    assert file_stem("data:image/png;base64,AAAA") == "inline"


def test_extract_candidates_prefers_srcset_and_lazy_attrs():
    raw = {
        "base": "https://e.com/p/",
        "imgs": [
            {"src": "small.jpg", "current": "https://e.com/p/small.jpg", "srcset": "small.jpg 300w, big.jpg 1200w",
             "nw": 300, "nh": 200, "pic": []},
            {"src": "placeholder.gif", "current": "https://e.com/p/placeholder.gif", "dsrc": "/lazy/real.jpg",
             "nw": 1, "nh": 1, "pic": []},
            {"src": "icon.png", "current": "https://e.com/p/icon.png", "nw": 32, "nh": 32, "pic": []},
            {"src": "fallback.jpg", "current": "https://e.com/p/fallback.jpg", "nw": 500, "nh": 500,
             "pic": ["pic-800.webp 800w, pic-1600.webp 1600w"]},
        ],
        "bgs": ['url("/bg/hero.jpg")', "linear-gradient(red, blue)"],
        "og": ["https://e.com/og.jpg"],
        "links": ["https://e.com/p/next#c", "https://e.com/p/next", "mailto:x@e.com"],
    }
    images, links = extract_candidates(raw, "https://e.com/p/")
    urls = [c.url for c in images]
    assert urls == [
        "https://e.com/p/big.jpg",
        "https://e.com/lazy/real.jpg",
        "https://e.com/p/icon.png",
        "https://e.com/p/pic-1600.webp",
        "https://e.com/bg/hero.jpg",
        "https://e.com/og.jpg",
    ]
    assert images[2].short_edge == 32  # tiny hint only when the shown image is the one picked
    assert images[0].short_edge is None
    assert links == ["https://e.com/p/next"]


def test_original_candidates_known_resize_patterns():
    wm = "https://upload.wikimedia.org/wikipedia/commons/thumb/a/ab/Red_Fox.jpg/500px-Red_Fox.jpg"
    assert original_candidates(wm) == ["https://upload.wikimedia.org/wikipedia/commons/a/ab/Red_Fox.jpg"]
    new_host = "https://thumb.wikimedia.org/wikipedia/commons/thumb/a/ab/Red_Fox.jpg/500px-Red_Fox.jpg?utm_source=x"
    assert original_candidates(new_host) == ["https://upload.wikimedia.org/wikipedia/commons/a/ab/Red_Fox.jpg"]
    wp = "https://site.com/wp-content/uploads/2024/01/photo-300x200.jpg"
    assert original_candidates(wp) == ["https://site.com/wp-content/uploads/2024/01/photo.jpg"]
    assert original_candidates("https://site.com/u/big-scaled.jpg") == ["https://site.com/u/big.jpg"]
    assert original_candidates("https://cdn.site.com/a.jpg?w=400&h=300&sig=abc") == ["https://cdn.site.com/a.jpg?sig=abc"]
    assert original_candidates("https://lh3.googleusercontent.com/abc=w400-h300-c") == [
        "https://lh3.googleusercontent.com/abc=s0"
    ]
    assert original_candidates("https://1.bp.blogspot.com/x/y/s400/pic.jpg") == ["https://1.bp.blogspot.com/x/y/s0/pic.jpg"]
    # Best guess first; the size-suffixed name without the width param stays as a second try.
    assert original_candidates("https://cdn.shopify.com/s/files/1/p/shirt_400x400.jpg?v=1&width=400")[0] == (
        "https://cdn.shopify.com/s/files/1/p/shirt.jpg?v=1"
    )


def test_original_candidates_leave_plain_urls_alone():
    assert original_candidates("https://site.com/img/photo.jpg") == []
    assert original_candidates("https://site.com/img/photo.jpg?v=3") == []
    assert original_candidates("data:image/png;base64,AAAA") == []


def test_extract_candidates_tries_linked_original_first():
    raw = {
        "base": "https://s.com/",
        "imgs": [
            {"src": "/t/cat-thumb.jpg", "nw": 150, "nh": 100, "link": "https://s.com/full/cat.jpg"},
            {"src": "/t/dog.jpg", "nw": 50, "nh": 50, "link": "https://s.com/dogs.html"},
        ],
    }
    imgs, _ = extract_candidates(raw, "https://s.com/")
    assert imgs[0].url == "https://s.com/full/cat.jpg"
    assert imgs[0].fallbacks == ("https://s.com/t/cat-thumb.jpg",) and imgs[0].short_edge is None
    assert imgs[1].url == "https://s.com/t/dog.jpg" and imgs[1].short_edge == 50  # link is a page, not an image

    plain, _ = extract_candidates(raw, "https://s.com/", prefer_originals=False)
    assert plain[0].url == "https://s.com/t/cat-thumb.jpg" and plain[0].fallbacks == ()


def test_rate_limited_host_falls_back_and_skips_originals_while_cooling():
    import asyncio
    import io

    from PIL import Image

    from crawler_app.crawler import ImageCandidate
    from crawler_app.fetcher import HostGate, fetch_image

    buf = io.BytesIO()
    Image.new("RGB", (200, 150), "red").save(buf, "JPEG")
    jpeg = buf.getvalue()
    calls: list[str] = []

    class Resp:
        def __init__(self, status, body=b"", headers=None):
            self.status, self._body, self.headers = status, body, headers or {}

        async def body(self):
            return self._body

        async def dispose(self):
            pass

    class Request:
        async def get(self, url, **kw):
            calls.append(url)
            if "orig" in url:
                return Resp(429, headers={"retry-after": "120"})
            return Resp(200, jpeg, {"content-type": "image/jpeg"})

    class Context:
        request = Request()

    notices = []
    gate = HostGate(on_limited=lambda host, s: notices.append((host, s)))
    cand = lambda i: ImageCandidate(f"https://s.com/orig/{i}.jpg", "https://s.com/", None, (f"https://s.com/t/{i}.jpg",))

    first = asyncio.run(fetch_image(Context(), cand(1), 5, 10**7, gate))
    assert first.url == "https://s.com/t/1.jpg" and not first.is_original
    assert notices == [("s.com", 120.0)]
    second = asyncio.run(fetch_image(Context(), cand(2), 5, 10**7, gate))
    assert second.url == "https://s.com/t/2.jpg"
    assert "https://s.com/orig/2.jpg" not in calls  # not even tried while the host cools down
