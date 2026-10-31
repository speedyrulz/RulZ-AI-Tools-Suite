"""Shared fixtures: a generated local test site served over HTTP, and a fake DetectionService.

The fake judges by colour, so the tests need real Chromium (Playwright) but no GPU or model.
"""

from __future__ import annotations

import base64
import functools
import http.server
import io
import threading
from pathlib import Path

import pytest
from image_detection import ModelServerDown
from PIL import Image, ImageDraw


def _shape(path: Path, kind: str, color, size=(640, 480), fmt=None) -> None:
    im = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(im)
    w, h = size
    box = (w * 0.25, h * 0.2, w * 0.75, h * 0.8)
    (d.ellipse if kind == "circle" else d.rectangle)(box, fill=color)
    im.save(path, format=fmt, quality=92)


def build_site(root: Path) -> None:
    img = root / "img"
    img.mkdir(parents=True)
    _shape(img / "red-circle-small.jpg", "circle", "red", (320, 240))
    _shape(img / "red-circle-large.jpg", "circle", "red", (1280, 960))
    _shape(img / "blue-square.jpg", "square", "blue")
    _shape(img / "green-square.png", "square", "green")
    _shape(img / "red-circle-lazy.jpg", "circle", (200, 20, 20))
    _shape(img / "bg-red-circle.jpg", "circle", (230, 0, 0))
    _shape(img / "tiny.png", "circle", "red", (50, 50))
    _shape(img / "My Photo (1).jpg", "circle", (220, 10, 10))
    _shape(img / "page2-blue-circle.jpg", "circle", "blue")
    _shape(img / "page2-red-circle.webp", "circle", "red", fmt="WEBP")
    # Full-size originals: a thumbnail linking to its original, and a WordPress-style resized copy.
    (img / "thumbs").mkdir()
    (img / "full").mkdir()
    _shape(img / "thumbs" / "red-gallery-thumb.jpg", "circle", "red", (200, 150))
    _shape(img / "full" / "red-gallery.jpg", "circle", "red", (1600, 1200))
    _shape(img / "red-photo-300x225.jpg", "circle", "red", (300, 225))
    _shape(img / "red-photo.jpg", "circle", "red", (1200, 900))
    _shape(img / "blue-only-300x225.jpg", "square", "blue", (300, 225))  # no original: falls back
    (img / "dup-of-red.jpg").write_bytes((img / "red-circle-lazy.jpg").read_bytes())
    (img / "logo.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200"><circle cx="100" cy="100" r="80" fill="red"/></svg>'
    )
    (root / "index.html").write_text(
        """<!doctype html><html><head><meta property="og:image" content="/img/green-square.png"></head><body>
<img src="/img/red-circle-small.jpg" srcset="/img/red-circle-small.jpg 320w, /img/red-circle-large.jpg 1280w" width="320">
<img src="/img/blue-square.jpg">
<img src="/img/tiny.png">
<img src="/img/logo.svg" width="200">
<img src="/img/My%20Photo%20(1).jpg">
<a href="/img/full/red-gallery.jpg"><img src="/img/thumbs/red-gallery-thumb.jpg"></a>
<img src="/img/red-photo-300x225.jpg">
<img src="/img/blue-only-300x225.jpg">
<div style="height:2500px"></div>
<img loading="lazy" src="data:image/gif;base64,R0lGODlhAQABAAAAACw=" data-src="/img/red-circle-lazy.jpg">
<img src="/img/dup-of-red.jpg">
<div style="width:400px;height:300px;background-image:url('/img/bg-red-circle.jpg')"></div>
<a href="/page2.html?utm_source=test#top">Page 2</a>
<a href="/files/report.pdf">PDF</a>
<a href="https://other.example.invalid/">Elsewhere</a>
</body></html>"""
    )
    (root / "page2.html").write_text(
        """<!doctype html><html><body>
<img src="/img/page2-blue-circle.jpg"><img src="/img/page2-red-circle.webp">
<a href="/">Home</a><a href="/page3.html">Page 3</a></body></html>"""
    )
    (root / "page3.html").write_text("<!doctype html><html><body><p>No images here.</p></body></html>")
    (root / "robots.txt").write_text("User-agent: *\nDisallow: /page3.html\n")


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args) -> None:
        pass


@pytest.fixture(scope="session")
def site_url(tmp_path_factory) -> str:
    root = tmp_path_factory.mktemp("site")
    build_site(root)
    handler = functools.partial(_QuietHandler, directory=str(root))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/"
    server.shutdown()


def _dominant(data: bytes) -> tuple[tuple[int, int, int], tuple[int, int]]:
    im = Image.open(io.BytesIO(data))
    im.seek(0)
    rgb = im.convert("RGB")
    small = rgb.resize((64, 48))
    colors = sorted(small.getcolors(64 * 48), reverse=True)
    for _, c in colors:
        if min(c) < 235:  # first clearly non-white colour
            return c, im.size
    return (255, 255, 255), im.size


class FakeService:
    """Stands in for image_detection.DetectionService. Red shapes match with 0.95, green shapes match
    with 0.75 (to exercise the threshold), everything else is a confident no."""

    def __init__(self):
        self.down = False
        self.calls = 0
        self.seen_ids: list[str] = []

    async def health(self):
        return {"model": "fake", "server": "down" if self.down else "up", "slots_free": 2, "stats": {}}

    async def expand_tag(self, text: str):
        if self.down:
            raise ModelServerDown("fake down")
        return {"id": "red_shape", "rubric": "SUBJECT: a shape. REQUIRE: it is red. EXCLUDE: blue. FALSE IF: not visible."}

    async def classify(self, images, tags, threshold=None):
        if self.down:
            raise ModelServerDown("fake down")
        self.calls += 1
        threshold = 0.7 if threshold is None else threshold
        out = {}
        for img in images:
            self.seen_ids.append(img.id)
            data = img.data if img.data is not None else base64.b64decode(img.data_b64)
            (r, g, b), (w, h) = _dominant(data)
            if min(w, h) < 96:
                out[img.id] = {"status": "skipped", "cached": False, "message": "too small"}
                continue
            if r > 150 and g < 90 and b < 90:
                match, conf, ev = True, 0.95, "red shape"
            elif g > 90 and r < 90 and b < 90:
                match, conf, ev = True, 0.75, "greenish shape"
            else:
                match, conf, ev = False, 0.9, "not red"
            out[img.id] = {
                "status": "ok",
                "cached": False,
                "tags": {t.id: {"match": match and conf >= threshold, "confidence": conf, "evidence": ev} for t in tags},
            }
        return out


@pytest.fixture
def fake_service() -> FakeService:
    return FakeService()
