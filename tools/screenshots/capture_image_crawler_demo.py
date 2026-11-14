r"""Website-mode screenshots of RulZ Image Crawler without a GPU or a model.

    VENV\python.exe -B capture_image_crawler_demo.py --images R:\Pictures\Wallpapers --work R:\ic-demo ^
        --out R:\out\rulz-image-crawler

VENV is any venv with the app's packages, Playwright and its Chromium, for example the launcher's install
("%LOCALAPPDATA%\Programs\RulZ AI Tools\RulZ Image Crawler\.venv"); the app itself is imported from this checkout.

* The detection engine is replaced by a scripted stand-in: "Generate checklist" returns a fixed checklist for
  "landscapes with mountains", and each picture is judged by the name make_demo_images.py gave it (mountain
  titles match, the rest don't). No llama-server is started and no model is loaded.
* A tiny gallery site made of the demo pictures (two pages, one broken image link for the Problems list) is served
  on 127.0.0.1, so the crawl never leaves this PC (Playwright's downloader can't resolve *.localhost names).
* 04-website.png: the website options filled in (https://example.com/gallery is only typed, never crawled).
  05-website-run.png: the finished crawl of the demo gallery. Output goes to --work, nothing else is written.
"""
from __future__ import annotations

import argparse
import asyncio
import functools
import hashlib
import html
import http.server
import sys
import threading
import time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
APP = HERE.parent.parent / "apps" / "rulz-image-crawler"
sys.path.insert(0, str(APP))  # crawler_app and image_detection from this checkout, not the install's

ap = argparse.ArgumentParser()
ap.add_argument("--images", required=True, help="demo pictures from make_demo_images.py --names titles")
ap.add_argument("--work", required=True, help="a new folder: the demo site and the app's output go here")
ap.add_argument("--out", required=True)
ap.add_argument("--port", type=int, default=18791)
ap.add_argument("--site-port", type=int, default=18792)
ap.add_argument("--site-host", default="127.0.0.1")
ap.add_argument("--typed-url", default="https://example.com/gallery")
ap.add_argument("--prompt", default="landscapes with mountains")
ap.add_argument("--no-run", action="store_true", help="only 04-website.png")
a = ap.parse_args()

work, out = Path(a.work), Path(a.out)
if work.exists() and any(work.iterdir()):
    sys.exit(f"{work} is not empty: use a new folder")
out.mkdir(parents=True, exist_ok=True)

import uvicorn  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

from crawler_app.config import AppSettings  # noqa: E402
from crawler_app.server import create_app  # noqa: E402

for mod in ("crawler_app.server", "image_detection"):
    assert Path(sys.modules[mod].__file__).resolve().is_relative_to(APP.resolve()), f"{mod} is not this checkout's"

# ---- the demo site: two gallery pages of demo pictures ----
pictures = sorted(p for p in Path(a.images).iterdir() if p.suffix.lower() == ".jpg")
site = work / "site"
(site / "gallery" / "photos").mkdir(parents=True)
names: dict[str, str] = {}  # sha256 -> picture name, for the stand-in's verdicts
for p in pictures:
    data = p.read_bytes()
    names[hashlib.sha256(data).hexdigest()] = p.stem
    (site / "gallery" / "photos" / p.name.replace(" ", "-").lower()).write_bytes(data)
half = (len(pictures) + 1) // 2


def page(title: str, pics: list[Path], extra: str) -> str:
    imgs = "\n".join(
        f'<figure><img src="photos/{html.escape(p.name.replace(" ", "-").lower())}" width="320">'
        f"<figcaption>{html.escape(p.stem)}</figcaption></figure>" for p in pics)
    return f"<!doctype html><html><head><title>{title}</title></head><body><h1>{title}</h1>\n{imgs}\n{extra}</body></html>"


(site / "gallery" / "index.html").write_text(
    page("Demo gallery", pictures[:half], '<a href="page-2.html">Next page</a> <a href="/about.html">About</a>'),
    encoding="utf-8")
(site / "gallery" / "page-2.html").write_text(
    page("Demo gallery, page 2", pictures[half:],
         '<img src="photos/sunset-99.jpg" width="320"><a href="index.html">First page</a>'), encoding="utf-8")
(site / "about.html").write_text("<!doctype html><p>A demo site.</p>", encoding="utf-8")


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args) -> None:
        pass


site_server = http.server.ThreadingHTTPServer(("127.0.0.1", a.site_port), functools.partial(Quiet, directory=str(site)))
threading.Thread(target=site_server.serve_forever, daemon=True).start()

# ---- the stand-in for image_detection.DetectionService ----
RUBRIC = (
    "SUBJECT: a natural outdoor scene. REQUIRE: at least one mountain must be visible; the mountain must be a "
    "natural landform with a steep slope or peak; the scene must be outdoors and not indoors or digitally "
    "generated. EXCLUDE: urban scenes with buildings or roads; hills that are too low or rounded to be "
    "mountains; mountains obscured by clouds or fog; artificial structures like ski resorts or dams on the "
    "mountain; mountains in paintings or illustrations. FALSE IF: the image is indoors or a close-up of an "
    "object; no natural landform is visible; the mountain is too small or indistinct to be identified as a mountain."
)
MOUNTAINS = ("Ridge Lines", "Alpine Haze", "Lakeside", "Mountain Light", "Peaks")
YES = ["Visible mountains, natural outdoor scene, no artificial structures", "Mountain peak visible above trees",
       "Multiple mountain ranges visible, natural outdoor scene", "Steep peaks reflected in a lake"]
NO = {"Meadow": "Rolling hills, too low to be mountains", "Hillside": "Rounded hills, no steep peaks",
      "Dunes": "Sand dunes, no mountains", "Desert": "Sand dunes, no mountains",
      "Aurora": "Night sky with aurora, no mountain visible", "Northern Lights": "Aurora over flat land",
      "Bokeh": "Blurred city lights, not a landscape", "City Lights": "Blurred city lights, not a landscape",
      "Shapes": "Abstract shapes, not a landscape", "Composition": "Abstract shapes, not a landscape"}


class DemoService:
    async def health(self):
        return {"model": "demo", "server": "up", "slots_free": 2, "stats": {}}

    async def expand_tag(self, text: str):
        await asyncio.sleep(0.5)
        return {"id": "landscapes_with_mountains", "rubric": RUBRIC}

    async def classify(self, images, tags, threshold=None):
        threshold = 0.7 if threshold is None else threshold
        out = {}
        for img in images:
            await asyncio.sleep(0.25)
            name = names.get(img.id, "")
            h = int(img.id[:8], 16)
            if name.startswith(MOUNTAINS):
                match, conf, ev = True, 0.80 + (h % 20) / 100, YES[h % len(YES)]
            else:
                match, conf = False, 0.85 + (h % 14) / 100
                ev = next((v for k, v in NO.items() if name.startswith(k)), "Open sea and sky, no mountains")
            out[img.id] = {"status": "ok", "cached": False,
                           "tags": {t.id: {"match": match and conf >= threshold, "confidence": conf, "evidence": ev}
                                    for t in tags}}
        return out

    async def aclose(self):
        pass


# ---- the app, with its own (real) headless Chromium for the crawl ----
app = create_app(AppSettings(output_dir=work / "output"), service=DemoService(), browser=None if not a.no_run else object())
server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=a.port, log_level="warning"))
threading.Thread(target=server.run, daemon=True).start()
while not server.started:
    time.sleep(0.1)

site_url = f"http://{a.site_host}:{a.site_port}/gallery/"
try:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        pg = browser.new_page(viewport={"width": 1600, "height": 960})

        def shot(name: str) -> None:
            pg.wait_for_timeout(600)
            pg.screenshot(path=str(out / name))
            print("saved", out / name, flush=True)

        def fill(url: str) -> None:
            pg.goto(f"http://127.0.0.1:{a.port}/")
            pg.wait_for_selector("#modelLine:has-text('Model ready')")
            pg.check("input[type=radio][value=web]")
            pg.fill("#url", url)
            pg.fill("#prompt", a.prompt)
            pg.click("#genBtn")
            pg.wait_for_function("document.querySelector('#rubric').value.length > 20")
            pg.check("#stayInPath")
            pg.evaluate("document.activeElement.blur(); scrollTo(0, 0)")

        fill(a.typed_url)
        shot("04-website.png")
        if not a.no_run:
            fill(site_url)
            pg.fill("#delay", "0.5")
            pg.click("#startBtn")
            pg.wait_for_function("!document.querySelector('#newBtn').classList.contains('hidden')", timeout=300000)
            pg.click("#errorsBox summary")
            pg.wait_for_timeout(1500)
            pg.evaluate("scrollTo(0, document.querySelector('#progress').getBoundingClientRect().top + scrollY - 20)")
            shot("05-website-run.png")
        browser.close()
finally:
    server.should_exit = True
    time.sleep(1.5)
    site_server.shutdown()
