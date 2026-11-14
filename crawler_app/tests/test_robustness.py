"""Redirects, robots.txt encodings, 429 holds, matches.json writes, failed saves, streamed zips, folder
listing limits and a stopped browser."""

from __future__ import annotations

import asyncio
import errno
import http.server
import io
import json
import threading
import time
import zipfile
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

from conftest import _shape
from playwright.async_api import async_playwright
from test_fetching import _Context, _jpeg_bytes, _Resp
from test_runs import _folder_run, _past_run

from crawler_app import fetcher
from crawler_app import storage as storage_mod
from crawler_app.config import AppSettings
from crawler_app.crawler import ImageCandidate, Robots
from crawler_app.folder import list_image_files
from crawler_app.runs import DISK_FULL_MESSAGE, Run, RunManager, RunOptions
from crawler_app.storage import RunStorage

# ---- a site with redirects and a Latin-1 robots.txt (real Chromium) ---------------------------------


def _jpeg(color) -> bytes:
    from PIL import Image, ImageDraw

    im = Image.new("RGB", (640, 480), "white")
    ImageDraw.Draw(im).ellipse((160, 96, 480, 384), fill=color)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=92)
    return buf.getvalue()


def _serve(
    routes: dict[str, tuple[int, dict[str, str], bytes]], seen: list[tuple[str, str]] | None = None
) -> tuple[http.server.HTTPServer, int]:
    """Serves `routes` on 127.0.0.1; `seen` gets (host name the browser used, path) for each request."""

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if seen is not None:
                seen.append((self.headers.get("Host", "").rsplit(":", 1)[0], self.path))
            status, headers, body = routes.get(self.path.split("?")[0], (404, {}, b"not found"))
            self.send_response(status)
            for k, v in headers.items():
                self.send_header(k, v.replace("{port}", str(port)))
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args) -> None:
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, port


def _html(body: str) -> tuple[int, dict[str, str], bytes]:
    return 200, {"Content-Type": "text/html"}, f"<!doctype html><html><body>{body}</body></html>".encode()


def test_redirects_out_of_the_site_are_not_crawled_and_latin1_robots_is_read(tmp_path, fake_service):
    jpg = {"Content-Type": "image/jpeg"}
    routes = {
        # A Latin-1 comment: Playwright's strict text() used to raise and fail the whole run.
        "/robots.txt": (200, {"Content-Type": "text/plain"}, "User-agent: *\n# Café\nDisallow: /private.html\n".encode("latin-1")),
        "/index.html": _html(
            '<img src="/img/red.jpg"><a href="/go">out</a><a href="/private.html">private</a>'
            '<a href="/alias">home again</a><a href="/page2.html">page 2</a>'
        ),
        "/go": (302, {"Location": "http://localhost:{port}/offsite.html"}, b""),  # another site (host name)
        "/alias": (302, {"Location": "/index.html"}, b""),  # a page already read
        "/page2.html": _html('<img src="/img/red2.jpg">'),
        "/offsite.html": _html('<img src="/img/offsite.jpg">'),
        "/private.html": _html('<img src="/img/private.jpg">'),
        "/img/red.jpg": (200, jpg, _jpeg((255, 0, 0))),
        "/img/red2.jpg": (200, jpg, _jpeg((235, 0, 0))),
        "/img/offsite.jpg": (200, jpg, _jpeg((215, 0, 0))),
        "/img/private.jpg": (200, jpg, _jpeg((195, 0, 0))),
    }
    seen: list[tuple[str, str]] = []
    server, port = _serve(routes, seen)
    opts = RunOptions(
        url=f"http://127.0.0.1:{port}/index.html", tag_id="red_shape", rubric="red", threshold=0.8,
        page_delay_s=0.0, respect_robots=True, max_pages=10, max_depth=2,
    )
    try:
        run = asyncio.run(_crawl_run(tmp_path, fake_service, opts))
    finally:
        server.shutdown()
    assert run.status == "done", run.message
    assert run.stats["pages_crawled"] == 2  # index and page 2; the alias led back to index
    assert run.stats["pages_blocked"] == 2  # /private.html (robots.txt) and /go (left the site)
    saved = sorted(m["url"].rsplit("/", 1)[1] for m in run.matches)
    assert saved == ["red.jpg", "red2.jpg"]
    assert any("blocked by robots.txt" in e and "private" in e for e in run.errors)
    assert ("localhost", "/robots.txt") not in seen  # the other site's robots.txt is none of its business


async def _crawl_run(tmp_path, service, opts) -> Run:
    async with async_playwright() as pw:
        browser = await pw.chromium.launch()
        try:
            run = Run("r1", opts, AppSettings(output_dir=tmp_path / "out", max_scroll_steps=3), service, browser)
            run.start()
            await asyncio.wait_for(run.wait(), 90)
            return run
        finally:
            await browser.close()


def test_pages_that_redirect_away_still_count_toward_max_pages(tmp_path, fake_service):
    links = "".join(f'<a href="/out{i}">out {i}</a>' for i in range(8))
    routes = {
        "/index.html": _html(f'<img src="/img/red.jpg">{links}<a href="/page2.html">page 2</a>'),
        "/page2.html": _html("<p>never reached</p>"),
        "/elsewhere.html": _html("<p>another site</p>"),
        "/img/red.jpg": (200, {"Content-Type": "image/jpeg"}, _jpeg((255, 0, 0))),
        **{f"/out{i}": (302, {"Location": "http://localhost:{port}/elsewhere.html"}, b"") for i in range(8)},
    }
    server, port = _serve(routes)
    opts = RunOptions(
        url=f"http://127.0.0.1:{port}/index.html", tag_id="red_shape", rubric="red", threshold=0.8,
        page_delay_s=0.0, max_pages=3, max_depth=2,
    )
    try:
        run = asyncio.run(_crawl_run(tmp_path, fake_service, opts))
    finally:
        server.shutdown()
    assert run.status == "done", run.message
    assert run.stats["pages_crawled"] == 1 and run.stats["pages_blocked"] == 2  # three loads, then it stops


# ---- robots.txt and 429 holds (no browser) ---------------------------------------------------------


def test_robots_txt_in_latin1_or_unreadable_never_fails_the_run():
    latin1 = "User-agent: *\n# Café\nDisallow: /private\n".encode("latin-1")
    rules = Robots(_Context(lambda url: _Resp(200, latin1)), 5)
    assert asyncio.run(rules.allowed("https://s.com/a")) and not asyncio.run(rules.allowed("https://s.com/private/x"))

    class Broken(_Resp):
        async def body(self):
            raise RuntimeError("connection reset while reading")

    assert asyncio.run(Robots(_Context(lambda url: Broken(200)), 5).allowed("https://s.com/a"))  # counts as missing


def test_downloads_queued_behind_a_busy_host_wait_out_its_new_hold(monkeypatch):
    monkeypatch.setattr(fetcher, "_retry_after", lambda value: 0.4)
    jpeg = _jpeg_bytes()
    calls: list[tuple[str, float]] = []

    class Request:
        async def get(self, url, **kw):
            calls.append((url, time.monotonic()))
            await asyncio.sleep(0.05)  # holds the host's only slot a moment
            if url.endswith("/1.jpg") and sum(1 for u, _ in calls if u == url) == 1:
                return _Resp(429)
            return _Resp(200, jpeg, {"content-type": "image/jpeg"})

    class Ctx:
        request = Request()

    gate = fetcher.HostGate(per_host=1)

    async def go():
        first = asyncio.create_task(fetcher.fetch_image(Ctx(), ImageCandidate("https://s.com/1.jpg", "https://s.com/"), 5, 10**7, gate))
        await asyncio.sleep(0.01)  # queued for the slot before the 429 arrives
        second = await fetcher.fetch_image(Ctx(), ImageCandidate("https://s.com/2.jpg", "https://s.com/"), 5, 10**7, gate)
        return await first, second

    asyncio.run(go())
    first_ask = calls[0][1]
    second_ask = next(t for u, t in calls if u.endswith("/2.jpg"))
    assert second_ask - first_ask >= 0.35  # it waited out the hold instead of going out right after the 429


def test_retry_after_reads_seconds_and_http_dates():
    soon = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=120), usegmt=True)
    assert 100 <= fetcher._retry_after(soon) <= 121
    assert fetcher._retry_after(format_datetime(datetime.now(timezone.utc) - timedelta(hours=1), usegmt=True)) == 5.0
    assert fetcher._retry_after("30") == 30.0 and fetcher._retry_after("1") == 5.0
    assert fetcher._retry_after("soon") == 60.0 and fetcher._retry_after(None) == 60.0


# ---- matches.json, saves and zips -----------------------------------------------------------------


def _image(color, name):
    return fetcher.check_image_bytes(f"http://e.com/{name}.jpg", "http://e.com/", _jpeg(color), 10**7)


def test_matches_json_is_written_even_while_a_reader_has_it_open(tmp_path):
    folder = tmp_path / "run"
    folder.mkdir()
    st = RunStorage(folder, {"status": "running"})
    st.write_manifest()
    st.save(_image("red", "red"), 0.9, "red")
    reader = open(folder / "matches.json", "rb")  # the history list reading it (Windows refuses the replace)
    threading.Timer(0.15, reader.close).start()
    assert st.flush() and not st.dirty
    assert len(json.loads((folder / "matches.json").read_text(encoding="utf-8"))["matches"]) == 1
    assert not list(folder.glob("*.part"))


def test_matches_json_stays_dirty_when_it_cannot_be_replaced(tmp_path, monkeypatch):
    folder = tmp_path / "run"
    folder.mkdir()
    st = RunStorage(folder, {"status": "running"})
    st.write_manifest()
    st.save(_image("red", "red"), 0.9, "red")
    real = storage_mod.os.replace

    def refuse(src, dst):
        if str(dst).endswith("matches.json"):
            raise PermissionError(13, "in use")
        return real(src, dst)

    monkeypatch.setattr(storage_mod, "REPLACE_TRIES", 2)
    monkeypatch.setattr(storage_mod.os, "replace", refuse)
    try:
        st.flush()
    except PermissionError:
        pass
    assert st.dirty  # the next flush tries again instead of believing it was written
    monkeypatch.setattr(storage_mod.os, "replace", real)
    assert st.flush() and not st.dirty
    assert len(json.loads((folder / "matches.json").read_text(encoding="utf-8"))["matches"]) == 1


def test_a_file_that_cannot_be_saved_loses_only_that_match(tmp_path, fake_service, monkeypatch):
    src = tmp_path / "photos"
    src.mkdir()
    for name, red in (("red-1", 200), ("red-2", 220), ("red-3", 240)):
        _shape(src / f"{name}.jpg", "circle", (red, 0, 0))
    real = RunStorage.save
    refused: list[str] = []

    def save(self, img, confidence, evidence):
        if "red-2" in img.url:
            refused.append(img.url)
            raise PermissionError(13, "Access is denied", img.url)
        return real(self, img, confidence, evidence)

    monkeypatch.setattr(RunStorage, "save", save)

    async def go():
        run = _folder_run(tmp_path, fake_service, src)
        run.start()
        await asyncio.wait_for(run.wait(), 30)
        return run

    run = asyncio.run(go())
    assert run.status == "done", run.message
    assert run.stats["matches"] == 2 and run.stats["errors"] == 1 and refused
    assert any("could not save" in e for e in run.errors)


def test_a_full_disk_pauses_the_run_and_keeps_the_match(tmp_path, fake_service, monkeypatch):
    src = tmp_path / "photos"
    src.mkdir()
    _shape(src / "red.jpg", "circle", "red")
    real = RunStorage.save
    full = {"on": True}

    def save(self, img, confidence, evidence):
        if full["on"]:
            raise OSError(errno.ENOSPC, "No space left on device")
        return real(self, img, confidence, evidence)

    monkeypatch.setattr(RunStorage, "save", save)

    async def go():
        run = _folder_run(tmp_path, fake_service, src)
        run.start()
        for _ in range(200):
            if run.status == "paused":
                break
            await asyncio.sleep(0.05)
        assert run.status == "paused" and run.message == DISK_FULL_MESSAGE
        full["on"] = False  # space was freed
        run.resume()
        await asyncio.wait_for(run.wait(), 30)
        return run

    run = asyncio.run(go())
    assert run.status == "done" and run.stats["matches"] == 1 and run.stats["errors"] == 0


def test_cancel_waits_for_a_save_in_progress_before_the_last_manifest(tmp_path, fake_service, monkeypatch):
    src = tmp_path / "photos"
    src.mkdir()
    _shape(src / "red.jpg", "circle", "red")
    real = RunStorage.save
    started = threading.Event()

    def slow_save(self, img, confidence, evidence):
        started.set()
        time.sleep(0.5)  # still writing when Cancel arrives
        return real(self, img, confidence, evidence)

    monkeypatch.setattr(RunStorage, "save", slow_save)

    async def go():
        run = _folder_run(tmp_path, fake_service, src)
        run.start()
        while not started.is_set():
            await asyncio.sleep(0.01)
        run.cancel()
        await asyncio.sleep(0.1)
        run.cancel()  # clicked again while the run waits for the save: must not cut its cleanup short
        await asyncio.wait_for(run.wait(), 30)
        return run

    run = asyncio.run(go())
    assert run.status == "cancelled" and run.finished_at is not None
    doc = json.loads((run.storage.folder / "matches.json").read_text(encoding="utf-8"))
    assert doc["status"] == "cancelled"
    saved = [p.name for p in run.storage.folder.glob("red_*.jpg")]
    assert saved and [m["file"] for m in doc["matches"]] == saved  # the file that was written is listed


def test_a_final_manifest_that_cannot_be_written_is_reported(tmp_path, fake_service, monkeypatch):
    src = tmp_path / "photos"
    src.mkdir()
    _shape(src / "red.jpg", "circle", "red")
    real = RunStorage.write_manifest

    def write_manifest(self):
        if self.meta.get("status") != "running":  # only the last write, with the final status
            raise OSError(errno.EIO, "I/O error")
        return real(self)

    monkeypatch.setattr(RunStorage, "write_manifest", write_manifest)

    async def go():
        run = _folder_run(tmp_path, fake_service, src)
        run.start()
        await asyncio.wait_for(run.wait(), 30)
        return run

    run = asyncio.run(go())
    assert run.status == "done" and "could not be written" in run.message
    assert any("could not write matches.json" in e for e in run.errors)


def test_zip_is_streamed_entry_by_entry_without_a_temporary_file(tmp_path):
    out = tmp_path / "out"
    st = _past_run(out)  # red 0.7, blue 0.8, green 0.9
    st.save(_image("white", "w"), 0.95, "w")  # saved but not flushed yet
    gone = next(e["file"] for e in st.entries if e["file"].startswith("blue"))
    (st.folder / gone).unlink()  # deleted by hand
    before = {p for p in tmp_path.rglob("*")}
    chunks = list(st.iter_zip(0.8))
    assert {p for p in tmp_path.rglob("*")} == before  # nothing written anywhere
    assert len(chunks) >= 3  # one piece per image, then the manifest and directory
    with zipfile.ZipFile(io.BytesIO(b"".join(chunks))) as zf:
        assert zf.testzip() is None
        names = sorted(n.split("_")[0] for n in zf.namelist())
        assert names == ["green", "matches.json", "w"]
        assert all(not (i.flag_bits & 0x08) for i in zf.infolist())  # sizes in the local headers, no data descriptors
        assert len(json.loads(zf.read("matches.json"))["matches"]) == 4  # the current list, flushed or not


def test_matches_json_names_must_be_plain_file_names(tmp_path):
    folder = tmp_path / "run"
    folder.mkdir()
    (tmp_path / "secret.txt").write_text("x")
    (folder / "ok.jpg").write_bytes(b"x")
    doc = {"matches": [{"file": "../secret.txt"}, {"file": str(tmp_path / "secret.txt")}, {"file": "ok.jpg"}, {"file": 3}]}
    (folder / "matches.json").write_text(json.dumps(doc), encoding="utf-8")
    st = RunStorage.load(folder)
    assert [e["file"] for e in st.entries] == ["ok.jpg"] and not st.has_file("../secret.txt")


# ---- folder listing and the browser -------------------------------------------------------------


def test_folder_listing_stops_at_the_limit_and_skips_the_output_folder(tmp_path):
    root = tmp_path / "pics"
    (root / "b-sub").mkdir(parents=True)
    (root / "output").mkdir()
    for name in ("c.jpg", "a.png", "notes.txt", "b-sub/d.jpg", "output/e.jpg"):
        (root / name).write_bytes(b"x")
    everything = list_image_files(root, True, root / "output")
    assert [p.relative_to(root).as_posix() for p in everything] == ["a.png", "c.jpg", "b-sub/d.jpg"]
    assert len(list_image_files(root, True, root / "output", limit=2)) == 2
    assert [p.name for p in list_image_files(root, False)] == ["a.png", "c.jpg"]


def test_a_stopped_browser_is_started_again_for_the_next_run():
    class Browser:
        def __init__(self, up):
            self.up = up

        def is_connected(self):
            return self.up

    launched: list[Browser] = []

    async def relaunch():
        launched.append(Browser(True))
        return launched[-1]

    async def go():
        mgr = RunManager(AppSettings(), None, Browser(True), relaunch=relaunch)
        await mgr.ensure_browser()
        assert not launched  # still running: left alone
        mgr.browser.up = False  # Chromium crashed
        await asyncio.gather(mgr.ensure_browser(), mgr.ensure_browser())
        assert len(launched) == 1 and mgr.browser is launched[0]  # started once, even when asked twice

    asyncio.run(go())
