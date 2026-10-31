"""Pipeline and API tests against a real local site and real Chromium, with a fake detection service."""

from __future__ import annotations

import asyncio
import io
import json
import time
import zipfile

from playwright.async_api import async_playwright

from crawler_app.config import AppSettings
from crawler_app.runs import Run, RunOptions
from crawler_app.server import create_app


def _settings(tmp_path) -> AppSettings:
    return AppSettings(output_dir=tmp_path / "out", chunk_wait_s=0.1, fetch_concurrency=3, max_scroll_steps=15)


def _opts(site_url: str, **kw) -> RunOptions:
    base = dict(url=site_url, tag_id="red_shape", rubric="red", threshold=0.8, page_delay_s=0.0)
    return RunOptions(**{**base, **kw})


async def _with_run(tmp_path, service, opts, during=None) -> Run:
    """Start a Run in a fresh browser; `during(run)` can poke at it while it runs."""
    async with async_playwright() as pw:
        browser = await pw.chromium.launch()
        try:
            run = Run("t1", opts, _settings(tmp_path), service, browser)
            run.start()
            if during:
                await during(run)
            await asyncio.wait_for(run.wait(), 60)
            return run
        finally:
            await browser.close()


async def _wait_for(pred, timeout=20.0):
    end = time.monotonic() + timeout
    while not pred():
        if time.monotonic() > end:
            raise AssertionError("condition not reached in time")
        await asyncio.sleep(0.05)


def test_full_crawl_finds_saves_and_filters(tmp_path, site_url, fake_service):
    run = asyncio.run(_with_run(tmp_path, fake_service, _opts(site_url, respect_robots=True)))
    assert run.status == "done", run.message

    files = sorted(m["file"] for m in run.matches)
    stems = sorted(f.rsplit("_", 1)[0] for f in files)
    # Largest srcset candidate, lazy data-src, CSS background, page-2 webp, the oddly named file,
    # and two full-size originals (linked from a thumbnail; WordPress-style "-300x225" removed).
    assert stems == [
        "bg-red-circle", "my-photo-1", "page2-red-circle", "red-circle-large", "red-circle-lazy",
        "red-gallery", "red-photo",
    ]
    assert not any("red-circle-small" in f for f in files)
    assert any(f.endswith(".webp") for f in files)

    s = run.stats
    assert s["pages_crawled"] == 2  # index + page2; page3 is disallowed by robots.txt, PDF/external never followed
    assert s["pages_blocked"] == 1
    assert s["duplicates"] == 1  # dup-of-red.jpg has the same bytes as red-circle-lazy.jpg
    assert s["skipped"] == 1  # tiny.png, by its rendered size, before any fetch
    assert s["fetch_errors"] == 1  # logo.svg
    assert s["matches"] == 7
    assert s["originals"] == 2  # blue-only-300x225 has no original and fell back to itself
    # green square matched at 0.75, below this run's 0.8 threshold, so it was classified but not saved
    assert s["classified"] == 11  # 9 distinct decodable images on index + 2 on page2

    # The engine was asked with threshold 0 for raw answers, and never saw the duplicate twice.
    assert len(fake_service.seen_ids) == len(set(fake_service.seen_ids))

    folder = run.storage.folder
    on_disk = sorted(p.name for p in folder.iterdir() if p.suffix != ".json")
    assert on_disk == files
    manifest = json.loads((folder / "matches.json").read_text(encoding="utf-8"))
    assert len(manifest["matches"]) == 7
    assert all(m["confidence"] >= 0.8 for m in manifest["matches"])
    assert manifest["status"] == "done" and manifest["stats"]["matches"] == 7
    gallery = next(m for m in manifest["matches"] if m["file"].startswith("red-gallery_"))
    assert gallery["original"] is True and (gallery["width"], gallery["height"]) == (1600, 1200)
    for name in files:
        assert all(ch.isalnum() or ch in "-_." for ch in name)


def test_lower_threshold_saves_green(tmp_path, site_url, fake_service):
    run = asyncio.run(_with_run(tmp_path, fake_service, _opts(site_url, threshold=0.7, max_pages=1)))
    assert any(m["file"].startswith("green-square_") for m in run.matches)


def test_model_down_pauses_then_resume_finishes(tmp_path, site_url, fake_service):
    fake_service.down = True

    async def during(run: Run):
        await _wait_for(lambda: run.status == "paused")
        assert "model" in run.message.lower()
        await asyncio.sleep(0.3)
        assert run.status == "paused"  # stays paused; no silent empty result
        fake_service.down = False
        run.resume()

    run = asyncio.run(_with_run(tmp_path, fake_service, _opts(site_url), during))
    assert run.status == "done"
    assert run.stats["matches"] == 7


def test_cancel_stops_cleanly(tmp_path, site_url, fake_service):
    async def during(run: Run):
        await _wait_for(lambda: run.stats["pages_crawled"] >= 1)
        start = time.monotonic()
        run.cancel()
        await asyncio.wait_for(run.wait(), 5)
        assert time.monotonic() - start < 5

    # A long page delay keeps the crawl busy so cancel lands mid-run.
    run = asyncio.run(_with_run(tmp_path, fake_service, _opts(site_url, page_delay_s=30), during))
    assert run.status == "cancelled"
    assert run.finished_at is not None


def test_max_images_caps_the_crawl(tmp_path, site_url, fake_service):
    run = asyncio.run(_with_run(tmp_path, fake_service, _opts(site_url, max_images=2)))
    assert run.status == "done"
    assert run.stats["images_found"] == 2
    assert run.stats["pages_crawled"] == 1


def test_api_end_to_end(tmp_path, site_url, fake_service):
    from fastapi.testclient import TestClient

    app = create_app(_settings(tmp_path), service=fake_service)
    with TestClient(app) as client:
        assert client.get("/api/health").json()["server"] == "up"

        r = client.post("/api/rubric", json={"prompt": "red hands"}).json()
        assert r["id"] == "red_shape" and r["suggested_threshold"] == 0.8  # body-detail prompt

        bad = client.post("/api/runs", json={"url": "ftp://x", "tag": {"id": "Bad Id", "rubric": "x"}})
        assert bad.status_code == 422

        host_only = site_url.removeprefix("http://").rstrip("/")  # no scheme: https first, then http fallback
        run_id = client.post(
            "/api/runs",
            json={"url": host_only, "tag": {"id": r["id"], "rubric": r["rubric"]}, "threshold": 0.8, "page_delay_s": 0},
        ).json()["run_id"]

        events = []
        with client.stream("GET", f"/api/runs/{run_id}/events") as stream:
            for line in stream.iter_lines():
                if line.startswith("data: "):
                    events.append(json.loads(line[6:]))
        types = [e["type"] for e in events]
        assert types[0] == "snapshot"
        assert types.count("match") + len(events[0]["run"]["matches"]) >= 5
        assert any(e["type"] == "status" and e["status"] == "done" for e in events)

        snap = client.get(f"/api/runs/{run_id}").json()
        assert snap["status"] == "done" and snap["stats"]["matches"] == 7

        name = snap["matches"][0]["file"]
        assert client.get(f"/api/runs/{run_id}/files/{name}").status_code == 200
        assert client.get(f"/api/runs/{run_id}/files/..%2Fmatches.json").status_code == 404

        z = client.get(f"/api/runs/{run_id}/zip", params={"min_confidence": 0.9})
        names = zipfile.ZipFile(io.BytesIO(z.content)).namelist()
        assert len(names) == 8 and "matches.json" in names  # 7 images (all 0.95) + manifest

        # History: the finished run is listed from disk and can be reopened, served and zipped.
        (tmp_path / "out" / "old_interrupted").mkdir()
        (tmp_path / "out" / "old_interrupted" / "matches.json").write_text(
            json.dumps({"start_url": "http://x/", "status": "running", "matches": []})
        )
        hist = client.get("/api/history").json()
        folder = snap["folder"].replace("\\", "/").rsplit("/", 1)[-1]
        item = next(h for h in hist if h["id"] == folder)
        assert item["status"] == "done" and item["matches"] == 7 and item["run_id"] == run_id
        assert next(h for h in hist if h["id"] == "old_interrupted")["status"] == "interrupted"
        past = client.get(f"/api/history/{folder}").json()
        assert len(past["matches"]) == 7 and past["tag"]["id"] == r["id"]
        assert client.get(f"/api/history/{folder}/files/{past['matches'][0]['file']}").status_code == 200
        assert client.get(f"/api/history/{folder}/files/matches.json").status_code == 404
        assert client.get("/api/history/..").status_code == 404
        assert client.get("/api/history/nope").status_code == 404
        hz = client.get(f"/api/history/{folder}/zip", params={"min_confidence": 0.9})
        assert len(zipfile.ZipFile(io.BytesIO(hz.content)).namelist()) == 8

        # Model down: new runs are refused with 503 rather than starting and finding nothing.
        fake_service.down = True
        down = client.post("/api/runs", json={"url": site_url, "tag": {"id": "x", "rubric": "x"}})
        assert down.status_code == 503 and down.json() == {"error": "model server down"}
        assert client.post("/api/rubric", json={"prompt": "x"}).status_code == 503


def test_originals_off_keeps_displayed_copies(tmp_path, site_url, fake_service):
    run = asyncio.run(_with_run(tmp_path, fake_service, _opts(site_url, max_pages=1, prefer_originals=False)))
    stems = {m["file"].rsplit("_", 1)[0] for m in run.matches}
    assert {"red-gallery-thumb", "red-photo-300x225"} <= stems
    assert run.stats["originals"] == 0


def test_min_megapixels_skips_small_images(tmp_path, site_url, fake_service):
    run = asyncio.run(_with_run(tmp_path, fake_service, _opts(site_url, min_megapixels=1.0)))
    assert run.status == "done"
    # Only red-circle-large (1.23 MP), red-gallery (1.92 MP) and red-photo (1.08 MP) are big enough.
    assert sorted(m["file"].rsplit("_", 1)[0] for m in run.matches) == ["red-circle-large", "red-gallery", "red-photo"]
    assert all(m["width"] * m["height"] >= 1_000_000 for m in run.matches)
    assert run.stats["classified"] == 3 and run.stats["too_small_mp"] == 8


def test_folder_scan(tmp_path, fake_service):
    from conftest import _shape

    src = tmp_path / "photos"
    (src / "sub").mkdir(parents=True)
    _shape(src / "red one.jpg", "circle", "red")
    _shape(src / "blue.png", "square", "blue")
    _shape(src / "sub" / "red-two.webp", "circle", "red", fmt="WEBP")
    (src / "z-copy.jpg").write_bytes((src / "red one.jpg").read_bytes())  # duplicate
    _shape(src / "icon.png", "circle", "red", (40, 40))  # too small
    (src / "broken.jpg").write_bytes(b"not an image")
    (src / "notes.txt").write_text("hello")
    # The app's own output inside the scanned folder is never re-read.
    settings_out = src / "out"
    (settings_out / "old-run").mkdir(parents=True)
    _shape(settings_out / "old-run" / "red-saved.jpg", "circle", "red")

    async def go(opts):
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            try:
                s = AppSettings(output_dir=settings_out, chunk_wait_s=0.1)
                run = Run("f1", opts, s, fake_service, browser)
                run.start()
                await asyncio.wait_for(run.wait(), 60)
                return run
            finally:
                await browser.close()

    opts = RunOptions(url="", tag_id="red_shape", rubric="red", threshold=0.8, folder=str(src))
    run = asyncio.run(go(opts))
    assert run.status == "done", run.message
    assert sorted(m["file"].rsplit("_", 1)[0] for m in run.matches) == ["red-one", "red-two"]
    s = run.stats
    assert s["images_found"] == 6 and s["duplicates"] == 1 and s["fetch_errors"] == 1 and s["skipped"] == 1
    assert s["classified"] == 3  # red one, blue, red-two
    assert run.storage.folder.name.split("_")[2] == "folder-photos"
    assert run.matches[0]["url"].endswith(".jpg") or run.matches[0]["url"].endswith(".webp")
    assert (src / "red one.jpg").exists()  # originals are copied, never moved

    flat = asyncio.run(go(RunOptions(url="", tag_id="red_shape", rubric="red", threshold=0.8, folder=str(src),
                                     recursive=False)))
    assert [m["file"].rsplit("_", 1)[0] for m in flat.matches] == ["red-one"]


def test_api_folder_validation(tmp_path, fake_service):
    from fastapi.testclient import TestClient

    app = create_app(_settings(tmp_path), service=fake_service)
    tag = {"id": "x", "rubric": "x"}
    with TestClient(app) as client:
        assert client.post("/api/runs", json={"tag": tag}).status_code == 422  # neither
        both = {"url": "https://a.com", "folder": str(tmp_path), "tag": tag}
        assert client.post("/api/runs", json=both).status_code == 422
        missing = client.post("/api/runs", json={"folder": str(tmp_path / "nope"), "tag": tag})
        assert missing.status_code == 422 and "folder not found" in missing.text
        ok = client.post("/api/runs", json={"folder": f'"{tmp_path}"', "tag": tag})  # pasted with quotes
        assert ok.status_code == 200


def test_storage_flushes_manifest_once_and_zips_a_snapshot(tmp_path):
    from crawler_app.fetcher import check_image_bytes
    from crawler_app.storage import RunStorage

    from PIL import Image

    def jpeg(color):
        buf = io.BytesIO()
        Image.new("RGB", (200, 150), color).save(buf, "JPEG")
        return buf.getvalue()

    folder = tmp_path / "run"
    folder.mkdir()
    st = RunStorage(folder, {"status": "running"})
    st.write_manifest()
    for i, c in enumerate(("red", "blue", "green")):
        st.save(check_image_bytes(f"http://x/{c}.jpg", "http://x/", jpeg(c), 10**7), 0.5 + i * 0.2, c)
    # saves don't rewrite matches.json each time; flush writes it once
    assert json.loads((folder / "matches.json").read_text(encoding="utf-8"))["matches"] == []
    assert st.dirty and st.flush() and not st.dirty and not st.flush()
    assert len(json.loads((folder / "matches.json").read_text(encoding="utf-8"))["matches"]) == 3

    st.save(check_image_bytes("http://x/w.jpg", "http://x/", jpeg("white"), 10**7), 0.95, "w")  # not flushed
    with zipfile.ZipFile(st.build_zip(0.8)) as zf:
        names = sorted(zf.namelist())
        doc = json.loads(zf.read("matches.json"))
    assert len(names) == 3 and "matches.json" in names  # green (0.9) and white (0.95) + manifest
    assert len(doc["matches"]) == 4  # the zip's manifest is the current list, flushed or not
    assert not list(folder.glob("*.part"))
