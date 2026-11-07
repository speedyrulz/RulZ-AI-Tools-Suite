"""Run robustness (folder scans, so no browser is needed), zips, the history cache and the host check."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import zipfile

from conftest import _shape
from PIL import Image

from crawler_app.config import AppSettings
from crawler_app.fetcher import check_image_bytes
from crawler_app.runs import Run, RunManager, RunOptions
from crawler_app.server import create_app
from crawler_app.storage import RunStorage

LOCAL = "http://127.0.0.1:8000"


def _folder_run(tmp_path, service, folder, **settings) -> Run:
    opts = RunOptions(url="", tag_id="red_shape", rubric="red", threshold=0.8, folder=str(folder))
    return Run("f1", opts, AppSettings(output_dir=tmp_path / "out", **settings), service, browser=None)


def test_huge_and_failing_images_fail_only_themselves(tmp_path, fake_service, monkeypatch):
    # Pillow opens at most 2 x MAX_IMAGE_PIXELS; 1000x800 is over that here, 640x480 is fine.
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 310_000)
    src = tmp_path / "photos"
    src.mkdir()
    _shape(src / "red-a.jpg", "circle", "red")
    _shape(src / "red-huge.png", "circle", "red", (1000, 800))
    _shape(src / "red-boom.jpg", "circle", (210, 0, 0))
    boom = hashlib.sha256((src / "red-boom.jpg").read_bytes()).hexdigest()
    real = fake_service.classify

    async def classify(images, tags, threshold=None):
        if any(i.id == boom for i in images):
            raise RuntimeError("a bug in the engine")
        return await real(images, tags, threshold)

    fake_service.classify = classify

    async def go():
        run = _folder_run(tmp_path, fake_service, src)
        run.start()
        await asyncio.wait_for(run.wait(), 30)
        return run

    run = asyncio.run(go())
    assert run.status == "done", run.message
    assert [m["file"].rsplit("_", 1)[0] for m in run.matches] == ["red-a"]
    assert run.stats["fetch_errors"] == 1 and run.stats["errors"] == 1
    assert any("too large to check" in e for e in run.errors)
    assert any("RuntimeError" in e for e in run.errors)


def test_pause_while_starting_stays_paused_until_resume(tmp_path, fake_service):
    src = tmp_path / "photos"
    src.mkdir()
    _shape(src / "red.jpg", "circle", "red")

    async def go():
        run = _folder_run(tmp_path, fake_service, src)
        run.start()
        run.pause()  # pressed while the page still says "starting"
        await asyncio.sleep(0.5)
        assert run.status == "paused" and run.stats["images_found"] == 0
        run.resume()
        await asyncio.wait_for(run.wait(), 30)
        return run

    run = asyncio.run(go())
    assert run.status == "done" and run.stats["matches"] == 1


def test_each_result_shows_without_waiting_for_slower_images(tmp_path, fake_service):
    src = tmp_path / "photos"
    src.mkdir()
    for name, red in (("red-1", 200), ("red-2", 220), ("red-slow", 240)):
        _shape(src / f"{name}.jpg", "circle", (red, 0, 0))
    slow = hashlib.sha256((src / "red-slow.jpg").read_bytes()).hexdigest()
    real = fake_service.classify
    gate: dict[str, asyncio.Event] = {}

    async def classify(images, tags, threshold=None):
        if any(i.id == slow for i in images):
            await gate["release"].wait()
        return await real(images, tags, threshold)

    fake_service.classify = classify

    async def go():
        gate["release"] = asyncio.Event()
        run = _folder_run(tmp_path, fake_service, src)
        run.start()
        for _ in range(200):  # the other two are saved while the slow one is still being judged
            if run.stats["matches"] == 2:
                break
            await asyncio.sleep(0.05)
        assert run.stats["matches"] == 2 and run.status == "running"
        gate["release"].set()
        await asyncio.wait_for(run.wait(), 30)
        return run

    assert asyncio.run(go()).stats["matches"] == 3


def test_run_manager_keeps_only_the_newest_finished_runs():
    class FakeRun:
        def __init__(self, i, finished):
            self.id, self.finished = str(i), finished

    mgr = RunManager(AppSettings(), None, None)
    for i in range(15):
        mgr.runs[str(i)] = FakeRun(i, i != 3)  # run 3 is still going
    mgr._prune()
    assert "3" in mgr.runs and "0" not in mgr.runs and "14" in mgr.runs
    assert sum(r.finished for r in mgr.runs.values()) == RunManager.KEEP_FINISHED


# ---- storage and API ------------------------------------------------------------------------------


def _jpeg(color) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (200, 150), color).save(buf, "JPEG")
    return buf.getvalue()


def _past_run(out, name="2026-11-01_1200_e-com_red_shape") -> RunStorage:
    folder = out / name
    folder.mkdir(parents=True)
    st = RunStorage(folder, {"status": "done", "started": "2026-11-01T12:00:00", "tag": {"id": "red_shape"}})
    for i, c in enumerate(("red", "blue", "green")):
        st.save(check_image_bytes(f"http://e.com/{c}.jpg", "http://e.com/", _jpeg(c), 10**7), 0.7 + i * 0.1, c)
    st.write_manifest()
    return st


def test_zips_get_their_own_file_outside_the_run_folder(tmp_path):
    st = _past_run(tmp_path / "out")
    zips = tmp_path / "out" / ".zips"
    first = st.build_zip(0.8, zips)
    with open(first, "rb"):  # still being downloaded: a second download of the same zip doesn't clash
        second = st.build_zip(0.8, zips)
    assert first != second
    with zipfile.ZipFile(second) as zf:
        assert sorted(n.split("_")[0] for n in zf.namelist()) == ["blue", "green", "matches.json"]
    assert not list(st.folder.glob("*.zip"))


def test_api_zip_history_cache_and_this_pc_only(tmp_path, fake_service):
    from fastapi.testclient import TestClient

    out = tmp_path / "out"
    st = _past_run(out)
    name = st.folder.name
    (out / ".zips").mkdir()
    (out / ".zips" / "left-over.zip").write_bytes(b"x")  # from a download cut off before a restart
    app = create_app(AppSettings(output_dir=out), service=fake_service)

    with TestClient(app, base_url=LOCAL) as c:
        assert not (out / ".zips" / "left-over.zip").exists()

        for _ in range(2):
            z = c.get(f"/api/history/{name}/zip", params={"min_confidence": 0.8})
            assert z.status_code == 200 and len(zipfile.ZipFile(io.BytesIO(z.content)).namelist()) == 3
        assert not list((out / ".zips").glob("*.zip")) and not list(st.folder.glob("*.zip"))  # deleted once sent

        files = [m["file"] for m in c.get(f"/api/history/{name}").json()["matches"]]
        assert all(c.get(f"/api/history/{name}/files/{f}").status_code == 200 for f in files)
        (st.folder / files[0]).unlink()  # deleted by hand while the listing is cached
        assert c.get(f"/api/history/{name}/files/{files[0]}").status_code == 404
        # matches.json changed on disk: the cached listing is read again
        doc = json.loads((st.folder / "matches.json").read_text(encoding="utf-8"))
        doc["matches"] = doc["matches"][1:]
        (st.folder / "matches.json").write_text(json.dumps(doc), encoding="utf-8")
        assert len(c.get(f"/api/history/{name}").json()["matches"]) == 2
        assert [h["matches"] for h in c.get("/api/history").json()] == [2]

        # Other sites' pages can't send requests (the Origin check)...
        evil = c.post("/api/rubric", json={"prompt": "x"}, headers={"Origin": "https://evil.example"})
        assert evil.status_code == 403
        assert c.post("/api/rubric", json={"prompt": "x"}, headers={"Origin": LOCAL}).status_code == 200

    # ...and a web page whose domain was switched to 127.0.0.1 can't read anything (the Host check).
    with TestClient(app, base_url="http://evil.example:8000") as c:
        assert c.get("/api/health").status_code == 403
        assert c.get("/api/history").status_code == 403
    with TestClient(create_app(AppSettings(output_dir=out), service=fake_service, allowed_hosts=["*"])) as c:
        assert c.get("/api/health").status_code == 200  # e.g. started with --host 0.0.0.0
