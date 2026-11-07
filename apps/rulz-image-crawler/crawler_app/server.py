"""FastAPI app for the crawler UI. Bind to 127.0.0.1 only.

The detection engine runs in-process (image_detection.DetectionService); only llama-server is a
separate process.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from image_detection import DetectionService, ModelError, ModelServerDown
from image_detection.rubrics import TAG_ID_RE
from pydantic import BaseModel, Field, field_validator, model_validator
from starlette.background import BackgroundTask

from .config import AppSettings
from .folder import pick_folder
from .runs import FINAL_STATUSES, Run, RunManager, RunOptions
from .storage import RunStorage, sweep_zips
from .urls import normalize_page_url

log = logging.getLogger("crawler_app.server")

STATIC_DIR = Path(__file__).resolve().parent / "static"
# Zips being downloaded are built here (inside the output folder, so on the same drive) and deleted once sent.
ZIP_DIR_NAME = ".zips"
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

# The spec: the model is overconfident on small regions, so body-detail tags default to 0.8.
_BODY_DETAIL = re.compile(
    r"\b(hands?|fingers?|finger ?nails?|nails?|eyes?|eyelashes|eyebrows?|teeth|tooth|lips?|ears?|nose|"
    r"toes?|feet|foot|freckles?|tattoos?|piercings?|wrists?|knuckles?|moles?|scars?)\b",
    re.IGNORECASE,
)


def suggest_threshold(text: str, rubric: str = "") -> float:
    return 0.8 if _BODY_DETAIL.search(text) or _BODY_DETAIL.search(rubric.split("EXCLUDE:")[0]) else 0.7


class RubricIn(BaseModel):
    prompt: str = Field(min_length=1, max_length=300)


class TagIn(BaseModel):
    id: str
    rubric: str = Field(min_length=1, max_length=2000)

    @field_validator("id")
    @classmethod
    def _valid_id(cls, v: str) -> str:
        if not TAG_ID_RE.match(v):
            raise ValueError("tag id must be snake_case: lowercase letters, digits, underscores (max 64)")
        return v


class RunIn(BaseModel):
    url: str | None = Field(default=None, max_length=2000)  # a website, or...
    folder: str | None = Field(default=None, max_length=1000)  # ...a folder on this PC
    recursive: bool = True
    tag: TagIn
    threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    max_pages: int = Field(default=10, ge=1, le=1000)
    max_depth: int = Field(default=3, ge=0, le=20)
    max_images: int = Field(default=300, ge=1, le=10000)
    respect_robots: bool = False
    stay_in_path: bool = False
    page_delay_s: float = Field(default=1.0, ge=0.0, le=30.0)
    prefer_originals: bool = True
    min_megapixels: float = Field(default=0.0, ge=0.0, le=200.0)

    @field_validator("url")
    @classmethod
    def _valid_url(cls, v: str | None) -> str | None:
        if v is None or not v.strip():
            return None
        v = v.strip()
        if "://" not in v:
            v = "https://" + v
        url = normalize_page_url(v)
        if not url:
            raise ValueError("enter an http(s) website address")
        return url

    @field_validator("folder")
    @classmethod
    def _valid_folder(cls, v: str | None) -> str | None:
        if v is None or not v.strip():
            return None
        path = Path(v.strip().strip('"')).expanduser()
        if not path.is_dir():
            raise ValueError(f"folder not found: {path}")
        return str(path.resolve())

    @model_validator(mode="after")
    def _one_source(self) -> RunIn:
        if bool(self.url) == bool(self.folder):
            raise ValueError("give either a website address or a folder")
        return self


_FOLDER_RE = re.compile(r"^[\w.-]{1,200}$")


def _summary(name: str, doc: dict[str, Any]) -> dict[str, Any]:
    tag = doc.get("tag") or {}
    return {
        "id": name,
        "started": doc.get("started"),
        "finished": doc.get("finished"),
        "start_url": doc.get("start_url"),
        "source": doc.get("source", "web"),
        "tag": tag.get("id"),
        "threshold": doc.get("threshold"),
        "status": doc.get("status", "done"),
        "matches": len(doc.get("matches") or []),
        "stats": doc.get("stats"),
    }


def _stamp(manifest: Path) -> tuple[int, int]:
    st = manifest.stat()
    return st.st_mtime_ns, st.st_size


class HistoryIndex:
    """The run folders under output_dir, read from disk and cached until their matches.json changes:
    a past search with thousands of matches is parsed once, not once per thumbnail, and the history
    list only re-reads the runs that changed."""

    def __init__(self, output_dir: Path, keep_runs: int = 16):
        self.output_dir = output_dir
        self.keep_runs = keep_runs
        self._lock = threading.Lock()
        self._runs: OrderedDict[str, tuple[tuple[int, int], RunStorage]] = OrderedDict()
        self._summaries: dict[str, tuple[tuple[int, int], dict[str, Any]]] = {}

    def storage(self, name: str) -> RunStorage:
        """Raises FileNotFoundError if the folder has no matches.json, ValueError if it is unreadable."""
        folder = self.output_dir / name
        stamp = _stamp(folder / "matches.json")
        with self._lock:
            hit = self._runs.get(name)
            if hit is not None and hit[0] == stamp:
                self._runs.move_to_end(name)
                return hit[1]
        st = RunStorage.load(folder)
        with self._lock:
            self._runs[name] = (stamp, st)
            self._runs.move_to_end(name)
            while len(self._runs) > self.keep_runs:
                self._runs.popitem(last=False)
        return st

    def summaries(self, limit: int = 300) -> list[dict[str, Any]]:
        """One summary per run folder (those with a matches.json), newest first."""
        items: list[dict[str, Any]] = []
        found: set[str] = set()
        if self.output_dir.is_dir():
            for folder in self.output_dir.iterdir():
                manifest = folder / "matches.json"
                try:
                    stamp = _stamp(manifest)
                except OSError:
                    continue
                found.add(folder.name)
                with self._lock:
                    hit = self._summaries.get(folder.name)
                if hit is None or hit[0] != stamp:
                    try:
                        doc = json.loads(manifest.read_text(encoding="utf-8"))
                    except (OSError, ValueError):
                        continue
                    if not isinstance(doc, dict):
                        continue
                    hit = (stamp, _summary(folder.name, doc))
                    with self._lock:
                        self._summaries[folder.name] = hit
                items.append(dict(hit[1]))  # a copy: callers add the live run's state to it
        with self._lock:
            for gone in set(self._summaries) - found:
                del self._summaries[gone]
        items.sort(key=lambda i: (i["started"] or "", i["id"]), reverse=True)
        return items[:limit]


def _host_name(host_header: str) -> str:
    try:
        return (urlsplit("//" + host_header).hostname or "").lower()
    except ValueError:
        return ""


def _remove(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:  # pragma: no cover - swept on the next download or start
        pass


def _sse(event: dict[str, Any]) -> str:
    return f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"


def create_app(
    settings: AppSettings | None = None,
    service: DetectionService | None = None,
    browser: Any | None = None,
    allowed_hosts: Iterable[str] | None = None,
) -> FastAPI:
    """`service` and `browser` can be injected (tests); otherwise the real engine and Chromium are started.
    `allowed_hosts`: host names the app answers to besides this PC's own (127.0.0.1, localhost, ::1);
    "*" allows any. Default: CRAWLER_ALLOWED_HOSTS, comma-separated."""
    settings = settings or AppSettings()
    if allowed_hosts is None:
        allowed_hosts = os.environ.get("CRAWLER_ALLOWED_HOSTS", "").split(",")
    hosts = LOOPBACK_HOSTS | {h.strip().lower() for h in allowed_hosts if h.strip()}
    history = HistoryIndex(settings.output_dir)
    zip_dir = settings.output_dir / ZIP_DIR_NAME

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await asyncio.to_thread(sweep_zips, zip_dir, 0)  # no download is running yet
        svc = service or DetectionService()
        pw = None
        b = browser
        if b is None:
            from playwright.async_api import async_playwright

            pw = await async_playwright().start()
            b = await pw.chromium.launch(headless=settings.headless)
        app.state.manager = RunManager(settings, svc, b)
        try:
            yield
        finally:
            await app.state.manager.shutdown()
            if pw is not None:
                await b.close()
                await pw.stop()
            if service is None:
                await svc.aclose()

    app = FastAPI(title="RulZ Image Crawler", lifespan=lifespan)

    @app.middleware("http")
    async def _this_pc_only(request: Request, call_next):
        """Only pages served by this app may use it. The Host check stops DNS rebinding (a web page whose
        domain is switched to 127.0.0.1 to read the API and start folder scans); the Origin check stops
        other sites' pages from sending POSTs (starting runs, opening the folder picker)."""
        if "*" not in hosts:
            host = request.headers.get("host", "")
            if _host_name(host) not in hosts:
                return JSONResponse(status_code=403, content={"error": "unknown host"})
            origin = request.headers.get("origin")
            if origin is not None and request.method not in ("GET", "HEAD", "OPTIONS"):
                if urlsplit(origin).netloc.lower() != host.lower():
                    return JSONResponse(status_code=403, content={"error": "cross-site request refused"})
        return await call_next(request)

    def _manager(request: Request) -> RunManager:
        return request.app.state.manager

    def _run(request: Request, run_id: str) -> Run:
        run = _manager(request).get(run_id)
        if run is None:
            raise HTTPException(404, "no such run")
        return run

    @app.exception_handler(ModelServerDown)
    async def _down(_: Request, exc: ModelServerDown):
        log.warning("model server down: %s", exc)
        return JSONResponse(status_code=503, content={"error": "model server down"})

    @app.get("/api/health")
    async def health(request: Request):
        return await _manager(request).svc.health()

    @app.post("/api/rubric")
    async def rubric(body: RubricIn, request: Request):
        try:
            out = await _manager(request).svc.expand_tag(body.prompt)
        except ValueError as e:
            raise HTTPException(422, str(e))
        except ModelError as e:
            raise HTTPException(502, str(e))
        return {**out, "suggested_threshold": suggest_threshold(body.prompt, out["rubric"])}

    @app.post("/api/runs")
    async def create_run(body: RunIn, request: Request):
        mgr = _manager(request)
        if (await mgr.svc.health())["server"] != "up":
            raise ModelServerDown("health check failed")
        run = mgr.create(
            RunOptions(
                url=body.url or "",
                folder=body.folder,
                recursive=body.recursive,
                tag_id=body.tag.id,
                rubric=body.tag.rubric,
                threshold=body.threshold,
                max_pages=body.max_pages,
                max_depth=body.max_depth,
                max_images=body.max_images,
                respect_robots=body.respect_robots,
                stay_in_path=body.stay_in_path,
                page_delay_s=body.page_delay_s,
                prefer_originals=body.prefer_originals,
                min_megapixels=body.min_megapixels,
            )
        )
        return {"run_id": run.id}

    @app.post("/api/pick-folder")
    async def pick_folder_dialog():
        try:
            return {"folder": await asyncio.to_thread(pick_folder)}
        except Exception as e:  # noqa: BLE001 - no desktop session, Tk missing, ...
            raise HTTPException(501, f"folder picker unavailable: {e}")

    @app.get("/api/runs")
    async def list_runs(request: Request):
        return [
            {"id": r.id, "status": r.status, "url": r.opts.folder or r.opts.url, "tag": r.opts.tag_id,
             "matches": r.stats["matches"]}
            for r in _manager(request).runs.values()
        ]

    @app.get("/api/runs/{run_id}")
    async def get_run(run_id: str, request: Request):
        return _run(request, run_id).snapshot()

    @app.get("/api/runs/{run_id}/events")
    async def events(run_id: str, request: Request):
        run = _run(request, run_id)

        async def stream():
            q = run.subscribe()  # subscribe before the snapshot so no event is missed
            try:
                yield _sse({"type": "snapshot", "run": run.snapshot()})
                if run.finished:
                    return
                while True:
                    if await request.is_disconnected():
                        return
                    try:
                        # short, so a closed tab is noticed (and unsubscribed) within seconds
                        ev = await asyncio.wait_for(q.get(), 5)
                    except TimeoutError:
                        yield ": keep-alive\n\n"
                        continue
                    yield _sse(ev)
                    if ev["type"] == "status" and ev["status"] in FINAL_STATUSES:
                        # the stats event published with it follows immediately
                        while not q.empty():
                            yield _sse(q.get_nowait())
                        return
            finally:
                run.unsubscribe(q)

        return StreamingResponse(
            stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        )

    @app.post("/api/runs/{run_id}/pause")
    async def pause(run_id: str, request: Request):
        run = _run(request, run_id)
        run.pause()
        return {"status": run.status}

    @app.post("/api/runs/{run_id}/resume")
    async def resume(run_id: str, request: Request):
        run = _run(request, run_id)
        if run.status == "paused" and (await run.svc.health())["server"] != "up":
            raise ModelServerDown("health check failed on resume")
        run.resume()
        return {"status": run.status}

    @app.post("/api/runs/{run_id}/cancel")
    async def cancel(run_id: str, request: Request):
        run = _run(request, run_id)
        run.cancel()
        await run.wait()
        return {"status": run.status}

    async def _send_zip(st: RunStorage, min_confidence: float) -> FileResponse:
        await asyncio.to_thread(sweep_zips, zip_dir)  # leftovers of downloads that were cut off
        path = await asyncio.to_thread(st.build_zip, min_confidence, zip_dir)
        return FileResponse(
            path,
            media_type="application/zip",
            filename=f"{st.folder.name}.zip",
            background=BackgroundTask(_remove, path),  # each download gets its own file, deleted once sent
        )

    @app.get("/api/runs/{run_id}/zip")
    async def download_zip(run_id: str, request: Request, min_confidence: float = 0.0):
        run = _run(request, run_id)
        if run.storage is None:
            raise HTTPException(409, "run has no output yet")
        return await _send_zip(run.storage, min_confidence)

    @app.get("/api/runs/{run_id}/files/{name}")
    async def download_file(run_id: str, name: str, request: Request):
        run = _run(request, run_id)
        # Only names this run saved are served, so no path can escape the run folder.
        if run.storage is None or not run.storage.has_file(name):
            raise HTTPException(404, "no such file")
        if not await asyncio.to_thread((run.storage.folder / name).is_file):  # deleted by hand
            raise HTTPException(404, "no such file")
        return FileResponse(run.storage.folder / name)

    # ---- history: every run folder on disk, including ones from before this app was started ----

    def _history_storage(name: str) -> RunStorage:
        if not _FOLDER_RE.match(name) or name.strip(".") == "":
            raise HTTPException(404, "no such run folder")
        try:
            return history.storage(name)
        except FileNotFoundError:
            raise HTTPException(404, "no such run folder")
        except (OSError, ValueError) as e:
            raise HTTPException(500, f"could not read matches.json: {e}")

    @app.get("/api/history")
    async def history_list(request: Request):
        live = {r.storage.folder.name: r for r in _manager(request).runs.values() if r.storage}
        items = await asyncio.to_thread(history.summaries)
        for item in items:
            run = live.get(item["id"])
            if run is not None:
                item.update(status=run.status, run_id=run.id, matches=run.stats["matches"])
            elif item["status"] in ("running", "paused", "starting"):
                item["status"] = "interrupted"  # the app stopped while it ran; its matches are kept
        return items

    @app.get("/api/history/{name}")
    async def history_run(name: str):
        st = await asyncio.to_thread(_history_storage, name)
        return {**st.meta, "id": name, "folder": str(st.folder), "matches": st.entries}

    @app.get("/api/history/{name}/files/{file}")
    async def history_file(name: str, file: str):
        st = await asyncio.to_thread(_history_storage, name)
        # Only names listed in matches.json, so no path can escape; the listing is cached, so check
        # the file is still there (it may have been deleted by hand since).
        if not st.has_file(file) or not await asyncio.to_thread((st.folder / file).is_file):
            raise HTTPException(404, "no such file")
        return FileResponse(st.folder / file)

    @app.get("/api/history/{name}/zip")
    async def history_zip(name: str, min_confidence: float = 0.0):
        st = await asyncio.to_thread(_history_storage, name)
        return await _send_zip(st, min_confidence)

    if (STATIC_DIR / "index.html").exists():
        app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="ui")
    else:

        @app.get("/")
        async def root():
            return {"app": "prompt-gated image crawler", "ui": "not built yet (Phase 3)", "api": "/docs"}

    return app


def main() -> None:
    import argparse

    import uvicorn

    p = argparse.ArgumentParser(description="Prompt-gated image crawler (local web app)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--log-level", default="info")
    args = p.parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # The UI polls /api/health every 5 s; don't log every llama-server request that causes.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    # Listening on another address means the person wants to reach it by that name (or, for 0.0.0.0,
    # from anywhere on the network): let those requests through the host check.
    extra = os.environ.get("CRAWLER_ALLOWED_HOSTS", "").split(",")
    if args.host not in LOOPBACK_HOSTS:
        extra.append("*" if args.host in ("0.0.0.0", "::") else args.host)
    uvicorn.run(create_app(allowed_hosts=extra), host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
