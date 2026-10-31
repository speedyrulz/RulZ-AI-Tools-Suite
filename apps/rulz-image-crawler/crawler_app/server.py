"""FastAPI app for the crawler UI. Bind to 127.0.0.1 only.

The detection engine runs in-process (image_detection.DetectionService); only llama-server is a
separate process.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from image_detection import DetectionService, ModelError, ModelServerDown
from image_detection.rubrics import TAG_ID_RE
from pydantic import BaseModel, Field, field_validator, model_validator

from .config import AppSettings
from .folder import pick_folder
from .runs import FINAL_STATUSES, Run, RunManager, RunOptions
from .storage import RunStorage
from .urls import normalize_page_url

log = logging.getLogger("crawler_app.server")

STATIC_DIR = Path(__file__).resolve().parent / "static"

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


def scan_history(output_dir: Path, limit: int = 300) -> list[dict[str, Any]]:
    """Summaries of the run folders under output_dir (those with a matches.json), newest first."""
    items: list[dict[str, Any]] = []
    if not output_dir.is_dir():
        return items
    for folder in output_dir.iterdir():
        manifest = folder / "matches.json"
        if not manifest.is_file():
            continue
        try:
            doc = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        tag = doc.get("tag") or {}
        items.append(
            {
                "id": folder.name,
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
        )
    items.sort(key=lambda i: (i["started"] or "", i["id"]), reverse=True)
    return items[:limit]


def _sse(event: dict[str, Any]) -> str:
    return f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"


def create_app(
    settings: AppSettings | None = None,
    service: DetectionService | None = None,
    browser: Any | None = None,
) -> FastAPI:
    """`service` and `browser` can be injected (tests); otherwise the real engine and Chromium are started."""
    settings = settings or AppSettings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
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

    @app.get("/api/runs/{run_id}/zip")
    async def download_zip(run_id: str, request: Request, min_confidence: float = 0.0):
        run = _run(request, run_id)
        if run.storage is None:
            raise HTTPException(409, "run has no output yet")
        path = await asyncio.to_thread(run.storage.build_zip, min_confidence)
        return FileResponse(path, media_type="application/zip", filename=f"{run.storage.folder.name}.zip")

    @app.get("/api/runs/{run_id}/files/{name}")
    async def download_file(run_id: str, name: str, request: Request):
        run = _run(request, run_id)
        # Only names this run saved are served, so no path can escape the run folder.
        if run.storage is None or not run.storage.has_file(name):
            raise HTTPException(404, "no such file")
        return FileResponse(run.storage.folder / name)

    # ---- history: every run folder on disk, including ones from before this app was started ----

    def _history_storage(name: str) -> RunStorage:
        folder = settings.output_dir / name
        if not _FOLDER_RE.match(name) or name.strip(".") == "" or not (folder / "matches.json").is_file():
            raise HTTPException(404, "no such run folder")
        try:
            return RunStorage.load(folder)
        except (OSError, ValueError) as e:
            raise HTTPException(500, f"could not read matches.json: {e}")

    @app.get("/api/history")
    async def history(request: Request):
        live = {r.storage.folder.name: r for r in _manager(request).runs.values() if r.storage}
        items = await asyncio.to_thread(scan_history, settings.output_dir)
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
        if not st.has_file(file):  # only names listed in matches.json, so no path can escape
            raise HTTPException(404, "no such file")
        return FileResponse(st.folder / file)

    @app.get("/api/history/{name}/zip")
    async def history_zip(name: str, min_confidence: float = 0.0):
        st = await asyncio.to_thread(_history_storage, name)
        path = await asyncio.to_thread(st.build_zip, min_confidence)
        return FileResponse(path, media_type="application/zip", filename=f"{name}.zip")

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
    uvicorn.run(create_app(), host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
