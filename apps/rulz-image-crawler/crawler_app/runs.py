"""One crawl = one Run. Crawl, fetch and classify run concurrently, linked by bounded queues:

    crawl (BFS, 1 page at a time) ──▶ fetch workers (N) ──▶ classifier (one task per image) ──▶ save

Bounded queues give back-pressure: when the model is the bottleneck the crawler waits instead of
racing ahead. If the model server goes down the run pauses (nothing is lost) until resumed.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from image_detection import DetectionService, ImageInput, ModelServerDown, Tag
from image_detection.preprocess import BROWSER_UA
from playwright.async_api import Browser, BrowserContext

from .config import AppSettings
from .crawler import ImageCandidate, PageError, Robots, load_page
from .fetcher import FetchedImage, FetchError, HostGate, check_image_bytes, fetch_image
from .folder import list_image_files
from .storage import RunStorage, make_run_dir
from .urls import in_path_scope, looks_like_page, normalize_page_url, path_scope, site_key

log = logging.getLogger("crawler_app.runs")

FINAL_STATUSES = {"done", "cancelled", "failed"}
ENGINE_DOWN_MESSAGE = "The model server is not responding. Start the model, then press Resume."

def _duration(s: float) -> str:
    return f"{s:.0f} s" if s < 90 else f"{s / 60:.0f} min"


def _read_local_image(path: Path, root: Path, max_bytes: int) -> FetchedImage:
    """Folder scans: read, type-check and hash one file (in a worker thread)."""
    if path.stat().st_size > max_bytes:
        raise FetchError("image too large")
    return check_image_bytes(str(path), str(root), path.read_bytes(), max_bytes)


STAT_KEYS = (
    "pages_crawled", "pages_queued", "pages_failed", "pages_blocked",
    "images_found", "images_fetched", "fetch_errors", "duplicates",
    "classified", "cached", "skipped", "too_small_mp", "originals", "errors", "matches",
)


@dataclass
class RunOptions:
    url: str  # website to crawl; empty when scanning a folder
    tag_id: str
    rubric: str
    threshold: float = 0.7  # images with model match AND confidence >= this are saved
    max_pages: int = 10
    max_depth: int = 3
    max_images: int = 300
    respect_robots: bool = False
    stay_in_path: bool = False  # only follow links inside the start page's folder (and below)
    page_delay_s: float = 1.0
    prefer_originals: bool = True  # try each image's full-size original before the displayed copy
    min_megapixels: float = 0.0  # images smaller than this are not classified or saved (0 = no limit)
    folder: str | None = None  # scan this local folder instead of crawling `url`
    recursive: bool = True  # include subfolders of `folder`


class Run:
    def __init__(
        self, run_id: str, opts: RunOptions, settings: AppSettings, service: DetectionService, browser: Browser
    ):
        self.id = run_id
        self.opts = opts
        self.s = settings
        self.svc = service
        self.browser = browser
        self.status = "starting"
        self.message = ""
        self.stats: dict[str, int] = dict.fromkeys(STAT_KEYS, 0)
        self.current_page: str | None = None
        self.errors: deque[str] = deque(maxlen=100)
        self.storage: RunStorage | None = None
        self.started_at = time.time()
        self.finished_at: float | None = None

        self._resume = asyncio.Event()
        self._resume.set()
        self._subs: set[asyncio.Queue] = set()
        self._dirty = False
        self._seen_hashes: set[str] = set()
        self._gate = HostGate(
            per_host=2,
            on_limited=lambda host, s, held: self._error(
                f"{host} asked us to slow down; waiting {_duration(s)} before downloading more of its images"
                if held
                else f"{host} asked us to slow down; using the page's copies of its images for {_duration(s)}"
            ),
        )
        self._context: BrowserContext | None = None
        self._task: asyncio.Task | None = None

    # ---- control ---------------------------------------------------------------------

    def start(self) -> None:
        self._task = asyncio.create_task(self._main(), name=f"run-{self.id}")

    def pause(self, message: str = "Paused.") -> None:
        if self.status in FINAL_STATUSES or self.status == "paused":
            return
        self._resume.clear()
        self._set_status("paused", message)

    def resume(self) -> None:
        if self.status != "paused":
            return
        self._set_status("running", "")
        self._resume.set()

    def cancel(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()

    async def wait(self) -> None:
        if self._task:
            await asyncio.gather(self._task, return_exceptions=True)

    @property
    def finished(self) -> bool:
        return self.status in FINAL_STATUSES

    @property
    def matches(self) -> list[dict[str, Any]]:
        """The saved matches (the storage's own list, not a second copy of it)."""
        return self.storage.entries if self.storage is not None else []

    # ---- events ----------------------------------------------------------------------

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._subs.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subs.discard(q)

    def _publish(self, event: dict[str, Any]) -> None:
        for q in list(self._subs):
            q.put_nowait(event)

    def _bump(self, key: str, n: int = 1) -> None:
        self.stats[key] += n
        self._dirty = True

    def _error(self, text: str) -> None:
        self.errors.append(text)
        log.info("run %s: %s", self.id, text)

    def _set_status(self, status: str, message: str = "") -> None:
        self.status, self.message = status, message
        self._publish({"type": "status", "status": status, "message": message})
        self._publish(self._stats_event())

    def _mark_running(self) -> None:
        """Starting -> running, unless Pause was pressed meanwhile: then it stays paused until Resume."""
        if self.status == "starting":
            self._set_status("running")

    def _stats_event(self) -> dict[str, Any]:
        self._dirty = False
        return {"type": "stats", "stats": dict(self.stats), "current_page": self.current_page}

    def snapshot(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "message": self.message,
            "options": asdict(self.opts),
            "folder": str(self.storage.folder) if self.storage else None,
            "stats": dict(self.stats),
            "current_page": self.current_page,
            "matches": list(self.matches),
            "errors": list(self.errors),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }

    async def _ticker(self) -> None:
        while True:
            await asyncio.sleep(0.5)
            if self._dirty:
                self._publish(self._stats_event())
            if self.storage is not None and self.storage.dirty:
                try:
                    await asyncio.to_thread(self.storage.flush)
                except OSError as e:  # pragma: no cover - retried on the next tick
                    log.warning("run %s: could not write matches.json: %s", self.id, e)

    # ---- pipeline --------------------------------------------------------------------

    async def _main(self) -> None:
        ticker = asyncio.create_task(self._ticker())
        try:
            o = self.opts
            label = f"folder-{Path(o.folder).name or 'drive'}" if o.folder else None
            folder = await asyncio.to_thread(make_run_dir, self.s.output_dir, o.url, o.tag_id, None, label)
            self.storage = RunStorage(
                folder,
                {
                    "source": "folder" if o.folder else "web",
                    "start_url": o.folder or o.url,
                    "tag": {"id": self.opts.tag_id, "rubric": self.opts.rubric},
                    "threshold": self.opts.threshold,
                    "options": asdict(self.opts),
                    "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "status": "running",
                },
            )
            await asyncio.to_thread(self.storage.write_manifest)
            if o.folder:
                await self._main_folder()
                return
            self._context = await self.browser.new_context(
                user_agent=BROWSER_UA, viewport={"width": 1366, "height": 900}
            )
            self._mark_running()

            fetch_q: asyncio.Queue[ImageCandidate | None] = asyncio.Queue(maxsize=200)
            classify_q: asyncio.Queue[FetchedImage | None] = asyncio.Queue(maxsize=self._in_flight())
            async with asyncio.TaskGroup() as tg:
                tg.create_task(self._crawl(fetch_q))
                tg.create_task(self._fetch_stage(fetch_q, classify_q))
                tg.create_task(self._classify_loop(classify_q))
            if self.stats["pages_crawled"] == 0 and not self.message:
                self.message = self.errors[-1] if self.errors else "No pages could be crawled."
            self._finish("done", self.message)
        except asyncio.CancelledError:
            self._finish("cancelled", "Cancelled.")
        except BaseException as e:  # noqa: BLE001 - surface anything unexpected as a failed run
            leaf = e
            while isinstance(leaf, BaseExceptionGroup):
                leaf = leaf.exceptions[0]
            log.exception("run %s failed", self.id, exc_info=e)
            self._finish("failed", f"{type(leaf).__name__}: {leaf}")
        finally:
            ticker.cancel()
            if self._context is not None:
                try:
                    await self._context.close()
                except Exception:  # pragma: no cover - browser already gone
                    pass

    async def _main_folder(self) -> None:
        self._mark_running()
        classify_q: asyncio.Queue[FetchedImage | None] = asyncio.Queue(maxsize=self._in_flight())
        async with asyncio.TaskGroup() as tg:
            tg.create_task(self._scan_folder(classify_q))
            tg.create_task(self._classify_loop(classify_q))
        if self.stats["images_found"] == 0 and not self.message:
            self.message = self.errors[-1] if self.errors else "No images found in that folder."
        self._finish("done", self.message)

    async def _scan_folder(self, classify_q: asyncio.Queue) -> None:
        o, s = self.opts, self.s
        root = Path(o.folder)
        try:
            files = await asyncio.to_thread(list_image_files, root, o.recursive, s.output_dir)
            self.stats["pages_queued"] = 0
            for path in files:
                if self.stats["images_found"] >= o.max_images:
                    break
                await self._resume.wait()
                self.current_page = str(path)
                self._bump("images_found")
                try:
                    img = await asyncio.to_thread(_read_local_image, path, root, s.max_image_bytes)
                except (OSError, FetchError) as e:
                    self._bump("fetch_errors")
                    self._error(f"file {path}: {e}")
                    continue
                await self._accept(img, classify_q)
        finally:
            self.current_page = None
            self._dirty = True
        await classify_q.put(None)

    def _finish(self, status: str, message: str) -> None:
        self.current_page = None
        self.finished_at = time.time()
        if self.storage is not None:  # recorded for the history list
            self.storage.meta.update(
                status=status, message=message, stats=dict(self.stats), finished=time.strftime("%Y-%m-%dT%H:%M:%S")
            )
            try:
                self.storage.write_manifest()
            except OSError as e:  # pragma: no cover - disk gone; the run result is still in memory
                log.warning("run %s: could not write matches.json: %s", self.id, e)
        self._set_status(status, message)

    async def _crawl(self, fetch_q: asyncio.Queue) -> None:
        o, s = self.opts, self.s
        start = normalize_page_url(o.url)
        robots = Robots(self._context, s.fetch_timeout_s, self._error) if o.respect_robots else None
        queue: deque[tuple[str, int]] = deque([(start, 0)]) if start else deque()
        seen_pages = {start}
        seen_images: set[str] = set()
        site: str | None = None
        scope = "/"
        try:
            while queue and self.stats["pages_crawled"] < o.max_pages and self.stats["images_found"] < o.max_images:
                await self._resume.wait()
                url, depth = queue.popleft()
                self.stats["pages_queued"] = len(queue)
                if robots and not await robots.allowed(url):
                    self._bump("pages_blocked")
                    self._error(f"blocked by robots.txt: {url}")
                    continue

                self.current_page = url
                self._dirty = True
                try:
                    page = await load_page(
                        self._context, url, s.page_timeout_s, s.max_scroll_steps, o.prefer_originals
                    )
                except PageError as e:
                    if site is None and url.startswith("https://"):
                        # Start page failed over https (the default when no scheme is typed): try http once.
                        fallback = "http://" + url[len("https://"):]
                        if fallback not in seen_pages:
                            seen_pages.add(fallback)
                            queue.appendleft((fallback, depth))
                            self._error(f"page {url}: {e}; retrying over http")
                            continue
                    self._bump("pages_failed")
                    self._error(f"page {url}: {e}")
                    continue
                self._bump("pages_crawled")
                if site is None:
                    site = site_key(page.url)  # the start page after redirects defines the site
                    if o.stay_in_path:
                        scope = path_scope(page.url)

                for cand in page.images:
                    if any(u in seen_images for u in cand.ids):
                        continue
                    seen_images.update(cand.ids)
                    if cand.short_edge is not None and cand.short_edge < s.min_short_edge:
                        self._bump("skipped")
                        continue
                    if self.stats["images_found"] >= o.max_images:
                        break
                    self._bump("images_found")
                    await fetch_q.put(cand)

                if depth < o.max_depth:
                    for link in page.links:
                        if (
                            link not in seen_pages
                            and site_key(link) == site
                            and in_path_scope(link, scope)
                            and looks_like_page(link)
                        ):
                            seen_pages.add(link)
                            queue.append((link, depth + 1))
                self.stats["pages_queued"] = len(queue)

                if queue and self.stats["pages_crawled"] < o.max_pages:
                    delay = o.page_delay_s
                    if robots:
                        delay = max(delay, min(10.0, await robots.crawl_delay(url)))
                    await asyncio.sleep(delay)
        finally:
            self.current_page = None
            self._dirty = True
        for _ in range(self.s.fetch_concurrency):
            await fetch_q.put(None)

    async def _fetch_stage(self, fetch_q: asyncio.Queue, classify_q: asyncio.Queue) -> None:
        async with asyncio.TaskGroup() as tg:
            for _ in range(self.s.fetch_concurrency):
                tg.create_task(self._fetch_worker(fetch_q, classify_q))
        await classify_q.put(None)

    async def _fetch_worker(self, fetch_q: asyncio.Queue, classify_q: asyncio.Queue) -> None:
        while (cand := await fetch_q.get()) is not None:
            await self._resume.wait()
            try:
                img = await fetch_image(
                    self._context, cand, self.s.fetch_timeout_s, self.s.max_image_bytes, self._gate
                )
            except FetchError as e:
                self._bump("fetch_errors")
                self._error(f"image {cand.url[:200]}: {e}")
                continue
            await self._accept(img, classify_q)

    async def _accept(self, img: FetchedImage, classify_q: asyncio.Queue) -> None:
        """Dedupe by content and apply the size limit, then queue for the model."""
        if img.sha256 in self._seen_hashes:
            self._bump("duplicates")
            return
        self._seen_hashes.add(img.sha256)
        self._bump("images_fetched")
        if img.is_original:
            self._bump("originals")
        mp = img.megapixels
        if self.opts.min_megapixels > 0 and mp is not None and mp < self.opts.min_megapixels:
            self._bump("too_small_mp")
            return
        if img.width and img.height and min(img.width, img.height) < self.s.min_short_edge:
            self._bump("skipped")  # icons and thumbnails; the engine would skip them anyway
            return
        await classify_q.put(img)

    def _in_flight(self) -> int:
        """Images being prepared or judged at once: CLASSIFY_IN_FLIGHT, else three per model slot, so
        every slot (on every GPU) always has the next image decoded and waiting."""
        if self.s.classify_in_flight > 0:
            return self.s.classify_in_flight
        pool = getattr(self.svc, "pool", None)
        slots = len(pool.clients) * pool.parallel if pool is not None else 2
        return max(4, 3 * slots)

    async def _classify_loop(self, q: asyncio.Queue) -> None:
        """One task per image: each result shows as soon as it's ready, and a slow or broken image holds
        up nothing else."""
        tag = Tag(self.opts.tag_id, self.opts.rubric)
        in_flight = asyncio.Semaphore(self._in_flight())

        async def one(img: FetchedImage) -> None:
            try:
                await self._classify_one(img, tag)
            finally:
                in_flight.release()

        async with asyncio.TaskGroup() as tg:
            while (img := await q.get()) is not None:
                await in_flight.acquire()
                tg.create_task(one(img))

    async def _classify_one(self, img: FetchedImage, tag: Tag) -> None:
        # The hash is already known: the engine doesn't hash the bytes again.
        inp = ImageInput(img.sha256, mime=img.mime, data=img.data, sha256=img.sha256)
        while True:
            await self._resume.wait()
            try:
                # threshold=0 returns the raw model answer; this run applies its own threshold.
                res = (await self.svc.classify([inp], [tag], threshold=0.0))[img.sha256]
                break
            except ModelServerDown:
                self.pause(ENGINE_DOWN_MESSAGE)
            except Exception as e:  # noqa: BLE001 - one image's failure must not end the run
                log.exception("run %s: classifying %s failed", self.id, img.url[:200])
                res = {"status": "error", "message": f"internal error: {type(e).__name__}: {e}"}
                break

        status = res.get("status")
        if status == "skipped":
            self._bump("skipped")
            return
        if status != "ok":
            self._bump("errors")
            self._error(f"classify {img.url[:200]}: {res.get('message', 'error')}")
            return
        self._bump("classified")
        if res.get("cached"):
            self._bump("cached")
        t = res["tags"][tag.id]
        if t["match"] and t["confidence"] >= self.opts.threshold:
            entry = await asyncio.to_thread(self.storage.save, img, t["confidence"], t["evidence"])
            self._bump("matches")
            self._publish({"type": "match", "match": entry})


class RunManager:
    # Finished runs stay readable from disk (Past searches); only the newest few are kept in memory.
    KEEP_FINISHED = 10

    def __init__(self, settings: AppSettings, service: DetectionService, browser: Browser):
        self.s = settings
        self.svc = service
        self.browser = browser
        self.runs: dict[str, Run] = {}

    def _prune(self) -> None:
        finished = [r for r in self.runs.values() if r.finished]  # oldest first
        for r in finished[: max(0, len(finished) - self.KEEP_FINISHED)]:
            del self.runs[r.id]

    def create(self, opts: RunOptions) -> Run:
        self._prune()
        run = Run(uuid.uuid4().hex[:12], opts, self.s, self.svc, self.browser)
        self.runs[run.id] = run
        run.start()
        return run

    def get(self, run_id: str) -> Run | None:
        return self.runs.get(run_id)

    async def shutdown(self) -> None:
        for run in self.runs.values():
            run.cancel()
        await asyncio.gather(*(r.wait() for r in self.runs.values()), return_exceptions=True)
