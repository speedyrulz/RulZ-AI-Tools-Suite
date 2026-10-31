"""Core orchestration. Usable directly as a library, or through the FastAPI app in server.py."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Sequence, TypeVar

import httpx

from .cache import ResultCache, TagResult, rubric_hash
from .config import Settings
from .llm import LlamaClient, ModelError, ModelServerDown
from .preprocess import ImageError, ImageTooSmall, PreparedImage, decode_b64, fetch_url, prepare_image
from .rubrics import enforce_adult_subject, normalize_tag_id

# Bump when the expansion prompt or post-processing changes so old cached rubrics are not reused.
EXPANSION_VERSION = "v2"
# Part of the result-cache key: bump when the meaning of a cached confidence changes.
# "p1" = confidence is the model's probability of its answer, not the number it wrote.
SCORING_VERSION = "p1"
# Part of the raw-bytes -> prepared-hash key: bump when prepare_image's output for the same input changes.
PREPARE_VERSION = "r2"

T = TypeVar("T")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

log = logging.getLogger("image_detection.service")


@dataclass(frozen=True)
class ImageInput:
    id: str
    data_b64: str | None = None
    url: str | None = None
    mime: str | None = None  # informational; the decoder sniffs the real format
    data: bytes | None = None  # raw bytes, for in-process callers (skips the base64 round trip)


@dataclass(frozen=True)
class Tag:
    id: str
    rubric: str


def _first_leaf(eg: BaseExceptionGroup) -> BaseException:
    exc: BaseException = eg
    while isinstance(exc, BaseExceptionGroup):
        exc = exc.exceptions[0]
    return exc


def _chunks(seq: Sequence, n: int) -> list[Sequence]:
    return [seq[i : i + n] for i in range(0, len(seq), n)]


class LlamaPool:
    """Spreads model calls over one or more llama-servers (e.g. one per GPU), `parallel` at a time
    on each, least busy first. A server that stops answering is skipped for a while and the call is
    retried on another; only when every server is down does the caller see ModelServerDown."""

    DOWN_FOR_S = 15.0

    def __init__(self, clients: Sequence[LlamaClient], parallel: int):
        self.clients = list(clients)
        self.parallel = max(1, parallel)
        self.busy = [0] * len(self.clients)
        self._down_until = [0.0] * len(self.clients)
        self._changed = asyncio.Event()
        self._next = 0

    def _usable(self, exclude: set[int]) -> list[int]:
        now = time.monotonic()
        left = [i for i in range(len(self.clients)) if i not in exclude]
        live = [i for i in left if self._down_until[i] <= now]
        return live or left  # every one recently down: try anyway, it may be back

    async def _acquire(self, exclude: set[int]) -> int:
        while True:
            free = [i for i in self._usable(exclude) if self.busy[i] < self.parallel]
            if free:
                n = len(self.clients)  # least busy; ties go round-robin so idle servers share the work
                i = min(free, key=lambda k: (self.busy[k], (k - self._next) % n))
                self._next = (i + 1) % n
                self.busy[i] += 1
                return i
            self._changed.clear()
            await self._changed.wait()

    def _release(self, i: int) -> None:
        self.busy[i] -= 1
        self._changed.set()

    async def call(self, fn: Callable[[LlamaClient], Awaitable[T]]) -> T:
        tried: set[int] = set()
        while True:
            i = await self._acquire(tried)
            try:
                return await fn(self.clients[i])
            except ModelServerDown:
                self._down_until[i] = time.monotonic() + self.DOWN_FOR_S
                tried.add(i)
                if len(tried) == len(self.clients):
                    raise
                log.warning("model server %s is down; retrying on another", self.clients[i].url)
            finally:
                self._release(i)


class DetectionService:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        llama: LlamaClient | None = None,
        llamas: Sequence[LlamaClient] | None = None,
        cache: ResultCache | None = None,
        fetch_client: httpx.AsyncClient | None = None,
    ):
        self.s = settings or Settings()
        if llamas:
            self.llamas = list(llamas)
        elif llama:
            self.llamas = [llama]
        else:
            self.llamas = [LlamaClient(self.s, url=u) for u in self.s.llama_urls]
        self.llama = self.llamas[0]
        self.pool = LlamaPool(self.llamas, self.s.llama_parallel)
        self.cache = cache or ResultCache(self.s.cache_path)
        self._fetch = fetch_client or httpx.AsyncClient()
        self._owns_fetch = fetch_client is None
        self._fetch_slots = asyncio.Semaphore(self.s.fetch_concurrency)
        self._busy = 0

    async def aclose(self) -> None:
        for c in self.llamas:
            await c.aclose()
        if self._owns_fetch:
            await self._fetch.aclose()
        self.cache.close()

    # ---- public API ------------------------------------------------------------------

    async def classify(
        self, images: Sequence[ImageInput], tags: Sequence[Tag], threshold: float | None = None
    ) -> dict[str, dict[str, Any]]:
        """Returns {caller_id: {"status", "cached", "tags"|"message"}}.
        Raises ModelServerDown if llama-server is unreachable when a model call is needed."""
        threshold = self.s.default_threshold if threshold is None else threshold
        model_id = await self._model_id()
        if self.s.confidence_from_logprobs:
            model_id = f"{model_id}|{SCORING_VERSION}"  # cache key only
        try:
            async with asyncio.TaskGroup() as tg:
                tasks = {img.id: tg.create_task(self._process_image(img, tags, threshold, model_id)) for img in images}
        except ExceptionGroup as eg:
            down = eg.subgroup(ModelServerDown)
            if down is None:
                raise
            self._forget_model_id()
            raise _first_leaf(down) from None
        return {img_id: t.result() for img_id, t in tasks.items()}

    async def expand_tag(self, text: str) -> dict[str, str]:
        normalized = " ".join(text.strip().lower().split())
        if not normalized:
            raise ValueError("tag text is empty")
        key = f"{EXPANSION_VERSION}:{normalized}"
        model_id = await self._model_id()
        hit = self.cache.get_expansion(key, model_id)
        if hit:
            return {"id": hit[0], "rubric": hit[1]}
        try:
            raw_id, rubric = await self.pool.call(lambda c: c.expand_tag(text.strip()))
        except ModelServerDown:
            self._forget_model_id()
            raise
        tag_id = normalize_tag_id(raw_id)
        rubric = enforce_adult_subject(text, rubric)
        self.cache.put_expansion(key, model_id, tag_id, rubric)
        return {"id": tag_id, "rubric": rubric}

    async def health(self) -> dict[str, Any]:
        ups = await asyncio.gather(*(c.is_up() for c in self.llamas))
        up = any(ups)
        model = None
        if up:
            try:
                model = await self._model_id()
            except ModelServerDown:
                up = False
        if model and model.lower().endswith(".gguf"):
            model = model[: -len(".gguf")]
        slots = 0
        if up:
            free = await asyncio.gather(*(c.slots_free() for c, u in zip(self.llamas, ups) if u))
            busy = [b for b, u in zip(self.pool.busy, ups) if u]
            slots = sum(max(0, self.s.llama_parallel - b) if f is None else f for f, b in zip(free, busy))
        return {
            "model": model,
            "server": "up" if up else "down",
            "slots_free": slots,
            "servers": [{"url": c.url, "up": u} for c, u in zip(self.llamas, ups)],
            "stats": {
                "model_calls": sum(c.stats.calls for c in self.llamas),
                "retries": sum(c.stats.retries for c in self.llamas),
                "failures": sum(c.stats.failures for c in self.llamas),
            },
        }

    # ---- internals -------------------------------------------------------------------

    async def _model_id(self) -> str:
        """All servers run the same model; ask the first one that answers."""
        err: ModelServerDown | None = None
        for c in self.llamas:
            try:
                return await c.model_id()
            except ModelServerDown as e:
                err = e
        raise err or ModelServerDown("no model server configured")

    def _forget_model_id(self) -> None:
        # The server may come back with a different model; re-read the id next time.
        if not self.s.model_id:
            for c in self.llamas:
                c._model_id = None

    async def _raw(self, img: ImageInput) -> bytes:
        if img.data is not None:
            return img.data
        if img.data_b64:
            return decode_b64(img.data_b64)
        if img.url:
            async with self._fetch_slots:
                return await fetch_url(self._fetch, img.url, self.s.fetch_timeout_s, self.s.max_image_bytes)
        raise ImageError("image has neither data_b64 nor url")

    def _prep_key(self) -> str:
        """Everything that changes the prepared bytes for the same raw bytes."""
        return f"{PREPARE_VERSION}|{self.s.max_edge}|{self.s.jpeg_quality}"

    def _known_results(
        self, raw_hash: str, hashes: dict[str, str], model_id: str
    ) -> tuple[str | None, dict[str, TagResult]]:
        """(prepared hash, cached results) for raw bytes seen before, so a fully cached image skips
        the decode and resize."""
        prepared_hash = self.cache.get_prepared(raw_hash, self._prep_key())
        if prepared_hash is None:
            return None, {}
        return prepared_hash, self.cache.get_many(prepared_hash, hashes.values(), model_id)

    async def _run_chunk(self, prepared: PreparedImage, chunk: Sequence[Tag], model_id: str) -> dict[str, TagResult]:
        pairs = [(t.id, t.rubric) for t in chunk]
        self._busy += 1
        try:
            results, missing = await self.pool.call(lambda c: c.classify(prepared, pairs))
        finally:
            self._busy -= 1
        # Only fully answered tags are cached; a "missing" key is not a real answer.
        self.cache.put_many(
            prepared.sha256,
            model_id,
            {rubric_hash(t.rubric): results[t.id] for t in chunk if t.id not in missing},
        )
        return results

    async def _process_image(
        self, img: ImageInput, tags: Sequence[Tag], threshold: float, model_id: str
    ) -> dict[str, Any]:
        hashes = {t.id: rubric_hash(t.rubric) for t in tags}
        try:
            data = await self._raw(img)
        except ImageError as e:
            return {"status": "error", "cached": False, "message": str(e)}

        raw_hash = await asyncio.to_thread(_sha256, data)
        known, hits = self._known_results(raw_hash, hashes, model_id)
        if known is None or len(hits) < len(set(hashes.values())):
            try:
                prepared = await asyncio.to_thread(
                    prepare_image, data, self.s.max_edge, self.s.min_short_edge, self.s.jpeg_quality
                )
            except ImageTooSmall as e:
                return {"status": "skipped", "cached": False, "message": str(e)}
            except ImageError as e:
                return {"status": "error", "cached": False, "message": str(e)}
            if prepared.sha256 != known:
                self.cache.put_prepared(raw_hash, self._prep_key(), prepared.sha256)
                hits = self.cache.get_many(prepared.sha256, hashes.values(), model_id)
        else:
            prepared = None  # every tag cached: no decode needed

        raw: dict[str, TagResult] = {t.id: hits[hashes[t.id]] for t in tags if hashes[t.id] in hits}
        todo = [t for t in tags if t.id not in raw]

        if todo:
            chunks = _chunks(todo, self.s.max_tags_per_call)
            outcomes = await asyncio.gather(
                *(self._run_chunk(prepared, c, model_id) for c in chunks), return_exceptions=True
            )
            errors: list[str] = []
            for outcome in outcomes:
                if isinstance(outcome, ModelServerDown):
                    raise outcome
                if isinstance(outcome, ModelError):
                    errors.append(str(outcome))
                elif isinstance(outcome, BaseException):
                    log.exception("unexpected error classifying %s", img.id, exc_info=outcome)
                    errors.append(f"internal error: {type(outcome).__name__}")
                else:
                    raw.update(outcome)
            if errors:
                return {"status": "error", "cached": False, "message": "; ".join(errors)}

        return {
            "status": "ok",
            "cached": not todo,
            "tags": {
                t.id: {
                    "match": raw[t.id].match and raw[t.id].confidence >= threshold,
                    "confidence": round(raw[t.id].confidence, 4),
                    "evidence": raw[t.id].evidence,
                }
                for t in tags
            },
        }
