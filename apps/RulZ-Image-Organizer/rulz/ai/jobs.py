"""Background AI jobs: tag photos, look for something in photos, test the connection.

One job runs at a time, on plain Python threads (requests block for seconds; a QThread can't be left
running when the window closes, these daemon threads can). Results are written as they come in, by the
job's own thread and database connection, so a stopped or crashed job keeps what it already found.
"""
from __future__ import annotations

import queue
import threading
import time
import traceback
from dataclasses import dataclass, field

from PySide6.QtCore import QObject, Signal

from .. import config
from ..db import Database
from . import client, settings as ai_settings
from .client import AIError, EndpointDown, Stopped

SAVE_EVERY = 0.75      # seconds between database writes while results come in
PROGRESS_EVERY = 0.1   # seconds between progress signals
LOG_MAX_BYTES = 2_000_000  # ai.log starts over beyond this (llama-server prints thousands of lines per load)


def log_file():
    return config.data_dir() / "ai.log"


_log_lock = threading.Lock()


def write_log(line: str) -> None:
    """Append a line to ai.log (llama-server output, failures). Called from any thread."""
    with _log_lock:
        try:
            path = log_file()
            mode = "w" if path.exists() and path.stat().st_size > LOG_MAX_BYTES else "a"
            with open(path, mode, encoding="utf-8") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {line}\n")
        except OSError:
            pass


@dataclass
class Job:
    kind: str                            # "tag", "look" or "test"
    photo_ids: list[int] | None = None   # None = every photo in the library
    queries: list[str] = field(default_factory=list)  # look: what to look for (normalized)
    force: bool = False                  # redo photos that already have a current result

    def title(self) -> str:
        if self.kind == "tag":
            return "AI tagging"
        if self.kind == "look":
            return (f"Looking for “{self.queries[0]}”" if len(self.queries) == 1
                    else f"Looking for {len(self.queries)} things")
        return "Testing the AI connection"


@dataclass
class _Item:
    photo_id: int
    path: str
    fp: str
    query: str = ""
    search_id: int = 0


class AIController(QObject):
    started = Signal(object)              # Job
    progress = Signal(int, int, str)      # done, total (0 = unknown), message
    saved = Signal()                      # results were written to the library
    finished = Signal(object, str, bool)  # Job, message for the user, ok (False: failed or stopped)
    log = Signal(str)                     # llama-server output and similar details (also written to ai.log)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.job: Job | None = None
        self.last_message = ""
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._servers = None
        self.finished.connect(self._done)  # first, so other slots already see running == False

    # ---- control -------------------------------------------------------------------------------
    @property
    def running(self) -> bool:
        return self.job is not None

    @property
    def stopping(self) -> bool:
        return self.running and self._stop.is_set()

    def start(self, job: Job) -> bool:
        """Start a job; False (nothing started) while another one runs."""
        if self.job is not None:
            return False
        self.job = job
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, args=(job,), daemon=True, name="rulz-ai")
        self.started.emit(job)
        self._thread.start()
        return True

    def stop(self) -> None:
        """Stop the running job. Requests already sent finish (or time out) first; nothing new is sent."""
        self._stop.set()

    def servers(self):
        """RulZ's own llama-server copies (Training Captioner managed mode), created on first use."""
        if self._servers is None:
            from .server import LlamaServers

            self._servers = LlamaServers()
        return self._servers

    def server_running(self) -> bool:
        return self._servers is not None and self._servers.running()

    def stop_server(self) -> None:
        if self._servers is not None:
            self._servers.stop()

    def shutdown(self) -> None:
        """The window is closing: stop the job (its threads die with the app) and the llama-server copies."""
        self._stop.set()
        self.stop_server()

    def wait(self, timeout: float = 10) -> bool:
        """Wait for the job thread to end (tests). True when it did."""
        t = self._thread
        if t is not None:
            t.join(timeout)
            return not t.is_alive()
        return True

    def _done(self, job, msg, ok):
        self.last_message = msg
        if self.job is job:
            self.job = None
            self._thread = None

    # ---- the job thread -------------------------------------------------------------------------
    def _run(self, job: Job) -> None:
        msg, ok = "", False
        db = None
        try:
            s = ai_settings.load()
            conn = ai_settings.resolve(s)
            if conn.problem:
                raise AIError(conn.problem)

            def endpoints():
                return self._managed_endpoints(conn, s) if conn.kind == "managed" else conn.endpoints

            if job.kind == "test":
                msg = self._test(endpoints(), conn, s)
            else:
                db = Database()
                msg = self._tag(db, job, endpoints, conn, s) if job.kind == "tag" else \
                    self._look(db, job, endpoints, conn, s)
            ok = not self._stop.is_set()
        except Stopped:
            msg = "AI stopped."
        except AIError as e:
            msg = str(e)
        except Exception as e:  # noqa: BLE001 - the server module's errors, and anything unexpected
            if type(e).__name__ == "Cancelled":
                msg = "AI stopped."
            elif type(e).__name__ == "ServerError":
                msg = str(e)
            else:
                traceback.print_exc()
                msg = f"AI job failed: {e}"
        finally:
            if db is not None:
                db.close()
            self._log(f"{job.title()}: {msg}")
            self._emit(self.finished, job, msg, ok)

    @staticmethod
    def _emit(signal, *args) -> None:
        try:
            signal.emit(*args)
        except RuntimeError:  # the window (and this object) is gone: the app is closing
            pass

    def _log(self, line: str) -> None:
        write_log(line)
        self._emit(self.log, line)

    def _managed_endpoints(self, conn, s) -> list[dict]:
        self._emit(self.progress, 0, 0, "Starting the AI model (llama-server)…")
        endpoints, source = self.servers().ensure(conn.captioner, s["managed_port"], s["disable_thinking"],
                                                  self._log, self._stop.is_set)
        self._log("Using the Training Captioner's running llama-server" if source == "captioner"
                  else "Using RulZ's own llama-server")
        return endpoints

    def _test(self, endpoints, conn, s) -> str:
        ep = endpoints[0]
        self._emit(self.progress, 0, 0, f"Testing {ep['url']}…")
        models = []
        if conn.kind == "external":
            try:
                models = client.list_models(ep["url"], conn.api_key, timeout=min(30, s["timeout"]))
            except AIError:
                pass  # not every server lists models; the request below is what counts
        try:
            sees, detail = client.vision_test(ep, conn.api_key, timeout=s["timeout"],
                                              disable_thinking=s["disable_thinking"])
        except (AIError, OSError) as e:
            text = str(e) if isinstance(e, AIError) else client._unreachable(ep["url"], e)
            if models and conn.model and conn.model not in models:
                text += f"\n\nModels on this server: {', '.join(models[:12])}"
            raise AIError(text) from None
        if sees:
            return f"Connected: the model sees images (it answered {detail})."
        return (f"The server answered ({detail}) but didn't name the colour of a red test picture. Check that "
                "the model can see images (a vision model; for llama.cpp, the mmproj file is set).")

    def _tag(self, db: Database, job: Job, endpoints, conn, s) -> str:
        items = [_Item(*row) for row in db.ai_todo_tags(job.photo_ids, job.force)]
        system, user = ai_settings.prompt(s, "tag_system"), ai_settings.prompt(s, "tag_user")
        max_tags = s["max_tags"]
        done, failed, first_error = self._process(
            items, endpoints, conn, s, lambda it: (system, user), s["tag_max_tokens"],
            lambda reply, finish: client.parse_tags(reply, max_tags, cut_off=finish == "length"),
            lambda batch: db.ai_save_tags([(it.photo_id, it.fp, tags, conn.model) for it, tags in batch]),
            job.title())
        if not items:
            return "Every photo in this selection already has AI tags." if job.photo_ids is not None else \
                "Every photo already has AI tags."
        msg = f"AI tagged {done:,} photo{'s' if done != 1 else ''}"
        return msg + _failures(failed, first_error) + ("" if not self._stop.is_set() else " (stopped)")

    def _look(self, db: Database, job: Job, endpoints, conn, s) -> str:
        items: list[_Item] = []
        searches: list[tuple[str, int]] = []
        for q in job.queries:
            sid = db.ai_search_id(q, create=True)
            if sid is None:  # nothing left after normalizing
                continue
            searches.append((q, sid))
            items += [_Item(*row, query=q, search_id=sid) for row in db.ai_todo_search(sid, job.photo_ids, job.force)]
        system = ai_settings.prompt(s, "look_system")

        def save(batch):
            by_search: dict[int, list] = {}
            for it, found in batch:
                by_search.setdefault(it.search_id, []).append((it.photo_id, it.fp, found, conn.model))
            for sid, rows in by_search.items():
                db.ai_save_answers(sid, rows)

        done, failed, first_error = self._process(
            items, endpoints, conn, s, lambda it: (system, ai_settings.look_user_prompt(s, it.query)),
            s["look_max_tokens"], client.parse_answer_reply, save, job.title())
        scope = set(job.photo_ids) if job.photo_ids is not None else None
        parts = []
        for q, sid in searches:
            found = db.ai_found(sid)
            n = len(found if scope is None else found & scope)
            parts.append(f"“{q}” found in {n:,} photo{'s' if n != 1 else ''}")
        msg = "; ".join(parts) + (f" ({done:,} checked now)" if items else " (all checked earlier)")
        return msg + _failures(failed, first_error) + ("" if not self._stop.is_set() else " (stopped)")

    def _process(self, items: list[_Item], get_endpoints, conn, s, prompts, max_tokens: int, parse, save,
                 title: str) -> tuple[int, int, str]:
        """Send every item to the endpoints (as many at a time as they have slots) and save the parsed replies.
        Returns (done, failed, first error). The endpoints are asked for only when there is work: a llama-server
        isn't started just to find everything was done before."""
        total = len(items)
        if not total:
            return 0, 0, ""
        endpoints = get_endpoints()
        slots: queue.Queue = queue.Queue()
        for ep in endpoints:
            for _ in range(max(1, int(ep.get("slots", 1)))):
                slots.put(ep)
        workers = slots.qsize()
        todo: queue.Queue = queue.Queue()
        for it in items:
            todo.put(it)
        results: queue.Queue = queue.Queue()
        down: set[str] = set()
        names = {ep["name"] for ep in endpoints}
        lock = threading.Lock()
        all_down = threading.Event()
        last_down = [""]

        def give_up() -> bool:
            return self._stop.is_set() or all_down.is_set()

        def work(it: _Item):
            ep = None
            while ep is None:  # wait for a free slot on a server that still answers
                if give_up():
                    return None
                try:
                    ep = slots.get(timeout=0.25)
                except queue.Empty:
                    continue
                if ep["name"] in down:
                    ep = None  # that server stopped answering; its slots are not handed back
            try:
                b64 = client.encode_photo(it.path, s["max_image_side"])
            except Exception as e:  # noqa: BLE001 - unreadable or vanished file
                slots.put(ep)
                return it, None, AIError(f"Couldn't read the photo: {e}")
            try:
                system, user = prompts(it)
                reply, finish = client.chat_with_retry(
                    ep, conn.api_key, system, user, b64, wait=self._stop.wait, max_tokens=max_tokens,
                    temperature=s["temperature"], disable_thinking=s["disable_thinking"], timeout=s["timeout"])
                value = parse(reply, finish)
            except EndpointDown as e:
                with lock:
                    down.add(ep["name"])
                    last_down[0] = str(e)
                    if down >= names:
                        all_down.set()  # nothing can answer the rest: stop instead of failing every photo
                        return it, None, e
                todo.put(it)  # another server still answers: it takes this photo
                return None
            except Exception as e:  # noqa: BLE001
                slots.put(ep)
                return it, None, e
            slots.put(ep)
            return it, value, None

        def worker():
            # Daemon threads, not a ThreadPoolExecutor: Python waits for executor threads at exit, so a request
            # still waiting on a slow or silent server kept the app's process alive after its window closed.
            try:
                while not give_up():
                    try:
                        it = todo.get_nowait()
                    except queue.Empty:
                        break
                    r = work(it)
                    if r is not None:
                        results.put(r)
            finally:
                results.put(None)  # this worker is done

        for _ in range(workers):
            threading.Thread(target=worker, daemon=True, name="rulz-ai-request").start()
        done = failed = 0
        first_error = ""
        batch: list = []
        last_save = last_progress = time.monotonic()
        self._emit(self.progress, 0, total, f"{title} · 0 of {total:,}")
        running = workers
        try:
            while running:
                r = results.get()
                if r is None:
                    running -= 1
                    continue
                it, value, err = r
                if err is None:
                    batch.append((it, value))
                    done += 1
                elif not isinstance(err, Stopped):
                    failed += 1
                    if not first_error and not isinstance(err, EndpointDown):
                        first_error = str(err)
                now = time.monotonic()
                if batch and (now - last_save >= SAVE_EVERY):
                    save(batch)
                    batch = []
                    last_save = now
                    self._emit(self.saved)
                if now - last_progress >= PROGRESS_EVERY:
                    last_progress = now
                    self._emit(self.progress, done + failed, total, f"{title} · {done + failed:,} of {total:,}")
            # requests already on their way finished above; their results are kept
        finally:
            if batch:
                save(batch)
                self._emit(self.saved)
        if all_down.is_set() and not self._stop.is_set():
            raise AIError(f"No AI server is answering, so the job stopped after {done:,} photo"
                          f"{'s' if done != 1 else ''}. {last_down[0]}")
        return done, failed, first_error

def _failures(failed: int, first_error: str) -> str:
    if not failed:
        return "."
    return f". {failed:,} photo{'s' if failed != 1 else ''} failed" + (f": {first_error}" if first_error else ".")
