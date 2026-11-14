"""llama.cpp server management and the multi-worker captioning engine."""
import atexit
import base64
import ipaddress
import os
import queue
import re
import shlex
import socket
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlsplit

import requests
from PySide6.QtCore import QBuffer, QByteArray, QIODevice, QObject, Qt, Signal
from PySide6.QtGui import QColor, QImage, QImageReader, QPainter

from . import winjob

CREATE_NO_WINDOW = 0x08000000
# Qt refuses to decode images that need more memory than this. Its default, 256 MB, rejects PNG/TIFF/
# WebP/BMP files over ~67 megapixels (an 8192x8192 upscale), for captioning and thumbnails alike.
# JPEGs are unaffected because they are scaled down while decoding.
IMAGE_MEMORY_LIMIT_MB = 1024


def allow_large_images():
    QImageReader.setAllocationLimit(IMAGE_MEMORY_LIMIT_MB)


def port_in_use(port, host="127.0.0.1"):
    """True when another socket already holds the port. A llama-server started there would fail to
    bind, and its health check would be answered by whatever holds the port instead."""
    with socket.socket() as sock:
        try:
            sock.bind((host, port))
        except OSError:
            return True
    return False


class StartCancelled(RuntimeError):
    """Server start was stopped by the user (not a failure)."""


def is_loopback(url):
    """True for a server on this computer (localhost, 127.x.x.x, ::1)."""
    host = (urlsplit(url).hostname or "").lower()
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def local_session(url="http://127.0.0.1"):
    """A requests session that, for a server on this computer, ignores proxy settings. requests
    takes the system proxy from Windows' settings or HTTP_PROXY, and Windows' "don't use the proxy
    for local addresses" doesn't cover 127.0.0.1 there, so every request to a local llama-server
    went to the proxy: the health check never passed and the app gave up after 15 minutes."""
    session = requests.Session()
    if is_loopback(url):
        session.trust_env = False
    return session


# --------------------------------------------------------------------------- servers
class ServerManager:
    """Launches llama-server processes (one per GPU, or one spanning GPUs)."""

    def __init__(self):
        self.procs = []
        self.endpoints = []
        self._stop = False
        self._gen = 0          # bumped by every start/stop; a start whose number is stale gives up
        self._lock = threading.Lock()
        atexit.register(self.stop)

    @staticmethod
    def build_plan(s):
        """Return a list of (port, env_overrides, extra_cli_args) describing each server."""
        gpus = [str(g) for g in s["gpus"]]
        base = int(s["base_port"])
        if not gpus:
            return [(base, {}, [])]
        if s["gpu_mode"] == "split" and len(gpus) > 1:
            args = ["--split-mode", "layer"]
            ts = s["tensor_split"].strip()
            if ts:
                args += ["--tensor-split", ts]
            return [(base, {"CUDA_VISIBLE_DEVICES": ",".join(gpus)}, args)]
        return [(base + i, {"CUDA_VISIBLE_DEVICES": g}, []) for i, g in enumerate(gpus)]

    @staticmethod
    def server_env(env_over):
        """Environment for one llama-server. The GPU list in Settings uses nvidia-smi's numbering (PCI
        bus order), but CUDA numbers the fastest card first unless CUDA_DEVICE_ORDER says otherwise, so
        on mixed cards CUDA_VISIBLE_DEVICES=1 could pick nvidia-smi's GPU 0."""
        env = dict(os.environ)
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env.update(env_over)
        return env

    def begin(self):
        """Claim the next start and return its ticket for start(). From here on, a Stop
        (cancel()/stop()) cancels that start. Call it where Stop is handled (the UI thread) before
        handing start() to another thread: resetting the stop flag inside start() wiped out a Stop
        pressed before that thread got going, or while the old servers were being killed (up to 10 s)."""
        with self._lock:
            self._stop = False
            self._gen += 1
            old, self.procs, self.endpoints = self.procs, [], []
            return self._gen, old

    def start(self, s, log, progress=None, ticket=None):
        """Start servers and block until all report healthy. Returns endpoint list.

        Raises StartCancelled when stop()/cancel() is called meanwhile (after begin(), when a ticket
        is passed). On any failure the processes this call launched are killed here, so the caller
        doesn't need to stop() (which could kill servers a newer start already launched)."""
        gen, old = ticket if ticket is not None else self.begin()
        self._kill([p for p, _ in old])  # the servers this start replaces
        procs, endpoints = [], []
        try:
            eps = self._start(s, log, progress, gen, procs, endpoints)
            with self._lock:
                if self._cancelled(gen):  # stopped after the last health check: don't report ready
                    raise StartCancelled("Cancelled")
            return eps
        except BaseException:
            self._kill([p for p, _ in procs])
            with self._lock:
                if self._gen == gen:
                    self.procs, self.endpoints = [], []
            raise

    def _cancelled(self, gen):
        return self._stop or self._gen != gen

    def _start(self, s, log, progress, gen, procs, endpoints):
        exe, model = s["llama_server"], s["model"]
        if not os.path.isfile(exe):
            raise RuntimeError("llama-server.exe not found: set its path in Settings.")
        if not os.path.isfile(model):
            raise RuntimeError("Model (.gguf) not found: set its path in Settings.")
        if s["mmproj"] and not os.path.isfile(s["mmproj"]):
            raise RuntimeError("mmproj file not found.")
        slots = max(1, int(s["slots_per_server"]))
        extra = shlex.split(s["extra_args"], posix=False) if s["extra_args"].strip() else []
        extra = [a.strip('"') for a in extra]
        if not s["mmproj"]:
            log("note: no vision projector (mmproj) is set; without it most models can't see the images "
                "and every image fails. Set it in Settings ▸ Backend.")
        plan = self.build_plan(s)
        busy = [port for port, _, _ in plan if port_in_use(port)]
        if busy:
            raise RuntimeError(f"Port {', '.join(map(str, busy))} is already in use by another program "
                               "(possibly a llama-server that is still running). Close it, or change "
                               "'First port' in Settings ▸ Backend.")

        for port, env_over, plan_args in plan:
            cmd = [exe, "-m", model, "-ngl", str(s["ngl"]), "-c", str(int(s["ctx_size"]) * slots),
                   "-np", str(slots), "--host", "127.0.0.1", "--port", str(port)]
            if s["mmproj"]:
                cmd += ["--mmproj", s["mmproj"]]
            if s["disable_thinking"]:
                cmd += ["--reasoning", "off"]
            else:
                cmd += ["--reasoning-budget", str(int(s["thinking_budget"]))]
            cmd += plan_args + extra
            env = self.server_env(env_over)
            tag = f"[:{port}"
            tag += f" GPU {env_over['CUDA_VISIBLE_DEVICES']}]" if env_over else "]"
            log(f"{tag} launching: {' '.join(cmd)}")
            with self._lock:  # stop() can't miss a process launched while it runs
                if self._cancelled(gen):
                    raise StartCancelled("Cancelled")
                p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, creationflags=CREATE_NO_WINDOW,
                                     cwd=os.path.dirname(exe) or None)
                p.job = winjob.adopt(p.pid)
                if p.job is None:
                    log(f"{tag} note: couldn't tie llama-server to the app; if the app crashes, end it in Task Manager")
                tail = []
                procs.append((p, tail))
                endpoints.append({"url": f"http://127.0.0.1:{port}/v1", "model": "default",
                                  "slots": slots, "name": tag.strip("[]")})
                self.procs, self.endpoints = list(procs), list(endpoints)
            threading.Thread(target=self._pump, args=(p, tag, log, tail), daemon=True).start()

        deadline = time.time() + 900
        pending = list(range(len(procs)))
        with local_session() as health:
            while pending:
                if self._cancelled(gen):
                    raise StartCancelled("Cancelled")
                if time.time() > deadline:
                    raise RuntimeError("Timed out waiting for llama-server to load the model.")
                for i in list(pending):
                    p, tail = procs[i]
                    if p.poll() is not None:
                        if self._cancelled(gen):  # killed by stop(), not a crash
                            raise StartCancelled("Cancelled")
                        raise RuntimeError("llama-server exited (code %s):\n%s" % (p.returncode, "\n".join(tail[-12:])))
                    url = endpoints[i]["url"].replace("/v1", "/health")
                    try:
                        if health.get(url, timeout=2).status_code == 200:
                            pending.remove(i)
                            log(f"[{endpoints[i]['name']}] ready")
                    except requests.RequestException:
                        pass
                if progress:
                    progress(len(procs) - len(pending), len(procs))
                time.sleep(0.25)
        return list(endpoints)

    @staticmethod
    def _pump(p, tag, log, tail):
        try:
            for line in iter(p.stdout.readline, b""):
                text = line.decode("utf-8", "replace").rstrip()
                tail.append(text)
                del tail[:-50]
                log(f"{tag} {text}")
        except (OSError, ValueError):
            pass
        finally:
            try:
                p.stdout.close()
            except OSError:
                pass

    def cancel(self):
        self._stop = True

    def running(self):
        procs = self.procs
        return bool(procs) and all(p.poll() is None for p, _ in procs)

    @staticmethod
    def _kill(procs):
        for p in procs:
            job = getattr(p, "job", None)
            if job is not None and job.terminate():
                continue  # the server and anything it started, at once
            if p.poll() is None:  # not in a job: taskkill finds its child processes itself (~0.75 s)
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)],
                               capture_output=True, creationflags=CREATE_NO_WINDOW)
        deadline = time.monotonic() + 10
        for p in procs:  # its port and VRAM are only free once it has exited; a restart needs both
            try:
                p.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                pass
            job = getattr(p, "job", None)
            if job is not None:
                # and so has anything it started: a .cmd wrapper exits before the server inside it
                job.wait(max(0.1, deadline - time.monotonic()))
                job.close()

    def stop(self):
        with self._lock:
            self._stop = True
            self._gen += 1
            procs, self.procs, self.endpoints = self.procs, [], []
        self._kill([p for p, _ in procs])


def external_endpoints(s):
    out = []
    for line in s["external_urls"].replace(",", "\n").splitlines():
        u = line.strip().rstrip("/")
        if u.endswith("/chat/completions"):  # the full request URL pasted from a server's docs
            u = u[:-len("/chat/completions")].rstrip("/")
        if u:
            if not u.endswith("/v1"):
                u += "/v1"
            out.append({"url": u, "model": s["external_model"] or "default",
                        "slots": max(1, int(s["slots_per_server"])), "name": u})
    return out


# --------------------------------------------------------------------------- captioning
# formats every JPEG decoder reads the same way (3-channel colour, or grey)
JPEG_READY_FORMATS = (QImage.Format_RGB32, QImage.Format_RGB888, QImage.Format_Grayscale8)


def encode_image(path, max_side):
    """Load (honouring EXIF), downscale and return base64 JPEG."""
    r = QImageReader(path)
    r.setAutoTransform(True)
    sz = r.size()  # stored size; the EXIF rotation is applied after scaling, so the long side still matches
    if sz.isValid() and max(sz.width(), sz.height()) > max_side:
        sz.scale(max_side, max_side, Qt.KeepAspectRatio)
        r.setScaledSize(sz)  # decoders like JPEG scale while decoding: far less work than a full decode
    img = r.read()
    if img.isNull():
        raise RuntimeError("Could not read image: " + r.errorString())
    if max(img.width(), img.height()) > max_side:
        img = img.scaled(max_side, max_side, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    if img.hasAlphaChannel():  # JPEG has no transparency: show see-through parts as white, not black
        flat = QImage(img.size(), QImage.Format_RGB32)
        flat.fill(QColor("white"))
        pt = QPainter(flat)
        pt.drawImage(0, 0, img)
        pt.end()
        img = flat
    elif img.format() not in JPEG_READY_FORMATS:
        # Qt 6.8+ reads CMYK JPEGs (print and stock photos) as CMYK and would save them that way, and
        # decoders disagree on CMYK JPEGs (stb_image, used by llama.cpp, expects Adobe's inverted ink
        # values), so the server could see other colours than the thumbnail shows. Send plain RGB.
        img = img.convertToFormat(QImage.Format_RGB32)
    ba = QByteArray()
    buf = QBuffer(ba)
    buf.open(QIODevice.WriteOnly)
    img.save(buf, "JPEG", 92)
    return base64.b64encode(bytes(ba)).decode("ascii")


_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
_THINK_OPEN = re.compile(r"<think>", re.I)
_THINK_CLOSE = re.compile(r"</think>", re.I)

OUT_OF_TOKENS = ("Ran out of tokens before answering (the model was still thinking). "
                 "Tick 'Disable thinking' or raise 'Max new tokens' in Settings ▸ Generation.")
CUT_OFF = ("Caption cut off at 'Max new tokens' and saved as it is. Raise the limit in "
           "Settings ▸ Generation for full-length captions.")

MAX_ATTEMPTS = 4          # one try plus three retries
RETRY_STATUSES = {408, 429}  # plus every 5xx
# (connect, read) seconds. Connecting to a running server is instant; a reply can take minutes on a
# slow model, so only the read gets the long limit.
REQUEST_TIMEOUT = (10, 900)


class EndpointDown(RuntimeError):
    """The server refused or dropped the connection on every attempt: it isn't running."""


class Stopped(RuntimeError):
    """The job was stopped while a request waited to be retried."""


class HTTPStatusError(RuntimeError):
    def __init__(self, status, text, retry_after=None):
        super().__init__(f"HTTP {status}: {text}")
        self.status = status
        self.retry_after = retry_after


def is_retryable(exc):
    """Connection problems, 408/429 and 5xx are worth another go; other errors are not. Neither is a
    reply that took the whole 15-minute read timeout: three more tries could hold one image for an hour."""
    if isinstance(exc, HTTPStatusError):
        return exc.status in RETRY_STATUSES or 500 <= exc.status <= 599
    # a 200 reply that isn't JSON (a web page at that URL) comes back the same every time; requests
    # files it under RequestException, so it was asked four times for each image
    if isinstance(exc, (requests.ReadTimeout, requests.exceptions.InvalidJSONError)):
        return False
    return isinstance(exc, requests.RequestException)


def retry_delay(attempt, retry_after=None):
    """Seconds to wait before retry number `attempt` (1-based): 1, 2, 4… capped at 30, or the
    server's Retry-After (seconds) when it sent one."""
    try:
        if retry_after is not None:
            return max(0.0, min(30.0, float(retry_after)))
    except (TypeError, ValueError):
        pass  # an HTTP date: fall back to the backoff
    return float(min(30, 2 ** (attempt - 1)))


def clean_caption(text, s):
    """The model's reply as a caption, with the prefix and suffix added. Raises when the reply holds
    no caption, so a prefix alone or unfinished thinking is never saved as one."""
    text = _THINK.sub("", text)
    text = _THINK_CLOSE.split(text)[-1]  # the chat template opened <think> in the prompt
    if _THINK_OPEN.search(text):  # thinking that never finished: the token limit hit first
        raise RuntimeError(OUT_OF_TOKENS)
    text = text.strip()
    if s["single_line"]:
        text = " ".join(text.split())
    if not text:
        raise RuntimeError("Model returned an empty caption")
    return f"{s['prefix']}{text}{s['suffix']}"


def request_caption(ep, image_b64, system, user, s, session):
    """POST one image. Returns (reply text, finish_reason)."""
    msgs = []
    if system.strip():
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": [
        {"type": "text", "text": user or "Describe this image."},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + image_b64}},
    ]})
    headers = {"Authorization": "Bearer " + s["api_key"]} if s["api_key"] else {}
    body = {"model": ep["model"], "messages": msgs, "temperature": float(s["temperature"]), "stream": False}
    budget = int(s["thinking_budget"])
    if s["disable_thinking"]:
        body["max_tokens"] = int(s["max_tokens"])
        body["chat_template_kwargs"] = {"enable_thinking": False}
    elif budget >= 0:
        body["max_tokens"] = int(s["max_tokens"]) + budget  # leave room for the answer after the thinking
    # else unlimited thinking: no cap at all, or the thinking would use up 'Max new tokens' by itself
    # (the context size still bounds the reply)
    r = session.post(ep["url"] + "/chat/completions", json=body, headers=headers, timeout=REQUEST_TIMEOUT)
    if r.status_code != 200:
        raise HTTPStatusError(r.status_code, r.text[:300], r.headers.get("Retry-After"))
    choice = r.json()["choices"][0]
    content = choice["message"].get("content") or ""
    finish = choice.get("finish_reason")
    if not content.strip() and finish == "length":
        raise RuntimeError(OUT_OF_TOKENS)
    return content, finish


def request_with_retry(ep, image_b64, system, user, s, session, wait=time.sleep):
    """request_caption, retried with backoff on errors is_retryable() accepts.

    `wait(seconds)` does the pause and returns True to give up (e.g. threading.Event.wait
    for a cancel flag)."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return request_caption(ep, image_b64, system, user, s, session)
        except Exception as e:  # noqa: BLE001
            if not is_retryable(e):
                raise
            if attempt == MAX_ATTEMPTS:
                kind = EndpointDown if isinstance(e, requests.ConnectionError) else RuntimeError
                raise kind(f"{ep['name']}: {e} (gave up after {MAX_ATTEMPTS} attempts)") from e
            if wait(retry_delay(attempt, getattr(e, "retry_after", None))):
                raise Stopped(f"{ep['name']}: {e} (stopped)") from e


class CaptionEngine(QObject):
    started = Signal(str)          # path: its request is on its way to a server
    requeued = Signal(str)         # path: handed back to the queue (its server stopped responding)
    captioned = Signal(str, str)   # path, caption
    failed = Signal(str, str)      # path, error
    warning = Signal(str, str)     # path ("" = about the whole job), message; not a failure
    progress = Signal(int, int)    # done, total
    finished = Signal()

    def __init__(self):
        super().__init__()
        self._q = queue.Queue()
        self._cancel = threading.Event()
        self._threads = []
        self._lock = threading.Lock()
        self._done = 0
        self._total = 0
        self._running = 0
        self._names = set()   # every endpoint of the job
        self._down = set()    # endpoints that stopped responding
        self._dropped = set()  # paths taken off the list during the job: not worth the GPU time
        self.active = False
        self.stop_reason = ""  # set when the engine stopped the job itself (no server responding)

    def start(self, paths, endpoints, system, user, s):
        if self.active:
            return
        self.stop_reason = ""
        if not endpoints:
            return
        self._cancel.clear()
        self._q = queue.Queue()
        for p in paths:
            self._q.put(p)
        self._done, self._total = 0, len(paths)
        self._names, self._down, self._dropped = {ep["name"] for ep in endpoints}, set(), set()
        self.active = True
        self.progress.emit(0, self._total)
        self._threads = []
        self._running = sum(ep["slots"] for ep in endpoints)
        for ep in endpoints:
            for _ in range(ep["slots"]):
                t = threading.Thread(target=self._worker, args=(ep, system, user, dict(s)), daemon=True)
                self._threads.append(t)
                t.start()

    def cancel(self):
        self._cancel.set()

    def drop(self, paths):
        """Skip `paths` if they are still waiting (they were removed from the list); requests already
        on their way finish, and the window ignores their results."""
        with self._lock:
            self._dropped.update(paths)

    def is_dropped(self, path):
        """True when `path` was removed from the list during this job (its result is unwanted, even if
        the image was added again since)."""
        return path in self._dropped

    def _take(self, pool, max_side):
        """Next queued path and a future encoding it, or None when done/cancelled."""
        while not self._cancel.is_set():
            try:
                path = self._q.get_nowait()
            except queue.Empty:
                return None
            with self._lock:
                dropped = path in self._dropped
                if dropped:
                    self._done += 1
                    done = self._done
            if not dropped:
                return path, pool.submit(encode_image, path, max_side)
            self.progress.emit(done, self._total)
        return None

    def _worker(self, ep, system, user, s):
        session = local_session(ep["url"])
        max_side = int(s["max_image_side"])
        last = False
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                while True:
                    if self._run(pool, ep, system, user, s, session, max_side):
                        return  # retired: _retire() already took this worker off the running count
                    with self._lock:
                        # a retiring worker can hand images back after this one found the queue empty;
                        # hand-backs happen under this lock, so looking again here can't miss one
                        if self._cancel.is_set() or self._q.empty():
                            self._running -= 1
                            last = self._running == 0
                            break
        finally:
            session.close()
            if last:  # last worker out reports completion
                self.active = False
                self.finished.emit()

    def _run(self, pool, ep, system, user, s, session, max_side):
        """Caption queued images until the queue is empty or the job stops. Returns True when this
        worker retired because its endpoint is down and its images went back in the queue."""
        nxt = self._take(pool, max_side)
        while nxt is not None and not self._cancel.is_set():
            path, fut = nxt
            nxt = self._take(pool, max_side)  # encode the next image while this one is in flight
            try:
                b64 = fut.result()
                self.started.emit(path)
                raw, finish = request_with_retry(ep, b64, system, user, s, session, wait=self._cancel.wait)
                self.captioned.emit(path, clean_caption(raw, s))
                if finish == "length":
                    self.warning.emit(path, CUT_OFF)
            except Stopped:
                pass  # stopped while waiting to retry: the image goes back to idle, it didn't fail
            except EndpointDown as e:
                if self._endpoint_down(ep, e, path, nxt):
                    return True
            except Exception as e:  # noqa: BLE001
                self.failed.emit(path, str(e))
            with self._lock:
                self._done += 1
                done = self._done
            self.progress.emit(done, self._total)
        if nxt is not None:  # cancelled with one encoded ahead: don't wait on it
            nxt[1].cancel()
        return False

    def _endpoint_down(self, ep, err, path, nxt):
        """Decide what a dead endpoint means for the job. Returns True when this worker retires."""
        if self._cancel.is_set():
            return False  # the user stopped the job (and maybe the servers): not a failure
        with self._lock:
            self._down.add(ep["name"])
            all_down = self._down >= self._names
        if all_down:
            # nothing can caption the rest; stop now instead of failing every image after its retries
            self.stop_reason = f"No server is responding, so captioning stopped.\n\nLast error: {err}"
            self._cancel.set()
            return False
        held = [path] + ([nxt[0]] if nxt is not None else [])
        # before the hand-back: once it is back in the queue another worker can report it started, and
        # a later "requeued" would arrive after that (the prefetched one never showed as started)
        self.requeued.emit(path)
        if self._retire(held):
            if nxt is not None:
                nxt[1].cancel()
            self.warning.emit("", f"{ep['name']} stopped responding; the other server(s) take over its images. ({err})")
            return True
        if self._cancel.is_set():
            # the last other server went down too and stopped the job meanwhile (it left, so the
            # hand-back above found nobody): the image stays uncaptioned, like the others
            return False
        # every other worker already finished (the queue was empty): nobody to hand this image to
        self.failed.emit(path, str(err))
        return False

    def _retire(self, paths):
        """Put `paths` back in the queue and leave, if another worker is still running to take them."""
        with self._lock:
            if self._running <= 1:
                return False
            for p in paths:
                self._q.put(p)
            self._running -= 1
            return True
