"""llama.cpp server management and the multi-worker captioning engine."""
import atexit
import base64
import os
import queue
import re
import shlex
import subprocess
import threading
import time

import requests
from PySide6.QtCore import QBuffer, QByteArray, QIODevice, QObject, Qt, Signal
from PySide6.QtGui import QColor, QImage, QImageReader, QPainter

CREATE_NO_WINDOW = 0x08000000


# --------------------------------------------------------------------------- servers
class ServerManager:
    """Launches llama-server processes (one per GPU, or one spanning GPUs)."""

    def __init__(self):
        self.procs = []
        self.endpoints = []
        self._stop = False
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

    def start(self, s, log, progress=None):
        """Start servers and block until all report healthy. Returns endpoint list."""
        self.stop()
        self._stop = False
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

        for port, env_over, plan_args in self.build_plan(s):
            cmd = [exe, "-m", model, "-ngl", str(s["ngl"]), "-c", str(int(s["ctx_size"]) * slots),
                   "-np", str(slots), "--host", "127.0.0.1", "--port", str(port)]
            if s["mmproj"]:
                cmd += ["--mmproj", s["mmproj"]]
            if s["disable_thinking"]:
                cmd += ["--reasoning", "off"]
            else:
                cmd += ["--reasoning-budget", str(int(s["thinking_budget"]))]
            cmd += plan_args + extra
            env = dict(os.environ)
            env.update(env_over)
            tag = f"[:{port}"
            tag += f" GPU {env_over['CUDA_VISIBLE_DEVICES']}]" if env_over else "]"
            log(f"{tag} launching: {' '.join(cmd)}")
            p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, creationflags=CREATE_NO_WINDOW,
                                 cwd=os.path.dirname(exe) or None)
            tail = []
            threading.Thread(target=self._pump, args=(p, tag, log, tail), daemon=True).start()
            self.procs.append((p, tail))
            self.endpoints.append({"url": f"http://127.0.0.1:{port}/v1", "model": "default",
                                   "slots": slots, "name": tag.strip("[]")})

        deadline = time.time() + 900
        pending = list(range(len(self.procs)))
        while pending:
            if self._stop:
                raise RuntimeError("Cancelled")
            if time.time() > deadline:
                raise RuntimeError("Timed out waiting for llama-server to load the model.")
            for i in list(pending):
                p, tail = self.procs[i]
                if p.poll() is not None:
                    raise RuntimeError("llama-server exited (code %s):\n%s" % (p.returncode, "\n".join(tail[-12:])))
                url = self.endpoints[i]["url"].replace("/v1", "/health")
                try:
                    if requests.get(url, timeout=2).status_code == 200:
                        pending.remove(i)
                        log(f"[{self.endpoints[i]['name']}] ready")
                except requests.RequestException:
                    pass
            if progress:
                progress(len(self.procs) - len(pending), len(self.procs))
            time.sleep(1)
        return list(self.endpoints)

    @staticmethod
    def _pump(p, tag, log, tail):
        for line in iter(p.stdout.readline, b""):
            text = line.decode("utf-8", "replace").rstrip()
            tail.append(text)
            del tail[:-50]
            log(f"{tag} {text}")

    def cancel(self):
        self._stop = True

    def running(self):
        return bool(self.procs) and all(p.poll() is None for p, _ in self.procs)

    def stop(self):
        for p, _ in self.procs:
            if p.poll() is None:
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)],
                               capture_output=True, creationflags=CREATE_NO_WINDOW)
        self.procs, self.endpoints = [], []


def external_endpoints(s):
    out = []
    for line in s["external_urls"].replace(",", "\n").splitlines():
        u = line.strip().rstrip("/")
        if u:
            if not u.endswith("/v1"):
                u += "/v1"
            out.append({"url": u, "model": s["external_model"] or "default",
                        "slots": max(1, int(s["slots_per_server"])), "name": u})
    return out


# --------------------------------------------------------------------------- captioning
def encode_image(path, max_side):
    """Load (honouring EXIF), downscale and return base64 JPEG."""
    r = QImageReader(path)
    r.setAutoTransform(True)
    img = r.read()
    if img.isNull():
        raise RuntimeError("Could not read image: " + r.errorString())
    if max(img.width(), img.height()) > max_side:
        img = img.scaled(max_side, max_side, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    flat = QImage(img.size(), QImage.Format_RGB32)
    flat.fill(QColor("white"))
    pt = QPainter(flat)
    pt.drawImage(0, 0, img)
    pt.end()
    ba = QByteArray()
    buf = QBuffer(ba)
    buf.open(QIODevice.WriteOnly)
    flat.save(buf, "JPEG", 92)
    return base64.b64encode(bytes(ba)).decode("ascii")


_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)


def clean_caption(text, s):
    text = _THINK.sub("", text)
    text = text.split("</think>")[-1].strip()
    if s["single_line"]:
        text = " ".join(text.split())
    return f"{s['prefix']}{text}{s['suffix']}"


def request_caption(ep, image_b64, system, user, s, session):
    msgs = []
    if system.strip():
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": [
        {"type": "text", "text": user or "Describe this image."},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + image_b64}},
    ]})
    headers = {"Authorization": "Bearer " + s["api_key"]} if s["api_key"] else {}
    max_tokens = int(s["max_tokens"])
    budget = int(s["thinking_budget"])
    if not s["disable_thinking"] and budget > 0:
        max_tokens += budget  # leave room for the answer after the thinking
    body = {"model": ep["model"], "messages": msgs, "temperature": float(s["temperature"]),
            "max_tokens": max_tokens, "stream": False}
    if s["disable_thinking"]:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    r = session.post(ep["url"] + "/chat/completions", json=body, headers=headers, timeout=900)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    choice = r.json()["choices"][0]
    content = choice["message"].get("content") or ""
    if not content.strip() and choice.get("finish_reason") == "length":
        raise RuntimeError("Ran out of tokens before answering (the model was still thinking). "
                           "Tick 'Disable thinking' or raise 'Max new tokens' in Settings ▸ Generation.")
    return content


class CaptionEngine(QObject):
    captioned = Signal(str, str)   # path, caption
    failed = Signal(str, str)      # path, error
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
        self.active = False

    def start(self, paths, endpoints, system, user, s):
        if self.active:
            return
        self._cancel.clear()
        self._q = queue.Queue()
        for p in paths:
            self._q.put(p)
        self._done, self._total = 0, len(paths)
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

    def _worker(self, ep, system, user, s):
        session = requests.Session()
        while not self._cancel.is_set():
            try:
                path = self._q.get_nowait()
            except queue.Empty:
                break
            try:
                b64 = encode_image(path, int(s["max_image_side"]))
                err = None
                for attempt in range(2):
                    try:
                        raw = request_caption(ep, b64, system, user, s, session)
                        break
                    except requests.RequestException as e:
                        err = e
                        time.sleep(1)
                else:
                    raise RuntimeError(f"{ep['name']}: {err}")
                text = clean_caption(raw, s)
                if not text.strip():
                    raise RuntimeError("Model returned an empty caption")
                self.captioned.emit(path, text)
            except Exception as e:  # noqa: BLE001
                self.failed.emit(path, str(e))
            with self._lock:
                self._done += 1
                done = self._done
            self.progress.emit(done, self._total)
        # last worker out reports completion
        with self._lock:
            self._running -= 1
            if self._running == 0:
                self.active = False
                self.finished.emit()
