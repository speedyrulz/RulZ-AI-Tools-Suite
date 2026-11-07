"""Run with:  python -m pytest tests   (from the app folder, inside its .venv)."""
import base64
import ctypes
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from ctypes import wintypes

import pytest
import requests

os.environ.setdefault("QT_QPA_PLATFORM", "minimal")  # no window needed; "offscreen" can hang on Windows
APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP_DIR)

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtGui import QColor, QGuiApplication, QImage, QImageReader  # noqa: E402

from captioner import backend  # noqa: E402
from captioner.model import ImageModel, caption_clashes, decode_caption  # noqa: E402

APP = QGuiApplication.instance() or QGuiApplication([])

EP = {"url": "http://127.0.0.1:1/v1", "model": "default", "slots": 1, "name": "test"}
S = {"api_key": "", "max_tokens": 10, "thinking_budget": 0, "disable_thinking": True, "temperature": 0.1}


class FakeResponse:
    def __init__(self, status, text="", headers=None, content="a cat", finish="stop"):
        self.status_code, self.text, self.headers = status, text, headers or {}
        self.content, self.finish = content, finish

    def json(self):
        return {"choices": [{"message": {"content": self.content}, "finish_reason": self.finish}]}


class FakeSession:
    def __init__(self, replies):
        self.replies, self.calls, self.bodies = list(replies), 0, []

    def post(self, *a, **k):
        self.calls += 1
        self.bodies.append(k.get("json"))
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


# ------------------------------------------------------------------ retry policy
@pytest.mark.parametrize("status,want", [(408, True), (429, True), (500, True), (503, True), (599, True),
                                         (400, False), (401, False), (404, False), (413, False)])
def test_is_retryable_status(status, want):
    assert backend.is_retryable(backend.HTTPStatusError(status, "x")) is want


def test_is_retryable_other_errors():
    assert backend.is_retryable(requests.ConnectionError("down"))
    assert backend.is_retryable(requests.Timeout("slow"))
    assert not backend.is_retryable(RuntimeError("Ran out of tokens"))


def test_retry_delay_backoff_and_retry_after():
    assert [backend.retry_delay(a) for a in (1, 2, 3, 4)] == [1, 2, 4, 8]
    assert backend.retry_delay(10) == 30
    assert backend.retry_delay(1, "5") == 5
    assert backend.retry_delay(1, "999") == 30
    assert backend.retry_delay(2, "Wed, 21 Oct 2015 07:28:00 GMT") == 2


def test_retries_503_then_succeeds():
    sess = FakeSession([FakeResponse(503, "loading"), FakeResponse(429, headers={"Retry-After": "3"}),
                        FakeResponse(200)])
    waits = []
    out = backend.request_with_retry(EP, "", "", "hi", S, sess, wait=lambda d: waits.append(d))
    assert out == ("a cat", "stop") and sess.calls == 3 and waits == [1, 3]


def test_no_retry_on_400():
    sess = FakeSession([FakeResponse(400, "bad request"), FakeResponse(200)])
    with pytest.raises(backend.HTTPStatusError):
        backend.request_with_retry(EP, "", "", "hi", S, sess, wait=lambda d: None)
    assert sess.calls == 1


def test_gives_up_after_max_attempts():
    sess = FakeSession([requests.ConnectionError("down")] * backend.MAX_ATTEMPTS)
    with pytest.raises(backend.EndpointDown, match="gave up"):  # refused every time: the server is down
        backend.request_with_retry(EP, "", "", "hi", S, sess, wait=lambda d: None)
    assert sess.calls == backend.MAX_ATTEMPTS


def test_gives_up_on_5xx_without_calling_the_server_down():
    sess = FakeSession([FakeResponse(503)] * backend.MAX_ATTEMPTS)
    with pytest.raises(RuntimeError, match="gave up") as info:
        backend.request_with_retry(EP, "", "", "hi", S, sess, wait=lambda d: None)
    assert not isinstance(info.value, backend.EndpointDown)  # it answered, so it is running


def test_cancel_during_backoff_stops():
    sess = FakeSession([FakeResponse(502), FakeResponse(200)])
    ev = threading.Event()
    ev.set()
    with pytest.raises(backend.Stopped, match="stopped"):
        backend.request_with_retry(EP, "", "", "hi", S, sess, wait=ev.wait)
    assert sess.calls == 1


# ------------------------------------------------------------------ request body and reply checks
@pytest.mark.parametrize("disable,budget,want", [(True, 1024, 10), (False, 1024, 1034), (False, 0, 10),
                                                 (False, -1, None)])
def test_max_tokens(disable, budget, want):
    sess = FakeSession([FakeResponse(200)])
    backend.request_caption(EP, "", "", "hi", dict(S, disable_thinking=disable, thinking_budget=budget), sess)
    assert sess.bodies[0].get("max_tokens") == want  # None: unlimited thinking sends no cap at all


def test_reply_reports_finish_reason():
    sess = FakeSession([FakeResponse(200, finish="length")])
    assert backend.request_caption(EP, "", "", "hi", S, sess) == ("a cat", "length")


CAP = {"single_line": True, "prefix": "ohwx person, ", "suffix": ""}


def test_clean_caption_adds_prefix():
    assert backend.clean_caption("<think>hm</think>\n A cat\non a mat. ", CAP) == "ohwx person, A cat on a mat."
    # the chat template opened <think> in the prompt, so only the closing tag is in the reply
    assert backend.clean_caption("reasoning…</think>A cat.", CAP) == "ohwx person, A cat."


@pytest.mark.parametrize("reply", ["", "   ", "<think>all thinking</think>", "<THINK>x</THINK>\n"])
def test_clean_caption_never_saves_a_bare_prefix(reply):
    with pytest.raises(RuntimeError, match="empty caption"):
        backend.clean_caption(reply, CAP)


def test_clean_caption_rejects_unfinished_thinking():
    with pytest.raises(RuntimeError, match="Ran out of tokens"):
        backend.clean_caption("<think>The user wants a caption. Let me look at", CAP)


# ------------------------------------------------------------------ servers
def test_server_env_numbers_gpus_like_nvidia_smi(monkeypatch):
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "FASTEST_FIRST")
    env = backend.ServerManager.server_env({"CUDA_VISIBLE_DEVICES": "1"})
    assert env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID" and env["CUDA_VISIBLE_DEVICES"] == "1"


def test_port_in_use():
    with socket.socket() as srv:
        srv.bind(("127.0.0.1", 0))
        srv.listen()
        port = srv.getsockname()[1]
        assert backend.port_in_use(port)
    assert not backend.port_in_use(port)


def _wait_exit(pid, ms):
    """True once process `pid` has exited (or never existed), waiting up to `ms`."""
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.OpenProcess.restype = wintypes.HANDLE
    k.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    k.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    k.CloseHandle.argtypes = (wintypes.HANDLE,)
    h = k.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
    if not h:
        return True
    try:
        return k.WaitForSingleObject(h, ms) == 0  # WAIT_OBJECT_0: it exited
    finally:
        k.CloseHandle(h)


@pytest.mark.skipif(os.name != "nt", reason="Windows job objects")
def test_adopted_process_dies_when_the_app_crashes():
    script = (
        "import os, subprocess, sys\n"
        f"sys.path.insert(0, {APP_DIR!r})\n"
        "from captioner import winjob\n"
        "d = subprocess.DEVNULL\n"
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], stdin=d, stdout=d, stderr=d)\n"
        "print(p.pid, winjob.adopt(p.pid), flush=True)\n"
        "os._exit(0)\n"  # no atexit, no cleanup: like a crash or a Task Manager kill
    )
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60)
    pid, adopted = out.stdout.split()
    pid = int(pid)
    try:
        assert adopted == "True"
        assert _wait_exit(pid, 10000), "the child outlived the app"
    finally:
        if not _wait_exit(pid, 0):
            os.kill(pid, signal.SIGTERM)


# ------------------------------------------------------------------ engine
ENGINE_S = {"max_image_side": 1024, "single_line": True, "prefix": "", "suffix": ""}


def _ep(name):
    return {"url": f"http://{name}/v1", "model": "default", "slots": 1, "name": name}


def run_engine(monkeypatch, endpoints, reply, n=12):
    """Run a real CaptionEngine with fake encoding and requests; return what it reported."""
    monkeypatch.setattr(backend, "encode_image", lambda path, side: "b64")
    monkeypatch.setattr(backend, "retry_delay", lambda attempt, retry_after=None: 0)
    eng = backend.CaptionEngine()
    monkeypatch.setattr(backend, "request_caption", lambda ep, *a: reply(eng, ep))
    got = {"captioned": [], "failed": [], "warning": []}
    done = threading.Event()
    for name in got:
        getattr(eng, name).connect(lambda *args, name=name: got[name].append(args), Qt.DirectConnection)
    eng.finished.connect(done.set, Qt.DirectConnection)
    paths = [f"img{i}.png" for i in range(n)]
    eng.start(paths, endpoints, "", "hi", ENGINE_S)
    assert done.wait(10), "the engine never finished"
    for t in eng._threads:
        t.join(5)
    assert not eng.active
    return eng, paths, got


def test_dead_endpoint_hands_its_images_to_the_others(monkeypatch):
    def reply(eng, ep):
        if ep["name"] == "dead":
            raise requests.ConnectionError("refused")
        time.sleep(0.02)
        return "a cat", "stop"
    eng, paths, got = run_engine(monkeypatch, [_ep("ok"), _ep("dead")], reply)
    assert sorted(p for p, _ in got["captioned"]) == sorted(paths)  # every image, none twice
    assert got["failed"] == [] and eng.stop_reason == ""
    assert len(got["warning"]) == 1 and "dead stopped responding" in got["warning"][0][1]


def test_every_endpoint_down_stops_instead_of_failing_each_image(monkeypatch):
    def reply(eng, ep):
        raise requests.ConnectionError("refused")
    eng, _, got = run_engine(monkeypatch, [_ep("a"), _ep("b")], reply)
    assert got["captioned"] == [] and got["failed"] == []
    assert eng.stop_reason.startswith("No server is responding")


def test_stopping_mid_request_is_not_a_failure(monkeypatch):
    def reply(eng, ep):  # the user stops the job, then the servers drop the connection
        eng.cancel()
        raise requests.ConnectionError("reset")
    eng, _, got = run_engine(monkeypatch, [_ep("a")], reply)
    assert got["captioned"] == [] and got["failed"] == [] and eng.stop_reason == ""


def test_cut_off_caption_is_saved_with_a_warning(monkeypatch):
    eng, paths, got = run_engine(monkeypatch, [_ep("a")], lambda eng, ep: ("a cat on a", "length"), n=1)
    assert got["captioned"] == [(paths[0], "a cat on a")]
    assert got["warning"] == [(paths[0], backend.CUT_OFF)]


def test_bad_reply_fails_that_image(monkeypatch):
    eng, paths, got = run_engine(monkeypatch, [_ep("a")], lambda eng, ep: ("<think>still going", "length"), n=2)
    assert got["captioned"] == [] and len(got["failed"]) == 2 and "Ran out of tokens" in got["failed"][0][1]


# ------------------------------------------------------------------ image encode
def test_encode_image_downscales(tmp_path):
    img = QImage(3000, 1500, QImage.Format_RGB32)
    img.fill(QColor("red"))
    p = str(tmp_path / "big.jpg")
    assert img.save(p, "JPEG")
    out = QImage.fromData(base64.b64decode(backend.encode_image(p, 1024)))
    assert (out.width(), out.height()) == (1024, 512)


def test_encode_image_small_untouched(tmp_path):
    img = QImage(300, 200, QImage.Format_ARGB32)
    img.fill(QColor(0, 0, 0, 0))
    p = str(tmp_path / "small.png")
    assert img.save(p, "PNG")
    out = QImage.fromData(base64.b64decode(backend.encode_image(p, 1024)))
    assert (out.width(), out.height()) == (300, 200)


def test_encode_image_reads_png_over_qt_default_limit(tmp_path):
    img = QImage(8200, 8200, QImage.Format_RGB32)  # 67 MP: needs 269 MB, over Qt's default 256 MB
    img.fill(QColor("gray"))
    p = str(tmp_path / "huge.png")
    assert img.save(p, "PNG")
    del img
    QImageReader.setAllocationLimit(256)
    with pytest.raises(RuntimeError, match="Could not read image"):
        backend.encode_image(p, 1024)
    backend.allow_large_images()
    out = QImage.fromData(base64.b64decode(backend.encode_image(p, 1024)))
    assert (out.width(), out.height()) == (1024, 1024)


# ------------------------------------------------------------------ captions / model
def test_decode_caption_fallbacks():
    assert decode_caption("café".encode("utf-8")) == "café"
    assert decode_caption(b"\xef\xbb\xbf" + "café".encode("utf-8")) == "café"
    assert decode_caption("café – x".encode("cp1252")) == "café – x"
    assert decode_caption(b"\x81\x8d") == "\x81\x8d"  # undefined in cp1252: latin-1


def test_add_paths_dedupes_case_insensitively(tmp_path):
    a = tmp_path / "A.png"
    a.write_bytes(b"")
    m = ImageModel()
    assert m.add_paths([str(a)]) == 1
    if os.name == "nt":
        assert m.add_paths([str(a).lower(), str(a).upper()]) == 0
    assert m.add_paths([str(a)]) == 0
    assert len(m.items) == 1


def test_caption_clashes():
    paths = [r"C:\d\photo.png", r"C:\d\photo.jpg", r"C:\d\other.png", r"C:\e\photo.webp", r"C:\d\PHOTO.webp"]
    groups = caption_clashes(paths)
    if os.name == "nt":  # case-insensitive: PHOTO.webp writes the same photo.txt
        assert groups == [[r"C:\d\photo.png", r"C:\d\photo.jpg", r"C:\d\PHOTO.webp"]]
    else:
        assert groups == [[r"C:\d\photo.png", r"C:\d\photo.jpg"]]


def test_server_stop_cancels_pending_start(monkeypatch):
    sm = backend.ServerManager()
    sm.stop()
    gen = sm._gen
    assert sm._cancelled(gen)  # stop() leaves _stop set
    with sm._lock:
        sm._stop = False
        sm._gen += 1
        gen = sm._gen
    assert not sm._cancelled(gen)
    sm.stop()
    assert sm._cancelled(gen)
