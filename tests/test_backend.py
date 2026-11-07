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

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage, QImageReader

from captioner import backend, model as model_mod, winjob
from captioner.model import (ImageModel, caption_clashes, collect_images, decode_caption, load_entries,
                             natural_key, sort_key)

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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
        self.timeout = k.get("timeout")
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
    assert backend.is_retryable(requests.ConnectTimeout("no answer to connect"))
    assert not backend.is_retryable(RuntimeError("Ran out of tokens"))


def test_full_read_timeout_is_not_retried():
    # three more 15-minute waits would hold one image for an hour
    assert not backend.is_retryable(requests.ReadTimeout("read timed out"))
    sess = FakeSession([requests.ReadTimeout("read timed out"), FakeResponse(200)])
    with pytest.raises(requests.ReadTimeout):
        backend.request_with_retry(EP, "", "", "hi", S, sess, wait=lambda d: None)
    assert sess.calls == 1


def test_request_timeout_is_short_for_connecting():
    sess = FakeSession([FakeResponse(200)])
    backend.request_caption(EP, "", "", "hi", S, sess)
    assert sess.timeout == (10, 900)


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
        "job = winjob.adopt(p.pid)\n"
        "print(p.pid, job is not None, flush=True)\n"
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


@pytest.mark.skipif(os.name != "nt", reason="Windows job objects")
def test_stop_ends_a_server_and_its_children_at_once(tmp_path):
    # a .cmd wrapper around the real server, like a user-made llama-server.cmd: its child must go too,
    # and the port it listens on must be free when the stop returns, or a restart can't use it
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    pidfile = tmp_path / "child.pid"
    code = ("import os, socket, time; s = socket.socket(); s.bind(('127.0.0.1', %d)); s.listen(); "
            "open(%r, 'w').write(str(os.getpid())); time.sleep(60)" % (port, str(pidfile)))
    d = subprocess.DEVNULL
    p = subprocess.Popen(["cmd", "/c", sys.executable, "-c", code], stdin=d, stdout=d, stderr=d,
                         creationflags=backend.CREATE_NO_WINDOW)
    p.job = winjob.adopt(p.pid)
    assert p.job is not None
    deadline = time.time() + 20
    while not (pidfile.exists() and pidfile.read_text()) and time.time() < deadline:
        time.sleep(0.05)
    child = int(pidfile.read_text())
    assert backend.port_in_use(port)
    try:
        t = time.perf_counter()
        backend.ServerManager._kill([p])
        took = time.perf_counter() - t
        assert not backend.port_in_use(port), "the port was still held when the stop returned"
        # Windows reports the job empty about a millisecond before the child's process object is
        # signalled; by then its port is already closed (checked above)
        assert p.poll() is not None and _wait_exit(child, 1000), "the server's child outlived the stop"
        assert took < 0.3, f"stop took {took:.2f}s (taskkill takes ~0.75s per server)"
    finally:
        for pid in (p.pid, child):
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
    got = {"captioned": [], "failed": [], "warning": [], "started": [], "requeued": []}
    got["events"] = []  # (signal, path) for every signal, in the order the workers sent them
    done = threading.Event()
    for name in [n for n in got if n != "events"]:
        def record(*args, name=name):
            got[name].append(args)
            got["events"].append((name, args[0]))
        getattr(eng, name).connect(record, Qt.DirectConnection)
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
    # the image the dead server held went back to queued, then showed as started again on the other one
    assert len(got["requeued"]) == 1
    back = got["requeued"][0][0]
    states = [kind for kind, p in got["events"] if p == back and kind in ("started", "requeued")]
    assert states[-1] == "started" and states.count("requeued") == 1


def test_each_image_shows_as_started_before_its_result(monkeypatch):
    eng, paths, got = run_engine(monkeypatch, [_ep("a"), _ep("b")], lambda eng, ep: ("a cat", "stop"), n=6)
    for p in paths:
        kinds = [kind for kind, q in got["events"] if q == p]
        assert kinds == ["started", "captioned"], (p, kinds)


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
def _sent_pixel(path):
    out = QImage.fromData(base64.b64decode(backend.encode_image(path, 1024)))
    return QColor(out.pixel(5, 5))


def test_encode_image_transparency_becomes_white(tmp_path):
    img = QImage(40, 40, QImage.Format_ARGB32)
    img.fill(QColor(0, 0, 0, 0))  # fully transparent: JPEG would show it black without the white backing
    p = str(tmp_path / "clear.png")
    img.save(p)
    c = _sent_pixel(p)
    assert min(c.red(), c.green(), c.blue()) > 245


def test_encode_image_opaque_colours_kept(tmp_path):
    img = QImage(40, 40, QImage.Format_RGB32)
    img.fill(QColor(200, 30, 30))
    p = str(tmp_path / "red.png")
    img.save(p)
    c = _sent_pixel(p)
    assert abs(c.red() - 200) < 8 and c.green() < 45 and c.blue() < 45


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


def test_natural_key_handles_any_digits():
    names = ["img10.png", "img2.png", "1²2.png", "img٣.png", "Img1.png"]  # ² is "digit" to isdigit() only
    assert sorted(names, key=natural_key) == ["1²2.png", "Img1.png", "img2.png", "img٣.png", "img10.png"]


def test_sort_key_keeps_folders_together():
    paths = [r"C:\d\b\1.png", r"C:\d\a\10.png", r"C:\d\a\2.png", r"C:\d\b\2.png", r"C:\d\top.png",
             r"C:\d\a2\1.png", r"C:\d\a10\1.png"]
    assert sorted(paths, key=sort_key) == [
        r"C:\d\top.png", r"C:\d\a\2.png", r"C:\d\a\10.png", r"C:\d\a2\1.png", r"C:\d\a10\1.png",
        r"C:\d\b\1.png", r"C:\d\b\2.png"]


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    return str(path)


def test_collect_images(tmp_path):
    top = _touch(tmp_path / "a.PNG")
    _touch(tmp_path / "notes.txt")
    deep = _touch(tmp_path / "sub" / "deeper" / "b.jpg")
    (tmp_path / "folder.png").mkdir()  # a folder with an image-like name is not an image
    single = _touch(tmp_path / "elsewhere" / "c.webp")
    assert sorted(collect_images([str(tmp_path)])) == [top]
    assert sorted(collect_images([str(tmp_path)], recursive=True)) == sorted([top, deep, single])
    assert collect_images([single, str(tmp_path / "missing.png")]) == [single]


def test_collect_images_skips_unreadable_folders(tmp_path, monkeypatch):
    locked = tmp_path / "locked"
    locked.mkdir()
    ok = _touch(tmp_path / "open" / "a.png")
    orig = os.scandir

    def scandir(d):
        if d == str(locked):
            raise PermissionError("Access is denied")
        return orig(d)
    monkeypatch.setattr(model_mod.os, "scandir", scandir)
    bad = []
    assert collect_images([str(tmp_path)], recursive=True, unreadable=bad) == [ok]  # the rest still adds
    assert bad == [str(locked)]


def test_load_entries_sorts_and_reads_captions(tmp_path):
    a = _touch(tmp_path / "x" / "2.png")
    b = _touch(tmp_path / "x" / "10.png")
    (tmp_path / "x" / "10.txt").write_text("a cat", encoding="utf-8")
    seen = []
    entries = load_entries([b, a], progress=seen.append)
    assert [e["path"] for e in entries] == [a, b]
    assert [(e["caption"], e["has_file"]) for e in entries] == [("", False), ("a cat", True)]


def test_load_entries_uses_the_folder_listing(tmp_path, monkeypatch):
    a = _touch(tmp_path / "x" / "a.png")
    b = _touch(tmp_path / "x" / "b.png")
    (tmp_path / "x" / "B.TXT").write_text("a dog", encoding="utf-8")  # Windows names ignore case
    listing = {}
    found = collect_images([str(tmp_path).replace("\\", "/")], recursive=True, listing=listing)  # dialog-style path
    opened = []
    real = model_mod.read_caption
    monkeypatch.setattr(model_mod, "read_caption", lambda p: opened.append(p) or real(p))
    entries = load_entries([os.path.normpath(p) for p in found], listing=listing)
    assert [(e["path"], e["caption"], e["has_file"]) for e in entries] == [(a, "", False), (b, "a dog", True)]
    assert opened == [b]  # a.txt isn't in the folder, so it was never opened


def test_add_entries_skips_images_listed_meanwhile(tmp_path):
    m = ImageModel()
    a = _touch(tmp_path / "a.png")
    entries = load_entries([a])
    assert m.add_paths([a]) == 1
    assert m.add_entries(entries) == 0  # a second scan of the same folder adds nothing
    assert len(m.items) == 1


def test_remove_rows_in_runs(tmp_path):
    m = ImageModel()
    m.add_entries([{"path": f"C:\\d\\{i:05}.png", "caption": "", "has_file": False, "status": "idle"}
                   for i in range(20)])
    removed = []
    m.rowsAboutToBeRemoved.connect(lambda parent, lo, hi: removed.append((lo, hi)))
    m.remove_rows([3, 4, 5, 9, 15, 16, 19, 0])
    assert removed == [(19, 19), (15, 16), (9, 9), (3, 5), (0, 0)]
    left = [int(os.path.basename(it["path"])[:5]) for it in m.items]
    assert left == [1, 2, 6, 7, 8, 10, 11, 12, 13, 14, 17, 18]
    assert all(m.index_of[it["path"]] == i for i, it in enumerate(m.items))


def test_remove_many_rows_is_fast():
    m = ImageModel()
    m.add_entries([{"path": f"C:\\d\\{i:06}.png", "caption": "", "has_file": False, "status": "idle"}
                   for i in range(20000)])
    t = time.perf_counter()
    m.remove_rows(range(0, 20000, 2))  # every other row: the worst case, 10,000 runs of one
    took = time.perf_counter() - t
    assert len(m.items) == 10000 and took < 0.5, f"took {took:.2f}s (1.07s before)"


def test_thumbnail_cache_is_capped_by_size(monkeypatch):
    m = ImageModel()
    paths = [f"C:\\d\\{i}.png" for i in range(10)]
    m.add_entries([{"path": p, "caption": "", "has_file": False, "status": "idle"} for p in paths])
    img = QImage(100, 100, QImage.Format_RGB32)  # 40,000 bytes each
    img.fill(QColor("gray"))
    monkeypatch.setattr(model_mod, "CACHE_BYTES", 3 * 40000)
    for p in paths:
        m._thumb_ready(p, img)
    assert list(m.cache) == paths[-3:] and m.cache_bytes == 3 * 40000
    m.data(m.index(7), Qt.DecorationRole)  # drawing one makes it the most recently used
    m._thumb_ready(paths[0], img)  # so 8, not 7, makes room for this one
    assert list(m.cache) == [paths[9], paths[7], paths[0]]
    m.remove_rows([9])
    assert list(m.cache) == [paths[7], paths[0]] and m.cache_bytes == 2 * 40000
    m._thumb_ready(paths[9], img)  # finished decoding after it was removed: not kept
    assert paths[9] not in m.cache


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
