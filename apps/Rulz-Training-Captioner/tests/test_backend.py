"""Run with:  python -m pytest tests   (from the app folder, inside its .venv)."""
import base64
import os
import sys
import threading

import pytest
import requests

os.environ.setdefault("QT_QPA_PLATFORM", "minimal")  # no window needed; "offscreen" can hang on Windows
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtGui import QColor, QGuiApplication, QImage  # noqa: E402

from captioner import backend  # noqa: E402
from captioner.model import ImageModel, decode_caption  # noqa: E402

APP = QGuiApplication.instance() or QGuiApplication([])

EP = {"url": "http://127.0.0.1:1/v1", "model": "default", "slots": 1, "name": "test"}
S = {"api_key": "", "max_tokens": 10, "thinking_budget": 0, "disable_thinking": True, "temperature": 0.1}


class FakeResponse:
    def __init__(self, status, text="", headers=None):
        self.status_code, self.text, self.headers = status, text, headers or {}

    def json(self):
        return {"choices": [{"message": {"content": "a cat"}, "finish_reason": "stop"}]}


class FakeSession:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), 0

    def post(self, *a, **k):
        self.calls += 1
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
    assert out == "a cat" and sess.calls == 3 and waits == [1, 3]


def test_no_retry_on_400():
    sess = FakeSession([FakeResponse(400, "bad request"), FakeResponse(200)])
    with pytest.raises(backend.HTTPStatusError):
        backend.request_with_retry(EP, "", "", "hi", S, sess, wait=lambda d: None)
    assert sess.calls == 1


def test_gives_up_after_max_attempts():
    sess = FakeSession([requests.ConnectionError("down")] * backend.MAX_ATTEMPTS)
    with pytest.raises(RuntimeError, match="gave up"):
        backend.request_with_retry(EP, "", "", "hi", S, sess, wait=lambda d: None)
    assert sess.calls == backend.MAX_ATTEMPTS


def test_cancel_during_backoff_stops():
    sess = FakeSession([FakeResponse(502), FakeResponse(200)])
    ev = threading.Event()
    ev.set()
    with pytest.raises(RuntimeError, match="stopped"):
        backend.request_with_retry(EP, "", "", "hi", S, sess, wait=ev.wait)
    assert sess.calls == 1


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
