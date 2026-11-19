import queue
import threading
import time

import pytest
import torch

from rulztrainer import probeworker as pw


class Connection:
    def __init__(self):
        self.incoming = queue.Queue()
        self.outgoing = queue.Queue()

    def recv(self):
        return self.incoming.get()

    def send(self, msg):
        self.outgoing.put(msg)


@pytest.mark.parametrize("action", ["probe", "failure", "stop"])
def test_sweep_interruption(monkeypatch, action):
    conn = Connection()
    started, resume = threading.Event(), threading.Event()
    calls = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("cache"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def picture(r, pic):
        calls.append(pic)
        if pic == "a":
            started.set()
            assert resume.wait(5)
        return {pic: pic}

    monkeypatch.setattr(pw, "_sweep_picture", picture)

    class Renderer:
        sweep_pictures = {"a": "a", "b": "b"}

        def _sweep_state(self, payload):
            calls.append("mirror")

        def render(self, jobs, payload, kind):
            calls.append("render")
            if action == "failure":
                raise RuntimeError("render failed")
            return ["image"], 1.0

    server = threading.Thread(target=pw._serve, args=(conn, Renderer()), daemon=True)
    server.start()
    conn.incoming.put(("sweep", 1, b"weights", ["a", "b"]))
    assert started.wait(5)
    conn.incoming.put(("stop",) if action == "stop" else ("probe", 2, [], b"weights", "probe"))
    # Ensure the receiver has queued the interruption before releasing the picture.
    deadline = time.monotonic() + 5
    while not conn.incoming.empty() and time.monotonic() < deadline:
        time.sleep(0.001)
    resume.set()
    if action == "probe":
        assert conn.outgoing.get(timeout=5) == ("reply", 2, ("done", ["image"], 1.0))
        reply = conn.outgoing.get(timeout=5)
        assert reply[0:2] == ("reply", 1) and reply[2][0:2] == ("sweep_done", {"a": "a", "b": "b"})
        assert calls == ["mirror", "a", "cache", "render", "cache", "mirror", "b", "cache"]
        conn.incoming.put(("stop",))
    elif action == "failure":
        assert "render failed" in conn.outgoing.get(timeout=5)[1]
        assert "b" not in calls
    else:
        assert conn.outgoing.empty() and "b" not in calls
    server.join(5)
    assert not server.is_alive()


def test_picture_by_picture_matches_full_sweep():
    from test_second_gpu_jobs import Toy, sweep_probes
    from rulztrainer.regionsweep import forward_readings

    model = Toy()
    pictures = sweep_probes(3, 1)
    expected = forward_readings(model, "cpu", pictures)
    renderer = type("Renderer", (), {"model": model, "device": "cpu"})()
    actual = {}
    for picture in pictures:
        actual.update(pw._sweep_picture(renderer, picture))
    assert actual == expected
