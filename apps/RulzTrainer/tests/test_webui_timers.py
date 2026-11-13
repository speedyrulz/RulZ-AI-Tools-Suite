"""A page's repeating timers rest while no browser is connected: a page outlives its browser for PAGE_KEEP_SECONDS."""
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import webui  # noqa: E402


def wrapped(monkeypatch, callback):
    client = SimpleNamespace(has_socket_connection=True)
    monkeypatch.setattr(webui.ui, "context", SimpleNamespace(client=client))
    return client, webui.while_connected(callback)


def test_timer_rests_without_a_browser(monkeypatch):
    calls = []
    client, tick = wrapped(monkeypatch, lambda: calls.append(1))
    tick()
    client.has_socket_connection = False
    tick()
    assert len(calls) == 1
    client.has_socket_connection = True  # the browser came back
    tick()
    assert len(calls) == 2


def test_async_callback_is_still_awaitable(monkeypatch):
    calls = []

    async def refresh():
        calls.append(1)

    client, tick = wrapped(monkeypatch, refresh)
    asyncio.run(tick())
    assert calls == [1]
    client.has_socket_connection = False
    assert tick() is None


def test_the_log_is_read_from_where_the_last_poll_stopped(tmp_path, monkeypatch):
    import webui

    log = tmp_path / "train.log"
    log.write_bytes(b"one\ntwo\npart")
    state = {"pos": 0, "rest": "", "decoder": None, "head": b""}
    resets = []
    assert webui.log_news(log, state, lambda: resets.append(1)) == ["one", "two"]  # "part" is still being written
    with open(log, "ab") as f:
        f.write(b"ial\r\n\xc3")  # a line ends, a two-byte character starts
    assert webui.log_news(log, state) == ["partial"]
    with open(log, "ab") as f:
        f.write(b"\xa9\n")
    assert webui.log_news(log, state) == ["\u00e9"] and webui.log_news(log, state) == []
    log.write_bytes(b"new run\n")  # written again: shorter than where we were
    assert webui.log_news(log, state, lambda: resets.append(1)) == ["new run"] and resets == [1]
    log.write_bytes(b"10:01 the next run, already longer than the last one was when it was read\n")
    assert webui.log_news(log, state, lambda: resets.append(1)) == [
        "10:01 the next run, already longer than the last one was when it was read"] and resets == [1, 1]


def test_a_long_log_opens_on_its_end(tmp_path, monkeypatch):
    import webui

    monkeypatch.setattr(webui, "LOG_FIRST_BYTES", 100)
    log = tmp_path / "train.log"
    log.write_text("".join(f"line {i}\n" for i in range(1000)), encoding="utf-8")
    lines = webui.log_news(log, {"pos": 0, "rest": "", "decoder": None, "head": b""})
    assert lines[-1] == "line 999" and 5 < len(lines) < 20 and all(line.startswith("line ") for line in lines)


def test_gpu_stats_for_timers_never_wait_for_nvidia_smi(monkeypatch):
    import threading

    import webui

    release = threading.Event()
    calls = []

    def slow():
        calls.append(1)
        release.wait(5)
        return [{"index": 0}]

    monkeypatch.setattr(webui, "gpu_stats", slow)
    monkeypatch.setattr(webui, "_gpu_cache", {"at": 0.0, "stats": [], "busy": False})
    assert webui.gpu_stats_cached() == [] and webui.gpu_stats_cached() == []  # one sample at a time
    release.set()
    for _ in range(100):
        if not webui._gpu_cache["busy"]:
            break
        __import__("time").sleep(0.02)
    assert webui.gpu_stats_cached() == [{"index": 0}] and calls == [1]
