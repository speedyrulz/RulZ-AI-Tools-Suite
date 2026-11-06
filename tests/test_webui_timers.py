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
