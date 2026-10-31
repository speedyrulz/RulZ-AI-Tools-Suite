from PIL import Image

from rulz.ui import thumbs


def _loader(qapp, monkeypatch):
    loader = thumbs.ThumbLoader()
    started = []
    monkeypatch.setattr(loader.pool, "start", lambda job: started.append(job))
    return loader, started


def test_newest_request_first(qapp, monkeypatch):
    loader, started = _loader(qapp, monkeypatch)
    for key in ("a", "b", "c"):
        loader.get(key)
    loader.get("a")  # scrolled back to it
    assert len(started) == 3  # one job per key, not per request
    order = [loader._take()[0] for _ in range(3)]
    assert order == ["a", "c", "b"]
    assert loader._take() is None


def test_scrolled_past_requests_are_dropped(qapp, monkeypatch):
    loader, _ = _loader(qapp, monkeypatch)
    loader.get("old")
    for i in range(thumbs.STALE_AFTER + 5):
        loader.get(f"new{i}")
    stale = {}
    while (item := loader._take()) is not None:
        stale[item[0]] = item[4]
    assert stale["old"] is True
    assert stale[f"new{thumbs.STALE_AFTER + 4}"] is False


def test_in_flight_key_is_not_queued_twice(qapp, monkeypatch):
    loader, started = _loader(qapp, monkeypatch)
    loader.get("a")
    assert loader._take()[0] == "a"  # a pool thread is reading it
    loader.get("a")
    assert loader._take() is None
    assert len(started) == 1


def test_cache_bounded_by_bytes(qapp, monkeypatch, tmp_path):
    loader, _ = _loader(qapp, monkeypatch)
    loader.budget = 3 * 100 * 100 * 4
    for i in range(6):
        p = tmp_path / f"{i}.png"
        Image.new("RGB", (100, 100)).save(p)
        loader.pending.add(str(p))
        loader._on_done(str(p), thumbs._read(str(p), None, 480), False)
    assert loader.cache_bytes <= loader.budget
    assert len(loader.cache) == 3
    assert str(tmp_path / "5.png") in loader.cache


def test_dropped_job_can_be_requested_again(qapp, monkeypatch):
    loader, started = _loader(qapp, monkeypatch)
    loader.get("x")
    loader._take()
    loader._on_done("x", thumbs.QImage(), True)
    assert "x" not in loader.failed and "x" not in loader.pending
    loader.get("x")
    assert len(started) == 2
