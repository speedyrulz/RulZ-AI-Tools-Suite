import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer.train_queue import TrainQueue  # noqa: E402


def _starter(log, ok=True):
    def start(config):
        log.append(config["job_name"])
        return ok, "Started" if ok else "Dataset folder not found"
    return start


def test_runs_start_in_order_once_the_current_one_ends(tmp_path):
    q = TrainQueue(tmp_path / "q.json")
    started = []
    q.add({"job_name": "a"})
    q.add({"job_name": "b"})
    assert q.advance(True, _starter(started)) is None and started == []  # the current run goes on
    assert q.advance(False, _starter(started))["job"] == "a"
    assert q.advance(True, _starter(started)) is None  # a runs
    q.advance(False, _starter(started))
    assert started == ["a", "b"] and q.snapshot()["waiting"] == []
    assert [r["job"] for r in q.snapshot()["recent"]] == ["b", "a"]


def test_a_restarted_webui_pauses_the_queue_and_kill_style_pause_holds(tmp_path):
    path = tmp_path / "q.json"
    TrainQueue(path).add({"job_name": "a"})
    q = TrainQueue(path)  # a restarted WebUI, nothing running: the queue pauses instead of starting a run
    started = []
    assert q.on_startup(running=False) and q.snapshot()["paused"]
    assert q.advance(False, _starter(started)) is None and started == []
    q.set_paused(False)  # Resume
    q.advance(False, _starter(started))
    assert started == ["a"]
    q.add({"job_name": "b"})
    q.set_paused(True)  # what Kill does
    assert q.advance(False, _starter(started)) is None and started == ["a"]


def test_a_run_that_cannot_start_stays_first_and_pauses_the_queue(tmp_path):
    q = TrainQueue(tmp_path / "q.json")
    q.add({"job_name": "a"})
    q.add({"job_name": "b"})
    done = q.advance(False, _starter([], ok=False))
    snap = q.snapshot()
    assert done["error"] == "Dataset folder not found" and snap["paused"]
    assert [e["job"] for e in snap["waiting"]] == ["a", "b"]


def test_move_and_remove(tmp_path):
    q = TrainQueue(tmp_path / "q.json")
    ids = [q.add({"job_name": n})["id"] for n in "abc"]
    assert q.move(ids[2], -1) and [e["job"] for e in q.snapshot()["waiting"]] == ["a", "c", "b"]
    assert not q.move(ids[0], -1)  # already first
    assert q.remove(ids[1]) and [e["job"] for e in q.snapshot()["waiting"]] == ["a", "c"]
    q.clear()
    assert q.snapshot()["waiting"] == []


def test_a_webui_restarted_while_a_job_runs_keeps_the_queue_going(tmp_path):
    q = TrainQueue(tmp_path / "q.json")
    q.add({"job_name": "a"})
    assert not q.on_startup(running=True) and not q.snapshot()["paused"]
