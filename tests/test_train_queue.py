import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

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


ADD_FROM_ANOTHER_PROCESS = """
import sys
sys.path.insert(0, sys.argv[1])
from rulztrainer.train_queue import TrainQueue
TrainQueue(sys.argv[2]).add({"job_name": "added"})
"""


def test_a_run_added_by_another_process_while_one_starts_is_not_lost(tmp_path):
    """advance holds the queue while a run starts: rulz_agent.py adds from an ssh session, a process of its own."""
    path = tmp_path / "q.json"
    q = TrainQueue(path)
    q.add({"job_name": "first"})
    other = []

    def start(config):
        other.append(subprocess.Popen([sys.executable, "-c", ADD_FROM_ANOTHER_PROCESS, str(ROOT), str(path)]))
        time.sleep(1.0)  # that process gets in its read and write now, unless the lock holds it back
        return True, "Started"

    q.advance(False, start)
    assert other[0].wait(30) == 0
    assert [e["job"] for e in TrainQueue(path).snapshot()["waiting"]] == ["added"]


def test_a_run_started_under_the_lock_does_not_keep_it(tmp_path):
    """A training process lives for hours: if it inherited the lock, nothing could edit the queue until it ended."""
    path = tmp_path / "q.json"
    q = TrainQueue(path)
    q.add({"job_name": "first"})
    job = []

    def start(config):
        job.append(subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"]))
        return True, "Started"

    q.advance(False, start)
    try:
        add = subprocess.run([sys.executable, "-c", ADD_FROM_ANOTHER_PROCESS, str(ROOT), str(path)], timeout=15)
        assert add.returncode == 0
    finally:
        job[0].kill()
        job[0].wait()


def test_the_lock_nests_within_a_thread_and_is_free_afterwards(tmp_path):
    q = TrainQueue(tmp_path / "q.json")
    with q.lock:
        with q.lock:
            q.add({"job_name": "a"})
        assert q.lock.file is not None  # still held by the outer one
    assert q.lock.file is None and q.lock.depth == 0


def test_threads_of_one_process_still_take_turns(tmp_path):
    q = TrainQueue(tmp_path / "q.json")

    def adds(n):
        for i in range(10):
            q.add({"job_name": f"{n}-{i}"})

    threads = [threading.Thread(target=adds, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(q.snapshot()["waiting"]) == 40
