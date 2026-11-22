"""Audit 2026-11-20, remote side: stop is never retried and can name its job (G1), a kill takes the probe child too
(G2), ssh calls and fleet looks have deadlines (G4), the probe process connect has one (G5), downloads time out and
keep a good earlier file (G6), atomic writes (G7, G9), one duplicate check inside the queue lock (G8), and a staged
dataset send keeps the old folder until the new one is in place (G10). Fakes only: no ssh, vast.ai or GPU."""
import json
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "vast"))

import rulz_agent as agent  # noqa: E402
import rulz_fleet  # noqa: E402
import rulz_remote as remote  # noqa: E402

from rulztrainer import atomic, checkrun, downloads, race, regions, trainmaps  # noqa: E402
from rulztrainer import probeworker as pw  # noqa: E402
from rulztrainer.config import TrainConfig  # noqa: E402
from rulztrainer.train_queue import DuplicateJob, TrainQueue  # noqa: E402

import test_vast_fleet as fleet_tests  # noqa: E402  (its fake machines and fleet)


# ---------------------------------------------------------------- the agent on a fake install


@pytest.fixture
def install(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(agent, "webui_running", lambda: True)
    (tmp_path / "output").mkdir()
    (tmp_path / "datasets" / "set").mkdir(parents=True)
    return tmp_path


def make_job(name):
    d = Path("output") / name
    d.mkdir(parents=True)
    TrainConfig(job_name=name, dataset_dir="datasets/set").save(d / "config.yaml")
    (d / "status.json").write_text(json.dumps({"step": 3, "total": 10, "done": False, "error": None}), encoding="utf-8")
    return d


def record_job(d, proc):
    Path(".webui_job.json").write_text(json.dumps({
        "pid": proc.pid, "create_time": psutil.Process(proc.pid).create_time(), "job_dir": str(d.resolve())}))


def run(capsys, *argv):
    code = agent.main(list(argv))
    out = capsys.readouterr()
    return code, out.out, out.err


# ---------------------------------------------------------------- G1: stop and kill are not retried


def test_the_command_line_stop_is_never_retried(monkeypatch, capsys):
    seen = {}

    class Fake:
        def __init__(self, *a, **k):
            pass

        def agent(self, *args, config=None, retry=True, timeout="auto"):
            seen[args[0]] = retry
            return subprocess.CompletedProcess([], 0, b"{}", b"")

    monkeypatch.setattr(remote, "Remote", Fake)
    for words in (["stop"], ["stop", "--kill"], ["start", "--job", "x"], ["status"]):
        remote.main(words)
    assert seen["stop"] is False and seen["start"] is False and seen["status"] is True


def test_the_fleets_stop_names_its_job_and_is_not_retried(tmp_path):
    world = fleet_tests.World(tmp_path, 1, 2)
    world.machines["2"].run("b")
    world.tick()
    fake = world.fleet.remote_for(world.fleet.instance(2))
    world.fleet.stop_run("b")
    assert world.machines["2"].calls[-1] == ("stop", "--job", "b") and fake.retry is False
    world.fleet.stop_run("2", kill=True)  # an instance, not a job: nothing to name
    assert world.machines["2"].calls[-1] == ("stop", "--kill")


def test_stop_for_another_job_touches_nothing(install, capsys):
    d = make_job("first")
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        record_job(d, proc)
        code, out, err = run(capsys, "stop", "--job", "second")
        assert code == 1 and "second" in err and not (d / "STOP").exists()
        code, out, err = run(capsys, "stop", "--kill", "--job", "second")
        assert code == 1 and proc.poll() is None
        assert run(capsys, "stop", "--job", "first")[0] == 0 and (d / "STOP").exists()
    finally:
        proc.kill()
        proc.wait()


# ---------------------------------------------------------------- G2: a kill takes the probe child along

CHILD_PARENT = ("import subprocess, sys, time\n"
                "c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
                "print(c.pid, flush=True)\n"
                "time.sleep(60)\n")


def test_kill_ends_the_probe_child_too(install, capsys):
    d = make_job("live")
    parent = subprocess.Popen([sys.executable, "-c", CHILD_PARENT], stdout=subprocess.PIPE, text=True)
    child = psutil.Process(int(parent.stdout.readline()))
    try:
        record_job(d, parent)
        assert run(capsys, "stop", "--kill")[0] == 0
        parent.wait(10)
        deadline = time.time() + 10
        while child.is_running() and time.time() < deadline:
            time.sleep(0.1)
        assert not child.is_running()
    finally:
        try:
            child.kill()
        except psutil.Error:
            pass
        parent.kill()
        parent.wait()


# ---------------------------------------------------------------- G4: deadlines on ssh calls and fleet looks


def new_remote():
    return remote.Remote(1, None, "/w", conn={"id": 1, "host": "h", "port": 22, "route": "direct"}, remember=False)


def test_a_hung_ssh_call_ends_with_a_failed_result(monkeypatch):
    seen = {}

    def hung(cmd, **kw):
        seen.update(kw)
        raise subprocess.TimeoutExpired(cmd, kw["timeout"])

    monkeypatch.setattr(remote.subprocess, "run", hung)
    done = new_remote().agent("snapshot")
    assert seen["timeout"] == remote.SHORT_CALL_TIMEOUT
    assert done.returncode == remote.TIMED_OUT and b"no answer" in done.stderr
    done = new_remote().run("true", timeout=0.5)
    assert seen["timeout"] == 0.5 and done.returncode == remote.TIMED_OUT


def test_start_and_update_get_no_timeout(monkeypatch):
    seen = []
    monkeypatch.setattr(remote.subprocess, "run", lambda cmd, **kw: seen.append(kw["timeout"]) or
                        subprocess.CompletedProcess(cmd, 0, b"", b""))
    r = new_remote()
    r.agent("start", "--job", "x")
    r.agent("update")
    assert seen == [None, None]


def test_one_hung_instance_does_not_stall_the_fleets_look(tmp_path, monkeypatch):
    world = fleet_tests.World(tmp_path, 1, 2)
    release = threading.Event()
    real = fleet_tests.FakeRemote.agent

    def agent_call(self, *args, **kw):
        if self.machine.id == 2:
            release.wait(30)
        return real(self, *args, **kw)

    monkeypatch.setattr(fleet_tests.FakeRemote, "agent", agent_call)
    monkeypatch.setattr(rulz_fleet, "POLL_DEADLINE", 0.5)
    try:
        started = time.monotonic()
        snaps = world.fleet.poll_all([world.fleet.instance(1), world.fleet.instance(2)])
        assert time.monotonic() - started < 10
        assert snaps["1"]["ok"] is True and snaps["2"]["ok"] is False and "no answer" in snaps["2"]["error"]
    finally:
        release.set()


# ---------------------------------------------------------------- G5: the probe process must connect in time


def test_a_hung_probe_process_is_killed_after_the_connect_deadline(tmp_path, monkeypatch):
    real_popen = subprocess.Popen
    procs = []

    def hung_child(cmd, **kw):
        proc = real_popen([sys.executable, "-c", "import sys, time; sys.stdin.read(); time.sleep(60)"], **kw)
        procs.append(proc)
        return proc

    monkeypatch.setattr(pw, "CONNECT_TIMEOUT", 1.0)
    monkeypatch.setattr(pw.subprocess, "Popen", hung_child)
    started = time.monotonic()
    try:
        with pytest.raises(pw.ProbeProcessError, match="did not connect within"):
            pw.ProbeProcess({}, None, "cpu", tmp_path / "probe.log")
        assert time.monotonic() - started < 20
        procs[0].wait(10)  # killed, not left hanging
    finally:
        for p in procs:
            p.kill()
            p.wait()


# ---------------------------------------------------------------- G6: downloads


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        mode = self.path.strip("/")
        body = b"x" * 100_000
        if mode == "ok":
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif mode == "short":
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body[:1000])
        elif mode == "stall":
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body[:1000])
            self.wfile.flush()
            time.sleep(3)


@pytest.fixture
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_a_complete_download_replaces_the_file_and_reports_progress(server, tmp_path):
    dest = tmp_path / "m.bin"
    seen = []
    downloads.download_url(f"{server}/ok", dest, "m", lambda text, frac: seen.append(frac))
    assert dest.stat().st_size == 100_000 and not (tmp_path / "m.bin.part").exists()
    assert seen[0] == 0.0 and seen[-1] == 1.0


def test_a_short_body_keeps_the_good_file_and_leaves_no_part_file(server, tmp_path):
    dest = tmp_path / "m.bin"
    dest.write_bytes(b"good")
    with pytest.raises(Exception):
        downloads.download_url(f"{server}/short", dest, "m")
    assert dest.read_bytes() == b"good" and not (tmp_path / "m.bin.part").exists()


def test_a_server_gone_quiet_times_out(server, tmp_path, monkeypatch):
    monkeypatch.setattr(downloads, "READ_TIMEOUT", 0.5)
    dest = tmp_path / "m.bin"
    dest.write_bytes(b"good")
    started = time.monotonic()
    with pytest.raises(Exception):
        downloads.download_url(f"{server}/stall", dest, "m")
    assert time.monotonic() - started < 2.5
    assert dest.read_bytes() == b"good" and not (tmp_path / "m.bin.part").exists()


# ---------------------------------------------------------------- G7, G9: atomic writes


@pytest.fixture
def spy(monkeypatch):
    targets = []
    real = atomic.writing

    def writing(path):
        targets.append(Path(path))
        return real(path)

    monkeypatch.setattr(atomic, "writing", writing)
    return targets


def test_bodies_json_is_written_atomically(tmp_path, spy):
    items = [SimpleNamespace(image=tmp_path / "a.png")]
    (tmp_path / "a.png").write_bytes(b"x")
    out = regions.analyze_bodies(items, lambda p: [0.5, 0.5], tmp_path / "bodies.json")
    assert tmp_path / "bodies.json" in spy and json.loads((tmp_path / "bodies.json").read_text())
    assert list(out.values()) == [[0.5, 0.5]]


def test_the_race_report_is_written_atomically(tmp_path, spy):
    race.write_report(tmp_path / "race.json", race.RaceSchedule(["a", "b"], warmup=1), [])
    assert tmp_path / "race.json" in spy and "configs" in json.loads((tmp_path / "race.json").read_text())


def test_the_check_run_manifest_is_written_atomically(tmp_path, spy):
    path = checkrun.save_manifest({"tag": "t1"}, root=tmp_path)
    assert path in spy and json.loads(path.read_text())["tag"] == "t1"


def test_the_training_maps_probes_json_is_written_atomically(tmp_path, spy):
    maps = trainmaps.TrainingMaps(tmp_path, [], every=10)
    maps.start(0, False)
    target = maps.dir / "probes.json"
    assert target in spy and json.loads(target.read_text())["every"] == 10


# ---------------------------------------------------------------- G8: one duplicate check, inside the queue lock


def test_the_queue_refuses_a_duplicate_name_inside_its_own_lock(tmp_path):
    q = TrainQueue(tmp_path / "_queue.json")
    q.add({"job_name": "Run1"}, refuse_duplicate=True)
    with pytest.raises(DuplicateJob):
        q.add({"job_name": "Run1"}, refuse_duplicate=True)
    assert len(q.snapshot()["waiting"]) == 1
    q.add({"job_name": "Run1"})  # the WebUI's callers do not ask for the check
    assert len(q.snapshot()["waiting"]) == 2


def test_two_overlapping_starts_queue_the_job_once(install, capsys, monkeypatch):
    """Both starts pass cmd_start's own look at the queue (it is empty for both); only one may get in."""
    make_job("old")
    q = TrainQueue(agent.QUEUE_FILE)
    real_snapshot = TrainQueue.snapshot
    gate = threading.Barrier(2, timeout=10)

    def snapshot(self):
        snap = real_snapshot(self)
        if threading.current_thread() is not threading.main_thread():
            try:
                gate.wait()  # both threads have looked before either adds
            except threading.BrokenBarrierError:
                pass
        return snap

    monkeypatch.setattr(TrainQueue, "snapshot", snapshot)
    monkeypatch.setattr(agent, "report_start", lambda q, entry, plan, busy: 0)
    results = []

    def start():
        try:
            results.append(agent.main(["start", "--job", "new", "--from", "old"]))
        except agent.Refused:
            results.append("refused")

    threads = [threading.Thread(target=start) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert sorted(map(str, results)) == ["0", "refused"] or sorted(map(str, results)) == ["0", "1"]
    assert [e["job"] for e in real_snapshot(q)["waiting"]] == ["new"]


# ---------------------------------------------------------------- G10: the staged send keeps the old folder aside


def test_a_staged_send_moves_the_old_copy_aside_instead_of_deleting_it_first(monkeypatch, tmp_path):
    shells = []

    class Proc:
        stdin = open(tmp_path / "sink", "wb")

        def wait(self, *a, **k):
            return 0

        def poll(self):
            return 0

        returncode = 0

    class FakeRemote:
        def command(self, shell, in_folder=True):
            shells.append(shell)
            return ["x"]

    monkeypatch.setattr(remote.subprocess, "Popen", lambda cmd, **kw: Proc())
    (tmp_path / "ds").mkdir()
    (tmp_path / "ds" / "a.txt").write_text("a")
    try:
        remote.send(FakeRemote(), [(tmp_path / "ds", "ds")], "datasets", staged=True)
    except Exception:
        pass  # the fake process gives no real exit status: only the built shell matters here
    shell = shells[0]
    delete_first = "rm -rf datasets/ds &&"
    assert delete_first not in shell and ".old-0" in shell
    # the old copy is only deleted after the new one is in place
    assert shell.index("mv datasets/.rulz_incoming") < shell.rindex("rm -rf datasets/.rulz_incoming")
    assert shell.index("mv datasets/ds") < shell.index("mv datasets/.rulz_incoming")


@pytest.mark.parametrize("arrives", [True, False])
def test_the_staged_send_shell_keeps_the_old_copy_when_the_new_one_cannot_go_in(monkeypatch, tmp_path, arrives):
    """The built shell run for real (bash, tar): the new copy replaces the old one, and when the new one is not there
    to move in (here an empty archive) the old copy is put back instead of going with the stage folder."""
    import io
    import shutil
    import tarfile

    bash = shutil.which("bash")
    if bash is None or shutil.which("tar") is None:
        pytest.skip("needs bash and tar")
    shells = []

    class FakeRemote:
        def command(self, shell, in_folder=True):
            shells.append(shell)
            return ["x"]

    class Proc:
        stdin = open(tmp_path / "sink", "wb")
        returncode = 0

        def wait(self, *a, **k):
            return 0

        def poll(self):
            return 0

    monkeypatch.setattr(remote.subprocess, "Popen", lambda cmd, **kw: Proc())
    (tmp_path / "ds").mkdir()
    (tmp_path / "ds" / "new.txt").write_text("new")
    try:
        remote.send(FakeRemote(), [(tmp_path / "ds", "ds")], "datasets", staged=True)
    except Exception:
        pass
    Proc.stdin.close()
    monkeypatch.undo()  # (the fake Popen is subprocess's own: the real one runs the shell below)
    box = tmp_path / "instance"
    (box / "datasets" / "ds").mkdir(parents=True)
    (box / "datasets" / "ds" / "old.txt").write_text("old")
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as tar:
        if arrives:
            tar.add(tmp_path / "ds", arcname="ds")
    done = subprocess.run([bash, "-c", shells[0]], cwd=box, input=archive.getvalue(), capture_output=True)
    names = sorted(p.name for p in (box / "datasets" / "ds").iterdir())
    assert names == (["new.txt"] if arrives else ["old.txt"])
    assert (done.returncode == 0) == arrives
    assert not list((box / "datasets" / ".rulz_incoming").glob("*"))  # the stage folder is gone either way
