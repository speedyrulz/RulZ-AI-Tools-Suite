"""LoRA test remote commands, without a GPU or network."""
import io
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP / "scripts" / "vast"))
import rulz_agent as agent
import rulz_remote as remote

ID = "t20261122-153012-a3f9"


@pytest.fixture
def install(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(agent, "job_process", lambda: None)
    monkeypatch.setattr(agent, "SENT_CONFIG", "")
    return tmp_path


def write_test(test_id=ID, state="queued", pid=None):
    d = Path("output/_tests") / test_id
    d.mkdir(parents=True)
    (d / "status.json").write_text(json.dumps({"test_id": test_id, "state": state, "pid": pid}), encoding="utf-8")
    return d


def start(monkeypatch, spec):
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(spec)))
    return agent.main(["test-start"])


def test_start_runs_new_and_detaches(install, monkeypatch, capsys):
    calls = []
    def new(cmd, **kwargs):
        calls.append((cmd, kwargs))
        assert json.loads(Path(cmd[-1]).read_text(encoding="utf-8"))["test_id"] == ID
        write_test()
        return SimpleNamespace(returncode=0, stdout=ID + "\n", stderr="")
    monkeypatch.setattr(agent.subprocess, "run", new)
    def popen(*a, **kw):
        assert kw["stdout"].name.endswith("run.log")
        calls.append((a[0], kw))
        return SimpleNamespace(pid=os.getpid())
    monkeypatch.setattr(agent.subprocess, "Popen", popen)
    assert start(monkeypatch, {"test_id": ID, "devices": ["cuda:0"]}) == 0
    assert json.loads(capsys.readouterr().out) == {"test_id": ID, "state": "queued"}
    assert calls[0][0][1:4] == ["-m", "rulztrainer.loratest", "new"]
    assert calls[1][0] == [sys.executable, "-m", "rulztrainer.loratest", "run", str(Path("output/_tests") / ID)]
    assert calls[1][1]["stdin"] == subprocess.DEVNULL
    assert calls[1][1]["stderr"] == subprocess.STDOUT
    assert calls[1][1]["start_new_session"] is True
    status = json.loads((Path("output/_tests") / ID / "status.json").read_text(encoding="utf-8"))
    assert status["pid"] == os.getpid() and status["pid_created"]
    assert (Path("output/_tests") / ID / "run.log").is_file()


def test_refuses_job_running_test_and_bad_id(install, monkeypatch, capsys):
    monkeypatch.setattr(agent, "job_process", lambda: object())
    assert start(monkeypatch, {"test_id": ID}) == 1
    assert "GPU" in capsys.readouterr().err
    monkeypatch.setattr(agent, "job_process", lambda: None)
    write_test(state="rendering", pid=42)
    monkeypatch.setattr(agent, "live_process", lambda pid, created=None: object())
    assert start(monkeypatch, {}) == 1
    assert "already running" in capsys.readouterr().err
    assert start(monkeypatch, {"test_id": "../bad"}) == 1
    assert "Bad test_id" in capsys.readouterr().err


def test_dead_test_does_not_block(install, monkeypatch, capsys):
    write_test(state="rendering", pid=42)
    monkeypatch.setattr(agent, "live_process", lambda pid, created=None: None)
    monkeypatch.setattr(agent.subprocess, "run", lambda *a, **kw: SimpleNamespace(
        returncode=1, stdout="", stderr="new was reached"))
    assert start(monkeypatch, {}) == 1
    assert "new was reached" in capsys.readouterr().err
    assert agent.main(["test-status", ID]) == 0
    assert json.loads(capsys.readouterr().out)["state"] == "failed"


def test_status_stop_and_files(install, capsys):
    d = write_test(state="done")
    other = write_test("t20261123-000000-abcd", state="done")
    (d / "images").mkdir()
    (d / "images" / "a.png").write_bytes(b"image")
    (other / "secret").write_text("other", encoding="utf-8")
    assert agent.main(["test-status"]) == 0
    assert [s["test_id"] for s in json.loads(capsys.readouterr().out)["tests"]] == [other.name, ID]
    assert agent.main(["test-status", ID]) == 0
    assert json.loads(capsys.readouterr().out)["test_id"] == ID
    assert agent.main(["test-stop", ID]) == 0
    assert json.loads(capsys.readouterr().out)["stopping"]
    assert (d / "STOP").exists()
    assert agent.main(["files", "--tests", ID]) == 0
    listing = json.loads(capsys.readouterr().out)
    assert {x["path"] for x in listing["files"]} == {"status.json", "STOP", "images/a.png"}
    assert Path(listing["base"]) == d.resolve()


def test_remote_start_and_pull(install, monkeypatch, capsys):
    calls = []
    class FakeRemote:
        conn = {"id": 17}
        def agent(self, *args, **kwargs):
            calls.append((args, kwargs))
            return subprocess.CompletedProcess(args, 0, b"{}", b"")
        def agent_json(self, *args):
            calls.append(args)
            return {"base": "/remote/test", "files": [{"path": "images/a.png", "size": 5}]}
    spec = install / "spec.json"
    spec.write_text("{}", encoding="utf-8")
    r = FakeRemote()
    monkeypatch.setattr(remote, "forward", lambda done: done.returncode)
    assert remote.cmd_test_start(SimpleNamespace(remote=r, spec=str(spec))) == 0
    call = calls.pop()
    assert call[0] == ("test-start",) and call[1]["retry"] is False
    assert re.fullmatch(r"t\d{8}-\d{6}-[0-9a-f]{4}", json.loads(call[1]["config"])["test_id"])
    assert "Test ID:" in capsys.readouterr().out
    monkeypatch.setattr(remote, "pulls_root", lambda: install)
    monkeypatch.setattr(remote, "fleet_move_in_progress", lambda: None)
    monkeypatch.setattr(remote, "fetch", lambda *a: calls.append(a))
    args = remote.parser().parse_args(["pull", "--tests", ID])
    args.remote = r
    assert remote.cmd_pull(args) == 0
    assert calls[0] == ("files", "--tests", ID)
    assert calls[1][1:] == ("/remote/test", ["images/a.png"], install / "17" / "_tests" / ID)
    assert json.loads(capsys.readouterr().out)["fetched"] == 1


def test_queued_age_pid_reuse_and_training_refusal(install, monkeypatch, capsys):
    d = write_test()
    assert start(monkeypatch, {}) == 1
    assert ID in capsys.readouterr().err
    assert agent.main(["start", "--job", "example"]) == 1
    assert ID in capsys.readouterr().err
    old = agent.time.time() - 121
    os.utime(d, (old, old))
    assert agent.test_status(d)["state"] == "failed"
    (d / "status.json").write_text(json.dumps({"state": "rendering", "pid": 42, "pid_created": 123}), encoding="utf-8")
    seen = []
    monkeypatch.setattr(agent, "live_process", lambda pid, created=None: seen.append((pid, created)))
    assert agent.test_status(d)["state"] == "failed"
    assert seen == [(42, 123)]


def test_resend_live_dead_and_final(install, monkeypatch, capsys):
    d = write_test(state="rendering", pid=42)
    monkeypatch.setattr(agent, "live_process", lambda pid, created=None: object())
    monkeypatch.setattr(agent.subprocess, "run", lambda *a, **kw: pytest.fail("new called on resend"))
    assert start(monkeypatch, {"test_id": ID}) == 0
    assert json.loads(capsys.readouterr().out)["state"] == "rendering"
    monkeypatch.setattr(agent, "live_process", lambda pid, created=None: None)
    calls = []
    monkeypatch.setattr(agent.subprocess, "Popen", lambda *a, **kw: calls.append(a) or SimpleNamespace(pid=os.getpid()))
    assert start(monkeypatch, {"test_id": ID}) == 0
    assert len(calls) == 1
    capsys.readouterr()
    (d / "status.json").write_text('{"state":"done"}', encoding="utf-8")
    assert start(monkeypatch, {"test_id": ID}) == 1
    assert "already ran" in capsys.readouterr().err


def test_status_order_by_timestamps(install, capsys):
    a = write_test("a", state="done")
    b = write_test("z", state="done")
    (a / "spec.json").write_text('{"created":"2026-12-01"}', encoding="utf-8")
    (b / "status.json").write_text('{"test_id":"z","state":"done","started":"2026-11-01"}', encoding="utf-8")
    assert agent.main(["test-status"]) == 0
    assert [s["test_id"] for s in json.loads(capsys.readouterr().out)["tests"]] == ["a", "z"]


def test_remote_main_routes_start_without_retry(install, monkeypatch, capsys):
    spec = install / "spec.json"
    spec.write_text('{}', encoding="utf-8")
    calls = []
    class FakeRemote:
        def __init__(self, *args):
            pass
        def agent(self, *args, **kwargs):
            calls.append((args, kwargs))
            return subprocess.CompletedProcess(args, 0, b"{}", b"")
    monkeypatch.setattr(remote, "Remote", FakeRemote)
    monkeypatch.setattr(remote, "forward", lambda result: result.returncode)
    assert remote.main(["test-start", "--spec", str(spec)]) == 0
    assert calls[0][0] == ("test-start",) and calls[0][1]["retry"] is False
    assert json.loads(calls[0][1]["config"])["test_id"] in capsys.readouterr().out


def test_symlinks_and_unsafe_listings(install, monkeypatch):
    outside = install / "outside"
    outside.mkdir()
    Path("output").mkdir()
    try:
        (Path("output") / "_tests").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks unavailable")
    with pytest.raises(agent.Refused):
        agent.test_dir(ID, must_exist=False)
    (Path("output") / "_tests").unlink()
    root = Path("output/_tests")
    root.mkdir()
    (root / "linked").symlink_to(outside, target_is_directory=True)
    assert agent.running_tests() == []
    with pytest.raises(agent.Refused):
        agent.test_dir("linked", must_exist=False)
    monkeypatch.setattr(remote, "pulls_root", lambda: install)
    monkeypatch.setattr(remote, "fleet_move_in_progress", lambda: None)
    class FakeRemote:
        conn = {"id": 17}
        path = "../escape"
        def agent_json(self, *args):
            return {"base": "/remote", "files": [{"path": self.path, "size": 1}]}
    r = FakeRemote()
    args = remote.parser().parse_args(["pull", "--tests", ID])
    args.remote = r
    for path in ("../escape", "/absolute", "x\ny", "C:\\absolute"):
        r.path = path
        with pytest.raises(SystemExit, match="Unsafe"):
            remote.cmd_pull(args)
    (install / "17").mkdir()
    (install / "17" / "_tests").symlink_to(outside, target_is_directory=True)
    with pytest.raises(SystemExit, match="symlink"):
        remote.cmd_pull(args)
