"""scripts/vast/rulz_remote.py and rulz_agent.py: finding the instance, what the agent reads and refuses, a run going
through the WebUI's queue, and the copy paths. No ssh and no GPU: the agent runs here on a fake install folder."""
import base64
import io
import json
import os
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "vast"))

import rulz_agent as agent  # noqa: E402
import rulz_remote as remote  # noqa: E402

from rulztrainer.config import TrainConfig  # noqa: E402
from rulztrainer.train_queue import TrainQueue  # noqa: E402


# ---------------------------------------------------------------- finding the instance

# the fields of `vastai show instances --raw` this tool reads, from a real instance
LIVE = {"id": 11, "actual_status": "running", "gpu_name": "RTX 5090", "dph_total": 0.8,
        "image_uuid": "vastai/pytorch:cuda-12.8.1-auto", "public_ipaddr": "199.0.0.1\n",
        "ssh_host": "ssh8.vast.ai", "ssh_port": 28486,
        "extra_env": [["-p 22:22", "1"], ["RULZ_REPO", "https://example/x.git"]],
        "ports": {"22/tcp": [{"HostIp": "0.0.0.0", "HostPort": "35529"}, {"HostIp": "::", "HostPort": "35529"}]}}


def plain(i=12, status="running"):
    return {"id": i, "actual_status": status, "gpu_name": "RTX 3090", "extra_env": [], "image_uuid": "vastai/base"}


def test_the_running_rulztrainer_instance_is_picked_over_others():
    assert remote.pick_instance([plain(), LIVE])["id"] == 11


def test_the_dockerfile_image_counts_as_rulztrainer():
    baked = dict(plain(13), image_uuid="me/rulztrainer:latest")
    assert remote.pick_instance([plain(), baked])["id"] == 13


def test_a_lone_running_instance_is_used_without_the_template_marks():
    assert remote.pick_instance([plain(), plain(14, "stopped")])["id"] == 12


def test_several_candidates_ask_for_an_id_and_none_says_what_there_is():
    with pytest.raises(remote.Unreachable, match="--instance: 11 .*, 15"):
        remote.pick_instance([LIVE, dict(LIVE, id=15)])
    with pytest.raises(remote.Unreachable, match="14: stopped"):
        remote.pick_instance([plain(14, "stopped")])
    with pytest.raises(remote.Unreachable, match="none rented"):
        remote.pick_instance([])


def test_a_named_instance_must_exist_and_run():
    assert remote.pick_instance([LIVE, plain()], 12)["id"] == 12
    with pytest.raises(remote.Unreachable, match="No instance 99"):
        remote.pick_instance([LIVE], 99)
    with pytest.raises(remote.Unreachable, match="'stopped'"):
        remote.pick_instance([plain(14, "stopped")], 14)


def test_the_direct_route_comes_first():
    """vast's proxy was down on a real instance whose direct port worked."""
    assert remote.routes(LIVE) == [{"route": "direct", "host": "199.0.0.1", "port": 35529},
                                   {"route": "proxy", "host": "ssh8.vast.ai", "port": 28486}]
    assert [r["route"] for r in remote.routes(dict(LIVE, ports={}))] == ["proxy"]


def test_each_rental_has_its_own_known_hosts(tmp_path, monkeypatch):
    monkeypatch.setattr(remote, "STATE", tmp_path)
    cmd = " ".join(remote.ssh_command({"id": 11, "host": "h", "port": 1}))
    assert "known_hosts_11" in cmd and "StrictHostKeyChecking=accept-new" in cmd and "BatchMode=yes" in cmd
    assert cmd.endswith("root@h")


def test_a_dropped_route_is_looked_up_again_once(monkeypatch):
    calls, returns = [], [255, 0]

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return SimpleNamespace(returncode=returns.pop(0), stdout=b"", stderr=b"")

    monkeypatch.setattr(remote.subprocess, "run", fake_run)
    monkeypatch.setattr(remote, "saved_connection", lambda wanted: {"id": 1, "host": "old", "port": 1})
    monkeypatch.setattr(remote, "find_connection", lambda *a: {"id": 1, "host": "new", "port": 2})
    assert remote.Remote(None, None, "/w").run("ls").returncode == 0
    assert "root@old" in calls[0] and "root@new" in calls[1]


def test_a_remote_commands_own_failure_is_not_a_dropped_route(monkeypatch):
    monkeypatch.setattr(remote.subprocess, "run", lambda cmd, **kw: SimpleNamespace(returncode=7, stdout=b"", stderr=b""))
    monkeypatch.setattr(remote, "saved_connection", lambda wanted: {"id": 1, "host": "h", "port": 1})
    monkeypatch.setattr(remote, "find_connection", lambda *a: pytest.fail("looked up again"))
    assert remote.Remote(None, None, "/w").run("false").returncode == 7


def test_the_saved_route_is_only_for_its_instance(tmp_path, monkeypatch):
    monkeypatch.setattr(remote, "CONNECTION", tmp_path / "c.json")
    (tmp_path / "c.json").write_text(json.dumps({"id": 1, "host": "h", "port": 2}), encoding="utf-8")
    assert remote.saved_connection(None)["id"] == 1 and remote.saved_connection(1)["id"] == 1
    assert remote.saved_connection(2) is None


# ---------------------------------------------------------------- the local side's commands


def test_agent_commands_reach_the_agent_with_global_options_taken_out(monkeypatch):
    seen = {}

    class Fake:
        def __init__(self, instance, key, folder):
            seen.update(instance=instance, folder=folder)

        def agent(self, *args, config=None):
            seen["args"] = args
            return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(remote, "Remote", Fake)
    assert remote.main(["--instance", "5", "--dir", "/w", "log", "job", "-n", "3", "--grep", "x y"]) == 0
    assert seen == {"instance": 5, "folder": "/w", "args": ("log", "job", "-n", "3", "--grep", "x y")}


def test_a_local_config_is_taken_out_of_the_arguments(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_bytes(b"steps: 5\n")
    args, config = remote.agent_args(["start", "--job", "a", "--config", str(path), "--set", "x=1"])
    assert args == ["start", "--job", "a", "--set", "x=1"] and config == b"steps: 5\n"
    assert remote.agent_args(["start", "--job", "a", "--from", "b"]) == (["start", "--job", "a", "--from", "b"], None)


def test_a_config_of_any_size_stays_off_the_ssh_command_line(monkeypatch):
    """Git for Windows' ssh cut the remote command at about 8190 characters, with exit status 0 and no error."""
    calls = []

    def fake_run(cmd, **kw):
        calls.append((cmd, kw["input"]))
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(remote.subprocess, "run", fake_run)
    monkeypatch.setattr(remote, "saved_connection", lambda wanted: {"id": 1, "host": "h", "port": 1})
    config = b"steps: 5\n" * 20000
    remote.Remote(None, None, "/w").agent("start", "--job", "a", config=config)
    (cmd, stdin), = calls
    assert sum(len(word) for word in cmd) < 1000 and base64.b64encode(config) in stdin


def test_pull_skips_what_is_here_at_its_size(tmp_path):
    (tmp_path / "j").mkdir()
    (tmp_path / "j" / "same.png").write_bytes(b"12345")
    (tmp_path / "j" / "train.log").write_bytes(b"12")
    files = [{"path": "j/same.png", "size": 5}, {"path": "j/train.log", "size": 9}, {"path": "j/new.png", "size": 1}]
    assert [f["path"] for f in remote.pull_plan(files, tmp_path)] == ["j/train.log", "j/new.png"]


def tar_stream(name: str, data: bytes) -> io.BytesIO:
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w") as tar:
        info = tarfile.TarInfo(name)
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    out.seek(0)
    return out


def test_a_pulled_archive_lands_in_its_folder_and_cannot_leave_it(tmp_path):
    remote.extract(tar_stream("job/train.log", b"hello"), tmp_path / "d")
    assert (tmp_path / "d" / "job" / "train.log").read_bytes() == b"hello"
    with pytest.raises(tarfile.TarError):
        remote.extract(tar_stream("../escaped.txt", b"x"), tmp_path / "d")
    assert not (tmp_path / "escaped.txt").exists()


def test_files_sent_from_windows_keep_scripts_runnable():
    def mode(name, type_=tarfile.REGTYPE):
        info = tarfile.TarInfo(name)
        info.type, info.uid, info.mode = type_, 1000, 0o666
        return remote.tar_info_for_instance(info)

    assert mode("scripts/vast/provision.sh").mode == 0o755 and mode("webui.py").mode == 0o644
    assert mode("scripts", tarfile.DIRTYPE).mode == 0o755 and mode("webui.py").uid == 0


def test_code_files_take_edits_and_new_files_but_not_ignored_ones(tmp_path):
    def git(*a):
        subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", *a], cwd=tmp_path, check=True,
                       capture_output=True)

    (tmp_path / "rulztrainer").mkdir()
    (tmp_path / "rulztrainer" / "a.py").write_text("1")
    (tmp_path / "webui.py").write_text("1")
    (tmp_path / "README.md").write_text("1")
    (tmp_path / ".gitignore").write_text("*.pyc\n")
    git("init")
    git("add", ".")
    git("commit", "-m", "x")
    (tmp_path / "rulztrainer" / "new.py").write_text("1")
    (tmp_path / "rulztrainer" / "a.pyc").write_text("1")
    assert remote.code_files(tmp_path) == ["rulztrainer/a.py", "rulztrainer/new.py", "webui.py"]


class FakeWaitRemote:
    """poll answers in turn; a None answer is a dropped connection."""

    def __init__(self, answers):
        self.answers = list(answers)

    def agent(self, *args):
        answer = self.answers.pop(0)
        if answer is None:
            return SimpleNamespace(returncode=255, stdout=b"", stderr=b"")
        if isinstance(answer, Exception):
            raise answer
        return SimpleNamespace(returncode=0, stdout=json.dumps(answer).encode(), stderr=b"")


def wait_args(answers, **kw):
    base = dict(name="j", step=None, every=0, timeout=0, max_down=600)
    return SimpleNamespace(remote=FakeWaitRemote(answers), **{**base, **kw})


RUNNING = {"job": "j", "state": "running", "step": 5, "total": 10}


def test_wait_rides_out_a_dropped_connection(capsys):
    args = wait_args([RUNNING, None, remote.Unreachable("gone"), {"job": "j", "state": "done"}])
    assert remote.cmd_wait(args) == 0


def test_wait_reports_a_failed_or_killed_run(capsys):
    assert remote.cmd_wait(wait_args([RUNNING, {"job": "j", "state": "error", "error": "boom"}])) == 1
    assert "boom" in capsys.readouterr().err
    assert remote.cmd_wait(wait_args([{"job": "j", "state": "ended"}])) == 1


def test_wait_can_stop_at_a_step_and_gives_up_in_time(capsys):
    assert remote.cmd_wait(wait_args([RUNNING, dict(RUNNING, step=7)], step=7)) == 0
    assert remote.cmd_wait(wait_args([None, None], max_down=-1)) == 3
    assert remote.cmd_wait(wait_args([RUNNING] * 3, timeout=-1)) == 2


# ---------------------------------------------------------------- the agent, on a fake install


@pytest.fixture
def install(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(agent, "webui_running", lambda: True)
    monkeypatch.setattr(agent, "START_WAIT", 3.0)
    (tmp_path / "output").mkdir()
    (tmp_path / "datasets" / "set").mkdir(parents=True)
    return tmp_path


def make_job(name="old", **status):
    d = Path("output") / name
    d.mkdir(parents=True)
    TrainConfig(job_name=name, dataset_dir="datasets/set").save(d / "config.yaml")
    (d / "status.json").write_text(json.dumps({"step": 10, "total": 10, "done": True, "error": None, **status}),
                                   encoding="utf-8")
    return d


def run(capsys, *argv):
    code = agent.main(list(argv))
    out = capsys.readouterr()
    return code, out.out, out.err


def run_json(capsys, *argv):
    code, out, err = run(capsys, *argv)
    return code, json.loads(out) if out.strip() else None


@pytest.fixture
def running_job():
    """A live process recorded as the job, the way the WebUI records one."""
    d = make_job("live", done=False, step=3)
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    Path(".webui_job.json").write_text(json.dumps({
        "pid": proc.pid, "create_time": psutil.Process(proc.pid).create_time(), "job_dir": str(d.resolve())}))
    yield d
    proc.kill()
    proc.wait()


def test_status_works_on_an_empty_install(install, capsys):
    code, data = run_json(capsys, "status")
    assert code == 0 and data["job_running"] is False and data["datasets"] == ["set"] and "job" not in data


def test_the_per_step_series_are_summed_up(install, capsys):
    d = make_job("a", losses=[0.1] * 500, step_times=[1] * 500, saved=["x", "y"],
                 probes=[{"step": 1, "combined_ema": 3.0, "images": ["a.png"]}])
    (d / "train.log").write_text("x")
    code, data = run_json(capsys, "job", "a")
    assert code == 0 and "losses" not in data and "step_times" not in data
    assert data["saved_count"] == 2 and data["last_saved"] == "y" and data["last_probe"] == {
        "step": 1, "combined_ema": 3.0}
    assert "train.log" in data["files"]
    assert len(run_json(capsys, "job", "a", "--full")[1]["losses"]) == 500


def test_jobs_lists_runs_not_the_outputs_other_folders(install, capsys):
    make_job("a", likeness={"best": 30.0, "best_step": 8})
    (Path("output") / "_history").mkdir()
    (Path("output") / "_history" / "status.json").write_text("{}")
    (Path("output") / "empty").mkdir()
    _, data = run_json(capsys, "jobs")
    assert [j["job"] for j in data] == ["a"] and data[0]["best"] == 30.0


def test_a_job_name_cannot_leave_the_output_folder(install, capsys):
    (install / "secret").mkdir()
    (install / "secret" / "status.json").write_text("{}")
    for name in ("../secret", "..", "a/b"):
        code, out, err = run(capsys, "log", name)
        assert code == 1 and "Not a job name" in err and not out


def test_log_tails_filters_and_falls_back_to_the_current_job(install, capsys):
    d = make_job("a")
    (d / "train.log").write_text("one\ntwo probe\nthree\nfour probe\n")
    (d / "console.log").write_text("Traceback\n")
    Path(".webui_job.json").write_text(json.dumps({"pid": 1, "job_dir": str(d.resolve())}))
    assert run(capsys, "log", "-n", "2")[1] == "three\nfour probe\n"
    assert run(capsys, "log", "a", "--grep", "probe", "-n", "0")[1] == "two probe\nfour probe\n"
    assert run(capsys, "log", "a", "--console")[1] == "Traceback\n"


def test_probes_are_a_table_and_a_torn_last_line_is_left_out(install, capsys):
    d = make_job("a")
    row = {"step": 1, "combined_ema": 2.0, "images": ["a.png"], "face_seeds": [1, 2], "gold_ema": None}
    (d / "likeness.jsonl").write_text(json.dumps(row) + "\n" + json.dumps(dict(row, step=2)) + '\n{"step": 3, "fa')
    _, data = run_json(capsys, "probes", "a")
    assert data == [{"step": 1, "combined_ema": 2.0}, {"step": 2, "combined_ema": 2.0}]
    assert run_json(capsys, "probes", "a", "--last", "1", "--full")[1][0]["images"] == ["a.png"]


def test_config_diff_shows_only_what_differs(install, capsys):
    make_job("a")
    d = make_job("b")
    TrainConfig(job_name="b", dataset_dir="datasets/set", steps=7).save(d / "config.yaml")
    _, data = run_json(capsys, "config", "a", "--diff", "b")
    assert data == {"job_name": ["a", "b"], "steps": [2000, 7]}


def test_files_pick_by_kind_and_step(install, capsys):
    d = make_job("a")
    (d / "probes").mkdir()
    (d / "race").mkdir()
    for name in ("train.log", "race/race.json", "probes/step000100_face_0.png", "probes/step000200_face_0.png",
                 "a_000100.safetensors", "a_000200.safetensors", "a_best.safetensors"):
        (d / name).write_bytes(b"x")

    def paths(*argv):
        return sorted(f["path"].split("/", 1)[1] for f in run_json(capsys, "files", "a", *argv)[1]["files"])

    assert paths() == ["config.yaml", "race/race.json", "status.json", "train.log"]
    assert paths("--what", "probes", "--step", "100") == ["probes/step000100_face_0.png"]
    assert paths("--what", "best") == ["a_best.safetensors"]
    assert paths("--what", "loras", "--step", "200") == ["a_000200.safetensors"]
    assert len(paths("--what", "all")) == 4 + 2 + 3


def test_files_maps_fetch_the_training_maps_folder(install, capsys):
    d = make_job("a")
    (d / "training_maps").mkdir()
    for name in ("maps.jsonl", "probes.json", "probe_0.jpg", "probe_1.jpg"):
        (d / "training_maps" / name).write_bytes(b"x")
    data = run_json(capsys, "files", "a", "--what", "maps", "--step", "100")[1]
    assert sorted(f["path"] for f in data["files"]) == [
        "a/training_maps/maps.jsonl", "a/training_maps/probe_0.jpg", "a/training_maps/probe_1.jpg",
        "a/training_maps/probes.json"]


def test_files_report_the_folder_they_are_under(install, capsys):
    make_job("a")
    data = run_json(capsys, "files", "a")[1]
    base = Path(data["base"])
    assert base.is_absolute() and base.name == "output" and data["files"][0]["path"].startswith("a/")


def test_poll_says_how_a_job_stands(install, capsys, running_job):
    make_job("won")
    make_job("failed", error="boom", done=False)
    make_job("killed", done=False)
    TrainQueue(agent.QUEUE_FILE).add({"job_name": "next"})
    state = lambda name=None: run_json(capsys, "poll", *([name] if name else []))[1]["state"]  # noqa: E731
    assert state() == state("live") == "running"
    assert state("next") == "queued" and state("won") == "done"
    assert state("failed") == "error" and state("killed") == "ended"


def test_a_pid_reused_by_another_process_is_not_the_job(install, capsys, running_job):
    data = json.loads(Path(".webui_job.json").read_text())
    Path(".webui_job.json").write_text(json.dumps(dict(data, create_time=data["create_time"] - 100)))
    assert agent.job_process() is None
    assert run_json(capsys, "status")[1]["job_running"] is False


# ---------------------------------------------------------------- starting and stopping


def picks_up(ok=True, message="Started"):
    """The WebUI's queue thread, once a run waits (a first start spends seconds importing torch to validate)."""
    started = []

    def tick():
        deadline = time.time() + 60
        while time.time() < deadline:
            snap = TrainQueue(agent.QUEUE_FILE).snapshot()
            if snap["waiting"] and not snap["paused"]:
                break
            time.sleep(0.05)
        TrainQueue(agent.QUEUE_FILE).advance(False, lambda config: (started.append(config) or ok, message))

    threading.Thread(target=tick, daemon=True).start()
    return started


def test_a_run_goes_through_the_queue_with_the_config_it_was_given(install, capsys):
    make_job("old")
    started = picks_up()
    code, data = run_json(capsys, "start", "--job", "new", "--from", "old", "--set", "steps=50",
                          "--set", "optimizer=hull", "--set", "hull_rho=0.01")
    assert code == 0 and data["state"] == "running" and data["changed"]["steps"] == [2000, 50]
    assert started[0]["job_name"] == "new" and started[0]["steps"] == 50 and started[0]["optimizer"] == "hull"
    TrainConfig.from_dict(started[0])  # what the WebUI's _start_queued does with it


def test_a_local_config_is_the_base_of_a_run(install, capsys, monkeypatch):
    text = yaml.safe_dump({"job_name": "ignored", "dataset_dir": "datasets/set", "steps": 9})
    monkeypatch.setattr(agent, "SENT_CONFIG", base64.b64encode(text.encode()).decode())
    started = picks_up()
    code, data = run_json(capsys, "start", "--job", "new")
    assert code == 0 and data["steps"] == 9 and started[0]["job_name"] == "new"


def test_a_run_the_webui_could_not_start_is_reported(install, capsys):
    make_job("old")
    picks_up(ok=False, message="Dataset folder not found: x")
    code, data = run_json(capsys, "start", "--job", "new", "--from", "old")
    assert code == 1 and data["state"] == "failed" and "not found" in data["message"]


def test_a_dry_run_queues_nothing_but_checks_everything(install, capsys):
    make_job("old")
    code, data = run_json(capsys, "start", "--job", "new", "--from", "old", "--dry-run")
    assert code == 0 and data["dry_run"] and not Path(agent.QUEUE_FILE).exists()
    assert run(capsys, "start", "--job", "old", "--from", "old", "--dry-run")[0] == 1


RACE_OF_HULLS = ["--set", "optimizer=hull", "--set", "hull_rho=0.01", "--set", "race=true", "--set", "probe_every=100",
                 "--set", "face_refs=datasets/set", "--set", 'race_configs=[{"name": "a"}, {"name": "b"}]']


@pytest.mark.parametrize("dry_run", [[], ["--dry-run"]], ids=["start", "dry-run"])
@pytest.mark.parametrize("extra, why", [
    (["--job", "old"], "already exists"),
    (["--job", "new", "--set", "nonsense=1"], "Unknown config keys"),
    (["--job", "new", "--set", "dataset_dir=datasets/gone"], "Dataset folder not found"),
    (["--job", "new", "--set", "broken"], "Not KEY=VALUE"),
    (["--job", "new", "--set", "loss_type=huber"], "loss_type must be"),  # refused only by validate, as training does
    (["--job", "new", *RACE_OF_HULLS], "needs face_gold: adaface"),  # a real run's: dry-run passed, then it failed
])
def test_what_a_start_refuses_queues_nothing(install, capsys, extra, why, dry_run):
    make_job("old")
    code, _, err = run(capsys, "start", "--from", "old", *extra, *dry_run)
    assert code == 1 and why in err and not Path(agent.QUEUE_FILE).exists()


def test_a_start_needs_a_base(install, capsys):
    assert "--from JOB or --config" in run(capsys, "start", "--job", "new")[2]


def test_overwrite_is_asked_for_by_name(install, capsys):
    make_job("old")
    picks_up()
    assert run_json(capsys, "start", "--job", "old", "--from", "old", "--overwrite")[0] == 0


def test_a_job_name_in_use_is_refused(install, capsys, running_job):
    assert "already running or queued" in run(capsys, "start", "--job", "live", "--from", "live", "--overwrite")[2]
    TrainQueue(agent.QUEUE_FILE).add({"job_name": "next"})
    assert "already running or queued" in run(capsys, "start", "--job", "next", "--from", "live")[2]


def test_a_run_waits_behind_a_running_job_without_blocking(install, capsys, running_job):
    begun = time.time()
    code, data = run_json(capsys, "start", "--job", "new", "--from", "live")
    assert code == 0 and data["state"] == "queued" and "ahead" in data["note"] and time.time() - begun < 2


def test_a_paused_queue_is_said_and_can_be_resumed_with_the_start(install, capsys):
    make_job("old")
    TrainQueue(agent.QUEUE_FILE).set_paused(True)
    code, data = run_json(capsys, "start", "--job", "new", "--from", "old")
    assert code == 0 and data["state"] == "queued" and data["paused"] and "paused" in data["note"]
    started = picks_up()
    code, data = run_json(capsys, "start", "--job", "new2", "--from", "old", "--resume-queue")
    assert code == 0 and started[0]["job_name"] == "new" and data["state"] == "queued"  # new2 is next in line


def test_without_the_webui_a_queued_run_says_so(install, capsys, monkeypatch):
    make_job("old")
    monkeypatch.setattr(agent, "webui_running", lambda: False)
    code, data = run_json(capsys, "start", "--job", "new", "--from", "old")
    assert code == 1 and "supervisorctl" in data["note"]


def test_stop_is_the_webuis_stop_file(install, capsys, running_job):
    assert run_json(capsys, "stop")[1]["state"] == "stopping"
    assert (running_job / "STOP").exists() and agent.job_process() is not None


def test_kill_ends_the_process_and_pauses_a_waiting_queue(install, capsys, running_job):
    TrainQueue(agent.QUEUE_FILE).add({"job_name": "next"})
    proc = agent.job_process()
    data = run_json(capsys, "stop", "--kill")[1]
    proc.wait(10)
    assert data["state"] == "killed" and data["queue"]["paused"] is True


def test_stop_without_a_job_is_refused(install, capsys):
    assert run(capsys, "stop")[0] == 1


def test_the_queue_can_be_shown_paused_and_edited(install, capsys):
    entry = TrainQueue(agent.QUEUE_FILE).add({"job_name": "next"})
    assert run_json(capsys, "queue")[1]["waiting"] == [{"id": entry["id"], "job": "next"}]
    assert run_json(capsys, "queue", "pause")[1]["paused"] is True
    assert run_json(capsys, "queue", "resume")[1]["paused"] is False
    assert run(capsys, "queue", "remove", "nope")[0] == 1
    assert run_json(capsys, "queue", "remove", entry["id"])[1]["waiting"] == []


# ---------------------------------------------------------------- the way it reaches an instance


def piped(cwd, *args, program=None, env=None):
    """As rulz_remote.py runs it: the script on stdin, no script file, the install as the working directory."""
    return subprocess.run([sys.executable, "-", *args], input=program or remote.agent_program(), cwd=cwd,
                          capture_output=True, env=env)


def test_the_agent_runs_piped_and_finds_the_repo_from_the_working_directory(install):
    (install / "rulztrainer").mkdir()
    (install / "rulztrainer" / "__init__.py").write_text("")
    (install / "rulztrainer" / "train_queue.py").write_text(
        (REPO / "rulztrainer" / "train_queue.py").read_text(encoding="utf-8"), encoding="utf-8")
    make_job("a")
    done = piped(install, "queue", "pause")
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout)["paused"] is True and json.loads(agent.QUEUE_FILE.read_text())["paused"] is True
    assert json.loads(piped(install, "jobs").stdout)[0]["job"] == "a"
    refused = piped(install, "log", "nope")
    assert refused.returncode == 1 and b"No job named" in refused.stderr


def test_a_config_too_long_for_a_command_line_arrives_whole(install):
    last = yaml.safe_dump({"dataset_dir": "datasets/set", "steps": 9})  # last: a cut would lose it
    config = ("# padding\n" * 20000 + last).encode()
    done = piped(install, "start", "--job", "new", "--dry-run", program=remote.agent_program(config),
                 env={**os.environ, "PYTHONPATH": str(REPO)})
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout)["steps"] == 9


def test_the_agent_stays_ascii_for_any_locale_on_the_instance():
    assert (REPO / "scripts" / "vast" / "rulz_agent.py").read_bytes().isascii()


def test_the_local_side_needs_only_the_standard_library():
    tree = (REPO / "scripts" / "vast" / "rulz_remote.py").read_text(encoding="utf-8")
    imports = {line.split()[1].split(".")[0] for line in tree.splitlines() if line.startswith(("import ", "from "))}
    assert imports <= set(sys.stdlib_module_names) | {"__future__"}, imports
