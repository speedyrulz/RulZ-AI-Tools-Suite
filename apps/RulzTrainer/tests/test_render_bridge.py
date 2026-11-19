"""The instance's bridge (scripts/vast/rulz_render_bridge.py), run the way the helper runs it: a subprocess started with
the python -c bootstrap, the script sent length-prefixed, then frames over its pipes, in a tmp "install folder" with
an output/ and a .webui_job.json (what rulz_agent.py reads to know the running job)."""
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import pytest

from rulztrainer.renderhelper import BOOTSTRAP

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/vast/rulz_render_bridge.py"
BASE = "job/remote_render/"


def frame(op, body=b"", **fields):
    header = json.dumps({"op": op, **fields}).encode()
    return b"RZRB" + struct.pack(">I", len(header)) + header + struct.pack(">Q", len(body)) + body


class Session:
    def __init__(self, folder: Path):
        self.folder = folder
        self.proc = subprocess.Popen([sys.executable, "-c", BOOTSTRAP], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     cwd=folder)
        script = SCRIPT.read_bytes()
        self.proc.stdin.write(struct.pack(">I", len(script)) + script)
        self.proc.stdin.flush()

    def __call__(self, op, body=b"", **fields):
        self.proc.stdin.write(frame(op, body, **fields))
        self.proc.stdin.flush()
        assert self.proc.stdout.read(4) == b"RZRB"
        reply = json.loads(self.proc.stdout.read(struct.unpack(">I", self.proc.stdout.read(4))[0]))
        return reply, self.proc.stdout.read(struct.unpack(">Q", self.proc.stdout.read(8))[0])


def link_dir(link: Path, target: Path) -> None:
    """A symlink, or on Windows without the privilege a junction (which resolve() follows too)."""
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        if sys.platform != "win32":
            raise
        subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], check=True, capture_output=True)


def start_job(folder: Path, name="job", pid=None, run=True):
    """A running job in the folder's output/, as the WebUI records one (this test's own process stands for it)."""
    job = folder / "output" / name
    (job / "remote_render").mkdir(parents=True, exist_ok=True)
    (folder / ".webui_job.json").write_text(json.dumps({"pid": os.getpid() if pid is None else pid,
                                                        "job_dir": str(job)}))
    if run:
        (job / "remote_render" / "run.json").write_text(json.dumps({"run_token": "tok", "job": name}))
    return job


@pytest.fixture
def bridge(tmp_path):
    (tmp_path / "output").mkdir()
    s = Session(tmp_path)
    yield s
    s.proc.stdin.close()
    s.proc.wait(timeout=10)
    assert s.proc.returncode == 0


def test_bootstrap_hello_and_protocol(bridge, tmp_path):
    reply, _ = bridge("hello", protocol=1)
    assert reply["ok"] and reply["protocol"] == 1 and reply["time"] > 0
    assert Path(reply["output_root"]) == (tmp_path / "output").resolve()
    reply, _ = bridge("hello", protocol=2)
    assert not reply["ok"] and "protocol" in reply["error"]
    assert not bridge("nonsense")[0]["ok"]  # unknown op is an answer, not the end of the bridge
    assert bridge("hello", protocol=1)[0]["ok"]


def test_files_and_listing(bridge, tmp_path):
    job = start_job(tmp_path)
    (job / "remote_render/rounds/000001_probe_1").mkdir(parents=True)  # the trainer's round
    blob = os.urandom(3 * 1024 * 1024)  # more than a pipe buffer
    assert bridge("put", blob, path=BASE + "rounds/000001_probe_1/lora.safetensors")[0]["ok"]
    assert bridge("get", path=BASE + "rounds/000001_probe_1/lora.safetensors")[1] == blob
    assert bridge("write_json", path=BASE + "prepare/ready.json", data={"a": 1})[0]["ok"]
    assert json.loads(bridge("get", path=BASE + "prepare/ready.json")[1]) == {"a": 1}
    assert bridge("list", path=BASE + "rounds")[0]["entries"] == ["000001_probe_1"]
    assert bridge("list", path="job/remote_render")[0]["entries"] == ["prepare", "rounds", "run.json"]
    assert not bridge("get", path=BASE + "nothing")[0]["ok"]
    assert not list((tmp_path / "output/job/remote_render/rounds/000001_probe_1").glob("*.tmp"))


def test_size_and_ranges(bridge, tmp_path):
    job = start_job(tmp_path)
    (job / "remote_render/data").write_bytes(b"0123456789")
    path = BASE + "data"
    assert bridge("size", path=path)[0]["size"] == 10
    assert bridge("get", path=path, offset=3, length=4)[1] == b"3456"
    assert bridge("get", path=path, offset=10, length=0)[1] == b""
    for offset, length in [(9, 2), (-1, 1), (0, -1), (True, 1)]:
        assert not bridge("get", path=path, offset=offset, length=length)[0]["ok"]
    for op in ("size", "get"):
        assert not bridge(op, path="job/remote_render/../secret", offset=0, length=1)[0]["ok"]
    (job / "remote_render/folder").mkdir()
    assert not bridge("size", path=BASE + "folder")[0]["ok"]  # a folder has no size to fetch


def test_runs_reports_only_the_running_job_with_run_json(bridge, tmp_path):
    assert bridge("runs")[0]["runs"] == []  # no .webui_job.json
    job = start_job(tmp_path, run=False)
    assert bridge("runs")[0]["runs"] == []  # running, but no run.json: not a remote-render run
    start_job(tmp_path)
    (job / "remote_render/rounds/000002_sample_10").mkdir(parents=True)
    (job / "remote_render/rounds/000001_probe_5").mkdir()
    (job / "remote_render/rounds/.tmp-3").mkdir()  # the trainer's half-written round
    (job / "remote_render/rounds/stray.txt").write_text("x")
    (job / "remote_render/claimed/000000_probe_0").mkdir(parents=True)
    (job / "remote_render/prepare").mkdir()
    (job / "remote_render/prepare/ready.json").write_text('{"run_token": "tok"}')
    runs = bridge("runs")[0]["runs"]
    assert len(runs) == 1
    r = runs[0]
    assert r["job"] == "job" and r["run"] == {"run_token": "tok", "job": "job"}
    assert r["rounds"] == ["000001_probe_5", "000002_sample_10"] and r["claimed"] == ["000000_probe_0"]
    assert r["ready"] == {"run_token": "tok"} and r["not_ready"] is None and r["has_prepare"] is False
    (job / "remote_render/prepare/prepare.json").write_text("{}")
    assert bridge("runs")[0]["runs"][0]["has_prepare"] is True


def test_runs_ignores_a_dead_job(bridge, tmp_path):
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    start_job(tmp_path, pid=dead.pid)
    assert bridge("runs")[0]["runs"] == []
    (tmp_path / ".webui_job.json").write_text("not json")
    assert bridge("runs")[0]["runs"] == []


def test_a_job_with_its_own_output_folder(bridge, tmp_path):
    own = tmp_path / "elsewhere" / "myjob"
    (own / "remote_render").mkdir(parents=True)
    (own / "remote_render/run.json").write_text('{"run_token": "t2"}')
    (tmp_path / ".webui_job.json").write_text(json.dumps({"pid": os.getpid(), "job_dir": str(own)}))
    assert bridge("runs")[0]["runs"][0]["job"] == "myjob"
    assert bridge("put", b"x", path="myjob/remote_render/helper.txt")[0]["ok"]
    assert (own / "remote_render/helper.txt").read_bytes() == b"x"


def test_heartbeat_writes_the_instance_clock(bridge, tmp_path):
    job = start_job(tmp_path)
    assert bridge("heartbeat", job="job", run_token="tok", host="pc", gpu="4090")[0]["ok"]
    beat = json.loads((job / "remote_render/helper.json").read_text())
    assert beat["run_token"] == "tok" and beat["host"] == "pc" and beat["gpu"] == "4090"
    import time

    assert abs(beat["time"] - time.time()) < 30
    assert not bridge("heartbeat", job="nojob", run_token="t", host="h", gpu="g")[0]["ok"]  # no remote_render there


def test_rename_never_replaces(bridge, tmp_path):
    job = start_job(tmp_path)
    (job / "remote_render/rounds/000001_probe_1").mkdir(parents=True)
    assert bridge("put", b"w", path=BASE + "rounds/000001_probe_1/round.json")[0]["ok"]
    src, dst = BASE + "rounds/000001_probe_1", BASE + "claimed/000001_probe_1"
    assert bridge("rename", src=src, dst=dst)[0]["renamed"] is True
    assert bridge("rename", src=src, dst=dst)[0]["renamed"] is False  # src is gone
    # a second round with the same name: the destination exists, so the later rename loses
    (job / "remote_render/rounds/000001_probe_1").mkdir()
    assert bridge("put", b"y", path=BASE + "rounds/000001_probe_1/round.json")[0]["ok"]
    assert bridge("rename", src=src, dst=dst)[0]["renamed"] is False
    assert (tmp_path / "output/job/remote_render/rounds/000001_probe_1/round.json").read_bytes() == b"y"
    assert (tmp_path / "output/job/remote_render/claimed/000001_probe_1/round.json").read_bytes() == b"w"
    # an empty destination folder is not replaced either (a plain rename would, on Linux)
    (tmp_path / "output/job/remote_render/done/000001_probe_1").mkdir(parents=True)
    assert bridge("rename", src=dst, dst=BASE + "done/000001_probe_1")[0]["renamed"] is False
    assert (tmp_path / "output/job/remote_render/claimed/000001_probe_1/round.json").exists()


def test_a_late_write_cannot_bring_a_taken_back_round_to_life(bridge, tmp_path):
    job = start_job(tmp_path)
    claimed = job / "remote_render/claimed/000001_probe_1"
    claimed.mkdir(parents=True)
    rr = BASE + "claimed/000001_probe_1/"
    assert bridge("put", b"png", path=rr + "out/probes/5_0.png")[0]["ok"]  # folders below the round are made
    assert (claimed / "out/probes/5_0.png").read_bytes() == b"png"
    (job / "remote_render/abandoned").mkdir()
    os.rename(claimed, job / "remote_render/abandoned/000001_probe_1")  # the trainer's deadline
    assert not bridge("put", b"png2", path=rr + "out/probes/5_1.png")[0]["ok"]
    assert not bridge("write_json", path=rr + "done.json", data={})[0]["ok"]
    assert not claimed.exists()  # nothing came back to life, so the helper's rename to done/ has nothing to win
    assert bridge("rename", src=rr[:-1], dst=BASE + "done/000001_probe_1")[0]["renamed"] is False
    assert not (job / "remote_render/done").exists()
    assert not bridge("put", b"x", path=BASE + "claimed/just_a_file")[0]["ok"]  # a round is a folder


@pytest.mark.parametrize("bad", ["../secret", "/etc/passwd", "job/../x", "job/remote_render/../x", "job/elsewhere",
                                 "job/remote_render/../../x", "job//remote_render/x", "job/remote_render//x",
                                 "job/remote_render/./x", "job\\remote_render\\x", "", "job", "..", "./job/remote_render/x",
                                 "job/remote_render/a\\..\\b"])
def test_path_escapes_are_refused(bridge, tmp_path, bad):
    start_job(tmp_path)
    (tmp_path / "secret").write_text("s")
    assert not bridge("get", path=bad)[0]["ok"]
    assert not bridge("put", b"x", path=bad)[0]["ok"]
    assert not bridge("list", path=bad)[0]["ok"]
    assert not (tmp_path / "x").exists() and not (tmp_path / "output/x").exists()


def test_writes_cannot_touch_the_folder_itself_or_cross_jobs(bridge, tmp_path):
    start_job(tmp_path)
    (tmp_path / "output/other/remote_render/rounds/a").mkdir(parents=True)
    assert not bridge("rename", src="job/remote_render", dst="job/remote_render2")[0]["ok"]
    assert not bridge("rename", src="other/remote_render/rounds/a", dst=BASE + "claimed/a")[0]["ok"]
    assert not bridge("put", b"x", path="job/remote_render")[0]["ok"]
    for bad_job in ("..", ".", "a/b"):
        assert not bridge("heartbeat", job=bad_job, run_token="t", host="h", gpu="g")[0]["ok"]


def test_symlinks_cannot_lead_out(bridge, tmp_path):
    job = start_job(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "file").write_text("secret")
    link_dir(job / "remote_render/link", outside)
    paths = ["link/file", "link/new"]
    try:
        (job / "remote_render/filelink").symlink_to(outside / "file")
        paths.append("filelink")
    except OSError:
        pass  # (a file symlink needs a privilege on Windows)
    for p in paths:
        assert not bridge("get", path=BASE + p)[0]["ok"]
        assert not bridge("put", b"x", path=BASE + p)[0]["ok"]
    assert not bridge("list", path=BASE + "link")[0]["ok"]
    assert not bridge("rename", src=BASE + "link", dst=BASE + "moved")[0]["ok"]
    assert (outside / "file").read_text() == "secret" and not (outside / "new").exists()


def test_remote_render_folder_symlink_and_job_symlink_are_refused(bridge, tmp_path):
    outside = tmp_path / "outside"
    (outside / "remote_render").mkdir(parents=True)
    (outside / "remote_render/run.json").write_text('{"run_token": "t"}')
    link_dir(tmp_path / "output/linked", outside)
    (tmp_path / "output/job").mkdir()
    link_dir(tmp_path / "output/job/remote_render", outside / "remote_render")
    assert not bridge("get", path="linked/remote_render/run.json")[0]["ok"]  # the job folder leads out of output/
    assert not bridge("get", path="job/remote_render/run.json")[0]["ok"]  # remote_render itself is a link
    (tmp_path / ".webui_job.json").write_text(json.dumps({"pid": os.getpid(), "job_dir": str(tmp_path / "output/job")}))
    assert bridge("runs")[0]["runs"] == []


def test_bad_magic_ends_the_bridge(tmp_path):
    (tmp_path / "output").mkdir()
    s = Session(tmp_path)
    s.proc.stdin.write(b"XXXX" + b"\0" * 12)
    s.proc.stdin.flush()
    assert s.proc.wait(timeout=10) == 0


def test_stdin_end_ends_the_bridge_and_stray_output_stays_out_of_frames(tmp_path):
    (tmp_path / "output").mkdir()
    s = Session(tmp_path)
    assert s("hello", protocol=1)[0]["ok"]
    s.proc.stdin.close()
    assert s.proc.wait(timeout=10) == 0
