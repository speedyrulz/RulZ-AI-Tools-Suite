"""The WebUI's control of the "render on this PC" helper: command line, one GPU per helper, stop, status words.
No GPU, no network: the process is a fake."""
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import renderhelper_ctl as ctl  # noqa: E402

GPUS = [{"index": 0, "name": "RTX 5090", "total_gb": 32.0, "free_gb": 30.0},
        {"index": 1, "name": "RTX 5060 Ti", "total_gb": 16.0, "free_gb": 15.0}]


class FakeIO:
    def __init__(self, proc=None):
        self.closed, self.proc = False, proc

    def close(self):
        self.closed = True
        if self.proc is not None and self.proc.exits_on_stdin_close:
            self.proc.code = 1 if self.proc.exit_code_on_close is None else self.proc.exit_code_on_close


class FakeProc:
    exits_on_stdin_close = False
    exit_code_on_close = None

    def __init__(self, cmd, **kw):
        self.cmd, self.kw, self.code = cmd, kw, None
        self.stdin = FakeIO(self)
        self.terminated = self.killed = False

    def poll(self):
        return self.code

    def terminate(self):
        self.terminated = True
        self.code = 0

    def kill(self):
        self.killed = True
        self.code = -9

    def wait(self, timeout=None):
        if self.code is None and self.hangs:
            raise subprocess.TimeoutExpired("x", timeout)
        if self.code is None:  # a helper that leaves once it was asked
            self.code = 0
        return self.code

    hangs = False


@pytest.fixture
def helpers(tmp_path):
    made = []

    def popen(cmd, **kw):
        p = FakeProc(cmd, **kw)
        made.append(p)
        return p

    h = ctl.RenderHelpers(tmp_path, popen=popen)
    h.made = made
    return h


def test_parse_gpus_skips_log_lines():
    text = "some warning\n" + json.dumps(GPUS) + "\n"
    assert [g["name"] for g in ctl.parse_gpus(text)] == ["RTX 5090", "RTX 5060 Ti"]
    with pytest.raises(ValueError):
        ctl.parse_gpus("nothing")


def test_command_line(tmp_path):
    cmd = ctl.helper_command(7, 1, tmp_path / "7.json")
    assert cmd[1:] == ["-m", "rulztrainer.renderhelper", "--instance", "7", "--device", "cuda:1",
                       "--status-file", str(tmp_path / "7.json")]
    cmd = ctl.helper_command(7, 0, tmp_path / "7.json", "C:/m/model.safetensors", "int8")
    assert cmd[-4:] == ["--sample-model", "C:/m/model.safetensors", "--sample-model-quant", "int8"]
    assert "--sample-model-quant" not in ctl.helper_command(7, 0, tmp_path / "x", "m.safetensors", "auto")
    assert "--sample-model-quant" not in ctl.helper_command(7, 0, tmp_path / "x", "", "int8")  # quant only with a model


def test_start_refuses_a_gpu_another_helper_uses_and_stop_ends_it(helpers):
    helpers.start(1, GPUS[0])
    assert helpers.running(1) and helpers.gpu_of(1) == 0
    with pytest.raises(ctl.Refused, match="already rendering for instance 1"):
        helpers.start(2, GPUS[0])
    with pytest.raises(ctl.Refused, match="already running"):
        helpers.start(1, GPUS[1])
    helpers.start(2, GPUS[1])  # another GPU is fine
    assert helpers.made[0].kw["stdin"] == subprocess.PIPE  # closing it stops a helper whose WebUI died
    helpers.made[0].hangs = True
    helpers.stop(1, wait=0.01)  # stdin closed first, then terminate when it does not leave
    assert helpers.made[0].stdin.closed and helpers.made[0].terminated and not helpers.running(1)
    helpers.start(3, GPUS[0])  # the GPU is free again
    helpers.stop_all()
    assert not helpers.running(2) and not helpers.running(3)


def test_stop_closes_stdin_first_and_is_not_an_error(helpers):
    helpers.start(1, GPUS[0])
    p = helpers.made[0]
    p.exits_on_stdin_close = True  # a clean stop; Windows would report exit code 1 for terminate()
    helpers.stop(1)
    assert p.stdin.closed and not p.terminated and not p.killed
    v = helpers.view(1)
    assert v["word"] == "Stopped" and v["level"] == "" and v["text"] == ""


def test_stop_kills_a_helper_that_ignores_terminate(helpers):
    helpers.start(1, GPUS[0])
    p = helpers.made[0]
    p.terminate = lambda: None
    p.hangs = True
    helpers.stop(1, wait=0.01)
    assert p.killed


def test_restart_closes_the_crashed_helpers_handles(helpers):
    helpers.start(1, GPUS[0])
    old = helpers.made[0]
    old.code = 3  # crashed
    assert helpers.view(1)["word"] == "Error"
    helpers.start(1, GPUS[0])
    assert old.stdin.closed and helpers.running(1)


def test_missing_sampling_model_is_refused(helpers, tmp_path):
    with pytest.raises(ctl.Refused, match="not found"):
        helpers.start(1, GPUS[0], sample_model=str(tmp_path / "nope.safetensors"))
    assert not helpers.made
    real = tmp_path / "m.safetensors"
    real.write_bytes(b"x")
    cmd = helpers.start(1, GPUS[0], sample_model=str(real), quant="nvfp4")
    assert str(real) in cmd and "nvfp4" in cmd


def test_start_clears_an_old_status_file(helpers):
    helpers.dir.mkdir(parents=True, exist_ok=True)
    helpers.status_file(1).write_text(json.dumps({"state": "ready"}), encoding="utf-8")
    helpers.start(1, GPUS[0])
    assert helpers.read_status(1) is None
    assert helpers.view(1)["word"] == "Connecting"


def test_list_gpus_runs_the_cli_once_and_caches(tmp_path):
    calls = []

    def run(cmd, **kw):
        calls.append((cmd, kw))
        return SimpleNamespace(returncode=0, stdout=json.dumps(GPUS), stderr="")

    h = ctl.RenderHelpers(tmp_path, run=run)
    assert h.cached_gpus() == []  # nothing runs until asked
    gpus, err = h.list_gpus()
    assert len(gpus) == 2 and not err
    h.list_gpus()
    assert len(calls) == 1 and "--list-gpus" in calls[0][0] and calls[0][1]["timeout"] > 0
    h.list_gpus(force=True)
    assert len(calls) == 2 and len(h.cached_gpus()) == 2


def test_list_gpus_failures_are_plain(tmp_path):
    def timeout(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, 1)

    gpus, err = ctl.RenderHelpers(tmp_path, run=timeout).list_gpus()
    assert gpus == [] and "too long" in err

    def broken(cmd, **kw):
        return SimpleNamespace(returncode=1, stdout="", stderr="Traceback\nImportError: no torch")

    gpus, err = ctl.RenderHelpers(tmp_path, run=broken).list_gpus()
    assert gpus == [] and err == "ImportError: no torch"


def test_status_words_icons_and_text():
    now = 1000.0
    v = ctl.status_view({"state": "rendering", "job": "ada", "rounds_done": 3, "last_seconds": 4.25, "behind": 1,
                         "time": now}, True, now=now)
    assert (v["word"], v["icon"], v["level"]) == ("Rendering", "autorenew", "ok")
    assert v["text"] == "ada · 3 rounds done · last 4.2 s · 1 behind" or v["text"].startswith("ada · 3 rounds done")
    v = ctl.status_view({"state": "not_ready", "message": "versions differ"}, True, now=now)
    assert (v["word"], v["level"], v["text"]) == ("Not ready", "warn", "versions differ")
    prep = ctl.status_view({"state": "preparing", "message": "loading the model"}, True)
    assert (prep["word"], prep["icon"]) == ("Preparing", "downloading")
    assert ctl.status_view({"state": "connecting"}, True)["word"] == "Connecting"
    assert ctl.status_view({"state": "ready"}, False, exit_code=1, stopped=True)["word"] == "Stopped"
    assert ctl.status_view({"state": "idle"}, True)["word"] == "Ready"
    assert ctl.status_view({"state": "error", "message": "boom"}, True)["level"] == "err"
    stale = ctl.status_view({"state": "ready", "time": now - 90}, True, now=now)
    assert stale["level"] == "warn" and "no news for 90 s" in stale["text"]
    assert ctl.status_view(None, True)["word"] == "Connecting"  # no status file yet
    gone = ctl.status_view({"state": "ready"}, False, exit_code=0)
    assert gone["word"] == "Stopped"
    crashed = ctl.status_view(None, False, exit_code=1)
    assert crashed["word"] == "Error" and "exit code 1" in crashed["text"]
    for word, icon, _ in list(ctl.STATES.values()) + [ctl.OFF]:
        assert word and icon  # a word and an icon for every state


def test_view_names_the_gpu(helpers):
    helpers.start(5, GPUS[1])
    v = helpers.view(5)
    assert v["running"] and v["gpu_name"] == "RTX 5060 Ti"
    helpers.status_file(5).write_text(json.dumps({"state": "ready", "time": 9e12}), encoding="utf-8")
    assert helpers.view(5)["word"] == "Ready"
    helpers.stop(5)
    assert helpers.view(5)["word"] == "Stopped" and not helpers.view(5)["running"]
