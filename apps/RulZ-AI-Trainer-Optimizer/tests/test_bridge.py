import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from rulz_optimizer import bridge


def make_trainer(tmp_path: Path, venv: bool = True) -> Path:
    root = tmp_path / "RulzTrainer"
    (root / "rulztrainer").mkdir(parents=True)
    (root / "rulztrainer" / "config.py").write_text("", encoding="utf-8")
    (root / "rulztrainer" / "__init__.py").write_text('__version__ = "3.28.0"\n', encoding="utf-8")
    (root / "scripts" / "vast").mkdir(parents=True)
    (root / "scripts" / "vast" / "rulz_fleet.py").write_text("", encoding="utf-8")
    if venv:
        py = bridge.venv_python(root)
        py.parent.mkdir(parents=True)
        py.write_text("", encoding="utf-8")
    return root


def test_find_trainer_uses_hint_and_reads_version(tmp_path):
    root = make_trainer(tmp_path)
    inst = bridge.find_trainer(root)
    assert inst and inst.root == root and inst.version == "3.28.0"
    assert inst.python == bridge.venv_python(root)


def test_find_trainer_rejects_non_trainer(tmp_path):
    assert bridge.install_from(tmp_path) is None


def test_no_venv_falls_back_to_this_python(tmp_path):
    inst = bridge.install_from(make_trainer(tmp_path, venv=False))
    assert inst and inst.python == Path(sys.executable)


class Recorder:
    def __init__(self, returncode=0, stdout="", stderr="", raises=None):
        self.calls = []
        self.result = subprocess.CompletedProcess([], returncode, stdout, stderr)
        self.raises = raises

    def __call__(self, args, **kw):
        self.calls.append((args, kw))
        if self.raises:
            raise self.raises
        return self.result


def test_validate_runs_helper_with_list_args_and_timeout(tmp_path, monkeypatch):
    inst = bridge.install_from(make_trainer(tmp_path))
    rec = Recorder(0, "Valid: x on krea2\n")
    monkeypatch.setattr(bridge.subprocess, "run", rec)
    ok, msg = bridge.validate(inst, tmp_path / "c.yaml")
    assert ok and "Valid" in msg
    args, kw = rec.calls[0]
    assert isinstance(args, list) and args[1:3] == [str(bridge.HELPER), "validate"]
    assert kw["timeout"] == bridge.VALIDATE_TIMEOUT and kw["cwd"] == str(inst.root)
    assert not kw.get("shell")


def test_validate_failure_message(tmp_path, monkeypatch):
    inst = bridge.install_from(make_trainer(tmp_path))
    monkeypatch.setattr(bridge.subprocess, "run", Recorder(1, "ValueError: learning_rate must be > 0"))
    ok, msg = bridge.validate(inst, tmp_path / "c.yaml")
    assert not ok and "learning_rate" in msg


def test_timeout_and_start_failure(tmp_path, monkeypatch):
    inst = bridge.install_from(make_trainer(tmp_path))
    monkeypatch.setattr(bridge.subprocess, "run",
                        Recorder(raises=subprocess.TimeoutExpired(cmd="x", timeout=1)))
    ok, msg = bridge.validate(inst, tmp_path / "c.yaml")
    assert not ok and "Timed out" in msg
    monkeypatch.setattr(bridge.subprocess, "run", Recorder(raises=FileNotFoundError("nope")))
    ok, msg = bridge.queue_local(inst, tmp_path / "c.yaml", "lbl")
    assert not ok and "Could not start" in msg


def test_queue_local_mentions_webui(tmp_path, monkeypatch):
    inst = bridge.install_from(make_trainer(tmp_path))
    rec = Recorder(0, "Queued job (id abc) in q")
    monkeypatch.setattr(bridge.subprocess, "run", rec)
    ok, msg = bridge.queue_local(inst, tmp_path / "c.yaml", "my label")
    assert ok and "WebUI" in msg
    assert rec.calls[0][0][-2:] == [str(tmp_path / "c.yaml"), "my label"]


def test_queue_vast_command_and_exit_codes(tmp_path, monkeypatch):
    inst = bridge.install_from(make_trainer(tmp_path))
    ds = [tmp_path / "set1", tmp_path / "set2"]
    cmd = bridge.fleet_command(inst, tmp_path / "c.yaml", "job1", ds)
    assert cmd[2:5] == ["queue", "add", "--job"] or cmd[1].endswith("rulz_fleet.py")
    assert cmd.count("--push") == 2 and cmd[cmd.index("--job") + 1] == "job1"

    out = json.dumps({"queued": [{"id": "1", "job": "job1"}]})
    monkeypatch.setattr(bridge.subprocess, "run", Recorder(0, out))
    ok, msg = bridge.queue_vast(inst, tmp_path / "c.yaml", "job1", ds)
    assert ok and "job1" in msg and "Fleet" in msg

    for code, word in ((1, "refused"), (2, "timed out"), (3, "unreachable")):
        monkeypatch.setattr(bridge.subprocess, "run", Recorder(code, "", "[x] reason"))
        ok, msg = bridge.queue_vast(inst, tmp_path / "c.yaml", "job1", ds)
        assert not ok and word in msg and "reason" in msg


def test_queue_vast_without_fleet_script(tmp_path):
    root = make_trainer(tmp_path)
    (root / "scripts" / "vast" / "rulz_fleet.py").unlink()
    ok, msg = bridge.queue_vast(bridge.install_from(root), tmp_path / "c.yaml", "j", [])
    assert not ok and "rulz_fleet.py" in msg


def test_fleet_instances_reads_state(tmp_path, monkeypatch):
    monkeypatch.setenv("RULZ_VAST_HOME", str(tmp_path))
    assert bridge.fleet_instances() == []
    (tmp_path / "fleet").mkdir()
    (tmp_path / "fleet" / "state.json").write_text(json.dumps(
        {"instances": {"12": {"id": 12, "gpu": "RTX 5090", "status": "running", "label": "a"},
                       "13": {"id": 13, "gone": 5}}}), encoding="utf-8")
    items = bridge.fleet_instances()
    assert [i["id"] for i in items] == [12, 13]
    assert items[0]["gpu"] == "RTX 5090" and items[1]["gone"] is True


def test_open_webui_when_already_running(tmp_path, monkeypatch):
    inst = bridge.install_from(make_trainer(tmp_path))
    monkeypatch.setattr(bridge, "webui_running", lambda port=bridge.WEBUI_PORT: True)
    popen = []
    monkeypatch.setattr(bridge.subprocess, "Popen", lambda *a, **k: popen.append(a))
    assert "already running" in bridge.open_webui(inst)
    assert popen == []


def test_open_webui_starts_script(tmp_path, monkeypatch):
    root = make_trainer(tmp_path)
    (root / ("run_webui.bat" if bridge.IS_WINDOWS else "run_webui.sh")).write_text("", encoding="utf-8")
    inst = bridge.install_from(root)
    monkeypatch.setattr(bridge, "webui_running", lambda port=bridge.WEBUI_PORT: False)
    popen = []
    monkeypatch.setattr(bridge.subprocess, "Popen", lambda args, **k: popen.append((args, k)))
    assert "Starting" in bridge.open_webui(inst)
    assert isinstance(popen[0][0], list) and not popen[0][1].get("shell")


def test_helper_runs_for_real_with_a_fake_trainer(tmp_path):
    """The helper script against a tiny fake trainer package (no real trainer, no network)."""
    root = tmp_path / "t"
    (root / "rulztrainer").mkdir(parents=True)
    (root / "rulztrainer" / "__init__.py").write_text("", encoding="utf-8")
    (root / "rulztrainer" / "config.py").write_text(
        "import yaml\n"
        "class TrainConfig:\n"
        "    def __init__(self, d): self.d = d; self.job_name = d.get('job_name', ''); self.model_arch = d.get('model_arch')\n"
        "    @classmethod\n"
        "    def load(cls, p): return cls(yaml.safe_load(open(p, encoding='utf-8')) or {})\n"
        "    def validate(self):\n"
        "        if self.d.get('bad'): raise ValueError('bad config')\n"
        "    def to_dict(self): return dict(self.d)\n", encoding="utf-8")
    (root / "rulztrainer" / "train_queue.py").write_text(
        "import json\n"
        "class TrainQueue:\n"
        "    def __init__(self, path): self.path = path\n"
        "    def add(self, config, label=''):\n"
        "        self.path.parent.mkdir(parents=True, exist_ok=True)\n"
        "        self.path.write_text(json.dumps({'config': config, 'label': label}))\n"
        "        return {'id': 'x1', 'job': config.get('job_name', '')}\n", encoding="utf-8")
    good = tmp_path / "good.yaml"
    good.write_text("job_name: j1\nmodel_arch: krea2\n", encoding="utf-8")
    bad = tmp_path / "bad.yaml"
    bad.write_text("job_name: j2\nbad: true\n", encoding="utf-8")
    inst = bridge.TrainerInstall(root, Path(sys.executable), None)
    assert bridge.validate(inst, good)[0] is True
    ok, msg = bridge.validate(inst, bad)
    assert not ok and "bad config" in msg
    ok, msg = bridge.queue_local(inst, good, "lbl")
    assert ok and "x1" in msg
    queued = json.loads((root / "output" / "_queue.json").read_text())
    assert queued["config"]["job_name"] == "j1" and queued["label"] == "lbl"
    ok, _ = bridge.queue_local(inst, bad, "lbl")
    assert not ok
