"""Audit fixes in the evaluation and tooling modules: face scoring's CPU threads under a container quota (facesim),
history files two processes write (history), the queue's replace retry (train_queue), and resume state written whole
(regionwatch, regionsweep, selftrain)."""
import json
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import facesim, history, regionsweep, regionwatch, selftrain, train_queue  # noqa: E402
from rulztrainer.config import TrainConfig  # noqa: E402
from rulztrainer.dataset import Item  # noqa: E402


# ---------------------------------------------------------------- face scoring's CPU threads


def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b, n = n & 0x7F, n >> 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _field(num: int, value) -> bytes:
    if isinstance(value, int):
        return _varint(num << 3) + _varint(value)
    data = value.encode() if isinstance(value, str) else value
    return _varint(num << 3 | 2) + _varint(len(data)) + data


def _identity_onnx(path: Path) -> Path:
    """A one-node ONNX model (y = x, float[4]) encoded by hand: the onnx package is not needed to make one."""
    tensor = _field(1, _field(1, 1) + _field(2, _field(1, _field(1, 4))))
    info = lambda name: _field(1, name) + _field(2, tensor)  # noqa: E731
    node = _field(1, "x") + _field(2, "y") + _field(3, "id") + _field(4, "Identity")
    graph = _field(1, node) + _field(2, "g") + _field(11, info("x")) + _field(12, info("y"))
    path.write_bytes(_field(1, 8) + _field(7, graph) + _field(8, _field(1, "") + _field(2, 13)))
    return path


def test_the_thread_cap_follows_the_trainer_only_under_a_cpu_quota(monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 24)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(get_num_threads=lambda: 11))
    assert facesim.cpu_thread_cap(11.5) == 11  # the trainer's cap (trainer.cap_cpu_threads)
    monkeypatch.setattr("rulztrainer.machine.cpu_quota", lambda: None)
    assert facesim.cpu_thread_cap() is None  # no quota: onnxruntime's default is PyTorch's, nothing to change
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(get_num_threads=lambda: 24))
    assert facesim.cpu_thread_cap(11.5) == 11  # a process that did not cap PyTorch (the WebUI): the quota
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(get_num_threads=lambda: 6))
    assert facesim.cpu_thread_cap(11.5) == 6  # PyTorch held lower still (OMP_NUM_THREADS): that
    monkeypatch.delitem(sys.modules, "torch")
    assert facesim.cpu_thread_cap(11.5) == 11  # without PyTorch loaded: the quota
    assert facesim.cpu_thread_cap(30.0) is None  # a quota above the CPUs shown: nothing to change
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(get_num_threads=lambda: 1 / 0))
    assert facesim.cpu_thread_cap(11.5) is None  # never a reason for the scorer to fail


def test_cpu_sessions_are_rebuilt_with_the_thread_cap(tmp_path, monkeypatch):
    ort = pytest.importorskip("onnxruntime")
    np = pytest.importorskip("numpy")
    path = _identity_onnx(tmp_path / "det.onnx")
    det = SimpleNamespace(session=ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]))
    gpu = SimpleNamespace(session=SimpleNamespace(get_providers=lambda: ["CUDAExecutionProvider",
                                                                         "CPUExecutionProvider"]))
    broken = SimpleNamespace(session=None)
    set_threads = []
    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(setNumThreads=set_threads.append))
    app = SimpleNamespace(models={"detection": det, "recognition": gpu, "other": broken})
    keep_gpu = gpu.session
    assert facesim.limit_threads(app, 3) == 1 and set_threads == [3]
    assert det.session.get_session_options().intra_op_num_threads == 3
    assert det.session.get_providers() == ["CPUExecutionProvider"]
    assert det.session.run(None, {"x": np.arange(4, dtype=np.float32)})[0].tolist() == [0.0, 1.0, 2.0, 3.0]
    assert gpu.session is keep_gpu and broken.session is None  # a GPU session and one without a session: as they were
    again = det.session
    assert facesim.limit_threads(app, 4) == 0 and det.session is again  # already within the cap: not reloaded


def test_the_scorer_caps_its_threads_and_never_fails_for_it(tmp_path, monkeypatch):
    """FaceScorer with a stand-in for insightface (not installed here): its sessions are rebuilt under a quota, and
    an insightface whose models look otherwise still gives a scorer."""
    ort = pytest.importorskip("onnxruntime")
    path = _identity_onnx(tmp_path / "rec.onnx")
    models_of = {"good": lambda: {"recognition": SimpleNamespace(
        session=ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]))},
        "odd": lambda: ["not", "a dict"]}

    class FaceAnalysis:
        kind = "good"

        def __init__(self, name, providers, allowed_modules):
            self.models = models_of[FaceAnalysis.kind]()

        def prepare(self, ctx_id, det_size):
            self.ctx = ctx_id

    app_mod = types.ModuleType("insightface.app")
    app_mod.FaceAnalysis = FaceAnalysis
    pkg = types.ModuleType("insightface")
    pkg.app = app_mod
    monkeypatch.setitem(sys.modules, "insightface", pkg)
    monkeypatch.setitem(sys.modules, "insightface.app", app_mod)
    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(setNumThreads=lambda n: None))
    monkeypatch.setattr("rulztrainer.downloads.ensure_insightface_pack", lambda *a, **k: None)
    monkeypatch.setattr(facesim, "cpu_thread_cap", lambda: 2)
    scorer = facesim.FaceScorer("cpu")
    assert scorer.app.models["recognition"].session.get_session_options().intra_op_num_threads == 2
    FaceAnalysis.kind = "odd"
    assert facesim.FaceScorer("cpu").app.models == ["not", "a dict"]
    monkeypatch.setattr(facesim, "cpu_thread_cap", lambda: None)
    FaceAnalysis.kind = "good"
    plain = facesim.FaceScorer("cpu")  # no quota: the sessions stay as insightface made them
    assert plain.app.models["recognition"].session.get_session_options().intra_op_num_threads == 0


# ---------------------------------------------------------------- history files written by two processes


STATUS = {"step": 600, "total": 2000, "face_scores": [], "probes": [
    {"step": 0, "face_norm_ema": 0.0, "body_norm_ema": 0.0, "combined_ema": 0.0},
    {"step": 25, "face_norm_ema": 12.3, "body_norm_ema": 5.0, "combined_ema": 9.0}]}


def test_a_label_set_while_the_pictures_are_copied_is_kept(tmp_path, monkeypatch):
    """save_run read the label, copied the probe pictures (slow), then wrote: a label the WebUI set meanwhile was
    written over with the one read before it."""
    cfg = TrainConfig(job_name="job")
    history.save_run(cfg, "job-1", 1.0, STATUS, root=tmp_path)

    def copying(run_id, status, root):
        assert history.set_label(run_id, "named during the copy", root)  # the WebUI, in between
        return []

    monkeypatch.setattr(history, "keep_probe_images", copying)
    history.save_run(cfg, "job-1", 1.0, {**STATUS, "step": 900}, root=tmp_path)
    (run,) = history.list_runs(tmp_path)
    assert run["label"] == "named during the copy" and run["step"] == 900


def test_each_history_write_has_its_own_temporary_file_and_retries_the_replace(tmp_path, monkeypatch):
    real = os.replace
    seen, refusals = [], [2]

    def replace(src, dst):
        seen.append(Path(src).name)
        if refusals[0]:
            refusals[0] -= 1
            raise PermissionError("a reader holds it (Windows)")
        real(src, dst)

    monkeypatch.setattr(os, "replace", replace)
    monkeypatch.setattr(history.time, "sleep", lambda s: None)
    path = tmp_path / "run-1.json"
    history._write(path, {"id": "run-1", "label": "a"})
    assert json.loads(path.read_text(encoding="utf-8"))["label"] == "a" and len(seen) == 3
    history._write(path, {"id": "run-1", "label": "b"})
    names = set(seen)
    assert len(names) == 2 and "run-1.tmp" not in names and all(not n.endswith(".json") for n in names)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["run-1.json"]  # nothing left behind
    refusals[0] = history.REPLACE_TRIES
    with pytest.raises(PermissionError):
        history._write(path, {"id": "run-1", "label": "c"})
    assert json.loads(path.read_text(encoding="utf-8"))["label"] == "b"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["run-1.json"]


def test_the_queue_retries_a_refused_replace(tmp_path, monkeypatch):
    real = os.replace
    refusals = [3]

    def replace(src, dst):
        if refusals[0]:
            refusals[0] -= 1
            raise PermissionError("a reader holds it (Windows)")
        real(src, dst)

    monkeypatch.setattr(os, "replace", replace)
    monkeypatch.setattr(train_queue.time, "sleep", lambda s: None)
    q = train_queue.TrainQueue(tmp_path / "_queue.json")
    entry = q.add({"job_name": "a"})
    assert [e["id"] for e in q.snapshot()["waiting"]] == [entry["id"]] and refusals == [0]
    refusals[0] = train_queue.REPLACE_TRIES
    with pytest.raises(PermissionError):
        q.set_paused(True)
    assert q.snapshot()["paused"] is False and not (tmp_path / "_queue.tmp").exists()


# ---------------------------------------------------------------- resume state written whole


class Killed(BaseException):
    """The process dying halfway through a write."""


def _torn(monkeypatch):
    """Path.write_text writes half its text and dies."""
    def write_text(self, data, encoding=None, errors=None, newline=None):
        with open(self, "w", encoding=encoding) as f:
            f.write(data[:len(data) // 2])
        raise Killed

    monkeypatch.setattr(Path, "write_text", write_text)


def _items(tmp_path, n):
    return [Item(image=tmp_path / f"img{i}.png", caption=f"cap {i}", width=512, height=512,
                 latent_path=tmp_path / "l", latent_flip_path=None, text_path=tmp_path / "t") for i in range(n)]


def test_region_watch_state_survives_a_kill_mid_write(tmp_path, monkeypatch):
    watch = regionwatch.RegionWatch(_items(tmp_path, 3), tmp_path)
    watch.epoch = 4
    watch._save()
    before = (tmp_path / regionwatch.STATE_FILE).read_text(encoding="utf-8")
    watch.epoch = 5
    _torn(monkeypatch)
    with pytest.raises(Killed):
        watch._save()
    monkeypatch.undo()
    assert (tmp_path / regionwatch.STATE_FILE).read_text(encoding="utf-8") == before
    assert json.loads(before)["epoch"] == 4 and sorted(p.name for p in tmp_path.iterdir()) == [regionwatch.STATE_FILE]


def test_region_sweep_state_survives_a_kill_mid_write(tmp_path, monkeypatch):
    probes = [{"key": "train0.png", "held": False, "parts": None, "load": None}]
    sweep = regionsweep.RegionSweep(tmp_path, probes, every=10)
    sweep.base = {"train0.png": {"face": 1.0}}
    sweep._save()
    state = tmp_path / regionsweep.STATE_FILE
    before = state.read_text(encoding="utf-8")
    sweep.base = {"train0.png": {"face": 2.0}}
    _torn(monkeypatch)
    with pytest.raises(Killed):
        sweep._save()
    monkeypatch.undo()
    assert state.read_text(encoding="utf-8") == before and not state.with_name(state.name + ".tmp").exists()


def test_self_training_rounds_survive_a_kill_mid_write(tmp_path, monkeypatch):
    path = tmp_path / "selftrain.json"
    keep = tmp_path / "kept.png"
    keep.write_bytes(b"png")
    cand = selftrain.Candidate(str(keep), "a prompt", 50.0, faces=1, det=0.9, sharp=0.01, verdict="kept")
    selftrain.record(path, 1, 100, 40.0, [cand])
    _torn(monkeypatch)
    with pytest.raises(Killed):
        selftrain.record(path, 2, 200, 40.0, [cand])
    monkeypatch.undo()
    assert [r["round"] for r in json.loads(path.read_text(encoding="utf-8"))] == [1]  # round 1 is still there
    assert selftrain.last_kept(path, 300) == (1, [(str(keep), "a prompt")])
    assert sorted(p.name for p in tmp_path.iterdir()) == ["kept.png", "selftrain.json"]
