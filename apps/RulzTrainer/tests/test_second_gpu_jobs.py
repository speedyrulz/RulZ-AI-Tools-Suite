"""probe_async / sample_async / maps_async / sweep_async: samples, region sweeps and training maps on the second GPU. The schedule and recording split (begin / record),
the process's job queue and its answers (a fake connection), what the worker computes (a toy model, no GPU), and the
trainer queueing, recording late, saving the held-out best from the swept weights, waiting, and falling back (the
process faked: no model, no GPU)."""
import contextlib
import io
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rulztrainer import lora as L  # noqa: E402
from rulztrainer import probeworker as PW  # noqa: E402
from rulztrainer import regionsweep as RS  # noqa: E402
from rulztrainer import trainmaps as TM  # noqa: E402
from rulztrainer.config import TrainConfig  # noqa: E402
from test_probe_async import _lora  # noqa: E402
from test_regionsweep import Toy, _probes as sweep_probes  # noqa: E402
from test_trainmaps import ToyModel, _probes as map_probes  # noqa: E402


def _cfg(**kw):
    kw.setdefault("dataset_dir", "ds")
    return TrainConfig(**kw)


# ---------------------------------------------------------------- the setting


def test_the_setting_is_off_by_default_and_round_trips():
    keys = ("probe_async", "sample_async", "maps_async", "sweep_async")
    assert not any(getattr(_cfg(), k) for k in keys) and not hasattr(_cfg(), "analysis_async")
    on = _cfg(sample_async=True, maps_async=True, sweep_async=True)
    on.validate()
    got = TrainConfig.from_dict(on.to_dict())
    assert (got.sample_async, got.maps_async, got.sweep_async, got.probe_async) == (True, True, True, False)


# ---------------------------------------------------------------- the schedule and the recording, split


def test_a_sweep_recorded_later_is_the_sweep_run_now(tmp_path):
    model = Toy()
    here = RS.RegionSweep(tmp_path / "here", sweep_probes(2, 1), every=10)
    (tmp_path / "here").mkdir()
    (tmp_path / "away").mkdir()
    away = RS.RegionSweep(tmp_path / "away", sweep_probes(2, 1), every=10)
    want = here.run(0, 0.0, model, "cpu")
    chosen = away.begin(0)
    assert not away.due(0) and not away.due(9) and away.due(10)  # moved on at queue time: not due while it is out
    readings = RS.forward_readings(model, "cpu", chosen)
    got = away.record(0, 0.0, [p["key"] for p in chosen], readings, 1.5)
    assert {k: v for k, v in got.items() if k != "seconds"} == {k: v for k, v in want.items() if k != "seconds"}
    assert got["seconds"] == 1.5 and away.latest == here.latest and away.history == [got]
    assert len((tmp_path / "away" / RS.LOG_FILE).read_text(encoding="utf-8").splitlines()) == 1
    assert json.loads((tmp_path / "away" / RS.STATE_FILE).read_text(encoding="utf-8"))["base"]


def test_a_failed_sweep_turns_the_sweeps_off(tmp_path):
    sweep = RS.RegionSweep(tmp_path, sweep_probes(1), every=10)
    sweep.fail(0, RuntimeError("boom"))
    assert sweep.failed and not sweep.due(10)


def test_maps_recorded_later_are_the_maps_captured_now(tmp_path):
    here = TM.TrainingMaps(tmp_path / "here", map_probes(), every=10)
    away = TM.TrainingMaps(tmp_path / "away", map_probes(), every=10)
    for m in (here, away):
        m.start(0, resume=False)
    model = ToyModel()
    here.capture(0, 0.0, model, {"l": model.lora}, "cpu")
    assert away.begin(0) is False  # the baseline has no effect to read
    assert not away.due(0) and not away.due(9) and away.due(10)
    out = away._run(model, {"l": model.lora}, "cpu", None, effect=False)
    away.record(0, 0.0, out, 0.5)
    rows = [json.loads(x) for x in (TM.maps_dir(tmp_path / "away") / "maps.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows == [json.loads(x) for x in (TM.maps_dir(tmp_path / "here") / "maps.jsonl").read_text(encoding="utf-8").splitlines()]
    assert away.begin(10) is True  # later captures read the LoRA's effect
    away.fail(10, RuntimeError("boom"))
    assert away.failed and not away.due(99)


# ---------------------------------------------------------------- the process's queue


class FakeConn:
    def __init__(self):
        self.inbox, self.sent = [], []

    def send(self, msg):
        self.sent.append(msg)

    def poll(self, timeout=0):
        return bool(self.inbox)

    def recv(self):
        return self.inbox.pop(0)

    def close(self):
        pass


def _process(analysis=True):
    w = PW.ProbeProcess.__new__(PW.ProbeProcess)
    w.device, w.probes, w.queue, w.waited, w.rendered, w.count = "cuda:1", True, [], 0.0, 0.0, 0
    w.analysed, w.analysis_count, w.analysis_wanted, w.analysis_error = 0.0, 0, analysis, None
    w.conn, w.proc, w.log_path, w.started, w.memory = FakeConn(), SimpleNamespace(poll=lambda: None), "x.log", time.time(), None
    return w


def _state():
    return L.lora_state(_lora(0))


def test_jobs_of_every_kind_queue_and_each_answer_reaches_its_job():
    w = _process()
    w.conn.inbox.append(("ready", "5 GB", None))
    w.submit_analysis("sweep", 10, _state(), keys=["a.png", "b.png"], extra={"epoch": 1.0})
    w.submit_analysis("maps", 10, _state(), extra={"epoch": 1.0, "effect": True})
    w.submit(10, ["job"], _state())
    assert [m[0] for m in w.conn.sent] == ["sweep", "maps", "probe"]
    assert w.conn.sent[0][2] == ["a.png", "b.png"] and w.conn.sent[1][2] is True  # keys; whether to read the effect
    with pytest.raises(RuntimeError, match="still queued"):  # one job of each kind at a time
        w.submit_analysis("sweep", 20, _state(), keys=[])
    with pytest.raises(RuntimeError, match="still queued"):
        w.submit(20, [], _state())
    assert w.take_analysis("sweep", False) is None and w.take(False) is None  # nothing in yet; ready is read on the way
    assert w.memory == "5 GB" and w.analysis_ok and w.analysis_error is None
    w.conn.inbox += [("sweep_done", {"a.png": [1]}, 2.0), ("maps_done", {"0": {}}, 3.0), ("done", ["x.png"], 4.0)]
    p, images = w.take(False)  # the probe is last in line: its answer is read through the other two
    assert (p.step, images) == (10, ["x.png"]) and w.pending is None
    assert w.ready_analysis("sweep") and w.ready_analysis("maps")
    p, readings, why = w.take_analysis("maps", False)
    assert (p.kind, readings, why) == ("maps", {"0": {}}, "") and p.extra["effect"] is True
    p, readings, why = w.take_analysis("sweep", True)
    assert (p.jobs, readings, why) == (["a.png", "b.png"], {"a.png": [1]}, "")
    assert (w.rendered, w.analysed, w.count, w.analysis_count) == (4.0, 5.0, 1, 2) and w.queue == []


def test_waiting_for_a_job_counts_as_waited_and_blocks_until_its_answer():
    w = _process()
    w.memory = "mem"
    w.submit_analysis("sweep", 5, _state(), keys=["a"], extra={})
    w.conn.inbox.append(("sweep_done", {"a": 1}, 1.0))
    assert w.take_analysis("sweep", True)[1] == {"a": 1}
    assert w.waited >= 0.0


def test_a_computation_failure_is_reported_for_that_job_and_a_process_failure_is_raised():
    w = _process()
    w.memory = "mem"
    w.submit_analysis("sweep", 5, _state(), keys=["a"], extra={})
    w.conn.inbox.append(("analysis_failed", "sweep", "RuntimeError: parts do not fit", 0.1))
    p, readings, why = w.take_analysis("sweep", True)
    assert readings is None and "parts do not fit" in why
    w.submit_analysis("maps", 5, _state(), extra={"effect": False})
    w.conn.inbox.append(("error", "Traceback ...\nCUDA out of memory"))
    with pytest.raises(PW.ProbeProcessError, match="out of memory"):
        w.take_analysis("maps", True)
    assert [p.kind for p in w.outstanding()] == ["maps"]  # still out: the trainer reads it again itself
    w.queue = [PW.Pending(1, [], {}, kind="probe")]
    w.conn.inbox.append(("sweep_done", {}, 1.0))
    with pytest.raises(PW.ProbeProcessError, match="answered"):  # an answer of the wrong kind: out of step
        w.take(True)


def test_a_failed_analysis_model_is_reported_by_the_ready_message():
    w = _process()
    w.conn.inbox.append(("ready", "5 GB", "Traceback\nCUDA out of memory"))
    w.wait_ready()
    assert not w.analysis_ok and "out of memory" in w.analysis_error
    assert not _process(analysis=False).analysis_ok


# ---------------------------------------------------------------- what the worker computes


def _renderer(model, injected, sweep=None, maps=None, shared=False):
    r = PW.Renderer.__new__(PW.Renderer)
    r.model, r.train_injected, r.device, r.shared, r.extras = model, injected, "cpu", shared, None
    r.sweep_pictures = {p["key"]: p for p in (sweep or [])}
    r.maps = maps
    return r


def _payload(injected):
    buf = io.BytesIO()
    torch.save(L.lora_state(injected), buf)
    return buf.getvalue()


def test_the_worker_reads_what_the_trainer_would(tmp_path):
    model = Toy()
    pictures = sweep_probes(2, 1)
    keys = [p["key"] for p in pictures]
    maps = TM.TrainingMaps(".", map_probes(), 1)
    injected = _lora(1)
    r = _renderer(model, injected, pictures, maps)
    tag, readings, seconds = r.analyse(("sweep", _payload(injected), keys))
    assert tag == "sweep_done" and readings == RS.forward_readings(model, "cpu", pictures) and seconds >= 0
    # the maps: the LoRA state of the job is mirrored into the worker's own slots before the forward
    src, dst = _lora(3), _lora(4)
    r = _renderer(ToyModel(), dst, maps=TM.TrainingMaps(".", map_probes(), 1))
    r.model.lora = SimpleNamespace(multiplier=1.0)
    seen = {}
    real = r.maps._run

    def run(model, inj, device, averaged, effect):
        seen["same"] = all(torch.equal(p, src[n].adapter_params()[k]) for n, m in inj.items()
                           for k, p in m.adapter_params().items())
        seen["effect"] = effect
        return real(model, inj, device, averaged, effect)

    r.maps._run = run
    tag, out, _ = r.analyse(("maps", _payload(src), True))
    assert tag == "maps_done" and seen == {"same": True, "effect": True} and set(out["0"]) == {"0.25", "0.5", "0.75"}
    assert "effect" not in TM.TrainingMaps(".", map_probes(), 1)._run(ToyModel(), {}, "cpu", None, False)["0"]["0.5"]


def test_a_failing_computation_is_an_answer_not_a_crash():
    model, injected = Toy(), _lora(1)
    r = _renderer(model, injected, sweep_probes(1))
    tag, kind, why, _ = r.analyse(("sweep", _payload(injected), ["not-a-picture.png"]))
    assert (tag, kind) == ("analysis_failed", "sweep") and "KeyError" in why
    r.maps = None
    assert r.analyse(("maps", _payload(injected), False))[0] == "analysis_failed"


def test_the_shared_model_switches_its_sampling_loras_off_for_an_analysis_and_on_for_a_render(monkeypatch):
    calls = []
    monkeypatch.setattr(L, "set_extras_enabled", lambda model, on: calls.append(on))
    r = _renderer(Toy(), _lora(1), sweep_probes(1), shared=True)
    r.slot = SimpleNamespace(model=r.model)
    injected = r.train_injected
    r.analyse(("sweep", _payload(injected), ["train0.png"]))
    r.analyse(("sweep", _payload(injected), ["train0.png"]))  # (already off: not switched again)
    r._set_extras(True)
    assert calls == [False, True]
    unshared = _renderer(Toy(), _lora(1), shared=False)
    unshared._set_extras(False)
    assert calls == [False, True]  # a model of its own has no switch to flip


# ---------------------------------------------------------------- the trainer


class FakeWorker:
    """ProbeProcess's interface for sweeps and maps; an answer is in when it is put in `done`."""

    def __init__(self, analysis_error=None, ready=True):
        self.device, self.queue, self.done, self.fail_with = "cuda:1", [], {}, None
        self.memory = "5 GB" if ready else None
        self.analysis_wanted, self.analysis_error = True, analysis_error
        self.count = self.analysis_count = 0
        self.rendered = self.analysed = self.waited = 0.0
        self.closed = False
        self.waits = []

    @property
    def analysis_ok(self):
        return self.memory is not None and self.analysis_error is None

    def wait_ready(self):
        if self.fail_with:
            raise PW.ProbeProcessError(self.fail_with)
        self.memory = "5 GB"

    def pending_of(self, kind):
        return next((p for p in self.queue if p.kind == kind), None)

    def submit(self, step, jobs, state, kind="probe"):
        assert self.pending_of(kind) is None
        self.queue.append(PW.Pending(step, list(jobs), state, kind=kind))

    def take_sample(self, wait):
        p = self.pending_of("sample")
        if p is None or not (wait or "sample" in self.done):
            return None
        self.waits.append(("sample", wait))
        if self.fail_with and wait:
            raise PW.ProbeProcessError(self.fail_with)
        self.queue.remove(p)
        return p, self.done.pop("sample")

    def submit_analysis(self, kind, step, state, keys=None, extra=None):
        assert self.pending_of(kind) is None
        self.queue.append(PW.Pending(step, list(keys or []), state, kind=kind, extra=dict(extra or {})))

    def ready_analysis(self, kind):
        return kind in self.done

    def take_analysis(self, kind, wait):
        p = self.pending_of(kind)
        if p is None or not (wait or kind in self.done):
            return None
        self.waits.append((kind, wait))
        if self.fail_with and wait:
            raise PW.ProbeProcessError(self.fail_with)
        self.queue.remove(p)
        result = self.done.pop(kind)
        p.reply = (f"{kind}_done", result, 2.0)
        if isinstance(result, str):
            return p, None, result
        self.analysis_count += 1
        return p, result, ""

    def outstanding(self):
        return [p for p in self.queue if p.kind != "probe"]

    def close(self):
        self.closed = True


def _trainer(tmp_path, **cfg):
    from rulztrainer import trainer as T

    tr = T.Trainer.__new__(T.Trainer)
    cfg.setdefault("maps_async", True)
    cfg.setdefault("sweep_async", True)
    tr.cfg = _cfg(holdout_track_best=True, **cfg)
    tr.controller, tr.race_rows, tr.ft, tr.ema, tr.device = None, None, None, None, "cpu"
    tr.job_dir = tmp_path
    status = {}
    tr.status = SimpleNamespace(update=lambda **k: status.update(k), data={"saved": []})
    tr.region_watch, tr.holdout_stop = None, False
    tr.injected = _lora(0)
    tr.replicas = [(Toy(), tr.injected, "cpu")]
    tr._maps_epoch_steps = lambda: (10, 1)
    tr._free_for_rendering = lambda: None
    tr.sweep = RS.RegionSweep(tmp_path, sweep_probes(2, 2), every=10, patience=tr.cfg.holdout_patience)
    maps_dir = tmp_path / "m"
    maps_dir.mkdir(parents=True, exist_ok=True)
    tr.maps = TM.TrainingMaps(maps_dir, map_probes(), every=10)
    tr.maps.start(0, resume=False)
    tr.analysis_worker = tr.probe_worker = tr.sample_worker = None
    tr.saved_states = []

    def save_best(injected, step, score, averaged=True):  # what a best save writes: the weights in `injected` now
        tr.saved_states.append((step, averaged, L.lora_state(injected)))

    tr._save_holdout_best = save_best
    return T, tr, status


def _bump(injected, by):
    with torch.no_grad():
        for m in injected.values():
            for p in m.adapter_params().values():
                p.add_(by)


def _same(a, b):
    return all(torch.equal(a[n]["params"][k], b[n]["params"][k]) for n in a for k in a[n]["params"])


def test_a_sweep_is_queued_with_the_averaged_lora_and_not_due_again(tmp_path):
    T, tr, status = _trainer(tmp_path)
    tr.analysis_worker = w = FakeWorker()
    raw = L.lora_state(tr.injected)

    @contextlib.contextmanager
    def averaged(replicas=False):  # the weight average: other weights inside, the raw ones back after
        before = L.lora_state(tr.injected)
        _bump(tr.injected, 0.25)
        try:
            yield
        finally:
            L.mirror_lora(L.lora_snapshots(before), tr.injected)

    tr._averaged = averaged
    assert tr.sweep.due(0)
    tr._run_sweep(tr.replicas[0][0], tr.injected, 0)  # (queued: nothing is read here)
    p = w.pending_of("sweep")
    assert p is not None and p.step == 0 and sorted(p.jobs) == sorted(x["key"] for x in tr.sweep.probes) and p.extra["epoch"] == 0.0
    assert not _same(p.state, raw) and all(torch.allclose(p.state[n]["params"][k], raw[n]["params"][k] + 0.25)
                                           for n in raw for k in raw[n]["params"])  # the average, not the raw weights
    assert _same(L.lora_state(tr.injected), raw) and not tr.sweep.due(0) and not tr.sweep.due(9)
    assert tr.sweep.history == [] and "sweep of step 0 on cuda:1" in status["stage"]
    tr._capture_maps(None, tr.injected, 0)
    assert w.pending_of("maps").extra == {"epoch": 0.0, "effect": False} and not tr.maps.due(5) and tr.maps.last_step == 0


def test_a_result_is_recorded_when_it_comes_back_and_a_best_saves_the_swept_weights(tmp_path):
    T, tr, status = _trainer(tmp_path, holdout_patience=2)
    tr.analysis_worker = w = FakeWorker()
    model = tr.replicas[0][0]
    swept = L.lora_state(tr.injected)
    for step, gain in ((0, 0.0), (10, 0.5)):
        with torch.no_grad():
            model.gain[:] = gain
        tr._run_sweep(model, tr.injected, step)
        p = w.pending_of("sweep")
        w.done["sweep"] = RS.forward_readings(model, "cpu", [pr for pr in tr.sweep.probes if pr["key"] in p.jobs])
        if step == 10:
            swept = p.state
        _bump(tr.injected, 1.0)  # training goes on
        live = L.lora_state(tr.injected)
        tr._analysis_take_all(step + 3)  # (recorded three steps later)
        assert _same(L.lora_state(tr.injected), live)  # training's weights are back
    assert [r["step"] for r in tr.sweep.history] == [0, 10] and tr.sweep.history[0]["first"]
    assert tr.sweep.history[1]["new_best"] and len(tr.saved_states) == 1
    step, averaged, saved = tr.saved_states[0]
    assert step == 10 and averaged is False and _same(saved, swept) and not _same(saved, live)  # the state swept
    assert status["region_sweep"]["step"] == 10 and not tr.holdout_stop


def test_a_sweep_that_acts_late_can_still_stop_the_run(tmp_path):
    T, tr, _ = _trainer(tmp_path, holdout_patience=1)
    tr.analysis_worker = w = FakeWorker()
    model = tr.replicas[0][0]
    for step, gain in ((0, 0.0), (10, 0.5), (20, 0.5)):  # the third sweep is no better
        with torch.no_grad():
            model.gain[:] = gain
        tr._run_sweep(model, tr.injected, step)
        w.done["sweep"] = RS.forward_readings(model, "cpu", tr.sweep.probes)
        tr._analysis_take("sweep", step + 1)
    assert tr.holdout_stop and tr.sweep.best[1] == 10


def test_the_next_one_waits_for_the_last_to_come_in(tmp_path):
    T, tr, _ = _trainer(tmp_path)
    tr.analysis_worker = w = FakeWorker()
    model = tr.replicas[0][0]
    tr._run_sweep(model, tr.injected, 0)
    assert w.pending_of("sweep").step == 0
    readings = RS.forward_readings(model, "cpu", tr.sweep.probes)
    w.done["sweep"] = readings  # (it came in; the take below blocks for it)
    tr._run_sweep(model, tr.injected, 10)
    assert ("sweep", True) in w.waits and [r["step"] for r in tr.sweep.history] == [0]  # recorded before the next goes
    assert w.pending_of("sweep").step == 10 and len(w.queue) == 1


def test_at_the_end_everything_still_out_is_waited_for_and_recorded(tmp_path):
    T, tr, _ = _trainer(tmp_path)
    tr.analysis_worker = w = FakeWorker()
    model, mm = tr.replicas[0][0], ToyModel()
    tr._run_sweep(model, tr.injected, 30)
    tr._capture_maps(mm, tr.injected, 30)
    w.done["sweep"] = RS.forward_readings(model, "cpu", tr.sweep.probes)
    w.done["maps"] = tr.maps._run(mm, {"l": mm.lora}, "cpu", None, True)
    tr._analysis_take_all(30, wait=True)
    assert [r["step"] for r in tr.sweep.history] == [30] and w.queue == []
    assert [json.loads(x)["step"] for x in (TM.maps_dir(tmp_path / "m") / "maps.jsonl").read_text(encoding="utf-8").splitlines()] == [30]


def test_a_computation_failure_turns_that_analysis_off_and_the_process_stays(tmp_path):
    T, tr, _ = _trainer(tmp_path)
    tr.analysis_worker = w = FakeWorker()
    tr._run_sweep(tr.replicas[0][0], tr.injected, 0)
    w.done["sweep"] = "RuntimeError: the parts map does not fit"
    tr._analysis_take("sweep", 1, wait=True)
    assert tr.sweep.failed and tr.analysis_worker is w and not w.closed
    tr._capture_maps(None, tr.injected, 0)  # the maps still run on the second GPU
    assert w.pending_of("maps") is not None


def test_a_process_that_fails_hands_everything_back_and_the_lost_sweep_is_read_here(tmp_path):
    T, tr, status = _trainer(tmp_path)
    tr.analysis_worker = w = FakeWorker()
    model = tr.replicas[0][0]
    queued = L.lora_state(tr.injected)
    tr._run_sweep(model, tr.injected, 0)
    _bump(tr.injected, 1.0)  # training went on
    live = L.lora_state(tr.injected)
    w.fail_with = "CUDA out of memory"
    tr._analysis_take("sweep", 2, wait=True)
    assert tr.analysis_worker is None and tr.probe_worker is None and w.closed
    assert [r["step"] for r in tr.sweep.history] == [0] and tr.sweep.history[0]["first"]  # read again, here
    assert _same(L.lora_state(tr.injected), live) and not _same(live, queued)  # training's weights are back
    assert tr._analysis_start("sweep", 10, tr.injected) is False  # from now on they run on the training GPU


def test_a_lost_sweep_is_not_read_again_when_its_answer_had_come(tmp_path):
    T, tr, _ = _trainer(tmp_path)
    tr.analysis_worker = w = FakeWorker()
    model = tr.replicas[0][0]
    tr._run_sweep(model, tr.injected, 0)
    w.queue[0].reply = ("sweep_done", RS.forward_readings(model, "cpu", tr.sweep.probes), 1.0)
    model.calls = 0
    tr._close_probe_worker(redo=True)
    assert model.calls == 0 and [r["step"] for r in tr.sweep.history] == [0]
    tr._close_probe_worker(redo=True)  # (a second close is nothing)


def test_an_analysis_model_that_did_not_load_sends_sweeps_and_maps_back_to_the_training_gpu(tmp_path):
    T, tr, _ = _trainer(tmp_path)
    tr.analysis_worker = w = FakeWorker(analysis_error="Traceback\ntorch.OutOfMemoryError: CUDA out of memory", ready=False)
    tr.probe_worker = w  # (the process renders probes too: it stays for them)
    assert tr._analysis_start("sweep", 0, tr.injected) is False
    assert tr.analysis_worker is None and tr.probe_worker is w and not w.closed and w.queue == []
    assert tr.sweep.due(0)  # nothing was queued: the sweep still runs here
    tr.probe_worker = None
    tr.analysis_worker = w2 = FakeWorker(analysis_error="out of memory")  # analysis only: nothing else for it to do
    assert tr._analysis_start("maps", 0, tr.injected) is False and w2.closed


def test_a_process_that_could_not_load_at_all_hands_everything_back(tmp_path):
    T, tr, _ = _trainer(tmp_path)
    tr.analysis_worker = w = FakeWorker(ready=False)
    w.fail_with = "CUDA out of memory"
    assert tr._analysis_start("sweep", 0, tr.injected) is False
    assert tr.analysis_worker is None and w.closed


def test_with_the_setting_off_everything_runs_here_as_before(tmp_path):
    T, tr, status = _trainer(tmp_path, maps_async=False, sweep_async=False)
    assert tr._worker_wants() == (False, False, False, False) and tr._probe_async_device() == ""
    assert tr._analysis_start("sweep", 0, tr.injected) is False and tr._analysis_start("maps", 0, tr.injected) is False
    tr._averaged = lambda replicas=False: contextlib.nullcontext()
    tr._run_sweep(tr.replicas[0][0], tr.injected, 0)  # the old path: read here, recorded at once
    assert [r["step"] for r in tr.sweep.history] == [0] and tr.sweep.last_step == 0
    assert "region_sweep" in status


def test_when_the_second_gpu_is_used(tmp_path, monkeypatch):
    T, tr, _ = _trainer(tmp_path)
    tr.cfg.device = "cuda:0"
    tr.device = "cuda:0"
    monkeypatch.setattr(T.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(T.torch.cuda, "device_count", lambda: 2)
    assert tr._worker_wants() == (False, False, True, True) and tr._probe_async_device() == "cuda:1"  # no probe_async
    tr.cfg.probe_async, tr.controller, tr.cfg.probe_every = True, object(), 10
    assert tr._worker_wants() == (True, False, True, True)
    tr.maps = tr.sweep = None  # nothing to read: only the probes
    assert tr._worker_wants() == (True, False, False, False)
    tr.cfg.probe_async = False
    assert tr._probe_async_device() == ""
    T2, tr, _ = _trainer(tmp_path / "again")
    tr.device = "cuda:0"
    for change in ({"race_rows": [("a", None, {})]}, {"ft": object()}):  # the same exclusions as probe_async
        for k, v in change.items():
            setattr(tr, k, v)
        assert tr._probe_async_device() == ""
        tr.race_rows, tr.ft = None, None
    tr.cfg.multi_gpu = "data_parallel"
    assert tr._probe_async_device() == ""
    tr.cfg.multi_gpu = "off"
    monkeypatch.setattr(T.torch.cuda, "device_count", lambda: 1)  # one GPU: here, as before
    assert tr._probe_async_device() == ""


def test_the_second_gpu_gets_the_pictures_as_plain_data(tmp_path):
    T, tr, _ = _trainer(tmp_path)
    for p in tr.sweep.probes:
        p["latent_path"], p["text_path"] = f"{p['key']}.latent", f"{p['key']}.text"
    tr.maps.probes[0]["off"] = {"embeds": torch.zeros(1, 5, 4), "mask": torch.ones(1, 5, dtype=torch.bool)}
    data = tr._analysis_data()
    assert [d["key"] for d in data["sweep"]] == [p["key"] for p in tr.sweep.probes]
    assert data["sweep"][0]["latent_path"].endswith(".latent") and "load" not in data["sweep"][0]
    assert data["sweep"][0]["parts"].shape == tr.sweep.probes[0]["parts"].shape
    m = data["maps"]
    assert m["sigmas"] == list(tr.maps.sigmas) and m["seed"] == tr.maps.seed
    assert set(m["probes"][0]) == {"id", "latents", "embeds", "mask", "trigger", "off"} and "item" not in m["probes"][0]
    assert m["probes"][0]["off"]["embeds"].shape == (1, 5, 4)
    import pickle

    pickle.loads(pickle.dumps(data))  # (our own data, round-tripped here: it goes over the process's connection)
    tr.maps = None
    assert tr._analysis_data()["maps"] is None


# ---------------------------------------------------------------- the worker's models


def _loading_shell(tmp_path, shared):
    from safetensors.torch import save_file

    from rulztrainer.moss.model import MOSSImageTransformer
    from test_probe_async import TINY

    built, attached = [], []

    class Arch:
        def get_transformer(self, cfg, mdir, device=None):
            built.append(device)
            return MOSSImageTransformer(**TINY)

        def pad_embeds(self, embeds):
            e = embeds[0][None]
            return e, torch.ones(e.shape[:2], dtype=torch.bool)

    cfg = _cfg(lora_rank=4, lora_alpha=4.0, lora_targets=["q", "v"], lora_dropout=0.1)
    tr = SimpleNamespace(cfg=cfg, arch=Arch(),
                         _lora_names=lambda m, blocks: [n for n, _ in m.named_modules() if n.endswith(("proj.q", "proj.v"))],
                         _attach_training_adapter=lambda m, log_it=False: attached.append(m))
    r = PW.Renderer.__new__(PW.Renderer)
    r.tr, r.mdir, r.device, r.shared, r.extras = tr, None, "cpu", shared, None
    r.slot = r.injected = r.model = r.train_injected = r.maps = None
    r.sweep_pictures = {}
    if shared:  # the one model the probes render with
        m0 = MOSSImageTransformer(**TINY)
        r.injected = L.inject_lora(m0, 4, 4.0, ["q", "v"], names=tr._lora_names(m0, None))
        r.slot = SimpleNamespace(model=m0)
    save_file({"latent": torch.zeros(3, 4, 8)}, str(tmp_path / "a.latent"))
    save_file({"embeds": torch.zeros(2, 4)}, str(tmp_path / "a.text"))
    analysis = {"sweep": [{"key": "a.png", "held": False, "latent_path": str(tmp_path / "a.latent"),
                           "text_path": str(tmp_path / "a.text"), "parts": torch.zeros(7, 4, 8, dtype=torch.uint8)}],
                "maps": {"probes": map_probes(), "sigmas": [0.25, 0.5], "seed": 99}}
    return r, built, attached, analysis


def test_the_worker_builds_a_second_model_when_the_sampling_model_is_another(tmp_path):
    r, built, attached, analysis = _loading_shell(tmp_path, shared=False)
    r._load_analysis(analysis)
    assert built == ["cpu"] and attached == [r.model]  # the training base, the training adapter attached
    assert not r.model.training and set(r.train_injected) == {n for n, _ in r.model.named_modules()
                                                              if n.endswith(("proj.q", "proj.v"))}
    assert all(m.rank == 4 and getattr(m.dropout, "p", 0.0) == 0.1 for m in r.train_injected.values())
    latents, embeds, mask = r.sweep_pictures["a.png"]["load"]()  # rebuilt from the paths
    assert latents.shape == (1, 3, 4, 8) and embeds.shape == (1, 2, 4) and mask.shape == (1, 2)
    assert r.maps.sigmas == (0.25, 0.5) and r.maps.seed == 99 and len(r.maps.probes) == 1


def test_the_worker_uses_the_one_model_for_both_when_there_is_no_sampling_model(tmp_path):
    r, built, attached, analysis = _loading_shell(tmp_path, shared=True)
    r._load_analysis(analysis)
    assert built == [] and r.model is r.slot.model and r.train_injected is r.injected  # nothing more to load
    assert attached == [r.model]  # (the training adapter follows the extras' switch: off while they render)


# ---------------------------------------------------------------- review fixes


def test_an_error_while_waiting_for_an_answer_falls_back(tmp_path):
    T, tr, _ = _trainer(tmp_path)
    tr.analysis_worker = w = FakeWorker()
    tr._run_sweep(tr.replicas[0][0], tr.injected, 0)

    def boom(kind):
        raise PW.ProbeProcessError("CUDA out of memory")

    w.ready_analysis = boom  # the check before the take pumps replies: its error must be caught too
    tr._analysis_take("sweep", 1, wait=True)
    assert tr.analysis_worker is None and w.closed and [r["step"] for r in tr.sweep.history] == [0]
    # the same for a probe
    tr.probe_worker = pw = SimpleNamespace(pending=SimpleNamespace(step=5), device="cuda:1", closed=False,
                                           ready=boom, take=lambda wait: None, count=0, queue=[],
                                           close=lambda: None)
    pw.ready = lambda: (_ for _ in ()).throw(PW.ProbeProcessError("gone"))
    assert tr._probe_async_take(6, None, None, wait=True) is False and tr.probe_worker is None


class BrokenConn(FakeConn):
    def send(self, msg):
        raise BrokenPipeError("pipe")

    def recv(self):
        raise EOFError


def test_a_closed_connection_is_a_process_error_and_the_job_runs_here(tmp_path):
    w = _process()
    w.conn = BrokenConn()
    w.memory = "mem"
    with pytest.raises(PW.ProbeProcessError, match="closed its connection"):
        w.submit_analysis("sweep", 1, _state(), keys=["a"])
    with pytest.raises(PW.ProbeProcessError, match="closed its connection"):
        w.submit(1, [], _state())
    w.conn.inbox = [1]
    w.queue = [PW.Pending(1, [], {}, kind="sweep")]
    with pytest.raises(PW.ProbeProcessError, match="closed its connection"):
        w._recv()
    # the trainer: begin() ran before the send failed; the schedule and the rotation cursor go back
    T, tr, _ = _trainer(tmp_path)
    tr.sweep = RS.RegionSweep(tmp_path, sweep_probes(6, 1), every=10, rotate=2)
    tr.sweep.run(0, 0.0, Toy(), "cpu")
    before = (tr.sweep.cursor, tr.sweep.last_step, tr.sweep.next_step)
    fw = FakeWorker()

    def fail(*a, **k):
        raise PW.ProbeProcessError("closed its connection")

    fw.submit_analysis = fail
    tr.analysis_worker = fw
    assert tr._analysis_start("sweep", 10, tr.injected) is False
    assert (tr.sweep.cursor, tr.sweep.last_step, tr.sweep.next_step) == before and tr.analysis_worker is None
    assert fw.closed and tr.sweep.due(10)  # the local run is still due, and takes the same pictures


def test_a_probe_lost_with_the_process_is_logged(tmp_path, caplog):
    T, tr, _ = _trainer(tmp_path)
    tr.probe_worker = tr.analysis_worker = w = FakeWorker()
    w.queue.append(PW.Pending(40, [], {}, kind="probe"))
    with caplog.at_level("WARNING", logger="rulztrainer"):
        tr._close_probe_worker(redo=True)
    assert any("probe of step 40" in r.message and "lost" in r.message for r in caplog.records)


def test_close_reads_what_the_worker_is_still_sending():
    w = _process()
    w.log_file = SimpleNamespace(close=lambda: None)
    state = {"polls": 0}
    w.conn.inbox = ["big reply 1", "big reply 2"]
    exited = []

    def poll():  # the process only exits once its replies are drained
        return None if w.conn.inbox else 0

    w.proc = SimpleNamespace(poll=poll, wait=lambda timeout=None: None, kill=lambda: exited.append("killed"))
    w.close()
    assert w.conn.inbox == [] and w.conn.sent[-1] == ("stop",) and exited == [] and state["polls"] == 0
    w2 = _process()
    w2.log_file = SimpleNamespace(close=lambda: None)
    w2.conn = BrokenConn()
    w2.proc = SimpleNamespace(poll=lambda: None, wait=lambda timeout=None: None, kill=lambda: exited.append("killed"))
    w2.close()  # a connection already gone: killed, no exception
    assert exited == ["killed"]


# ---------------------------------------------------------------- one switch per job


def _switches(tmp_path, monkeypatch, **cfg):
    T, tr, _ = _trainer(tmp_path, **cfg)
    tr.cfg.device = tr.device = "cuda:0"
    monkeypatch.setattr(T.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(T.torch.cuda, "device_count", lambda: 2)
    tr.controller, tr.cfg.probe_every = object(), 10
    tr.cfg.sample_prompts, tr.cfg.sample_every = ["a photo"], 10
    return T, tr


KEYS = ("probe_async", "sample_async", "maps_async", "sweep_async")


@pytest.mark.parametrize("on", KEYS)
def test_each_switch_alone_wants_only_its_own_job(tmp_path, monkeypatch, on):
    T, tr = _switches(tmp_path, monkeypatch, **{k: k == on for k in KEYS})
    assert tr._worker_wants() == tuple(k == on for k in KEYS)
    assert tr._probe_async_device() == "cuda:1"


def test_probes_and_sweeps_without_maps_leave_the_maps_on_the_training_gpu(tmp_path, monkeypatch):
    T, tr = _switches(tmp_path, monkeypatch, probe_async=True, sweep_async=True, maps_async=False)
    assert tr._worker_wants() == (True, False, False, True)
    for q in tr.sweep.probes:
        q["latent_path"] = q["text_path"] = "x"
    data = tr._analysis_data(False, True)
    assert data["maps"] is None and data["sweep"] is not None
    tr.analysis_worker = tr.probe_worker = w = FakeWorker()
    assert tr._analysis_start("maps", 0, tr.injected) is False and w.queue == [] and tr.maps.due(0)
    tr._run_sweep(tr.replicas[0][0], tr.injected, 0)
    assert w.pending_of("sweep") is not None  # the sweep did go to the worker
    tr._averaged = lambda replicas=False: contextlib.nullcontext()
    tr._capture_maps(ToyModel(), tr.injected, 0)  # here, on the training GPU: nothing is queued for it
    assert w.pending_of("maps") is None and tr.maps.last_step == 0


def test_samples_alone_load_no_training_model_and_all_on_wants_everything(tmp_path, monkeypatch):
    T, tr = _switches(tmp_path, monkeypatch, sample_async=True, maps_async=False, sweep_async=False)
    assert tr._worker_wants() == (False, True, False, False)
    assert tr._analysis_data(False, False) == {"sweep": None, "maps": None}
    T, tr = _switches(tmp_path / "all", monkeypatch, probe_async=True, sample_async=True)
    assert tr._worker_wants() == (True, True, True, True)
    tr.cfg.sample_prompts = []  # nothing to sample: that switch has no job
    assert tr._worker_wants()[1] is False


def test_a_sample_round_is_queued_with_the_averaged_lora_and_finished_when_it_is_in(tmp_path):
    T, tr, status = _trainer(tmp_path, sample_async=True)
    tr.sample_worker = w = FakeWorker()
    tr.remote = None
    tr._sample_jobs = lambda step, n=1: ["job0.png"]
    finished = []
    tr._sample_finish = lambda paths, step: finished.append((step, paths))
    tr._averaged = lambda replicas=False: contextlib.nullcontext()
    tr.sample_images = lambda model, step: finished.append(("here", step))
    tr._sample_round(None, 10)
    p = w.pending_of("sample")
    assert p.step == 10 and p.jobs == ["job0.png"] and finished == [] and "samples of step 10" in status["stage"]
    tr._sample_take(11)  # not in yet
    assert finished == []
    w.done["sample"] = ["samples/step000010_0.png"]
    tr._sample_take(13)
    assert finished == [(10, ["samples/step000010_0.png"])] and w.queue == []
    # the next one is due before the last is in: training waits for it
    tr._sample_round(None, 20)
    w.done["sample"] = ["samples/step000020_0.png"]
    tr._sample_round(None, 30)
    assert ("sample", True) in w.waits and finished[-1] == (20, ["samples/step000020_0.png"])
    assert w.pending_of("sample").step == 30


def test_a_lost_sample_round_is_rendered_again_here_from_its_own_weights(tmp_path):
    T, tr, _ = _trainer(tmp_path)
    tr.sample_worker = w = FakeWorker()
    tr._sample_jobs = lambda step, n=1: ["job"]
    tr._averaged = lambda replicas=False: contextlib.nullcontext()
    finished, rendered = [], []
    tr._sample_finish = lambda paths, step: finished.append((step, paths))

    def render(model, step, state, jobs, kind):
        rendered.append((step, kind, state))
        return ["again.png"]

    tr._render_from_state = render
    tr._sample_round(None, 10)
    queued = w.queue[0].state
    w.fail_with = "CUDA out of memory"
    tr._sample_take(12, wait=True)
    assert tr.sample_worker is None and w.closed and finished == [(10, ["again.png"])]
    assert rendered[0][:2] == (10, "sample") and rendered[0][2] is queued
    # a round whose pictures had already come is not rendered again
    tr.sample_worker = w2 = FakeWorker()
    tr._sample_round(None, 20)
    w2.queue[0].reply = ("done", ["came.png"], 1.0)
    tr._close_probe_worker(redo=True)
    assert finished[-1] == (20, ["came.png"]) and len(rendered) == 1


def test_a_failed_send_of_a_sample_round_renders_it_here(tmp_path):
    T, tr, _ = _trainer(tmp_path)
    tr.sample_worker = w = FakeWorker()
    tr._sample_jobs = lambda step, n=1: ["job"]
    tr._averaged = lambda replicas=False: contextlib.nullcontext()
    tr.remote = None
    here = []
    tr.sample_images = lambda model, step: here.append(step)

    def fail(*a, **k):
        raise PW.ProbeProcessError("closed")

    w.submit = fail
    tr._sample_round(None, 10)
    assert here == [10] and tr.sample_worker is None and w.closed


def test_a_sample_worker_wins_over_a_remote_helper(tmp_path):
    T, tr, _ = _trainer(tmp_path)
    tr.sample_worker = w = FakeWorker()
    tr._sample_jobs = lambda step, n=1: ["job"]
    tr._averaged = lambda replicas=False: contextlib.nullcontext()
    tr.remote = SimpleNamespace(poll=lambda: pytest.fail("the helper must not be asked"))
    tr._sample_round(None, 10)
    assert w.pending_of("sample") is not None


def test_the_process_queues_sample_rounds_beside_probes_and_sweeps():
    w = _process()
    w.memory = "mem"
    w.submit(10, ["p"], _state())
    w.submit(10, ["s"], _state(), kind="sample")
    w.submit_analysis("sweep", 10, _state(), keys=["a"])
    assert [m[0] for m in w.conn.sent] == ["probe", "probe", "sweep"]
    assert (w.conn.sent[0][3], w.conn.sent[1][3]) == ("probe", "sample")
    with pytest.raises(RuntimeError, match="still queued"):
        w.submit(20, ["s"], _state(), kind="sample")
    w.conn.inbox += [("done", ["probe.png"], 1.0), ("done", ["sample.png"], 2.0), ("sweep_done", {}, 1.0)]
    assert w.take_sample(True)[1] == ["sample.png"]  # read through the probe's answer
    assert w.take(False)[1] == ["probe.png"]
    assert [p.kind for p in w.outstanding()] == ["sweep"]
    w.submit(30, ["s"], _state(), kind="sample")
    assert [p.kind for p in w.outstanding()] == ["sweep", "sample"]  # a lost sample is read again, a probe is not


# ---------------------------------------------------------------- review fixes: no deadlock, a dead process is handled


def test_the_worker_reads_the_next_job_while_it_is_blocked_sending_a_large_reply():
    import threading
    from multiprocessing.connection import Client, Listener

    big = 24 * 2**20
    key = b"k" * 16
    listener = Listener(("127.0.0.1", 0), authkey=key)
    worker_side = []
    t = threading.Thread(target=lambda: worker_side.append(Client(listener.address, authkey=key)), daemon=True)
    t.start()
    trainer = listener.accept()
    t.join(10)
    seen = []

    class R:  # a renderer whose answer is far bigger than the socket buffers
        def analyse(self, msg):
            seen.append(msg[0])
            return ("maps_done", b"r" * big, 0.0)

    server = threading.Thread(target=PW._serve, args=(worker_side[0], R()), daemon=True)
    server.start()
    sent = []

    def send_jobs():  # job 1's reply blocks the worker (nobody reads it yet); job 2 is as large as a reply
        trainer.send(("maps", b"a", False))
        trainer.send(("maps", b"p" * big, False))
        trainer.send(("stop",))
        sent.append(True)

    sender = threading.Thread(target=send_jobs, daemon=True)
    sender.start()
    sender.join(30)
    assert not sender.is_alive(), "deadlock: the trainer's send never completed"
    replies = [trainer.recv(), trainer.recv()]
    server.join(30)
    assert not server.is_alive() and sent == [True] and seen == ["maps", "maps"]
    assert [len(r[1]) for r in replies] == [big, big]
    trainer.close()
    listener.close()


def test_a_probe_that_cannot_be_sent_renders_on_the_training_gpu(tmp_path):
    T, tr, status = _trainer(tmp_path)
    tr.probe_worker = tr.analysis_worker = w = FakeWorker()
    tr._probe_jobs = lambda step: ["job"]
    tr._averaged = lambda replicas=False: contextlib.nullcontext()
    assert tr._probe_async_start(10, tr.injected) is True and w.pending_of("probe").step == 10
    assert "probe of step 10" in status["stage"]
    w.queue.clear()

    def fail(*a, **k):
        raise PW.ProbeProcessError("closed its connection")

    w.submit = fail
    assert tr._probe_async_start(20, tr.injected) is False  # the caller renders it here
    assert tr.probe_worker is None and tr.analysis_worker is None and w.closed


def test_a_failed_sample_worker_never_hands_the_round_to_a_remote_helper(tmp_path):
    T, tr, _ = _trainer(tmp_path)
    tr._sample_jobs = lambda step, n=1: ["job"]
    tr._averaged = lambda replicas=False: contextlib.nullcontext()
    tr.remote = SimpleNamespace(poll=lambda: pytest.fail("the helper must not be asked"))
    here, finished = [], []
    tr.sample_images = lambda model, step: here.append(step)
    tr._sample_finish = lambda paths, step: finished.append((step, paths))
    tr._render_from_state = lambda model, step, state, jobs, kind: ["again.png"]
    # the send fails
    tr.sample_worker = w = FakeWorker()

    def fail(*a, **k):
        raise PW.ProbeProcessError("closed")

    w.submit = fail
    tr._sample_round(None, 10)
    assert here == [10] and tr.sample_worker is None and w.closed
    # the last round fails while this one waits for it: the lost round is redone and this one renders here
    tr.sample_worker = w2 = FakeWorker()
    tr._sample_round(None, 20)
    w2.fail_with = "CUDA out of memory"
    tr._sample_round(None, 30)
    assert here == [10, 30] and finished == [(20, ["again.png"])] and tr.sample_worker is None and w2.closed


def test_a_poll_error_and_a_failed_setup_send_are_process_errors(tmp_path, monkeypatch):
    class PollConn(FakeConn):
        def poll(self, timeout=0):
            raise OSError("socket closed")

    w = _process()
    w.conn = PollConn()
    with pytest.raises(PW.ProbeProcessError, match="closed its connection"):
        w._recv()
    w.memory = "mem"
    w.queue = [PW.Pending(1, [], {}, kind="sweep")]
    with pytest.raises(PW.ProbeProcessError, match="closed its connection"):
        w._pump()
    with pytest.raises(PW.ProbeProcessError, match="closed its connection"):
        w.ready_analysis("sweep")

    killed = []

    class FakeListener:
        address = ("127.0.0.1", 1234)

        def __init__(self, *a, **k):
            pass

        def accept(self):
            return BrokenConn()

        def close(self):
            pass

    class FakePopen:
        returncode = None

        def __init__(self, *a, **k):
            self.stdin = io.BytesIO()

        def poll(self):
            return None

        def kill(self):
            killed.append(True)

    monkeypatch.setattr(PW, "Listener", FakeListener)
    monkeypatch.setattr(PW.subprocess, "Popen", FakePopen)
    with pytest.raises(PW.ProbeProcessError, match="closed its connection"):
        PW.ProbeProcess({}, None, "cuda:1", tmp_path / "probe.log")
    assert killed == [True]
