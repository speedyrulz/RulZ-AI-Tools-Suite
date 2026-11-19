"""sweep_split: each region sweep's pictures divided between the second GPU and the training GPU, then merged into one
sweep. The sizing (regionsweep.SplitSizer), and the trainer queueing the second GPU's part, reading its own at once,
merging, falling back and abandoning (the second GPU's process faked: no model, no GPU)."""
import contextlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rulztrainer import lora as L  # noqa: E402
from rulztrainer import probeworker as PW  # noqa: E402
from rulztrainer import regionsweep as RS  # noqa: E402
from rulztrainer.config import TrainConfig  # noqa: E402
from test_probe_async import _lora  # noqa: E402
from test_regionsweep import Toy, _probes as sweep_probes  # noqa: E402


def _cfg(**kw):
    kw.setdefault("dataset_dir", "ds")
    return TrainConfig(**kw)


# ---------------------------------------------------------------- the setting


def test_the_setting_is_off_by_default_needs_sweep_async_and_round_trips():
    assert _cfg().sweep_split is False
    with pytest.raises(ValueError, match="sweep_async"):
        _cfg(sweep_split=True).validate()
    on = _cfg(sweep_async=True, sweep_split=True)
    on.validate()
    assert TrainConfig.from_dict(on.to_dict()).sweep_split is True


# ---------------------------------------------------------------- the sizing


def _sized(remote_spp=None, local_spp=None):
    z = RS.SplitSizer()
    z.remote_spp, z.local_spp = remote_spp, local_spp
    return z


def test_nothing_measured_splits_in_half_and_the_held_out_pictures_stay_on_the_second_gpu():
    assert _sized().remote_count(10, 0, 50, None, 0.0) == 5
    assert _sized().remote_count(10, 7, 50, None, 0.0) == 7  # (more than half are held out)
    assert _sized().remote_count(0, 0, 50, None, 0.0) == 0


def test_the_budget_is_a_share_of_the_interval_less_the_other_work_and_it_never_exceeds_the_pictures():
    z = _sized(remote_spp=2.0, local_spp=1.0)
    # 0.8 * 50 steps * 1.0 s = 40 s of the second GPU's time: 20 pictures at 2 s
    assert z.remote_count(30, 2, 50, 1.0, 0.0) == 20
    assert z.remote_count(30, 2, 50, 1.0, 10.0) == 15  # 10 s went to probes and maps
    assert z.remote_count(12, 2, 50, 1.0, 0.0) == 12  # it keeps up: nothing is read on the training GPU
    assert z.remote_count(30, 2, 50, 1.0, 100.0) == 2  # no budget left: the held-out pictures are still all there
    assert z.remote_count(30, 5, 50, 0.0, 0.0) == 10  # no step time known yet: by speed (a third of them)


def test_the_first_and_the_last_sweep_split_by_speed():
    z = _sized(remote_spp=2.0, local_spp=1.0)  # the training GPU is twice as fast: it takes two thirds
    assert z.remote_count(30, 0, 50, 1.0, 0.0, last=True) == 10
    assert z.remote_count(30, 15, 50, 1.0, 0.0, last=True) == 15  # the held-out pictures first
    assert _sized(remote_spp=3.0).remote_count(10, 0, 50, 1.0, 0.0, last=True) == 5  # an unmeasured side: equal


def test_seconds_per_picture_are_smoothed_and_an_overrun_shrinks_the_budget_until_sweeps_are_in_time():
    z = RS.SplitSizer()
    z.note_remote(10.0, 5)
    z.note_remote(20.0, 5)
    assert z.remote_spp == pytest.approx(3.0) and z.remote_total == 30.0
    z.note_remote(0.0, 0)  # nothing read: nothing learned
    assert z.remote_spp == pytest.approx(3.0)
    z.note_wait(10.0, 40.0)  # training waited 10 s on a 40 s interval
    assert z.scale == pytest.approx(0.8)
    z.note_wait(0.0, 40.0)
    assert z.scale == pytest.approx(0.88)
    for _ in range(10):
        z.note_wait(0.0, 40.0)
    assert z.scale == 1.0
    z.note_wait(5.0, None)  # (no step time yet: nothing to scale by)
    assert z.scale == 1.0


def test_the_other_work_is_what_the_second_gpu_did_besides_the_sweeps_since_the_last_sweep():
    z = RS.SplitSizer()
    assert z.other_work(12.0) == 0.0  # the first call only sets the mark
    z.note_remote(30.0, 10)
    assert z.other_work(12.0 + 30.0 + 6.0) == pytest.approx(6.0)  # 6 s of probes since
    assert z.other_work(12.0 + 30.0 + 6.0) == 0.0


# ---------------------------------------------------------------- the trainer


class FakeWorker:
    """ProbeProcess's interface for sweeps; an answer is in when it is put in `done`."""

    def __init__(self, ready=True):
        self.device, self.queue, self.done, self.fail_with, self.send_fails = "cuda:1", [], {}, None, False
        self.memory = "5 GB" if ready else None
        self.analysis_wanted, self.analysis_error = True, None
        self.count = self.analysis_count = 0
        self.rendered = self.analysed = self.waited = 0.0
        self.closed, self.waits = False, []

    @property
    def analysis_ok(self):
        return self.memory is not None

    def wait_ready(self):
        self.memory = "5 GB"

    def pending_of(self, kind):
        return next((p for p in self.queue if p.kind == kind), None)

    def submit_analysis(self, kind, step, state, keys=None, extra=None):
        if self.send_fails:
            raise PW.ProbeProcessError("the connection closed")
        assert self.pending_of(kind) is None
        self.queue.append(PW.Pending(step, list(keys or []), state, kind=kind, extra=dict(extra or {})))

    def ready_analysis(self, kind):
        return kind in self.done

    def take_analysis(self, kind, wait):
        p = self.pending_of(kind)
        if p is None or not (wait or kind in self.done):
            return None
        self.waits.append((kind, wait))
        self.queue.remove(p)
        result = self.done.pop(kind)
        p.reply = (f"{kind}_done", result, 3.0)
        if isinstance(result, str):
            return p, None, result
        self.analysis_count += 1
        return p, result, ""

    def outstanding(self):
        return [p for p in self.queue if p.kind != "probe"]

    def close(self):
        self.closed = True


def _trainer(tmp_path, n_train=6, n_held=2, steps=1000, **cfg):
    from rulztrainer import trainer as T

    tr = T.Trainer.__new__(T.Trainer)
    cfg.setdefault("sweep_async", True)
    cfg.setdefault("sweep_split", True)
    tr.cfg = _cfg(holdout_track_best=True, steps=steps, **cfg)
    tr.controller, tr.race_rows, tr.ft, tr.ema, tr.device = None, None, None, None, "cpu"
    tr.job_dir = tmp_path
    status = {}
    tr.status = SimpleNamespace(update=lambda **k: status.update(k), data={"saved": []})
    tr.region_watch, tr.holdout_stop = None, False
    tr.injected = _lora(0)
    tr.replicas = [(Toy(), tr.injected, "cpu")]
    tr._maps_epoch_steps = lambda: (10, 1)
    tr._free_for_rendering = lambda: None
    tr.sweep = RS.RegionSweep(tmp_path, sweep_probes(n_train, n_held), every=10)
    tr.maps = None
    tr.analysis_worker = tr.probe_worker = tr.sample_worker = None
    tr.recent_step_times = []
    tr.saved_states = []

    def save_best(injected, step, score, averaged=True):
        tr.saved_states.append((step, averaged, L.lora_state(injected)))

    tr._save_holdout_best = save_best
    return T, tr, status


def _readings(tr, model, keys):
    return RS.forward_readings(model, "cpu", [p for p in tr.sweep.probes if p["key"] in keys])


def _answer(w, tr, model):
    """The second GPU's answer to the sweep it was sent: the readings of the pictures it was given."""
    w.done["sweep"] = _readings(tr, model, w.pending_of("sweep").jobs)


def test_a_split_sweep_is_the_unsplit_sweep_with_its_parts_read_where_they_were_sent(tmp_path):
    (tmp_path / "plain").mkdir()
    plain = RS.RegionSweep(tmp_path / "plain", sweep_probes(6, 2), every=10)
    T, tr, status = _trainer(tmp_path)
    tr.analysis_worker = w = FakeWorker()
    model = tr.replicas[0][0]
    with torch.no_grad():
        model.gain[:] = 0.3
    want = plain.run(0, 0.0, model, "cpu")
    assert tr.sweep.due(0)
    tr._run_sweep(model, tr.injected, 0)  # (first sweep, nothing measured: half each, the held-out ones remote)
    p = w.pending_of("sweep")
    assert p.jobs == ["held0.png", "held1.png", "train0.png", "train1.png"]
    assert p.extra["all_keys"] == p.jobs + [f"train{i}.png" for i in range(2, 6)]
    assert sorted(p.extra["local"]) == [f"train{i}.png" for i in range(2, 6)]
    assert not tr.sweep.due(5) and tr.sweep.history == [] and status["stage"].startswith("training; region sweep")
    _answer(w, tr, model)
    tr._analysis_take("sweep", 3)
    row = tr.sweep.history[0]
    assert {k: v for k, v in row.items() if k != "seconds"} == {k: v for k, v in want.items() if k != "seconds"}
    assert tr.sweep.latest == plain.latest and row["swept"] == 8
    assert status["region_sweep"]["split"]["remote"] == 4 and status["region_sweep"]["split"]["local"] == 4
    assert w.queue == [] and tr._sweep_sizer().remote_spp == pytest.approx(3.0 / 4)
    assert tr._sweep_sizer().local_spp is not None


def test_the_part_read_here_uses_the_weight_average_that_was_sent(tmp_path):
    T, tr, _ = _trainer(tmp_path)
    tr.analysis_worker = w = FakeWorker()
    seen = []

    @contextlib.contextmanager
    def averaged(replicas=False):
        before = L.lora_state(tr.injected)
        with torch.no_grad():
            for m in tr.injected.values():
                for q in m.adapter_params().values():
                    q.add_(0.25)
        try:
            yield
        finally:
            L.mirror_lora(L.lora_snapshots(before), tr.injected)

    def forward(model, device, chosen, averaged=None):
        with averaged():
            seen.append((len(chosen), L.lora_state(tr.injected)))
        return RS.forward_readings(model, device, chosen)

    tr._averaged = averaged
    tr.sweep._forward = forward
    raw = L.lora_state(tr.injected)
    tr._run_sweep(tr.replicas[0][0], tr.injected, 0)
    sent = w.pending_of("sweep").state
    assert seen and all(torch.equal(seen[0][1][n]["params"][k], sent[n]["params"][k]) for n in sent
                        for k in sent[n]["params"])
    assert all(torch.equal(L.lora_state(tr.injected)[n]["params"][k], raw[n]["params"][k]) for n in raw
               for k in raw[n]["params"])  # training's weights are back


def test_when_the_second_gpu_keeps_up_nothing_is_read_here(tmp_path):
    T, tr, _ = _trainer(tmp_path)
    tr.analysis_worker = w = FakeWorker()
    tr.recent_step_times = [1.0] * 10
    z = tr._sweep_sizer()
    z.note_remote(8.0, 8)  # 1 s a picture; 0.8 * 10 steps * 1 s = 8 s: all eight fit
    tr.sweep.next_step = 0
    reads = []
    tr.sweep._forward = lambda *a, **k: reads.append(1) or {}
    tr._run_sweep(tr.replicas[0][0], tr.injected, 10)
    p = w.pending_of("sweep")
    assert len(p.jobs) == 8 and reads == [] and p.extra["local"] == {}
    _answer(w, tr, tr.replicas[0][0])
    tr._analysis_take("sweep", 10)
    assert tr.sweep.history[0]["swept"] == 8


def test_when_nothing_fits_on_the_second_gpu_but_the_held_out_pictures_it_reads_those_and_this_reads_the_rest(tmp_path):
    T, tr, _ = _trainer(tmp_path)
    tr.analysis_worker = w = FakeWorker()
    tr.recent_step_times = [1.0] * 10
    tr._sweep_sizer().note_remote(40.0, 2)  # 20 s a picture: no budget for any
    tr._run_sweep(tr.replicas[0][0], tr.injected, 10)
    p = w.pending_of("sweep")
    assert p.jobs == ["held0.png", "held1.png"] and len(p.extra["local"]) == 6


def test_with_no_picture_for_the_second_gpu_the_sweep_is_read_and_recorded_here(tmp_path):
    T, tr, status = _trainer(tmp_path, n_held=0)
    tr.analysis_worker = w = FakeWorker()
    tr.recent_step_times = [1.0] * 10
    tr._sweep_sizer().note_remote(40.0, 2)
    tr._run_sweep(tr.replicas[0][0], tr.injected, 10)
    assert w.queue == [] and [r["step"] for r in tr.sweep.history] == [10] and tr.sweep.history[0]["swept"] == 6
    assert status["region_sweep"]["split"]["remote"] == 0 and status["stage"] == "training"


def test_the_final_sweep_is_split_by_speed(tmp_path):
    T, tr, _ = _trainer(tmp_path, n_held=0, steps=30)
    tr.analysis_worker = w = FakeWorker()
    tr.recent_step_times = [1.0] * 10
    z = tr._sweep_sizer()
    z.remote_spp, z.local_spp = 1.0, 2.0  # the second GPU is twice as fast: it takes two thirds
    tr._run_sweep(tr.replicas[0][0], tr.injected, 30)
    assert len(w.pending_of("sweep").jobs) == 4 and len(w.pending_of("sweep").extra["local"]) == 2


def test_the_next_sweep_waits_for_the_last_and_an_overrun_shrinks_the_budget(tmp_path):
    T, tr, _ = _trainer(tmp_path)
    tr.analysis_worker = w = FakeWorker()
    tr.recent_step_times = [1.0] * 10
    model = tr.replicas[0][0]
    tr._run_sweep(model, tr.injected, 0)
    _answer(w, tr, model)
    tr._run_sweep(model, tr.injected, 10)  # the take blocks for the answer (it is "in" at once)
    assert ("sweep", True) in w.waits and [r["step"] for r in tr.sweep.history] == [0]
    assert w.pending_of("sweep").step == 10
    z = tr._sweep_sizer()
    z.scale = 1.0
    z.note_wait(10.0, 10.0)
    assert z.scale == pytest.approx(0.5)


def test_a_local_failure_turns_the_sweeps_off_and_the_second_gpus_answer_is_dropped(tmp_path):
    T, tr, status = _trainer(tmp_path)
    tr.analysis_worker = w = FakeWorker()
    model = tr.replicas[0][0]

    def boom(*a, **k):
        raise RuntimeError("the parts map does not fit")

    tr.sweep._forward = boom
    tr._run_sweep(model, tr.injected, 0)
    assert tr.sweep.failed and w.pending_of("sweep").extra["abandoned"] is True and tr.sweep.history == []
    _answer(w, tr, model)
    tr._analysis_take_all(2)
    assert tr.sweep.history == [] and w.queue == [] and status["stage"] == "training"  # (no KeyError on the local keys)


def test_a_failed_send_puts_the_schedule_back_and_the_whole_sweep_runs_here(tmp_path):
    T, tr, _ = _trainer(tmp_path)
    tr.analysis_worker = w = FakeWorker()
    w.send_fails = True
    assert tr._analysis_start("sweep", 0, tr.injected, model=tr.replicas[0][0]) is False
    assert tr.analysis_worker is None and w.closed and tr.sweep.due(0) and tr.sweep.last_step is None


def test_a_second_gpu_that_fails_has_only_its_part_read_again_here_and_merged_with_the_kept_part(tmp_path):
    (tmp_path / "plain").mkdir()
    plain = RS.RegionSweep(tmp_path / "plain", sweep_probes(6, 2), every=10)
    T, tr, _ = _trainer(tmp_path)
    tr.analysis_worker = w = FakeWorker()
    model = tr.replicas[0][0]
    want = plain.run(0, 0.0, model, "cpu")
    tr._run_sweep(model, tr.injected, 0)
    p = w.pending_of("sweep")
    remote_keys, local_keys = list(p.jobs), set(p.extra["local"])
    asked = []
    real = tr.sweep._forward
    tr.sweep._forward = lambda m, d, chosen, averaged: asked.append([c["key"] for c in chosen]) or real(m, d, chosen,
                                                                                                      averaged)
    tr._close_probe_worker(redo=True)  # (the process failed: everything out is finished here)
    assert asked == [remote_keys] and not (set(remote_keys) & local_keys)
    row = tr.sweep.history[0]
    assert {k: v for k, v in row.items() if k not in ("seconds",)} == {k: v for k, v in want.items() if k != "seconds"}


def test_a_lost_split_sweep_whose_answer_had_come_is_not_read_again(tmp_path):
    T, tr, _ = _trainer(tmp_path)
    tr.analysis_worker = w = FakeWorker()
    model = tr.replicas[0][0]
    tr._run_sweep(model, tr.injected, 0)
    p = w.pending_of("sweep")
    p.reply = ("sweep_done", _readings(tr, model, p.jobs), 2.0)
    tr.sweep._forward = lambda *a, **k: pytest.fail("read again although its answer had come")
    tr._close_probe_worker(redo=True)
    assert [r["step"] for r in tr.sweep.history] == [0] and tr.sweep.history[0]["swept"] == 8


def test_with_the_setting_off_a_sweep_is_queued_whole_as_before(tmp_path):
    T, tr, _ = _trainer(tmp_path, sweep_split=False)
    tr.analysis_worker = w = FakeWorker()
    tr._run_sweep(tr.replicas[0][0], tr.injected, 0)
    p = w.pending_of("sweep")
    assert len(p.jobs) == 8 and "all_keys" not in p.extra
    _answer(w, tr, tr.replicas[0][0])
    tr._analysis_take("sweep", 1)
    assert tr.sweep.history[0]["swept"] == 8


def test_one_picture_still_measures_the_second_gpu():
    from rulztrainer.regionsweep import SplitSizer

    sizer = SplitSizer()
    assert sizer.remote_count(1, 0, 10, None, 0.0) == 1  # half of one rounds to 0: the second GPU would never be timed
    sizer.note_remote(2.0, 1)
    sizer.note_local(1.0, 1)
    assert sizer.remote_count(1, 0, 10, None, 0.0, last=True) == 0  # once measured, the speed split decides
