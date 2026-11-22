"""Region sweeps and held-out photos in a configuration race (report only): one sweep per configuration on its own steps,
the same photos held out of all of them, files and status per configuration, and the parallel race's path. CPU only: a
toy model and a Trainer built without its constructor, as test_regionsweep does."""
import dataclasses
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from test_regionsweep import Scalar, _stub_trainer  # noqa: E402

from rulztrainer import race as R  # noqa: E402
from rulztrainer import regionsweep as RS  # noqa: E402
from rulztrainer.dataset import BucketBatcher  # noqa: E402


class Recording(Scalar):
    """As Scalar; remembers the gain each forward pass read (what weights the sweep saw)."""

    def __init__(self):
        super().__init__()
        self.seen = []

    def forward(self, noisy, sigma, embeds, mask):
        self.seen.append(round(float(self.gain), 4))
        return super().forward(noisy, sigma, embeds, mask)


class Broken(Scalar):
    def forward(self, noisy, sigma, embeds, mask):
        raise RuntimeError("out of memory")


def _race(tmp_path, parallel=False, **kw):
    t, items = _stub_trainer(tmp_path, 12, holdout_images=2, **kw)
    base = t.cfg
    t.race_rows = [(n, dataclasses.replace(base), {}) for n in ("a", "b")]
    t.race_parallel, t.race_base = parallel, base
    train = t._split_holdout(base, items)  # (once, over the shared items)
    model = Recording()
    model.train()
    t.replicas = [(model, {}, "cpu")]
    variants = [SimpleNamespace(idx=i, name=n, cfg=c, state={"items": R.variant_items(train, c)}, started=True,
                                overrides={}, alive=True) for i, (n, c, _) in enumerate(t.race_rows)]
    t.race = SimpleNamespace(active=variants[0], variants=variants)
    return t, items, train, variants, model


def _start(t, v):
    """What a configuration's first start does for its sweep (its image queue is there)."""
    t.cfg = v.cfg
    t.batcher = BucketBatcher(v.state["items"], 1, 0)
    t._race_make_sweep(v)


def _lines(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]


def _status(t):
    return json.loads((t.job_dir / "status.json").read_text(encoding="utf-8"))


def test_the_same_photos_are_held_out_of_every_configuration(tmp_path):
    t, items, train, variants, _ = _race(tmp_path)
    held = {str(it.image) for it in t.holdout_items}
    assert len(train) == 10 and len(held) == 2
    for v in variants:  # each configuration's items are made from the already split items
        assert not held & {str(it.image) for it in v.state["items"]} and len(v.state["items"]) == 10


def test_a_tagged_sweep_writes_its_own_files_with_the_extra_keys(tmp_path):
    sweep = RS.RegionSweep(tmp_path, [], 5, tag="race3", extra={"config": "x", "index": 3})
    assert (sweep.state_name, sweep.log_name) == ("region_sweep.race3.json", "region_sweep.race3.jsonl")
    assert RS.file_names() == (RS.STATE_FILE, RS.LOG_FILE) == ("region_sweep.json", "region_sweep.jsonl")
    assert RS.RegionSweep(tmp_path, [], 5).state_name == "region_sweep.json" and RS.RegionSweep(tmp_path, [], 5).extra == {}


def test_each_configuration_is_swept_on_its_own_steps_against_its_own_first_sweep(tmp_path):
    t, items, train, (a, b), model = _race(tmp_path)
    _start(t, a)
    _start(t, b)
    sa, sb = a.state["sweep"], b.state["sweep"]
    assert sa is not sb and sa.every == sb.every == 10 and (sa.tag, sb.tag) == ("race0", "race1")
    assert len(sa.probes) == 12 and sum(p["held"] for p in sa.probes) == 2 and sa.patience == sb.patience == 0
    # a: its first sweep at its own step 0, then 10 steps later (own steps)
    t.sweep, t.race.active = sa, a
    assert sa.due(0) and not sb.due(-1)
    t._run_sweep(model, {}, 0)
    assert not sa.due(9) and sa.due(10) and sa.due(3, last=True)
    with torch.no_grad():
        model.gain.fill_(0.5)
    t._run_sweep(model, {}, 10)
    # b takes over: its own first sweep is at ITS step 0, with the weights it has then (gain 0.5), whatever a's step
    t.sweep, t.race.active = sb, b
    assert sb.due(0) and not sa.due(10)
    t._run_sweep(model, {}, 0)
    with torch.no_grad():
        model.gain.fill_(0.5)  # b has not learned anything since its own first sweep
    t._run_sweep(model, {}, 10)
    ra, rb = _lines(tmp_path / "job" / "region_sweep.race0.jsonl"), _lines(tmp_path / "job" / "region_sweep.race1.jsonl")
    assert [r["step"] for r in ra] == [0, 10] and [r["step"] for r in rb] == [0, 10]
    assert ra[0]["first"] and rb[0]["first"] and not ra[1]["first"]
    assert all(r["config"] == "a" and r["index"] == 0 for r in ra) and all(r["config"] == "b" and r["index"] == 1 for r in rb)
    assert ra[1]["held"]["subject"] == pytest.approx(0.75, abs=1e-3)  # a learned since its first sweep
    assert rb[1]["held"]["subject"] == pytest.approx(0.0, abs=1e-3)  # b did not: measured from its own start
    for i in (0, 1):
        state = json.loads((tmp_path / "job" / f"region_sweep.race{i}.json").read_text(encoding="utf-8"))
        assert state["history"][-1]["index"] == i and len(state["base"]) == 12
    # the plain run's files are not written in a race
    assert not (tmp_path / "job" / "region_sweep.json").exists() and not (tmp_path / "job" / "region_sweep.jsonl").exists()
    configs = _status(t)["region_sweep_race"]["configs"]
    assert [c["index"] for c in configs] == [0, 1] and [c["config"] for c in configs] == ["a", "b"]
    assert configs[0]["step"] == 10 and [h["step"] for h in configs[0]["history"]] == [0, 10]
    assert set(configs[0]) >= {"line", "history", "train", "held", "gap", "first", "swept", "pictures"}
    assert "region_sweep" not in _status(t)  # (the plain entry)


def test_the_sweeps_report_only_nothing_stops_or_is_saved(tmp_path):
    t, items, train, (a, b), model = _race(tmp_path, holdout_patience=1, holdout_track_best=True)
    saved = []
    t._save_holdout_best = lambda *args, **kw: saved.append(args)
    _start(t, a)
    t.sweep = a.state["sweep"]
    t._run_sweep(model, {}, 0)
    with torch.no_grad():
        model.gain.fill_(0.5)
    t._run_sweep(model, {}, 10)  # a new best: with holdout_track_best a plain run would save it
    t._run_sweep(model, {}, 20)  # no better: with a patience of 1 a plain run would stop
    rows = _lines(tmp_path / "job" / "region_sweep.race0.jsonl")
    assert rows[1]["new_best"] and not any(r["stop"] for r in rows)
    assert saved == [] and t.holdout_stop is False and a.state["sweep"].patience == 0


def test_a_failed_sweep_stops_only_that_configurations_sweeps(tmp_path, caplog):
    t, items, train, (a, b), model = _race(tmp_path)
    _start(t, a)
    _start(t, b)
    t.sweep, t.race.active = a.state["sweep"], a
    t.replicas = [(Broken(), {}, "cpu")]
    with caplog.at_level(logging.WARNING, logger="rulztrainer"):
        t._run_sweep(t.replicas[0][0], {}, 0)
    assert a.state["sweep"].failed and not b.state["sweep"].failed
    assert not a.state["sweep"].due(10)
    t.replicas = [(model, {}, "cpu")]
    t.sweep, t.race.active = b.state["sweep"], b
    t._run_sweep(model, {}, 0)  # b goes on
    configs = {c["config"]: c for c in _status(t)["region_sweep_race"]["configs"]}
    assert configs["a"]["failed"] is True and "failed" not in configs["b"] and configs["b"]["step"] == 0
    assert not (tmp_path / "job" / "region_sweep.race0.jsonl").exists()
    assert (tmp_path / "job" / "region_sweep.race1.jsonl").exists()


def test_a_dropped_configuration_keeps_its_entry_and_files(tmp_path):
    t, items, train, (a, b), model = _race(tmp_path)
    _start(t, a)
    _start(t, b)
    for v in (a, b):
        t.sweep, t.race.active = v.state["sweep"], v
        t._run_sweep(model, {}, 0)
    t._race_sweep_publish(a, dropped=True)
    t._race_sweep_publish(b, finished=True)
    configs = _status(t)["region_sweep_race"]["configs"]
    assert configs[0]["dropped"] is True and "dropped" not in configs[1] and configs[1]["finished"] is True
    assert len(configs) == 2 and (tmp_path / "job" / "region_sweep.race0.json").exists()


def test_the_race_says_what_its_sweeps_cost_and_what_stays_off(tmp_path, caplog):
    t, items, train, variants, _ = _race(tmp_path, holdout_patience=2, holdout_track_best=True, sweep_async=True,
                                         region_sweep_every=2)
    with caplog.at_level(logging.INFO, logger="rulztrainer"):
        t._race_sweep_plan(train)
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert ("Region sweeps in the race: 2 configurations, each with its own sweep of 10 training pictures and 2 "
            "held-out, every 2 epochs of its own steps (36 forward passes") in text
    assert "holdout_patience and holdout_track_best are off" in text and "run on the training GPU" in text


def test_a_race_without_sweeps_asked_for_makes_none(tmp_path):
    t, items = _stub_trainer(tmp_path, 12)
    t.race_rows = [("a", t.cfg, {})]
    v = SimpleNamespace(idx=0, name="a", cfg=t.cfg, state={}, started=True)
    t.batcher = BucketBatcher(items, 1, 0)
    t._race_make_sweep(v)
    assert "sweep" not in v.state
    t.race = SimpleNamespace(active=v)
    t._setup_sweep(t.cfg, 0)
    assert t.sweep is None


def test_the_active_configurations_sweep_is_the_one_the_loop_runs(tmp_path):
    t, items, train, (a, b), model = _race(tmp_path)
    _start(t, a)
    _start(t, b)
    t.sweep = None
    t._setup_sweep(t.cfg, 0)  # (the first configuration trains at the start)
    assert t.sweep is a.state["sweep"]
    t.race.active = b
    t._setup_sweep(b.cfg, 0)
    assert t.sweep is b.state["sweep"]


def _parallel(tmp_path):
    t, items, train, variants, model = _race(tmp_path, parallel=True)
    t.controller = t.loss_watch = t.batch_iter = t.curriculum_w = t.id_loss = None
    t.ts_logit_shift, t.all_items = 0.0, []
    gains = {"a": 0.25, "b": 0.75}
    loaded = []

    def load_render(v):  # the configuration's weights go into the first copy's wrappers
        loaded.append(v.name)
        with torch.no_grad():
            model.gain.fill_(gains[v.name])

    t._rp_load_render = load_render
    for v in variants:  # (each configuration's first start, with its own bound batcher)
        t.cfg = v.cfg
        with t._rp_bound(v, writeback=False):
            t.batcher = BucketBatcher(v.state["items"], 1, 0)
            t._race_make_sweep(v)
    return t, variants, model, loaded, gains


def test_the_parallel_race_sweeps_each_configuration_with_its_weights_loaded(tmp_path):
    t, (a, b), model, loaded, gains = _parallel(tmp_path)
    assert a.state["sweep"].every == 10 and a.state["sweep_epoch"] == 10  # a lane's step is its own batches
    for v in (a, b):
        v.state["step"] = 0
    assert t._rp_first_sweeps([a, b]) is True
    assert loaded == ["a", "b"]
    n = len(model.seen) // 2
    assert set(model.seen[:n]) == {0.25} and set(model.seen[n:]) == {0.75}  # a's sweep read a's weights, b's b's
    first = [_lines(tmp_path / "job" / f"region_sweep.race{i}.jsonl") for i in (0, 1)]
    assert [len(r) for r in first] == [1, 1] and first[0][0]["config"] == "a" and first[1][0]["index"] == 1
    assert t._rp_first_sweeps([a, b]) is False  # (each configuration's first sweep is once)
    assert not (tmp_path / "job" / "region_sweep.json").exists()
    assert {c["config"] for c in _status(t)["region_sweep_race"]["configs"]} == {"a", "b"}
    assert t.race.active is a  # (the binding is undone)


def test_a_lanes_events_sweep_on_its_own_step_count(tmp_path):
    t, (a, b), model, loaded, gains = _parallel(tmp_path)
    for v in (a, b):
        v.cfg.save_every, v.cfg.steps, v.cfg.sample_prompts = 1000, 1000, []
        v.state.update(step=0, loss_watch=None)
    t._rp_first_sweeps([a, b])
    loaded.clear()
    a.state["step"], b.state["step"] = 10, 4  # a lane is at its own step: only a's sweep is due
    assert t._rp_events(a) is True and t._rp_events(b) is False
    assert loaded == ["a"]
    assert [r["step"] for r in _lines(tmp_path / "job" / "region_sweep.race0.jsonl")] == [0, 10]
    assert [r["step"] for r in _lines(tmp_path / "job" / "region_sweep.race1.jsonl")] == [0]
    b.state["step"] = 1000  # its last step: a sweep whatever the schedule
    assert t._rp_events(b) is True
    assert [r["step"] for r in _lines(tmp_path / "job" / "region_sweep.race1.jsonl")] == [0, 1000]


def test_a_failed_weight_load_in_the_parallel_race_turns_only_that_sweep_off(tmp_path):
    t, (a, b), model, loaded, gains = _parallel(tmp_path)

    def broken_load(v):
        if v.name == "a":
            raise torch.OutOfMemoryError("mirroring")
        loaded.append(v.name)

    t._rp_load_render = broken_load
    for v in (a, b):
        v.state["step"] = 0
    assert t._rp_first_sweeps([a, b]) is True  # (the caller still places the lanes again)
    assert a.state["sweep"].failed and not b.state["sweep"].failed and loaded == ["b"]
    configs = {c["config"]: c for c in _status(t)["region_sweep_race"]["configs"]}
    assert configs["a"]["failed"] is True and configs["b"]["step"] == 0
    assert not a.state["sweep"].due(10)


def test_an_error_after_the_sweep_read_also_turns_that_sweep_off(tmp_path):
    t, items, train, (a, b), model = _race(tmp_path)
    _start(t, a)
    _start(t, b)
    t._race_sweep_publish = lambda v, row=None, **kw: (_ for _ in ()).throw(OSError("disk")) if row else None
    t._race_sweep_run(a, model, 0)
    assert a.state["sweep"].failed and not b.state["sweep"].failed and not a.state["sweep"].due(10)


def test_a_race_start_clears_old_sweep_files_but_a_resume_keeps_them(tmp_path):
    t, items, train, (a, b), model = _race(tmp_path)
    job = t.job_dir
    names = ["region_sweep.json", "region_sweep.jsonl", "region_sweep.race0.json", "region_sweep.race1.jsonl",
             "region_sweep.race7.json"]
    for n in names + ["status.json.keep"]:
        (job / n).write_text("{}", encoding="utf-8")
    (job / "sub").mkdir()
    (job / "sub" / "region_sweep.race0.json").write_text("{}", encoding="utf-8")
    t._race_sweep_plan(train)
    assert [n for n in names if (job / n).exists()] == []
    assert (job / "status.json.keep").exists() and (job / "sub" / "region_sweep.race0.json").exists()
    # a plain run set up fresh clears the race files only; on a resume nothing is deleted
    for n in names:
        (job / n).write_text("{}", encoding="utf-8")
    t.race_rows, t.race, t.cfg.holdout_images = None, None, 2
    t.batcher = BucketBatcher(train, 1, 0)
    t.cfg.resume_state, t.cfg.region_sweep_every = "x", 1
    t._setup_sweep(t.cfg, 5)
    assert all((job / n).exists() for n in names)
    t.cfg.resume_state = ""
    t._setup_sweep(t.cfg, 0)
    assert t.sweep is not None and not (job / "region_sweep.jsonl").exists()  # (the new sweep clears its own pair)
    assert not any((job / n).exists() for n in names if "race" in n)
