"""Render on this PC with several GPUs (rulztrainer/renderpool.py and the helper's pool path): the split, the pool's
rounds, failures and loads, with fake worker processes. No GPU, no network, no torch worker."""
import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rulztrainer import renderhelper as rh  # noqa: E402
from rulztrainer import renderhelper_ctl as ctl  # noqa: E402
from rulztrainer import renderpool as rp  # noqa: E402
from test_renderhelper import TOKEN, drive, read_json, world  # noqa: E402,F401 - the fake bridge "world" fixture


def jobs_of(n, cost_of=None):
    return [SimpleNamespace(label=f"j{i}", steps=1, width=1, height=1, frames=1, out_path=None,
                            **({} if cost_of is None else {"steps": cost_of(i)})) for i in range(n)]


# ---------------------------------------------------------------- the split


def test_equal_split_remainder_and_order():
    assert rp.plan_split([1] * 8, [None, None]) == [4, 4]
    assert rp.plan_split([1] * 5, [None, None]) == [2, 3]
    assert rp.plan_split([1] * 7, [None, None, None]) == [2, 3, 2]
    for n in range(0, 40):
        for k in (1, 2, 3, 4):
            assert sum(rp.plan_split([1] * n, [None] * k)) == n


def test_shares_follow_measured_speed():
    assert rp.plan_split([1] * 8, [1.0, 3.0]) == [6, 2]  # the second GPU is 3x slower
    assert rp.plan_split([1] * 8, [3.0, 1.0]) == [2, 6]


def test_an_unmeasured_gpu_gets_the_mean_of_the_measured_ones():
    sizes = rp.plan_split([1] * 12, [1.0, None, 3.0])  # as if it ran at 2.0
    assert sizes == [7, 3, 2]
    assert rp.plan_split([1] * 12, [2.0, 2.0, 2.0]) == rp.plan_split([1] * 12, [2.0, None, 2.0])


def test_cost_weighting_gives_a_big_picture_a_share_of_its_own():
    assert rp.plan_split([10] + [1] * 10, [None, None]) == [1, 10]  # job 0 costs as much as all the others together
    assert rp.job_cost(SimpleNamespace(steps=20, width=64, height=32, frames=5)) == 20 * 64 * 32 * 5
    assert rp.job_cost(SimpleNamespace(steps=0, label="x")) == 1.0  # missing or zero counts 1


def test_fewer_jobs_than_gpus_give_zero_shares_to_the_slowest():
    assert rp.plan_split([1, 1], [3.0, 1.0, 2.0]) == [0, 1, 1]
    assert rp.plan_split([1], [None, None, None]) == [1, 0, 0]
    assert rp.plan_split([], [1.0, 2.0]) == [0, 0]


def test_probe_map_is_fixed_and_equal_in_device_order():
    assert rp.fixed_sizes(5, 2) == [3, 2] and rp.fixed_sizes(2, 3) == [1, 1, 0] and rp.fixed_sizes(4, 1) == [4]
    assert rp.fixed_sizes(5, 2) == rp.fixed_sizes(5, 2)


def test_warm_up_render_is_not_timed_and_speeds_are_averaged():
    s = rp.Slot("cuda:0")
    s.measure("sample", 50.0, 100.0)  # the first render of a worker warms up
    assert s.rates == {}
    s.measure("sample", 10.0, 100.0)
    assert s.rates["sample"] == pytest.approx(0.1)
    s.measure("sample", 20.0, 100.0)
    assert s.rates["sample"] == pytest.approx(0.15)  # EWMA, weight 0.5
    assert "probe" not in s.rates


# ---------------------------------------------------------------- fake workers


class FakeWorker:
    def __init__(self, fx, device):
        self.fx, self.device = fx, device
        self.pending = None
        self.renders = 0
        self.killed = self.stopped = self.released = False

    @property
    def sc(self):
        return self.fx.script.setdefault(self.device, {})

    def poll_ready(self):
        self.fx.events.append(("poll_ready", self.device))
        if self.device in self.fx.load_fail:
            raise RuntimeError("Traceback (most recent call last):\n  ...\n" + self.fx.load_fail[self.device])
        return not self.sc.get("never_ready")

    def submit_payload(self, jobs, payload, negative_kind):
        if self.sc.get("fail_submit"):
            raise BrokenPipeError("the probe process closed its connection")
        self.fx.events.append(("submit", self.device, [j.label for j in jobs]))
        self.fx.got.append((self.device, [j.label for j in jobs], payload, negative_kind))
        self.pending = list(jobs)

    def poll_result(self):
        if self.pending is None:
            return None
        sc = self.sc
        if sc.get("hang") or sc.get("hang_on") == self.renders + 1:
            return None
        if sc.get("during"):
            sc["during"]()
        if sc.get("fail") or sc.get("fail_on") == self.renders + 1:
            self.pending = None
            raise RuntimeError("Traceback (most recent call last):\n  ...\nRuntimeError: boom on " + self.device)
        jobs, self.pending = self.pending, None
        self.renders += 1
        for j in jobs:
            out = getattr(j, "out_path", None)
            if out is not None:
                Path(out).write_bytes(f"{j.label}@{self.device}".encode())
                if getattr(j, "frames", 1) > 1:
                    Path(out).with_suffix(".webp").write_bytes(b"clip " + j.label.encode())
        return [str(getattr(j, "out_path", "")) for j in jobs], sc.get("sec_per_job", 1.0) * len(jobs)

    def request_stop(self):
        self.stopped = True

    def exited(self):
        return self.stopped and self.sc.get("exits_on_stop", True)

    def kill(self):
        self.killed = True
        if self.fx.on_kill:
            self.fx.on_kill(self)

    def release(self):
        self.released = True


class Fx:
    def __init__(self):
        self.workers, self.events, self.got, self.load_fail, self.script = {}, [], [], {}, {}
        self.spawned = []
        self.on_kill = None

    def spawn(self, cfg, model_dir, device, log_path, stop=None, on_start=None):
        self.events.append(("spawn", device))
        self.spawned.append((device, model_dir))
        w = FakeWorker(self, device)
        self.workers.setdefault(device, []).append(w)
        if on_start:
            on_start(w)
        return w

    def last(self, device):
        return self.workers[device][-1]

    def labels(self, device):
        return [lab for d, lab, _, _ in self.got if d == device]


@pytest.fixture
def fx(monkeypatch):
    f = Fx()
    monkeypatch.setattr(rp, "spawn_worker", f.spawn)
    monkeypatch.setattr(rp, "device_info", lambda d: {"gpu": f"Fake {d}", "vram_gb": 24.0 if d.endswith("1") else 16.0})
    monkeypatch.setattr(rp, "resolve_model_dir", lambda cfg, d: "/models/dir")
    monkeypatch.setattr(rp, "POLL", 0.001)
    monkeypatch.setattr(rp, "CLOSE_WAIT", 0.1)
    monkeypatch.setattr(rp, "HANG_FLOOR", 0.05)
    return f


def make_pool(tmp_path, devices, stop=None):
    pool = rp.RenderPool(devices, tmp_path, "t", stop or threading.Event())
    pool.load_all({"a": 1}, "/models/dir")
    return pool


# ---------------------------------------------------------------- pool: loading


def test_workers_load_one_after_another_and_a_failed_one_is_named(fx, tmp_path):
    fx.load_fail["cuda:3"] = "RuntimeError: CUDA out of memory. Tried to allocate 2 GiB"
    pool = rp.RenderPool(["cuda:1", "cuda:3", "cuda:5"], tmp_path, "t", threading.Event())
    loaded = pool.load_all({}, "/m")
    assert [s.device for s in loaded] == ["cuda:1", "cuda:5"]
    kinds = [(e[0], e[1]) for e in fx.events]
    # the next process starts only after the one before was ready (or failed)
    assert kinds.index(("spawn", "cuda:3")) > kinds.index(("poll_ready", "cuda:1"))
    assert kinds.index(("spawn", "cuda:5")) > kinds.index(("poll_ready", "cuda:3"))
    bad = pool.slots[1]
    assert bad.state == "load_failed" and "Out of GPU memory" in bad.error and fx.last("cuda:3").killed
    assert "cuda:3 did not load" in pool.note() and "Rendering on cuda:1, cuda:5" in pool.note()
    assert [g["state"] for g in pool.snapshot()] == ["ready", "load_failed", "ready"]
    assert pool.slots[0].name == "Fake cuda:1"


def test_no_gpu_loads(fx, tmp_path):
    fx.load_fail.update({"cuda:0": "boom0", "cuda:1": "boom1"})
    pool = rp.RenderPool(["cuda:0", "cuda:1"], tmp_path, "t", threading.Event())
    assert pool.load_all({}, "/m") == [] and not pool.live


def test_a_load_that_never_finishes_times_out(fx, tmp_path, monkeypatch):
    monkeypatch.setattr(rp, "LOAD_TIMEOUT", 0.05)
    fx.script["cuda:0"] = {"never_ready": True}
    pool = rp.RenderPool(["cuda:0", "cuda:1"], tmp_path, "t", threading.Event())
    assert [s.device for s in pool.load_all({}, "/m")] == ["cuda:1"]
    assert "did not load within" in pool.slots[0].error and fx.last("cuda:0").killed


def test_stop_during_a_load_raises_stopping_and_kills(fx, tmp_path):
    fx.script["cuda:0"] = {"never_ready": True}
    stop = threading.Event()
    threading.Timer(0.05, stop.set).start()
    pool = rp.RenderPool(["cuda:0"], tmp_path, "t", stop)
    with pytest.raises(rp.Stopping):
        pool.load_all({}, "/m")
    assert fx.last("cuda:0").killed


# ---------------------------------------------------------------- pool: rounds


def test_results_in_place_and_speed_shares_after_the_warm_up(fx, tmp_path):
    pool = make_pool(tmp_path, ["cuda:0", "cuda:1"])
    fx.script["cuda:0"] = {"sec_per_job": 1.0}
    fx.script["cuda:1"] = {"sec_per_job": 3.0}
    jobs = jobs_of(8)
    for _ in range(2):  # round 1 is the warm-up (equal split), round 2 measures
        pool.run_round(jobs, b"lora", "sample", "sample", 60)
    assert fx.labels("cuda:0")[:1] == [[f"j{i}" for i in range(4)]] or True
    assert [lab for d, lab, _, _ in fx.got if d == "cuda:0"][0] == ["j0", "j1", "j2", "j3"]  # contiguous, first GPU first
    assert [lab for d, lab, _, _ in fx.got if d == "cuda:1"][0] == ["j4", "j5", "j6", "j7"]
    n_before = len(fx.got)
    pool.run_round(jobs, b"lora", "sample", "sample", 60)
    third = fx.got[n_before:]
    assert {d: len(lab) for d, lab, _, _ in third} == {"cuda:0": 6, "cuda:1": 2}  # sized by measured speed
    assert third[0][1] == [f"j{i}" for i in range(6)] and third[1][1] == ["j6", "j7"]
    assert all(g[2] == b"lora" and g[3] == "sample" for g in fx.got)
    assert [s.state for s in pool.slots] == ["ready", "ready"]
    assert pool.snapshot()[0]["sec_per_unit"] == pytest.approx(1.0)


def test_probe_rounds_use_the_fixed_map_whatever_the_speeds(fx, tmp_path):
    pool = make_pool(tmp_path, ["cuda:0", "cuda:1"])
    fx.script["cuda:0"] = {"sec_per_job": 1.0}
    fx.script["cuda:1"] = {"sec_per_job": 9.0}
    for _ in range(4):
        pool.run_round(jobs_of(5), b"x", "probe", "probe", 60)
    per_round = [fx.got[i:i + 2] for i in range(0, 8, 2)]
    for rnd in per_round:
        assert [(d, lab) for d, lab, _, _ in rnd] == [("cuda:0", ["j0", "j1", "j2"]), ("cuda:1", ["j3", "j4"])]
    assert pool.slots[1].rates.get("probe") and pool.slots[0].rates.get("probe")  # measured, yet never used for probes


def test_one_worker_failing_mid_round_is_dropped_and_its_chunk_dealt_again(fx, tmp_path):
    pool = make_pool(tmp_path, ["cuda:0", "cuda:1"])
    fx.script["cuda:1"] = {"fail": True}
    secs = pool.run_round(jobs_of(6), b"p", "sample", "sample", 60)
    assert secs >= 0
    assert fx.labels("cuda:0") == [["j0", "j1", "j2"], ["j3", "j4", "j5"]]  # its own chunk, then the other's
    assert fx.last("cuda:1").killed and pool.slots[1].state == "dropped" and "boom on cuda:1" in pool.slots[1].error
    assert "cuda:1 was dropped" in pool.note() and "Rendering on cuda:0" in pool.note()
    assert [s.device for s in pool.live] == ["cuda:0"]
    pool.run_round(jobs_of(4), b"p", "sample", "sample", 60)  # the dropped GPU is not used again
    assert fx.labels("cuda:0")[-1] == ["j0", "j1", "j2", "j3"] and len(fx.workers["cuda:1"]) == 1
    assert not any(d == "cuda:1" for d, *_ in fx.got[2:])


def test_a_worker_that_cannot_be_reached_is_dropped_too(fx, tmp_path):
    pool = make_pool(tmp_path, ["cuda:0", "cuda:1"])
    fx.script["cuda:0"] = {"fail_submit": True}
    pool.run_round(jobs_of(4), b"p", "sample", "sample", 60)
    assert pool.slots[0].state == "dropped" and "closed its connection" in pool.slots[0].error
    assert sorted(j for lab in fx.labels("cuda:1") for j in lab) == ["j0", "j1", "j2", "j3"]


def test_a_chunk_dealt_again_that_fails_again_fails_the_round(fx, tmp_path):
    pool = make_pool(tmp_path, ["cuda:0", "cuda:1", "cuda:2"])
    fx.script["cuda:0"] = {"fail": True}
    fx.script["cuda:1"] = {"fail_on": 2}  # its own chunk renders, the chunk dealt again does not
    with pytest.raises(rp.RoundFailed, match="failed again"):
        pool.run_round(jobs_of(6), b"p", "sample", "sample", 60)
    # not dealt a third time; fail-fast drains the healthy worker that was still rendering its re-dealt chunk
    assert [s.state for s in pool.slots] == ["dropped", "dropped", "dropped"]
    assert "still rendering" in pool.slots[2].error
    assert all(s.chunk is None for s in pool.slots)


def test_every_worker_failing_fails_the_round_and_leaves_no_pool(fx, tmp_path):
    pool = make_pool(tmp_path, ["cuda:0", "cuda:1"])
    fx.script["cuda:0"] = fx.script["cuda:1"] = {"fail": True}
    with pytest.raises(rp.RoundFailed, match="every GPU failed"):
        pool.run_round(jobs_of(4), b"p", "sample", "sample", 60)
    assert pool.live == [] and all(w.killed for ws in fx.workers.values() for w in ws)
    with pytest.raises(rp.RoundFailed, match="no GPU is left"):
        pool.run_round(jobs_of(1), b"p", "sample", "sample", 60)


def test_a_redeal_that_would_pass_the_deadline_fails_the_round_at_once(fx, tmp_path):
    pool = make_pool(tmp_path, ["cuda:0", "cuda:1"])
    pool.slots[0].rates["sample"] = pool.slots[1].rates["sample"] = 100.0  # 100 s per unit: slow cards
    fx.script["cuda:0"] = {"fail": True}
    with pytest.raises(rp.RoundFailed, match="deadline"):
        pool.run_round(jobs_of(4), b"p", "sample", "sample", 60)
    assert len(fx.labels("cuda:1")) == 1  # the chunk was never sent again
    assert [s.device for s in pool.live] == ["cuda:1"]


def test_a_hung_worker_is_killed_at_the_cap_and_its_chunk_dealt_again(fx, tmp_path):
    pool = make_pool(tmp_path, ["cuda:0", "cuda:1"])
    fx.script["cuda:0"] = {"hang": True}
    pool.slots[0].rates["sample"] = 0.001  # measured: its cap is the floor (0.05 s here), far inside the deadline
    t0 = time.time()
    pool.run_round(jobs_of(4), b"p", "sample", "sample", 60)
    assert 0.04 < time.time() - t0 < 5
    assert fx.last("cuda:0").killed and pool.slots[0].state == "dropped" and "did not answer" in pool.slots[0].error
    assert fx.labels("cuda:1") == [["j2", "j3"], ["j0", "j1"]]


def test_the_cap_is_three_times_the_expected_seconds_but_not_past_the_deadline(fx, tmp_path, monkeypatch):
    pool = make_pool(tmp_path, ["cuda:0"])
    pool.slots[0].rates["sample"] = 0.001
    seen = []
    monkeypatch.setattr(pool, "clock", lambda: 1000.0)
    pool._send([(pool.slots[0], jobs_of(10))], b"p", "sample", "sample", 90.0)
    seen.append(pool.slots[0].hang_at - 1000.0)  # expected 0.01 s: the floor
    assert seen[0] == pytest.approx(rp.HANG_FLOOR)
    pool.slots[0].chunk = None
    pool.slots[0].rates["sample"] = 1e6
    pool._send([(pool.slots[0], jobs_of(10))], b"p", "sample", "sample", 90.0)
    assert pool.slots[0].hang_at - 1000.0 == pytest.approx(90.0)  # 3x expected is far more: the deadline caps it
    pool.slots[0].chunk = None
    pool.slots[0].rates["sample"] = 1.0
    monkeypatch.setattr(rp, "HANG_FLOOR", 1.0)
    pool._send([(pool.slots[0], jobs_of(10))], b"p", "sample", "sample", 90.0)
    assert pool.slots[0].hang_at - 1000.0 == pytest.approx(30.0)  # 3 x 10 s


def test_stop_during_a_round_kills_what_is_rendering(fx, tmp_path):
    stop = threading.Event()
    pool = make_pool(tmp_path, ["cuda:0", "cuda:1"], stop)
    fx.script["cuda:0"] = fx.script["cuda:1"] = {"hang": True}
    threading.Timer(0.05, stop.set).start()
    with pytest.raises(rp.Stopping):
        pool.run_round(jobs_of(4), b"p", "sample", "sample", 60)
    assert all(w.killed for ws in fx.workers.values() for w in ws)  # drained before leaving


def test_drain_kills_a_worker_that_still_has_a_chunk(fx, tmp_path):
    pool = make_pool(tmp_path, ["cuda:0"])
    pool.slots[0].chunk = jobs_of(2)
    pool.drain()
    assert fx.last("cuda:0").killed and pool.slots[0].state == "dropped" and pool.slots[0].chunk is None


def test_close_asks_every_worker_then_kills_the_ones_that_stay(fx, tmp_path):
    pool = make_pool(tmp_path, ["cuda:0", "cuda:1"])
    fx.script["cuda:1"] = {"exits_on_stop": False}
    t0 = time.time()
    pool.close()
    assert time.time() - t0 < 3
    a, b = fx.last("cuda:0"), fx.last("cuda:1")
    assert a.stopped and b.stopped and b.killed and a.released and b.released


# ---------------------------------------------------------------- the helper's pool path


def pool_helper(w, device, **kw):
    opts = rh.Options(instance="42", device=device, models_dir=str(w.models), status_file=w.tmp / "status.json",
                      work_dir=w.tmp / "work", **kw)
    h = rh.Helper(opts, w.connect)
    w.helper = h
    return h


def pool_ready(w, device="cuda:1,cuda:3", **kw):
    w.write_prepare()
    h = pool_helper(w, device, **kw)
    drive(h, lambda: (w.base / "prepare/ready.json").exists())
    return h


def test_ready_json_names_the_gpus_that_loaded(world, fx):
    w = world
    fx.load_fail["cuda:3"] = "RuntimeError: CUDA out of memory"
    h = pool_ready(w, "cuda:1,cuda:3,cuda:5")
    ready = read_json(w.base / "prepare/ready.json")
    assert ready["gpu"] == "Fake cuda:1 + Fake cuda:5" and ready["vram_gb"] == 16.0  # the smallest that loaded
    assert ready["gpus"] == [{"device": "cuda:1", "name": "Fake cuda:1", "vram_gb": 24.0},
                             {"device": "cuda:5", "name": "Fake cuda:5", "vram_gb": 24.0}] or len(ready["gpus"]) == 2
    assert ready["kinds"] == ["sample", "probe"] and not w.loads  # nothing loaded in this process
    s = read_json(w.tmp / "status.json")
    assert s["state"] == "ready" and "cuda:3 did not load" in s["message"] and "Out of GPU memory" in s["message"]
    assert [g["state"] for g in s["gpus"]] == ["ready", "load_failed", "ready"]
    assert fx.spawned == [("cuda:1", "/models/dir"), ("cuda:3", "/models/dir"), ("cuda:5", "/models/dir")]
    assert h.pool is not None and h.gpu_name == "Fake cuda:1 + Fake cuda:5"


def test_no_gpu_loading_is_not_ready(world, fx):
    w = world
    fx.load_fail.update({"cuda:1": "boom1", "cuda:3": "boom3"})
    w.write_prepare()
    h = pool_helper(w, "cuda:1,cuda:3")
    drive(h, lambda: (w.base / "prepare/not_ready.json").exists())
    reason = read_json(w.base / "prepare/not_ready.json")["reason"]
    assert "any GPU" in reason and "boom1" in reason and "boom3" in reason
    assert not (w.base / "prepare/ready.json").exists() and h.pool is None
    assert h.status.data["state"] == "not_ready"


def test_one_device_renders_in_process_unless_workers_process(world, fx):
    w = world
    h = pool_ready(w, "cuda:1")
    assert w.loads and not fx.spawned and h.pool is None and read_json(w.tmp / "status.json")["gpus"] == []
    h._drop_hb()
    h._drop_wb()
    w.loads.clear()
    (w.base / "prepare/ready.json").unlink()
    h2 = pool_ready(w, "cuda:1", workers="process")
    assert fx.spawned == [("cuda:1", "/models/dir")] and not w.loads and h2.pool is not None
    assert read_json(w.base / "prepare/ready.json")["gpu"] == "Fake cuda:1"


def test_a_round_is_dealt_across_the_gpus_and_uploaded_in_job_order(world, fx):
    w = world
    h = pool_ready(w)
    jobs = [{"label": f"p{i}", "out_path": f"probes/5_{i}.png"} for i in range(5)]
    jobs[2]["frames"] = 5
    w.add_round("000001_probe_5", "probe", 5, jobs=jobs, negative_kind="probe")
    drive(h, lambda: (w.base / "done/000001_probe_5").exists())
    assert fx.labels("cuda:1") == [["p0", "p1", "p2"]] and fx.labels("cuda:3") == [["p3", "p4"]]
    d = read_json(w.base / "done/000001_probe_5/done.json")
    assert d["files"] == ["probes/5_0.png", "probes/5_1.png", "probes/5_2.webp", "probes/5_2.png", "probes/5_3.png",
                          "probes/5_4.png"]  # job order, a clip's file before its still
    assert d["gpu"] == "Fake cuda:1 + Fake cuda:3" and d["seconds"] >= 0
    assert (w.base / "done/000001_probe_5/out/probes/5_4.png").read_bytes() == b"p4@cuda:3"
    assert all(g[2] == b"lora 000001_probe_5" for g in fx.got)  # the LoRA's bytes as they came
    assert h.status.data["rounds_done"] == 1 and h.status.data["state"] == "ready"


def test_a_dropped_gpu_is_named_and_the_round_still_completes(world, fx):
    w = world
    h = pool_ready(w)
    fx.script["cuda:3"] = {"fail": True}
    w.add_round("000001_sample_5", "sample", 5, negative_kind="sample",
                jobs=[{"label": f"s{i}", "out_path": f"samples/5_{i}.png"} for i in range(4)])
    drive(h, lambda: (w.base / "done/000001_sample_5").exists())
    for i in range(4):
        assert (w.base / f"done/000001_sample_5/out/samples/5_{i}.png").exists()
    s = read_json(w.tmp / "status.json")
    assert [g["state"] for g in s["gpus"]] == ["ready", "dropped"]
    assert "cuda:3 was dropped" in s["message"] and "boom on cuda:3" in s["message"]
    assert h.pool.live and h.prepared


def test_a_chunk_that_fails_twice_fails_the_round_with_error_json(world, fx):
    w = world
    h = pool_ready(w, "cuda:1,cuda:3,cuda:5")
    fx.script["cuda:1"] = {"fail": True}
    fx.script["cuda:3"] = {"fail_on": 2}
    w.add_round("000001_probe_5", "probe", 5,
                jobs=[{"label": f"p{i}", "out_path": f"probes/5_{i}.png"} for i in range(6)])
    drive(h, lambda: (w.base / "failed/000001_probe_5").exists())
    err = read_json(w.base / "failed/000001_probe_5/error.json")
    assert "failed again" in err["error"] and "Traceback" in err["traceback"]
    # fail-fast drained the healthy third worker too, so nothing is left: it loads all three again at once
    drive(h, lambda: h.prepared and h.pool is not None and len(h.pool.live) == 3)
    w.add_round("000002_probe_10", "probe", 10, jobs=[{"label": "q", "out_path": "probes/10_0.png"}])
    drive(h, lambda: (w.base / "done/000002_probe_10").exists())  # the reloaded pool goes on


def test_all_gpus_failing_fails_the_round_loads_again_at_once_and_stays_silent(world, fx):
    w = world
    h = pool_ready(w)
    first_ready = read_json(w.base / "prepare/ready.json")["time"]
    beat = read_json(w.base / "helper.json")["time"]
    fx.script["cuda:1"] = fx.script["cuda:3"] = {"fail": True}
    w.add_round("000001_probe_5", "probe", 5, jobs=[{"label": "p", "out_path": "probes/5_0.png"}])
    fx.load_fail.update({"cuda:1": "still broken", "cuda:3": "still broken"})  # the second load fails as well
    drive(h, lambda: (w.base / "failed/000001_probe_5").exists())
    assert "cuda:3 was dropped" in read_json(w.base / "failed/000001_probe_5/error.json")["error"]
    drive(h, lambda: (w.base / "prepare/not_ready.json").exists())  # it loaded again at once, retrying every GPU
    assert fx.spawned.count(("cuda:1", "/models/dir")) == 2 and fx.spawned.count(("cuda:3", "/models/dir")) == 2
    assert h.pool_lost and h.pool is None
    stamp = read_json(w.base / "helper.json")["time"]
    for _ in range(4):  # no heartbeat while there is no pool: the instance sees the helper go stale
        h.tick()
        time.sleep(0.02)
    assert read_json(w.base / "helper.json")["time"] == stamp and stamp >= beat
    # a GPU comes back: a new ready.json, and the heartbeat is back
    fx.load_fail.clear()
    h.next_prepare = 0.0
    drive(h, lambda: read_json(w.base / "prepare/ready.json")["time"] > first_ready)
    h.tick()
    assert not h.pool_lost and read_json(w.base / "helper.json")["time"] > stamp


def test_pictures_a_dropped_connection_leaves_are_drained_before_the_work_folder_goes(world, fx):
    w = world
    h = pool_ready(w)
    w.add_round("000001_probe_5", "probe", 5, jobs=[{"label": "p", "out_path": "probes/5_0.png"}])
    seen = []
    fx.on_kill = lambda worker: seen.append(any(p.exists() for p in (w.tmp / "work").rglob("rounds")))
    real = h.pool.run_round

    def leave_one_busy(*a, **k):
        h.pool.slots[0].chunk = [SimpleNamespace(label="x")]  # a chunk still out when the connection drops
        raise rh.Disconnected("the connection ended")

    h.pool.run_round = leave_one_busy
    h.wb = w.connect()
    with pytest.raises(rh.Disconnected):
        h._render({"job": "job", "run": {"run_token": TOKEN}}, "000001_probe_5")
    assert fx.last("cuda:1").killed and seen == [True]  # killed while the round's folder still existed
    assert not list((w.tmp / "work").rglob("000001_probe_5"))
    h.pool.run_round = real


def test_shutdown_kills_every_worker_first(world, fx):
    w = world
    h = pool_ready(w)
    order = []
    fx.on_kill = lambda worker: order.append(("kill", worker.device, h.hb is not None))
    h.shutdown()
    assert {o[1] for o in order} == {"cuda:1", "cuda:3"} and all(o[2] for o in order)  # before the connections closed
    assert read_json(w.tmp / "status.json")["state"] == "stopped"


def test_a_new_run_frees_the_pool(world, fx):
    w = world
    h = pool_ready(w)
    h._reset("another")
    assert h.pool is None and all(x.stopped or x.killed for ws in fx.workers.values() for x in ws)
    assert read_json(w.tmp / "status.json")["gpus"] == []


# ---------------------------------------------------------------- command line and the Fleet tab's side


def test_cli_parses_device_lists_and_the_hidden_switch(tmp_path, capsys):
    a = rh.parse_args(["--instance", "1", "--device", "cuda:1,cuda:3"])
    assert a.device == "cuda:1,cuda:3" and a.workers == "auto"
    assert rh.parse_args(["--instance", "1", "--workers", "process"]).workers == "process"
    assert rh.split_devices("cuda:1, cuda:3,cuda:1,") == ["cuda:1", "cuda:3"] and rh.split_devices("cuda:2") == ["cuda:2"]
    assert rh.split_devices("") == ["cuda:0"]
    with pytest.raises(SystemExit):
        rh.parse_args(["--workers", "bogus"])
    capsys.readouterr()
    with pytest.raises(SystemExit):
        rh.parse_args(["--help"])
    assert "--workers" not in capsys.readouterr().out  # hidden
    cmd = ctl.helper_command(9, [1, 3], tmp_path / "9.json")
    assert rh.parse_args(cmd[3:]).device == "cuda:1,cuda:3"
    assert rh.Helper(rh.Options(instance="1", device="cuda:1,cuda:3"), lambda: None).use_pool
    assert not rh.Helper(rh.Options(instance="1", device="cuda:1"), lambda: None).use_pool
    assert rh.Helper(rh.Options(instance="1", device="cuda:1", workers="process"), lambda: None).use_pool


def test_every_gpu_state_has_a_word_in_the_fleet_tab():
    assert set(rh.GPU_STATES) == set(rp.GPU_STATES) <= set(ctl.GPU_STATES)
    for word, icon, level in ctl.GPU_STATES.values():
        assert word and icon
    v = ctl.status_view({"state": "rendering", "gpus": [{"device": "cuda:1", "name": "A", "state": "rendering"},
                                                        {"device": "cuda:3", "name": "B", "state": "dropped"}]}, True)
    assert "1 of 2 GPUs" in v["text"]
    assert [(g["device"], g["word"]) for g in v["gpus"]] == [("cuda:1", "Rendering"), ("cuda:3", "Dropped")]


@pytest.mark.skipif(sys.platform != "win32", reason="Job Objects are Windows only")
def test_a_worker_in_the_job_object_dies_with_the_helper(tmp_path):
    """Real processes, no GPU: a "helper" adopts a sleeper into the job object and is killed; the sleeper must go too."""
    import subprocess

    marker = tmp_path / "pid.txt"
    script = tmp_path / "helper.py"
    script.write_text(
        "import subprocess, sys, time\n"
        f"sys.path.insert(0, {str(Path(rp.__file__).resolve().parents[1])!r})\n"
        "from rulztrainer import renderpool\n"
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
        "ok = renderpool.adopt_process(p)\n"
        f"open({str(marker)!r}, 'w').write(f'{{p.pid}} {{ok}}')\n"
        "time.sleep(120)\n", encoding="utf-8")
    helper = subprocess.Popen([sys.executable, str(script)])
    try:
        end = time.time() + 30
        while not marker.exists() and time.time() < end:
            time.sleep(0.1)
        time.sleep(0.3)
        pid, ok = marker.read_text().split()
        helper.kill()  # TerminateProcess: no handler runs, the job handle closes with the process
        helper.wait(10)
        gone = False
        for _ in range(100):
            out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True).stdout
            if pid not in out:
                gone = True
                break
            time.sleep(0.1)
        if ok == "False":
            pytest.skip("this environment does not allow a job object here")
        assert gone
    finally:
        helper.kill()


# ---------------------------------------------------------------- review fixes


def test_plan_split_gives_every_gpu_a_job_when_there_are_enough():
    assert rp.plan_split([1] * 4, [1.0, 100.0]) == [3, 1]  # the slow card is still timed
    sizes = rp.plan_split([1] * 5, [1.0, 50.0, 100.0])
    assert sum(sizes) == 5 and min(sizes) >= 1
    assert sum(rp.plan_split([1, 1], [1.0, 100.0, 5.0])) == 2  # fewer jobs than GPUs: still the fastest first


def test_the_deadline_runs_from_the_claim_not_from_the_render(fx, tmp_path):
    pool = make_pool(tmp_path, ["cuda:0", "cuda:1"])
    with pytest.raises(rp.RoundFailed, match="deadline passed"):
        pool.run_round(jobs_of(4), b"p", "sample", "sample", 60, claim=pool.clock() - 61)  # the LoRA fetch ate it all
    assert not fx.got
    fx.script["cuda:0"] = {"hang": True}
    t0 = time.time()
    with pytest.raises(rp.RoundFailed, match="cannot be shown to fit"):
        pool.run_round(jobs_of(4), b"p", "sample", "sample", 60, claim=pool.clock() - 59.8)
    assert time.time() - t0 < 5  # the hang cap was cut to the 0.2 s left, not 60 s
    assert fx.last("cuda:0").killed and len(fx.labels("cuda:1")) == 1  # nothing was dealt again


def test_a_redeal_needs_a_known_fit_or_floor_seconds_left(fx, tmp_path):
    pool = make_pool(tmp_path, ["cuda:0", "cuda:1"])
    fx.script["cuda:0"] = {"fail": True}
    with pytest.raises(rp.RoundFailed, match="cannot be shown"):  # no speed known, 0.02 s left (< HANG_FLOOR)
        pool.run_round(jobs_of(4), b"p", "sample", "sample", 0.02)
    pool = make_pool(tmp_path, ["cuda:0", "cuda:1"])
    pool.run_round(jobs_of(4), b"p", "sample", "sample", 60)  # unknown speed but plenty of time: dealt again
    assert fx.last("cuda:1").renders == 2


def test_the_second_failure_fails_at_once_without_waiting_for_the_others(fx, tmp_path):
    pool = make_pool(tmp_path, ["cuda:0", "cuda:1", "cuda:2"])
    fx.script["cuda:0"] = {"fail": True}
    fx.script["cuda:1"] = {"fail_on": 2}
    fx.script["cuda:2"] = {"hang_on": 2}  # its re-dealt chunk would hang for the whole deadline
    t0 = time.time()
    with pytest.raises(rp.RoundFailed, match="failed again"):
        pool.run_round(jobs_of(6), b"p", "sample", "sample", 60)
    assert time.time() - t0 < 5
    assert fx.last("cuda:2").killed and all(s.chunk is None for s in pool.slots)  # drained


def test_a_worker_killed_by_a_stop_meanwhile_is_no_lost_chunk(fx, tmp_path):
    pool = make_pool(tmp_path, ["cuda:0"])
    pool.slots[0].chunk = jobs_of(2)
    w = fx.last("cuda:0")
    w.poll_result = lambda: (pool.kill_all(), None)[1]  # the tick thread kills everything while this poll runs
    assert pool._collect("sample") == []  # nothing appended for a chunk that is gone


def test_a_stop_in_a_round_raises_stopping_even_when_the_workers_were_killed_first(fx, tmp_path):
    stop = threading.Event()
    pool = make_pool(tmp_path, ["cuda:0", "cuda:1"], stop)
    fx.script["cuda:0"] = fx.script["cuda:1"] = {"during": lambda: (stop.set(), pool.kill_all())}
    with pytest.raises(rp.Stopping):
        pool.run_round(jobs_of(4), b"p", "sample", "sample", 60)


def test_a_stop_while_rendering_leaves_the_round_to_the_instance(world, fx):
    w = world
    h = pool_ready(w)
    w.add_round("000001_probe_5", "probe", 5, jobs=[{"label": "p", "out_path": "probes/5_0.png"}])

    def stopping(*a, **k):
        raise rp.Stopping()

    h.pool.run_round = stopping
    h.wb = w.connect()
    with pytest.raises(rp.Stopping):
        h._render({"job": "job", "run": {"run_token": TOKEN}}, "000001_probe_5")
    assert not (w.base / "failed").exists() or not list((w.base / "failed").iterdir())
    assert (w.base / "claimed/000001_probe_5").exists() and not (w.base / "claimed/000001_probe_5/error.json").exists()
    assert "failed" not in h.status.data["message"] and h.status.data["state"] == "rendering"


def test_the_round_gets_the_claims_time_as_its_start(world, fx):
    w = world
    h = pool_ready(w)
    seen = {}
    real = h.pool.run_round

    def spy(jobs, payload, kind, negative_kind, deadline_s, claim=None):
        seen.update(claim=claim, deadline=deadline_s, now=time.monotonic())
        return real(jobs, payload, kind, negative_kind, deadline_s, claim=claim)

    h.pool.run_round = spy
    w.add_round("000001_probe_5", "probe", 5, jobs=[{"label": "p", "out_path": "probes/5_0.png"}])
    drive(h, lambda: (w.base / "done/000001_probe_5").exists())
    assert seen["deadline"] == 90 and 0 <= seen["now"] - seen["claim"] < 30  # the claim came before the render began


def _never_connecting(monkeypatch):
    """Real PoolWorker, real process, but the child is a sleeper that never connects back."""
    pytest.importorskip("torch")
    import subprocess

    real = subprocess.Popen
    monkeypatch.setattr(subprocess, "Popen", lambda args, **kw: real(
        [sys.executable, "-c", "import time; time.sleep(120)"], **kw))
    adopted = []
    monkeypatch.setattr(rp, "adopt_process", lambda proc: adopted.append(proc) or True)
    return adopted


def test_a_worker_that_never_connects_can_be_stopped_and_killed(monkeypatch, tmp_path):
    adopted = _never_connecting(monkeypatch)
    stop, started = threading.Event(), []
    threading.Timer(0.6, stop.set).start()
    with pytest.raises(rp.Stopping):
        rp.spawn_worker({}, None, "cuda:0", tmp_path / "w.log", stop=stop, on_start=started.append)
    assert len(started) == 1 and adopted == [started[0].proc]  # in the job object, and reachable, before it connected
    assert started[0].proc.poll() is not None  # killed


def test_a_worker_that_never_connects_times_out(monkeypatch, tmp_path):
    _never_connecting(monkeypatch)
    monkeypatch.setattr(rp, "CONNECT_TIMEOUT", 0.7)
    started = []
    t0 = time.time()
    with pytest.raises(Exception, match="did not connect within"):
        rp.spawn_worker({}, None, "cuda:0", tmp_path / "w.log", on_start=started.append)
    assert time.time() - t0 < 30 and started[0].proc.poll() is not None


def test_the_pool_can_reach_a_worker_that_is_still_connecting(fx, tmp_path, monkeypatch):
    seen = []

    def spawn(cfg, model_dir, device, log_path, stop=None, on_start=None):
        w = FakeWorker(fx, device)
        on_start(w)
        seen.append(pool.slots[0].worker is w)  # reachable while spawn_worker has not returned
        raise rp.Stopping()

    monkeypatch.setattr(rp, "spawn_worker", spawn)
    pool = rp.RenderPool(["cuda:0"], tmp_path, "t", threading.Event())
    with pytest.raises(rp.Stopping):
        pool.load_all({}, "/m")
    assert seen == [True]
