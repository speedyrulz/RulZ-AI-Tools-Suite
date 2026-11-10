"""scripts/vast/rulz_fleet.py: several instances as one fleet. Finding them, recording their runs, the queue that
hands the next run to the first idle instance, a race split over instances, the per-kind download rules, the watches
and the wait that reports them. No ssh and no GPU: the instances are fakes that answer the agent's commands, and
the agent's own new commands run here on a fake install folder."""
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "vast"))

import rulz_agent as agent  # noqa: E402
import rulz_fleet as fleet_module  # noqa: E402
import rulz_remote as remote  # noqa: E402
from rulz_fleet import Fleet, Refused  # noqa: E402

from rulztrainer.config import TrainConfig  # noqa: E402


# ---------------------------------------------------------------- fakes


class Clock:
    def __init__(self):
        self.t = 1_000_000.0

    def __call__(self):
        self.t += 0.001  # no two moments alike, as on a real clock
        return self.t


class Machine:
    """An instance: its jobs, what runs, and what it answers to the agent's commands."""

    def __init__(self, iid, datasets=("set",)):
        self.id, self.datasets = iid, list(datasets)
        self.jobs: dict[str, dict] = {}
        self.running = None
        self.down = False
        self.refuse = ""  # start's refusal
        self.calls: list[tuple] = []
        self.configs: dict[str, dict] = {}
        self.disk, self.temp = 80.0, 60
        self.n = 0

    def run(self, name, **status):
        self.n += 1
        self.jobs[name] = {"job": name, "run_id": f"{name}-{self.n}", "step": 0, "total": 100, "elapsed": 10.0,
                           "avg_loss": 0.3, **status}
        self.running = name
        return self.jobs[name]

    def finish(self, state="done", **status):
        job = self.jobs[self.running]
        job.update(step=job["total"] if state == "done" else job["step"], **status)
        job["final"] = state
        self.running = None

    def snapshot(self):
        jobs = [{k: v for k, v in dict(j, running=name == self.running,
                                       state="running" if name == self.running else j.get("final", "done")).items()
                 if k != "final"} for name, j in self.jobs.items()]
        snap = {"webui_running": True, "disk_free_gb": self.disk, "disk_total_gb": 200.0, "datasets": self.datasets,
                "gpus": [{"name": "RTX 5090", "memory_used_mb": 9000, "memory_total_mb": 32000, "utilization": 97,
                          "temperature": self.temp}],
                "job_running": self.running is not None, "queue": {"paused": False, "waiting": [], "recent": []},
                "jobs": jobs}
        if self.jobs:
            name = self.running or list(self.jobs)[-1]
            job = self.jobs[name]
            snap["job"] = {**{k: v for k, v in job.items() if k != "final"}, "loss": job["avg_loss"], "lr": 1e-4}
            snap["series"] = {"losses": [[job["step"], job["avg_loss"]]], "probes": []}
        return snap

    def answer(self, args, config):
        self.calls.append(tuple(args))
        if self.down:
            return 255, b"", b""
        command = args[0]
        if command == "snapshot":
            return 0, self.snapshot(), b""
        if command == "start":
            name = args[args.index("--job") + 1]
            if self.refuse:
                return 1, b"", f"[refused] {self.refuse}".encode()
            if "--dry-run" in args:
                return 0, {"job": name, "dry_run": True, "steps": 100, "arch": "krea2", "changed": {}}, b""
            self.configs[name] = json.loads(config)
            self.run(name)
            return 0, {"job": name, "state": "running"}, b""
        if command == "record":
            job = self.jobs[args[1]]
            return 0, {"job": args[1], "run_id": job["run_id"], "config": {"optimizer": "keel", "steps": job["total"]},
                       "status": {"step": job["step"], "likeness": {"best": job.get("best"), "best_step": 80}},
                       "probes": [{"step": 50, "combined_ema": 40.0}], "losses": [[1, 0.5]]}, b""
        if command == "files":
            kinds = [args[i + 1] for i, a in enumerate(args) if a == "--what"]
            return 0, {"base": "/out", "files": [{"path": f"{args[1]}/{kind}.bin", "size": 5} for kind in kinds]}, b""
        if command == "stop":
            return 0, {"job": self.running, "state": "stopping"}, b""
        if command == "config":
            return 0, {"steps": 50, "optimizer": "adamw"}, b""
        raise AssertionError(f"unexpected {args}")


class FakeRemote:
    def __init__(self, machine):
        self.machine = machine
        self.conn = {"id": machine.id, "host": "h", "port": 1}

    def agent(self, *args, config=None):
        code, out, err = self.machine.answer(args, config)
        return SimpleNamespace(returncode=code, stdout=out if isinstance(out, bytes) else json.dumps(out).encode(),
                               stderr=err)

    def agent_json(self, *args):
        done = self.agent(*args)
        if done.returncode != 0:
            raise SystemExit(done.stderr.decode() or "failed")
        return json.loads(done.stdout)


def listing(machines, **status):
    return [{"id": m.id, "actual_status": status.get(str(m.id), "running"), "gpu_name": "RTX 5090", "dph_total": 0.9,
             "extra_env": [["RULZ_REPO", "x"]], "image_uuid": "vastai/pytorch"} for m in machines]


class World:
    """A fleet over fake machines, with the files it fetched and sent written down."""

    def __init__(self, tmp_path, *ids):
        self.machines = {str(i): Machine(i) for i in ids}
        self.listed = lambda: listing(self.machines.values())
        self.fetched: list[tuple] = []
        self.sent: list[tuple] = []
        self.clock = Clock()
        self.fleet = self.new_fleet(tmp_path)
        self.fleet.discover()

    def new_fleet(self, tmp_path):
        def fetch(target, base, files, dest):
            self.fetched.append((target.machine.id, sorted(files)))
            for name in files:
                (dest / name).parent.mkdir(parents=True, exist_ok=True)
                (dest / name).write_bytes(b"12345")

        return Fleet(home=tmp_path / "fleet", connect=lambda inst: FakeRemote(self.machines[str(inst["id"])]),
                     list_instances=lambda: self.listed(), fetch=fetch,
                     send=lambda target, items, where, hidden=True: self.sent.append(
                         (target.machine.id, [name for _, name in items], where)), now=self.clock)

    def tick(self, seconds=30.0):
        self.clock.t += seconds
        snaps = self.fleet.tick()
        self.fleet.drain()
        return snaps

    def kinds(self, *wanted):
        return [e for e in self.fleet.events() if not wanted or e["kind"] in wanted]

    def starts(self, iid):
        return [c[c.index("--job") + 1] for c in self.machines[str(iid)].calls
                if c[0] == "start" and "--dry-run" not in c]


@pytest.fixture
def world(tmp_path):
    return World(tmp_path, 1, 2)


# ---------------------------------------------------------------- instances


def test_running_rulztrainer_instances_join_and_a_destroyed_one_is_marked_gone(world):
    insts = world.fleet.read()["instances"]
    assert sorted(insts) == ["1", "2"] and insts["1"]["gpu"] == "RTX 5090" and insts["1"]["dph"] == 0.9
    world.machines["2"].run("a")
    world.tick()
    world.listed = lambda: listing([world.machines["1"]])
    world.fleet.discover()
    assert world.fleet.read()["instances"]["2"]["gone"]
    lost = world.fleet.runs(instance=2)[0]
    assert lost["state"] == "lost" and "destroyed" in lost["error"]
    assert [e["kind"] for e in world.kinds("instance_gone", "run_lost")] == ["instance_gone", "run_lost"]
    assert "2" not in world.tick()  # a gone instance is not looked at


def test_a_removed_instance_stays_out_until_it_is_added_again(world):
    world.fleet.remove_instance(2)
    world.fleet.discover()
    assert sorted(world.fleet.read()["instances"]) == ["1"]
    world.fleet.add_instance(2, label="spare")
    assert world.fleet.read()["instances"]["2"]["label"] == "spare"


def test_vast_out_of_reach_is_a_note_not_a_failure(world):
    def nothing():
        raise remote.Unreachable("no api key")

    world.listed = nothing
    assert "no api key" in world.fleet.discover()
    assert world.fleet.overview()["discover_error"] == "no api key"
    assert sorted(world.tick()) == ["1", "2"]


def test_an_instance_out_of_reach_is_said_once_and_keeps_its_last_look(world):
    world.machines["1"].run("a", step=40)
    world.tick()
    world.machines["1"].down = True
    world.tick()
    world.tick()
    assert [e["kind"] for e in world.kinds("instance_down", "instance_up")] == ["instance_down"]
    view = world.fleet.overview()["instances"][0]
    assert view["state"] == "unreachable" and view["disk_free_gb"] == 80.0
    assert world.fleet.runs(instance=1)[0]["state"] == "running"  # nobody knows better
    world.machines["1"].down = False
    world.tick()
    assert world.kinds("instance_down", "instance_up")[-1]["kind"] == "instance_up"
    assert world.fleet.overview()["instances"][0]["state"] == "running"


# ---------------------------------------------------------------- runs on record


def test_a_run_is_recorded_while_it_goes_and_closed_when_it_ends(world):
    machine = world.machines["1"]
    machine.run("a", step=10)
    world.tick()
    rec = world.fleet.runs()[0]
    assert rec["state"] == "running" and rec["job"] == "a" and rec["instance"] == 1 and rec["step"] == 10
    assert rec["config"] == {"optimizer": "keel", "steps": 100}  # fetched once the run was seen
    assert rec["cost"] == pytest.approx(10 / 3600 * 0.9, abs=1e-3)
    machine.finish(best=55.5)
    world.tick()
    rec = world.fleet.runs()[0]
    assert rec["state"] == "done" and rec["step"] == 100 and rec["best"] == 55.5 and rec["ended"]
    assert rec["probes"] == [{"step": 50, "combined_ema": 40.0}]
    assert [e["kind"] for e in world.kinds("run_started", "run_done")] == ["run_started", "run_done"]
    assert world.kinds("run_done")[0]["best"] == 55.5


def test_a_job_name_used_again_is_a_new_run(world):
    machine = world.machines["1"]
    machine.run("a")
    world.tick()
    machine.finish()
    world.tick()
    machine.run("a")
    world.tick()
    assert sorted(r["state"] for r in world.fleet.runs(job="a")) == ["done", "running"]


def test_runs_already_on_an_instance_are_taken_into_the_history_without_events(world):
    machine = world.machines["1"]
    machine.run("old")
    machine.finish("error", error="boom")
    world.tick()
    rec = world.fleet.find_run("old")
    assert rec["state"] == "error" and rec["imported"] and not rec.get("final_pull")
    assert not world.kinds("run_started", "run_failed") and not world.fetched


def test_a_failed_and_a_killed_run_say_so(world):
    world.machines["1"].run("a")
    world.machines["2"].run("b")
    world.tick()
    world.machines["1"].finish("error", error="CUDA out of memory")
    world.machines["2"].finish("ended")
    world.tick()
    by_kind = {e["kind"]: e for e in world.kinds("run_failed", "run_ended")}
    assert by_kind["run_failed"]["error"] == "CUDA out of memory" and by_kind["run_ended"]["job"] == "b"


def test_a_run_can_be_forgotten_but_not_while_it_goes(world):
    world.machines["1"].run("a")
    world.tick()
    with pytest.raises(Refused, match="still going"):
        world.fleet.forget_run("a")
    world.machines["1"].finish()
    world.tick()
    world.fleet.forget_run("a")
    assert world.fleet.runs() == []


# ---------------------------------------------------------------- the queue


def test_a_queued_run_goes_to_the_first_idle_instance_and_the_next_to_whoever_finishes(world):
    world.machines["1"].run("busy1")
    world.machines["2"].run("busy2")
    world.tick()
    for name in ("q1", "q2", "q3"):
        world.fleet.enqueue(name, {"steps": 100})
    world.tick()
    assert world.starts(1) == [] and world.starts(2) == []  # nobody idle
    world.machines["2"].finish()
    world.tick()
    assert world.starts(2) == ["q1"] and world.starts(1) == []
    world.tick()
    assert world.starts(2) == ["q1"]  # it is busy with q1 now
    world.machines["1"].finish()
    world.tick()
    assert world.starts(1) == ["q2"]
    world.machines["2"].finish()
    world.tick()
    assert world.starts(2) == ["q1", "q3"]
    assert world.fleet.read()["queue"]["waiting"] == []
    assert [e["job"] for e in world.kinds("dispatched")] == ["q1", "q2", "q3"]
    assert world.kinds("queue_empty")
    assert world.machines["2"].configs["q1"] == {"steps": 100}


def test_two_idle_instances_take_two_runs_in_one_look(world):
    world.fleet.enqueue("q1", {"steps": 1})
    world.fleet.enqueue("q2", {"steps": 1})
    world.tick()
    assert world.starts(1) == ["q1"] and world.starts(2) == ["q2"]


def test_a_run_for_one_instance_waits_for_it(world):
    world.machines["2"].run("busy")
    world.fleet.enqueue("only2", {"steps": 1}, instance=2)
    world.fleet.enqueue("any", {"steps": 1})
    world.tick()
    assert world.starts(1) == ["any"] and world.starts(2) == []
    world.machines["2"].finish()
    world.tick()
    assert world.starts(2) == ["only2"]


def test_a_paused_queue_and_an_instance_on_hold_start_nothing(world):
    world.fleet.enqueue("q1", {"steps": 1})
    world.fleet.queue_edit("pause")
    world.tick()
    assert world.starts(1) == []
    world.fleet.queue_edit("resume")
    world.fleet.set_instance(1, accept=False)
    world.tick()
    assert world.starts(1) == [] and world.starts(2) == ["q1"]


def test_an_instance_with_its_own_queue_waiting_or_no_webui_is_not_idle(world):
    snap = dict(world.machines["1"].snapshot(), ok=True, polled=world.clock())
    inst = world.fleet.read()["instances"]["1"]
    assert world.fleet.idle(inst, snap)
    assert not world.fleet.idle(inst, dict(snap, webui_running=False))
    assert not world.fleet.idle(inst, dict(snap, queue={"waiting": [{"id": "x", "job": "theirs"}]}))
    assert not world.fleet.idle(inst, dict(snap, ok=False))
    assert not world.fleet.idle(dict(inst, settled=snap["polled"] + 1), snap)  # a look from before our last start


def test_a_refused_run_tries_the_next_instance_and_refused_by_all_is_set_aside(world):
    world.machines["1"].refuse = "Dataset folder not found: datasets/x"
    world.fleet.enqueue("q1", {"steps": 1})
    world.tick()
    assert world.starts(2) == [] and world.fleet.read()["queue"]["waiting"][0]["refused_by"] == ["1"]
    world.machines["2"].refuse = "Dataset folder not found: datasets/x"
    world.tick()
    q = world.fleet.read()["queue"]
    assert q["waiting"] == [] and "Dataset folder not found" in q["recent"][0]["error"]
    failed = world.kinds("dispatch_failed")
    assert len(failed) == 1 and failed[0]["job"] == "q1" and failed[0]["instance"] == 2


def test_a_run_sent_to_an_instance_that_dropped_out_goes_back_to_the_head_of_the_queue(world):
    world.fleet.set_instance(2, enabled=False)
    world.fleet.enqueue("q1", {"steps": 1})
    world.fleet.enqueue("q2", {"steps": 1})
    machine = world.machines["1"]
    answer = machine.answer
    machine.answer = lambda args, config: (255, b"", b"") if args[0] == "start" else answer(args, config)
    world.tick()
    assert [e["job"] for e in world.fleet.read()["queue"]["waiting"]] == ["q1", "q2"]
    assert not world.kinds("dispatch_failed")
    machine.answer = answer
    world.tick()
    assert world.starts(1) == ["q1"]


def test_a_dataset_is_pushed_first_when_the_instance_lacks_it(world, tmp_path):
    (tmp_path / "newset").mkdir()
    (tmp_path / "set").mkdir()
    world.fleet.set_instance(2, enabled=False)
    world.fleet.enqueue("q1", {"steps": 1}, push=[tmp_path / "newset", tmp_path / "set"])
    world.tick()
    assert world.sent == [(1, ["newset"], "datasets")] and world.starts(1) == ["q1"]
    with pytest.raises(Refused, match="Not found here"):
        world.fleet.enqueue("q2", {"steps": 1}, push=[tmp_path / "nowhere"])


def test_queue_names_cannot_repeat_and_entries_move_and_leave(world):
    world.fleet.queue_edit("pause")
    ids = [world.fleet.enqueue(name, {"steps": 1})[0]["id"] for name in ("a", "b", "c")]
    with pytest.raises(Refused, match="Already queued: b"):
        world.fleet.enqueue("b", {"steps": 1})
    world.fleet.queue_edit("move", ids[2], -2)
    assert [e["job"] for e in world.fleet.queue_view()["waiting"]] == ["c", "a", "b"]
    world.fleet.queue_edit("remove", ids[0])
    assert [e["job"] for e in world.fleet.queue_view()["waiting"]] == ["c", "b"]
    with pytest.raises(Refused, match="No waiting run"):
        world.fleet.queue_edit("remove", "nope")
    assert world.fleet.queue_edit("clear")["waiting"] == []


def test_runs_being_sent_when_the_looker_died_are_queued_again(world):
    world.fleet.enqueue("q1", {"steps": 1})
    with world.fleet.edit() as st:
        st["queue"]["active"].append(st["queue"]["waiting"].pop())
        st["instances"]["1"]["dispatching"] = {"entry": "x", "since": world.clock()}
    world.fleet.recover()
    st = world.fleet.read()
    assert [e["job"] for e in st["queue"]["waiting"]] == ["q1"] and "dispatching" not in st["instances"]["1"]


# ---------------------------------------------------------------- a race over several instances

RACE = {"steps": 100, "race": True, "race_configs": [{"name": "lo", "learning_rate": 5e-5}, {"learning_rate": 1e-4},
                                                     {"name": "hi", "learning_rate": 2e-4}]}


def test_a_race_is_dealt_over_the_instances_one_part_each(world):
    entries = world.fleet.enqueue("r", RACE, split=2)
    assert [(e["job"], e["race_rows"], e["group"]) for e in entries] == [("r_part1", ["lo", "hi"], "r"),
                                                                        ("r_c2", ["c2"], "r")]
    world.tick()
    starts = {m: next(c for c in world.machines[m].calls if c[0] == "start") for m in ("1", "2")}
    assert starts["1"][starts["1"].index("--race-rows") + 1] == "lo,hi"
    assert starts["2"][starts["2"].index("--race-rows") + 1] == "c2"
    world.tick()  # the runs are seen: they carry the group
    world.machines["1"].finish(best=60.0)
    world.machines["2"].finish(best=71.0)
    world.tick()
    group = world.fleet.group("r")
    assert group["leader"] == "r_c2" and group["finished"] and [p["job"] for p in group["parts"]] == ["r_c2", "r_part1"]
    assert world.kinds("run_done")[0]["group"] == "r"


def test_more_parts_than_configurations_gives_one_run_each(world):
    assert [e["job"] for e in world.fleet.plan("r", RACE, split=9)] == ["r_lo", "r_c2", "r_hi"]


def test_what_cannot_be_split_is_refused(world):
    with pytest.raises(Refused, match="Only a configuration race"):
        world.fleet.plan("r", {"steps": 1}, split=2)
    with pytest.raises(Refused, match="relay"):
        world.fleet.plan("r", dict(RACE, race_follow=True), split=2)
    with pytest.raises(Refused, match="two or more instances"):
        world.fleet.plan("r", RACE, split=1)
    with pytest.raises(Refused, match="names repeat"):
        world.fleet.plan("r", dict(RACE, race_configs=[{"name": "a"}, {"name": "a"}]), split=2)


# ---------------------------------------------------------------- downloads


def test_each_kind_follows_its_own_rule(world):
    world.fleet.set_instance(2, enabled=False)
    world.fleet.set_download({"logs": "during", "probes": "during", "samples": "end", "maps": "off", "best": "end",
                              "loras": "off"}, every=300)
    machine = world.machines["1"]
    machine.run("a")
    world.tick()
    assert world.fetched == [(1, ["a/logs.bin", "a/probes.bin"])]  # the first look at a running run
    world.tick(60)
    assert len(world.fetched) == 1  # not due yet
    world.fetched.clear()
    machine.finish()
    world.tick()
    assert world.fetched == [(1, ["a/best.bin", "a/samples.bin"])]  # the rest came before, at their size
    rec = world.fleet.find_run("a")
    assert rec["downloaded"] == ["best", "logs", "probes", "samples"] and not rec["final_pull"]
    assert Path(rec["folder"]).name == "a" and (Path(rec["folder"]) / "best.bin").is_file()
    world.fetched.clear()
    world.tick(1000)
    assert world.fetched == []  # an ended run is done with


def test_a_run_can_have_its_own_rule_and_a_bad_rule_is_refused(world):
    world.fleet.set_instance(2, enabled=False)
    world.fleet.set_download({k: "off" for k in fleet_module.DOWNLOAD_KINDS})
    world.fleet.enqueue("q1", {"steps": 1}, download={"loras": "end"})
    world.tick()
    world.tick()
    assert world.fetched == []
    world.machines["1"].finish()
    world.tick()
    assert world.fetched == [(1, ["q1/loras.bin"])]
    with pytest.raises(Refused, match="Not a download rule: loras=sometimes"):
        world.fleet.set_download({"loras": "sometimes"})
    with pytest.raises(Refused, match="Not a download rule"):
        world.fleet.plan("q2", {"steps": 1}, download={"everything": "end"})
    assert world.fleet.set_download({"maps": "end"}, run="q1")["download"] == {"loras": "end", "maps": "end"}


def test_a_download_that_keeps_failing_gives_up_and_says_so(world):
    world.fleet.set_instance(2, enabled=False)

    def broken(*a):
        raise OSError("disk full here")

    world.fleet.fetch = broken
    world.machines["1"].run("a")
    world.tick()
    world.machines["1"].finish()
    for _ in range(fleet_module.PULL_RETRIES + 2):
        world.tick()
    failed = world.kinds("download_failed")
    assert len(failed) == 1 and "disk full here" in failed[0]["error"]
    rec = world.fleet.find_run("a")
    assert not rec["final_pull"] and "disk full" in rec["pull_error"]


# ---------------------------------------------------------------- watches


def test_a_watch_fires_once_per_run_the_moment_the_number_is_on_that_side(world):
    world.fleet.add_watch("avg_loss", "above", 0.5, note="diverging?", by="claude")
    machine = world.machines["1"]
    machine.run("a", avg_loss=0.3, step=10)
    world.tick()
    assert world.kinds("watch") == []
    machine.jobs["a"].update(avg_loss=0.7, step=20)
    world.tick()
    machine.jobs["a"].update(avg_loss=0.9, step=30)
    world.tick()
    fired = world.kinds("watch")
    assert len(fired) == 1
    assert {k: fired[0][k] for k in ("metric", "op", "value", "actual", "instance", "job", "step", "note")} == {
        "metric": "avg_loss", "op": "above", "value": 0.5, "actual": 0.7, "instance": 1, "job": "a", "step": 20.0,
        "note": "diverging?"}
    world.machines["2"].run("b", avg_loss=0.8)  # another run: its own notice
    world.tick()
    assert [e["job"] for e in world.kinds("watch")] == ["a", "b"]


def test_a_repeating_watch_fires_again_when_the_number_comes_back(world):
    world.fleet.add_watch("avg_loss", "below", 0.2, repeat=True)
    machine = world.machines["1"]
    machine.run("a", avg_loss=0.1)
    for loss in (0.1, 0.1, 0.3, 0.15):
        machine.jobs["a"]["avg_loss"] = loss
        world.tick()
    assert [e["actual"] for e in world.kinds("watch")] == [0.1, 0.15]


def test_watches_read_the_last_probe_and_the_machine_and_keep_to_their_job_and_instance(world):
    world.fleet.add_watch("combined_ema", "below", 40.0, job="keep*")
    world.fleet.add_watch("disk_free_gb", "below", 10.0)
    world.fleet.add_watch("gpu_temp", "above", 85, instance=2)
    for iid, name in (("1", "keep_a"), ("2", "other")):
        world.machines[iid].run(name, last_probe={"step": 50, "combined_ema": 35.0}, probes_count=1)
    world.machines["1"].temp = 90
    world.machines["1"].disk = 4.0
    world.tick()
    assert [(e["metric"], e["instance"], e.get("job")) for e in world.kinds("watch")] == [
        ("combined_ema", 1, "keep_a"), ("disk_free_gb", 1, None)]
    world.machines["2"].temp = 91
    world.tick()
    assert world.kinds("watch")[-1]["metric"] == "gpu_temp"


def test_a_watch_needs_a_known_metric_and_can_be_removed(world):
    with pytest.raises(Refused, match="No metric 'losss'"):
        world.fleet.add_watch("losss", "above", 1)
    with pytest.raises(Refused, match="'above' or 'below'"):
        world.fleet.add_watch("loss", "over", 1)
    watch = world.fleet.add_watch("loss", "above", 1)
    assert world.fleet.remove_watch(watch["id"]) == 1 and world.fleet.read()["watches"] == []
    with pytest.raises(Refused):
        world.fleet.remove_watch(watch["id"])


def test_wait_gives_each_event_once_and_only_the_kinds_asked_for(world):
    world.fleet.add_watch("avg_loss", "above", 0.5)
    world.machines["1"].run("a", avg_loss=0.9)
    world.fleet.set_instance(2, enabled=False)
    world.clock.t += 60

    def sleep(seconds):
        world.clock.t += 30

    found = world.fleet.wait(sleep=sleep)
    assert [e["kind"] for e in found] == ["watch"]  # run_started is not something to wake up for
    assert world.fleet.wait(timeout=100, sleep=sleep) == []  # nothing new: the timeout
    world.machines["1"].finish()
    assert [e["kind"] for e in world.fleet.wait(timeout=1000, sleep=sleep)] == ["run_done"]
    assert [e["kind"] for e in world.fleet.wait(reader="other", kinds=("run_started",), sleep=sleep)] == ["run_started"]
    world.fleet.release()


def test_metrics_of_a_look():
    snap = {"disk_free_gb": 50.5, "gpus": [{"temperature": 70, "utilization": 99, "memory_used_mb": 100},
                                           {"temperature": 75, "utilization": 3, "memory_used_mb": 200}],
            "job_running": True,
            "job": {"step": 50, "total": 200, "avg_loss": 0.25, "elapsed": 1800.0, "done": False, "eta": None,
                    "last_probe": {"face_ema": 61.0, "since_improve": 2}}}
    m = fleet_module.metrics_of(snap, 2.0)
    assert m == {"disk_free_gb": 50.5, "gpu_temp": 75, "gpu_util": 3, "gpu_mem_mb": 200, "step": 50.0, "total": 200.0,
                 "avg_loss": 0.25, "elapsed": 1800.0, "progress": 25.0, "face_ema": 61.0, "since_improve": 2.0,
                 "cost": 1.0}
    assert set(m) <= set(fleet_module.METRICS)
    idle = fleet_module.metrics_of(dict(snap, job_running=False))
    assert "avg_loss" not in idle and idle["gpu_temp"] == 75  # a finished run's numbers are not watched


# ---------------------------------------------------------------- one looker, and the views


def test_only_one_fleet_looks_at_a_time(world, tmp_path):
    other = world.new_fleet(tmp_path)
    assert world.fleet.hold() and world.fleet.someone_looks()
    assert not other.hold() and other.someone_looks()
    assert other.step() is False
    world.fleet.release()
    assert not other.someone_looks()
    assert other.hold()
    other.release()


def test_step_looks_only_when_due_and_when_there_is_something_to_look_at(world, tmp_path):
    world.clock.t += 100
    assert world.fleet.step() is True
    assert world.fleet.step() is False  # just looked
    world.clock.t += 100
    assert world.fleet.step() is True
    world.fleet.release()
    empty = Fleet(home=tmp_path / "none", list_instances=lambda: [], now=world.clock)
    assert empty.step() is False
    empty.release()


def test_the_overview_names_each_instances_state(world):
    world.machines["1"].run("a", step=5)
    world.fleet.add_instance(3)
    world.fleet.set_instance(3, enabled=False)
    world.tick()
    view = world.fleet.overview()
    assert [(i["id"], i["state"]) for i in view["instances"]] == [(1, "running"), (2, "idle"), (3, "off")]
    assert view["instances"][0]["job"]["job"] == "a" and view["instances"][0]["metrics"]["step"] == 5
    assert view["queue"]["waiting"] == [] and view["looking"] is False and view["looked"] is not None


def test_stop_reaches_the_instance_that_runs_the_job(world):
    world.machines["2"].run("b")
    world.tick()
    assert world.fleet.stop_run("b") == {"job": "b", "state": "stopping", "instance": 2}
    assert world.machines["2"].calls[-1] == ("stop",)
    assert world.fleet.stop_run("2", kill=True)["instance"] == 2 and world.machines["2"].calls[-1] == ("stop", "--kill")
    with pytest.raises(Refused, match="No running job"):
        world.fleet.stop_run("nothing")


# ---------------------------------------------------------------- the command line


@pytest.fixture
def cli(tmp_path, monkeypatch, capsys):
    world = World(tmp_path, 1, 2)
    monkeypatch.setattr(fleet_module, "Fleet", lambda **kw: world.fleet)

    def call(*argv):
        code = fleet_module.main(list(argv))
        out = capsys.readouterr()
        return code, (json.loads(out.out) if out.out.strip() else None), out.err

    return world, call


def test_cli_queue_add_checks_on_an_instance_then_queues(cli, tmp_path):
    world, call = cli
    world.tick()
    path = tmp_path / "c.yaml"
    path.write_text("steps: 100\noptimizer: keel\n", encoding="utf-8")
    code, data, _ = call("queue", "add", "--job", "t1", "--config", str(path), "--set", "steps=300",
                         "--download", "loras=end")
    assert code == 0 and data["queued"][0]["job"] == "t1" and data["checks"][0]["checked"] == 1
    entry = world.fleet.read()["queue"]["waiting"][0]
    assert entry["config"] == {"steps": 100, "optimizer": "keel"} and entry["set"] == ["steps=300"]
    assert entry["download"] == {"loras": "end"}
    world.machines["1"].refuse = world.machines["2"].refuse = "A setting is invalid: steps must be positive"
    code, _, err = call("queue", "add", "--job", "t2", "--config", str(path))
    assert code == 1 and "steps must be positive" in err
    assert [e["job"] for e in world.fleet.read()["queue"]["waiting"]] == ["t1"]
    assert call("queue", "add", "--job", "t3")[0] == 1  # nothing to start from


def test_cli_split_alone_means_one_part_per_instance(cli, tmp_path):
    world, call = cli
    path = tmp_path / "race.json"
    path.write_text(json.dumps(RACE), encoding="utf-8")
    code, data, _ = call("queue", "add", "--job", "r", "--config", str(path), "--split", "--no-check")
    assert code == 0 and [q["job"] for q in data["queued"]] == ["r_part1", "r_c2"]


def test_cli_from_takes_an_earlier_runs_settings(cli):
    world, call = cli
    world.machines["1"].run("base")
    world.tick()
    code, data, _ = call("queue", "add", "--job", "again", "--from", "base", "--no-check")
    assert code == 0 and world.fleet.read()["queue"]["waiting"][0]["config"] == {"optimizer": "keel", "steps": 100}
    code, data, _ = call("queue", "add", "--job", "again2", "--from", "base@2", "--no-check")
    assert world.fleet.read()["queue"]["waiting"][1]["config"] == {"steps": 50, "optimizer": "adamw"}
    assert call("queue", "add", "--job", "x", "--from", "unknown", "--no-check")[0] == 1


def test_cli_watch_wait_and_views(cli):
    world, call = cli
    code, watch, _ = call("watch", "add", "avg_loss", "above", "0.5", "--note", "check it", "--by", "claude")
    assert code == 0 and watch["metric"] == "avg_loss" and watch["value"] == 0.5
    assert call("watch", "add", "avg_loss", "sideways", "1")[0] == 1
    assert call("watch", "add", "avg_loss", "above", "much")[0] == 1
    world.machines["1"].run("a", avg_loss=0.9)
    world.tick()
    code, found, _ = call("wait", "--timeout", "5")
    assert code == 0 and found[0]["kind"] == "watch" and found[0]["note"] == "check it"
    code, status, _ = call("status", "--no-poll")
    assert status["instances"][0]["state"] == "running" and status["watches"][0]["fired"] == 1
    assert "series" not in status["instances"][0]
    assert call("runs")[1][0]["job"] == "a"
    assert call("run", "a")[1]["instance"] == 1
    assert call("metrics")[1]["now"]["1"]["avg_loss"] == 0.9
    assert call("events", "--kind", "watch")[1][0]["actual"] == 0.9
    assert call("download", "loras=end", "--every", "120")[1] == {
        "download": dict(fleet_module.DEFAULT_SETTINGS["download"], loras="end"), "every": 120}
    assert call("download", "loras")[0] == 1
    assert call("pull", "a", "--what", "samples")[1]["fetched"] == 1
    assert call("watch", "remove", "all")[1] == {"removed": 1}
    assert call("hold", "2")[1]["accept"] is False
    assert call("run", "nothing")[0] == 1


def test_cli_status_looks_once_when_nobody_else_does(cli):
    world, call = cli
    world.machines["1"].run("a")
    code, status, _ = call("status")
    assert status["instances"][0]["state"] == "running" and world.starts(1) == []
    assert world.fleet.ticker is None  # and lets go again


# ---------------------------------------------------------------- the agent's side


@pytest.fixture
def install(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(agent, "webui_running", lambda: True)
    (tmp_path / "output").mkdir()
    (tmp_path / "datasets" / "set").mkdir(parents=True)
    return tmp_path


def make_job(name, **status):
    d = Path("output") / name
    d.mkdir(parents=True)
    TrainConfig(job_name=name, dataset_dir="datasets/set").save(d / "config.yaml")
    (d / "status.json").write_text(json.dumps({"step": 10, "total": 10, "done": True, "error": None, **status}),
                                   encoding="utf-8")
    return d


def agent_json(capsys, *argv):
    code = agent.main(list(argv))
    out = capsys.readouterr().out
    return code, json.loads(out) if out.strip() else None


def test_a_series_is_thinned_to_a_few_points_with_the_last_one_kept():
    assert agent.thin([]) == []
    assert agent.thin([0.5, 0.25]) == [[1, 0.5], [2, 0.25]]
    points = agent.thin([i / 1000 for i in range(1001)], limit=100)
    assert len(points) <= 101 and points[-1] == [1001, 1.0] and points[0][0] == 11


def test_snapshot_is_the_machine_every_job_and_the_current_jobs_series(install, capsys):
    make_job("failed", error="boom", done=False, run_id="failed-1")
    d = make_job("live", done=False, step=3, run_id="live-2", losses=[0.5, 0.4, 0.3], step_times=[0.5, 0.7],
                 probes=[{"step": 2, "combined_ema": 41.23456, "images": ["x.png"], "face_seeds": [1, 2]}])
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        Path(".webui_job.json").write_text(json.dumps({
            "pid": proc.pid, "create_time": psutil.Process(proc.pid).create_time(), "job_dir": str(d.resolve())}))
        code, snap = agent_json(capsys, "snapshot")
    finally:
        proc.kill()
        proc.wait()
    assert code == 0 and snap["job_running"] and snap["datasets"] == ["set"]
    states = {j["job"]: (j["state"], j["run_id"]) for j in snap["jobs"]}
    assert states == {"live": ("running", "live-2"), "failed": ("error", "failed-1")}
    assert snap["job"]["job"] == "live" and snap["job"]["running"] and snap["job"]["step_time"] == 0.6
    assert "losses" not in snap["job"] and snap["job"]["last_probe"]["combined_ema"] == 41.23456
    assert snap["series"] == {"losses": [[1, 0.5], [2, 0.4], [3, 0.3]], "probes": [{"step": 2, "combined_ema": 41.235}]}
    assert fleet_module.metrics_of(snap)["combined_ema"] == 41.23456


def test_record_is_one_runs_settings_numbers_and_scores(install, capsys):
    make_job("a", run_id="a-1", losses=[0.5], probes=[{"step": 5, "face": 50.0}], likeness={"best": 50.0})
    code, rec = agent_json(capsys, "record", "a")
    assert code == 0 and rec["run_id"] == "a-1" and rec["state"] == "done" and rec["config"]["job_name"] == "a"
    assert rec["probes"] == [{"step": 5, "face": 50.0}] and rec["losses"] == [[1, 0.5]]
    assert rec["status"]["likeness"] == {"best": 50.0}
    code, cfg = agent_json(capsys, "config", "a", "--json")
    assert cfg["dataset_dir"] == "datasets/set"


RACE_SETS = ["--set", "race=true", "--set", "probe_every=100", "--set", "face_refs=datasets/set", "--set",
             'race_configs=[{"name": "lo", "learning_rate": 5e-5}, {"name": "hi", "learning_rate": 2e-4, '
             '"lora_rank": 8}, {"learning_rate": 1e-4}]']


def test_one_race_row_alone_starts_as_a_plain_run_with_its_settings(install, capsys):
    make_job("base")
    code, plan = agent_json(capsys, "start", "--job", "solo", "--from", "base", *RACE_SETS, "--race-rows", "hi",
                            "--dry-run")
    assert code == 0, plan
    changed = plan["changed"]
    assert changed["learning_rate"][1] == 2e-4 and changed["lora_rank"][1] == 8
    assert "race" not in changed and "race_configs" not in changed  # the run is no race any more


def test_several_race_rows_stay_a_race_among_themselves(install, capsys):
    make_job("base")
    code, plan = agent_json(capsys, "start", "--job", "pair", "--from", "base", *RACE_SETS, "--race-rows", "lo,c3",
                            "--dry-run")
    assert code == 0, plan
    assert [row["name"] for row in plan["changed"]["race_configs"][1]] == ["lo", "c3"]


def test_unknown_race_rows_and_a_relay_are_refused(install, capsys):
    make_job("base")
    assert agent.main(["start", "--job", "x", "--from", "base", *RACE_SETS, "--race-rows", "nope", "--dry-run"]) == 1
    assert "No race configuration named nope" in capsys.readouterr().err
    assert agent.main(["start", "--job", "x", "--from", "base", "--race-rows", "lo", "--dry-run"]) == 1
    assert "no race" in capsys.readouterr().err
    assert agent.main(["start", "--job", "x", "--from", "base", *RACE_SETS, "--set", "race_follow=true",
                       "--race-rows", "lo", "--dry-run"]) == 1
    assert "relay" in capsys.readouterr().err


def test_the_fleets_route_is_not_the_single_instance_tools_route(tmp_path, monkeypatch):
    """rulz_remote.py remembers one instance in connection.json: the fleet's looks must not move it around."""
    monkeypatch.setattr(remote, "STATE", tmp_path)
    monkeypatch.setattr(remote, "CONNECTION", tmp_path / "connection.json")
    monkeypatch.setattr(remote, "list_instances", lambda: [dict(listing([Machine(7)])[0], public_ipaddr="1.2.3.4",
                                                                ports={"22/tcp": [{"HostPort": "22"}]})])
    monkeypatch.setattr(remote, "reachable", lambda conn, key: True)
    target = remote.Remote(7, None, "/w", remember=False)
    assert target.conn["host"] == "1.2.3.4" and not (tmp_path / "connection.json").exists()
    remote.Remote(7, None, "/w")
    assert (tmp_path / "connection.json").exists()


def test_settings_that_point_into_a_pushed_folder_point_at_its_place_on_the_instance(world, tmp_path):
    local = tmp_path / "sets" / "anna"
    (local / "refs").mkdir(parents=True)
    config = {"dataset_dir": str(local), "face_refs": str(local / "refs"), "datasets": [{"path": str(local)}],
              "model_path": str(tmp_path / "models" / "m.safetensors"), "sample_prompts": ["anna, a photo / portrait"],
              "steps": 5}
    entry = world.fleet.plan("q", config, push=[local])[0]
    assert entry["config"] == dict(config, dataset_dir="datasets/anna", face_refs="datasets/anna/refs",
                                   datasets=[{"path": "datasets/anna"}])
    assert world.fleet.plan("q", config)[0]["config"] == config


def test_a_run_that_started_and_ended_between_two_looks_is_still_closed_and_downloaded(world):
    world.fleet.set_instance(2, enabled=False)
    world.tick()  # the instance has been looked at: what appears from now on ran in the fleet's time
    world.machines["1"].run("quick")
    world.machines["1"].finish(best=40.0)
    world.tick()
    assert [e["kind"] for e in world.kinds("run_started", "run_done")] == ["run_started", "run_done"]
    rec = world.fleet.find_run("quick")
    assert not rec["imported"] and rec["downloaded"] and rec["state"] == "done"
