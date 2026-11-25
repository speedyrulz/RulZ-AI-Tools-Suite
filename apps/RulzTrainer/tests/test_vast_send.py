"""scripts/vast/rulz_send.py: the SendManager (contract sections 7, 9 and 10). Everything is a fake: vast.ai (the market),
the instances (answering the agent's commands), the file transfers and the clock. Nothing here rents, starts, stops or
destroys anything real, touches the network or uses a GPU. tick() runs with the market locked: a rent or a start from
it fails the test."""
import hashlib
import inspect
import json
import os as rulz_send_os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "vast"))

import rulz_files  # noqa: E402
import rulz_fleet as fleet_module  # noqa: E402
import rulz_market  # noqa: E402
import rulz_remote  # noqa: E402
import rulz_send as send_module  # noqa: E402
from rulz_fleet import Fleet  # noqa: E402
from rulz_market import MarketError  # noqa: E402
from rulz_send import SendError, SendManager  # noqa: E402

# ---------------------------------------------------------------- fakes

CHECKED: set[str] = set()


def bound(name, real, *args, **kwargs):
    """The call rulz_send makes, bound to the REAL function's signature (a mismatch with the real module raises), and
    the arguments by name with their defaults. `real` of a method takes a placeholder for self."""
    CHECKED.add(name)
    got = inspect.signature(real).bind(*args, **kwargs)
    got.apply_defaults()
    return got.arguments


class Died(BaseException):
    """The program was killed at this point (not an Exception: nothing in rulz_send may catch it)."""


class Clock:
    def __init__(self):
        self.t = 1_000_000.0

    def __call__(self):
        self.t += 0.001
        return self.t


class Machine:
    """An instance: what it holds and how it answers the agent's commands."""

    def __init__(self, iid):
        self.id = iid
        self.datasets: list[str] = []
        self.marks: dict[str, str] = {}
        self.jobs: dict[str, dict] = {}
        self.running = None
        self.down = False
        self.ssh_ok = True
        self.refuse = ""
        self.calls: list[tuple] = []
        self.configs: dict[str, dict] = {}
        self.disk, self.webui, self.version = 500.0, True, "4.2.0"
        self.queue_waiting: list[dict] = []
        self.tests = 0
        self.tests_dir_empty = True
        self.watchdog = {"armed": False}
        self.can_stop = True
        self.autostop = False
        self.behind = 0
        self.extra_jobs: list[str] = []
        self.tree: dict[str, list[tuple[str, int]]] = {}
        self.arm_fails = False
        self.error_text = ""
        self.hook = lambda name: None
        self.n = 0

    def run(self, name):
        self.n += 1
        self.jobs[name] = {"job": name, "run_id": f"{name}-{self.n}", "step": 10, "total": 100, "elapsed": 10.0,
                           "avg_loss": 0.3}
        self.running = name

    def finish(self, state="done"):
        job = self.jobs[self.running]
        job.update(step=job["total"] if state == "done" else job["step"], final=state)
        self.running = None

    def snapshot(self):
        jobs = [{**{k: v for k, v in j.items() if k != "final"}, "running": n == self.running,
                 "state": "running" if n == self.running else j.get("final", "done")} for n, j in self.jobs.items()]
        snap = {"webui_running": self.webui, "disk_free_gb": self.disk, "disk_total_gb": 1000.0,
                "datasets": list(self.datasets), "dataset_marks": dict(self.marks),
                "gpus": [{"name": "RTX 5090", "memory_used_mb": 9000, "memory_total_mb": 32000, "utilization": 97,
                          "temperature": 60}],
                "job_running": self.running is not None,
                "queue": {"paused": False, "waiting": list(self.queue_waiting), "recent": []}, "jobs": jobs,
                "trainer": {"version": self.version}, "tests_running": self.tests, "watchdog": dict(self.watchdog),
                "autostop": {"armed": self.autostop}}
        if self.jobs:
            name = self.running or list(self.jobs)[-1]
            job = self.jobs[name]
            snap["job"] = {**{k: v for k, v in job.items() if k != "final"}, "loss": 0.3}
            snap["series"] = {"losses": [], "probes": []}
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
                self.hook("check")
                return 0, {"job": name, "dry_run": True, "steps": 100, "arch": "krea2", "changed": {}}, b""
            self.configs[name] = yaml.safe_load(config)
            self.run(name)
            return 0, {"job": name, "state": "running"}, b""
        if command == "record":
            job = self.jobs[args[1]]
            return 0, {"job": args[1], "run_id": job["run_id"], "config": {"steps": job["total"]},
                       "status": {"step": job["step"]}, "probes": [], "losses": []}, b""
        if command == "files":
            kinds = [args[i + 1] for i, a in enumerate(args) if a == "--what"]
            if kinds == ["tree"]:
                return 0, {"base": "/out", "files": [{"path": f"{args[1]}/{n}", "size": s}
                                                     for n, s in self.tree.get(args[1], [])]}, b""
            return 0, {"base": "/out", "files": [{"path": f"{args[1]}/{k}.bin", "size": 5} for k in kinds]}, b""
        if command == "jobs":
            return 0, [{"job": n} for n in [*self.jobs, *self.extra_jobs]], b""
        if command == "watchdog":
            self.watch_calls.append(args[1])
            if args[1] == "arm":
                if self.arm_fails:
                    return 1, b"", b"[refused] no folder yet"
                self.watchdog = {"armed": True, "fired": False, "minutes": int(args[3])}
                return 0, dict(self.watchdog, can_stop=self.can_stop), b""
            if args[1] == "disarm":
                self.watchdog = {"armed": False}
            return 0, dict(self.watchdog), b""
        if command == "update":
            if "--check" in args:
                return 0, {"latest": {"version": "4.3.0", "commit": "abc"}, "behind": self.behind,
                           "current": {"commit": "old"}}, b""
            self.hook("update")
            self.behind = 0
            self.version = "4.3.0"
            return 0, {"state": "done", "to": {"version": "4.3.0", "commit": "abc"}}, b""
        if command == "stop":
            return 0, {"job": self.running, "state": "stopping"}, b""
        raise AssertionError(f"unexpected {args}")

    watch_calls: list = []

    def __post_init__(self):
        pass


class FakeRemote:
    def __init__(self, machine):
        self.machine = machine
        self.conn = {"id": machine.id, "host": "h", "port": 1}

    def run(self, *args, **kwargs):
        bound("remote.run", rulz_remote.Remote.run, self, *args, **kwargs)
        shell = args[0]
        m = self.machine
        if m.down or not m.ssh_ok:
            return SimpleNamespace(returncode=255, stdout=b"", stderr=b"ssh: connect refused")
        if "_tests" in shell:
            return SimpleNamespace(returncode=0 if m.tests_dir_empty else 1, stdout=b"", stderr=b"")
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    def agent(self, *args, config=None, retry=True, timeout="auto"):
        bound("remote.agent", rulz_remote.Remote.agent, self, *args, config=config, retry=retry, timeout=timeout)
        code, out, err = self.machine.answer(args, config)
        return SimpleNamespace(returncode=code, stdout=out if isinstance(out, bytes) else json.dumps(out).encode(),
                               stderr=err)

    def agent_json(self, *args):
        bound("remote.agent_json", rulz_remote.Remote.agent_json, self, *args)
        done = self.agent(*args)
        if done.returncode != 0:
            raise SystemExit(done.stderr.decode() or "failed")
        return json.loads(done.stdout)


def mrow(i, **kw):
    """An instance as Market.instances() returns it."""
    row = {"id": i, "label": f"label-{i}", "status": "running", "actual_status": "running",
           "intended_status": "running", "gpu_name": "RTX 5090", "num_gpus": 1, "gpu_ram_mb": 32000, "dph_total": 0.8,
           "storage_per_hour": 0.1, "per_hour": 0.8, "billing": True, "is_rulz": True, "ours": False}
    row.update(kw)
    return row


class FakeMarket:
    """vast.ai: records what was asked of it. `locked` while the manager ticks: renting or starting then is a bug."""

    def __init__(self):
        self.rows: dict[int, dict] = {}
        self.rented: list[dict] = []
        self.intent_rows: list[dict] = []
        self.calls: list[tuple] = []
        self.locked = False
        self.stop_works = True
        self.rent_error = None
        self.start_error = None
        self.stop_errors: list = []
        self.next_id = 500

    def instances(self):
        bound("market.instances", rulz_market.Market.instances, self)
        return [dict(r) for r in self.rows.values()]

    def rentals(self):
        bound("market.rentals", rulz_market.Market.rentals, self)
        return [dict(r) for r in self.rented]

    def intents(self):
        bound("market.intents", rulz_market.Market.intents, self)
        return [dict(r) for r in self.intent_rows]

    def settle(self):
        bound("market.settle", rulz_market.Market.settle, self)
        self.calls.append(("settle",))
        return []

    def rent(self, *args, **kwargs):
        send_id = bound("market.rent", rulz_market.Market.rent, self, *args, **kwargs)["send_id"]
        assert not self.locked, "tick() rented"
        self.calls.append(("rent", send_id))
        if self.rent_error is not None:
            err = self.rent_error
            if err.kind == "pending":
                self.intent_rows.append({"label": err.data["label"], "send": send_id, "state": "checking",
                                         "message": "checking your account"})
            raise err
        iid = self.next_id
        self.next_id += 1
        label = f"rulz-trainer-{iid}"
        self.rows[iid] = mrow(iid, label=label, status="loading", actual_status="loading", ours=True)
        rental = {"id": iid, "label": label, "send": send_id, "price": 0.9, "real_price": 0.9, "price_over": False,
                  "note": ""}
        self.rented.append(rental)
        return dict(rental)

    def start(self, *args, **kwargs):
        iid = bound("market.start", rulz_market.Market.start, self, *args, **kwargs)["instance_id"]
        assert not self.locked, "tick() started an instance"
        self.calls.append(("start", iid))
        if self.start_error is not None:
            raise self.start_error
        self.rows[iid].update(status="loading", actual_status="loading", billing=True)
        return {"id": iid, "started": True}

    def stop(self, *args, **kwargs):
        iid = bound("market.stop", rulz_market.Market.stop, self, *args, **kwargs)["instance_id"]
        self.calls.append(("stop", iid))
        if self.stop_errors:
            raise self.stop_errors.pop(0)
        if iid not in self.rows:
            raise MarketError("gone", "gone", True)
        if self.stop_works:
            self.rows[iid].update(status="exited", actual_status="exited", billing=False, per_hour=0.1)
        return {"id": iid, "ok": True}

    def destroy(self, *args, **kwargs):
        got = bound("market.destroy", rulz_market.Market.destroy, self, *args, **kwargs)
        iid, require_rental = got["instance_id"], got["require_rental"]
        self.calls.append(("destroy", iid, require_rental))
        if require_rental and not any(r["id"] == iid for r in self.rented):
            raise MarketError("not rented by this app", "refused", True)
        self.rows.pop(iid, None)
        return {"id": iid, "ok": True}

    def of(self, name):
        return [c for c in self.calls if c[0] == name]


class FakeFiles:
    """rulz_files' functions without ssh or a bucket: they record what was asked."""

    FilesError = rulz_files.FilesError
    Cancelled = rulz_files.Cancelled
    check_bucket_id = staticmethod(rulz_files.check_bucket_id)

    def __init__(self):
        self.calls: list[tuple] = []
        self.on_upload = None
        self.fail_with = None

    def target_for(self, path, **kw):
        bound("files.target_for", rulz_files.target_for, path, **kw)
        p = Path(path)
        size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.is_dir() else p.stat().st_size
        sha = hashlib.sha256(f"{p.name}{size}".encode()).hexdigest()
        name = rulz_files.clean_name(p.name)
        return {"kind": "folder" if p.is_dir() else "file", "name": name, "sha256": sha, "d12": sha[:12],
                "dest": rulz_files.sent_dest(sha[:12], name), "size": size, "files": []}

    def _go(self, what, *args, progress=None, cancel=None, size=10):
        self.calls.append((what, *args))
        if self.fail_with is not None:
            raise self.fail_with
        if self.on_upload:
            self.on_upload(what)
        if cancel and cancel():
            raise rulz_files.Cancelled()
        if progress:
            progress(size, size)

    def upload_file(self, *args, **kw):
        got = bound("files.upload_file", rulz_files.upload_file, *args, **kw)
        self._go("upload_file", str(got["path"]), got["dest"], progress=got["progress"], cancel=got["cancel"])

    def upload_tree(self, *args, **kw):
        got = bound("files.upload_tree", rulz_files.upload_tree, *args, **kw)
        self._go("upload_tree", str(got["path"]), got["target"]["dest"], progress=got["progress"],
                 cancel=got["cancel"])

    def upload_to_bucket(self, *args, **kw):
        got = bound("files.upload_to_bucket", rulz_files.upload_to_bucket, *args, **kw)
        target = got["target"]
        self._go("upload_to_bucket", got["bucket_id"], str(got["path"]), progress=got["progress"],
                 cancel=got["cancel"])
        return {"bucket": got["bucket_id"], "kind": target["kind"], "root": "r", "size": target["size"],
                "sha256": target["sha256"], "files": [{"rel": "", "bucket_path": f"rulz-trainer/{target['name']}",
                                                       "size": target["size"], "sha256": target["sha256"]}]}

    def fetch_from_bucket(self, *args, **kw):
        got = bound("files.fetch_from_bucket", rulz_files.fetch_from_bucket, *args, **kw)
        self._go("fetch_from_bucket", got["bucket_id"], got["dest"], progress=got["progress"], cancel=got["cancel"])

    def fetch_tree_from_bucket(self, *args, **kw):
        got = bound("files.fetch_tree_from_bucket", rulz_files.fetch_tree_from_bucket, *args, **kw)
        self._go("fetch_tree_from_bucket", got["dest"], progress=got["progress"], cancel=got["cancel"])


class World:
    """A fleet over fake machines, a fake vast.ai, a manager over both, and a clock: tick() is one look of each."""

    def __init__(self, tmp_path, ids=(1,), executor="inline"):
        self.tmp = tmp_path
        self.clock = Clock()
        self.machines = {str(i): Machine(i) for i in ids}
        self.market = FakeMarket()
        for i in ids:
            self.market.rows[i] = mrow(i)
        self.fetched: list[tuple] = []
        self.sent: list[tuple] = []
        self.files = FakeFiles()
        self.jobs_deferred: list = []
        self.executor_kind = executor
        self.die_at, self.died = None, False
        for m in self.machines.values():
            m.hook = self.hook
        self.sleeps: list[float] = []
        self.listed = lambda: []
        self.root = tmp_path / "work"
        self.root.mkdir(parents=True, exist_ok=True)
        self.fleet = self.new_fleet()
        self.mgr = self.new_manager()

    def hook(self, name):
        if self.die_at == name and not self.died:
            self.died = True
            raise Died(name)

    def new_fleet(self):
        def fetch(target, base, files, dest):
            self.fetched.append((target.machine.id, sorted(files)))
            for name in files:
                (dest / name).parent.mkdir(parents=True, exist_ok=True)
                (dest / name).write_bytes(b"x" * self.size_of(target.machine, name))

        def send(*args, **kw):
            got = bound("fleet.send", rulz_remote.send, *args, **kw)
            target, items, where, progress, kw = got["remote"], got["items"], got["target"], got["progress"], {
                "marks": got["marks"]}
            self.hook("datasets")
            self.sent.append((target.machine.id, [n for _, n in items], where, kw.get("marks")))
            if progress:
                progress(1000, 0)
            for _, name in items:
                target.machine.datasets.append(name)
                if (kw.get("marks") or {}).get(name):
                    target.machine.marks[name] = kw["marks"][name]

        fleet = Fleet(home=self.tmp / "fleet", connect=lambda inst: FakeRemote(self.machines[str(inst["id"])]),
                      list_instances=lambda: self.listed(), fetch=fetch, send=send, now=self.clock)
        with fleet.edit() as st:
            st["settings"]["auto_discover"] = False
        return fleet

    @staticmethod
    def size_of(machine, name):
        for files in machine.tree.values():
            for n, s in files:
                if name.endswith("/" + n):
                    return s
        return 5

    def new_manager(self):
        def inline(iid, mark, func):
            if self.executor_kind == "inline" or (self.executor_kind == "dcheck_deferred" and mark[0] == "send"):
                func()
            else:
                self.jobs_deferred.append(func)

        return SendManager(home=self.tmp, fleet=self.fleet, market=self.market, files=self.files, now=self.clock,
                           sleep=lambda s: self.sleeps.append(s), executor=inline)

    def restart(self):
        """The WebUI closes and opens again: a new Fleet and a new manager over the same files on disk."""
        self.fleet = self.new_fleet()
        self.mgr = self.new_manager()
        return self.mgr

    def tick(self, seconds=30.0):
        self.clock.t += seconds
        self.fleet.tick()
        self.fleet.drain()
        self.market.locked = True
        try:
            self.mgr.tick()
        finally:
            self.market.locked = False
        assert not self.mgr.errors, self.mgr.errors
        return self.mgr

    def until(self, predicate, limit=120):
        for _ in range(limit):
            if predicate():
                return
            self.tick()
        raise AssertionError("never happened: " + json.dumps(self.mgr.sends(), default=str)[:1500])

    def view(self, sid):
        return self.mgr.get(sid)

    def finish_runs(self):
        for m in self.machines.values():
            if m.running:
                m.finish()

    def run_to_end(self, sid, limit=200, finish=True):
        for _ in range(limit):
            if self.view(sid)["state"] in ("done", "failed", "cancelled"):
                return self.view(sid)
            if finish:
                self.finish_runs()
            self.tick()
        raise AssertionError("never ended: " + json.dumps(self.view(sid), default=str)[:1500])

    def dataset(self, name="anna", files=2):
        folder = self.root / "sets" / name
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(files):
            if not (folder / f"{i}.png").exists():
                (folder / f"{i}.png").write_bytes(b"png" * (i + 1))
        return folder

    def config(self, **extra):
        return {"dataset_dir": str(self.dataset()), "steps": 100, "output_dir": "C:/out", **extra}

    def create(self, job="t1", config=None, **kw):
        kw.setdefault("instance", 1)
        return self.mgr.create(job, config or self.config(), self.root, **kw)


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


def events(world, *kinds):
    return [e for e in world.fleet.events() if not kinds or e["kind"] in kinds]


# ---------------------------------------------------------------- pure decisions


FULL = {key: True for key, _ in send_module.CONDITIONS + send_module.DESTROY_CONDITIONS}


@pytest.mark.parametrize("fact", [k for k, _ in send_module.CONDITIONS])
def test_every_condition_of_the_after_run_action_holds_it_back(fact):
    assert send_module.after_decision("stop", FULL)["go"]
    decision = send_module.after_decision("stop", dict(FULL, **{fact: False}))
    assert not decision["go"] and decision["wait"]
    assert not send_module.after_decision("destroy", dict(FULL, **{fact: False}))["go"]


@pytest.mark.parametrize("fact", [k for k, _ in send_module.DESTROY_CONDITIONS])
def test_a_destroy_that_misses_a_condition_becomes_a_stop_with_the_reason(fact):
    assert send_module.after_decision("destroy", FULL) == {"go": True, "action": "destroy", "wait": [],
                                                           "downgrade": ""}
    decision = send_module.after_decision("destroy", dict(FULL, **{fact: False}))
    assert decision["go"] and decision["action"] == "stop" and decision["downgrade"]
    # the extras are no reason to hold a plain stop
    assert send_module.after_decision("stop", dict(FULL, **{fact: False}))["go"]


def test_absent_facts_count_as_not_holding_and_leave_does_nothing():
    assert not send_module.after_decision("stop", {})["go"]
    assert send_module.after_decision("leave", FULL) == {"go": False, "action": None, "wait": [], "downgrade": ""}


def test_the_mildest_choice_of_several_sends_wins():
    assert send_module.mildest(["destroy", "stop"]) == "stop"
    assert send_module.mildest(["destroy", "leave", "stop"]) == "leave"
    assert send_module.mildest(["destroy"]) == "destroy"
    assert send_module.mildest([]) == "leave"


def test_choosing_destroy_raises_every_download_kind_to_at_least_end():
    modes = {"logs": "during", "probes": "off", "samples": "end", "maps": "off", "best": "end", "loras": "off"}
    assert send_module.kinds_for("stop", modes) == {}
    got = send_module.kinds_for("destroy", modes)
    assert got == {"logs": "during", "probes": "end", "samples": "end", "maps": "end", "best": "end", "loras": "end"}


def test_only_pinned_sent_to_and_uncertain_entries_count_as_waiting_for_an_instance():
    st = {"queue": {"waiting": [{"id": "a", "instance": None}, {"id": "b", "instance": 1},
                                {"id": "c", "instance": None, "uncertain": "1"}, {"id": "d", "instance": 2}],
                    "active": [{"id": "e", "sent_to": 1}, {"id": "f", "sent_to": 2}]}}
    assert sorted(e["id"] for e in send_module.entries_on(st, 1)) == ["b", "c", "e"]


def test_a_tree_is_missing_what_is_not_here_at_its_size(tmp_path):
    (tmp_path / "job").mkdir()
    (tmp_path / "job" / "a.bin").write_bytes(b"12345")
    (tmp_path / "job" / "b.bin").write_bytes(b"12")
    listing = {"files": [{"path": "job/a.bin", "size": 5}, {"path": "job/b.bin", "size": 9},
                         {"path": "job/c.bin", "size": 1}]}
    assert send_module.tree_missing(listing, tmp_path) == ["job/b.bin", "job/c.bin"]


# ---------------------------------------------------------------- an existing instance, start to finish


def test_a_send_to_a_running_instance_goes_through_every_step_and_stops_it_after_the_countdown(world):
    send = world.create(after="stop")
    assert send["steps"][0]["state"] == "skipped" and send["state"] == "going"
    m = world.machines["1"]
    done = world.run_to_end(send["id"])
    assert done["state"] == "done" and "stopped" in done["result"]
    assert m.configs["t1"]["dataset_dir"] == "datasets/anna" and m.configs["t1"]["output_dir"] == "output"
    assert [c for c in m.calls if c[0] == "start" and "--dry-run" in c], "the dry run was made"
    assert [c for c in m.calls if c[0] == "start" and "--dry-run" not in c] == [
        ("start", "--job", "t1", "--resume-queue")]
    assert world.sent[0][2] == "datasets" and world.sent[0][1] == ["anna"]
    assert world.market.of("stop") == [("stop", 1)] and not world.market.of("rent") and not world.market.of("start")
    world.tick(130)  # the stop is followed until vast.ai no longer reports the instance running
    kinds = [e["kind"] for e in events(world, "send_started", "instance_stopped_after_run")]
    assert kinds == ["send_started", "instance_stopped_after_run"]
    assert world.sleeps == []  # a tick never sleeps


def test_the_run_is_matched_by_its_entry_id_never_by_job_name(world):
    # an older run of the same job name on record, of another entry: nothing to do with this send
    world.fleet.save_run({"key": "1-t1-old", "instance": 1, "job": "t1", "state": "running", "entry": "someone",
                          "first_seen": 1.0})
    send = world.create(job="t1")
    assert send["job"] == "t1"
    world.until(lambda: world.view(send["id"])["run"])
    recs = world.fleet.runs(instance=1, job="t1")
    mine = [r for r in recs if r.get("entry") == send["id"]]
    assert len(recs) == 2 and len(mine) == 1 and mine[0]["send"] == send["id"]
    assert world.view(send["id"])["run"] == mine[0]["key"]


def test_the_entry_has_the_send_id_no_push_the_final_config_and_is_pinned(world):
    send = world.create()
    world.until(lambda: world.view(send["id"])["step"] == "queue")
    world.tick()
    entry = next(e for e in world.fleet.read()["queue"]["waiting"] + world.fleet.read()["queue"]["active"]
                 if e["id"] == send["id"]) if not world.fleet.runs() else None
    if entry is not None:
        assert entry["push"] == [] and entry["instance"] == 1 and entry["send"] == send["id"]
    world.until(lambda: world.view(send["id"])["run"])
    assert world.fleet.runs()[0]["entry"] == send["id"]
    assert world.machines["1"].configs["t1"]["dataset_dir"] == "datasets/anna"


def test_a_paused_fleet_queue_is_said_on_the_send_and_the_send_waits(world):
    send = world.create()
    world.fleet.queue_edit("pause")
    world.until(lambda: world.view(send["id"])["step"] == "queue")
    world.tick()
    view = world.view(send["id"])
    assert any("paused" in w for w in view["warnings"]) and view["state"] == "waiting"
    world.fleet.queue_edit("resume")
    world.until(lambda: world.view(send["id"])["step"] == "run")
    assert not any("paused" in w for w in world.view(send["id"])["warnings"])


def test_a_refused_dry_run_fails_the_send_with_the_agents_message(world):
    world.machines["1"].refuse = "Job name 't1' already exists"
    send = world.create()
    done = world.run_to_end(send["id"])
    assert done["state"] == "failed" and "already exists" in done["error"] and done["step"] == "check"
    assert not world.fleet.read()["queue"]["waiting"]  # nothing was queued
    assert world.market.of("stop") == []  # an instance this app did not rent is left running
    assert "left running" in done["error"]
    assert [e["kind"] for e in events(world, "send_failed")] == ["send_failed"]


def test_the_error_shown_has_no_address_or_token(world):
    world.machines["1"].refuse = "see https://example.org/x?token=SECRET and hf_abcdefghijklmnop1234"
    send = world.create()
    done = world.run_to_end(send["id"])
    assert "SECRET" not in done["error"] and "hf_abcdefghijklmnop1234" not in done["error"]
    assert "SECRET" not in json.dumps(events(world, "send_failed"))
    assert "SECRET" not in (world.tmp / "sends.json").read_text(encoding="utf-8")


def test_a_dispatch_the_instance_refuses_fails_the_send(world):
    send = world.create()
    world.until(lambda: world.view(send["id"])["step"] == "queue")
    world.machines["1"].refuse = ""
    orig = world.machines["1"].answer

    def refuse_real(args, config):
        if args[0] == "start" and "--dry-run" not in args:
            return 1, b"", b"[refused] A setting is invalid: boom"
        return orig(args, config)

    world.machines["1"].answer = refuse_real
    done = world.run_to_end(send["id"])
    assert done["state"] == "failed" and "boom" in done["error"]


def test_a_taken_job_name_and_a_wrong_choice_are_refused_at_create(world, tmp_path):
    world.create(job="t1")
    with pytest.raises(SendError, match="already being sent"):
        world.create(job="t1")
    with pytest.raises(SendError, match="Not a job name"):
        world.create(job="../x")
    with pytest.raises(SendError, match="one instance, or an offer"):
        world.mgr.create("t2", world.config(), world.root)
    with pytest.raises(SendError, match="After the run"):
        world.create(job="t3", after="explode")
    with pytest.raises(SendError, match="only for an instance this app rented"):
        world.create(job="t4", after="destroy")
    missing = world.config(dataset_dir=str(world.root / "nowhere"))
    with pytest.raises(SendError, match="cannot be sent"):
        world.create(job="t5", config=missing)


def test_an_instance_that_is_not_runnable_is_refused_at_create(world):
    world.market.rows[2] = mrow(2, is_rulz=False)
    world.market.rows[3] = mrow(3, status="exited", actual_status="exited", billing=False)
    world.market.rows[4] = mrow(4, status="loading", billing=True)
    with pytest.raises(SendError, match="not set up"):
        world.create(job="a", instance=2)
    with pytest.raises(SendError, match="starting it bills"):
        world.create(job="b", instance=3)
    with pytest.raises(SendError, match="not available|loading"):
        world.create(job="c", instance=4)
    with pytest.raises(SendError, match="not on your vast.ai account"):
        world.create(job="d", instance=99)


# ---------------------------------------------------------------- model files and datasets


def test_a_model_file_is_uploaded_and_the_config_reads_where_it_landed(world):
    model = world.root / "models" / "unet.safetensors"
    model.parent.mkdir()
    model.write_bytes(b"w" * 100)
    cfg = world.config(transformer_file=str(model), model_source="files")
    send = world.create(config=cfg, choices={str(model.resolve()): {"how": "upload"}})
    done = world.run_to_end(send["id"])
    assert done["state"] == "done"
    sent = world.files.calls[0]
    assert sent[0] == "upload_file" and sent[2].startswith("models/sent/") and sent[2].endswith("/unet.safetensors")
    assert world.machines["1"].configs["t1"]["transformer_file"] == sent[2]


def test_a_model_file_by_bucket_goes_up_then_the_instance_fetches_it(world):
    model = world.root / "unet.safetensors"
    model.write_bytes(b"w" * 100)
    send = world.create(config=world.config(transformer_file=str(model)),
                        choices={str(model.resolve()): {"how": "bucket", "bucket": "me/mine"}})
    world.run_to_end(send["id"])
    assert [c[0] for c in world.files.calls] == ["upload_to_bucket", "fetch_from_bucket"]
    assert world.files.calls[0][1] == "me/mine"
    assert world.machines["1"].configs["t1"]["transformer_file"] == world.files.calls[1][2]


def test_a_model_file_needs_a_choice_and_a_bucket_needs_a_name(world):
    model = world.root / "unet.safetensors"
    model.write_bytes(b"w")
    with pytest.raises(SendError, match="Choose how to send unet"):
        world.create(config=world.config(transformer_file=str(model)))
    with pytest.raises(SendError, match="owner/name"):
        world.create(config=world.config(transformer_file=str(model)),
                     choices={str(model.resolve()): {"how": "bucket", "bucket": "nonsense"}})


def test_a_model_folder_is_uploaded_as_a_tree(world):
    folder = world.root / "tok"
    folder.mkdir()
    (folder / "a.json").write_text("{}")
    send = world.create(config=world.config(tokenizer_path=str(folder), model_source="files"),
                        choices={str(folder.resolve()): {"how": "upload"}})
    world.run_to_end(send["id"])
    assert world.files.calls[0][0] == "upload_tree"


def test_upload_progress_is_shown_on_the_step(world):
    model = world.root / "unet.safetensors"
    model.write_bytes(b"w" * 100)
    seen = []
    world.files.on_upload = lambda what: seen.append(world.view(send["id"])["steps"][5]["state"])
    send = world.create(config=world.config(transformer_file=str(model)),
                        choices={str(model.resolve()): {"how": "upload"}})
    world.run_to_end(send["id"])
    assert seen == ["going"]
    assert world.view(send["id"])["steps"][5]["frac"] == 1.0


def test_a_dataset_is_pushed_once_and_a_second_send_of_the_same_folder_finds_its_mark(world):
    first = world.create(job="t1", after="leave")
    world.run_to_end(first["id"])
    assert len(world.sent) == 1 and world.machines["1"].marks["anna"] == world.sent[0][3]["anna"]
    second = world.create(job="t2", after="leave")
    world.run_to_end(second["id"])
    assert len(world.sent) == 1  # unchanged digest: skipped
    assert world.machines["1"].configs["t2"]["dataset_dir"] == "datasets/anna"


def test_a_changed_dataset_goes_beside_the_old_one_under_a_versioned_name(world):
    world.run_to_end(world.create(job="t1", after="leave")["id"])
    folder = world.root / "sets" / "anna"
    (folder / "new.png").write_bytes(b"more pictures")
    send = world.create(job="t2", after="leave")
    world.run_to_end(send["id"])
    versioned = world.sent[1][1][0]
    assert versioned.startswith("anna_") and len(versioned) == len("anna_") + 8
    assert world.machines["1"].configs["t2"]["dataset_dir"] == f"datasets/{versioned}"
    assert world.machines["1"].configs["t1"]["dataset_dir"] == "datasets/anna"


def test_a_dataset_folder_without_a_mark_is_not_taken_as_ours_when_the_local_folder_exists(world):
    m = world.machines["1"]
    m.datasets.append("anna")  # put there by hand, or by an older version: no mark
    world.run_to_end(world.create()["id"])
    versioned = world.sent[0][1][0]
    assert versioned.startswith("anna_") and world.machines["1"].configs["t1"]["dataset_dir"] == f"datasets/{versioned}"


def test_the_marks_are_asked_again_from_a_fresh_snapshot_for_every_dataset(world):
    second = world.dataset("bob")
    cfg = world.config(datasets=[{"path": str(world.root / "sets" / "anna")}, {"path": str(second)}])
    cfg.pop("dataset_dir")
    send = world.create(config=cfg)
    m = world.machines["1"]
    snaps_before = []
    orig = world.fleet.send

    def counting(target, items, where, **kw):
        snaps_before.append(len([c for c in m.calls if c[0] == "snapshot"]))
        return orig(target, items, where, **kw)

    world.fleet.send = counting
    world.run_to_end(send["id"])
    assert [s[1][0] for s in world.sent] == ["anna", "bob"]
    assert snaps_before[1] > snaps_before[0]  # the second decision looked again, after the first push


def test_a_references_set_is_staged_and_sent_as_one_folder(world, tmp_path):
    ref = world.root / "refs" / "face.png"
    ref.parent.mkdir()
    ref.write_bytes(b"face")
    cfg = world.config(face_refs=[str(ref)])
    send = world.create(config=cfg)
    world.run_to_end(send["id"])
    names = [s[1][0] for s in world.sent]
    assert any(n.startswith("refs_") for n in names)
    cfgs = world.machines["1"].configs["t1"]
    assert cfgs["face_refs"][0].startswith("datasets/refs_") and cfgs["face_refs"][0].endswith("/face.png")


def test_a_disk_too_small_for_what_goes_up_fails_before_any_upload(world):
    model = world.root / "unet.safetensors"
    model.write_bytes(b"w" * 100)
    world.machines["1"].disk = 0.0000001
    send = world.create(config=world.config(transformer_file=str(model)),
                        choices={str(model.resolve()): {"how": "upload"}})
    done = world.run_to_end(send["id"])
    assert done["state"] == "failed" and "not enough disk" in done["error"] and world.files.calls == []


# ---------------------------------------------------------------- renting


def offer():
    return {"id": 77, "num_gpus": 1}


def test_a_rented_send_waits_for_boot_joins_the_fleet_only_when_ssh_answers_and_arms_the_watchdog(world):
    world.machines["500"] = Machine(500)
    send = world.create(instance=None, rent={"offer": offer(), "disk": 200}, after="stop")
    assert world.market.of("rent") == [("rent", send["id"])] and send["instance"] == 500 and send["rented"]
    assert world.market.rows[500]["label"] == send["label"]
    world.tick()
    assert world.view(send["id"])["step"] == "boot" and "500" not in world.fleet.read()["instances"]
    # discovery must not take it meanwhile (its label is an open rental of a send that has not reached it)
    world.listed = lambda: [{"id": 500, "actual_status": "running", "label": send["label"], "gpu_name": "x",
                             "extra_env": [["RULZ_REPO", "x"]]}]
    world.fleet.discover()
    assert "500" not in world.fleet.read()["instances"]
    world.market.rows[500].update(status="running", actual_status="running")
    world.machines["500"].ssh_ok = False
    world.tick()
    world.tick()
    assert world.view(send["id"])["step"] == "reach" and "500" not in world.fleet.read()["instances"]
    world.machines["500"].ssh_ok = True
    world.until(lambda: world.view(send["id"])["step"] not in ("reach",))
    inst = world.fleet.read()["instances"]["500"]
    assert inst["reserved"] == send["id"]
    world.until(lambda: world.machines["500"].watchdog.get("armed"))
    assert world.machines["500"].watchdog["minutes"] == 60
    done = world.run_to_end(send["id"])
    assert done["state"] == "done" and ("stop", 500) in world.market.calls
    assert "reserved" not in world.fleet.read()["instances"]["500"]


def test_a_rent_is_made_once_after_the_send_was_saved_and_a_clear_refusal_drops_the_send(world):
    world.market.rent_error = MarketError("The price rose", "price", True)
    with pytest.raises(MarketError):
        world.create(instance=None, rent={"offer": offer()})
    assert world.mgr.sends() == [] and len(world.market.of("rent")) == 1


def test_a_rent_with_no_clear_answer_keeps_the_send_waiting_for_the_intent(world):
    world.market.rent_error = MarketError("no clear answer", "pending", False, label="rulz-trainer-abc12345")
    with pytest.raises(MarketError) as e:
        world.create(instance=None, rent={"offer": offer()})
    assert e.value.kind == "pending"
    send = world.mgr.sends()[0]
    assert send["state"] == "waiting" and send["step"] == "rent" and send["label"] == "rulz-trainer-abc12345"
    world.tick()
    assert world.view(send["id"])["state"] == "waiting"
    # the market settles it: the rental turns up
    world.market.intent_rows.clear()
    world.machines["501"] = Machine(501)
    world.market.rows[501] = mrow(501, label="rulz-trainer-abc12345", ours=True)
    world.market.rented.append({"id": 501, "label": "rulz-trainer-abc12345", "send": send["id"], "price": 0.5})
    world.tick()
    assert world.view(send["id"])["instance"] == 501 and world.view(send["id"])["step"] != "rent"
    assert len(world.market.of("rent")) == 1  # never rented again


def test_a_rent_whose_intent_vanished_without_a_rental_fails_the_send_and_is_not_retried(world):
    world.market.rent_error = MarketError("no clear answer", "pending", False, label="rulz-trainer-abc12345")
    with pytest.raises(MarketError):
        world.create(instance=None, rent={"offer": offer()})
    world.market.intent_rows.clear()
    send = world.mgr.sends()[0]
    done = world.run_to_end(send["id"])
    assert done["state"] == "failed" and done["step"] == "rent"
    with pytest.raises(SendError, match="Renting is not tried again"):
        world.mgr.retry(send["id"])
    assert len(world.market.of("rent")) == 1


def test_a_stopped_instance_is_started_only_by_create_and_only_with_start_true(world):
    world.market.rows[1].update(status="exited", actual_status="exited", billing=False)
    with pytest.raises(SendError, match="starting it bills"):
        world.create()
    send = world.create(start=True)
    assert world.market.of("start") == [("start", 1)]
    assert send["state"] == "going" and send["step"] == "boot"
    world.tick()
    world.tick()
    assert len(world.market.of("start")) == 1  # waiting for vast.ai, never starting again
    world.market.rows[1].update(status="running", actual_status="running")
    done = world.run_to_end(send["id"])
    assert done["state"] == "done" and ("stop", 1) in world.market.calls  # it started it: it may stop it


def test_a_boot_that_takes_too_long_fails_and_stops_what_this_app_started(world):
    world.market.rows[1].update(status="exited", actual_status="exited", billing=False)
    send = world.create(start=True)
    world.market.rows[1].update(status="loading", billing=True)
    for _ in range(60):
        world.tick(60)
        if world.view(send["id"])["state"] == "failed":
            break
    view = world.view(send["id"])
    assert view["state"] == "failed" and "25 minutes" in view["error"] and "stopped" in view["error"]
    assert ("stop", 1) in world.market.calls


def test_a_reach_that_takes_too_long_fails_and_stops_the_rental(world):
    world.machines["500"] = Machine(500)
    world.machines["500"].ssh_ok = False
    send = world.create(instance=None, rent={"offer": offer()})
    world.market.rows[500].update(status="running", actual_status="running")
    for _ in range(40):
        world.tick(60)
        if world.view(send["id"])["state"] == "failed":
            break
    view = world.view(send["id"])
    assert view["state"] == "failed" and "10 minutes" in view["error"] and ("stop", 500) in world.market.calls
    assert not world.market.of("destroy")  # a failed setup never destroys


def test_a_setup_that_takes_too_long_fails_and_stops_the_rental_but_not_when_leave_was_chosen(tmp_path):
    for after, stops in (("stop", True), ("leave", False)):
        w = World(tmp_path / after, ids=())
        w.machines["500"] = Machine(500)
        w.machines["500"].webui = False
        w.config_ = None
        send = w.create(instance=None, rent={"offer": offer()}, after=after)
        w.market.rows[500].update(status="running", actual_status="running")
        for _ in range(80):
            w.tick(60)
            if w.view(send["id"])["state"] == "failed":
                break
        view = w.view(send["id"])
        assert view["state"] == "failed" and "45 minutes" in view["error"] and view["step"] == "setup"
        assert (("stop", 500) in w.market.calls) is stops
        assert not w.market.of("destroy")


def test_a_failed_send_to_an_instance_another_send_uses_does_not_stop_it(tmp_path):
    w = World(tmp_path, ids=())
    w.machines["500"] = Machine(500)
    first = w.create(job="a", instance=None, rent={"offer": offer()}, after="stop")
    w.market.rows[500].update(status="running", actual_status="running")
    w.until(lambda: w.view(first["id"])["step"] not in ("boot", "reach"))
    second = w.create(job="b", instance=500, after="stop")
    w.machines["500"].refuse = "boom"
    w.until(lambda: w.view(second["id"])["state"] == "failed")
    assert ("stop", 500) not in w.market.calls


def test_the_queue_deadline_counts_only_while_the_instance_is_idle_and_free(world):
    send = world.create()
    world.until(lambda: world.view(send["id"])["step"] == "queue")
    world.fleet.queue_edit("pause")
    # a run that is going keeps the instance busy: it takes no time off the deadline
    world.machines["1"].run("other")
    for _ in range(40):
        world.tick(60)
    assert world.view(send["id"])["state"] != "failed"
    world.machines["1"].finish()
    for _ in range(40):
        world.tick(60)
        if world.view(send["id"])["state"] == "failed":
            break
    view = world.view(send["id"])
    assert view["state"] == "failed" and "30 minutes" in view["error"] and "paused" in view["error"]
    assert not world.fleet.read()["queue"]["waiting"]  # the entry was taken back out


# ---------------------------------------------------------------- the update step


def test_an_update_is_made_only_when_asked_and_the_instance_is_idle_and_older(world):
    m = world.machines["1"]
    m.behind = 3
    m.run("busy")
    send = world.create(update=True)
    world.until(lambda: world.view(send["id"])["step"] == "update")
    world.tick()
    view = world.view(send["id"])
    assert view["step"] == "update" and view["state"] == "waiting"
    assert not [c for c in m.calls if c[0] == "update" and "--check" not in c]
    m.finish()
    done = world.run_to_end(send["id"])
    assert done["state"] == "done" and m.version == "4.3.0"
    assert len([c for c in m.calls if c[0] == "update" and "--check" not in c]) == 1


def test_no_update_without_the_flag_and_none_when_the_version_is_current(world):
    m = world.machines["1"]
    m.behind = 3
    world.run_to_end(world.create(job="a", after="leave")["id"])
    assert not [c for c in m.calls if c[0] == "update"]
    m.behind = 0
    m.version = "4.3.0"
    world.run_to_end(world.create(job="b", update=True)["id"])
    assert [c for c in m.calls if c[0] == "update"] == [("update", "--check")]


# ---------------------------------------------------------------- tick never rents and never blocks


def test_uploads_run_in_a_worker_so_a_tick_does_not_wait_for_them(tmp_path):
    w = World(tmp_path, executor="deferred")
    send = w.create()
    w.until(lambda: w.view(send["id"])["step"] == "files")
    w.tick()
    assert w.jobs_deferred and w.files.calls == [] and w.sent == []
    job = w.jobs_deferred.pop(0)
    job()
    assert w.sent and w.view(send["id"])["step"] in ("check", "queue")


def test_nothing_in_tick_rents_or_starts_in_any_scenario_including_a_restart(world):
    send = world.create(after="stop")
    world.until(lambda: world.view(send["id"])["step"] == "run")
    world.restart()
    world.run_to_end(send["id"])
    world.restart()
    for _ in range(10):
        world.tick()
    assert world.market.of("rent") == [] and world.market.of("start") == []


@pytest.mark.parametrize("step", ["boot", "reach", "setup", "update", "files", "datasets", "check", "queue", "run",
                                  "download", "after"])
def test_a_restart_of_the_webui_at_every_step_goes_on_and_never_queues_twice(tmp_path, step):
    w = World(tmp_path)
    model = w.root / "unet.safetensors"
    model.write_bytes(b"w" * 10)
    w.machines["1"].behind = 1
    w.die_at = step if step in ("update", "files", "datasets", "check") else None  # killed in the middle of the step
    w.files.on_upload = lambda what: w.hook("files")
    send = w.create(config=w.config(transformer_file=str(model)), choices={str(model.resolve()): {"how": "upload"}},
                    update=True)
    w.market.rows[1].update(status="running", actual_status="running")
    if step == "boot":
        w.market.rows[1].update(status="loading", actual_status="loading")
    w.machines["1"].ssh_ok = step != "reach"
    w.machines["1"].webui = step != "setup"
    for _ in range(100):
        if w.died or (w.die_at is None and w.view(send["id"])["step"] == step):
            break
        try:
            w.tick()
        except Died:
            break
        if step in ("download", "after") and w.view(send["id"])["step"] in ("run", "download", "after"):
            w.finish_runs()
    assert w.view(send["id"])["step"] == step
    w.restart()
    w.die_at = None
    w.machines["1"].ssh_ok = True
    w.machines["1"].webui = True
    w.market.rows[1].update(status="running", actual_status="running")
    done = w.run_to_end(send["id"])
    assert done["state"] == "done"
    starts = [c for c in w.machines["1"].calls if c[0] == "start" and "--dry-run" not in c]
    assert len(starts) == 1
    assert len([r for r in w.fleet.runs(instance=1) if r.get("entry") == send["id"]]) == 1
    assert w.machines["1"].configs["t1"]["transformer_file"].startswith("models/sent/")


def test_the_manager_that_does_not_hold_the_lock_does_nothing_and_only_one_holds_it(world, tmp_path):
    other = SendManager(home=world.tmp, fleet=world.fleet, market=world.market, files=world.files, now=world.clock)
    assert world.mgr.hold() and not other.hold()
    assert other.step() is False
    assert world.mgr.step() is True and ("settle",) in world.market.calls
    world.mgr.release()
    assert other.hold()
    other.release()


# ---------------------------------------------------------------- after the run


def finish_run_and_reach_after(w, sid):
    w.until(lambda: w.view(sid)["step"] == "run" and w.view(sid)["run"])
    w.finish_runs()
    w.until(lambda: w.view(sid)["step"] == "after")


def test_the_countdown_is_three_minutes_shown_and_ends_with_the_stop(world):
    send = world.create(after="stop")
    finish_run_and_reach_after(world, send["id"])
    world.tick(20)
    world.tick(20)
    view = world.view(send["id"])
    assert view["action"] == "stop" and 0 < view["countdown"] <= 180 and view["state"] in ("going", "waiting")
    assert not world.market.of("stop")
    world.tick(60)
    assert not world.market.of("stop") and world.view(send["id"])["countdown"] > 0
    world.tick(200)
    assert world.market.of("stop") == [("stop", 1)]
    assert world.view(send["id"])["state"] == "done"


def test_keep_running_during_the_countdown_cancels_the_action_and_switches_the_watchdog_off(world):
    send = world.create(after="stop")
    finish_run_and_reach_after(world, send["id"])
    world.tick(20)
    assert world.view(send["id"])["countdown"] is not None
    assert world.machines["1"].watchdog.get("armed")
    kept = world.mgr.keep(send["id"])
    assert kept["after"] == "leave" and kept["countdown"] is None
    world.tick(300)
    done = world.view(send["id"])
    assert done["state"] == "done" and "left running" in done["result"] and not world.market.of("stop")
    assert not world.machines["1"].watchdog.get("armed")


def test_a_condition_that_appears_before_the_countdown_ends_stops_the_stop(world):
    send = world.create(after="stop")
    finish_run_and_reach_after(world, send["id"])
    world.tick(20)
    assert world.view(send["id"])["countdown"] is not None
    world.machines["1"].run("someone_elses_run")  # a run was started there by hand
    world.tick(200)
    assert not world.market.of("stop")
    view = world.view(send["id"])
    assert view["countdown"] is None and view["state"] == "waiting"
    world.machines["1"].finish()
    world.tick(30)
    world.tick(30)
    assert world.view(send["id"])["countdown"] is not None  # a new countdown, from its start
    world.tick(200)
    assert world.market.of("stop") == [("stop", 1)]


def test_a_restart_of_the_webui_starts_the_countdown_again_from_the_beginning(world):
    send = world.create(after="stop")
    finish_run_and_reach_after(world, send["id"])
    world.tick(20)
    world.tick(150)
    assert world.view(send["id"])["countdown"] is not None
    world.restart()
    assert world.mgr.hold()
    assert world.view(send["id"])["countdown"] is None
    world.tick(30)
    world.tick(30)
    view = world.view(send["id"])
    assert view["countdown"] is not None and view["countdown"] > 100
    assert not world.market.of("stop")


@pytest.mark.parametrize("block", ["queue", "tests", "entry", "download", "final"])
def test_each_condition_holds_the_action_back_in_the_real_flow(world, block):
    send = world.create(after="stop")
    finish_run_and_reach_after(world, send["id"])
    m = world.machines["1"]
    if block == "queue":
        m.queue_waiting = [{"job": "x"}]
    elif block == "tests":
        m.tests = 1
    elif block == "entry":
        world.fleet.enqueue("pinned", {"steps": 1}, instance=1)
    elif block == "final":
        key = world.fleet.runs()[0]["key"]
        world.fleet.update_run(key, lambda r: r.update(final_pull=True, pull_fails=0, state="done"))
        world.fleet.pull = lambda *a, **k: None  # the download never finishes
    elif block == "download":
        world.fleet.downloads = lambda keep=15.0: [{"instance": 1, "state": "going", "run": "x"}]
    for _ in range(12):
        world.tick(60)
    assert not world.market.of("stop"), block
    assert world.view(send["id"])["countdown"] is None


def test_an_unpinned_entry_does_not_hold_the_stop(world):
    send = world.create(after="stop")
    finish_run_and_reach_after(world, send["id"])
    world.fleet.queue_edit("pause")
    world.fleet.enqueue("loose", {"steps": 1})
    world.tick(30)
    world.tick(30)
    assert world.view(send["id"])["countdown"] is not None


def test_a_stop_that_keeps_failing_to_work_is_followed_without_blocking_and_ends_in_a_loud_alert(world):
    world.market.stop_works = False
    send = world.create(after="stop")
    finish_run_and_reach_after(world, send["id"])
    world.run_to_end(send["id"])
    start = world.clock.t
    for _ in range(40):
        world.tick(60)
        if events(world, "send_alert"):
            break
    assert events(world, "send_alert") and world.clock.t - start >= 600
    assert "still billing" in events(world, "send_alert")[0]["error"]
    assert world.sleeps == []
    world.market.stop_works = True
    world.market.rows[1].update(status="exited", actual_status="exited", billing=False)
    world.tick(130)
    assert events(world, "instance_stopped_after_run")


def test_with_several_sends_the_mildest_choice_wins_and_all_must_be_over(tmp_path):
    w = World(tmp_path)
    a = w.create(job="a", after="stop")
    b = w.create(job="b", after="leave")
    w.run_to_end(a["id"], finish=True)
    w.run_to_end(b["id"], finish=True)
    assert not w.market.of("stop")
    assert "left running" in w.view(a["id"])["result"]


def test_a_stop_waits_for_the_other_send_to_be_over(tmp_path):
    w = World(tmp_path)
    a = w.create(job="a", after="stop")
    w.until(lambda: w.view(a["id"])["step"] == "run")
    b = w.create(job="b", after="stop")
    w.finish_runs()
    w.until(lambda: w.view(a["id"])["step"] == "after")
    for _ in range(4):
        w.tick(60)
    assert not w.market.of("stop") and w.view(b["id"])["step"] != "after"
    w.run_to_end(b["id"])
    w.run_to_end(a["id"])
    assert len(w.market.of("stop")) == 1


def test_the_final_download_failing_still_stops_the_instance_and_says_the_files_are_there(world):
    def failing_fetch(*a, **k):
        raise OSError("disk full")
    world.fleet.fetch = failing_fetch
    send = world.create(after="stop")
    done = world.run_to_end(send["id"])
    assert done["state"] == "done" and world.market.of("stop") == [("stop", 1)]
    assert "files are still on its disk" in done["result"]


# ---------------------------------------------------------------- destroy


def rented_world(tmp_path, tree=True):
    w = World(tmp_path, ids=())
    w.machines["500"] = Machine(500)
    send = w.create(instance=None, rent={"offer": offer()}, after="destroy")
    w.market.rows[500].update(status="running", actual_status="running")
    return w, send


def test_destroy_checks_the_tree_downloads_it_and_destroys_only_what_this_app_rented(tmp_path):
    w, send = rented_world(tmp_path)
    w.machines["500"].tree = {"t1": [("t1_best.safetensors", 7), ("log.txt", 3)]}
    done = w.run_to_end(send["id"])
    assert done["state"] == "done" and w.market.of("destroy") == [("destroy", 500, True)]
    assert not w.market.of("stop")
    assert [e["kind"] for e in events(w, "instance_destroyed_after_run")] == ["instance_destroyed_after_run"]
    rec = w.fleet.runs()[0]
    assert all(rec["download"][k] != "off" for k in fleet_module.DOWNLOAD_KINDS)  # every kind at least "end"
    folder = w.fleet.pull_folder(rec)
    assert (folder / "t1_best.safetensors").stat().st_size == 7


@pytest.mark.parametrize("case", ["failed_run", "foreign_job", "tests", "missing_file"])
def test_a_destroy_that_cannot_be_proven_safe_stops_instead_with_the_reason(tmp_path, case):
    w, send = rented_world(tmp_path)
    m = w.machines["500"]
    m.tree = {"t1": [("a.bin", 4)]}
    if case == "foreign_job":
        m.extra_jobs = ["somebody_elses"]
    if case == "tests":
        m.tests_dir_empty = False
    if case == "missing_file":
        w.fleet.pull_now = lambda key, kinds: {"error": "dropped"}
    w.until(lambda: w.view(send["id"])["step"] == "run" and w.view(send["id"])["run"])
    w.machines["500"].finish("error" if case == "failed_run" else "done")
    done = w.run_to_end(send["id"], finish=False)
    assert done["state"] == "done" and not w.market.of("destroy") and w.market.of("stop") == [("stop", 500)]
    assert "not destroyed" in done["result"]


def test_destroy_refused_by_the_market_falls_back_to_stop(tmp_path):
    w, send = rented_world(tmp_path)
    w.market.rented.clear()  # no longer a rental of this app
    w.machines["500"].tree = {"t1": [("a.bin", 4)]}
    done = w.run_to_end(send["id"])
    assert done["state"] == "done" and w.market.of("stop") == [("stop", 500)]
    assert "destroy was refused" in done["result"]


# ---------------------------------------------------------------- the watchdog


def test_the_watchdog_is_armed_for_stop_and_destroy_never_for_leave(tmp_path):
    for after, armed in (("stop", True), ("destroy", True), ("leave", False)):
        w = World(tmp_path / after, ids=())
        w.machines["500"] = Machine(500)
        send = w.create(instance=None, rent={"offer": offer()}, after=after, idle_minutes=90)
        w.market.rows[500].update(status="running", actual_status="running")
        w.until(lambda: w.view(send["id"])["step"] == "queue")
        assert bool(w.machines["500"].watchdog.get("armed")) is armed
        if armed:
            assert w.machines["500"].watchdog["minutes"] == 90


def test_the_alive_file_is_touched_while_the_send_is_before_the_run_and_a_gone_watchdog_is_armed_again(world):
    m = world.machines["1"]
    m.watch_calls = []
    model = world.root / "unet.safetensors"
    model.write_bytes(b"w")
    world.files.on_upload = lambda what: None
    send = world.create(after="stop")
    world.until(lambda: m.watchdog.get("armed"))
    m.watchdog = {"armed": False, "fired": False}  # its process is gone
    world.fleet.queue_edit("pause")
    world.until(lambda: world.view(send["id"])["step"] == "queue")
    for _ in range(4):
        world.tick(30)
    assert m.watch_calls.count("arm") >= 2 and "alive" in m.watch_calls


def test_a_watchdog_that_cannot_stop_the_instance_is_a_warning_and_the_send_goes_on(world):
    world.machines["1"].can_stop = False
    send = world.create(after="stop")
    done = world.run_to_end(send["id"])
    assert done["state"] == "done"
    # the warning shows while the send is under way
    world.market.rows[1].update(status="running", actual_status="running", billing=True)
    send2 = world.create(job="t2", after="stop")
    world.until(lambda: world.view(send2["id"])["warnings"])
    assert any("cannot stop itself" in w for w in world.view(send2["id"])["warnings"])


def test_an_autostop_that_is_armed_there_is_reported(world):
    world.machines["1"].autostop = True
    send = world.create(after="stop")
    world.until(lambda: any("AutoStop" in w for w in world.view(send["id"])["warnings"]))


def test_a_watchdog_that_cannot_be_armed_yet_is_tried_again(world):
    m = world.machines["1"]
    m.arm_fails = True
    send = world.create(after="stop")
    for _ in range(6):
        world.tick()
    assert not m.watchdog.get("armed")
    m.arm_fails = False
    world.until(lambda: m.watchdog.get("armed"))
    assert send


# ---------------------------------------------------------------- interrupted, cancelled, retry


def test_an_interrupted_instance_is_shown_and_watched_and_failed_when_back_with_no_run(tmp_path):
    w = World(tmp_path, ids=())
    w.machines["500"] = Machine(500)
    send = w.create(instance=None, rent={"offer": offer()}, after="destroy")
    w.market.rows[500].update(status="running", actual_status="running")
    w.fleet.queue_edit("pause")
    w.until(lambda: w.view(send["id"])["step"] == "queue")
    w.market.rows[500].update(status="exited", actual_status="exited", billing=False)
    w.tick()
    w.tick()
    view = w.view(send["id"])
    assert view["state"] == "interrupted" and any("interrupted" in x for x in view["warnings"])
    assert not w.market.of("stop") and not w.market.of("destroy")
    w.market.rows[500].update(status="running", actual_status="running", billing=True)
    w.tick()
    w.tick()
    view = w.view(send["id"])
    assert view["state"] == "failed" and "Retry on the instance's own page" in view["error"]
    assert view["after"] == "stop"  # after-run falls to stop
    assert ("stop", 500) in w.market.calls and not w.market.of("destroy")


def test_an_instance_that_left_the_account_fails_the_send_without_a_stop(world):
    send = world.create()
    world.until(lambda: world.view(send["id"])["step"] == "files")
    world.market.rows.pop(1)
    world.tick()
    world.tick()
    view = world.view(send["id"])
    assert view["state"] == "failed" and "no longer on your vast.ai account" in view["error"]
    assert not world.market.of("stop")


def test_cancel_before_the_queue_ends_the_send_and_can_stop_a_rental(tmp_path):
    w = World(tmp_path, ids=())
    w.machines["500"] = Machine(500)
    send = w.create(instance=None, rent={"offer": offer()}, after="stop")
    w.market.rows[500].update(status="running", actual_status="running")
    w.until(lambda: w.view(send["id"])["step"] in ("setup", "files"))
    view = w.mgr.cancel(send["id"], instance_action="stop")
    assert view["state"] == "cancelled" and ("stop", 500) in w.market.calls
    w.tick()
    assert w.view(send["id"])["state"] == "cancelled"
    assert "reserved" not in w.fleet.read()["instances"]["500"]


def test_cancel_with_keep_leaves_the_instance_and_switches_the_watchdog_off(tmp_path):
    w = World(tmp_path, ids=())
    w.machines["500"] = Machine(500)
    send = w.create(instance=None, rent={"offer": offer()}, after="stop")
    w.market.rows[500].update(status="running", actual_status="running")
    w.until(lambda: w.machines["500"].watchdog.get("armed"))
    w.mgr.cancel(send["id"], instance_action="keep")
    assert not w.market.of("stop") and not w.machines["500"].watchdog.get("armed")


def test_cancel_during_an_upload_stops_the_worker_without_failing_the_send(world):
    model = world.root / "unet.safetensors"
    model.write_bytes(b"w" * 10)
    send = world.create(config=world.config(transformer_file=str(model)),
                        choices={str(model.resolve()): {"how": "upload"}})
    world.files.on_upload = lambda what: world.mgr.cancel(send["id"])
    world.until(lambda: world.view(send["id"])["state"] == "cancelled")
    for _ in range(3):
        world.tick()
    assert world.view(send["id"])["state"] == "cancelled" and len(world.files.calls) == 1


def test_cancel_after_the_run_is_queued_is_the_fleets_job(world):
    send = world.create()
    world.until(lambda: world.view(send["id"])["step"] in ("run",))
    with pytest.raises(SendError, match="Fleet"):
        world.mgr.cancel(send["id"])


def test_a_cancel_click_that_loses_to_the_queue_is_refused(world, monkeypatch):
    send = world.create()
    world.until(lambda: world.view(send["id"])["step"] == "queue")
    stale = world.mgr._raw(send["id"])
    world.until(lambda: world.view(send["id"])["step"] == "run")
    monkeypatch.setattr(world.mgr, "_raw", lambda sid: stale)  # what the click read before the run was queued
    with pytest.raises(SendError, match="Fleet"):
        world.mgr.cancel(send["id"])
    monkeypatch.undo()
    assert world.view(send["id"])["state"] == "going" and not world.mgr._cancelled(send["id"])


def test_retry_goes_on_from_the_failed_step(world):
    world.machines["1"].refuse = "boom"
    send = world.create()
    done = world.run_to_end(send["id"])
    assert done["state"] == "failed" and done["step"] == "check"
    world.machines["1"].refuse = ""
    again = world.mgr.retry(send["id"])
    assert again["state"] == "going" and again["step"] == "check" and again["error"] is None
    assert world.run_to_end(send["id"])["state"] == "done"
    assert len(world.sent) == 1  # the datasets were not pushed again


def test_retry_of_something_that_did_not_fail_is_refused(world):
    send = world.create()
    with pytest.raises(SendError, match="failed"):
        world.mgr.retry(send["id"])


def test_a_dropped_connection_in_an_upload_is_tried_again_up_to_five_times_in_a_row(world):
    model = world.root / "unet.safetensors"
    model.write_bytes(b"w" * 10)
    world.files.fail_with = rulz_files.FilesError("could not be reached", "ssh")
    send = world.create(config=world.config(transformer_file=str(model)),
                        choices={str(model.resolve()): {"how": "upload"}})
    done = world.run_to_end(send["id"])
    assert done["state"] == "failed" and "keeps failing" in done["error"]
    assert len(world.files.calls) == send_module.RETRIES + 1
    world.files.fail_with = None
    world.mgr.retry(send["id"])
    assert world.run_to_end(send["id"])["state"] == "done"


def test_a_checksum_failure_in_an_upload_is_final_at_once(world):
    model = world.root / "unet.safetensors"
    model.write_bytes(b"w" * 10)
    world.files.fail_with = rulz_files.FilesError("arrived changed twice", "checksum")
    send = world.create(config=world.config(transformer_file=str(model)),
                        choices={str(model.resolve()): {"how": "upload"}})
    done = world.run_to_end(send["id"])
    assert done["state"] == "failed" and len(world.files.calls) == 1


def test_a_lost_run_is_not_final_while_a_look_still_shows_it(world):
    send = world.create()
    world.until(lambda: world.view(send["id"])["step"] == "run" and world.view(send["id"])["run"])
    key = world.view(send["id"])["run"]
    world.fleet.update_run(key, lambda r: r.update(state="lost"))
    world.tick()
    world.tick()
    assert world.view(send["id"])["step"] == "run"  # the machine still runs it: the Fleet sets it running again
    world.machines["1"].finish()
    world.fleet.update_run(key, lambda r: r.update(state="lost"))
    world.machines["1"].jobs["t1"]["final"] = "done"
    world.tick()
    assert world.view(send["id"])["step"] in ("download", "after", "run")


# ---------------------------------------------------------------- the views


def test_the_views_have_the_shape_the_ui_uses(world):
    send = world.create()
    view = world.view(send["id"])
    assert set(view) >= {"id", "job", "created", "instance", "label", "after", "idle_minutes", "state", "step", "steps",
                         "error", "warnings", "per_hour", "rented", "run", "entry", "countdown", "action", "result"}
    assert [s["key"] for s in view["steps"]] == list(send_module.STEP_KEYS)
    assert set(view["steps"][0]) == {"key", "label", "state", "text", "frac", "started", "ended"}
    assert world.mgr.sends()[0]["id"] == send["id"] and world.mgr.sends(active=True)
    world.run_to_end(send["id"])
    assert world.mgr.sends(active=True) == []
    with pytest.raises(SendError):
        world.mgr.get("nope")
    world.mgr.dismiss(send["id"])
    assert world.mgr.sends() == []


def test_instances_joins_vast_ais_list_with_the_fleets_last_look(world):
    world.market.rows[2] = mrow(2, status="exited", actual_status="exited", billing=False, per_hour=0.1)
    world.market.rows[3] = mrow(3, is_rulz=False)
    world.market.rows[4] = mrow(4, status="loading")
    world.machines["1"].run("busy")
    world.fleet.add_instance(1)
    world.tick()
    world.mgr._rows(fresh=True)
    rows = {r["id"]: r for r in world.mgr.instances()}
    assert rows[1]["selectable"] and rows[1]["job"] == "busy" and rows[1]["version"] == "4.2.0"
    assert rows[1]["free_gb"] == 500.0
    assert rows[2]["selectable"] and "starting it bills $0.800/hour" in rows[2]["reason"]
    assert not rows[3]["selectable"] and "RulZ AI Trainer" in rows[3]["reason"]
    assert not rows[4]["selectable"] and "loading" in rows[4]["reason"]
    assert set(rows[1]) >= {"id", "gpu_name", "num_gpus", "gpu_ram_mb", "per_hour", "dph_total", "storage_per_hour",
                            "status", "label", "ours", "is_rulz", "selectable", "reason", "free_gb", "version", "job",
                            "waiting", "sends"}


def test_set_after_changes_the_choice_and_destroy_is_only_for_rentals(tmp_path):
    w, send = rented_world(tmp_path)
    assert w.mgr.set_after(send["id"], "stop")["after"] == "stop"
    assert w.mgr.set_after(send["id"], "destroy")["after"] == "destroy"
    with pytest.raises(SendError):
        w.mgr.set_after(send["id"], "explode")
    other = World(tmp_path / "o").create(job="x")
    assert other["rented"] is False
    w2 = World(tmp_path / "p")
    s2 = w2.create()
    with pytest.raises(SendError, match="only for an instance this app rented"):
        w2.mgr.set_after(s2["id"], "destroy")


def test_the_sends_file_never_holds_a_token_an_address_or_an_environment(tmp_path):
    w, send = rented_world(tmp_path)
    w.run_to_end(send["id"])
    text = (w.tmp / "sends.json").read_text(encoding="utf-8")
    assert "http" not in text and "hf_" not in text and "extra_env" not in text


def test_a_stop_somebody_else_sent_can_be_followed(world):
    world.mgr.follow_stop(1)
    world.tick(130)
    assert not events(world, "send_alert")
    world.tick(130)
    assert world.mgr._read_stops() == [x for x in world.mgr._read_stops() if x["instance"] == 1]
    world.market.rows[1].update(status="exited", actual_status="exited", billing=False)
    world.tick(130)
    assert world.mgr._read_stops() == []


# ---------------------------------------------------------------- review fixes


def test_a_sends_file_that_cannot_be_read_is_never_overwritten(world):
    send = world.create()
    path = world.tmp / "sends.json"
    (world.tmp / "sends.json.bak").unlink(missing_ok=True)  # with no good copy to go back to
    path.write_text("{ this is not json", encoding="utf-8")
    with pytest.raises(SendError, match="not valid JSON"):
        world.mgr.sends()
    world.mgr.tick()
    assert path.read_text(encoding="utf-8") == "{ this is not json"
    assert any("not valid JSON" in e for e in world.mgr.errors)
    world.mgr.errors.clear()
    path.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(SendError, match="not what was expected"):
        world.mgr.get(send["id"])
    path.unlink()  # only a missing file starts empty
    assert world.mgr.sends() == []


def test_a_sharing_violation_is_tried_again_and_then_refused(world, monkeypatch):
    world.create()
    real = Path.read_text
    seen = []

    def flaky(self, *a, **k):
        if self.name == "sends.json":
            seen.append(1)
            if len(seen) < 3:
                raise PermissionError("in use")
        return real(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", flaky)
    assert len(world.mgr.sends()) == 1 and len(seen) == 3

    def always(self, *a, **k):
        if self.name == "sends.json":
            raise PermissionError("in use")
        return real(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", always)
    with pytest.raises(SendError, match="could not be read"):
        world.mgr.sends()


def test_when_the_lock_cannot_be_had_nothing_is_written(tmp_path):
    (tmp_path / "sends.json.lock").mkdir()  # the lock file cannot be opened
    w = World(tmp_path)
    with pytest.raises(SendError, match="could not be locked"):
        w.create()
    assert not (tmp_path / "sends.json").exists()


def test_a_start_whose_answer_is_unclear_keeps_the_send_and_boots_deadline_stops_it(world):
    world.market.rows[1].update(status="exited", actual_status="exited", billing=False)
    world.market.start_error = MarketError("the connection dropped", "network", False)
    send = world.create(start=True)
    assert send["state"] == "waiting" and send["step"] == "boot"
    assert any("no clear answer" in w for w in send["warnings"])
    assert world.mgr._raw(send["id"])["started"] is True
    world.market.rows[1].update(status="loading", billing=True)  # it was started after all
    for _ in range(60):
        world.tick(60)
        if world.view(send["id"])["state"] == "failed":
            break
    assert world.view(send["id"])["state"] == "failed" and ("stop", 1) in world.market.calls


def test_a_start_that_was_clearly_refused_drops_the_send(world):
    world.market.rows[1].update(status="exited", actual_status="exited", billing=False)
    world.market.start_error = MarketError("no", "client", True)
    with pytest.raises(MarketError):
        world.create(start=True)
    assert world.mgr.sends() == []


@pytest.mark.parametrize("error", [MarketError("server error", "server", False), OSError("reset by peer")])
def test_a_rent_error_that_is_not_definite_keeps_the_send_in_the_rent_step(world, error):
    world.market.rent_error = error
    with pytest.raises(MarketError) as e:
        world.create(instance=None, rent={"offer": offer()})
    assert not e.value.definite
    send = world.mgr.sends()[0]
    assert send["step"] == "rent" and send["state"] == "waiting"
    world.machines["501"] = Machine(501)
    world.market.rows[501] = mrow(501, label="rulz-trainer-x", ours=True)
    world.market.rented.append({"id": 501, "label": "rulz-trainer-x", "send": send["id"], "price": 0.5})
    world.tick()
    assert world.view(send["id"])["instance"] == 501 and len(world.market.of("rent")) == 1


def test_cancel_while_the_rent_is_unsettled_is_refused(world):
    world.market.rent_error = MarketError("no clear answer", "pending", False, label="rulz-trainer-abc12345")
    with pytest.raises(MarketError):
        world.create(instance=None, rent={"offer": offer()})
    send = world.mgr.sends()[0]
    with pytest.raises(SendError, match="has not answered the rent yet"):
        world.mgr.cancel(send["id"])
    world.market.rent_error = MarketError("server", "server", False)
    world.market.intent_rows.clear()
    with pytest.raises(MarketError):
        world.create(job="t2", instance=None, rent={"offer": offer()})
    second = next(x for x in world.mgr.sends() if x["job"] == "t2")
    with pytest.raises(SendError, match="has not answered the rent yet"):
        world.mgr.cancel(second["id"])


def test_a_rental_no_send_owns_and_vast_still_bills_is_one_loud_alert_and_never_stopped_silently(tmp_path):
    w = World(tmp_path, ids=())
    w.machines["500"] = Machine(500)
    send = w.create(instance=None, rent={"offer": offer()}, after="stop")
    # the send ends before it reached the instance, and the instance stays billed
    w.mgr._mut(send["id"], lambda x: x.update(state="failed", error="x", keep_ok=False))
    w.market.rows[500].update(status="running", actual_status="running", billing=True)
    assert w.mgr.hold()
    w.mgr.step()
    w.mgr.step()
    alerts = events(w, "send_alert")
    assert len(alerts) == 1 and alerts[0]["instance"] == 500
    assert any("no send is using it" in x for x in w.view(send["id"])["warnings"])
    assert not w.market.of("stop")
    w.mgr._drop(send["id"])  # the send no longer exists: still billed, already alerted once
    w.mgr.step()
    assert len(events(w, "send_alert")) == 1
    w.mgr.release()


def rented_with_upload(tmp_path):
    w = World(tmp_path, ids=())
    m = Machine(500)
    w.machines["500"] = m
    model = w.root / "unet.safetensors"
    model.write_bytes(b"w" * 10)

    def hook(what):
        m.down = True
        raise rulz_files.FilesError("arrived changed twice", "checksum")
    w.files.on_upload = hook
    send = w.create(instance=None, rent={"offer": offer()}, after="stop",
                    config=w.config(transformer_file=str(model)), choices={str(model.resolve()): {"how": "upload"}})
    w.market.rows[500].update(status="running", actual_status="running")
    return w, send


def test_a_failure_with_the_instance_out_of_reach_waits_for_a_look_before_stopping(tmp_path):
    w, send = rented_with_upload(tmp_path)
    w.until(lambda: w.view(send["id"])["state"] == "failed")
    assert not w.market.of("stop") and "as soon as it can see" in w.view(send["id"])["error"]
    assert w.mgr._read()["pending_stops"]
    for _ in range(3):
        w.tick(60)
    assert not w.market.of("stop")
    w.machines["500"].down = False  # a look works: nothing runs there
    w.tick(60)
    w.tick(60)
    assert w.market.of("stop") == [("stop", 500)] and not w.mgr._read()["pending_stops"]


def test_a_failure_waits_no_longer_than_30_minutes_and_stops_when_no_run_was_ever_seen(tmp_path):
    w, send = rented_with_upload(tmp_path)
    w.until(lambda: w.view(send["id"])["state"] == "failed")
    for _ in range(40):
        w.tick(60)
        if w.market.of("stop"):
            break
    assert len(events(w, "send_alert")) == 1 and w.market.of("stop") == [("stop", 500)]


def test_a_failure_that_cannot_look_only_alerts_when_a_run_was_seen_there(tmp_path):
    w, send = rented_with_upload(tmp_path)
    w.until(lambda: w.view(send["id"])["state"] == "failed")
    w.fleet.save_run({"key": "500-x", "instance": 500, "job": "x", "state": "running", "first_seen": w.clock.t + 1,
                      "entry": "other"})
    for _ in range(40):
        w.tick(60)
    assert len(events(w, "send_alert")) == 1 and not w.market.of("stop")


@pytest.mark.parametrize("what", ["job", "queue", "tests", "download"])
def test_a_failure_leaves_the_instance_alone_when_something_is_going_there(tmp_path, what):
    w = World(tmp_path, ids=())
    m = Machine(500)
    w.machines["500"] = m
    model = w.root / "unet.safetensors"
    model.write_bytes(b"w")

    def hook(_):
        if what == "job":
            m.run("somebody")
        elif what == "queue":
            m.queue_waiting = [{"job": "x"}]
        elif what == "tests":
            m.tests = 1
        else:
            w.fleet.downloads = lambda keep=15.0: [{"instance": 500, "state": "going", "run": "x"}]
        raise rulz_files.FilesError("changed", "checksum")
    w.files.on_upload = hook
    send = w.create(instance=None, rent={"offer": offer()}, after="stop", config=w.config(transformer_file=str(model)),
                    choices={str(model.resolve()): {"how": "upload"}})
    w.market.rows[500].update(status="running", actual_status="running")
    w.until(lambda: w.view(send["id"])["state"] == "failed")
    assert not w.market.of("stop") and "left running" in w.view(send["id"])["error"]


def to_files_step(world, send):
    world.mgr._mut(send["id"], lambda x: x.update(rented=True))
    world.until(lambda: world.view(send["id"])["step"] in ("reach", "setup", "files"))


def test_a_stop_that_failed_is_tried_again_every_two_minutes_and_alerted_once(world):
    world.market.stop_errors = [MarketError("server error", "server", False)] * 2
    send = world.create(after="stop")
    to_files_step(world, send)
    world.mgr.cancel(send["id"], instance_action="stop")
    assert len(world.market.of("stop")) == 1 and len(events(world, "send_alert")) == 1
    world.tick(130)
    assert len(world.market.of("stop")) == 2
    world.tick(130)
    assert len(world.market.of("stop")) == 3 and world.market.rows[1]["billing"] is False
    assert len(events(world, "send_alert")) == 1


def test_a_stop_vast_accepted_but_the_instance_still_runs_is_sent_again_every_ten_minutes(world):
    world.market.stop_works = False
    send = world.create(after="stop")
    to_files_step(world, send)
    world.mgr.cancel(send["id"], instance_action="stop")
    for _ in range(25):
        world.tick(60)
    assert len(world.market.of("stop")) == 3  # at once, after 10 minutes, after 20
    assert len(events(world, "send_alert")) == 1


def test_an_instance_stopped_before_the_last_download_ends_the_send(world):
    send = world.create(after="stop")
    world.until(lambda: world.view(send["id"])["run"])
    world.finish_runs()
    world.tick()  # the run is seen to end
    world.market.rows[1].update(status="exited", actual_status="exited", billing=False)  # outbid, or the watchdog
    done = world.run_to_end(send["id"])
    assert done["state"] == "done" and "stopped before the last download" in done["result"]
    assert not world.market.of("stop")


def test_an_instance_gone_in_the_download_or_after_step_ends_the_send(world):
    send = world.create(after="stop")
    world.until(lambda: world.view(send["id"])["run"])
    world.finish_runs()
    world.tick()  # the run is seen to end
    world.market.rows.pop(1)
    world.machines["1"].down = True  # destroyed under the send: nothing answers any more
    done = world.run_to_end(send["id"])
    assert done["state"] == "done" and "gone" in done["result"]


def test_an_instance_that_stopped_during_the_after_step_is_already_stopped(world):
    send = world.create(after="stop")
    finish_run_and_reach_after(world, send["id"])
    world.market.rows[1].update(status="exited", actual_status="exited", billing=False)
    world.tick()
    done = world.view(send["id"])
    assert done["state"] == "done" and "already stopped" in done["steps"][-1]["text"] and not world.market.of("stop")


def test_a_running_instance_that_does_not_answer_for_30_minutes_is_an_alert_and_is_stopped_when_the_run_is_over(world):
    send = world.create(after="stop")
    finish_run_and_reach_after(world, send["id"])
    world.machines["1"].down = True
    for _ in range(40):
        world.tick(60)
        if world.market.of("stop"):
            break
    alerts = events(world, "send_alert")
    assert len(alerts) == 1 and "30 minutes" in alerts[0]["error"] and world.market.of("stop") == [("stop", 1)]
    assert any("not destroyed" in x for x in world.view(send["id"])["warnings"])
    assert not world.market.of("destroy")


def test_an_unreachable_instance_is_not_stopped_when_the_choice_is_leave(world):
    send = world.create(after="leave")
    finish_run_and_reach_after(world, send["id"])
    world.machines["1"].down = True
    for _ in range(3):
        world.tick(60)
    assert world.view(send["id"])["state"] == "done"  # leave: nothing to do
    assert not world.market.of("stop")


def test_a_worker_the_lane_did_not_take_does_not_stay_marked_as_running(tmp_path):
    w = World(tmp_path)
    w.mgr.executor = lambda iid, mark, func: False
    send = w.create()
    for _ in range(8):
        w.tick()
        if w.view(send["id"])["step"] == "files":
            break
    w.tick()
    assert w.mgr._running == set()
    w.mgr.executor = lambda iid, mark, func: func()
    w.run_to_end(send["id"])
    assert w.view(send["id"])["state"] == "done"


def test_a_new_holder_restarts_what_a_dead_worker_left_going(world):
    send = world.create()
    world.mgr._mut(send["id"], lambda x: x.update(tree="going", dcheck={"state": "going", "at": None}))
    world.restart()
    assert world.mgr.hold()
    got = world.mgr._raw(send["id"])
    assert got["tree"] is None and got["dcheck"] is None


@pytest.mark.parametrize("job", ["my run é 1", "Ünïcode name", "a b", "x" * 100])
def test_job_names_the_webui_accepts_are_accepted(world, job):
    assert world.create(job=job)["job"] == job


@pytest.mark.parametrize("job", ["", " ", ".hidden", "-flag", "trailing.", "trailing ", "a/b", "a\\b", "x" * 101,
                                 "ctrl\x01name", ".", ".."])
def test_job_names_that_could_leave_their_folder_are_refused(world, job):
    with pytest.raises(SendError, match="Not a job name"):
        world.create(job=job)


def test_job_names_are_compared_without_regard_to_case(world):
    world.create(job="Run One")
    with pytest.raises(SendError, match="already being sent"):
        world.create(job="run one")


def test_the_checks_before_a_destroy_run_in_a_worker_and_the_countdown_waits_for_a_fresh_answer(tmp_path):
    w = World(tmp_path, ids=(), executor="dcheck_deferred")
    w.machines["500"] = Machine(500)
    w.machines["500"].tree = {"t1": [("a.bin", 4)]}
    send = w.create(instance=None, rent={"offer": offer()}, after="destroy")
    w.market.rows[500].update(status="running", actual_status="running")
    w.until(lambda: w.view(send["id"])["step"] == "run" and w.view(send["id"])["run"])
    w.finish_runs()
    w.until(lambda: w.view(send["id"])["step"] == "after")
    for _ in range(3):
        w.tick(30)
    m = w.machines["500"]
    assert w.jobs_deferred and not [c for c in m.calls if c[:2] == ("jobs", "--all")]  # tick did not ask
    assert w.view(send["id"])["countdown"] is None
    w.jobs_deferred.pop(0)()
    assert [c for c in m.calls if c[:2] == ("jobs", "--all")]
    w.tick(30)
    assert w.view(send["id"])["countdown"] is not None
    w.tick(200)  # the countdown is over: a fresh check is asked for, and the destroy waits for it
    assert w.jobs_deferred and not w.market.of("destroy") and w.view(send["id"])["countdown"] is not None
    w.jobs_deferred.pop(0)()
    w.tick(30)
    assert w.market.of("destroy") == [("destroy", 500, True)]


def test_every_call_rulz_send_makes_matches_the_real_signature(tmp_path):
    """The fakes of this file bind each call to the real function's signature (a mismatch raises, so every test above
    checks it); this runs the flows that reach every one of them."""
    w = World(tmp_path / "a", ids=(), executor="inline")
    w.machines["500"] = Machine(500)
    w.machines["500"].behind = 1
    w.machines["500"].tree = {"t1": [("a.bin", 4)]}
    unet, tok = w.root / "unet.safetensors", w.root / "tok"
    unet.write_bytes(b"w")
    tok.mkdir()
    (tok / "a.json").write_text("{}")
    folder = w.root / "folder.safetensors"
    folder.write_bytes(b"f")
    cfg = w.config(transformer_file=str(unet), tokenizer_path=str(tok), vae_file=str(folder), model_source="files")
    choices = {str(unet.resolve()): {"how": "bucket", "bucket": "me/mine"},
               str(tok.resolve()): {"how": "bucket", "bucket": "me/mine"}, str(folder.resolve()): {"how": "upload"}}
    send = w.create(instance=None, rent={"offer": offer()}, after="destroy", config=cfg, choices=choices, update=True)
    w.market.rows[500].update(status="running", actual_status="running")
    w.mgr.hold()
    w.mgr.step()
    w.run_to_end(send["id"])
    w2 = World(tmp_path / "b")
    w2.market.rows[1].update(status="exited", actual_status="exited", billing=False)
    w2.create(start=True)
    w2.market.rows[1].update(status="running", actual_status="running")
    w2.run_to_end(w2.mgr.sends()[0]["id"])
    w3 = World(tmp_path / "c", ids=())
    w3.market.rent_error = MarketError("no clear answer", "pending", False, label="rulz-trainer-abc12345")
    with pytest.raises(MarketError):
        w3.create(instance=None, rent={"offer": offer()})
    w3.tick()
    expected = {"files.target_for", "files.upload_file", "files.upload_to_bucket", "files.fetch_from_bucket",
                "files.fetch_tree_from_bucket", "market.rent", "market.start", "market.stop", "market.destroy",
                "market.settle", "market.rentals", "market.intents", "market.instances", "remote.run",
                "remote.agent", "remote.agent_json", "fleet.send"}
    assert expected <= CHECKED, expected - CHECKED


def test_a_start_is_refused_when_the_price_rose_above_the_one_shown(world):
    world.market.rows[1].update(status="exited", actual_status="exited", billing=False, dph_total=0.8)
    with pytest.raises(SendError, match=r"price rose from \$0.700 to \$0.800"):
        world.create(start=True, shown=0.7)
    assert not world.market.of("start") and world.mgr.sends() == []
    world.create(start=True, shown=0.8 / 1.004)  # within 0.5%: the shown price holds
    assert world.market.of("start") == [("start", 1)]


def test_two_dialogs_creating_the_same_job_at_once_cannot_both_pass(world):
    real_read, inner = world.fleet.read, []

    def racing_read():
        if not inner:  # the other dialog gets in between the check of the queue and the save
            inner.append(None)
            inner[0] = world.mgr.create("t1", world.config(), world.root, instance=1)
        return real_read()

    world.fleet.read = racing_read
    with pytest.raises(SendError, match="already being sent"):
        world.mgr.create("T1", world.config(), world.root, instance=1)
    assert len(world.mgr.sends()) == 1


# ---------------------------------------------------------------- join review


def test_upload_progress_makes_a_handful_of_writes_of_sends_json_not_one_per_tar_write(world, tmp_path):
    import io
    import tarfile

    send = world.create()
    writes, calls = [], []
    real = world.mgr._write_file
    world.mgr._write_file = lambda data: (writes.append(1), real(data))[1]
    big = tmp_path / "big.bin"
    big.write_bytes(b"x" * 20_000_000)
    done = [0]

    def progress(count, _files):
        calls.append(count)
        done[0] += count
        world.mgr._progress(send["id"], "big: uploading", min(0.99, done[0] / 20_000_000))

    writer = rulz_remote.CountingWriter(io.BytesIO(), progress)
    with tarfile.open(fileobj=writer, mode="w|") as tar:
        tar.add(big, arcname="big.bin")
    writer.finish()
    world.mgr._progress(send["id"], "big: uploading", 1.0)
    assert len(calls) <= 4 and len(writes) <= 6, (len(calls), len(writes))
    assert sum(calls) >= 20_000_000


def test_a_counting_writer_adds_up_the_bytes_between_its_calls():
    import io
    t = [0.0]
    got = []
    w = rulz_remote.CountingWriter(io.BytesIO(), lambda b, f: got.append(b), every=2.0, clock=lambda: t[0])
    for _ in range(10):
        w.write(b"abc")
    assert got == []
    t[0] = 2.5
    w.write(b"abc")
    assert got == [33]
    w.write(b"zz")
    w.finish()
    assert got == [33, 2]


@pytest.mark.parametrize("when", ["before", "after"])
def test_a_cancel_between_the_decision_to_queue_and_its_save_leaves_no_entry_in_the_queue(world, when):
    send = world.create()
    world.until(lambda: world.view(send["id"])["step"] == "queue")
    real = world.fleet.enqueue

    def racing(*a, **k):
        if when == "before":
            world.mgr.cancel(send["id"])
        got = real(*a, **k)
        if when == "after":
            world.mgr.cancel(send["id"])
        return got

    world.fleet.enqueue = racing
    world.tick()
    for _ in range(3):
        world.tick()
    st = world.fleet.read()["queue"]
    assert world.view(send["id"])["state"] == "cancelled"
    assert not [e for e in st["waiting"] + st["active"] if e["id"] == send["id"]]
    assert not [c for c in world.machines["1"].calls if c[0] == "start" and "--dry-run" not in c]


def stale_stop(world, **kw):
    stop = {"instance": 1, "requested": world.clock.t - 1000, "next": 0, "alerted": False, "after_run": False,
            "sends": [], "sent_at": world.clock.t - 1000, "failed": False}
    stop.update(kw)
    with world.mgr._edit() as data:
        data["stops"].append(stop)


def test_a_start_by_create_clears_a_stop_asked_for_earlier(world):
    stale_stop(world)
    world.market.rows[1].update(status="exited", actual_status="exited", billing=False)
    world.create(start=True)
    assert world.mgr._read_stops() == []


def test_the_clear_stop_the_ui_calls_after_a_start_drops_both_lists(world):
    stale_stop(world)
    with world.mgr._edit() as data:
        data["pending_stops"].append({"instance": 1, "send": "x", "since": 0, "next": 0, "alerted": False, "created": 0})
    world.mgr.clear_stop(1)
    assert world.mgr._read_stops() == [] and world.mgr._read()["pending_stops"] == []


@pytest.mark.parametrize("how", ["newer_send", "started_later", "failed_retry", "run_going"])
def test_a_stale_stop_does_not_stop_an_instance_started_again(world, how):
    world.market.stop_works = False
    stale_stop(world, failed=how == "failed_retry")
    if how in ("newer_send", "failed_retry"):
        world.create()
    elif how == "started_later":
        world.market.rows[1]["started"] = world.clock.t + 100
    elif how == "run_going":
        world.machines["1"].run("x")
        world.fleet.add_instance(1)
        world.fleet.tick()
    for _ in range(3):
        world.tick(130)
    assert not world.market.of("stop")
    assert world.mgr._read_stops() == []


def test_names_with_a_plus_and_other_odd_characters_make_a_place_every_side_accepts(tmp_path):
    import rulz_agent
    import rulz_plan
    for name in ["model+v2.safetensors", "a b+c.safetensors", "+x", "..hidden", "été (1).safetensors",
                 "x" * 300 + ".bin", "a%b#c@d.pt"]:
        dest = rulz_files.sent_dest("0123456789ab", name)
        assert rulz_plan.remote_ok(dest), dest
        assert rulz_files.check_dest(dest) == dest
        assert rulz_agent.sent_parts(dest)


def test_a_model_file_with_a_plus_in_its_name_goes_up_and_is_named_in_the_config(world):
    model = world.root / "model+v2.safetensors"
    model.write_bytes(b"w" * 10)
    send = world.create(config=world.config(transformer_file=str(model)),
                        choices={str(model.resolve()): {"how": "upload"}})
    done = world.run_to_end(send["id"])
    assert done["state"] == "done"
    assert world.machines["1"].configs["t1"]["transformer_file"].endswith("/model+v2.safetensors")


def test_a_place_the_instance_would_refuse_is_refused_before_any_upload(world):
    model = world.root / "unet.safetensors"
    model.write_bytes(b"w" * 10)
    real = world.files.target_for
    world.files.target_for = lambda path, **kw: dict(real(path, **kw), dest="models/sent/zz/unet bad.safetensors")
    send = world.create(config=world.config(transformer_file=str(model)),
                        choices={str(model.resolve()): {"how": "upload"}})
    done = world.run_to_end(send["id"])
    assert done["state"] == "failed" and "cannot be sent" in done["error"] and world.files.calls == []


def test_keep_running_clicked_after_the_decision_still_wins_at_the_action(world):
    send = world.create(after="stop")
    finish_run_and_reach_after(world, send["id"])
    world.tick(20)
    assert world.view(send["id"])["countdown"] is not None
    real, seen = world.mgr._after_facts, []

    def clicked(*a, **k):
        got = real(*a, **k)
        seen.append(1)
        world.mgr.keep(send["id"])  # the click lands right after the conditions were looked at
        return got

    world.mgr._after_facts = clicked
    world.tick(300)
    assert seen and not world.market.of("stop") and not world.market.of("destroy")
    world.mgr._after_facts = real
    world.tick(60)
    assert world.view(send["id"])["state"] == "done" and "left running" in world.view(send["id"])["result"]


def test_a_check_before_a_destroy_that_raises_anything_falls_back_to_a_stop(tmp_path):
    w, send = rented_world(tmp_path)
    w.machines["500"].tree = {"t1": [("a.bin", 4)]}
    w.fleet.pull_now = lambda key, kinds: (_ for _ in ()).throw(RuntimeError("odd"))
    done = w.run_to_end(send["id"])
    assert done["state"] == "done" and w.market.of("stop") == [("stop", 500)] and not w.market.of("destroy")


def test_a_check_before_a_destroy_that_never_answers_counts_as_failed_after_ten_minutes(tmp_path):
    w = World(tmp_path, ids=(), executor="dcheck_deferred")
    w.machines["500"] = Machine(500)
    send = w.create(instance=None, rent={"offer": offer()}, after="destroy")
    w.market.rows[500].update(status="running", actual_status="running")
    w.until(lambda: w.view(send["id"])["step"] == "run" and w.view(send["id"])["run"])
    w.finish_runs()
    w.until(lambda: w.view(send["id"])["step"] == "after")
    for _ in range(60):
        w.tick(300)
        w.jobs_deferred.clear()  # the worker never runs
        if w.market.of("stop"):
            break
    assert w.market.of("stop") == [("stop", 500)] and not w.market.of("destroy")


def test_started_is_in_the_view_and_a_started_instance_left_billing_is_one_alert(world):
    world.market.rows[1].update(status="exited", actual_status="exited", billing=False)
    send = world.create(start=True)
    assert send["started"] is True
    world.mgr._mut(send["id"], lambda x: x.update(state="failed", error="x", after="stop"))
    world.market.rows[1].update(status="running", actual_status="running", billing=True)
    assert world.mgr.hold()
    world.clock.t += 30  # the list is read again
    world.mgr.step()
    world.mgr.step()
    assert len(events(world, "send_alert")) == 1 and "started by a send" in events(world, "send_alert")[0]["error"]
    assert any("still bills" in w for w in world.view(send["id"])["warnings"])
    world.mgr.release()


def test_a_cancel_of_a_started_instance_can_stop_it(world):
    world.market.rows[1].update(status="exited", actual_status="exited", billing=False)
    send = world.create(start=True)
    world.market.rows[1].update(status="running", actual_status="running", billing=True)
    world.mgr.cancel(send["id"], instance_action="stop")
    assert world.market.of("stop") == [("stop", 1)]


def test_retry_after_the_entry_was_taken_out_of_the_queue_queues_it_once_more(world):
    world.fleet.queue_edit("pause")
    send = world.create()
    world.until(lambda: world.view(send["id"])["step"] == "queue")
    world.tick()
    world.fleet.queue_edit("remove", send["id"])
    world.tick()
    world.tick()
    assert world.view(send["id"])["state"] == "failed"
    world.fleet.queue_edit("resume")
    world.mgr.retry(send["id"])
    done = world.run_to_end(send["id"])
    assert done["state"] == "done"
    assert len([c for c in world.machines["1"].calls if c[0] == "start" and "--dry-run" not in c]) == 1


def test_retry_does_not_queue_a_run_that_is_still_queued(world):
    world.fleet.queue_edit("pause")
    send = world.create()
    world.until(lambda: world.view(send["id"])["step"] == "queue")
    world.tick()
    world.tick()
    world.mgr._fail(send["id"], "something else went wrong", stop=False, step="queue")
    world.mgr.retry(send["id"])
    world.tick()
    world.tick()
    queue = world.fleet.read()["queue"]
    assert len([e for e in queue["waiting"] if e["id"] == send["id"]]) == 1


def test_retry_on_an_instance_the_failure_stopped_says_to_start_it_and_never_dead_ends(tmp_path):
    w = World(tmp_path, ids=())
    w.machines["500"] = Machine(500)
    w.machines["500"].refuse = "boom"
    send = w.create(instance=None, rent={"offer": offer()}, after="stop")
    w.market.rows[500].update(status="running", actual_status="running")
    done = w.run_to_end(send["id"])
    assert done["state"] == "failed" and ("stop", 500) in w.market.calls
    with pytest.raises(SendError, match="Start it from the Rent GPUs tab"):
        w.mgr.retry(send["id"])
    assert w.view(send["id"])["state"] == "failed" and not w.market.of("start")


def test_sends_json_is_fsynced_and_the_last_good_copy_is_kept(world, monkeypatch):
    synced = []
    real = rulz_send_os.fsync
    monkeypatch.setattr(rulz_send_os, "fsync", lambda fd: (synced.append(1), real(fd))[1])
    send = world.create()
    world.tick()
    assert synced
    bak = world.tmp / "sends.json.bak"
    assert bak.exists() and json.loads(bak.read_text(encoding="utf-8"))["sends"][0]["id"] == send["id"]


@pytest.mark.parametrize("damage", ["garbage", "empty"])
def test_an_unreadable_sends_json_is_replaced_by_the_last_good_copy_with_an_alert(world, damage):
    send = world.create()
    world.tick()
    path = world.tmp / "sends.json"
    path.write_text("{ not json" if damage == "garbage" else "", encoding="utf-8")
    assert [x["id"] for x in world.mgr.sends()] == [send["id"]]
    assert list(world.tmp.glob("sends.json.bad-*")) and json.loads(path.read_text(encoding="utf-8"))["sends"]
    assert events(world, "send_alert") and "last good copy" in events(world, "send_alert")[0]["error"]
    world.mgr.errors.clear()


def test_an_instance_missing_from_one_list_is_not_gone_until_a_later_list_agrees(world):
    send = world.create()
    world.until(lambda: world.view(send["id"])["step"] == "files")
    row = world.market.rows.pop(1)  # an odd answer reads as an empty list
    world.tick()
    assert world.view(send["id"])["state"] != "failed"
    world.market.rows[1] = row
    for _ in range(3):
        world.tick()
    assert world.view(send["id"])["state"] != "failed" and not world.mgr._gone_seen  # a later list showed it


def test_a_stop_answered_gone_while_the_list_still_shows_the_instance_is_not_taken_for_stopped(world):
    send = world.create(after="stop")
    world.mgr._mut(send["id"], lambda x: x.update(rented=True))
    world.until(lambda: world.view(send["id"])["step"] in ("reach", "setup", "files"))
    world.market.stop_errors = [MarketError("not on your account", "gone", True)]
    world.mgr.cancel(send["id"], instance_action="stop")
    assert events(world, "send_alert") and world.mgr._read_stops()[0]["failed"] is True


def test_the_conftest_guard_records_a_refusal_even_when_code_catches_it(request):
    refused = request.getfixturevalue("_no_real_vast_or_hugging_face")
    try:
        rulz_market.Api(key="k").get("/x")
    except Exception:  # noqa: BLE001  (what code that catches everything does)
        pass
    assert refused
    refused.clear()  # so that this test itself does not fail in the fixture's teardown
