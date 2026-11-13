"""An instance's own controls from the fleet: reboot it (vast.ai stops and starts its container), update its
RulzTrainer to the newest and restart its WebUI, and open its WebUI signed in. The fleet side runs over fake
machines (tests/test_vast_fleet.py's World); the instance side (rulz_agent.py update and webui) runs here on real
git checkouts of a local "origin", with install.sh and supervisorctl stood in for."""
import json
import socket
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "vast"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import rulz_agent as agent  # noqa: E402
import rulz_fleet as fleet_module  # noqa: E402
from rulz_fleet import REBOOT_GRACE, Refused  # noqa: E402
from test_vast_fleet import Machine, World, listing  # noqa: E402


# ---------------------------------------------------------------- the fleet's side


class ControlMachine(Machine):
    """A fake instance that also answers update and webui, and knows its install and when it booted."""

    def __init__(self, iid):
        super().__init__(iid)
        self.booted = 5000.0
        self.trainer = {"version": "3.16.0", "commit": "aaaaaaaaaa", "branch": "app/rulz-trainer"}
        self.newest = {"version": "3.17.0", "commit": "bbbbbbbbbb"}
        self.update = None  # what the snapshot reports of the last update
        self.update_answer = None  # (code, reply, stderr) for update, else a done update
        self.webui = {"port": 7870, "path": "/", "scheme": "http", "ip": "1.2.3.4", "host_port": 41000,
                      "token": "t0k/en", "webui_running": True}

    def snapshot(self):
        snap = super().snapshot()
        snap.update(trainer=dict(self.trainer), booted=self.booted)
        if self.update:
            snap["update"] = dict(self.update)
        return snap

    def answer(self, args, config):
        if args[0] == "update" and not self.down:
            self.calls.append(tuple(args))
            if "--check" in args:
                return 0, {"checkout": True, "branch": "app/rulz-trainer", "current": dict(self.trainer),
                           "latest": dict(self.newest), "behind": 3, "ahead": 0, "up_to_date": False, "changed": [],
                           "changed_count": 0, "installs": False, "job_running": self.running is not None,
                           "job": self.running}, b""
            if self.update_answer is not None:
                return self.update_answer
            self.update = {"state": "done", "started": 7000.0, "ended": 7010.0, "from": dict(self.trainer),
                           "to": dict(self.newest)}
            self.trainer = dict(self.newest, branch="app/rulz-trainer")
            return 0, dict(self.update, restarted=True, installed=False), b""
        if args[0] == "webui" and not self.down:
            self.calls.append(tuple(args))
            return 0, dict(self.webui), b""
        return super().answer(args, config)


class Vast:
    """The vastai CLI: what it was asked, and what it says."""

    def __init__(self):
        self.calls, self.says = [], "Rebooting instance {}."

    def __call__(self, *args):
        self.calls.append(args)
        return self.says.format(args[-1]) + "\n"


@pytest.fixture
def world(tmp_path):
    w = World(tmp_path, 1, 2)
    w.machines = {i: ControlMachine(int(i)) for i in w.machines}
    w.listed = lambda: listing(w.machines.values())
    w.vast = Vast()
    w.fleet.vast = w.vast
    w.tick()
    return w


def state_of(world, iid):
    return next(i for i in world.fleet.overview()["instances"] if i["id"] == iid)


def test_a_reboot_is_waited_for_and_not_taken_for_the_instance_going_down_or_away(world):
    world.fleet.enqueue("next", {"steps": 10}, instance=1)
    world.fleet.queue_edit("pause")
    out = world.fleet.reboot(1)
    assert out == {"instance": 1, "state": "rebooting", "killed": None, "said": "Rebooting instance 1."}
    assert world.vast.calls == [("reboot", "instance", "1")]
    assert state_of(world, 1)["state"] == "rebooting"
    world.fleet.queue_edit("resume")
    world.tick()  # the old container still answers: nothing is sent to it meanwhile
    assert world.starts(1) == [] and state_of(world, 1)["state"] == "rebooting"
    world.machines["1"].down = True
    world.listed = lambda: listing(world.machines.values(), **{"1": "loading"})
    world.fleet.discover()
    world.tick()
    assert not world.fleet.read()["instances"]["1"].get("gone")
    assert world.kinds("instance_down", "instance_gone") == []
    with pytest.raises(Refused, match="already rebooting"):
        world.fleet.reboot(1)
    machine = world.machines["1"]
    machine.down, machine.booted = False, 9000.0
    world.listed = lambda: listing(world.machines.values())
    world.tick()
    assert [e["kind"] for e in world.kinds("instance_rebooting", "instance_rebooted", "instance_up")] == \
        ["instance_rebooting", "instance_rebooted"]
    world.tick()
    assert world.starts(1) == ["next"]


def test_a_new_container_ends_the_reboot_even_when_no_look_missed_it(world):
    world.fleet.reboot(2)
    world.machines["2"].booted = 6000.0
    world.tick()
    assert state_of(world, 2)["state"] == "idle" and world.kinds("instance_rebooted")[0]["instance"] == 2


def test_a_reboot_that_never_comes_back_counts_as_down_after_its_grace(world):
    world.fleet.reboot(1)
    world.machines["1"].down = True
    world.tick()
    assert world.kinds("instance_down") == []
    world.tick(REBOOT_GRACE + 1)
    assert world.kinds("instance_down")[0]["instance"] == 1 and state_of(world, 1)["state"] == "unreachable"
    world.listed = lambda: listing(world.machines.values(), **{"1": "exited"})
    world.fleet.discover()
    assert world.fleet.read()["instances"]["1"]["gone"]


def test_a_reboot_kills_a_run_only_when_forced_and_what_vast_refuses_is_said(world):
    world.machines["1"].run("a", step=30)
    world.tick()
    with pytest.raises(Refused, match="a is running on instance 1: a reboot kills it"):
        world.fleet.reboot(1)
    assert world.vast.calls == []
    assert world.fleet.reboot(1, force=True)["killed"] == "a"
    assert world.kinds("instance_rebooting")[0]["job"] == "a"
    world.vast.says = "Instance {} is not running"  # the CLI ends with 0 and says why on stdout
    with pytest.raises(Refused, match="did not reboot instance 2: Instance 2 is not running"):
        world.fleet.reboot(2)
    assert "rebooting" not in world.fleet.read()["instances"]["2"]
    world.listed = lambda: listing([world.machines["1"]])
    world.fleet.discover()
    with pytest.raises(Refused, match="destroyed"):
        world.fleet.reboot(2)


def test_a_check_says_what_an_update_changes_and_the_card_shows_the_newer_version(world):
    plan = world.fleet.check_update(1)
    assert plan["latest"]["version"] == "3.17.0" and plan["behind"] == 3
    assert world.machines["1"].calls[-1] == ("update", "--check")
    line = state_of(world, 1)
    assert line["trainer"]["version"] == "3.16.0" and line["newer"] == {"version": "3.17.0", "behind": 3}
    assert "newer" not in state_of(world, 2)


def test_an_update_restarts_the_webui_on_the_newest_and_the_queue_waits_for_it(world):
    world.fleet.check_update(1)
    reply = world.fleet.update_trainer(1)
    assert reply["state"] == "done" and reply["to"]["version"] == "3.17.0" and reply["instance"] == 1
    assert world.machines["1"].calls[-1] == ("update",)
    updated = world.kinds("instance_updated")
    assert len(updated) == 1 and updated[0]["was"] == "3.16.0" and updated[0]["version"] == "3.17.0"
    world.tick()
    assert len(world.kinds("instance_updated")) == 1  # the look that sees the same update says nothing again
    line = state_of(world, 1)
    assert line["trainer"]["version"] == "3.17.0" and "newer" not in line and line["state"] == "idle"
    world.machines["2"].update = {"state": "running", "started": 8000.0}  # one started elsewhere
    world.fleet.enqueue("r", {"steps": 10}, instance=2)
    world.tick()
    assert state_of(world, 2)["state"] == "updating" and world.starts(2) == []
    with pytest.raises(Refused, match="already updating"):
        world.fleet.update_trainer(2)
    world.machines["2"].update = {"state": "done", "started": 8000.0, "to": {"version": "3.17.0"}}
    world.tick()
    world.tick()
    assert world.starts(2) == ["r"] and world.kinds("instance_updated")[-1]["instance"] == 2


def test_a_refused_a_failed_and_a_cut_off_update(world):
    machine = world.machines["1"]
    machine.update_answer = (1, b"", b"[refused] a is running: an update changes the code under it.")
    with pytest.raises(Refused, match="^a is running"):
        world.fleet.update_trainer(1, force=False)
    assert world.kinds("update_failed", "instance_updated") == [] and state_of(world, 1)["state"] == "idle"
    machine.update_answer = (1, {"state": "failed", "started": 7100.0, "error": "requirements failed (exit 1)"}, b"")
    with pytest.raises(Refused, match="requirements failed"):
        world.fleet.update_trainer(1, discard=True, reinstall=True)
    assert machine.calls[-1] == ("update", "--discard", "--reinstall")
    assert world.kinds("update_failed")[0]["error"] == "requirements failed (exit 1)"
    machine.update_answer = (255, b"", b"Connection reset")
    with pytest.raises(Refused, match="dropped out of reach during the update"):
        world.fleet.update_trainer(1)
    assert world.fleet.remotes["1"].retry is False  # never sent twice by a reconnect
    world.fleet.enqueue("meanwhile", {"steps": 10}, instance=1)
    world.tick()  # nothing says yet that it began: the queue still keeps off it
    assert state_of(world, 1)["state"] == "updating" and world.starts(1) == []
    machine.update = {"state": "done", "started": 7200.0, "from": {"version": "3.16.0"}, "to": {"version": "3.17.0"}}
    world.tick()  # it went on there: the next look says how it ended
    assert world.kinds("instance_updated")[-1]["version"] == "3.17.0"
    world.tick()
    assert world.starts(1) == ["meanwhile"]


def test_a_cut_off_update_that_never_began_frees_the_instance_after_a_while(world):
    machine = world.machines["2"]
    machine.update_answer = (255, b"", b"Connection reset")
    with pytest.raises(Refused, match="dropped out of reach"):
        world.fleet.update_trainer(2)
    world.tick()
    assert state_of(world, 2)["state"] == "updating"
    world.tick(fleet_module.UPDATE_SETTLE + 1)
    assert state_of(world, 2)["state"] == "idle" and world.kinds("update_failed", "instance_updated") == []


def test_an_update_cut_short_here_frees_the_instance_once_a_look_settles_it(world):
    machine = world.machines["1"]

    def interrupted(args, config):
        raise KeyboardInterrupt  # Ctrl+C on the command line while the instance installs

    machine.answer = interrupted
    with pytest.raises(KeyboardInterrupt):
        world.fleet.update_trainer(1)
    assert world.fleet.read()["instances"]["1"]["updating"]["unsure"]
    machine.answer = lambda args, config: ControlMachine.answer(machine, args, config)
    machine.update = {"state": "done", "started": 7300.0, "to": {"version": "3.17.0"}}
    world.tick()
    assert state_of(world, 1)["state"] == "idle" and world.kinds("instance_updated")[-1]["instance"] == 1


def test_a_stale_route_is_not_taken_for_a_dropped_update(world):
    world.machines["1"].probe_fails = True  # ssh does not get in at all: the update never ran
    with pytest.raises(Refused) as refused:
        world.fleet.update_trainer(1)
    assert "connect refused" in str(refused.value) and "dropped" not in str(refused.value)
    assert "updating" not in world.fleet.read()["instances"]["1"]
    assert not [c for c in world.machines["1"].calls if c[0] == "update"]


def test_a_failed_look_alone_does_not_end_a_reboot_while_the_old_container_answers(world):
    world.fleet.reboot(1)
    world.machines["1"].down = True
    world.tick()
    world.machines["1"].down = False  # the old container again (vast.ai has not stopped it yet)
    world.tick()
    assert state_of(world, 1)["state"] == "rebooting" and world.kinds("instance_rebooted") == []


def test_a_reboot_without_a_recent_look_goes_by_the_instance_dropping_out(world):
    world.clock.t += 600  # the last look is old: the container may have restarted since
    world.fleet.reboot(2)
    assert world.fleet.read()["instances"]["2"]["rebooting"]["booted"] is None
    world.machines["2"].down = True
    world.tick()
    world.machines["2"].down = False
    world.tick()
    assert world.kinds("instance_rebooted")[0]["instance"] == 2


def test_killing_a_run_is_not_forcing_past_the_other_guards(world):
    world.machines["1"].run("a")
    world.tick()
    world.fleet.set_instance(1, dispatching={"entry": "x", "job": "j", "since": world.clock.t})
    with pytest.raises(Refused, match="being sent"):
        world.fleet.reboot(1, kill=True)
    world.fleet.set_instance(1, dispatching=None)
    assert world.fleet.reboot(1, kill=True)["killed"] == "a"


def test_the_webui_link_is_a_plain_http_address_or_nothing():
    url = fleet_module.portal_url
    assert url("https", "::1", "443", "/", "a b") == "https://[::1]:443/?token=a%20b"
    assert url("http", "1.2.3.4", 41000, "/%0Ajs", None) == "http://1.2.3.4:41000/"
    for bad in (("javascript", "1.2.3.4", 80), ("http", "evil.example", 80), ("http", "1.2.3.4", 0),
                ("http", "1.2.3.4", "80; x"), ("file", "1.2.3.4", 80)):
        with pytest.raises(Refused, match="Not an address"):
            url(*bad, "/", "t")


def test_a_reboot_is_taken_under_the_lock_and_given_back_when_vast_refuses(world):
    world.fleet.set_instance(1, dispatching={"entry": "x", "job": "j", "since": world.clock.t})
    with pytest.raises(Refused, match="being sent"):
        world.fleet.reboot(1)
    assert world.vast.calls == []
    world.vast.says = "error 500"
    with pytest.raises(Refused, match="did not reboot instance 2: error 500"):
        world.fleet.reboot(2)
    assert "rebooting" not in world.fleet.read()["instances"]["2"]

    def seen_while_asking(*args):  # what the looker would read while vast.ai is being asked
        assert world.fleet._rebooting(world.fleet.read()["instances"]["2"])
        return "Rebooting instance 2."

    world.fleet.vast = seen_while_asking
    assert world.fleet.reboot(2)["state"] == "rebooting"


def test_an_update_that_ended_before_the_fleet_first_looked_makes_no_event(world, tmp_path):
    late = World(tmp_path / "late", 3)
    late.machines = {"3": ControlMachine(3)}
    late.machines["3"].update = {"state": "done", "started": 1.0, "to": {"version": "3.16.0"}}
    late.tick()
    assert late.kinds("instance_updated", "update_failed") == []


def test_an_update_waits_for_a_reboot_and_a_dispatch(world):
    world.fleet.reboot(1)
    with pytest.raises(Refused, match="rebooting"):
        world.fleet.update_trainer(1)
    world.fleet.set_instance(2, dispatching={"entry": "x", "job": "j", "since": world.clock.t})
    with pytest.raises(Refused, match="being sent"):
        world.fleet.update_trainer(2)


def test_the_webui_link_signs_the_browser_in_and_falls_back_to_vasts_listing(world):
    link = world.fleet.webui_link(1)
    assert link == {"instance": 1, "url": "http://1.2.3.4:41000/?token=t0k%2Fen", "signed_in": True, "https": False,
                    "webui_running": True}
    machine = world.machines["2"]
    machine.webui = dict(machine.webui, ip=None, host_port=None, token=None, scheme="https")

    def listed():
        rows = listing(world.machines.values())
        rows[1].update(public_ipaddr="5.6.7.8 ", ports={"7870/tcp": [{"HostIp": "0.0.0.0", "HostPort": "42000"}]})
        return rows

    world.listed = listed
    assert world.fleet.webui_link(2)["url"] == "https://5.6.7.8:42000/"
    world.listed = lambda: listing(world.machines.values())
    with pytest.raises(Refused, match="maps no public port"):
        world.fleet.webui_link(2)
    machine.down = True
    with pytest.raises(Refused, match="did not say where its WebUI is"):
        world.fleet.webui_link(2)


def test_cli_update_reboot_and_webui(world, monkeypatch, capsys):
    monkeypatch.setattr(fleet_module, "Fleet", lambda **kw: world.fleet)

    def call(*argv):
        code = fleet_module.main(list(argv))
        out = capsys.readouterr()
        return code, (json.loads(out.out) if out.out.strip() else None), out.err

    code, plan, _ = call("update", "1", "--check")
    assert code == 0 and plan["behind"] == 3
    code, reply, err = call("update", "1", "--discard")
    assert code == 0 and reply["state"] == "done" and "updating instance 1" in err
    code, link, _ = call("webui", "2")
    assert code == 0 and link["url"].startswith("http://1.2.3.4:41000/")
    world.machines["2"].run("b")
    world.tick()
    code, _, err = call("reboot", "2")
    assert code == 1 and "kills it" in err
    code, out, _ = call("reboot", "2", "--kill")
    assert code == 0 and out["killed"] == "b"
    code, _, err = call("reboot", "2", "--kill")
    assert code == 1 and "already rebooting" in err
    assert call("reboot", "2", "--force")[0] == 0
    assert call("update", "9")[0] == 1  # not in the fleet


# ---------------------------------------------------------------- the instance's side: rulz_agent.py update


def run_git(cwd, *args):
    done = subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.com", "-c", "core.autocrlf=false",
                           *args], cwd=cwd, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


def write_release(folder: Path, version: str, requirements: str = "psutil\n") -> None:
    (folder / "rulztrainer").mkdir(parents=True, exist_ok=True)
    (folder / "rulztrainer" / "__init__.py").write_text(f'__version__ = "{version}"\n', encoding="utf-8")
    (folder / "requirements.txt").write_text(requirements, encoding="utf-8")
    (folder / "install.sh").write_text("echo install\n", encoding="utf-8")
    (folder / "webui.py").write_text(f"# {version}\n", encoding="utf-8")
    (folder / "scripts" / "vast").mkdir(parents=True, exist_ok=True)
    (folder / "scripts" / "vast" / "rulztrainer.sh").write_text(f"# service {version}\n", encoding="utf-8")
    (folder / ".gitignore").write_text("output/\ndatasets/\n", encoding="utf-8")


def publish(work: Path, version: str, **kw) -> None:
    write_release(work, version, **kw)
    run_git(work, "add", "-A")
    run_git(work, "commit", "-q", "-m", version)
    run_git(work, "push", "-q", "origin", "HEAD:refs/heads/app/rulz-trainer")


@pytest.fixture
def origin(tmp_path):
    """A bare "origin" with RulzTrainer 1.0.0 on app/rulz-trainer, and a work tree that publishes to it."""
    bare, work = tmp_path / "origin.git", tmp_path / "work"
    run_git(tmp_path, "init", "-q", "--bare", str(bare))
    work.mkdir()
    run_git(work, "init", "-q")
    run_git(work, "remote", "add", "origin", str(bare))
    publish(work, "1.0.0")
    return bare, work


@pytest.fixture
def stand_ins(tmp_path, monkeypatch):
    """install.sh, supervisorctl and the supervisor folder, each writing down that it ran."""
    done = tmp_path / "ran.txt"
    script = tmp_path / "stand_in.py"
    script.write_text("import sys\nopen(sys.argv[1], 'a').write(' '.join(sys.argv[2:]) + '\\n')\n"
                      "sys.exit(1 if 'fail' in sys.argv[2:] else 0)\n", encoding="utf-8")
    service_dir = tmp_path / "supervisor-scripts"
    service_dir.mkdir()
    monkeypatch.setattr(agent, "INSTALL_COMMAND", [sys.executable, str(script), str(done), "install"])
    monkeypatch.setattr(agent, "supervisorctl", lambda: [sys.executable, str(script), str(done), "ctl"])
    monkeypatch.setattr(agent, "SUPERVISOR_FILES", (("scripts/vast/rulztrainer.sh",
                                                     str(service_dir / "rulztrainer.sh"), 0o755),))
    monkeypatch.setattr(agent, "webui_running", lambda: True)
    monkeypatch.setattr(agent, "container_env", lambda *a, **k: {})
    return done, service_dir / "rulztrainer.sh"


def agent_call(capsys, *argv):
    code = agent.main(list(argv))
    out = capsys.readouterr()
    return code, (json.loads(out.out) if out.out.strip() else None), out.err


def test_update_brings_a_checkout_to_the_newest_installs_what_changed_and_restarts_the_webui(
        origin, stand_ins, tmp_path, monkeypatch, capsys):
    bare, work = origin
    install = tmp_path / "install"
    run_git(tmp_path, "clone", "-q", "--branch", "app/rulz-trainer", str(bare), str(install))
    monkeypatch.chdir(install)
    code, plan, _ = agent_call(capsys, "update", "--check")
    assert code == 0 and plan["up_to_date"] and plan["behind"] == 0 and plan["current"]["version"] == "1.0.0"
    code, reply, _ = agent_call(capsys, "update")
    assert code == 0 and reply["state"] == "current"
    publish(work, "1.1.0", requirements="psutil\npyyaml\n")
    code, plan, _ = agent_call(capsys, "update", "--check")
    assert plan["behind"] == 1 and plan["latest"]["version"] == "1.1.0" and plan["installs"] is True
    assert plan["repo"] == str(bare) and plan["branch"] == "app/rulz-trainer" and plan["changed"] == []
    assert (install / "webui.py").read_text(encoding="utf-8") == "# 1.0.0\n"  # a check changes nothing
    ran, service = stand_ins
    code, reply, _ = agent_call(capsys, "update")
    assert code == 0 and reply["state"] == "done" and reply["to"]["version"] == "1.1.0" and reply["restarted"]
    assert reply["installed"] and reply["from"]["version"] == "1.0.0"
    assert reply["steps"] == ["code", "requirements", "service", "restart"]
    assert (install / "webui.py").read_text(encoding="utf-8") == "# 1.1.0\n"
    assert ran.read_text(encoding="utf-8").splitlines() == ["install", "ctl update", "ctl restart rulztrainer"]
    assert service.read_text(encoding="utf-8") == "# service 1.1.0\n"  # the supervisor file renewed
    assert agent.update_state()["state"] == "done" and Path("output/_update.log").is_file()
    publish(work, "1.2.0", requirements="psutil\npyyaml\n")  # the requirements stay: no install this time
    ran.write_text("", encoding="utf-8")
    code, reply, _ = agent_call(capsys, "update")
    assert reply["steps"] == ["code", "service", "restart"] and not reply["installed"]
    assert reply["to"]["version"] == "1.2.0" and agent.install_info()["branch"] == "app/rulz-trainer"
    service.write_text("# service 1.3.0\n", encoding="utf-8")  # the same file: no supervisor reload
    publish(work, "1.3.0", requirements="psutil\npyyaml\n")
    code, reply, _ = agent_call(capsys, "update")
    assert reply["steps"] == ["code", "restart"]


def test_the_instances_own_queue_is_held_while_the_code_changes(origin, stand_ins, tmp_path, monkeypatch, capsys):
    bare, work = origin
    install = tmp_path / "install"
    run_git(tmp_path, "clone", "-q", "--branch", "app/rulz-trainer", str(bare), str(install))
    monkeypatch.chdir(install)
    publish(work, "1.1.0")
    paused = []

    class Queue:
        def snapshot(self):
            return {"paused": False, "waiting": [{"id": "q1", "job": "next"}], "recent": []}

        def set_paused(self, value):
            paused.append(value)

    monkeypatch.setattr(agent, "queue", Queue)
    code, reply, _ = agent_call(capsys, "update")
    assert code == 0 and paused == [True, False]  # held for the update, then as it was


def test_update_keeps_changed_files_and_a_running_job_unless_told(origin, stand_ins, tmp_path, monkeypatch, capsys):
    bare, work = origin
    install = tmp_path / "install"
    run_git(tmp_path, "clone", "-q", "--branch", "app/rulz-trainer", str(bare), str(install))
    monkeypatch.chdir(install)
    publish(work, "1.1.0")
    (install / "webui.py").write_text("# synced from a laptop\n", encoding="utf-8")
    code, _, err = agent_call(capsys, "update")
    assert code == 1 and "1 file(s) changed here" in err and "webui.py" in err and "--discard" in err
    assert (install / "webui.py").read_text(encoding="utf-8") == "# synced from a laptop\n"
    monkeypatch.setattr(agent, "running_dir", lambda: install / "output" / "busy")
    code, _, err = agent_call(capsys, "update", "--discard")
    assert code == 1 and "busy is running" in err
    code, reply, _ = agent_call(capsys, "update", "--discard", "--force")
    assert code == 0 and (install / "webui.py").read_text(encoding="utf-8") == "# 1.1.0\n"


def test_a_folder_that_is_no_checkout_becomes_one(origin, stand_ins, tmp_path, monkeypatch, capsys):
    bare, _ = origin
    install = tmp_path / "copy"
    write_release(install, "0.9.0")
    (install / "datasets" / "anna").mkdir(parents=True)
    monkeypatch.chdir(install)
    monkeypatch.setattr(agent, "container_env", lambda *a, **k: {"RULZ_REPO": str(bare)})
    code, plan, _ = agent_call(capsys, "update", "--check")
    assert code == 0 and plan["checkout"] is False and plan["latest"]["version"] == "1.0.0"
    assert plan["current"] == {"version": "0.9.0"}
    code, reply, _ = agent_call(capsys, "update")
    assert code == 0 and reply["to"]["version"] == "1.0.0" and reply["to"]["branch"] == "app/rulz-trainer"
    assert (install / "datasets" / "anna").is_dir()


def test_a_failed_step_is_recorded_and_said(origin, stand_ins, tmp_path, monkeypatch, capsys):
    bare, work = origin
    install = tmp_path / "install"
    run_git(tmp_path, "clone", "-q", "--branch", "app/rulz-trainer", str(bare), str(install))
    monkeypatch.chdir(install)
    publish(work, "1.1.0", requirements="psutil\nbroken\n")
    ran, _ = stand_ins
    monkeypatch.setattr(agent, "INSTALL_COMMAND", agent.INSTALL_COMMAND + ["fail"])
    code, reply, _ = agent_call(capsys, "update")
    assert code == 1 and reply["state"] == "failed" and "requirements failed (exit 1)" in reply["error"]
    assert agent.update_state()["state"] == "failed"
    assert "ctl" not in ran.read_text(encoding="utf-8")  # the WebUI keeps running the code it has loaded
    # the code is already the newest, but the update did not finish: the next one does the rest
    monkeypatch.setattr(agent, "INSTALL_COMMAND", agent.INSTALL_COMMAND[:-1])
    code, plan, _ = agent_call(capsys, "update", "--check")
    assert plan["up_to_date"] and plan["unfinished"] and plan["installs"]
    code, reply, _ = agent_call(capsys, "update")
    assert code == 0 and reply["state"] == "done" and reply["steps"][:2] == ["code", "requirements"]
    assert reply["restarted"]
    code, reply, _ = agent_call(capsys, "update")
    assert reply["state"] == "current"


def test_files_the_newest_release_brings_over_untracked_ones_count_as_changes(
        origin, stand_ins, tmp_path, monkeypatch, capsys):
    bare, work = origin
    install = tmp_path / "install"
    run_git(tmp_path, "clone", "-q", "--branch", "app/rulz-trainer", str(bare), str(install))
    monkeypatch.chdir(install)
    (work / "rulztrainer" / "new_module.py").write_text("# released\n", encoding="utf-8")
    publish(work, "1.1.0")
    (install / "rulztrainer" / "new_module.py").write_text("# synced before the release\n", encoding="utf-8")
    code, plan, _ = agent_call(capsys, "update", "--check")
    assert plan["changed"] == ["rulztrainer/new_module.py"] and plan["changed_count"] == 1
    assert agent_call(capsys, "update")[0] == 1
    code, reply, _ = agent_call(capsys, "update", "--discard")
    assert code == 0 and (install / "rulztrainer" / "new_module.py").read_text(encoding="utf-8") == "# released\n"


def test_a_second_update_while_one_holds_the_lock_is_refused(origin, stand_ins, tmp_path, monkeypatch, capsys):
    bare, _ = origin
    install = tmp_path / "install"
    run_git(tmp_path, "clone", "-q", "--branch", "app/rulz-trainer", str(bare), str(install))
    monkeypatch.chdir(install)
    holder = subprocess.Popen([sys.executable, "-c", (
        "import sys, time; sys.path.insert(0, sys.argv[1]); import rulz_agent as a; "
        "f = a.hold_lock(a.UPDATE_LOCK); print('held' if f else 'no', flush=True); time.sleep(30)"),
        str(REPO / "scripts" / "vast")], cwd=install, stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "held"
        code, _, err = agent_call(capsys, "update")
        assert code == 1 and "already going" in err
    finally:
        holder.kill()
        holder.wait()
    assert agent_call(capsys, "update", "--check")[0] == 0


def test_an_update_that_died_reads_as_interrupted_and_one_going_refuses_another(install_dir, capsys):
    agent.write_json(agent.UPDATE_STATE, {"state": "running", "pid": 999999999, "started": 1.0})
    assert agent.update_state()["state"] == "interrupted"
    me = agent.live_process(__import__("os").getpid())
    agent.write_json(agent.UPDATE_STATE, {"state": "running", "pid": me.pid, "pid_created": me.create_time(),
                                          "started": 1.0})
    code, _, err = agent_call(capsys, "update", "--check")
    assert code == 1 and "already going" in err


@pytest.fixture
def install_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "output").mkdir()
    monkeypatch.setattr(agent, "webui_running", lambda: True)
    return tmp_path


# ---------------------------------------------------------------- the instance's side: where its WebUI is


def test_the_containers_variables_come_from_its_files_and_this_process(tmp_path):
    env_file, first = tmp_path / "environment", tmp_path / "environ"
    env_file.write_text('OPEN_BUTTON_TOKEN="abc"\nexport PUBLIC_IPADDR=1.2.3.4\n# a note\nVAST_TCP_PORT_7870=41000\n',
                        encoding="utf-8")
    first.write_bytes(b"PUBLIC_IPADDR=5.6.7.8\0ENABLE_HTTPS=true\0")
    env = agent.container_env(env_file, first, own={"ENABLE_HTTPS": "false"})
    assert env == {"OPEN_BUTTON_TOKEN": "abc", "PUBLIC_IPADDR": "5.6.7.8", "VAST_TCP_PORT_7870": "41000",
                   "ENABLE_HTTPS": "false"}
    assert agent.container_env(tmp_path / "none", tmp_path / "none", own={}) == {}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_webui_says_the_portals_port_scheme_address_and_token(install_dir, monkeypatch, capsys):
    port = free_port()  # nothing listens: http or https comes from ENABLE_HTTPS
    portal = install_dir / "portal.yaml"
    portal.write_text(f"applications:\n  Instance Portal:\n    external_port: 1111\n    name: Instance Portal\n"
                      f"  RulzTrainer:\n    hostname: localhost\n    external_port: {port}\n    internal_port: 17870\n"
                      f"    open_path: /\n    name: RulzTrainer\n", encoding="utf-8")
    monkeypatch.setattr(agent, "PORTAL_FILE", portal)
    monkeypatch.setattr(agent, "container_env", lambda *a, **k: {
        "OPEN_BUTTON_TOKEN": "tok", "PUBLIC_IPADDR": "1.2.3.4", f"VAST_TCP_PORT_{port}": "41000", "ENABLE_HTTPS": "true"})
    code, info, _ = agent_call(capsys, "webui")
    assert code == 0 and info == {"port": port, "path": "/", "scheme": "https", "ip": "1.2.3.4", "host_port": 41000,
                                  "token": "tok", "webui_running": True}
    assert agent.portal_entry("", "localhost:1111:11111:/:Instance Portal|localhost:7871:17870:/:RulzTrainer") == \
        (7871, "/")
    assert agent.portal_entry("not: [yaml", "") == (agent.PORTAL_PORT, "/")


def test_a_port_that_answers_without_tls_is_http(monkeypatch):
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]
        import threading

        def answer():
            conn, _ = server.accept()
            conn.recv(1024)
            conn.sendall(b"HTTP/1.1 400 Bad Request\r\n\r\n")
            conn.close()

        threading.Thread(target=answer, daemon=True).start()
        assert agent.portal_scheme(port, {"ENABLE_HTTPS": "true"}) == "http"
