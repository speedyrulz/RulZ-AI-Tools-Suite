"""The watchdog of docs/vast-send-contract.md section 6 and amendments 1, 2 and 9: its pure decision function, one look
(`step`) with a fake clock and a fake stop request, what it observes on a fake install, the stop request itself, and the
agent's `watchdog` commands. No instance, no vast.ai: requests go through fake openers."""
import contextlib
import io
import json
import os
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "vast"))

import rulz_agent as agent  # noqa: E402

watch = agent.embedded(agent.WATCH_PROGRAM, "rulz_watch_test")

T0 = 1_000_000.0
BOOT = 500_000.0
MIN = 60


def obs(**kw):
    base = {"running": False, "waiting": 0, "paused": False, "tests": False, "arriving": False, "alive": None,
            "booted": BOOT}
    base.update(kw)
    return base


def state(**kw):
    base = {"armed": True, "minutes": 10, "idle_since": None, "seen_running": False, "booted": BOOT}
    base.update(kw)
    return base


def walk(st, steps):
    """Feed (seconds since T0, observation) pairs through decide; returns the last (state, action) and every action."""
    actions = []
    for at, o in steps:
        st, action = watch.decide(T0 + at, st, o)
        actions.append(action)
    return st, actions


# ---------------------------------------------------------------- decide


def test_it_stops_after_n_idle_minutes_once_a_run_was_seen():
    st, acts = walk(state(), [(0, obs(running=True)), (10, obs()), (10 * MIN + 9, obs()), (11 * MIN, obs())])
    assert acts == [None, None, None, "stop"] and st["seen_running"] is True and st["idle_since"] == T0 + 10


def test_a_run_a_waiting_entry_a_test_or_arriving_files_reset_the_idle_clock():
    st = state(seen_running=True)
    for busy in (obs(running=True), obs(waiting=1), obs(tests=True), obs(arriving=True)):
        st, acts = walk(st, [(0, obs()), (9 * MIN, busy), (11 * MIN, obs())])
        assert acts == [None, None, None] and st["idle_since"] == T0 + 11 * MIN  # counting from the last busy look


def test_nothing_ever_started_it_fires_too_after_n_minutes():
    """Amendment 1: not seen running does not mean it waits for ever."""
    st, acts = walk(state(), [(0, obs()), (9 * MIN, obs()), (10 * MIN + 1, obs())])
    assert acts == [None, None, "stop"]


def test_a_manager_that_looked_in_lately_holds_the_early_rule_back_but_not_the_late_one():
    alive = lambda at: obs(alive=T0 + at)  # noqa: E731
    st, acts = walk(state(), [(0, alive(0)), (9 * MIN, alive(9 * MIN)), (15 * MIN, alive(15 * MIN)),
                              (30 * MIN, alive(15 * MIN))])  # the file goes 10+ minutes old at the last look
    assert acts == [None, None, None, None] and st["idle_since"] == T0 + 30 * MIN
    st, acts = walk(state(), [(31 * MIN, obs(alive=T0 + 30 * MIN)), (42 * MIN, obs(alive=T0 + 30 * MIN)),
                              (53 * MIN, obs(alive=T0 + 30 * MIN))])
    assert acts == [None, None, "stop"]  # the file was fresh at 31 minutes; idle counts from 42
    late = state(seen_running=True)  # after a run was seen the manager's looks no longer hold it
    _, acts = walk(late, [(0, alive(0)), (11 * MIN, alive(11 * MIN))])
    assert acts == [None, "stop"]


def test_a_paused_queue_holds_it_back_until_the_container_restarted():
    st = state(seen_running=True)
    _, acts = walk(st, [(0, obs(waiting=2, paused=True)), (30 * MIN, obs(waiting=2, paused=True))])
    assert acts == [None, None]
    restarted = obs(waiting=2, paused=True, booted=BOOT + 1000)
    st, acts = walk(state(seen_running=True), [(0, restarted)])
    assert st["rebooted"] is True and st["booted"] == BOOT + 1000  # idle counts from the restart
    st, acts = walk(st, [(1, restarted), (11 * MIN, restarted)])
    assert acts == [None, "stop"]
    st, _ = walk(st, [(12 * MIN, obs(running=True, booted=BOOT + 1000))])  # a new run: the flag goes
    assert st["rebooted"] is False
    _, acts = walk(st, [(13 * MIN, obs(waiting=1, paused=True, booted=BOOT + 1000)),
                        (50 * MIN, obs(waiting=1, paused=True, booted=BOOT + 1000))])
    assert acts == [None, None]


def test_a_waiting_queue_that_is_not_paused_holds_it_even_after_a_restart():
    st, acts = walk(state(seen_running=True), [(0, obs(waiting=1, booted=BOOT + 1000)),
                                               (30 * MIN, obs(waiting=1, booted=BOOT + 1000))])
    assert acts == [None, None]


def test_a_run_that_was_going_when_the_container_restarted_counts_as_ended():
    st = state(seen_running=True, idle_since=None)
    st, acts = walk(st, [(0, obs(running=True)), (5, obs(booted=BOOT + 800)), (10 * MIN + 6, obs(booted=BOOT + 800))])
    assert acts == [None, None, "stop"]  # idle from the restart, not from the run


def test_a_fired_watchdog_never_fires_again_and_a_restart_after_it_spends_it():
    st = state(seen_running=True, fired=True, fired_boot=BOOT)
    for at in (0, 60 * MIN):
        st, action = watch.decide(T0 + at, st, obs())
        assert action is None and not st.get("spent")
    st, action = watch.decide(T0 + 61 * MIN, st, obs(booted=BOOT + 5000))
    assert action is None and st["spent"] is True  # the container came back: nothing until re-armed


def test_a_clock_wobble_in_the_boot_time_is_not_a_restart():
    st, _ = walk(state(), [(0, obs(booted=BOOT + 2.5))])
    assert not st.get("rebooted")


def test_decide_does_not_change_its_input():
    original = state(seen_running=True, idle_since=T0)
    frozen = dict(original)
    watch.decide(T0 + 3600, original, obs())
    assert original == frozen


# ---------------------------------------------------------------- one look with a fake stop request


@pytest.fixture
def root(tmp_path):
    (tmp_path / "output" / "_watchdog").mkdir(parents=True)
    return tmp_path


def write_state(root, **kw):
    (root / "output" / "_watchdog" / "state.json").write_text(json.dumps(state(**kw)), encoding="utf-8")


def read_state(root):
    return json.loads((root / "output" / "_watchdog" / "state.json").read_text(encoding="utf-8"))


def test_fired_is_marked_before_the_request_is_sent(root):
    write_state(root, seen_running=True, idle_since=T0)
    seen = {}

    def stop():
        seen["state"] = read_state(root)
        return True, "Stop sent to vast.ai."

    assert watch.step(root, T0 + 11 * MIN, obs(), stop) == "stopped"
    assert seen["state"]["fired"] is True and seen["state"]["fired_boot"] == BOOT  # on disk before the request
    assert read_state(root)["fired"] is True and read_state(root)["last"] == "Stop sent to vast.ai."
    assert watch.step(root, T0 + 15 * MIN, obs(), lambda: pytest.fail("fired twice")) is None


def test_a_failed_request_unfires_and_is_tried_again_every_five_minutes_while_idle(root):
    write_state(root, seen_running=True, idle_since=T0)
    calls = []

    def stop():
        calls.append(1)
        return False, "vast.ai answered HTTP 500."

    assert watch.step(root, T0 + 11 * MIN, obs(), stop) == "failed"
    st = read_state(root)
    assert st["fired"] is False and st["next_try"] == T0 + 11 * MIN + 300 and st["fired_boot"] is None
    assert watch.step(root, T0 + 14 * MIN, obs(), stop) is None and len(calls) == 1  # not yet
    assert watch.step(root, T0 + 16 * MIN + 1, obs(), stop) == "failed" and len(calls) == 2
    assert watch.step(root, T0 + 22 * MIN, obs(running=True), stop) is None  # a run came: no more tries
    assert read_state(root)["idle_since"] is None and read_state(root)["next_try"] is None
    ok = lambda: (True, "ok")  # noqa: E731
    write_state(root, seen_running=True, idle_since=T0, next_try=T0 + 100)
    assert watch.step(root, T0 + 11 * MIN, obs(), ok) == "stopped"


def test_a_disarmed_state_does_nothing_and_a_look_never_overwrites_the_agents_fields(root):
    write_state(root, armed=False, seen_running=True, idle_since=T0)
    assert watch.step(root, T0 + 3600, obs(), lambda: pytest.fail("stopped while disarmed")) == "disarmed"
    write_state(root, minutes=10)
    p = root / "output" / "_watchdog" / "state.json"
    data = json.loads(p.read_text())
    data["minutes"] = 90  # re-armed with another time between the program's read and write
    original = watch.read_json

    def reading(path, default=None):
        out = original(path, default)
        if Path(path) == p:
            p.write_text(json.dumps(data), encoding="utf-8")
        return out

    watch.read_json = reading
    try:
        watch.step(root, T0, obs(running=True), lambda: (True, ""))
    finally:
        watch.read_json = original
    assert read_state(root)["minutes"] == 90 and read_state(root)["seen_running"] is True


# ---------------------------------------------------------------- what it observes


def test_observe_reads_runs_queue_tests_arriving_files_fetches_and_the_alive_file(root):
    now = time.time()
    assert watch.observe(root, now)["running"] is False
    (root / "output" / "_queue.json").write_text(json.dumps({"waiting": [{"id": "a"}, {"id": "b"}], "paused": True}))
    got = watch.observe(root, now)
    assert got["waiting"] == 2 and got["paused"] is True and got["arriving"] is False and got["alive"] is None
    sent = root / "models" / "sent" / "0123456789ab"
    sent.mkdir(parents=True)
    f = sent / "x.bin.part"
    f.write_bytes(b"x")
    os.utime(f, (now - 60, now - 60))
    assert watch.observe(root, now)["arriving"] is True  # a file grew a minute ago
    os.utime(f, (now - 600, now - 600))
    assert watch.observe(root, now + 3600)["arriving"] is False  # an hour later: not arriving (ctime is real time)
    incoming = root / "datasets" / ".rulz_incoming" / "s1"
    incoming.mkdir(parents=True)
    (incoming / "a.png").write_bytes(b"1")
    assert watch.observe(root, time.time())["arriving"] is True  # a dataset being unpacked (real file time)
    (root / "output" / "_watchdog" / "alive").touch()
    assert abs(watch.observe(root, now)["alive"] - time.time()) < 10


def test_observe_sees_a_running_file_fetch_a_live_job_and_a_running_test(root):
    fetch = root / "output" / "_fetch"
    fetch.mkdir()
    (fetch / "a.json").write_text(json.dumps({"state": "running", "pid": os.getpid(), "updated": T0}))
    assert watch.observe(root, T0 + 10_000)["arriving"] is True
    (fetch / "a.json").write_text(json.dumps({"state": "running", "pid": 999999999, "updated": T0}))
    assert watch.observe(root, T0 + 10_000)["arriving"] is False  # its process is gone
    (fetch / "a.json").write_text(json.dumps({"state": "starting", "pid": None, "updated": T0}))
    assert watch.observe(root, T0 + 10)["arriving"] is True and watch.observe(root, T0 + 1000)["arriving"] is False
    (root / ".webui_job.json").write_text(json.dumps({"pid": os.getpid()}))
    assert watch.observe(root, T0)["running"] is True
    (root / ".webui_job.json").write_text(json.dumps({"pid": 999999999}))
    assert watch.observe(root, T0)["running"] is False
    test = root / "output" / "_tests" / "t1"
    test.mkdir(parents=True)
    (test / "status.json").write_text(json.dumps({"state": "running", "pid": os.getpid()}))
    assert watch.observe(root, T0)["tests"] is True
    (test / "status.json").write_text(json.dumps({"state": "done"}))
    assert watch.observe(root, T0)["tests"] is False


# ---------------------------------------------------------------- the stop request


class Reply:
    def __init__(self, body=b"{}"):
        self.body = body

    def read(self):
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_the_stop_request_is_a_put_with_the_key_in_a_header_only():
    seen = {}

    def opener(request, timeout=0):
        seen.update(url=request.full_url, method=request.get_method(), body=request.data, auth=request.get_header(
            "Authorization"))
        return Reply(b'{"success": true}')

    env = {"CONTAINER_ID": "123", "CONTAINER_API_KEY": "sekret-key"}
    ok, message = watch.stop_instance(env, opener)
    assert ok and seen["url"] == "https://console.vast.ai/api/v0/instances/123/" and seen["method"] == "PUT"
    assert json.loads(seen["body"]) == {"state": "stopped"} and seen["auth"] == "Bearer sekret-key"
    assert "sekret" not in message
    ok, _ = watch.stop_instance({"VAST_CONTAINERLABEL": "C.77", "CONTAINER_API_KEY": "k"}, opener)
    assert ok and seen["url"].endswith("/instances/77/")


def test_failed_stop_requests_say_why_without_the_key_or_an_address():
    import urllib.error

    def refuse(request, timeout=0):
        return Reply(b'{"success": false, "msg": "sekret-key is wrong"}')

    def http(request, timeout=0):
        raise urllib.error.HTTPError(request.full_url, 403, "Forbidden", {}, None)

    def down(request, timeout=0):
        raise urllib.error.URLError("https://console.vast.ai/api/v0/instances/123/ sekret-key unreachable")

    env = {"CONTAINER_ID": "123", "CONTAINER_API_KEY": "sekret-key"}
    for opener in (refuse, http, down):
        ok, message = watch.stop_instance(env, opener)
        assert not ok and "sekret" not in message and "vast.ai/api" not in message
    assert "403" in watch.stop_instance(env, http)[1]
    assert watch.stop_instance({"CONTAINER_ID": "5"}, refuse)[0] is False
    assert watch.stop_instance({"CONTAINER_API_KEY": "k"}, refuse)[0] is False


def test_the_key_is_found_in_etc_environment_then_the_first_process(tmp_path, monkeypatch):
    env_file, first = tmp_path / "environment", tmp_path / "environ"
    env_file.write_text("export CONTAINER_ID='42'\nCONTAINER_API_KEY=from-file\n# note\n", encoding="utf-8")
    first.write_bytes(b"CONTAINER_API_KEY=from-pid1\0OTHER=1\0")
    monkeypatch.setattr(watch, "ENV_FILE", env_file)
    monkeypatch.setattr(watch, "FIRST_PROCESS_ENV", first)
    monkeypatch.delenv("CONTAINER_API_KEY", raising=False)
    monkeypatch.delenv("CONTAINER_ID", raising=False)
    values = watch.container_env()
    assert values["CONTAINER_ID"] == "42" and values["CONTAINER_API_KEY"] == "from-pid1"
    monkeypatch.setenv("CONTAINER_API_KEY", "from-env")
    assert watch.container_env()["CONTAINER_API_KEY"] == "from-env"
    monkeypatch.setattr(watch, "FIRST_PROCESS_ENV", tmp_path / "missing")
    monkeypatch.delenv("CONTAINER_API_KEY")
    assert watch.container_env()["CONTAINER_API_KEY"] == "from-file"


# ---------------------------------------------------------------- the agent's commands


@pytest.fixture
def install(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "output").mkdir()
    monkeypatch.setattr(agent, "webui_running", lambda: True)
    monkeypatch.setattr(agent, "supervisorctl", lambda: None)
    monkeypatch.setattr(agent, "WATCH_CONF", tmp_path / "rulz-watch.conf")
    return tmp_path


def agent_json(*args):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = agent.main(list(args))
    return code, (json.loads(out.getvalue()) if out.getvalue().strip() else None), err.getvalue()


class VastKey:
    """A fake vast.ai: answers the key check, remembers what was asked."""

    def __init__(self, status=200, body=b'{"instances": {"id": 9}}'):
        self.status, self.body, self.seen = status, body, []

    def __call__(self, request, timeout=0):
        import urllib.error

        self.seen.append((request.get_method(), request.full_url, request.get_header("Authorization")))
        if self.status != 200:
            raise urllib.error.HTTPError(request.full_url, self.status, "x", {}, None)
        return Reply(self.body)


ENV = {"CONTAINER_ID": "9", "CONTAINER_API_KEY": "sekret-key"}


def arm_with(monkeypatch, vast, spawned):
    monkeypatch.setattr(agent, "container_env", lambda *a, **k: dict(ENV))
    monkeypatch.setattr(agent, "URLOPEN", vast)
    monkeypatch.setattr(agent, "start_watch_process", lambda root: spawned.append(root))
    monkeypatch.setattr(agent, "WATCH_START_WAIT", 0.0)


def test_arming_checks_the_key_with_a_read_writes_the_program_and_starts_it(install, monkeypatch):
    vast, spawned = VastKey(), []
    arm_with(monkeypatch, vast, spawned)
    code, data, _ = agent_json("watchdog", "arm", "--idle-minutes", "30")
    assert code == 0 and data["can_stop"] is True and data["minutes"] == 30 and data["supervised"] is False
    assert vast.seen == [("GET", "https://console.vast.ai/api/v0/instances/9/", "Bearer sekret-key")]
    assert (install / "output" / "_watchdog" / "watch.py").read_text(encoding="utf-8") == agent.WATCH_PROGRAM
    st = json.loads((install / "output" / "_watchdog" / "state.json").read_text())
    assert st["armed"] is True and st["minutes"] == 30 and st["fired"] is False and st["seen_running"] is False
    assert spawned == [install]
    assert "sekret" not in json.dumps(st) + (install / "output" / "_watchdog" / "watch.py").read_text()


def test_a_key_that_cannot_stop_this_instance_is_reported_and_nothing_starts(install, monkeypatch):
    spawned = []
    arm_with(monkeypatch, VastKey(status=401), spawned)
    code, data, _ = agent_json("watchdog", "arm")
    assert code == 0 and data["can_stop"] is False and data["armed"] is False and "401" in data["reason"]
    assert "sekret" not in json.dumps(data) and not spawned
    assert not (install / "output" / "_watchdog" / "watch.py").exists()
    monkeypatch.setattr(agent, "container_env", lambda *a, **k: {"CONTAINER_ID": "9"})
    assert agent_json("watchdog", "arm")[1]["can_stop"] is False  # no key at all
    for bad in ("5", "5000"):
        assert agent_json("watchdog", "arm", "--idle-minutes", bad)[0] == 1  # outside 10..1440


def test_arming_again_updates_the_minutes_and_starts_no_second_process(install, monkeypatch):
    spawned = []
    arm_with(monkeypatch, VastKey(), spawned)
    agent_json("watchdog", "arm", "--idle-minutes", "30")
    st_path = install / "output" / "_watchdog" / "state.json"
    st = json.loads(st_path.read_text())
    st.update(pid=os.getpid())  # as the program writes its own
    st_path.write_text(json.dumps(st))
    monkeypatch.setattr(agent, "pid_running", lambda pid, created=None: True)
    code, data, _ = agent_json("watchdog", "arm", "--idle-minutes", "90")
    assert data["minutes"] == 90 and data["armed"] is True and spawned == [install]


def test_rearming_after_a_fire_clears_it(install, monkeypatch):
    spawned = []
    arm_with(monkeypatch, VastKey(), spawned)
    agent_json("watchdog", "arm")
    st_path = install / "output" / "_watchdog" / "state.json"
    st = json.loads(st_path.read_text())
    st.update(fired=True, fired_boot=1.0, spent=True, seen_running=True, rebooted=True)
    st_path.write_text(json.dumps(st))
    monkeypatch.setattr(agent, "pid_running", lambda pid, created=None: True)
    assert agent_json("watchdog", "status")[1]["armed"] is False  # fired: not armed
    agent_json("watchdog", "arm")
    st = json.loads(st_path.read_text())
    assert (st["fired"], st["spent"], st["seen_running"], st["rebooted"]) == (False, False, False, False)


def test_status_armed_needs_the_process_to_be_alive_and_not_fired(install, monkeypatch):
    p = install / "output" / "_watchdog"
    p.mkdir(parents=True)
    (p / "state.json").write_text(json.dumps({"armed": True, "minutes": 60, "pid": 4242, "idle_since": T0,
                                              "seen_running": True}))
    monkeypatch.setattr(agent, "pid_running", lambda pid, created=None: False)
    block = agent_json("watchdog", "status")[1]
    assert block["armed"] is False and block["alive"] is False and block["minutes"] == 60 and block["fired"] is False
    monkeypatch.setattr(agent, "pid_running", lambda pid, created=None: True)
    block = agent_json("watchdog", "status")[1]
    assert block["armed"] is True and block["idle_since"] == T0 and block["seen_running"] is True
    assert set(block) >= {"armed", "idle_since", "minutes", "fired"}


def test_alive_touches_the_file_and_status_reports_its_age(install):
    assert agent_json("watchdog", "status")[1]["alive_age"] is None
    assert agent_json("watchdog", "alive")[1]["alive"] is True
    age = agent_json("watchdog", "status")[1]["alive_age"]
    assert age is not None and age < 10


def test_disarm_ends_it(install, monkeypatch):
    spawned = []
    arm_with(monkeypatch, VastKey(), spawned)
    agent_json("watchdog", "arm")
    st_path = install / "output" / "_watchdog" / "state.json"
    killed = []

    class Proc:
        def terminate(self):
            killed.append(1)

    st = json.loads(st_path.read_text())
    st.update(pid=4242, created=None)
    st_path.write_text(json.dumps(st))
    monkeypatch.setattr(agent, "live_process", lambda pid, created=None: Proc())
    monkeypatch.setattr(agent, "pid_running", lambda pid, created=None: True)
    data = agent_json("watchdog", "disarm")[1]
    assert data["armed"] is False and killed == [1]
    assert json.loads(st_path.read_text())["armed"] is False


def test_with_supervisor_it_becomes_a_supervisor_program_that_restarts(install, monkeypatch):
    calls = []
    monkeypatch.setattr(agent, "supervisorctl", lambda: ["/usr/bin/supervisorctl"])
    monkeypatch.setattr(agent, "ctl_run", lambda ctl, *words: calls.append(words) or True)
    monkeypatch.setattr(agent, "container_env", lambda *a, **k: dict(ENV))
    monkeypatch.setattr(agent, "URLOPEN", VastKey())
    spawned = []
    monkeypatch.setattr(agent, "start_watch_process", lambda root: spawned.append(root))
    monkeypatch.setattr(agent, "WATCH_START_WAIT", 0.0)
    agent_json("watchdog", "arm")
    conf = (install / "rulz-watch.conf").read_text(encoding="utf-8")
    assert "[program:rulz-watch]" in conf and "autostart=true" in conf and "autorestart=true" in conf
    assert "output/_watchdog/watch.py" in conf and "--supervised" in conf
    assert ("reread",) in calls and ("update",) in calls and ("start", "rulz-watch") in calls
    assert spawned == [install]  # supervisor did not bring it up (fake): the detached fallback
    st = json.loads((install / "output" / "_watchdog" / "state.json").read_text())
    assert st["supervised"] is False
    calls.clear()
    agent_json("watchdog", "disarm")
    assert ("stop", "rulz-watch") in calls and not (install / "rulz-watch.conf").exists()


def test_the_program_runs_for_real_and_records_its_pid(install, monkeypatch):
    """The detached program, started as a process: it writes its pid and, disarmed, ends."""
    import subprocess

    (install / "output" / "_watchdog").mkdir(parents=True)
    program = install / "output" / "_watchdog" / "watch.py"
    program.write_text(agent.WATCH_PROGRAM, encoding="utf-8")
    state = install / "output" / "_watchdog" / "state.json"
    state.write_text(json.dumps({"armed": False, "minutes": 10}))
    done = subprocess.run([sys.executable, str(program), "--root", str(install)], timeout=60, capture_output=True)
    assert done.returncode == 0 and json.loads(state.read_text())["pid"] is not None


def test_snapshot_carries_the_watchdog_tests_and_autostop(install, monkeypatch):
    monkeypatch.setattr(agent, "pid_running", lambda pid, created=None: True)
    code, data, _ = agent_json("snapshot")
    assert data["watchdog"]["armed"] is False and data["watchdog"]["fired"] is False
    assert data["tests_running"] == 0 and data["autostop"] == {"armed": False}
    (install / "output" / "_autostop.json").write_text(json.dumps({"armed": True, "seen_running": True}))
    t = install / "output" / "_tests" / "t1"
    t.mkdir(parents=True)
    (t / "status.json").write_text(json.dumps({"state": "running", "pid": os.getpid()}))
    monkeypatch.undo()
    monkeypatch.chdir(install)
    monkeypatch.setattr(agent, "webui_running", lambda: True)
    data = agent_json("snapshot")[1]
    assert data["autostop"] == {"armed": True} and data["tests_running"] == 1
    p = install / "output" / "_watchdog"
    p.mkdir(parents=True)
    (p / "state.json").write_text(json.dumps({"armed": True, "minutes": 45, "pid": os.getpid(), "idle_since": T0,
                                              "fired": False}))
    data = agent_json("snapshot")[1]["watchdog"]
    assert data["armed"] is True and data["minutes"] == 45 and data["idle_since"] == T0 and data["autostop_armed"] is True


# ---------------------------------------------------------------- review fixes


def test_provisioning_counts_only_young_non_viewer_processes_naming_a_setup_step():
    pp = watch.provisioning_process
    assert pp(["bash", "/opt/x/provision.sh"], "bash", 60)
    assert pp(["/bin/bash", "scripts/vast/download_models.sh"], "bash", 60)
    assert pp(["/venv/main/bin/pip", "install", "-r", "requirements.txt"], "pip", 60)
    assert pp(["python", "-m", "pip", "install", "x"], "python", 60)
    assert pp(["bash", "install.sh", "system"], "bash", 60)
    assert not pp(["tail", "-F", "/var/log/portal/provision.sh"], "tail", 60)  # a log viewer
    assert not pp(["less", "provisioner.log"], "less", 60) and not pp(["grep", "install.sh", "x"], "grep", 60)
    assert not pp(["bash", "/opt/x/provision.sh"], "bash", 7200 + 1)  # older than two hours: not going on
    assert not pp(["python", "portal.py", "--note", "provisioning-done"], "python", 60)
    assert not pp([], "x", 0) and not pp(["pip", "list"], "pip", 5)


def test_a_stop_vast_accepted_but_that_did_not_happen_is_asked_again_every_ten_minutes():
    st = state(seen_running=True, fired=True, fired_at=T0, fired_boot=BOOT)
    st, action = watch.decide(T0 + 9 * MIN, st, obs())
    assert action is None
    st, action = watch.decide(T0 + 10 * MIN + 1, st, obs())
    assert action == "stop"
    st, action = watch.decide(T0 + 10 * MIN + 1, dict(st, fired_at=T0 + 10 * MIN), obs(running=True))
    assert action is None and st["fired_at"] == T0 + 10 * MIN + 1  # busy again: the wait starts over
    st, action = watch.decide(T0 + 60 * MIN, dict(st, fired_at=T0), obs(booted=BOOT + 900))
    assert action is None and st["spent"] is True  # a restart after the fire: never until re-armed
    assert watch.decide(T0 + 99 * MIN, st, obs(booted=BOOT + 900))[1] is None


def test_step_asks_again_after_ten_idle_minutes_following_a_fire(root):
    write_state(root, seen_running=True, idle_since=T0)
    calls = []
    stop = lambda: calls.append(1) or (True, "ok")  # noqa: E731
    assert watch.step(root, T0 + 11 * MIN, obs(), stop) == "stopped"
    assert watch.step(root, T0 + 15 * MIN, obs(), stop) is None
    assert watch.step(root, T0 + 21 * MIN + 1, obs(), stop) == "stopped" and len(calls) == 2


def test_newest_mtime_counts_ctime_directories_and_every_file(tmp_path):
    folder = tmp_path / "f"
    (folder / "sub").mkdir(parents=True)
    old = time.time() - 86400
    for i in range(300):
        f = folder / "sub" / f"{i:04d}.bin"
        f.write_bytes(b"x")
        os.utime(f, (old, old))  # tar restores old mtimes: ctime is the real time of the write
    assert watch.newest_mtime(folder) > time.time() - 120
    quiet = tmp_path / "q"
    quiet.mkdir()
    assert watch.newest_mtime(quiet) > time.time() - 120  # the folder itself counts
    assert watch.newest_mtime(tmp_path / "missing") is None
    assert watch.newest_mtime(folder, fresh_after=time.time() - 120) > time.time() - 120  # early stop


@pytest.mark.skipif(sys.platform == "win32", reason="Windows refuses concurrent replaces of one file; the lock serialises them on the instance")
def test_every_state_change_runs_under_the_lock_and_temp_names_are_unique(root, monkeypatch):
    entered = []

    class Lock:
        def __init__(self, folder):
            entered.append(Path(folder))

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(watch, "locked", Lock)
    write_state(root, seen_running=True)
    watch.step(root, T0, obs(running=True), lambda: (True, ""))
    watch.save_owned(root / "output" / "_watchdog" / "state.json", {"last": "x"})
    assert len(entered) == 2 and all(p == root / "output" / "_watchdog" for p in entered)
    import threading

    errors = []

    def writer(n):
        try:
            for i in range(30):
                watch.write_json(root / "output" / "_watchdog" / "shared.json", {"n": n, "i": i})
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=writer, args=(n,)) for n in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors and not list((root / "output" / "_watchdog").glob("*.tmp"))


def test_the_agent_changes_state_under_the_same_lock(install, monkeypatch):
    entered = []
    real = agent.embedded_watch().locked

    def spy(folder):
        entered.append(Path(folder))
        return real(folder)

    monkeypatch.setattr(agent.embedded_watch(), "locked", spy)
    agent.update_watch_state(armed=True)
    agent.watchdog_disarm()
    assert len(entered) == 2 and entered[0] == agent.WATCH_DIR


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fcntl locks need a POSIX system")
def test_the_lock_excludes_a_second_holder(tmp_path):
    import threading

    order = []
    with watch.locked(tmp_path):
        def other():
            with watch.locked(tmp_path):
                order.append("second")
        t = threading.Thread(target=other)
        t.start()
        time.sleep(0.3)
        order.append("first")
    t.join(5)
    assert order == ["first", "second"]


def test_the_supervisor_program_is_what_vast_images_need(install):
    conf = agent.watch_conf_text("/venv/main/bin/python", Path("/workspace/RulzTrainer"))
    assert "command=/venv/main/bin/python /workspace/RulzTrainer/output/_watchdog/watch.py --root /workspace/RulzTrainer" in conf
    assert "--supervised" in conf and "directory=/workspace/RulzTrainer" in conf
    assert "autostart=true" in conf and "autorestart=true" in conf and "user=" not in conf
    assert "stdout_logfile=/dev/null" in conf and "redirect_stderr=true" in conf  # no log file that could grow


def test_a_supervised_program_stays_up_when_disarmed_so_autorestart_does_not_loop(install):
    import subprocess

    folder = install / "output" / "_watchdog"
    folder.mkdir(parents=True)
    (folder / "watch.py").write_text(agent.WATCH_PROGRAM, encoding="utf-8")
    (folder / "state.json").write_text(json.dumps({"armed": False}))
    proc = subprocess.Popen([sys.executable, str(folder / "watch.py"), "--root", str(install), "--supervised"])
    try:
        time.sleep(2.0)
        assert proc.poll() is None  # sleeping, not exiting (a normal exit would be restarted at once)
    finally:
        proc.kill()
        proc.wait()
    plain = subprocess.run([sys.executable, str(folder / "watch.py"), "--root", str(install)], timeout=60)
    assert plain.returncode == 0  # unsupervised and disarmed: ends


def test_everything_the_agent_ships_compiles_for_python_3_10():
    import ast
    import tokenize

    src = (REPO / "scripts" / "vast" / "rulz_agent.py").read_text(encoding="utf-8")
    for text in (src, agent.WATCH_PROGRAM, agent.FETCH_PROGRAM):
        ast.parse(text, feature_version=(3, 10))
        stack = []  # an f-string quoted inside another f-string with the same quote is a syntax error before 3.12
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type == getattr(tokenize, "FSTRING_START", -1):
                quote = tok.string.lstrip("rRbBfF")
                assert quote not in [q for q in stack], f"nested same-quote f-string at line {tok.start[0]}"
                stack.append(quote)
            elif tok.type == getattr(tokenize, "FSTRING_END", -1):
                stack.pop()
    for banned in ("datetime.UTC", "tomllib", "typing.Self", "ExceptionGroup", "except*", "StrEnum"):
        assert banned not in src + agent.WATCH_PROGRAM + agent.FETCH_PROGRAM
