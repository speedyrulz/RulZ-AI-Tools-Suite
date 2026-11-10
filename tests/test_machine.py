"""The machine panel's readings and the stop-when-done switch for a rented instance."""
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import machine  # noqa: E402

NO_FILE = Path("/nonexistent/environment")


def test_disks_share_a_row_per_drive_and_accept_missing_folders(tmp_path):
    rows = machine.disks({"runs": tmp_path / "output", "models": tmp_path / "models" / "deep"})
    assert len(rows) == 1
    row = rows[0]
    assert row["label"] == "runs, models"
    assert row["total"] > 0 and 0 <= row["free"] <= row["total"]
    assert row["level"] == machine.disk_level(row["free"])


def test_disk_levels():
    assert machine.disk_level(100) == "ok"
    assert machine.disk_level(machine.DISK_WARN_GB - 0.1) == "warning"
    assert machine.disk_level(machine.DISK_CRITICAL_GB - 0.1) == "critical"


def test_memory_reads_sensible_numbers():
    mem = machine.memory()
    assert mem is None or 0 < mem["used"] <= mem["total"]


def test_vast_instance_from_the_container_variables(tmp_path):
    assert machine.vast_instance({}, NO_FILE) is None
    assert machine.vast_instance({"CONTAINER_ID": "abc"}, NO_FILE) is None
    info = machine.vast_instance({"CONTAINER_ID": "123", "CONTAINER_API_KEY": "k"}, NO_FILE)
    assert info == {"id": 123, "label": "C.123", "can_stop": True, "ip": ""}
    assert machine.vast_instance({"VAST_CONTAINERLABEL": "C.77"}, NO_FILE)["can_stop"] is False
    # a service that did not inherit them: /etc/environment has them
    env_file = tmp_path / "environment"
    env_file.write_text('PATH=/usr/bin\nCONTAINER_ID="55"\nexport CONTAINER_API_KEY=secret\n', encoding="utf-8")
    assert machine.vast_instance({}, env_file) == {"id": 55, "label": "C.55", "can_stop": True, "ip": ""}


class Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_stop_instance_sends_the_instances_own_key():
    seen = {}

    def opener(request, timeout):
        seen.update(url=request.full_url, method=request.get_method(), body=json.loads(request.data),
                    auth=request.get_header("Authorization"))
        return Response(b'{"success": true}')

    ok, message = machine.stop_instance({"CONTAINER_ID": "123", "CONTAINER_API_KEY": "k"}, NO_FILE, opener)
    assert ok and "123" in message
    assert seen == {"url": f"{machine.VAST_API}/instances/123/", "method": "PUT", "body": {"state": "stopped"},
                    "auth": "Bearer k"}


def test_stop_instance_refuses_without_a_key_and_reports_a_failure(monkeypatch):
    called = []
    ok, _ = machine.stop_instance({"CONTAINER_ID": "123"}, NO_FILE, lambda *a, **k: called.append(1))
    assert not ok and not called
    monkeypatch.setattr(machine.shutil, "which", lambda name: None)
    ok, message = machine.stop_instance({"CONTAINER_ID": "123", "CONTAINER_API_KEY": "k"}, NO_FILE,
                                        lambda *a, **k: Response(b'{"success": false, "msg": "no such instance"}'))
    assert not ok and "no such instance" in message


def test_bucket_download_states(tmp_path):
    assert machine.bucket_download(tmp_path, max_age=0) is None
    (tmp_path / "bucket").mkdir()
    (tmp_path / "bucket_download.log").write_text("== Syncing hf://buckets/x -> y\n", encoding="utf-8")
    (tmp_path / "bucket" / "a.safetensors").write_bytes(b"x" * 1024)
    state = machine.bucket_download(tmp_path, max_age=0)
    assert state["state"] == "running" and state["gb"] > 0 and "Syncing" in state["line"]
    (tmp_path / "bucket_download.log").write_text("x\n[ERROR] model download failed after 3 attempts\n",
                                                  encoding="utf-8")
    assert machine.bucket_download(tmp_path, max_age=0)["state"] == "failed"
    (tmp_path / "bucket" / ".download_complete").write_text("now", encoding="utf-8")
    assert machine.bucket_download(tmp_path, max_age=0)["state"] == "done"


def auto(tmp_path, calls):
    return machine.AutoStop(tmp_path / "_autostop.json", grace=100,
                            stop=lambda: (calls.append(1), (True, "stopped"))[1])


def test_autostop_waits_for_a_run_then_the_grace_time(tmp_path):
    calls = []
    a = auto(tmp_path, calls)
    assert a.tick(False, 0, now=0) is None  # not armed
    a.arm()
    assert a.tick(False, 0, now=1) is None and a.seconds_left(1) is None  # no run seen yet: it never stops an idle one
    assert a.tick(True, 1, now=2) is None
    assert a.tick(False, 1, now=3) is None and a.seconds_left(3) is None  # the queue's next run is about to start
    assert a.tick(True, 0, now=4) is None
    assert a.tick(False, 0, now=10) is None and a.seconds_left(10) == 100
    assert a.tick(False, 0, now=60) is None and a.seconds_left(60) == 50
    assert not calls
    assert a.tick(False, 0, now=111) == (True, "stopped")
    assert calls == [1] and not a.armed
    assert a.tick(False, 0, now=500) is None and calls == [1]  # once


def test_autostop_starts_over_when_a_run_starts_in_the_grace_time(tmp_path):
    calls = []
    a = auto(tmp_path, calls)
    a.arm(running=True)
    a.tick(False, 0, now=0)
    a.tick(True, 0, now=50)  # started by hand
    assert a.seconds_left(50) is None
    a.tick(False, 0, now=60)
    assert a.tick(False, 0, now=120) is None and not calls
    assert a.tick(False, 0, now=161) == (True, "stopped")


def test_autostop_is_disarmed_on_disk_before_the_stop_and_survives_a_restart(tmp_path):
    path = tmp_path / "_autostop.json"
    at_stop = {}

    def stop():
        at_stop.update(json.loads(path.read_text(encoding="utf-8")))
        return True, "stopped"

    a = machine.AutoStop(path, grace=10, stop=stop)
    a.arm(running=True)
    again = machine.AutoStop(path, grace=10, stop=stop)  # a restarted WebUI still knows
    assert again.armed and again.state["seen_running"]
    again.tick(False, 0, now=0)
    again.tick(False, 0, now=11)
    assert at_stop["armed"] is False  # a restarted instance must not stop again
    assert not machine.AutoStop(path).armed
    a.disarm()
    assert a.seconds_left() is None
