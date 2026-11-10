"""The WebUI's Fleet tab: how it words an instance's card, an event and a past run, and its small chart."""
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
pytest.importorskip("nicegui")
pytest.importorskip("psutil")

import webui  # noqa: E402


def test_a_cards_run_line_says_progress_loss_time_left_and_cost():
    inst = {"job": {"job": "a", "running": True, "step": 1240, "total": 2000, "avg_loss": 0.2075, "eta": 640.0},
            "metrics": {"cost": 0.281}}
    assert webui.fleet_job_line(inst) == "step 1,240 / 2,000 · loss 0.2075 · 11 min left · $0.28 so far"
    ended = {"job": {"job": "a", "running": False, "step": 2000, "total": 2000, "done": True}}
    assert webui.fleet_job_line(ended) == "step 2,000 / 2,000 · finished"
    assert webui.fleet_job_line({"job": {"running": False, "step": 5, "total": 9, "error": "x"}}).endswith("failed")
    assert webui.fleet_job_line({"job": {"running": False, "step": 5, "total": 9}}).endswith("ended early")
    assert webui.fleet_job_line({}) == ""


def test_the_last_probe_reads_as_one_line():
    probe = {"step": 1200, "face_ema": 73.61, "body_ema": 48.2, "combined_ema": 47.0, "best": 50.0, "best_step": 1000}
    assert webui.fleet_probe_text(probe) == "step 1,200: face 73.6 · body 48.2 · combined 47.0 · best 50.0 at 1,000"
    assert webui.fleet_probe_text({"face_ema": 60.0}) == "face 60.0"
    assert webui.fleet_probe_text(None) == ""


def test_the_machine_line_names_disk_gpu_and_the_looks_age():
    inst = {"disk_free_gb": 80.2, "disk_total_gb": 200.0, "age": 12,
            "gpus": [{"utilization": 97, "temperature": 60, "memory_used_mb": 9216, "memory_total_mb": 32768}]}
    assert webui.fleet_machine_line(inst) == "disk 80 GB free of 200 · GPU 97% · 60°C · 9.0 / 32 GB · seen 12 s ago"
    assert webui.fleet_machine_line({"age": 600}) == "seen 10 min ago"


def test_every_kind_of_event_reads_as_a_sentence():
    import rulz_fleet

    base = {"instance": 7, "job": "a", "error": "boom", "status": "exited", "step": 40}
    for kind in rulz_fleet.EVENT_KINDS:
        if kind != "watch":
            text = webui.fleet_event_text(dict(base, kind=kind))
            assert text and text != kind and "None" not in text, kind
    watch = {"kind": "watch", "metric": "avg_loss", "op": "above", "value": 0.5, "actual": 0.71, "instance": 7,
             "job": "a", "step": 120.0, "note": "diverging?"}
    assert webui.fleet_event_text(watch) == \
        "avg_loss went above 0.5: now 0.71 (a on instance 7, step 120) · diverging?"
    machine = {"kind": "watch", "metric": "disk_free_gb", "op": "below", "value": 15.0, "actual": 4.0, "instance": 7}
    assert webui.fleet_event_text(machine) == "disk_free_gb went below 15: now 4 (instance 7)"
    assert webui.fleet_event_text({"kind": "run_done", "job": "a", "instance": 7, "best": 55.52}) == \
        "a finished on instance 7, best likeness 55.5"


def test_past_runs_become_table_rows():
    rows = webui.fleet_run_rows([
        {"key": "7-a-1", "job": "a", "state": "done", "instance": 7, "gpu": "RTX 5090", "step": 2000, "total": 2000,
         "avg_loss": 0.2075, "best": 49.96, "elapsed": 1740.0, "cost": 0.4444, "ended": 1_000_000.0, "group": "r",
         "downloaded": ["logs", "probes"]},
        {"key": "7-b", "job": "b", "state": "running", "instance": 7}])
    assert rows[0] == dict(rows[0], job="a", state="done", where="7 · RTX 5090", steps="2,000 / 2,000", loss="0.2075",
                           best=50.0, time="29 min", cost="$0.44", group="r", files="logs, probes")
    assert rows[0]["ended"].startswith("1970-01-1")
    assert rows[1]["best"] is None and rows[1]["steps"] == "0 / 0" and rows[1]["cost"] == ""
    assert {c["field"] for c in webui.FLEET_RUN_COLUMNS} <= set(rows[0])


def test_the_small_chart_has_the_loss_and_the_likeness_on_its_own_axis():
    options = webui.fleet_spark({"losses": [[10, 0.5], [20, 0.4]],
                                 "probes": [{"step": 10, "combined_ema": 30.0}, {"step": 20, "combined": 35.0},
                                            {"step": 30}]})
    loss, likeness = options["series"]
    assert loss["data"] == [[10, 0.5], [20, 0.4]] and likeness["data"] == [[10, 30.0], [20, 35.0]]
    assert likeness["yAxisIndex"] == 1 and len(options["yAxis"]) == 2
    assert webui.fleet_spark(None)["series"][0]["data"] == []


def test_the_fleet_is_one_object_for_the_whole_webui(tmp_path, monkeypatch):
    monkeypatch.setattr(webui, "_FLEET", None)
    import rulz_remote

    monkeypatch.setattr(rulz_remote, "STATE", tmp_path)
    first = webui.fleet()
    assert webui.fleet() is first and first.home == tmp_path / "fleet"
