"""Previous Runs lists the training runs the Fleet tracked on vast.ai instances (history.import_fleet_runs)."""
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import history  # noqa: E402


def probe(step, with_images=False):
    p = {"step": step, "face": 40.0 + step / 100, "body": 30.0, "combined": 35.0 + step / 100,
         "face_ema": 40.0, "body_ema": 30.0, "combined_ema": 35.0 + step / 100,
         "face_norm_ema": 45.0, "body_norm_ema": 33.0}
    if with_images:
        p["images"] = [f"output/job/probes/step{step:06d}_face_0.png"]
    return p


def fleet_record(tmp_path, folder=None, **over):
    rec = {"key": "123-job-20261101-120000", "instance": 123, "job": "job", "run_id": "job-20261101-120000",
           "gpu": "RTX 5090", "state": "done", "first_seen": 1000.0, "started": 1100.0, "ended": 2000.0,
           "modified": 1999, "step": 500, "total": 500, "config": {"job_name": "job", "steps": 500},
           "probes": [probe(0), probe(250), probe(500)], "folder": str(folder) if folder else "",
           "downloaded": []}
    rec.update(over)
    fleet = tmp_path / "fleet"
    fleet.mkdir(parents=True, exist_ok=True)
    (fleet / f"{rec['key']}.json").write_text(json.dumps(rec), encoding="utf-8")
    return fleet, rec


def test_imports_from_the_fleet_record_without_a_folder(tmp_path):
    fleet, rec = fleet_record(tmp_path)
    root = tmp_path / "hist"
    assert history.import_fleet_runs(fleet, root) == 1
    runs = history.list_runs(root)
    assert len(runs) == 1
    r = runs[0]
    assert r["id"] == "vast-" + rec["key"] and r["state"] == "finished" and r["job_dir"] == ""
    assert r["started"] == 1100.0 and r["updated"] == 2000.0 and r["step"] == 500 and r["total"] == 500
    assert r["source"]["kind"] == "vast" and r["source"]["gpu"] == "RTX 5090" and r["source"]["instance"] == 123
    assert r["probes"]["combined"][-1] == [500, 40.0]
    assert r["probe_images"] == []


def test_state_mapping(tmp_path):
    for state, want in (("done", "finished"), ("error", "error"), ("ended", "stopped"), ("running", "running")):
        fleet, _ = fleet_record(tmp_path / state, state=state)
        root = tmp_path / state / "hist"
        history.import_fleet_runs(fleet, root)
        assert history.list_runs(root)[0]["state"] == want


def test_a_run_without_likeness_data_is_skipped(tmp_path):
    fleet, _ = fleet_record(tmp_path, probes=[])
    assert history.import_fleet_runs(fleet, tmp_path / "hist") == 0


def test_folder_status_gives_thumbnails_from_its_probes_folder(tmp_path):
    PIL = pytest.importorskip("PIL.Image")
    folder = tmp_path / "dl"
    (folder / "probes").mkdir(parents=True)
    for step in (0, 250):
        PIL.new("RGB", (2000, 1000), (200, 10, 10)).save(folder / "probes" / f"step{step:06d}_face_0.png")
    status = {"step": 500, "total": 500, "probes": [probe(0, True), probe(250, True), probe(500, True)]}
    (folder / "status.json").write_text(json.dumps(status), encoding="utf-8")
    (folder / "config.yaml").write_text("job_name: job\nsteps: 500\nlora_rank: 16\n", encoding="utf-8")
    fleet, rec = fleet_record(tmp_path, folder=folder, probes=[])
    root = tmp_path / "hist"
    assert history.import_fleet_runs(fleet, root) == 1
    r = history.list_runs(root)[0]
    assert r["job_dir"] == str(folder) and r["config"].get("lora_rank") == 16
    assert {e["step"] for e in r["probe_images"]} == {0, 250}  # step 500's file was not downloaded
    for e in r["probe_images"]:
        thumb = root / e["file"]
        assert thumb.is_file()
        with PIL.open(thumb) as im:
            assert max(im.size) <= history.PROBE_THUMB_PX


def test_unchanged_run_is_not_rewritten_and_a_label_is_kept_on_update(tmp_path):
    fleet, rec = fleet_record(tmp_path, state="running", ended=None, step=250)
    root = tmp_path / "hist"
    assert history.import_fleet_runs(fleet, root) == 1
    path = root / f"vast-{rec['key']}.json"
    before = path.stat().st_mtime_ns
    assert history.import_fleet_runs(fleet, root) == 0
    assert path.stat().st_mtime_ns == before
    assert history.set_label("vast-" + rec["key"], "my label", root)
    fleet_record(tmp_path, state="done", step=500, modified=2500)
    assert history.import_fleet_runs(fleet, root) == 1
    r = history.list_runs(root)[0]
    assert r["label"] == "my label" and r["state"] == "finished"


def test_delete_hides_the_run_and_the_import_does_not_bring_it_back(tmp_path):
    fleet, rec = fleet_record(tmp_path)
    root = tmp_path / "hist"
    history.import_fleet_runs(fleet, root)
    sig = history.signature(root)
    assert history.delete_runs(["vast-" + rec["key"]], root) == 1
    assert history.list_runs(root) == [] and (root / history.HIDDEN_VAST).is_file()
    assert all(not name.startswith("_") for name, _, _ in history.signature(root))
    assert history.import_fleet_runs(fleet, root) == 0
    assert history.list_runs(root) == []
    assert (fleet / f"{rec['key']}.json").is_file()  # the Fleet's record is untouched
    assert sig != history.signature(root)


def test_missing_fleet_dir(tmp_path):
    assert history.import_fleet_runs(tmp_path / "nope", tmp_path / "hist") == 0
    assert history.fleet_signature(tmp_path / "nope") == ()


def test_fleet_signature_changes_with_the_fleet(tmp_path):
    fleet, rec = fleet_record(tmp_path)
    a = history.fleet_signature(fleet)
    fleet_record(tmp_path, step=499, modified=2100)
    # a rewrite of the same size within one tick of a coarse file clock keeps the old mtime (seen on the Windows CI
    # runner); the Fleet's next write is seen then, so here the rewrite is dated a second later
    path = fleet / f"{rec['key']}.json"
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    assert history.fleet_signature(fleet) != a


def test_history_rows_say_where_a_run_was_trained(tmp_path):
    pytest.importorskip("psutil")
    import webui

    fleet, rec = fleet_record(tmp_path)
    root = tmp_path / "hist"
    history.import_fleet_runs(fleet, root)
    vast = history.list_runs(root)[0]
    local = {**vast, "id": "local-1", "source": None}
    local.pop("source")
    rows = {r["id"]: r for r in webui.history_rows([vast, local])}
    assert rows[vast["id"]]["where"] == "vast.ai · RTX 5090"
    assert rows["local-1"]["where"] == "This PC"
    vast["source"]["gpu"] = None
    assert webui.history_rows([vast])[0]["where"] == "vast.ai"
    assert any(c["name"] == "where" and c["label"] == "Where" for c in webui.HISTORY_COLUMNS)
