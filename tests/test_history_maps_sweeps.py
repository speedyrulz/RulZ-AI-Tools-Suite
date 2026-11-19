"""Previous Runs keeps a run's region sweeps and training maps after the job folder is reused, and compares them."""
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import history  # noqa: E402
from rulztrainer.config import TrainConfig  # noqa: E402


def sweep_row(step, train=0.1, held=None):
    row = {"step": step, "epoch": 1.0, "train": {"face": train, "subject": train / 2}, "gap": {}}
    if held is not None:
        row["held"] = {"face": held, "subject": held / 2}
    return row


def grid(*vals, w=2, h=2):
    return {"w": w, "h": h, "v": list(vals)}


def write_maps(job: Path, steps=(0, 100), pictures=1):
    d = job / "training_maps"
    d.mkdir(parents=True, exist_ok=True)
    probes = [{"id": str(i), "key": f"k{i}", "image": f"probe_{i}.jpg", "source": f"a{i}.png", "words": [],
               "pair": False, "size": [64, 64]} for i in range(pictures)]
    (d / "probes.json").write_text(json.dumps({"probes": probes, "sigmas": [0.25, 0.5, 0.75], "words": []}),
                                   encoding="utf-8")
    from PIL import Image

    for i in range(pictures):
        Image.new("RGB", (64, 64), (90, 90, 90)).save(d / f"probe_{i}.jpg")
    with open(d / "maps.jsonl", "w", encoding="utf-8") as f:
        for n, s in enumerate(steps):
            per = {str(i): {"0.5": {"loss": grid(4.0 - n, 3.0 - n, 2.0, 1.0)}} for i in range(pictures)}
            f.write(json.dumps({"step": s, "epoch": float(n), "maps": per}) + "\n")
    return d


def status(sweeps=None, step=100):
    st = {"step": step, "total": 100, "face_scores": [],
          "probes": [{"step": 0, "face": 40.0, "body": 30.0, "combined": 35.0, "combined_ema": 35.0,
                      "face_ema": 40.0, "body_ema": 30.0},
                     {"step": 100, "face": 50.0, "body": 30.0, "combined": 40.0, "combined_ema": 40.0,
                      "face_ema": 50.0, "body_ema": 30.0}]}
    if sweeps is not None:
        st["region_sweep"] = {"history": sweeps}
    return st


# ---------------------------------------------------------------- sweeps


def test_sweeps_come_from_the_status_deduped_by_step_and_rounded():
    rows = [sweep_row(0, 0.0, 0.0), sweep_row(100, 0.123456789, 0.1), sweep_row(100, 0.2, 0.15), {"epoch": 3},
            {"step": 200, "train": {"face": "x"}}]
    out = history.sweep_series({"region_sweep": {"history": rows}})
    assert [e["step"] for e in out] == [0, 100]
    assert out[1]["train"]["face"] == 0.2 and out[1]["held"] == {"face": 0.15, "subject": 0.075}
    out = history.sweep_series({"region_sweep": {"history": [sweep_row(5, 0.123456789)]}})
    assert out[0]["train"]["face"] == 0.12346 and out[0]["held"] == {}
    assert history.sweep_series({}) == [] and history.sweep_series({"region_sweep": {}}, None) == []


def test_sweeps_fall_back_to_the_folders_jsonl(tmp_path):
    with open(tmp_path / "region_sweep.jsonl", "w", encoding="utf-8") as f:
        for r in (sweep_row(0, 0.0), sweep_row(50, 0.1), sweep_row(50, 0.3), sweep_row(80, 0.4)):
            f.write(json.dumps(r) + "\n")
        f.write("{half a line")
    out = history.sweep_series({"region_sweep": {"history": []}}, tmp_path)
    assert [(e["step"], e["train"]["face"]) for e in out] == [(0, 0.0), (50, 0.3), (80, 0.4)]
    rec = history.build_record({"job_name": "j", "output_dir": str(tmp_path.parent), "steps": 9}, "", "j-1", 1.0,
                               status(), "finished", folder=tmp_path)
    assert [e["step"] for e in rec["sweeps"]] == [0, 50, 80]
    assert history.build_record({"job_name": "j"}, "", "j-1", 1.0, status(), "finished", folder=None)["sweeps"] == []


def test_save_run_keeps_the_sweeps(tmp_path):
    cfg = TrainConfig(job_name="s", output_dir=str(tmp_path / "out"))
    history.save_run(cfg, "s-1", 1.0, status([sweep_row(0, 0.0), sweep_row(100, 0.4)]), "finished", tmp_path / "h")
    (run,) = history.list_runs(tmp_path / "h")
    assert [e["step"] for e in run["sweeps"]] == [0, 100]


# ---------------------------------------------------------------- maps


def test_maps_are_copied_on_the_final_save_not_while_running(tmp_path):
    job = tmp_path / "out" / "m"
    write_maps(job)
    cfg = TrainConfig(job_name="m", output_dir=str(tmp_path / "out"))
    root = tmp_path / "h"
    history.save_run(cfg, "m-1", 1.0, status(), "running", root)
    assert not (root / "m-1" / "maps").exists() and "maps" not in history.list_runs(root)[0]
    history.save_run(cfg, "m-1", 1.0, status(), "finished", root)
    kept = root / "m-1" / "maps"
    assert sorted(p.name for p in kept.iterdir()) == ["maps.jsonl", "probe_0.jpg", "probes.json"]
    assert history.list_runs(root)[0]["maps"] is True
    # a later save keeps the mark, and an unchanged copy is not written again
    before = (kept / "maps.jsonl").stat().st_mtime_ns
    history.save_run(cfg, "m-1", 1.0, status(), "running", root)
    assert history.list_runs(root)[0]["maps"] is True
    history.save_run(cfg, "m-1", 1.0, status(), "stopped", root)
    assert (kept / "maps.jsonl").stat().st_mtime_ns == before


def test_a_job_without_maps_keeps_none(tmp_path):
    cfg = TrainConfig(job_name="n", output_dir=str(tmp_path / "out"))
    history.save_run(cfg, "n-1", 1.0, status(), "finished", tmp_path / "h")
    assert "maps" not in history.list_runs(tmp_path / "h")[0] and not (tmp_path / "h" / "n-1").exists()


def fleet_folder(tmp_path):
    folder = tmp_path / "dl"
    folder.mkdir()
    (folder / "status.json").write_text(json.dumps(status([sweep_row(0, 0.0), sweep_row(100, 0.5, 0.3)])),
                                        encoding="utf-8")
    write_maps(folder)
    return folder


def fleet_record(tmp_path, folder):
    rec = {"key": "9-job-1", "instance": 9, "job": "job", "gpu": "RTX 5090", "state": "done", "started": 1100.0,
           "ended": 2000.0, "modified": 1999, "step": 100, "total": 100, "config": {"job_name": "job"},
           "probes": [], "folder": str(folder) if folder else "", "downloaded": []}
    fleet = tmp_path / "fleet"
    fleet.mkdir(exist_ok=True)
    (fleet / f"{rec['key']}.json").write_text(json.dumps(rec), encoding="utf-8")
    return fleet


def test_fleet_import_keeps_maps_and_sweeps_from_the_download(tmp_path):
    folder = fleet_folder(tmp_path)
    fleet, root = fleet_record(tmp_path, folder), tmp_path / "h"
    assert history.import_fleet_runs(fleet, root) == 1
    (run,) = history.list_runs(root)
    assert run["id"] == "vast-9-job-1" and run["maps"] is True
    assert [e["step"] for e in run["sweeps"]] == [0, 100] and run["sweeps"][1]["held"]["face"] == 0.3
    assert (root / run["id"] / "maps" / "maps.jsonl").is_file()
    # the download folder is gone: the kept copy and sweeps stay with the record
    assert (root / run["id"] / "maps" / "probe_0.jpg").is_file()


def test_fleet_run_without_a_download_has_no_sweeps_or_maps(tmp_path):
    fleet = fleet_record(tmp_path, None)
    root = tmp_path / "h"
    st = {"probes": [{"step": 0, "combined_ema": 1.0, "face_ema": 1.0}]}
    rec = json.loads((fleet / "9-job-1.json").read_text(encoding="utf-8"))
    rec["probes"] = st["probes"]
    (fleet / "9-job-1.json").write_text(json.dumps(rec), encoding="utf-8")
    assert history.import_fleet_runs(fleet, root) == 1
    (run,) = history.list_runs(root)
    assert run["sweeps"] == [] and "maps" not in run


def test_fleet_import_brings_sweeps_to_a_run_imported_by_3_46(tmp_path):
    """A finished vast.ai run imported under 3.46.0 (no sweeps, no maps, the old stamp) gets its sweeps and maps from
    the downloaded folder the next time the import runs, though its Fleet record never changes again. The folder
    holds the sweeps the way the Fleet page reads them: a status.json without a sweep history, plus
    region_sweep.jsonl."""
    folder = tmp_path / "dl"
    folder.mkdir()
    (folder / "status.json").write_text(json.dumps(status()), encoding="utf-8")
    with open(folder / "region_sweep.jsonl", "w", encoding="utf-8") as f:
        for row in (sweep_row(0, 0.0, 0.0), sweep_row(100, 0.4, 0.25), sweep_row(200, 0.6, 0.3)):
            f.write(json.dumps(row) + "\n")
    write_maps(folder)
    fleet, root = fleet_record(tmp_path, folder), tmp_path / "h"
    assert history.import_fleet_runs(fleet, root) == 1
    # turn the record into what 3.46.0 wrote: no sweeps or maps field, no kept maps, the stamp without the version
    path = root / "vast-9-job-1.json"
    old = json.loads(path.read_text(encoding="utf-8"))
    old.pop("sweeps")
    old.pop("maps")
    fr = json.loads((fleet / "9-job-1.json").read_text(encoding="utf-8"))
    old["source"]["stamp"] = [fr["modified"], fr["step"], fr["state"], fr["ended"],
                              (folder / "status.json").stat().st_mtime_ns, []]
    old["label"] = "kept label"
    path.write_text(json.dumps(old), encoding="utf-8")
    import shutil

    shutil.rmtree(root / "vast-9-job-1" / "maps")
    assert history.import_fleet_runs(fleet, root) == 1
    (run,) = history.list_runs(root)
    assert [e["step"] for e in run["sweeps"]] == [0, 100, 200] and run["sweeps"][2]["held"]["face"] == 0.3
    assert run["maps"] is True and (root / run["id"] / "maps" / "maps.jsonl").is_file()
    assert run["label"] == "kept label"
    # imported once more, then left alone again while the Fleet record stays the same
    assert history.import_fleet_runs(fleet, root) == 0


def test_import_jobs_adds_sweeps_and_maps_to_an_older_record(tmp_path):
    out, root = tmp_path / "out", tmp_path / "h"
    job = out / "old"
    job.mkdir(parents=True)
    st = status([sweep_row(0, 0.0), sweep_row(100, 0.4)])
    (job / "status.json").write_text(json.dumps(st), encoding="utf-8")
    (job / "config.yaml").write_text("job_name: old\nsteps: 100\n", encoding="utf-8")
    write_maps(job)
    # a record saved before sweeps and maps were kept
    rec = history.build_record({"job_name": "old", "output_dir": str(out)}, "", "old-1", 1.0, st, "finished",
                               folder=None)
    del rec["sweeps"]
    rec["probe_images"] = []
    history._write(root / "old-1.json", rec)
    assert history.import_jobs(out, root) == 1
    (run,) = history.list_runs(root)
    assert [e["step"] for e in run["sweeps"]] == [0, 100] and run["maps"] is True
    assert (root / "old-1" / "maps" / "maps.jsonl").is_file()
    assert history.import_jobs(out, root) == 0  # nothing more to add


def test_delete_runs_removes_the_maps_copy(tmp_path):
    folder = fleet_folder(tmp_path)
    root = tmp_path / "h"
    history.import_fleet_runs(fleet_record(tmp_path, folder), root)
    assert (root / "vast-9-job-1" / "maps").is_dir()
    assert history.delete_runs(["vast-9-job-1"], root) == 1
    assert not (root / "vast-9-job-1").exists()


def test_keep_maps_skips_identical_files_and_picks_up_changes(tmp_path):
    job, root = tmp_path / "j", tmp_path / "h"
    write_maps(job)
    assert history.keep_maps("r", job, root) and not history.keep_maps("r2", tmp_path / "none", root)
    dest = root / "r" / "maps" / "maps.jsonl"
    marker = dest.stat().st_mtime_ns
    history.keep_maps("r", job, root)
    assert dest.stat().st_mtime_ns == marker
    src = job / "training_maps" / "maps.jsonl"
    with open(src, "a", encoding="utf-8") as f:
        f.write(json.dumps({"step": 200, "epoch": 2.0, "maps": {}}) + "\n")
    os.utime(src, ns=(marker + 10**9, marker + 10**9))
    history.keep_maps("r", job, root)
    assert dest.read_text(encoding="utf-8").count("\n") == 3


# ---------------------------------------------------------------- the WebUI's side

psutil = pytest.importorskip("psutil")

import webui  # noqa: E402


def test_history_maps_dir_prefers_the_kept_copy(tmp_path, monkeypatch):
    root = tmp_path / "h"
    monkeypatch.setattr(history, "HISTORY_DIR", root)
    job = tmp_path / "job"
    write_maps(job)
    (job / "status.json").write_text(json.dumps({"run_id": "r-1"}), encoding="utf-8")
    rec = {"id": "r-1", "job": "job", "job_dir": str(job), "started": 1.0}
    assert webui.history_maps_dir(rec, [rec]) == job / "training_maps"  # still running: the job folder
    history.keep_maps("r-1", job, root)
    assert webui.history_maps_dir(rec, [rec]) == root / "r-1" / "maps"  # kept: the copy
    (job / "status.json").write_text(json.dumps({"run_id": "later"}), encoding="utf-8")
    assert webui.history_maps_dir(rec, [rec]) == root / "r-1" / "maps"
    assert webui.history_maps_dir({"id": "r-2", "job": "job", "job_dir": str(job), "started": 1.0}, []) is None


def test_read_maps_dir_reads_a_kept_copy(tmp_path):
    job = tmp_path / "job"
    write_maps(job, steps=(0, 100, 200))
    history.keep_maps("r", job, tmp_path / "h")
    data = webui.read_maps_dir(tmp_path / "h" / "r" / "maps")
    assert [c["step"] for c in data["captures"]] == [0, 100, 200] and data["dir"].endswith("maps")
    assert data["probes"][0]["image"] == "probe_0.jpg" and webui.map_picture(data, data["probes"][0]) is not None
    assert webui.read_maps_dir(tmp_path / "nothing") is None


def test_sweep_lines_one_colour_per_run_held_out_dashed():
    runs = [{"id": "a", "job": "a", "label": "A", "started": 1, "sweeps": [
                {"step": 0, "train": {"face": 0.0}, "held": {"face": 0.0}},
                {"step": 100, "train": {"face": 0.3333}, "held": {"face": 0.2}}]},
            {"id": "b", "job": "b", "label": "B", "started": 2, "sweeps": []},
            {"id": "c", "job": "c", "label": "C", "started": 3, "sweeps": [{"step": 50, "train": {"face": 0.5}}]}]
    lines = webui.history_sweep_lines(runs, "face")
    assert [ln["id"] for ln in lines] == ["A · training", "A · held-out", "C · training"]
    assert lines[0]["data"] == [[0, 0.0], [100, 33.3]] and lines[1]["lineStyle"]["type"] == "dashed"
    assert lines[0]["itemStyle"]["color"] == webui.HISTORY_COLORS[0] == lines[1]["itemStyle"]["color"]
    assert lines[2]["itemStyle"]["color"] == webui.HISTORY_COLORS[2]  # the same colour as on the other charts
    assert [ln["id"] for ln in webui.history_sweep_lines(runs, "face", ("held",))] == ["A · held-out"]
    assert webui.history_sweep_lines(runs, "hair") == []


def test_maps_capture_index_is_the_latest_at_or_before_the_step():
    caps = [{"step": 100}, {"step": 200}, {"step": 300}]
    assert webui.maps_capture_index(caps, 250) == 1 and webui.maps_capture_index(caps, 300) == 2
    assert webui.maps_capture_index(caps, 50) == 0 and webui.maps_capture_index([], 50) is None


def test_maps_compare_picks_each_runs_capture_and_one_shared_scale(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    write_maps(a, steps=(0, 100, 200), pictures=2)
    write_maps(b, steps=(0, 150), pictures=1)
    entries = [("A", webui.read_maps_dir(a / "training_maps")), ("B", webui.read_maps_dir(b / "training_maps"))]
    assert webui.maps_steps(entries) == [0, 100, 150, 200]
    items, top = webui.maps_compare(entries, 0, "loss", "all", "fixed", 100)
    assert [(i["label"], i["capture"]["step"]) for i in items] == [("A", 100), ("B", 0)]
    assert items[0]["note"] == "" and items[1]["note"] == "its latest: step 0" and top > 0
    assert items[0]["a"].shape == (2, 2)
    items, top = webui.maps_compare(entries, 1, "learned", "all", "fixed", 100)  # B has no second picture
    assert items[1] == {"label": "B", "missing": "has no picture 2"} and top == 0.5  # fixed scale
    items, _ = webui.maps_compare(entries, 0, "loss", "all", "fixed", -5)  # before any capture: the first
    assert items[0]["note"] == "its first: step 0" 
    # one scale for every picture: the larger of the runs' own
    big = webui.read_maps_dir(a / "training_maps")
    big["captures"][0]["maps"]["0"]["0.5"]["loss"]["v"] = [40.0, 30.0, 20.0, 10.0]
    _, top_shared = webui.maps_compare([("A", big), ("B", entries[1][1])], 0, "loss", "all", "amount", 0)
    _, top_b = webui.maps_compare([("B", entries[1][1])], 0, "loss", "all", "amount", 0)
    assert top_shared > top_b
