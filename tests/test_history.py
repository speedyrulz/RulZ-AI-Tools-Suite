"""Run history (output/_history): the series kept from a run's scores, saving while it trains, labels, removal, importing
earlier job folders, and the Previous Runs table and overlay series."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import webui  # noqa: E402
from rulztrainer import history  # noqa: E402
from rulztrainer.config import TrainConfig  # noqa: E402

SCORES = [{"step": 300, "face_norm": 10.0, "body_norm": 0.0, "combined_norm": 5.0, "percent": 20.0},
          {"step": 300, "face_norm": 20.0, "body_norm": None, "combined_norm": 15.0, "percent": 30.0},
          {"step": 600, "face_norm": 40.0, "body_norm": 30.0, "combined_norm": 35.0, "percent": 50.0}]
PROBES = [{"step": 0, "face_norm_ema": 0.0, "body_norm_ema": 0.0, "combined_ema": 0.0},
          {"step": 25, "face_norm_ema": 12.34, "body_norm_ema": 5.0, "combined_ema": 9.0},
          {"step": 50, "face_ema": 60.0, "body_ema": 55.0, "combined_ema": 58.0}]  # an older record: raw only


def _status(**kw):
    return {"step": 600, "total": 2000, "face_scores": SCORES, "probes": PROBES, **kw}


def test_series_are_means_per_step_and_fall_back_to_raw_scores():
    ident = history.identity_series(SCORES)
    assert ident["normalized"] and ident["face"] == [[300, 15.0], [600, 40.0]] and ident["body"] == [[300, 0.0], [600, 30.0]]
    raw = history.identity_series([{"step": 1, "percent": 30.0, "body_percent": 50.0}])
    assert not raw["normalized"] and raw["face"] == [[1, 30.0]] and raw["body"] == [[1, 50.0]]
    probes = history.probe_series(PROBES)
    assert probes["face"] == [[0, 0.0], [25, 12.3], [50, 60.0]] and probes["combined"][-1] == [50, 58.0]
    assert history.probe_series([]) == {"face": [], "body": [], "combined": []}


def test_a_run_is_saved_while_it_trains_keeps_its_label_and_needs_data(tmp_path):
    cfg = TrainConfig(job_name="my lora", lora_rank=32, steps=2000)
    rid = history.new_run_id(cfg.job_name, 1_700_000_000)
    assert rid.startswith("my_lora-2023")
    assert history.save_run(cfg, rid, 1_700_000_000, {"step": 5, "face_scores": [], "probes": []}, root=tmp_path) is None
    path = history.save_run(cfg, rid, 1_700_000_000, _status(), root=tmp_path)
    assert path == tmp_path / f"{rid}.json"
    assert history.set_label(rid, "  rank 32 test ", root=tmp_path) and not history.set_label("nope", "x", root=tmp_path)
    history.save_run(cfg, rid, 1_700_000_000, _status(step=900), "finished", root=tmp_path)  # a later save
    (run,) = history.list_runs(tmp_path)
    assert run["label"] == "rank 32 test" and run["state"] == "finished" and run["step"] == 900
    assert run["config"]["lora_rank"] == 32 and run["total"] == 2000 and run["arch"] == "moss_image"
    assert history.delete_runs([rid, "../x"], tmp_path) == 1 and history.list_runs(tmp_path) == []


def test_signature_changes_when_the_history_does(tmp_path):
    cfg = TrainConfig(job_name="a")
    empty = history.signature(tmp_path)
    history.save_run(cfg, "a-1", 1.0, _status(), root=tmp_path)
    first = history.signature(tmp_path)
    assert first != empty
    history.set_label("a-1", "x", tmp_path)
    assert history.signature(tmp_path) != first


def _job(output: Path, name: str, **status):
    job = output / "group" / name
    job.mkdir(parents=True)
    (job / "config.yaml").write_text(f"job_name: {name}\noutput_dir: {output / 'group'}\nsteps: 2000\n"
                                     "network_type: lokr\n", encoding="utf-8")
    (job / "status.json").write_text(json.dumps({"step": 2000, "total": 2000, "done": True, **status}),
                                     encoding="utf-8")
    return job


def test_earlier_job_folders_are_imported_once(tmp_path):
    out, root = tmp_path / "output", tmp_path / "output" / "_history"
    _job(out, "scored", face_scores=SCORES, probes=PROBES)
    _job(out, "unscored")
    _job(out, "_crops_like_job_without_scores")
    assert history.import_jobs(out, root) == 1
    assert history.import_jobs(out, root) == 0  # the job already has a run
    (run,) = history.list_runs(root)
    assert run["job"] == "scored" and run["state"] == "finished" and run["id"].endswith("-imported")
    assert run["config"]["network_type"] == "lokr" and run["probes"]["combined"]


def test_the_table_and_the_overlay_series():
    runs = [{"id": "a", "job": "krea", "started": 1_700_000_000, "label": "", "state": "finished", "step": 2000,
             "total": 2000, "config": {"network_type": "lokr", "lokr_factor": 8, "learning_rate": 1e-4},
             "identity": {"face": [[300, 15.0]], "body": [[300, 5.0]]},
             "probes": {"face": [[25, 40.0], [50, 60.0]], "body": [[25, 10.0]], "combined": [[25, 20.0], [50, 70.0]]}},
            {"id": "b", "job": "moss", "started": 1_700_100_000, "label": "rank 16", "state": "running", "step": 10,
             "total": 100, "config": {"lora_rank": 16, "finetune": False},
             "identity": {"face": [], "body": []}, "probes": {"face": [], "body": [], "combined": []}}]
    rows = webui.history_rows(runs)
    assert rows[0]["adapter"] == "LoKr f8" and rows[0]["steps"] == "2,000 / 2,000" and rows[0]["lr"] == "1.0e-04"
    assert (rows[0]["face"], rows[0]["body"], rows[0]["combined"], rows[0]["best"]) == ("60.0%", "10.0%", "70.0%", "70.0%")
    assert rows[1]["run"] == "rank 16" and rows[1]["adapter"] == "LoRA r16" and rows[1]["best"] == "-"
    probes = webui.history_series(runs, "probes", ["face", "body", "combined"])
    assert [s["name"].rsplit(" - ", 1)[1] for s in probes] == ["face", "body", "combined"]  # run b has no points
    assert probes[1]["lineStyle"]["type"] == "dashed" and probes[2]["lineStyle"]["width"] == 3
    assert all(s["itemStyle"]["color"] == webui.HISTORY_COLORS[0] for s in probes)
    both = webui.history_series(runs, "identity", ["face"])
    assert len(both) == 1 and both[0]["data"] == [[300, 15.0]] and both[0]["showSymbol"]
    assert webui.history_series(runs, "probes", []) == []


def test_a_race_keeps_each_configuration_s_probes_apart(tmp_path):
    def rec(name, step, comb):
        prefix = f"{name}_" if name else ""
        return {"step": step, "face_norm_ema": comb - 5, "body_norm_ema": comb + 5, "combined_ema": comb,
                "images": [f"out/probes/{prefix}step{step:06d}_face_0.png"]}

    status = {"race": {"names": ["lora16", "lokr8"]},
              "probes": [rec("lora16", 50, 10.0), rec("lokr8", 50, 14.0), rec("lokr8", 100, 20.0), rec("", 150, 25.0)]}
    series = history.probe_series(status["probes"], history.race_names(status))
    assert series["combined"] == [[50, 10.0], [50, 14.0], [100, 20.0], [150, 25.0]]  # all together, as before
    assert list(series["racers"]) == ["lora16", "lokr8", "winner"]
    assert series["racers"]["lokr8"]["combined"] == [[50, 14.0], [100, 20.0]]
    assert series["racers"]["winner"]["face"] == [[150, 20.0]]
    run = {"id": "r", "job": "race", "label": "my race", "probes": series, "identity": {}, "config": {"race": True}}
    lines = webui.history_series([run], "probes", ["combined"])
    assert [s["name"] for s in lines] == ["my race lora16 - combined", "my race lokr8 - combined",
                                          "my race winner - combined"]
    assert len({s["itemStyle"]["color"] for s in lines}) == 3  # one colour per configuration
    # a race saved before the split gets it from its job folder on import
    job = tmp_path / "output" / "race"
    job.mkdir(parents=True)
    (job / "status.json").write_text(json.dumps(status))
    (job / "config.yaml").write_text("job_name: race\nrace: true\n")
    root = tmp_path / "history"
    old = {**run, "probes": {k: v for k, v in series.items() if k != "racers"}}
    history._write(root / "r.json", old)
    assert history.import_jobs(tmp_path / "output", root) == 1
    upgraded = history.list_runs(root)
    assert len(upgraded) == 1 and list(upgraded[0]["probes"]["racers"]) == ["lora16", "lokr8", "winner"]
    assert upgraded[0]["label"] == "my race"
