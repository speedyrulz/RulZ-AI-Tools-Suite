"""rulztrainer/runreport.py: a run's summary picture and PDF, made from a job folder's own files."""
import json
import sys
from pathlib import Path

import pytest
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("matplotlib")

from rulztrainer import runreport as R  # noqa: E402
from rulztrainer.config import TrainConfig  # noqa: E402


def _png(path: Path, size, color):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def _job(tmp_path: Path, probes: bool = True, maps: bool = True, dataset: bool = True, racer: str = "") -> Path:
    """A small finished run: install/output/job with the files a real run leaves."""
    root = tmp_path / "install"
    job = root / "output" / "demo run"
    job.mkdir(parents=True)
    cfg = TrainConfig(job_name="demo run", datasets=[{"path": "datasets/people"}], trigger_word="tok", steps=40,
                      learning_rate=5e-05, lora_rank=8, probe_every=10, probe_seeds=3, region_watch=True).to_dict()
    (job / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    records = []
    prefix = racer + "_" if racer else ""
    for i, step in enumerate((0, 10, 20, 30, 40)):
        body = [20.0 + 5 * i, None if step >= 30 else 25.0 + 5 * i, 30.0 + i]
        images = [f"output/demo run/probes/{prefix}step{step:06d}_{kind}_{k}.png" for kind in ("face", "body")
                  for k in range(3)]
        records.append({"step": step, "face": 50.0 + i, "body": 25.0, "combined": 30.0 + 4 * i, "face_ema": 50.0 + i,
                        "body_ema": 25.0, "combined_ema": 28.0 + 4 * i - (9 if step == 40 else 0), "best": 40.0,
                        "best_step": 30, "face_norm": 10.0 * i, "body_norm": 25.0, "face_norm_ema": 10.0 * i,
                        "body_norm_ema": 25.0, "face_norm_seeds": [10.0 * i, 12.0 * i + 1, 8.0 * i],
                        "body_norm_seeds": body, "body_seeds": [None if v is None else 80.0 + v / 10 for v in body],
                        "copy": 0.4 + 0.07 * i, "body_probes": 3, "body_missing": sum(v is None for v in body),
                        "images": images})
        if racer:
            # the other configuration, probed after the winner at the same own steps each round until it left for
            # drift after round 5: the file's last record is its, with its own best
            records.append({"step": step, "face": 40.0, "body": 20.0, "combined": 22.0, "combined_ema": 22.0,
                            "best": 22.0, "best_step": 10, "body_norm_seeds": [1.0, 2.0, 3.0],
                            "images": [p.replace(prefix, "other_") for p in images]})
        if probes:
            for k in range(3):
                _png(job / "probes" / f"{racer + '_' if racer else ''}step{step:06d}_face_{k}.png", (64, 64), (40 * k, 90, 120))
                _png(job / "probes" / f"{racer + '_' if racer else ''}step{step:06d}_body_{k}.png", (64, 96), (120, 40 * k, 90))
    (job / "likeness.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    status = {"stage": "finished", "step": 40, "total": 40, "elapsed": 135.0, "avg_loss": 0.21, "done": True,
              "losses": [0.3 - 0.002 * i for i in range(40)], "saved": ["output/demo run/demo run.safetensors"],
              "run_id": "demo_run-20260102-030405", "peak_mem": {"cuda:0": 9.5},
              "near_duplicates": {"count": 2}}
    if racer:  # the keys RaceSchedule.state() writes (the trainer's _race_status)
        status["race"] = {"names": [racer, "other"], "winner": racer, "last": racer, "alive": [racer],
                          "dropped": {"other": 5}, "drifted": ["other"], "manual": [], "stale": [], "finished": [],
                          "best": {racer: 40.0, "other": 22.0}, "best_step": {racer: 30, "other": 10}}
    (job / "status.json").write_text(json.dumps(status), encoding="utf-8")
    (job / "train.log").write_text(
        "10:00:00 INFO step 10/40 loss 0.25 avg 0.26 lr 5.00e-05 grad 0.02 0.4s/it\n"
        "10:00:05 WARNING Near-duplicate check: 2 pair(s)\n10:00:06 WARNING Near-duplicate check: 2 pair(s)\n"
        "10:00:09 ERROR something went sideways\n", encoding="utf-8")
    (job / "region_sweep.jsonl").write_text("\n".join(json.dumps(r) for r in (
        {"step": 0, "first": True},
        {"step": 20, "train": {"subject": 0.1, "face": 0.2}, "held": {"subject": 0.08, "face": 0.1}},
        {"step": 40, "train": {"subject": 0.2, "face": 0.3}, "held": {"subject": 0.05, "face": 0.1}})), encoding="utf-8")
    if maps:
        _png(job / "training_maps" / "probe_0.jpg", (48, 72), (90, 90, 160))
        grid = {"w": 3, "h": 4, "v": [float(i) for i in range(12)]}
        (job / "training_maps" / "probes.json").write_text(json.dumps(
            {"probes": [{"id": "0", "image": "probe_0.jpg", "size": [48, 72]}]}), encoding="utf-8")
        (job / "training_maps" / "maps.jsonl").write_text("\n".join(json.dumps(
            {"step": s, "maps": {"0": {"0.25": {"loss": grid}, "0.5": {"loss": grid, "effect": grid, "attn": grid}}}})
            for s in (0, 20, 40)), encoding="utf-8")
    if dataset:
        for i in range(5):
            _png(root / "datasets" / "people" / f"p{i}.jpg", (60, 80), (30 * i, 100, 100))
            (root / "datasets" / "people" / f"p{i}.txt").write_text(f"tok, a person number {i}", encoding="utf-8")
    return job


def test_a_run_is_read_from_its_folder(tmp_path):
    run = R.load(_job(tmp_path))
    assert run.job == "demo run" and run.started == "2026-01-02 03:04"
    assert len(run.probes) == 5 and len(run.dataset) == 5 and len(run.sweeps) == 2  # (the first sweep is the zero)
    assert sorted(run.images[("", 20, "body")]) == [0, 1, 2]
    assert run.lr == [(10, 5e-05)]
    assert [n[0] for n in run.notes] == ["WARNING", "ERROR"]  # a repeated line once
    assert R.best(run) == (40.0, 30)
    assert R.steps_shown(run, 3) == [0, 30, 40] and R.steps_shown(run, 9) == [0, 10, 20, 30, 40]
    assert R.seed_scores(R.record_at(run, 30), "body") == [35.0, None, 33.0]
    kinds = [k for k, _ in R.verdict(run)]
    assert kinds.count("warn") >= 2  # the lost framing and the copy score, besides the error line's absence
    text = " ".join(t for _, t in R.verdict(run))
    assert "step 30 of 40" in text and "1 of 3 renders without a body from step 30" in text and "Held-out" in text
    changed = dict(R.changed_settings(run.cfg))
    assert changed["lora_rank"] == "8" and changed["learning_rate"] == "5e-05" and "batch_size" not in changed
    rows = dict(R.settings_rows(run))
    assert rows["Dataset"] == "people (5 images)" and rows["Network"].startswith("LoRA, rank 8") and \
        "region watch" in rows["Also on"]


def test_region_data_and_reports(tmp_path):
    job = _job(tmp_path, maps=False)
    (job / "region_sweep.jsonl").write_text("\n".join(json.dumps(row) for row in (
        {"step": 20, "train": {"subject": 0.1, "face": 0.2}, "held": {"subject": 0.08, "face": 0.1}},
        {"step": 40, "train": {"subject": 0.2, "face": 0.3}, "held": {"subject": -0.01, "face": 0.1}})), encoding="utf-8")
    picture = "photos/person.jpg"
    (job / "region_watch.json").write_text(json.dumps({
        "epoch": 2, "frozen": True, "history": [{"epoch": 2, "step": 40,
        "learned": {"face": 0.4}, "contrast": {}}],
        "pictures": {picture: {"seen": 2, "contrast_log": {"background": 0.0, "clothes": 0.69314718056}}}
    }), encoding="utf-8")
    (job / "region_sweep.json").write_text(json.dumps({"latest": {picture: {
        "contrast": {"background": 0.8, "clothes": 1.2}}}}), encoding="utf-8")
    run = R.load(job)
    assert R.sweep_series(run.sweeps, "face", "train") == [(20, 20.0), (40, 30.0)]
    assert R.sweep_gaps(run.sweeps)["face"] == [(20, 10.0), (40, 20.0)]
    assert R.growing_gap(run.sweeps)[0] == "subject"
    row = run.watch["rows"][0]
    assert row["image"] == "person.jpg" and row["background"] == 1.0
    assert row["clothes"] == pytest.approx(2.0) and row["sweep_background"] == 0.8
    assert run.watch["summary"]["learned"]["face"] == 0.4
    assert any("memorised" in text for _, text in R.verdict(run))
    assert all(path.is_file() for path in R.generate(job).values())


def test_region_sweeps_without_held_out(tmp_path):
    job = _job(tmp_path, maps=False)
    (job / "region_sweep.jsonl").write_text(json.dumps({"step": 10, "train": {"subject": 0.2}}), encoding="utf-8")
    run = R.load(job)
    assert R.sweep_gaps(run.sweeps) == {} and R.growing_gap(run.sweeps) is None
    assert all(path.is_file() for path in R.generate(job).values())


def test_both_files_are_written_and_a_bare_run_still_gets_them(tmp_path):
    job = _job(tmp_path)
    done = R.generate(job)
    assert done["image"] == job / "report" / "demo_run_report.png" and done["pdf"].name == "demo_run_report.pdf"
    with Image.open(done["image"]) as im:
        assert im.size[0] >= 2000 and im.size[1] > im.size[0]  # a tall page
    data = done["pdf"].read_bytes()
    assert data.startswith(b"%PDF") and data.count(b"/Type /Page\n") + data.count(b"/Type /Page ") >= 6
    # nothing but a folder with a status file: no probes, no maps, no dataset, no config
    bare = tmp_path / "bare" / "output" / "old"
    bare.mkdir(parents=True)
    (bare / "status.json").write_text(json.dumps({"step": 3, "total": 9}), encoding="utf-8")
    done = R.generate(bare, out_dir=tmp_path / "elsewhere", pdf_file=False)
    assert list(done) == ["image"] and done["image"].parent == tmp_path / "elsewhere" and done["image"].stat().st_size > 20000
    with pytest.raises(R.ReportError):
        R.generate(tmp_path / "no such run")


def test_a_race_shows_its_winner_and_the_command_line_writes_one_file(tmp_path, capsys):
    job = _job(tmp_path, maps=False, dataset=False, racer="seed43")
    run = R.load(job)
    assert R.racers(run) == ["seed43"] and R.steps_shown(run, 3, "seed43") == [0, 30, 40]
    assert any("Race won by seed43" in t for _, t in R.verdict(run))
    # the trainer writes "winner"; the file's last record is the other configuration's, with its own best
    assert R.race_winner(run) == "seed43" and run.probes[-1]["best"] == 22.0
    assert R.best(run) == (40.0, 30)
    # both configurations were probed at step 30 of their own: each record is found by its racer
    assert R.seed_scores(R.record_at(run, 30, "seed43"), "body") == [35.0, None, 33.0]
    assert R.record_at(run, 30, "other")["best"] == 22.0 and R.record_at(run, 30) == {}
    assert R.is_best(run, "seed43", 30) and not R.is_best(run, "other", 30) and not R.is_best(run, "seed43", 20)
    assert R.generate(job, out_dir=tmp_path / "picture", pdf_file=False)["image"].is_file()
    assert R.main([str(job), "--pdf"]) == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1 and out[0].endswith("demo_run_report.pdf") and Path(out[0]).is_file()
    assert not (job / "report" / "demo_run_report.png").exists()
    assert R.main([str(tmp_path / "missing")]) == 1 and "no report" in capsys.readouterr().err


def test_the_best_checkpoint_is_the_race_winner_s():
    def rec(name, step, best, best_step):
        prefix = f"{name}_" if name else ""
        return {"step": step, "best": best, "best_step": best_step,
                "images": [f"output/job/probes/{prefix}step{step:06d}_face_0.png"]}

    probes = [rec("a", 10, 30.0, 10), rec("b", 10, 50.0, 10), rec("a", 20, 35.0, 20), rec("", 30, 45.0, 30)]
    race = {"names": ["a", "b"], "winner": "a", "last": "a"}
    run = R.Run(Path("job"), Path("."), status={"race": race}, probes=probes)
    # a won and trained on after the race: its probes then carry no name and hold its latest best
    assert R.best_racers(run) == {"a", ""} and R.best(run) == (45.0, 30) and R.is_best(run, "", 30)
    # a race decided at the end that b won on an earlier best, while a was the last one racing
    run = R.Run(Path("job"), Path("."), status={"race": {**race, "winner": "b"}}, probes=probes)
    assert R.best_racers(run) == {"b"} and R.best(run) == (50.0, 10) and not R.is_best(run, "", 30)
    # no winner decided yet: the latest probe's configuration, as before
    run = R.Run(Path("job"), Path("."), status={"race": {**race, "winner": None}}, probes=probes[:3])
    assert R.best_racers(run) == {"a"} and R.best(run) == (35.0, 20)
    # the winner's records fell out of status.json's last probes: the race's own best
    run = R.Run(Path("job"), Path("."), probes=probes[1:2],
                status={"race": {**race, "winner": "a", "best": {"a": 44.0}, "best_step": {"a": 25}}})
    assert R.best(run) == (44.0, 25)
    # outside a race every record is the run's own
    run = R.Run(Path("job"), Path("."), probes=[rec("", 10, 12.0, 10), rec("", 20, 15.0, 20)])
    assert R.best_racers(run) == {""} and R.best(run) == (15.0, 20) and R.record_at(run, 10)["best"] == 12.0


def test_the_setting_is_off_by_default_and_round_trips():
    assert TrainConfig().run_report is False
    assert TrainConfig.from_dict(TrainConfig(run_report=True).to_dict()).run_report is True


def _both(job):
    done = R.generate(job)
    assert done["image"].is_file() and done["pdf"].is_file()
    return done


@pytest.mark.parametrize("name", ["train_list", "history_dict", "pictures_list", "latest_list", "sweep_list"])
def test_odd_region_files_never_stop_the_report(tmp_path, name):
    job = _job(tmp_path, maps=False)
    picture = "photos/person.jpg"
    watch = {"epoch": 2, "frozen": True, "history": [{"epoch": 2, "step": 40, "learned": {"face": 0.4}, "contrast": {}}],
             "pictures": {picture: {"seen": 2, "contrast_log": {"background": 0.0}}}}
    sweep = {"latest": {picture: {"contrast": {"background": 0.8}}}}
    rows = [{"step": 20, "train": {"subject": 0.1}, "held": {"subject": 0.08}},
            {"step": 40, "train": {"subject": 0.2}, "held": {"subject": 0.05}}]
    if name == "train_list":
        rows[1]["train"] = [0.2]
    elif name == "history_dict":
        watch["history"] = {"epoch": 2}
    elif name == "pictures_list":
        watch["pictures"] = [picture]
    elif name == "latest_list":
        sweep["latest"] = [picture]
    else:
        sweep = [1, 2]
    (job / "region_sweep.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    (job / "region_watch.json").write_text(json.dumps(watch), encoding="utf-8")
    (job / "region_sweep.json").write_text(json.dumps(sweep), encoding="utf-8")
    _both(job)


def test_odd_nested_watch_values_give_data_or_nothing(tmp_path):
    job = _job(tmp_path, maps=False)
    (job / "region_watch.json").write_text(json.dumps({"history": [[1]], "pictures": {"a.jpg": {"seen": 1, "contrast_log": [1]}}}),
                                           encoding="utf-8")
    (job / "region_sweep.json").write_text(json.dumps({"latest": {"a.jpg": [1]}}), encoding="utf-8")
    data = R.watch_data(job)
    assert data["rows"][0]["background"] is None and data["rows"][0]["sweep_background"] is None
    (job / "region_sweep.json").write_text(json.dumps({"latest": {"a.jpg": {"contrast": [1]}}}), encoding="utf-8")
    assert R.watch_data(job)["rows"][0]["sweep_clothes"] is None
    _both(job)


def test_an_empty_watch_state_has_no_page(tmp_path):
    job = _job(tmp_path, maps=False)
    (job / "region_watch.json").write_text(json.dumps({"epoch": None, "frozen": False, "history": [], "pictures": {}}),
                                           encoding="utf-8")
    assert R.watch_data(job) == {}
    (job / "region_watch.json").write_text(json.dumps({"pictures": {"a.jpg": {"seen": 0}}}), encoding="utf-8")
    assert R.watch_data(job) == {}
    assert R.load(job).watch == {}
    _both(job)


def _rows(*gaps, late=False):
    return [{"step": 10 * (i + 1), "train": {"subject": g / 100}, "held": {"subject": 0.0}, **({"late": True} if late else {})}
            for i, g in enumerate(gaps)]


def test_growing_gap_edges():
    assert R.growing_gap(_rows(0, 15)) == ("subject", pytest.approx(15.0), pytest.approx(15.0))
    assert R.growing_gap(_rows(10, 14.9)) is None  # the gap is below 15
    assert R.growing_gap(_rows(-5, 10)) is None  # a rise of 15 but a gap under 15
    assert R.growing_gap(_rows(30, 40)) is None  # a gap over 15 that did not rise by 15
    assert R.growing_gap(_rows(20)) is None


def test_a_resumed_baseline_counts_from_its_own_rows():
    old = _rows(0, 30)  # grew 30 points before the resume
    new = [{**r, "step": r["step"] + 100} for r in _rows(20, 22, late=True)]
    assert R.growing_gap(old + new) is None  # the new baseline's gap only rose 2 points
    assert [p[0] for p in R.sweep_gaps(old + new)["subject"]] == [110, 120]
    assert R.growing_gap(old) is not None
    assert R.since_baseline(_rows(1, 2, late=True)) == _rows(1, 2, late=True)  # a late run from the start: all of it


def test_the_memorised_warning_survives_a_long_sweep_line(tmp_path):
    job = _job(tmp_path, maps=False)
    # a real summary row: describe() makes a long line from the train / held shares
    shares = {c: 0.5 for c in ("face", "hair", "skin", "clothes", "background", "subject")}
    rows = [{"step": 20, "train": {"subject": 0.1, "face": 0.2}, "held": {"subject": 0.08, "face": 0.1}},
            {"step": 40, "train": dict(shares), "held": {c: 0.0 for c in shares}, "gap": dict(shares),
             "pictures": 30, "held_pictures": 8, "swept": 38, "seconds": 12.5, "best": [0.08, 20]}]
    (job / "region_sweep.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    run = R.load(job)
    assert len(R.regionsweep.describe(run.sweeps[-1])) > 240  # long enough to fill four wrapped lines
    text = R.sweeps_text(run)
    assert "may be memorised" in text and "Held-out photos peaked" in text and "Sweep at step 40" in text
    assert len(text.splitlines()) <= 5
    assert any("may be memorised" in t for _, t in R.verdict(run))
    _both(job)


def test_many_sweeps_and_pictures_paginate(tmp_path):
    job = _job(tmp_path, maps=False)
    rows = [{"step": 10 * i, "train": {"subject": i / 100, "face": i / 100}, "held": {"subject": i / 200}}
            for i in range(1, 41)]
    (job / "region_sweep.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    pictures = {f"photos/p{i}.jpg": {"seen": 3, "contrast_log": {"background": 0.01 * i, "clothes": -0.01 * i}}
                for i in range(37)}
    (job / "region_watch.json").write_text(json.dumps({"epoch": 3, "frozen": True, "pictures": pictures,
                                                       "history": [{"epoch": 3, "step": 40, "learned": {"face": 0.1}}]}),
                                           encoding="utf-8")
    assert len(R.watch_data(job)["rows"]) == 37
    done = _both(job)
    data = done["pdf"].read_bytes()
    assert data.count(b"/Type /Page\n") + data.count(b"/Type /Page ") >= 10


def test_a_watch_contrast_summary_goes_through_describe(tmp_path):
    job = _job(tmp_path, maps=False)
    (job / "region_watch.json").write_text(json.dumps({
        "epoch": 3, "frozen": True, "pictures": {"photos/a.jpg": {"seen": 3, "contrast_log": {"background": 0.1}}},
        "history": [{"epoch": 3, "step": 40, "learned": {"face": 0.4},
                     "contrast": {"background": {"median": 0.8, "pictures": 3}}}]}), encoding="utf-8")
    summary = R.watch_data(job)["summary"]
    line = R.regionwatch.describe(summary)
    assert "background against subject x0.80 (median of 3 pictures)" in line
    _both(job)
