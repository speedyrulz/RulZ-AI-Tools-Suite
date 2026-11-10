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
    for i, step in enumerate((0, 10, 20, 30, 40)):
        body = [20.0 + 5 * i, None if step >= 30 else 25.0 + 5 * i, 30.0 + i]
        records.append({"step": step, "face": 50.0 + i, "body": 25.0, "combined": 30.0 + 4 * i, "face_ema": 50.0 + i,
                        "body_ema": 25.0, "combined_ema": 28.0 + 4 * i - (9 if step == 40 else 0), "best": 40.0,
                        "best_step": 30, "face_norm": 10.0 * i, "body_norm": 25.0, "face_norm_ema": 10.0 * i,
                        "body_norm_ema": 25.0, "face_norm_seeds": [10.0 * i, 12.0 * i + 1, 8.0 * i],
                        "body_norm_seeds": body, "copy": 0.4 + 0.07 * i, "body_probes": 3,
                        "body_missing": sum(v is None for v in body)})
        if probes:
            for k in range(3):
                _png(job / "probes" / f"{racer + '_' if racer else ''}step{step:06d}_face_{k}.png", (64, 64), (40 * k, 90, 120))
                _png(job / "probes" / f"{racer + '_' if racer else ''}step{step:06d}_body_{k}.png", (64, 96), (120, 40 * k, 90))
    (job / "likeness.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    status = {"stage": "finished", "step": 40, "total": 40, "elapsed": 135.0, "avg_loss": 0.21, "done": True,
              "losses": [0.3 - 0.002 * i for i in range(40)], "saved": ["output/demo run/demo run.safetensors"],
              "run_id": "demo_run-20260102-030405", "peak_mem": {"cuda:0": 9.5},
              "near_duplicates": {"count": 2}}
    if racer:
        status["race"] = {"names": [racer, "other"], "champion": racer, "dropped": {"other": 2},
                          "left": {"other": "drift"}, "best": {racer: 40.0, "other": 22.0}}
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
    assert R.main([str(job), "--pdf"]) == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1 and out[0].endswith("demo_run_report.pdf") and Path(out[0]).is_file()
    assert not (job / "report" / "demo_run_report.png").exists()
    assert R.main([str(tmp_path / "missing")]) == 1 and "no report" in capsys.readouterr().err


def test_the_setting_is_off_by_default_and_round_trips():
    assert TrainConfig().run_report is False
    assert TrainConfig.from_dict(TrainConfig(run_report=True).to_dict()).run_report is True
