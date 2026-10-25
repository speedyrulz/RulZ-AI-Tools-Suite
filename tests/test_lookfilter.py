"""The look consistency filter: scores against baselines, the outlier fence, saved scans (ours and Fizgig's), moving
pictures out and back with their captions, training skipping excluded_by_look, and the look warm-up loss weights."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import dataset as D  # noqa: E402
from rulztrainer import lookfilter as LF  # noqa: E402
from rulztrainer.config import TrainConfig  # noqa: E402
from rulztrainer.trainer import Trainer  # noqa: E402


def _unit(angle):
    return np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)


def test_scores_fence_and_suggestions():
    vecs = {"b1": _unit(0.0), "b2": _unit(0.1), "x1": _unit(0.05), "x2": _unit(0.3), "x3": _unit(0.4),
            "x4": _unit(0.5), "far": _unit(1.4), "none": None}
    scores = LF.score_images(list(vecs), ["b1", "b2"], vecs.get)
    assert scores["none"] is None
    assert scores["x1"] == pytest.approx((np.cos(0.05) + np.cos(0.05)) / 2, abs=1e-6)  # mean over the baselines
    med, cut = LF.fence(scores)
    vals = sorted(v for v in scores.values() if v is not None)
    q1, q2, q3 = np.percentile(vals, [25, 50, 75])
    assert med == pytest.approx(q2) and cut == pytest.approx(max(q2 - 1.5 * (q3 - q1), 0.25))
    assert LF.suggest(scores, ["b1", "b2"]) == {"far"}  # never a baseline, never no face
    assert LF.fence({"a": 0.9, "b": 0.8, "c": None}) == (None, None) and LF.suggest({"a": 0.1}) == set()
    assert LF.fence({k: 0.9 for k in "abcd"})[1] == pytest.approx(0.9)  # no spread: the fence is the median
    assert LF.fence({"a": 0.3, "b": 0.3, "c": 0.3, "d": 0.3})[1] == pytest.approx(0.3)
    assert LF.fence({"a": 0.26, "b": 0.27, "c": 0.9, "d": 0.95})[1] == 0.25  # never under the floor
    assert [LF.verdict(v) for v in (0.5, 0.35, 0.1, None)] == ["match", "borderline", "drift", "no face"]
    with pytest.raises(ValueError, match="No face found in baseline none"):
        LF.score_images(["b1"], ["b1", "none"], vecs.get)


def _dataset(tmp_path):
    root = tmp_path / "ds"
    (root / "sub").mkdir(parents=True)
    for rel in ("a.png", "b.png", "c.png", "d.png", "e.png", "sub/f.png"):
        Image.new("RGB", (64, 64), (90, 90, 90)).save(root / rel)
        (root / rel).with_suffix(".txt").write_text(f"caption of {rel}", encoding="utf-8")
    return root


def test_save_load_and_outliers(tmp_path):
    root = _dataset(tmp_path)
    scores = {str(root / "a.png"): 0.7, str(root / "b.png"): 0.68, str(root / "c.png"): 0.66,
              str(root / "d.png"): 0.69, str(root / "e.png"): None, str(root / "sub" / "f.png"): 0.2}
    LF.save_scores(root, [root / "a.png"], scores)
    saved = json.loads((root / LF.SCORES_FILE).read_text(encoding="utf-8"))
    assert saved["baselines"] == ["a.png"] and "sub/f.png" in saved["scores"]  # paths inside the dataset
    data = LF.load_scores(root)
    assert data["scores"] == {str(root / k): v for k, v in saved["scores"].items()}
    assert data["cutoff"] == pytest.approx(LF.fence(scores)[1])
    assert LF.outliers(root) == {str((root / "sub" / "f.png").resolve())}
    # Fizgig's file (scores by stem) is read when there is none of ours
    (root / LF.SCORES_FILE).unlink()
    (root / LF.FIZGIG_FILE).write_text(json.dumps({"baselines": ["a.png"], "cutoff": 0.5,
                                                   "scores": {"a": 0.8, "b": 0.3, "gone": 0.1}}), encoding="utf-8")
    fz = LF.load_scores(root)
    assert fz["scores"] == {str(root / "a.png"): 0.8, str(root / "b.png"): 0.3} and fz["cutoff"] == 0.5
    assert LF.outliers(root) == {str((root / "b.png").resolve())}
    assert LF.load_scores(tmp_path) is None and LF.outliers(tmp_path) == set()


def test_move_out_and_back_with_captions(tmp_path):
    root = _dataset(tmp_path)
    moved, errors = LF.move_out(root, [root / "b.png", root / "sub" / "f.png", tmp_path / "elsewhere.png"])
    assert errors == [] and len(moved) == 2
    ex = root / LF.EXCLUDED_DIR
    assert (ex / "b.png").is_file() and (ex / "b.txt").read_text(encoding="utf-8") == "caption of b.png"
    assert (ex / "sub" / "f.png").is_file() and (ex / "sub" / "f.txt").is_file()
    assert not (root / "b.png").exists() and not (root / "b.txt").exists()
    # training and the WebUI's listing skip the folder
    cfg = TrainConfig(dataset_dir=str(root), cache_dir=str(tmp_path / "cache"))
    names = sorted({it.image.name for it in D.scan_dataset(cfg)})
    assert names == ["a.png", "c.png", "d.png", "e.png"]
    assert D.skipped(ex / "b.png", root) and not D.skipped(root / "a.png", root)
    # a picture with the same name added since is not overwritten when the moved one comes back
    Image.new("RGB", (8, 8)).save(root / "b.png")
    back, errors = LF.move_back(root)
    assert errors == [] and len(back) == 2 and not ex.exists()
    assert (root / "b_2.png").is_file() and (root / "b_2.txt").read_text(encoding="utf-8") == "caption of b.png"
    assert (root / "sub" / "f.png").is_file() and (root / "sub" / "f.txt").is_file()


def test_warmup_weights(tmp_path):
    assert LF.warmup_factor(0, 0.4, 4) == pytest.approx(0.4)
    assert LF.warmup_factor(2, 0.4, 4) == pytest.approx(0.7)
    assert LF.warmup_factor(9, 0.4, 4) == 1.0 and LF.warmup_factor(0, 0.4, 0) == 1.0
    tr = Trainer.__new__(Trainer)
    tr.look_outliers = {str(Path("ds/odd.png"))}
    items = [SimpleNamespace(image=Path("ds/odd.png"), repeats=1), SimpleNamespace(image=Path("ds/ok.png"), repeats=3)]
    batcher = SimpleNamespace(items=items)  # 4 images per epoch (repeats counted)
    cfg = TrainConfig(look_warmup=True, look_warmup_start=0.4, look_warmup_epochs=4)
    assert tr._look_factors(items, 1, cfg, batcher, 2) == [pytest.approx(0.4), 1.0]  # step 1: epoch 0
    assert tr._look_factors(items, 5, cfg, batcher, 2) == [pytest.approx(0.7), 1.0]  # 8 images in: epoch 2
    assert tr._look_factors(items, 9, cfg, batcher, 2) is None  # epoch 4: full weight
    assert tr._look_factors(items, 1, TrainConfig(), batcher, 2) is None  # off


def test_trainer_finds_the_outliers(tmp_path, caplog):
    root = _dataset(tmp_path)
    cfg = TrainConfig(dataset_dir=str(root), cache_dir=str(tmp_path / "cache"), look_warmup=True)
    items = D.scan_dataset(cfg)
    tr = Trainer.__new__(Trainer)
    tr.race_rows = None
    assert tr._look_outlier_images(cfg, items) == set()  # no scan yet: off, with a warning
    assert "no look consistency scan" in caplog.text
    LF.save_scores(root, [root / "a.png"], {str(root / n): s for n, s in (
        ("a.png", 0.7), ("b.png", 0.69), ("c.png", 0.68), ("d.png", 0.67), ("e.png", 0.1))})
    assert tr._look_outlier_images(cfg, items) == {str(it.image) for it in items if it.image.name == "e.png"}
    assert tr._look_outlier_images(TrainConfig(dataset_dir=str(root)), items) == set()  # look_warmup off
