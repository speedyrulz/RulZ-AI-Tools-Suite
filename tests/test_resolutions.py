"""Multiple training resolutions and the resolution step-up controller."""
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import dataset as D  # noqa: E402
from rulztrainer.config import MEGAPIXEL_CHOICES, TrainConfig, mp_to_side, side_to_mp  # noqa: E402
from rulztrainer.likeness import LikenessController  # noqa: E402


def test_megapixel_conversions_and_sides():
    assert mp_to_side(1.0) == 1024 and mp_to_side(0.25) == 512 and mp_to_side(4.0) == 2048
    assert side_to_mp(1024) == 1.0 and side_to_mp(512) == 0.25
    assert len(MEGAPIXEL_CHOICES) == 16 and MEGAPIXEL_CHOICES[0] == 0.25 and MEGAPIXEL_CHOICES[-1] == 4.0
    assert TrainConfig().training_sides() == [1024]
    assert TrainConfig(resolution=768).training_sides() == [768]
    assert TrainConfig(resolutions=[4.0, 0.25, 1.0, 1.0]).training_sides() == [512, 1024, 2048]


def test_validate_step_up_requirements():
    TrainConfig(resolutions=[0.25, 1.0]).validate()
    with pytest.raises(ValueError):
        TrainConfig(resolutions=[0.25, 1.0], resolution_step_up=True).validate()  # no probes
    with pytest.raises(ValueError):
        TrainConfig(resolutions=[0.25, 1.0], resolution_step_up=True, probe_every=25,
                    face_refs=["a.png"], step_up_patience=0).validate()
    TrainConfig(resolutions=[0.25, 1.0], resolution_step_up=True, probe_every=25, face_refs=["a.png"]).validate()
    with pytest.raises(ValueError):
        TrainConfig(resolutions=[0.0]).validate()


def test_bucket_size_per_side_keeps_aspect():
    cfg = TrainConfig(min_bucket_side=256, max_bucket_side=4096)  # limits never reached: area and aspect hold
    for side in (512, 1024, 2048):
        for w, h in ((1000, 1500), (1500, 1000), (1024, 1024), (1600, 900)):
            bw, bh = D.bucket_size(w, h, cfg, side)
            assert bw % 16 == 0 and bh % 16 == 0
            assert abs(bw * bh - side * side) / (side * side) < 0.08, (side, w, h, bw, bh)
            assert abs(bw / bh - w / h) < 0.08
    # 0.25 MP portrait with the default 512 minimum: it used to be cropped to 512x624; now the whole bucket is
    # scaled up so the aspect ratio survives
    cfg = TrainConfig(min_bucket_side=512, max_bucket_side=2048)
    bw, bh = D.bucket_size(1000, 1500, cfg, 512)
    assert bh >= 512 and bw >= 512 and abs(bw / bh - 2 / 3) < 0.06, (bw, bh)
    # the default (cfg.resolution) path is unchanged for ordinary aspects
    assert D.bucket_size(1000, 1500, cfg) == D.bucket_size(1000, 1500, cfg, 1024)
    assert D.bucket_size(1000, 1500, TrainConfig(enable_buckets=False), 512) == (512, 512)


def test_scan_dataset_items_per_resolution(tmp_path):
    ds = tmp_path / "ds"
    ds.mkdir()
    Image.new("RGB", (640, 480), "red").save(ds / "a.png")
    Image.new("RGB", (300, 500), "blue").save(ds / "b.jpg")
    (ds / "a.txt").write_text("a red thing", encoding="utf-8")
    cfg = TrainConfig(dataset_dir=str(ds), resolutions=[0.25, 1.0], min_bucket_side=256)
    items = D.scan_dataset(cfg)
    assert len(items) == 4 and {it.side for it in items} == {512, 1024}
    assert len({it.latent_path for it in items}) == 4  # one latent cache per image and resolution
    low, high = D.items_for_side(items, 512), D.items_for_side(items, 1024)
    assert len(low) == 2 and len(high) == 2
    for lo, hi in zip(low, high):
        assert lo.key == hi.key  # per-image state (loss watch, curriculum) is shared across resolutions
        assert lo.width * lo.height < hi.width * hi.height
    assert D.describe_sides(cfg) == "0.25 MP, 1 MP"
    single = D.scan_dataset(TrainConfig(dataset_dir=str(ds)))
    assert len(single) == 2 and all(it.side == 1024 for it in single)


def _ctrl(**kw):
    base = dict(resolutions=[0.25, 1.0, 4.0], resolution_step_up=True, probe_every=10, probe_ema=0.0,
                likeness_noise_band=1.0, likeness_track_best=False, likeness_patience=1, face_refs=["x.png"],
                face_weight=1.0, body_weight=0.0)
    base.update(kw)
    return LikenessController(TrainConfig(**base))


def test_step_up_controller_advances_and_stops():
    c = _ctrl()
    assert c.n_stages == 3 and c.stage == 0
    rec, a = c.update(0, 50.0, None)  # stage 0 baseline
    assert "step_up" not in a and rec.stage == 0 and rec.stage_best == 50.0
    rec, a = c.update(10, 55.0, None)  # improves: stays
    assert "step_up" not in a and rec.stage_since_improve == 0
    rec, a = c.update(20, 55.5, None)  # within the noise band: stalled once -> patience 1 -> step up
    assert a.get("step_up") == 1 and rec.stage == 0 and rec.stage_since_improve == 1
    assert c.stage == 1 and c.stage_best is None  # fresh baseline for the new stage
    rec, a = c.update(30, 40.0, None)  # a dip right after the switch is the new baseline, not a stall
    assert "step_up" not in a and "stop" not in a and rec.stage == 1 and rec.stage_best == 40.0
    rec, a = c.update(40, 39.0, None)  # stalled -> stage 2 (the highest)
    assert a.get("step_up") == 2 and c.last_stage
    c.update(50, 60.0, None)
    rec, a = c.update(60, 60.0, None)  # stalled at the top -> stop (stop patience 1)
    assert a.get("stop") is True and c.stopped
    # likeness_patience (1) never fired on its own while step-up is on, even though scores stalled earlier
    assert rec.best == 60.0


def test_step_up_patience_settings_are_separate():
    c = _ctrl(step_up_patience=2, step_up_stop_patience=3)
    c.update(0, 50.0, None)
    assert "step_up" not in c.update(10, 50.0, None)[1]  # 1 stall of 2
    assert c.update(20, 50.0, None)[1].get("step_up") == 1  # 2 stalls -> next stage
    c.enter_stage(2)  # jump to the top
    c.update(30, 50.0, None)
    for i in range(2):
        assert "stop" not in c.update(40 + 10 * i, 50.0, None)[1]
    assert c.update(60, 50.0, None)[1].get("stop") is True
    assert c.state()["stage"] == 2 and c.state()["n_stages"] == 3


def test_controller_without_step_up_unchanged():
    c = _ctrl(resolution_step_up=False, likeness_patience=2)
    c.update(0, 50.0, None)
    c.update(10, 50.0, None)
    rec, a = c.update(20, 50.0, None)
    assert a.get("stop") is True and "step_up" not in a and rec.stage == 0
