"""Region sweep (regionsweep.py): fixed-noise readings per picture, the held-out pictures, their best and their stop.
A toy model that removes the error where it is told to; parts tensors made by hand."""
import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import regionsweep as RS  # noqa: E402
from rulztrainer.dataset import Item  # noqa: E402

H, W = 4, 8


class Toy(torch.nn.Module):
    """Predicts `gain` x the noise per cell (the latents are zero, so the noise is noisy / sigma): gain 1 on a cell
    is no error there, gain 0 the whole error."""

    def __init__(self):
        super().__init__()
        self.gain = torch.nn.Parameter(torch.zeros(H, W))
        self.calls = 0

    def forward(self, noisy, sigma, embeds, mask):
        self.calls += 1
        return self.gain * noisy.float() / (sigma.view(-1, 1, 1, 1) / 1000.0)


def _parts():
    """Background on the left half, face on columns 4-5, clothes on columns 6-7."""
    p = torch.zeros(7, H, W, dtype=torch.uint8)
    p[0][:, :4], p[3][:, 4:6], p[4][:, 6:] = 255, 255, 255
    return p


def _probes(n_train, n_held=0):
    def load():
        return torch.zeros(1, 3, H, W), torch.zeros(1, 2, 4), torch.ones(1, 2, dtype=torch.bool)

    return ([{"key": f"train{i}.png", "held": False, "parts": _parts(), "load": load} for i in range(n_train)]
            + [{"key": f"held{i}.png", "held": True, "parts": _parts(), "load": load} for i in range(n_held)])


def _items(tmp_path, n, **kw):
    return [Item(image=tmp_path / f"img{i:02d}.png", caption="c", width=64, height=64, latent_path=tmp_path / "l",
                 latent_flip_path=None, text_path=tmp_path / "t", **kw) for i in range(n)]


def test_holdout_pick_is_stable_and_takes_a_photos_crops_out_of_training(tmp_path):
    photos = _items(tmp_path, 20)
    crops = [Item(image=p.image, caption="c", width=64, height=64, latent_path=tmp_path / "l", latent_flip_path=None,
                  text_path=tmp_path / "t", category="face_crop") for p in photos]
    pairs = _items(tmp_path / "pairs", 2, pair_side="left")
    items = photos + crops + pairs
    held = RS.pick_holdout(items, 2)
    assert held == RS.pick_holdout(list(reversed(items)), 2) and len(held) == 2  # by name, whatever the order
    assert held == {str(photos[5].image), str(photos[15].image)}  # the middles of two equal stretches
    train, out = RS.split_holdout(items, 2)
    assert [str(it.image) for it in out] == [str(photos[5].image), str(photos[15].image)]
    assert len(train) == len(items) - 4 and not any(str(it.image) in held for it in train)  # photos and their crops
    assert all(p in train for p in pairs)
    assert RS.split_holdout(items, 0) == (items, [])
    RS.check_holdout(20, 4)
    for n_photos, n in ((9, 1), (20, 5), (10, 3)):
        with pytest.raises(ValueError):
            RS.check_holdout(n_photos, n)
    RS.check_holdout(10, 2) and RS.check_holdout(12, 1)


def test_a_sweep_reads_each_region_against_its_own_first_sweep(tmp_path):
    model = Toy()
    model.train()
    sweep = RS.RegionSweep(tmp_path, _probes(2, 1), every=10)
    assert sweep.due(0)
    torch.manual_seed(7)
    before = torch.get_rng_state()
    first = sweep.run(0, 0.0, model, "cpu")
    assert bool((torch.get_rng_state() == before).all()) and model.training  # nothing of training is touched
    assert first["first"] and first["swept"] == 3 and model.calls == 9  # 3 pictures x 3 noise levels
    assert first["train"]["face"] == 0.0 and first["held"]["background"] == 0.0
    assert "first sweep of 3 pictures" in RS.describe(first)
    assert not sweep.due(0) and not sweep.due(9) and sweep.due(10) and sweep.due(5, last=True)
    same = sweep.run(10, 1.0, model, "cpu")
    assert same["train"]["face"] == pytest.approx(0.0, abs=1e-6)  # the same noise: an unchanged model reads 0
    with torch.no_grad():
        model.gain[:, 4:6] = 1.0  # the face is learned whole
        model.gain[:, :4] = 0.5  # the background's error halves in amplitude: a quarter left
    row = sweep.run(20, 2.0, model, "cpu")
    assert row["train"]["face"] == pytest.approx(1.0, abs=1e-3) and row["held"]["face"] == pytest.approx(1.0, abs=1e-3)
    assert row["train"]["background"] == pytest.approx(0.75, abs=1e-3)
    assert row["train"]["clothes"] == pytest.approx(0.0, abs=1e-6) and row["train"]["hair"] is None
    assert row["gap"]["face"] == pytest.approx(0.0, abs=1e-6)
    assert "training pictures learned face 100%" in RS.describe(row) and "held-out pictures learned" in RS.describe(row)
    lines = (tmp_path / RS.LOG_FILE).read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["step"] for line in lines] == [0, 10, 20]


def test_contrast_is_the_background_against_the_subject(tmp_path):
    model = Toy()
    sweep = RS.RegionSweep(tmp_path, _probes(2), every=10)
    sweep.run(0, 0.0, model, "cpu")
    assert sweep.contrasts() == {"train0.png": {"background": 1.0, "clothes": 1.0},
                                 "train1.png": {"background": 1.0, "clothes": 1.0}}
    with torch.no_grad():
        model.gain[:, :4] = 0.5  # the background learned, the subject (the face) not
    sweep.run(10, 1.0, model, "cpu")
    c = sweep.contrasts()["train0.png"]
    assert c["background"] == pytest.approx(0.25, abs=1e-3) and c["clothes"] == pytest.approx(1.0, abs=1e-3)


def test_rotation_takes_every_picture_first_and_then_a_few_in_turn(tmp_path):
    model = Toy()
    sweep = RS.RegionSweep(tmp_path, _probes(6, 2), every=10, rotate=2)
    assert sweep.run(0, 0.0, model, "cpu")["swept"] == 8  # the first sweep: everything
    seen = []
    for k in range(1, 4):
        picked = [p["key"] for p in sweep.chosen()]
        seen += [key for key in picked if key.startswith("train")]
        assert [key for key in picked if key.startswith("held")] == ["held0.png", "held1.png"]  # always
        assert len(picked) == 4
    assert seen == [f"train{i}.png" for i in range(6)]  # every picture once in three sweeps


def test_held_out_best_flat_regions_and_the_stop(tmp_path):
    model = Toy()
    sweep = RS.RegionSweep(tmp_path, _probes(1, 2), every=10, patience=2)
    row = sweep.run(0, 0.0, model, "cpu")
    assert row["best"] is None and not row["new_best"] and not row["stop"]
    with torch.no_grad():
        model.gain[:, 4:6] = 0.5
    row = sweep.run(10, 1.0, model, "cpu")
    assert row["new_best"] and row["best"] == [pytest.approx(0.75, abs=1e-3), 10] and row["since_best"] == 0
    assert row["flat"]["face"] == 0
    row = sweep.run(20, 2.0, model, "cpu")  # nothing better
    assert not row["new_best"] and row["since_best"] == 1 and not row["stop"] and row["flat"]["face"] == 1
    with torch.no_grad():
        model.gain[:, 4:6] = 0.4  # worse on the held-out face
    row = sweep.run(30, 3.0, model, "cpu")
    assert row["since_best"] == 2 and row["stop"] and row["flat"]["face"] == 2 and row["best"][1] == 10
    assert "held-out subject best 75% at step 10" in RS.describe(row)
    with torch.no_grad():
        model.gain[:, 4:6] = 0.9
    row = sweep.run(40, 4.0, model, "cpu")
    assert row["new_best"] and row["since_best"] == 0 and not row["stop"] and row["flat"]["face"] == 0
    no_patience = RS.RegionSweep(tmp_path / "b", _probes(1, 1), every=10)
    no_patience.run(0, 0.0, Toy(), "cpu")
    assert not no_patience.run(10, 1.0, Toy(), "cpu")["stop"]


def test_resume_reads_the_first_sweep_back_and_notices_other_held_pictures(tmp_path):
    model = Toy()
    sweep = RS.RegionSweep(tmp_path, _probes(2, 1), every=10)
    sweep.run(0, 0.0, model, "cpu")
    with torch.no_grad():
        model.gain[:, 4:6] = 0.5
    sweep.run(10, 1.0, model, "cpu")
    again = RS.RegionSweep(tmp_path, _probes(2, 1), every=10, resume=True)
    assert not again.late and again.best == sweep.best and again.base.keys() == sweep.base.keys()
    assert again.next_step == 20 and not again.due(15) and again.due(20)
    row = again.run(20, 2.0, model, "cpu")
    assert row["train"]["face"] == pytest.approx(0.75, abs=1e-3)  # against the saved first sweep, not a new one
    other = RS.RegionSweep(tmp_path, _probes(2, 2), every=10, resume=True)  # another set held out
    assert other.late and not other.base
    fresh = RS.RegionSweep(tmp_path, _probes(2, 1), every=10)
    assert not (tmp_path / RS.STATE_FILE).exists() and not fresh.base
    assert RS.RegionSweep(tmp_path / "none", _probes(1), every=10, resume=True).late


def test_a_failing_sweep_turns_itself_off_and_pictures_without_parts_are_left_out(tmp_path):
    class Broken(torch.nn.Module):
        def forward(self, *a):
            raise RuntimeError("boom")

    probes = _probes(2)
    probes[1]["parts"] = None
    sweep = RS.RegionSweep(tmp_path, probes, every=10)
    assert [p["key"] for p in sweep.probes] == ["train0.png"]
    broken = Broken()
    broken.train()
    assert sweep.run(0, 0.0, broken, "cpu") is None and sweep.failed and broken.training
    assert not sweep.due(10) and sweep.run(10, 1.0, Toy(), "cpu") is None
    assert not RS.RegionSweep(tmp_path / "e", [], every=10).due(0)


class Scalar(torch.nn.Module):
    """As Toy, one gain for every cell and any grid."""

    def __init__(self):
        super().__init__()
        self.gain = torch.nn.Parameter(torch.zeros(()))

    def forward(self, noisy, sigma, embeds, mask):
        return self.gain * noisy.float() / (sigma.view(-1, 1, 1, 1) / 1000.0)


class FakeArch:
    def pad_embeds(self, rows):
        e = torch.stack(rows)
        return e, torch.ones(e.shape[:2], dtype=torch.bool)


def _stub_trainer(tmp_path, n, **cfg_kw):
    from safetensors.torch import save_file

    from rulztrainer.config import TrainConfig
    from rulztrainer.regions import RegionMap, save_region_map
    from rulztrainer.trainer import Status, Trainer

    items = []
    for i in range(n):
        lat, txt, reg = (tmp_path / f"p{i:02d}.{kind}.safetensors" for kind in ("lat", "txt", "reg"))
        save_file({"latent": torch.zeros(4, 3, 4)}, str(lat))
        save_file({"embeds": torch.zeros(2, 4)}, str(txt))
        parts = torch.zeros(7, 3, 4, dtype=torch.uint8)
        parts[0][:, :2], parts[3][:, 2:] = 255, 255  # background left, face right
        save_region_map(reg, RegionMap(torch.ones(1, 3, 4), torch.zeros(1, 3, 4, dtype=torch.bool), None, None, True,
                                       parts, "p"))
        items.append(Item(image=tmp_path / f"p{i:02d}.jpg", caption="c", width=64, height=48, latent_path=lat,
                          latent_flip_path=None, text_path=txt, region_path=reg))
    t = Trainer.__new__(Trainer)
    t.cfg = TrainConfig(job_name="t", output_dir=str(tmp_path), **cfg_kw)
    t.job_dir = tmp_path / "job"
    t.job_dir.mkdir()
    t.arch, t.status = FakeArch(), Status(t.job_dir / "status.json")
    t.race_rows = t.ft = t.ema = t.dp = t.region_watch = t.sweep = t.batcher = None
    t.region_watch_off, t.holdout_items, t.holdout_stop = "", [], False
    return t, items


def test_trainer_holds_photos_out_sweeps_them_keeps_the_best_and_stops(tmp_path):
    from rulztrainer.dataset import BucketBatcher

    t, items = _stub_trainer(tmp_path, 12, holdout_images=2, holdout_track_best=True, holdout_patience=1)
    train = t._split_holdout(t.cfg, items)
    assert len(train) == 10 and [it.image.name for it in t.holdout_items] == ["p03.jpg", "p09.jpg"]
    t.batcher = BucketBatcher(train, 1, 0)
    model = Scalar()
    model.train()
    t.replicas = [(model, {}, "cpu")]
    saved, watched = [], []
    t._save_holdout_best = lambda injected, step, score: saved.append((step, round(score, 3)))
    t.region_watch = type("W", (), {"set_sweep": lambda self, c, gap, flat, learned=None: watched.append(
        (len(c), gap["face"], len(learned)))})()
    t._setup_sweep(t.cfg, 0)
    assert t.sweep is not None and t.sweep.every == 10  # 10 training pictures, one a step: one epoch
    assert len(t.sweep.probes) == 12 and sum(p["held"] for p in t.sweep.probes) == 2
    t._run_sweep(model, {}, 0)
    status = json.loads((t.job_dir / "status.json").read_text(encoding="utf-8"))
    assert status["region_sweep"]["first"] and "first sweep of 12 pictures" in status["region_sweep"]["line"]
    assert saved == [] and model.training
    with torch.no_grad():
        model.gain.fill_(0.5)
    t._run_sweep(model, {}, 10)
    assert saved == [(10, 0.75)] and not t.holdout_stop
    assert watched[-1] == (10, pytest.approx(0.0, abs=1e-3), 10)  # contrasts, the gap, each picture learned share
    t._run_sweep(model, {}, 20)  # no better: with a patience of 1 the run is told to stop
    assert saved == [(10, 0.75)] and t.holdout_stop
    assert json.loads((t.job_dir / "status.json").read_text(encoding="utf-8"))["region_sweep"]["stop"] is True


def test_trainer_sweeps_only_what_is_asked_and_stays_off_when_it_cannot(tmp_path):
    from rulztrainer.dataset import BucketBatcher

    t, items = _stub_trainer(tmp_path, 12, holdout_images=1, region_sweep_pictures=-1)
    train = t._split_holdout(t.cfg, items)
    t.batcher = BucketBatcher(train, 1, 0)
    t.replicas = [(Scalar(), {}, "cpu")]
    t._setup_sweep(t.cfg, 0)
    assert [p["held"] for p in t.sweep.probes] == [True]  # the held-out photo alone
    t.cfg.region_sweep_pictures, t.cfg.region_sweep_every = 4, 3
    t._setup_sweep(t.cfg, 0)
    assert t.sweep.rotate == 4 and t.sweep.every == 33 and len(t.sweep.probes) == 12  # every 3 epochs of 11 steps
    t.race_rows = [("a", t.cfg)]
    t._setup_sweep(t.cfg, 0)
    assert t.sweep is None and t._split_holdout(t.cfg, items) == items and t.holdout_items == []  # a race: all train
    t.race_rows, t.region_watch_off = None, "RuntimeError: no parser"
    assert t._split_holdout(t.cfg, items) == items
    t.region_watch_off = ""
    with pytest.raises(ValueError, match="at least 10 photos"):
        t._split_holdout(t.cfg, items[:6])
    t.cfg.holdout_images = 0
    t.cfg.region_sweep_every = 0
    t._setup_sweep(t.cfg, 0)
    assert t.sweep is None and t._split_holdout(t.cfg, items) == items
