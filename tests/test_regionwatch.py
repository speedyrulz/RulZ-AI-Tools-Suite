"""Region watch: the training steps' loss split by the picture's parts, the frozen baseline, the learned share per
region and the per-picture contrast (regionwatch.py). The parts tensors are made by hand: no parser needed."""
import json
import random
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import regionwatch as RW  # noqa: E402
from rulztrainer.config import TrainConfig  # noqa: E402
from rulztrainer.dataset import Item  # noqa: E402
from rulztrainer.regions import RegionMap, flip_region_map  # noqa: E402

FACE, HAIR, SKIN, CLOTHES, BACKGROUND, SUBJECT = (RW.COLUMNS.index(c) for c in RW.COLUMNS)


def _items(tmp_path, n, **kw):
    return [Item(image=tmp_path / f"img{i}.png", caption=f"cap {i}", width=512, height=512,
                 latent_path=tmp_path / "l", latent_flip_path=None, text_path=tmp_path / "t", **kw) for i in range(n)]


def _parts(h, w, classes):
    """A parts tensor [7, h, w] from a class per cell (the parser's numbering), no ring."""
    p = torch.zeros(7, h, w, dtype=torch.uint8)
    for c in range(6):
        p[c][classes == c] = 255
    return p


def test_region_losses_split_by_class():
    h, w = 4, 6
    classes = torch.zeros(h, w, dtype=torch.long)  # background
    classes[:, 3:] = 3  # face skin on the right half
    classes[0, 0], classes[0, 1], classes[1, 0] = 4, 5, 1  # clothes, others (counted as clothes), hair
    sq = torch.ones(3, 2, h, w)
    sq[0, :, :, 3:] = 5.0  # the face cells of sample 0 are 5x as wrong
    sq[0, :, 0, 0], sq[0, :, 0, 1] = 2.0, 4.0
    losses, cells = RW.region_losses(sq, [_parts(h, w, classes), None, _parts(h + 1, w, torch.zeros(h + 1, w))])
    assert losses[1] is None and cells[1] is None  # no parts map
    assert losses[2] is None  # a parts map of another size
    row, size = losses[0], cells[0]
    assert row[FACE] == pytest.approx(5.0) and size[FACE] == pytest.approx(12.0)
    assert row[CLOTHES] == pytest.approx(3.0) and size[CLOTHES] == pytest.approx(2.0)  # clothes + others
    assert row[HAIR] == pytest.approx(1.0) and size[HAIR] == pytest.approx(1.0)
    assert size[SKIN] == 0.0
    assert row[BACKGROUND] == pytest.approx(1.0) and size[BACKGROUND] == pytest.approx(9.0)
    assert size[SUBJECT] == pytest.approx(13.0)  # face + hair + skin
    assert row[SUBJECT] == pytest.approx((12 * 5.0 + 1.0) / 13)


def _row(face, background, clothes=None, base=1.0):
    """One sample's losses and cells: a face, hair and skin learned alike, a background and (optionally) clothes."""
    c = base if clothes is None else clothes
    losses = [face, face, face, c, background, face]
    cells = [20.0, 30.0, 40.0, 50.0 if clothes is not None else 0.0, 200.0, 90.0]
    return losses, cells


def _run(watch, items, epochs, loss_of, rng, start_epoch=0, step=0):
    """Feeds `epochs` epochs: loss_of(i, epoch) -> (face factor, background factor) on a loss that grows with noise."""
    summary = None
    for epoch in range(start_epoch, start_epoch + epochs):
        for i, it in enumerate(items):
            sigma = rng.random()
            base = 0.1 + 0.9 * sigma
            f, b = loss_of(i, epoch)
            losses, cells = _row(base * f, base * b)
            step += 1
            watch.observe([it.key], [sigma], [losses], [cells])
        assert watch.epoch_complete()
        summary = watch.end_epoch(step)
    return summary, step


def test_learned_per_region_against_the_frozen_baseline(tmp_path):
    items = _items(tmp_path, 20)
    watch = RW.RegionWatch(items, tmp_path)
    rng = random.Random(0)
    summary, step = _run(watch, items, 3, lambda i, e: (1.0, 1.0), rng)
    assert not summary["frozen"] and "collecting the baseline" in RW.describe(summary)  # 60 of 64 samples
    summary, step = _run(watch, items, 1, lambda i, e: (1.0, 1.0), rng, 3, step)
    assert summary["frozen"] and not summary["learned"]  # the baseline itself reads as nothing learned
    assert "baseline set" in RW.describe(summary)
    # the face loses 40% of its error, the background 10%
    summary, step = _run(watch, items, 6, lambda i, e: (0.6, 0.9), rng, 4, step)
    learned = summary["learned"]
    assert learned["face"] == pytest.approx(0.4, abs=0.05) and learned["hair"] == pytest.approx(0.4, abs=0.05)
    assert learned["background"] == pytest.approx(0.1, abs=0.05)
    assert learned["clothes"] is None  # no picture has clothes cells
    assert learned["subject"] == pytest.approx(0.4, abs=0.05)
    assert "face 40%" in RW.describe(summary) or "face 4" in RW.describe(summary)
    rows = [json.loads(line) for line in (tmp_path / RW.LOG_FILE).read_text(encoding="utf-8").splitlines()]
    assert [r["epoch"] for r in rows] == [5, 6, 7, 8, 9, 10]
    assert RW.read(tmp_path)["frozen"] is True


def test_contrast_follows_a_picture_whose_background_runs_ahead(tmp_path):
    items = _items(tmp_path, 20)
    watch = RW.RegionWatch(items, tmp_path)
    rng = random.Random(1)

    def loss_of(i, epoch):
        if i == 0:  # its background is learned away while its subject stays: memorised scenery
            return 1.0, max(0.3, 1.0 - 0.08 * max(0, epoch - 6))
        if i == 1:  # an easy plain background from the start that does not change: not over-learned
            return 1.0, 0.4
        return 1.0, 1.0

    summary, _ = _run(watch, items, 24, loss_of, rng)
    ahead, easy, plain = (watch.stats[items[i].key].contrast("background") for i in (0, 1, 2))
    assert ahead < 0.6
    assert easy == pytest.approx(1.0, abs=0.1) and plain == pytest.approx(1.0, abs=0.1)
    assert summary["rows"][0]["image"] == "img0.png"  # the furthest ahead first
    assert summary["contrast"]["background"]["pictures"] == 20
    assert summary["contrast"]["background"]["median"] == pytest.approx(1.0, abs=0.1)
    assert watch.stats[items[0].key].contrast("clothes") is None  # no clothes cells: no contrast


def test_small_regions_dropped_captions_and_pairs_are_not_read(tmp_path):
    items = _items(tmp_path, 4) + _items(tmp_path / "pairs", 1, pair_side="left")
    watch = RW.RegionWatch(items, tmp_path)
    assert len(watch.stats) == 4 and watch.per_epoch == 5  # the pair counts towards the epoch only
    losses, cells = _row(1.0, 1.0)
    cells[FACE] = 1.0  # under MIN_CELLS
    watch.frozen = True
    for c in RW.COLUMNS:
        watch.loss_curves[c].add(0.5, 1.0)
    watch.observe([items[0].key, items[1].key, items[4].key], [0.5, 0.5, 0.5], [losses, losses, losses],
                  [cells, cells, cells], skip=[False, True, False])
    assert watch.epoch_obs == 3
    assert watch.epoch_n["face"] == 0 and watch.epoch_n["hair"] == 1  # the small face is left out, once
    assert watch.stats[items[1].key].seen == 0  # the dropped caption
    watch.observe([items[2].key], [0.5], [None], [None])  # no parts map
    watch.observe([items[3].key], [0.5], [[float("nan")] * 6], [cells])
    assert watch.unread == 2 and watch.epoch_complete()
    assert "2 samples had no parts map" in RW.describe(watch.end_epoch(5))


def test_resume_reads_the_saved_state_and_a_fresh_run_clears_it(tmp_path):
    items = _items(tmp_path, 20)
    watch = RW.RegionWatch(items, tmp_path)
    rng = random.Random(2)
    _run(watch, items, 8, lambda i, e: (0.8, 1.0), rng)
    again = RW.RegionWatch(items, tmp_path, resume=True)
    assert again.frozen and not again.late and again.epoch == 8
    assert again.loss_curves["face"].counts == watch.loss_curves["face"].counts
    assert again.stats[items[0].key].contrast("background") == watch.stats[items[0].key].contrast("background")
    assert len(again.history) == len(watch.history)
    fresh = RW.RegionWatch(items, tmp_path)  # the same job started again
    assert not fresh.frozen and not (tmp_path / RW.STATE_FILE).exists() and not (tmp_path / RW.LOG_FILE).exists()
    late = RW.RegionWatch(items, tmp_path, resume=True)  # a resume with nothing saved: counts from here, and says so
    assert late.late and not late.frozen


def test_flip_carries_the_parts_map():
    classes = torch.zeros(2, 4, dtype=torch.long)
    classes[:, 0] = 3
    rm = RegionMap(torch.ones(1, 2, 4), torch.zeros(1, 2, 4, dtype=torch.bool), None, None, True,
                   _parts(2, 4, classes), "print", "weights")
    flipped = flip_region_map(rm, 64)
    assert flipped.parts is not None and flipped.parts_print == "print"
    assert bool((flipped.parts[3][:, 3] == 255).all()) and bool((flipped.parts[3][:, 0] == 0).all())
    plain = flip_region_map(RegionMap(torch.ones(1, 2, 4), torch.zeros(1, 2, 4, dtype=torch.bool), None, None, False), 64)
    assert plain.parts is None


def test_config_asks_for_parts_maps_without_touching_hull():
    cfg = TrainConfig(region_watch=True)
    assert cfg.wants_part_maps() and cfg.wants_regions()
    assert not cfg.wants_parts()  # Hull's custom backward (and its no-compile rule) stays Hull's
    assert not cfg.wants_person_masks()
    assert not TrainConfig().wants_part_maps()


def test_a_batch_brings_its_parts_map_flipped_with_the_latent(tmp_path):
    from safetensors.torch import save_file

    from rulztrainer.dataset import load_batch
    from rulztrainer.regions import save_region_map

    cfg = TrainConfig(region_watch=True, flip_augment=True, caption_dropout=0.0)
    save_file({"latent": torch.zeros(128, 4, 8)}, str(tmp_path / "l.safetensors"))
    save_file({"latent": torch.ones(128, 4, 8)}, str(tmp_path / "lf.safetensors"))
    save_file({"embeds": torch.zeros(3, 4096, dtype=torch.bfloat16)}, str(tmp_path / "t.safetensors"))
    classes = torch.zeros(4, 8, dtype=torch.long)
    classes[:, :2] = 3  # the face on the left
    save_region_map(tmp_path / "r.safetensors",
                    RegionMap(torch.ones(1, 4, 8), torch.zeros(1, 4, 8, dtype=torch.bool), None, None, True,
                              _parts(4, 8, classes), "print"))
    it = Item(image=tmp_path / "x.png", caption="c", width=128, height=64, latent_path=tmp_path / "l.safetensors",
              latent_flip_path=tmp_path / "lf.safetensors", text_path=tmp_path / "t.safetensors",
              region_path=tmp_path / "r.safetensors")

    class RngFlip:
        def random(self):
            return 0.0  # always pick the flip

    latents, _, _, regions = load_batch([it], cfg, RngFlip())
    assert regions is not None and float(latents.mean()) == 1.0
    parts = regions.meta[0].parts
    assert parts is not None and bool((parts[3][:, 6:] == 255).all()) and bool((parts[3][:, :6] == 0).all())
    sq = torch.ones(1, 128, 4, 8)
    sq[..., 6:] = 3.0  # the flipped face's cells
    losses, cells = RW.region_losses(sq, [parts])
    assert losses[0][FACE] == pytest.approx(3.0) and cells[0][FACE] == pytest.approx(8.0)
    assert losses[0][BACKGROUND] == pytest.approx(1.0)
    assert str(it.text_path) == regions.text_paths[0]  # no dropped caption: the trainer reads this sample


def test_trainer_makes_the_watch_only_when_it_can(tmp_path):
    from rulztrainer.trainer import Trainer

    items = _items(tmp_path, 3)
    t = Trainer.__new__(Trainer)
    t.job_dir, t.race_rows, t.region_watch_off = tmp_path, None, ""
    t.cfg = TrainConfig()
    assert t._make_region_watch(items, 0) is None
    t.cfg = TrainConfig(region_watch=True)
    watch = t._make_region_watch(items, 0)
    assert watch is not None and len(watch.stats) == 3 and not watch.late
    assert t._make_region_watch(items, 500).late is False  # (not a resume: no resume_state)
    t.race_rows = [("a", t.cfg)]
    assert t._make_region_watch(items, 0) is None  # a race takes turns on one model
    t.race_rows, t.region_watch_off = None, "RuntimeError: no parser"
    assert t._make_region_watch(items, 0) is None
