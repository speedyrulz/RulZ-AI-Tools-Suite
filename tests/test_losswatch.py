import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer.config import TrainConfig  # noqa: E402
from rulztrainer.dataset import Item  # noqa: E402
from rulztrainer.losswatch import STUCK_LADDER, LossWatch, outlier_fence, read_verdicts  # noqa: E402


def _items(tmp_path, n):
    return [Item(image=tmp_path / f"img{i}.png", caption=f"cap {i}", width=512, height=512,
                 latent_path=tmp_path / "l", latent_flip_path=None, text_path=tmp_path / "t") for i in range(n)]


def test_verdicts_throttle_and_exclusion(tmp_path):
    cfg = TrainConfig(loss_watch=True, per_image_lr=True, loss_watch_exclude_after=1, loss_watch_improve_tol=0.03)
    items = _items(tmp_path, 6)
    watch = LossWatch(cfg, items, tmp_path, tmp_path / "excluded.json")
    # sigma-dependent base loss so normalisation matters; image 0 is hard and flat, image 5 is easy and flat,
    # image 1 is hard but improving, the rest are normal.
    import random

    rng = random.Random(0)
    step = 0
    for epoch in range(14):
        for i, it in enumerate(items):
            sigma = rng.random()
            base = 0.1 + 0.9 * sigma  # expected loss grows with sigma
            factor = {0: 2.0, 1: max(0.9, 2.5 / (1 + epoch)), 5: 0.4}.get(i, 1.0)
            step += 1
            watch.observe(step, str(it.image), base * factor, sigma)
        assert watch.epoch_complete()
        summary = watch.end_epoch(step)
    s0, s1, s5 = (watch.stats[str(items[i].image)] for i in (0, 1, 5))
    assert s0.verdict == "excluded" and s0.multiplier == 0.0 and s0.excluded
    assert s1.verdict in ("suspect", "easy", "healthy")  # improving image is never throttled
    assert s5.verdict == "exhausted" and s5.multiplier == 0.6
    assert watch.multiplier(str(items[5].image)) == 0.6 and watch.multiplier(str(items[0].image)) == 0.0
    assert summary["counts"]["excluded"] == 1 and (tmp_path / "per_image_loss.json").exists()
    assert read_verdicts(tmp_path)[str(items[0].image)] == "excluded"
    # the exclusion persists for the same caption and is cleared when the caption changes
    watch2 = LossWatch(cfg, items, tmp_path, tmp_path / "excluded.json")
    assert watch2.stats[str(items[0].image)].excluded
    items[0].caption = "edited caption"
    watch3 = LossWatch(cfg, items, tmp_path, tmp_path / "excluded.json")
    assert not watch3.stats[str(items[0].image)].excluded
    assert watch2.sampling_mask()[str(items[0].image)] == 0.0


def test_stuck_ladder(tmp_path):
    cfg = TrainConfig(loss_watch=True, per_image_lr=True)
    items = _items(tmp_path, 4)
    watch = LossWatch(cfg, items, tmp_path)
    step = 0
    ladder = []
    for epoch in range(8):
        for i, it in enumerate(items):
            step += 1
            watch.observe(step, str(it.image), 1.0 * (3.0 if i == 0 else 1.0), 0.5)
        watch.end_epoch(step)
        ladder.append(watch.stats[str(items[0].image)].multiplier)
    assert 0.5 in ladder and ladder[-1] == 0.25 and min(ladder) == 0.25  # the ladder stops at x0.25
    cfg2 = TrainConfig(loss_watch=True, per_image_lr=False)
    watch2 = LossWatch(cfg2, items, tmp_path)
    assert watch2.multiplier(str(items[0].image)) == 1.0  # observe-only never scales


def _flat_run(tmp_path, cfg, factors, epochs=8):
    """Every image at its own constant multiple of the loss, so nothing improves: {index: verdict} at the end."""
    items = _items(tmp_path, len(factors))
    watch = LossWatch(cfg, items, tmp_path)
    step = 0
    for _ in range(epochs):
        for it, factor in zip(items, factors):
            step += 1
            watch.observe(step, str(it.image), factor, 0.5)
        summary = watch.end_epoch(step)
    return [watch.stats[str(it.image)].verdict for it in items], summary


def test_outlier_fence_spares_the_hard_end_of_a_spread(tmp_path):
    """A dataset whose losses are simply spread out, none improving (late in a run): the median rule alone calls the
    whole hard end stuck; with the fence only the image far outside the rest is."""
    factors = [0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.1, 1.2, 1.6, 1.7, 1.8, 4.0]
    old, _ = _flat_run(tmp_path / "old", TrainConfig(loss_watch=True, per_image_lr=True, loss_watch_fence=0.0), factors)
    new, summary = _flat_run(tmp_path / "new", TrainConfig(loss_watch=True, per_image_lr=True), factors)
    assert [i for i, v in enumerate(old) if v == "stuck"] == [8, 9, 10, 11]
    assert [i for i, v in enumerate(new) if v == "stuck"] == [11]
    assert [new[i] for i in (8, 9, 10)] == ["suspect"] * 3  # hard and watched, at the full rate
    assert summary["fence"] is not None and summary["fence"] > summary["median"]


def test_outlier_fence_values():
    import math

    assert outlier_fence([1, 2, 3, 4, 5], 3, 1.5) == 3 + 1.5 * (4 - 2)
    assert outlier_fence([1, 2, 3, 4, 5], 3, 0) == -math.inf  # off
    assert outlier_fence([1, 2, 9], 2, 1.5) == -math.inf  # too few images for quartiles: the median rule alone


def test_long_stuck_image_recovers_a_rung_an_epoch(tmp_path):
    """An image held at the lowest rung for longer than the ladder is long, then improving: back up one rung per
    epoch (before 3.7.1 this read past the ladder's end and stopped the run with an IndexError)."""
    cfg = TrainConfig(loss_watch=True, per_image_lr=True)
    items = _items(tmp_path, 6)
    watch = LossWatch(cfg, items, tmp_path)
    hard = watch.stats[str(items[0].image)]
    step, mults = 0, []
    for epoch in range(24):
        for i, it in enumerate(items):
            step += 1
            factor = 1.0 if i else (4.0 if epoch < 8 else 4.0 * 0.6 ** (epoch - 7))  # flat, then easier every epoch
            watch.observe(step, str(it.image), factor, 0.5)
        watch.end_epoch(step)
        mults.append(hard.multiplier)
    last_low = max(i for i, m in enumerate(mults) if m == 0.25)
    assert mults.count(0.25) > len(STUCK_LADDER) + 1  # held at the lowest rung for longer than the ladder is long
    assert mults[last_low + 1:last_low + 3] == [0.5, 1.0], mults
    assert all(m == 1.0 for m in mults[last_low + 2:]), mults
