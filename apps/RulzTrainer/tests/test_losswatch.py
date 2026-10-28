import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer.config import TrainConfig  # noqa: E402
from rulztrainer.dataset import Item  # noqa: E402
from rulztrainer.losswatch import LossWatch, read_verdicts  # noqa: E402


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
    assert 0.5 in ladder and 0.25 in ladder and ladder[-1] == 0.125
    cfg2 = TrainConfig(loss_watch=True, per_image_lr=False)
    watch2 = LossWatch(cfg2, items, tmp_path)
    assert watch2.multiplier(str(items[0].image)) == 1.0  # observe-only never scales
