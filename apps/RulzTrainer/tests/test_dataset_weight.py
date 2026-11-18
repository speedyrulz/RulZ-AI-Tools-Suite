"""Per-dataset loss weight and per-epoch sample cap: the spec's numbers and their validation, old configs without the
keys, the scanner carrying the weight on every item (crops and clips too), and the batcher drawing a fixed number of
items per epoch, freshly and deterministically."""
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import dataset as D  # noqa: E402
from rulztrainer.config import DatasetSpec, TrainConfig  # noqa: E402


def _folder(root: Path, name: str, sizes) -> Path:
    d = root / name
    d.mkdir()
    for i, (w, h) in enumerate(sizes):
        Image.new("RGB", (w, h), (40 * i, 90, 160)).save(d / f"{name}_{i}.png")
        (d / f"{name}_{i}.txt").write_text(f"{name} picture {i}", encoding="utf-8")
    return d


def _item(tmp_path, ds, i, w=512, h=512, repeats=1, scale=1.0):
    return D.Item(image=tmp_path / f"{ds}{i}.png", caption="", width=w, height=h, latent_path=tmp_path / "l",
                  latent_flip_path=None, text_path=tmp_path / "t", dataset=ds, repeats=repeats, loss_scale=scale)


def _epoch(batcher):
    return [it for batch in batcher.epoch() for it in batch]


def test_old_configs_load_unchanged_and_new_keys_round_trip():
    cfg = TrainConfig(datasets=[{"path": "data/a", "repeats": 2}, "data/b"])
    a, b = cfg.dataset_specs()
    assert (a.weight, a.samples, b.weight, b.samples) == (1.0, 0, 1.0, 0)
    assert a == DatasetSpec("data/a", (), 2)
    cfg = TrainConfig(datasets=[{"path": "data/a", "weight": 0.5, "samples": 30}, {"path": "data/b", "weight": None,
                                                                                    "samples": None}])
    a, b = cfg.dataset_specs()
    assert (a.weight, a.samples, b.weight, b.samples) == (0.5, 30, 1.0, 0)


def test_validation(tmp_path):
    for entry, msg in (({"path": "x", "weight": 0}, "weight"), ({"path": "x", "weight": -1}, "weight"),
                       ({"path": "x", "samples": -2}, "samples")):
        with pytest.raises(ValueError, match=msg):
            TrainConfig(datasets=[entry]).validate_datasets()
    TrainConfig(datasets=[{"path": "x", "weight": 0.25, "samples": 12}]).validate_datasets()
    # a parallel race applies the weight like the single-copy step (race_parallel.forward_backward)
    TrainConfig(datasets=[{"path": "x", "weight": 0.5}], race=True, race_parallel=True).validate_datasets()


def test_scan_gives_every_item_its_datasets_weight_crops_too(tmp_path):
    a = _folder(tmp_path, "a", [(512, 512), (512, 512)])
    b = _folder(tmp_path, "b", [(512, 512)])
    kw = dict(resolutions=[0.25], min_bucket_side=64, crop_augment="face", crop_augment_min_px=8,
              crop_skip_face_frac=1.0, crop_skip_cover=1.0, crop_face_scale=2.0)
    cfg = TrainConfig(datasets=[{"path": str(a)}, {"path": str(b), "weight": 0.4, "samples": 5}], **kw)
    items = D.scan_dataset(cfg)
    assert {it.loss_scale for it in items if it.dataset == str(a)} == {1.0}
    assert {it.loss_scale for it in items if it.dataset == str(b)} == {0.4}
    analysis = {str(it.image.resolve()): {"bbox": [200, 200, 300, 300], "det": 0.9, "W": 512, "H": 512}
                for it in items}
    crops = [it for it in D.augment_items(items, cfg, analysis) if it.category == "face_crop"]
    assert crops and {c.loss_scale for c in crops if c.dataset == str(b)} == {0.4}


def test_scan_logs_a_line_per_non_default_dataset(tmp_path, caplog):
    import logging

    a = _folder(tmp_path, "a", [(512, 512)])
    b = _folder(tmp_path, "b", [(512, 512)])
    cfg = TrainConfig(resolutions=[0.25], min_bucket_side=64,
                      datasets=[{"path": str(a)}, {"path": str(b), "weight": 0.5, "samples": 3, "repeats": 4}])
    with caplog.at_level(logging.INFO, logger="rulztrainer"):
        D.scan_dataset(cfg)
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert f"Dataset {b}: loss weight 0.5" in text
    assert f"Dataset {b}: at most 3 of its 4 pictures" in text
    assert f"Dataset {a}: loss weight" not in text and f"Dataset {a}: at most" not in text


def test_samples_cap_the_epoch_and_draw_afresh_each_epoch(tmp_path):
    big = [_item(tmp_path, "big", i) for i in range(40)]
    small = [_item(tmp_path, "small", i) for i in range(6)]
    batcher = D.BucketBatcher(big + small, batch_size=4, seed=3, dataset_samples={"big": 10})
    assert batcher.epoch_images() == 16
    seen = []
    for _ in range(6):
        ep = _epoch(batcher)
        assert len(ep) == 16 and sum(1 for it in ep if it.dataset == "big") == 10
        assert sorted(it.key for it in ep if it.dataset == "small") == sorted(it.key for it in small)
        assert len({it.key for it in ep}) == 16  # no picture twice in one epoch
        seen.append(frozenset(it.key for it in ep if it.dataset == "big"))
    assert len(set(seen)) > 1  # a different draw each epoch
    assert len(batcher.epoch()) == 4  # 16 items in batches of 4: the same step count every epoch


def test_samples_are_deterministic_for_a_seed_and_differ_between_seeds(tmp_path):
    items = [_item(tmp_path, "d", i) for i in range(30)]

    def draws(seed):
        b = D.BucketBatcher(items, batch_size=1, seed=seed, dataset_samples={"d": 7})
        return [sorted(it.key for it in _epoch(b)) for _ in range(3)]

    assert draws(5) == draws(5) and draws(5) != draws(6)


def test_cap_counts_after_repeats_and_splits_over_size_buckets(tmp_path):
    sq = [_item(tmp_path, "d", i, 512, 512, repeats=2) for i in range(10)]  # 20 entries
    wide = [_item(tmp_path, "d", 100 + i, 768, 384, repeats=2) for i in range(5)]  # 10 entries
    batcher = D.BucketBatcher(sq + wide, batch_size=1, seed=0, dataset_samples={"d": 15})
    for _ in range(4):
        ep = _epoch(batcher)
        assert len(ep) == 15
        assert sum(1 for it in ep if it.width == 512) == 10 and sum(1 for it in ep if it.width == 768) == 5
    assert batcher.epoch_images() == 15


def test_cap_at_or_above_the_total_changes_nothing_and_draws_nothing(tmp_path):
    items = [_item(tmp_path, "d", i) for i in range(8)]
    plain = D.BucketBatcher(items, batch_size=2, seed=1)
    capped = D.BucketBatcher(items, batch_size=2, seed=1, dataset_samples={"d": 8, "other": 3})
    assert [[it.key for it in b] for b in plain.epoch()] == [[it.key for it in b] for b in capped.epoch()]
    assert capped.epoch_images() == 8


def test_cap_composes_with_sampling_weights_and_regularisation(tmp_path):
    big = [_item(tmp_path, "big", i) for i in range(20)]
    reg = [D.Item(image=tmp_path / f"r{i}.png", caption="", width=512, height=512, latent_path=tmp_path / "l",
                  latent_flip_path=None, text_path=tmp_path / "t", reg=True) for i in range(30)]
    batcher = D.BucketBatcher(big + reg, batch_size=1, seed=2, reg_per_epoch=-1, dataset_samples={"big": 6})
    assert batcher.epoch_images() == 12  # 6 subject + as many regularisation pictures as the subject trains
    batcher.set_weights({it.key: 1.0 for it in big})
    ep = _epoch(batcher)
    assert len(ep) == 12 and sum(1 for it in ep if it.reg) == 6



def test_cap_holds_with_a_high_sampling_weight(tmp_path):
    """The curriculum's weights act inside a capped dataset's share: the dataset still gives exactly its quota, and the
    bucket's other entries keep their count, so the epoch's step count does not change."""
    big = [_item(tmp_path, "big", i) for i in range(30)]
    small = [_item(tmp_path, "small", i) for i in range(10)]
    batcher = D.BucketBatcher(big + small, batch_size=1, seed=4, dataset_samples={"big": 8})
    batcher.set_weights({**{it.key: 1.0 for it in small}, **{it.key: 1.0 for it in big}, big[0].key: 50.0})
    for _ in range(8):
        ep = _epoch(batcher)
        assert len(ep) == 18 and sum(1 for it in ep if it.dataset == "big") == 8
        assert sum(1 for it in ep if it.dataset == "small") == 10
    assert batcher.epoch_images() == 18


def test_bucket_sizes_give_the_capped_epoch(tmp_path):
    items = [_item(tmp_path, "d", i) for i in range(20)]
    batcher = D.BucketBatcher(items, batch_size=1, seed=0, dataset_samples={"d": 6})
    assert sum(batcher.bucket_sizes.values()) == 6 == batcher.epoch_images()
