"""Several datasets per run: the config's dataset list (fallback to dataset_dir, per-dataset resolutions, the union of
training resolutions, validation, caches), the scanner (each dataset at its own resolutions, with its repeats and its
own latent cache), repeats in the batcher, and crop items inheriting both."""
import random
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


def test_dataset_specs_fall_back_to_dataset_dir_and_sides_are_the_union():
    cfg = TrainConfig(dataset_dir="data/one", resolutions=[1.0])
    assert cfg.dataset_specs() == [DatasetSpec("data/one")] and cfg.training_sides() == [1024]
    cfg = TrainConfig(resolutions=[1.0], datasets=[
        {"path": "data/a"},  # the run's resolutions, once per epoch
        {"path": "data/b", "resolutions": [0.25, 4.0], "repeats": 3},
        "data/c"])
    a, b, c = cfg.dataset_specs()
    assert a == DatasetSpec("data/a") and b == DatasetSpec("data/b", (0.25, 4.0), 3) and c.path == "data/c"
    assert cfg.dataset_sides(a) == [1024] and cfg.dataset_sides(b) == [512, 2048]
    assert cfg.training_sides() == [512, 1024, 2048]
    assert cfg.resolved_cache_dir() == Path("data/a/.rulz_cache")  # the run's own files: the first dataset's
    assert cfg.dataset_cache_dir("data/b") == Path("data/b/.rulz_cache")
    shared = TrainConfig(cache_dir="cache", datasets=[{"path": "data/a"}, {"path": "data/b"}])
    assert shared.resolved_cache_dir() == Path("cache") and shared.dataset_cache_dir("data/b") == Path("cache")


def test_dataset_list_validation(tmp_path):
    TrainConfig(datasets=[{"path": "a"}, {"path": "b", "repeats": 2, "resolutions": [0.5]}]).validate()
    for bad, msg in (([{"path": "a"}, {"path": "./a"}], "listed twice"),
                     ([{"path": str(tmp_path)}, {"path": str(tmp_path / "sub")}], "lies inside"),
                     ([{"path": "a", "repeats": 0}], "repeats"),
                     ([{"path": "a", "resolutions": [20.0]}], "megapixel"),
                     ([{"path": "a", "repat": 2}], "unknown: repat"),
                     ([{"path": ""}], "No dataset")):
        with pytest.raises(ValueError, match=msg):
            TrainConfig(datasets=bad).validate()
    with pytest.raises(ValueError, match="No dataset"):
        TrainConfig(dataset_dir="").validate()


def test_scan_each_dataset_at_its_resolutions_with_its_repeats_and_cache(tmp_path):
    a = _folder(tmp_path, "a", [(640, 480), (480, 640)])
    b = _folder(tmp_path, "b", [(800, 800)])
    cfg = TrainConfig(resolutions=[1.0], min_bucket_side=256, datasets=[
        {"path": str(a)}, {"path": str(b), "resolutions": [0.25, 1.0], "repeats": 3}])
    items = D.scan_dataset(cfg)
    by = {name: [it for it in items if it.dataset == str(tmp_path / name)] for name in ("a", "b")}
    assert len(by["a"]) == 2 and {it.side for it in by["a"]} == {1024} and all(it.repeats == 1 for it in by["a"])
    assert len(by["b"]) == 2 and {it.side for it in by["b"]} == {512, 1024} and all(it.repeats == 3 for it in by["b"])
    assert all(it.latent_path.parent == a / ".rulz_cache" / "latents" for it in by["a"])  # each dataset its own cache
    assert all(it.latent_path.parent == b / ".rulz_cache" / "latents" for it in by["b"])
    assert all(it.text_path.parent == a / ".rulz_cache" / "text" for it in items)  # captions: the run's cache
    assert by["a"][0].caption == "a picture 0"
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="No images found"):
        D.scan_dataset(TrainConfig(datasets=[{"path": str(a)}, {"path": str(tmp_path / "empty")}]))
    with pytest.raises(FileNotFoundError):
        D.scan_dataset(TrainConfig(datasets=[{"path": str(tmp_path / "missing")}]))


def test_repeats_in_the_batcher_and_crop_items(tmp_path):
    a = _folder(tmp_path, "a", [(512, 512)])
    b = _folder(tmp_path, "b", [(512, 512)])
    cfg = TrainConfig(resolutions=[0.25], min_bucket_side=256,
                      datasets=[{"path": str(a)}, {"path": str(b), "repeats": 3}])
    items = D.scan_dataset(cfg)
    batcher = D.BucketBatcher(items, batch_size=1, seed=0)
    epoch = [x for batch in batcher.epoch() for x in batch]
    assert len(epoch) == 4 and sum(1 for it in epoch if it.dataset == str(b)) == 3
    batcher.set_weights({items[0].key: 1.0})  # the curriculum's weighted draw keeps the repeated entries
    assert len([x for batch in batcher.epoch() for x in batch]) == 4
    crop_cfg = TrainConfig(resolutions=[0.25], min_bucket_side=64, crop_augment="face", crop_augment_min_px=8,
                           crop_skip_face_frac=1.0, crop_skip_cover=1.0, crop_face_scale=2.0,
                           datasets=[{"path": str(a)}, {"path": str(b), "repeats": 3}])
    base = D.scan_dataset(crop_cfg)
    analysis = {str(it.image.resolve()): {"bbox": [200, 200, 300, 300], "det": 0.9, "W": 512, "H": 512}
                for it in base}
    crops = [it for it in D.augment_items(base, crop_cfg, analysis) if it.category == "face_crop"]
    assert len(crops) == 2
    for crop in crops:
        src = next(it for it in base if it.image == crop.image)
        assert crop.repeats == src.repeats and crop.dataset == src.dataset
        assert crop.latent_path.parent == src.latent_path.parent  # next to its image's latents
    random.seed(0)
