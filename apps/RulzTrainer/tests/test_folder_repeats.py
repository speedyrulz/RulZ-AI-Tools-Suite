"""Repeats from folder names (folder_repeats, kohya style): a picture in a sub-folder named "<N>_<name>" trains N times per
epoch, times its dataset's repeats; the nearest such folder below the dataset folder counts, the dataset folder never."""
import logging
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import dataset as D  # noqa: E402
from rulztrainer.config import TrainConfig  # noqa: E402


def _pic(folder: Path, name: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64)).save(folder / f"{name}.png")
    (folder / f"{name}.txt").write_text("a picture", encoding="utf-8")
    return folder / f"{name}.png"


def _tree(root: Path) -> None:
    _pic(root, "top")
    _pic(root / "10_portraits", "a")
    _pic(root / "10_portraits" / "3_close", "b")  # the nearest folder wins
    _pic(root / "10_portraits" / "plain", "c")  # no rule of its own: the outer one
    _pic(root / "0_none", "d")  # N must be 1 or more
    _pic(root / "5_", "e")  # no name after the number
    _pic(root / "2x_things", "f")  # not "<N>_"
    _pic(root / "outer" / "4_inner", "g")


def _repeats(cfg) -> dict:
    return {it.image.stem: it.repeats for it in D.scan_dataset(cfg)}


def test_off_by_default_every_picture_trains_once(tmp_path):
    _tree(tmp_path)
    assert TrainConfig().folder_repeats is False
    assert set(_repeats(TrainConfig(dataset_dir=str(tmp_path), min_bucket_side=64)).values()) == {1}


def test_folder_names_give_the_repeats(tmp_path):
    _tree(tmp_path)
    cfg = TrainConfig(dataset_dir=str(tmp_path), min_bucket_side=64, folder_repeats=True)
    assert _repeats(cfg) == {"top": 1, "a": 10, "b": 3, "c": 10, "d": 1, "e": 1, "f": 1, "g": 4}


def test_the_dataset_folder_itself_never_counts_and_repeats_multiply(tmp_path):
    root = tmp_path / "7_dataset"
    _pic(root, "top")
    _pic(root / "2_sub", "a")
    cfg = TrainConfig(min_bucket_side=64, folder_repeats=True, datasets=[{"path": str(root), "repeats": 3}])
    assert _repeats(cfg) == {"top": 3, "a": 6}


def test_the_folders_are_logged_and_the_batcher_counts_them(tmp_path, caplog):
    _tree(tmp_path)
    cfg = TrainConfig(dataset_dir=str(tmp_path), min_bucket_side=64, folder_repeats=True)
    with caplog.at_level(logging.INFO, logger="rulztrainer"):
        items = D.scan_dataset(cfg)
    assert "folder 10_portraits: 2 pictures, 10 repeats" in caplog.text
    assert "folder 10_portraits/3_close: 1 picture, 3 repeats" in caplog.text
    assert D.BucketBatcher(items, 1, 0).epoch_images() == 1 + 10 + 3 + 10 + 1 + 1 + 1 + 4


def test_the_regularisation_set_ignores_folder_names(tmp_path):
    subject, reg = tmp_path / "subject", tmp_path / "reg"
    _pic(subject, "s")
    _pic(reg / "10_reg", "r")
    cfg = TrainConfig(dataset_dir=str(subject), reg_dir=str(reg), min_bucket_side=64, folder_repeats=True)
    assert {it.repeats for it in D.scan_reg(cfg)} == {1}
