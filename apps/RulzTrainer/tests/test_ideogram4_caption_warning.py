"""Ideogram 4: the one-time warning at dataset load when the captions are mostly short plain text."""
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import dataset as D  # noqa: E402
from rulztrainer.config import TrainConfig  # noqa: E402

LONG = " ".join(f"word{i}" for i in range(40))


def _folder(root: Path, captions: list, json_captions: dict | None = None) -> Path:
    d = root / "data"
    d.mkdir()
    for i, text in enumerate(captions):
        Image.new("RGB", (64, 64), (10 * i, 90, 160)).save(d / f"p{i}.png")
        (d / f"p{i}.txt").write_text(text, encoding="utf-8")
    for name, body in (json_captions or {}).items():
        Image.new("RGB", (64, 64), (200, 90, 10)).save(d / f"{name}.png")
        (d / f"{name}.json").write_text(json.dumps(body), encoding="utf-8")
    return d


def _warning(root: Path, arch="ideogram4", **kw):
    cfg = TrainConfig(model_arch=arch, datasets=[{"path": str(root)}], resolutions=[0.25], min_bucket_side=64, **kw)
    return D.ideogram4_caption_warning(D.scan_dataset(cfg), cfg)


def test_short_plain_captions_warn(tmp_path):
    msg = _warning(_folder(tmp_path, ["a woman in a red dress", "a man on a beach", "a dog"]))
    assert msg and "3 of 3 captions" in msg and "refusal" in msg and "JSON" in msg


def test_rich_plain_captions_do_not_warn(tmp_path):
    assert _warning(_folder(tmp_path, [LONG, LONG + " more", LONG])) is None


def test_the_median_decides_not_a_single_short_caption(tmp_path):
    (tmp_path / "x").mkdir()
    (tmp_path / "y").mkdir()
    assert _warning(_folder(tmp_path / "x", ["short one", LONG, LONG, LONG])) is None
    assert _warning(_folder(tmp_path / "y", ["short one", "another short", "tiny", LONG])) is not None


def test_json_captions_do_not_count_as_plain(tmp_path):
    body = {"high_level_description": "a cat", "compositional_deconstruction": {"background": "", "elements": []}}
    root = _folder(tmp_path, ["short plain"], {"j1": body, "j2": body})
    assert _warning(root) is None  # two of the three pictures have structured captions: mostly structured


def test_only_ideogram4_runs_are_checked(tmp_path):
    root = _folder(tmp_path, ["a woman", "a man", "a dog"])
    assert _warning(root, arch="flux2_klein") is None


def test_the_trigger_word_does_not_make_a_caption_long(tmp_path):
    msg = _warning(_folder(tmp_path, ["a woman", "a man", "a dog"]), trigger_word="ohwx person")
    assert msg is not None
