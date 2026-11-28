"""Regularisation buckets worth under one picture still train (the leftover places are drawn per epoch), and the
picture scan does not decode a PNG that has no eXIf chunk."""
import sys
from pathlib import Path

from PIL import Image, PngImagePlugin

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import dataset as D  # noqa: E402
from rulztrainer import facesim  # noqa: E402


def _item(name, size=(64, 64), reg=False):
    p = Path(name)
    return D.Item(image=p, caption=name, width=size[0], height=size[1], latent_path=p, latent_flip_path=None,
                  text_path=p, repeats=1, reg=reg)


def _counts(items, per_epoch, epochs, seed=0, subject_n=1):
    subject = [_item(f"s{i}") for i in range(subject_n)]
    b = D.BucketBatcher(subject + items, 1, seed, reg_per_epoch=per_epoch)
    seen = {}
    totals = set()
    for _ in range(epochs):
        flat = [it for batch in b.epoch() for it in batch if it.reg]
        totals.add(len(flat))
        for it in flat:
            seen[it.caption[0]] = seen.get(it.caption[0], 0) + 1
    return seen, totals


def test_three_one_picture_buckets_each_get_their_share():
    reg = [_item("a0", (64, 64), True), _item("b0", (96, 64), True), _item("c0", (64, 96), True)]
    seen, totals = _counts(reg, 1, 60)
    assert totals == {1}
    assert sum(seen.values()) == 60 and all(15 <= seen.get(k, 0) <= 25 for k in "abc"), seen


def test_small_buckets_are_drawn_in_proportion():
    reg = [_item(f"a{i}", (64, 64), True) for i in range(98)] + [_item("b0", (96, 64), True), _item("c0", (64, 96), True)]
    seen, totals = _counts(reg, 10, 100)
    assert totals == {10}
    assert seen["b"] > 0 and seen["c"] > 0
    assert 3 <= seen["b"] <= 25 and 3 <= seen["c"] <= 25, seen  # expected 10 each (1% of 1000)


def test_same_seed_same_draws_and_total_unchanged():
    reg = [_item("a0", (64, 64), True), _item("b0", (96, 64), True), _item("c0", (64, 96), True)]
    one = []
    for seed in (3, 3, 4):
        b = D.BucketBatcher([_item("s0")] + reg, 1, seed, reg_per_epoch=1)
        one.append([sorted(it.caption for batch in b.epoch() for it in batch if it.reg) for _ in range(20)])
    assert one[0] == one[1] and one[0] != one[2]


def test_capped_dataset_buckets_get_their_share_too():
    items = [_item(f"a{i}") for i in range(2)] + [_item("b0", (96, 64)), _item("c0", (64, 96))]
    for it in items:
        it.dataset = "ds"
    b = D.BucketBatcher(items, 1, 5, dataset_samples={"ds": 1})
    seen = {}
    for _ in range(80):
        flat = [it for batch in b.epoch() for it in batch]
        assert len(flat) == 1
        seen[flat[0].caption[0]] = seen.get(flat[0].caption[0], 0) + 1
    assert seen.get("b", 0) > 5 and seen.get("c", 0) > 5, seen


# ---------------------------------------------------------------- PNG scan


def _count_loads(monkeypatch):
    calls = []
    orig = Image.Image.load

    def counting(self, *a, **k):
        calls.append(type(self).__name__)
        return orig(self, *a, **k)

    monkeypatch.setattr(Image.Image, "load", counting)
    monkeypatch.setattr(PngImagePlugin.PngImageFile, "load", counting)
    return calls


def _exif(orientation):
    ex = Image.Exif()
    ex[0x0112] = orientation
    return ex.tobytes()


def test_png_without_exif_is_not_decoded(tmp_path, monkeypatch):
    p = tmp_path / "plain.png"
    Image.new("RGB", (40, 20), (1, 2, 3)).save(p)
    calls = _count_loads(monkeypatch)
    assert D.upright_header(p)[:2] == (40, 20)
    assert facesim.image_size(p) == (40, 20)
    assert not any(c == "PngImageFile" for c in calls), calls


def test_png_with_exif_orientation_swaps(tmp_path):
    p = tmp_path / "rot.png"
    Image.new("RGB", (40, 20), (1, 2, 3)).save(p, exif=_exif(6))
    assert D.upright_header(p)[:2] == (20, 40)
    assert D.upright_size(p) == (20, 40)
    assert facesim.image_size(p) == (20, 40)


def test_png_exif_chunk_walk(tmp_path):
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    Image.new("RGB", (8, 8)).save(a)
    Image.new("RGB", (8, 8)).save(b, exif=_exif(3))
    assert D.png_has_exif_chunk(a) is False and D.png_has_exif_chunk(b) is True


def test_jpeg_orientation_unchanged(tmp_path):
    p = tmp_path / "rot.jpg"
    Image.new("RGB", (40, 20), (9, 9, 9)).save(p, exif=_exif(8))
    q = tmp_path / "flat.jpg"
    Image.new("RGB", (40, 20)).save(q)
    assert D.upright_header(p)[:2] == (20, 40) and facesim.image_size(p) == (20, 40)
    assert D.upright_header(q)[:2] == (40, 20) and facesim.image_size(q) == (40, 20)
