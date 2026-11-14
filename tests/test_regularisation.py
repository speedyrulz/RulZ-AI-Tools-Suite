"""The regularisation dataset (reg_dir): a second folder of ordinary captioned pictures trained beside the subject's.
The config's checks, the scan (its own items, captions without a trigger word, captions required), each epoch's share
(a fixed count per size bucket, the folder walked through before any picture comes up again), captions that are never
dropped, and a real run on a tiny model: the loss watch and everything that reads the subject never see its pictures."""
import dataclasses
import json
import logging
import random
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import dataset as D  # noqa: E402
from rulztrainer import trainer as T  # noqa: E402
from rulztrainer.config import TrainConfig  # noqa: E402

from test_run_safety import _go, _job, _picture, _status, rig  # noqa: E402,F401  (rig: the tiny-model fixture)


def _folder(root: Path, name: str, sizes, caption="a person, picture {i}") -> Path:
    d = root / name
    d.mkdir()
    for i, (w, h) in enumerate(sizes):
        Image.new("RGB", (w, h), (30 * i % 255, 90, 160)).save(d / f"{name}_{i}.png")
        if caption is not None:
            (d / f"{name}_{i}.txt").write_text(caption.format(i=i), encoding="utf-8")
    return d


def _item(name: str, size=(64, 64), reg=False, repeats=1) -> D.Item:
    p = Path(name)
    return D.Item(image=p, caption=name, width=size[0], height=size[1], latent_path=p, latent_flip_path=None,
                  text_path=p, repeats=repeats, reg=reg)


# ---------------------------------------------------------------- the config


def test_off_by_default_and_the_numbers_are_checked(tmp_path):
    assert TrainConfig().reg_dir == "" and D.scan_reg(TrainConfig()) == []
    TrainConfig(dataset_dir="a", reg_dir="b").validate()
    TrainConfig(dataset_dir="a", reg_dir="b", reg_weight=0.3, reg_per_epoch=0).validate()
    for kw, msg in ((dict(reg_weight=0.0), "reg_weight"), (dict(reg_weight=-1.0), "reg_weight"),
                    (dict(reg_per_epoch=-2), "reg_per_epoch"), (dict(isolation=True, trigger_word="ohwx"), "Isolation"),
                    (dict(finetune=True), "fine-tune")):
        with pytest.raises(ValueError, match=msg):
            TrainConfig(dataset_dir="a", reg_dir="b", **kw).validate()
    # the same settings are fine while the regularisation dataset is off
    TrainConfig(dataset_dir="a", reg_weight=0.0, reg_per_epoch=-5).validate()


def test_the_folder_is_apart_from_every_dataset(tmp_path):
    sub = tmp_path / "set" / "more"
    for reg, msg in ((tmp_path / "set", "itself"), (sub, "lies inside the dataset"), (tmp_path, "lies inside reg_dir")):
        with pytest.raises(ValueError, match=msg):
            TrainConfig(dataset_dir=str(tmp_path / "set"), reg_dir=str(reg)).validate()
        with pytest.raises(ValueError, match=msg):
            TrainConfig(datasets=[{"path": str(tmp_path / "other")}, {"path": str(tmp_path / "set")}],
                        reg_dir=str(reg)).validate()
    TrainConfig(dataset_dir=str(tmp_path / "set"), reg_dir=str(tmp_path / "set2")).validate()  # a name it starts with


# ---------------------------------------------------------------- the scan


def test_scan_gives_plain_items_of_their_own(tmp_path):
    subject = _folder(tmp_path, "subject", [(640, 480)])
    reg = _folder(tmp_path, "reg", [(640, 480), (480, 640), (512, 512)])
    (reg / "deeper").mkdir()
    Image.new("RGB", (300, 300)).save(reg / "deeper" / "x.png")
    (reg / "deeper" / "x.txt").write_text("someone else again", encoding="utf-8")
    (reg / "clip.mp4").write_bytes(b"not a picture")
    cfg = TrainConfig(dataset_dir=str(subject), reg_dir=str(reg), trigger_word="ohwx", resolutions=[0.25, 1.0],
                      min_bucket_side=256, reg_weight=0.4, flip_augment=True)
    assert all(not it.reg and it.caption.startswith("ohwx") for it in D.scan_dataset(cfg))  # the datasets: untouched
    items = D.scan_reg(cfg)
    assert len({it.image for it in items}) == 4 and len(items) == 8  # every picture at each resolution of the run
    assert {it.side for it in items} == {512, 1024}
    for it in items:
        assert it.reg and it.loss_scale == 0.4 and it.repeats == 1 and it.category == "image"
        assert "ohwx" not in it.caption and it.caption  # its own caption, no trigger word
        assert it.region_path is None and it.dop_caption is None
        assert it.latent_path.parent == reg / ".rulz_cache" / "latents"  # the folder's own cache
        assert it.latent_flip_path is not None and it.text_path == D.text_cache_path(cfg, it.caption)
        assert it.dataset == str(reg)
    assert not (reg / ".rulz_cache" / "latents" / "nothing").exists() and D.scan_reg(cfg)  # (its cache folder is skipped)


def test_scan_refuses_pictures_without_a_caption_and_names_them(tmp_path):
    subject = _folder(tmp_path, "subject", [(64, 64)])
    reg = _folder(tmp_path, "reg", [(64, 64)] * 2)
    for i in range(7):
        Image.new("RGB", (64, 64)).save(reg / f"bare_{i}.png")
    (reg / "bare_0.txt").write_text("  \n", encoding="utf-8")  # an empty caption file is no caption
    cfg = TrainConfig(dataset_dir=str(subject), reg_dir=str(reg))
    with pytest.raises(ValueError, match=r"7 regularisation picture\(s\).*no caption: bare_0.png.*and 2 more"):
        D.scan_reg(cfg)
    # a family whose captions are JSON wraps an empty one: the caption file itself is what counts
    with pytest.raises(ValueError, match="no caption"):
        D.scan_reg(dataclasses.replace(cfg, model_arch="ideogram4"))
    for p in reg.glob("bare_*.png"):
        p.with_suffix(".txt").write_text("someone else", encoding="utf-8")
    (reg / "reg_0.json").write_text("{}", encoding="utf-8")  # a JSON caption that says nothing (it wins over the .txt)
    with pytest.raises(ValueError, match=r"1 regularisation picture\(s\).*no caption: reg_0.png"):
        D.scan_reg(dataclasses.replace(cfg, model_arch="ideogram4"))
    assert len(D.scan_reg(cfg)) == 9  # a family that reads the .txt captions: every picture has one


def test_scan_missing_and_empty_folders(tmp_path):
    subject = _folder(tmp_path, "subject", [(64, 64)])
    with pytest.raises(FileNotFoundError, match="Regularisation folder not found"):
        D.scan_reg(TrainConfig(dataset_dir=str(subject), reg_dir=str(tmp_path / "missing")))
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="No images found in the regularisation folder"):
        D.scan_reg(TrainConfig(dataset_dir=str(subject), reg_dir=str(tmp_path / "empty")))


def test_scan_warns_about_a_trigger_word_in_a_caption(tmp_path, caplog):
    subject = _folder(tmp_path, "subject", [(64, 64)])
    other = _folder(tmp_path, "other", [(64, 64)])
    reg = _folder(tmp_path, "reg", [(64, 64)] * 3)
    (reg / "reg_1.txt").write_text("a photo of OHWX at the beach", encoding="utf-8")
    (reg / "reg_2.txt").write_text("sks person, smiling", encoding="utf-8")
    cfg = TrainConfig(datasets=[{"path": str(subject), "trigger": "ohwx"}, {"path": str(other), "trigger": "sks"}],
                      reg_dir=str(reg))
    assert D.reg_triggers(cfg) == ["ohwx", "sks"]
    with caplog.at_level(logging.WARNING, logger="rulztrainer"):
        assert len(D.scan_reg(cfg)) == 3  # they still train: a warning, not a refusal
    assert "2 caption(s) use a trigger word" in caplog.text and "reg_1.png, reg_2.png" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="rulztrainer"):
        D.scan_reg(TrainConfig(dataset_dir=str(subject), reg_dir=str(reg)))  # no trigger word: nothing to warn about
    assert "trigger" not in caplog.text


# ---------------------------------------------------------------- each epoch's share


def test_quotas_per_bucket():
    sizes = {"a": 6, "b": 3, "c": 1}
    assert D.reg_quotas(sizes, subject=5, per_epoch=0) == sizes  # 0: all of them
    assert D.reg_quotas(sizes, subject=50, per_epoch=-1) == sizes  # never more than there are
    assert D.reg_quotas(sizes, subject=5, per_epoch=-1) == {"a": 3, "b": 2, "c": 0}  # largest remainders first
    assert D.reg_quotas(sizes, subject=5, per_epoch=4) == {"a": 2, "b": 1, "c": 1}
    assert D.reg_quotas(sizes, subject=0, per_epoch=-1) == {"a": 0, "b": 0, "c": 0}
    assert D.reg_quotas({}, subject=5, per_epoch=-1) == {}
    rng = random.Random(0)
    for _ in range(200):
        sizes = {k: rng.randint(1, 40) for k in range(rng.randint(1, 6))}
        want = rng.randint(1, 60)
        got = D.reg_quotas(sizes, subject=0, per_epoch=want)
        assert sum(got.values()) == min(want, sum(sizes.values()))
        assert all(0 <= got[k] <= sizes[k] for k in sizes)


def test_a_queue_without_regularisation_pictures_draws_as_before():
    items = [_item(f"s{i}", (64, 64) if i % 2 else (96, 64), repeats=1 + i % 3) for i in range(9)]
    old = random.Random(7)
    buckets: dict = {}
    for it in items:  # the queue as it was before the regularisation dataset
        buckets.setdefault((it.width, it.height, it.frames), []).extend([it] * it.repeats)
    b = D.BucketBatcher(items, 2, 7)
    for _ in range(3):
        want = []
        for group in buckets.values():
            group = group[:]
            old.shuffle(group)
            want += [group[i:i + 2] for i in range(0, len(group), 2)]
        old.shuffle(want)
        assert b.epoch() == want
    assert b.reg_images() == 0 and b.epoch_images() == sum(it.repeats for it in items)


def test_an_epoch_takes_its_share_and_the_folder_is_walked_through():
    subject = [_item(f"s{i}", repeats=2) for i in range(3)]  # 6 subject pictures an epoch
    reg = [_item(f"r{i}", reg=True) for i in range(10)] + [_item(f"wide{i}", (96, 64), reg=True) for i in range(5)]
    b = D.BucketBatcher(subject + reg, 4, 0)  # -1: as many as the subject's
    assert b.reg_quota == {(64, 64, 1): 4, (96, 64, 1): 2} and b.reg_images() == 6 and b.epoch_images() == 12
    assert set(b.buckets) == {(64, 64, 1)} and sum(len(v) for v in b.buckets.values()) == 6  # the subject's only
    seen: list[str] = []
    for n in range(1, 11):
        batches = b.epoch()
        assert all(len({(it.width, it.height) for it in batch}) == 1 for batch in batches)  # one size per batch
        flat = [it for batch in batches for it in batch]
        drawn = [it.caption for it in flat if it.reg]
        assert len(flat) == 12 and len(drawn) == 6 and len(set(drawn)) == 6  # never twice in one epoch
        assert sorted(it.caption for it in flat if not it.reg) == sorted(s.caption for s in subject for _ in range(2))
        seen += drawn
        for names, per, size in (([c for c in seen if c.startswith("r")], 4, 10),
                                 ([c for c in seen if c.startswith("wide")], 2, 5)):
            counts = [names.count(c) for c in set(names)]
            # every picture of the bucket comes up before any comes up again
            assert max(counts) - min(counts) <= 1 and (n * per < size or len(set(names)) == size)
    assert any(any(it.reg for it in batch) and not all(it.reg for it in batch) for batch in b.epoch())  # mixed batches


def test_the_queue_shows_the_subject_alone_and_a_rebuilt_one_goes_on_with_the_pass():
    subject = [_item(f"s{i}") for i in range(2)]
    reg = [_item(f"r{i}", reg=True) for i in range(8)]
    b = D.BucketBatcher(subject + reg, 1, 0)
    # what reads the queue's pictures (training maps, region sweeps, the curriculum) gets the subject's only
    assert b.items == subject and b.reg_items == reg
    plain = D.BucketBatcher(subject, 1, 0)
    assert plain.items is subject and plain.reg_items == []
    first = {it.caption for batch in b.epoch() for it in batch if it.reg}
    again = D.BucketBatcher(subject + reg, 1, 5)  # self-training rebuilds the queue every round
    again.carry_reg_pass(b)
    seen = set(first)
    for _ in range(3):  # the six not drawn yet come up before any of the first two does again
        drawn = {it.caption for batch in again.epoch() for it in batch if it.reg}
        assert not drawn & seen
        seen |= drawn
    assert len(seen) == 8
    other = D.BucketBatcher(subject + reg[:4], 1, 5)  # other pictures in the bucket: its own pass
    other.carry_reg_pass(b)
    assert other._reg_decks == {}
    other.carry_reg_pass(None)


def _stage_stub(subject, reg, old=None):
    """What Trainer._enter_stage reads, without a model."""
    from types import SimpleNamespace

    return SimpleNamespace(cfg=TrainConfig(), sides=[64], res_stage=0, all_items=subject, reg_items=reg, batcher=old,
                           batch_iter=None, balancer=None, status=SimpleNamespace(update=lambda **kw: None),
                           _refresh_batcher_weights=lambda: None, _stage_label=lambda: "64")


def test_a_race_configuration_starts_its_own_pass():
    """A rebuilt queue of the same run goes on with the pass; a race configuration's first queue is built while the
    configuration before it is still the trainer's queue, and must not take that one's pass over."""
    subject = [_item(f"s{i}") for i in range(2)]
    reg = [_item(f"r{i}", reg=True) for i in range(8)]
    before = D.BucketBatcher(subject + reg, 1, 0)
    list(before.epoch())  # the configuration before: two of the eight are drawn
    stub = _stage_stub(subject, reg, old=before)
    T.Trainer._enter_stage(stub, 0, 0, initial=True, carry_reg=False)
    assert stub.batcher._reg_decks == {}
    fresh = D.BucketBatcher(subject + reg, 1, stub.cfg.seed)
    drawn = [[it.caption for it in batch] for batch in stub.batcher.epoch()]
    assert drawn == [[it.caption for it in batch] for batch in fresh.epoch()]  # as if nothing ran before it
    stub = _stage_stub(subject, reg, old=before)
    T.Trainer._enter_stage(stub, 0, 5, initial=True)  # the same run's queue, built again
    assert stub.batcher._reg_decks


def test_trigger_off_texts_are_cached_for_the_subject_pictures_the_maps_pick(tmp_path):
    """prepare_cache hands every picture to cache, the regularisation ones too; the maps pick from the subject's."""
    from types import SimpleNamespace

    from rulztrainer import trainmaps

    subject = [dataclasses.replace(_item(f"s{i}"), caption=f"zxq person {i}") for i in range(3)]
    reg = [dataclasses.replace(_item(f"r{i}", reg=True), dataset="reg", caption=f"zxq stranger {i}")
           for i in range(6)]
    cfg = TrainConfig(training_maps=True, training_maps_trigger_off=True, training_maps_images=3)
    stub = SimpleNamespace(race_rows=None, holdout_items=[], _trigger_words=lambda cfgs: ["zxq"])
    texts = T.Trainer._trigger_off_prompts(stub, cfg, subject + reg)
    picked = trainmaps.pick_items(subject, cfg.training_maps_images)
    assert texts == sorted({trainmaps.strip_trigger(it.caption, ["zxq"]) for it in picked})
    assert not any("stranger" in t for t in texts)


def test_per_epoch_settings_and_sampling_weights_leave_the_share_alone():
    subject = [_item(f"s{i}") for i in range(4)]
    reg = [_item(f"r{i}", reg=True) for i in range(10)]
    assert D.BucketBatcher(subject + reg, 1, 0, reg_per_epoch=0).reg_images() == 10
    assert D.BucketBatcher(subject + reg, 1, 0, reg_per_epoch=3).reg_images() == 3
    assert D.BucketBatcher(subject + reg, 1, 0, reg_per_epoch=99).reg_images() == 10
    b = D.BucketBatcher(subject + reg, 1, 0)
    b.set_weights({"s0": 0.0, "s1": 0.0, "s2": 0.0, "r0": 0.0, "r1": 0.0})  # the loss watch's exclusions, say
    flat = [it for batch in b.epoch() for it in batch]
    assert {it.caption for it in flat if not it.reg} == {"s3"}
    assert len([it for it in flat if it.reg]) == 4  # still its share, whatever the subject's weights
    drawn = {it.caption for _ in range(3) for batch in b.epoch() for it in batch if it.reg}
    assert {"r0", "r1"} <= drawn | {it.caption for it in flat if it.reg}  # a weight never keeps one of them out


def test_a_regularisation_caption_is_never_dropped(tmp_path):
    cfg = TrainConfig(dataset_dir=str(tmp_path), caption_dropout=1.0, trigger_word="ohwx")
    paths = {}
    for name, value in (("lat", None), ("own", 1.0), ("dropped", 2.0)):
        paths[name] = tmp_path / f"{name}.safetensors"
        save_file({"latent": torch.zeros(128, 4, 4)} if value is None else
                  {"embeds": torch.full((3, 4096), value).to(torch.bfloat16)}, str(paths[name]))
    D.dropout_text_path(cfg).parent.mkdir(parents=True, exist_ok=True)
    save_file({"embeds": torch.full((3, 4096), 2.0).to(torch.bfloat16)}, str(D.dropout_text_path(cfg)))
    items = [D.Item(image=tmp_path / f"{i}.png", caption="c", width=64, height=64, latent_path=paths["lat"],
                    latent_flip_path=None, text_path=paths["own"], reg=reg) for i, reg in enumerate((False, True))]
    _, embeds, _, _ = D.load_batch(items, cfg, random.Random(0))
    assert float(embeds[0].float().mean()) == 2.0  # the subject's picture: dropped to the trigger word alone
    assert float(embeds[1].float().mean()) == 1.0  # the regularisation picture: its own caption


# ---------------------------------------------------------------- the trainer


def test_epoch_lengths_count_the_share_not_the_folder():
    t = T.Trainer.__new__(T.Trainer)
    t.cfg = TrainConfig(batch_size=2, gradient_accumulation=1)
    t.replicas = [None]
    t.batcher = D.BucketBatcher([_item(f"s{i}") for i in range(4)] + [_item(f"r{i}", reg=True) for i in range(30)], 2, 0)
    assert t._epoch_images(t.batcher) == 8  # 4 of the subject's and 4 of the 30
    assert t._maps_epoch_steps() == (4, 2)


def _prepare(cfg) -> None:
    """The caches of a run with a regularisation folder, written ahead (test_run_safety._prepare, with its pictures)."""
    from test_run_safety import _prepare as prepare_run

    prepare_run(cfg)
    for it in D.scan_reg(cfg):
        it.latent_path.parent.mkdir(parents=True, exist_ok=True)
        save_file({"latent": torch.randn(128, it.height // 16, it.width // 16)}, str(it.latent_path))
        save_file({"embeds": torch.randn(5, 32).to(torch.bfloat16)}, str(it.text_path))


def test_a_run_trains_them_and_the_loss_watch_never_sees_them(rig, tmp_path, caplog):
    cfg, _, _ = rig
    reg = tmp_path / "reg"
    reg.mkdir()
    for i in range(5):
        _picture(reg / f"other_{i}.png", 30 * i)
        (reg / f"other_{i}.txt").write_text(f"someone else, number {i}", encoding="utf-8")
    cfg = dataclasses.replace(cfg, reg_dir=str(reg), reg_weight=0.5, steps=12, loss_watch=True, caption_dropout=0.5,
                              trigger_word="ohwx")
    _prepare(cfg)
    trained: list[str] = []
    real = D.load_batch

    def load(batch, c, rng):
        trained.extend(("reg:" if it.reg else "") + it.image.name for it in batch)
        return real(batch, c, rng)

    T.load_batch, keep = load, T.load_batch
    try:
        with caplog.at_level(logging.INFO, logger="rulztrainer"):
            tr = T.Trainer(cfg)
            tr.run()
    finally:
        T.load_batch = keep
    st = _status(cfg)
    assert st["stage"] == "finished" and st["step"] == 12
    assert st["reg_images"] == 5 and st["reg_per_epoch"] == 3  # as many as the subject's three
    assert len(tr.reg_items) == 5 and all(not it.reg for it in tr.all_items) and len(tr.all_items) == 3
    assert tr.batcher.epoch_images() == 6
    n_reg = sum(name.startswith("reg:") for name in trained)
    assert len(trained) == 12 and n_reg == 6  # two epochs of 3 + 3
    assert len({name for name in trained[:6] if name.startswith("reg:")}) == 3
    assert len({name for name in trained if name.startswith("reg:")}) == 5  # the second epoch took the other two first
    watched = [json.loads(line)["image"] for line in
               (_job(cfg) / "per_image_loss.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(watched) == 6 and all(Path(k).parent == Path(cfg.dataset_dir) for k in watched)
    assert set(tr.loss_watch.stats) == {str(Path(cfg.dataset_dir) / f"{i}.png") for i in range(3)}
    assert "Regularisation dataset: 5 pictures" in caplog.text and "50% of the steps" in caplog.text


def test_a_run_refuses_a_folder_without_captions_before_it_loads_anything(rig, tmp_path):
    cfg, _, _ = rig
    reg = tmp_path / "reg"
    reg.mkdir()
    _picture(reg / "bare.png", 10)
    cfg = dataclasses.replace(cfg, reg_dir=str(reg))
    with pytest.raises(ValueError, match="no caption: bare.png"):
        _go(cfg)
    assert _status(cfg)["stage"] == "error" and not list(_job(cfg).glob("*.safetensors"))
