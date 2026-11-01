"""The SSCD copy check: its config, the limit read against the photos' own match, the self-training gate and the
probes' logged copy score (the model faked: no download)."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from rulztrainer import copycheck as CC
from rulztrainer import selftrain as ST
from rulztrainer.config import TrainConfig


def _cfg(**kw):
    kw.setdefault("dataset_dir", "ds")
    return TrainConfig(**kw)


def _unit(*v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


# photos: two from one shoot (cosine 0.6 to each other) and one apart; renders named after what they copy
VECS = {"a.png": _unit(1, 0, 0), "b.png": _unit(0.6, 0.8, 0), "c.png": _unit(0, 0, 1),
        "copy_of_a.png": _unit(0.95, 0.05, 0.3), "new.png": _unit(0.4, 0.2, 0.4), "far.png": _unit(-1, 0, 0)}


def _checker():
    cc = CC.CopyChecker.with_embedder(lambda p: VECS[str(p).replace("\\", "/").rsplit("/", 1)[-1]])
    cc.set_photos(["a.png", "b.png", "c.png", "a.png"])  # a photo at two resolutions is described once
    return cc


def test_config():
    assert not _cfg().copy_check_on() and _cfg(selftrain=True).copy_check_on()
    assert _cfg(copy_check="on").copy_check_on() and not _cfg(copy_check="off", selftrain=True).copy_check_on()
    with pytest.raises(ValueError, match="copy_check"):
        _cfg(copy_check="maybe").validate()
    with pytest.raises(ValueError, match="copy_max"):
        _cfg(copy_max=0.0).validate()
    # YAML and race rows read a bare on / off as true / false
    import yaml

    loaded = TrainConfig.from_dict({"dataset_dir": "ds", **yaml.safe_load("copy_check: on\nface_gold: off")})
    assert (loaded.copy_check, loaded.face_gold) == ("on", "off")
    loaded.validate()
    from rulztrainer import race

    rows = [{"name": "a", **race.parse_overrides("copy_check=off")}, {"name": "b", **race.parse_overrides("copy_max=0.6")}]
    out = race.variant_configs(_cfg(race=True, race_configs=rows, selftrain=True, face_refs=["x"]))
    assert [(c.copy_check, c.copy_max, c.copy_check_on()) for _, c, _ in out] == [("off", 0.5, False),
                                                                                 ("auto", 0.6, True)]


def test_the_limit_rises_with_the_photos_own_match():
    assert CC.copy_limit(0.5, None) == 0.5
    assert CC.copy_limit(0.5, 0.2) == 0.5
    assert CC.copy_limit(0.5, 0.7) == pytest.approx(0.8)  # one shoot: a render like one more photo of it is no copy


def test_the_checker_finds_the_photo_a_render_copies(tmp_path):
    cc = _checker()
    assert cc.photos == ["a.png", "b.png", "c.png"]
    assert cc.baseline == pytest.approx(0.6)  # each photo's best match among the others: 0.6, 0.6, 0 -> median
    score, photo = cc.score("copy_of_a.png")
    assert photo == "a.png" and score > 0.9
    assert cc.closest(["new.png", "copy_of_a.png", "far.png"]) == (pytest.approx(score), "a.png", "copy_of_a.png")
    assert CC.CopyChecker.with_embedder(lambda p: None).score("x.png") == (None, "")  # no photos
    # the model's input: the whole picture squeezed to 320 x 320, ImageNet-normalized
    Image.new("RGB", (640, 480), (124, 116, 104)).save(tmp_path / "grey.png")
    x = CC.load_image(tmp_path / "grey.png")
    assert x.shape == (3, 320, 320) and np.abs(x).max() < 0.1


def test_self_training_leaves_out_a_picture_that_copies_a_photo(tmp_path):
    cands = [ST.Candidate("copy.png", "p", 90.0, 1, 0.9, sharp=1.0, copy=0.85, copy_of="a.png"),
             ST.Candidate("new.png", "p", 80.0, 1, 0.9, sharp=1.0, copy=0.4, copy_of="b.png"),
             ST.Candidate("unchecked.png", "p", 70.0, 1, 0.9, sharp=1.0)]
    kept, reasons = ST.select(cands, need_face=True, bar=50, cap=10, copy_max=0.8)
    assert [c.path for c in kept] == ["new.png", "unchecked.png"] and reasons == {"copies a photo": 1}
    assert cands[0].verdict == "copies a photo" and "1 copies a photo" in ST.describe(reasons)
    kept, _ = ST.select(cands, need_face=True, bar=50, cap=10)  # no copy check: the copy is kept, best first
    assert kept[0].path == "copy.png"
    ST.record(tmp_path / "selftrain.json", 1, 50, 60.0, cands, 0.8)
    rec = json.loads((tmp_path / "selftrain.json").read_text())[0]
    assert rec["copy_limit"] == 0.8 and rec["candidates"][0]["copy"] == 0.85
    assert rec["candidates"][0]["copy_of"] == "a.png" and "copy" not in rec["candidates"][2]


def _trainer(cfg):
    from rulztrainer import trainer as T

    tr = T.Trainer.__new__(T.Trainer)
    tr.cfg, tr.controller, tr.face_scorer, tr.body_scorer, tr.race_rows = cfg, object(), object(), None, None
    status = {}
    tr.status = SimpleNamespace(update=lambda **k: status.update(k), data={})
    return tr, status


def test_the_trainer_loads_the_check_only_when_wanted(monkeypatch):
    made = []

    class Fake(CC.CopyChecker):
        def __init__(self, models_dir, device, progress=None):
            made.append(device)
            self.photos, self.descs, self.baseline = [], None, None

        def embed(self, path):
            return VECS[path]

    monkeypatch.setattr(CC, "CopyChecker", Fake)
    items = [SimpleNamespace(image="a.png", category="image"), SimpleNamespace(image="b.png", category="image"),
             SimpleNamespace(image="a.png", category="face_crop")]
    tr, status = _trainer(_cfg(probe_every=100))
    assert tr._make_copy_checker(items) is None and made == []  # auto without self-training: off
    tr.cfg = _cfg(probe_every=100, copy_check="on", face_provider="cpu")
    cc = tr._make_copy_checker(items)
    assert made == ["cpu"] and cc.photos == ["a.png", "b.png"] and status["copy_baseline"] == pytest.approx(0.6)
    tr.cfg = _cfg(selftrain=True, noise_files=["x.safetensors"])  # the noise check renders no new pictures to check
    assert tr._make_copy_checker(items) is None
    tr.cfg, tr.face_scorer = _cfg(selftrain=True), None  # nothing scores the pictures: no probes, no self-training
    assert tr._make_copy_checker(items) is None
    # a race where one configuration self-trains: on for all of them, the gate only for that one
    tr.face_scorer, tr.cfg = object(), _cfg(probe_every=100)
    tr.race_rows = [("a", _cfg(probe_every=100), {}), ("b", _cfg(probe_every=100, selftrain=True), {})]
    tr.copy_checker = tr._make_copy_checker(items)
    assert tr.copy_checker is not None and tr._selftrain_copy_limit() is None
    tr.cfg = tr.race_rows[1][1]
    assert tr._selftrain_copy_limit() == pytest.approx(0.7)  # max(0.5, the photos' 0.6 + 0.1)


def test_a_probe_logs_its_closest_photo():
    from rulztrainer.likeness import ProbeRecord

    tr, _ = _trainer(_cfg(copy_check="on"))
    rec = ProbeRecord(100, 50.0, None, 50.0, 50.0, None, 50.0, 1.0, 0.0, 50.0, 100, 0)
    tr.copy_checker = None
    assert tr._copy_probe(rec, ["copy_of_a.png"]) == "" and rec.copy is None
    tr.copy_checker = _checker()
    note = tr._copy_probe(rec, ["new.png", "copy_of_a.png"])
    assert rec.copy > 0.9 and rec.copy_of == "a.png" and rec.copy_baseline == pytest.approx(0.6)
    assert "copy_of_a.png to a.png" in note and "above the copy limit 0.70" in note
    assert "copy limit 0.70" in tr._copy_probe(rec, ["new.png"]) and rec.copy < 0.7


def test_a_self_training_candidate_gets_its_copy_score(tmp_path):
    tr, _ = _trainer(_cfg(selftrain=True))
    tr.face_scorer, tr.copy_checker = None, _checker()
    path = tmp_path / "copy_of_a.png"
    Image.fromarray(np.random.default_rng(0).integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(path)
    c = tr._selftrain_candidate(str(path), "photo of zk", None)
    assert c.copy > 0.9 and c.copy_of == "a.png"
    tr.cfg = _cfg(selftrain=True, copy_check="off")  # this configuration's check is off: no score, no gate
    assert tr._selftrain_candidate(str(path), "photo of zk", None).copy is None


def test_the_copy_check_comes_after_the_quality_filter_and_before_the_bar():
    assert ST.REASONS.index("blurry") < ST.REASONS.index("copies a photo") < ST.REASONS.index("below the bar")


def test_the_monitor_shows_a_probe_copy_score():
    import webui

    rec = {"face_norm": 40.0, "body_norm": 20.0, "combined": 33.0, "scale": {"face": [1, 2]}}
    assert "copy" not in webui.probe_score_line(rec)
    assert webui.probe_score_line({**rec, "copy": 0.4321}).endswith("| copy 0.43")
