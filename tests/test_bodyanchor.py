"""The anchored body score around a stand-in embedder and window: 0 at the strangers' mean and 100 at the references'
mean, the leave-one-out reference level, no score before the strangers or with too few of them, one try at the
strangers, saving them for a resumed run; the canonical window (colour grade and exposure gone, background grey);
its linear scale in the controller, the spreadsheet and Previous Runs; and the trainer taking the first step-0 probe's
body renders as the strangers, with more body seeds for that probe."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import bodyanchor as BA  # noqa: E402
from rulztrainer import facesim  # noqa: E402
from rulztrainer import likeness as L  # noqa: E402
from rulztrainer.bodysim import WINDOW_GREY  # noqa: E402
from rulztrainer.config import TrainConfig  # noqa: E402


@pytest.fixture(autouse=True)
def _fake_paths(monkeypatch):
    monkeypatch.setattr(facesim, "expand_refs", lambda paths: list(paths))  # the stand-in paths are not files


def unit(*v):
    v = np.asarray(v, np.float64)
    return v / np.linalg.norm(v)


# references around +x, strangers around +y; the same vectors for both backbones, so their mean is either one
VECS = {"r1": unit(1, 0.1, 0), "r2": unit(1, -0.1, 0), "r3": unit(1, 0, 0.2), "s1": unit(0.1, 1, 0),
        "s2": unit(-0.1, 1, 0), "s3": unit(0, 1, 0.1), "between": unit(1, 1, 0), "close": unit(1, 0.05, 0.05),
        "portrait": unit(1, 0, 0)}


def _scorer(covers=None):
    covers = covers or {}
    return BA.AnchoredScorer.with_embedder(
        lambda im: {b: VECS[im] for b in BA.BACKBONES},
        lambda p: None if p == "nobody" else (p, covers.get(p, 1.0)))


def test_zero_at_the_strangers_and_100_at_the_references():
    s = _scorer({"portrait": 0.4})
    info = s.set_references(["r1", "r2", "r3", "nobody"])
    assert s.ready and s.ref_count == 3 and [i["person"] for i in info] == [True, True, True, False]
    assert s.needs_strangers and not s.anchored and s.score("close") == (None, 0)  # no zero yet
    assert s.ref_self_similarity is None
    assert s.set_strangers(["s1", "s2", "s3", "nobody"]) == 3 and s.anchored and not s.needs_strangers
    refs = np.stack([VECS[k] for k in ("r1", "r2", "r3")])
    strs = np.stack([VECS[k] for k in ("s1", "s2", "s3")])
    assert abs(np.mean([s.score(k)[0] for k in ("s1", "s2", "s3")])) < 1e-9  # the strangers average 0
    assert abs(np.mean([s.score(k)[0] for k in ("r1", "r2", "r3")]) - 100.0) < 1e-9  # the references 100
    d = refs.mean(0) - strs.mean(0)
    want = 100 * (VECS["between"] - strs.mean(0)) @ d / (d @ d)
    assert abs(s.score("between")[0] - want) < 1e-9 and s.score("between")[1] == 3
    assert 0 < s.score("between")[0] < 100 < s.score("close")[0] + 10  # halfway is in between; linear, uncapped
    assert s.score("portrait") == (None, 0)  # too little of the window in the frame
    # a typical reference scored without itself (the median of the leave-one-out scores)
    loo = []
    for i in range(3):
        keep = [j for j in range(3) if j != i]
        loo.append(float(BA.anchored_scores(refs[i], refs[keep], strs)))
    assert abs(s.ref_self_similarity - np.median(loo) / 100) < 1e-9


def test_too_few_strangers_leave_the_score_off_and_are_tried_once(tmp_path):
    s = _scorer()
    s.set_references(["r1", "r2"])
    assert s.set_strangers(["s1", "nobody"]) == 1 and not s.anchored
    assert not s.needs_strangers  # tried: a later step-0 probe (a soup's LoRA) is no base model
    assert s.score("close") == (None, 0)
    s2 = _scorer()
    s2.set_references(["r1", "r2"])
    s2.set_strangers(["s1", "s2", "s3"])
    s2.save_strangers(tmp_path / BA.STRANGERS_FILE)
    s3 = _scorer()
    s3.set_references(["r1", "r2"])
    assert not s3.load_strangers(tmp_path / "missing.npz") and s3.needs_strangers
    assert s3.load_strangers(tmp_path / BA.STRANGERS_FILE) and s3.stranger_count == 3
    assert abs(s3.score("between")[0] - s2.score("between")[0]) < 1e-9


def test_the_canonical_window_drops_the_colour_grade_and_keeps_the_grey():
    from PIL import Image

    rng = np.random.default_rng(0)
    a = np.empty((120, 80, 3), np.float32)
    a[:] = WINDOW_GREY
    body = rng.uniform(60, 200, (80, 40, 1)) * np.array([1.0, 0.9, 0.8])
    a[20:100, 20:60] = body
    graded = a.copy()
    graded[20:100, 20:60] = body * np.array([0.7, 0.8, 1.1]) + 12  # darker, cooler, lifted: a different look
    c1 = np.asarray(BA.canonical_window(Image.fromarray(a.astype(np.uint8))), np.float32)
    c2 = np.asarray(BA.canonical_window(Image.fromarray(np.clip(graded, 0, 255).astype(np.uint8))), np.float32)
    assert np.abs(c1 - c2).mean() < 2.0  # the same person, whatever the grade
    assert np.abs(c1[2, 2] - np.asarray(WINDOW_GREY)).max() <= 1  # the background stays grey
    assert BA.canonical_window(Image.new("RGB", (40, 60), WINDOW_GREY)) is None  # nobody in it
    p = BA.person_pixels(Image.fromarray(a.astype(np.uint8)))
    assert p[50, 40] and not p[5, 5]


def test_the_anchored_body_scale_is_linear_from_0_to_100():
    cfg = TrainConfig(body_score="anchored")
    assert L.body_shape(cfg) == (1.0, 1.0) and L.body_shape(TrainConfig()) == (L.BODY_SPAN_DEFAULT,
                                                                                 L.BODY_CURVE_DEFAULT)
    c = L.LikenessController(cfg, face_ceiling=70.0, body_ceiling=79.0)  # the passed level is not its 100
    assert c.scale("body") == (0.0, 100.0)
    c.update(0, 20.0, 0.3)  # the step-0 probe: the floors
    assert c.scale("body") == (0.0, 100.0) and c.norm("body", 24.0) == 24.0
    assert c.norm("body", -5.0) == 0.0 and c.norm("body", 130.0) == 100.0
    assert c.scale_record()["curve"]["body"] == 1.0 and c.scale_record()["span"]["body"] == 1.0
    assert "references' mean" in c.describe_scale()
    old = L.LikenessController(TrainConfig(), body_ceiling=79.0)
    assert old.scale("body")[1] != 100.0 and old._shape("body") == (1.35, 1.5)


def test_the_spreadsheet_and_previous_runs_keep_the_anchored_body_linear(tmp_path):
    from rulztrainer import history
    from rulztrainer import likeness_export as X

    cfg = TrainConfig(body_score="anchored", job_name="anch")
    scale = {"face": [10.0, 91.0], "body": [0.0, 100.0], "curve": {"face": 1.5, "body": 1.0},
             "span": {"face": 1.35, "body": 1.0}}
    recs = [{"step": 0, "face": 10.0, "body": 0.2, "scale": scale}, {"step": 100, "face": 50.0, "body": 20.0,
                                                                     "scale": scale}]
    sc = X.build_workbook_from(recs, cfg)["Scale"]
    assert [sc[f"B{r}"].value for r in (7, 8, 9, 10, 11)] == [0.0, 100.0, 1.0, 1.0, 0.0]
    rec = {"config": {"body_score": "anchored", "probe_ema": 0.0}, "probe_raw": {"records": recs}}
    series = history.rescore(rec, 1.35, 1.5, 1.35, 1.5, 0.7, 0.3)  # the common body span and curve don't apply
    assert series["body"][-1] == [100, 20.0]
    rec["config"]["body_score"] = "appearance"
    assert history.rescore(rec, 1.35, 1.5, 1.35, 1.5, 0.7, 0.3)["body"][-1][1] != 20.0
    assert json.dumps(series)


class _Status:
    def __init__(self):
        self.data = {}

    def update(self, **kw):
        self.data.update(kw)


def _trainer(tmp_path, scorer, **cfg):
    from rulztrainer.trainer import Trainer

    tr = Trainer.__new__(Trainer)
    tr.cfg = TrainConfig(body_score="anchored", probe_every=50, probe_body_prompt="a woman standing", **cfg)
    tr.body_scorer, tr.face_scorer, tr.race = scorer, None, None
    tr.job_dir, tr.status = tmp_path, _Status()
    (tmp_path / "probes").mkdir(exist_ok=True)
    return tr


def test_the_first_step0_probe_sets_the_strangers_with_more_body_seeds(tmp_path):
    s = _scorer()
    s.set_references(["r1", "r2", "r3"])
    tr = _trainer(tmp_path, s, probe_seeds=2, probe_floor_seeds=4, probe_at_start=False)
    assert tr._probe_at_start()  # probe_at_start off, but the anchored score needs the step-0 probe
    jobs = tr._probe_jobs(0)
    assert sum(j.kind == "body" for j in jobs) == BA.STRANGER_SEEDS
    assert sum(j.kind == "body" for j in tr._probe_jobs(50)) == 2
    jobs = [SimpleNamespace(kind="body") for _ in range(3)]
    face, body, seeds = tr._score_probe(jobs, ["s1", "s2", "s3"], 0)
    assert s.anchored and abs(body) < 1e-9 and len(seeds["body"]) == 3
    assert (tmp_path / "probes" / BA.STRANGERS_FILE).is_file()
    assert tr.status.data["body_ref_similarity"] == s.ref_self_similarity
    assert not tr._probe_at_start() and sum(j.kind == "body" for j in tr._probe_jobs(0)) == 4
    _, body, _ = tr._score_probe([SimpleNamespace(kind="body")], ["close"], 0)  # a soup's step-0 probe: no refit
    assert s.stranger_count == 3 and body > 90

    s2 = _scorer()  # a resumed run reads the strangers back
    s2.set_references(["r1", "r2", "r3"])
    tr2 = _trainer(tmp_path, s2)
    tr2._resume_body_strangers(400)
    assert s2.anchored and abs(s2.score("between")[0] - s.score("between")[0]) < 1e-9
