"""The held-out gold face scorer (face_gold): when it turns on, FaceScorer's second recognizer and its ceiling, the
controller's gold scale and decisions (best, stop and race on the gold face; lr control on ArcFace), the trainer's
ceilings, setup and reward-stage warning, the noise report's gold kinds and the Monitor's gold line."""
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import facesim  # noqa: E402
from rulztrainer import likeness_stats as S  # noqa: E402
from rulztrainer.config import TrainConfig  # noqa: E402
from rulztrainer.likeness import LikenessController  # noqa: E402


def test_when_the_gold_scorer_turns_on():
    assert TrainConfig().gold_backend() == ""  # nothing trains against ArcFace
    assert TrainConfig(reward_steps=10).gold_backend() == "facenet"
    assert TrainConfig(id_align=True).gold_backend() == "facenet"
    assert TrainConfig(id_loss_weight=0.2).gold_backend() == "facenet"  # arcface backend
    assert TrainConfig(noise_files=["a.safetensors"]).gold_backend() == "facenet"  # the noise check measures it
    assert TrainConfig(id_loss_weight=0.2, id_loss_backend="facenet").gold_backend() == ""  # trains on FaceNet
    assert TrainConfig(id_loss_weight=0.2, id_loss_backend="facenet", id_align=True).gold_backend() == ""
    assert TrainConfig(reward_steps=10, face_gold="off").gold_backend() == ""
    assert TrainConfig(face_gold="facenet").gold_backend() == "facenet"
    with pytest.raises(ValueError, match="face_gold must be"):
        TrainConfig(face_gold="adaface").validate()
    with pytest.raises(ValueError, match="no held-out judge"):
        TrainConfig(face_gold="facenet", id_loss_weight=0.2, id_loss_backend="facenet").validate()


def _scorer(tmp_path, vecs, gold_vecs, cut=()):
    """A FaceScorer without InsightFace: each image's face embeds to vecs[name], and the gold recognizer to
    gold_vecs[name]."""
    from PIL import Image

    paths = []
    for name in vecs:
        paths.append(tmp_path / f"{name}.png")
        Image.new("RGB", (100, 100)).save(paths[-1])
    s = facesim.FaceScorer.__new__(facesim.FaceScorer)
    s.refs, s.gold, s.gold_refs = [], None, None
    s.faces = lambda p: [SimpleNamespace(
        bbox=np.array([-30, 5, 95, 120] if Path(p).stem in cut else [10, 10, 90, 90], dtype=np.float32),
        kps=np.zeros((5, 2), np.float32) + len(Path(p).stem), det_score=0.9,
        normed_embedding=np.asarray(vecs[Path(p).stem], dtype=np.float32))]
    s._gold_embed = lambda p, kps: np.asarray(gold_vecs[Path(p).stem], dtype=np.float32)
    return s, paths


def test_the_gold_recognizer_scores_the_same_face_against_its_own_references(tmp_path):
    vecs = {"a": [1.0, 0.0], "b": [0.8, 0.6], "c": [0.6, 0.8], "cut": [0.0, 1.0], "probe": [0.8, 0.6]}
    gold = {"a": [1.0, 0.0, 0.0], "b": [0.6, 0.8, 0.0], "c": [0.0, 1.0, 0.0], "cut": [0.0, 0.0, 1.0],
            "probe": [0.0, 0.6, 0.8]}
    s, paths = _scorer(tmp_path, vecs, gold, cut={"cut"})
    s.set_references(paths[:4])
    assert s.gold_refs is None and s.score(paths[4], gold=True).gold_percent is None  # no gold recognizer yet
    s.set_gold(object())
    assert s.gold_refs.shape == (4, 3)
    assert s.gold_self_similarity == pytest.approx(0.8)  # best matches .6 .8 .8, the cut-off face left out
    sc = s.score(paths[4], gold=True)
    assert sc.cosine == pytest.approx(1.0) and sc.gold_cosine == pytest.approx(0.8)  # best gold match: c
    assert sc.gold_percent == pytest.approx(80.0)
    assert s.score(paths[4]).gold_percent is None  # only when asked (the reward stage and samples skip it)
    s.set_references(paths[:2])  # new references: the gold ones follow
    assert s.gold_refs.shape == (2, 3) and s.gold_self_similarity == pytest.approx(0.6)


def _cfg(**kw):
    return TrainConfig(**{"face_weight": 1.0, "body_weight": 0.0, "probe_ema": 0.0, "likeness_noise_band": 1.0,
                          "face_scale_span": 1.0, "face_scale_curve": 1.0, **kw})


def _probe(ctrl, step, face, gold):
    return ctrl.update(step, face, None, seeds={"face": [face, face], "body": [], "gold": [gold, gold]})


def test_the_gold_face_makes_the_decisions_and_arcface_the_lr():
    ctrl = LikenessController(_cfg(likeness_lr_control=True), face_ceiling=70.0, gold_ceiling=80.0)
    rec, _ = _probe(ctrl, 0, 20.0, 30.0)  # the base model: both scales' zero
    assert ctrl.scale("gold") == (30.0, 80.0) and rec.gold_norm == 0.0 and rec.gold_select
    assert "gold face 30.0 -> 80.0" in ctrl.describe_scale() and set(rec.scale) >= {"face", "gold"}
    rec, act = _probe(ctrl, 100, 45.0, 55.0)
    assert act["save_best"] and ctrl.best == pytest.approx(50.0) and rec.gold_norm_ema == pytest.approx(50.0)
    # ArcFace jumps (the reward games it) while the gold face stays put: no new best, no progress
    rec, act = _probe(ctrl, 200, 70.0, 55.0)
    assert not act.get("save_best") and ctrl.since_improve == 1 and ctrl.combined_ema == pytest.approx(50.0)
    assert rec.face_norm_ema == pytest.approx(100.0) and rec.arc_combined_ema == pytest.approx(100.0)
    assert act["lr_ratio"] > 1  # the lr control still reads ArcFace's trend
    assert rec.combined_seeds == [pytest.approx(50.0)] * 2 and rec.gold_norm_seeds == [pytest.approx(50.0)] * 2
    assert ctrl.last_seeds == rec.combined_seeds  # paired race drops read the gold face too
    st = ctrl.state()
    assert st["gold_select"] and st["gold_norm_ema"] == pytest.approx(50.0) and st["arc_combined_ema"] == 100.0


def test_with_select_off_the_gold_face_is_only_kept():
    ctrl = LikenessController(_cfg(face_gold_select=False), face_ceiling=70.0, gold_ceiling=80.0)
    _probe(ctrl, 0, 20.0, 30.0)
    _probe(ctrl, 100, 45.0, 55.0)
    rec, act = _probe(ctrl, 200, 70.0, 55.0)
    assert act["save_best"] and ctrl.best == pytest.approx(100.0) and not rec.gold_select
    assert rec.gold_norm_ema == pytest.approx(50.0) and rec.combined_seeds == [pytest.approx(100.0)] * 2


def test_without_gold_scores_nothing_changes():
    ctrl = LikenessController(_cfg(), face_ceiling=70.0)
    ctrl.update(0, 20.0, None, seeds={"face": [20.0], "body": []})
    rec, _ = ctrl.update(100, 45.0, None, seeds={"face": [45.0], "body": []})
    assert not ctrl.has_gold and rec.gold is None and rec.gold_norm_seeds == [] and set(rec.scale) == {
        "face", "body", "curve", "span"}
    assert rec.arc_combined_ema is None and ctrl.arc_combined_ema == ctrl.combined_ema == pytest.approx(50.0)


def _trainer(cfg, controller=None):
    from rulztrainer import trainer as T

    tr = T.Trainer.__new__(T.Trainer)
    tr.cfg, tr.controller, tr.body_scorer = cfg, controller, None
    status = {}
    tr.status = SimpleNamespace(update=lambda **k: status.update(k), data={"saved": []})
    return tr, status


def test_the_trainer_sets_the_gold_scale_and_loads_it_only_when_wanted(monkeypatch):
    from rulztrainer import idloss

    tr, status = _trainer(_cfg())
    tr.face_scorer = SimpleNamespace(ref_self_similarity=0.6, gold_self_similarity=0.75)
    assert tr._controller_ceilings() == {"face_ceiling": 60.0, "gold_ceiling": 75.0}

    loads = []

    class Emb:
        def to(self, dev):
            loads.append(dev)
            return self

    monkeypatch.setattr(idloss, "facenet_embedder", lambda *a, **k: Emb())
    sc = SimpleNamespace(gold="stale", gold_refs=None, gold_self_similarity=0.75)
    sc.set_gold = lambda emb: setattr(sc, "gold", emb) or setattr(sc, "gold_refs", np.zeros((2, 4)))
    tr._setup_gold(sc)  # nothing trains against ArcFace
    assert sc.gold is None and loads == []
    tr.cfg = _cfg(reward_steps=5)
    tr._setup_gold(sc)
    assert isinstance(sc.gold, Emb) and loads == ["cpu"] and status["face_gold"] == "facenet"
    # a race where one configuration trains on FaceNet: no held-out judge for all of them
    tr.race_rows = [("a", _cfg(id_loss_weight=0.2), {}), ("b", _cfg(id_loss_weight=0.2, id_loss_backend="facenet"), {})]
    tr._setup_gold(sc)
    assert sc.gold is None


def test_the_reward_stage_warns_when_arcface_runs_ahead_of_the_gold_face(caplog):
    cfg = _cfg(reward_steps=5)
    ctrl = LikenessController(cfg, face_ceiling=70.0, gold_ceiling=80.0)
    tr, status = _trainer(cfg, ctrl)
    _probe(ctrl, 0, 20.0, 30.0)
    _probe(ctrl, 100, 45.0, 55.0)
    tr.stage_tag, tr._gold_gap0, tr._gold_gap_warned = "_reward", tr._gold_gap(), False
    assert tr._gold_gap0 == pytest.approx(0.0)
    _probe(ctrl, 110, 50.0, 55.0)  # 10 points ahead: within the limit (max(10, 3 bands))
    tr._check_gold_gap(110)
    assert "gold_gap_warning" not in status
    _probe(ctrl, 120, 60.0, 55.0)
    with caplog.at_level(logging.WARNING, logger="rulztrainer"):
        tr._check_gold_gap(120)
        tr._check_gold_gap(130)  # once
    assert status["gold_gap_warning"] == 120 and sum("gaming ArcFace" in r.message for r in caplog.records) == 1


def test_the_noise_report_measures_the_gold_face_too():
    files = {"mid": {"face": [40, 55, 45, 60], "body": [], "combined": [40, 55, 45, 60],
                     "gold": [30, 50, 35, 52], "combined_gold": [30, 50, 35, 52]},
             "best": {"face": [45, 60, 50, 65], "body": [], "combined": [45, 60, 50, 65],
                      "gold": [36, 51, 44, 55], "combined_gold": [36, 51, 44, 55]}}
    rep = S.noise_report(files, 2, 0.5)
    assert set(rep["kinds"]) == {"face", "body", "combined", "gold", "combined_gold"}
    assert rep["kinds"]["face"]["sd_diff"] == pytest.approx(0.0) and rep["kinds"]["gold"]["sd_diff"] > 2
    assert "suggested_noise_band_gold" in rep and rep["suggested_noise_band_gold"] > rep["suggested_noise_band"]
    lines = S.describe(rep)
    assert any(line.startswith("gold face (FaceNet):") for line in lines)
    assert any("gold face in charge" in line for line in lines)
    no_gold = S.noise_report({k: {kk: v[kk] for kk in ("face", "body", "combined")} for k, v in files.items()}, 2, 0.5)
    assert set(no_gold["kinds"]) == {"face", "body", "combined"} and "suggested_noise_band_gold" not in no_gold
    recs = [{"step": s, "face_norm_seeds": [s, s + 4], "body_norm_seeds": [], "gold_norm_seeds": [s, s + 8],
             "combined_seeds": [s, s + 8]} for s in (100, 200)]
    run = S.run_noise(recs)
    assert run["gold"]["sigma"] > run["face"]["sigma"]
    assert "gold" not in S.run_noise([{k: v for k, v in r.items() if k != "gold_norm_seeds"} for r in recs])


def test_the_monitor_draws_the_gold_face():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import webui

    probes = [{"step": 100, "face_norm_ema": 50.0, "body_norm_ema": None, "combined_ema": 40.0, "gold_norm_ema": 40.0,
               "face_norm": 50.0, "gold_norm": 40.0, "combined": 40.0, "scale": {"face": [20, 70]}}]
    series, legend = webui.probe_chart_series(probes, [])
    assert legend[-1] == "face (FaceNet)" and series[-1]["data"] == [[100, 40.0]]
    assert "FaceNet face 40%" in webui.probe_score_line(probes[0])
    _, plain = webui.probe_chart_series([{k: v for k, v in probes[0].items() if not k.startswith("gold")}], [])
    assert "face (FaceNet)" not in plain


def test_the_noise_check_reports_both_recognizers(tmp_path):
    import contextlib
    import json

    from rulztrainer import archs

    cfg = _cfg(job_name="job", noise_files=["a/mid.safetensors", "b/best.safetensors"], noise_seeds=4, probe_seeds=2,
               probe_ema=0.5)
    ctrl = LikenessController(cfg, face_ceiling=70.0, gold_ceiling=80.0)
    tr, status = _trainer(cfg, ctrl)
    tr.arch, tr.job_dir, tr.devices = archs.arch_by_name("moss_image"), tmp_path, ["cpu"]
    tr._free_for_rendering = lambda: None
    tr._sampling_slots = lambda model, step: contextlib.nullcontext([])
    tr._probe_with = lambda slots, step, tag="", n_seeds=None: (20.0, None, [], {"face": [20.0] * 4, "body": [],
                                                                                   "gold": [30.0] * 4})

    def probe(model, path, tag, n_seeds=None):
        bump = 5.0 if "best" in path else 0.0
        return None, None, {"face": [40.0 + bump, 55.0 + bump, 45.0 + bump, 60.0 + bump], "body": [],
                            "gold": [35.0 + bump, 50.0 + bump * 2, 40.0, 55.0 + bump]}

    tr._soup_probe = probe
    tr._noise_stage(None)
    rep = json.loads((tmp_path / "noise" / "noise.json").read_text())
    assert ctrl.floor["gold"] == 30.0 and rep["scale"]["gold"] == [30.0, 80.0]
    assert rep["per_seed"]["mid.safetensors"]["combined"] == rep["per_seed"]["mid.safetensors"]["face"]  # ArcFace
    assert rep["per_seed"]["mid.safetensors"]["combined_gold"] == rep["per_seed"]["mid.safetensors"]["gold"]
    assert rep["raw"]["best.safetensors"]["gold"] == [40.0, 60.0, 40.0, 60.0]
    assert rep["kinds"]["face"]["sd_diff"] == pytest.approx(0.0, abs=1e-9) and rep["kinds"]["gold"]["sd_diff"] > 1
    assert status["noise"]["gold_sd_diff"] == rep["kinds"]["gold"]["sd_diff"] and "suggested_noise_band_gold" in rep
