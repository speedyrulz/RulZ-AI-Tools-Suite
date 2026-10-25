"""The appearance body score (DINOv2 on the person crop) around a stand-in embedder: best-match scoring, the
leave-one-out self-similarity ceiling, no person -> no score, and the config switch."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import bodysim, facesim  # noqa: E402
from rulztrainer.config import TrainConfig, apply_preset  # noqa: E402


@pytest.fixture(autouse=True)
def _fake_paths(monkeypatch):
    monkeypatch.setattr(facesim, "expand_refs", lambda paths: list(paths))  # the stand-in paths are not files

VECS = {"a": np.array([1.0, 0.0, 0.0]), "b": np.array([0.8, 0.6, 0.0]), "c": np.array([0.0, 1.0, 0.0]),
        "probe_like_a": np.array([0.96, 0.28, 0.0]), "stranger": np.array([0.0, 0.0, 1.0]),
        "opposite": np.array([-1.0, 0.0, 0.0])}


def _scorer():
    return bodysim.AppearanceScorer.with_embedder(lambda im: VECS[im], lambda p: None if p == "nobody" else p)


def test_best_match_scoring_and_the_leave_one_out_ceiling():
    s = _scorer()
    assert not s.ready and s.score("a") == (None, 0)
    info = s.set_references(["a", "b", "c", "nobody"])
    assert s.ready and s.ref_count == 3 and [i["person"] for i in info] == [True, True, True, False]
    # each reference's best match among the others: a-b 0.8, b-a 0.8, c-b 0.6 -> mean 0.7333
    assert abs(s.ref_self_similarity - (0.8 + 0.8 + 0.6) / 3) < 1e-9
    pct, n = s.score("probe_like_a")
    assert n == 3 and abs(pct - 96.0) < 1e-9  # the best match (a) at cosine 0.96
    assert s.score("stranger") == (0.0, 3)  # no match at all: 0, never negative
    assert s.score("opposite") == (0.0, 3)
    assert s.score("nobody") == (None, 0)


def test_one_reference_has_no_ceiling():
    s = _scorer()
    s.set_references(["a"])
    assert s.ready and s.ref_self_similarity is None and s.score("b") == (80.0, 1)


def test_body_score_config_default_and_preset():
    assert TrainConfig().body_score == "appearance" and TrainConfig(body_score="proportions").body_score == "proportions"
    data = apply_preset(TrainConfig(body_score="proportions").to_dict(), "identity")
    assert data["body_score"] == "proportions" and data["body_backend"] == "mediapipe+sam3"  # masks keep SAM 3
