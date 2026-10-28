"""The appearance body score (DINOv2 on the body window) around a stand-in embedder and window: best-match scoring
against the references that show the window, the median ceiling, the fallback to partial windows, no score for an
image with too little window, the window's geometry on synthetic images (upright, scaled by the torso, background and
out-of-frame grey, coverage), and the config switch."""
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


VECS = {"a": [1.0, 0.0, 0.0], "b": [0.8, 0.6, 0.0], "c": [0.0, 1.0, 0.0], "d": [0.6, 0.8, 0.0],
        "odd": [0.0, 0.0, -1.0], "probe_like_a": [0.96, 0.28, 0.0], "stranger": [0.0, 0.0, 1.0],
        "opposite": [-1.0, 0.0, 0.0], "half": [1.0, 0.0, 0.0]}


def _scorer(covers=None):
    covers = covers or {}
    return bodysim.AppearanceScorer.with_embedder(
        lambda im: np.asarray(VECS[im]), lambda p: None if p == "nobody" else (p, covers.get(p, 1.0)))


def test_best_match_scoring_and_the_median_ceiling():
    s = _scorer({"half": 0.5})
    assert not s.ready and s.score("a") == (None, 0)
    info = s.set_references(["a", "b", "c", "d", "odd", "nobody"])
    assert s.ready and s.ref_count == 5 and [i["person"] for i in info] == [True] * 5 + [False]
    # each reference's best match among the others: a .8, b .96, c .8, d .96, odd 0 -> the median .8 (the mean .70
    # would let one odd photo drag the 100% down)
    assert abs(s.ref_self_similarity - 0.8) < 1e-6
    pct, n = s.score("probe_like_a")
    assert n == 5 and abs(pct - 96.0) < 1e-6  # the best match (a) at cosine 0.96
    assert s.score("stranger") == (0.0, 5)  # no match at all: 0, never negative
    assert s.score("opposite") == (0.0, 5)
    assert s.score("nobody") == (None, 0)
    assert s.score("half") == (None, 0)  # half of the window outside the frame: a portrait, not scored


def test_only_photos_showing_the_window_are_references_unless_too_few():
    s = _scorer({"c": 0.7})
    info = s.set_references(["a", "b", "c"])
    assert s.ref_count == 2 and [i["used"] for i in info] == [True, True, False]  # c shows too little of the body
    assert abs(s.ref_self_similarity - 0.8) < 1e-6 and abs(s.score("d")[0] - 96.0) < 1e-6  # b, not c (0.8)
    s = _scorer({"c": 0.7, "d": 0.7})
    info = s.set_references(["a", "c", "d"])  # only one photo shows the whole window: all three count
    assert s.ref_count == 3 and all(i["used"] for i in info)
    assert abs(s.ref_self_similarity - 0.8) < 1e-6  # a .6, c .8, d .8


def test_one_reference_has_no_ceiling():
    s = _scorer()
    s.set_references(["a"])
    assert s.ready and s.ref_self_similarity is None and abs(s.score("b")[0] - 80.0) < 1e-6 and s.score("b")[1] == 1


RED, BG = (220, 30, 30), (30, 140, 40)


def _scene(size, body_box, sho, hip):
    """An image with a red body on green, its person mask, and pose points with both shoulders and hips."""
    from PIL import Image, ImageDraw

    im = Image.new("RGB", size, BG)
    ImageDraw.Draw(im).rectangle(body_box, fill=RED)
    mask = np.zeros((size[1], size[0]), bool)
    x0, y0, x1, y1 = body_box
    mask[y0:y1 + 1, x0:x1 + 1] = True
    pts = np.zeros((33, 2), np.float32)
    vis = np.zeros(33, np.float32)
    for i, p in zip((bodysim.L_SHO, bodysim.R_SHO, bodysim.L_HIP, bodysim.R_HIP), (*sho, *hip)):
        pts[i], vis[i] = p, 1.0
    return im, pts, vis, mask


def _red(win):
    a = np.asarray(win).astype(int)
    return (np.abs(a - np.array(RED)).sum(-1) < 60)


def test_body_window_is_upright_scaled_by_the_torso_and_cut_out():
    # standing: shoulders at y 200, hips at y 300 (torso 100 px), the body 80 px wide from y 190 to 400
    im, pts, vis, mask = _scene((400, 600), (160, 190, 240, 400), ((170, 200), (230, 200)), ((180, 300), (220, 300)))
    win, cover = bodysim.body_window(im, pts, vis, mask)
    assert abs(win.width - 160) <= 1 and abs(win.height - 190) <= 1 and cover == 1.0  # 1.6 x 1.9 torso lengths
    red = _red(win)
    cols, rows = np.where(red.any(0))[0], np.where(red.any(1))[0]
    assert abs(cols.min() - 40) <= 2 and abs(cols.max() - 120) <= 2 and rows.min() <= 6  # centred, from the neck
    assert tuple(np.asarray(win)[100, 5]) == bodysim.WINDOW_GREY  # the background is gone
    # lying down (shoulders left, hips right): turned upright, the same window
    im2, pts2, vis2, mask2 = _scene((600, 400), (190, 160, 400, 240), ((200, 170), (200, 230)),
                                    ((300, 180), (300, 220)))
    win2, cover2 = bodysim.body_window(im2, pts2, vis2, mask2)
    red2 = _red(win2)
    assert abs(win2.width - win.width) <= 1 and abs(win2.height - win.height) <= 1 and cover2 == 1.0
    assert abs(int(red2.sum()) - int(red.sum())) / red.sum() < 0.05
    rows2, cols2 = np.where(red2.any(1))[0], np.where(red2.any(0))[0]
    assert rows2.max() - rows2.min() > 2 * (cols2.max() - cols2.min())  # taller than wide: upright


def test_body_window_coverage_and_visibility():
    # the hips near the bottom edge: the window runs 57 px past the photo, which stays grey
    im, pts, vis, mask = _scene((400, 300), (165, 190, 235, 299), ((170, 200), (230, 200)), ((180, 290), (220, 290)))
    win, cover = bodysim.body_window(im, pts, vis, mask)
    assert 0.6 < cover < 0.7
    assert tuple(np.asarray(win)[-3, win.width // 2]) == bodysim.WINDOW_GREY  # below the photo: grey, not black
    # without a person mask the background stays, out of the frame still grey
    win, _ = bodysim.body_window(im, pts, vis, None)
    a = np.asarray(win)
    assert tuple(a[50, 3]) == BG and tuple(a[-3, win.width // 2]) == bodysim.WINDOW_GREY
    vis[bodysim.R_HIP] = 0.2  # a hip not visible: no window
    assert bodysim.body_window(im, pts, vis, mask) is None


def test_body_score_config_default_and_preset():
    assert TrainConfig().body_score == "appearance" and TrainConfig(body_score="proportions").body_score == "proportions"
    data = apply_preset(TrainConfig(body_score="proportions").to_dict(), "identity")
    assert data["body_score"] == "proportions" and data["body_backend"] == "mediapipe+sam3"  # masks keep SAM 3
