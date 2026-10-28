"""Pose-independent body-shape scores (bodyshape.py): the calibrated profile, 3D bone ratios that do not change when a
limb bends, the upright / frontal gate for silhouette widths, girths from mesh slices, and the SMPL-X option's
manual download."""
import math

import numpy as np
import pytest

from rulztrainer import bodyshape as BS
from rulztrainer.bodysim import (L_ANK, L_EAR, L_ELB, L_HIP, L_KNEE, L_SHO, L_WRI, NOSE, R_ANK, R_EAR, R_ELB, R_HIP,
                                 R_KNEE, R_SHO, R_WRI)


def test_a_steady_measurement_counts_more_than_a_noisy_one():
    refs = [{"steady": 1.00 + d, "noisy": 1.0 + 10 * d} for d in (-0.01, 0.0, 0.01, 0.005, -0.005)]
    prof = BS.Profile(refs)
    assert prof.sd["steady"] == pytest.approx(0.03)  # the floor: 3% of the median
    assert prof.sd["noisy"] > 0.05
    off_steady, _ = prof.similarity({"steady": 1.10, "noisy": 1.0, "x": 1}, min_keys=2)
    off_noisy, _ = prof.similarity({"steady": 1.0, "noisy": 1.10}, min_keys=2)
    assert off_steady < off_noisy  # 10% off a steady measurement is far; 10% off a noisy one is not
    assert prof.similarity({"steady": 1.0}, min_keys=2) == (None, 1)


def test_the_references_own_level_leaves_each_one_out():
    refs = [{"a": 1.0 + d, "b": 2.0 - d, "c": 0.5} for d in (-0.02, 0.0, 0.02, 0.01)]
    own = BS.Profile.self_similarity(refs)
    assert 0.0 < own < 1.0
    assert BS.Profile.self_similarity(refs[:2]) is None  # one other picture is no profile


def _standing() -> tuple[np.ndarray, np.ndarray]:
    """A 3D skeleton (y up here, metres) standing straight: [33, 3] and all visible."""
    xyz = np.zeros((33, 3), dtype=np.float64)
    xyz[NOSE] = (0, 0.65, 0.05)
    xyz[L_EAR], xyz[R_EAR] = (0.07, 0.62, 0), (-0.07, 0.62, 0)
    xyz[L_SHO], xyz[R_SHO] = (0.19, 0.5, 0), (-0.19, 0.5, 0)
    xyz[L_ELB], xyz[R_ELB] = (0.21, 0.22, 0), (-0.21, 0.22, 0)
    xyz[L_WRI], xyz[R_WRI] = (0.22, -0.02, 0), (-0.22, -0.02, 0)
    xyz[L_HIP], xyz[R_HIP] = (0.13, 0.0, 0), (-0.13, 0.0, 0)
    xyz[L_KNEE], xyz[R_KNEE] = (0.12, -0.42, 0), (-0.12, -0.42, 0)
    xyz[L_ANK], xyz[R_ANK] = (0.12, -0.84, 0), (-0.12, -0.84, 0)
    return xyz, np.ones(33)


def test_3d_bone_ratios_do_not_change_when_a_leg_bends():
    xyz, vis = _standing()
    straight = BS.bone_ratios_3d(xyz, vis)
    bent = xyz.copy()  # sitting: the shins swing forward 90 degrees around the knees, same lengths
    for knee, ank in ((L_KNEE, L_ANK), (R_KNEE, R_ANK)):
        shin = np.linalg.norm(xyz[ank] - xyz[knee])
        bent[ank] = xyz[knee] + (0, 0, shin)
    assert BS.bone_ratios_3d(bent, vis) == pytest.approx(straight)
    assert {"3d shoulder/hip width", "3d torso/leg length", "3d thigh/shin"} <= set(straight)
    vis2 = vis.copy()
    vis2[L_KNEE] = vis2[R_KNEE] = 0.0  # knees hidden: no leg ratios, the rest stays
    part = BS.bone_ratios_3d(xyz, vis2)
    assert "3d thigh/shin" not in part and "3d shoulder/hip width" in part


def test_silhouette_widths_only_where_standing_upright_and_facing_the_camera():
    xyz, vis = _standing()
    pts = np.stack([300 + xyz[:, 0] * 400, 500 - xyz[:, 1] * 400], 1)  # image y grows downward
    assert BS.upright_frontal(pts, xyz, vis, size=(600, 900))
    turned = xyz.copy()
    turned[L_SHO, 2], turned[L_HIP, 2] = 0.3, 0.3  # one side far nearer the camera
    assert not BS.upright_frontal(pts, turned, vis, size=(600, 900))
    kneel = xyz.copy()
    kneel[L_ANK] = xyz[L_KNEE] + (0, 0, -0.42)
    assert not BS.upright_frontal(pts, kneel, vis, size=(600, 900))
    assert not BS.upright_frontal(pts, xyz, vis, size=(600, 600))  # feet out of the frame


def test_a_girth_is_the_convex_perimeter_of_a_slice():
    t = np.linspace(0, 2 * math.pi, 200, endpoint=False)
    circle = np.stack([0.15 * np.cos(t), 0.15 * np.sin(t)], 1)
    assert BS._convex_perimeter(circle) == pytest.approx(2 * math.pi * 0.15, rel=1e-3)
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5]])
    assert BS._convex_perimeter(square) == pytest.approx(4.0)


def test_tpose_measurements_of_a_cylinder_person():
    """A crude body: a torso cylinder (part 0) between y 0.9 and 1.5, a thigh cylinder (part 1) below, 1.6 m tall."""
    t = np.linspace(0, 2 * math.pi, 40, endpoint=False)
    verts, parts = [], []
    for y in np.linspace(0.9, 1.5, 61):
        r = 0.16 if y < 1.1 else 0.12 if y < 1.3 else 0.15  # hips, waist, bust
        verts += [(r * math.cos(a), y, r * math.sin(a)) for a in t]
        parts += [0] * len(t)
    for y in np.linspace(0.5, 0.9, 41):
        verts += [(0.09 + 0.08 * math.cos(a), y, 0.08 * math.sin(a)) for a in t]
        parts += [1] * len(t)
    verts += [(0, 0.0, 0), (0, 1.6, 0)]  # soles and the top of the head
    parts += [7, 15]
    joints = np.zeros((55, 3))
    for j, y in ((BS.PELVIS, 0.95), (BS.HIP_L, 0.9), (BS.HIP_R, 0.9), (BS.SPINE1, 1.05), (BS.SPINE2, 1.2),
                 (BS.SPINE3, 1.35), (BS.NECK, 1.5), (BS.KNEE_L, 0.5), (BS.KNEE_R, 0.5), (BS.ANKLE_L, 0.1),
                 (BS.ANKLE_R, 0.1)):
        joints[j, 1] = y
    m = BS.tpose_measurements(np.array(verts), joints, np.array(parts))
    assert m["smplx waist/height"] == pytest.approx(2 * math.pi * 0.12 / 1.6, rel=0.02)
    assert m["smplx hip/height"] == pytest.approx(2 * math.pi * 0.16 / 1.6, rel=0.02)
    assert m["smplx waist/hip"] == pytest.approx(0.12 / 0.16, rel=0.03)
    assert m["smplx thigh/height"] == pytest.approx(2 * math.pi * 0.08 / 1.6, rel=0.02)


def test_smplx_needs_nlf_downloaded_by_hand(tmp_path):
    assert BS.find_nlf("", tmp_path) is None
    with pytest.raises(FileNotFoundError, match="noncommercial"):
        BS.SMPLXScorer("", tmp_path, "cpu")
    (tmp_path / "nlf").mkdir()
    f = tmp_path / "nlf" / "nlf_l_multi_0.3.2.patch6.torchscript"
    f.write_bytes(b"")
    assert BS.find_nlf("", tmp_path) == f
    assert BS.find_nlf(str(f), "elsewhere") == f


def test_body_score_must_be_known():
    from rulztrainer.config import TrainConfig

    for name in ("appearance", "proportions", "skeleton3d", "smplx"):
        TrainConfig(body_score=name).validate()
    with pytest.raises(ValueError, match="body_score"):
        TrainConfig(body_score="smpl").validate()
