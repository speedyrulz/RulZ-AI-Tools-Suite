"""Tests for the shadow lift (python -m unittest discover tests)."""
import unittest

import cv2
import numpy as np

from rulz.filters import process
from rulz.shadows import ShadowEdit, Stroke, lift_shadows, render_mask

LUMA = np.array([0.0722, 0.7152, 0.2126], np.float32)


def _scene(h=400, w=600):
    """A textured surface with a soft-edged, dark, bluish elliptical shadow; returns (shot, truth, masks)."""
    rng = np.random.default_rng(0)
    tex = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), 1.5)
    tex = (tex - tex.mean()) / tex.std()
    lin = np.array([0.17, 0.27, 0.37], np.float32)[None, None] * np.exp(0.25 * tex)[..., None]
    yy, xx = np.mgrid[:h, :w]
    e = (xx - w / 2) ** 2 / (w / 4) ** 2 + (yy - h / 2) ** 2 / (h / 4) ** 2
    soft = cv2.GaussianBlur((e < 1).astype(np.float32), (0, 0), 3)
    illum = 1 - soft[..., None] * (1 - np.array([0.30, 0.22, 0.18], np.float32))
    shot = np.power(lin * illum, 1 / 2.2).astype(np.float32)
    truth = np.power(lin, 1 / 2.2).astype(np.float32)
    return shot, truth, e < 0.8, e > 1.6


def _log_y(img):
    return np.log(np.power(img, 2.2) @ LUMA + 1e-4)


def _over_shadow(h=400, w=600, overshoot=1.3):
    pts = tuple((x / w, 0.5) for x in np.linspace(w / 2 - w / 4 * overshoot, w / 2 + w / 4 * overshoot, 8))
    return ShadowEdit((Stroke(pts, (h / 4) * overshoot / max(h, w)),), strength=100)


class ShadowTests(unittest.TestCase):
    def test_lifts_to_the_surroundings_and_keeps_texture(self):
        shot, truth, inner, outer = _scene()
        out = lift_shadows(shot, _over_shadow())
        lo, lt = _log_y(out), _log_y(truth)
        self.assertAlmostEqual(lo[inner].mean(), lt[inner].mean(), delta=0.08)  # brightness matched
        self.assertGreater(lo[inner].std() / lt[inner].std(), 0.9)  # texture contrast kept
        self.assertLess(np.abs(out[inner].mean(0) - truth[inner].mean(0)).max(), 0.03)  # blue tint removed
        # lit surface the brush ran over is left nearly alone (only dark texture blobs move a little)
        change = np.abs(out[outer] - shot[outer]).max(1)
        self.assertLess(change.mean(), 0.01)
        self.assertLess(np.percentile(change, 99), 0.08)

    def test_strength_scales_the_lift(self):
        shot, _, inner, _ = _scene()
        full = _log_y(lift_shadows(shot, _over_shadow()))[inner].mean()
        half = _log_y(lift_shadows(shot, ShadowEdit(_over_shadow().strokes, strength=50)))[inner].mean()
        before = _log_y(shot)[inner].mean()
        self.assertAlmostEqual(half - before, (full - before) / 2, delta=0.1)

    def test_nothing_painted_or_lit_area_painted(self):
        shot, _, _, _ = _scene()
        self.assertIs(lift_shadows(shot, ShadowEdit()), shot)
        lit = ShadowEdit((Stroke(((0.05, 0.1), (0.2, 0.1)), 0.04),))
        self.assertLess(np.abs(lift_shadows(shot, lit) - shot).max(), 0.02)

    def test_erase_and_mask(self):
        paint = Stroke(((0.2, 0.5), (0.8, 0.5)), 0.05)
        erase = Stroke(((0.5, 0.2), (0.5, 0.8)), 0.05, erase=True)
        m = render_mask(ShadowEdit((paint, erase), blend=0), 200, 300)
        self.assertGreater(m[100, 90], 0.99)
        self.assertLess(m[100, 150], 0.01)
        self.assertEqual(m[10, 10], 0.0)
        self.assertFalse(ShadowEdit((erase,)).active)

    def test_round_trip_and_pipeline(self):
        edit = ShadowEdit((Stroke(((0.1, 0.2), (0.3, 0.4)), 0.05), Stroke(((0.5, 0.5),), 0.02, True)), 40, 70)
        self.assertEqual(ShadowEdit.from_dict(edit.to_dict()), edit)
        shot, _, inner, _ = _scene()
        img16 = np.round(shot * 65535).astype(np.uint16)
        out = process(img16, {"color": 0, "jpeg": 0, "detail": 0}, shadow=_over_shadow())
        self.assertEqual(out.dtype, np.uint16)
        self.assertGreater(out[inner].mean(), img16[inner].mean() * 1.3)
        gray = process(img16[..., 1], {"color": 0}, shadow=_over_shadow())
        self.assertEqual(gray.shape, img16.shape[:2])


if __name__ == "__main__":
    unittest.main()
