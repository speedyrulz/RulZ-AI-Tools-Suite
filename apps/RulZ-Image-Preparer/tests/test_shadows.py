"""Tests for the shadow lift (python -m unittest discover tests)."""
import unittest

import cv2
import numpy as np

from rulz import filters, shadows
from rulz.filters import process
from rulz.shadows import ShadowEdit, Stroke, lift_shadows, render_mask

LUMA = np.array([0.0722, 0.7152, 0.2126], np.float32)


def _strength(h=400, w=600, soft=3.0):
    """How much of the shadow each pixel of _scene is in, 0..1; soft is its edge's blur in pixels."""
    yy, xx = np.mgrid[:h, :w]
    e = (xx - w / 2) ** 2 / (w / 4) ** 2 + (yy - h / 2) ** 2 / (h / 4) ** 2
    return cv2.GaussianBlur((e < 1).astype(np.float32), (0, 0), soft)


def _scene(h=400, w=600, soft=3.0, albedo=None):
    """A textured surface with a soft-edged, dark, bluish elliptical shadow; returns (shot, truth, masks).
    albedo: an h x w factor on the surface's brightness, to put bright things in the shadow."""
    rng = np.random.default_rng(0)
    tex = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), 1.5)
    tex = (tex - tex.mean()) / tex.std()
    lin = np.array([0.17, 0.27, 0.37], np.float32)[None, None] * np.exp(0.25 * tex)[..., None]
    if albedo is not None:
        lin = lin * albedo[..., None]
    yy, xx = np.mgrid[:h, :w]
    e = (xx - w / 2) ** 2 / (w / 4) ** 2 + (yy - h / 2) ** 2 / (h / 4) ** 2
    soft = _strength(h, w, soft)
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

    def test_soft_edge_painted_past_gets_no_halo(self):
        """Brushed past a soft-edged shadow, its edge is lifted only as far as it was dimmed. It used to get the
        whole shadow's lift, which left a bright ring around the shadow."""
        for soft in (3.0, 12.0):
            shot, truth, _, _ = _scene(soft=soft)
            strength = _strength(soft=soft)
            edge, middle = (strength > 0.1) & (strength < 0.9), strength > 0.99
            err = _log_y(lift_shadows(shot, _over_shadow(overshoot=1.6))) - _log_y(truth)
            with self.subTest(soft=soft):  # the old code: 0.33 and 0.28 brighter than the middle
                self.assertLess(err[edge].mean() - err[middle].mean(), 0.18)
                self.assertLess(np.percentile(err[edge], 99), 0.45)

    def test_bright_things_in_the_shadow_get_its_lift(self):
        """A bright object in the shadow is dimmed as much as the rest, so it should be lifted as much, but its
        edges look like less shadow to the lighting. Enclosed, small or large, it gets the full lift; reaching
        out of the shadow, or with a middle that looks lit, it's lifted about as far as before 0.5.1."""
        h, w = 400, 600
        yy, xx = np.mgrid[:h, :w]
        r2 = (xx - w * 0.42) ** 2 + (yy - h / 2) ** 2
        strength = _strength()
        cases = []
        for size, least in ((32, -0.06), (96, -0.1)):  # discs 2.5x as bright as the ground, barely clipped
            disc = r2 < (size / 2) ** 2
            cases.append((f"disc {size}", np.where(disc, 1.5, 0.6), r2 < (size * 0.35) ** 2, least))
        for width, least in ((16, -0.45), (48, -0.45)):  # bars from the lit ground into the middle (old: -0.4)
            bar = (xx >= 250) & (xx < 250 + width) & (yy >= 30) & (yy < 200)
            inside = bar & (strength > 0.99) & (xx >= 250 + width * 0.2) & (xx < 250 + width * 0.8)
            cases.append((f"bar {width}", np.where(bar, 1.5, 0.6), inside, least))
        cases.append(("6x disc", np.where(r2 < 16 ** 2, 1.5, 0.25), r2 < 11 ** 2, -0.95))  # middle looks lit (old: -0.87)
        for name, albedo, measured, least in cases:
            shot, truth, _, _ = _scene(albedo=albedo.astype(np.float32))
            err = _log_y(lift_shadows(shot, _over_shadow(overshoot=1.3))) - _log_y(truth)
            with self.subTest(name):
                self.assertGreater(err[measured].mean(), least)

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

    def test_working_in_the_painted_box_changes_nothing(self):
        """Full-size steps run only around the painting; the result must match working on the whole picture."""
        shot, _, _, _ = _scene(500, 900)
        shot[0, 0] = 1.2  # out of range, as a Lab round trip can give
        yy, xx = np.mgrid[:500, :900]
        checks = np.where((xx // 12 + yy // 12) % 2, 1.2, 0.2).astype(np.float32)
        tiled, _, _, _ = _scene(500, 900, albedo=checks)  # strong texture the size of the lighting's detail
        whole = lambda m: (slice(0, m.shape[0]), slice(0, m.shape[1]))  # noqa: E731
        small = ShadowEdit((Stroke(((0.2, 0.47), (0.33, 0.52)), 0.04),))  # small, over the shadow's edge
        for shot, edit in ((shot, small),
                           (shot, ShadowEdit((Stroke(((0.0, 0.5), (0.45, 0.5)), 0.05),), 80, 100)),  # off the edge
                           (shot, _over_shadow(500, 900)),
                           (tiled, small)):
            got = lift_shadows(shot, edit)
            self.assertIsNot(got, shot)  # it found a shadow to lift
            saved = shadows._extent
            shadows._extent = whole
            try:
                want = lift_shadows(shot, edit)
            finally:
                shadows._extent = saved
            np.testing.assert_allclose(got, want, rtol=0, atol=2e-6)
            box = shadows._extent(render_mask(edit, *shot.shape[:2]))
            outside = np.ones(shot.shape[:2], bool)
            outside[box] = False
            np.testing.assert_array_equal(got[outside], np.clip(shot, 0, 1)[outside])

    def test_kept_lighting_changes_nothing(self):
        """A brush stroke reuses the picture's measured lighting; the result must be the very same."""
        shot, _, _, _ = _scene(500, 900)
        shot[0, 0], shot[1, 1] = 1.2, -0.1  # out of range, as a Lab round trip can give
        corners = np.ascontiguousarray(np.roll(shot, (250, 450), (0, 1)))  # the shadow split over the corners
        gray = filters._float_bgr(np.round(np.clip(shot[..., 1], 0, 1) * 65535).astype(np.uint16), True)
        paint = Stroke(((0.3, 0.5), (0.5, 0.5), (0.7, 0.5)), 0.12)
        cases = ((shot, ShadowEdit((paint,))),  # paint only
                 (shot, ShadowEdit((paint, Stroke(((0.5, 0.2), (0.5, 0.8)), 0.04, True)), 60, 20)),  # and erase
                 (shot, ShadowEdit((Stroke(((0.0, 0.5), (0.45, 0.5)), 0.05),), 80, 100)),  # off the picture's edge
                 (shot, _over_shadow(500, 900)),
                 (corners, ShadowEdit((Stroke(((0.0, 0.08), (0.12, 0.0)), 0.06),), 90, 0)),  # in a corner
                 (corners, ShadowEdit((Stroke(((0.0, 0.08), (0.12, 0.0)), 0.06),
                                       Stroke(((1.0, 0.92), (0.88, 1.0)), 0.06)), 90)),  # two corners
                 (gray, _over_shadow(500, 900)))  # a gray picture as the pipeline hands it over
        lightings = {}
        for img, edit in cases:
            lighting = lightings.setdefault(id(img), shadows.measure(img))
            before = [a.copy() for a in (lighting.light, lighting.light_s, lighting.tint)]
            want = lift_shadows(img, edit)
            self.assertIsNot(want, img)  # it found a shadow to lift
            np.testing.assert_array_equal(lift_shadows(img, edit, lighting), want)
            for a, b in zip((lighting.light, lighting.light_s, lighting.tint), before):
                np.testing.assert_array_equal(a, b)  # left as it was for the next stroke
        lighting = lightings[id(shot)]
        self.assertIs(lift_shadows(shot, ShadowEdit(), lighting), shot)
        lit = ShadowEdit((Stroke(((0.05, 0.1), (0.2, 0.1)), 0.04),))
        np.testing.assert_array_equal(lift_shadows(shot, lit, lighting), lift_shadows(shot, lit))
        with self.assertRaises(ValueError):
            lift_shadows(shot[:400], _over_shadow(400, 900), lighting)

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
