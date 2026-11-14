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


def _gap_scene(width, notch=False, soft=3.0, h=400, w=600):
    """A soft-edged rectangular shadow on _scene's surface with a lit strip `width` px wide down its middle,
    cutting it in two, or (notch) ending inside it; returns (shot, truth, strength)."""
    _, truth, _, _ = _scene(h, w)  # its surface, re-shadowed below
    yy, xx = np.mgrid[:h, :w]
    shade = ((yy >= 100) & (yy < 300) & (xx >= 120) & (xx < 480)).astype(np.float32)
    shade[(np.abs(xx - w / 2) < width / 2) & (yy < (220 if notch else 300))] = 0
    strength = cv2.GaussianBlur(shade, (0, 0), soft)
    lin = np.power(truth, 2.2)
    illum = 1 - strength[..., None] * (1 - np.array([0.30, 0.22, 0.18], np.float32))
    return np.power(lin * illum, 1 / 2.2).astype(np.float32), truth, strength


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

    def test_lit_gaps_between_shadows_are_left_lit(self):
        """A lit strip between two painted shadows, or cutting into one, is lit ground: it and its soft edges
        are lifted only as far as they're shaded. Before 0.5.2 a strip narrower than about an eighth of the
        picture was lifted like a bright object, its edges about 0.6 brighter than the shadow's middle."""
        edit = ShadowEdit((Stroke(tuple((x / 600, 0.5) for x in np.linspace(90, 510, 8)), 0.25),), strength=100)
        yy, xx = np.mgrid[:400, :600]
        for width, notch, soft in ((16, False, 3.0), (32, False, 3.0), (16, True, 3.0), (32, True, 8.0)):
            shot, truth, strength = _gap_scene(width, notch, soft)
            err = _log_y(lift_shadows(shot, edit)) - _log_y(truth)
            along = (yy > 130) & (yy < (190 if notch else 270))
            lit = (strength < 0.05) & (np.abs(xx - 300) < width / 2) & along
            edges = (strength > 0.1) & (strength < 0.9) & (np.abs(xx - 300) < width / 2 + 3 * soft) & along
            middle = (strength > 0.99) & (np.abs(xx - 300) > width / 2 + 40)
            with self.subTest(width=width, notch=notch, soft=soft):  # before 0.5.2: 0.57 to 0.66
                self.assertLess(err[edges].mean() - err[middle].mean(), 0.2)
                if lit.any():  # before 0.5.2: 0.83 at 16 px
                    self.assertLess(err[lit].mean(), 0.2)

    def test_lit_ground_by_hand(self):
        """_lit_ground on small fields worked out by hand: growth from outside, the steepest way down, a row's
        end not running on into the next row, a bright thing taking its shaded part first, odd shapes and
        non-finite values."""
        def lit(level, outside, depth=1.0):
            return shadows._lit_ground(np.array(level, np.float32), np.array(outside), depth)

        # lit ground from the outside cell; 0.5 and 0.9 lead down to it; 1.0 leads down to 0.7, which leads nowhere
        np.testing.assert_array_equal(lit([[0.0, 0.1, 0.5, 0.9, 1.0, 0.7]], [[True] + [False] * 5]),
                                      [[True, True, True, True, False, False]])
        # 0.6 ends a row and has nothing lower beside it: it doesn't run on into the lit 0.0 that starts the next
        # row (the 0.9s beside that 0.0 do lead down to it)
        np.testing.assert_array_equal(lit([[0.9, 0.5, 0.9], [0.9, 0.9, 0.6], [0.0, 0.9, 0.9]],
                                          [[False] * 3, [False] * 3, [True, False, False]]),
                                      [[False] * 3, [True, False, False], [True, True, False]])
        # a bright thing (-0.6) reaches the 0.1 beside it before the lit ground can (that needs 0.3), and wins the
        # 0.3 they both reach at once
        painted = [[True, False, False, False, False]]
        np.testing.assert_array_equal(lit([[0.0, 0.3, 0.3, 0.1, -0.6]], painted), [[True, True, False, False, False]])
        # only a little brighter (-0.4, never past BRIGHT_PEAK) it's texture, and the lit ground gets it all
        np.testing.assert_array_equal(lit([[0.0, 0.3, 0.3, 0.1, -0.4]], painted), [[True, True, True, True, False]])
        # in a shallow shadow the floors count: -0.6 of a depth of 0.3 is only 0.18 brighter
        np.testing.assert_array_equal(lit([[0.0, 0.3, 0.3, 0.1, -0.6]], painted, 0.3), [[True, True, True, True, False]])
        # a bright thing outside the painting comes in too
        np.testing.assert_array_equal(lit([[-0.6, 0.1, 0.3, 0.0]], [[True, False, False, True]]),
                                      [[False, False, False, True]])
        self.assertFalse(lit([[0.0]], [[False]]).any())  # nothing outside: no lit ground
        np.testing.assert_array_equal(lit([[0.0], [0.2], [np.nan], [np.inf], [0.1]], [[True]] + [[False]] * 4),
                                      [[True], [True], [False], [False], [False]])
        # a cell that isn't a number beside 1.0 doesn't stop it going down through 0.5 to the lit 0.0
        np.testing.assert_array_equal(lit([[0.0, 0.5], [np.nan, 1.0]], [[True, False], [False, False]]),
                                      [[True, True], [False, True]])

    def test_lit_ground_depends_only_on_the_painting(self):
        """_lit_ground works on the painting's box, so field around it (here a far-off bright cell, and the
        box-vs-whole tests' patched _extent) changes nothing, even when a long winding painted path runs past
        the growing's cap."""
        level = np.ones((40, 40), np.float32)
        outside = np.ones((40, 40), bool)
        for row in range(2, 38, 2):  # a one-cell-wide painted path winding down the field, 702 cells long
            outside[row, 1:39] = False
            outside[row + 1, 38 if row % 4 == 2 else 1] = False
        level[~outside] = 0.3
        level[2, 0] = 0.0  # where it starts, on lit ground outside
        far = level.copy()
        far[39, 39] = -0.6  # a bright cell outside the painting, far from it
        want = shadows._lit_ground(level, outside, 1.0)
        self.assertLess(want[~outside].sum(), (~outside).sum())  # the cap is reached
        saved = shadows._extent
        shadows._extent = lambda m: (slice(0, m.shape[0]), slice(0, m.shape[1]))
        try:
            for got in (shadows._lit_ground(far, outside, 1.0), shadows._lit_ground(level, outside, 1.0)):
                np.testing.assert_array_equal(got, want)
        finally:
            shadows._extent = saved

    def test_lit_level_leaves_bright_things_out(self):
        """The lit ground's light is measured without the bright things (and dark fringes) outside the painting,
        so next to the painting a bright thing reads as brighter than the lit ground, not as lit ground."""
        light = np.zeros((40, 60), np.float32)
        outside = np.ones_like(light, bool)
        outside[10:30, 15:45] = False  # the painting
        light[0:10, 25:32] = 1.0  # a bright thing outside it, up against it
        light[30:40, 0:10] = -0.6  # a darker fringe outside it
        level = shadows._lit_level(light, outside, 1.0)
        assert level is not None
        np.testing.assert_allclose(level[0:10, 25:32], -1.0, atol=1e-5)
        np.testing.assert_allclose(level[30:40, 0:10], 0.6, atol=1e-5)
        np.testing.assert_allclose(level[10:30, 15:45], 0.0, atol=1e-5)  # the painting: filled from plain lit ground
        self.assertIsNone(shadows._lit_level(light, np.zeros_like(outside), 1.0))

    def test_bright_things_in_the_shadow_get_its_lift(self):
        """A bright object in the shadow is dimmed as much as the rest, so it should be lifted as much, but its
        edges look like less shadow to the lighting. Enclosed, small or large, it gets the full lift; reaching
        out of the shadow, or with a middle that looks lit, it's lifted about as far as before 0.5.1. That holds
        for one about as many times brighter than the ground as the shadow is darker, whose shaded part is at the
        lit level, so it isn't taken for lit ground (0.5.2): also with the painting stopped just past a soft
        shadow edge, or with the object's own edge soft (where a first try of 0.5.2 lifted them about 0.5 less)."""
        h, w = 400, 600
        yy, xx = np.mgrid[:h, :w]
        r2 = (xx - w * 0.42) ** 2 + (yy - h / 2) ** 2
        cases = []  # name, albedo, where it's measured (a function of the shadow's strength), least, soft, overshoot
        for size, least in ((32, -0.06), (96, -0.1)):  # discs 2.5x as bright as the ground, barely clipped
            disc = r2 < (size / 2) ** 2
            cases.append((f"disc {size}", np.where(disc, 1.5, 0.6), lambda _, n=size: r2 < (n * 0.35) ** 2,
                          least, 3.0, 1.3))
        # bars from the lit ground into the middle, 2.5x (old: -0.4) and 4x as bright as the ground (old: -0.49
        # and -1.01; taken for lit ground they'd be -1.3); then a 4x bar with the painting stopped just past a
        # soft shadow edge (old: -0.74, first try: -1.19), a 3x bar with a soft edge itself (old: -0.46, first
        # try: -0.99), and a wide one from the picture's top edge, standing on the lit ground (old: -0.49, second
        # try: -0.93)
        for width, ratio, least, soft, over, blur, top in (
                (16, 2.5, -0.45, 3.0, 1.3, 0, 30), (48, 2.5, -0.45, 3.0, 1.3, 0, 30), (16, 4.0, -0.6, 3.0, 1.3, 0, 30),
                (48, 4.0, -1.15, 3.0, 1.3, 0, 30), (48, 4.0, -0.85, 16.0, 1.05, 0, 30), (48, 3.0, -0.6, 8.0, 1.3, 6.0, 30),
                (72, 3.5, -0.6, 8.0, 1.05, 0, 0)):
            bar = ((xx >= 250) & (xx < 250 + width) & (yy >= top) & (yy < 200)).astype(np.float32)
            if blur:
                bar = cv2.GaussianBlur(bar, (0, 0), blur)
            albedo = (1.2 / ratio) * np.exp(np.log(ratio) * bar)  # 1.2 on the bar, 1.2 / ratio around it
            cases.append((f"bar {width} {ratio}x soft {soft} over {over} blur {blur}", albedo,
                          lambda s, b=bar, n=width: (b > 0.99) & (s > 0.99) & (xx >= 250 + n * 0.2) & (xx < 250 + n * 0.8),
                          least, soft, over))
        cases.append(("6x disc", np.where(r2 < 16 ** 2, 1.5, 0.25), lambda _: r2 < 11 ** 2, -0.95, 3.0, 1.3))  # old: -0.87
        for name, albedo, measured, least, soft, over in cases:
            shot, truth, _, _ = _scene(soft=soft, albedo=albedo.astype(np.float32))
            err = _log_y(lift_shadows(shot, _over_shadow(overshoot=over))) - _log_y(truth)
            with self.subTest(name):
                self.assertGreater(err[measured(_strength(soft=soft))].mean(), least)

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
        gapped, _, _ = _gap_scene(32)  # lit ground beside the painting's box, reaching into it
        for shot, edit in ((shot, small),
                           (shot, ShadowEdit((Stroke(((0.0, 0.5), (0.45, 0.5)), 0.05),), 80, 100)),  # off the edge
                           (shot, _over_shadow(500, 900)),
                           (tiled, small),
                           (gapped, ShadowEdit((Stroke(((0.3, 0.5), (0.55, 0.5)), 0.08),)))):
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
