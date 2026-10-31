"""Tests for the pipeline shortcuts: crop-first clean-up and the editor's cache (python -m unittest discover tests)."""
import unittest

import cv2
import numpy as np

from rulz.filters import BALANCED, process, process_cached, process_crop
from rulz.shadows import ShadowEdit, Stroke


def _jpeg_picture(h=520, w=700, quality=25):
    """A textured picture with real JPEG blocks."""
    rng = np.random.default_rng(0)
    base = cv2.GaussianBlur(rng.random((h // 8, w // 8, 3)).astype(np.float32), (0, 0), 1.0)
    img = cv2.resize(base, (w, h), interpolation=cv2.INTER_CUBIC)
    img += 0.15 * cv2.GaussianBlur(rng.random((h, w, 3)).astype(np.float32), (0, 0), 0.8)
    img = np.clip(img / img.max() * 255, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


BOXES = [(150, 120, 431, 377), (0, 0, 200, 150), (500, 333, 700, 520), (3, 7, 699, 515)]


class ProcessCropTests(unittest.TestCase):
    def assert_close(self, got, want, msg=""):
        self.assertEqual(got.shape, want.shape, msg)
        diff = np.abs(got.astype(int) - want.astype(int))
        self.assertLessEqual(diff.max(), 3, msg)
        self.assertLess(diff.mean(), 0.05, msg)

    def test_same_as_cleaning_the_whole_picture(self):
        img = _jpeg_picture()
        for settings in ({"jpeg": 60, "detail": 0}, {"jpeg": 100, "detail": 50}, {"jpeg": 0, "detail": 80},
                         {"jpeg": 35, "detail": 100, "color": 0}):
            full = process(img, settings, BALANCED)
            for x0, y0, x1, y1 in BOXES:
                self.assert_close(process_crop(img, (x0, y0, x1, y1), settings, BALANCED), full[y0:y1, x0:x1],
                                  f"{settings} {(x0, y0, x1, y1)}")

    def test_16_bit_and_alpha(self):
        img = _jpeg_picture().astype(np.uint16) * 257
        bgra = np.concatenate([img, np.full(img.shape[:2] + (1,), 40000, np.uint16)], axis=2)
        settings = {"jpeg": 70, "detail": 40}
        full = process(bgra, settings, BALANCED)
        got = process_crop(bgra, BOXES[0], settings, BALANCED)
        x0, y0, x1, y1 = BOXES[0]
        self.assertEqual(got.dtype, np.uint16)
        diff = np.abs(got.astype(int) - full[y0:y1, x0:x1].astype(int))
        self.assertLessEqual(diff.max(), 3 * 257)

    def test_color_noise_and_shadows_use_the_whole_picture(self):
        img = _jpeg_picture()
        x0, y0, x1, y1 = BOXES[0]
        settings = {"color": 50, "jpeg": 40, "detail": 20}
        np.testing.assert_array_equal(process_crop(img, BOXES[0], settings), process(img, settings)[y0:y1, x0:x1])
        shadow = ShadowEdit((Stroke(((0.4, 0.4), (0.6, 0.5)), 0.1),))
        settings = {"jpeg": 40}
        np.testing.assert_array_equal(process_crop(img, BOXES[0], settings, shadow=shadow),
                                      process(img, settings, shadow=shadow)[y0:y1, x0:x1])

    def test_nothing_to_do(self):
        img = _jpeg_picture()
        x0, y0, x1, y1 = BOXES[0]
        np.testing.assert_array_equal(process_crop(img, BOXES[0], {"jpeg": 0}), img[y0:y1, x0:x1])


class ProcessCachedTests(unittest.TestCase):
    def test_same_result_as_process_and_reuses_the_stage(self):
        img = _jpeg_picture(200, 260)
        shadow = ShadowEdit((Stroke(((0.3, 0.3), (0.6, 0.6)), 0.15),))
        cache = None
        stages = []
        for settings, sh in (({"color": 0, "jpeg": 50, "detail": 0}, None),
                             ({"color": 40, "jpeg": 50, "detail": 0}, None),
                             ({"color": 40, "jpeg": 50, "detail": 70}, None),
                             ({"color": 40, "jpeg": 50, "detail": 70}, shadow),
                             ({"color": 10, "jpeg": 50, "detail": 70}, shadow),
                             ({"color": 10, "jpeg": 20, "detail": 70}, shadow)):
            out, cache = process_cached(img, settings, BALANCED, sh, cache)
            np.testing.assert_array_equal(out, process(img, settings, BALANCED, shadow=sh))
            stages.append(cache[1])
        self.assertIs(stages[0], stages[1])
        self.assertIs(stages[1], stages[2])
        self.assertIsNot(stages[2], stages[3])  # shadow areas changed
        self.assertIs(stages[3], stages[4])
        self.assertIsNot(stages[4], stages[5])  # JPEG changed

    def test_noop(self):
        img = _jpeg_picture(64, 64)
        out, cache = process_cached(img, {"color": 0, "jpeg": 0, "detail": 0})
        self.assertIs(out, img)
        self.assertIsNone(cache)


if __name__ == "__main__":
    unittest.main()
