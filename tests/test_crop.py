"""Tests for the crop box (python -m unittest discover tests)."""
import unittest

import numpy as np

from rulz.crop import CropSettings, compute_crop_box, crop_box


def _mask(h, w, y0, y1, x0, x1):
    m = np.zeros((h, w), np.float32)
    m[y0:y1, x0:x1] = 1
    return m


class CropBoxTests(unittest.TestCase):
    def test_empty_mask(self):
        self.assertIsNone(compute_crop_box(np.zeros((10, 10)), 10, 1, 0.0))

    def test_padding_against_crop(self):
        # 80 px mask, 10% of the final crop each side: crop = 80 / 0.8 = 100, so 10 px per side.
        box = compute_crop_box(_mask(400, 400, 100, 180, 100, 180), 10, 1, 0.0, max_aspect_ratio=None)
        self.assertEqual(box, (90, 90, 190, 190))

    def test_padding_against_mask_and_image(self):
        m = _mask(400, 400, 100, 180, 100, 180)
        self.assertEqual(compute_crop_box(m, 10, 1, 0.0, padding_reference="mask"), (92, 92, 188, 188))
        self.assertEqual(compute_crop_box(m, 10, 1, 0.0, padding_reference="image"), (60, 60, 220, 220))

    def test_per_edge_and_clamping(self):
        m = _mask(200, 200, 0, 100, 50, 150)
        box = compute_crop_box(m, (20, 0, 0, 0), 1, 0.0, padding_reference="mask")
        self.assertEqual(box, (50, 0, 150, 100))  # the top border is clamped to the image

    def test_aspect_limit_widens_a_tall_crop(self):
        m = _mask(1000, 1000, 100, 700, 450, 550)  # 100 x 600
        x0, y0, x1, y1 = compute_crop_box(m, 0, 1, 0.0, max_aspect_ratio=1.5)
        self.assertEqual((y0, y1), (100, 700))
        self.assertEqual(x1 - x0, 400)
        self.assertLessEqual(x0, 450)
        self.assertGreaterEqual(x1, 550)

    def test_divisible_by(self):
        x0, y0, x1, y1 = compute_crop_box(_mask(500, 500, 10, 111, 20, 133), 0, 64, 0.0)
        self.assertEqual(((x1 - x0) % 64, (y1 - y0) % 64), (0, 0))
        self.assertTrue(x0 <= 20 and x1 >= 133 and y0 <= 10 and y1 >= 111)

    def test_divisible_by_never_cuts_into_the_mask(self):
        x0, _, x1, _ = compute_crop_box(_mask(100, 100, 0, 100, 0, 90), 0, 64, 0.0)
        self.assertEqual((x0, x1), (0, 100))

    def test_low_resolution_mask_is_scaled_to_the_frame(self):
        box = compute_crop_box(_mask(100, 100, 25, 50, 25, 50), 0, 1, 0.0, 400, 400)
        self.assertEqual(box, (100, 100, 200, 200))

    def test_threshold_and_batch_union(self):
        m = np.zeros((2, 50, 50), np.float32)
        m[0, 10:20, 10:20] = 1.0
        m[1, 30:40, 30:40] = 0.4
        self.assertEqual(compute_crop_box(m, 0, 1, 0.5), (10, 10, 20, 20))
        self.assertEqual(compute_crop_box(m, 0, 1, 0.0), (10, 10, 40, 40))

    def test_settings(self):
        s = CropSettings(padding_mode="per_edge", padding_top=5, aspect_limit="off")
        self.assertEqual(s.padding(), (5, 10.0, 10.0, 10.0))
        self.assertEqual(crop_box(_mask(100, 100, 40, 60, 40, 60), CropSettings(aspect_limit="off"), 100, 100),
                         (37, 37, 63, 63))  # 2.5 px rounds up
        back = CropSettings.from_dict({**s.to_dict(), "aspect_limit": "7:1", "divisible_by": "x", "junk": 1})
        self.assertEqual((back.padding_mode, back.aspect_limit, back.divisible_by), ("per_edge", "2:3 / 3:2", 1))


if __name__ == "__main__":
    unittest.main()
