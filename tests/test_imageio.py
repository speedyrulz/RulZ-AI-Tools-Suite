"""Tests for loading pictures: one decode, EXIF orientation, alpha and 16-bit (python -m unittest discover tests)."""
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from rulz import imageio

try:
    from PIL import Image
except ImportError:  # Pillow comes with the Auto crop extras only
    Image = None


def _picture(h=40, w=60, channels=3):
    """Asymmetric, so every rotation and flip looks different."""
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, (h, w, channels), dtype=np.uint8)
    img[:10, :10, :3] = (0, 0, 255)
    return img


def _exif(orientation):
    ex = Image.Exif()
    ex[0x0112] = orientation
    return ex


@unittest.skipIf(Image is None, "needs Pillow to write EXIF")
class OrientationTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.dir.name)

    def tearDown(self):
        self.dir.cleanup()

    def _save(self, name, bgr, orientation, **kw):
        mode = cv2.COLOR_BGRA2RGBA if bgr.shape[2] == 4 else cv2.COLOR_BGR2RGB
        path = self.tmp / name
        Image.fromarray(cv2.cvtColor(bgr, mode)).save(path, exif=_exif(orientation), **kw)
        return path

    def test_orient_matches_opencv(self):
        img = _picture()
        for o in range(1, 9):
            path = self._save(f"o{o}.png", img, o)
            want = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)  # OpenCV rotates this one
            np.testing.assert_array_equal(imageio.orient(img, o), want, err_msg=f"orientation {o}")

    def test_png_and_webp_with_alpha_are_rotated(self):
        img = _picture(channels=4)
        for ext, kw in ((".png", {}), (".webp", {"lossless": True, "exact": True})):
            for o in (3, 6, 8):
                got = imageio.load(self._save(f"a{o}{ext}", img, o, **kw))
                self.assertEqual(got.shape[2], 4, ext)
                np.testing.assert_array_equal(got, imageio.orient(img, o), err_msg=f"{ext} orientation {o}")

    def test_png_without_alpha(self):
        img = _picture()
        np.testing.assert_array_equal(imageio.load(self._save("rgb.png", img, 6)), imageio.orient(img, 6))
        np.testing.assert_array_equal(imageio.load(self._save("plain.png", img, 1)), img)

    def test_jpeg(self):
        img = cv2.resize(_picture(), (600, 400), interpolation=cv2.INTER_CUBIC)  # smooth, so JPEG keeps it
        path = self._save("o6.jpg", img, 6, quality=95)
        got = imageio.load(path)
        self.assertEqual(got.shape, (600, 400, 3))
        self.assertLess(np.abs(got.astype(int) - imageio.orient(img, 6)).mean(), 12)

    def test_reduced_jpeg_thumbnail_keeps_orientation(self):
        img = cv2.resize(_picture(40, 60), (1200, 800), interpolation=cv2.INTER_NEAREST)
        path = self._save("big.jpg", img, 6, quality=95)
        full = imageio.load(path)
        thumb = imageio.load(path, max_side=250)
        self.assertEqual(full.shape, (1200, 800, 3))
        self.assertEqual(thumb.shape, (300, 200, 3))  # 1/4: the smallest still at least 250 px long
        ref = cv2.resize(full, (200, 300), interpolation=cv2.INTER_AREA)
        self.assertLess(np.abs(thumb.astype(int) - ref).mean(), 8)

    def test_tiff(self):
        img = _picture()
        np.testing.assert_array_equal(imageio.load(self._save("o6.tif", img, 6)), imageio.orient(img, 6))


class DepthTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.dir.name)

    def tearDown(self):
        self.dir.cleanup()

    def _roundtrip(self, name, img):
        path = self.tmp / name
        self.assertTrue(cv2.imwrite(str(path), img))
        return imageio.load(path)

    def test_16_bit_png_and_tiff(self):
        rng = np.random.default_rng(2)
        for channels in (3, 4):
            img = rng.integers(0, 65536, (30, 50, channels), dtype=np.uint16)
            for ext in (".png", ".tif"):
                got = self._roundtrip(f"d{channels}{ext}", img)
                self.assertEqual(got.dtype, np.uint16, ext)
                np.testing.assert_array_equal(got, img, err_msg=f"{ext} {channels} channels")

    def test_gray_and_alpha_png(self):
        rng = np.random.default_rng(3)
        gray = rng.integers(0, 256, (30, 50), dtype=np.uint8)
        np.testing.assert_array_equal(self._roundtrip("g.png", gray), gray)
        bgra = rng.integers(0, 256, (30, 50, 4), dtype=np.uint8)
        np.testing.assert_array_equal(self._roundtrip("a.png", bgra), bgra)

    def test_gray_jpeg_thumbnail_stays_gray(self):
        gray = np.tile(np.arange(0, 256, dtype=np.uint8), (512, 4))  # 512 x 1024
        path = self.tmp / "g.jpg"
        cv2.imwrite(str(path), gray)
        got = imageio.load(path, max_side=200)
        self.assertEqual(got.ndim, 2)
        self.assertEqual(got.shape, (128, 256))  # 1/4

    def test_unreadable(self):
        path = self.tmp / "bad.jpg"
        path.write_bytes(b"\xff\xd8not a picture")
        with self.assertRaises(ValueError):
            imageio.load(path)

    def test_to_qimage_bgr(self):
        img = np.zeros((4, 6, 3), np.uint8)
        img[0, 0] = (255, 0, 0)  # blue in BGR
        q = imageio.to_qimage(img)
        c = q.pixelColor(0, 0)
        self.assertEqual((c.red(), c.green(), c.blue()), (0, 0, 255))
        self.assertEqual((q.width(), q.height()), (6, 4))


if __name__ == "__main__":
    unittest.main()
