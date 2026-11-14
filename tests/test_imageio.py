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

    def test_reduced_load_tells_the_full_size(self):
        """Auto crop decodes a big JPEG small but needs the picture's real size for the crop box."""
        img = cv2.resize(_picture(40, 60), (1203, 805), interpolation=cv2.INTER_NEAREST)  # odd: sizes round up
        gray = self.tmp / "sized_gray.jpg"
        Image.fromarray(img[:, :, 0]).save(gray, exif=_exif(6), quality=90)
        small, size = imageio.load_sized(gray, max_side=250)
        self.assertEqual((small.ndim, size), (2, imageio.load(gray).shape))
        for o in range(1, 9):
            path = self._save(f"sized{o}.jpg", img, o, quality=90)
            full = imageio.load(path)
            small, size = imageio.load_sized(path, max_side=250)
            self.assertLess(max(small.shape[:2]), max(full.shape[:2]), f"orientation {o}")  # really reduced
            self.assertEqual(size, full.shape[:2], f"orientation {o}")
            self.assertEqual(imageio.load_sized(path)[1], full.shape[:2])
        png = self._save("sized.png", img, 6)
        self.assertEqual(imageio.load_sized(png, max_side=250)[1], (1203, 805))
        saved = imageio._full_size
        imageio._full_size = lambda *_: None  # can't tell how far it was reduced: decoded at full size
        try:
            got, size = imageio.load_sized(self.tmp / "sized6.jpg", max_side=250)
        finally:
            imageio._full_size = saved
        self.assertEqual((got.shape[:2], size), ((1203, 805), (1203, 805)))


    def test_tiff(self):
        img = _picture()
        np.testing.assert_array_equal(imageio.load(self._save("o6.tif", img, 6)), imageio.orient(img, 6))

    def test_straight_alpha_tiff_keeps_its_colors(self):
        """Pillow (like Photoshop and GIMP) stores straight alpha; OpenCV hands it back multiplied by alpha."""
        img = _picture(channels=4)
        for o in (1, 6):
            got = imageio.load(self._save(f"sa{o}.tif", img, o)).astype(int)
            want = imageio.orient(img, o).astype(int)
            np.testing.assert_array_equal(got[..., 3], want[..., 3])
            a = want[..., 3]
            err = np.abs(got[..., :3] - want[..., :3]).max(axis=2)
            # what multiplying by alpha rounded away can't come back: at most about 255 / (2 * alpha)
            self.assertTrue(np.all(err[a > 0] <= 255 / (2 * a[a > 0]) + 1), f"orientation {o}")
            self.assertTrue(np.all(err[a >= 128] <= 1), f"orientation {o}")
            self.assertLess(err.mean(), 3.0)  # multiplied by alpha, this random picture is off by 63 on average

    def test_can_copy_unchanged(self):
        img = cv2.resize(_picture(), (600, 400), interpolation=cv2.INTER_CUBIC)
        self.assertTrue(imageio.can_copy_unchanged(self._save("plain.jpg", img, 1, quality=90)))
        self.assertFalse(imageio.can_copy_unchanged(self._save("o6.jpg", img, 6, quality=90)))  # rotation to bake in
        self.assertTrue(imageio.can_copy_unchanged(self._save("plain.png", img, 1)))
        self.assertFalse(imageio.can_copy_unchanged(self._save("o6.png", img, 6)))
        self.assertFalse(imageio.can_copy_unchanged(self._save("o8.tif", img, 8)))
        cmyk = self.tmp / "cmyk.jpg"
        Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert("CMYK").save(cmyk, quality=90)
        self.assertFalse(imageio.can_copy_unchanged(cmyk))  # load() turns CMYK into RGB
        no_exif = self.tmp / "no_exif.jpg"
        cv2.imwrite(str(no_exif), img)
        self.assertTrue(imageio.can_copy_unchanged(no_exif))


def _profile(space=b"RGB ", size=3000):
    """A stand-in ICC profile: the header fields that matter here, then filler."""
    head = bytearray(128)
    head[0:4] = size.to_bytes(4, "big")
    head[16:20] = space
    head[20:24] = b"XYZ "
    head[36:40] = b"acsp"
    return bytes(head) + bytes(4) + bytes((i * 7) % 251 for i in range(size - 132))  # 0 tags, then filler


@unittest.skipIf(Image is None, "needs Pillow to read color profiles back")
class ColorProfileTests(unittest.TestCase):
    """A cleaned or cropped picture keeps its source's color profile (copied files always did)."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.dir.name)

    def tearDown(self):
        self.dir.cleanup()

    def test_profile_is_kept_in_jpeg_png_and_webp(self):
        rgb, rgba = _picture(), _picture(channels=4)
        for size in (3000, 70001):  # the second takes two JPEG segments, and an odd-sized WebP chunk
            icc = _profile(size=size)
            for name, img, kw in (("a.jpg", rgb, {"quality": 95}), ("a.png", rgb, {}), ("b.png", rgba, {}),
                                  ("a.webp", rgb, {"lossless": True}), ("b.webp", rgba, {"lossless": True})):
                src = self.tmp / f"src_{size}_{name}"
                mode = cv2.COLOR_BGRA2RGBA if img.shape[2] == 4 else cv2.COLOR_BGR2RGB
                Image.fromarray(cv2.cvtColor(img, mode)).save(src, icc_profile=icc, **kw)
                self.assertEqual(imageio.icc_profile(src), icc, name)
                dst = self.tmp / f"out_{size}_{name}"
                plain = self.tmp / f"plain_{size}_{name}"
                pic = imageio.load(src)
                imageio.save(dst, pic, imageio.icc_profile(src))
                imageio.save(plain, pic)
                with Image.open(dst) as im:
                    self.assertEqual(im.info.get("icc_profile"), icc, name)
                    im.load()
                self.assertEqual(imageio.icc_profile(dst), icc, name)
                self.assertIsNone(imageio.icc_profile(plain))
                np.testing.assert_array_equal(imageio.load(dst), imageio.load(plain), err_msg=name)

    def test_a_jpeg_profile_with_a_piece_missing_is_left_out(self):
        img = _picture()
        src = self.tmp / "two_pieces.jpg"
        imageio.save(src, img, _profile(size=70001))  # stored in two APP2 segments
        data = bytearray(src.read_bytes())
        first = data.index(b"ICC_PROFILE\0") - 4  # the first segment's marker
        length = int.from_bytes(data[first + 2:first + 4], "big")
        del data[first:first + 2 + length]  # drop piece 1 of 2
        cut = self.tmp / "one_piece.jpg"
        cut.write_bytes(bytes(data))
        self.assertIsNone(imageio.icc_profile(cut))
        out = self.tmp / "out.jpg"
        imageio.save(out, imageio.load(cut), imageio.icc_profile(cut))
        self.assertIsNone(imageio.icc_profile(out))
        self.assertFalse(imageio._icc_fits(_profile()[:-10], img))  # shorter than its header says

    def test_mismatched_profiles_are_left_out(self):
        img = _picture()
        dst = self.tmp / "cmyk_profile.jpg"
        imageio.save(dst, img, _profile(b"CMYK"))  # a CMYK file's profile doesn't describe its RGB pixels
        self.assertIsNone(imageio.icc_profile(dst))
        gray = self.tmp / "gray.png"
        imageio.save(gray, img[:, :, 0], _profile(b"RGB "))
        self.assertIsNone(imageio.icc_profile(gray))
        imageio.save(gray, img[:, :, 0], _profile(b"GRAY"))
        self.assertEqual(imageio.icc_profile(gray), _profile(b"GRAY"))
        tif = self.tmp / "a.tif"  # OpenCV can't write one into a TIFF: saved without it, not broken
        imageio.save(tif, img, _profile())
        np.testing.assert_array_equal(imageio.load(tif), img)

    def test_save_all_keeps_the_profile(self):
        from rulz.ui.workers import SaveAll

        img = cv2.resize(_picture(), (300, 200), interpolation=cv2.INTER_CUBIC)
        src = self.tmp / "p3.jpg"
        Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).save(src, quality=90, icc_profile=_profile())
        out = self.tmp / "out.jpg"
        saver = SaveAll([(src, out, {"color": 40, "jpeg": 0, "detail": 0}, None)], "balanced")
        saver.run()
        self.assertEqual(imageio.icc_profile(out), _profile())


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

    def test_empty_file_gives_a_plain_message(self):
        for name in ("empty.jpg", "empty.png"):
            path = self.tmp / name
            path.write_bytes(b"")
            with self.assertRaises(ValueError) as cm:
                imageio.load(path)
            self.assertEqual(str(cm.exception), f"Can't read {name}: the file is empty")

    def test_8_bit_rgba_tiff_from_opencv_round_trips(self):
        """OpenCV writes no alpha type: that must not be taken for straight alpha."""
        bgra = np.random.default_rng(4).integers(0, 256, (30, 50, 4), dtype=np.uint8)
        np.testing.assert_array_equal(self._roundtrip("a8.tif", bgra), bgra)

    def test_to_qimage_bgr(self):
        img = np.zeros((4, 6, 3), np.uint8)
        img[0, 0] = (255, 0, 0)  # blue in BGR
        q = imageio.to_qimage(img)
        c = q.pixelColor(0, 0)
        self.assertEqual((c.red(), c.green(), c.blue()), (0, 0, 255))
        self.assertEqual((q.width(), q.height()), (6, 4))

    def test_to_qimage_gray_alpha_and_16_bit(self):
        from PySide6.QtGui import QImage

        gray = np.full((3, 5), 200, np.uint8)
        c = imageio.to_qimage(gray).pixelColor(4, 2)
        self.assertEqual((c.red(), c.green(), c.blue(), c.alpha()), (200, 200, 200, 255))
        bgra = np.zeros((3, 5, 4), np.uint8)
        bgra[1, 2] = (0, 0, 240, 128)  # half see-through red
        q = imageio.to_qimage(bgra)
        self.assertEqual(q.format(), QImage.Format_ARGB32_Premultiplied)
        c = q.pixelColor(2, 1)
        self.assertEqual(c.alpha(), 128)
        self.assertLessEqual(abs(c.red() - 240), 2)
        self.assertEqual((c.green(), c.blue()), (0, 0))
        deep = np.array([[0, 128, 32768, 65535]], np.uint16)
        np.testing.assert_array_equal(imageio.to_8bit(deep), [[0, 0, 128, 255]])  # rounded

    def test_jpeg_keeps_full_color_resolution(self):
        """Saved 4:4:4: OpenCV's default 4:2:0 halves the color resolution the clean-up just restored."""
        img = np.zeros((64, 64, 3), np.uint8)
        img[:, ::2] = (0, 0, 255)  # one-pixel red and black stripes
        path = self.tmp / "stripes.jpg"
        imageio.save(path, img)
        data = np.fromfile(str(path), np.uint8)
        sof = next(start for marker, start, _ in imageio._jpeg_segments(data) if marker == 0xC0)
        self.assertEqual([int(data[sof + 6 + 3 * i + 1]) for i in range(3)], [0x11] * 3)  # 1x1 for every channel
        back = cv2.imdecode(data, cv2.IMREAD_COLOR).astype(int)
        self.assertLess(np.abs(back - img).mean(), 6)


if __name__ == "__main__":
    unittest.main()
