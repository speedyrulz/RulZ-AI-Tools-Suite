import numpy as np
import pytest
from PIL import Image, JpegImagePlugin

from rulz import imaging


def test_16bit_gray_png_is_scaled_not_clipped(tmp_path):
    a = np.full((40, 60), 32768, np.uint16)
    a[:, :20] = 0
    path = tmp_path / "deep.png"
    Image.fromarray(a).save(path)
    with Image.open(path) as im:
        assert im.mode in imaging._HIGH_DEPTH
    li = imaging.load(str(path), 1600)
    px = np.asarray(li.image)
    assert li.image.mode == "RGB"
    assert px[0, 0].tolist() == [0, 0, 0]
    assert 120 <= px[0, 50, 0] <= 135  # mid-gray, not white


def test_float_tiff_in_0_1_range(tmp_path):
    a = np.full((30, 30), 0.5, np.float32)
    path = tmp_path / "float.tif"
    Image.fromarray(a, "F").save(path)
    px = np.asarray(imaging.load(str(path), 1600).image)
    assert 120 <= px[0, 0, 0] <= 135


def test_8bit_values_in_wide_mode_untouched():
    im = Image.fromarray(np.full((10, 10), 200, np.int32), "I")
    assert np.asarray(imaging._to_8bit(im))[0, 0, 0] == 200


def test_mpo_uses_jpeg_draft(tmp_path, monkeypatch):
    path = tmp_path / "phone.mpo"
    frames = [Image.new("RGB", (3200, 2400), (200, 30, 30)), Image.new("RGB", (3200, 2400), (30, 200, 30))]
    try:
        frames[0].save(path, "MPO", save_all=True, append_images=frames[1:])
    except (KeyError, ValueError, OSError):
        pytest.skip("this Pillow can't write MPO")
    drafts = []
    real = JpegImagePlugin.JpegImageFile.draft

    def draft(self, mode, size):
        drafts.append(self.format)
        return real(self, mode, size)

    monkeypatch.setattr(JpegImagePlugin.JpegImageFile, "draft", draft)
    li = imaging.load(str(path), 800)
    assert drafts[:1] == ["MPO"]
    assert max(li.image.size) == 800
    assert li.full_size == (3200, 2400)
    assert li.image.getpixel((10, 10))[0] > 150  # first frame
