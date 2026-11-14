"""Crop box geometry (frame size changes, boxes drawn at an edge) and the colour match's
reference loading (ICC profiles) and lookup tables.

Run with:  python -m pytest tests   (from the app folder, inside its .venv).
"""
import numpy as np
import pytest

from test_core import APP  # noqa: F401  (sets up Qt, the settings file and sys.path)

from PySide6.QtCore import QEvent, QPointF, QRectF, Qt  # noqa: E402
from PySide6.QtGui import QColor, QColorSpace, QImage, QMouseEvent  # noqa: E402

from simplecut import colormatch  # noqa: E402
from simplecut.colormatch import (Correction, apply_tables, build_tables, load_image,  # noqa: E402
                                  qimage_to_rgb, rgb_to_qimage, to_srgb)
from simplecut.cropview import VideoView  # noqa: E402


# ------------------------------------------------------------------ crop box
def make_view(fw=1920, fh=1080, aspect=None, size=(960, 540)):
    v = VideoView()
    v.resize(*size)
    v.set_frame(preview(), fw, fh)
    v.set_crop_enabled(True)
    if aspect:
        v.set_aspect(aspect)
    return v


def preview():
    img = QImage(64, 36, QImage.Format_RGB888)
    img.fill(QColor(90, 90, 90))
    return img


def assert_inside(v, r, eps=1e-6):
    fw, fh = v.frame_size
    assert r.x() >= -eps and r.y() >= -eps, (r, v.frame_size)
    assert r.right() <= fw + eps and r.bottom() <= fh + eps, (r, v.frame_size)


@pytest.mark.parametrize("aspect", [9 / 16, 1.0, 16 / 9, 2.39])
@pytest.mark.parametrize("where", ["centre", "right", "bottom-left"])
def test_locked_aspect_box_keeps_its_aspect_when_the_frame_shrinks(aspect, where):
    """A repair's auto-crop turning 1920x1080 into 1800x1012 used to clip a 9:16 box of
    607.5x1080 to 607.5x1012 (aspect 0.600), which "Scale to 1080x1920" then stretched."""
    v = make_view(aspect=aspect)
    r = v.crop
    if where != "centre":  # a smaller box pushed against an edge, so clipping cuts one side only
        w, h = r.width() * 0.8, r.height() * 0.8
        v.crop = r = QRectF(1920 - w if where == "right" else 0, 0 if where == "right" else 1080 - h, w, h)
    assert r.width() / r.height() == pytest.approx(aspect, abs=1e-6)

    v.set_frame(preview(), 1800, 1012)
    r = v.crop
    assert r.width() / r.height() == pytest.approx(aspect, abs=1e-3)
    assert_inside(v, r)
    x, y, w, h = v.crop_rect_int()
    assert x >= 0 and y >= 0 and x + w <= 1800 and y + h <= 1012


def test_the_reported_9_16_box_is_refitted_not_clipped():
    v = make_view(aspect=9 / 16)
    assert (v.crop.width(), v.crop.height()) == pytest.approx((607.5, 1080))
    v.set_frame(preview(), 1800, 1012)
    assert (v.crop.width(), v.crop.height()) == pytest.approx((1012 * 9 / 16, 1012))
    assert_inside(v, v.crop)


def test_free_box_is_still_just_clipped_when_the_frame_shrinks():
    v = make_view()
    v.crop = QRectF(1000, 100, 900, 900)
    v.set_frame(preview(), 1800, 1012)
    assert (v.crop.x(), v.crop.y(), v.crop.width(), v.crop.height()) == (1000, 100, 800, 900)


def test_growing_frame_leaves_a_locked_box_alone():
    v = make_view(fw=1800, fh=1012, aspect=1.0)
    before = (v.crop.x(), v.crop.y(), v.crop.width(), v.crop.height())
    v.set_frame(preview(), 1920, 1080)
    assert (v.crop.x(), v.crop.y(), v.crop.width(), v.crop.height()) == pytest.approx(before)


def drag(v, p0, p1):
    """Press at frame point p0, move to p1 and release, through the widget's mouse handlers."""
    s = v._scale()
    to_w = lambda p: QPointF(v._disp.x() + p[0] * s, v._disp.y() + p[1] * s)  # noqa: E731
    a, b = to_w(p0), to_w(p1)
    v.mousePressEvent(QMouseEvent(QEvent.MouseButtonPress, a, a, Qt.LeftButton, Qt.LeftButton, Qt.NoModifier))
    v.mouseMoveEvent(QMouseEvent(QEvent.MouseMove, b, b, Qt.NoButton, Qt.LeftButton, Qt.NoModifier))
    v.mouseReleaseEvent(QMouseEvent(QEvent.MouseButtonRelease, b, b, Qt.LeftButton, Qt.NoButton, Qt.NoModifier))


def test_square_box_started_at_the_right_edge_ends_inside_the_frame():
    """Pressing at x=1915 of 1920 with 1:1 locked drew a 16 px box from 1915 to 1931, while
    crop_rect_int exported x=1904: what was drawn was not what got exported."""
    v = make_view(aspect=1.0)
    drag(v, (1915, 500), (1925, 505))
    r = v.crop
    assert_inside(v, r)
    assert r.width() == pytest.approx(r.height())
    assert (r.x(), r.y(), r.width(), r.height()) == (1904, 500, 16, 16)
    assert v.crop_rect_int() == (1904, 500, 16, 16)


@pytest.mark.parametrize("aspect, p0, p1", [
    (1.0, (1915, 500), (1925, 505)),       # right edge, dragging out of the frame
    (9 / 16, (1910, 1075), (1930, 1090)),  # bottom-right corner, tall box
    (16 / 9, (3, 2), (-10, -6)),           # top-left corner, dragging up-left
    (2.39, (5, 1076), (-4, 1090)),         # bottom-left corner, wide box
])
def test_minimum_size_box_at_an_edge_is_shifted_inside_keeping_the_aspect(aspect, p0, p1):
    v = make_view(aspect=aspect)
    v.crop = QRectF(860, 440, 200, 200 / aspect)  # small and central: every press draws a new box
    drag(v, p0, p1)
    r = v.crop
    assert min(r.width(), r.height()) == pytest.approx(16)  # the MIN_SIZE fallback box
    assert_inside(v, r)
    assert r.width() / r.height() == pytest.approx(aspect, abs=1e-6)
    x, y, w, h = v.crop_rect_int()
    # the exported rectangle is the drawn one, give or take the even-pixel rounding
    assert abs(x - r.x()) <= 1 and abs(y - r.y()) <= 1
    assert abs(w - r.width()) <= 2 and abs(h - r.height()) <= 2


# ------------------------------------------------------------------ reference photo colour space
RAW = (180, 120, 100)   # a skin tone; as Display P3 it is (190, 117, 96) in sRGB


def tagged(rgb=RAW, space=None, fmt=QImage.Format_RGB32):
    img = QImage(8, 6, fmt)
    img.fill(QColor(*rgb))
    if space is not None:
        img.setColorSpace(QColorSpace(space))
    return img


def spread(px):
    px = [int(c) for c in px]
    return max(px) - min(px)


def test_display_p3_reference_comes_back_more_saturated_in_srgb():
    rgb = qimage_to_rgb(tagged(space=QColorSpace.NamedColorSpace.DisplayP3))
    px = rgb[0, 0]
    assert (rgb == px).all()
    assert spread(px) > spread(RAW)            # more saturated than its raw numbers
    assert px[0] > RAW[0] and px[2] < RAW[2]
    assert np.abs(px.astype(int) - (190, 117, 96)).max() <= 2


def test_adobe_rgb_reference_is_converted_too():
    px = qimage_to_rgb(tagged(space=QColorSpace.NamedColorSpace.AdobeRgb))[0, 0]
    assert spread(px) > spread(RAW)


def test_load_image_converts_an_embedded_p3_profile(tmp_path):
    path = str(tmp_path / "p3.png")
    assert tagged(space=QColorSpace.NamedColorSpace.DisplayP3).save(path)
    img = load_image(path)
    assert img.colorSpace() == QColorSpace(QColorSpace.NamedColorSpace.SRgb)
    px = qimage_to_rgb(img)[0, 0]
    assert spread(px) > spread(RAW)
    assert np.abs(px.astype(int) - (190, 117, 96)).max() <= 2


def test_images_without_a_colour_space_are_left_alone(tmp_path):
    img = tagged()
    assert not img.colorSpace().isValid()
    assert to_srgb(img) is img
    assert tuple(qimage_to_rgb(img)[0, 0]) == RAW

    path = str(tmp_path / "plain.png")
    assert img.save(path)
    loaded = load_image(path)
    assert tuple(qimage_to_rgb(loaded)[0, 0]) == RAW

    srgb = tagged(space=QColorSpace.NamedColorSpace.SRgb)
    assert to_srgb(srgb) is srgb
    assert tuple(qimage_to_rgb(srgb)[0, 0]) == RAW


def test_video_frame_round_trip_is_unchanged():
    """The app hands the dialog an RGB888 frame with no colour space: its pixels must not move."""
    rng = np.random.default_rng(3)
    frame = rng.integers(0, 256, (37, 53, 3), dtype=np.uint8)
    img = rgb_to_qimage(frame)
    assert not img.colorSpace().isValid()
    assert np.array_equal(qimage_to_rgb(img), frame)


# ------------------------------------------------------------------ lookup tables
def numpy_tables(rgb, tables):
    """apply_tables as it was before cv2.LUT."""
    out = np.empty_like(rgb)
    for c in range(3):
        out[..., c] = tables[c][rgb[..., c]]
    return out


def table_sets():
    rng = np.random.default_rng(7)
    yield rng.integers(0, 256, (3, 256), dtype=np.uint8)
    yield build_tables(Correction((1.3, 0.9, 0.7), 1.2))
    yield build_tables(Correction((1.0, 1.0, 1.0), 1.0))


@pytest.mark.parametrize("use_cv2", [True, False])
def test_apply_tables_matches_numpy_indexing(use_cv2, monkeypatch):
    if use_cv2:
        assert colormatch.cv2 is not None, "OpenCV is a dependency; the fast path must be live"
    else:  # OpenCV missing (an old source-install .venv): numpy does the job
        monkeypatch.setattr(colormatch, "cv2", None)
    rng = np.random.default_rng(11)
    rgb = rng.integers(0, 256, (123, 77, 3), dtype=np.uint8)
    for tables in table_sets():
        out = apply_tables(rgb, tables)
        assert out.dtype == np.uint8 and out.shape == rgb.shape
        assert np.array_equal(out, numpy_tables(rgb, tables))


@pytest.mark.parametrize("use_cv2", [True, False])
def test_apply_tables_on_a_padded_read_only_frame_buffer(use_cv2, monkeypatch):
    """The app passes a view into the decoded plane (rows padded to line_size), and wraps the
    result in a QImage with stride width*3, so the output must be a fresh contiguous array."""
    if not use_cv2:
        monkeypatch.setattr(colormatch, "cv2", None)
    rng = np.random.default_rng(5)
    h, w, line = 41, 90, 90 * 3 + 30
    buf = rng.integers(0, 256, (h, line), dtype=np.uint8)
    buf.setflags(write=False)
    view = buf[:, :w * 3].reshape(h, w, 3)
    assert not view.flags["C_CONTIGUOUS"]
    tables = next(table_sets())
    out = apply_tables(view, tables)
    assert out.flags["C_CONTIGUOUS"]
    assert np.array_equal(out, numpy_tables(view, tables))
    assert np.array_equal(qimage_to_rgb(QImage(out.data, w, h, w * 3, QImage.Format_RGB888).copy()), out)


def test_apply_tables_on_an_empty_image():
    tables = next(table_sets())
    assert apply_tables(np.zeros((0, 4, 3), np.uint8), tables).shape == (0, 4, 3)
