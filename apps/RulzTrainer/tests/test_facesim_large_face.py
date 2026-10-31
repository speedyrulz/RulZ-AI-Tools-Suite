"""A face that fills the frame is missed by the detector (it rescales every input to det_size, and its anchors stop
at a face size relative to that frame); faces() then detects on the image shrunk to a half and a quarter at the
centre of a canvas of its own size and maps the boxes back to source pixels."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import facesim  # noqa: E402

pytest.importorskip("cv2")


class _App:
    """A face `frac` of the picture's width, centred, seen only when it covers at most `limit` of the frame. The
    picture is whatever is not the canvas grey. Records each call's (frame width, picture width)."""

    def __init__(self, frac: float, limit: float = 0.45):
        self.frac, self.limit, self.calls = frac, limit, []

    def get(self, arr):
        w = arr.shape[1]
        pic = np.argwhere((arr != 114).any(-1))
        (y0, x0), (y1, x1) = pic.min(0), pic.max(0) + 1
        s = float(x1 - x0)
        self.calls.append((w, int(s)))
        if self.frac * s / w > self.limit:
            return []
        cx, cy, half = (x0 + x1) / 2, (y0 + y1) / 2, self.frac * s / 2
        return [SimpleNamespace(bbox=np.array([cx - half, cy - half, cx + half, cy + half], dtype=np.float32),
                                kps=np.array([[x0 + 0.3 * s, y0 + 0.4 * s]], dtype=np.float32), det_score=0.9)]


def _scorer(tmp_path, size, frac, limit=0.45):
    from PIL import Image

    s = facesim.FaceScorer.__new__(facesim.FaceScorer)
    s.app = _App(frac, limit)
    s.refs = []
    p = tmp_path / "probe.png"
    Image.new("RGB", (size, size), (120, 90, 80)).save(p)
    return s, p


def test_large_face_found_at_half_size_with_boxes_in_source_pixels(tmp_path):
    s, p = _scorer(tmp_path, 1024, 0.8)
    faces = s.faces(p)
    assert s.app.calls == [(1024, 1024), (1024, 512)] and len(faces) == 1  # the canvas stays the image's size
    np.testing.assert_allclose(faces[0].bbox, [102.4, 102.4, 921.6, 921.6], atol=1e-3)  # 0.1..0.9 of the source
    np.testing.assert_allclose(faces[0].kps, [[307.2, 409.6]], atol=1e-3)


def test_quarter_size_and_the_thumbnail_scale_compose(tmp_path):
    s, p = _scorer(tmp_path, 4096, 0.8, limit=0.25)  # thumbnailed to 2048 for the detector; seen at a quarter
    faces = s.faces(p)
    assert len(faces) == 1 and s.app.calls == [(2048, 2048), (2048, 1024), (2048, 512)]
    np.testing.assert_allclose(faces[0].bbox, [409.6, 409.6, 3686.4, 3686.4], atol=1e-2)  # back in the 4096 source


def test_the_embedding_comes_from_the_source_pixels(tmp_path):
    s, p = _scorer(tmp_path, 1024, 0.8)
    seen = []
    s.app.models = {"recognition": SimpleNamespace(get=lambda img, f: seen.append((img.shape, f.kps.copy())))}
    s.faces(p)
    assert seen[0][0] == (1024, 1024, 3)
    np.testing.assert_allclose(seen[0][1], [[307.2, 409.6]], atol=1e-3)


def test_small_face_needs_no_canvas(tmp_path):
    s, p = _scorer(tmp_path, 1024, 0.3125)  # a face that is 31% of the frame
    assert len(s.faces(p)) == 1 and s.app.calls == [(1024, 1024)]


def test_insightface_runs_on_the_configured_gpu():
    assert facesim.cuda_index("cpu", "cuda:1") is None
    assert facesim.cuda_index("cuda", "cuda:1") == 1 and facesim.cuda_index("cuda") == 0
    assert facesim.cuda_index("cuda:2", "cuda:1") == 2  # an explicit provider index wins


def test_cut_off_faces_and_the_displayed_size(tmp_path):
    from PIL import Image

    assert not facesim.cut_off([10, 10, 90, 90], (100, 100))
    assert not facesim.cut_off([-3, 10, 90, 103], (100, 100))  # within 5% of the box: a tight crop, still whole
    assert facesim.cut_off([-20, 10, 90, 90], (100, 100)) and facesim.cut_off([10, 10, 90, 130], (100, 100))
    p = tmp_path / "rotated.jpg"
    exif = Image.Exif()
    exif[0x0112] = 6  # stored landscape, shown portrait
    Image.new("RGB", (300, 200)).save(p, exif=exif)
    assert facesim.image_size(p) == (200, 300)


def test_cut_off_faces_are_left_out_of_the_ceiling(tmp_path):
    """The ceiling is the median of each reference's best match among the others over the whole faces: a face the
    frame cuts off (an unreliable embedding, here unlike every other) is still a reference but not in the median."""
    from PIL import Image

    vecs = {"a": [1.0, 0.0, 0.0], "b": [0.8, 0.6, 0.0], "c": [0.0, 1.0, 0.0], "d": [0.6, 0.8, 0.0],
            "cut": [0.0, 0.0, -1.0]}
    paths = []
    for name in vecs:
        paths.append(tmp_path / f"{name}.png")
        Image.new("RGB", (100, 100)).save(paths[-1])
    s = facesim.FaceScorer.__new__(facesim.FaceScorer)
    s.refs = []
    s.faces = lambda p: [SimpleNamespace(
        bbox=np.array([-30, 5, 95, 120] if Path(p).stem == "cut" else [10, 10, 90, 90], dtype=np.float32),
        normed_embedding=np.asarray(vecs[Path(p).stem], dtype=np.float32), det_score=0.9)]
    info = s.set_references(paths)
    assert len(s.refs) == 5 and [i["cut_off"] for i in info] == [False] * 4 + [True]
    assert abs(s.ref_self_similarity - 0.88) < 1e-6  # .8 .96 .8 .96 (with the cut-off face: .8)
