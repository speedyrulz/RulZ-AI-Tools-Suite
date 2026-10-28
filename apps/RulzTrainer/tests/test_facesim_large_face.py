"""A face that fills the frame is missed by the detector (it rescales every input to det_size, and its anchors stop
at a face size relative to that frame); faces() then detects on a canvas 2x and 4x the image's size and shifts the
boxes back to source pixels."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import facesim  # noqa: E402


class _App:
    """Sees the face (80% of the original image, centred) only when it covers at most 45% of the frame."""

    def __init__(self, orig: int):
        self.orig, self.calls = orig, []

    def get(self, arr):
        h, w = arr.shape[:2]
        self.calls.append(w)
        face = 0.8 * self.orig
        if face / w > 0.45:
            return []
        ox, oy = (w - self.orig) / 2, (h - self.orig) / 2  # the image sits at the canvas centre
        return [SimpleNamespace(bbox=np.array([ox + 0.1 * self.orig, oy + 0.1 * self.orig, ox + 0.9 * self.orig,
                                               oy + 0.9 * self.orig], dtype=np.float32),
                                kps=np.array([[ox + 0.3 * self.orig, oy + 0.4 * self.orig]], dtype=np.float32),
                                det_score=0.9)]


def _scorer(tmp_path, size, seen):
    from PIL import Image

    s = facesim.FaceScorer.__new__(facesim.FaceScorer)
    s.app = _App(seen)
    s.refs = []
    p = tmp_path / "probe.png"
    Image.new("RGB", (size, size), (120, 90, 80)).save(p)
    return s, p


def test_large_face_found_on_the_2x_canvas_with_boxes_in_source_pixels(tmp_path):
    s, p = _scorer(tmp_path, 1024, 1024)
    faces = s.faces(p)
    assert s.app.calls == [1024, 2048] and len(faces) == 1
    np.testing.assert_allclose(faces[0].bbox, [102.4, 102.4, 921.6, 921.6], atol=1e-3)  # 0.1..0.9 of the source
    np.testing.assert_allclose(faces[0].kps, [[307.2, 409.6]], atol=1e-3)


def test_4x_canvas_and_the_thumbnail_scale_compose(tmp_path):
    s, p = _scorer(tmp_path, 4096, 2048)  # thumbnailed to 2048 for the detector, which needs the 4x canvas here
    s.app.get = lambda arr, _g=s.app.get: _g(arr) if arr.shape[1] >= 8192 or arr.shape[1] == 2048 else []
    faces = s.faces(p)
    assert len(faces) == 1 and s.app.calls[-1] == 8192
    np.testing.assert_allclose(faces[0].bbox, [409.6, 409.6, 3686.4, 3686.4], atol=1e-2)  # back in the 4096 source


def test_small_face_needs_no_canvas(tmp_path):
    s, p = _scorer(tmp_path, 1024, 400)  # a face that is 32% of the frame
    assert len(s.faces(p)) == 1 and s.app.calls == [1024]


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
