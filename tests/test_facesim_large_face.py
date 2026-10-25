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
