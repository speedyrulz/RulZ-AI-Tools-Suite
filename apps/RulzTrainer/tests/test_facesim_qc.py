"""Face subject QC (score(qc=True)): a picture whose largest face has a rival of similar size, or is too small to
embed, is left unscored with the reason, instead of scoring what may be the wrong person."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import facesim  # noqa: E402


def _face(x0, y0, side):
    return SimpleNamespace(bbox=np.array([x0, y0, x0 + side, y0 + side], dtype=np.float32), det_score=0.9,
                           normed_embedding=np.array([1.0, 0.0], dtype=np.float32), kps=None)


def _scorer(faces):
    s = facesim.FaceScorer.__new__(facesim.FaceScorer)
    s.refs = [("ref.png", np.array([1.0, 0.0], dtype=np.float32))]
    s.gold = None
    s.faces = lambda image: faces
    return s


def test_subject_problem_names_rivals_and_tiny_faces():
    assert facesim.subject_problem([]) == ""
    assert facesim.subject_problem([_face(0, 0, 200)]) == ""
    assert facesim.subject_problem([_face(0, 0, 200), _face(300, 0, 100)]) == ""  # a quarter of the area: background
    assert facesim.subject_problem([_face(0, 0, 200), _face(300, 0, 170)]).startswith("ambiguous")  # 72%
    assert facesim.subject_problem([_face(0, 0, 40)]).startswith("face too small")


def test_qc_leaves_the_picture_unscored_only_when_asked():
    two = [_face(0, 0, 200), _face(300, 0, 180)]
    s = _scorer(two)
    plain = s.score("p.png")
    assert plain.percent == 100.0 and plain.skipped == ""  # callers without qc (reward, self-training) are unchanged
    checked = s.score("p.png", qc=True)
    assert checked.percent is None and checked.cosine is None and checked.faces == 2
    assert checked.skipped.startswith("ambiguous")
    assert _scorer([_face(0, 0, 200)]).score("p.png", qc=True).percent == 100.0


def test_preview_samples_use_the_qc(tmp_path):
    s = _scorer([_face(0, 0, 30)])
    out = facesim.score_samples(s, [str(tmp_path / "s.png")], 5, tmp_path)
    assert out[0].percent is None and out[0].skipped.startswith("face too small")


def test_a_face_model_of_its_own_leaves_the_shared_detector_alone(monkeypatch):
    """face_model antelopev2: the likeness scorer is a FaceScorer of its own; the dataset's detector (face analysis
    cache, identity alignment) keeps buffalo_l. buffalo_l shares the one scorer as before."""
    from rulztrainer import trainer as T
    from rulztrainer.config import TrainConfig

    made = []

    class Fake:
        def __init__(self, provider, det_size, model_name="buffalo_l", progress=None):
            made.append(model_name)
            self.model_name, self.refs, self.ref_self_similarity = model_name, [], None

        def set_references(self, paths):
            self.refs = [(p, None) for p in paths]

    monkeypatch.setattr(facesim, "available", lambda: (True, ""))
    monkeypatch.setattr(facesim, "FaceScorer", Fake)
    for model, shared in (("buffalo_l", True), ("antelopev2", False)):
        tr = T.Trainer.__new__(T.Trainer)
        tr.cfg = TrainConfig(face_refs=["a.png"], face_model=model)
        tr.face_detector = detector = Fake("cpu", 640)
        tr.status = SimpleNamespace(update=lambda **k: None)
        tr._setup_gold = lambda scorer: None
        scorer = tr._make_face_scorer()
        assert scorer.model_name == model and (scorer is detector) == shared
        assert tr.face_detector is detector and detector.model_name == "buffalo_l"


def test_a_pack_zipped_inside_a_folder_is_flattened(tmp_path, monkeypatch):
    """antelopev2.zip holds its files under antelopev2/; FaceAnalysis looks for them in the pack folder itself."""
    import zipfile

    from rulztrainer import downloads

    def fake_download(url, path, label, progress=None):
        path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("antelopev2/glintr100.onnx", b"r")
            zf.writestr("antelopev2/scrfd_10g_bnkps.onnx", b"d")

    monkeypatch.setattr(downloads, "download_url", fake_download)
    pack = downloads.ensure_insightface_pack("antelopev2", root=str(tmp_path))
    assert sorted(f.name for f in pack.glob("*.onnx")) == ["glintr100.onnx", "scrfd_10g_bnkps.onnx"]
