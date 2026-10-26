"""The ComfyUI likeness nodes (comfyui/rulztrainer_likeness): references from files, folders and connected images;
per-image scores, averages as the trainer averages a probe, the normalized scale and its sources, the spreadsheet
export, and bodysim's mediapipe 1.0.x mask workaround. Stand-in scorers: no models are loaded."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "comfyui"))

import rulztrainer_likeness as pkg  # noqa: E402
from rulztrainer import bodysim  # noqa: E402
from rulztrainer.facesim import FaceScore  # noqa: E402
from rulztrainer.likeness import (CEILING_DEFAULT, FACE_CURVE_DEFAULT, FACE_SPAN_DEFAULT, FLOOR_DEFAULT,  # noqa: E402
                                  MIN_SPAN, LikenessController, combine)
from rulztrainer_likeness import core, nodes  # noqa: E402

FACE = {"a.png": 60.0, "b.png": 40.0, "c.png": None, "ref1.png": 70.0, "ref2.png": 66.0, "base.png": 20.0}
BODY = {"a.png": 85.0, "b.png": None, "c.png": 83.0, "base.png": 81.0}


class FakeFace:
    """FaceScorer stand-in: the score comes from the file name."""

    def __init__(self):
        self.refs, self.ref_self_similarity = [], None

    def set_references(self, paths):
        from rulztrainer.facesim import expand_refs

        paths = expand_refs(paths)
        self.refs = [(str(p), None) for p in paths]
        self.ref_self_similarity = 0.68 if len(paths) >= 2 else None
        return [{"image": str(p), "faces": 1} for p in paths]

    def score(self, image, step=0, prompt_index=0):
        v = FACE.get(Path(image).name)
        return FaceScore(step, str(image), prompt_index, 0 if v is None else 1, None if v is None else v / 100,
                         v, self.refs[0][0] if v is not None else None)


class FakeBody:
    def __init__(self):
        self.refs, self.ref_count, self.ref_self_similarity = [], 0, None

    @property
    def ready(self):
        return bool(self.refs)

    def set_references(self, paths):
        from rulztrainer.facesim import expand_refs

        paths = expand_refs(paths)
        self.refs = [(str(p), None) for p in paths]
        self.ref_count = len(paths)
        self.ref_self_similarity = 0.9 if len(paths) >= 2 else None
        return [{"image": str(p), "person": True, "used": True, "cover": 1.0} for p in paths]

    def score(self, path):
        v = BODY.get(Path(path).name)
        return (v, self.ref_count) if v is not None else (None, 0)


@pytest.fixture
def fakes(monkeypatch, tmp_path):
    monkeypatch.setattr(core, "face_model", lambda *a, **k: FakeFace())
    monkeypatch.setattr(core, "body_model", lambda *a, **k: FakeBody())
    nodes.REPO_DIR = ROOT
    for name in list(FACE) + ["d.png"]:
        Image.new("RGB", (8, 8), (100, 120, 140)).save(tmp_path / name)
    (tmp_path / "refs").mkdir()
    for name in ("ref1.png", "ref2.png"):
        (tmp_path / name).replace(tmp_path / "refs" / name)
    return tmp_path


def test_package_finds_the_checkout():
    assert pkg.REPO == ROOT.resolve() and set(pkg.NODE_CLASS_MAPPINGS) == {
        "RZLikenessReferences", "RZLikenessScore", "RZLikenessExport"}


def test_scores_and_averages_like_the_trainer(fakes):
    refs, info = nodes.RZLikenessReferences().build(str(fakes / "refs"), "", "", "appearance", "mediapipe+sam3",
                                                    "cpu", "cpu", 640, "facebook/sam3")
    assert refs.face_count == 2 and refs.body_count == 2 and "2 of 2 have a face" in info
    images = "\n".join(str(fakes / n) for n in ("a.png", "b.png", "c.png")) + "\n# a comment\n"
    report, summary, details, overall, face, body = nodes.RZLikenessScore().score(
        refs, images, 0.7, 0.3, True, "", "", -1, -1, -1, -1)
    # the trainer's controller with the same ceilings and default floors gives the same numbers
    ctl = LikenessController(SimpleNamespace(likeness_normalize=True, resolution_step_up=False),
                             face_ceiling=68.0, body_ceiling=90.0)
    face_mean, body_mean = (60.0 + 40.0) / 2, (85.0 + 83.0) / 2  # raw scores of the images that have one
    assert face == pytest.approx(ctl.norm("face", face_mean)) and body == pytest.approx(ctl.norm("body", body_mean))
    assert overall == pytest.approx(combine(face, body, 0.7, 0.3))
    rows = report["rows"]
    assert [r["image"] for r in rows] == ["a.png", "b.png", "c.png"]
    assert rows[1]["body"] is None and rows[1]["overall"] == pytest.approx(rows[1]["face"])  # face only
    assert rows[2]["face"] is None and rows[2]["faces"] == 0 and "no face" in details
    assert "Overall" in summary and "2 of 3 with a face" in summary and "2 of 3 with a usable body" in summary
    assert report["scale"].sources["face"] == {"floor": "default", "ceiling": "references"}


def test_connected_images_and_separate_face_and_body_references(fakes):
    batch = torch.from_numpy(np.full((2, 8, 8, 3), 0.5, dtype=np.float32))
    refs, _ = nodes.RZLikenessReferences().build("", str(fakes / "refs" / "ref1.png"), "", "appearance",
                                                 "mediapipe+sam3", "cpu", "cpu", 640, "x", body_images=batch)
    assert refs.face_count == 1 and refs.body_count == 2  # face from its path, body from the connected batch
    assert [refs.label(i["image"]) for i in refs.body_info] == ["body reference image 1", "body reference image 2"]
    report = nodes.RZLikenessScore().score(refs, "", 0.7, 0.3, True, "", "", -1, -1, -1, -1,
                                           images=batch[:1])[0]
    assert report["rows"][0]["image"] == "image" and report["rows"][0]["path"] == "(connected image)"
    with pytest.raises(ValueError, match="Nothing to score"):
        nodes.RZLikenessScore().score(refs, "", 0.7, 0.3, True, "", "", -1, -1, -1, -1)
    with pytest.raises(ValueError, match="No references"):
        nodes.RZLikenessReferences().build("", "", "", "appearance", "mediapipe+sam3", "cpu", "cpu", 640, "x")


def test_the_scale_sources_in_order(fakes, tmp_path):
    refs, _ = nodes.RZLikenessReferences().build(str(fakes / "refs" / "ref1.png"), "", "", "appearance",
                                                 "mediapipe+sam3", "cpu", "cpu", 640, "x")
    s = core.build_scale(refs)  # one reference: no measured ceiling
    span = FACE_SPAN_DEFAULT  # the face scale is stretched, as in the trainer
    assert s.face == (FLOOR_DEFAULT["face"], FLOOR_DEFAULT["face"] + (CEILING_DEFAULT["face"] - FLOOR_DEFAULT["face"]) * span)
    assert s.sources["face"]["ceiling"] == "default" and s.curve == {"face": FACE_CURVE_DEFAULT, "body": 1.0}
    s = core.build_scale(refs, baseline_rows=[{"face_raw": 20.0, "body_raw": 81.0}, {"face_raw": 30.0,
                                                                                     "body_raw": None}])
    assert s.face[0] == 25.0 and s.body[0] == 81.0 and s.sources["face"]["floor"] == "baseline images (2)"
    assert s.body[1] == 90.0  # the default ceiling, at least MIN_SPAN above the floor
    s = core.build_scale(refs, baseline_rows=[{"face_raw": 70.0}])
    assert s.face == (70.0, 70.0 + MIN_SPAN["face"] * span)  # a floor above the ceiling lifts the ceiling, as the trainer
    job = tmp_path / "job"
    job.mkdir()
    (job / "likeness.jsonl").write_text("\n".join(json.dumps(r) for r in (
        {"step": 0, "scale": {"face": [13.8, 69.2], "body": [73.3, 81.3]}},
        {"step": 60, "scale": {"face": [13.8, 69.2], "body": [73.3, 81.3]}})))
    s = core.build_scale(refs, baseline_rows=[{"face_raw": 30.0}], job=core.job_scale(job), body_floor=75.0)
    assert s.face == (13.8, 69.2) and s.sources["face"]["floor"] == "trainer run"  # the run beats the baseline
    assert s.curve == {"face": 1.0, "body": 1.0}  # a run from before the curve was linear
    new_run = {"face": [13.8, 94.7], "body": [73.3, 81.3], "curve": {"face": 1.5, "body": 1.0}}
    s2 = core.build_scale(refs, job=new_run)
    assert s2.face == (13.8, 94.7) and s2.curve["face"] == 1.5  # the run's own numbers and curve, as logged
    assert abs(s2.norm("face", 67.7) - 100 * ((67.7 - 13.8) / (94.7 - 13.8)) ** 1.5) < 1e-9
    assert s.body == (75.0, 83.0) and s.sources["body"]["floor"] == "typed in"  # typed in beats the run
    assert core.build_scale(refs, normalize=False).norm("face", 37.5) == 37.5
    with pytest.raises(FileNotFoundError):
        core.job_scale(tmp_path)


def test_spreadsheet_export(fakes, tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    refs, _ = nodes.RZLikenessReferences().build(str(fakes / "refs"), "", "", "appearance", "mediapipe+sam3",
                                                 "cpu", "cpu", 640, "x")
    report = nodes.RZLikenessScore().score(refs, str(fakes / "a.png") + "\n" + str(fakes / "b.png"), 0.7, 0.3,
                                           True, "", "", -1, -1, -1, -1)[0]
    files = core.export(report["rows"], report["summary"], report["scale"], refs, tmp_path / "out" / "lk", "xlsx")
    wb = openpyxl.load_workbook(files[0])
    assert wb.sheetnames == ["Summary", "Images", "References"]
    head = [c.value for c in wb["Images"][1]]
    assert head[:5] == ["#", "Image", "Overall", "Face", "Body"] and wb["Images"].max_row == 3
    assert wb["Summary"]["C2"].value == pytest.approx(report["summary"]["face"], abs=1e-3)
    csvs = core.export(report["rows"], report["summary"], report["scale"], refs, tmp_path / "out" / "lk", "csv")
    assert [f.name for f in csvs] == ["lk_images.csv", "lk_summary.csv", "lk_references.csv"]
    assert csvs[0].read_text(encoding="utf-8").splitlines()[1].startswith("1,a.png,")


def test_export_node_toggle(fakes, tmp_path, monkeypatch):
    out = SimpleNamespace(get_output_directory=lambda: str(tmp_path),
                          get_save_image_path=lambda prefix, d: (str(Path(d) / "likeness"), "run", 7, "", prefix))
    monkeypatch.setitem(sys.modules, "folder_paths", out)
    refs, _ = nodes.RZLikenessReferences().build(str(fakes / "refs"), "", "", "appearance", "mediapipe+sam3",
                                                 "cpu", "cpu", 640, "x")
    report = nodes.RZLikenessScore().score(refs, str(fakes / "a.png"), 0.7, 0.3, True, "", "", -1, -1, -1, -1)[0]
    off = nodes.RZLikenessExport().export(report, False, "csv", "likeness/run")
    assert off["result"] == ("",) and not (tmp_path / "likeness").exists()
    on = nodes.RZLikenessExport().export(report, True, "csv", "likeness/run")
    assert (tmp_path / "likeness" / "run_00007_images.csv").is_file() and "run_00007_images.csv" in on["result"][0]


def test_mediapipe_1_0_mask_workaround(monkeypatch):
    """mediapipe 1.0.x aborts reading a float mask with padded rows: the image is padded to a width that is a multiple
    of 4, the landmarks scaled by the padded width and the mask cut back to the image."""
    import mediapipe

    monkeypatch.setattr(mediapipe, "__version__", "1.0.1")
    assert bodysim.mask_width_multiple() == 4
    monkeypatch.setattr(mediapipe, "__version__", "1.1.1")
    assert bodysim.mask_width_multiple() == 1
    seen = {}

    class Landmarker:
        def detect(self, image):
            arr = image.numpy_view()
            seen["shape"] = arr.shape
            h, w = arr.shape[:2]
            lm = [SimpleNamespace(x=(w - 1) / w, y=0.5, visibility=0.9)] * 33
            mask = SimpleNamespace(numpy_view=lambda: np.ones((h, w, 1), dtype=np.float32))
            return SimpleNamespace(pose_landmarks=[lm], segmentation_masks=[mask])

    scorer = bodysim.BodyScorer.__new__(bodysim.BodyScorer)
    scorer._mp, scorer.landmarker, scorer.pad_to = mediapipe, Landmarker(), 4
    pts, vis, mask = scorer._detect(Image.new("RGB", (1233, 40)))
    assert seen["shape"][:2] == (40, 1236) and mask.shape == (40, 1233)
    assert pts[0][0] == pytest.approx(1235.0)  # the last padded column, in the padded image's pixels
    scorer.pad_to = 1
    pts, vis, mask = scorer._detect(Image.new("RGB", (1233, 40)))
    assert seen["shape"][:2] == (40, 1233) and mask.shape == (40, 1233)
