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
from rulztrainer.likeness import (BODY_CURVE_DEFAULT, BODY_SPAN_DEFAULT, CEILING_DEFAULT,  # noqa: E402
                                  FACE_CURVE_DEFAULT, FACE_SPAN_DEFAULT, FLOOR_DEFAULT,
                                  MIN_SPAN, LikenessController, combine)
from rulztrainer_likeness import core, nodes  # noqa: E402

FACE = {"a.png": 60.0, "b.png": 40.0, "c.png": None, "ref1.png": 70.0, "ref2.png": 66.0, "base.png": 20.0}
BODY = {"a.png": 85.0, "b.png": None, "c.png": 83.0, "base.png": 81.0}
GOLD = {"a.png": 50.0, "b.png": 30.0, "base.png": 10.0}  # the held-out gold face (c.png has no face)


class FakeFace:
    """FaceScorer stand-in: the score comes from the file name (the gold face's too, after set_gold)."""

    def __init__(self):
        self.refs, self.ref_self_similarity = [], None
        self.gold = self.gold_refs = self.gold_self_similarity = None

    def set_gold(self, embedder):
        self.gold = embedder
        self.gold_refs = np.zeros((len(self.refs), 4), np.float32)
        self.gold_self_similarity = 0.6 if len(self.refs) >= 2 else None

    def set_references(self, paths):
        from rulztrainer.facesim import expand_refs

        paths = expand_refs(paths)
        self.refs = [(str(p), None) for p in paths]
        self.ref_self_similarity = 0.68 if len(paths) >= 2 else None
        return [{"image": str(p), "faces": 1} for p in paths]

    def score(self, image, step=0, prompt_index=0, gold=False, qc=False):
        v = FACE.get(Path(image).name)
        s = FaceScore(step, str(image), prompt_index, 0 if v is None else 1, None if v is None else v / 100,
                      v, self.refs[0][0] if v is not None else None)
        if gold and self.gold is not None and v is not None:
            s.gold_percent = GOLD.get(Path(image).name)
            s.gold_cosine = None if s.gold_percent is None else s.gold_percent / 100
        return s


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
    monkeypatch.setattr(core, "gold_model", lambda *a, **k: object())
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
    report, summary, details, overall, face, body, gold = nodes.RZLikenessScore().score(
        refs, images, 0.7, 0.3, True, "", "", -1, -1, -1, -1)
    # the trainer's controller with the same ceilings and default floors gives the same numbers
    ctl = LikenessController(SimpleNamespace(likeness_normalize=True, resolution_step_up=False),
                             face_ceiling=68.0, body_ceiling=90.0, gold_ceiling=60.0)
    face_mean, body_mean = (60.0 + 40.0) / 2, (85.0 + 83.0) / 2  # raw scores of the images that have one
    assert face == pytest.approx(ctl.norm("face", face_mean)) and body == pytest.approx(ctl.norm("body", body_mean))
    assert overall == pytest.approx(combine(face, body, 0.7, 0.3))  # ArcFace's face: no run, overall_face auto
    assert gold == pytest.approx(ctl.norm("gold", (50.0 + 30.0) / 2))  # the gold scale is shaped like the face's
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
    assert refs.has_gold and refs.gold_ceiling is None  # one reference: no measured gold ceiling either
    s = core.build_scale(refs)  # one reference: no measured ceiling
    span = FACE_SPAN_DEFAULT  # the face scale is stretched, as in the trainer
    assert s.face == (FLOOR_DEFAULT["face"], FLOOR_DEFAULT["face"] + (CEILING_DEFAULT["face"] - FLOOR_DEFAULT["face"]) * span)
    assert s.sources["face"]["ceiling"] == "default"
    assert s.curve == {"face": FACE_CURVE_DEFAULT, "body": BODY_CURVE_DEFAULT, "gold": FACE_CURVE_DEFAULT}
    gold_lo = FLOOR_DEFAULT["gold"]  # the gold face's own default scale, stretched like the face's
    assert s.gold == (gold_lo, gold_lo + (CEILING_DEFAULT["gold"] - gold_lo) * span)
    s = core.build_scale(refs, baseline_rows=[{"face_raw": 20.0, "body_raw": 81.0}, {"face_raw": 30.0,
                                                                                     "body_raw": None}])
    assert s.face[0] == 25.0 and s.body[0] == 81.0 and s.sources["face"]["floor"] == "baseline images (2)"
    # the default ceiling (90), lifted to halfway from the floor to 100 (90.5), then stretched like the face
    assert abs(s.body[1] - (81.0 + 9.5 * BODY_SPAN_DEFAULT)) < 1e-9
    s = core.build_scale(refs, baseline_rows=[{"face_raw": 70.0}])
    assert s.face == (70.0, 70.0 + MIN_SPAN["face"] * span)  # a floor above the ceiling lifts the ceiling, as the trainer
    job = tmp_path / "job"
    job.mkdir()
    (job / "likeness.jsonl").write_text("\n".join(json.dumps(r) for r in (
        {"step": 0, "scale": {"face": [13.8, 69.2], "body": [73.3, 81.3]}},
        {"step": 60, "scale": {"face": [13.8, 69.2], "body": [73.3, 81.3]}})))
    s = core.build_scale(refs, baseline_rows=[{"face_raw": 30.0}], job=core.job_scale(job), body_floor=75.0)
    assert s.face == (13.8, 69.2) and s.sources["face"]["floor"] == "trainer run"  # the run beats the baseline
    assert s.curve["face"] == 1.0 and s.curve["body"] == 1.0  # a run from before the curve was linear
    assert s.sources["gold"]["floor"] == "default"  # a run without the gold face: its scale from the references
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


def test_the_gold_face_in_the_overall_matches_the_trainer(fakes, tmp_path):
    """overall_face gold: the held-out gold face takes ArcFace's place as the trainer's face_gold_select does; the
    overall equals the controller's own probe record for the same raw scores, and the ArcFace overall is kept."""
    from rulztrainer.config import TrainConfig

    refs, info = nodes.RZLikenessReferences().build(str(fakes / "refs"), "", "", "appearance", "mediapipe+sam3",
                                                    "cpu", "cpu", 640, "x")
    assert refs.has_gold and refs.gold_ceiling == pytest.approx(60.0) and "Gold face (AdaFace" in info
    images = "\n".join(str(fakes / n) for n in ("a.png", "b.png", "c.png"))
    report, summary, details, overall, face, body, gold = nodes.RZLikenessScore().score(
        refs, images, 0.7, 0.3, True, "", "", -1, -1, -1, -1, "gold")
    ctl = LikenessController(TrainConfig(face_weight=0.7, body_weight=0.3), face_ceiling=68.0, body_ceiling=90.0,
                             gold_ceiling=60.0)
    rec, _ = ctl.update(10, (60.0 + 40.0) / 2, (85.0 + 83.0) / 2,
                        seeds={"face": [60.0, 40.0, None], "body": [85.0, None, 83.0], "gold": [50.0, 30.0, None]})
    assert rec.gold_select and overall == pytest.approx(rec.combined) and gold == pytest.approx(rec.gold_norm)
    assert report["summary"]["overall_arcface"] == pytest.approx(combine(face, body, 0.7, 0.3))
    assert [r["overall"] for r in report["rows"]] == pytest.approx(
        [x for x in rec.combined_seeds])  # each image as the record's per-seed combined score
    assert "in charge of the overall score" in summary and "Overall with ArcFace's face" in summary
    assert "gold face" in details.splitlines()[0]
    # a trainer run whose probes had the gold face in charge: auto follows it, with the run's gold scale
    job = tmp_path / "job"
    job.mkdir()
    # a run from before 3.14.0 (no gold_name: FaceNet): its gold scale is not AdaFace's, so it is not taken
    rec_old = {"step": 0, "gold_select": True, "scale": {
        "face": [13.8, 94.7], "body": [73.3, 90.0], "gold": [8.0, 80.0],
        "curve": {"face": 1.5, "body": 1.5, "gold": 1.5}}}
    (job / "likeness.jsonl").write_text(json.dumps(rec_old))
    report_old = nodes.RZLikenessScore().score(refs, images, 0.7, 0.3, True, "", str(job), -1, -1, -1, -1, "auto")[0]
    assert report_old["scale"].gold != (8.0, 80.0) and report_old["scale"].sources["gold"]["floor"] != "trainer run"
    assert report_old["scale"].sources["face"]["floor"] == "trainer run"
    (job / "likeness.jsonl").write_text(json.dumps({**rec_old, "gold_name": "adaface"}))
    report2, _, _, overall2, _, _, gold2 = nodes.RZLikenessScore().score(
        refs, images, 0.7, 0.3, True, "", str(job), -1, -1, -1, -1, "auto")
    assert report2["scale"].gold == (8.0, 80.0) and report2["scale"].sources["gold"]["floor"] == "trainer run"
    assert gold2 == pytest.approx(100 * ((40.0 - 8.0) / 72.0) ** 1.5)
    assert overall2 == pytest.approx(combine(gold2, report2["summary"]["body"], 0.7, 0.3))
    # typed in beats the run, as for the face and the body
    report3 = nodes.RZLikenessScore().score(refs, images, 0.7, 0.3, True, "", str(job), -1, -1, -1, -1, "arcface",
                                            10.0, 70.0)[0]
    assert report3["scale"].gold == (10.0, 70.0) and not report3["summary"]["gold_select"]
    assert report3["summary"]["overall"] == pytest.approx(report3["summary"]["overall_arcface"])
    # the baseline images set the gold floor like the step-0 probe
    s = core.build_scale(refs, baseline_rows=[{"face_raw": 20.0, "gold_raw": 12.0}])
    assert s.gold[0] == 12.0 and s.sources["gold"]["floor"] == "baseline images (1)"


def test_gold_face_off_or_unavailable(fakes, monkeypatch):
    refs, info = nodes.RZLikenessReferences().build(str(fakes / "refs"), "", "", "appearance", "mediapipe+sam3",
                                                    "cpu", "cpu", 640, "x", "off")
    assert not refs.has_gold and "Gold face" not in info
    report, summary, _, overall, face, body, gold = nodes.RZLikenessScore().score(
        refs, str(fakes / "a.png"), 0.7, 0.3, True, "", "", -1, -1, -1, -1, "gold")
    assert gold == nodes.NA and report["scale"].gold is None and "Gold face" not in summary
    assert overall == pytest.approx(combine(face, body, 0.7, 0.3))  # no gold face: ArcFace's face is used

    def broken(*a, **k):
        raise RuntimeError("no network")

    monkeypatch.setattr(core, "gold_model", broken)
    refs, info = nodes.RZLikenessReferences().build(str(fakes / "refs"), "", "", "appearance", "mediapipe+sam3",
                                                    "cpu", "cpu", 640, "x")
    assert not refs.has_gold and "could not load (no network)" in info


def test_spreadsheet_has_the_gold_face(fakes, tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    refs, _ = nodes.RZLikenessReferences().build(str(fakes / "refs"), "", "", "appearance", "mediapipe+sam3",
                                                 "cpu", "cpu", 640, "x")
    report = nodes.RZLikenessScore().score(refs, str(fakes / "a.png"), 0.7, 0.3, True, "", "", -1, -1, -1, -1,
                                           "gold")[0]
    wb = openpyxl.load_workbook(core.export(report["rows"], report["summary"], report["scale"], refs,
                                            tmp_path / "lk", "xlsx")[0])
    head = [c.value for c in wb["Images"][1]]
    assert head[5] == "Gold face" and wb["Images"].cell(2, 6).value == pytest.approx(
        report["rows"][0]["gold"], abs=1e-3)
    assert wb["Summary"]["E1"].value == "Gold face (AdaFace)"
    assert wb["Summary"]["E2"].value == pytest.approx(report["summary"]["gold"], abs=1e-3)
    labels = [r[0].value for r in wb["Summary"].iter_rows()]
    assert "Gold face scale (raw floor, ceiling)" in labels and "Face in the overall score" in labels


def test_a_run_of_another_face_model_or_body_score_keeps_the_references_scale(fakes, tmp_path):
    """A run's face scale is one InsightFace pack's raw numbers and its body scale one body score's: the nodes apply
    them only to the same kind (the trainer records face_model / body_score since 3.19.0; older records count as
    buffalo_l with the body score unknown)."""
    build = nodes.RZLikenessReferences().build
    refs, _ = build(str(fakes / "refs"), "", "", "appearance", "mediapipe+sam3", "cpu", "cpu", 640, "x")
    assert refs.face_model == "buffalo_l"
    job = tmp_path / "job"
    job.mkdir()
    (job / "likeness.jsonl").write_text(json.dumps({"step": 0, "scale": {"face": [13.8, 69.2], "body": [73.3, 81.3]},
                                                    "face_model": "antelopev2", "body_score": "smplx_ratio"}))
    run = core.job_scale(job)
    assert run["face_model"] == "antelopev2" and run["body_score"] == "smplx_ratio"
    s = core.build_scale(refs, job=run)
    assert s.sources["face"]["floor"] == "default" and s.sources["body"]["floor"] == "default"
    r100, _ = build(str(fakes / "refs"), "", "", "appearance", "mediapipe+sam3", "cpu", "cpu", 640, "x",
                    face_model="antelopev2")
    assert r100.face_model == "antelopev2"
    s = core.build_scale(r100, job={**run, "body_score": "appearance"})
    assert s.face == (13.8, 69.2) and s.body == (73.3, 81.3)  # the same face model and body score: the run's scale
    (job / "likeness.jsonl").write_text(json.dumps({"step": 0, "scale": {"face": [13.8, 69.2], "body": [73.3, 81.3]}}))
    old = core.job_scale(job)  # a run from before 3.19.0
    assert old["face_model"] == "buffalo_l" and old["body_score"] == ""
    assert core.build_scale(refs, job=old).sources["body"]["floor"] == "trainer run"


def test_an_ambiguous_or_tiny_face_stays_unscored_as_in_the_probes(fakes, monkeypatch):
    """The nodes score with the probes' subject check (FaceScorer.score qc): the reason shows in the details."""
    refs, _ = nodes.RZLikenessReferences().build(str(fakes / "refs"), "", "", "appearance", "mediapipe+sam3", "cpu",
                                                 "cpu", 640, "x")
    asked, real = [], FakeFace.score

    def score(self, image, step=0, prompt_index=0, gold=False, qc=False):
        asked.append(qc)
        if qc and Path(image).name == "b.png":
            return FaceScore(step, str(image), prompt_index, 2, None, None, None, skipped="ambiguous: 2 faces")
        return real(self, image, step, prompt_index, gold, qc)

    monkeypatch.setattr(FakeFace, "score", score)
    rows = core.score_rows(refs, [("a.png", fakes / "a.png"), ("b.png", fakes / "b.png")])
    assert asked and all(asked)
    assert rows[0]["face_raw"] == 60.0 and rows[1]["face_raw"] is None and rows[1]["face_skipped"] == "ambiguous: 2 faces"
    text = core.details_text(core.apply_scale(rows, core.build_scale(refs)))
    assert "face unscored (ambiguous: 2 faces)" in text
