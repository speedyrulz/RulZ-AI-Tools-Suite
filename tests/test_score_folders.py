"""Score sample folders: a picture's kind and step from its names, the run list, the scoring loop around stand-in
scorers (the baseline's bodies become the anchored score's strangers, its file numbers are no steps), the printed
table, and the workbook's formulas, scale and chart data."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import bodyanchor as BA  # noqa: E402
from rulztrainer import bodysim, facesim  # noqa: E402
from rulztrainer import score_folders as SF  # noqa: E402


def test_kind_and_step_from_the_names(tmp_path):
    root = tmp_path / "run"
    assert SF.kind_of(root / "fizgig face" / "200.png", root) == "face"
    assert SF.kind_of(root / "x body" / "200.png", root) == "body"
    assert SF.kind_of(root / "step000200_body_0.png", root) == "body"
    assert SF.kind_of(root / "faces" / "step000200_body_0.png", root) == "body"  # the closest name decides
    assert SF.kind_of(root / "samples" / "200.png", root) == "both"
    assert SF.kind_of(tmp_path / "body runs" / "a" / "200.png", tmp_path / "body runs") == "both"  # the run's own
    assert SF.step_of(Path("200.png")) == 200
    assert SF.step_of(Path("lora16_step000400_face_1.png")) == 400  # after "step", not the race prefix
    assert SF.step_of(Path("1730000000000__000000600_0.jpg")) == 600  # a 13-digit timestamp skipped
    assert SF.step_of(Path("final.png")) is None
    import re

    assert SF.step_of(Path("e3-s800.png"), re.compile(r"s(\d+)")) == 800
    assert SF.parse_run("keel = F:/runs/keel") == ("keel", "F:/runs/keel")
    assert SF.parse_run(str(tmp_path)) == (tmp_path.name, str(tmp_path))  # a bare folder names itself


def _png(path: Path):
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8)).save(path)


def unit(*v):
    v = np.asarray(v, np.float64)
    return v / np.linalg.norm(v)


VECS = {"r1": unit(1, 0.1, 0), "r2": unit(1, -0.1, 0), "s1": unit(0.1, 1, 0), "s2": unit(-0.1, 1, 0),
        "s3": unit(0, 1, 0.1), "half": unit(1, 1, 0), "near": unit(1, 0.05, 0)}
FACES = {"face": 12.0, "f200": 50.0, "f400": 70.0}


class FakeFaces:
    def __init__(self, *a, **k):
        self.refs, self.ref_self_similarity, self.gold_self_similarity, self.gold = [], None, None, None

    def set_references(self, paths):
        self.refs = [(str(p), None) for p in paths]
        self.ref_self_similarity = 0.7

    def score(self, path, gold=False, qc=False):
        v = FACES.get(Path(path).stem)
        return SimpleNamespace(percent=v, gold_percent=None, faces=int(v is not None))


@pytest.fixture
def folders(tmp_path, monkeypatch):
    monkeypatch.setattr(facesim, "expand_refs", lambda paths: list(paths))
    monkeypatch.setattr(facesim, "FaceScorer", FakeFaces)
    monkeypatch.setattr(facesim, "available", lambda: (True, ""))
    monkeypatch.setattr(bodysim, "available", lambda: (True, ""))

    real = BA.AnchoredScorer

    def body_scorer(*a, **k):
        def window(path):
            stem = Path(path).stem
            return None if stem not in VECS else (stem, 1.0)

        return real.with_embedder(lambda key: {b: VECS[key] for b in BA.BACKBONES}, window)

    monkeypatch.setattr(BA, "AnchoredScorer", body_scorer)
    base, run = tmp_path / "base", tmp_path / "keel"
    for name in ("s1", "s2", "s3"):
        _png(base / f"body_seed_10{name[-1]}" / f"{name}.png")  # the seed folders' numbers are no steps
    _png(base / "face.png")
    for rel in ("keel face/f200.png", "keel face/f400.png", "keel body/half.png", "keel body/near.png"):
        _png(run / rel)
    # the steps come from the file names: name them by step
    (run / "keel face" / "f200.png").rename(run / "keel face" / "200_f200.png")
    (run / "keel face" / "f400.png").rename(run / "keel face" / "400_f400.png")
    (run / "keel body" / "half.png").rename(run / "keel body" / "200_half.png")
    (run / "keel body" / "near.png").rename(run / "keel body" / "400_near.png")
    FACES.update({"200_f200": 50.0, "400_f400": 70.0})
    VECS.update({"200_half": VECS["half"], "400_near": VECS["near"]})
    return base, run


def test_scoring_runs_against_the_baseline(folders):
    base, run = folders
    res = SF.score(["face_ref.png"], ["r1", "r2"], base, [("keel", str(run))], gold=False)
    pics = {(p["run"], Path(p["rel"]).stem): p for p in res["pictures"]}
    assert all(p["step"] is None for p in res["pictures"] if p["run"] == SF.BASELINE)
    assert abs(np.mean([pics[(SF.BASELINE, s)]["body"] for s in ("s1", "s2", "s3")])) < 1e-9  # the zero
    assert pics[(SF.BASELINE, "face")]["face"] == 12.0 and pics[(SF.BASELINE, "face")]["body"] is None
    assert pics[("keel", "200_f200")]["step"] == 200 and pics[("keel", "200_f200")]["body"] is None  # face only
    assert 0 < pics[("keel", "200_half")]["body"] < pics[("keel", "400_near")]["body"]
    assert res["refs"]["body_strangers"] == 3 and res["runs"] == ["keel"]
    rows = SF.by_step(res)
    assert [(r["run"], r["step"]) for r in rows] == [(SF.BASELINE, None), ("keel", 200), ("keel", 400)]
    assert SF.late_from(res) == 200
    lines = SF.summary_lines(res)
    assert lines[1].startswith(SF.BASELINE) and "0.0" in lines[1] and lines[2].startswith("keel")
    with pytest.raises(ValueError, match="Two runs"):
        SF.score([], [], None, [("a", str(run)), ("a", str(run))])
    with pytest.raises(FileNotFoundError):
        SF.collect("x", run / "missing")


def test_no_baseline_bodies_means_no_body_score(folders):
    _, run = folders
    res = SF.score(["face_ref.png"], ["r1", "r2"], None, [("keel", str(run))], gold=False)
    assert all(p["body"] is None for p in res["pictures"]) and "body_strangers" not in res["refs"]
    assert any(p["face"] for p in res["pictures"])


def test_a_missing_package_skips_that_score(folders, monkeypatch):
    base, run = folders
    monkeypatch.setattr(facesim, "available", lambda: (False, "face scoring needs 'insightface'"))
    res = SF.score(["face_ref.png"], ["r1", "r2"], base, [("keel", str(run))], gold=False)
    assert all(p["face"] is None for p in res["pictures"]) and any(p["body"] for p in res["pictures"])


def test_the_workbook_is_formulas_on_the_raw_scores(folders, tmp_path):
    from openpyxl import load_workbook

    base, run = folders
    res = SF.score(["face_ref.png"], ["r1", "r2"], base, [("keel", str(run)), ("other", str(run))], gold=False)
    out = SF.export(res, tmp_path / "out" / "cmp.xlsx", face_weight=0.6)
    assert out.with_suffix(".json").is_file()
    wb = load_workbook(out)
    assert wb.sheetnames == ["Summary", "By step", "Pictures", "Chart data", "Scale", "Notes"]
    sc, bs, sm, cd = wb["Scale"], wb["By step"], wb["Summary"], wb["Chart data"]
    assert sc["B5"].value.startswith("=IFERROR(AVERAGEIFS(Pictures!$E$5")  # the face floor: the baseline's mean
    assert abs(sc["C5"].value - 70.0) < 1e-9 and sc["B10"].value == 0.6 and sc["B12"].value == 200
    assert [sc[f"{c}7"].value for c in "BCDEF"] == [0.0, 100.0, 0.0, 1.0, 1.0]  # the anchored body: fixed
    assert bs["A5"].value == SF.BASELINE and bs["B5"].value == "none" and bs["B6"].value == 200
    assert bs["E6"].value.startswith("=IFERROR(AVERAGEIFS(Pictures!$E$5:$E$")
    assert "Scale!$B$5" in bs["H6"].value and "Scale!$B$7" in bs["J6"].value
    assert sm["A5"].value == SF.BASELINE and sm["A6"].value == "keel" and "MAXIFS" in sm["I6"].value
    assert cd["A5"].value == 200 and cd["B4"].value == "keel" and cd["B5"].value.startswith("='By step'!H")
    assert len(sm._charts) == 3 and wb.calculation.fullCalcOnLoad
