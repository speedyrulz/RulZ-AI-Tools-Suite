"""Score folders of finished samples for likeness, from any trainer, on RulzTrainer's probe scores: a spreadsheet that
compares runs checkpoint by checkpoint.

    python -m rulztrainer.score_folders --face-refs <folder> --body-refs <folder> --baseline <folder>
        --run "keel_fb=<folder>" --run "ai-toolkit=<folder>" [--out likeness.xlsx]

Runs. Each run is a folder of pictures, sub-folders included: one per checkpoint, or several seeds per checkpoint.
    kind   a picture whose own name or a sub-folder name below the run folder says "face" is a face picture (scored for
           the face), "body" a body picture (scored for the body); the name closest to the file decides, and a
           picture named neither gets both scores
    step   the number after "step" in the file name, else its first number of 1-9 digits (a 13-digit timestamp is
           skipped), else none; --step-pattern takes a regular expression whose first group is the step

Baseline. The base model's renders of the same prompts without a LoRA (the trainer's step-0 probe): the face scale's
zero (their mean face score) and the anchored body score's strangers (bodyanchor.py: their body windows set its zero;
4 to 8 seeds of the body prompt). Without baseline body pictures there is no body score.

Scores. face: ArcFace's best match among the face references (facesim); gold face: AdaFace's (--gold facenet: FaceNet's; nothing trains on it,
so it checks a run that trained against ArcFace); body: the anchored score (0 = the baseline's bodies, 100 = the body
references' mean). Each is put on the trainer's likeness scale (likeness.py: the face from its baseline to the
references' own level, stretched and curved by face_scale_span / _curve; the anchored body as it is) and combined with
the face and body weights. A checkpoint's score is its pictures' mean raw score on that scale, as a probe's seeds are.

The workbook. Summary (each run's late checkpoints, its best one, charts), By step, Pictures (every raw score), Chart
data, Scale (the floors, levels, span, curve, weights and the late steps: blue inputs, every score follows them) and
Notes. Every derived number is a formula. A .json with the raw numbers goes next to it.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

log = logging.getLogger("rulztrainer")

BASELINE = "Base model (no LoRA)"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
FONT = "Arial"
# the categorical palette in its fixed order (validated for colour-blind separation), one colour per run
COLORS = ["2A78D6", "EB6834", "1BAF7A", "EDA100", "E87BA4", "008300", "6250D6", "E34948"]
MARKERS = ["circle", "square", "triangle", "diamond", "x", "star", "dash", "plus"]
_STEP_AFTER_WORD = re.compile(r"step[_\- ]?(\d+)", re.IGNORECASE)
_FIRST_NUMBER = re.compile(r"(?<!\d)(\d{1,9})(?!\d)")


@dataclass
class Picture:
    run: str
    path: str
    rel: str  # the path below the run folder, for the sheet
    kind: str  # face | body | both
    step: int | None
    face: float | None = None  # ArcFace best match, raw %
    gold: float | None = None  # the gold recognizer's (AdaFace, or FaceNet) best match, raw %
    faces: int | None = None  # faces found
    body: float | None = None  # anchored body score
    cover: float | None = None  # the share of the body window inside the picture


def kind_of(path: Path, root: Path) -> str:
    """face | body | both: the name closest to the file that says exactly one of "face" and "body" decides."""
    rel = path.relative_to(root)
    for part in [rel.stem, *reversed(rel.parts[:-1])]:  # the file, then its folders from the nearest
        low = part.lower()
        face, body = "face" in low, "body" in low
        if face != body:
            return "face" if face else "body"
    return "both"


def step_of(path: Path, pattern: re.Pattern | None = None) -> int | None:
    stem = path.stem
    if pattern is not None:
        m = pattern.search(stem)
        return int(m.group(1)) if m else None
    m = _STEP_AFTER_WORD.search(stem) or _FIRST_NUMBER.search(stem)
    return int(m.group(1)) if m else None


def collect(run: str, folder: str | Path, pattern: re.Pattern | None = None) -> list[Picture]:
    """The pictures in a run folder (sub-folders included), in step and then name order."""
    root = Path(folder)
    if not root.is_dir():
        raise FileNotFoundError(f"{run}: no folder {root}")
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    out = [Picture(run, str(p), p.relative_to(root).as_posix(), kind_of(p, root), step_of(p, pattern)) for p in files]
    out.sort(key=lambda q: (q.step is None, q.step or 0, q.rel))
    if not out:
        raise FileNotFoundError(f"{run}: no pictures in {root}")
    return out


def parse_run(text: str) -> tuple[str, str]:
    """"name=folder" -> (name, folder); a bare folder is named after itself."""
    name, sep, folder = text.partition("=")
    if not sep or not folder.strip() or Path(text).exists():
        folder = text
        name = Path(text.rstrip("/\\")).name
    return name.strip(), folder.strip().strip('"')


# ------------------------------------------------------------ scoring


def score(face_refs: list[str], body_refs: list[str], baseline: str | Path | None, runs: list[tuple[str, str]],
          models_dir: str | Path = "models", device: str = "cuda:0", face_provider: str = "cpu",
          gold: bool | str = True, step_pattern: str = "", progress=None) -> dict:
    """Score every picture: {"pictures": [Picture as dict], "refs": {...}, "runs": [names]}; progress(text,
    fraction). gold: the gold face recognizer, "adaface" (True) or "facenet"; False / "" for none."""
    gold = "adaface" if gold is True else (str(gold).lower() if gold else "")
    from . import bodyanchor, bodysim, facesim
    from .bodysim import WINDOW_MIN_SCORE_COVER

    def say(text, frac=None):
        log.info(text)
        if progress:
            progress(text, frac)

    pattern = re.compile(step_pattern) if step_pattern else None
    pics: list[Picture] = []
    if baseline:
        pics += collect(BASELINE, baseline, pattern)
        for p in pics:  # one zero for the whole comparison: its file numbers are seeds, not steps
            p.step = None
    names = []
    for name, folder in runs:
        if name in names or name == BASELINE:
            raise ValueError(f"Two runs are named {name!r}: give each its own name (name=folder)")
        names.append(name)
        pics += collect(name, folder, pattern)
    faces = [p for p in pics if p.kind != "body"]
    bodies = [p for p in pics if p.kind != "face"]
    refs: dict = {"face": [], "body": []}

    fs = None
    face_ok, face_why = facesim.available() if face_refs and faces else (False, "")
    if face_why:
        log.warning("%s; no face scores", face_why)
    if face_ok:
        say("Loading the face scorer (ArcFace)...")
        fs = facesim.FaceScorer(face_provider, 640, progress=progress)
        fs.set_references(face_refs)
        if not fs.refs:
            log.warning("No face in any face reference: no face scores")
            fs = None
        elif gold:
            from . import idloss

            try:
                fs.set_gold(idloss.gold_embedder(gold, models_dir, progress=progress))
            except Exception as e:  # noqa: BLE001
                log.warning("No gold face scores (%s could not load: %s)", idloss.gold_label(gold), e)
                fs.gold = None
        if fs is not None:
            refs["face"] = [name for name, _ in fs.refs]
            refs["face_level"] = fs.ref_self_similarity
            refs["gold_level"] = fs.gold_self_similarity if getattr(fs, "gold", None) is not None else None
            refs["gold_name"] = gold if getattr(fs, "gold", None) is not None else ""

    bs = None
    body_ok, body_why = bodysim.available() if body_refs and bodies else (False, "")
    if body_why:
        log.warning("%s; no body scores", body_why)
    elif body_ok and not any(p.run == BASELINE for p in bodies):
        log.warning("No baseline body pictures (the base model without a LoRA): the body score has no zero, so no "
                    "body scores")
        body_ok = False
    if body_ok:
        say("Loading the body scorer (SigLIP 2 + DINOv2)...")
        bs = bodyanchor.AnchoredScorer(models_dir, device, progress=progress)
        bs.set_references(body_refs)
        if not bs.ready:
            log.warning("No body reference shows the body from the neck to mid-thigh: no body scores")
            bs = None
    total = (len(bodies) if bs else 0) + (len(faces) if fs else 0)
    done = 0
    if bs is not None:
        embs = {}
        for p in bodies:  # each body window once; the baseline's are the strangers too
            done += 1
            say(f"Body windows: {p.run}: {p.rel}", done / total)
            win = bs.window(p.path)
            p.cover = None if win is None else round(float(win[1]), 3)
            if win is not None and win[1] >= WINDOW_MIN_SCORE_COVER:
                embs[p.path] = bs.embed(win[0])
        base = [embs.get(p.path) for p in bodies if p.run == BASELINE]
        if bs.set_stranger_embeddings([e for e in base if e is not None], len(base)) < bodyanchor.MIN_STRANGERS:
            bs = None
        else:
            for p in bodies:
                if embs.get(p.path) is not None:
                    p.body = bs.score_embedding(embs[p.path])
            refs.update(body=bs.ref_names, body_strangers=bs.stranger_count, body_level=bs.ref_self_similarity)
    for p in faces if fs is not None else []:
        done += 1
        say(f"Faces: {p.run}: {p.rel}", done / total)
        sc = fs.score(p.path, gold=True, qc=True)  # with the probes' subject check
        p.face, p.gold, p.faces = sc.percent, sc.gold_percent, sc.faces
    return {"pictures": [asdict(p) for p in pics], "refs": refs, "runs": names,
            "baseline": bool(baseline), "steps_from": step_pattern or "file names"}


# ------------------------------------------------------------ the scale (likeness.py's, for the printout)


def scale_of(result: dict, face_weight: float = 0.7):
    """A LikenessController on the result's scale: the baseline's floors, the references' levels, the anchored body."""
    from .config import TrainConfig
    from .likeness import LikenessController

    refs = result["refs"]
    cfg = TrainConfig(body_score="anchored", face_weight=face_weight, body_weight=1.0 - face_weight)
    ctrl = LikenessController(cfg, face_ceiling=_pct(refs.get("face_level")), gold_ceiling=_pct(refs.get("gold_level")))
    base = [p for p in result["pictures"] if p["run"] == BASELINE]
    ctrl.has_gold = refs.get("gold_level") is not None
    ctrl._set_floors(_mean(p["face"] for p in base), _mean(p["body"] for p in base), _mean(p["gold"] for p in base))
    return ctrl


def _pct(v):
    return None if v is None else float(v) * 100.0


def _mean(values):
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def by_step(result: dict) -> list[dict]:
    """Each run's checkpoints: [{"run", "step", "face", "gold", "body", "n_face", "n_body"}], raw means."""
    groups: dict[tuple, list[dict]] = {}
    for p in result["pictures"]:
        groups.setdefault((p["run"], p["step"]), []).append(p)
    out = []
    for (run, step), ps in groups.items():
        out.append({"run": run, "step": step, "face": _mean(p["face"] for p in ps),
                    "gold": _mean(p["gold"] for p in ps), "body": _mean(p["body"] for p in ps),
                    "n_face": sum(p["face"] is not None for p in ps), "n_body": sum(p["body"] is not None for p in ps)})
    order = [BASELINE, *result["runs"]]
    out.sort(key=lambda r: (order.index(r["run"]), r["step"] is None, r["step"] or 0))
    return out


def late_from(result: dict) -> int:
    """The first late step: half of the largest step any run reached (0 without steps)."""
    steps = [p["step"] for p in result["pictures"] if p["step"] is not None and p["run"] != BASELINE]
    return max(steps) // 2 if steps else 0


def summary_lines(result: dict, face_weight: float = 0.7) -> list[str]:
    """A plain-text table of each run's late checkpoints on the likeness scale."""
    from .likeness import combine

    ctrl = scale_of(result, face_weight)
    rows = by_step(result)
    late = late_from(result)
    lines = [f"{'run':28s} {'face':>6s} {'gold':>6s} {'body':>6s} {'comb.':>6s}  (late: steps >= {late}; 0 = the base "
             "model, the face's 100 past the references' own level, the body's 100 = the references' mean)"]
    for run in [BASELINE, *result["runs"]]:
        mine = [r for r in rows if r["run"] == run]
        pick = [r for r in mine if r["step"] is not None and r["step"] >= late] or mine
        if not pick:
            continue
        f = _mean(ctrl.norm("face", r["face"]) for r in pick)
        g = _mean(ctrl.norm("gold", r["gold"]) for r in pick) if ctrl.has_gold else None
        b = _mean(ctrl.norm("body", r["body"]) for r in pick)
        c = _mean(combine(ctrl.norm("face", r["face"]), ctrl.norm("body", r["body"]), face_weight, 1 - face_weight)
                  for r in pick)
        fmt = lambda v: "   n/a" if v is None else f"{v:6.1f}"  # noqa: E731
        lines.append(f"{run[:28]:28s} {fmt(f)} {fmt(g)} {fmt(b)} {fmt(c)}")
    return lines


# ------------------------------------------------------------ the workbook


def _gold_label(refs: dict) -> str:
    from .idloss import gold_label

    return gold_label(refs.get("gold_name"))


def build_workbook(result: dict, face_weight: float = 0.7, title: str = "Likeness of finished runs"):
    from openpyxl import Workbook
    from openpyxl.chart import LineChart, Reference
    from openpyxl.chart.shapes import GraphicalProperties
    from openpyxl.drawing.line import LineProperties
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter

    from .config import TrainConfig
    from .likeness import CEILING_DEFAULT, FLOOR_DEFAULT, MIN_SPAN

    d = TrainConfig()
    h1, h2 = Font(name=FONT, size=14, bold=True), Font(name=FONT, size=11, bold=True)
    txt, bold = Font(name=FONT, size=10), Font(name=FONT, size=10, bold=True)
    muted = Font(name=FONT, size=9, italic=True, color="666666")
    hdr, hfill = Font(name=FONT, size=10, bold=True, color="FFFFFF"), PatternFill("solid", fgColor="1F2937")
    blue, black = Font(name=FONT, size=10, color="0000FF"), Font(name=FONT, size=10, color="000000")
    green, yellow = Font(name=FONT, size=10, color="008000"), PatternFill("solid", fgColor="FFFF00")
    band, box = PatternFill("solid", fgColor="F2F4F7"), Border(bottom=Side(style="thin", color="D0D4DA"))
    wrap, center = Alignment(wrap_text=True, vertical="top"), Alignment(horizontal="center", vertical="center",
                                                                          wrap_text=True)
    refs, pics, runs = result["refs"], result["pictures"], result["runs"]
    steps_rows = by_step(result)

    wb = Workbook()
    summary = wb.active
    summary.title = "Summary"
    bystep = wb.create_sheet("By step")
    pictures = wb.create_sheet("Pictures")
    chart_data = wb.create_sheet("Chart data")
    scale = wb.create_sheet("Scale")
    notes = wb.create_sheet("Notes")

    def header(ws, row, labels, widths=None, height=42):
        for j, lab in enumerate(labels, 1):
            c = ws.cell(row, j, lab)
            c.font, c.fill, c.alignment = hdr, hfill, center
        ws.row_dimensions[row].height = height
        for j, w in enumerate(widths or [], 1):
            ws.column_dimensions[get_column_letter(j)].width = w

    # -------------------------------------------------------- Pictures: the raw scores (inputs)
    p0, p1 = 5, 4 + max(1, len(pics))
    pictures["A1"] = "Every picture's raw scores"
    pictures["A1"].font = h1
    pictures["A2"] = ("Measured with RulzTrainer's scorers (blue). Face and gold face: the best match among the face "
                      "references, in %. Body: the anchored score, 0 = the baseline's bodies, 100 = the body "
                      "references' mean. Empty: no face found, or no body window in the picture.")
    pictures["A2"].font = muted
    header(pictures, 4, ["Run", "Step", "Kind", "Picture (below the run folder)", "Face raw (ArcFace %)",
                         f"Gold face raw ({_gold_label(refs)} %)", "Body raw (anchored)", "Faces found", "Body window in frame"],
           [26, 8, 7, 44, 12, 12, 12, 9, 11])
    for i, p in enumerate(pics):
        r = p0 + i
        vals = [p["run"], "none" if p["step"] is None else p["step"], p["kind"], p["rel"], p["face"], p["gold"],
                p["body"], p["faces"], p["cover"]]
        for j, v in enumerate(vals, 1):
            c = pictures.cell(r, j, round(v, 3) if isinstance(v, float) else v)
            c.font = blue if j >= 5 else txt
            if j in (5, 6, 7):
                c.number_format = "0.0"
            elif j == 9:
                c.number_format = "0%"
    pictures.freeze_panes = "B5"
    pictures.auto_filter.ref = f"A4:I{p1}"

    def prange(col):
        return f"Pictures!${col}${p0}:${col}${p1}"

    # -------------------------------------------------------- Scale (inputs and the scale's tops)
    scale["A1"] = "Likeness scale"
    scale["A1"].font = h1
    scale["A2"] = ("RulzTrainer's normalize() (likeness.py): score = MIN(100, 100 x MAX(0, (raw - floor) / "
                   "(top - floor)) ^ curve), top = floor + (MAX(level, floor + minimum span) - floor) x span. "
                   "0 = the base model without a LoRA. The anchored body score is on its own scale already "
                   "(0 = the baseline's bodies, 100 = the references' mean), so its floor, top and curve are fixed.")
    scale["A2"].font, scale["A2"].alignment = txt, wrap
    scale.merge_cells("A2:I2")
    scale.row_dimensions[2].height = 44
    header(scale, 4, ["Score", "Floor (raw)", "References' own level (raw)", "Minimum span", "Span", "Curve",
                      "Top of the scale (raw)", "Your references' level reads", "Where they come from"],
           [26, 12, 14, 10, 9, 9, 13, 13, 70], 44)
    base_face = f'IFERROR(AVERAGEIFS({prange("E")},{prange("A")},"{BASELINE}"),{FLOOR_DEFAULT["face"]:g})'
    base_gold = f'IFERROR(AVERAGEIFS({prange("F")},{prange("A")},"{BASELINE}"),{FLOOR_DEFAULT["gold"]:g})'
    rows = [("Face (ArcFace)", f"={base_face}", _pct(refs.get("face_level")) or CEILING_DEFAULT["face"],
             MIN_SPAN["face"], d.face_scale_span, d.face_scale_curve,
             "Floor: the baseline face pictures' mean (the trainer's default without them). Level: each face "
             "reference's best match among the others, the median."),
            (f"Gold face ({_gold_label(refs)})", f"={base_gold}", _pct(refs.get("gold_level")) or CEILING_DEFAULT["gold"],
             MIN_SPAN["gold"], "=E5", "=F5", f"The same with {_gold_label(refs)}; shaped like the face, as the trainer does."),
            ("Body (anchored)", 0.0, 100.0, 0.0, 1.0, 1.0,
             "Fixed by construction: the baseline's bodies average 0, the body references' mean is 100, linear. "
             "Leave these as they are.")]
    for i, (name, floor, level, mspan, span, curve, src) in enumerate(rows):
        r = 5 + i
        scale.cell(r, 1, name).font = bold
        for col, v in ((2, floor), (3, level), (4, mspan), (5, span), (6, curve)):
            c = scale.cell(r, col, round(v, 4) if isinstance(v, float) else v)
            is_formula = isinstance(v, str)
            c.font = (green if col == 2 else black) if is_formula else blue
            c.number_format = "0.0" if col <= 4 else "0.00"
            if i == 0 and col in (5, 6):
                c.fill = yellow
        c = scale.cell(r, 7, f"=B{r}+(MAX(C{r},B{r}+D{r})-B{r})*E{r}")
        c.font, c.number_format = black, "0.0"
        c = scale.cell(r, 8, f"=MIN(100,100*MAX(0,(C{r}-B{r})/(G{r}-B{r}))^F{r})")
        c.font, c.number_format = black, "0.0"
        c = scale.cell(r, 9, src)
        c.font, c.alignment = txt, wrap
        scale.row_dimensions[r].height = 30
    scale["A9"] = "Weights and late checkpoints"
    scale["A9"].font = h2
    for r, (lab, v, fmt, note) in enumerate((
            ("Face weight", face_weight, "0%", "face_weight: the combined score is the face and body's weighted mean."),
            ("Body weight", round(1.0 - face_weight, 4), "0%", "body_weight."),
            ("Late checkpoints from step", late_from(result), "0",
             "The Summary's late means take the checkpoints from this step on (half of the largest step).")), start=10):
        scale.cell(r, 1, lab).font = bold
        c = scale.cell(r, 2, v)
        c.font, c.fill, c.number_format = blue, yellow, fmt
        scale.cell(r, 3, note).font = txt
    scale["A14"] = ("Legend: blue = measured or default input; yellow fill = a lever (every score follows it); black = "
                    "formula; green = a formula on another sheet.")
    scale["A14"].font = muted
    scale.freeze_panes = "B5"

    def norm(cell, srow):
        return (f'IF({cell}="","",MIN(100,100*MAX(0,({cell}-Scale!$B${srow})/(Scale!$G${srow}-Scale!$B${srow}))'
                f'^Scale!$F${srow}))')

    def combined(f, b):
        return (f'IF(AND({f}="",{b}=""),"",(IF({f}="",0,{f}*Scale!$B$10)+IF({b}="",0,{b}*Scale!$B$11))'
                f'/(IF({f}="",0,Scale!$B$10)+IF({b}="",0,Scale!$B$11)))')

    # -------------------------------------------------------- By step: each checkpoint (formulas on Pictures)
    s0, s1 = 5, 4 + max(1, len(steps_rows))
    bystep["A1"] = "Every checkpoint"
    bystep["A1"].font = h1
    bystep["A2"] = ("A checkpoint's raw scores are its pictures' means; the 0-100 columns put them on the Scale "
                    "sheet's scale, as the trainer does a probe's seeds.")
    bystep["A2"].font = muted
    header(bystep, 4, ["Run", "Step", "Face pictures", "Body pictures", "Face raw", "Gold face raw", "Body raw",
                       "Face (0-100)", "Gold face (0-100)", "Body (0-100)", "Combined (0-100)", "Late (1 = yes)",
                       "Body: squared gap to the run's late mean", "Body: counted in the spread", "Step as a number"],
           [26, 8, 9, 9, 10, 10, 10, 11, 11, 11, 12, 9, 12, 12, 10])
    row_of: dict[tuple, int] = {}
    for i, sr in enumerate(steps_rows):
        r = s0 + i
        row_of[(sr["run"], sr["step"])] = r
        bystep.cell(r, 1, sr["run"]).font = txt
        bystep.cell(r, 2, "none" if sr["step"] is None else sr["step"]).font = txt
        crit = f'{prange("A")},$A{r},{prange("B")},$B{r}'
        bystep.cell(r, 3, f'=COUNTIFS({crit},{prange("E")},"<>")')
        bystep.cell(r, 4, f'=COUNTIFS({crit},{prange("G")},"<>")')
        for col, src in ((5, "E"), (6, "F"), (7, "G")):
            bystep.cell(r, col, f'=IFERROR(AVERAGEIFS({prange(src)},{crit}),"")')
        bystep.cell(r, 8, f"={norm(f'E{r}', 5)}")
        bystep.cell(r, 9, f"={norm(f'F{r}', 6)}")
        bystep.cell(r, 10, f"={norm(f'G{r}', 7)}")
        bystep.cell(r, 11, f"={combined(f'H{r}', f'J{r}')}")
        # late: from the Scale sheet's late step on; a run without steps counts all its pictures
        bystep.cell(r, 12, f'=IF(COUNTIFS($A${s0}:$A${s1},$A{r},$B${s0}:$B${s1},">="&Scale!$B$12)=0,1,'
                           f'IF(AND(ISNUMBER(B{r}),N(B{r})>=Scale!$B$12),1,0))')
        bystep.cell(r, 14, f"=IF(AND(L{r}=1,ISNUMBER(J{r})),1,0)")
        bystep.cell(r, 13, f"=IF(N{r}=1,(N(J{r})-AVERAGEIFS($J${s0}:$J${s1},$A${s0}:$A${s1},$A{r},"
                           f"$L${s0}:$L${s1},1))^2,0)")
        bystep.cell(r, 15, f"=IF(ISNUMBER(B{r}),B{r},0)")
        for j in range(3, 16):
            c = bystep.cell(r, j)
            c.font, c.number_format = black, ("0" if j <= 4 or j in (12, 14, 15) else "0.0")
        if ([BASELINE, *runs].index(sr["run"])) % 2:
            for j in range(1, 16):
                bystep.cell(r, j).fill = band
    for letter in "MNO":  # helpers for the Summary
        bystep.column_dimensions[letter].hidden = True
    bystep.freeze_panes = "C5"
    bystep.auto_filter.ref = f"A4:L{s1}"

    def srange(col):
        return f"'By step'!${col}${s0}:${col}${s1}"

    # -------------------------------------------------------- Chart data: steps x runs (only the checkpoints there are)
    numeric = sorted({sr["step"] for sr in steps_rows if sr["step"] is not None and sr["run"] != BASELINE})
    chart_data["A1"] = "Chart data (formulas on the By step sheet)"
    chart_data["A1"].font = h1
    blocks, row0 = {}, 3
    for title_, col, key in (("Face (0-100)", "H", "face"), ("Body (0-100)", "J", "body"),
                             ("Combined (0-100)", "K", None)):
        chart_data.cell(row0, 1, title_).font = h2
        chart_data.cell(row0 + 1, 1, "Step").font = bold
        for j, run in enumerate(runs, 2):
            chart_data.cell(row0 + 1, j, run).font = bold
        for i, step in enumerate(numeric):
            rr = row0 + 2 + i
            chart_data.cell(rr, 1, step).font = txt
            for j, run in enumerate(runs, 2):
                src = row_of.get((run, step))
                sr = steps_rows[src - s0] if src else None
                has = sr is not None and (sr["face"] is not None or sr["body"] is not None if key is None
                                          else sr[key] is not None)
                if has:  # an empty cell is a gap in the line; a formula's "" would plot as 0
                    c = chart_data.cell(rr, j, f"='By step'!{col}{src}")
                    c.font, c.number_format = green, "0.0"
        blocks[title_] = row0
        row0 += len(numeric) + 4
    chart_data.column_dimensions["A"].width = 8
    for j in range(2, 2 + len(runs)):
        chart_data.column_dimensions[get_column_letter(j)].width = 20

    # -------------------------------------------------------- Summary
    summary["A1"] = title
    summary["A1"].font = h1
    summary["A2"] = ("Scored with RulzTrainer's probe scorers on its likeness scale: 0 = the base model without a LoRA "
                     "(the baseline pictures); the face's 100 sits past the face references' own level (the Scale "
                     "sheet says where that level reads), the body's 100 is the body references' mean. Late = the "
                     "checkpoints from the Scale sheet's late step on (all of a run's pictures without steps). "
                     "Change the Scale sheet's yellow cells and everything follows.")
    summary["A2"].font, summary["A2"].alignment = txt, wrap
    summary.merge_cells("A2:K2")
    summary.row_dimensions[2].height = 44
    header(summary, 4, ["Run", "Pictures", "Checkpoints", "Face (late)", "Gold face (late)", "Body (late)",
                        "Combined (late)", "Body checkpoint spread (late SD)", "Best combined checkpoint",
                        "Best at step", "Combined (all checkpoints)"],
           [26, 9, 11, 10, 10, 10, 11, 13, 12, 9, 12], 54)
    every = [BASELINE, *runs] if result.get("baseline") else list(runs)
    for i, run in enumerate(every):
        r = 5 + i
        summary.cell(r, 1, run).font = bold
        summary.cell(r, 2, f'=COUNTIFS({prange("A")},$A{r})')
        summary.cell(r, 3, f'=COUNTIFS({srange("A")},$A{r})')
        for col, src in ((4, "H"), (5, "I"), (6, "J"), (7, "K")):
            summary.cell(r, col, f'=IFERROR(AVERAGEIFS({srange(src)},{srange("A")},$A{r},{srange("L")},1),"")')
        summary.cell(r, 8, f'=IFERROR(SQRT(SUMIFS({srange("M")},{srange("A")},$A{r})'
                           f'/(SUMIFS({srange("N")},{srange("A")},$A{r})-1)),"")')
        summary.cell(r, 9, f'=IF(COUNTIFS({srange("A")},$A{r},{srange("K")},">=0")=0,"",'
                           f'_xlfn.MAXIFS({srange("K")},{srange("A")},$A{r}))')
        summary.cell(r, 10, f'=IF(I{r}="","",SUMPRODUCT(MAX(({srange("A")}=$A{r})*({srange("K")}=I{r})'
                            f'*{srange("O")})))')
        summary.cell(r, 11, f'=IFERROR(AVERAGEIFS({srange("K")},{srange("A")},$A{r}),"")')
        for j in range(2, 12):
            c = summary.cell(r, j)
            c.font, c.number_format = black, ("0" if j in (2, 3, 10) else "0.0")
        for j in range(1, 12):
            summary.cell(r, j).border = box
    below = 5 + len(every) + 1
    summary.cell(below, 1, "Body checkpoint spread: how much one checkpoint's body score moves by chance; gaps between "
                           "runs smaller than about twice it are noise.").font = muted

    def line_chart(title_, anchor, y_title):
        ch = LineChart()
        ch.title, ch.style = title_, 2
        ch.height, ch.width = 9.5, 15.5
        ch.y_axis.title, ch.x_axis.title = y_title, "Training step"
        ch.y_axis.scaling.min = 0
        ch.y_axis.majorGridlines.spPr = GraphicalProperties(ln=LineProperties(solidFill="E3E5E8", w=6350))
        ch.x_axis.delete = ch.y_axis.delete = False
        r0 = blocks[title_]
        ch.add_data(Reference(chart_data, min_col=2, max_col=1 + len(runs), min_row=r0 + 1,
                              max_row=r0 + 1 + len(numeric)), titles_from_data=True)
        ch.set_categories(Reference(chart_data, min_col=1, min_row=r0 + 2, max_row=r0 + 1 + len(numeric)))
        for k, s in enumerate(ch.series):
            color, marker = COLORS[k % len(COLORS)], MARKERS[k % len(MARKERS)]
            s.graphicalProperties.line.solidFill = color
            s.graphicalProperties.line.width = 22225  # 1.75 pt
            s.marker.symbol, s.marker.size = marker, 7
            if marker in ("x", "plus", "dash", "star"):  # line-drawn markers: no fill, the outline in the colour
                s.marker.graphicalProperties = GraphicalProperties(noFill=True)
            else:
                s.marker.graphicalProperties = GraphicalProperties(solidFill=color)
            s.marker.graphicalProperties.line.solidFill = color
            s.smooth = False
        ch.legend.position, ch.legend.overlay = "b", False
        ch.y_axis.number_format = "0"
        for t in (ch.title, ch.x_axis.title, ch.y_axis.title):
            t.overlay = False
        summary.add_chart(ch, anchor)

    if numeric and runs:
        top = below + 2
        line_chart("Face (0-100)", f"A{top}", "Face (0-100)")
        line_chart("Body (0-100)", f"G{top}", "Body (0-100)")
        line_chart("Combined (0-100)", f"A{top + 20}", "Combined (0-100)")
        summary.cell(top + 20, 7, "The charts read the Chart data sheet, which reads the By step sheet.").font = muted
    summary.freeze_panes = "B5"

    # -------------------------------------------------------- Notes
    notes["A1"] = "How these scores are made"
    notes["A1"].font = h1
    body_note = (f"{refs.get('body_strangers', 0)} baseline body pictures set its zero" if refs.get("body")
                 else "no body score (no body references, no baseline body pictures, or no body window in them)")
    level = refs.get("body_level")
    lines = [
        ("Face", "ArcFace (InsightFace buffalo_l): the largest face's best cosine match among the face references, "
                 f"as a %. {len(refs.get('face') or [])} face references have a face."),
        ("Gold face", ("AdaFace (ViT-Base, WebFace4M)" if refs.get("gold_name") != "facenet" else
                       "FaceNet (VGGFace2)") + " on the same face. Nothing in RulzTrainer trains on it, so it checks "
                      "runs that trained against ArcFace."),
        ("Body", "The anchored body score (bodyanchor.py): the body from the neck to mid-thigh, upright, cut out, "
                 "its colour and sharpness normalised, embedded by SigLIP 2 and DINOv2; how far it moved from the "
                 "baseline's bodies toward the body references, 0 = the baseline's mean, 100 = the references' mean. "
                 f"{len(refs.get('body') or [])} body references show that window; {body_note}."
                 + (f" A typical reference scores {level * 100:.0f} against the others; a probe in one fixed outfit "
                    "and scene reaches less, so compare runs with each other, not with 100." if level else "")),
        ("Kinds", "A picture whose own name or a folder name below its run folder says face is a face picture, body "
                  "a body picture; a picture named neither gets both scores."),
        ("Steps", f"From the file names ({result.get('steps_from', 'file names')}): the number after \"step\", else "
                  "the first number of 1-9 digits. Pictures without one are in the row \"none\"."),
        ("Baseline", "The base model's renders of the same prompts without a LoRA (the trainer's step-0 probe): the "
                     "face's zero and the body's strangers. Its own rows should read about 0."),
        ("Noise", "One picture per checkpoint moves a few points by chance (the Summary's spread). Several seeds per "
                  "checkpoint, all in the run folder, average it down."),
    ]
    for i, (k, v) in enumerate(lines):
        r = 3 + i
        notes.cell(r, 1, k).font = bold
        c = notes.cell(r, 2, v)
        c.font, c.alignment = txt, wrap
        notes.row_dimensions[r].height = 48
    notes.column_dimensions["A"].width = 12
    notes.column_dimensions["B"].width = 110

    wb.calculation.fullCalcOnLoad = True  # openpyxl writes no cached values: the reader computes them on opening
    return wb


def export(result: dict, out: str | Path, face_weight: float = 0.7) -> Path:
    """Write the workbook and the raw numbers (<out>.json next to it)."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    build_workbook(result, face_weight).save(out)
    out.with_suffix(".json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m rulztrainer.score_folders", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--face-refs", nargs="*", default=[], help="face reference pictures or folders")
    ap.add_argument("--body-refs", nargs="*", default=[], help="body reference pictures or folders")
    ap.add_argument("--baseline", default="", help="the base model's renders of the same prompts, no LoRA")
    ap.add_argument("--run", action="append", default=[], metavar="NAME=FOLDER", help="a run's folder (repeat)")
    ap.add_argument("--out", default="likeness_runs.xlsx")
    ap.add_argument("--face-weight", type=float, default=0.7)
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--device", default="cuda:0", help="for SigLIP 2 and DINOv2")
    ap.add_argument("--face-provider", default="cpu", help="cpu | cuda (InsightFace)")
    ap.add_argument("--gold", default="adaface", choices=("adaface", "facenet"),
                    help="the gold face recognizer (default adaface; facenet: the one before 3.14.0)")
    ap.add_argument("--no-gold", action="store_true", help="skip the gold face")
    ap.add_argument("--step-pattern", default="", help="a regular expression whose first group is the step")
    a = ap.parse_args(argv)
    if not a.run:
        ap.error("give at least one --run NAME=FOLDER")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = score(a.face_refs, a.body_refs, a.baseline or None, [parse_run(r) for r in a.run], a.models_dir,
                   a.device, a.face_provider, "" if a.no_gold else a.gold, a.step_pattern)
    out = export(result, a.out, a.face_weight)
    print("\n".join(summary_lines(result, a.face_weight)))
    print(f"Saved {out} and {out.with_suffix('.json').name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
