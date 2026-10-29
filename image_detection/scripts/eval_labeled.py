"""Precision / recall per tag over a labeled image set.

Folder mode (one tag; every image in --yes should match, every image in --no should not):
    python scripts/eval_labeled.py --yes F:\\yes --no F:\\no --text "women with blonde hair"
    python scripts/eval_labeled.py --yes F:\\yes --no F:\\no --id woman_blonde_hair --rubric "SUBJECT: ..."
--text generates the rubric with /expand_tag; --rubric uses yours as-is.

labels.json mode:
    python scripts/eval_labeled.py [--dir tests/labeled]

Expects <dir>/labels.json:
    {
      "tags": {"man_blonde_hair": "<rubric>", ...},          # optional; defaults to REFERENCE_RUBRICS
      "thresholds": {"woman_long_fingers": 0.8},             # optional per-tag override
      "images": {"img01.jpg": ["man_blonde_hair"], "img02.jpg": [], ...}   # positives per image
    }
Every listed image is a negative for every tag not in its list.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from image_detection.rubrics import REFERENCE_RUBRICS  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".avif", ".heic"}


def _folder_images(d: Path) -> list[Path]:
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(Path(__file__).resolve().parents[1] / "tests" / "labeled"))
    ap.add_argument("--yes", help="folder of images that should match")
    ap.add_argument("--no", help="folder of images that should not match")
    ap.add_argument("--text", help="tag text to expand into a rubric via /expand_tag")
    ap.add_argument("--id", help="tag id (with --rubric)")
    ap.add_argument("--rubric", help="rubric to use as-is (with --id)")
    ap.add_argument("--service", default="http://127.0.0.1:8090")
    ap.add_argument("--threshold", type=float, default=0.7)
    a = ap.parse_args()

    c = httpx.Client(base_url=a.service, timeout=900)
    paths: dict[str, Path] = {}
    labels: dict[str, list[str]] = {}
    thresholds: dict[str, float] = {}

    if a.yes or a.no:
        if a.rubric and a.id:
            rubrics = {a.id: a.rubric}
        elif a.text:
            r = c.post("/expand_tag", json={"text": a.text})
            r.raise_for_status()
            rubrics = {r.json()["id"]: r.json()["rubric"]}
        else:
            ap.error("folder mode needs --text, or --id with --rubric")
        (tag_id,) = rubrics
        print(f"tag: {tag_id}\nrubric: {rubrics[tag_id]}\n")
        for folder, positive in ((a.yes, True), (a.no, False)):
            if not folder:
                continue
            for p in _folder_images(Path(folder)):
                key = f"{'yes' if positive else 'no'}/{p.name}"
                paths[key] = p
                labels[key] = [tag_id] if positive else []
    else:
        root = Path(a.dir)
        spec = json.loads((root / "labels.json").read_text(encoding="utf-8"))
        rubrics = spec.get("tags") or REFERENCE_RUBRICS
        thresholds = spec.get("thresholds", {})
        labels = spec["images"]
        paths = {name: root / name for name in labels}

    tags = [{"id": t, "rubric": r} for t, r in rubrics.items()]
    images = [{"id": k, "data_b64": base64.b64encode(p.read_bytes()).decode()} for k, p in paths.items()]
    results: dict = {}
    for i in range(0, len(images), 64):
        # threshold 0 -> get raw match/confidence back and threshold per tag here
        r = c.post("/classify", json={"images": images[i : i + 64], "tags": tags, "threshold": 0.0})
        r.raise_for_status()
        results.update(r.json()["results"])

    print(f"{'tag':<24}{'thr':>5}{'TP':>4}{'FP':>4}{'FN':>4}{'TN':>4}{'prec':>7}{'recall':>8}")
    for t in rubrics:
        thr = thresholds.get(t, a.threshold)
        tp = fp = fn = tn = 0
        for name, pos in labels.items():
            res = results[name]
            if res["status"] != "ok":
                print(f"  ! {name}: {res['status']} {res.get('message', '')}")
                continue
            got = res["tags"][t]["match"] and res["tags"][t]["confidence"] >= thr
            want = t in pos
            tp += got and want
            fp += got and not want
            fn += want and not got
            tn += not got and not want
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        print(f"{t:<24}{thr:>5.2f}{tp:>4}{fp:>4}{fn:>4}{tn:>4}{prec:>7.2f}{rec:>8.2f}")

    print("\nper-image detail:")
    for name, res in results.items():
        if res["status"] != "ok":
            continue
        for t, v in res["tags"].items():
            want = t in labels[name]
            got = v["match"] and v["confidence"] >= thresholds.get(t, a.threshold)
            flag = "  " if got == want else ("FN" if want else "FP")
            print(f"  {flag} {name:<28} {t}: match={v['match']!s:<5} conf={v['confidence']:.2f}  {v['evidence']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
