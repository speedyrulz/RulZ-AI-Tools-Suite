"""Live acceptance checks from the spec's test checklist, run against a running service.

    python scripts/acceptance.py [--service http://127.0.0.1:8090] [--calls 50]

Covers: /health up + VRAM after warm-up, N consecutive classify calls with zero retries,
cache re-send + one-new-tag, 50 px image skipped, 8 tags under 60 s.
The 503 check (stop llama-server mid-run) is manual; see README.
"""

from __future__ import annotations

import argparse
import base64
import io
import random
import subprocess
import sys
import time

import httpx
from PIL import Image, ImageDraw

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from image_detection.rubrics import REFERENCE_RUBRICS  # noqa: E402

COLORS = ["red", "blue", "green", "yellow", "purple", "orange", "black"]
SHAPE_TAGS = [
    {"id": "red_circle", "rubric": "SUBJECT: a circle or disc. REQUIRE: the circle is clearly visible and filled red. "
     "EXCLUDE: squares or triangles; orange, pink or purple circles. FALSE IF: no round shape is visible."},
    {"id": "blue_square", "rubric": "SUBJECT: a square or rectangle. REQUIRE: the shape is clearly visible and filled blue. "
     "EXCLUDE: circles or triangles; purple or green shapes. FALSE IF: no four-sided shape is visible."},
]


def synthetic(seed: int, size=(800, 600)) -> tuple[str, dict[str, bool]]:
    rng = random.Random(seed)
    im = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(im)
    truth = {"red_circle": False, "blue_square": False}
    for _ in range(rng.randint(1, 3)):
        color, kind = rng.choice(COLORS), rng.choice(["circle", "square", "triangle"])
        x, y, r = rng.randint(120, 680), rng.randint(120, 480), rng.randint(50, 110)
        if kind == "circle":
            d.ellipse((x - r, y - r, x + r, y + r), fill=color)
            truth["red_circle"] |= color == "red"
        elif kind == "square":
            d.rectangle((x - r, y - r, x + r, y + r), fill=color)
            truth["blue_square"] |= color == "blue"
        else:
            d.polygon([(x, y - r), (x - r, y + r), (x + r, y + r)], fill=color)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode(), truth


def vram() -> str:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.used", "--format=csv,noheader"], text=True
        ).strip()
    except Exception as e:  # pragma: no cover
        return f"nvidia-smi unavailable: {e}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--service", default="http://127.0.0.1:8090")
    ap.add_argument("--calls", type=int, default=50)
    a = ap.parse_args()
    c = httpx.Client(base_url=a.service, timeout=600)
    ok = True

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal ok
        ok &= cond
        print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")

    h = c.get("/health").json()
    check("health reports up", h.get("server") == "up", str(h))
    seed0 = int(time.time())  # fresh images so the cache does not hide model calls
    img, _ = synthetic(seed0)
    c.post("/classify", json={"images": [{"id": "warm", "data_b64": img}], "tags": SHAPE_TAGS})
    print("VRAM after warm-up:\n  " + vram().replace("\n", "\n  "))

    retries0 = c.get("/health").json()["stats"]["retries"]
    t0, statuses, agree, total = time.time(), [], 0, 0
    for i in range(a.calls):
        img, truth = synthetic(seed0 + 1 + i)
        r = c.post("/classify", json={"images": [{"id": "x", "data_b64": img}], "tags": SHAPE_TAGS}).json()
        res = r["results"]["x"]
        statuses.append(res["status"])
        if res["status"] == "ok":
            for t, v in truth.items():
                total += 1
                agree += res["tags"][t]["match"] == v
    retries = c.get("/health").json()["stats"]["retries"] - retries0
    dt = time.time() - t0
    check(f"{a.calls} consecutive calls all ok", all(s == "ok" for s in statuses), f"({statuses.count('ok')} ok)")
    check("zero retries logged", retries == 0, f"(retries={retries})")
    print(f"       {dt / a.calls:.2f} s/image, synthetic-shape accuracy {agree}/{total}")

    page = [{"id": f"p{i}", "data_b64": synthetic(seed0 + 1 + i)[0]} for i in range(5)]
    calls0 = c.get("/health").json()["stats"]["model_calls"]
    r = c.post("/classify", json={"images": page, "tags": SHAPE_TAGS}).json()["results"]
    calls1 = c.get("/health").json()["stats"]["model_calls"]
    check("re-sent page fully cached", all(v["cached"] for v in r.values()) and calls1 == calls0)
    extra = SHAPE_TAGS + [{"id": "man_blonde_hair", "rubric": REFERENCE_RUBRICS["man_blonde_hair"]}]
    c.post("/classify", json={"images": page, "tags": extra})
    calls2 = c.get("/health").json()["stats"]["model_calls"]
    check("one new tag -> one model call per image", calls2 - calls1 == len(page), f"({calls2 - calls1} calls)")

    tiny = synthetic(1, (50, 50))[0]
    r = c.post("/classify", json={"images": [{"id": "t", "data_b64": tiny}], "tags": SHAPE_TAGS}).json()
    calls3 = c.get("/health").json()["stats"]["model_calls"]
    check("50 px image skipped, never sent", r["results"]["t"]["status"] == "skipped" and calls3 == calls2)

    eight = [{"id": f"tag_{i}", "rubric": t["rubric"]} for i, t in enumerate((SHAPE_TAGS + [
        {"rubric": REFERENCE_RUBRICS["man_blonde_hair"]}, {"rubric": REFERENCE_RUBRICS["woman_long_fingers"]}]) * 2)]
    img, _ = synthetic(seed0 + 10_000)
    t0 = time.time()
    r = c.post("/classify", json={"images": [{"id": "e", "data_b64": img}], "tags": eight}).json()
    dt = time.time() - t0
    check("8 tags on one image under 60 s", r["results"]["e"]["status"] == "ok" and dt < 60, f"({dt:.2f} s)")

    print("VRAM at end:\n  " + vram().replace("\n", "\n  "))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
