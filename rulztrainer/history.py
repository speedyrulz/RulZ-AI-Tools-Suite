"""Run history: what a training run's likeness scores looked like, kept after the job folder is reused.

Each run with likeness data writes one small JSON file in output/_history/ while it trains (so a crash keeps it) and
once more when it ends. The WebUI's Previous Runs tab lists them and overlays any number of them, like TensorBoard.
What is kept is what the Monitor's two likeness charts plot:

  * identity: the mean face and the mean body score of the sample images at each sampling step;
  * probes:   the smoothed face, body and combined score of the likeness probes at each probe step, and for a race
              each configuration's own (probes.racers, by name; "winner" for the probes after the race);

on the normalized scale (0 = a stranger, 100 = the references) where the run had one, else the raw percentages.
The probe images are kept too, as small JPEG copies in output/_history/<run id>/ (a later run with the same job name
overwrites the job folder's own), each with its step, kind, seed, configuration and that probe's scores
(probe_images), for the Previous Runs tab's side-by-side comparison. Runs from before the history existed are imported
from their job folders' status.json."""
from __future__ import annotations

import json
import re
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HISTORY_DIR = ROOT / "output" / "_history"
SCAN_DIRS = ("_history", "_crops")  # folders under output/ that are not jobs
CONFIG_KEYS = ("job_name", "preset", "network_type", "lora_rank", "lokr_factor", "oft_block_size", "dora",
               "attn_localize", "id_align", "learning_rate", "optimizer", "steps", "batch_size", "lora_ema",
               "face_weight", "body_weight", "face_loss_weight", "body_loss_weight", "crop_augment", "isolation",
               "finetune", "race", "timestep_sampling", "likeness_curriculum", "face_scale_span", "face_scale_curve",
               "body_scale_span", "body_scale_curve", "probe_face_prompt", "probe_body_prompt", "probe_seed",
               "probe_seeds", "probe_size", "probe_steps", "probe_cfg", "turbo_lora", "resolutions",
               "reward_steps", "body_align", "selftrain", "probe_ema", "body_score", "lora_plus_ratio", "race_follow",
               "race_patience_rising")


def new_run_id(job_name: str, now: float | None = None) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", job_name).strip("._") or "job"
    return f"{safe}-{datetime.fromtimestamp(now or time.time()):%Y%m%d-%H%M%S}"


def _mean_by_step(rows: list[tuple[int, float | None]]) -> list[list]:
    per: dict[int, list[float]] = {}
    for step, v in rows:
        if v is not None:
            per.setdefault(int(step), []).append(float(v))
    return [[s, round(sum(v) / len(v), 1)] for s, v in sorted(per.items())]


def identity_series(face_scores: list[dict]) -> dict:
    """The mean face and body score per sampling step (the Monitor's Identity similarity chart), normalized when any
    sample carries a normalized score, else the raw percentages."""
    normalized = any(sc.get("combined_norm") is not None for sc in face_scores)
    pick = (lambda sc, n, r: sc.get(n)) if normalized else (lambda sc, n, r: sc.get(r))  # noqa: E731
    return {"normalized": normalized,
            "face": _mean_by_step([(sc["step"], pick(sc, "face_norm", "percent")) for sc in face_scores]),
            "body": _mean_by_step([(sc["step"], pick(sc, "body_norm", "body_percent")) for sc in face_scores])}


_RACE_IMAGE = re.compile(r"^(?P<name>.+?)_step(?P<step>\d{6})_(?P<rest>.+)\.png$")


def probe_racer(rec: dict, names) -> str:
    """The race configuration a probe record belongs to (its images carry the name as a prefix); "" for a probe
    outside the race (the winner training on after it)."""
    for p in rec.get("images") or []:
        m = _RACE_IMAGE.match(Path(p).name)
        if m and m["name"] in names:
            return m["name"]
    return ""


def probe_series(probes: list[dict], names=()) -> dict:
    """The smoothed face, body and combined score per probe (the Likeness probes chart), all probes together; with a
    race's configuration names also each configuration's own under "racers" (the winner's after the race:
    "winner")."""
    def pts(key: str, fallback: str | None = None) -> list[list]:
        out = []
        for pr in probes:
            v = pr.get(key)
            if v is None and fallback:
                v = pr.get(fallback)
            if v is not None:
                out.append([pr["step"], round(v, 1)])
        return out

    out = {"face": pts("face_norm_ema", "face_ema"), "body": pts("body_norm_ema", "body_ema"),
           "combined": pts("combined_ema")}
    if names:
        by: dict[str, list[dict]] = {}
        for pr in probes:
            by.setdefault(probe_racer(pr, names), []).append(pr)
        out["racers"] = {(n or "winner"): probe_series(by[n]) for n in [*names, ""] if n in by}
    return out


_PROBE_IMAGE = re.compile(r"step(?P<step>\d{6})_(?P<kind>face|body)_(?P<seed>\d+)\.png$")
PROBE_THUMB_PX = 768  # the copies' longest side (probes render at probe_size, 512 by default)


def _source(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def keep_probe_images(run_id: str, status: dict, root: Path = HISTORY_DIR) -> list[dict]:
    """Copy the run's probe images into root/<run id>/ (JPEG, at most PROBE_THUMB_PX; already copied ones are kept)
    and return their entries: {"step", "kind", "seed", "racer", "file" (relative to root), "face", "body",
    "combined"} with that probe's scores (normalized where the run had them)."""
    names = race_names(status)
    out = []
    for rec in status.get("probes") or []:
        racer = probe_racer(rec, names) if names else ""
        norm = rec.get("face_norm") is not None or rec.get("body_norm") is not None
        scores = {"face": rec.get("face_norm") if norm else rec.get("face"),
                  "body": rec.get("body_norm") if norm else rec.get("body"), "combined": rec.get("combined")}
        for img in rec.get("images") or []:
            m = _PROBE_IMAGE.search(Path(img).name)
            if not m:
                continue
            name = f"{racer + '_' if racer else ''}step{int(m['step']):06d}_{m['kind']}_{m['seed']}.jpg"
            dest = root / run_id / name
            if not dest.is_file():
                src = _source(img)
                if not src.is_file():
                    continue
                try:
                    from PIL import Image

                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with Image.open(src) as im:
                        im = im.convert("RGB")
                        im.thumbnail((PROBE_THUMB_PX, PROBE_THUMB_PX))
                        im.save(dest, quality=90)
                except OSError:
                    continue
            out.append({"step": int(rec.get("step", m["step"])), "kind": m["kind"], "seed": int(m["seed"]),
                        "racer": racer, "file": f"{run_id}/{name}", **scores})
    return out


def probe_raw(status: dict) -> dict:
    """The raw probe scores, enough to rebuild the likeness spreadsheet with any scale after the job folder is gone:
    {"names": the race's configurations, "records": [{"step", "face", "body", "racer" (a race), "scale" (the logged
    scale, where it changes)}]}. A probe logged twice (the step-0 probe) is kept once, the later copy."""
    names = race_names(status)
    out: list[dict] = []
    last_scale: dict[str, dict] = {}
    for rec in status.get("probes") or []:
        racer = probe_racer(rec, names) if names else ""
        e = {"step": rec.get("step"), "face": rec.get("face"), "body": rec.get("body")}
        if names:
            e["racer"] = racer
        scale = rec.get("scale")
        if out and out[-1]["step"] == e["step"] and out[-1].get("racer") == e.get("racer"):
            dropped = out.pop()
            if "scale" in dropped:
                last_scale.pop(racer, None)
        if scale and last_scale.get(racer) != scale:
            e["scale"] = scale
            last_scale[racer] = scale
        out.append(e)
    return {"names": names, "records": out}


def rescore(record: dict, face_span: float, face_curve: float, body_span: float, body_curve: float,
            face_weight: float, body_weight: float) -> dict | None:
    """The run's probe series (probe_series' shape) rebuilt from its raw scores on other scale settings: its own
    floors and references' levels, the given span / curve per kind and weights, its own smoothing. None for a run
    without raw scores (probe_raw)."""
    from .likeness import MIN_ROOM_SHARE, MIN_SPAN, combine, normalize
    from .likeness_export import _groups, _reference_levels

    raw = record.get("probe_raw") or {}
    records = raw.get("records") or []
    if not records:
        return None
    names = raw.get("names") or []
    a = float((record.get("config") or {}).get("probe_ema", 0.5))
    shape = {"face": (face_span, face_curve), "body": (body_span, body_curve)}

    def tops(rows: list[dict]) -> dict:
        out = {}
        for kind, (floor, ref) in _reference_levels(rows, None).items():
            least = max(MIN_SPAN[kind], MIN_ROOM_SHARE[kind] * (100.0 - floor))
            out[kind] = (floor, floor + (max(ref, floor + least) - floor) * shape[kind][0])
        return out

    def points(rows: list[dict]) -> list[tuple]:
        """(step, face, body, combined) per probe, smoothed within these rows."""
        scale, ema, out = tops(rows), {"face": None, "body": None}, []
        for r in rows:
            norm = {}
            for kind in ("face", "body"):
                v = r.get(kind)
                if v is not None:
                    ema[kind] = v if ema[kind] is None else a * ema[kind] + (1 - a) * v
                norm[kind] = normalize(ema[kind], *scale[kind], shape[kind][1])
            out.append((r["step"], norm["face"], norm["body"],
                        combine(norm["face"], norm["body"], face_weight, body_weight)))
        return out

    def series(pts: list[tuple]) -> dict:
        return {key: [[p[0], round(p[i], 1)] for p in pts if p[i] is not None]
                for i, key in ((1, "face"), (2, "body"), (3, "combined"))}

    if not names:
        return series(points(records))
    by_record: dict[int, tuple] = {}
    racers = {}
    for label, rows in _groups(records, names):
        pts = points(rows)  # each configuration smoothed on its own, as its controller did
        racers[label] = series(pts)
        by_record.update({id(r): p for r, p in zip(rows, pts)})
    out = series([by_record[id(r)] for r in records])  # all probes in order (the table's last / best)
    out["racers"] = racers
    return out


def race_names(status: dict) -> list[str]:
    return [str(n) for n in ((status.get("race") or {}).get("names") or [])]


def build_record(cfg: dict, arch: str, run_id: str, started: float, status: dict, state: str,
                 updated: float | None = None) -> dict:
    return {"version": 1, "id": run_id, "job": cfg.get("job_name", ""), "label": "", "arch": arch,
            "started": started, "updated": updated or time.time(), "state": state,
            "step": status.get("step") or 0, "total": status.get("total") or cfg.get("steps") or 0,
            "config": {k: cfg[k] for k in CONFIG_KEYS if k in cfg},
            "identity": identity_series(status.get("face_scores") or []),
            "probes": probe_series(status.get("probes") or [], race_names(status)),
            "probe_raw": probe_raw(status)}


def has_data(record: dict) -> bool:
    return any(record["identity"].get(k) for k in ("face", "body")) or any(
        record["probes"].get(k) for k in ("face", "body", "combined"))


def _write(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record), encoding="utf-8")
    tmp.replace(path)


def save_run(cfg, run_id: str, started: float, status: dict, state: str = "running",
             root: Path = HISTORY_DIR) -> Path | None:
    """Write (or update) this run's history file; nothing for a run that has no likeness data yet. A label the user
    gave the run is kept."""
    record = build_record(cfg.to_dict(), cfg.arch_name(), run_id, started, status, state)
    if not has_data(record):
        return None
    path = root / f"{run_id}.json"
    try:
        record["label"] = json.loads(path.read_text(encoding="utf-8")).get("label", "")
    except (OSError, ValueError):
        pass
    record["probe_images"] = keep_probe_images(run_id, status, root)
    _write(path, record)
    return path


def list_runs(root: Path = HISTORY_DIR) -> list[dict]:
    """Every saved run, newest first."""
    runs = []
    for p in sorted(root.glob("*.json")) if root.is_dir() else []:
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(rec, dict) and rec.get("id") and "identity" in rec and "probes" in rec:
                runs.append(rec)
        except (OSError, ValueError):
            continue
    return sorted(runs, key=lambda r: r.get("started") or 0, reverse=True)


def signature(root: Path = HISTORY_DIR) -> tuple:
    """Changes whenever a history file is added, changed or removed (the page's cheap poll)."""
    return tuple((p.name, p.stat().st_mtime_ns) for p in sorted(root.glob("*.json"))) if root.is_dir() else ()


def set_label(run_id: str, label: str, root: Path = HISTORY_DIR) -> bool:
    path = root / f"{run_id}.json"
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    rec["label"] = label.strip()
    _write(path, rec)
    return True


def delete_runs(run_ids: list[str], root: Path = HISTORY_DIR) -> int:
    n = 0
    for rid in run_ids:
        p = root / f"{rid}.json"
        if p.is_file() and p.parent == root:
            p.unlink()
            n += 1
        images = root / rid
        if images.is_dir() and images.parent == root:
            import shutil

            shutil.rmtree(images, ignore_errors=True)
    return n


def import_jobs(output: Path | None = None, root: Path = HISTORY_DIR) -> int:
    """History for the job folders (output/**/status.json) from before the history existed: one run per folder whose
    job has no history yet and whose status.json holds likeness scores. A race run saved before the history kept each
    configuration's probes apart gets them from its folder, and a run saved before the history kept probe images or
    raw probe scores (probe_raw, for its spreadsheet) gets them when its folder still holds that same run (the same
    probe scores). Returns how many runs were added or updated."""
    import yaml

    output = output or ROOT / "output"
    runs = list_runs(root)
    have = {r["job"] for r in runs}
    unsplit = {r["job"]: r for r in runs if (r.get("config") or {}).get("race") and "racers" not in r["probes"]}
    incomplete: dict[str, list[dict]] = {}  # without probe images or raw scores; a job name can have several runs
    for r in runs:
        if "probe_images" not in r or "probe_raw" not in r:
            incomplete.setdefault(r["job"], []).append(r)
    added = 0
    for status_path in sorted(output.rglob("status.json")) if output.is_dir() else []:
        job_dir = status_path.parent
        if any(part in SCAN_DIRS for part in job_dir.relative_to(output).parts):
            continue
        cfg_path = job_dir / "config.yaml"
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        except (OSError, ValueError, yaml.YAMLError):
            continue
        job = cfg.get("job_name") or job_dir.name
        changed = False
        series = probe_series(status.get("probes") or [], race_names(status))
        if job in unsplit and race_names(status):
            rec = unsplit.pop(job)
            rec["probes"] = series
            changed = True
        same = [r for r in incomplete.get(job, []) if series.get("combined")
                and r["probes"].get("combined") == series.get("combined")]
        if same:
            rec = same[0]  # the folder still holds this very run: its probe images and scores are its own
            incomplete[job].remove(rec)
            if "probe_images" not in rec:
                images = keep_probe_images(rec["id"], status, root)
                if images:
                    rec["probe_images"] = images
                    changed = True
            if "probe_raw" not in rec:
                raw = probe_raw(status)
                if raw["records"]:
                    rec["probe_raw"] = raw
                    changed = True
        if changed:
            _write(root / f"{rec['id']}.json", rec)
            added += 1
            continue
        if job in have:
            continue
        try:
            from .config import TrainConfig

            arch = TrainConfig.from_dict(cfg).arch_name()
        except Exception:  # noqa: BLE001 - an old config the current code cannot read
            arch = ""
        state = "error" if status.get("error") else "finished" if status.get("done") else "stopped"
        started, updated = cfg_path.stat().st_mtime, status_path.stat().st_mtime
        record = build_record({**cfg, "job_name": job}, arch, new_run_id(job, started) + "-imported", started, status,
                              state, updated)
        if has_data(record):
            record["probe_images"] = keep_probe_images(record["id"], status, root)
            _write(root / f"{record['id']}.json", record)
            have.add(job)
            added += 1
    return added
