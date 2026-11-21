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
(probe_images), for the Previous Runs tab's side-by-side comparison. The region sweeps are kept in the record
(sweeps: per step the training and held-out shares per region) and the training maps as a copy of the job's
training_maps/ files in output/_history/<run id>/maps/ (record "maps": true; copied when the run ends). Runs from before the history existed are imported
from their job folders' status.json."""
from __future__ import annotations

import contextlib
import json
import os
import re
import time
import uuid
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HISTORY_DIR = ROOT / "output" / "_history"
SCAN_DIRS = ("_history", "_crops")  # folders under output/ that are not jobs
CONFIG_KEYS = ("job_name", "preset", "network_type", "lora_rank", "lokr_factor", "oft_block_size", "dora",
               "attn_localize", "id_align", "learning_rate", "optimizer", "lr_scheduler", "lr_min_factor",
               "adam_beta2", "steps", "batch_size", "lora_ema", "weight_noise", "weight_noise_mode",
               "weight_noise_bound_norm",
               "face_weight", "body_weight", "face_loss_weight", "body_loss_weight", "crop_augment", "isolation",
               "finetune", "race", "timestep_sampling", "resolution_shift", "likeness_curriculum", "face_scale_span", "face_scale_curve",
               "body_scale_span", "body_scale_curve", "probe_face_prompt", "probe_body_prompt", "probe_seed",
               "probe_seeds", "probe_size", "probe_steps", "probe_cfg", "turbo_lora", "resolutions",
               "reward_steps", "body_align", "selftrain", "probe_ema", "body_score", "face_model", "lora_plus_ratio", "race_follow",
               "race_patience_rising", "race_pick")


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


def _source(path: str, base: Path | None = None) -> Path:
    """Where a probe image is: the trainer's folder for a local run; with base (a downloaded vast.ai run's folder)
    its probes/ folder, by file name, since the status.json paths are the instance's."""
    if base is not None:
        return base / "probes" / Path(path).name
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def keep_probe_images(run_id: str, status: dict, root: Path = HISTORY_DIR, base: Path | None = None) -> list[dict]:
    """Copy the run's probe images into root/<run id>/ (JPEG, at most PROBE_THUMB_PX; already copied ones are kept)
    and return their entries: {"step", "kind", "seed", "racer", "file" (relative to root), "face", "body",
    "combined"} with that probe's scores (normalized where the run had them). base: the folder a vast.ai run was
    downloaded to (its images are looked for in base/probes/)."""
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
                src = _source(img, base)
                if not src.is_file():
                    continue
                try:
                    from PIL import Image
                    from .pictures import to_rgb

                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with Image.open(src) as im:
                        im = to_rgb(im)
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
    from .likeness import ANCHORED_BODY_SCALE, MIN_ROOM_SHARE, MIN_SPAN, combine, normalize
    from .likeness_export import _groups, _reference_levels

    raw = record.get("probe_raw") or {}
    records = raw.get("records") or []
    if not records:
        return None
    names = raw.get("names") or []
    a = float((record.get("config") or {}).get("probe_ema", 0.5))
    anchored = (record.get("config") or {}).get("body_score") == "anchored"
    if anchored:  # the anchored body score's scale is fixed: 0 to the references' mean, linear
        body_span = body_curve = 1.0
    shape = {"face": (face_span, face_curve), "body": (body_span, body_curve)}

    def tops(rows: list[dict]) -> dict:
        out = {}
        for kind, (floor, ref) in _reference_levels(rows, None).items():
            least = max(MIN_SPAN[kind], MIN_ROOM_SHARE[kind] * (100.0 - floor))
            out[kind] = (floor, floor + (max(ref, floor + least) - floor) * shape[kind][0])
        if anchored:
            out["body"] = ANCHORED_BODY_SCALE
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


def sweep_series(status: dict, folder=None) -> list[dict]:
    """The run's region sweeps, compact: [{"step", "train": {region: share}, "held": {region: share}}] oldest first,
    from the status's region_sweep history, else (a status written without one) the folder's region_sweep.jsonl. A step
    seen twice keeps the later row; the shares (of the starting error that is gone) are rounded to 5 decimals."""
    configs = list((status.get("region_sweep_race") or {}).get("configs") or [])
    if folder is not None:
        races = []
        for path in Path(folder).glob("region_sweep.race*.jsonl"):
            match = re.fullmatch(r"region_sweep\.race(\d+)\.jsonl", path.name)
            if match:
                races.append((int(match.group(1)), path))
        for index, path in sorted(races):  # by the integer index: race10 after race2
            try:
                entry = next((c for c in configs if c.get("index") == index), None)
                if entry is None:
                    entry = {"index": index}
                    configs.append(entry)
                if not entry.get("history"):
                    with path.open(encoding="utf-8") as f:
                        parsed = []
                        for line in f:
                            try:
                                parsed.append(json.loads(line))
                            except ValueError:
                                continue
                    entry = dict(entry, history=parsed)
                    configs = [entry if c.get("index") == index else c for c in configs]
            except (OSError, ValueError):
                continue
    if configs:
        configs.sort(key=lambda c: c.get("index") if isinstance(c.get("index"), int) else 0)
        eligible = [c for c in configs if not c.get("dropped") and not c.get("failed")] or configs
        def last_step(c):
            steps = [r.get("step", 0) for r in c.get("history") or [] if isinstance(r, dict)
                     and isinstance(r.get("step"), (int, float))]
            return max(steps, default=c.get("step") or 0)
        chosen = max(eligible, key=last_step)
        rows = list(chosen.get("history") or ([chosen] if "step" in chosen else []))
    else:
        rows = list(((status.get("region_sweep") or {}).get("history")) or [])
    if not rows and folder is not None:
        try:
            with open(Path(folder) / "region_sweep.jsonl", encoding="utf-8") as f:
                for line in f:
                    try:
                        rows.append(json.loads(line))
                    except ValueError:
                        continue  # a line still being written
        except OSError:
            pass
    by: dict[int, dict] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            step = int(row["step"])
        except (KeyError, TypeError, ValueError):
            continue
        entry: dict = {"step": step}
        for key in ("train", "held"):
            vals = row.get(key)
            entry[key] = {str(r): round(float(v), 5) for r, v in (vals.items() if isinstance(vals, dict) else ())
                          if isinstance(v, (int, float)) and not isinstance(v, bool)}
        if entry["train"] or entry["held"]:
            by[step] = entry
    return [by[s] for s in sorted(by)]


MAPS_SUBDIR = "maps"  # root/<run id>/maps/: the kept copy of the job's training_maps/ folder
MAPS_DIRNAME = "training_maps"  # trainmaps.DIRNAME (not imported here)


def keep_maps(run_id: str, job_dir, root: Path = HISTORY_DIR) -> bool:
    """Copy the job's training_maps/ files (probes.json, maps.jsonl, probe_*.jpg) into root/<run id>/maps/; a file
    already there with the same size and modification time is left alone. True when the kept copy has maps."""
    import shutil

    src = Path(job_dir) / MAPS_DIRNAME if job_dir else None
    dest = root / run_id / MAPS_SUBDIR
    if src is not None and (src / "maps.jsonl").is_file():
        for p in [src / "probes.json", src / "maps.jsonl", *sorted(src.glob("probe_*.jpg"))]:
            try:
                if not p.is_file():
                    continue
                st, target = p.stat(), dest / p.name
                if target.is_file() and (target.stat().st_size, target.stat().st_mtime_ns) == (st.st_size,
                                                                                              st.st_mtime_ns):
                    continue
                dest.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, target)
            except OSError:
                continue
    return (dest / "maps.jsonl").is_file()


def build_record(cfg: dict, arch: str, run_id: str, started: float, status: dict, state: str,
                 updated: float | None = None, folder=...) -> dict:
    """folder: where the run's files are when that is not the config's output_dir/job_name (a vast.ai download), or None
    when they are not here; its region_sweep.jsonl stands in for a status without the sweep history."""
    job_dir = Path(cfg.get("output_dir") or "output") / cfg.get("job_name", "")
    return {"version": 1, "id": run_id, "job": cfg.get("job_name", ""), "job_dir": str(job_dir.resolve()),
            "label": "", "arch": arch,
            "started": started, "updated": updated or time.time(), "state": state,
            "step": status.get("step") or 0, "total": status.get("total") or cfg.get("steps") or 0,
            "config": {k: cfg[k] for k in CONFIG_KEYS if k in cfg},
            "identity": identity_series(status.get("face_scores") or []),
            "probes": probe_series(status.get("probes") or [], race_names(status)),
            "probe_raw": probe_raw(status),
            "sweeps": sweep_series(status, job_dir if folder is ... else folder)}


def has_data(record: dict) -> bool:
    return any(record["identity"].get(k) for k in ("face", "body")) or any(
        record["probes"].get(k) for k in ("face", "body", "combined"))


VAST_PREFIX = "vast-"
HIDDEN_VAST = "_hidden_vast.json"
FLEET_RUNS_DIR = Path(os.environ.get("RULZ_VAST_HOME") or Path.home() / ".rulz_vast") / "fleet" / "runs"
# the Fleet's run state -> the history's
VAST_STATES = {"done": "finished", "error": "error", "ended": "stopped", "running": "running"}
# the shape of an imported vast.ai record, kept in its stamp: raise it whenever import_fleet_runs starts keeping
# something new (2: sweeps and maps, 3.47.0), so the runs imported before that are imported once more. Without it a
# finished run's stamp never changes and its record never gets the new fields (3.47.0 left the runs imported under
# 3.46.0 without region sweeps on Previous Runs).
VAST_IMPORT_VERSION = 2

REPLACE_TRIES = 6  # on Windows a reader (the Previous Runs poll, a scan) can briefly hold the target open


def _replace(tmp: Path, path: Path) -> None:
    """os.replace, tried again for a moment while Windows refuses it (as atomic._replace, which this module does not
    import: atomic loads PyTorch, and nothing else here needs it)."""
    for attempt in range(REPLACE_TRIES):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == REPLACE_TRIES - 1:
                raise
            time.sleep(0.05 * (attempt + 1))


def _write(path: Path, record) -> None:
    """A history file written whole: into a temporary file beside it, which then takes its place. The trainer
    (save_run) and the WebUI (set_label) write the same file from two processes, so each write has a temporary name
    of its own: with one shared <id>.tmp a writer could move the other's half-written file into place, or find its
    own gone. The name does not end in .json, so list_runs and signature never read one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}-{uuid.uuid4().hex[:8]}.tmp")
    try:
        tmp.write_text(json.dumps(record), encoding="utf-8")
        _replace(tmp, path)
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)


def save_run(cfg, run_id: str, started: float, status: dict, state: str = "running",
             root: Path = HISTORY_DIR) -> Path | None:
    """Write (or update) this run's history file; nothing for a run that has no likeness data yet. A label the user
    gave the run is kept."""
    record = build_record(cfg.to_dict(), cfg.arch_name(), run_id, started, status, state)
    if not has_data(record):
        return None
    path = root / f"{run_id}.json"
    record["probe_images"] = keep_probe_images(run_id, status, root)
    # the maps are copied once the run is over (this runs often while it trains); a copy already kept stays marked
    if state != "running":
        keep_maps(run_id, record["job_dir"], root)
    if (root / run_id / MAPS_SUBDIR / "maps.jsonl").is_file():
        record["maps"] = True
    # the label is read last, right before the write: copying the pictures takes a while, and a label the WebUI set
    # meanwhile (set_label) was written over with the one read before it
    try:
        record["label"] = json.loads(path.read_text(encoding="utf-8")).get("label", "")
    except (OSError, ValueError):
        pass
    _write(path, record)
    return path


def list_runs(root: Path = HISTORY_DIR) -> list[dict]:
    """Every saved run, newest first."""
    runs = []
    for p in sorted(root.glob("*.json")) if root.is_dir() else []:
        if p.name.startswith("_"):  # _hidden_vast.json and the like: not a run
            continue
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(rec, dict) and rec.get("id") and "identity" in rec and "probes" in rec:
                runs.append(rec)
        except (OSError, ValueError):
            continue
    return sorted(runs, key=lambda r: r.get("started") or 0, reverse=True)


def signature(root: Path = HISTORY_DIR) -> tuple:
    """Changes whenever a history file is added, changed or removed (the page's cheap poll). With the size too: two
    writes can share a time stamp where the clock is coarse (Windows)."""
    if not root.is_dir():
        return ()
    return tuple((p.name, st.st_mtime_ns, st.st_size) for p, st in ((p, p.stat()) for p in sorted(root.glob("*.json"))
                                                  if not p.name.startswith("_")))


def set_label(run_id: str, label: str, root: Path = HISTORY_DIR) -> bool:
    path = root / f"{run_id}.json"
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    rec["label"] = label.strip()
    _write(path, rec)
    return True


def hidden_vast(root: Path = HISTORY_DIR) -> set[str]:
    """The vast.ai runs the user removed from the list (the import does not bring them back)."""
    try:
        data = json.loads((root / HIDDEN_VAST).read_text(encoding="utf-8"))
        return {str(x) for x in data} if isinstance(data, list) else set()
    except (OSError, ValueError):
        return set()


def delete_runs(run_ids: list[str], root: Path = HISTORY_DIR) -> int:
    """Remove runs from the history (the record and the probe thumbnails). A vast.ai run's id is also put on the
    hidden list, so the next import of the Fleet's records leaves it out; the Fleet's record and the downloaded files
    are never touched."""
    n = 0
    vast = [r for r in run_ids if str(r).startswith(VAST_PREFIX)]
    if vast:
        _write(root / HIDDEN_VAST, sorted(hidden_vast(root) | set(vast)))
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


JOB_IMAGE_DIRS = ("samples", "probes")  # a job folder's picture folders: thousands of files, no job folders inside


def status_files(output: Path) -> list[Path]:
    """Every status.json under output, sorted. Like output.rglob("status.json"), but it does not walk the folders
    that are not jobs (SCAN_DIRS) or, inside a job folder, its samples/ and probes/ pictures (one listing of those
    outweighs everything else on a long-lived output tree)."""
    found: list[Path] = []
    for here, dirs, files in os.walk(output):
        if "status.json" in files:
            found.append(Path(here) / "status.json")
            dirs[:] = [d for d in dirs if d not in JOB_IMAGE_DIRS]
        dirs[:] = [d for d in dirs if not (Path(here) == output and d in SCAN_DIRS)]
    return sorted(found)


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
        if "probe_images" not in r or "probe_raw" not in r or "sweeps" not in r or "maps" not in r:
            incomplete.setdefault(r["job"], []).append(r)
    added = 0
    for status_path in status_files(output) if output.is_dir() else []:
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
            if "sweeps" not in rec:
                rec["sweeps"] = sweep_series(status, job_dir)
                changed = True
            if "maps" not in rec and keep_maps(rec["id"], job_dir, root):
                rec["maps"] = True
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
                              state, updated, folder=job_dir)
        record["job_dir"] = str(job_dir.resolve())
        if has_data(record):
            record["probe_images"] = keep_probe_images(record["id"], status, root)
            if keep_maps(record["id"], job_dir, root):
                record["maps"] = True
            _write(root / f"{record['id']}.json", record)
            have.add(job)
            added += 1
    return added


def fleet_signature(fleet_dir: Path = FLEET_RUNS_DIR) -> tuple:
    """Changes whenever a Fleet run record is added, changed or removed (the page's cheap poll); () without a Fleet."""
    try:
        return tuple((p.name, st.st_mtime_ns, st.st_size)
                     for p, st in ((p, p.stat()) for p in sorted(fleet_dir.glob("*.json"))))
    except OSError:
        return ()


def _mtime(path: Path) -> int:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return 0


def import_fleet_runs(fleet_dir: Path = FLEET_RUNS_DIR, root: Path = HISTORY_DIR) -> int:
    """History for the training runs the Fleet tracked on vast.ai instances (one JSON per run in the Fleet's runs
    folder): id "vast-<key>", the likeness data from the run's downloaded status.json where there is one, else from
    the Fleet's own record; probe images from the download's probes/ folder. A run is rewritten only when its Fleet
    data changed or its record was imported by an older version (VAST_IMPORT_VERSION); a label the user gave it is kept; runs the user removed (delete_runs) are not brought back.
    Returns how many runs were added or updated."""
    if not fleet_dir.is_dir():
        return 0
    hidden = hidden_vast(root)
    added = 0
    for path in sorted(fleet_dir.glob("*.json")):
        try:
            fr = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(fr, dict) or not fr.get("key"):
            continue
        run_id = VAST_PREFIX + re.sub(r"[^A-Za-z0-9._-]+", "_", str(fr["key"]))
        if run_id in hidden:
            continue
        folder = Path(fr["folder"]) if fr.get("folder") else None
        if folder is not None and not folder.is_dir():
            folder = None
        status_path = folder / "status.json" if folder else None
        stamp = [VAST_IMPORT_VERSION, fr.get("modified"), fr.get("step"), fr.get("state"), fr.get("ended"),
                 _mtime(status_path) if status_path else 0, sorted(fr.get("downloaded") or [])]
        out_path = root / f"{run_id}.json"
        old: dict = {}
        try:
            old = json.loads(out_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        if not isinstance(old, dict):
            old = {}
        if (old.get("source") or {}).get("stamp") == stamp:
            continue
        status, cfg = None, None
        if status_path and status_path.is_file():
            try:
                st = json.loads(status_path.read_text(encoding="utf-8"))
                if isinstance(st, dict) and st.get("probes"):
                    status = st
            except (OSError, ValueError):
                pass
            if status is not None:
                try:
                    import yaml

                    cfg = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
                except Exception:  # noqa: BLE001 - missing or unreadable: the Fleet record's copy
                    cfg = None
        if status is None:
            status = {"probes": fr.get("probes") or [], "step": fr.get("step"), "total": fr.get("total")}
        if not isinstance(cfg, dict) or not cfg:
            cfg = fr.get("config") if isinstance(fr.get("config"), dict) else {}
        cfg = {**cfg, "job_name": cfg.get("job_name") or fr.get("job") or ""}
        try:
            from .config import TrainConfig

            arch = TrainConfig.from_dict(cfg).arch_name()
        except Exception:  # noqa: BLE001 - a config the current code cannot read
            arch = ""
        started = fr.get("started") or fr.get("first_seen") or 0
        record = build_record(cfg, arch, run_id, started, status, VAST_STATES.get(fr.get("state"), "stopped"),
                              fr.get("ended") or fr.get("modified") or started, folder=folder)
        if not has_data(record):
            continue
        record["job_dir"] = str(folder) if folder else ""
        record["source"] = {"kind": "vast", "instance": fr.get("instance"), "gpu": fr.get("gpu"), "key": fr["key"],
                            "stamp": stamp}
        record["probe_images"] = keep_probe_images(run_id, status, root, folder) if folder else []
        if folder and keep_maps(run_id, folder, root):
            record["maps"] = True
        record["label"] = old.get("label", "")
        try:
            _write(out_path, record)
        except OSError:
            continue
        added += 1
    return added
