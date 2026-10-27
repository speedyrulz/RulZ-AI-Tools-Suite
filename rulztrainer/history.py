"""Run history: what a training run's likeness scores looked like, kept after the job folder is reused.

Each run with likeness data writes one small JSON file in output/_history/ while it trains (so a crash keeps it) and
once more when it ends. The WebUI's Previous Runs tab lists them and overlays any number of them, like TensorBoard.
What is kept is what the Monitor's two likeness charts plot:

  * identity: the mean face and the mean body score of the sample images at each sampling step;
  * probes:   the smoothed face, body and combined score of the likeness probes at each probe step, and for a race
              each configuration's own (probes.racers, by name; "winner" for the probes after the race);

on the normalized scale (0 = a stranger, 100 = the references) where the run had one, else the raw percentages.
Runs from before the history existed are imported from their job folders' status.json."""
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
               "body_scale_span", "body_scale_curve",
               "reward_steps", "body_align", "selftrain")


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


def race_names(status: dict) -> list[str]:
    return [str(n) for n in ((status.get("race") or {}).get("names") or [])]


def build_record(cfg: dict, arch: str, run_id: str, started: float, status: dict, state: str,
                 updated: float | None = None) -> dict:
    return {"version": 1, "id": run_id, "job": cfg.get("job_name", ""), "label": "", "arch": arch,
            "started": started, "updated": updated or time.time(), "state": state,
            "step": status.get("step") or 0, "total": status.get("total") or cfg.get("steps") or 0,
            "config": {k: cfg[k] for k in CONFIG_KEYS if k in cfg},
            "identity": identity_series(status.get("face_scores") or []),
            "probes": probe_series(status.get("probes") or [], race_names(status))}


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
    return n


def import_jobs(output: Path | None = None, root: Path = HISTORY_DIR) -> int:
    """History for the job folders (output/**/status.json) from before the history existed: one run per folder whose
    job has no history yet and whose status.json holds likeness scores. A race run saved before the history kept each
    configuration's probes apart gets them from its folder. Returns how many runs were added or updated."""
    import yaml

    output = output or ROOT / "output"
    runs = list_runs(root)
    have = {r["job"] for r in runs}
    unsplit = {r["job"]: r for r in runs if (r.get("config") or {}).get("race") and "racers" not in r["probes"]}
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
        if job in unsplit and race_names(status):
            rec = unsplit.pop(job)
            rec["probes"] = probe_series(status.get("probes") or [], race_names(status))
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
            _write(root / f"{record['id']}.json", record)
            have.add(job)
            added += 1
    return added
