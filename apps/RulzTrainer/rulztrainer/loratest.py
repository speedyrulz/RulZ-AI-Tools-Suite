"""LoRA test: render finished LoRAs of one base model side by side, across strengths, prompts and seeds, and score every
picture on the trainer's likeness scale.

    python -m rulztrainer.loratest new --spec <file>      validate a spec, make output/_tests/<test_id>/, print the id
    python -m rulztrainer.loratest run <test_dir>         render, score and write results.json (the process owns the GPUs)
    python -m rulztrainer.loratest list                   one JSON line per test folder

The rules (spec, refusals, status.json, results.json) are in docs/lora-test-contract.md, sections 1 to 4. This module is
the run side: the WebUI tab, the exports and the vast.ai agent only read and write the files it describes.

Why it is built this way:
- Refusals come first and need no GPU: a spec is checked in full (files, base model, sizes) before anything loads, and a
  refused test says why in status.json, so the tab and the instance agent can show the reason.
- Heavy imports (torch, the Renderer, the scorers) happen inside the functions that need them, so `new`, `list` and the
  validation import fast on a machine without a GPU, and the tests run on fakes.
- Everything that touches a GPU or a model is reached through `Hooks` (the runner, the prompt cache, the soup, the
  scorer). The defaults are the real code; the tests replace them with fakes and run the whole `run` flow.
- One strength path: `lora.set_multiplier` on the Renderer's own slots (never the sampling LoRAs), which is exactly
  neutral at 1.0, so a strength 1.0 picture is the probes' picture. Two or more GPUs go through renderpool, whose
  workers take only a LoRA state (not a multiplier), so there the strength is baked into the state's scale (see
  payload_from_file).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
import re
import secrets
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from . import atomic

log = logging.getLogger("rulztrainer")

INSTALL_DIR = Path(__file__).resolve().parent.parent  # the folder holding webui.py
TESTS_SUBDIR = Path("output") / "_tests"
SPEC_VERSION = 1
RESULTS_VERSION = 1
MAX_CELLS = 2000
STRENGTH_RANGE = (0.0, 2.0)
STATUS_EVERY = 5.0  # seconds: status.json is rewritten at least this often while the run is going
FINAL_STATES = ("refused", "done", "stopped", "failed")
KINDS = ("face", "body", "both")
TEST_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
DEVICE_RE = re.compile(r"^cuda:\d+$")

# What two contestants must share so that one Renderer serves the whole test (contract section 2, "Refusals"):
# the base model, how it is loaded and the kind of LoRA.
BASE_KEYS = ("model_arch", "model_source", "model_path", "transformer_file", "transformer_quant", "model_file_quant",
             "sample_model_path", "sample_model_quant", "network_type", "dora")
# ... and the shape of the LoRA slots the Renderer injects once from the first contestant's config: a file of another
# rank or layer set would not load into them (lora.load_lora_weights raises on a rank mismatch), so that is refused
# up front as well. Not in the contract's list: reported to the orchestrator.
LAYOUT_KEYS = ("lora_targets", "lora_blocks", "lora_rank", "lokr_factor", "oft_block_size")
# the settings that decide which likeness scale a run's pictures were scored on (comfyui/rulztrainer_likeness/README.md)
SCALE_KEYS = (("face_model", "buffalo_l"), ("body_score", "appearance"), ("face_gold", "auto"))


class SpecError(ValueError):
    """A spec the run refuses; the message is the reason that goes into status.json (it names the file or field)."""


# ---------------------------------------------------------------- small helpers


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_test_id(now: datetime | None = None) -> str:
    """t + UTC YYYYMMDD-HHMMSS + - + 4 lowercase hex characters (t20261122-153012-a3f9)."""
    now = now or datetime.now(timezone.utc)
    return f"t{now.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(2)}"


def tests_root(base: Path | None = None) -> Path:
    return Path(base or INSTALL_DIR) / TESTS_SUBDIR


def write_json(path: Path, data) -> None:
    """The file whole or not at all (atomic.py); readers poll status.json while it is being rewritten."""
    with atomic.writing(path) as tmp:
        Path(tmp).write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8", newline="\n")


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


PID_CREATED_TOLERANCE = 1.0  # seconds between a recorded process start and psutil's create_time
QUEUED_START_GAP = 120.0  # seconds a queued status without a pid may be old before it counts as failed


def pid_alive(pid, created=None) -> bool:
    """Whether the process `pid` runs. With `created` (its start time as psutil's create_time, which the vast agent
    records as pid_created) a process that has the pid but started at another time is a reused pid: not the run.
    Without psutil on Windows it cannot be told (os.kill(pid, 0) would terminate the process there), so it counts as
    alive."""
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        import psutil

        if not psutil.pid_exists(pid):
            return False
        if isinstance(created, (int, float)) and not isinstance(created, bool):
            try:
                return abs(psutil.Process(pid).create_time() - float(created)) <= PID_CREATED_TOLERANCE
            except psutil.Error:
                return False
        return True
    except ImportError:
        if sys.platform == "win32":
            return True
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except OSError:
            return True
        return True


def effective_state(status: dict, age: float | None = None) -> str:
    """The state readers show: a pid that no longer runs (or is another process that reused it: pid_created) while the
    state is not final means the run died (failed). A queued status with no pid is the gap between the folder being
    made and the process starting: it is failed only once status.json is older than QUEUED_START_GAP seconds (`age`,
    unknown: still queued)."""
    state = status.get("state") or "queued"
    if state in FINAL_STATES:
        return state
    if state == "queued" and status.get("pid") is None:
        return "failed" if age is not None and age > QUEUED_START_GAP else state
    if not pid_alive(status.get("pid"), status.get("pid_created")):
        return "failed"
    return state


def resolve_path(base: Path, text, what: str) -> Path:
    """A path of the spec: relative to the install folder, or absolute. A relative one must not climb out (`..`)."""
    if not isinstance(text, str) or not text.strip():
        raise SpecError(f"{what}: a path is needed")
    p = Path(text)
    if p.is_absolute():
        return p
    if ".." in p.parts:
        raise SpecError(f"{what}: {text!r} climbs out of the install folder (..)")
    return Path(base) / p


def read_config(path: Path) -> dict:
    """A run's config.yaml as a plain dict (no torch)."""
    import yaml

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError) as e:
        raise SpecError(f"{path}: cannot read the config ({e})") from e
    if not isinstance(data, dict):
        raise SpecError(f"{path}: the config is not a YAML mapping")
    return data


def _cfg_get(cfg: dict, key: str):
    """A config's value, the TrainConfig default when the file does not have it (imported only then)."""
    if key in cfg and cfg[key] is not None:
        return cfg[key]
    try:
        import dataclasses

        from .config import TrainConfig

        for f in dataclasses.fields(TrainConfig):
            if f.name == key:
                if f.default is not dataclasses.MISSING:
                    return f.default
                if f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
                    return f.default_factory()  # type: ignore[misc]
    except Exception:  # noqa: BLE001 - a default only completes the picture
        pass
    return None


def _same(cfgs: list[tuple[str, dict]], keys, what: str) -> None:
    """Raise SpecError when the configs (id, dict) differ in any of the keys."""
    first_id, first = cfgs[0]
    for key in keys:
        want = json.dumps(_cfg_get(first, key), sort_keys=True, default=str)
        for cid, cfg in cfgs[1:]:
            got = json.dumps(_cfg_get(cfg, key), sort_keys=True, default=str)
            if got != want:
                raise SpecError(f"{what}: {first_id} and {cid} differ in {key} "
                                f"({_cfg_get(first, key)!r} against {_cfg_get(cfg, key)!r})")


# ---------------------------------------------------------------- the spec


@dataclass
class Plan:
    """A validated spec with its paths resolved: what the run works from."""
    spec: dict
    base: Path
    files: list[dict]  # contestants: {id, label, lora, config (text as in the spec), lora_path, config_path, cfg}
    soups: list[dict]  # {id, label, of, weights (normalized), out}
    prompts: list[str]
    kinds: list[str]
    seeds: list[int]
    strengths: list[float]
    devices: list[str]
    from_id: str
    total: int

    @property
    def contestants(self) -> list[dict]:
        """Every contestant of the grid in order: the files, then the soups."""
        return self.files + self.soups

    def cfg_of(self, cid: str) -> dict:
        for c in self.files:
            if c["id"] == cid:
                return c["cfg"]
        for s in self.soups:  # a soup's base model is its ingredients'
            if s["id"] == cid:
                return self.cfg_of(s["of"][0])
        raise KeyError(cid)


def strength_name(s: float) -> str:
    return f"{s:.2f}"


def image_rel(cid: str, strength: float, pi: int, seed: int) -> str:
    """images/<contestant_id>/s<strength>_p<prompt_index>_seed<seed>.png, strength with two decimals."""
    return f"images/{cid}/s{strength_name(strength)}_p{pi}_seed{seed}.png"


def validate(spec: dict, base: Path | None = None, folder_name: str | None = None,
             config_reader: Callable[[Path], dict] | None = None) -> Plan:
    """Check a spec in full without touching a GPU; raises SpecError (the reason) or returns the Plan.

    folder_name: the test folder's name, which must be the spec's test_id (run); new passes None."""
    base = Path(base or INSTALL_DIR)
    config_reader = config_reader or read_config
    if not isinstance(spec, dict):
        raise SpecError("spec.json: not a JSON object")
    tid = spec.get("test_id")
    if not isinstance(tid, str) or not TEST_ID_RE.match(tid):
        raise SpecError(f"test_id: {tid!r} is not a valid id (only a-z, 0-9 and -)")
    if folder_name is not None and folder_name != tid:
        raise SpecError(f"test_id: {tid!r} is not the test folder's name {folder_name!r}")

    for key in ("scoring", "render"):
        if spec.get(key) is not None and not isinstance(spec[key], dict):
            raise SpecError(f"{key}: must be an object")
    if spec.get("soups") is not None and not isinstance(spec["soups"], list):
        raise SpecError("soups: must be a list")
    for key in ("face_refs", "body_refs"):
        v = (spec.get("scoring") or {}).get(key)
        if v is not None and (not isinstance(v, list) or not all(isinstance(x, str) for x in v)):
            raise SpecError(f"scoring.{key}: must be null or a list of paths")
    prompts = spec.get("prompts")
    if not isinstance(prompts, list) or not prompts or not all(isinstance(p, str) and p.strip() for p in prompts):
        raise SpecError("prompts: a non-empty list of prompts is needed")
    seeds = spec.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        raise SpecError("seeds: a non-empty list of seeds is needed")
    if not all(isinstance(s, int) and not isinstance(s, bool) for s in seeds):
        raise SpecError("seeds: every seed must be a whole number")
    if len(set(seeds)) != len(seeds):
        raise SpecError("seeds: a seed is listed twice (the pictures would overwrite each other)")
    strengths = spec.get("strengths")
    if not isinstance(strengths, list) or not strengths:
        raise SpecError("strengths: a non-empty list of strengths is needed")
    for s in strengths:
        if isinstance(s, bool) or not isinstance(s, (int, float)):
            raise SpecError(f"strengths: {s!r} is not a number")
        if not STRENGTH_RANGE[0] <= s <= STRENGTH_RANGE[1]:
            raise SpecError(f"strengths: {s} is outside {STRENGTH_RANGE[0]:g} to {STRENGTH_RANGE[1]:g}")
    strengths = [float(s) for s in strengths]
    if len({strength_name(s) for s in strengths}) != len(strengths):
        raise SpecError("strengths: two strengths are the same at two decimals (the pictures would overwrite each other)")
    kinds = spec.get("prompt_kinds")
    if kinds is None:
        kinds = ["both"] * len(prompts)
    if not isinstance(kinds, list) or len(kinds) != len(prompts):
        raise SpecError("prompt_kinds: needs one entry per prompt")
    if any(k not in KINDS for k in kinds):
        raise SpecError(f"prompt_kinds: each must be one of {', '.join(KINDS)}")

    devices = spec.get("devices") or ["cuda:0"]
    if not isinstance(devices, list) or not all(isinstance(d, str) and DEVICE_RE.match(d) for d in devices):
        raise SpecError("devices: a list of cuda:N names is needed")
    if len(set(devices)) != len(devices):
        raise SpecError("devices: a device is listed twice")

    raw = spec.get("contestants")
    if not isinstance(raw, list) or not raw:
        raise SpecError("contestants: at least one contestant is needed")
    files: list[dict] = []
    seen: set[str] = set()
    for i, c in enumerate(raw):
        if not isinstance(c, dict):
            raise SpecError(f"contestants[{i}]: not an object")
        cid = c.get("id")
        if not isinstance(cid, str) or not re.fullmatch(r"[a-z0-9]+", cid or ""):
            raise SpecError(f"contestants[{i}].id: {cid!r} is not a valid id")
        if cid in seen:
            raise SpecError(f"contestants[{i}].id: {cid!r} is used twice")
        seen.add(cid)
        lora = resolve_path(base, c.get("lora"), f"contestants[{i}].lora ({cid})")
        if not lora.is_file():
            raise SpecError(f"contestants[{i}].lora: the LoRA file {c.get('lora')} of {cid} does not exist")
        cpath = resolve_path(base, c.get("config"), f"contestants[{i}].config ({cid})")
        if not cpath.is_file():
            raise SpecError(f"contestants[{i}].config: the config file {c.get('config')} of {cid} does not exist")
        files.append({"id": cid, "label": str(c.get("label") or cid), "lora": c["lora"], "config": c["config"],
                      "lora_path": lora, "config_path": cpath, "cfg": config_reader(cpath),
                      "run_id": c.get("run_id")})

    soups: list[dict] = []
    by_id = {f["id"]: f for f in files}
    for i, s in enumerate(spec.get("soups") or []):
        sid = s.get("id") if isinstance(s, dict) else None
        if not isinstance(sid, str) or not re.fullmatch(r"[a-z0-9]+", sid or ""):
            raise SpecError(f"soups[{i}].id: {sid!r} is not a valid id")
        if sid in seen:
            raise SpecError(f"soups[{i}].id: {sid!r} is used twice")
        seen.add(sid)
        of = s.get("of")
        if not isinstance(of, list) or len(of) < 2:
            raise SpecError(f"soups[{i}].of: a soup ({sid}) needs two or more contestants")
        for o in of:
            if o not in by_id:  # a soup of soups, or an id that is not a contestant
                raise SpecError(f"soups[{i}].of: {o!r} is not a file contestant (a soup is made of files only)")
        if len(set(of)) != len(of):
            raise SpecError(f"soups[{i}].of: {sid} lists a contestant twice")
        _same([(o, by_id[o]["cfg"]) for o in of], BASE_KEYS + LAYOUT_KEYS, f"soups[{i}] ({sid}) mixes base models")
        w = s.get("weights")
        if w is None:
            weights = [1.0 / len(of)] * len(of)
        else:
            if (not isinstance(w, list) or len(w) != len(of) or any(isinstance(x, bool) or not isinstance(x, (int, float))
                                                                  or x < 0 for x in w) or sum(w) <= 0):
                raise SpecError(f"soups[{i}].weights: {sid} needs null or one number (0 or more, not all 0) per entry of of")
            weights = [float(x) / sum(w) for x in w]
        soups.append({"id": sid, "label": str(s.get("label") or sid), "of": list(of), "weights": weights,
                      "out": f"soups/{sid}.safetensors"})

    for f in files:  # video models render clips: the test grid is of still pictures
        frames = _cfg_get(f["cfg"], "probe_frames")
        if isinstance(frames, (int, float)) and frames > 1:
            raise SpecError(f"{f['config']}: probe_frames is {frames}, so its probes render clips; LoRA tests "
                            "render single pictures only (set probe_frames to 1 in a copy of the config)")
    _same([(f["id"], f["cfg"]) for f in files], BASE_KEYS, "contestants have different base models")
    _same([(f["id"], f["cfg"]) for f in files], LAYOUT_KEYS, "contestants have different LoRA layouts")

    if len(devices) >= 2:  # renderpool's workers take a state, not a multiplier: strength is baked into LoRA / LoKr
        for f in files:
            if str(_cfg_get(f["cfg"], "network_type")) not in ("lora", "lokr") or _cfg_get(f["cfg"], "dora"):
                raise SpecError(f"devices: two or more GPUs can apply a strength only to plain LoRA and LoKr files "
                                f"(OFT and DoRA need one device); {f['id']} is not one")

    def file_kind(path: Path) -> str:
        """"kohya" (lora_unet_ keys), "oft_dora", or "" (comfy / PEFT LoRA or LoKr): from the key names only."""
        from safetensors import safe_open

        try:
            with safe_open(str(path), framework="pt") as f:
                keys = list(f.keys())
        except Exception:  # noqa: BLE001 - not readable here: the run reports it when it loads
            return ""
        if any(k.endswith((".oft_blocks", ".dora_scale")) for k in keys):
            return "oft_dora"
        return "kohya" if any(k.startswith("lora_unet_") for k in keys) else ""

    for sp in soups:
        for o in sp["of"]:
            kind = file_kind(by_id[o]["lora_path"])
            if kind:
                raise SpecError(f"soup {sp['id']}: {by_id[o]['lora']} is a {'kohya' if kind == 'kohya' else 'OFT / DoRA'}"
                                " file, which cannot be averaged from the file alone; soup comfy or PEFT LoRA / LoKr files")
    if len(devices) >= 2:
        for f in files:
            if file_kind(f["lora_path"]) == "kohya":
                raise SpecError(f"devices: {f['lora']} is a kohya file, which two or more GPUs cannot load; use one device")

    total = (len(files) + len(soups)) * len(strengths) * len(prompts) * len(seeds)
    if total > MAX_CELLS:
        raise SpecError(f"{total} pictures is more than the {MAX_CELLS} a test may render")

    from_id = (spec.get("scoring") or {}).get("from_contestant") or files[0]["id"]
    if from_id not in by_id:
        raise SpecError(f"scoring.from_contestant: {from_id!r} is not a file contestant")
    return Plan(spec=spec, base=base, files=files, soups=soups, prompts=list(prompts), kinds=list(kinds),
                seeds=list(seeds), strengths=strengths, devices=list(devices), from_id=from_id, total=total)


def render_settings(plan: Plan) -> dict:
    """width, height, steps, cfg, negative: the spec's `render` fields, else the first contestant's sample settings."""
    cfg = plan.files[0]["cfg"]
    given = plan.spec.get("render") or {}
    out = {}
    for key, cfg_key in (("width", "sample_width"), ("height", "sample_height"), ("steps", "sample_steps"),
                         ("cfg", "sample_cfg"), ("negative", "sample_negative")):
        v = given.get(key)
        out[key] = v if v is not None else _cfg_get(cfg, cfg_key)
    return out


# ---------------------------------------------------------------- the scale and the references


def resolve_refs(plan: Plan) -> tuple[list[str], list[str]]:
    """(face reference paths, body reference paths): the spec's lists, else from_contestant's config (relative paths
    are the install folder's). Both empty: the test still renders and the scores are null."""
    scoring = plan.spec.get("scoring") or {}
    cfg = next(f["cfg"] for f in plan.files if f["id"] == plan.from_id)

    def paths(given, key):
        items = given if given is not None else (cfg.get(key) or [])
        return [str(resolve_path(plan.base, p, f"scoring.{key}")) for p in items]

    return paths(scoring.get("face_refs"), "face_refs"), paths(scoring.get("body_refs"), "body_refs")


def recorded_scale_problems(plan: Plan, first: dict) -> list[str]:
    """Why from_contestant's likeness.jsonl cannot be this test's scale (empty: it can). Never raises: a scale that
    cannot be read only means the references' own scale, never a failed test after the rendering."""
    try:
        job = _load_core(plan.base).job_scale(first["config_path"].parent)
    except Exception as e:  # noqa: BLE001 - missing file, no scale, no scoring code
        return [f"{first['id']}'s run has no usable likeness.jsonl ({str(e).splitlines()[0][:120] if str(e) else type(e).__name__})"]
    out = []
    cfg = first["cfg"]
    if str(job.get("face_model") or "buffalo_l") != str(_cfg_get(cfg, "face_model") or "buffalo_l"):
        out.append(f"the run recorded face_model {job.get('face_model')}, its config says {_cfg_get(cfg, 'face_model')}")
    if job.get("body_score") and str(job["body_score"]) != str(_cfg_get(cfg, "body_score") or "appearance"):
        out.append(f"the run recorded body_score {job['body_score']}, its config says {_cfg_get(cfg, 'body_score')}")
    gold = str(_cfg_get(cfg, "face_gold") or "auto")
    if gold not in ("auto", "off", "none", "") and gold != str(job.get("gold_name") or "facenet"):
        out.append(f"the run recorded the {job.get('gold_name')} gold face, its config says {gold}")
    return out


def decide_scale(plan: Plan) -> tuple[str, str | None]:
    """("run" | "references", note). "run": every contestant's run shares face_model, body_score and the gold face, the
    references come from from_contestant's config and that run's likeness.jsonl is there to read the scale from. Else
    the references' own scale, and the note says why (scores on the two scales are not comparable)."""
    scoring = plan.spec.get("scoring") or {}
    why = []
    first = next(f for f in plan.files if f["id"] == plan.from_id)
    for key, default in SCALE_KEYS:
        values = {f["id"]: str(_cfg_get(f["cfg"], key) or default) for f in plan.files}
        if len(set(values.values())) > 1:
            why.append(f"the runs differ in {key} ({', '.join(f'{i}: {v}' for i, v in values.items())})")
    if scoring.get("face_refs") is not None or scoring.get("body_refs") is not None:
        why.append("the references were given in the test, not taken from the run's config")
    if not why:  # the recorded scale must be readable and be the scale this scorer will use
        why += recorded_scale_problems(plan, first)
    if why:
        return "references", "Scored on the references' own scale: " + "; ".join(why) + "."
    return "run", None


# ---------------------------------------------------------------- status.json


class StatusWriter:
    """status.json, rewritten atomically after every change and by a heartbeat thread at least every STATUS_EVERY
    seconds while the run is going (a long load, one slow picture or the scoring would otherwise leave it silent, and a
    reader cannot tell a slow run from a dead one except by `updated`)."""

    def __init__(self, test_dir: Path, test_id: str, every: float = STATUS_EVERY, clock=time.time):
        self.path = Path(test_dir) / "status.json"
        self.every, self.clock = every, clock
        self.lock = threading.Lock()
        self.data = {"state": "queued", "test_id": test_id, "pid": os.getpid(), "started": utc_now(),
                     "updated": utc_now(), "done": 0, "total": 0, "current": None, "eta_s": None, "reason": None,
                     "error": None}
        try:  # keys another writer put there (the agent's pid_created, anything unknown) stay
            old = read_json(self.path)
            if isinstance(old, dict):
                self.data.update({k: v for k, v in old.items() if k not in self.data})
        except (OSError, ValueError):
            old = None
        # this process's start time next to its pid, so a reused pid is told from the run after a crash; the launcher's
        # own record (the vast agent's) stays when it is for this same pid
        if not (isinstance(old, dict) and old.get("pid") == self.data["pid"] and old.get("pid_created") is not None):
            try:
                import psutil

                self.data["pid_created"] = psutil.Process(os.getpid()).create_time()
            except Exception:  # noqa: BLE001 - no psutil: no start time
                self.data.pop("pid_created", None)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._pictures: list[float] = []  # seconds each picture took

    def update(self, **kw) -> None:
        with self.lock:
            if kw.get("pid", 0) is None and self.data.get("pid") is not None:  # a pid is never written back to null
                del kw["pid"]
            self.data.update(kw)
            self._write()

    def _write(self) -> None:
        self.data["updated"] = utc_now()
        try:
            write_json(self.path, self.data)
        except OSError:  # a poll holding the file on Windows: the next write (at most `every` later) lands
            log.debug("status.json not written", exc_info=True)

    def picture_done(self, seconds: float, current: str | None) -> None:
        """One more picture: done + 1, and the ETA from the mean seconds per picture so far."""
        with self.lock:
            self._pictures.append(seconds)
            left = max(0, self.data["total"] - self.data["done"] - 1)
            mean = sum(self._pictures) / len(self._pictures)
            self.data.update(done=self.data["done"] + 1, current=current, eta_s=int(round(mean * left)))
            self._write()

    def start(self) -> None:
        def beat():
            while not self._stop.wait(self.every):
                with self.lock:
                    self._write()

        self._thread = threading.Thread(target=beat, daemon=True, name="loratest-status")
        self._thread.start()

    def finish(self, state: str, **kw) -> None:
        """The final state; stops the heartbeat first so it cannot rewrite an older state over it."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self.update(state=state, eta_s=None, **kw)


# ---------------------------------------------------------------- the real renderers (heavy imports inside)


def _dump(state: dict) -> bytes:
    import torch

    buf = io.BytesIO()
    torch.save(state, buf)
    return buf.getvalue()


def blank_slots(injected: dict) -> None:
    """Zero the LoRA / LoKr factors (and set OFT blocks to the identity) before a file is loaded, so a layer the file
    does not have contributes nothing: lora.load_lora_weights skips layers missing from a file and would otherwise leave
    the previous contestant's weights there. DoRA's magnitude is left (a zero magnitude would silence the layer)."""
    import torch

    with torch.no_grad():
        for m in injected.values():
            for part, p in m.adapter_params().items():
                if part not in ("dora_scale",):
                    p.zero_()


def payload_from_file(path: Path, cfg: dict, strength: float) -> bytes:
    """The bytes renderpool's workers take (what Renderer.render loads: torch.save of lora.lora_state) built from a
    LoRA or LoKr file without a model: the layers soup.read_ingredient reads, with `strength` folded into each layer's
    scale (a LoRA / LoKr change is linear in its scale, so this is exactly what lora.set_multiplier does to it). Used
    for two or more GPUs, where there is no model in this process to load the file into. OFT / DoRA files are refused
    (soup.read_ingredient raises); a layer the file lacks keeps whatever the worker held, so contestants of one test
    must cover the same layers (they share lora_targets and lora_blocks)."""
    from . import soup

    entries = soup.read_ingredient(path)
    factor = int(_cfg_get(cfg, "lokr_factor") or 8)
    block = int(_cfg_get(cfg, "oft_block_size") or 32)
    rank = int(_cfg_get(cfg, "lora_rank") or 1)
    state = {}
    for name, (kind, first, second, scale) in entries.items():
        if kind == "lokr":
            params, r = {"lokr_w1": first.float().contiguous(), "lokr_w2": second.float().contiguous()}, rank
        else:
            params, r = {"lora_down": first.float().contiguous(), "lora_up": second.float().contiguous()}, first.shape[0]
        state[name] = {"kind": kind, "dora": False, "block": block, "params": params, "alpha": float(scale) * r,
                       "scale": float(scale) * float(strength), "rank": int(r), "factor": factor, "constraint": 0.0,
                       "trainable": True, "dropout": 0.0}
    return _dump(state)


class LocalRunner:
    """One Renderer on one GPU. Each contestant's file becomes the state Renderer.render mirrors into its own `injected`
    slots (see use: built from the file for comfy / PEFT LoRA, LoKr and soups, so a rank change is resized by
    mirror_lora; lora.load_lora_weights for kohya, OFT and DoRA files). The strength is lora.set_multiplier on those
    slots before each render: mirror_lora copies weights, alpha and scale only, never `multiplier`, and the sampling
    LoRAs are separate extras, so neither is touched."""

    def __init__(self, cfg: dict, device: str, test_dir: Path):
        from . import renderhelper

        self.renderer, info = renderhelper.load_renderer(trainer_cfg_dict(cfg), device)
        self.gpus = [info["gpu"]]
        self.payload = b""

    def use(self, path: Path, cfg: dict) -> None:
        """Plain LoRA / LoKr files and soups (comfy / PEFT layouts) become a state built from the file
        (payload_from_file) that Renderer.render's mirror_lora copies into the slots, resizing a layer whose rank
        differs: correct in any contestant order. Layers the file lacks get zeroed copies of the slots' own entries, so
        nothing of the previous contestant stays. A file that route cannot read or that maps to no slot (kohya SDXL
        keys, OFT, DoRA) goes through lora.load_lora_weights, after the slots are put back to the config's rank."""
        import io

        import torch

        from . import lora

        inj = self.renderer.injected
        if getattr(self, "pristine", None) is None:  # the slots as the Renderer built them: config rank, no weights
            self.pristine = lora.lora_state(inj)
            for entry in self.pristine.values():
                entry["params"] = {k: torch.zeros_like(v) for k, v in entry["params"].items()}
        state = None
        if str(_cfg_get(cfg, "network_type")) in ("lora", "lokr") and not _cfg_get(cfg, "dora"):
            try:
                state = torch.load(io.BytesIO(payload_from_file(path, cfg, 1.0)), weights_only=True)
            except ValueError:  # OFT / DoRA keys or no layers: not readable without a model
                state = None
            state = {n: e for n, e in (state or {}).items() if n in inj} or None  # kohya names map to no slot
        if state is not None:
            for name, entry in lora.lora_state(inj).items():
                if name not in state:
                    entry["params"] = {k: torch.zeros_like(v) for k, v in entry["params"].items()}
                    state[name] = entry
            self.payload = _dump(state)
            return
        # load_lora_weights raises on a rank mismatch and a soup may have resized the slots: back to the config's rank
        lora.mirror_lora(lora.lora_snapshots(self.pristine), inj)
        blank_slots(inj)
        if lora.load_lora_weights(inj, path) == 0:
            raise ValueError(f"{path}: none of its layers fit this model (wrong base model or layer names?)")
        self.payload = _dump(lora.lora_state(inj))

    def render(self, jobs: list, strength: float) -> list[str]:
        from . import lora

        lora.set_multiplier(self.renderer.injected, strength)
        images, _ = self.renderer.render(jobs, self.payload)
        return images

    def close(self) -> None:
        from . import renderhelper

        self.renderer = None
        renderhelper.release_gpu()


class PoolRunner:
    """renderpool.RenderPool over two or more GPUs: one worker process per GPU, a round's pictures dealt across them
    (the probes' fixed map). The pool's workers take only a LoRA state, so the payload is built from the file with the
    strength folded in (payload_from_file). A column (one contestant at one strength) is one round, so a STOP waits for
    the round in hand rather than a single picture."""

    def __init__(self, cfg: dict, devices: list[str], test_dir: Path):
        from . import renderpool

        cfg = trainer_cfg_dict(cfg)
        self.pool = renderpool.RenderPool(devices, Path(test_dir) / "pool_logs", "loratest", threading.Event())
        Path(test_dir, "pool_logs").mkdir(parents=True, exist_ok=True)
        if not self.pool.load_all(cfg, renderpool.resolve_model_dir(cfg, devices[0])):
            raise RuntimeError(f"no GPU loaded the model: {self.pool.note()}")
        self.gpus = [s.name or s.device for s in self.pool.live]
        self.path: Path | None = None
        self.cfg: dict = {}

    def use(self, path: Path, cfg: dict) -> None:
        self.path, self.cfg = path, cfg

    def render(self, jobs: list, strength: float) -> list[str]:
        self.pool.run_round(jobs, payload_from_file(self.path, self.cfg, strength), "probe")
        return [str(j.out_path) for j in jobs]

    def close(self) -> None:
        self.pool.close()


def trainer_cfg_dict(cfg: dict) -> dict:
    """A run's config as TrainConfig.from_dict takes it: keys this version no longer has are dropped (a config saved
    by an older version), missing ones take their defaults."""
    import dataclasses

    from .config import TrainConfig

    known = {f.name for f in dataclasses.fields(TrainConfig)}
    return {k: v for k, v in cfg.items() if k in known}


def make_runner(cfg: dict, devices: list[str], test_dir: Path):
    return PoolRunner(cfg, devices, test_dir) if len(devices) > 1 else LocalRunner(cfg, devices[0], test_dir)


def make_job(**fields):
    from .trainer import RenderJob

    return RenderJob(**fields)


def prepare_prompts(cfg: dict, prompts: list[str], device: str) -> None:
    """Cache the prompts' text embeds under the first contestant's config, as the trainer does before probes: the
    Renderer has no text encoder and reads each prompt (and the negative) from the cache. A prompt already there costs
    nothing; otherwise the text encoder is loaded, the missing ones are encoded and it is unloaded again before the
    sampling model loads."""
    from . import renderhelper
    from .archs import get_arch
    from .config import TrainConfig
    from .dataset import cache_text, text_cache_path
    from .trainer import free_cuda, unload_model

    tc = TrainConfig.from_dict({**trainer_cfg_dict(cfg), "device": device})
    arch = get_arch(tc)
    # the negative Renderer.render asks for (Trainer.negative_prompt('probe') for stills): the config's, else the arch's
    wanted = list(prompts) + [tc.sample_negative or arch.default_negative]
    if not any(not text_cache_path(tc, p).exists() for p in wanted):
        return
    model_dir = arch.resolve_for_config(tc)
    te, tok = arch.get_text_encoder(tc, model_dir)
    cache_text([], tc, te, tok, extra_prompts=wanted, encode_batch_fn=arch.encode_prompts)
    unload_model(te)
    del te, tok
    free_cuda()
    renderhelper.release_gpu()


def cook_soup(paths: list[Path], out: Path, weights: list[float], cfg: dict, device: str = "cpu") -> dict:
    """soup.make_soup on the CPU: exact (rank 0) for LoRA files, the config's rank for LoKr (not low-rank)."""
    from . import soup

    kind = str(_cfg_get(cfg, "network_type"))
    rank = 0 if kind == "lora" else int(_cfg_get(cfg, "lora_rank") or 32)
    out.parent.mkdir(parents=True, exist_ok=True)
    return soup.make_soup([str(p) for p in paths], out, rank=rank, weights=weights, device=device)


# ---------------------------------------------------------------- the real scorer


def _load_core(base: Path):
    """comfyui/rulztrainer_likeness/core.py by path: the nodes' own scoring code (references, the scale, per-image
    scores) with no ComfyUI imports; importing the package would run its __init__ and need ComfyUI's modules."""
    import importlib.util

    path = next((q for q in (Path(base) / "comfyui" / "rulztrainer_likeness" / "core.py",
                             INSTALL_DIR / "comfyui" / "rulztrainer_likeness" / "core.py") if q.is_file()),
                Path(base) / "comfyui" / "rulztrainer_likeness" / "core.py")
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing: the likeness scoring code is not in this install")
    spec = importlib.util.spec_from_file_location("rulztrainer_likeness_core", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class LikenessScorer:
    """Scores a picture on the trainer's likeness scale: the face (ArcFace or antelopev2), the gold face, the body, the
    face / body weighting (likeness.combine) of the run `from_contestant` came from. `scale` and `note` say which scale
    the numbers are on. Pictures of kind "face" get no body score, kind "body" no face score."""

    def __init__(self, plan: Plan, scale: str, note: str | None, face_refs: list[str], body_refs: list[str],
                 device: str):
        from .likeness import combine

        self.combine = combine
        cfg = next(f["cfg"] for f in plan.files if f["id"] == plan.from_id)
        core = self.core = _load_core(plan.base)
        notes = [note] if note else []
        body_kind = str(_cfg_get(cfg, "body_score") or "appearance")
        if body_kind not in ("appearance", "proportions"):
            if body_refs:
                notes.append(f"body_score {body_kind} is not available in tests: no body scores.")
            body_refs = []
        gold = str(_cfg_get(cfg, "face_gold") or "auto")
        job = None
        if scale == "run":  # decide_scale read it already; "auto" gold is the recognizer the run recorded
            job = core.job_scale(next(f["config_path"] for f in plan.files if f["id"] == plan.from_id).parent)
        if gold == "auto":
            gold = str(job.get("gold_name") or "facenet") if job else "adaface"
        elif gold in ("off", "none", ""):
            gold = "off"
        self.refs = core.build_references(
            face_refs, body_refs, face_provider=str(_cfg_get(cfg, "face_provider") or "cpu"),
            det_size=int(_cfg_get(cfg, "face_det_size") or 640), body_kind=body_kind if body_refs else "appearance",
            body_backend=str(_cfg_get(cfg, "body_backend") or "mediapipe+sam3"), body_device=device,
            sam3_model=str(_cfg_get(cfg, "sam3_model") or "facebook/sam3"),
            models_dir=str(plan.base / str(_cfg_get(cfg, "models_dir") or "models")), gold=gold,
            face_model_name=str(_cfg_get(cfg, "face_model") or "buffalo_l"))
        self.scale_info = core.build_scale(self.refs, normalize=True, job=job)
        self.fw, self.bw = float(_cfg_get(cfg, "face_weight")), float(_cfg_get(cfg, "body_weight"))
        self.gold_select = bool(_cfg_get(cfg, "face_gold_select"))
        self.scale = scale
        self.scale_note = " ".join(notes) or None
        self.notes = self.refs.notes

    def score(self, path: Path, kind: str) -> dict:
        rows = self.core.score_rows(self.refs, [(Path(path).name, path)])
        if kind == "face":
            rows[0]["body_raw"] = None
        elif kind == "body":
            rows[0]["face_raw"] = rows[0]["gold_raw"] = None
        r = self.core.apply_scale(rows, self.scale_info, self.fw, self.bw, self.gold_select)[0]
        return {"face": r["face"], "body": r["body"], "gold": r["gold"], "combined": r["overall"]}

    def close(self) -> None:
        self.refs = None


def make_scorer(plan: Plan, scale: str, note: str | None, face_refs: list[str], body_refs: list[str]):
    return LikenessScorer(plan, scale, note, face_refs, body_refs, plan.devices[0])


@dataclass
class Hooks:
    """Everything the run reaches a GPU, a model or a scorer through. The defaults are the real implementations; a test
    passes fakes (a runner that writes small pictures, a scorer that reads a number from the file name)."""
    read_config: Callable | None = None
    make_runner: Callable | None = None  # (cfg, devices, test_dir) -> runner: .gpus, .use(path, cfg), .render(jobs, strength), .close()
    make_job: Callable | None = None  # (**RenderJob fields)
    prepare_prompts: Callable | None = None  # (cfg, prompts, device)
    cook_soup: Callable | None = None  # (paths, out, weights, cfg) -> report
    make_scorer: Callable | None = None  # (plan, scale, note, face_refs, body_refs) -> scorer: .score(path, kind) -> dict, .close()
    release_gpu: Callable | None = None
    status_every: float = STATUS_EVERY

    def filled(self) -> "Hooks":
        def release():
            from .renderhelper import release_gpu

            release_gpu()

        return Hooks(read_config=self.read_config or read_config, make_runner=self.make_runner or make_runner,
                     make_job=self.make_job or make_job, prepare_prompts=self.prepare_prompts or prepare_prompts,
                     cook_soup=self.cook_soup or cook_soup, make_scorer=self.make_scorer or make_scorer,
                     release_gpu=self.release_gpu or release, status_every=self.status_every)


# ---------------------------------------------------------------- results


def _mean(values: list[float]):
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def _r(v):
    return None if v is None else round(float(v), 2)


def build_columns(cells: list[dict], order: list[str]) -> list[dict]:
    """One column per (contestant, strength) that has cells: the means over its scored cells, n = how many cells have a
    combined score. Contestants in the grid's order, strengths ascending."""
    groups: dict[tuple, list[dict]] = {}
    for c in cells:
        groups.setdefault((c["contestant"], c["strength"]), []).append(c)
    columns = []
    for (cid, s), cs in sorted(groups.items(), key=lambda kv: (order.index(kv[0][0]), kv[0][1])):
        col = {"contestant": cid, "strength": s}
        for k in ("face", "body", "gold", "combined"):
            col[k] = _mean([c[k] for c in cs])
        col["n"] = sum(c["combined"] is not None for c in cs)
        columns.append(col)
    return columns


def pick_best(columns: list[dict], order: list[str]) -> tuple[dict, dict | None]:
    """({contestant: its best strength}, the winner column). The highest column `combined`; ties (to six decimals) go to
    the lower strength, then the earlier contestant. A contestant or test with no scored column gets None."""
    def key(c):
        return (-round(c["combined"], 6), c["strength"], order.index(c["contestant"]))

    scored = [c for c in columns if c["combined"] is not None]
    best = {cid: None for cid in order}
    for cid in order:
        mine = [c for c in scored if c["contestant"] == cid]
        if mine:
            best[cid] = min(mine, key=key)["strength"]
    winner = min(scored, key=key) if scored else None
    return best, winner


def _round_cells(cells: list[dict]) -> list[dict]:
    """The cells as written: scores rounded to two decimals (aggregating and ranking use the full values)."""
    return [{**c, **{k: _r(c[k]) for k in ("face", "body", "gold", "combined") if k in c}} for c in cells]


def _round_columns(columns: list[dict]) -> list[dict]:
    return [{**c, **{k: _r(c[k]) for k in ("face", "body", "gold", "combined")}} for c in columns]


# ---------------------------------------------------------------- the run


def refuse(test_dir: Path, test_id: str, reason: str) -> int:
    StatusWriter(test_dir, test_id).update(state="refused", reason=reason)
    log.error("Refused: %s", reason)
    return 1


def tail_of(text: str, lines: int = 12) -> str:
    return "\n".join(text.strip().splitlines()[-lines:])


def run(test_dir, base: Path | None = None, hooks: Hooks | None = None) -> int:
    """Run one test to its end: refusals, soups, render, score, results.json. Returns the exit status (0 done or
    stopped; 1 refused or failed). The state is in status.json throughout."""
    test_dir = Path(test_dir)
    base = Path(base or INSTALL_DIR)
    h = (hooks or Hooks()).filled()
    try:
        spec = read_json(test_dir / "spec.json")
    except (OSError, ValueError) as e:
        return refuse(test_dir, test_dir.name, f"spec.json: cannot read it ({e})")
    test_id = spec.get("test_id") if isinstance(spec, dict) and isinstance(spec.get("test_id"), str) else test_dir.name
    try:
        plan = validate(spec, base, folder_name=test_dir.name, config_reader=h.read_config)
    except SpecError as e:
        return refuse(test_dir, test_id, str(e))

    status = StatusWriter(test_dir, test_id, every=h.status_every)
    status.update(state="loading", total=plan.total, current="starting")
    status.start()
    started = time.time()
    try:
        return _run(test_dir, plan, h, status, started)
    except SpecError as e:  # found after the soups were cooked, still before any GPU work
        status.finish("refused", reason=str(e))
        return 1
    except Exception:  # noqa: BLE001 - whatever it is ends the test as failed, with the traceback's end
        text = traceback.format_exc()
        log.error("The test failed:\n%s", text)
        status.finish("failed", error=tail_of(text))
        return 1
    finally:
        status._stop.set()


def pool_layer_check(plan: Plan, paths: dict[str, Path]) -> None:
    """Two or more GPUs send each worker a state built from the file alone (payload_from_file), and a worker keeps its
    previous weights in any layer the state lacks; the parent has no slots to complete it from. So every contestant
    and soup must have the same set of layers (they share lora_targets and lora_blocks, so in practice they do), or the
    test is refused before any GPU work. A layer the workers' model lacks still fails the round with the worker's error
    (the parent cannot see the slots to check that earlier). One GPU has no such limit: LocalRunner completes the state
    from its own slots."""
    from . import soup

    layers: dict[str, frozenset] = {}
    for cid, path in paths.items():
        try:
            layers[cid] = frozenset(soup.read_ingredient(path))
        except Exception as e:  # noqa: BLE001
            raise SpecError(f"devices: {path.name} ({cid}) cannot be read for a multi-GPU test: {e}") from e
    first = next(iter(layers))
    for cid, got in layers.items():
        if got != layers[first]:
            raise SpecError(f"devices: {cid} and {first} do not have the same layers ({len(got)} against "
                            f"{len(layers[first])}); with two or more GPUs every LoRA must cover the same layers "
                            "(use one device)")


def _run(test_dir: Path, plan: Plan, h: Hooks, status: StatusWriter, started: float) -> int:
    order = [c["id"] for c in plan.contestants]
    paths: dict[str, Path] = {f["id"]: f["lora_path"] for f in plan.files}

    # contestants as they are in the results (file hashes first: they are what the test is about)
    out_contestants = []
    for f in plan.files:
        out_contestants.append({"id": f["id"], "label": f["label"], "lora": f["lora"], "sha256": sha256_of(f["lora_path"]),
                                "kind": "file"})
    # soups: cooked on the CPU before any GPU work
    for s in plan.soups:
        status.update(state="loading", current=f"cooking soup {s['id']}")
        out = test_dir / s["out"]
        h.cook_soup([paths[o] for o in s["of"]], out, s["weights"], plan.cfg_of(s["of"][0]))
        paths[s["id"]] = out
        out_contestants.append({"id": s["id"], "label": s["label"], "lora": s["out"], "sha256": sha256_of(out),
                                "kind": "soup", "of": s["of"], "weights": s["weights"]})

    if len(plan.devices) > 1:
        pool_layer_check(plan, paths)
    cfg = dict(plan.files[0]["cfg"])
    settings = render_settings(plan)
    if (plan.spec.get("render") or {}).get("negative") is not None:
        cfg["sample_negative"] = settings["negative"]
    status.update(state="loading", current="caching the prompts' text")
    h.prepare_prompts(cfg, plan.prompts, plan.devices[0])
    status.update(state="loading", current="loading the model")
    runner = h.make_runner(cfg, plan.devices, test_dir)
    stopped = False
    render_seconds = 0.0
    cells: list[dict] = []
    try:
        status.update(state="rendering")
        for cid in order:
            if stopped:
                break
            runner.use(paths[cid], plan.cfg_of(cid))
            for strength in plan.strengths:
                if stopped:
                    break
                pairs = [(pi, seed) for pi in range(len(plan.prompts)) for seed in plan.seeds]
                # the pool renders a whole column per round; one GPU renders picture by picture
                batches = [pairs] if len(plan.devices) > 1 else [[p] for p in pairs]
                for batch in batches:
                    if (test_dir / "STOP").exists():
                        stopped = True
                        break
                    jobs = []
                    for pi, seed in batch:
                        rel = image_rel(cid, strength, pi, seed)
                        (test_dir / rel).parent.mkdir(parents=True, exist_ok=True)
                        jobs.append(h.make_job(
                            prompt=plan.prompts[pi], width=int(settings["width"]), height=int(settings["height"]),
                            steps=int(settings["steps"]), cfg_scale=float(settings["cfg"]), seed=int(seed),
                            out_path=test_dir / rel, label=f"{cid} at {strength_name(strength)}, prompt {pi}, seed {seed}",
                            kind=plan.kinds[pi] if plan.kinds[pi] != "both" else "sample",
                            shown=plan.prompts[pi]))
                    t0 = time.time()
                    runner.render(jobs, strength)
                    took = time.time() - t0
                    render_seconds += took
                    for pi, seed in batch:
                        cells.append({"contestant": cid, "strength": strength, "prompt": pi, "seed": seed,
                                      "image": image_rel(cid, strength, pi, seed)})
                        status.picture_done(took / len(batch),
                                            f"{cid} at {strength_name(strength)}, prompt {pi}, seed {seed}")
                    if (test_dir / "STOP").exists():  # one that came while this batch was in hand, the last included
                        stopped = True
                        break
        gpus = list(getattr(runner, "gpus", []))
    finally:
        runner.close()
        h.release_gpu()

    # score what exists
    status.update(state="scoring", current="scoring the pictures")
    scale, note = decide_scale(plan)
    face_refs, body_refs = resolve_refs(plan)
    scorer = h.make_scorer(plan, scale, note, face_refs, body_refs) if (face_refs or body_refs) else None
    try:
        if scorer is not None:
            scale, note = getattr(scorer, "scale", scale), getattr(scorer, "scale_note", note)
            for n, cell in enumerate(cells):
                status.update(current=f"scoring {n + 1} of {len(cells)}")
                s = scorer.score(test_dir / cell["image"], plan.kinds[cell["prompt"]])
                cell.update({k: s.get(k) for k in ("face", "body", "gold", "combined")})  # full precision
        else:
            for cell in cells:
                cell.update(face=None, body=None, gold=None, combined=None)
            note = (note + " " if note else "") + "No references: the pictures are not scored."
    finally:
        if scorer is not None and hasattr(scorer, "close"):
            scorer.close()
        h.release_gpu()

    columns = build_columns(cells, order)
    best, winner = pick_best(columns, order)
    results = {
        "version": RESULTS_VERSION, "test_id": plan.spec["test_id"], "complete": not stopped, "scale": scale,
        "scale_note": note, "contestants": out_contestants, "cells": _round_cells(cells),
        "columns": _round_columns(columns),
        "best_strength": best,
        "winner": None if winner is None else {"contestant": winner["contestant"], "strength": winner["strength"],
                                               "combined": _r(winner["combined"])},
        "render": {"width": settings["width"], "height": settings["height"], "steps": settings["steps"],
                   "cfg": settings["cfg"], "seconds": round(render_seconds, 1), "gpus": gpus}}
    write_json(test_dir / "results.json", results)
    status.finish("stopped" if stopped else "done", current=None)
    return 0


# ---------------------------------------------------------------- new / list and the command line


def cmd_new(spec_text: str, base: Path | None = None, config_reader=None) -> str:
    """Validate a spec, make output/_tests/<test_id>/ with spec.json (test_id, created and version filled when missing)
    and a queued status.json; returns the test_id. SpecError when the spec is refused (nothing is made)."""
    base = Path(base or INSTALL_DIR)
    try:
        spec = json.loads(spec_text)
    except ValueError as e:
        raise SpecError(f"spec: not valid JSON ({e})") from e
    if not isinstance(spec, dict):
        raise SpecError("spec: not a JSON object")
    spec.setdefault("version", SPEC_VERSION)
    if not spec.get("test_id"):
        spec["test_id"] = new_test_id()
    spec.setdefault("created", utc_now())
    validate(spec, base, config_reader=config_reader)
    folder = tests_root(base) / spec["test_id"]
    if folder.exists():
        raise SpecError(f"test_id: {spec['test_id']} already exists")
    folder.mkdir(parents=True)
    write_json(folder / "spec.json", spec)
    status = StatusWriter(folder, spec["test_id"])
    status.data.pop("pid_created", None)
    status.data["pid"] = None  # nothing runs yet (set directly: update never writes a pid back to null)
    status.update(state="queued", total=0)
    return spec["test_id"]


def list_tests(base: Path | None = None) -> list[dict]:
    """One entry per test folder, newest first: id, name, state, created (state as readers show it: a dead pid is
    failed). A folder without a readable spec is listed by its name with state unknown."""
    root = tests_root(base)
    out = []
    for d in sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name, reverse=True) if root.is_dir() else []:
        try:
            spec = read_json(d / "spec.json")
        except (OSError, ValueError):
            spec = {}
        try:
            state = effective_state(read_json(d / "status.json"),
                                    age=time.time() - (d / "status.json").stat().st_mtime)
        except (OSError, ValueError):
            state = "unknown"
        out.append({"id": d.name, "test_id": d.name, "name": spec.get("name"), "state": state,
                    "created": spec.get("created")})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m rulztrainer.loratest", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("new", help="validate a spec and make its test folder")
    p.add_argument("--spec", required=True, help="the spec JSON file ('-' reads stdin)")
    p = sub.add_parser("run", help="run a test folder")
    p.add_argument("test_dir")
    sub.add_parser("list", help="one JSON line per test")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    if args.cmd == "new":
        text = sys.stdin.read() if args.spec == "-" else Path(args.spec).read_text(encoding="utf-8")
        try:
            print(cmd_new(text))
        except SpecError as e:
            print(f"refused: {e}", file=sys.stderr)
            return 1
        return 0
    if args.cmd == "list":
        for t in list_tests():
            print(json.dumps(t, ensure_ascii=False))
        return 0
    return run(args.test_dir)


if __name__ == "__main__":
    sys.exit(main())
