"""Offline inventory and rewriting of a trainer config for a remote send."""
from __future__ import annotations

import copy
import os
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath

PATH_KEYS = {"output_dir", "models_dir", "cache_dir", "dataset_dir", "reg_dir", "model_path",
             "transformer_file", "text_encoder_file", "vae_file", "tokenizer_path", "sample_model_path",
             "turbo_lora", "sample_lora", "training_adapter", "resume_from", "finetune_resume",
             "resume_state", "nlf_model", "sam3_model", "soup_files", "noise_files", "face_refs", "body_refs"}
NOT_PATHS = {"save_state", "compile_model", "face_model"}
_MODELS = PATH_KEYS - {"output_dir", "models_dir", "cache_dir", "dataset_dir", "reg_dir", "face_refs", "body_refs"}
_REPO_KEYS = {"model_path", "sample_model_path", "sam3_model"}
_REFS = {"face_refs", "body_refs"}


@dataclass
class SendPlan:
    datasets: list[dict] = field(default_factory=list)
    files: list[dict] = field(default_factory=list)
    changes: list[dict] = field(default_factory=list)
    blockers: list[dict] = field(default_factory=list)
    total_bytes: int = 0
    config: dict = field(default_factory=dict, repr=False)
    _rewrites: list = field(default_factory=list, repr=False)


# One rule for a model file's place on the instance: rulz_files.clean_name makes the name (letters, digits and . _ + -
# only, no leading dot), models/sent/<12 hex>/<name> is what rulz_files.check_dest and the agent's sent_parts accept.
REMOTE_RE = re.compile(r"models/sent/[0-9a-f]{12}/[A-Za-z0-9._+-]+")


def remote_ok(target) -> bool:
    return isinstance(target, str) and bool(REMOTE_RE.fullmatch(target)) and PurePosixPath(target).name not in (".", "..")


def _safe(name):
    return re.sub(r"[^A-Za-z0-9._ -]", "_", name).strip(" .") or "item"


def _identity(path):
    return str(path.resolve()).replace("\\", "/").casefold().rstrip("/")


def _inside(path, parent):
    a, b = _identity(path), _identity(parent)
    return a == b or a.startswith(b + "/")


def _relative(path, parent):
    # Identity comparison above is case insensitive even on a non-Windows test host.
    return "/".join(path.parts[len(parent.parts):])


def _repo(value):
    return bool(re.fullmatch(r"[\w.-]+/[\w.-]+(?:#[\w.-]+)?", value))


def _absolute(value):
    return Path(value).is_absolute() or PureWindowsPath(value).is_absolute()


def _inventory(path):
    if path.is_file():
        with path.open("rb"):
            pass
        return 1, path.stat().st_size
    count = size = 0
    def fail(exc):
        raise exc
    for base, dirs, names in os.walk(path, onerror=fail):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for name in names:
            if name.startswith("."):
                continue
            item = Path(base) / name
            with item.open("rb"):
                pass
            count += 1
            size += item.stat().st_size
    return count, size


def _standard(config, key, path):
    """Ask the trainer's own download planner, with this setting reset to its default."""
    try:
        from rulztrainer.config import TrainConfig
        from rulztrainer.comfy_files import planned_downloads
        fields = TrainConfig.__dataclass_fields__
        values = {k: v for k, v in config.items() if k in fields}
        values[key] = "auto" if key in {"turbo_lora", "sample_lora", "training_adapter"} else ""
        cfg = TrainConfig(**values)
        for field_name, repo, filename in planned_downloads(cfg):
            if field_name == key and Path(filename).name.casefold() == path.name.casefold():
                return {"repo": repo, "file": filename, "bytes": None}
    except (ImportError, ValueError, KeyError, TypeError, AttributeError):
        pass
    return None


def plan_send(config: dict, root: Path, gpus: int = 1) -> SendPlan:
    root = Path(root).resolve()
    plan = SendPlan(config=copy.deepcopy(config))
    dataset_requests, references, model_requests = [], [], []

    def location(value):
        p = Path(value).expanduser()
        return (p if p.is_absolute() else root / p).resolve()

    def blocker(key, value, reason):
        plan.blockers.append({"key": key, "path": value, "reason": reason})

    def gather(row, prefix=(), effective=None):
        effective = effective or row
        def dataset(value, loc, required=False):
            if not value:
                if required:
                    blocker("dataset_dir", value, "empty path")
                return
            p = location(value)
            if not p.is_dir():
                blocker(str(loc[-1]), value, "missing dataset folder")
                return
            dataset_requests.append((p, loc))
        if "datasets" in row and row["datasets"]:
            for i, item in enumerate(row["datasets"]):
                dataset(item.get("path", ""), prefix + ("datasets", i, "path"), True)
        elif "dataset_dir" in row and not row.get("datasets"):
            dataset(row["dataset_dir"], prefix + ("dataset_dir",), True)
        if row.get("reg_dir"):
            dataset(row["reg_dir"], prefix + ("reg_dir",))
        source = effective.get("model_source", "files")
        unused = {"model_path"} if source == "files" else {"transformer_file", "text_encoder_file", "vae_file",
                                                            "tokenizer_path"}
        for key in _REFS | _MODELS:
            if key not in row or key in unused:  # (the other model source's files are not read by the run)
                continue
            value = row[key]
            for index, val in enumerate(value if isinstance(value, list) else [value]):
                loc = prefix + (key,) + ((index,) if isinstance(value, list) else ())
                if not val or val in ("auto", "none"):
                    continue
                p = location(val)
                if not p.exists():
                    if key in _REPO_KEYS and not _absolute(val) and _repo(val):
                        continue
                    blocker(key, val, "missing file or folder")
                    continue
                (references if key in _REFS else model_requests).append((p, loc, key, effective))
        for key, new in (("output_dir", "output"), ("models_dir", "models")):
            if key in row and row[key] != new:
                plan.changes.append({"key": key, "path": list(prefix + (key,)), "old": row[key], "new": new, "reason": "instance folder"})
        if row.get("cache_dir") and _absolute(row["cache_dir"]):
            plan.changes.append({"key": "cache_dir", "path": list(prefix + ("cache_dir",)), "old": row["cache_dir"], "new": "", "reason": "local cache"})
        if gpus <= 1 and row.get("multi_gpu") not in (None, "off", False):
            plan.changes.append({"key": "multi_gpu", "path": list(prefix + ("multi_gpu",)), "old": row["multi_gpu"], "new": "off", "reason": "one GPU"})
        for key in ("device", "devices", "text_encoder_device", "body_device", "probe_device", "split_plan"):
            if isinstance(row.get(key), str):
                new = re.sub(r"cuda:(\d+)", lambda m: "cuda:0" if int(m.group(1)) >= max(1, gpus) else m.group(0), row[key])
                if new != row[key]:
                    plan.changes.append({"key": key, "path": list(prefix + (key,)), "old": row[key], "new": new, "reason": "GPU unavailable"})

    gather(config)
    for i, row in enumerate(config.get("race_configs") or []):
        gather(row, ("race_configs", i), {**config, **row})
    # Parents precede children so an inner dataset is sent only once.
    dataset_requests.sort(key=lambda item: len(item[0].parts))
    used = set()
    def name_for(path, initial=None):
        base = _safe(initial or path.name)
        candidate = base
        if candidate.casefold() in used:
            candidate = _safe(path.parent.name + "_" + base)
        stem = candidate
        n = 2
        while candidate.casefold() in used:
            candidate = f"{stem}_{n}"
            n += 1
        used.add(candidate.casefold())
        return candidate
    for p, loc in dataset_requests:
        parent = next((d for d in plan.datasets if _inside(p, Path(d["path"]))), None)
        if parent is None:
            try:
                count, size = _inventory(p)
            except OSError as exc:
                blocker(str(loc[-1]), str(p), f"unreadable: {exc}")
                continue
            parent = {"path": str(p), "name": name_for(p), "files": count, "bytes": size, "kind": "folder"}
            plan.datasets.append(parent)
        suffix = _relative(p, Path(parent["path"]))
        plan._rewrites.append((loc, "datasets/" + parent["name"] + ("/" + suffix if suffix else "")))
    outside = []
    for p, loc, key, _ in references:
        parent = next((d for d in plan.datasets if _inside(p, Path(d["path"]))), None)
        if parent:
            plan._rewrites.append((loc, "datasets/" + parent["name"] + "/" + _relative(p, Path(parent["path"]))))
        else:
            outside.append((p, loc, key))
    if outside:
        name = name_for(root, "refs_" + _safe(str(config.get("job_name") or "job")))
        entries, seen = [], {}
        for p, loc, key in outside:
            identity = _identity(p)
            if identity not in seen:
                try:
                    _, size = _inventory(p)
                    if not p.is_file() or p.name.startswith("."):
                        raise OSError("reference is not a visible file")
                except OSError as exc:
                    blocker(key, str(p), f"unreadable: {exc}")
                    continue
                target = _safe(p.name)
                stem, ext = os.path.splitext(target)
                n = 2
                while target.casefold() in {e["name"].casefold() for e in entries}:
                    target = f"{stem}_{n}{ext}"
                    n += 1
                seen[identity] = target
                entries.append({"path": str(p), "name": target, "bytes": size})
            plan._rewrites.append((loc, f"datasets/{name}/{seen[identity]}"))
        if entries:
            plan.datasets.append({"path": "", "name": name, "kind": "files", "files": entries, "bytes": sum(e["bytes"] for e in entries)})
    by_path = {}
    for p, loc, key, effective in model_requests:
        ident = _identity(p)
        if ident not in by_path:
            try:
                _, size = _inventory(p)
            except OSError as exc:
                blocker(key, str(p), f"unreadable: {exc}")
                continue
            entry = {"keys": [], "path": str(p), "bytes": size, "kind": "folder" if p.is_dir() else "file", "standard": _standard(effective, key, p)}
            by_path[ident] = entry
            plan.files.append(entry)
        entry = by_path[ident]
        if key not in entry["keys"]:
            entry["keys"].append(key)
        standard = _standard(effective, key, p)
        if entry["standard"] != standard:
            entry["standard"] = None
        plan._rewrites.append((loc, ("file", entry["path"])))
    plan.total_bytes = sum(d["bytes"] for d in plan.datasets) + sum(f["bytes"] for f in plan.files)
    return plan


def apply(plan: SendPlan, choices: dict) -> dict:
    if plan.blockers:
        raise ValueError("Cannot apply a plan with blockers")
    result = copy.deepcopy(plan.config)
    def set_at(loc, value):
        obj = result
        for part in loc[:-1]:
            obj = obj[part]
        obj[loc[-1]] = value
    for change in plan.changes:
        set_at(change["path"], change["new"])
    for loc, target in plan._rewrites:
        if isinstance(target, tuple):
            path = target[1]
            entry = next(f for f in plan.files if f["path"] == path)
            choice = choices.get(path) or next((v for k, v in choices.items() if str(k).casefold() == path.casefold()), None)
            if not choice:
                choice = {"how": "standard"} if entry["standard"] else None
            if not choice or choice.get("how") not in ("upload", "bucket", "standard"):
                raise ValueError(f"Choose how to send {path}")
            if choice["how"] == "standard":
                if not entry["standard"]:
                    raise ValueError(f"No standard file for {path}")
                target = "auto" if loc[-1] in ("turbo_lora", "sample_lora", "training_adapter") else ""
            else:
                target = choice.get("remote", "")
                if not remote_ok(target):
                    raise ValueError(f"Invalid remote path for {path}")
        set_at(loc, target)
    return result
