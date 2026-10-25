"""RulzTrainer WebUI (NiceGUI). Run: python webui.py [--port 7870] [--open]

Training runs in a subprocess so Stop frees GPU memory cleanly; the UI polls
<output>/<job>/status.json and train.log. The running job (pid + job dir) is recorded in .webui_job.json, so a
reloaded page, a second browser or a restarted WebUI reattaches to it.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
from contextlib import contextmanager
from pathlib import Path

from nicegui import app, run, ui

from rulztrainer.config import (ARCH_DEFAULT_TARGETS, ARCH_SAMPLE_DEFAULTS, MEGAPIXEL_CHOICES, MODEL_PRESETS,
                                PRESET_PROBE_PROMPTS, PRESETS, TrainConfig, apply_preset, mp_to_side, side_to_mp)
from rulztrainer import bodysim, facesim
from rulztrainer.losswatch import read_verdicts

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
CUSTOM = "Custom path / repo id"
HF_SOURCE = "Hugging Face repo / diffusers folder"
FILE_SOURCE = "Single safetensors files (ComfyUI)"
TARGETS = ["q", "k", "v", "o", "ffn_up_gate", "ffn_down"]
# model-specific LoRA target groups (see rulztrainer.archs): name -> (families that have it, tooltip)
EXTRA_TARGETS = {
    "cross_attn": (("fibo2",), "text cross-attention in the 5 injection blocks"),
    "adaln": (("fibo2",), "per-block modulation Linear"),
    "refiner": (("fibo2",), "the 2 noise-refiner blocks"),
    "perceiver": (("fibo2",), "the gist-token resampler"),
    "attn_gate": (("krea2",), "the sigmoid output gate of every attention (Krea's own LoRAs train it too; on by "
                              "default)"),
    "text_fusion": (("krea2",), "the text fusion blocks that mix the 12 text encoder layers (on by default: the usual "
                                "recipe leaves them under-trained)"),
    "embed": (("fibo2", "krea2"), "image / text / timestep embedders and the final layer"),
}
N_BLOCKS = {"moss_image": 42, "fibo2": 44, "krea2": 28}
COMPILE_CHOICES = {"auto": "auto: runs of 100+ steps", True: "on", False: "off"}
N_BLOCKS_MAX = max(N_BLOCKS.values())


def preset_arch(preset: str, model_path: str = "") -> str:
    """The model family behind a preset name (or a custom path / repo id)."""
    path = (model_path or "") if preset == CUSTOM else MODEL_PRESETS.get(preset, preset or "")
    return TrainConfig(model_path=path).arch_name()


def is_fibo_preset(preset: str, model_path: str = "") -> bool:
    return preset_arch(preset, model_path) == "fibo2"
QUANTS = ["auto", "none", "int8", "int8_convrot", "nvfp4"]
MP_OPTIONS = {mp: f"{mp:g} MP ({mp_to_side(mp)}^2)" for mp in MEGAPIXEL_CHOICES}
SAFETENSORS = [("safetensors", "*.safetensors"), ("all files", "*.*")]

_proc: subprocess.Popen | None = None  # started by this WebUI process
_adopted = None  # psutil.Process of a job started by an earlier WebUI process (reattached from JOB_FILE)
_job_dir: Path | None = None
JOB_FILE = ROOT / ".webui_job.json"


# ---------------------------------------------------------------- backend helpers


def _gpus() -> list[str]:
    try:
        import torch

        return [f"cuda:{i}" for i in range(torch.cuda.device_count())] or ["cpu"]
    except Exception:  # noqa: BLE001
        return ["cuda:0"]


def _running() -> bool:
    if _proc is not None:
        return _proc.poll() is None
    if _adopted is not None:
        try:
            import psutil

            return _adopted.is_running() and _adopted.status() != psutil.STATUS_ZOMBIE
        except Exception:  # noqa: BLE001  (process vanished between the checks)
            return False
    return False


def _record_job(pid: int, job_dir: Path) -> None:
    """Remember the job on disk so a reloaded page or a restarted WebUI can reattach to it."""
    try:
        import psutil

        created = psutil.Process(pid).create_time()
    except Exception:  # noqa: BLE001
        created = None
    data = {"pid": pid, "create_time": created, "job_dir": str(job_dir.resolve())}
    try:
        JOB_FILE.write_text(json.dumps(data, indent=1), encoding="utf-8")
    except OSError:
        pass


def reattach_job() -> bool:
    """Pick up the last job from JOB_FILE when this process has none. Returns True if that job is still running.

    The job dir is adopted even when the process has ended, so the monitor shows the last run's final status.
    The pid only counts as the same job when its create time matches (pids get reused)."""
    global _adopted, _job_dir
    if _proc is not None or _adopted is not None:
        return _running()
    try:
        data = json.loads(JOB_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    job_dir = Path(data.get("job_dir") or "")
    if not job_dir.is_dir():
        return False
    _job_dir = job_dir
    try:
        import psutil

        proc = psutil.Process(int(data["pid"]))
        created = data.get("create_time")
        if created is not None and abs(proc.create_time() - float(created)) > 1.0:
            return False
        _adopted = proc
    except Exception:  # noqa: BLE001  (gone, access denied, psutil missing)
        return False
    return _running()


def _relative(path: str) -> str:
    """Store paths inside the project relative to it, so configs stay portable."""
    try:
        return str(Path(path).resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return path


# native pick dialogs need a desktop: always on Windows/macOS, only with a display on Linux (not on a headless
# server such as a vast.ai instance, where Browse opens the in-page browser of the server's files, server_pick)
HAS_DISPLAY = sys.platform in ("win32", "darwin") or bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
# ...and a viewer sitting at that desktop: a dialog opened for a remote browser (Tailscale, LAN) would pop up on the
# server's screen. Set per page load by build_page.
_native_dialogs = HAS_DISPLAY


def is_local_viewer(request) -> bool:
    """True when the page was opened on this machine. `tailscale serve` connects from 127.0.0.1 too, but adds the
    Tailscale-User-Login / X-Forwarded-For headers."""
    host = request.client.host if request.client else ""
    proxied = "tailscale-user-login" in request.headers or "x-forwarded-for" in request.headers
    return host in ("127.0.0.1", "::1", "localhost") and not proxied


def _is_repo_example(path: str) -> bool:
    """configs/example.yaml ships with the repo; never modify it (a local change would block `git pull`)."""
    p = Path(path)
    p = p if p.is_absolute() else ROOT / p
    try:
        return p.resolve() == (ROOT / "configs" / "example.yaml").resolve()
    except OSError:
        return False


def dialog_start_dir(initial: str = "") -> str:
    """Folder a pick dialog opens in: the path itself if it is a folder, else its folder when that exists (also
    for a file not created yet), else the project folder. Relative paths are taken from the project folder."""
    init = Path(initial) if initial else ROOT
    if not init.is_absolute():
        init = ROOT / init
    if not init.is_dir():
        init = init.parent if init.parent.is_dir() else ROOT
    return str(init)


def tk_dialog(kind: str, initial: str = "", filetypes=None) -> str:
    """Native pick dialog (runs in a worker thread). kind: 'dir' | 'file'."""
    import tkinter as tk
    from tkinter import filedialog

    init = dialog_start_dir(initial)
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        if kind == "dir":
            return filedialog.askdirectory(initialdir=init, parent=root) or ""
        return filedialog.askopenfilename(initialdir=init, filetypes=filetypes or SAFETENSORS, parent=root) or ""
    finally:
        root.destroy()


# ---------------------------------------------------------------- in-page browser of the server's files

BROWSE_LIMIT = 2000  # entries listed per folder
BROWSE_TIP = {True: "Browse", False: "Browse the server's files"}  # by whether the native dialog is used


def file_suffixes(filetypes=None) -> set[str] | None:
    """The suffixes of a tk filetypes list's first (default) entry: {".safetensors"}; None for any file."""
    spec = (filetypes or SAFETENSORS)[0][1]
    pats = [p.strip() for p in spec.split(";") if p.strip()]
    if not pats or any(p in ("*", "*.*") for p in pats):
        return None
    return {p.lstrip("*").lower() for p in pats}


def list_folder(folder: Path, suffixes: set[str] | None, kind: str = "file", show_hidden: bool = False):
    """(folders, files, error) of a server folder, sorted by name: files only when picking a file, filtered by
    suffix (None = all). Folders that cannot be read give an error, not an exception."""
    dirs, files = [], []
    try:
        with os.scandir(folder) as it:
            for e in it:
                if not show_hidden and e.name.startswith("."):
                    continue
                try:
                    if e.is_dir():
                        dirs.append(Path(e.path))
                    elif (kind == "file" and e.is_file()
                          and (suffixes is None or Path(e.name).suffix.lower() in suffixes)):
                        files.append((Path(e.path), e.stat().st_size))
                except OSError:
                    continue
    except OSError as e:
        return [], [], f"{type(e).__name__}: {e.strerror or e}"
    dirs.sort(key=lambda p: p.name.casefold())
    files.sort(key=lambda t: t[0].name.casefold())
    return dirs[:BROWSE_LIMIT], files[:BROWSE_LIMIT], ""


def browse_shortcuts() -> list[tuple[str, Path]]:
    """Folders worth one click: the project and its models / datasets / output, the vast.ai workspace, home, a
    ComfyUI models folder, and the drives (Windows) or / (Linux). Only existing ones, each once."""
    workspace = Path(os.environ.get("WORKSPACE") or "/workspace")
    cands = [("RulzTrainer", ROOT), ("models", ROOT / "models"), ("datasets", ROOT / "datasets"),
             ("output", ROOT / "output"), ("workspace", workspace), ("home", Path.home())]
    cands += [("ComfyUI models", base / "ComfyUI" / "models") for base in (workspace, Path.home(), ROOT.parent)]
    if sys.platform == "win32":
        cands += [(f"{d}:", Path(f"{d}:\\")) for d in "CDEFGHIJ" if Path(f"{d}:\\").exists()]
    else:
        cands.append(("/", Path("/")))
    out, seen = [], set()
    for name, p in cands:
        try:
            key = p.resolve()
        except OSError:
            continue
        if p.is_dir() and key not in seen:
            seen.add(key)
            out.append((name, p))
    return out


def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


async def server_pick(kind: str, initial: str = "", filetypes=None) -> str:
    """Pick a file or folder on the server from the page itself (for a remote browser, such as a vast.ai instance,
    where a native dialog would open on the server's screen, or on none). Returns "" when cancelled."""
    suffixes = file_suffixes(filetypes)
    start = Path(dialog_start_dir(initial))
    state = {"dir": start, "picked": None, "all": False}
    with ui.dialog() as dlg, ui.card().classes("w-[52rem] max-w-full"):
        ui.label("Choose a folder on the server" if kind == "dir" else "Choose a file on the server") \
            .classes("text-base font-medium")
        with ui.row().classes("w-full items-center gap-1 no-wrap"):
            ui.button(icon="arrow_upward", on_click=lambda: go(state["dir"].parent)).props("flat dense round") \
                .tooltip("Up one folder")
            where = ui.input("Folder").classes("flex-grow min-w-0").props("dense")
            where.on("keydown.enter", lambda: go(Path(where.value.strip() or "/").expanduser()))
        with ui.row().classes("w-full gap-1"):
            for name, path in browse_shortcuts():
                ui.button(name, on_click=lambda p=path: go(p)).props("flat dense no-caps size=sm")
        listing = ui.column().classes("w-full gap-0 overflow-auto").style("height: 50vh")
        with ui.row().classes("w-full items-center justify-between no-wrap"):
            chosen = ui.label("").classes("rz-hint truncate min-w-0")
            with ui.row().classes("items-center gap-1 no-wrap"):
                if kind == "file" and suffixes is not None:
                    ui.switch("All files", on_change=lambda e: (state.update(all=e.value), go(state["dir"])))
                ui.button("Cancel", on_click=lambda: dlg.submit("")).props("flat")
                ok = ui.button("Select this folder" if kind == "dir" else "Select", on_click=lambda: finish()) \
                    .props("unelevated")

    def finish():
        if kind == "dir":
            dlg.submit(str(state["dir"]))
        elif state["picked"] is not None:
            dlg.submit(str(state["picked"]))

    def pick(path: Path):
        state["picked"] = path
        chosen.text = str(path)
        ok.enable()

    def go(folder: Path):
        if not folder.is_dir():
            ui.notify(f"Not a folder: {folder}", type="warning")
            return
        state["dir"], state["picked"] = folder, None
        where.value = str(folder)
        chosen.text = str(folder) if kind == "dir" else "No file selected"
        if kind == "file":
            ok.disable()
        dirs, files, error = list_folder(folder, None if state["all"] else suffixes, kind)
        listing.clear()
        with listing:
            if error:
                ui.label(error).classes("rz-hint")
            for d in dirs:
                with ui.item(on_click=lambda p=d: go(p)).props("dense clickable").classes("w-full"):
                    with ui.item_section().props("avatar"):
                        ui.icon("folder")
                    ui.item_section(d.name)
            for p, size in files:
                with ui.item(on_click=lambda p=p: pick(p)).props("dense clickable").classes("w-full") as row:
                    with ui.item_section().props("avatar"):
                        ui.icon("description")
                    ui.item_section(p.name)
                    with ui.item_section().props("side"):
                        ui.label(_human_size(size)).classes("rz-hint")
                row.on("dblclick", lambda p=p: (pick(p), finish()))
            if not dirs and not files and not error:
                ui.label("Empty folder" if kind == "dir" or state["all"] or suffixes is None
                         else f"No {', '.join(sorted(suffixes))} files here").classes("rz-hint")
            if len(dirs) == BROWSE_LIMIT or len(files) == BROWSE_LIMIT:
                ui.label(f"Only the first {BROWSE_LIMIT} entries are listed").classes("rz-hint")

    go(start)
    result = await dlg
    dlg.delete()
    return result or ""


async def pick_path(kind: str, initial: str = "", filetypes=None, native: bool = False) -> str:
    """A Browse button's pick: the native dialog for a viewer at this machine's desktop (native), else the in-page
    browser of the server's files. "" when nothing was picked."""
    if native:
        try:
            return await run.io_bound(tk_dialog, kind, initial, filetypes)
        except Exception:  # noqa: BLE001  (no tkinter, no display): the in-page browser instead
            pass
    return await server_pick(kind, initial, filetypes)


def start_training(cfg: TrainConfig) -> str:
    global _proc, _adopted, _job_dir
    reattach_job()
    if _running():
        return "A training job is already running."
    if not Path(cfg.dataset_dir).is_dir():
        return f"Dataset dir not found: {cfg.dataset_dir}"
    if cfg.model_source == "files":
        for n in ("transformer_file", "text_encoder_file", "vae_file"):
            if getattr(cfg, n) and not Path(getattr(cfg, n)).is_file():
                return f"{n} not found: {getattr(cfg, n)!r}"
    _job_dir = cfg.job_dir()
    _job_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = _job_dir / "config.yaml"
    cfg.save(cfg_path)
    for stale in ("status.json", "STOP"):
        p = _job_dir / stale
        if p.exists():
            p.unlink()
    (_job_dir / "train.log").write_text("", encoding="utf-8")
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    console = open(_job_dir / "console.log", "w", encoding="utf-8")  # noqa: SIM115
    # detached from the WebUI (own process group / session, no console): closing or restarting the WebUI leaves the
    # job running, and the next WebUI reattaches to it through JOB_FILE
    if sys.platform == "win32":
        detach = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW}
    else:
        detach = {"start_new_session": True}
    _proc = subprocess.Popen([sys.executable, "-m", "rulztrainer.train", "--config", str(cfg_path)],
                             cwd=str(ROOT), env=env, stdout=console, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, **detach)
    _adopted = None
    _record_job(_proc.pid, _job_dir)
    return f"Started job '{cfg.job_name}' (pid {_proc.pid}). Config saved to {cfg_path}"


def stop_training() -> str:
    reattach_job()
    if not _running():
        return "No job running."
    (_job_dir / "STOP").touch()
    return "Stop requested; the trainer will save the LoRA after the current step."


def kill_training() -> str:
    reattach_job()
    if not _running():
        return "No job running."
    if _proc is not None:
        _proc.kill()
    else:
        _adopted.kill()
    return "Process killed (no final save)."


def read_status() -> dict | None:
    if _job_dir is None:
        return None
    p = _job_dir / "status.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def read_log_tail(n: int = 80) -> str:
    if _job_dir is None:
        return ""
    text = ""
    lp = _job_dir / "train.log"
    if lp.exists():
        text = "\n".join(lp.read_text(encoding="utf-8", errors="replace").splitlines()[-n:])
    st = read_status() or {}
    if not _running() and (st.get("error") or not st.get("done")):
        cp = _job_dir / "console.log"
        if cp.exists():
            tail = cp.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
            text += "\n--- console.log ---\n" + "\n".join(tail)
    return text


def download_model(model_path: str) -> str:
    from rulztrainer.moss.loading import resolve_model_dir

    try:
        return f"Ready: {resolve_model_dir(model_path, 'models')}"
    except Exception as e:  # noqa: BLE001
        return f"Failed: {e}"


def scan_dataset(cfg: TrainConfig):
    from rulztrainer.dataset import scan_dataset as _scan

    items = _scan(cfg)
    sizes: dict[str, int] = {}
    for it in items:
        key = f"{it.width}x{it.height}"
        sizes[key] = sizes.get(key, 0) + 1
    missing = sum(1 for it in items if not it.caption)
    info = f"{len(items)} images. Buckets: {sizes}. Images without caption: {missing}."
    return info, [(str(it.image), it.caption[:120]) for it in items[:24]]


_task_progress = {"text": "", "frac": None, "active": False}


def set_progress(text: str, frac=None) -> None:
    """Called from worker threads (model downloads / loading); the UI timer shows it."""
    _task_progress.update(text=text, frac=frac)


def face_preview(paths: list[str], provider: str, det_size: int) -> tuple[str, list[tuple[str, str]]]:
    """Detect reference faces and save crops under output/_face_refs for the UI. Returns (info, [(crop, caption)])."""
    ok, msg = facesim.available()
    if not ok:
        return msg, []
    scorer = facesim.FaceScorer(provider, int(det_size), progress=set_progress)
    set_progress("Detecting faces...", None)
    paths = [str(p) for p in facesim.expand_refs(paths)]
    if not paths:
        return "No image files found in the listed paths (folders are expanded to the images inside).", []
    info = scorer.set_references(paths)
    out_dir = ROOT / "output" / "_face_refs"
    out_dir.mkdir(parents=True, exist_ok=True)
    crops = []
    for i, item in enumerate(info):
        if item.get("faces"):
            crop = scorer.crop_face(item["image"], item["bbox"], out_dir / f"ref_{i}.jpg")
            crops.append((str(crop), f"{Path(item['image']).name}: {item['faces']} face(s), det {item['det_score']:.2f}"))
        else:
            crops.append(("", f"{Path(item['image']).name}: NO FACE FOUND"))
    self_sim = scorer.ref_self_similarity
    text = (f"{len(scorer.refs)} of {len(paths)} reference images have a face. "
            + (f"Reference self-similarity: {self_sim * 100:.1f}% (the realistic ceiling for samples)."
               if self_sim is not None else "Add 2+ references to see the self-similarity ceiling."))
    return text, crops


def body_preview(paths: list[str], backend: str = "mediapipe", sam3_model: str = "facebook/sam3",
                 device: str = "cuda:0") -> str:
    ok, msg = bodysim.available()
    if not ok:
        return msg
    sil = None
    note = ""
    if backend == "mediapipe+sam3":
        try:
            sil = bodysim.SilhouetteScorer(sam3_model, device, progress=set_progress)
        except Exception as e:  # noqa: BLE001
            note = f" (SAM 3 unavailable: {str(e)[:120]}; skeleton only)"
    scorer = bodysim.BodyScorer("models", silhouette=sil, progress=set_progress)
    set_progress("Detecting poses...", None)
    paths = [str(p) for p in facesim.expand_refs(paths)]
    if not paths:
        return "No image files found in the listed paths (folders are expanded to the images inside)."
    info = scorer.set_references(paths)
    found = [i for i in info if i.get("ratios")]
    text = f"{len(found)} of {len(paths)} body references have a usable pose."
    if scorer.profile:
        text += " Profile: " + ", ".join(f"{k} {v:.2f}" for k, v in scorer.profile.items())
    if scorer.ref_self_similarity is not None:
        text += f". Reference self-similarity {scorer.ref_self_similarity * 100:.1f}%."
    return text + note


def score_caption(sc: dict) -> str:
    parts = []
    parts.append(f"face {sc['percent']:.0f}%" if sc.get("percent") is not None else "no face")
    if sc.get("body_percent") is not None:
        parts.append(f"body {sc['body_percent']:.0f}%")
    if sc.get("combined_percent") is not None:
        parts.append(f"combined {sc['combined_percent']:.0f}%")
    return "  |  " + ", ".join(parts)


def face_scores_by_image(job_dir: Path | None) -> dict[str, dict]:
    if job_dir is None:
        return {}
    return {Path(sc["image"]).name: sc for sc in facesim.read_scores(job_dir)}


def analyse_and_preview(cfg_dict: dict, what: str) -> tuple[str, list[tuple[str, str]]]:
    """Run face analysis on the dataset in the UI process; render region overlays or crop thumbnails."""
    from rulztrainer import regions as R
    from rulztrainer.dataset import augment_items, load_image_fit, scan_dataset

    cfg = TrainConfig.from_dict(cfg_dict)
    ok, msg = facesim.available()
    if not ok:
        return msg, []
    items = scan_dataset(cfg)
    det = facesim.FaceScorer(cfg.face_provider, cfg.face_det_size, progress=set_progress)
    set_progress("Analysing faces...", None)
    analysis = R.analyze_faces(items, det, cfg.resolved_cache_dir() / "faces.json",
                               progress=lambda i, n: set_progress(f"Analysing faces {i}/{n}", i / max(1, n)))
    n_face = sum(1 for r in analysis.values() if r.get("bbox"))
    out_dir = ROOT / "output" / ("_regions" if what == "regions" else "_crops")
    out_dir.mkdir(parents=True, exist_ok=True)
    thumbs: list[tuple[str, str]] = []
    if what == "crops":
        all_items = augment_items(items, cfg, analysis)
        crops = [it for it in all_items if it.category != "image"]
        for i, it in enumerate(crops[:24]):
            img = load_image_fit(it.image, it.width, it.height, it.src_box)
            img.thumbnail((256, 256))
            pth = out_dir / f"crop_{i}.jpg"
            img.save(pth, "JPEG", quality=85)
            thumbs.append((str(pth), f"{it.category} of {it.image.name}: {it.caption[:60]}"))
        text = (f"{n_face} of {len(items)} images have a face; crop_augment={cfg.crop_augment} would add "
                f"{sum(1 for c in crops if c.category == 'face_crop')} face and "
                f"{sum(1 for c in crops if c.category == 'bust_crop')} bust crops.")
    else:
        for i, it in enumerate(items[:12]):
            rec = analysis.get(str(it.image.resolve()))
            bbox = R.remap_bbox(rec["bbox"], rec["W"], rec["H"], it.width, it.height) if rec and rec.get("bbox") else None
            rm = R.build_weight_map(it.width, it.height, bbox, None, cfg.face_loss_weight, cfg.body_loss_weight,
                                    cfg.background_loss_weight, cfg.face_margin, cfg.region_feather_cells)
            img = load_image_fit(it.image, it.width, it.height)
            ov = R.render_overlay(img, rm, it.width, it.height)
            ov.thumbnail((256, 256))
            pth = out_dir / f"region_{i}.jpg"
            ov.save(pth, "JPEG", quality=85)
            thumbs.append((str(pth), f"{it.image.name}: {'face' if rm.face_cells else 'no face'}"))
        text = (f"{n_face} of {len(items)} images have a face. Overlay: brighter red = higher loss weight "
                f"(face {cfg.face_loss_weight:g} / background {cfg.background_loss_weight:g}; person masks are "
                f"built during training).")
    return text, thumbs


def gpu_stats() -> list[dict]:
    """Per-GPU utilization and memory via nvidia-smi (no CUDA context in the UI process)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3).stdout
    except Exception:  # noqa: BLE001
        return []
    stats = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            stats.append({"index": int(parts[0]), "name": parts[1], "util": int(float(parts[2])),
                          "used": float(parts[3]), "total": float(parts[4])})
        except ValueError:
            continue
    return stats


# ---------------------------------------------------------------- UI


class RaceRows:
    """The configuration race's rows: a name and the settings that differ from the rest of the page. .value is the
    race_configs list, so the Form reads and fills it like any other widget."""

    def __init__(self, on_change=None):
        self.rows: list[tuple] = []  # (row element, name input, settings input, check label)
        self.on_change = on_change
        self.box = ui.column().classes("w-full gap-1")

    @property
    def value(self) -> list[dict]:
        from rulztrainer import race

        out = []
        for _, name, text, _ in self.rows:
            over = race.parse_overrides(text.value or "")
            out.append({"name": (name.value or "").strip() or f"c{len(out) + 1}", **over})
        return out

    @value.setter
    def value(self, rows) -> None:
        from rulztrainer import race

        for row, _, _, check in self.rows:
            row.delete()
            check.delete()
        self.rows = []
        for r in rows or []:
            self.add(str(r.get("name") or ""), race.format_overrides(race.row_overrides(r)))
        self.changed()

    def add(self, name: str = "", text: str = "") -> None:
        with self.box:
            with ui.row().classes("w-full items-center gap-2 no-wrap") as row:
                n = ui.input("Name", value=name or f"c{len(self.rows) + 1}").classes("w-44")
                t = ui.input("Settings that differ: setting=value, ...", value=text).classes("flex-grow")
                ui.button(icon="close", on_click=lambda _, r=row: self.remove(r)).props("flat round dense") \
                    .tooltip("Remove this configuration")
            check = ui.label("").classes("text-xs rz-hint")
        self.rows.append((row, n, t, check))
        for widget in (n, t):
            widget.on_value_change(lambda _: self.changed())
        self.changed()

    def remove(self, row) -> None:
        for entry in list(self.rows):
            if entry[0] is row:
                entry[0].delete()
                entry[3].delete()
                self.rows.remove(entry)
        self.changed()

    def changed(self) -> None:
        """Check every row as it is typed: unreadable text, settings a race shares, or what the row changes."""
        from rulztrainer import race

        for _, _, text, check in self.rows:
            try:
                over = race.parse_overrides(text.value or "")
            except ValueError as e:
                check.text = f"Cannot read this: {e}"
                continue
            bad = [k for k in over if k not in race.ALLOWED]
            check.text = (f"A race shares these, so a configuration cannot change them: {', '.join(bad)}" if bad else
                          f"Changes {', '.join(over)}" if over else "The page's own settings")
        if self.on_change is not None:
            self.on_change()


class Form:
    """Holds the NiceGUI widgets and converts to/from TrainConfig."""

    def __init__(self):
        self.w: dict[str, object] = {}
        self.preset_sync = False  # True while from_config moves the preset switch (its on_change must not re-apply)
        self.loading = False  # True while from_config fills the widgets
        self.target_family = TrainConfig().arch_name()  # the family whose default LoRA targets the boxes started from

    def value(self, key):
        return self.w[key].value

    def ticked_targets(self) -> set[str]:
        return {t for t in TARGETS + list(EXTRA_TARGETS)
                if self.w.get(f"target_{t}") is not None and self.w[f"target_{t}"].value}

    def follow_family_targets(self) -> None:
        """On a model change, tick the new family's default LoRA targets, unless the user changed the ticks from
        the previous family's defaults (their own choice stays)."""
        family = preset_arch(self.w["model_preset"].value, self.w["model_path"].value)
        if family == self.target_family:
            return
        if not self.loading and self.ticked_targets() == set(ARCH_DEFAULT_TARGETS[self.target_family]):
            for t in TARGETS + list(EXTRA_TARGETS):
                if self.w.get(f"target_{t}") is not None:
                    self.w[f"target_{t}"].value = t in ARCH_DEFAULT_TARGETS[family]
        # the sample steps / CFG likewise: each family samples at its own (Krea 2 Raw at CFG 5.5, MOSS-Image at 3)
        steps, cfg = self.w.get("sample_steps"), self.w.get("sample_cfg")
        if not self.loading and steps is not None and cfg is not None:
            try:
                current = (int(steps.value), float(cfg.value))
            except (TypeError, ValueError):
                current = None
            if current == ARCH_SAMPLE_DEFAULTS[self.target_family]:
                steps.value, cfg.value = ARCH_SAMPLE_DEFAULTS[family]
        self.target_family = family

    def to_config(self) -> TrainConfig:
        v = self.value
        preset = v("model_preset")
        model_path = (v("model_path") or "").strip() if preset == CUSTOM else MODEL_PRESETS[preset]
        prompts = [p.strip() for p in (v("sample_prompts") or "").splitlines() if p.strip()]
        s = lambda k: (v(k) or "").strip()  # noqa: E731
        return TrainConfig(
            preset="identity" if self.w.get("preset_toggle") is not None and self.w["preset_toggle"].value else "",
            job_name=s("job_name") or "my_lora", output_dir=s("output_dir") or "output", seed=int(v("seed")),
            model_source="files" if v("model_source") == FILE_SOURCE else "hf",
            model_path=model_path, transformer_file=s("transformer_file"), text_encoder_file=s("text_encoder_file"),
            vae_file=s("vae_file"), tokenizer_path=s("tokenizer_path"),
            device=v("device"), text_encoder_device=v("text_encoder_device"),
            multi_gpu=v("multi_gpu"), devices=(v("devices") or "").strip(), split_plan=(v("split_plan") or "").strip(),
            parallel_jobs=bool(v("parallel_jobs")), dp_balance=v("dp_balance"),
            transformer_quant=v("transformer_quant"), text_encoder_quant=v("text_encoder_quant"),
            gradient_checkpointing=bool(v("gradient_checkpointing")),
            checkpoint_keep_blocks=int(v("checkpoint_keep_blocks")),
            int8_matmul=bool(v("int8_matmul")), int8_backward=v("int8_backward"),
            nvfp4_matmul=bool(v("nvfp4_matmul")), nvfp4_backward=v("nvfp4_backward"),
            attention_backend=v("attention_backend"),
            dataset_dir=s("dataset_dir"), resolutions=sorted(float(x) for x in (v("resolutions") or [1.0])),
            resolution_step_up=bool(v("resolution_step_up")), step_up_patience=int(v("step_up_patience")),
            step_up_stop_patience=int(v("step_up_stop_patience")),
            enable_buckets=bool(v("enable_buckets")), flip_augment=bool(v("flip_augment")),
            trigger_word=s("trigger_word"), caption_dropout=float(v("caption_dropout")),
            caption_dropout_keep_trigger=bool(v("caption_dropout_keep_trigger")),
            max_token_length=int(v("max_token_length")), recache=bool(v("recache")),
            lora_rank=int(v("lora_rank")), lora_alpha=float(v("lora_alpha")),
            lora_targets=[t for t in TARGETS + list(EXTRA_TARGETS) if self.w.get(f"target_{t}") is not None
                          and self.w[f"target_{t}"].value],
            train_text_embedder=bool(v("train_text_embedder")), resume_from=s("resume_from"),
            lora_blocks=s("lora_blocks"),
            lora_save_dtype=v("lora_save_dtype"), lora_ema=float(v("lora_ema") or 0),
            save_state=bool(v("save_state")), resume_state=s("resume_state"),
            save_state_every=int(v("save_state_every")), keep_states=int(v("keep_states")),
            compile_model=v("compile_model"), sample_seed=int(v("sample_seed")),
            sample_walk_seed=bool(v("sample_walk_seed")), sample_before_training=bool(v("sample_before_training")),
            steps=int(v("steps")), batch_size=int(v("batch_size")),
            gradient_accumulation=int(v("gradient_accumulation")), learning_rate=float(v("learning_rate")),
            optimizer=v("optimizer"), lr_scheduler=v("lr_scheduler"), warmup_steps=int(v("warmup_steps")),
            max_grad_norm=float(v("max_grad_norm")), timestep_sampling=v("timestep_sampling"),
            automagic_min_lr=float(v("automagic_min_lr")), automagic_max_lr=float(v("automagic_max_lr")),
            automagic_polarity_history=int(v("automagic_polarity_history")), automagic_groups=v("automagic_groups"),
            automagic_clip_threshold=float(v("automagic_clip_threshold")),
            timestep_shift=float(v("timestep_shift")), sigmoid_scale=float(v("sigmoid_scale")),
            logit_mean=float(v("logit_mean")), logit_std=float(v("logit_std")), loss_weighting=v("loss_weighting"),
            save_every=int(v("save_every")), sample_every=int(v("sample_every")), sample_prompts=prompts,
            sample_width=int(v("sample_width")), sample_height=int(v("sample_height")),
            sample_steps=int(v("sample_steps")), sample_cfg=float(v("sample_cfg")),
            sample_shift=float(v("sample_shift")), sample_negative=v("sample_negative") or "",
            turbo_lora=s("turbo_lora"), turbo_lora_strength=float(v("turbo_lora_strength")),
            sample_model_path=s("sample_model_path"), sample_model_quant=v("sample_model_quant"),
            sample_lora=s("sample_lora"), sample_lora_strength=float(v("sample_lora_strength")),
            face_refs=[p.strip() for p in (v("face_refs") or "").splitlines() if p.strip()],
            face_provider=v("face_provider"), face_det_size=int(v("face_det_size")),
            body_refs=[p.strip() for p in (v("body_refs") or "").splitlines() if p.strip()],
            face_weight=float(v("face_weight")), body_weight=float(v("body_weight")),
            body_backend=v("body_backend"), sam3_model=(v("sam3_model") or "facebook/sam3").strip(),
            body_device=v("body_device"),
            probe_every=int(v("probe_every")), probe_at_start=bool(v("probe_at_start")),
            probe_face_prompt=s("probe_face_prompt"), probe_body_prompt=s("probe_body_prompt"),
            probe_seeds=int(v("probe_seeds")), probe_seed=int(v("probe_seed")), probe_size=int(v("probe_size")),
            probe_steps=int(v("probe_steps")), probe_cfg=float(v("probe_cfg")), probe_ema=float(v("probe_ema")),
            likeness_noise_band=float(v("likeness_noise_band")), likeness_track_best=bool(v("likeness_track_best")),
            likeness_normalize=bool(v("likeness_normalize")),
            likeness_patience=int(v("likeness_patience")), likeness_lr_control=bool(v("likeness_lr_control")),
            likeness_lr_min=float(v("likeness_lr_min")), likeness_lr_max=float(v("likeness_lr_max")),
            likeness_lr_step=float(v("likeness_lr_step")), likeness_curriculum=bool(v("likeness_curriculum")),
            likeness_curriculum_strength=float(v("likeness_curriculum_strength")),
            loss_watch=bool(v("loss_watch")), per_image_lr=bool(v("per_image_lr")),
            loss_watch_buckets=int(v("loss_watch_buckets")), loss_watch_beta=float(v("loss_watch_beta")),
            loss_watch_high=float(v("loss_watch_high")), loss_watch_low=float(v("loss_watch_low")),
            loss_watch_improve_tol=float(v("loss_watch_improve_tol")),
            loss_watch_healthy_mult=float(v("loss_watch_healthy_mult")),
            loss_watch_exclude_after=int(v("loss_watch_exclude_after")),
            region_weights=bool(v("region_weights")), face_loss_weight=float(v("face_loss_weight")),
            body_loss_weight=float(v("body_loss_weight")), background_loss_weight=float(v("background_loss_weight")),
            face_margin=float(v("face_margin")), region_feather_cells=int(v("region_feather_cells")),
            region_prior_weight=float(v("region_prior_weight")),
            id_loss_weight=float(v("id_loss_weight")), id_loss_sigma_max=float(v("id_loss_sigma_max")),
            id_loss_every=int(v("id_loss_every")), id_loss_backend=v("id_loss_backend"),
            id_loss_margin_cells=int(v("id_loss_margin_cells")),
            id_loss_min_face_cells=int(v("id_loss_min_face_cells")), id_loss_random_ref=bool(v("id_loss_random_ref")),
            content_power=float(v("content_power")), sigma_min=float(v("sigma_min")), sigma_max=float(v("sigma_max")),
            lora_plus_ratio=float(v("lora_plus_ratio")),
            crop_augment=v("crop_augment"), crop_augment_min_px=int(v("crop_augment_min_px")),
            crop_face_scale=float(v("crop_face_scale")), crop_bust_scale=float(v("crop_bust_scale")),
            crop_skip_face_frac=float(v("crop_skip_face_frac")), crop_skip_cover=float(v("crop_skip_cover")),
            crop_face_caption=v("crop_face_caption") or "", crop_bust_caption=v("crop_bust_caption") or "",
            crop_augment_weight=float(v("crop_augment_weight")),
            race=bool(v("race")), race_warmup_probes=int(v("race_warmup_probes") or 0),
            race_configs=self.w["race_configs"].value,
        )

    def from_config(self, cfg: TrainConfig) -> None:
        preset = next((k for k, v in MODEL_PRESETS.items() if v == cfg.model_path), CUSTOM)
        values = {**cfg.to_dict(), "model_preset": preset, "sample_prompts": "\n".join(cfg.sample_prompts),
                  "face_refs": "\n".join(cfg.face_refs), "body_refs": "\n".join(cfg.body_refs),
                  "model_source": FILE_SOURCE if cfg.model_source == "files" else HF_SOURCE}
        targets = cfg.resolved_lora_targets()  # an empty list means the family's defaults
        for t in TARGETS + list(EXTRA_TARGETS):
            values[f"target_{t}"] = t in targets
        if not cfg.resolutions:  # legacy single `resolution`: snap to the nearest megapixel choice
            values["resolutions"] = [min(MEGAPIXEL_CHOICES, key=lambda m: abs(m - side_to_mp(cfg.resolution)))]
        self.loading = True
        try:
            for key, widget in self.w.items():
                if key in values:
                    widget.value = values[key]
        finally:
            self.loading = False
        self.target_family = cfg.arch_name()
        toggle = self.w.get("preset_toggle")
        if toggle is not None and toggle.value != (cfg.preset == "identity"):
            # the switch mirrors the loaded config's preset in both directions; setting .value fires on_change,
            # so flag it as a sync and not a user click (no re-apply, no restore)
            self.preset_sync = True
            try:
                toggle.value = cfg.preset == "identity"
            finally:
                self.preset_sync = False


# ---------------------------------------------------------------- look and feel

ASSETS = ROOT / "assets"
RED = "#E10600"
CHART_COLORS = ["#FF3B30", "#F4F4F5", "#F59E0B", "#A1A1AA", "#FB7185", "#60A5FA", "#34D399", "#C084FC"]

THEME_CSS = """
:root {
  --rz-red: #E10600; --rz-red-bright: #FF3B30; --rz-red-soft: rgba(225, 6, 0, 0.14);
  --rz-bg: #0A0A0A; --rz-surface: #131313; --rz-surface-2: #1A1A1A;
  --rz-border: #262626; --rz-border-strong: #3A3A3A;
  --rz-text: #EDEDED; --rz-muted: #A1A1AA; --rz-ok: #22C55E;
}
body, .q-page-container, .q-layout { background: var(--rz-bg); }
body {
  color: var(--rz-text); font-size: 15px; line-height: 1.5; -webkit-font-smoothing: antialiased;
  font-family: "Inter", "Segoe UI", system-ui, -apple-system, "Helvetica Neue", Roboto, Arial, sans-serif;
}
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-thumb { background: #2A2A2A; border-radius: 10px; }
::-webkit-scrollbar-track { background: transparent; }

/* header: logo, status pill, tabs */
.nicegui-header.rz-header {
  background: #000; border-bottom: 1px solid var(--rz-border); padding: 0; gap: 0;
  flex-direction: column; align-items: stretch;
}
.rz-header-row { max-width: 1440px; margin: 0 auto; padding: 12px 20px 2px; }
.rz-logo { height: 30px; width: auto; display: block; }
.rz-wordmark { letter-spacing: .34em; font-weight: 600; font-size: 12px; color: var(--rz-muted); padding-top: 7px; }
.rz-tagline { color: #71717A; font-size: 13px; padding-top: 6px; margin-left: 8px; }
.rz-tabs { max-width: 1440px; margin: 0 auto; width: 100%; padding: 0 8px; }
.rz-tabs .q-tab { color: var(--rz-muted); min-height: 44px; padding: 0 14px; }
.rz-tabs .q-tab__label { font-weight: 500; font-size: 14px; }
.rz-tabs .q-tab__icon { font-size: 19px; }
.rz-tabs .q-tab--active { color: #fff; }
.rz-tabs .q-tab--active .q-tab__icon { color: var(--rz-red-bright); }
.rz-tabs .q-tab__indicator { background: var(--rz-red); height: 3px; border-radius: 3px 3px 0 0; }
.rz-pill {
  border: 1px solid var(--rz-border-strong); border-radius: 999px; padding: 4px 12px; font-size: 13px;
  background: #0D0D0D; cursor: pointer; max-width: 320px;
}
.rz-pill:hover { border-color: #52525B; }
.rz-pill-text { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.rz-dot { width: 8px; height: 8px; border-radius: 50%; background: #52525B; display: inline-block; flex-shrink: 0; }
.rz-dot.live { background: var(--rz-red-bright); animation: rz-pulse 1.6s infinite; }
.rz-dot.done { background: var(--rz-ok); }
.rz-dot.error { background: #F59E0B; }
@keyframes rz-pulse {
  0% { box-shadow: 0 0 0 0 rgba(255, 59, 48, .6); }
  70% { box-shadow: 0 0 0 8px rgba(255, 59, 48, 0); }
  100% { box-shadow: 0 0 0 0 rgba(255, 59, 48, 0); }
}

/* page and panels (visible overflow so the section list can stay in view) */
.rz-panels { background: transparent; overflow: visible !important; }
.rz-panels > .q-panel, .rz-panels > .q-panel > div { overflow: visible; }
.rz-panels .q-tab-panel { padding: 20px 20px 28px; }
.rz-page { width: 100%; max-width: 1440px; margin: 0 auto; }

/* cards and settings sections */
.rz-card { background: var(--rz-surface); border: 1px solid var(--rz-border); border-radius: 14px; padding: 18px 20px; }
.rz-section { background: var(--rz-surface); border: 1px solid var(--rz-border); border-radius: 14px; scroll-margin-top: 104px; }
.rz-section > .q-expansion-item__container > .q-item { padding: 12px 18px; min-height: 64px; border-radius: 14px; }
.rz-section > .q-expansion-item__container > .q-item .q-item__section--avatar { min-width: 0; padding-right: 14px; }
.rz-section > .q-expansion-item__container > .q-item .q-item__section--avatar .q-icon {
  color: var(--rz-red-bright); background: var(--rz-red-soft); border-radius: 10px; font-size: 20px;
  width: 38px; height: 38px;
}
.rz-section > .q-expansion-item__container > .q-item .q-item__label { font-size: 16px; font-weight: 600; color: #fff; }
.rz-section > .q-expansion-item__container > .q-item .q-item__label--caption {
  font-size: 13px; color: var(--rz-muted); font-weight: 400; margin-top: 2px;
}
.rz-section .nicegui-expansion-content { padding: 2px 18px 20px; gap: 14px; }
.rz-advanced { border-top: 1px dashed var(--rz-border); }
.rz-advanced > .q-expansion-item__container > .q-item { padding: 10px 0 4px; min-height: 0; color: #C4C4CC; font-weight: 500; }
.rz-advanced > .q-expansion-item__container > .q-item .q-item__section--avatar { min-width: 0; padding-right: 8px; }
.rz-advanced > .q-expansion-item__container > .q-item .q-icon { color: var(--rz-muted); font-size: 18px; }
.rz-section .rz-advanced .nicegui-expansion-content { padding: 8px 0 6px; gap: 14px; }
.rz-title { font-size: 16px; font-weight: 600; color: #fff; }
.rz-step-title { font-size: 20px; font-weight: 700; color: #fff; }
.rz-subhead { font-size: 12px; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; color: var(--rz-muted); margin-top: 4px; }
.rz-hint { font-size: 13px; color: var(--rz-muted); line-height: 1.55; max-width: 980px; }
.rz-warn { color: #FBBF24; }
.rz-fields { gap: 14px 16px; align-items: center; flex-wrap: wrap; }

/* form fields and buttons */
.q-field--outlined .q-field__control { border-radius: 9px; }
.q-field--outlined.q-field--dark .q-field__control:before { border-color: var(--rz-border-strong); }
.q-field--dark .q-field__label { color: #B5B5BD; }
.q-field--dark .q-field__native, .q-field--dark .q-field__input, .q-field--dark .q-select__dropdown-icon { color: #F4F4F5; }
.q-btn { text-transform: none; font-weight: 600; letter-spacing: .01em; border-radius: 9px; }
.q-toggle__label, .q-checkbox__label, .q-radio__label { color: #E4E4E7; }
.q-tooltip { background: #1F1F1F; border: 1px solid var(--rz-border-strong); color: #F4F4F5; font-size: 13px; max-width: 440px; line-height: 1.45; }
.q-menu { background: #161616; border: 1px solid var(--rz-border-strong); }

/* Train tab: section list on the left */
.rz-nav { position: sticky; top: 108px; width: 210px; flex-shrink: 0; gap: 2px; }
.rz-nav-title { font-size: 12px; text-transform: uppercase; letter-spacing: .08em; color: var(--rz-muted); padding: 2px 12px 6px; }
.rz-nav-btn { width: 100%; color: #D4D4D8; font-weight: 500; }
.rz-nav-btn .q-btn__content { justify-content: flex-start; }
.rz-nav-btn .q-icon { color: var(--rz-red-bright); font-size: 18px; }

/* action bar */
.nicegui-footer.rz-footer {
  background: rgba(0, 0, 0, .94); backdrop-filter: blur(6px); border-top: 1px solid var(--rz-border); padding: 0;
}
.rz-footer-row { max-width: 1440px; margin: 0 auto; padding: 10px 20px; }
.rz-status-msg {
  font-size: 13px; color: #C4C4CC; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;
  overflow: hidden;
}

/* Monitor */
.rz-tile { background: var(--rz-surface); border: 1px solid var(--rz-border); border-radius: 12px; padding: 12px 16px; min-width: 150px; flex: 1 1 150px; }
.rz-tile-label { font-size: 11px; color: var(--rz-muted); text-transform: uppercase; letter-spacing: .08em; }
.rz-tile-value { font-size: 20px; font-weight: 600; color: #fff; font-variant-numeric: tabular-nums; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 100%; }
.rz-tile-sub { font-size: 12px; color: var(--rz-muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 100%; }
.rz-banner { border-radius: 10px; padding: 10px 14px; font-size: 14px; width: 100%; }
.rz-banner.error { background: rgba(225, 6, 0, .12); border: 1px solid rgba(255, 59, 48, .45); color: #FFB4AE; }
.rz-banner.warning { background: rgba(245, 158, 11, .10); border: 1px solid rgba(245, 158, 11, .40); color: #FCD34D; }
.rz-banner.info { background: #161616; border: 1px solid var(--rz-border-strong); color: #D4D4D8; }
.rz-banner.ok { background: rgba(34, 197, 94, .10); border: 1px solid rgba(34, 197, 94, .40); color: #86EFAC; }
.q-uploader { background: var(--rz-surface); border: 1px dashed var(--rz-border-strong); border-radius: 12px; }
.rz-log { background: #050505; border: 1px solid var(--rz-border); border-radius: 10px; color: #D4D4D8; }
.rz-thumb { background: var(--rz-surface-2); border: 1px solid var(--rz-border); border-radius: 10px; overflow: hidden; }
.rz-mono { font-family: ui-monospace, "Cascadia Mono", Consolas, Menlo, monospace; font-size: 12px; color: #C4C4CC; }

/* Wizard */
.rz-stepper { background: transparent; box-shadow: none; }
.rz-stepper .q-stepper__header { border-bottom: 1px solid var(--rz-border); }
.rz-stepper .q-stepper__tab { padding: 14px 8px; }
.rz-stepper .q-stepper__title { font-size: 13px; font-weight: 500; color: #D4D4D8; }
.rz-stepper .q-stepper__tab--active .q-stepper__title { color: #fff; font-weight: 600; }
.rz-stepper .q-stepper__dot { width: 30px; min-width: 30px; height: 30px; }
.rz-stepper .q-stepper__tab:not(.q-stepper__tab--active):not(.q-stepper__tab--done) .q-stepper__dot { background: #2A2A2A; }
.rz-stepper .q-stepper__step-inner { padding: 22px 4px 8px; }
.rz-stepper .q-stepper__nav { padding: 18px 4px 4px; border-top: 1px solid var(--rz-border); margin-top: 8px; width: 100%; }
.rz-setting {
  display: grid; grid-template-columns: minmax(0, 1fr) minmax(240px, 380px); gap: 10px 32px; width: 100%;
  padding: 18px 0; border-bottom: 1px solid var(--rz-border); align-items: start;
}
.rz-setting-title { font-weight: 600; font-size: 15px; color: #fff; }
.rz-rec { font-size: 13px; color: #FCA5A5; margin-top: 2px; }
.rz-rec .q-icon { color: var(--rz-red-bright); font-size: 17px; }
.rz-rec.ok { color: #86EFAC; }
.rz-rec.ok .q-icon { color: var(--rz-ok); }
.rz-apply { background: var(--rz-red-soft); border: 1px solid rgba(255, 59, 48, .35); border-radius: 12px; padding: 10px 14px; }
.rz-choices { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 14px; width: 100%; }
.rz-choice {
  background: var(--rz-surface-2); border: 1px solid var(--rz-border-strong); border-radius: 12px; padding: 16px;
  cursor: pointer; transition: border-color .15s, background .15s; gap: 8px;
}
.rz-choice:hover { border-color: #71717A; }
.rz-choice.selected {
  border-color: var(--rz-red); background: rgba(225, 6, 0, .16);
  box-shadow: inset 0 0 0 1px var(--rz-red), 0 0 0 3px rgba(225, 6, 0, .18);
}
.rz-choice-icon { color: var(--rz-red-bright); font-size: 24px; }
.rz-check { display: none; color: var(--rz-red-bright); font-size: 22px; }
.rz-choice.selected .rz-check { display: inline-flex; }
.rz-badge { font-size: 11px; font-weight: 600; letter-spacing: .04em; padding: 1px 8px; border-radius: 999px; background: var(--rz-red); color: #fff; }
.rz-badge.muted { background: #3F3F46; }
.rz-summary { display: grid; grid-template-columns: max-content minmax(0, 1fr); gap: 8px 24px; width: 100%; font-size: 14px; }
.rz-summary .k { color: var(--rz-muted); }
.rz-summary .v { color: #F4F4F5; overflow-wrap: anywhere; }
@media (max-width: 760px) {
  .rz-setting { grid-template-columns: 1fr; }
  .rz-panels .q-tab-panel { padding: 14px 12px 24px; }
  .rz-footer-row, .rz-header-row { padding-left: 12px; padding-right: 12px; }
}

/* Help */
.rz-help { max-width: 920px; }
.rz-help h2 { font-size: 20px; font-weight: 700; color: #fff; margin: 28px 0 10px; padding-bottom: 6px; border-bottom: 1px solid var(--rz-border); }
.rz-help h2:first-child { margin-top: 0; }
.rz-help p, .rz-help li { color: #D4D4D8; line-height: 1.65; }
.rz-help strong { color: #fff; }
.rz-help code { background: #1F1F1F; color: #FCA5A5; padding: 1px 6px; border-radius: 5px; font-size: 13px; }
.rz-help pre { background: #050505; border: 1px solid var(--rz-border); border-radius: 10px; padding: 12px 14px; overflow-x: auto; }
.rz-help pre code { background: transparent; color: #E4E4E7; padding: 0; }
.rz-help a { color: var(--rz-red-bright); }
"""


def setup_theme() -> None:
    """Once per process, before the first page: logo files, the stylesheet and outlined form fields."""
    app.add_static_files("/assets", ASSETS)
    ui.add_css(THEME_CSS, shared=True)
    for element in (ui.input, ui.number, ui.textarea):
        element.default_props("outlined dense")
    ui.select.default_props("outlined dense options-dense")


def theme_colors() -> None:
    ui.colors(primary=RED, secondary="#27272A", accent="#FF3B30", dark="#131313", dark_page="#0A0A0A",
              positive="#22C55E", negative="#FF453A", info="#3F3F46", warning="#F59E0B")


def section(title: str, caption: str, icon: str, open_: bool = True):
    """A collapsible settings card (Train and Monitor tabs)."""
    return ui.expansion(title, caption=caption, icon=icon, value=open_).classes("rz-section w-full")


def advanced(title: str = "Advanced options"):
    """Rarely changed options inside a section, folded away by default."""
    return ui.expansion(title, icon="tune").props("dense").classes("rz-advanced w-full")


def fields():
    """A wrapping row of form fields."""
    return ui.row().classes("rz-fields w-full")


def hint(text: str = "", warn: bool = False):
    return ui.label(text).classes("rz-hint" + (" rz-warn" if warn else ""))


def subhead(text: str):
    return ui.label(text).classes("rz-subhead")


def card():
    return ui.card().props("flat").classes("rz-card w-full")


def stat_tile(label: str):
    """A Monitor tile: small caption, big value, sub line. Returns (value label, sub label)."""
    with ui.column().classes("rz-tile gap-0"):
        ui.label(label).classes("rz-tile-label")
        value = ui.label("-").classes("rz-tile-value")
        sub = ui.label("").classes("rz-tile-sub")
    return value, sub


def _axis(name: str, **kw) -> dict:
    return {"type": "value", "name": name, "axisLine": {"lineStyle": {"color": "#3F3F46"}},
            "axisLabel": {"color": "#A1A1AA"}, "splitLine": {"lineStyle": {"color": "#222222"}},
            "nameTextStyle": {"color": "#A1A1AA"}, **kw}


def chart_options(x: dict, y, series: list, legend: list[str]) -> dict:
    """ECharts options in the dark theme: transparent background, muted axes, the brand palette first."""
    return {"backgroundColor": "transparent", "color": CHART_COLORS, "textStyle": {"color": "#D4D4D8"},
            "xAxis": x, "yAxis": y, "series": series,
            "legend": {"data": legend, "type": "scroll", "top": 0, "textStyle": {"color": "#D4D4D8"},
                       "pageTextStyle": {"color": "#A1A1AA"}},
            "tooltip": {"trigger": "axis", "backgroundColor": "#18181B", "borderColor": "#3F3F46",
                        "textStyle": {"color": "#F4F4F5"}},
            "grid": {"left": 60, "right": 60, "top": 40, "bottom": 40}}


def _line(name: str, color: str, **kw) -> dict:
    return {"name": name, "type": "line", "showSymbol": False, "data": [], "itemStyle": {"color": color},
            "lineStyle": {"color": color, **kw.pop("lineStyle", {})}, **kw}


def path_field(label: str, kind: str, value: str = "", width: str = "w-96", filetypes=None):
    """Text input with a Browse button: a native dialog for a viewer at this machine's desktop, the in-page browser of
    the server's files otherwise (a vast.ai instance, a remote browser). width "w-full" makes the input fill its
    container."""
    full = width == "w-full"
    native = _native_dialogs  # this page's viewer, read while the page is built
    with ui.row().classes("items-center gap-1 no-wrap max-w-full" + (" w-full" if full else "")):
        inp = ui.input(label, value=value).classes("flex-grow min-w-0" if full else f"{width} min-w-0")

        async def browse():
            picked = await pick_path(kind, inp.value or "", filetypes, native)
            if picked:
                inp.value = _relative(picked)

        ui.button(icon="folder_open" if kind == "dir" else "description", on_click=browse) \
            .props("flat dense round").tooltip(BROWSE_TIP[native])
    return inp


def path_input(f: Form, key: str, label: str, kind: str, value: str = "", width: str = "w-96", filetypes=None):
    """A path_field registered in the form under `key`."""
    f.w[key] = path_field(label, kind, value, width, filetypes)
    return f.w[key]


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
PAGE_SIZE = 24
THUMB_SIZE = 512


def list_dataset_images(folder: str) -> list[Path]:
    root = Path(folder)
    if not root.is_dir():
        return []
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS and ".rulz_cache" not in p.parts)


def dataset_summary(folder: str) -> dict | None:
    """Image count, captions and typical image size of a dataset folder, for the wizard. None if it does not exist."""
    from PIL import Image

    if not folder or not Path(folder).is_dir():
        return None
    images = list_dataset_images(folder)
    captioned = sum(1 for p in images if p.with_suffix(".txt").is_file() or p.with_suffix(".json").is_file())
    sizes = []
    for p in images[:300]:  # the header is enough for the size
        try:
            with Image.open(p) as im:
                sizes.append(im.size[0] * im.size[1] / 2 ** 20)
        except Exception:  # noqa: BLE001
            continue
    sizes.sort()
    return {"folder": folder, "images": len(images), "captioned": captioned,
            "median_mp": sizes[len(sizes) // 2] if sizes else None, "sample": images[:8]}


def make_thumbs(images: list[Path], root: Path) -> list[Path]:
    """Create (or reuse) small JPEG thumbnails under <dataset>/.rulz_cache/thumbs. Runs in a worker thread."""
    import hashlib

    from PIL import Image, ImageOps

    tdir = root / ".rulz_cache" / "thumbs"
    tdir.mkdir(parents=True, exist_ok=True)
    out = []
    for img in images:
        try:
            st = img.stat()
            key = hashlib.sha1(f"{img.resolve()}|{st.st_mtime_ns}|{st.st_size}|sq".encode()).hexdigest()[:16]
            t = tdir / f"{key}.jpg"
            if not t.exists():
                with Image.open(img) as im:
                    im = ImageOps.exif_transpose(im).convert("RGB")
                    im = ImageOps.fit(im, (THUMB_SIZE, THUMB_SIZE), Image.Resampling.LANCZOS)  # square center crop
                    im.save(t, "JPEG", quality=85)
            out.append(t)
        except Exception:  # noqa: BLE001
            out.append(img)
    return out


def image_viewer():
    """Full-window image viewer dialog. Returns show(path) callable."""
    with ui.dialog().props("maximized") as dlg, ui.card().tight().classes("w-full h-full bg-black"):
        with ui.row().classes("w-full items-center justify-between px-3 py-1"):
            title = ui.label("").classes("text-sm text-white truncate")
            ui.button(icon="close", on_click=dlg.close).props("flat dense round color=white")
        big = ui.image("").props("fit=contain").classes("w-full").style("height: calc(100vh - 48px)")
        big.on("click", dlg.close)

    def show(path: Path):
        title.text = str(path)
        big.set_source(str(path))
        dlg.open()

    return show


def build_dataset_tab(f: Form):
    """Upload images/captions, view them in a grid and edit captions (saved to .txt on change)."""
    state = {"page": 0, "images": [], "busy": False}
    with card():
        with fields():
            folder = ui.input("Dataset folder", value=f.w["dataset_dir"].value).classes("w-96 max-w-full")
            folder.bind_value(f.w["dataset_dir"], "value")  # keep in sync with the Train tab

            native = _native_dialogs

            async def browse():
                picked = await pick_path("dir", folder.value or "", native=native)
                if picked:
                    folder.value = _relative(picked)
                    await refresh()

            ui.button(icon="folder_open", on_click=browse).props("flat dense round").tooltip(BROWSE_TIP[native])
            ui.button("Load / refresh", icon="refresh", on_click=lambda: refresh()).props("unelevated")
            spinner = ui.spinner(size="sm")
            spinner.visible = False
            info = ui.label("").classes("text-sm")
        hint("Click a picture to see it full size. Captions are saved to the image's .txt file as you type. "
             "Face ref / Body ref pick the images that character LoRAs are scored against.")
    with ui.row().classes("w-full items-center gap-3"):
        def on_upload(e):
            dest = Path(folder.value or "datasets/new_dataset")
            dest.mkdir(parents=True, exist_ok=True)
            name = Path(e.name).name
            if Path(name).suffix.lower() not in IMAGE_EXTS | {".txt"}:
                ui.notify(f"Skipped {name}: not an image or .txt", type="warning")
                return
            (dest / name).write_bytes(e.content.read())

        ui.upload(label="Upload images and .txt captions (drop files here)", multiple=True, auto_upload=True,
                  on_upload=on_upload, on_multi_upload=lambda e: refresh()) \
            .props("accept=.png,.jpg,.jpeg,.webp,.bmp,.txt flat bordered color=grey-10").classes("w-[36rem] max-w-full")
        ui.space()
        with ui.row().classes("items-center gap-1 no-wrap"):
            ui.button(icon="chevron_left", on_click=lambda: page(-1)).props("flat dense round")
            page_label = ui.label("").classes("text-sm")
            ui.button(icon="chevron_right", on_click=lambda: page(1)).props("flat dense round")
            columns = ui.select([2, 3, 4, 5, 6], value=4, label="Columns").classes("w-28 ml-2")
    grid = ui.grid(columns=4).classes("w-full gap-3")
    show_full = image_viewer()

    def ref_set(key: str) -> set[str]:
        return {p.strip() for p in (f.w[key].value or "").splitlines() if p.strip()}

    def toggle_ref(key: str, path: Path, on: bool):
        refs = ref_set(key)
        rel = _relative(str(path))
        (refs.add if on else refs.discard)(rel)
        f.w[key].value = "\n".join(sorted(refs))

    def save_caption(path: Path, text: str, status_label):
        cap = path.with_suffix(".txt")
        try:
            cap.write_text(text, encoding="utf-8")  # creates the file if it did not exist
            status_label.text = "saved" if text.strip() else "saved (empty)"
        except OSError as e:
            status_label.text = f"error: {e}"
            ui.notify(f"Could not save {cap.name}: {e}", type="negative")

    async def render():
        imgs = state["images"]
        n_pages = max(1, (len(imgs) + PAGE_SIZE - 1) // PAGE_SIZE)
        state["page"] = min(state["page"], n_pages - 1)
        page_label.text = f"page {state['page'] + 1} / {n_pages}"
        start = state["page"] * PAGE_SIZE
        page_imgs = imgs[start:start + PAGE_SIZE]
        spinner.visible = True
        thumbs = await run.io_bound(make_thumbs, page_imgs, Path(folder.value))
        spinner.visible = False
        verdicts = read_verdicts(_job_dir)
        badge_color = {"stuck": "red", "excluded": "grey", "suspect": "orange", "exhausted": "blue", "easy": "green",
                       "healthy": "teal"}
        grid.style(f"grid-template-columns: repeat({columns.value}, minmax(0, 1fr))")
        grid.clear()
        with grid:
            for img, thumb in zip(page_imgs, thumbs):
                cap = img.with_suffix(".txt")
                text = cap.read_text(encoding="utf-8") if cap.is_file() else ""
                with ui.card().tight().classes("w-full rz-thumb"):
                    ui.image(str(thumb)).props("fit=cover ratio=1").classes("w-full cursor-pointer") \
                        .on("click", lambda _, p=img: show_full(p)).tooltip("Click to view full size")
                    with ui.row().classes("w-full items-center justify-between px-2 pt-1 gap-1"):
                        ui.label(img.name).classes("text-xs truncate")
                        vd = verdicts.get(str(img)) or verdicts.get(_relative(str(img)))
                        if vd and vd != "new":
                            ui.badge(vd, color=badge_color.get(vd, "grey")).props("dense").tooltip("loss watch verdict")
                        ui.checkbox("Face ref", value=_relative(str(img)) in ref_set("face_refs"),
                                    on_change=lambda e, p=img: toggle_ref("face_refs", p, e.value)).props("dense") \
                            .classes("text-xs").tooltip("Use this image's face as a similarity reference")
                        ui.checkbox("Body ref", value=_relative(str(img)) in ref_set("body_refs"),
                                    on_change=lambda e, p=img: toggle_ref("body_refs", p, e.value)).props("dense") \
                            .classes("text-xs").tooltip("Use this image's body proportions as a reference")
                        status_label = ui.label("" if cap.is_file() else "no caption yet").classes("text-xs text-grey")
                    ui.textarea(value=text, placeholder="type a caption; it is saved to the .txt automatically",
                                on_change=lambda e, p=img, sl=status_label: save_caption(p, e.value, sl)) \
                        .props("dense outlined autogrow").classes("w-full px-2 pb-2 text-xs")

    async def refresh():
        if state["busy"]:
            return
        state["busy"] = True
        try:
            state["images"] = await run.io_bound(list_dataset_images, folder.value or "")
            missing = sum(1 for p in state["images"] if not p.with_suffix(".txt").is_file())
            info.text = (f"{len(state['images'])} images, {missing} without caption" if state["images"]
                         else "No images found in that folder")
            await render()
        finally:
            state["busy"] = False

    async def page(delta: int):
        state["page"] = max(0, state["page"] + delta)
        await render()

    columns.on_value_change(lambda e: render())
    ui.timer(0.1, refresh, once=True)


def build_samples_tab(f: Form):
    """Live grid of generated sample images for the current (or any) job folder."""
    state = {"files": [], "columns": None}
    with card():
        with fields():
            folder = ui.input("Job folder (output/<job>)", value="").classes("w-96 max-w-full")

            native = _native_dialogs

            async def browse():
                picked = await pick_path("dir", folder.value or str(ROOT / "output"), native=native)
                if picked:
                    folder.value = _relative(picked)
                    state["files"] = []

            ui.button(icon="folder_open", on_click=browse).props("flat dense round").tooltip(BROWSE_TIP[native])
            follow = ui.switch("Follow current job", value=True)
            columns = ui.select([2, 3, 4, 6], value=4, label="Columns").classes("w-28")
            info = ui.label("").classes("rz-hint")
    grid = ui.grid(columns=4).classes("w-full gap-3")
    show_full = image_viewer()

    def samples_dir() -> Path | None:
        if follow.value:
            jd = _job_dir or (Path(f.w["output_dir"].value or "output") / (f.w["job_name"].value or "my_lora"))
            rel = _relative(str(jd))
            if folder.value != rel:
                folder.value = rel
        if not folder.value:
            return None
        p = Path(folder.value)
        return p if p.name == "samples" else p / "samples"

    def refresh():
        sd = samples_dir()
        files = sorted(sd.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True) if sd and sd.is_dir() else []
        sig = [(str(p), p.stat().st_mtime) for p in files]
        fs_file = (sd.parent / "face_scores.jsonl") if sd else None
        if fs_file is not None and fs_file.exists():
            sig.append(("face_scores", fs_file.stat().st_mtime))
        if sig == state["files"] and state["columns"] == columns.value:
            return
        state["files"] = sig
        state["columns"] = columns.value
        grid.style(f"grid-template-columns: repeat({columns.value}, minmax(0, 1fr))")
        info.text = f"{len(files)} samples in {sd}" if sd else "No job selected"
        scores = face_scores_by_image(sd.parent if sd else None)
        grid.clear()
        with grid:
            for p in files[:120]:
                with ui.card().tight().classes("w-full rz-thumb"):
                    ui.image(str(p)).props("fit=contain").classes("w-full h-64 bg-black cursor-pointer") \
                        .on("click", lambda _, q=p: show_full(q)).tooltip("Click to view full size")
                    sc = scores.get(p.name)
                    label = p.name + (score_caption(sc) if sc is not None else "")
                    ui.label(label).classes("text-xs px-2 py-1")

    columns.on_value_change(lambda e: (state.__setitem__("files", []), refresh()))
    ui.timer(3.0, refresh)
    refresh()


# ---------------------------------------------------------------- wizard

GOALS = {
    "character": ("face", "Character / person",
                  "A specific person or character. Turns on the measured identity recipe: region-weighted loss, face "
                  "and bust crops, and likeness probes that keep the best checkpoint and stop early."),
    "style": ("palette", "Style",
              "An art style, medium, lighting or look that should carry over to any subject."),
    "concept": ("category", "Concept / object",
                "An object, outfit, pose or idea. Also the right pick when you are not sure."),
}

MODEL_NOTES = {
    "MOSS-Image-Base": ("Recommended",
                        "OpenMOSS's 5B base model and the usual one to train on: its LoRAs also work on SFT and load in "
                        "ComfyUI with the standard Load LoRA node. About 30 GB to download."),
    "MOSS-Image-SFT": ("", "The fine-tuned MOSS-Image checkpoint. Train on it only if you will generate with SFT alone; "
                           "a Base LoRA can still preview its samples on SFT (Train tab, sampling model)."),
    "FIBO-2.0-dev": ("", "Bria FIBO 2.0 (8B), the full non-distilled model. Gated: accept its terms on Hugging Face and "
                         "run hf auth login first. LoRAs load in diffusers, also onto Edit Turbo; ComfyUI has no FIBO "
                         "support yet."),
    "FIBO-2.0-edit-turbo": ("Not for training", "The 4-step distilled edit model. Train on FIBO-2.0-dev instead: its "
                                                "LoRAs load onto Edit Turbo."),
    "Krea-2-Raw": ("", "Krea 2 (12B), the base checkpoint Krea publishes for training. Gated: accept its license on "
                       "Hugging Face and log in first. Its LoRAs are meant for Krea 2 Turbo, like Krea's own. About 35 GB "
                       "to download; needs a 24 GB card or two 16 GB cards for int8 (one 16 GB card: nvfp4)."),
    "Krea-2-Turbo": ("Not for training", "The 8-step distilled Krea 2. Train on Krea-2-Raw and use Turbo to preview "
                                         "and generate with the LoRA."),
}


def wizard_recommendations(cfg: TrainConfig, goal: str | None, gpus: list[dict], data: dict | None) -> dict[str, tuple]:
    """Recommended (value, reason) per setting key, from the goal, the GPUs (nvidia-smi rows) and the dataset scan."""
    r: dict[str, tuple] = {}
    fibo = cfg.arch_name() == "fibo2"
    krea = cfg.arch_name() == "krea2"
    krea_turbo = krea and "krea-2-turbo" in (cfg.sample_model_path or "").lower()
    character = goal == "character"
    vram = min(g["total"] for g in gpus) / 1024 if gpus else None
    gpu_name = "your smallest GPU" if len(gpus) > 1 else "your GPU"
    # hardware
    if len(gpus) >= 2 and krea and vram is not None and vram < 22:
        r["multi_gpu"] = ("model_split", f"{len(gpus)} GPUs of {vram:.0f} GB: split, the 12B model trains in int8; a "
                                         "full copy on each card would only fit in nvfp4")
    elif len(gpus) >= 2:
        r["multi_gpu"] = ("data_parallel", f"{len(gpus)} GPUs found: a model copy on each trains about 1.7-2x faster")
    if len(gpus) >= 2:
        r["dp_balance"] = ("auto", "measures the cards and splits each step's images by speed, so the faster one "
                                   "does not wait; the images per step stay the same")
    elif gpus:
        r["multi_gpu"] = ("off", "one GPU found")
    if cfg.model_source == "files":
        r["transformer_quant"] = ("auto", "keeps the format of your file (a ComfyUI int8 or nvfp4 file stays quantized)")
        r["text_encoder_quant"] = ("auto", "keeps the format of your file")
    elif vram is not None:
        if fibo:
            r["transformer_quant"] = ("auto", "FIBO switches itself to int8 on GPUs under 22 GB")
        elif krea:
            r["transformer_quant"] = ("auto", "Krea 2 picks for the memory it runs in: int8 (13.5 GB) from 22 GB, "
                                              "e.g. two 16 GB cards in model split, else nvfp4 (8 GB); bf16 is 25 GB")
        elif vram >= 22:
            r["transformer_quant"] = ("auto", f"{gpu_name} has {vram:.0f} GB, so the 10 GB bf16 model fits as it is")
        elif vram >= 11:
            r["transformer_quant"] = ("int8_convrot", f"{gpu_name} has {vram:.0f} GB: int8 halves the 10 GB model and "
                                                      "its W8A8 compute is as fast as bf16, at under 1% error")
        else:
            r["transformer_quant"] = ("nvfp4", f"{gpu_name} has {vram:.0f} GB: nvfp4 quarters the model so it fits "
                                               "(slower, fp4 weights are expanded every step)")
        te_gb = 9 if fibo or krea else 19
        if vram >= te_gb + 4:
            r["text_encoder_quant"] = ("auto", f"the {te_gb} GB text encoder fits on {gpu_name} as it is")
        else:
            r["text_encoder_quant"] = ("int8", f"halves the {te_gb} GB text encoder so it fits on the GPU for caching "
                                               "instead of spilling into system RAM")
    r["gradient_checkpointing"] = (True, "saves most of the activation memory for a modest slowdown")
    r["checkpoint_keep_blocks"] = (-1, "measured at the start: as many blocks as fit skip the recompute")
    r["compile_model"] = ("auto", "measured 26% (MOSS-Image) and 12% (Krea 2) faster per step after a warm-up of "
                                  "under a minute; auto skips it for short test runs")
    # dataset
    target = 1.0 if vram is None or vram >= 12 else 0.5
    reason = ("1 MP (1024x1024) is the size the model was trained at" if target == 1.0
              else f"0.5 MP keeps memory low on a {vram:.0f} GB GPU")
    if data and data.get("median_mp") and data["median_mp"] < target * 0.9:
        med = data["median_mp"]
        target = max(0.25, int(med * 4) / 4)
        reason = f"half of your images are under {med:.2f} MP; training above their size only upscales them"
    r["resolutions"] = ([target], reason)
    r["enable_buckets"] = (True, "keeps every image's aspect ratio instead of cropping it square")
    if character:
        r["flip_augment"] = (False, "faces are not symmetric (hair parting, moles, logos); mirrored copies blur them")
    elif goal == "style":
        r["flip_augment"] = (True, "a mirrored image has the same style, so this doubles the variety for free")
    else:
        r["flip_augment"] = (False, "only worth it if your subject looks the same mirrored (no text or logos)")
    if character and cfg.dataset_dir:
        r["face_refs"] = (cfg.dataset_dir, "every image with a face becomes a reference (a folder means all its images)")
    # training
    n = (data or {}).get("images") or 0
    if n:
        steps = int(min(4000, max(1000, round(n * 100, -2))))
        r["steps"] = (steps, f"about 100 steps per image for {n} images"
                             + ("; the likeness probes can stop it earlier" if character else ""))
    else:
        r["steps"] = (2000, "a good start; scan your dataset (Dataset step) for a number that fits it")
    r["learning_rate"] = (1e-4, "the usual LoRA learning rate; go lower (5e-5) if samples look burnt or noisy")
    r["optimizer"] = ("adamw8bit", "AdamW with 8-bit state: the same results as adamw with less memory")
    rank = 32 if goal == "style" else 16
    r["lora_rank"] = (rank, "styles carry a lot of varied detail" if goal == "style"
                      else "plenty for a character or concept; higher ranks mostly add file size")
    r["lora_alpha"] = (float(rank), "alpha equal to rank keeps the adapter's scale at 1")
    r["batch_size"] = (1, "small batches with more steps beat large batches for LoRAs")
    if krea:
        r["target_text_fusion"] = (True, "the attention + feed-forward recipe leaves Krea's text fusion under-trained: "
                                         "Fizgig measured 2-7 likeness points from boosting it on finished LoRAs")
        r["target_attn_gate"] = (True, "Krea's own LoRAs train the attention output gate too")
    if character:
        r["region_weights"] = (True, "the best measured single setting for face likeness")
    # saving and samples
    total = int(cfg.steps or 2000)
    every = max(100, int(round(total / 8 / 50)) * 50)
    r["save_every"] = (every, f"about 8 checkpoints over {total} steps to choose from afterwards")
    r["lora_ema"] = (0.98, "measured by Fizgig: 5 likeness points above no average late in a run, with half the swing "
                           "between checkpoints; longer averages (0.995) lag behind")
    r["sample_every"] = (every, "a sample set at every checkpoint shows which one to keep")
    if krea:
        r["sample_model_path"] = ("krea/Krea-2-Turbo", "Krea trains LoRAs on Raw and uses them on Turbo: previews and "
                                                       "probes in 8 steps without guidance instead of 28+ with it "
                                                       "(Turbo is gated separately: accept its license on Hugging "
                                                       "Face too; about 35 GB)")
    if krea_turbo:
        r["sample_steps"] = (8, "Krea 2 Turbo is an 8-step distill")
        r["sample_cfg"] = (1.0, "Turbo samples without guidance")
    elif cfg.turbo_lora:
        r["sample_steps"] = (8, "most turbo LoRAs are 8-step distills; match yours")
        r["sample_cfg"] = (1.0, "distilled turbo LoRAs sample without guidance")
    elif krea:
        r["sample_steps"] = (28, "Krea's pipeline samples Raw at 28 steps")
        r["sample_cfg"] = (5.5, "Krea's guidance 4.5 for Raw, in this trainer's convention (1 = off)")
    elif fibo:
        r["sample_steps"] = (30, "Bria samples dev at 50 steps; 20-30 is plenty for previews")
        r["sample_cfg"] = (5.0, "Bria's guidance scale for FIBO 2.0 dev")
    else:
        r["sample_steps"] = (20, "the reference MOSS-Image sampler")
        r["sample_cfg"] = (3.0, "the reference MOSS-Image guidance scale")
    if character:
        r["probe_every"] = (25, "frequent quick checks give the best-checkpoint and early-stop decisions enough points")
        if cfg.turbo_lora or krea_turbo:
            r["probe_steps"] = (8, "fast probes with the turbo model" if krea_turbo else "fast probes with the turbo LoRA")
            r["probe_cfg"] = (1.0, "turbo models sample without guidance")
        else:
            r["probe_steps"] = (r["sample_steps"][0], "without a turbo LoRA a probe needs the full sampler steps")
            r["probe_cfg"] = (r["sample_cfg"][0], "without a turbo LoRA a probe needs normal guidance")
        face_p, body_p = PRESET_PROBE_PROMPTS["identity"]
        subject = cfg.trigger_word or "the character"
        r["probe_face_prompt"] = (face_p.format(trigger=subject), "a plain headshot scores the face reliably")
        r["probe_body_prompt"] = (body_p.format(trigger=subject), "an upright full-body shot scores proportions")
    return r


def review_issues(cfg: TrainConfig, goal: str | None, data: dict | None, n_gpus: int) -> list[tuple[str, str]]:
    """What the wizard's last step points out before starting: (error | warning | info, text)."""
    out: list[tuple[str, str]] = []
    if _running():
        out.append(("error", "A training job is already running. Stop it (Monitor tab) before starting another."))
    if not Path(cfg.dataset_dir).is_dir():
        out.append(("error", f"Dataset folder not found: {cfg.dataset_dir or '(empty)'}"))
    elif data is not None:
        if not data["images"]:
            out.append(("error", "The dataset folder has no images (png, jpg, webp or bmp)."))
        elif data["captioned"] < data["images"] and not cfg.trigger_word:
            out.append(("warning", f"{data['images'] - data['captioned']} of {data['images']} images have no caption "
                                   "and there is no trigger word, so they train on an empty prompt."))
    try:
        cfg.validate()
    except ValueError as e:
        out.append(("error", str(e)))
    if cfg.model_source == "files" and cfg.arch_name() != "moss_image":
        out.append(("error", "Single safetensors files are supported for MOSS-Image only: pick the Hugging Face source "
                             "for this model."))
    elif cfg.model_source == "files":
        for key, label in (("transformer_file", "Diffusion model"), ("text_encoder_file", "Text encoder"),
                           ("vae_file", "VAE")):
            if getattr(cfg, key) and not Path(getattr(cfg, key)).is_file():
                out.append(("error", f"{label} file not found: {getattr(cfg, key)}"))
    elif "edit-turbo" in cfg.model_path.lower():
        out.append(("warning", "FIBO-2.0-edit-turbo is a distilled model. Train on FIBO-2.0-dev: its LoRAs load "
                               "onto Edit Turbo."))
    elif "krea-2-turbo" in cfg.model_path.lower():
        out.append(("warning", "Krea-2-Turbo is a distilled model. Train on Krea-2-Raw and set Turbo as the sampling "
                               "model: LoRAs trained on Raw are meant for Turbo."))
    elif cfg.arch_name() == "krea2":
        out.append(("info", "Krea 2 is gated: accept its license on Hugging Face and log in with hf auth login before "
                            "the first download." + (" Krea-2-Turbo (the sampling model) has a gate of its own: accept "
                                                     "its license as well." if "krea-2-turbo" in
                                                     (cfg.sample_model_path or "").lower() else "")))
    elif cfg.arch_name() == "fibo2":
        out.append(("info", "FIBO 2.0 is gated: accept its terms on Hugging Face and log in with hf auth login "
                            "before the first download."))
    if cfg.multi_gpu != "off" and n_gpus < 2:
        out.append(("error", "Multi-GPU is on, but only one GPU was found."))
    if goal == "character" or cfg.preset == "identity":
        if not (cfg.face_refs or cfg.body_refs):
            out.append(("warning", "No face or body references: likeness scores, the best checkpoint and early stop "
                                   "need them (Dataset step)."))
        elif (cfg.probe_every > 0 and cfg.probe_steps <= 10
              and not (cfg.turbo_lora or "turbo" in (cfg.sample_model_path or "").lower())):
            out.append(("warning", f"Probes render in {cfg.probe_steps} steps without a turbo LoRA, so they come out "
                                   "unfinished and their likeness scores mean little. Set a turbo LoRA or raise the "
                                   "probe steps (Samples step)."))
    if cfg.sample_every <= 0 or not cfg.sample_prompts:
        out.append(("info", "No sample prompts (or sampling is off): there will be no preview images while training."))
    if cfg.save_every > cfg.steps:
        out.append(("info", "Save every is more than the number of steps, so only the final LoRA is saved."))
    return out


def suggested_prompts(goal: str | None, trigger: str) -> list[str]:
    if goal == "character":
        t = trigger or "the character"
        return [f"photo of {t} smiling in a sunny park, natural light",
                f"close-up portrait of {t}, studio lighting, plain grey background",
                f"full body photo of {t} walking down a city street at night, neon lights"]
    if goal == "style":
        t = f"{trigger} style" if trigger else "the trained style"
        return [f"a lighthouse on a rocky coast at sunset, {t}",
                f"portrait of an old fisherman with a pipe, {t}",
                f"a busy market street full of people, {t}"]
    t = trigger or "the subject"
    return [f"photo of {t} on a wooden table, soft daylight",
            f"{t} in a snowy forest, cinematic lighting",
            f"close-up photo of {t}, studio lighting, black background"]


def _same(current, rec) -> bool:
    """Does the current widget value match a recommended value?"""
    if isinstance(rec, bool):
        return bool(current) == rec
    if isinstance(rec, (int, float)):
        try:
            return abs(float(current) - rec) <= 1e-6 * max(abs(float(current)), abs(rec), 1e-12)
        except (TypeError, ValueError):
            return False
    if isinstance(rec, list):
        try:
            return sorted(float(x) for x in (current or [])) == sorted(float(x) for x in rec)
        except (TypeError, ValueError):
            return False
    if isinstance(rec, str):
        return (current or "").strip() == rec.strip()
    return current == rec


def _show(widget, value) -> str:
    """A recommended value the way the user sees it (the option label of a select, on/off, megapixels)."""
    options = getattr(widget, "options", None)
    if isinstance(options, dict) and value in options:
        return str(options[value])
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, list):
        return ", ".join(f"{v:g} MP" for v in value)
    if isinstance(value, float):
        return f"{value:g}"
    text = str(value)
    return text if len(text) <= 48 else text[:45] + "..."


class Wizard:
    """Guided setup on its own tab: one step per topic, each setting explained, with a recommended value.

    Its controls are bound to the full form's widgets (Train tab), so both views edit the same settings and a run
    started from either is the same run."""

    STEPS = (("goal", "Goal", "flag"), ("model", "Model", "model_training"), ("hardware", "Hardware", "memory"),
             ("dataset", "Dataset", "photo_library"), ("training", "Training", "school"),
             ("samples", "Samples", "image"), ("review", "Review & start", "rocket_launch"))

    def __init__(self, f: Form, gpus: list[str], start, goto):
        self.f, self.w = f, f.w
        self.gpus = gpus  # torch device names for the device select
        self.start = start  # start_job(): starts training from the form, True when it started
        self.goto = goto  # goto(tab name)
        self.goal: str | None = "character" if self.w["preset_toggle"].value else None
        self.smi: list[dict] = []  # nvidia-smi rows (memory per GPU)
        self.data: dict | None = None  # last dataset scan
        self.recs: dict[str, tuple] = {}
        self.rec_rows: list[tuple] = []
        self.goal_cards: dict[str, ui.card] = {}
        self.model_cards: dict[str, ui.card] = {}
        self.steps: dict[str, ui.step] = {}

    # ------------------------------------------------------------ building blocks

    def bind(self, widget, key: str):
        """Bind a wizard control to the form widget `key` and keep the recommendations current."""
        widget.bind_value(self.w[key], "value")
        widget.on_value_change(lambda _: self.refresh())
        return widget

    @contextmanager
    def setting(self, key: str | None, title: str, text: str, show_if=None):
        """One setting: title, explanation and recommendation on the left, the control (built in the block) on the
        right. show_if=(form key, predicate) shows it only while the predicate holds for that widget's value."""
        box = ui.element("div").classes("rz-setting")
        if show_if is not None:
            box.bind_visibility_from(self.w[show_if[0]], "value", show_if[1])
        with box:
            with ui.column().classes("gap-1 min-w-0"):
                ui.label(title).classes("rz-setting-title")
                ui.label(text).classes("rz-hint")
                if key:
                    self.rec_row(key)
            with ui.column().classes("gap-2 w-full min-w-0 justify-center"):
                yield

    def rec_row(self, key: str):
        with ui.row().classes("rz-rec items-start gap-2 no-wrap") as row:
            icon = ui.icon("lightbulb").classes("mt-[2px] shrink-0")
            label = ui.label("").classes("min-w-0")
            btn = ui.button("Use", on_click=lambda: self.apply([key])) \
                .props("outline dense size=sm no-caps color=white").classes("shrink-0 px-2") \
                .tooltip("Use the recommended value")
        self.rec_rows.append((key, row, icon, label, btn))

    def apply_bar(self, keys: list[str]):
        with ui.row().classes("rz-apply w-full items-center gap-3"):
            ui.icon("auto_awesome").classes("text-xl").style("color: var(--rz-red-bright)")
            ui.label("Not sure? Take every recommendation on this page.").classes("text-sm")
            ui.space()
            ui.button("Apply recommended", icon="done_all", on_click=lambda: self.apply(keys)).props("unelevated")

    def nav(self, back: bool = True):
        with ui.stepper_navigation():
            if back:
                ui.button("Back", icon="arrow_back", on_click=self.stepper.previous).props("flat")
            ui.space()
            ui.button("Next", on_click=self.stepper.next).props("unelevated icon-right=arrow_forward")

    def heading(self, title: str, text: str):
        ui.label(title).classes("rz-step-title")
        hint(text)

    # ------------------------------------------------------------ recommendations

    def refresh(self):
        try:
            cfg = self.f.to_config()
        except (TypeError, ValueError):
            return  # a field is mid-edit (e.g. an empty number); keep the last recommendations
        self.recs = wizard_recommendations(cfg, self.goal, self.smi, self.data)
        for key, row, icon, label, btn in self.rec_rows:
            rec = self.recs.get(key)
            row.visible = rec is not None
            if rec is None:
                continue
            value, reason = rec
            same = _same(self.w[key].value, value)
            if same:
                row.classes(add="ok")
                label.text = f"{reason[:1].upper()}{reason[1:]}."
            else:
                row.classes(remove="ok")
                label.text = f"Recommended: {_show(self.w[key], value)}. {reason[:1].upper()}{reason[1:]}."
            icon.name = "check_circle" if same else "lightbulb"
            btn.visible = not same

    def apply(self, keys: list[str]):
        for key in keys:
            rec = self.recs.get(key)
            if rec is not None:
                self.w[key].value = list(rec[0]) if isinstance(rec[0], list) else rec[0]
            self.refresh()  # later recommendations can depend on earlier ones (save every on steps)

    # ------------------------------------------------------------ steps

    def build(self):
        with ui.column().classes("rz-page gap-4"):
            with card():
                with ui.row().classes("w-full items-center gap-4"):
                    with ui.column().classes("gap-1 flex-1 min-w-[16rem]"):
                        ui.label("Guided setup").classes("rz-title")
                        hint("Walks through a training run one topic at a time. Every setting is explained and comes "
                             "with a recommendation for your GPUs and dataset: click Use to take it, or Apply "
                             "recommended for a whole page. These are the same settings as on the Train tab, which "
                             "has every option.")
                    ui.button("All settings", icon="tune", on_click=lambda: self.goto("Train")).props("outline")
            with ui.stepper(on_value_change=self.on_step).props("header-nav alternative-labels animated flat") \
                    .classes("rz-stepper w-full") as self.stepper:
                for name, title, icon in self.STEPS:
                    with ui.step(name, title=title, icon=icon).props(f"active-icon={icon}") as step:
                        self.steps[name] = step
                        getattr(self, f"step_{name}")()
        self.w["preset_toggle"].on_value_change(lambda e: self.sync_goal(e.value))
        self.w["model_preset"].on_value_change(lambda _: (self.mark_model(), self.refresh()))
        self.mark_goal()
        self.mark_model()
        self.refresh()
        ui.timer(0.2, self.detect_gpus, once=True)

    async def on_step(self, e):
        name = e.value if isinstance(e.value, str) else e.value.props.get("name")
        names = [s[0] for s in self.STEPS]
        idx = names.index(name) if name in names else 0
        for i, n in enumerate(names):
            self.steps[n].props(add="done") if i < idx else self.steps[n].props(remove="done")
        if name == "dataset" and self.data is None and Path(self.w["dataset_dir"].value or "").is_dir():
            await self.scan()
        if name == "review":
            await self.render_review()

    # ---- 1 goal
    def step_goal(self):
        self.heading("What are you training?", "This picks the recipe for the rest of the wizard. You can change "
                                               "anything afterwards.")
        with ui.element("div").classes("rz-choices"):
            for key, (icon, title, text) in GOALS.items():
                with ui.card().props("flat").classes("rz-choice") as c:
                    with ui.row().classes("w-full items-center gap-2 no-wrap"):
                        ui.icon(icon).classes("rz-choice-icon")
                        ui.label(title).classes("rz-setting-title")
                        ui.space()
                        ui.icon("check_circle").classes("rz-check")
                    ui.label(text).classes("rz-hint")
                c.on("click", lambda _, k=key: self.set_goal(k))
                self.goal_cards[key] = c
        self.goal_note = ui.label("").classes("rz-banner info")
        with self.setting(None, "Job name", "The name of this run. Checkpoints are saved as "
                                            "output/<job name>/<job name>_<step>.safetensors and the final LoRA as "
                                            "output/<job name>/<job name>.safetensors."):
            self.bind(ui.input("Job name"), "job_name").classes("w-full")
        self.nav(back=False)

    def set_goal(self, goal: str):
        self.goal = goal
        want = goal == "character"
        if bool(self.w["preset_toggle"].value) != want:
            self.w["preset_toggle"].value = want  # applies the identity recipe, or restores the settings from before
        self.mark_goal()
        self.refresh()

    def sync_goal(self, preset_on: bool):
        """The identity preset was switched elsewhere (Train tab, a loaded config)."""
        if preset_on:
            self.goal = "character"
        elif self.goal == "character":
            self.goal = None
        self.mark_goal()
        self.refresh()

    def mark_goal(self):
        for key, c in self.goal_cards.items():
            c.classes(add="selected") if key == self.goal else c.classes(remove="selected")
        if self.goal == "character":
            self.goal_note.text = ("Identity recipe on: region-weighted loss (face 3 / person 1.5 / background 0.3), "
                                   "LoRA+ 4 with adamw8bit, face and bust crops, likeness probes every 25 steps with "
                                   "best checkpoint and early stop, curriculum, per-image loss watch.")
        elif self.goal:
            self.goal_note.text = f"{GOALS[self.goal][1]}: general settings, tuned on the next pages."
        else:
            self.goal_note.text = "Pick one to get recommendations that fit it."

    # ---- 2 model
    def step_model(self):
        self.heading("Which model do you want to train on?", "The base model the LoRA is trained for.")
        with self.setting(None, "Where the model comes from",
                          "Hugging Face downloads the model once into models/ and reuses it after that. Single files "
                          "uses the diffusion model, text encoder and VAE files you already have for ComfyUI, so "
                          "nothing is downloaded (MOSS-Image only)."):
            self.bind(ui.radio([HF_SOURCE, FILE_SOURCE]), "model_source")
        with ui.column().classes("w-full gap-3 pt-4") \
                .bind_visibility_from(self.w["model_source"], "value", lambda x: x == HF_SOURCE):
            with ui.element("div").classes("rz-choices"):
                for name, (badge, text) in [*MODEL_NOTES.items(), (CUSTOM, ("", "Any local diffusers folder "
                                                                                 "(transformer/, text_encoder/, vae/) "
                                                                                 "or Hugging Face repo id."))]:
                    with ui.card().props("flat").classes("rz-choice") as c:
                        with ui.row().classes("w-full items-center gap-2"):
                            ui.label(name).classes("rz-setting-title")
                            if badge:
                                ui.label(badge).classes("rz-badge" + ("" if badge == "Recommended" else " muted"))
                            ui.space()
                            ui.icon("check_circle").classes("rz-check")
                        ui.label(text).classes("rz-hint")
                    c.on("click", lambda _, n=name: setattr(self.w["model_preset"], "value", n))
                    self.model_cards[name] = c
            with self.setting(None, "Custom model", "A local folder in diffusers layout, or a Hugging Face repo id "
                                                    "such as OpenMOSS-Team/MOSS-Image-Base.",
                              show_if=("model_preset", lambda p: p == CUSTOM)):
                self.bind(ui.input("Path or repo id"), "model_path").classes("w-full")
            with self.setting(None, "Download and check",
                              "Fetches what is missing and checks the files already there. Training would also do "
                              "this on start, but doing it now shows problems (gated repo, no login, disk space) "
                              "before you start."):
                status = hint("")

                async def download():
                    status.text = "Downloading / verifying... (about 30 GB on the first download)"
                    status.text = await run.io_bound(download_model, self.f.to_config().model_path)

                ui.button("Download / verify", icon="download", on_click=download).props("outline")
        with ui.column().classes("w-full gap-0 pt-2") \
                .bind_visibility_from(self.w["model_source"], "value", lambda x: x == FILE_SOURCE):
            for key, title, text, kind in (
                    ("transformer_file", "Diffusion model", "ComfyUI models/diffusion_models file: bf16, fp8, int8 or "
                                                            "nvfp4.", "file"),
                    ("text_encoder_file", "Text encoder", "ComfyUI models/text_encoders file: Qwen3.5-9B for "
                                                          "MOSS-Image, Qwen3-VL-4B for Krea 2.", "file"),
                    ("vae_file", "VAE", "ComfyUI models/vae file: the Flux2 VAE for MOSS-Image, the Qwen-Image VAE for "
                                        "Krea 2.", "file"),
                    ("tokenizer_path", "Tokenizer folder (optional)", "Left empty, the tokenizer is downloaded once "
                                                                      "(a few MB).", "dir")):
                with self.setting(None, title, text + " Any file left empty is taken from the Hugging Face model."):
                    self.bind(path_field(title, kind, width="w-full"), key)
        self.nav()

    def mark_model(self):
        for name, c in self.model_cards.items():
            c.classes(add="selected") if name == self.w["model_preset"].value else c.classes(remove="selected")

    # ---- 3 hardware
    async def detect_gpus(self):
        self.smi = await run.io_bound(gpu_stats)
        self.gpu_box.clear()
        with self.gpu_box:
            if not self.smi:
                hint("No NVIDIA GPU found with nvidia-smi. The recommendations below assume a 16 GB+ card.", warn=True)
            for g in self.smi:
                with ui.column().classes("rz-tile gap-0"):
                    ui.label(f"GPU {g['index']}").classes("rz-tile-label")
                    ui.label(g["name"]).classes("rz-tile-value text-base")
                    ui.label(f"{g['total'] / 1024:.0f} GB memory, {g['used'] / 1024:.1f} GB in use").classes("rz-tile-sub")
        self.refresh()

    def step_hardware(self):
        self.heading("Your GPUs", "How the model is spread over your GPUs and how much memory it may take.")
        self.gpu_box = ui.row().classes("w-full gap-3")
        with self.gpu_box:
            ui.spinner(size="sm")
        self.apply_bar(["multi_gpu", "dp_balance", "transformer_quant", "text_encoder_quant", "gradient_checkpointing",
                        "checkpoint_keep_blocks", "compile_model"])
        with self.setting("multi_gpu", "Multi-GPU",
                          "Data parallel puts a full model copy on each GPU, each trains its own images, and the "
                          "results are averaged: about 1.7-2x the speed with two cards (each must fit the model). "
                          "Model split divides the model's blocks over the GPUs: half the memory each, not faster."):
            self.bind(ui.select(dict(self.w["multi_gpu"].options), label="Multi-GPU"), "multi_gpu").classes("w-full")
        with self.setting(None, "Training GPU", "The GPU that trains. Pick the one with the most memory "
                                                "(listed in CUDA order).",
                          show_if=("multi_gpu", lambda x: x == "off")):
            self.bind(ui.select(self.gpus, label="Training GPU"), "device").classes("w-full")
        with self.setting(None, "GPUs to use", "Comma-separated, the first is the main GPU (it also saves and "
                                               "samples).", show_if=("multi_gpu", lambda x: x != "off")):
            self.bind(ui.input("Devices"), "devices").classes("w-full")
        with self.setting("dp_balance", "Balance unequal GPUs",
                          "With two different cards the faster one waits for the slower one every step. auto measures "
                          "them during the first steps and splits each step's images by speed (it needs a few images "
                          "per step: gradient accumulation 3 or more). grow lets the faster card train more images in "
                          "the same time instead, so each step holds more images.",
                          show_if=("multi_gpu", lambda x: x == "data_parallel")):
            self.bind(ui.select(dict(self.w["dp_balance"].options), label="Balance"), "dp_balance").classes("w-full")
        with self.setting("transformer_quant", "Model quantization",
                          "Stores the frozen base model in fewer bits to save memory; the LoRA itself always trains "
                          "in full precision. auto keeps the model as it is. int8_convrot halves it at under 1% error "
                          "and is as fast as bf16 or faster; nvfp4 quarters it but is slower."):
            self.bind(ui.select(QUANTS, label="Diffusion model quant"), "transformer_quant").classes("w-full")
        with self.setting("text_encoder_quant", "Text encoder quantization",
                          "The text encoder only runs once before training, to turn your captions into embeddings, "
                          "and is unloaded afterwards. Quantizing it lets it fit on the GPU, so that step is quick."):
            self.bind(ui.select(QUANTS, label="Text encoder quant"), "text_encoder_quant").classes("w-full")
        with self.setting("gradient_checkpointing", "Gradient checkpointing",
                          "Recomputes intermediate results during the backward pass instead of keeping them in "
                          "memory: far less memory for some extra time per step."):
            self.bind(ui.switch("Gradient checkpointing"), "gradient_checkpointing")
        with self.setting("checkpoint_keep_blocks", "Blocks that skip the recompute",
                          "Spends spare GPU memory on speed: these blocks keep their intermediate results instead of "
                          "recomputing them, about a third faster each. -1 measures your largest images at the start "
                          "and keeps as many as fit; 0 recomputes every block.",
                          show_if=("gradient_checkpointing", bool)):
            self.bind(ui.number("Kept blocks (-1 = auto)", min=-1, precision=0), "checkpoint_keep_blocks") \
                .classes("w-full")
        with self.setting("compile_model", "Compile the model",
                          "torch.compile turns the small steps between the big matrix multiplies into fewer, fused GPU "
                          "kernels: measured 26% faster per step on MOSS-Image and 12% on Krea 2, after a warm-up of "
                          "under a minute. auto does it for runs of 100 steps or more."):
            self.bind(ui.select(COMPILE_CHOICES, label="Compile"), "compile_model").classes("w-full")
        self.nav()

    # ---- 4 dataset
    def step_dataset(self):
        self.heading("Your images", "A folder of images, each with an optional same-name .txt caption. 15-50 sharp, "
                                    "varied images are typical for a character; styles and concepts like 20-100.")
        with self.setting(None, "Dataset folder", "Upload images and write captions on the Dataset tab, or point "
                                                  "this at a folder you already have. Sub-folders are included."):
            folder = self.bind(path_field("Dataset folder", "dir", width="w-full"), "dataset_dir")
            folder.on_value_change(lambda _: self.forget_scan())
            with ui.row().classes("gap-2"):
                ui.button("Scan folder", icon="search", on_click=self.scan).props("unelevated")
                ui.button("Dataset tab", icon="photo_library", on_click=lambda: self.goto("Dataset")).props("flat")
        self.scan_box = ui.column().classes("w-full gap-3")
        self.apply_bar(["resolutions", "enable_buckets", "flip_augment", "face_refs"])
        with self.setting(None, "Trigger word",
                          "Put in front of every caption, and then used in your prompts to call up what the LoRA "
                          "learned. Pick a word the model does not know yet, such as 'zxq woman' or 'inkwash style'. "
                          "Leave it empty if your captions already contain one."):
            self.bind(ui.input("Trigger word"), "trigger_word").classes("w-full")
        with self.setting(None, "Captions",
                          "Describe in each caption what should stay changeable: clothes, background, pose, "
                          "lighting. What you leave out is what the LoRA learns as part of the trigger word, so for a "
                          "character do not describe the face; for a style, describe the content but not the style."):
            ui.button("Edit captions", icon="edit_note", on_click=lambda: self.goto("Dataset")).props("outline")
        with self.setting("resolutions", "Training resolution",
                          "The image area training works at, in megapixels (1 MP = 1024x1024), with each image's "
                          "aspect ratio kept. Higher learns finer detail but takes more memory and time. Picking "
                          "several mixes them."):
            self.bind(ui.select(MP_OPTIONS, multiple=True, label="Resolutions").props("use-chips"), "resolutions") \
                .classes("w-full")
        with self.setting("enable_buckets", "Aspect-ratio buckets",
                          "Groups images by shape and trains each at its own aspect ratio. Off, every image is cropped "
                          "to a square."):
            self.bind(ui.switch("Aspect-ratio buckets"), "enable_buckets")
        with self.setting("flip_augment", "Mirror images",
                          "Also trains a left-right mirrored copy of every image: twice the variety, but anything "
                          "asymmetric (a hair parting, a logo, text) gets blurred."):
            self.bind(ui.switch("Horizontal flip"), "flip_augment")
        with self.setting("face_refs", "Face references",
                          "Images whose face is the identity to match. Samples and probes are compared with them "
                          "(ArcFace; 50%+ generally means the same person), which picks the best checkpoint and "
                          "stops early. A folder counts as all its images; you can also tick Face ref on the Dataset "
                          "tab.", show_if=("preset_toggle", bool)):
            self.bind(ui.textarea("Face references (one path per line)"), "face_refs").classes("w-full")
        with self.setting(None, "Body references (optional)",
                          "Full-body shots in an upright, fairly neutral pose, to score body proportions as well. "
                          "Tick Body ref on such images on the Dataset tab, or list them here.",
                          show_if=("preset_toggle", bool)):
            self.bind(ui.textarea("Body references (one path per line)"), "body_refs").classes("w-full")
        self.nav()

    def forget_scan(self):
        if self.data is not None and self.data.get("folder") != (self.w["dataset_dir"].value or ""):
            self.data = None
            self.scan_box.clear()
            self.refresh()

    async def scan(self):
        folder = (self.w["dataset_dir"].value or "").strip()
        self.scan_box.clear()
        with self.scan_box:
            ui.spinner(size="sm")
        self.data = await run.io_bound(dataset_summary, folder)
        thumbs = await run.io_bound(make_thumbs, self.data["sample"], Path(folder)) if self.data else []
        self.scan_box.clear()
        with self.scan_box:
            if self.data is None:
                ui.label(f"Folder not found: {folder or '(empty)'}").classes("rz-banner error")
            else:
                n, cap, mp = self.data["images"], self.data["captioned"], self.data["median_mp"]
                with ui.row().classes("w-full gap-3"):
                    for label, value, sub in (("Images", str(n), "png, jpg, webp, bmp"),
                                              ("Captioned", f"{cap} / {n}", ".txt or .json next to the image"),
                                              ("Typical size", f"{mp:.2f} MP" if mp else "-", "median image area")):
                        v, s = stat_tile(label)
                        v.text, s.text = value, sub
                if not n:
                    ui.label("No images in this folder yet. Add them on the Dataset tab (drag and drop) or copy "
                             "them into the folder, then scan again.").classes("rz-banner warning")
                elif cap < n:
                    ui.label(f"{n - cap} images have no caption yet. They train on the trigger word alone, which "
                             "works for a quick character LoRA; captions give a more flexible one.") \
                        .classes("rz-banner warning")
                if thumbs:
                    with ui.row().classes("w-full gap-2"):
                        for t in thumbs:
                            ui.image(str(t)).classes("w-24 h-24 rz-thumb")
        self.refresh()

    # ---- 5 training
    def step_training(self):
        self.heading("How to train", "How long and how hard to train, and how big the LoRA is.")
        self.apply_bar(["steps", "learning_rate", "optimizer", "lora_rank", "lora_alpha", "batch_size",
                        "target_text_fusion", "target_attn_gate", "region_weights"])
        with self.setting("steps", "Training steps",
                          "Each step trains on one batch of images. Too few and the LoRA is weak; too many and it "
                          "starts copying the training images (overfitting). Checkpoints are saved on the way, so "
                          "you can always go back to an earlier one."):
            self.bind(ui.number("Steps", min=1, precision=0), "steps").classes("w-full")
        with self.setting("learning_rate", "Learning rate",
                          "How big each update is. Too high gives burnt, noisy samples; too low learns slowly. LoRAs "
                          "use about 10x the learning rate of a full fine-tune, almost regardless of rank."):
            self.bind(ui.number("Learning rate", format="%.1e", step=1e-5, min=0), "learning_rate").classes("w-full")
        with self.setting("optimizer", "Optimizer",
                          "adamw8bit is AdamW with its state stored in 8 bits: the default choice. adamw keeps full "
                          "precision state (same results, more memory). adafactor is a lighter alternative. "
                          "automagic3 tunes its own learning rate per weight."):
            self.bind(ui.select(["adamw", "adamw8bit", "adafactor", "automagic3"], label="Optimizer"), "optimizer") \
                .classes("w-full")
        with self.setting("lora_rank", "LoRA rank",
                          "The size of the adapter. Higher ranks can hold more detail but give bigger files and "
                          "memorise the training images more easily."):
            self.bind(ui.number("Rank", min=1, max=256, precision=0), "lora_rank").classes("w-full")
        with self.setting("lora_alpha", "LoRA alpha",
                          "Scales the adapter's effect (alpha / rank). Keeping alpha equal to rank is the safe "
                          "default."):
            self.bind(ui.number("Alpha", min=0), "lora_alpha").classes("w-full")
        with self.setting("batch_size", "Batch size",
                          "Images per step. Larger batches need more memory and, for LoRAs, rarely help."):
            self.bind(ui.number("Batch size", min=1, precision=0), "batch_size").classes("w-full")
        is_krea = ("model_preset", lambda p: preset_arch(p, self.w["model_path"].value) == "krea2")
        with self.setting("target_text_fusion", "Train Krea's text fusion",
                          "Krea 2 mixes 12 layers of its text encoder in four text-fusion blocks before the main "
                          "model. The usual attention + feed-forward LoRA leaves them under-trained: boosting them on "
                          "finished LoRAs lifted likeness 2-7 points in Fizgig's tests. Training them adds about 3.5M "
                          "LoRA weights, and the backward then runs through every block.", show_if=is_krea):
            self.bind(ui.switch("Text fusion"), "target_text_fusion")
        with self.setting("target_attn_gate", "Train the attention gate",
                          "Every Krea 2 attention has a sigmoid gate on its output. Krea's own LoRAs train it.",
                          show_if=is_krea):
            self.bind(ui.switch("Attention gate"), "target_attn_gate")
        with self.setting("region_weights", "Region-weighted loss",
                          "Weights what the model learns by region: the face counts most, the rest of the person "
                          "next, the background least, so the LoRA spends its capacity on the identity.",
                          show_if=("preset_toggle", bool)):
            self.bind(ui.switch("Region-weighted loss"), "region_weights")
        hint("Timesteps, schedules, LoRA layers and blocks, and per-image loss watch are on the Train tab.")
        self.nav()

    # ---- 6 samples
    def step_samples(self):
        self.heading("Checkpoints and preview images", "What gets saved along the way, and the pictures that show "
                                                       "how training is going.")
        # the sampling model first: the step and guidance recommendations depend on it
        self.apply_bar(["sample_model_path", "save_every", "lora_ema", "sample_every", "sample_steps", "sample_cfg",
                        "probe_every", "probe_steps", "probe_cfg", "probe_face_prompt", "probe_body_prompt"])
        with self.setting("save_every", "Save a checkpoint every",
                          "Steps between saved LoRAs (output/<job>/<job>_<step>.safetensors). Several of them let you "
                          "pick the last one before it started to overfit."):
            self.bind(ui.number("Save every", min=1, precision=0, suffix="steps"), "save_every").classes("w-full")
        with self.setting("lora_ema", "Weight average",
                          "Each step pulls the LoRA toward the last image it saw, so a checkpoint carries a bit of "
                          "chance. Saving a running average of the recent steps instead (about the last 50 images at "
                          "0.98) steadies it: checkpoints, previews and likeness probes use the average while training "
                          "goes on as before. 0 turns it off."):
            self.bind(ui.number("Decay", min=0, max=0.999, step=0.005, format="%.3f"), "lora_ema").classes("w-full")
        with self.setting("sample_every", "Preview images every",
                          "Steps between sample sets (0 = never). Training pauses while they render; they show up "
                          "on the Monitor and Samples tabs."):
            self.bind(ui.number("Sample every", min=0, precision=0, suffix="steps"), "sample_every").classes("w-full")
        with self.setting(None, "Sample prompts",
                          "One prompt per line, rendered at every sample step with a fixed seed, so the pictures are "
                          "comparable over time. Use your trigger word and vary the scene to see whether the LoRA "
                          "carries over to new situations."):
            self.bind(ui.textarea("Sample prompts (one per line)"), "sample_prompts").classes("w-full")
            ui.button("Suggest prompts", icon="lightbulb", on_click=self.suggest).props("flat")
        is_krea = ("model_preset", lambda p: preset_arch(p, self.w["model_path"].value) == "krea2")
        with self.setting("sample_model_path", "Preview with Krea 2 Turbo",
                          "Renders previews and probes with Krea-2-Turbo carrying your LoRA, while training stays on "
                          "Raw. That is how Krea uses LoRAs trained on Raw, and Turbo needs 8 steps without guidance "
                          "where Raw needs 28 or more with it. Empty = preview on Raw.", show_if=is_krea):
            self.bind(ui.input("Sampling model (repo id or folder)"), "sample_model_path").classes("w-full")
        with self.setting(None, "Turbo LoRA (optional)",
                          "A step-distilled LoRA (for example an 8-step DMD distill of MOSS-Image-Base), used only "
                          "while rendering samples and probes, never in training. Previews then take seconds; set the "
                          "steps and guidance below to match it.",
                          show_if=("model_preset", lambda p: preset_arch(p, self.w["model_path"].value) != "krea2")):
            self.bind(path_field("Turbo LoRA", "file", width="w-full"), "turbo_lora")
        with self.setting("sample_steps", "Sampling steps", "Denoising steps per preview image. More is slower and "
                                                            "a little cleaner."):
            self.bind(ui.number("Sample steps", min=1, precision=0), "sample_steps").classes("w-full")
        with self.setting("sample_cfg", "Guidance (CFG)", "How strictly previews follow the prompt."):
            self.bind(ui.number("Sample CFG", min=0, step=0.5), "sample_cfg").classes("w-full")
        is_identity = ("preset_toggle", bool)
        with self.setting("probe_every", "Likeness probe every",
                          "A quick, small headshot and full-body render every N steps, scored against your "
                          "references. The scores keep the best checkpoint (<job>_best.safetensors) and can stop "
                          "training once it stops improving.", show_if=is_identity):
            self.bind(ui.number("Probe every", min=0, precision=0, suffix="steps"), "probe_every").classes("w-full")
        with self.setting("probe_face_prompt", "Face probe prompt",
                          "A plain headshot of your character, rendered with a fixed seed.", show_if=is_identity):
            self.bind(ui.textarea("Face probe prompt").props("autogrow"), "probe_face_prompt").classes("w-full")
        with self.setting("probe_body_prompt", "Body probe prompt",
                          "A full-body, upright shot, for the body proportions score.", show_if=is_identity):
            self.bind(ui.textarea("Body probe prompt").props("autogrow"), "probe_body_prompt").classes("w-full")
        with self.setting("probe_steps", "Probe steps", "Denoising steps per probe.", show_if=is_identity):
            self.bind(ui.number("Probe steps", min=1, precision=0), "probe_steps").classes("w-full")
        with self.setting("probe_cfg", "Probe guidance (CFG)", "Guidance for the probes.", show_if=is_identity):
            self.bind(ui.number("Probe CFG", min=0, step=0.5), "probe_cfg").classes("w-full")
        with self.setting(None, "Early stop", "Stop after this many probes in a row without improvement (0 = never: "
                                              "run all the steps). The best checkpoint is kept either way.",
                          show_if=is_identity):
            self.bind(ui.number("Probes without improvement", min=0, precision=0), "likeness_patience") \
                .classes("w-full")
        self.nav()

    def suggest(self):
        replaced = bool((self.w["sample_prompts"].value or "").strip())
        self.w["sample_prompts"].value = "\n".join(suggested_prompts(self.goal, (self.w["trigger_word"].value or "")
                                                                     .strip()))
        if replaced:
            ui.notify("Replaced the sample prompts with suggestions")

    # ---- 7 review
    def step_review(self):
        self.heading("Review and start", "A last look at the run before it starts.")
        self.review_box = ui.column().classes("w-full gap-3")
        with ui.row().classes("w-full items-center gap-3 pt-2"):
            self.cfg_path = ui.input("Save as", value="").classes("w-80 max-w-full")
            ui.button("Save config", icon="save", on_click=self.save).props("outline")
            ui.space()
            ui.button("Start training", icon="play_arrow", on_click=self.start).props("unelevated size=lg")
        with ui.stepper_navigation():
            ui.button("Back", icon="arrow_back", on_click=self.stepper.previous).props("flat")

    def save(self):
        try:
            cfg = self.f.to_config()
        except (TypeError, ValueError) as e:
            ui.notify(f"A setting is invalid: {e}", type="negative")
            return
        path = (self.cfg_path.value or "").strip() or f"configs/{cfg.job_name}.yaml"
        if _is_repo_example(path):
            ui.notify("configs/example.yaml ships with RulzTrainer; save under another name", type="warning")
            return
        cfg.save(path)
        ui.notify(f"Saved {path}", type="positive")

    async def render_review(self):
        try:
            cfg = self.f.to_config()
        except (TypeError, ValueError) as e:
            self.review_box.clear()
            with self.review_box:
                ui.label(f"A setting is invalid: {e}").classes("rz-banner error")
            return
        if self.data is None or self.data.get("folder") != cfg.dataset_dir:
            self.data = await run.io_bound(dataset_summary, cfg.dataset_dir)
        if not self.cfg_path.value:
            self.cfg_path.value = f"configs/{cfg.job_name}.yaml"
        issues = review_issues(cfg, self.goal, self.data, len(self.gpus))
        data = self.data or {}
        multi = {"off": f"single GPU ({cfg.device})", "data_parallel": f"data parallel on {cfg.devices}",
                 "model_split": f"model split over {cfg.devices}"}.get(cfg.multi_gpu, cfg.multi_gpu)
        model = (Path(cfg.transformer_file).name or "Hugging Face model") if cfg.model_source == "files" \
            else next((k for k, v in MODEL_PRESETS.items() if v == cfg.model_path), cfg.model_path)
        rows = [
            ("Goal", GOALS[self.goal][1] if self.goal else "not chosen (general settings)"),
            ("Job", f"{cfg.job_name}  ->  {_relative(str(cfg.job_dir()))}/"),
            ("Model", f"{model} ({'single files' if cfg.model_source == 'files' else 'Hugging Face'})"),
            ("GPUs", f"{multi}; model quant {cfg.transformer_quant}, text encoder quant {cfg.text_encoder_quant}"),
            ("Dataset", f"{cfg.dataset_dir} ({data.get('images', '?')} images, {data.get('captioned', '?')} captioned)"),
            ("Trigger word", cfg.trigger_word or "(none)"),
            ("Resolution", ", ".join(f"{mp:g} MP" for mp in cfg.resolutions)),
            ("Training", f"{cfg.steps} steps, batch {cfg.batch_size}, lr {cfg.learning_rate:g}, {cfg.optimizer}"),
            ("LoRA", f"rank {cfg.lora_rank}, alpha {cfg.lora_alpha:g}"),
            ("Saving", f"every {cfg.save_every} steps; samples every {cfg.sample_every} steps "
                       f"({len(cfg.sample_prompts)} prompts)"),
        ]
        if cfg.preset == "identity":
            rows.append(("Likeness", f"{len(cfg.face_refs)} face / {len(cfg.body_refs)} body reference entries, "
                                     f"probes every {cfg.probe_every} steps"))
        self.review_box.clear()
        with self.review_box:
            if not issues:
                ui.label("Everything looks ready.").classes("rz-banner ok")
            for level, text in sorted(issues, key=lambda x: ("error", "warning", "info").index(x[0])):
                ui.label(text).classes(f"rz-banner {level}")
            with card():
                with ui.element("div").classes("rz-summary"):
                    for k, v in rows:
                        ui.label(k).classes("k")
                        ui.label(v).classes("v")


# ---------------------------------------------------------------- page


def build_page(local: bool = True):
    global _native_dialogs
    _native_dialogs = HAS_DISPLAY and local  # read while the widgets are built below
    d = TrainConfig()
    gpus = _gpus()
    f = Form()
    w = f.w
    theme_colors()
    ui.page_title("RulzTrainer")
    job_running = reattach_job()

    with ui.header().classes("rz-header"):
        with ui.row().classes("rz-header-row w-full items-center no-wrap gap-2"):
            ui.element("img").props('src="/assets/logo_dark.png" alt="Rulz"').classes("rz-logo")
            ui.label("TRAINER").classes("rz-wordmark")
            ui.label("LoRA training for MOSS-Image, Bria FIBO 2.0 and Krea 2").classes("rz-tagline gt-sm")
            ui.space()
            with ui.row().classes("rz-pill items-center gap-2 no-wrap") as pill:
                pill_dot = ui.element("span").classes("rz-dot")
                pill_text = ui.label("Idle").classes("rz-pill-text")
        with ui.tabs().props("inline-label no-caps align=left dense").classes("rz-tabs") as tabs:
            tab_wizard = ui.tab("Wizard", icon="auto_awesome")
            tab_train = ui.tab("Train", icon="tune")
            tab_dataset = ui.tab("Dataset", icon="photo_library")
            tab_monitor = ui.tab("Monitor", icon="show_chart")
            tab_samples = ui.tab("Samples", icon="collections")
            tab_help = ui.tab("Help", icon="help_outline")

    def goto(name: str):
        tabs.value = name

    def start_job() -> bool:
        try:
            cfg = f.to_config()
            if cfg.race:  # the race's own checks before a model loads (step budget, settings, probes)
                from rulztrainer import race

                race.validate(cfg)
        except (TypeError, ValueError) as e:
            status_msg.text = f"A setting is invalid: {e}"
            ui.notify(status_msg.text, type="negative")
            return False
        status_msg.text = start_training(cfg)
        started = status_msg.text.startswith("Started")
        ui.notify(status_msg.text, type="positive" if started else "warning")
        if started:
            tabs.value = tab_monitor
        return started

    pill.on("click", lambda: goto("Monitor"))
    pill.tooltip("Open the Monitor")

    with ui.tab_panels(tabs, value=tab_monitor if job_running else tab_wizard).classes("w-full rz-panels") \
            .props("keep-alive"):
        # ------------------------------------------------------------ Train
        with ui.tab_panel(tab_train):
            with ui.row().classes("rz-page no-wrap items-start gap-6"):
                nav = ui.column().classes("rz-nav gt-sm")
                with ui.column().classes("gap-4 min-w-0 flex-1"):
                    nav_items: list[tuple[str, str, object]] = []

                    with card():
                        with ui.row().classes("w-full items-center gap-4"):
                            with ui.column().classes("gap-1 flex-1 min-w-[16rem]"):
                                ui.label("Training job").classes("rz-title")
                                hint("Every setting, grouped by topic; rarely changed ones are folded under Advanced. "
                                     "New to this? The Wizard walks through the important ones with recommendations.")
                            ui.button("Open the wizard", icon="auto_awesome", on_click=lambda: goto("Wizard")) \
                                .props("outline")
                        with fields():
                            w["job_name"] = ui.input("Job name", value=d.job_name).classes("w-56")
                            path_input(f, "output_dir", "Output folder", "dir", d.output_dir, "w-64")
                            w["seed"] = ui.number("Seed", value=d.seed, precision=0).classes("w-32")
                            preset_state = {"snapshot": None, "applying": False}

                            def on_preset_toggle(e):
                                if f.preset_sync and not e.value:
                                    preset_note.text = ""  # a loaded config without the preset
                                if preset_state["applying"] or f.preset_sync:
                                    return
                                preset_state["applying"] = True
                                try:
                                    if e.value:
                                        snapshot = f.to_config()
                                        snapshot.preset = ""  # the switch already reads on; the snapshot is pre-preset
                                        preset_state["snapshot"] = snapshot
                                        f.from_config(TrainConfig.from_dict(apply_preset(snapshot.to_dict(), "identity")))
                                        keys = sorted(PRESETS["identity"])
                                        preset_note.text = (f"Identity preset applied ({len(keys)} settings: region "
                                                            "loss, LoRA+ 4, adamw8bit, crops without upscaling, SAM 3 "
                                                            "masks, probes every 25, best/early-stop, curriculum 0.5, "
                                                            "loss watch + per-image lr). Set face/body references and "
                                                            "check the probe prompts.")
                                    elif preset_state["snapshot"] is not None:
                                        f.from_config(preset_state["snapshot"])
                                        preset_note.text = "Previous settings restored."
                                finally:
                                    preset_state["applying"] = False

                            w["preset_toggle"] = ui.switch("Identity training preset", value=False,
                                                           on_change=on_preset_toggle) \
                                .tooltip("Applies the measured recipe for character LoRAs; switch off to restore your "
                                         "settings")
                        preset_note = hint("")
                        with ui.row().classes("items-center gap-1 pt-1"):
                            cfg_path = ui.input("Config file", value="configs/my_job.yaml").classes("w-72")

                            def on_load():
                                path = cfg_path.value.strip()
                                try:
                                    cfg, missing, unknown = TrainConfig.load_report(path)
                                    f.from_config(cfg)
                                except Exception as e:  # noqa: BLE001
                                    status_msg.text = f"Could not load config: {e}"
                                    return
                                msg = f"Loaded {path}"
                                if missing:
                                    names = ", ".join(missing[:12]) + (f" and {len(missing) - 12} more" if len(missing) > 12 else "")
                                    if _is_repo_example(path):
                                        msg += f". {len(missing)} newer options use their defaults: {names}"
                                    else:
                                        try:
                                            TrainConfig.append_missing(path, cfg, missing)
                                            msg += f". Added {len(missing)} newer options to the file with their defaults: {names}"
                                        except OSError as e:
                                            msg += f". {len(missing)} newer options use their defaults ({names}); could not " \
                                                   f"add them to the file: {e}"
                                    ui.notify(f"{len(missing)} options are new since this config was saved; they are now in the "
                                              "form with their defaults", type="info")
                                if unknown:
                                    msg += f". Ignored options this version no longer has: {', '.join(unknown)}"
                                status_msg.text = msg

                            def on_save():
                                cfg = f.to_config()
                                path = cfg_path.value.strip() or f"configs/{cfg.job_name}.yaml"
                                if _is_repo_example(path):
                                    status_msg.text = "configs/example.yaml ships with RulzTrainer; save under another name."
                                    return
                                cfg.save(path)
                                status_msg.text = f"Saved {path}"

                            async def browse_cfg():
                                start = cfg_path.value.strip()
                                if not start or not (ROOT / start if not Path(start).is_absolute() else Path(start)).is_file():
                                    (ROOT / "configs").mkdir(exist_ok=True)
                                    start = str(ROOT / "configs")  # the configs folder unless the box names an existing file
                                picked = await pick_path("file", start, [("YAML", "*.yaml;*.yml")], native_cfg)
                                if picked:
                                    cfg_path.value = _relative(picked)

                            native_cfg = _native_dialogs
                            ui.button(icon="description", on_click=browse_cfg).props("flat dense round") \
                                .tooltip(BROWSE_TIP[native_cfg])
                            ui.button("Load", icon="file_open", on_click=on_load).props("flat")
                            ui.button("Save", icon="save", on_click=on_save).props("flat")

                    # ---- Model
                    with section("Model", "Base model, model files and GPUs", "model_training") as sec:
                        nav_items.append(("Model", "model_training", sec))
                        w["model_source"] = ui.radio([HF_SOURCE, FILE_SOURCE], value=HF_SOURCE).props("inline")
                        with fields():  # the model (family) in both modes: with files it is also the HF fallback
                            w["model_preset"] = ui.select(list(MODEL_PRESETS) + [CUSTOM], value="MOSS-Image-Base",
                                                          label="Model").classes("w-60") \
                                .tooltip("The model family. With single files, pick the model the files belong to: "
                                         "anything left empty comes from it on Hugging Face, and the page's "
                                         "model-specific settings (blocks, LoRA layers) follow it.")
                            w["model_path"] = ui.input("Custom path or HF repo id", value=d.model_path) \
                                .classes("w-96 max-w-full") \
                                .bind_visibility_from(w["model_preset"], "value", lambda p: p == CUSTOM)
                            for key in ("model_preset", "model_path"):  # each family has its own default targets
                                w[key].on_value_change(lambda _: f.follow_family_targets())
                            with ui.row().bind_visibility_from(w["model_source"], "value", lambda x: x == HF_SOURCE):
                                async def on_download():
                                    dl_status.text = "Downloading / verifying... (about 30 GB on first download)"
                                    cfg = f.to_config()
                                    dl_status.text = await run.io_bound(download_model, cfg.model_path)

                                ui.button("Download / verify model", icon="download", on_click=on_download) \
                                    .props("outline")
                        dl_status = hint("")
                        with ui.column().classes("w-full gap-3").bind_visibility_from(
                                w["model_source"], "value", lambda x: x == FILE_SOURCE):
                            path_input(f, "transformer_file", "Diffusion model (.safetensors: bf16, fp8, int8, nvfp4)",
                                       "file", "", "w-[40rem]")
                            path_input(f, "text_encoder_file", "Text encoder (.safetensors: Qwen3.5-9B for MOSS-Image, "
                                                               "Qwen3-VL-4B for Krea 2)", "file", "", "w-[40rem]")
                            path_input(f, "vae_file", "VAE (.safetensors: Flux2 for MOSS-Image, Qwen-Image for Krea 2)",
                                       "file", "", "w-[40rem]")
                            path_input(f, "tokenizer_path", "Tokenizer folder (optional; auto-downloads if empty)",
                                       "dir", "", "w-[40rem]")
                            hint("ComfyUI's files for MOSS-Image and Krea 2 (bf16, scaled fp8, int8, nvfp4; the family "
                                 "is read from the diffusion model file). Any file left empty is taken from the model "
                                 "picked above on Hugging Face. FIBO 2.0 has no single-file loader.")
                        subhead("GPUs and memory")
                        with fields():
                            w["multi_gpu"] = ui.select({"off": "single GPU",
                                                        "data_parallel": "2+ GPUs: data parallel (faster)",
                                                        "model_split": "2+ GPUs: model split (memory only)"},
                                                       value=d.multi_gpu, label="Multi-GPU").classes("w-72") \
                                .tooltip("data parallel: a model copy per GPU, each trains its own images, LoRA grads "
                                         "are averaged (about 1.7-2x throughput). model split: blocks divided across "
                                         "GPUs, half the memory per GPU, not faster.")
                            w["device"] = ui.select(gpus, value=gpus[0], label="Training GPU").classes("w-40") \
                                .bind_visibility_from(w["multi_gpu"], "value", lambda x: x == "off")
                            w["devices"] = ui.input("Devices (first = primary)",
                                                    value=",".join(gpus[:2]) if len(gpus) > 1 else d.devices) \
                                .classes("w-56").bind_visibility_from(w["multi_gpu"], "value", lambda x: x != "off")
                            w["split_plan"] = ui.input("Split plan (empty = auto-balance)", value=d.split_plan) \
                                .classes("w-80").props('placeholder="cuda:0=0-25;cuda:1=26-41"') \
                                .bind_visibility_from(w["multi_gpu"], "value", lambda x: x == "model_split") \
                                .tooltip("e.g. cuda:0=0-25;cuda:1=26-41. Give the faster GPU more blocks. Auto-balance "
                                         "counts blocks before the first trained block as forward-only (cheap) and "
                                         "every later block as forward+backward.")
                            w["parallel_jobs"] = (
                                ui.switch("Split samples, probes and caching across GPUs", value=d.parallel_jobs)
                                .bind_visibility_from(w["multi_gpu"], "value", lambda x: x != "off")
                                .tooltip("Each GPU renders its own share of the sample and probe images, encodes its "
                                         "share of the latent cache, and builds its share of the SAM 3 region masks; a "
                                         "second text encoder copy is loaded when a quantized one fits on each GPU. In "
                                         "model split the training model spans both GPUs, so samples and probes split "
                                         "only when a separate sampling model is set."))
                            w["dp_balance"] = (
                                ui.select({"auto": "auto: split each step by GPU speed",
                                           "grow": "grow: faster GPUs fill the slowest one's time",
                                           "off": "off: lockstep"}, value=d.dp_balance,
                                          label="Data parallel balance").classes("w-80")
                                .bind_visibility_from(w["multi_gpu"], "value", lambda x: x == "data_parallel")
                                .tooltip("Unequal GPUs (an RTX 50 card runs nvfp4 on FP4 tensor cores, an RTX 40 card "
                                         "dequantizes) wait for the slowest each step. auto measures them and splits "
                                         "the step's batches by speed (gradient accumulation 4 on two cards: 5+3 "
                                         "instead of 4+4), so the images per step stay the same. grow lets the faster "
                                         "cards run more batches in the slowest one's time (Krea 2 nvfp4: 3+2 instead "
                                         "of 1+1), so a step holds more images: lower the steps to match."))
                        with fields():
                            w["transformer_quant"] = ui.select(QUANTS, value="auto", label="Diffusion model quant") \
                                .classes("w-48").tooltip("auto keeps the file's format; int8/int8_convrot/nvfp4 "
                                                         "quantize on load")
                            w["text_encoder_quant"] = ui.select(QUANTS, value="auto", label="Text encoder quant") \
                                .classes("w-48")
                            w["gradient_checkpointing"] = ui.switch("Gradient checkpointing", value=True)
                            w["checkpoint_keep_blocks"] = ui.number("Blocks kept without recompute (-1 = auto)",
                                                                    value=d.checkpoint_keep_blocks, min=-1,
                                                                    precision=0).classes("w-64") \
                                .bind_visibility_from(w["gradient_checkpointing"], "value") \
                                .tooltip("Checkpointing runs every block's forward twice per step to save memory. "
                                         "Blocks kept here store their activations instead and skip the recompute, "
                                         "about a third of that block's time. -1 measures the largest training "
                                         "bucket at the start and keeps as many of the last blocks as the free GPU "
                                         "memory allows (a step that still runs out keeps fewer); 0 = every block "
                                         "is checkpointed.")
                        with advanced():
                            with fields():
                                w["text_encoder_device"] = ui.select(["auto"] + gpus + ["cpu"], value="auto",
                                                                     label="Text encoder device (caching)") \
                                    .classes("w-56")
                                w["compile_model"] = ui.select(COMPILE_CHOICES, value=d.compile_model,
                                                               label="Compile model (torch.compile)").classes("w-72")                                     .tooltip("Fuses the work between the matrix multiplies into fewer GPU kernels. "
                                             "Measured at 1 MP on an RTX 5060 Ti: MOSS-Image 26% and Krea 2 12% faster "
                                             "per step, after a 25-40 s warm-up. auto compiles runs of 100+ steps "
                                             "(needs triton, installed with the requirements); a block that fails to "
                                             "compile then runs uncompiled.")
                            with fields():
                                w["int8_matmul"] = ui.select({True: "W8A8 int8 matmul (fast)",
                                                              False: "dequantize to bf16 (slow)"},
                                                             value=True, label="int8 layers compute").classes("w-72") \
                                    .tooltip("int8 / int8_convrot layers only. W8A8 quantizes the activations per "
                                             "token to int8 and multiplies them with the stored int8 weight "
                                             "(torch._int_mm), so the weight is never dequantized: about 1.2-1.5x "
                                             "faster than bf16 and far faster than the dequantize path, at under 1% "
                                             "extra forward error. nvfp4 layers have their own setting.")
                                w["int8_backward"] = ui.select({"bf16": "bf16 (exact)",
                                                                "int8": "int8 (faster, approximate)"},
                                                               value="bf16", label="int8 backward").classes("w-56") \
                                    .bind_visibility_from(w["int8_matmul"], "value", lambda x: bool(x)) \
                                    .tooltip("bf16: grad_x = grad_out @ W with the int8 weight widened to bf16 "
                                             "(exact). int8: grad_out is also quantized per token and multiplied with "
                                             "the int8 weight (about 1% gradient error, saves the bf16 weight copy in "
                                             "backward).")
                                w["nvfp4_matmul"] = ui.select({True: "native FP4 (RTX 50 / Blackwell)",
                                                               False: "dequantize to bf16"},
                                                              value=d.nvfp4_matmul, label="nvfp4 layers compute") \
                                    .classes("w-72") \
                                    .tooltip("nvfp4 layers only. Native: the activations are quantized to NVFP4 too "
                                             "and multiplied on the FP4 tensor cores (about 4x faster than a bf16 "
                                             "matmul; W4A4, so the forward error grows from about 9.5% to 13% on "
                                             "random data). The backward stays exact in bf16. GPUs without FP4 tensor "
                                             "cores (RTX 40 and older) always dequantize; pick the RTX 50 card as the "
                                             "training GPU to use it.")
                                w["nvfp4_backward"] = ui.select({"bf16": "bf16 (exact)",
                                                                 "fp4": "FP4 (faster, approximate)"},
                                                                value=d.nvfp4_backward, label="nvfp4 backward") \
                                    .classes("w-56") \
                                    .bind_visibility_from(w["nvfp4_matmul"], "value", lambda x: bool(x)) \
                                    .tooltip("Native nvfp4 layers only. bf16: grad_x = grad_out @ W with the weight "
                                             "expanded to bf16 (exact). FP4: grad_out is quantized to NVFP4 with "
                                             "stochastic rounding (unbiased) and multiplied with the weight requantized "
                                             "along the other axis, on the FP4 tensor cores. Krea 2 on an RTX 5060 Ti: "
                                             "9% faster per step, LoRA gradients at cosine 0.93-0.99 to the exact "
                                             "ones, same loss over 20 steps. Noisier gradients: compare samples before "
                                             "a final run.")
                                w["attention_backend"] = ui.select({"auto": "auto (cuDNN on RTX 50, else PyTorch's pick)",
                                                                    "torch": "PyTorch picks", "cudnn": "cuDNN",
                                                                    "flash": "flash", "efficient": "memory-efficient",
                                                                    "math": "math (slow, exact)"},
                                                                   value=d.attention_backend, label="Attention kernel")                                     .classes("w-72")                                     .tooltip("The attention kernel of the DiT blocks. cuDNN ran Krea 2 steps 15-22% "
                                             "faster on an RTX 5060 Ti at the same loss, so auto pins it on Blackwell "
                                             "GPUs (RTX 50, B200) and lets PyTorch pick elsewhere. A pinned kernel that "
                                             "rejects a call falls back to PyTorch's pick with a warning.")

                    # ---- Dataset
                    with section("Dataset", "Images, captions and resolution", "photo_library") as sec:
                        nav_items.append(("Dataset", "photo_library", sec))
                        with fields():
                            path_input(f, "dataset_dir", "Dataset folder (images + .txt captions)", "dir",
                                       d.dataset_dir)
                            w["trigger_word"] = ui.input("Trigger word (prepended to captions)").classes("w-64")

                            async def on_scan():
                                cfg = f.to_config()
                                scan_info.text = "Scanning..."
                                try:
                                    info, items = await run.io_bound(scan_dataset, cfg)
                                except Exception as e:  # noqa: BLE001
                                    scan_info.text = f"Error: {e}"
                                    return
                                scan_info.text = info
                                gallery.clear()
                                with gallery:
                                    for path, caption in items:
                                        with ui.card().tight().classes("w-28 rz-thumb"):
                                            ui.image(path).classes("w-28 h-28 object-cover")
                                            ui.tooltip(caption or "(no caption)")

                            ui.button("Scan dataset", icon="search", on_click=on_scan).props("outline")
                        scan_info = hint("")
                        gallery = ui.row().classes("w-full gap-2")
                        with fields():
                            w["resolutions"] = ui.select(MP_OPTIONS, multiple=True, value=[1.0],
                                                         label="Training resolutions (megapixels)") \
                                .classes("w-[30rem] max-w-full").props("use-chips") \
                                .tooltip("Area each image is bucketed to, aspect ratio preserved (1 MP = 1024x1024). "
                                         "Pick several to train them all (mixed in every epoch) or, with Resolution "
                                         "step-up, one after another from the lowest.")
                            w["resolution_step_up"] = ui.switch("Resolution step-up", value=d.resolution_step_up) \
                                .tooltip("Train the lowest selected resolution first. When the probe likeness score "
                                         "stops improving, move to the next selected resolution; when it stops "
                                         "improving at the highest one, stop training and save the final LoRA. Needs "
                                         "likeness probes (face/body references, probe every N steps) and replaces "
                                         "their early-stop patience.")
                        with fields().bind_visibility_from(w["resolution_step_up"], "value"):
                            w["step_up_patience"] = ui.number("Probes w/o improvement before next resolution",
                                                              value=d.step_up_patience, min=1, precision=0) \
                                .classes("w-80")
                            w["step_up_stop_patience"] = ui.number("Probes w/o improvement at the highest before "
                                                                   "stopping", value=d.step_up_stop_patience, min=1,
                                                                   precision=0).classes("w-96")
                        with fields():
                            w["enable_buckets"] = ui.switch("Aspect-ratio buckets", value=True)
                            w["flip_augment"] = ui.switch("Horizontal flip augment", value=False)
                        with advanced("Captions and cache"):
                            with fields():
                                w["caption_dropout"] = ui.number("Caption dropout", value=d.caption_dropout, min=0,
                                                                 max=0.5, step=0.01).classes("w-36")
                                w["caption_dropout_keep_trigger"] = ui.switch("Dropped captions keep the trigger word",
                                                                              value=d.caption_dropout_keep_trigger)                                     .tooltip("A dropped caption becomes the trigger word alone instead of the empty "
                                             "prompt, so the trigger by itself summons the subject, the way the LoRA "
                                             "is prompted. Off: the empty prompt (FIBO: its null caption).")
                                w["max_token_length"] = ui.number("Max caption tokens", value=d.max_token_length,
                                                                  min=64, max=4096, step=64, precision=0) \
                                    .classes("w-40")
                                w["recache"] = ui.switch("Force re-cache", value=False)
                        subhead("Automatic crops (character LoRAs)")
                        with fields():
                            w["crop_augment"] = ui.select(["none", "face", "face+bust"], value=d.crop_augment,
                                                          label="Auto crops").classes("w-48") \
                                .tooltip("Adds a face crop and/or a bust crop of every image with a detected face as "
                                         "extra training items, with templated captions. Needs insightface.")

                            async def on_crop_preview():
                                crop_info.text = "Working..."
                                _task_progress.update(text="Preparing face detector...", frac=None, active=True)
                                try:
                                    text, thumbs = await run.io_bound(analyse_and_preview, f.to_config().to_dict(),
                                                                      "crops")
                                except Exception as e:  # noqa: BLE001
                                    crop_info.text = f"Error: {e}"
                                    return
                                finally:
                                    _task_progress.update(active=False)
                                crop_info.text = text
                                crop_row.clear()
                                with crop_row:
                                    for pth, cap in thumbs:
                                        with ui.card().tight().classes("w-32 rz-thumb"):
                                            ui.image(pth).props("fit=contain").classes("w-32 h-32 bg-black")
                                            ui.tooltip(cap)

                            ui.button("Analyse faces & preview crops", icon="center_focus_strong",
                                      on_click=on_crop_preview).props("outline")
                        with advanced("Crop details").bind_visibility_from(w["crop_augment"], "value",
                                                                           lambda x: x != "none"):
                            with fields():
                                w["crop_augment_min_px"] = ui.number("Min crop px (0 = resolution)",
                                                                     value=d.crop_augment_min_px, precision=0) \
                                    .classes("w-48")
                                w["crop_face_scale"] = ui.number("Face crop x face size", value=d.crop_face_scale,
                                                                 step=0.5).classes("w-44")
                                w["crop_bust_scale"] = ui.number("Bust crop x face size", value=d.crop_bust_scale,
                                                                 step=0.5).classes("w-44")
                                w["crop_augment_weight"] = ui.number("Crop sampling weight",
                                                                     value=d.crop_augment_weight, step=0.1) \
                                    .classes("w-44")
                            with fields():
                                w["crop_face_caption"] = ui.input("Face crop caption ({caption}, {raw}, {trigger})",
                                                                  value=d.crop_face_caption).classes("w-96 max-w-full")
                                w["crop_bust_caption"] = ui.input("Bust crop caption", value=d.crop_bust_caption) \
                                    .classes("w-96 max-w-full")
                                w["crop_skip_face_frac"] = ui.number("Skip face crop if face > frac",
                                                                     value=d.crop_skip_face_frac, step=0.05) \
                                    .classes("w-52")
                                w["crop_skip_cover"] = ui.number("Skip crop covering > frac",
                                                                 value=d.crop_skip_cover, step=0.05).classes("w-52")
                        crop_info = hint("")
                        crop_row = ui.row().classes("w-full gap-2")

                    # ---- LoRA
                    with section("LoRA", "Adapter size, layers and blocks", "hub") as sec:
                        nav_items.append(("LoRA", "hub", sec))
                        with fields():
                            w["lora_rank"] = ui.number("Rank", value=d.lora_rank, min=1, max=256, precision=0) \
                                .classes("w-28")
                            w["lora_alpha"] = ui.number("Alpha", value=d.lora_alpha).classes("w-28")
                            w["lora_plus_ratio"] = ui.number("LoRA+ ratio (0 = off, 16 = paper)",
                                                             value=d.lora_plus_ratio, step=1).classes("w-60") \
                                .tooltip("lora_up learns at lr x ratio (faster convergence); ignored by automagic3")
                            w["lora_save_dtype"] = ui.select(["bf16", "fp16", "fp32"], value=d.lora_save_dtype,
                                                             label="Save LoRA as").classes("w-36")
                        subhead("Target layers")
                        with ui.row().classes("w-full items-center gap-1"):
                            for t in TARGETS:
                                w[f"target_{t}"] = ui.checkbox(t, value=t in d.resolved_lora_targets())
                            for t, (families, tip) in EXTRA_TARGETS.items():
                                w[f"target_{t}"] = ui.checkbox(t, value=t in d.resolved_lora_targets()) \
                                    .bind_visibility_from(w["model_preset"], "value",
                                                          lambda p, fs=families: preset_arch(p, w["model_path"].value)
                                                          in fs).tooltip(tip)
                            w["train_text_embedder"] = ui.checkbox("DiT text projection blocks", value=False) \
                                .bind_visibility_from(w["model_preset"], "value",
                                                      lambda p: preset_arch(p, w["model_path"].value) == "moss_image") \
                                .tooltip("Also train the 2-block text projection inside the DiT")
                        with advanced("Blocks to train"):
                            with fields():
                                w["lora_blocks"] = ui.input("Blocks (e.g. 0-41, 21-41, 10-31,40, even; empty = all)",
                                                            value=d.lora_blocks).classes("w-[30rem] max-w-full")
                            hint("Training only later blocks also skips the backward pass through earlier ones, "
                                 "unless LoRA layers sit before the blocks (Krea's text fusion, embedders, MOSS's "
                                 "text projection): their gradient passes through every block.")
                            block_boxes: list = []

                            def n_blocks() -> int:
                                return N_BLOCKS[preset_arch(w["model_preset"].value, w["model_path"].value)]

                            def blocks_from_text():
                                from rulztrainer.lora import parse_blocks
                                n = n_blocks()
                                try:
                                    bl = parse_blocks(w["lora_blocks"].value or "", n)
                                except ValueError:
                                    return
                                sel = set(range(n)) if bl is None else set(bl)
                                for i, cb in enumerate(block_boxes):
                                    if cb.value != (i in sel):
                                        cb.value = i in sel

                            def text_from_blocks():
                                from rulztrainer.lora import format_blocks
                                n = n_blocks()
                                sel = [i for i, cb in enumerate(block_boxes) if cb.value and i < n]
                                txt = "" if len(sel) == n else format_blocks(sel)
                                if (w["lora_blocks"].value or "") != txt:
                                    w["lora_blocks"].value = txt

                            def set_blocks(sel: set[int]):
                                for i, cb in enumerate(block_boxes):
                                    cb.value = i in sel
                                text_from_blocks()

                            block_presets = (("All", lambda n: set(range(n))), ("None", lambda n: set()),
                                             ("First half", lambda n: set(range(0, n // 2))),
                                             ("Second half", lambda n: set(range(n // 2, n))),
                                             ("Middle", lambda n: set(range(n // 4, n - n // 4))),
                                             ("Even", lambda n: set(range(0, n, 2))),
                                             ("Odd", lambda n: set(range(1, n, 2))))
                            with ui.row().classes("w-full items-center gap-1"):
                                ui.label("Presets:").classes("rz-hint")
                                for label, fn in block_presets:
                                    ui.button(label, on_click=lambda _, f_=fn: set_blocks(f_(n_blocks()))) \
                                        .props("dense flat size=sm")
                                ui.button("Invert", on_click=lambda: set_blocks({i for i, cb in enumerate(block_boxes)
                                                                                 if not cb.value and i < n_blocks()})) \
                                    .props("dense flat size=sm")
                            with ui.grid(columns=14).classes("w-full gap-0"):
                                for i in range(N_BLOCKS_MAX):
                                    cb = ui.checkbox(str(i), value=True, on_change=lambda e: text_from_blocks()) \
                                        .props("dense size=xs").classes("text-xs")
                                    if i >= min(N_BLOCKS.values()):  # blocks only some families have
                                        cb.bind_visibility_from(
                                            w["model_preset"], "value",
                                            lambda p, i=i: i < N_BLOCKS[preset_arch(p, w["model_path"].value)])
                                    block_boxes.append(cb)
                            w["lora_blocks"].on_value_change(lambda e: blocks_from_text())
                        with advanced("Resume and training states"):
                            path_input(f, "resume_from", "Resume from LoRA weights (.safetensors, optional)", "file",
                                       "", "w-[40rem]")
                            with fields():
                                w["save_state"] = ui.switch("Save training state (optimizer, scheduler, RNG)",
                                                            value=False) \
                                    .tooltip("The final state is always saved")
                                w["save_state_every"] = ui.number("State every N steps (0 = with each LoRA save)",
                                                                  value=d.save_state_every, precision=0) \
                                    .classes("w-80")
                                w["keep_states"] = ui.number("States to keep (rolling, 0 = all)", value=d.keep_states,
                                                             precision=0).classes("w-60")
                            path_input(f, "resume_state", "Resume from training state folder (state_XXXXXX, optional)",
                                       "dir", "", "w-[40rem]")

                    # ---- Optimization
                    with section("Optimization", "Steps, learning rate, optimizer and timesteps", "speed") as sec:
                        nav_items.append(("Optimization", "speed", sec))
                        with fields():
                            w["steps"] = ui.number("Steps", value=d.steps, precision=0).classes("w-28")
                            w["batch_size"] = ui.number("Batch size", value=d.batch_size, precision=0).classes("w-28")
                            w["learning_rate"] = ui.number("Learning rate", value=d.learning_rate, format="%.2e",
                                                           step=1e-5).classes("w-36")
                            w["optimizer"] = ui.select(["adamw", "adamw8bit", "adafactor", "automagic3"],
                                                       value="adamw8bit", label="Optimizer").classes("w-40")
                            w["lr_scheduler"] = ui.select(["constant", "cosine", "linear"], value=d.lr_scheduler,
                                                          label="LR schedule").classes("w-36")
                            w["warmup_steps"] = ui.number("Warmup steps", value=d.warmup_steps, precision=0) \
                                .classes("w-32")
                        with ui.column().classes("w-full gap-2").bind_visibility_from(
                                w["optimizer"], "value", lambda x: x == "automagic3"):
                            hint("Automagic v3 tunes its own learning rate; LR schedule and warmup are ignored.")
                            with fields():
                                w["automagic_min_lr"] = ui.number("Min lr", value=d.automagic_min_lr, format="%.1e") \
                                    .classes("w-32")
                                w["automagic_max_lr"] = ui.number("Max lr", value=d.automagic_max_lr, format="%.1e") \
                                    .classes("w-32")
                                w["automagic_polarity_history"] = ui.number("Polarity history",
                                                                            value=d.automagic_polarity_history,
                                                                            min=2, max=64, precision=0).classes("w-36")
                                w["automagic_clip_threshold"] = ui.number("Clip threshold",
                                                                          value=d.automagic_clip_threshold,
                                                                          step=0.1).classes("w-32")
                                w["automagic_groups"] = ui.select({"single": "one lr for every layer",
                                                                   "by_type": "an lr per layer type"},
                                                                  value=d.automagic_groups,
                                                                  label="Automagic lr groups").classes("w-56") \
                                    .tooltip("by_type: text-side layers (Krea's text fusion), attention, feed-forward "
                                             "and the rest each adapt their own learning rate, so a small, slower "
                                             "group is not outvoted by the large ones. Fizgig trains Krea 2 this way, "
                                             "with a polarity history of 16.")
                        with advanced("Timesteps, loss weighting and gradients"):
                            with fields():
                                w["gradient_accumulation"] = ui.number("Grad accumulation",
                                                                       value=d.gradient_accumulation,
                                                                       precision=0).classes("w-40")
                                w["max_grad_norm"] = ui.number("Max grad norm (0 = off)", value=d.max_grad_norm) \
                                    .classes("w-44")
                                w["loss_weighting"] = ui.select(["none", "min_snr", "sigma_sqrt"], value="none",
                                                                label="Loss weighting").classes("w-40")
                            with fields():
                                w["timestep_sampling"] = ui.select(["logit_normal", "sigmoid", "uniform", "identity",
                                                                    "content", "style", "reference"],
                                                                   value=d.timestep_sampling,
                                                                   label="Timestep sampling").classes("w-44") \
                                    .tooltip("identity: logit-normal mean -0.5 (detail/low noise). content: sigma = u^p "
                                             "toward low noise (ai-toolkit style). style: mirror toward high noise. "
                                             "reference: the model family's own training recipe (Bria's for FIBO 2.0, "
                                             "the MOSS-Image report's for MOSS).")
                                w["content_power"] = ui.number("Content power", value=d.content_power, step=0.5) \
                                    .classes("w-32").bind_visibility_from(w["timestep_sampling"], "value",
                                                                          lambda x: x in ("content", "style"))
                                w["sigma_min"] = ui.number("Sigma min", value=d.sigma_min, min=0, max=1, step=0.05) \
                                    .classes("w-28")
                                w["sigma_max"] = ui.number("Sigma max", value=d.sigma_max, min=0, max=1, step=0.05) \
                                    .classes("w-28")
                                w["sigmoid_scale"] = ui.number("Sigmoid scale", value=d.sigmoid_scale).classes("w-32") \
                                    .bind_visibility_from(w["timestep_sampling"], "value", lambda x: x == "sigmoid")
                                w["logit_mean"] = ui.number("Logit mean", value=d.logit_mean).classes("w-32") \
                                    .bind_visibility_from(w["timestep_sampling"], "value",
                                                          lambda x: x == "logit_normal")
                                w["logit_std"] = ui.number("Logit std", value=d.logit_std).classes("w-32") \
                                    .bind_visibility_from(w["timestep_sampling"], "value",
                                                          lambda x: x == "logit_normal")
                                w["timestep_shift"] = ui.number("Timestep shift", value=d.timestep_shift) \
                                    .classes("w-32")

                    # ---- Saving & samples
                    with section("Saving & samples", "Checkpoints and preview images", "save") as sec:
                        nav_items.append(("Saving & samples", "save", sec))
                        with fields():
                            w["save_every"] = ui.number("Save every N steps", value=d.save_every, precision=0) \
                                .classes("w-44")
                            w["sample_every"] = ui.number("Sample every N steps (0 = off)", value=d.sample_every,
                                                          precision=0).classes("w-60")
                            w["lora_ema"] = ui.number("Weight average decay (0 = off)", value=d.lora_ema, min=0,
                                                      max=0.999, step=0.005, format="%.3f").classes("w-56") \
                                .tooltip("Checkpoints, samples and likeness probes use a running average of the LoRA "
                                         "over about the last 1 / (1 - decay) images (50 at 0.98); training goes on "
                                         "with the raw weights. Fizgig measured 0.98 at +5 likeness points late in a "
                                         "run with half the swing between checkpoints; 0.995 lagged behind.")
                        w["sample_prompts"] = ui.textarea("Sample prompts (one per line; empty = no samples)") \
                            .classes("w-full")
                        with fields():
                            w["sample_width"] = ui.number("Width", value=d.sample_width, precision=0).classes("w-28")
                            w["sample_height"] = ui.number("Height", value=d.sample_height, precision=0) \
                                .classes("w-28")
                            w["sample_steps"] = ui.number("Steps", value=d.sample_steps, precision=0).classes("w-28")
                            w["sample_cfg"] = ui.number("CFG", value=d.sample_cfg).classes("w-28")
                            w["sample_shift"] = ui.number("Shift", value=d.sample_shift).classes("w-28")
                        hint("FIBO 2.0: Bria samples the dev model at 50 steps / CFG 5 (20-30 steps is fine for "
                             "previews); an empty negative prompt means the null caption it was trained with; sample "
                             "shift and the skip layer are MOSS-only. Prompts can be plain sentences or VGL JSON "
                             "captions.").bind_visibility_from(w["model_preset"], "value",
                                                               lambda p: is_fibo_preset(p, w["model_path"].value))
                        hint("Krea 2: Raw is a base for training and needs many steps with guidance (Krea's pipeline: "
                             "28 steps at their guidance 4.5, CFG 5.5 here). Faster: set "
                             "krea/Krea-2-Turbo as the sampling model (Advanced below) with 8 steps and CFG 1, the "
                             "way Krea uses LoRAs trained on Raw. Sample shift and the skip layer are MOSS-only.") \
                            .bind_visibility_from(w["model_preset"], "value",
                                                  lambda p: preset_arch(p, w["model_path"].value) == "krea2")
                        with fields():
                            path_input(f, "turbo_lora", "Turbo / step-distill LoRA for sampling (optional)", "file",
                                       "", "w-[34rem]")
                            w["turbo_lora_strength"] = ui.number("Turbo LoRA strength", value=d.turbo_lora_strength,
                                                                 step=0.05).classes("w-44")
                        with advanced("Seeds, negative prompt and sampling models"):
                            with fields():
                                w["sample_seed"] = ui.number("Sample seed", value=d.sample_seed, precision=0) \
                                    .classes("w-32")
                                w["sample_walk_seed"] = ui.switch("Walk seed (+1 per prompt)",
                                                                  value=d.sample_walk_seed)
                                w["sample_before_training"] = ui.switch("Samples before training starts",
                                                                        value=d.sample_before_training)
                            w["sample_negative"] = ui.input("Negative prompt (optional)", value=d.sample_negative) \
                                .classes("w-full")
                            with fields():
                                path_input(f, "sample_model_path", "Sampling model (optional; .safetensors, diffusers "
                                                                   "folder or repo id)", "file", "", "w-[34rem]")
                                w["sample_model_quant"] = ui.select(QUANTS, value="auto",
                                                                    label="Sampling model quant").classes("w-48")
                            hint("Use a different checkpoint (e.g. SFT) for sample images only; the trained LoRA is "
                                 "copied onto it. The training model is parked in system RAM while sampling.")
                            with fields():
                                path_input(f, "sample_lora", "Secondary LoRA for sampling (optional, applied with the "
                                                             "turbo LoRA)", "file", "", "w-[34rem]")
                                w["sample_lora_strength"] = ui.number("Secondary LoRA strength",
                                                                      value=d.sample_lora_strength, step=0.05) \
                                    .classes("w-48")

                    def _begin(text):
                        _task_progress.update(text=text, frac=None, active=True)

                    def _end():
                        _task_progress.update(active=False)

                    # ---- Identity similarity
                    with section("Identity similarity", "Face and body scoring for character LoRAs", "face",
                                 open_=False) as sec:
                        nav_items.append(("Identity similarity", "face", sec))
                        hint("Pick reference images on the Dataset tab (\"Face ref\" / \"Body ref\") or list paths "
                             "here. After each sampling pass, every sample's largest face is compared to the "
                             "reference faces with InsightFace (ArcFace; 50%+ generally means the same person), and "
                             "its body proportions (MediaPipe pose ratios) to the body references. Body references "
                             "should show the whole body in a fairly neutral, upright pose.")
                        w["face_refs"] = ui.textarea("Face reference images or folders (one path per line)") \
                            .classes("w-full")
                        with fields():
                            w["face_provider"] = ui.select(["cpu", "cuda"], value=d.face_provider,
                                                           label="Face model device").classes("w-40")
                            w["face_det_size"] = ui.number("Detector size", value=d.face_det_size, precision=0) \
                                .classes("w-32")

                            async def on_face_preview():
                                paths = [p.strip() for p in (w["face_refs"].value or "").splitlines() if p.strip()]
                                if not paths:
                                    face_info.text = "No reference images listed."
                                    return
                                face_info.text = "Working..."
                                _begin("Preparing face model...")
                                try:
                                    text, crops = await run.io_bound(face_preview, paths, w["face_provider"].value,
                                                                     w["face_det_size"].value)
                                except Exception as e:  # noqa: BLE001
                                    face_info.text = f"Error: {e}"
                                    return
                                finally:
                                    _end()
                                face_info.text = text
                                face_row.clear()
                                with face_row:
                                    for crop, caption in crops:
                                        with ui.card().tight().classes("w-32 rz-thumb"):
                                            if crop:
                                                ui.image(crop).props("fit=contain").classes("w-32 h-32 bg-black")
                                            ui.label(caption).classes("text-xs px-1")

                            ui.button("Preview reference faces", icon="face", on_click=on_face_preview) \
                                .props("outline")
                        face_info = hint("")
                        face_row = ui.row().classes("w-full gap-2")
                        w["body_refs"] = ui.textarea("Body reference images or folders (one path per line; "
                                                     "full-body shots)").classes("w-full")
                        with fields():
                            w["body_backend"] = ui.select(["mediapipe", "mediapipe+sam3"], value=d.body_backend,
                                                          label="Body backend").classes("w-48") \
                                .tooltip("mediapipe: skeleton proportion ratios. +sam3: also silhouette widths from a "
                                         "SAM 3 person mask")
                            w["sam3_model"] = ui.input("SAM 3 model (repo id or folder)", value=d.sam3_model) \
                                .classes("w-64") \
                                .bind_visibility_from(w["body_backend"], "value", lambda x: x == "mediapipe+sam3")
                            w["body_device"] = ui.select(["cuda:0", "cuda:1", "cpu"], value=d.body_device,
                                                         label="SAM 3 device").classes("w-36") \
                                .bind_visibility_from(w["body_backend"], "value", lambda x: x == "mediapipe+sam3")

                            async def on_body_preview():
                                paths = [p.strip() for p in (w["body_refs"].value or "").splitlines() if p.strip()]
                                if not paths:
                                    body_info.text = "No body reference images listed."
                                    return
                                body_info.text = "Working..."
                                _begin("Preparing body models...")
                                try:
                                    body_info.text = await run.io_bound(body_preview, paths, w["body_backend"].value,
                                                                        w["sam3_model"].value, w["body_device"].value)
                                except Exception as e:  # noqa: BLE001
                                    body_info.text = f"Error: {e}"
                                finally:
                                    _end()

                            ui.button("Check body references", icon="accessibility_new", on_click=on_body_preview) \
                                .props("outline")
                        body_info = hint("")
                        with fields():
                            w["face_weight"] = ui.number("Face weight (combined score)", value=d.face_weight,
                                                         step=0.1, min=0).classes("w-56")
                            w["body_weight"] = ui.number("Body weight (combined score)", value=d.body_weight,
                                                         step=0.1, min=0).classes("w-56")

                    # ---- Region-weighted loss
                    with section("Region-weighted loss", "Face, person and background weights; identity loss",
                                 "filter_center_focus", open_=False) as sec:
                        nav_items.append(("Region loss", "filter_center_focus", sec))
                        hint("Weights each training image's loss by region: the detected face counts most, the person "
                             "silhouette next (SAM 3 or MediaPipe), the background least. Maps are built once at "
                             "cache time (16 px cells) and normalised to mean 1, so the loss scale is unchanged.")
                        with fields():
                            w["region_weights"] = ui.switch("Region-weighted loss", value=d.region_weights)
                            w["face_loss_weight"] = ui.number("Face weight", value=d.face_loss_weight, step=0.5) \
                                .classes("w-32")
                            w["body_loss_weight"] = ui.number("Person weight", value=d.body_loss_weight, step=0.25) \
                                .classes("w-32")
                            w["background_loss_weight"] = ui.number("Background weight",
                                                                    value=d.background_loss_weight, step=0.1) \
                                .classes("w-40")

                            async def on_region_preview():
                                region_info.text = "Working..."
                                _task_progress.update(text="Preparing face detector...", frac=None, active=True)
                                try:
                                    text, thumbs = await run.io_bound(analyse_and_preview, f.to_config().to_dict(),
                                                                      "regions")
                                except Exception as e:  # noqa: BLE001
                                    region_info.text = f"Error: {e}"
                                    return
                                finally:
                                    _task_progress.update(active=False)
                                region_info.text = text
                                region_row.clear()
                                with region_row:
                                    for pth, cap in thumbs:
                                        with ui.card().tight().classes("w-32 rz-thumb"):
                                            ui.image(pth).props("fit=contain").classes("w-32 h-32 bg-black")
                                            ui.tooltip(cap)

                            ui.button("Preview region maps", icon="grid_on", on_click=on_region_preview) \
                                .props("outline")
                        region_info = hint("")
                        region_row = ui.row().classes("w-full gap-2")
                        with advanced():
                            with fields():
                                w["face_margin"] = ui.number("Face box margin", value=d.face_margin, step=0.05) \
                                    .classes("w-36")
                                w["region_feather_cells"] = ui.number("Feather (cells)", value=d.region_feather_cells,
                                                                      precision=0).classes("w-32")
                                w["region_prior_weight"] = ui.number("Background prior weight (0 = off; +35% step "
                                                                     "time)", value=d.region_prior_weight, step=0.1) \
                                    .classes("w-96 max-w-full") \
                                    .tooltip("Regularises background cells toward the base model's prediction (extra "
                                             "no-grad forward)")
                        with advanced("Experimental: identity loss"):
                            hint("On low-noise samples with a detected face, decodes the face of the model's predicted "
                                 "clean image (with gradients), aligns it and adds weight x (1 - cosine similarity) to a "
                                 "face reference embedding. Needs face references (Identity similarity). Applied "
                                 "weight is capped at 0.5; try 0.1-0.2 only once the rest of the recipe works. Costs a "
                                 "VAE decode + face network per eligible sample on the first GPU (data parallel: "
                                 "primary replica only). The likeness controller switches it off if the face probe "
                                 "score falls well below its best twice in a row.")
                            hint("Warning: with the arcface backend the face probe score is computed by the same "
                                 "network the loss optimises, so the probe metric becomes the training target and can "
                                 "rise without a real likeness gain. Use facenet to keep the probe an independent "
                                 "check, and judge the samples by eye.", warn=True)
                            with fields():
                                w["id_loss_weight"] = ui.number("Identity loss weight (0 = off)",
                                                                value=d.id_loss_weight, min=0, max=0.5, step=0.05) \
                                    .classes("w-56")
                                w["id_loss_backend"] = ui.select(["arcface", "facenet"], value=d.id_loss_backend,
                                                                 label="Backend").classes("w-36")
                                w["id_loss_sigma_max"] = ui.number("Max sigma", value=d.id_loss_sigma_max, min=0,
                                                                   max=1, step=0.05).classes("w-28")
                                w["id_loss_every"] = ui.number("Every k-th eligible", value=d.id_loss_every, min=1,
                                                               precision=0).classes("w-40")
                                w["id_loss_min_face_cells"] = ui.number("Min face (cells)",
                                                                        value=d.id_loss_min_face_cells, min=1,
                                                                        precision=0).classes("w-36")
                                w["id_loss_margin_cells"] = ui.number("Crop margin (cells)",
                                                                      value=d.id_loss_margin_cells, min=0,
                                                                      precision=0).classes("w-40")
                                w["id_loss_random_ref"] = ui.switch("Random reference per step",
                                                                    value=d.id_loss_random_ref)

                    # ---- Loss watch
                    with section("Per-image loss watch", "Per-image verdicts and learning rate", "rule",
                                 open_=False) as sec:
                        nav_items.append(("Loss watch", "rule", sec))
                        hint("Tracks every image's loss normalised by its timestep bucket and, at each epoch boundary, "
                             "classes it as easy / healthy / suspect / stuck / exhausted. Per-image LR then throttles "
                             "stuck images (x0.5, x0.25, x0.125), eases exhausted ones (x0.6) and can exclude images "
                             "that stay stuck. A true per-image learning rate needs batch size 1 (otherwise per-sample "
                             "loss weights). Not available with automagic3.")
                        with fields():
                            w["loss_watch"] = ui.switch("Loss watch (observe)", value=d.loss_watch)
                            w["per_image_lr"] = ui.switch("Per-image learning rate (act on verdicts)",
                                                          value=d.per_image_lr)
                            w["loss_watch_exclude_after"] = ui.number("Exclude after N epochs at lowest throttle "
                                                                      "(0 = never)", value=d.loss_watch_exclude_after,
                                                                      precision=0).classes("w-96 max-w-full")
                        with advanced("Thresholds"):
                            with fields():
                                w["loss_watch_buckets"] = ui.number("Sigma buckets", value=d.loss_watch_buckets,
                                                                    precision=0).classes("w-32")
                                w["loss_watch_beta"] = ui.number("Residual EMA", value=d.loss_watch_beta, min=0,
                                                                 max=0.99, step=0.05).classes("w-32")
                                w["loss_watch_high"] = ui.number("Hard above median x", value=d.loss_watch_high,
                                                                 step=0.05).classes("w-44")
                                w["loss_watch_low"] = ui.number("Easy below median x", value=d.loss_watch_low,
                                                                step=0.05).classes("w-44")
                                w["loss_watch_improve_tol"] = ui.number("Improving = drop >",
                                                                        value=d.loss_watch_improve_tol,
                                                                        step=0.01).classes("w-40")
                                w["loss_watch_healthy_mult"] = ui.number("Healthy multiplier",
                                                                         value=d.loss_watch_healthy_mult,
                                                                         step=0.05).classes("w-40")

                    # ---- Likeness probes
                    with section("Likeness probes & controller", "Best checkpoint, early stop and curriculum",
                                 "insights", open_=False) as sec:
                        nav_items.append(("Likeness probes", "insights", sec))
                        hint("Renders a fixed-seed headshot and/or full-body probe every N steps (small, turbo LoRA), "
                             "scores it against the references, and lets a controller act on the smoothed scores: "
                             "keep the best checkpoint and stop early, nudge the learning rate, and weight the dataset "
                             "/ timesteps toward whichever of face or body is lagging. Works with any optimizer. Needs "
                             "face/body references (Identity similarity).")
                        with fields():
                            w["probe_every"] = ui.number("Probe every N steps (0 = off)", value=d.probe_every,
                                                         precision=0).classes("w-60")
                            w["probe_at_start"] = ui.switch("Probe at step 0 (baseline)", value=d.probe_at_start)
                        w["probe_face_prompt"] = ui.input("Face probe prompt (headshot; empty = none)") \
                            .classes("w-full")
                        w["probe_body_prompt"] = ui.input("Body probe prompt (full body, upright; empty = none)") \
                            .classes("w-full")
                        with fields():
                            w["probe_steps"] = ui.number("Probe steps", value=d.probe_steps, precision=0) \
                                .classes("w-32")
                            w["probe_cfg"] = ui.number("Probe CFG", value=d.probe_cfg, step=0.5).classes("w-32")
                            w["probe_size"] = ui.number("Probe size (px)", value=d.probe_size, step=64,
                                                        precision=0).classes("w-36")                                 .tooltip("The face probe is size x size; the body probe is a 2:3 portrait with the "
                                         "same pixel count (1024: 832 x 1248), so it fits wherever samples of that "
                                         "size do")
                        subhead("Controller actions")
                        with fields():
                            w["likeness_normalize"] = ui.switch("Normalize face and body scores", value=d.likeness_normalize) \
                                .tooltip("Score each on one scale first: 0 = what a stranger scores (the step-0 probe), "
                                         "100 = what your references score against each other. Raw, a body score "
                                         "moves ~10 points over a run and a face score ~60, so the combined score, "
                                         "the curriculum and the noise band would follow the face alone.")
                            w["likeness_track_best"] = ui.switch("Keep best checkpoint (<job>_best.safetensors)",
                                                                 value=d.likeness_track_best)
                            w["likeness_patience"] = ui.number("Early stop after N probes w/o improvement (0 = never)",
                                                               value=d.likeness_patience, precision=0) \
                                .classes("w-96 max-w-full") \
                                .tooltip("Not used while Resolution step-up is on (its two counts replace it).")
                        with fields():
                            w["likeness_lr_control"] = ui.switch("LR multiplier from probe trend",
                                                                 value=d.likeness_lr_control)
                            w["likeness_lr_min"] = ui.number("Min factor", value=d.likeness_lr_min, step=0.1) \
                                .classes("w-28").bind_visibility_from(w["likeness_lr_control"], "value")
                            w["likeness_lr_max"] = ui.number("Max factor", value=d.likeness_lr_max, step=0.1) \
                                .classes("w-28").bind_visibility_from(w["likeness_lr_control"], "value")
                            w["likeness_lr_step"] = ui.number("Step (relative)", value=d.likeness_lr_step, step=0.05) \
                                .classes("w-32").bind_visibility_from(w["likeness_lr_control"], "value")
                        with fields():
                            w["likeness_curriculum"] = ui.switch("Face/body curriculum (dataset weighting + timestep "
                                                                 "bias)", value=d.likeness_curriculum)
                            w["likeness_curriculum_strength"] = ui.number("Strength",
                                                                          value=d.likeness_curriculum_strength,
                                                                          min=0, max=1, step=0.1).classes("w-28") \
                                .bind_visibility_from(w["likeness_curriculum"], "value")
                        with advanced("Probe scoring"):
                            with fields():
                                w["probe_seeds"] = ui.number("Seeds per probe", value=d.probe_seeds, min=1, max=8,
                                                             precision=0).classes("w-36")
                                w["probe_seed"] = ui.number("Probe seed", value=d.probe_seed, precision=0) \
                                    .classes("w-32")
                                w["probe_ema"] = ui.number("Score smoothing (EMA)", value=d.probe_ema, min=0,
                                                           max=0.95, step=0.05).classes("w-48")
                                w["likeness_noise_band"] = ui.number("Noise band (% points)",
                                                                     value=d.likeness_noise_band, step=0.5) \
                                    .classes("w-48")

                    # ---- Configuration race
                    with section("Configuration race", "Several configurations, the probes pick the winner",
                                 "emoji_events", open_=False) as sec:
                        nav_items.append(("Configuration race", "emoji_events", sec))
                        hint("Each configuration below trains until its next probe, then the next one does. Once all of "
                             "them have been probed (a round), the one with the lowest smoothed likeness score is "
                             "dropped; the last one left trains until Steps, which counts its own steps, so the race "
                             "adds the others' steps on top. They all start from the same LoRA initialisation and see "
                             "the same images and noise. A configuration changes only what its row lists; everything "
                             "else on this page is shared. Needs the likeness probes above.")
                        with fields():
                            w["race"] = ui.switch("Race these configurations", value=d.race)
                            w["race_warmup_probes"] = ui.number("Warm-up probe rounds (nobody is dropped)",
                                                                value=d.race_warmup_probes, min=0, precision=0) \
                                .classes("w-72")
                        race_summary = hint("")

                        def race_summary_update():
                            from rulztrainer import race

                            try:
                                rows = w["race_configs"].value
                            except ValueError as e:
                                race_summary.text = f"Check the rows: {e}"
                                return
                            steps, every = int(w["steps"].value or 0), int(w["probe_every"].value or 0)
                            warmup = int(w["race_warmup_probes"].value or 0)
                            if len(rows) < 2 or every <= 0:
                                race_summary.text = ("Add at least two configurations." if len(rows) < 2 else
                                                     "Set 'Probe every N steps' above: the race runs on the probes.")
                                return
                            est = race.estimate_steps(len(rows), warmup, every, steps)
                            race_summary.text = (
                                f"{len(rows)} configurations: {est['rounds']} rounds and {est['probes']} probes, about "
                                f"{est['total']} training steps in all ({est['ratio']:.1f}x the {steps}-step run)"
                                + (f". Steps must be at least {est['race_own_steps']} for the race to finish."
                                   if steps < est["race_own_steps"] else "."))

                        w["race_configs"] = RaceRows(on_change=race_summary_update)
                        w["race_configs"].value = d.race_configs
                        with ui.row().classes("items-center gap-2"):
                            ui.button("Add configuration", icon="add",
                                      on_click=lambda: w["race_configs"].add()).props("outline")
                        for key in ("steps", "probe_every", "race_warmup_probes"):
                            w[key].on_value_change(lambda _: race_summary_update())
                        with advanced("What a configuration may change"):
                            from rulztrainer.race import RACE_KEYS

                            for group, keys in RACE_KEYS.items():
                                ui.label(f"{group}: {', '.join(keys)}").classes("text-xs rz-mono")
                            hint('Lists in brackets (lora_targets=[q, k, v, o]); quote a value with a comma '
                                 '(lora_blocks="10-31,40"). Rank, layers and blocks may differ: each configuration '
                                 'keeps its own LoRA.')

                def jump(exp):
                    exp.value = True
                    ui.run_javascript(f"getHtmlElement({exp.id}).scrollIntoView({{behavior: 'smooth', block: 'start'}})")

                with nav:
                    ui.label("Sections").classes("rz-nav-title")
                    for title, icon, exp in nav_items:
                        ui.button(title, icon=icon, on_click=lambda _, e=exp: jump(e)) \
                            .props("flat no-caps align=left").classes("rz-nav-btn")

        # ------------------------------------------------------------ Wizard (bound to the Train tab's widgets)
        with ui.tab_panel(tab_wizard):
            Wizard(f, gpus, start_job, goto).build()

        # ------------------------------------------------------------ Dataset
        with ui.tab_panel(tab_dataset):
            with ui.column().classes("rz-page gap-4"):
                build_dataset_tab(f)

        # ------------------------------------------------------------ Monitor
        with ui.tab_panel(tab_monitor):
            with ui.column().classes("rz-page gap-4"):
                with ui.row().classes("w-full gap-3"):
                    tiles = {key: stat_tile(label) for key, label in (
                        ("state", "Status"), ("stage", "Stage"), ("step", "Step"), ("loss", "Loss"),
                        ("lr", "Learning rate"), ("eta", "Time left"))}
                job_bar = ui.linear_progress(value=0, show_value=False, size="6px").classes("w-full") \
                    .props("rounded track-color=grey-10")
                error_box = ui.label("").classes("rz-banner error")
                error_box.visible = False
                gpu_box = ui.row().classes("w-full gap-3")
                gpu_widgets: list[dict] = []
                with card():
                    with ui.row().classes("w-full items-center gap-4"):
                        ui.label("Loss").classes("rz-title")
                        ui.space()
                        ui.label("Smoothing").classes("rz-hint")
                        smoothing = ui.slider(min=0, max=0.98, step=0.02, value=0.8).classes("w-48")
                        ui.label().bind_text_from(smoothing, "value", lambda x: f"{x:.2f}").classes("rz-hint w-10")
                    loss_chart = ui.echart(chart_options(
                        _axis("step", min="dataMin"),
                        [_axis("loss", scale=True),
                         _axis("id loss", scale=True, show=False, splitLine={"show": False})],
                        [_line("raw", "#52525B", lineStyle={"width": 1}),
                         _line("smoothed", "#FF3B30", lineStyle={"width": 2}),
                         _line("id loss (smoothed)", "#F4F4F5", yAxisIndex=1,
                               lineStyle={"width": 2, "type": "dashed"})],
                        ["raw", "smoothed"])).classes("w-full h-72")
                with section("Identity similarity", "Face / body / combined score of every sample", "face") as face_exp:
                    face_label = hint("")
                    face_chart = ui.echart(chart_options(_axis("step", min="dataMin"),
                                                         _axis("similarity %", min=0, max=100), [], [])) \
                        .classes("w-full h-60")
                face_exp.visible = False
                with section("Per-image loss watch", "Verdict per training image", "rule") as lw_exp:
                    lw_label = hint("")
                    lw_table = ui.table(columns=[
                        {"name": "image", "label": "image", "field": "image", "align": "left", "sortable": True},
                        {"name": "verdict", "label": "verdict", "field": "verdict", "align": "left", "sortable": True},
                        {"name": "residual", "label": "residual (epoch)", "field": "residual", "sortable": True},
                        {"name": "ema", "label": "residual EMA", "field": "ema", "sortable": True},
                        {"name": "trend", "label": "trend %", "field": "trend", "sortable": True},
                        {"name": "multiplier", "label": "lr x", "field": "multiplier", "sortable": True},
                        {"name": "seen", "label": "seen", "field": "seen", "sortable": True},
                    ], rows=[], row_key="path", pagination=15).props("flat dense").classes("w-full text-xs")
                lw_exp.visible = False
                lw_state = {"sig": None}
                with section("Configuration race", "Smoothed probe score per configuration; the lowest leaves after "
                             "every round", "emoji_events") as race_exp:
                    race_label = hint("")
                    race_table = ui.table(columns=[
                        {"name": "name", "label": "Configuration", "field": "name", "align": "left"},
                        {"name": "state", "label": "Status", "field": "state", "align": "left"},
                        {"name": "score", "label": "Score now", "field": "score"},
                        {"name": "rounds", "label": "Score after each round", "field": "rounds", "align": "left"},
                        {"name": "settings", "label": "Settings", "field": "settings", "align": "left"}],
                        rows=[], row_key="name").classes("w-full")
                race_exp.visible = False
                with section("Likeness probes & controller", "Smoothed probe scores and controller state",
                             "insights") as probe_exp:
                    probe_label = hint("")
                    probe_chart = ui.echart(chart_options(
                        _axis("step", min="dataMin"), _axis("likeness %", min=0, max=100),
                        [_line("face", "#FF3B30"), _line("body", "#F4F4F5"),
                         _line("combined", "#F59E0B", lineStyle={"width": 3}),
                         {"name": "face (raw)", "type": "scatter", "symbolSize": 5, "data": [],
                          "itemStyle": {"color": "#FB7185"}},
                         {"name": "body (raw)", "type": "scatter", "symbolSize": 5, "data": [],
                          "itemStyle": {"color": "#A1A1AA"}}],
                        ["face", "body", "combined", "face (raw)", "body (raw)"])).classes("w-full h-60")
                    probe_row = ui.row().classes("w-full gap-2")
                probe_exp.visible = False
                shown_probes: list[str] = []
                with section("Latest samples", "Click an image to see it full size", "image"):
                    with ui.row().classes("w-full gap-2") as samples_row:
                        hint("No samples yet. They appear here every 'Sample every' steps.")
                with section("Log and saved files", "train.log, checkpoints and training states", "terminal"):
                    saved_label = ui.label("").classes("rz-mono whitespace-pre-wrap")
                    log_view = ui.log(max_lines=600).classes("w-full h-80 rz-log rz-mono")
            shown_samples: list[str] = []
            log_state = {"job": None, "n": 0, "console_shown": False}
            show_full = image_viewer()

            def show(key: str, value: str, sub: str = ""):
                tiles[key][0].text = value
                tiles[key][1].text = sub

            def update_gpus():
                stats = gpu_stats()
                if len(gpu_widgets) != len(stats):
                    gpu_box.clear()
                    gpu_widgets.clear()
                    with gpu_box:
                        for _ in stats:
                            with ui.column().classes("rz-tile gap-1 min-w-[18rem]"):
                                name = ui.label("").classes("text-sm font-semibold text-white")
                                bars = {}
                                for key, color in (("vram", "primary"), ("util", "grey-5")):
                                    with ui.row().classes("w-full items-center gap-2 no-wrap"):
                                        ui.label("VRAM" if key == "vram" else "Load").classes("rz-tile-sub w-10")
                                        bar = ui.linear_progress(value=0, show_value=False, size="6px",
                                                                 color=color).props("rounded track-color=grey-10") \
                                            .classes("flex-1")
                                        text = ui.label("").classes("rz-tile-sub w-28 text-right")
                                    bars[key] = (bar, text)
                            gpu_widgets.append({"name": name, **bars})
                for st, wdg in zip(stats, gpu_widgets):
                    wdg["name"].text = f"GPU {st['index']} · {st['name']}"
                    wdg["vram"][0].value = st["used"] / max(1, st["total"])
                    wdg["vram"][1].text = f"{st['used'] / 1024:.1f} / {st['total'] / 1024:.1f} GB"
                    wdg["util"][0].value = st["util"] / 100
                    wdg["util"][1].text = f"{st['util']}%"

            def update_log():
                if _job_dir is None:
                    return
                if log_state["job"] != str(_job_dir):
                    log_state.update(job=str(_job_dir), n=0, console_shown=False)
                    log_view.clear()
                lp = _job_dir / "train.log"
                if lp.exists():
                    lines = lp.read_text(encoding="utf-8", errors="replace").splitlines()
                    if len(lines) < log_state["n"]:
                        log_view.clear()
                        log_state["n"] = 0
                    for line in lines[log_state["n"]:]:
                        log_view.push(line)
                    log_state["n"] = len(lines)
                st = read_status() or {}
                if not _running() and not log_state["console_shown"] and (st.get("error") or not st.get("done")):
                    cp = _job_dir / "console.log"
                    if cp.exists():
                        tail = cp.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
                        if tail:
                            log_view.push("--- console.log ---")
                            for line in tail:
                                log_view.push(line)
                    log_state["console_shown"] = True

            def poll():
                update_gpus()
                st = read_status()
                running = _running()
                if st is None:
                    show("state", "Starting" if running else "Idle", _job_dir.name if _job_dir else "no job yet")
                    for key in ("stage", "step", "loss", "lr", "eta"):
                        show(key, "-")
                    return
                step, total = st.get("step", 0), st.get("total", 0) or 0
                eta = st.get("eta")
                loss = st.get("loss")
                state = ("Running" if running else "Error" if st.get("error") else "Finished" if st.get("done")
                         else "Stopped")
                show("state", state, _job_dir.name if _job_dir else "")
                show("stage", str(st.get("stage") or "-"), f"resolution {st['resolution']}"
                     if st.get("resolution") else "")
                show("step", f"{step:,} / {total:,}" if total else str(step),
                     f"{step / total * 100:.0f}% done" if total else "")
                if loss is not None:
                    show("loss", f"{loss:.4f}", f"average {st.get('avg_loss') or 0:.4f}")
                show("lr", f"{st['lr']:.2e}" if st.get("lr") is not None else "-")
                show("eta", f"{eta / 60:.1f} min" if eta and running else "-")
                error_box.text = f"Error: {st['error']}" if st.get("error") else ""
                error_box.visible = bool(st.get("error"))
                job_bar.value = (step / total) if total else 0
                losses = st.get("losses") or []
                if losses:
                    start = max(0, step - len(losses))
                    raw = [[start + i + 1, v] for i, v in enumerate(losses)]
                    a = float(smoothing.value)
                    sm, ema = [], None
                    for x, v in raw:
                        ema = v if ema is None else a * ema + (1 - a) * v
                        sm.append([x, ema])
                    loss_chart.options["series"][0]["data"] = raw
                    loss_chart.options["series"][1]["data"] = sm
                    id_pts = st.get("id_losses") or []  # [step, id loss] (identity loss, steps with a face only)
                    if id_pts:
                        ism, ema = [], None
                        for x, v in id_pts:
                            ema = v if ema is None else a * ema + (1 - a) * v
                            ism.append([x, ema])
                        loss_chart.options["series"][2]["data"] = ism
                        loss_chart.options["yAxis"][1]["show"] = True
                        loss_chart.options["legend"]["data"] = ["raw", "smoothed", "id loss (smoothed)"]
                    loss_chart.update()
                fscores = st.get("face_scores") or []
                if fscores:
                    face_exp.visible = True
                    by_prompt: dict[int, list] = {}
                    per_step: dict[int, dict[str, list]] = {}
                    for sc in fscores:
                        if sc.get("combined_percent") is not None:
                            by_prompt.setdefault(sc["prompt_index"], []).append([sc["step"], round(sc["combined_percent"], 1)])
                        d_ = per_step.setdefault(sc["step"], {"face": [], "body": []})
                        if sc.get("percent") is not None:
                            d_["face"].append(sc["percent"])
                        if sc.get("body_percent") is not None:
                            d_["body"].append(sc["body_percent"])
                    series = [{"name": f"combined p{k + 1}", "type": "line", "data": v}
                              for k, v in sorted(by_prompt.items())]
                    for key, name in (("face", "mean face"), ("body", "mean body")):
                        pts = [[stp, round(sum(v[key]) / len(v[key]), 1)] for stp, v in sorted(per_step.items()) if v[key]]
                        if pts:
                            series.append({"name": name, "type": "line", "data": pts, "lineStyle": {"type": "dotted"}})
                    ref_sim = st.get("face_ref_similarity")
                    if ref_sim is not None and series:
                        xs = [pt[0] for v in by_prompt.values() for pt in v] or [0, 1]
                        series.append({"name": "face reference self-similarity", "type": "line", "showSymbol": False,
                                       "lineStyle": {"type": "dashed"},
                                       "data": [[min(xs), ref_sim * 100], [max(xs), ref_sim * 100]]})
                    face_chart.options["legend"]["data"] = [sr["name"] for sr in series]
                    face_chart.options["series"] = series
                    face_chart.update()
                    latest_step = max(sc["step"] for sc in fscores)
                    latest = [sc for sc in fscores if sc["step"] == latest_step]
                    fv = [sc["percent"] for sc in latest if sc.get("percent") is not None]
                    bv = [sc["body_percent"] for sc in latest if sc.get("body_percent") is not None]
                    cv = [sc["combined_percent"] for sc in latest if sc.get("combined_percent") is not None]
                    parts = [f"Step {latest_step}:"]
                    parts.append(f"face {sum(fv) / len(fv):.1f}% ({len(fv)}/{len(latest)} with a face)" if fv
                                 else f"face: none in {len(latest)} sample(s)")
                    if st.get("body_refs"):
                        parts.append(f"body {sum(bv) / len(bv):.1f}% ({len(bv)}/{len(latest)} with a pose)" if bv
                                     else "body: no usable pose")
                    if cv:
                        parts.append(f"combined {sum(cv) / len(cv):.1f}%")
                    if ref_sim is not None:
                        parts.append(f"face ref self-sim {ref_sim * 100:.1f}%")
                    if st.get("body_ref_similarity") is not None:
                        parts.append(f"body ref self-sim {st['body_ref_similarity'] * 100:.1f}%")
                    face_label.text = " | ".join(parts)
                lw = st.get("loss_watch")
                if lw:
                    lw_exp.visible = True
                    sig = (lw.get("epoch"), lw.get("step"))
                    if sig != lw_state["sig"]:
                        lw_state["sig"] = sig
                        counts = ", ".join(f"{k} {v}" for k, v in (lw.get("counts") or {}).items())
                        med = lw.get("median")
                        lw_label.text = (f"Epoch {lw.get('epoch')} (step {lw.get('step')}): {counts} | median residual "
                                         f"{med:.3f} | {lw.get('active')}/{lw.get('total')} images active"
                                         if med is not None else f"Epoch {lw.get('epoch')}: waiting for the first epoch")
                        lw_table.rows = lw.get("rows") or []
                        lw_table.update()
                rc = st.get("race")
                race_exp.visible = bool(rc)
                if rc:
                    names, dropped = rc.get("names") or [], rc.get("dropped") or {}
                    f1 = lambda x: "n/a" if x is None else f"{x:.1f}%"  # noqa: E731
                    rows = []
                    for n in names:
                        state = ("winner" if rc.get("winner") == n else f"out after round {dropped[n]}" if n in dropped
                                 else "training" if rc.get("active") == n else "waiting its turn")
                        per_round = [h["scores"].get(n) for h in rc.get("history") or [] if n in h["scores"]]
                        rows.append({"name": n, "state": state, "score": f1((rc.get("scores") or {}).get(n)),
                                     "rounds": " / ".join(f1(s) for s in per_round) or "-",
                                     "settings": (rc.get("settings") or {}).get(n) or "the page's own settings"})
                    race_table.rows = rows
                    race_table.update()
                    est = rc.get("estimate") or {}
                    race_label.text = (f"{rc.get('winner')} won; it trains on to the end" if rc.get("winner") else
                                       f"Round {rc.get('round')}: {len(rc.get('alive') or [])} of {len(names)} "
                                       f"configurations left, about {est.get('total')} steps in all")
                probes = st.get("probes") or []
                if probes:
                    probe_exp.visible = True
                    def pts(key):
                        return [[pr["step"], round(pr[key], 1)] for pr in probes if pr.get(key) is not None]
                    probe_chart.options["series"][0]["data"] = pts("face_ema")
                    probe_chart.options["series"][1]["data"] = pts("body_ema")
                    probe_chart.options["series"][2]["data"] = pts("combined_ema")
                    probe_chart.options["series"][3]["data"] = pts("face")
                    probe_chart.options["series"][4]["data"] = pts("body")
                    probe_chart.update()
                    lk = st.get("likeness") or {}
                    last = probes[-1]
                    f2 = lambda x: "n/a" if x is None else f"{x:.1f}%"  # noqa: E731
                    norm = (f" (normalized {f2(last.get('face_norm_ema'))} / {f2(last.get('body_norm_ema'))})"
                            if last.get("face_norm_ema") is not None or last.get("body_norm_ema") is not None else "")
                    probe_label.text = (f"Probe @ step {last['step']}: face {f2(last.get('face_ema'))} body {f2(last.get('body_ema'))}"
                                        f"{norm} combined {f2(last.get('combined_ema'))} | best {f2(lk.get('best'))} @ step {lk.get('best_step')}"
                                        f" | {lk.get('since_improve', 0)} probe(s) since improvement | lr x{lk.get('lr_factor', 1.0):.2f}"
                                        f" | face bias {lk.get('face_bias', 0.0):+.2f}" + (" | STOPPED" if lk.get("stopped") else ""))
                    latest_imgs = list(last.get("images") or [])[:6]
                    if latest_imgs != shown_probes:
                        shown_probes[:] = latest_imgs
                        probe_row.clear()
                        with probe_row:
                            for p in latest_imgs:
                                with ui.card().tight().classes("rz-thumb"):
                                    ui.image(p).props("fit=contain").classes("w-40 h-40 bg-black cursor-pointer") \
                                        .on("click", lambda _, q=p: show_full(Path(q)))
                                    ui.label(Path(p).name).classes("text-xs px-1")
                score_by_name = {Path(sc["image"]).name: sc for sc in fscores}
                samples = (st.get("samples") or [])[:12]
                sample_sig = [(p, (score_by_name.get(Path(p).name) or {}).get("combined_percent")) for p in samples]
                if sample_sig != shown_samples:
                    shown_samples[:] = sample_sig
                    samples_row.clear()
                    with samples_row:
                        if not sample_sig:
                            hint("No samples yet. They appear here every 'Sample every' steps.")
                        for p, pct in sample_sig:
                            with ui.card().tight().classes("rz-thumb"):
                                ui.image(p).props("fit=contain").classes("w-48 h-48 bg-black cursor-pointer") \
                                    .on("click", lambda _, q=p: show_full(Path(q)))
                                caption = Path(p).name
                                sc_ = score_by_name.get(Path(p).name)
                                if sc_ is not None:
                                    caption += score_caption(sc_)
                                ui.label(caption).classes("text-xs px-1")
                saved = list(st.get("saved") or [])
                states = list(st.get("states") or [])
                saved_label.text = ("Saved:\n" + "\n".join(saved) if saved else "No LoRA saved yet.") \
                    + ("\nStates:\n" + "\n".join(states) if states else "")
                update_log()

            ui.timer(2.0, poll)

        # ------------------------------------------------------------ Samples
        with ui.tab_panel(tab_samples):
            with ui.column().classes("rz-page gap-4"):
                build_samples_tab(f)

        # ------------------------------------------------------------ Help
        with ui.tab_panel(tab_help):
            with ui.column().classes("rz-page"):
                with card():
                    ui.markdown(HELP).classes("rz-help")

    # ---------------------------------------------------------------- action bar (Train and Monitor tabs)
    with ui.footer().classes("rz-footer") as footer:
        with ui.row().classes("rz-footer-row w-full items-center gap-3"):
            with ui.column().classes("gap-1 min-w-[12rem] flex-1"):
                status_msg = ui.label("").classes("rz-status-msg")
                with ui.row().classes("items-center gap-2 no-wrap") as task_row:
                    ui.spinner(size="sm")
                    task_text = ui.label("").classes("text-sm")
                    task_bar = ui.linear_progress(value=0, show_value=False, size="6px").classes("w-48") \
                        .props("rounded track-color=grey-10")
                task_row.visible = False
            with ui.row().classes("items-center gap-2 no-wrap"):
                ui.button("Start training", icon="play_arrow", on_click=start_job).props("unelevated")
                ui.button("Stop", icon="stop", on_click=lambda: setattr(status_msg, "text", stop_training())) \
                    .props("outline color=white").tooltip("Stop after the current step and save the LoRA")
                ui.button("Kill", icon="dangerous", on_click=lambda: setattr(status_msg, "text", kill_training())) \
                    .props("flat color=negative").tooltip("End the process now, without a final save")
    footer.bind_value_from(tabs, "value", lambda t: t in ("Train", "Monitor", tab_train, tab_monitor))

    def task_tick():
        if not _task_progress["active"]:
            task_row.visible = False
            return
        task_row.visible = True
        task_text.text = _task_progress["text"]
        frac = _task_progress["frac"]
        task_bar.visible = frac is not None
        task_bar.value = frac or 0

    def pill_tick():
        st = read_status() or {}
        running = _running()
        state = "live" if running else "error" if st.get("error") else "done" if st.get("done") else ""
        pill_dot.classes(replace=f"rz-dot {state}")
        if running:
            stage = str(st.get("stage") or "starting")
            pill_text.text = (f"Training · step {st.get('step', 0):,} / {st.get('total') or 0:,}"
                              if stage == "training" else stage[:1].upper() + stage[1:])
        else:
            pill_text.text = {"error": "Last run failed", "done": "Last run finished"}.get(state, "Idle")

    ui.timer(0.5, task_tick)
    ui.timer(2.0, pill_tick)
    pill_tick()

    tab_names = ("Wizard", "Train", "Dataset", "Monitor", "Samples", "Help")

    def remember_tab(e):
        name = e.value if isinstance(e.value, str) else e.value.props.get("name")
        ui.run_javascript(f"try {{ localStorage.setItem('rulz.tab', {json.dumps(name)}) }} catch (e) {{}}")

    async def restore_tab():
        """Open the tab used last in this browser (the Monitor wins while a job runs)."""
        if job_running:
            return
        try:
            name = await ui.run_javascript("localStorage.getItem('rulz.tab')", timeout=3)
        except Exception:  # noqa: BLE001  (storage blocked, slow client)
            return
        if name in tab_names:
            tabs.value = name

    tabs.on_value_change(remember_tab)
    ui.timer(0.05, restore_tab, once=True)

    if job_running and _job_dir is not None and (_job_dir / "config.yaml").is_file():
        # reattached to a running job: show its settings instead of the defaults
        try:
            f.from_config(TrainConfig.load(_job_dir / "config.yaml"))
            status_msg.text = f"Reconnected to the running job '{_job_dir.name}' (settings loaded from its config)"
        except Exception as e:  # noqa: BLE001
            status_msg.text = f"Reconnected to the running job '{_job_dir.name}' (could not load its config: {e})"


HELP = """
## Wizard
New to LoRA training? The **Wizard** tab walks through a run one topic at a time: what you are training, the model,
your GPUs, the dataset, the training settings and the preview images, then a review with any problems it finds.
Every setting is explained and comes with a recommendation worked out from your GPUs (memory) and your dataset
(image count and size): *Use* takes one, *Apply recommended* takes every one on the page. The wizard edits the same
settings as the Train tab, so you can switch between the two at any time.

## Quick start
1. **Dataset**: a folder of images; each image may have a same-name `.txt` caption. Optional trigger word is prepended.
2. **Model**: either a Hugging Face repo (Base / SFT, about 30 GB, stored under `models/`) or three single
   ComfyUI-style files: diffusion model, Qwen3.5-9B text encoder and Flux2 VAE.
3. Click **Start training** (the bar at the bottom of the Train and Monitor tabs). First run caches VAE latents and
   text embeddings; the text encoder is loaded once for caching and unloaded before the diffusion model is loaded.
4. LoRAs are written to `output/<job>/<job>_<step>.safetensors` and `output/<job>/<job>.safetensors` (bf16 by default).

## Quantization
*Diffusion model quant* and *Text encoder quant*: `auto` keeps whatever the file already is (bf16 stays bf16, a
ComfyUI int8/nvfp4 file stays quantized), `none` dequantizes to bf16, and `int8`, `int8_convrot` or `nvfp4`
quantize on load. Only the frozen base weights are quantized; the LoRA itself always trains in fp32.
int8 roughly halves VRAM for the base model (about 5 GB for the DiT); nvfp4 quarters it.

*Blocks kept without recompute* (next to *Gradient checkpointing*): checkpointing runs each block's forward twice
per step to save memory. Kept blocks store their activations and skip the second run, with identical results.
`-1` (default) measures the largest bucket at the start and keeps as many of the last blocks as fit; `0` keeps
none. On 16 GB cards the gain is small (0-4% per step); cards with more free memory keep more blocks.

*int8 layers compute* (Model, Advanced): with **W8A8** (default) int8 / int8_convrot layers quantize the activations
per token to int8 and run an int8 x int8 matmul (`torch._int_mm`) on the stored weight, so training is faster than
bf16 instead of ~3x slower; the extra error is under 1% of the output. *int8 backward* `bf16` is exact (the weight
is widened to bf16 only for the gradient), `int8` also quantizes the gradient (faster, ~1% gradient error).
`dequantize to bf16` is the old slow path for A/B comparison.

*nvfp4 layers compute* (Model, Advanced): on RTX 50 / Blackwell GPUs `native FP4` quantizes the activations to NVFP4
as well and multiplies on the FP4 tensor cores (about 4x faster than a bf16 matmul, W4A4: more forward error than
dequantizing), with an exact bf16 backward. GPUs without FP4 tensor cores always dequantize (a fused kernel). Pick
the RTX 50 card as the training GPU to use it; the log says which GPUs run native. *nvfp4 backward* `FP4` (opt-in)
runs the gradient through those layers on the FP4 tensor cores too: about 9% faster per step on Krea 2, with
noisier gradients (cosine 0.93-0.99 to the exact ones); `bf16` stays exact. *Attention kernel*: `auto` pins cuDNN's
attention on Blackwell GPUs (RTX 50, B200), where it ran Krea 2 steps 15-22% faster at the same loss, and lets
PyTorch pick elsewhere; `torch`, `cudnn`, `flash`, `efficient`, `math` force one. A kernel that rejects a call falls
back to PyTorch's pick with a warning.

## Training resolutions
*Training resolutions* takes one or more megapixel values (1 MP = 1024x1024); each image is bucketed to every
selected area with its aspect ratio kept and latents are cached per resolution. Several values are mixed in every
epoch, or, with *Resolution step-up* (next to the resolutions), trained lowest first: when the probe likeness score
stops improving for *N probes* the next resolution starts, and at the highest resolution *N probes* without
improvement end training (two separate counts, default 1 each). Step-up needs probes (face/body references).

## Training states
Turn on *Save training state* (LoRA, Advanced) to also write `state_XXXXXX/` folders (LoRA in fp32 + optimizer +
scheduler + RNG). Point *Resume from training state folder* at one of them to continue exactly where it stopped.

## Unequal GPUs
*Data parallel balance* (Model, next to the GPU settings): in data parallel every step waits for the slowest GPU.
auto (default) times each GPU during the first steps and splits a step's images by speed when that is at least 5%
faster per image, keeping the images per step the same (it needs several per step: gradient accumulation 3 or
more). grow lets the faster GPU train more images in the slower one's time, so each step holds more images: lower
the steps to match. The log says what was measured and chosen.

## Compile
*Compile model* (Model, Advanced; default auto): torch.compile fuses the small steps between the matrix multiplies
into fewer GPU kernels. Measured at 1 MP on an RTX 5060 Ti: MOSS-Image 26% and Krea 2 12% faster per step, after a
warm-up of under a minute. auto compiles runs of 100 steps or more (triton is installed with the requirements).

## Weight average
*Weight average decay* (Saving & samples, default 0.98, 0 = off): checkpoints, samples and likeness probes use a
running average of the LoRA over about the last 1 / (1 - decay) images (50 at 0.98), while training goes on with the
raw weights. Fizgig measured 0.98 at 5 likeness points above no average late in a run, with half the swing between
checkpoints; longer averages (0.995) lag behind. Training states keep both the raw weights and the average.

## Sampling model
Optionally sample with a different checkpoint than you train on (for example train on Base, preview on SFT).
The trained LoRA is copied onto it before each sampling pass; the training model waits in system RAM meanwhile.

## Turbo LoRA
An optional ComfyUI-format LoRA (for example an 8-step distill) applied only while generating sample images,
never during the training step. Set *Sample steps* to match it.

## Identity training (character LoRAs)
The **Identity training preset** switch (top of the Train tab, or *Character / person* in the Wizard) applies the
measured recipe below in one go and restores your previous settings when switched off; the CLI equivalent is
`--preset identity`.
Measured on a small test set (120 steps, same seeds): region-weighted loss gave the best face likeness (+2 points
over baseline), LoRA+ at ratio 4 a small gain (ratio 16 on a 2e-4 lr was unstable), `identity` timesteps were
neutral, and auto crops *hurt* when the crops had to be upscaled. Suggested recipe: *Region-weighted loss* on
(face 3 / person 1.5 / background 0.3), *LoRA+ ratio* 4 with `adamw8bit`, *Auto crops* only when your photos are
high-resolution (crops are never upscaled by default), timesteps left at `logit_normal`.
LoRA rules of thumb: LoRA lr is ~10x a full fine-tune lr and nearly rank-independent; keep MLP layers in the
targets; small batches with more steps beat large batches. *Dropped captions keep the trigger word* (Dataset
section, default on): caption dropout trains on the trigger alone instead of the empty prompt, so the trigger by
itself summons the subject.

**Probe scores.** The face score (ArcFace: a stranger ~10-25, the references against each other ~60-80) and the body
score (proportions: every human ~80, the references ~90-96) are not on one scale, so with *Normalize face and body
scores* (default on) each is first mapped to 0 = a stranger's score (the step-0 probe, so keep *Probe at start* on)
and 100 = the references' own self-similarity. The combined score, best checkpoint, early stop, curriculum and race
decisions use these; the Monitor shows the raw scores next to them.

## Configuration race
*Configuration race* (Train tab, after the likeness probes) trains several configurations in turn: each trains until
its next probe, then the next one. After a round (every configuration probed once more) the one with the lowest
smoothed likeness score is dropped, after *Warm-up probe rounds* rounds without drops. The last one trains until
*Steps*, counted in its own steps, so the race adds the others' steps (the section shows the total). Each row names a
configuration and the settings it changes, for example `learning_rate=2e-4, lora_rank=32` or
`lora_targets=[q, k, v, o], timestep_sampling=identity`; the rest of the page is shared. All of them start from the
same LoRA initialisation and see the same images and noise. Files of a configuration carry its name while the race
runs (`<job>_<name>_<step>`, `<job>_<name>_best`); a dropped one is saved as `<job>_<name>_dropped`; the winner
continues with the run's own names, and `<job>_winner.yaml` holds its settings. Training states are only saved after
the race. The Monitor tab shows each configuration's score per round.

## Using in ComfyUI
ComfyUI supports MOSS-Image natively. Copy the `.safetensors` to `ComfyUI/models/loras/` and use the standard
**Load LoRA** node after *Load Diffusion Model*.

## FIBO 2.0 (Bria)

Pick **FIBO-2.0-dev** as the model (the gated `briaai/Fibo-2.0-dev` repo: accept its terms on Hugging Face and log
in with `hf auth login` first). Train on **dev**, the full non-distilled base; **Edit Turbo** shares its architecture,
so the LoRA loads onto it too. Captions can be plain `.txt` sentences or structured VGL `.json` captions (a `.json`
next to the image wins). The 8B transformer is quantized to int8 (W8A8) automatically on 16 GB cards; the Qwen3-VL
text encoder (8.9 GB) fits one card for caching and is unloaded afterwards.

FIBO LoRAs are saved in the diffusers / PEFT format (ComfyUI has no FIBO support yet):

```python
from diffusers import BriaFibo2Pipeline  # diffusers main (PR #14991)
pipe = BriaFibo2Pipeline.from_pretrained("briaai/Fibo-2.0-edit-turbo", dtype=torch.bfloat16).to("cuda")
pipe.transformer.load_lora_adapter("output/<job>", weight_name="<job>.safetensors")
edited = pipe(prompt=vgl(scene), image=source, num_inference_steps=4, guidance_scale=1.0).images[0]
```

## Krea 2

Pick **Krea-2-Raw** as the model (the gated `krea/Krea-2-Raw` repo: accept the Krea 2 Community License on Hugging
Face and log in with `hf auth login` first; about 35 GB). Raw is the checkpoint Krea publishes for training; its LoRAs
are meant for **Krea-2-Turbo**, the 8-step distill, as Krea's own Krea-2 LoRAs are. Set `krea/Krea-2-Turbo` as the
*sampling model* (Saving & samples, Advanced) with 8 steps and CFG 1 to preview the LoRA the way it will be used,
much faster than sampling Raw (28-52 steps with guidance). The 12B transformer is 25 GB in bf16; *Diffusion model
quant* `auto` makes it int8 (13.5 GB) when the GPUs it runs on have 22 GB or more together (one 24 GB card, or two
16 GB cards with *model split*) and nvfp4 (about 8 GB) on a single 16 GB card.

Krea 2 also loads from **single ComfyUI files** (Model, "Single safetensors files"): the diffusion model
(bf16, scaled fp8, int8 or nvfp4, finetunes and Turbo included), the Qwen3-VL-4B text encoder and the Qwen-Image VAE,
with no Hugging Face repo needed when all three are given. Pick Krea-2-Raw as the model so the page's Krea settings
show; a file name containing "turbo" marks the distilled model. The sampling model can be a Turbo single file too.

Krea 2 LoRAs train the attention gate and the **text fusion** blocks by default, on top of the usual attention and
feed-forward layers: the usual recipe leaves text fusion under-trained (boosting it on finished LoRAs lifted
likeness 2-7 points in Fizgig's tests). Untick *text_fusion* / *attn_gate* under LoRA, Target layers, for the old
set; ComfyUI loads both.

*Sample CFG* uses the same convention for every model (1 = off). Krea's own guidance scale g is CFG g + 1 here:
Krea's 3.5 for Raw is 4.5.

Krea 2 LoRAs are saved in the diffusers / PEFT format, with the same module names as Krea's published LoRAs:

```python
from diffusers import Krea2Pipeline
pipe = Krea2Pipeline.from_pretrained("krea/Krea-2-Turbo", torch_dtype=torch.bfloat16).to("cuda")
pipe.load_lora_weights("output/<job>/<job>.safetensors")
image = pipe("<trigger>, ...", num_inference_steps=8, guidance_scale=0.0).images[0]
```
"""


def _free_port(host: str, start: int, tries: int = 10) -> int:
    for port in range(start, start + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
                return port
            except OSError:
                print(f"Port {port} is busy, trying {port + 1}...")
    raise SystemExit(f"No free port in range {start}-{start + tries - 1}")


def _tailscale_exe() -> str | None:
    exe = shutil.which("tailscale")
    if exe is None and sys.platform == "win32":
        default = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Tailscale" / "tailscale.exe"
        exe = str(default) if default.is_file() else None
    return exe


def tailscale_serve(port: int) -> str:
    """Publish the WebUI to your tailnet with `tailscale serve`: HTTPS with a Tailscale certificate, reachable only
    from your own Tailscale devices, while the WebUI itself stays bound to 127.0.0.1. The HTTPS port is the WebUI's
    port, so a `tailscale serve` already on 443 (ComfyUI, for example) is left alone. Returns the URL."""
    exe = _tailscale_exe()
    if exe is None:
        raise RuntimeError("tailscale is not installed or not on PATH (https://tailscale.com/download)")
    try:
        state = json.loads(subprocess.run([exe, "status", "--json"], capture_output=True, text=True, timeout=15,
                                          check=True).stdout)
    except (subprocess.SubprocessError, json.JSONDecodeError) as e:
        raise RuntimeError(f"tailscale status failed ({e}); is Tailscale running and logged in?") from e
    if state.get("BackendState") != "Running":
        raise RuntimeError(f"Tailscale is {state.get('BackendState')}, not connected; run `tailscale up` first")
    res = subprocess.run([exe, "serve", "--bg", "--yes", f"--https={port}", f"http://127.0.0.1:{port}"],
                         capture_output=True, text=True, timeout=60)
    if res.returncode != 0:
        raise RuntimeError("tailscale serve failed (the tailnet needs HTTPS certificates: admin console > DNS > "
                           f"HTTPS Certificates):\n{(res.stderr or res.stdout).strip()}")
    return f"https://{state['Self']['DNSName'].rstrip('.')}:{port}"


def tailscale_park(port: int) -> None:
    """Point the WebUI's tailnet address at a short notice instead of the stopped server, so it says what is wrong
    rather than answering 502 Bad Gateway. The next start replaces it with the proxy again."""
    exe = _tailscale_exe()
    if exe:
        note = (f"RulzTrainer is not running on {socket.gethostname()}.\n\nStart run_webui_tailscale.bat there (or "
                "python webui.py --tailscale), keep its window open, then reload this page.\n\nTo remove this address "
                f"for good: tailscale serve --https={port} off")
        subprocess.run([exe, "serve", "--bg", "--yes", f"--https={port}", f"text:{note}"], capture_output=True,
                       timeout=20)


def on_exit_signals(callback) -> None:
    """Also run `callback` when the WebUI ends without a normal shutdown: its console window closed, logoff or system
    shutdown on Windows (about 5 s are allowed), a terminal hangup (SIGHUP) elsewhere. Ctrl+C keeps its usual path."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        def handler(event):
            if event in (2, 5, 6):  # CTRL_CLOSE_EVENT, CTRL_LOGOFF_EVENT, CTRL_SHUTDOWN_EVENT
                callback()
            return False  # carry on with the default handling (the process ends)

        on_exit_signals.handler = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)(handler)  # keep a reference
        ctypes.windll.kernel32.SetConsoleCtrlHandler(on_exit_signals.handler, True)
    else:
        import signal

        def hangup(*_):
            callback()
            os.kill(os.getpid(), signal.SIGTERM)  # then shut down the usual way

        signal.signal(signal.SIGHUP, hangup)


if __name__ in {"__main__", "__mp_main__"}:
    from fastapi import Request

    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7870)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--open", action="store_true", help="open the UI in the default browser once ready")
    ap.add_argument("--tailscale", action="store_true", default=os.environ.get("RULZ_TAILSCALE") == "1",
                    help="also serve the UI over HTTPS to your Tailscale devices (tailscale serve); "
                         "RULZ_TAILSCALE=1 does the same")
    a = ap.parse_args()
    port = _free_port(a.host, a.port)
    setup_theme()
    if a.tailscale:
        try:
            url = tailscale_serve(port)
        except (RuntimeError, OSError, subprocess.SubprocessError) as e:
            raise SystemExit(f"[tailscale] {e}") from e
        print(f"Tailscale: open {url} on any of your Tailscale devices (tailnet only)")
        parked = threading.Event()

        def park():
            if not parked.is_set():  # once, whichever way the WebUI ends
                parked.set()
                tailscale_park(port)

        app.on_shutdown(park)
        on_exit_signals(park)

    @ui.page("/")
    def index(request: Request):
        # A real page function, not NiceGUI's script mode: script mode re-executes this file for every page load,
        # which reset the job globals, so a reload lost the running job.
        build_page(local=is_local_viewer(request))

    ui.run(host=a.host, port=port, title="RulzTrainer", show=a.open, reload=False, dark=True,
           favicon=ASSETS / "icon.png")
