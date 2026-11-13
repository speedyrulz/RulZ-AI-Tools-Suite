"""RulzTrainer WebUI (NiceGUI). Run: python webui.py [--port 7870] [--open]

Training runs in a subprocess so Stop frees GPU memory cleanly; the UI polls
<output>/<job>/status.json and train.log. The running job (pid + job dir) is recorded in .webui_job.json, so a
reloaded page, a second browser or a restarted WebUI reattaches to it.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from nicegui import app, background_tasks, run, ui

from rulztrainer.archs import get_arch
from rulztrainer.config import (AUTOMAGIC, KEEL, OPTIMIZERS, ARCH_DEFAULT_TARGETS, ARCH_SAMPLE_DEFAULTS, ARCH_TURBO, MEGAPIXEL_CHOICES,
                                MODEL_PRESETS, ARCH_SAMPLE_LORA, VIDEO_ARCHS, VIDEO_EXTS,
                                PRESET_PROBE_PROMPTS, PRESETS, TrainConfig, apply_preset, mp_to_side, side_to_mp)
from rulztrainer import __version__, bodysim, facesim, history, lookfilter, machine, oom_recovery, promptcheck
from rulztrainer.train_queue import TrainQueue
from rulztrainer.dataset import skipped as dataset_skipped
from rulztrainer.pictures import to_rgb
from rulztrainer.losswatch import read_verdicts
from rulztrainer.regionwatch import describe as region_watch_line

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
    "text_in": (("qwen_image21", "flux3_klein", "flux2_klein"),
                "the text projection that brings the text encoder's features into the model"),
    "embed": (("fibo2", "krea2", "qwen_image21", "flux3_klein", "flux2_klein"),
              "image / text / timestep embedders and the final layer"),
    "mode_blocks": (("flux3_klein",), "the 5 text and 5 image per-mode blocks that run before the 28 joint ones (on by "
                                      "default: the attention / feed-forward groups apply there too)"),
    "modulation": (("flux3_klein", "flux2_klein"), "the shift / scale / gate projections of the timestep"),
    "token_refiner": (("minimax_h3",), "the 2 token-refiner blocks over the text (musubi-tuner leaves them out; the "
                                       "training adapters train them)"),
}
N_BLOCKS = {"moss_image": 42, "fibo2": 44, "krea2": 28, "qwen_image21": 32, "flux3_klein": 28, "flux2_klein": 32}
# Ideogram 4: 34 blocks; its per-block modulation, text projection and embedders are LoRA targets too
N_BLOCKS["ideogram4"] = 34
for _t in ("adaln", "text_in", "embed"):
    EXTRA_TARGETS[_t] = (EXTRA_TARGETS[_t][0] + ("ideogram4",), EXTRA_TARGETS[_t][1])
N_BLOCKS["minimax_h3"] = 50
N_BLOCKS["p_h3_image"] = 50  # P-H3-Image: H3's transformer
N_BLOCKS["minimax_h3_image"] = 50  # MiniMax-H3-Image: H3's transformer without the audio stream
EXTRA_TARGETS["token_refiner"] = (("minimax_h3", "p_h3_image", "minimax_h3_image"),
                                  EXTRA_TARGETS["token_refiner"][1])
N_BLOCKS["pan2"] = 44  # PAN 2: 44 joint blocks; its 4 text-refiner blocks are a LoRA target of their own
EXTRA_TARGETS["token_refiner"] = (EXTRA_TARGETS["token_refiner"][0] + ("pan2",),
                                  "the token-refiner blocks over the text (MiniMax H3: 2, which musubi-tuner leaves "
                                  "out and the training adapters train; PAN 2: 4)")
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
# the single files to download (Comfy-Org's repackages): which format
FILE_QUANT_LABELS = {"int8_convrot": "int8 ConvRot (about half of bf16)", "fp8_scaled": "fp8 scaled",
                     "bf16": "bf16 (full size)"}
FILE_QUANT_LABELS["nvfp4_mixed"] = "nvfp4 mixed (about a quarter of bf16)"  # Ideogram 4's
FILE_QUANT_LABELS["nvfp4"] = "nvfp4 (about a third of bf16)"  # Ideogram 4's Qwen3-VL-8B text encoder
TURBO_MODES = {"auto": "The model's turbo LoRA (downloaded)", "none": "None: full-quality sampling",
               "file": "A LoRA file"}
SAMPLE_LORA_MODES = {"auto": "The model's sampling LoRA (downloaded; MOSS-Image Base: the SFT LoRA)", "none": "None",
                     "file": "A LoRA file"}
ADAPTER_MODES = {"auto":"The model's training adapter (downloaded; Qwen-Image 2.1)", "none": "None",
                 "file": "A LoRA file"}
ADAPTER_MODES["auto"] = ("The model's training adapter (downloaded; Qwen-Image 2.1, MiniMax H3, P-H3-Image, "
                         "MiniMax-H3-Image)")
# MiniMax H3's Comfy-Org formats (h3/loading.py: the pruned files drop the ~13B AdaLN parameters)
FILE_QUANT_LABELS.update({"pruned_int8_convrot": "pruned int8 ConvRot (21.0 GB, for 16 GB cards)",
                          "pruned_bf16": "pruned bf16 (40.2 GB)", "nvfp4_awq": "NVFP4 AWQ (15.7 GB)"})
# MiniMax-H3-Image's 4-bit file (ComfyUI's asym_w4a8_int8): decoded on read to the int8 weight it stands for
FILE_QUANT_LABELS["w4a8"] = "w4a8 (4-bit file, 12.5 GB to download; int8 in memory)"
SAMPLE_ASPECTS = {"1:1": "1:1 square", "4:5": "4:5 portrait", "3:4": "3:4 portrait", "2:3": "2:3 portrait",
                  "9:16": "9:16 portrait", "5:4": "5:4 landscape", "4:3": "4:3 landscape", "3:2": "3:2 landscape",
                  "16:9": "16:9 landscape", "21:9": "21:9 wide"}


def file_quant_options(family: str, kind: str) -> dict[str, str]:
    """The formats Comfy-Org offers a family's diffusion model ("model") or text encoder ("text") in."""
    from rulztrainer.config import COMFY_DIFFUSION_QUANTS, COMFY_TEXT_ENCODER_QUANTS

    table = COMFY_DIFFUSION_QUANTS if kind == "model" else COMFY_TEXT_ENCODER_QUANTS
    return {q: FILE_QUANT_LABELS[q] for q in table.get(family, ("int8_convrot", "bf16"))}


def aspect_size(aspect: str, mp: float) -> tuple[int, int]:
    """(width, height) for an aspect ratio "w:h" and an area in megapixels (1 MP = 1024 x 1024), multiples of 16."""
    import math

    a, b = (float(x) for x in str(aspect).split(":"))
    area = float(mp) * 1024 * 1024
    w = math.sqrt(area * a / b)
    return max(16, int(round(w / 16)) * 16), max(16, int(round(w * b / a / 16)) * 16)


def size_aspect(width: int, height: int) -> tuple[str, float]:
    """The nearest listed aspect ratio and megapixel choice for a width x height (a config's sample size)."""
    import math

    ratio = width / max(1, height)
    aspect = min(SAMPLE_ASPECTS, key=lambda k: abs(math.log(ratio * float(k.split(":")[1]) / float(k.split(":")[0]))))
    mp = min(MEGAPIXEL_CHOICES, key=lambda m: abs(m - width * height / 2 ** 20))
    return aspect, mp


def download_files(cfg: TrainConfig) -> str:
    """Download (or find in the cache) the single files and turbo LoRA a config leaves to Comfy-Org. Runs in a worker."""
    from rulztrainer.comfy_files import planned_downloads, resolve_downloads

    try:
        plan = planned_downloads(cfg)
        if not plan:
            return "Nothing to download: every file is set."
        resolved = resolve_downloads(cfg, progress=set_progress)
    except Exception as e:  # noqa: BLE001
        return f"Download failed: {e}"
    return "Ready: " + "; ".join(f"{Path(getattr(resolved, field)).name}" for field, _, _ in plan)


SAFETENSORS = [("safetensors", "*.safetensors"), ("all files", "*.*")]

_proc: subprocess.Popen | None = None  # started by this WebUI process
_adopted = None  # psutil.Process of a job started by an earlier WebUI process (reattached from JOB_FILE)
_job_dir: Path | None = None
JOB_FILE = ROOT / ".webui_job.json"
FORM_FILE = ROOT / ".webui_form.json"  # the page's settings, so a reload or a restarted WebUI keeps them
# How long the server keeps a page whose browser went away. Browsers suspend background tabs, which drops the page's
# connection; NiceGUI's default of 3 s then deletes the page and the browser reloads it on return. Only this page's
# wait is raised, not ui.run's: there it also scales the connection's pings, so a dead link would go unnoticed for ages.
PAGE_KEEP_SECONDS = 3600.0
QUEUE = TrainQueue()  # runs waiting for the current one (output/_queue.json); queue_loop() starts them in turn
_start_lock = threading.RLock()  # the queue's thread and a page may start a run at the same moment
AUTOSTOP = machine.AutoStop()  # "stop this instance when the runs are done" (a vast.ai rental), queue_loop()
_watched: dict = {}  # the job this WebUI saw running ({"pid", "job_dir"}): when it ends, out-of-memory recovery looks
_killed_pids: set[int] = set()  # jobs you killed: never a memory failure to recover from


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
    The pid only counts as the same job when its create time matches (pids get reused). Once this process's job has
    ended, a newer job in JOB_FILE (another WebUI on this install started it, from its queue) is picked up: this
    one's queue must not see "nothing runs" and start a second job on the same GPU."""
    global _proc, _adopted, _job_dir
    if _proc is not None or _adopted is not None:
        if _running():
            return True
        try:
            data = json.loads(JOB_FILE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if int(data.get("pid") or 0) == (_proc.pid if _proc is not None else _adopted.pid):
            return False  # still the last job: it has ended
        _proc = _adopted = None
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


def load_form_state() -> dict | None:
    """The settings the page last held (None: nothing saved yet, or a file that cannot be read)."""
    try:
        state = json.loads(FORM_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return state if isinstance(state, dict) else None


def save_form_state(text: str) -> bool:
    """Write the file whole or not at all (a reload can read it at any moment). False: not written this time, which
    also covers Windows briefly locking the file while another page reads it."""
    tmp = FORM_FILE.with_name(f"{FORM_FILE.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, FORM_FILE)
        return True
    except OSError:
        tmp.unlink(missing_ok=True)
        return False


def config_file_name(job_name: str) -> str:
    """A job name as a file name that is safe on every system."""
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", job_name).strip(" .")
    return f"{name or 'config'}.yaml"


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
    with _start_lock:
        return _start_training(cfg)


def _start_training(cfg: TrainConfig, recovery_note: str | None = None) -> str:
    """Start a run. recovery_note: the start is out-of-memory recovery's (oom_recovery), and the new train.log opens
    with this note; any other start begins with no recovery tries."""
    global _proc, _adopted, _job_dir
    reattach_job()
    if _running():
        return "A training job is already running."
    missing = [sp.path or "(none picked)" for sp in cfg.dataset_specs() if not sp.path or not Path(sp.path).is_dir()]
    if missing:
        return f"Dataset folder not found: {', '.join(missing)}"
    foreign = cfg.foreign_sampling_files()
    if foreign:  # TrainConfig.validate refuses it in the job too: say so here instead of in its log
        return (f"Not started: this run trains {cfg.arch_name()}, but these settings name another model's files: "
                f"{'; '.join(foreign)}. Clear them (or set auto).")
    if cfg.model_source == "files":
        for n in ("transformer_file", "text_encoder_file", "vae_file"):
            if getattr(cfg, n) and not Path(getattr(cfg, n)).is_file():
                return f"{n} not found: {getattr(cfg, n)!r}"
    _job_dir = cfg.job_dir()
    _job_dir.mkdir(parents=True, exist_ok=True)
    if recovery_note is None:
        oom_recovery.reset(_job_dir)
    cfg_path = _job_dir / "config.yaml"
    cfg.save(cfg_path)
    for stale in ("status.json", "STOP", "RACE_DROP", "RACE_PICK"):  # (a drop or pick asked of an earlier run)
        p = _job_dir / stale
        if p.exists():
            p.unlink()
    (_job_dir / "train.log").write_text(recovery_note or "", encoding="utf-8")
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    # detached from the WebUI (own process group / session, no console): closing or restarting the WebUI leaves the
    # job running, and the next WebUI reattaches to it through JOB_FILE
    if sys.platform == "win32":
        detach = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW}
    else:
        detach = {"start_new_session": True}
    with open(_job_dir / "console.log", "w", encoding="utf-8") as console:  # the job keeps its own handle
        _proc = subprocess.Popen([sys.executable, "-m", "rulztrainer.train", "--config", str(cfg_path)],
                                 cwd=str(ROOT), env=env, stdout=console, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, **detach)
    _adopted = None
    _record_job(_proc.pid, _job_dir)
    return f"Started job '{cfg.job_name}' (pid {_proc.pid}). Config saved to {cfg_path}"


def request_race_drop(job_dir: Path, name: str) -> None:
    """Ask the running trainer to drop a race configuration (it reads RACE_DROP after every step)."""
    with open(Path(job_dir) / "RACE_DROP", "a", encoding="utf-8") as f:
        f.write(name.strip() + "\n")


def request_race_pick(job_dir: Path, name: str) -> None:
    """Tell a relay race waiting for your pick which racer everyone continues from (it reads RACE_PICK every few
    seconds while it waits)."""
    (Path(job_dir) / "RACE_PICK").write_text(name.strip() + "\n", encoding="utf-8")


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
    # a kill usually means "stop everything": the queued runs wait for Resume (paused first, so the queue's next tick
    # cannot start one between the kill and the pause)
    paused = bool(QUEUE.snapshot()["waiting"])
    if paused:
        QUEUE.set_paused(True)
    _killed_pids.add(_job_pid())
    if _proc is not None:
        _proc.kill()
    else:
        _adopted.kill()
    note = ""
    if AUTOSTOP.armed:  # a kill means "let me look": the rented instance must not stop under that
        AUTOSTOP.disarm("switched off by Kill")
        note = " Stopping the instance when the runs are done is switched off."
    if paused:
        return "Process killed (no final save). The queue is paused: Resume it on the Monitor to go on." + note
    return "Process killed (no final save)." + note


def validated_config(cfg: TrainConfig) -> TrainConfig:
    """The checks a run gets before it starts or is queued (a race's step budget, settings and probes); raises
    ValueError."""
    if cfg.race:
        from rulztrainer import race

        race.validate(cfg)
    return cfg


def _start_queued(config: dict) -> tuple[bool, str]:
    try:
        msg = start_training(validated_config(TrainConfig.from_dict(config)))
    except (TypeError, ValueError) as e:
        return False, f"A setting is invalid: {e}"
    except Exception as e:  # noqa: BLE001 - a full disk, a folder it cannot write: recorded, the queue pauses
        return False, f"Could not start: {type(e).__name__}: {e}"
    return msg.startswith("Started"), msg


def queue_run(cfg: TrainConfig) -> tuple[bool, str]:
    """Add a run to the queue (it starts once nothing runs). Refuses a job name the running job or a waiting run
    already uses: the later run would write over the earlier one's folder."""
    reattach_job()
    taken = {e["job"] for e in QUEUE.snapshot()["waiting"]}
    if _running() and _job_dir is not None:
        taken.add(_job_dir.name)
    if cfg.job_name in taken:
        return False, (f"Job name '{cfg.job_name}' is already running or queued; give this run another job name so "
                       "it doesn't overwrite that one's folder.")
    QUEUE.add(cfg.to_dict(), label=cfg.arch_name())
    n = len(QUEUE.snapshot()["waiting"])
    if not _running():
        return True, f"Queued '{cfg.job_name}': nothing is running, so it starts in a moment."
    return True, f"Queued '{cfg.job_name}' (#{n}): it starts when the runs before it are done."


def _job_pid() -> int | None:
    if _proc is not None:
        return _proc.pid
    return _adopted.pid if _adopted is not None else None


def _oom_block_count(cfg: TrainConfig) -> int:
    try:
        from rulztrainer.archs import guess_arch_name

        return int(N_BLOCKS.get(guess_arch_name(cfg), 0))
    except Exception:  # noqa: BLE001
        return int(N_BLOCKS.get(cfg.arch_name(), 0))


def recover_after_oom(job_dir: Path, returncode: int | None = None, killed: bool = False) -> str | None:
    """A run this WebUI watched has ended: when memory ran out, start it again as oom_recovery decides (under the
    start lock, before the queue's next run). Returns what it did for the log, or None when it was no memory failure."""
    job_dir = Path(job_dir)
    try:
        status = json.loads((job_dir / "status.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        status = {}
    if status.get("done") and not status.get("error"):
        return None  # finished or stopped
    kind = oom_recovery.classify(status.get("error"), oom_recovery.console_tail(job_dir), returncode, killed)
    if kind is None:
        return None
    try:
        cfg = TrainConfig.load(job_dir / "config.yaml")
    except Exception as e:  # noqa: BLE001
        return f"{job_dir.name}: out of memory, but its config could not be read again ({e}); not retried"
    tries = oom_recovery.read_tries(job_dir)
    d = oom_recovery.plan(cfg.to_dict(), status, kind, tries, _oom_block_count(cfg))
    what = "GPU memory" if kind == "gpu" else "system memory"
    step = status.get("step")
    if d["do"] in ("continue", "restart"):
        action = ("carried on from " + Path(d["save"]).name if d["do"] == "continue" else "started again") + \
            (f", {d['change']}" if d["change"] else ", same settings")
        try:
            new_cfg = validated_config(TrainConfig.from_dict(d["config"]))
        except (TypeError, ValueError) as e:
            d = {**d, "do": "stop", "why": f"the lighter settings are not valid ({e})"}
        else:
            tries = oom_recovery.add_try(job_dir, kind, step, action)
            kept = oom_recovery.keep_logs(job_dir, len(tries))
            note = (f"[oom recovery] try {len(tries)} of {cfg.oom_retries}: the last try ran out of {what} at step "
                    f"{step}; {d['why']}: {action}. Its logs: {', '.join(kept) or 'none'}.\n")
            msg = _start_training(new_cfg, recovery_note=note)
            if msg.startswith("Started"):
                return f"{job_dir.name}: out of {what} at step {step}; {d['why']}: {action}"
            d = {**d, "do": "stop", "why": f"it could not start again ({msg})"}
    text = (f"\n[oom recovery] The run ran out of {what} at step {step} and is not retried: {d['why']}.\n"
            f"{oom_recovery.advice(kind, d.get('save'))}\n{oom_recovery.summary(tries)}\n")
    try:
        with open(job_dir / "train.log", "a", encoding="utf-8") as f:
            f.write(text)
    except OSError:
        pass
    return f"{job_dir.name}: out of {what} at step {step}, not retried: {d['why']}"


def queue_tick() -> None:
    """One step of the queue: start the next waiting run when the job has ended (under the start lock, so a run the
    page starts in between is seen as running, not failed as "already running" with the queue paused). A watched run
    that ran out of memory is started again first (recover_after_oom); it then holds the queue like any run."""
    note = None
    with _start_lock:
        reattach_job()
        running = _running()
        pid = _job_pid()
        if running and pid is not None and _job_dir is not None:
            _watched.update(pid=pid, job_dir=_job_dir)
        elif _watched.get("pid") is not None:
            seen, _watched["pid"] = _watched["pid"], None
            code = _proc.poll() if _proc is not None and _proc.pid == seen else None
            try:
                note = recover_after_oom(_watched["job_dir"], code, seen in _killed_pids)
            except Exception as e:  # noqa: BLE001  (recovery must never stop the queue)
                note = f"recovery failed: {type(e).__name__}: {e}"
            running = _running()
        done = QUEUE.advance(running, _start_queued)
    if note:
        print(f"[oom recovery] {note}", flush=True)
    if done is not None:
        print(f"[queue] {done.get('job')}: {done.get('message')}", flush=True)


def queue_loop() -> None:
    reattach_job()
    if QUEUE.on_startup(_running()):
        print("[queue] runs are waiting and nothing runs: the queue is paused until you Resume it (Monitor)",
              flush=True)
    while True:
        try:
            queue_tick()
        except Exception as e:  # noqa: BLE001  (a bad tick must not end the queue)
            print(f"[queue] {e}", flush=True)
        try:
            stopped = AUTOSTOP.tick(_running(), len(QUEUE.snapshot()["waiting"]))
            if stopped is not None:
                print(f"[autostop] {stopped[1]}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[autostop] {e}", flush=True)
        time.sleep(3.0)


def read_status() -> dict | None:
    if _job_dir is None:
        return None
    p = _job_dir / "status.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):  # locked or torn mid-write (JSON / Unicode decode errors are ValueErrors)
        return None


def tail_lines(path: Path, n: int, chunk: int = 65536) -> list[str]:
    """The last n lines of a text file, read from its end: a long run's train.log is not read whole on every poll."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            pos = f.tell()
            data = b""
            while pos > 0 and data.count(b"\n") <= n:
                size = min(chunk, pos)
                pos -= size
                f.seek(pos)
                data = f.read(size) + data
    except OSError:
        return []
    lines = data.decode("utf-8", errors="replace").splitlines()
    if pos > 0:  # the first line read may start mid-line
        lines = lines[1:]
    return lines[-n:] if n > 0 else []


def read_log_tail(n: int = 80) -> str:
    if _job_dir is None:
        return ""
    text = ""
    lp = _job_dir / "train.log"
    if lp.exists():
        text = "\n".join(tail_lines(lp, n))
    st = read_status() or {}
    if not _running() and (st.get("error") or not st.get("done")):
        cp = _job_dir / "console.log"
        if cp.exists():
            tail = tail_lines(cp, 30)
            text += "\n--- console.log ---\n" + "\n".join(tail)
    return text


def download_model(model_path: str) -> str:
    from rulztrainer.moss.loading import resolve_model_dir

    if TrainConfig(model_path=model_path).arch_name() == "ideogram4":  # without its unconditional model (9.3 GB)
        from rulztrainer.ideogram4.loading import resolve_model_dir
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


def eta_text(seconds: float | None) -> str:
    """ "3 h 05 min", "42 min" or "under a minute" for a time left in seconds; "-" for none."""
    if not seconds or seconds <= 0:
        return "-"
    minutes = int(round(seconds / 60))
    if minutes < 1:
        return "under a minute"
    if minutes < 60:
        return f"{minutes} min"
    return f"{minutes // 60} h {minutes % 60:02d} min"


VAST_INSTANCE = machine.vast_instance()  # None unless this is a vast.ai instance


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
            + (f"Typical best match among them: {self_sim * 100:.1f}% (the normalized 100%; faces cut off by the "
               f"frame left out)." if self_sim is not None else "Add 2+ references to see the ceiling."))
    return text, crops


def body_preview(paths: list[str], backend: str = "mediapipe", sam3_model: str = "facebook/sam3",
                 device: str = "cuda:0", body_score: str = "appearance", nlf_model: str = "") -> str:
    ok, msg = bodysim.available()
    if not ok:
        return msg
    if body_score in ("skeleton3d", "smplx"):
        from rulztrainer import bodyshape

        if body_score == "smplx":
            if bodyshape.find_nlf(nlf_model, "models") is None:
                return bodyshape.missing_nlf_message("models")
            scorer = bodyshape.SMPLXScorer(nlf_model, "models", device, progress=set_progress)
        else:
            scorer = bodyshape.Skeleton3DScorer("models", progress=set_progress)
        set_progress("Measuring bodies...", None)
        paths = [str(p) for p in facesim.expand_refs(paths)]
        if not paths:
            return "No image files found in the listed paths (folders are expanded to the images inside)."
        if hasattr(scorer, "hold"):
            with scorer.hold():
                scorer.set_references(paths)
        else:
            scorer.set_references(paths)
        need = "the body down to the knees" if body_score == "smplx" else "a usable pose"
        text = f"{scorer.ref_count} of {len(paths)} body references show {need} and are measured."
        if scorer.profile:
            text += " Profile (median +/- the photos' own spread): " + ", ".join(
                f"{k} {v}" for k, v in scorer.profile.describe().items())
        if scorer.ref_self_similarity is not None:
            text += f". The references' own level {scorer.ref_self_similarity * 100:.1f}% (the normalized 100)."
        else:
            text += ". Add 3+ measurable photos to measure the references' own level."
        return text
    if body_score == "anchored":
        from rulztrainer import bodyanchor

        scorer = bodyanchor.AnchoredScorer("models", device, progress=set_progress)
        set_progress("Finding people...", None)
        paths = [str(p) for p in facesim.expand_refs(paths)]
        if not paths:
            return "No image files found in the listed paths (folders are expanded to the images inside)."
        info = scorer.set_references(paths)
        shown = sum(1 for i in info if i.get("person"))
        return (f"{shown} of {len(paths)} body references show shoulders and hips; {scorer.ref_count} show the body "
                "from the neck to mid-thigh and are compared (anchored score, SigLIP 2 + DINOv2). Its zero is set "
                "when training starts, by the step-0 probe's body renders (the base model without the LoRA).")
    if body_score == "appearance":
        scorer = bodysim.AppearanceScorer("models", device, progress=set_progress)
        set_progress("Finding people...", None)
        paths = [str(p) for p in facesim.expand_refs(paths)]
        if not paths:
            return "No image files found in the listed paths (folders are expanded to the images inside)."
        info = scorer.set_references(paths)
        shown = sum(1 for i in info if i.get("person"))
        text = (f"{shown} of {len(paths)} body references show shoulders and hips; {scorer.ref_count} show the body "
                "from the neck to mid-thigh and are compared (appearance score, DINOv2).")
        if scorer.ref_self_similarity is not None:
            text += f" Typical best match among them {scorer.ref_self_similarity * 100:.1f}% (the normalized 100%)."
        else:
            text += " Add 2+ three-quarter or full-length photos to measure the ceiling."
        return text
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
    """"face 47%, body 17%, combined 38%": the normalized scores (the scores themselves for samples recorded without a
    scale, normalization off)."""
    def one(label, norm_key, raw_key):
        v = sc.get(norm_key)
        if v is None:
            v = sc.get(raw_key)
        return f"{label} {v:.0f}%" if v is not None else None

    parts = [one("face", "face_norm", "percent") or "no face"]
    for label, nk, rk in (("body", "body_norm", "body_percent"), ("combined", "combined_norm", "combined_percent")):
        p = one(label, nk, rk)
        if p:
            parts.append(p)
    return "  |  " + ", ".join(parts)


_RACE_IMAGE = history._RACE_IMAGE  # "<configuration>_step000120_<rest>.png"


def race_names(job_dir: Path | None) -> list[str]:
    """The configurations of a job's race (race.json), [] without one."""
    try:
        data = json.loads((Path(job_dir) / "race.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return []
    names = data.get("names") if isinstance(data, dict) else None
    return [str(n) for n in names] if isinstance(names, list) else []


def generation_label(step: int, rest: str) -> str:
    """"step 240, sample 2" / "step 240, face probe 1" for the image-name tail after the step ("1" / "face_0")."""
    kind, _, seed = rest.rpartition("_")
    if kind and seed.isdigit():
        return f"step {step}, {kind} probe {int(seed) + 1}"
    return f"step {step}, sample {int(rest) + 1}" if rest.isdigit() else f"step {step}, {rest}"


def group_by_generation(paths, names: list[str]) -> list[tuple[int, str, list[tuple[str, str]]]]:
    """Race images grouped by generation: [(step, rest, [(contender, path), ...])], the newest step first, each
    group's contenders in race order. The same generation is the same step, prompt (or probe kind and seed) and seed,
    one image per configuration; an image of the winner after the race (no configuration prefix) is its own group."""
    groups: dict[tuple[int, str], dict[str, str]] = {}
    for p in paths:
        m = _RACE_IMAGE.match(Path(p).name)
        name = m["name"] if m and m["name"] in names else ""
        if m and name:
            key = (int(m["step"]), m["rest"])
        else:
            m2 = re.match(r"^step(\d{6})_(.+)\.png$", Path(p).name)
            if not m2:
                continue
            key = (int(m2.group(1)), m2.group(2))
        groups.setdefault(key, {})[name] = str(p)
    order = {n: i for i, n in enumerate(names)}
    out = []
    for (step, rest), by in sorted(groups.items(), key=lambda kv: (-kv[0][0], kv[0][1])):
        out.append((step, rest, sorted(by.items(), key=lambda kv: order.get(kv[0], len(order)))))
    return out


probe_racer = history.probe_racer


def _gold_name(rec: dict | None) -> str:
    """The display name of a probe record's gold face recognizer (records before 3.14.0 carry none: FaceNet)."""
    return {"adaface": "AdaFace"}.get(str((rec or {}).get("gold_name") or "").lower(), "FaceNet")


def _probe_points(probes: list[dict], key: str, fallback: str | None = None) -> list[list]:
    """[step, score] of each probe record for a key (normalized where the record has it, raw for older runs)."""
    out = []
    for pr in probes:
        v = pr.get(key)
        if v is None and fallback:
            v = pr.get(fallback)
        if v is not None:
            out.append([pr["step"], round(v, 1)])
    return out


PROBE_DEFAULT_SERIES = ["face", "body", "combined", "face (probe)", "body (probe)"]


def probe_chart_series(probes: list[dict], names: list[str],
                       shown=("combined", "face", "body")) -> tuple[list[dict], list[str]]:
    """(series, legend) of the Monitor's likeness chart. One run: the smoothed face, body and combined lines plus each
    probe's face and body dots. A race: the smoothed lines of every configuration, named after it ("lora16 face"),
    one colour per configuration and one line style per score (combined thick, face solid, body dashed), only the
    scores in `shown`; the winner's probes after the race are "winner"."""
    if not names:
        series = [_line("face", "#FF3B30", data=_probe_points(probes, "face_norm_ema", "face_ema")),
                  _line("body", "#F4F4F5", data=_probe_points(probes, "body_norm_ema", "body_ema")),
                  _line("combined", "#F59E0B", lineStyle={"width": 3}, data=_probe_points(probes, "combined_ema")),
                  {"name": "face (probe)", "type": "scatter", "symbolSize": 5,
                   "data": _probe_points(probes, "face_norm", "face"), "itemStyle": {"color": "#FB7185"}},
                  {"name": "body (probe)", "type": "scatter", "symbolSize": 5,
                   "data": _probe_points(probes, "body_norm", "body"), "itemStyle": {"color": "#A1A1AA"}}]
        legend = list(PROBE_DEFAULT_SERIES)
        gold = _probe_points(probes, "gold_norm_ema", "gold_ema")
        if gold:  # the held-out gold face (face_gold): drawn next to ArcFace's, so a gap between them shows
            name = f"face ({_gold_name(probes[-1])})"
            series.append(_line(name, "#22D3EE", lineStyle={"type": "dashed"}, data=gold))
            legend.append(name)
        return series, legend
    by: dict[str, list[dict]] = {}
    for pr in probes:
        by.setdefault(probe_racer(pr, names), []).append(pr)
    series, legend = [], []
    for i, name in enumerate([n for n in names if n in by] + ([""] if "" in by else [])):
        color = CHART_COLORS[i % len(CHART_COLORS)]
        label = name or "winner"
        for score, key, fallback, style in (("combined", "combined_ema", None, {"width": 3}),
                                            ("face", "face_norm_ema", "face_ema", {"width": 1.5}),
                                            ("body", "body_norm_ema", "body_ema", {"width": 1.5, "type": "dashed"})):
            if score not in shown:
                continue
            series.append(_line(f"{label} {score}", color, lineStyle=dict(style),
                                data=_probe_points(by[name], key, fallback)))
            legend.append(f"{label} {score}")
    return series, legend


SWEEP_COLORS = {"face": "#FF3B30", "hair": "#F59E0B", "skin": "#FB7185", "clothes": "#60A5FA",
                "background": "#34D399", "subject": "#F4F4F5"}


def sweep_history(sweep: dict | None, job_dir=None) -> list[dict]:
    """A run's sweeps, oldest first, for the Monitor's chart: the status's own list, else (a status written without
    one) the job folder's region_sweep.jsonl."""
    rows = list((sweep or {}).get("history") or [])
    if rows or not sweep or job_dir is None:
        return rows
    try:
        with open(Path(job_dir) / "region_sweep.jsonl", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, dict) and "step" in row:
                    rows.append(row)
    except OSError:
        return []
    return rows[-400:]


def sweep_gap_series(history: list[dict]) -> tuple[list[dict], list[str]]:
    """(series, legend) of the Monitor's sweep gap chart: per region, the training pictures' learned share less the
    held-out pictures', in points, at every sweep both were read. One line per region in its own colour, the subject
    (face, hair and skin together) thicker. Empty without held-out pictures (or with them only)."""
    series, legend = [], []
    for region, color in SWEEP_COLORS.items():
        pts = [[int(r["step"]), round(100 * ((r.get("train") or {})[region] - (r.get("held") or {})[region]), 1)]
               for r in history
               if (r.get("train") or {}).get(region) is not None and (r.get("held") or {}).get(region) is not None]
        if not pts:
            continue
        series.append(_line(region, color, data=pts, showSymbol=len(pts) < 60, symbolSize=4,
                            lineStyle={"width": 3 if region == "subject" else 1.5}))
        legend.append(region)
    return series, legend


def sweep_region_series(history: list[dict], region: str, best=None) -> tuple[list[dict], list[str]]:
    """(series, legend) of one region's sweep chart: the share of its starting error that is gone, in percent, at
    every sweep, on the training pictures (the region's colour) and on the held-out ones (grey, dashed). A set of
    pictures that was not swept has no line. best: (score, step) of the held-out subject's best sweep, marked on the
    subject's held-out line."""
    series, legend = [], []
    for key, name, color, style in (("train", "training", SWEEP_COLORS[region], {}),
                                    ("held", "held-out", "#A1A1AA", {"type": "dashed"})):
        pts = [[int(r["step"]), round(100 * (r.get(key) or {})[region], 1)] for r in history
               if (r.get(key) or {}).get(region) is not None]
        if not pts:
            continue
        line = _line(name, color, data=pts, showSymbol=len(pts) < 60, symbolSize=4, lineStyle={"width": 2, **style})
        if key == "held" and region == "subject" and best:
            line["markPoint"] = {"symbol": "pin", "symbolSize": 34, "itemStyle": {"color": "#F59E0B"},
                                 "label": {"formatter": "best", "color": "#0A0A0A", "fontSize": 9},
                                 "data": [{"coord": [int(best[1]), round(100 * float(best[0]), 1)]}]}
        series.append(line)
        legend.append(name)
    return series, legend


def probe_score_line(rec: dict | None) -> str:
    """"overall 38% | face 47% | body 17%": one probe's normalized scores (the scores themselves with normalization
    off)."""
    if not rec:
        return ""
    face, body, comb = rec.get("face_norm"), rec.get("body_norm"), rec.get("combined")
    gold = rec.get("gold_norm")
    if face is None and body is None and not rec.get("scale"):
        face, body, gold = rec.get("face"), rec.get("body"), rec.get("gold")
    parts = [f"{label} {v:.0f}%" for label, v in (("overall", comb), ("face", face),
                                                   (f"{_gold_name(rec)} face", gold),
                                                   ("body", body)) if v is not None]
    if rec.get("copy") is not None:  # the copy check: its closest training photo (SSCD cosine)
        parts.append(f"copy {rec['copy']:.2f}")
    kept = (rec.get("prompt_check") or {}).get("kept_share")
    if kept is not None:  # the prompt check: the share of renders still in their prompt's frame (the worst prompt)
        parts.append(f"on-prompt {100 * kept:.0f}%")
    return " | ".join(parts)


_PROBE_IMAGE = re.compile(r"_(face|body)_(\d+)\.png$")
_CHECK_IMAGE = re.compile(r"_check-([A-Za-z0-9_.+-]+)_(\d+)\.png$")  # an extra prompt of the prompt check
_SAMPLE_IMAGE = re.compile(r"step(\d{6})_(\d+)\.png$")


def probe_image_title(path) -> str:
    """"Face probe 2" for ..._face_1.png; "Prompt check full_walking 1" for ..._check-full_walking_0.png."""
    m = _PROBE_IMAGE.search(Path(path).name)
    if m:
        return f"{m.group(1).capitalize()} probe {int(m.group(2)) + 1}"
    m = _CHECK_IMAGE.search(Path(path).name)
    return f"Prompt check {m.group(1)} {int(m.group(2)) + 1}" if m else Path(path).name


def probe_steps(probes: list[dict], names: list[str]) -> dict[int, tuple[str, list]]:
    """The Monitor's probe images by step for a StepGallery: {step: (summary, [(group, path, title, scores)])}. One
    run: the probe's scores once as the summary, a title per image. A race: one group per probe image ("Face probe
    1"), every configuration's image side by side, each with its own probe's scores (the winner after the race:
    "winner")."""
    out: dict[int, tuple[str, list]] = {}
    for rec in probes:
        images = rec.get("images") or []
        if not images:
            continue
        line = probe_score_line(rec).replace(" | ", " · ").replace("overall", "combined")
        if names:
            who = probe_racer(rec, names) or "winner"
            _, items = out.setdefault(rec["step"], ("", []))
            items += [(probe_image_title(p), p, who, line) for p in images]
        else:
            out[rec["step"]] = (f"This probe: {line}" if line else "",
                                [("", p, probe_image_title(p), "") for p in images])
    if names:  # each group's configurations in race order
        order = {n: i for i, n in enumerate(names)}
        for step, (summary, items) in out.items():
            groups = list(dict.fromkeys(g for g, *_ in items))
            out[step] = (summary, sorted(items, key=lambda it: (groups.index(it[0]), order.get(it[2], len(order)))))
    return out


def sample_steps(samples: list[str], scores: list[dict], names: list[str]) -> dict[int, tuple[str, list]]:
    """The Monitor's sample images by step for a StepGallery: one run, a title per prompt with its likeness scores and
    the step's average as the summary; a race, one group per prompt with every configuration side by side."""
    by_name = {Path(sc["image"]).name: sc for sc in scores}
    out: dict[int, tuple[str, list]] = {}
    if names:
        for step, rest, members in group_by_generation(samples, names):
            what = generation_label(step, rest).split(", ", 1)[-1]
            _, items = out.setdefault(step, ("", []))
            items += [(what[:1].upper() + what[1:], p, who or "winner", short_scores(by_name.get(Path(p).name)))
                      for who, p in members]
        for step, (_, items) in out.items():  # prompt 1 first
            groups = sorted(dict.fromkeys(g for g, *_ in items), key=lambda g: (len(g), g))
            items.sort(key=lambda it: groups.index(it[0]))
        return out
    for p in samples:
        m = _SAMPLE_IMAGE.search(Path(p).name)
        if not m:
            continue
        step, i = int(m.group(1)), int(m.group(2))
        out.setdefault(step, ("", []))[1].append(("", p, f"Prompt {i + 1}", short_scores(by_name.get(Path(p).name))))
    for step, (_, items) in list(out.items()):
        items.sort(key=lambda it: int(it[2].split()[-1]))
        mine = [by_name[Path(p).name] for _, p, *_ in items if Path(p).name in by_name]
        avg = short_scores({k: sum(v) / len(v) for k in ("face_norm", "body_norm", "combined_norm", "percent",
                                                          "body_percent", "combined_percent")
                            if (v := [sc[k] for sc in mine if sc.get(k) is not None])}) if mine else ""
        out[step] = (f"Average of this step's samples: {avg}" if avg else "", items)
    return out


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


class DatasetPicker:
    """The datasets a run trains on: a row per known dataset with a tick, its own resolutions (none picked: the run's
    training resolutions) and its repeats (how many times each of its images comes up per epoch). .value is the
    config's `datasets` list (the ticked rows) and .selected their paths, so the Form, the wizard and the Dataset tab
    use it like any other widget."""

    def __init__(self, on_change=None):
        self.rows: dict[str, dict] = {}
        self.on_change = on_change
        self.listeners: list = []  # called when rows are added or removed (the wizard's dataset choices)
        self.box = ui.column().classes("w-full gap-1")
        with self.box:
            self.empty = ui.label("No datasets yet: create one on the Dataset tab, or add a folder you already have.") \
                .classes("rz-hint")

    @staticmethod
    def key(path) -> str:
        return os.path.normcase(os.path.abspath(str(path)))

    def paths(self) -> list[str]:
        return [r["path"] for r in self.rows.values()]

    @property
    def value(self) -> list[dict]:
        out = []
        for r in self.rows.values():
            if not r["use"].value:
                continue
            d = {"path": r["path"], "resolutions": sorted(float(x) for x in (r["res"].value or [])),
                 "repeats": max(1, int(r["rep"].value or 1))}
            for key, el in (("trigger", r["trig"]), ("class", r["cls"])):
                if str(el.value or "").strip():
                    d[key] = str(el.value).strip()
            if r["kind"].value and r["kind"].value != "auto":
                d["kind"] = r["kind"].value
            out.append(d)
        return out

    @value.setter
    def value(self, entries) -> None:
        for r in self.rows.values():
            r["use"].value = False
        for d in entries or []:
            d = d if isinstance(d, dict) else {"path": str(d)}
            path = str(d.get("path") or "").strip()
            if not path:
                continue
            r = self.ensure(path)
            r["res"].value = sorted({min(MEGAPIXEL_CHOICES, key=lambda m, x=float(x): abs(m - x))
                                     for x in (d.get("resolutions") or [])})
            r["rep"].value = int(d.get("repeats") or 1)
            r["trig"].value = str(d.get("trigger") or "")
            r["cls"].value = str(d.get("class") or "")
            r["kind"].value = str(d.get("kind") or "auto")
            r["use"].value = True
        self.changed()

    @property
    def selected(self) -> list[str]:
        return [r["path"] for r in self.rows.values() if r["use"].value]

    @selected.setter
    def selected(self, paths) -> None:
        want = {self.key(p) for p in paths or []}
        for p in paths or []:
            self.ensure(p)
        for k, r in self.rows.items():
            if r["use"].value != (k in want):
                r["use"].value = k in want

    def ensure(self, path: str) -> dict:
        """The row of a dataset, added (unticked) when it has none yet."""
        key = self.key(path)
        if key in self.rows:
            return self.rows[key]
        with self.box:
            with ui.row().classes("w-full items-center gap-3 no-wrap") as row:
                use = ui.checkbox(Path(path).name or path).classes("w-56 min-w-0").tooltip(path)
                info = ui.label(path).classes("rz-hint flex-grow min-w-0 truncate")
                res = ui.select(MP_OPTIONS, multiple=True, value=[], label="Own resolutions (none: the run's)") \
                    .props("use-chips dense").classes("w-80 max-w-full") \
                    .tooltip("This dataset's training resolutions. None picked: the Training resolutions below.")
                rep = ui.number("Repeats", value=1, min=1, max=1000, precision=0).classes("w-24") \
                    .tooltip("How many times each image of this dataset comes up per epoch: 2 counts a dataset "
                             "twice as much as one at 1.")
                trig = ui.input("Trigger", placeholder="the run's").props("dense").classes("w-28") \
                    .tooltip("This dataset's own trigger word (empty: the Trigger word of the run). Several "
                             "subjects, each with its own trigger, train together (LoRA Isolation).")
                cls = ui.input("Is a", placeholder="from captions").props("dense").classes("w-28") \
                    .tooltip("LoRA Isolation: what the subject is (man, woman, dog...). Empty: read from the "
                             "captions' \"<trigger>, a man\".")
                kind = ui.select({"auto": "auto", "person": "person", "thing": "animal / thing", "style": "style"},
                                 value="auto", label="Kind").props("dense").classes("w-32") \
                    .tooltip("LoRA Isolation: a person gets the fused trigger and the side-by-side pairs with "
                             "strangers, an animal or thing the fused trigger only, a style keeps its trigger as "
                             "written. auto: a person when the class is one.")
        for el in (res, rep, trig, cls, kind):
            el.bind_enabled_from(use, "value")
        for el in (use, res, rep, trig, cls, kind):
            el.on_value_change(lambda _: self.changed())
        self.rows[key] = {"path": path, "row": row, "use": use, "info": info, "res": res, "rep": rep, "trig": trig,
                          "cls": cls, "kind": kind}
        self.empty.visible = False
        for fn in self.listeners:
            fn()
        return self.rows[key]

    def refresh(self, paths) -> None:
        """Rows for the listed datasets: new ones added; unticked ones no longer listed removed (a ticked one stays
        until it is unticked)."""
        keys = {self.key(p) for p in paths}
        for p in paths:
            self.ensure(p)
        for k, r in list(self.rows.items()):
            if k not in keys and not r["use"].value:
                r["row"].delete()
                del self.rows[k]
        self.empty.visible = not self.rows
        for fn in self.listeners:
            fn()

    def set_counts(self, counts: dict) -> None:
        for r in self.rows.values():
            c = counts.get(r["path"])
            if c is not None:
                r["info"].text = f"{r['path']}: {images_text(c[0])}, {c[1]} captioned"

    def changed(self) -> None:
        if self.on_change is not None:
            self.on_change()


class Form:
    """Holds the NiceGUI widgets and converts to/from TrainConfig."""

    def __init__(self):
        self.w: dict[str, object] = {}
        self.extras: dict[str, object] = {}  # the page's other inputs that outlive a reload but are not config options
        self.quant_copies: dict[str, object] = {}  # the wizard's download format selects, bound to the form's
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
        # the sample steps / CFG likewise: each family samples at its own (its turbo LoRA's, or full quality)
        if not self.loading:
            self.clear_foreign_sampling_files(family)
            self.follow_sample_defaults(self.target_family, family, self.turbo_on(), self.turbo_on())
            self.set_quant_options(family)
        self.target_family = family

    def clear_foreign_sampling_files(self, family: str) -> None:
        """On a model change, empty the sampling model and set the sampling-only LoRA files back to auto when their
        names belong to another family: the previous model's (Krea 2's krea/Krea-2-Turbo preview model, its turbo
        LoRA) would otherwise stay in fields the new model's page hides, and its run sample on Krea 2 Turbo."""
        from rulztrainer.config import sampling_file_family

        def foreign(key):
            value = (self.w[key].value or "").strip() if self.w.get(key) is not None else ""
            named = sampling_file_family(key, value)
            return named is not None and named != family

        cleared = []
        if foreign("sample_model_path"):
            self.w["sample_model_path"].value = ""
            cleared.append("sampling model")
        for key, mode, label in (("turbo_lora", "turbo_mode", "turbo LoRA"),
                                 ("sample_lora", "sample_lora_mode", "sampling LoRA"),
                                 ("training_adapter", "adapter_mode", "training adapter")):
            if self.w.get(mode) is not None and self.w[mode].value == "file" and foreign(key):
                self.w[key].value = ""
                self.w[mode].value = "auto"
                cleared.append(label)
        if cleared:
            ui.notify(f"The previous model's {', '.join(cleared)} cleared (set back to auto) for the new model",
                      type="info")

    def turbo_on(self) -> bool:
        mode = self.w.get("turbo_mode")
        return mode is None or mode.value != "none"

    def follow_sample_defaults(self, old_family: str, new_family: str, old_turbo: bool, new_turbo: bool) -> None:
        """Move the sample steps / CFG to the new family's or mode's defaults (turbo LoRA: ARCH_SAMPLE_DEFAULTS; none:
        ARCH_FULL_SAMPLE), unless the user changed them from the old defaults (their own choice stays)."""
        from rulztrainer.config import ARCH_FULL_SAMPLE

        steps, cfg = self.w.get("sample_steps"), self.w.get("sample_cfg")
        if steps is None or cfg is None:
            return
        try:
            current = (int(steps.value), float(cfg.value))
        except (TypeError, ValueError):
            return
        def defaults(family, turbo):  # a family without a turbo LoRA (FLUX 3 Klein) samples at full quality
            return ARCH_SAMPLE_DEFAULTS.get(family, ARCH_FULL_SAMPLE[family]) if turbo else ARCH_FULL_SAMPLE[family]

        if current == defaults(old_family, old_turbo):
            steps.value, cfg.value = defaults(new_family, new_turbo)

    def set_quant_options(self, family: str) -> None:
        """The quant choices of the single files to download follow the family (Krea 2 also has fp8)."""
        for key, kind in (("model_file_quant", "model"), ("text_encoder_file_quant", "text")):
            sel = self.w.get(key)
            if sel is not None:
                opts = file_quant_options(family, kind)
                copy = self.quant_copies.get(key)
                if copy is not None:  # first: a bound select that lacks the value blanks it for both
                    copy.set_options(opts)
                value = sel.value if sel.value in opts else "int8_convrot" if "int8_convrot" in opts else next(iter(opts))
                if family == "minimax_h3" and kind == "model" and value == "int8_convrot":
                    value = "pruned_int8_convrot"  # H3: the pruned file (21 GB) instead of the full one (34 GB)
                sel.set_options(opts, value=value)

    def to_config(self) -> TrainConfig:
        v = self.value
        preset = v("model_preset")
        model_path = (v("model_path") or "").strip() if preset == CUSTOM else MODEL_PRESETS[preset]
        prompts = [p.strip() for p in (v("sample_prompts") or "").splitlines() if p.strip()]
        s = lambda k: (v(k) or "").strip()  # noqa: E731
        kw = dict(
            preset="identity" if self.w.get("preset_toggle") is not None and self.w["preset_toggle"].value else "",
            job_name=s("job_name") or "my_lora", output_dir=s("output_dir") or "output", seed=int(v("seed")),
            model_source="files" if v("model_source") == FILE_SOURCE else "hf",
            model_path=model_path, transformer_file=s("transformer_file"), text_encoder_file=s("text_encoder_file"),
            vae_file=s("vae_file"), tokenizer_path=s("tokenizer_path"),
            model_file_quant=v("model_file_quant"), text_encoder_file_quant=v("text_encoder_file_quant"),
            device=v("device"), text_encoder_device=v("text_encoder_device"),
            multi_gpu=v("multi_gpu"), devices=(v("devices") or "").strip(), split_plan=(v("split_plan") or "").strip(),
            parallel_jobs=bool(v("parallel_jobs")), dp_balance=v("dp_balance"),
            transformer_quant=v("transformer_quant"), text_encoder_quant=v("text_encoder_quant"),
            gradient_checkpointing=bool(v("gradient_checkpointing")),
            checkpoint_keep_blocks=int(v("checkpoint_keep_blocks")),
            int8_matmul=bool(v("int8_matmul")), int8_backward=v("int8_backward"),
            nvfp4_matmul=bool(v("nvfp4_matmul")), nvfp4_backward=v("nvfp4_backward"),
            attention_backend=v("attention_backend"),
            datasets=self.w["datasets"].value, dataset_dir=(self.w["datasets"].selected or [""])[0],
            resolutions=sorted(float(x) for x in (v("resolutions") or [1.0])),
            resolution_step_up=bool(v("resolution_step_up")), step_up_patience=int(v("step_up_patience")),
            step_up_stop_patience=int(v("step_up_stop_patience")),
            enable_buckets=bool(v("enable_buckets")), flip_augment=bool(v("flip_augment")),
            trigger_word=s("trigger_word"), caption_dropout=float(v("caption_dropout")),
            caption_dropout_keep_trigger=bool(v("caption_dropout_keep_trigger")),
            max_token_length=int(v("max_token_length")), recache=bool(v("recache")),
            lora_rank=int(v("lora_rank")), lora_alpha=float(v("lora_alpha")),
            network_type=v("network_type"), lokr_factor=int(v("lokr_factor") or 8),
            oft_block_size=int(v("oft_block_size") or 32), oft_constraint=float(v("oft_constraint") or 0.0),
            oft_scaled_lr=bool(v("oft_scaled_lr")),
            dora=bool(v("dora")),
            finetune=bool(v("finetune")), finetune_rotate_epochs=float(v("finetune_rotate_epochs") or 1),
            finetune_window_gb=float(v("finetune_window_gb") or 0), finetune_reserve_gb=float(v("finetune_reserve_gb") or 0),
            finetune_extract_ranks=parse_ranks(v("finetune_extract_ranks")),
            finetune_save_delta=bool(v("finetune_save_delta")), finetune_resume=s("finetune_resume"),
            finetune_anchor=float(v("finetune_anchor") or 0.0),
            lora_targets=[t for t in TARGETS + list(EXTRA_TARGETS) if self.w.get(f"target_{t}") is not None
                          and self.w[f"target_{t}"].value],
            train_text_embedder=bool(v("train_text_embedder")), resume_from=s("resume_from"),
            lora_blocks=s("lora_blocks"),
            lora_save_dtype=v("lora_save_dtype"), lora_ema=float(v("lora_ema") or 0),
            save_state=bool(v("save_state")), resume_state=s("resume_state"),
            save_state_every=int(v("save_state_every")), keep_states=int(v("keep_states")),
            oom_retries=3 if v("oom_retries") is None else max(0, int(v("oom_retries"))),
            compile_model=v("compile_model"), compile_cache=s("compile_cache") or "auto",
            sample_seed=int(v("sample_seed")),
            sample_walk_seed=bool(v("sample_walk_seed")), sample_before_training=bool(v("sample_before_training")),
            sample_frames=int(v("sample_frames") or 1), probe_frames=int(v("probe_frames") or 1),
            train_frames=int(v("train_frames") or 1), train_video_mode=v("train_video_mode") or "fps",
            video_resolution=int(v("video_resolution") or 0),
            steps=0 if v("soup_run") or v("noise_run") else int(v("steps")), batch_size=int(v("batch_size")),
            gradient_accumulation=int(v("gradient_accumulation")), learning_rate=float(v("learning_rate")),
            optimizer=v("optimizer"), lr_scheduler=v("lr_scheduler"), warmup_steps=int(v("warmup_steps")),
            max_grad_norm=float(v("max_grad_norm")), timestep_sampling=v("timestep_sampling"),
            automagic_min_lr=float(v("automagic_min_lr")), automagic_max_lr=float(v("automagic_max_lr")),
            automagic_polarity_history=int(v("automagic_polarity_history")), automagic_groups=v("automagic_groups"),
            automagic_clip_threshold=float(v("automagic_clip_threshold")),
            automagic_sensitivity_quantile=float(v("automagic_sensitivity_quantile")), automagic_vote=v("automagic_vote"),
            automagic_max_lr_change=float(v("automagic_max_lr_change")),
            hull_rho=float(v("hull_rho")), hull_regions=v("hull_regions"), hull_gain_body=float(v("hull_gain_body")),
            keel_gauge=bool(v("keel_gauge")), keel_gauge_eps=float(v("keel_gauge_eps")),
            keel_cautious=bool(v("keel_cautious")), keel_pair_sigma=bool(v("keel_pair_sigma")),
            keel_bands=int(v("keel_bands")), keel_band_weights=str(v("keel_band_weights") or "").strip(),
            keel_consensus=float(v("keel_consensus")), keel_consensus_power=float(v("keel_consensus_power")),
            keel_consensus_mode=str(v("keel_consensus_mode") or "auto"),
            keel_lookahead=bool(v("keel_lookahead")), keel_lookahead_pullback=float(v("keel_lookahead_pullback")),
            keel_lookahead_lr=float(v("keel_lookahead_lr")), keel_lookahead_grace=int(v("keel_lookahead_grace")),
            keel_face_gain=float(v("keel_face_gain")), keel_face_gain_max=float(v("keel_face_gain_max")),
            keel_face_gain_beta=float(v("keel_face_gain_beta")), keel_body_weight=float(v("keel_body_weight")),
            timestep_shift=float(v("timestep_shift")), timestep_buckets=int(v("timestep_buckets") or 0),
            sigmoid_scale=float(v("sigmoid_scale")),
            logsnr_mean=float(v("logsnr_mean")), logsnr_std=float(v("logsnr_std")),
            block_swap=int(v("block_swap") or 0), block_swap_compile=bool(v("block_swap_compile")),
            activation_offload=bool(v("activation_offload")),
            clean_share=str(v("clean_share") or "").strip(),
            logit_mean=float(v("logit_mean")), logit_std=float(v("logit_std")), loss_weighting=v("loss_weighting"),
            loss_type=v("loss_type") or "mse",
            run_report=bool(v("run_report")),
            save_every=int(v("save_every")), keep_checkpoints=int(v("keep_checkpoints") or 0),
            sample_every=int(v("sample_every")), sample_prompts=prompts,
            reward_steps=int(v("reward_steps") or 0), reward_lr=float(v("reward_lr") or 2e-5),
            reward_group=int(v("reward_group") or 4),
            reward_prompts=[p.strip() for p in (v("reward_prompts") or "").splitlines() if p.strip()],
            reward_sample_steps=int(v("reward_sample_steps") or 0), reward_train_steps=int(v("reward_train_steps") or 4),
            reward_noise=float(v("reward_noise") or 0.7), reward_kl=float(v("reward_kl") or 0.0),
            reward_size=int(v("reward_size") or 0),
            selftrain=bool(v("selftrain")), selftrain_at=s("selftrain_at") or "0.5",
            selftrain_images=int(v("selftrain_images") or 32),
            selftrain_min_likeness=float(v("selftrain_min_likeness") if v("selftrain_min_likeness") is not None
                                         else 60.0),
            selftrain_share=float(v("selftrain_share") or 0.2), selftrain_weight=float(v("selftrain_weight") or 0.5),
            selftrain_prompts=[p.strip() for p in (v("selftrain_prompts") or "").splitlines() if p.strip()],
            copy_check=v("copy_check") or "auto",
            copy_max=float(v("copy_max") if v("copy_max") is not None else 0.5),
            sample_width=aspect_size(v("sample_aspect"), v("sample_mp"))[0],
            sample_height=aspect_size(v("sample_aspect"), v("sample_mp"))[1],
            sample_steps=int(v("sample_steps")), sample_cfg=float(v("sample_cfg")),
            sample_shift=float(v("sample_shift")), sample_negative=v("sample_negative") or "",
            turbo_lora={"auto": "auto", "none": ""}.get(v("turbo_mode"), s("turbo_lora")),
            turbo_lora_strength=float(v("turbo_lora_strength")),
            sample_model_path=s("sample_model_path"), sample_model_quant=v("sample_model_quant"),
            sample_lora={"auto": "auto", "none": ""}.get(v("sample_lora_mode"), s("sample_lora")),
            sample_lora_strength=float(v("sample_lora_strength")),
            training_adapter={"auto": "auto", "none": ""}.get(v("adapter_mode"), s("training_adapter")),
            training_adapter_strength=float(v("training_adapter_strength") if v("training_adapter_strength") is not None
                                            else 1.0),
            face_refs=[p.strip() for p in (v("face_refs") or "").splitlines() if p.strip()],
            face_provider=v("face_provider"), face_det_size=int(v("face_det_size")),
            body_refs=[p.strip() for p in (v("body_refs") or "").splitlines() if p.strip()],
            face_weight=float(v("face_weight")), body_weight=float(v("body_weight")),
            body_backend=v("body_backend"), sam3_model=(v("sam3_model") or "facebook/sam3").strip(),
            body_score=v("body_score"), nlf_model=s("nlf_model"),
            body_device=v("body_device"),
            probe_every=int(v("probe_every")), probe_at_start=bool(v("probe_at_start")),
            probe_face_prompt=s("probe_face_prompt"), probe_body_prompt=s("probe_body_prompt"),
            probe_subject=s("probe_subject"), probe_face_expect=v("probe_face_expect") or "",
            probe_body_expect=v("probe_body_expect") or "",
            # the presets picked on the page, then the prompts of one's own the config file holds (no field for them)
            probe_checks=list(v("probe_checks") or []) + [
                e for e in getattr(getattr(self, "_base", None), "probe_checks", []) if isinstance(e, dict)],
            probe_check_seeds=int(v("probe_check_seeds") or 4),
            likeness_best_min_kept=float(v("likeness_best_min_kept") or 0),
            probe_seeds=int(v("probe_seeds")), probe_seed=int(v("probe_seed")), probe_size=int(v("probe_size")),
            probe_floor_seeds=int(v("probe_floor_seeds") or 4),
            probe_async=bool(v("probe_async")), probe_device=s("probe_device"),
            probe_steps=int(v("probe_steps")), probe_cfg=float(v("probe_cfg")), probe_ema=float(v("probe_ema")),
            likeness_noise_band=float(v("likeness_noise_band")), likeness_track_best=bool(v("likeness_track_best")),
            likeness_track_best_face=bool(v("likeness_track_best_face")),
            likeness_track_best_body=bool(v("likeness_track_best_body")),
            likeness_best_any_rise=bool(v("likeness_best_any_rise")),
            likeness_framing=bool(v("likeness_framing")),
            face_gold=v("face_gold") or "auto", face_gold_select=bool(v("face_gold_select")),
            likeness_normalize=bool(v("likeness_normalize")),
            face_scale_span=float(v("face_scale_span")), face_scale_curve=float(v("face_scale_curve")),
            body_scale_span=float(v("body_scale_span")), body_scale_curve=float(v("body_scale_curve")),
            likeness_patience=int(v("likeness_patience")), likeness_lr_control=bool(v("likeness_lr_control")),
            likeness_lr_min=float(v("likeness_lr_min")), likeness_lr_max=float(v("likeness_lr_max")),
            likeness_lr_step=float(v("likeness_lr_step")), likeness_curriculum=bool(v("likeness_curriculum")),
            likeness_curriculum_strength=float(v("likeness_curriculum_strength")),
            likeness_curriculum_framing=float(v("likeness_curriculum_framing") or 0),
            loss_watch=bool(v("loss_watch")), per_image_lr=bool(v("per_image_lr")),
            loss_watch_buckets=int(v("loss_watch_buckets")), loss_watch_beta=float(v("loss_watch_beta")),
            loss_watch_high=float(v("loss_watch_high")), loss_watch_low=float(v("loss_watch_low")),
            loss_watch_fence=float(v("loss_watch_fence") or 0),
            loss_watch_improve_tol=float(v("loss_watch_improve_tol")),
            loss_watch_healthy_mult=float(v("loss_watch_healthy_mult")),
            loss_watch_exclude_after=int(v("loss_watch_exclude_after")),
            region_watch=bool(v("region_watch")),
            region_class_weights=str(v("region_class_weights") or "").strip(),
            region_steer=bool(v("region_steer")),
            region_steer_regions=str(v("region_steer_regions") or "background,clothes"),
            region_steer_cap=float(v("region_steer_cap") or 0), region_steer_ahead=float(v("region_steer_ahead") or 0),
            region_steer_boost=float(v("region_steer_boost") or 0),
            region_steer_prior=float(v("region_steer_prior") or 0),
            region_steer_min=float(v("region_steer_min") or 0), region_steer_step=float(v("region_steer_step")),
            region_steer_strength=float(v("region_steer_strength")),
            region_steer_signal=str(v("region_steer_signal") or "steps"),
            region_steer_holdout=float(v("region_steer_holdout") or 0),
            region_steer_subject_ahead=float(v("region_steer_subject_ahead") or 0),
            region_sweep_every=int(v("region_sweep_every") or 0),
            region_sweep_pictures=int(v("region_sweep_pictures") or 0),
            holdout_images=int(v("holdout_images") or 0), holdout_track_best=bool(v("holdout_track_best")),
            holdout_patience=int(v("holdout_patience") or 0),
            training_maps=bool(v("training_maps")), training_maps_images=int(v("training_maps_images") or 3),
            training_maps_every=int(v("training_maps_every") or 0),
            training_maps_trigger_off=bool(v("training_maps_trigger_off")),
            isolation=bool(v("isolation")), isolation_class=s("isolation_class"),
            isolation_kind=v("isolation_kind") or "auto", dop_weight=float(v("dop_weight")),
            dop_every=int(v("dop_every") or 1), isolation_pairs=int(v("isolation_pairs") or 0),
            isolation_pair_weight=float(v("isolation_pair_weight")), isolation_balance=bool(v("isolation_balance")),
            look_warmup=bool(v("look_warmup")), look_warmup_start=float(v("look_warmup_start")),
            look_warmup_epochs=float(v("look_warmup_epochs")),
            region_weights=bool(v("region_weights")), face_loss_weight=float(v("face_loss_weight")),
            body_loss_weight=float(v("body_loss_weight")), background_loss_weight=float(v("background_loss_weight")),
            face_margin=float(v("face_margin")), region_feather_cells=int(v("region_feather_cells")),
            region_prior_weight=float(v("region_prior_weight")),
            attn_localize=bool(v("attn_localize")), attn_localize_weight=float(v("attn_localize_weight") or 0.0),
            attn_localize_layers=s("attn_localize_layers"), id_align=bool(v("id_align")),
            id_align_weight=float(v("id_align_weight") or 0.0), id_align_layer=int(v("id_align_layer")),
            body_align=bool(v("body_align")), body_align_weight=float(v("body_align_weight") or 0.0),
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
            race_drop_lowest=bool(v("race_drop_lowest")), race_patience_drop=bool(v("race_patience_drop")),
            race_patience=int(v("race_patience") or 3), race_patience_rising=bool(v("race_patience_rising")),
            race_parallel=bool(v("race_parallel")), race_follow=bool(v("race_follow")),
            race_pick=bool(v("race_pick")) and bool(v("race_follow")),
            race_pick_minutes=float(v("race_pick_minutes") or 0),
            race_drop_confidence=float(v("race_drop_confidence") or 0) if v("race_paired") else 0.0,
            race_max_close_rounds=int(v("race_max_close_rounds") or 0),
            race_drop_drift=float(v("race_drop_drift") or 0),
            race_drop_drift_probes=int(v("race_drop_drift_probes") or 2),
            race_configs=self.w["race_configs"].value,
            soup_files=([p.strip() for p in (v("soup_files") or "").splitlines() if p.strip()]
                        if v("soup_run") else []),
            soup_rank=int(v("soup_rank") or 0), soup_strengths=s("soup_strengths") or "1.0",
            noise_files=([p.strip().strip('"') for p in (v("noise_files") or "").splitlines() if p.strip()]
                         if v("noise_run") else []),
            noise_seeds=int(v("noise_seeds") or 16),
        )
        # the settings without a field on this page (weight_decay, lora_dropout, the hull_* tuning, cache_dir, ...)
        # stay as the loaded or uploaded config had them, not the defaults: Load then Start (or Save) keeps them
        # (model_arch: auto, so the family follows the model picked here)
        base = getattr(self, "_base", None)
        if base is not None:
            kw = {**{f.name: copy.deepcopy(getattr(base, f.name)) for f in dataclasses.fields(TrainConfig)
                     if f.name not in kw and f.name != "model_arch"}, **kw}
        return TrainConfig(**kw)

    def from_config(self, cfg: TrainConfig) -> None:
        self._base = copy.deepcopy(cfg)  # what to_config keeps of the settings this page has no field for
        preset = next((k for k, v in MODEL_PRESETS.items() if v == cfg.model_path), CUSTOM)
        values = {**cfg.to_dict(), "model_preset": preset, "sample_prompts": "\n".join(cfg.sample_prompts),
                  "reward_prompts": "\n".join(cfg.reward_prompts),
                  "selftrain_prompts": "\n".join(cfg.selftrain_prompts),
                  "soup_files": "\n".join(cfg.soup_files), "soup_run": bool(cfg.soup_files),
                  "noise_files": "\n".join(cfg.noise_files), "noise_run": bool(cfg.noise_files),
                  "race_paired": cfg.race_drop_confidence > 0,
                  "race_drop_confidence": cfg.race_drop_confidence or 0.9,
                  "face_refs": "\n".join(cfg.face_refs), "body_refs": "\n".join(cfg.body_refs),
                  # the page lists the presets; a prompt of one's own stays in the config (to_config keeps it)
                  "probe_checks": [e for e in cfg.probe_checks if isinstance(e, str)
                                   and any(p.key == e for p in promptcheck.PRESETS)],
                  "model_source": FILE_SOURCE if cfg.model_source == "files" else HF_SOURCE}
        values["datasets"] = [{"path": sp.path, "resolutions": list(sp.resolutions), "repeats": sp.repeats,
                               "trigger": sp.trigger, "class": sp.subject_class, "kind": sp.subject_kind}
                              for sp in cfg.dataset_specs() if sp.path]  # a single-folder config: that folder
        values["sample_aspect"], values["sample_mp"] = size_aspect(cfg.sample_width, cfg.sample_height)
        values["turbo_mode"] = ("auto" if cfg.turbo_lora == "auto" else "file" if cfg.turbo_lora else "none")
        values["finetune_extract_ranks"] = ", ".join(str(r) for r in cfg.finetune_extract_ranks)
        values["turbo_lora"] = "" if cfg.turbo_lora == "auto" else cfg.turbo_lora
        values["sample_lora_mode"] = ("auto" if cfg.sample_lora == "auto" else "file" if cfg.sample_lora else "none")
        values["sample_lora"] = "" if cfg.sample_lora == "auto" else cfg.sample_lora
        values["adapter_mode"] = ("auto" if cfg.training_adapter == "auto" else "file" if cfg.training_adapter
                                  else "none")
        values["training_adapter"] = "" if cfg.training_adapter == "auto" else cfg.training_adapter
        targets = cfg.resolved_lora_targets()  # an empty list means the family's defaults
        for t in TARGETS + list(EXTRA_TARGETS):
            values[f"target_{t}"] = t in targets
        if not cfg.resolutions:  # legacy single `resolution`: snap to the nearest megapixel choice
            values["resolutions"] = [min(MEGAPIXEL_CHOICES, key=lambda m: abs(m - side_to_mp(cfg.resolution)))]
        self.loading = True
        try:
            self.set_quant_options(cfg.arch_name())  # the loaded family's download formats, before its values
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


    def snapshot(self, previous: dict | None = None) -> dict:
        """Every input's raw value, for restore(). Raw and not through to_config(), which needs the form valid and
        drops what a run does not use (a half-typed number, the soup files while the soup is off). A value that cannot
        be read right now (a race row being typed) keeps its entry in `previous`."""
        out = dict(previous or {})
        for key, widget in {**self.w, **self.extras}.items():
            try:
                out[key] = widget.value
            except (TypeError, ValueError):
                pass
        return out

    def restore(self, values: dict) -> None:
        """Put a snapshot back without the side effects of a user's change, leaving alone whatever this version cannot
        take: an option it no longer has, a choice no longer offered (a GPU that is gone)."""
        widgets = {**self.w, **self.extras}
        model = {k: values[k] if k in values and fits_options(self.w[k], values[k]) else self.w[k].value
                 for k in ("model_preset", "model_path")}
        family = preset_arch(model["model_preset"], model["model_path"])
        self.loading = self.preset_sync = True
        try:
            self.set_quant_options(family)  # the saved family's download formats, before its values
            for key, widget in widgets.items():
                if key in values and fits_options(widget, values[key]):
                    try:
                        widget.value = values[key]
                    except (TypeError, ValueError, AttributeError):
                        pass
        finally:
            self.loading = self.preset_sync = False
        self.target_family = family


def fits_options(widget, value) -> bool:
    """A value a choice widget can hold (setting any other raises); other widgets take anything."""
    options = getattr(widget, "options", None)
    if options is None or value is None:
        return True
    try:
        return all(v in options for v in (value if isinstance(value, list) else [value]))
    except TypeError:  # an unhashable value against a dict of options
        return False


# ---------------------------------------------------------------- look and feel

ASSETS = ROOT / "assets"
RED = "#E10600"
CHART_COLORS = ["#FF3B30", "#F4F4F5", "#F59E0B", "#A1A1AA", "#FB7185", "#60A5FA", "#34D399", "#C084FC"]

THEME_CSS = """
:root {
  --rz-red: #E10600; --rz-red-bright: #FF3B30; --rz-red-soft: rgba(225, 6, 0, 0.14);
  --rz-bg: #0A0A0A; --rz-surface: #131313; --rz-surface-2: #1A1A1A;
  --rz-border: #262626; --rz-border-strong: #3A3A3A;
  --rz-text: #EDEDED; --rz-muted: #B4B4BD; --rz-ok: #22C55E;
  --rz-heading: #FFFFFF; --rz-text-2: #D4D4D8; --rz-text-3: #E4E4E7; --rz-tagline: #71717A;
  --rz-header-bg: #000000; --rz-footer-bg: rgba(0, 0, 0, .94); --rz-pill-bg: #0D0D0D; --rz-hover-border: #52525B;
  --rz-scroll: #2A2A2A; --rz-tooltip-bg: #1F1F1F; --rz-menu-bg: #161616; --rz-deep: #050505; --rz-thumb-bg: #0E0E0E;
  --rz-code-bg: #1F1F1F; --rz-code: #FCA5A5; --rz-dot-idle: #2A2A2A; --rz-badge-muted: #3F3F46; --rz-label: #B5B5BD;
  --rz-warn: #FBBF24; --rz-rec: #FCA5A5; --rz-ok-text: #86EFAC; --rz-err-text: #FFB4AE; --rz-warn-text: #FCD34D;
  --rz-track: #212121; --rz-img-bg: #000000; --rz-choice-bg: rgba(225, 6, 0, .16);
}
html[data-theme="light"] {
  --rz-bg: #FFFFFF; --rz-surface: #F5F5F7; --rz-surface-2: #EBEBEF;
  --rz-border: #D9D9DE; --rz-border-strong: #C4C4CC;
  --rz-text: #111114; --rz-muted: #55565E; --rz-ok: #16A34A;
  --rz-heading: #111114; --rz-text-2: #2E2E33; --rz-text-3: #1F1F23; --rz-tagline: #6B6B73;
  --rz-header-bg: #FFFFFF; --rz-footer-bg: rgba(255, 255, 255, .94); --rz-pill-bg: #F5F5F7; --rz-hover-border: #A1A1AA;
  --rz-scroll: #C9C9CF; --rz-tooltip-bg: #FFFFFF; --rz-menu-bg: #FFFFFF; --rz-deep: #F5F5F7; --rz-thumb-bg: #E4E4E8;
  --rz-code-bg: #EBEBEF; --rz-code: #B4000A; --rz-dot-idle: #A1A1AA; --rz-badge-muted: #8A8A93; --rz-label: #4A4A52;
  --rz-warn: #B45309; --rz-rec: #C8000B; --rz-ok-text: #15803D; --rz-err-text: #B4000A; --rz-warn-text: #92400E;
  --rz-track: #E4E4E8; --rz-img-bg: #EBEBEF; --rz-choice-bg: rgba(225, 6, 0, .08);
  color-scheme: light;
}
body, .q-page-container, .q-layout { background: var(--rz-bg); }
body {
  color: var(--rz-text); font-size: 15px; line-height: 1.5; -webkit-font-smoothing: antialiased;
  font-family: "Inter", "Segoe UI", system-ui, -apple-system, "Helvetica Neue", Roboto, Arial, sans-serif;
}
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-thumb { background: var(--rz-scroll); border-radius: 10px; }
::-webkit-scrollbar-track { background: transparent; }

/* header: logo and version, the tabs in their three groups, the machine chip; under it the run strip */
.nicegui-header.rz-header {
  background: var(--rz-header-bg); border-bottom: 1px solid var(--rz-border); padding: 0; gap: 0;
  flex-direction: column; align-items: stretch;
}
.rz-header-row { max-width: 1440px; margin: 0 auto; padding: 4px 20px 0; gap: 0 18px; flex-wrap: wrap; }
.rz-brand { padding: 6px 0; flex-shrink: 0; }
.rz-logo { height: 26px; width: auto; display: block; }
.rz-wordmark { letter-spacing: .34em; font-weight: 600; font-size: 12px; color: var(--rz-muted); padding-top: 5px; }
.rz-version {
  font-size: 11px; font-weight: 600; color: var(--rz-muted); border: 1px solid var(--rz-border-strong);
  border-radius: 999px; padding: 0 8px; margin-top: 4px; font-variant-numeric: tabular-nums;
}
.rz-tabs { flex: 1 1 0; min-width: 0; }
.rz-tabs .q-tab { color: var(--rz-muted); min-height: 46px; padding: 0 10px; }
.rz-tabs .q-tab__label { font-weight: 500; font-size: 15px; }
.rz-tabs .q-tab__icon { font-size: 19px; }
.rz-tabs .q-tab:hover { color: var(--rz-text); }
.rz-tabs .q-tab--active { color: var(--rz-heading); }
.rz-tabs .q-tab--active .q-tab__label { font-weight: 600; }
.rz-tabs .q-tab--active .q-tab__icon { color: var(--rz-red-bright); }
.rz-tabs .q-tab__indicator { background: var(--rz-red); height: 3px; border-radius: 3px 3px 0 0; }
.rz-tab-sep { width: 1px; height: 20px; background: var(--rz-border-strong); margin: 0 6px; align-self: center; flex-shrink: 0; }
.rz-header-tools { margin-left: auto; flex-shrink: 0; }
.rz-chip {
  border: 1px solid var(--rz-border-strong); border-radius: 999px; padding: 3px 12px; font-size: 13px;
  background: var(--rz-pill-bg); cursor: pointer; color: var(--rz-text-2); white-space: nowrap;
}
.rz-chip:hover { border-color: var(--rz-hover-border); }
.rz-chip .q-icon { font-size: 16px; color: var(--rz-muted); }
.rz-chip-text { font-variant-numeric: tabular-nums; }
.rz-chip.warning { border-color: rgba(245, 158, 11, .6); color: var(--rz-warn-text); }
.rz-chip.critical { border-color: rgba(255, 59, 48, .7); color: var(--rz-err-text); }
.rz-chip.warning .q-icon, .rz-chip.critical .q-icon { color: inherit; }
.rz-run { border-top: 1px solid var(--rz-border); background: var(--rz-surface); cursor: pointer; }
.rz-run:hover { background: var(--rz-surface-2); }
.rz-run-row { max-width: 1440px; margin: 0 auto; padding: 6px 20px; gap: 12px; min-height: 36px; color: var(--rz-text); }
.rz-run-state { font-weight: 600; font-size: 14px; color: var(--rz-heading); white-space: nowrap; }
.rz-run-job { font-size: 14px; color: var(--rz-text-2); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 30ch; }
.rz-run-step { font-size: 13px; color: var(--rz-muted); white-space: nowrap; font-variant-numeric: tabular-nums; overflow: hidden; text-overflow: ellipsis; }
.rz-run-bar { flex: 1 1 80px; min-width: 48px; max-width: 420px; }
.rz-run-note { margin-left: auto; font-size: 13px; color: var(--rz-text-2); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.rz-run-note.rz-warn { color: var(--rz-warn-text); font-weight: 600; }
.rz-run-go { color: var(--rz-muted); font-size: 18px; }
.rz-dot { width: 9px; height: 9px; border-radius: 50%; background: #52525B; display: inline-block; flex-shrink: 0; }
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
.rz-section > .q-expansion-item__container > .q-item .q-item__label { font-size: 16px; font-weight: 600; color: var(--rz-heading); }
.rz-section > .q-expansion-item__container > .q-item .q-item__label--caption {
  font-size: 14px; color: var(--rz-muted); font-weight: 400; margin-top: 2px;
}
.rz-section .nicegui-expansion-content { padding: 2px 18px 20px; gap: 14px; }
.rz-advanced { border-top: 1px dashed var(--rz-border); }
.rz-advanced > .q-expansion-item__container > .q-item { padding: 10px 0 4px; min-height: 0; color: var(--rz-text-2); font-weight: 500; }
.rz-advanced > .q-expansion-item__container > .q-item .q-item__section--avatar { min-width: 0; padding-right: 8px; }
.rz-advanced > .q-expansion-item__container > .q-item .q-icon { color: var(--rz-muted); font-size: 18px; }
.rz-section .rz-advanced .nicegui-expansion-content { padding: 8px 0 6px; gap: 14px; }
.rz-title { font-size: 16px; font-weight: 600; color: var(--rz-heading); }
.rz-step-title { font-size: 20px; font-weight: 700; color: var(--rz-heading); }
.rz-subhead { font-size: 12.5px; font-weight: 600; letter-spacing: .07em; text-transform: uppercase; color: var(--rz-muted); margin-top: 4px; }
.rz-hint { font-size: 14px; color: var(--rz-muted); line-height: 1.55; max-width: 980px; }
.rz-warn { color: var(--rz-warn); }
.rz-fields { gap: 14px 16px; align-items: center; flex-wrap: wrap; }

/* form fields and buttons */
.q-field--outlined .q-field__control { border-radius: 9px; }
.q-field--dense .q-field__control, .q-field--dense .q-field__marginal { height: 46px; }
.q-field--dense.q-field--auto-height .q-field__control, .q-field--dense.q-textarea .q-field__control { height: auto; min-height: 46px; }
.q-field--dense .q-field__label { font-size: 15px; top: 13px; }
.q-field--dense.q-field--float .q-field__label { transform: translateY(-46%) scale(.8); }
.q-field--dense .q-field__native, .q-field--dense .q-field__input, .q-field--dense .q-field__prefix, .q-field--dense .q-field__suffix { font-size: 15px; }
.q-field--dense.q-field--labeled .q-field__native, .q-field--dense.q-field--labeled .q-field__input { padding-top: 17px; padding-bottom: 3px; }
.q-field--dense.q-field--auto-height.q-field--labeled .q-field__control-container { padding-top: 18px; }
.q-field--dense.q-field--auto-height.q-field--labeled .q-field__native { padding-top: 0; padding-bottom: 2px; min-height: 26px; }
.q-field--outlined.q-field--dark .q-field__control:before { border-color: var(--rz-border-strong); }
.q-field__label { color: var(--rz-label); }
.q-field--dark .q-field__native, .q-field--dark .q-field__input, .q-field--dark .q-select__dropdown-icon { color: #F4F4F5; }
.q-field--outlined:not(.q-field--dark) .q-field__control:before { border-color: var(--rz-border-strong); }
.q-btn { text-transform: none; font-weight: 600; letter-spacing: .01em; border-radius: 9px; font-size: 14px; }
/* red is the one main action of a view and what is switched on: every other button is plain text with a red icon */
/* (Quasar's color classes are !important in a later cascade layer: their variable is set instead) */
.q-btn--flat.text-primary, .q-btn--outline.text-primary { --q-primary: var(--rz-text); }
.q-btn--flat.text-primary .q-icon, .q-btn--outline.text-primary .q-icon { color: var(--rz-red-bright); }
.q-btn--flat.text-primary:hover, .q-btn--outline.text-primary:hover { --q-primary: var(--rz-heading); }
.q-btn--outline.text-primary::before { border-color: var(--rz-border-strong); }
.q-btn--outline.text-primary:hover::before { border-color: var(--rz-red); }
.q-btn.rz-danger, .q-btn.rz-danger .q-icon { color: #FF453A; }
html[data-theme="light"] .q-btn.rz-danger, html[data-theme="light"] .q-btn.rz-danger .q-icon { color: #C8000B; }
.q-btn.disabled { opacity: .38 !important; }
.q-table th { font-size: 13px; color: var(--rz-muted); font-weight: 600; }
.q-table td { font-size: 14px; }
.q-table.text-xs td { font-size: 13px; }
.q-toggle__label, .q-checkbox__label, .q-radio__label { color: var(--rz-text-3); }
.q-tooltip { background: var(--rz-tooltip-bg); border: 1px solid var(--rz-border-strong); color: var(--rz-text); font-size: 14px; max-width: 460px; line-height: 1.5; padding: 8px 12px; }
.q-menu { background: var(--rz-menu-bg); border: 1px solid var(--rz-border-strong); }

/* Train tab: the section list on the left picks the one section shown on the right */
.rz-nav { position: sticky; top: 108px; width: 236px; flex-shrink: 0; gap: 1px; }
@media (max-width: 1400px) { .rz-nav { top: 152px; } }
.rz-nav-title { font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: .1em; color: var(--rz-muted); padding: 14px 12px 6px; }
.rz-nav-title:first-child { padding-top: 2px; }
.rz-adv-hidden { display: none !important; }
.rz-nav-btn { width: 100%; color: var(--rz-text-2); font-weight: 500; min-height: 38px; font-size: 14.5px; border-left: 3px solid transparent; border-radius: 0 9px 9px 0; }
.rz-nav-btn .q-btn__content { justify-content: flex-start; flex-wrap: nowrap; text-align: left; }
.rz-nav-btn .q-icon { color: var(--rz-muted); font-size: 18px; }
.rz-nav-btn.rz-on { background: var(--rz-red-soft); color: var(--rz-heading); border-left-color: var(--rz-red); }
.rz-nav-btn.rz-on .q-icon { color: var(--rz-red-bright); }
.rz-pane { background: var(--rz-surface); border: 1px solid var(--rz-border); border-radius: 14px; padding: 18px 24px 14px; gap: 16px; }
.rz-pane-head { padding-bottom: 16px; border-bottom: 1px solid var(--rz-border); }
.rz-pane-icon { color: var(--rz-red-bright); background: var(--rz-red-soft); border-radius: 11px; font-size: 22px; width: 42px; height: 42px; flex-shrink: 0; }
.rz-pane-title { font-size: 20px; font-weight: 700; color: var(--rz-heading); line-height: 1.25; }
.rz-pane-caption { font-size: 14px; color: var(--rz-muted); }
.rz-pane-foot { border-top: 1px solid var(--rz-border); padding-top: 10px; margin-top: 4px; }
.q-btn.rz-pane-link { color: var(--rz-text-2); font-weight: 500; }
.q-btn.rz-pane-link .q-icon { color: var(--rz-red-bright); }
.rz-intro { background: var(--rz-surface-2); border: 1px solid var(--rz-border); border-radius: 12px; padding: 10px 14px; }
.rz-pane > .rz-subhead { margin-top: 10px; padding-bottom: 6px; border-bottom: 1px solid var(--rz-border); width: 100%; }
/* a setting and its description side by side; the setting fills its column */
.rz-item { display: grid; grid-template-columns: minmax(0, 300px) minmax(0, 1fr); gap: 4px 28px; width: 100%; align-items: start; }
.rz-pane .rz-item > .q-field, .rz-pane .rz-item > .rz-path { width: 100%; max-width: 100%; }
.rz-desc { font-size: 14px; color: var(--rz-muted); line-height: 1.5; padding-top: 9px; max-width: 760px; }
.rz-desc.rz-clamp { display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; cursor: pointer; }
.rz-desc.rz-clamp:hover { color: var(--rz-text-2); }
.rz-desc.rz-clamp.rz-open { display: block; }
.rz-pane .rz-fields { gap: 14px 16px; }

/* action bar */
.nicegui-footer.rz-footer {
  background: var(--rz-footer-bg); backdrop-filter: blur(6px); border-top: 1px solid var(--rz-border); padding: 0;
}
.rz-footer-row { max-width: 1440px; margin: 0 auto; padding: 10px 20px; }
.rz-status-msg {
  font-size: 14px; color: var(--rz-text-2); display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;
  overflow: hidden;
}

/* Monitor */
.rz-views { background: var(--rz-surface); border: 1px solid var(--rz-border); border-radius: 12px; padding: 4px; gap: 4px; }
.rz-views .q-btn { border-radius: 8px; padding: 6px 20px; font-size: 14.5px; font-weight: 500; color: var(--rz-text-2); min-height: 38px; }
.rz-views .q-btn .q-icon { color: var(--rz-muted); font-size: 18px; }
.rz-views .q-btn.rz-on { background: var(--rz-red-soft); color: var(--rz-heading); font-weight: 600; box-shadow: inset 0 0 0 1px var(--rz-red); }
.rz-views .q-btn.rz-on .q-icon { color: var(--rz-red-bright); }
.rz-machine { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 12px; width: 100%; }
.rz-machine-gpus, .rz-machine-sys { display: contents; }
.rz-groups { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, var(--rz-group-min, 336px)), 1fr)); gap: 18px 16px; width: 100%; }
.rz-group { min-width: 0; display: flex; flex-direction: column; gap: 8px; }
.rz-groups .rz-gallery { grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); }
.rz-log-tall { height: min(68vh, 760px); }
.rz-fleet-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, 400px), 1fr)); gap: 14px; }
.rz-fleet-card { gap: 8px; min-width: 0; }
.rz-tile { background: var(--rz-surface); border: 1px solid var(--rz-border); border-radius: 12px; padding: 12px 16px; min-width: 150px; flex: 1 1 150px; }
.rz-tile-label { font-size: 12px; color: var(--rz-muted); text-transform: uppercase; letter-spacing: .08em; }
.rz-tile-value { font-size: 20px; font-weight: 600; color: var(--rz-heading); font-variant-numeric: tabular-nums; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 100%; }
.rz-tile-sub { font-size: 13px; color: var(--rz-muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 100%; }
.rz-banner { border-radius: 10px; padding: 10px 14px; font-size: 14px; width: 100%; }
.rz-banner.error { background: rgba(225, 6, 0, .12); border: 1px solid rgba(255, 59, 48, .45); color: var(--rz-err-text); }
.rz-banner.warning { background: rgba(245, 158, 11, .10); border: 1px solid rgba(245, 158, 11, .40); color: var(--rz-warn-text); }
.rz-banner.info { background: var(--rz-surface); border: 1px solid var(--rz-border-strong); color: var(--rz-text-2); }
.rz-banner.ok { background: rgba(34, 197, 94, .10); border: 1px solid rgba(34, 197, 94, .40); color: #86EFAC; }
.q-uploader { background: var(--rz-surface); border: 1px dashed var(--rz-border-strong); border-radius: 12px; }
.q-uploader__header { color: var(--rz-text); border-bottom: 1px solid var(--rz-border); }
.rz-log { background: var(--rz-deep); border: 1px solid var(--rz-border); border-radius: 10px; color: var(--rz-text-2); }
.rz-thumb { background: var(--rz-surface-2); border: 1px solid var(--rz-border); border-radius: 10px; overflow: hidden; }
.rz-tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; width: 100%; }
.rz-tiles .rz-tile { min-width: 0; }
.rz-gpu { flex: 1 1 18rem; min-width: 0; }
/* the Monitor's image galleries: square thumbnails in a grid, a title and the scores under each */
.rz-gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(170px, 1fr)); gap: 10px; width: 100%; }
.rz-shot { background: var(--rz-surface-2); border: 1px solid var(--rz-border); border-radius: 10px; overflow: hidden;
  min-width: 0; }
.rz-shot:hover { border-color: var(--rz-border-strong); }
.rz-shot-title { font-size: 13px; font-weight: 600; color: var(--rz-text); padding: 6px 8px 0; overflow-wrap: anywhere; }
.rz-shot-sub { font-size: 12px; color: var(--rz-muted); padding: 2px 8px 7px; line-height: 1.4; }
.rz-shot-title:last-child { padding-bottom: 7px; }
.rz-step-pick { width: 11rem; }
/* "label value" chips: a probe's or a sample step's scores */
.rz-kvs { display: flex; flex-wrap: wrap; gap: 6px; }
.rz-kv { display: inline-flex; align-items: baseline; gap: 6px; background: var(--rz-surface-2);
  border: 1px solid var(--rz-border); border-radius: 8px; padding: 3px 10px; font-size: 13px; }
.rz-kv .k { color: var(--rz-muted); font-size: 12px; }
.rz-kv .v { color: var(--rz-heading); font-weight: 600; font-variant-numeric: tabular-nums; }
.rz-kv.warn { border-color: rgba(245, 158, 11, .5); }
.rz-kv.warn .v { color: var(--rz-warn-text); }
/* Previous Runs: a probe image's "tick to compare" box over its corner, and the ticked ones outlined */
.rz-shot { position: relative; }
.rz-shot-pick { position: absolute; top: 4px; right: 4px; background: rgba(0, 0, 0, .55); color: #fff; }
.rz-shot.rz-picked { border: 2px solid var(--rz-red, #E10600); }
.rz-compare-half { width: 50%; }
.rz-compare-img { height: calc(100vh - 96px); }
.rz-ds-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 12px; width: 100%; }
.rz-look-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(130px, 1fr)); gap: 8px; }
.rz-ds-card { background: var(--rz-surface-2); border: 1px solid var(--rz-border); border-radius: 12px;
  overflow: hidden; transition: border-color .15s; min-width: 0; }
.rz-ds-card:hover { border-color: var(--rz-border-strong); }
.rz-ds-card.rz-ds-open { border-color: var(--rz-red); }
.rz-ds-thumbs { display: grid; grid-template-columns: repeat(4, 1fr); gap: 2px; background: var(--rz-thumb-bg); }
.rz-ds-empty { display: flex; align-items: center; justify-content: center; aspect-ratio: 4 / 1;
  color: var(--rz-muted); font-size: 13px; background: var(--rz-thumb-bg); }
.rz-mono { font-family: ui-monospace, "Cascadia Mono", Consolas, Menlo, monospace; font-size: 13px; line-height: 1.5; color: var(--rz-text-2); }

/* Wizard */
.rz-stepper { background: transparent; box-shadow: none; }
.rz-stepper .q-stepper__header { border-bottom: 1px solid var(--rz-border); }
.rz-stepper .q-stepper__tab { padding: 14px 8px; }
.rz-stepper .q-stepper__title { font-size: 13px; font-weight: 500; color: var(--rz-text-2); }
.rz-stepper .q-stepper__tab--active .q-stepper__title { color: var(--rz-heading); font-weight: 600; }
.rz-stepper .q-stepper__dot { width: 30px; min-width: 30px; height: 30px; }
.rz-stepper .q-stepper__tab:not(.q-stepper__tab--active):not(.q-stepper__tab--done) .q-stepper__dot { background: var(--rz-dot-idle); }
.rz-stepper .q-stepper__step-inner { padding: 22px 4px 8px; }
.rz-stepper .q-stepper__nav { padding: 18px 4px 4px; border-top: 1px solid var(--rz-border); margin-top: 8px; width: 100%; }
.rz-setting {
  display: grid; grid-template-columns: minmax(0, 1fr) minmax(240px, 380px); gap: 10px 32px; width: 100%;
  padding: 18px 0; border-bottom: 1px solid var(--rz-border); align-items: start;
}
.rz-setting-title { font-weight: 600; font-size: 15px; color: var(--rz-heading); }
.rz-rec { font-size: 14px; color: var(--rz-rec); margin-top: 2px; }
.rz-rec .q-icon { color: var(--rz-red-bright); font-size: 17px; }
.rz-rec.ok { color: var(--rz-ok-text); }
.rz-rec.ok .q-icon { color: var(--rz-ok); }
.rz-apply { background: var(--rz-red-soft); border: 1px solid rgba(255, 59, 48, .35); border-radius: 12px; padding: 10px 14px; }
.rz-choices { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 14px; width: 100%; }
.rz-racers { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 14px; }
.rz-racer {
  background: var(--rz-surface-2); border: 1px solid var(--rz-border-strong); border-radius: 12px; padding: 14px;
  gap: 6px;
}
.rz-choice {
  background: var(--rz-surface-2); border: 1px solid var(--rz-border-strong); border-radius: 12px; padding: 16px;
  cursor: pointer; transition: border-color .15s, background .15s; gap: 8px;
}
.rz-choice:hover { border-color: #71717A; }
.rz-choice.selected {
  border-color: var(--rz-red); background: var(--rz-choice-bg);
  box-shadow: inset 0 0 0 1px var(--rz-red), 0 0 0 3px rgba(225, 6, 0, .18);
}
.rz-choice-icon { color: var(--rz-red-bright); font-size: 24px; }
.rz-check { display: none; color: var(--rz-red-bright); font-size: 22px; }
.rz-choice.selected .rz-check { display: inline-flex; }
.rz-badge { font-size: 11px; font-weight: 600; letter-spacing: .04em; padding: 1px 8px; border-radius: 999px; background: var(--rz-red); color: #fff; }
.rz-badge.muted { background: var(--rz-badge-muted); }
.rz-evidence { font-size: 11px; font-weight: 600; padding: 1px 8px; border-radius: 999px; border: 1px solid; cursor: help; }
.rz-evidence.others, .rz-evidence.measured { color: var(--rz-ok-text); border-color: var(--rz-ok); }
.rz-evidence.mixed, .rz-evidence.cost, .rz-evidence.worse { color: var(--rz-warn-text); border-color: var(--rz-warn); }
.rz-evidence.untested { color: var(--rz-muted); border-color: var(--rz-border-strong); }
.rz-evidence.tool { color: var(--rz-text-2); border-color: var(--rz-border-strong); }
.rz-exp { padding-top: 10px; padding-bottom: 10px; }
.rz-summary { display: grid; grid-template-columns: max-content minmax(0, 1fr); gap: 8px 24px; width: 100%; font-size: 14px; }
.rz-summary .k { color: var(--rz-muted); }
.rz-summary .v { color: var(--rz-text-3); overflow-wrap: anywhere; }
@media (max-width: 600px) {
  .rz-tiles { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }
  .rz-tile { padding: 9px 12px; }
  .rz-tile-value { font-size: 17px; }
  .rz-gallery { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }
  .rz-monitor .rz-section .q-item__label--caption, .rz-history .rz-section .q-item__label--caption { display: none; }
  .rz-kv { font-size: 12px; padding: 2px 8px; }
  .rz-step-pick { width: auto; flex: 1 1 auto; min-width: 0; }
  .rz-grow-phone { flex: 1 1 auto; min-width: 0; }
  /* Previous Runs: the table's less needed columns hidden, the compared images one above the other */
  .rz-wide-col { display: none; }
  .rz-history .q-table td { white-space: normal; }
  .rz-history .q-table th, .rz-history .q-table td { padding-left: 6px; padding-right: 6px; }
  .rz-compare { flex-direction: column; flex-wrap: nowrap; }
  .rz-compare-half { width: 100%; }
  .rz-compare-img { height: calc(50vh - 64px); }
  /* the action bar: one row of compact buttons, the secondary ones as icons (their tooltips name them) */
  .rz-footer-row { gap: 4px 8px; padding: 6px 10px; }
  .rz-footer-row > .rz-footer-msg { min-width: 0; flex-basis: 100%; }
  .rz-footer-row > .rz-footer-btns { width: 100%; justify-content: space-between; }
  .rz-footer .q-btn { min-height: 38px; padding: 4px 10px; font-size: 13px; }
  .rz-footer .q-btn .q-btn__content { flex-wrap: nowrap; white-space: nowrap; }
  .rz-footer .q-btn.rz-icon-only .q-btn__content > .block { font-size: 0; max-width: 0; overflow: hidden; }
  .rz-footer .q-btn.rz-icon-only .q-icon { margin-right: 0; }
}
@media (max-width: 1560px) { .rz-wordmark { display: none; } }
@media (max-width: 1400px) { .rz-tabs { order: 3; flex-basis: 100%; margin: 0 -8px; } .rz-wordmark { display: block; } }
@media (max-width: 760px) {
  .rz-wordmark { display: none; }
  .rz-views { flex-wrap: nowrap; }
  .rz-views .q-btn .q-icon { display: none; }
  .rz-run-row { padding-left: 12px; padding-right: 12px; gap: 8px; }
  .rz-run-job, .rz-run-go { display: none; }
  .rz-run-bar { min-width: 32px; }
  .rz-views { width: 100%; }
  .rz-views .q-btn { flex: 1 1 auto; padding: 6px 6px; font-size: 14px; white-space: nowrap; }
  .rz-chip { padding: 3px 9px; }
  .rz-setting, .rz-item { grid-template-columns: 1fr; }
  .rz-desc { padding-top: 0; }
  .rz-pane { padding: 14px 14px 10px; }
  .rz-panels .q-tab-panel { padding: 14px 12px 24px; }
  .rz-footer-row, .rz-header-row { padding-left: 12px; padding-right: 12px; }
}

/* light look: progress tracks, image backdrops, theme toggle, logos */
.q-linear-progress__track { background: var(--rz-track); opacity: 1; }
.rz-imgbg { background: var(--rz-img-bg); }
.rz-strong { color: var(--rz-heading); }
.rz-header-row, .rz-footer-row { color: var(--rz-text); }  /* the header and footer bars carry Quasar text-white */
.rz-rec .q-btn { color: var(--rz-text-3); }
/* file buttons (saved LoRAs, the likeness spreadsheet): legible text on a surface pill, the icon in red */
.q-btn.rz-file-btn { background: var(--rz-surface-2); color: var(--rz-text); border: 1px solid var(--rz-border-strong);
  font-size: 13px; font-weight: 600; padding: 4px 12px; min-height: 32px; }
.q-btn.rz-file-btn .q-icon { color: var(--rz-red-bright); font-size: 18px; }
.q-btn.rz-metric-chip { color: var(--rz-muted); border: 1px solid var(--rz-border-strong); border-radius: 999px;
  padding: 2px 14px; min-height: 30px; font-size: 13px; }
.q-btn.rz-metric-chip .q-btn__content { gap: 8px; flex-wrap: nowrap; }
.q-btn.rz-metric-chip.rz-on { color: var(--rz-text); background: var(--rz-red-soft); border-color: var(--rz-red); }
.rz-line { display: inline-block; width: 22px; border-top: 2px solid currentColor; }
.rz-line-dashed { border-top-style: dashed; }
.rz-line-thick { border-top-width: 4px; }
.q-btn.rz-file-btn:hover { border-color: var(--rz-red); background: var(--rz-red-soft); }
.rz-light-only { display: none !important; }
html[data-theme="light"] .rz-light-only { display: inline-flex !important; }
html[data-theme="light"] .rz-dark-only { display: none !important; }
.rz-theme-btn { color: var(--rz-muted); }
.rz-theme-btn:hover { color: var(--rz-heading); }
.rz-look-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; width: 100%; }
.rz-look { border: 2px solid var(--rz-border-strong); border-radius: 14px; padding: 10px; cursor: pointer; gap: 10px;
  background: var(--rz-surface); transition: border-color .15s, box-shadow .15s; }
.rz-look:hover { border-color: var(--rz-hover-border); }
.rz-look.selected { border-color: var(--rz-red); box-shadow: 0 0 0 3px rgba(225, 6, 0, .18); }
.rz-look-name { font-weight: 600; font-size: 15px; color: var(--rz-heading); }
.rz-look .rz-check { font-size: 20px; }
.rz-look.selected .rz-check { display: inline-flex; }
.rz-mini { border-radius: 9px; height: 96px; padding: 10px; display: flex; flex-direction: column; gap: 7px;
  border: 1px solid #262626; overflow: hidden; align-self: stretch; }
.rz-mini.dark { background: #0A0A0A; }
.rz-mini.light { background: #FFFFFF; border-color: #D9D9DE; }
.rz-mini .bar { height: 8px; border-radius: 4px; }
.rz-mini.dark .bar { background: #1F1F1F; }
.rz-mini.light .bar { background: #EBEBEF; }
.rz-mini .bar.text { width: 60%; }
.rz-mini.dark .bar.text { background: #EDEDED; }
.rz-mini.light .bar.text { background: #111114; }
.rz-mini .bar.red { width: 34%; background: #E10600; height: 14px; margin-top: auto; }
.rz-mini .panel { border-radius: 6px; flex: 1; padding: 6px; display: flex; flex-direction: column; gap: 5px; }
.rz-mini.dark .panel { background: #131313; border: 1px solid #262626; }
.rz-mini.light .panel { background: #F5F5F7; border: 1px solid #D9D9DE; }

/* Help */
.rz-help { max-width: 920px; }
.rz-help h2 { font-size: 20px; font-weight: 700; color: var(--rz-heading); margin: 28px 0 10px; padding-bottom: 6px; border-bottom: 1px solid var(--rz-border); }
.rz-help h2:first-child { margin-top: 0; }
.rz-help p, .rz-help li { color: var(--rz-text-2); line-height: 1.65; }
.rz-help strong { color: var(--rz-heading); }
.rz-help code { background: var(--rz-code-bg); color: var(--rz-code); padding: 1px 6px; border-radius: 5px; font-size: 13px; }
.rz-help pre { background: var(--rz-deep); border: 1px solid var(--rz-border); border-radius: 10px; padding: 12px 14px; overflow-x: auto; }
.rz-help pre code { background: transparent; color: var(--rz-text-3); padding: 0; }
.rz-help a { color: var(--rz-red-bright); }
"""


# The look (dark or light) is per browser: localStorage "rulz.theme". This runs in <head>, before anything is drawn,
# so the page starts in the saved look; rzSetTheme switches it live (Quasar components and the CSS variables).
THEME_KEY = "rulz.theme"
THEME_HEAD = """<script>
(function () {
  var t = 'dark';
  try { t = localStorage.getItem('%(key)s') === 'light' ? 'light' : 'dark'; } catch (e) {}
  document.documentElement.dataset.theme = t;
  window.rzDark = t !== 'light';
  window.rzSetTheme = function (theme, save) {
    if (save) { try { localStorage.setItem('%(key)s', theme); } catch (e) {} }
    document.documentElement.dataset.theme = theme;
    window.rzDark = theme !== 'light';
    if (typeof setDark === 'function') setDark(window.rzDark);
  };
})();
</script>
<style>html[data-theme="light"], html[data-theme="light"] body { background: #FFFFFF; }</style>""" % {"key": THEME_KEY}

# ECharts colors are options, not CSS: the light look swaps these (and back). Keys and values must not overlap.
CHART_LIGHT = {"#F4F4F5": "#1C1C21", "#D4D4D8": "#3A3A40", "#A1A1AA": "#6E6E77", "#3F3F46": "#B8B8C0",
               "#222222": "#E6E6EA", "#18181B": "#FFFFFE", "#52525B": "#9A9AA3", "#F59E0B": "#D97706",
               "#34D399": "#059669", "#60A5FA": "#2563EB", "#C084FC": "#9333EA", "#FB7185": "#E11D48"}
CHART_DARK = {v: k for k, v in CHART_LIGHT.items()}


def _recolor(obj, table: dict):
    if isinstance(obj, str):
        return table.get(obj, obj)
    if isinstance(obj, dict):
        return {k: (v if k == "data" else _recolor(v, table)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_recolor(v, table) for v in obj]
    return obj


def recolor_chart(chart, light: bool) -> None:
    """Switch an ECharts chart built with chart_options() to the light or the dark look (no-op if already there)."""
    if getattr(chart, "_rz_light", False) == light:
        return
    chart._rz_light = light
    chart.options.update(_recolor(dict(chart.options), CHART_LIGHT if light else CHART_DARK))
    chart.update()


def setup_theme() -> None:
    """Once per process, before the first page: logo files, the stylesheet and outlined form fields."""
    app.add_static_files("/assets", ASSETS)
    ui.add_head_html(THEME_HEAD, shared=True)
    # Quasar starts in the look saved in this browser (set by THEME_HEAD) instead of always dark, so a reload in
    # the light look does not flash dark first
    app.config.vue_config_script = app.config.vue_config_script.replace(
        "setDark(dark);", "setDark(window.rzDark ?? dark);")
    ui.add_css(THEME_CSS, shared=True)
    for element in (ui.input, ui.number, ui.textarea):
        element.default_props("outlined dense")
    ui.select.default_props("outlined dense options-dense")


def theme_colors() -> None:
    ui.colors(primary=RED, secondary="#27272A", accent="#FF3B30", dark="#131313", dark_page="#0A0A0A",
              positive="#22C55E", negative="#FF453A", info="#3F3F46", warning="#F59E0B")


CLEAN_SLIDE = "60:7:30:8:3"  # Fizgig's MOSS-Image Slide until 8 November 2026 (rulztrainer.clean_share.SLIDE)
CLEAN_SLIDE_MUON = "11:7:40:8:3"  # its Slide since 10 November 2026, for Muon (rulztrainer.clean_share.SLIDE_MUON)
CLEAN_SLIDE_FT = "11:4:30:8:2"  # its full fine-tune default, in rotations (rulztrainer.clean_share.FT_SLIDE)
CLEAN_H3_IMAGE = "40"  # Fizgig's MiniMax-H3-Image presets: 40% of the steps at the last of its 6 levels (0.232)


def parse_ranks(text) -> list[int]:
    """"64, 128" -> [64, 128]; empty -> [64]."""
    out = []
    for part in str(text or "").replace(";", ",").split(","):
        part = part.strip()
        if part:
            out.append(int(float(part)))
    return out or [64]


def clean_share_hint(form) -> str:
    """What the timestep settings put at the clean end (sigma below 0.5), and what clean_share makes of it."""
    from rulztrainer import clean_share as cs
    from rulztrainer.archs import get_arch

    try:
        cfg = form.to_config()
        schedule = cs.parse(cfg.clean_share)
        own = cs.measured_share(cfg, 0.0, 4096, get_arch(cfg))
    except NotImplementedError:
        return "This model has no reference timestep recipe: pick another timestep sampling."
    except (ValueError, TypeError) as e:
        return f"Check the clean-end share: {e}"
    text = f"These timestep settings put {own:.0f}% of steps at the clean end on their own."
    if schedule is None:
        return text
    first, until, rest, end, every = schedule
    if first == rest:
        return text + f" With the schedule: {first:g}% all run."
    unit = "rotation" if cfg.finetune else "epoch"
    plan = (f"{first:g}% until {unit} {until:g}, then {rest:g}%" if end <= until else
            f"{first:g}% until {unit} {until:g}, sliding to {rest:g}% by {unit} {end:g}")
    hi, lo = max(first, rest), min(first, rest)
    nth = {1: "", 2: "2nd ", 3: "3rd "}.get(every, f"{every}th ")
    again = (f", then {hi:g}% with every {nth}{unit} at {lo:g}% at half the learning rate" if every > 0 else "")
    return text + " With the schedule: " + plan + again + "."


def while_connected(callback):
    """A ui.timer callback that rests while its page has no browser connected. A page is kept for PAGE_KEEP_SECONDS
    after its browser goes away, so a closed or reloaded one must not go on reading files and redrawing all that time;
    the first tick after a reconnect brings a page back up to date."""
    client = ui.context.client
    return lambda: callback() if client.has_socket_connection else None


def section(title: str, caption: str, icon: str, open_: bool = True):
    """A collapsible settings card (Train and Monitor tabs)."""
    return ui.expansion(title, caption=caption, icon=icon, value=open_).classes("rz-section w-full")


class SectionNav:
    """The Train tab's settings sections, one shown at a time: the list on the left (a dropdown on narrow screens)
    picks which, and each section ends with links to its neighbours."""

    def __init__(self):
        self.items: list[tuple[str, str, str, object]] = []  # (group, title, icon, section column)
        self.buttons: list = []
        self.picker = None
        self.current = 0
        self.group_labels: dict = {}

    @contextmanager
    def section(self, group: str, title: str, caption: str, icon: str):
        with ui.column().classes("rz-pane w-full") as box:
            with ui.row().classes("rz-pane-head w-full items-center no-wrap gap-3"):
                ui.icon(icon).classes("rz-pane-icon")
                with ui.column().classes("gap-0 min-w-0"):
                    ui.label(title).classes("rz-pane-title")
                    ui.label(caption).classes("rz-pane-caption")
            yield box
        box.visible = not self.items
        self.items.append((group, title, icon, box))

    def build(self, nav, picker_row) -> None:
        """The section list into `nav`, the narrow-screen dropdown into `picker_row`, the neighbour links into each
        section."""
        group = None
        with nav:
            for i, (grp, title, icon, _) in enumerate(self.items):
                if grp != group:
                    group = grp
                    self.group_labels[grp] = ui.label(grp).classes("rz-nav-title")
                self.buttons.append(ui.button(title, icon=icon, on_click=lambda _, i=i: self.show(i), color=None)
                                    .props("flat no-caps align=left").classes("rz-nav-btn"))
        with picker_row:
            self.picker = ui.select({i: f"{grp}: {title}" for i, (grp, title, _, _) in enumerate(self.items)},
                                    value=0, label="Section", on_change=lambda e: self.show(e.value)) \
                .classes("w-full")
        self.foots: list = []
        self.set_hidden(set())  # the neighbour links
        self.show(0, scroll=False)

    def set_hidden(self, hidden: set[int]) -> None:
        """Hide these sections (the Advanced switch): their list buttons, dropdown entries and the neighbour links to
        them, which then lead to the next section shown."""
        self.hidden = hidden
        for i, btn in enumerate(self.buttons):
            btn.classes(add="rz-adv-hidden") if i in hidden else btn.classes(remove="rz-adv-hidden")
        shown = [i for i in range(len(self.items)) if i not in hidden]
        for grp, label in self.group_labels.items():  # a group with nothing shown loses its heading too
            if any(self.items[i][0] == grp for i in shown):
                label.classes(remove="rz-adv-hidden")
            else:
                label.classes(add="rz-adv-hidden")
        self.picker.set_options({i: f"{self.items[i][0]}: {self.items[i][1]}" for i in shown},
                                value=self.current if self.current in shown else None)
        for row in self.foots:
            row.delete()
        self.foots = []
        for pos, i in enumerate(shown):
            with self.items[i][3], ui.row().classes("rz-pane-foot w-full items-center no-wrap") as row:
                if pos > 0:
                    prev = shown[pos - 1]
                    ui.button(self.items[prev][1], icon="chevron_left", on_click=lambda _, j=prev: self.show(j),
                              color=None).props("flat no-caps").classes("rz-pane-link")
                ui.space()
                if pos + 1 < len(shown):
                    nxt = shown[pos + 1]
                    ui.button(self.items[nxt][1], on_click=lambda _, j=nxt: self.show(j), color=None) \
                        .props("flat no-caps icon-right=chevron_right").classes("rz-pane-link")
            self.foots.append(row)
        if self.current in hidden:
            self.show(shown[0] if shown else 0, scroll=False)

    def show(self, index: int, scroll: bool = True) -> None:
        if index is None:
            return
        self.current = index
        for i, (_, _, _, box) in enumerate(self.items):
            box.visible = i == index
            if i == index:
                self.buttons[i].classes(add="rz-on")
            else:
                self.buttons[i].classes(remove="rz-on")
        if self.picker.value != index:
            self.picker.value = index
        if scroll:
            ui.run_javascript("window.scrollTo({top: 0, behavior: 'smooth'})")


def inline_descriptions(root) -> None:
    """Shows the tooltip of every setting under `root` as a description beside the setting (the Train tab): the two
    become one row, hidden together with the setting. Buttons keep their tooltips."""
    from nicegui.elements.mixins.value_element import ValueElement

    elements = {el.html_id: el for el in root.descendants(include_self=True)}
    for tip in [el for el in elements.values() if isinstance(el, ui.tooltip)]:
        target = tip.props.get("target")  # NiceGUI 3 places the tooltip beside its element; 2 inside it
        target = elements.get(target.lstrip("#")) if isinstance(target, str) else tip.parent_slot.parent
        if not isinstance(target, ValueElement):
            continue
        control = target.parent_slot.parent if "rz-path" in target.parent_slot.parent.classes else target
        slot = control.parent_slot
        text = tip.text
        tip.delete()
        with slot:
            item = ui.element("div").classes("rz-item")
        item.move(slot.parent, slot.children.index(control), target_slot=slot.name)
        control.move(item)
        with item:
            desc = ui.label(text).classes("rz-desc")
        if len(text) > 240:  # long ones show three lines until clicked
            desc.classes("rz-clamp").on("click", js_handler="(e) => e.currentTarget.classList.toggle('rz-open')")
        item.bind_visibility_from(control, "visible")


MONITOR_VIEWS = {"overview": ("Overview", "dashboard"), "images": ("Images", "photo_library"),
                 "details": ("Details", "query_stats"), "log": ("Log & files", "terminal")}
TRAIN_EXP_SECTIONS = {"LoRA Isolation": "isolation", "Configuration race": "race", "Seed soup": None,
                      "Post-hoc EMA": None, "Probe noise check": None}  # title: the switch that keeps it shown
TRAIN_EXP_KEYS = ("dora", "likeness_lr_control", "resolution_step_up")  # experimental rows outside an Advanced panel


class TrainAdvanced:
    """The Train tab's half of the Advanced switch (the wizard's is the other): off, the Advanced options panels,
    the experimental settings and the tool sections are hidden, except those holding an experimental setting that is
    switched on, so nothing is on out of sight."""

    def __init__(self, root, sections: SectionNav, w: dict):
        self.sections, self.w, self.on = sections, w, False
        self.off = {ex["key"]: ex["off"] for ex in EXPERIMENTS}
        panels = [e for e in root.descendants() if isinstance(e, ui.expansion) and "rz-advanced" in e.classes]
        self.panels = [(p, [k for k in self.off if self.w[k] in set(p.descendants())]) for p in panels]
        self.rows = []
        for key in TRAIN_EXP_KEYS:
            ctl = w[key]
            parent = ctl.parent_slot.parent
            self.rows.append((parent if "rz-item" in parent.classes else ctl, key))
        for key in self.off:
            w[key].on_value_change(lambda _: self.apply())

    def is_on(self, key: str | None) -> bool:
        return key is not None and not _same(self.w[key].value, self.off.get(key, False))

    def apply(self, on: bool | None = None) -> None:
        if on is not None:
            self.on = bool(on)
        for el, keep in [(p, any(self.is_on(k) for k in keys)) for p, keys in self.panels] + \
                [(el, self.is_on(k)) for el, k in self.rows]:
            el.classes(remove="rz-adv-hidden") if self.on or keep else el.classes(add="rz-adv-hidden")
        self.sections.set_hidden(set() if self.on else {
            i for i, (_, title, _, _) in enumerate(self.sections.items)
            if title in TRAIN_EXP_SECTIONS and not self.is_on(TRAIN_EXP_SECTIONS[title])})


class ViewSwitch:
    """A row of buttons that shows one view of a tab at a time (the Monitor's): {key: (label, icon)}. on_change(key)
    is called when another is picked."""

    def __init__(self, views: dict, value: str):
        self.value, self.on_change, self.buttons = value, None, {}
        with ui.row().classes("rz-views items-center"):
            for key, (label, icon) in views.items():
                self.buttons[key] = ui.button(label, icon=icon, on_click=lambda _, k=key: self.pick(k), color=None) \
                    .props("flat no-caps")
        self.buttons[value].classes(add="rz-on")

    def pick(self, key: str) -> None:
        if key == self.value or key not in self.buttons:
            return
        self.buttons[self.value].classes(remove="rz-on")
        self.buttons[key].classes(add="rz-on")
        self.value = key
        if self.on_change is not None:
            self.on_change(key)


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
    """A value axis; labels that would overlap are left out (a step axis on a phone)."""
    return {"type": "value", "name": name, "axisLine": {"lineStyle": {"color": "#3F3F46"}},
            "axisLabel": {"color": "#A1A1AA", "hideOverlap": True}, "splitLine": {"lineStyle": {"color": "#222222"}},
            "nameTextStyle": {"color": "#A1A1AA"}, **kw}


def chart_options(x: dict, y, series: list, legend: list[str]) -> dict:
    """ECharts options in the dark theme: transparent background, muted axes, the brand palette first. The legend
    wraps onto more rows instead of paging and the step axis is named under it (fit_chart makes room for the
    legend's rows; ECharts 6 keeps the axis labels inside the chart)."""
    if x.get("name") == "step":
        x = {**x, "nameLocation": "middle", "nameGap": 26}
    return {"backgroundColor": "transparent", "color": CHART_COLORS, "textStyle": {"color": "#D4D4D8"},
            "xAxis": x, "yAxis": y, "series": series,
            "legend": {"data": legend, "type": "plain", "top": 0, "itemGap": 12, "textStyle": {"color": "#D4D4D8"},
                       "pageTextStyle": {"color": "#A1A1AA"}},
            "tooltip": {"trigger": "axis", "confine": True, "backgroundColor": "#18181B", "borderColor": "#3F3F46",
                        "textStyle": {"color": "#F4F4F5"}},
            "grid": {"left": 52, "right": 24, "top": 40, "bottom": 44}}


def fit_chart(chart) -> None:
    """Fit a chart_options() chart to its browser's width (chart._rz_width, the window's): the legend's rows are
    counted the way ECharts wraps them and the plot starts below them, with smaller text and margins on a phone."""
    width = getattr(chart, "_rz_width", 1280)
    narrow = width < 600
    plot = min(width, 1440) - (100 if narrow else 160)  # less the page's and the card's padding (a 375 px phone: 282)
    legend = chart.options.get("legend") or {}
    names = [n["name"] if isinstance(n, dict) else str(n) for n in legend.get("data") or []]
    font, icon, gap = (11, 16, 8) if narrow else (12, 25, 12)
    rows, used = (1 if names else 0), 0.0
    for name in names:
        w = icon + 5 + len(name) * font * 0.62 + gap
        if used and used + w > plot:
            rows, used = rows + 1, 0.0
        used += w
    legend.update(itemGap=gap, itemWidth=icon, textStyle={**legend.get("textStyle", {}), "fontSize": font})
    y_axes = chart.options.get("yAxis")
    named = any(a.get("name") for a in (y_axes if isinstance(y_axes, list) else [y_axes or {}]))
    for i, axis in enumerate(y_axes if isinstance(y_axes, list) else [y_axes or {}]):
        if axis.get("name"):  # the name runs inwards from the axis instead of centred on it (clipped at the edge)
            right = axis.get("position", "right" if i else "left") == "right"  # a second axis is on the right
            axis.setdefault("nameTextStyle", {})["align"] = "right" if right else "left"
    # a legend row is about 25 px high, and the y axis name sits about 30 px above the plot
    chart.options["grid"].update(top=rows * 25 + (34 if named else 10), left=40 if narrow else 52,
                                 right=14 if narrow else 24)
    # the chart grows by the legend's extra rows, so a long legend (a race on a phone) leaves the plot its height
    base = next((int(c[2:]) * 4 for c in chart.classes if re.fullmatch(r"h-\d+", c)), None)
    if base is not None:
        extra = max(0, rows - 1) * 25
        chart.style(replace=f"height: {base + extra}px" if extra else "")


class RzChart(ui.echart):
    """An ECharts chart that fits its legend and margins to the browser's width on every update."""

    def update(self) -> None:
        if "grid" in self.options and not getattr(self, "_rz_fitting", False):
            self._rz_fitting = True  # the options are observed: changing them calls update() again
            try:
                fit_chart(self)
            finally:
                self._rz_fitting = False
        super().update()


def _line(name: str, color: str, **kw) -> dict:
    return {"name": name, "type": "line", "showSymbol": False, "data": [], "itemStyle": {"color": color},
            "lineStyle": {"color": color, **kw.pop("lineStyle", {})}, **kw}


def path_field(label: str, kind: str, value: str = "", width: str = "w-96", filetypes=None):
    """Text input with a Browse button: a native dialog for a viewer at this machine's desktop, the in-page browser of
    the server's files otherwise (a vast.ai instance, a remote browser). width "w-full" makes the input fill its
    container."""
    full = width == "w-full"
    native = _native_dialogs  # this page's viewer, read while the page is built
    with ui.row().classes("rz-path items-center gap-1 no-wrap max-w-full" + (" w-full" if full else "")):
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
LOOK_SHOWN = 48  # look filter results shown before Show all
LOOK_COLORS = {"match": "positive", "borderline": "warning", "drift": "negative", "no face": "grey",
               "no body": "grey"}


@app.get("/rz-image")
def _serve_image(path: str):
    """An image file of this machine for the page (image_url). Image files only."""
    from fastapi import HTTPException
    from fastapi.responses import FileResponse

    p = Path(path)
    if p.suffix.lower() not in IMAGE_EXTS or not p.is_file():
        raise HTTPException(status_code=404, detail="Not Found")
    return FileResponse(p, headers={"Cache-Control": "max-age=86400"})


@app.get("/rz-lora")
def _serve_lora(path: str):
    """A LoRA a run saved, as a download: .safetensors files inside a job folder (one with a status.json) only."""
    from fastapi import HTTPException
    from fastapi.responses import FileResponse

    p = Path(path)
    if p.suffix.lower() != ".safetensors" or not p.is_file() or not (p.parent / "status.json").is_file():
        raise HTTPException(status_code=404, detail="Not Found")
    return FileResponse(p, filename=p.name, media_type="application/octet-stream")


@app.get("/rz-likeness-xlsx")
def _serve_likeness_xlsx(job: str):
    """The likeness spreadsheet of a job folder (one with a status.json), built now: the run's probes so far."""
    from fastapi import HTTPException
    from fastapi.responses import FileResponse

    from rulztrainer import likeness_export

    d = Path(job)
    if not (d / "status.json").is_file() or not (d / "likeness.jsonl").is_file():
        raise HTTPException(status_code=404, detail="No likeness probes in this job yet")
    try:
        out = likeness_export.export(d)
    except (OSError, ValueError) as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    return FileResponse(out, filename=out.name,
                        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


COMPARE_DIR = ROOT / "output" / "_compare"  # Score sample folders' spreadsheets


@app.get("/rz-compare-xlsx")
def _serve_compare_xlsx(name: str):
    """A Score sample folders spreadsheet: a file name in COMPARE_DIR only."""
    from fastapi import HTTPException
    from fastapi.responses import FileResponse

    p = COMPARE_DIR / Path(name).name
    if p.suffix.lower() != ".xlsx" or not p.is_file():
        raise HTTPException(status_code=404, detail="Not Found")
    return FileResponse(p, filename=p.name,
                        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


@app.get("/rz-history-xlsx")
def _serve_history_xlsx(run: str):
    """The likeness spreadsheet of a Previous Runs record, built from its saved raw probe scores."""
    import tempfile

    from fastapi import HTTPException
    from fastapi.responses import FileResponse
    from starlette.background import BackgroundTask

    from rulztrainer import likeness_export

    rec = next((r for r in history.list_runs() if r["id"] == run), None)
    if rec is None:
        raise HTTPException(status_code=404, detail="No such run in the history")
    # a temporary file of its own, removed once sent (two runs of one name exported at once never share one)
    fd, tmp = tempfile.mkstemp(suffix=".xlsx")
    os.close(fd)
    try:
        likeness_export.export_run(rec, Path(tmp))
    except FileNotFoundError as e:
        os.unlink(tmp)
        raise HTTPException(status_code=404, detail=str(e)) from e
    except (OSError, ValueError) as e:
        os.unlink(tmp)
        raise HTTPException(status_code=500, detail=str(e)) from e
    return FileResponse(tmp, filename=f"{history_file_name(rec)}_likeness.xlsx",
                        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        background=BackgroundTask(os.unlink, tmp))


def history_file_name(rec: dict) -> str:
    """A file name for a history run: its own name if it has one, else the job and its start."""
    started = time.strftime("%Y%m%d-%H%M", time.localtime(rec.get("started") or 0))
    base = rec.get("label") or f"{rec.get('job') or 'run'}_{started}"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._") or "run"


def lora_url(path) -> str:
    """The download URL of a LoRA a run saved (served by /rz-lora)."""
    from urllib.parse import quote

    return f"/rz-lora?path={quote(str(Path(path).resolve()))}"


LOG_SUFFIXES = {".log", ".yaml", ".yml", ".json", ".jsonl", ".txt", ".csv"}


def job_loras(job_dir) -> list[tuple[Path, int]]:
    """[(file, bytes)] of the LoRA / LoKr files a job folder holds (checkpoints, best, final, a race's), newest first."""
    rows = []
    for p in Path(job_dir).glob("*.safetensors") if job_dir else []:
        try:
            st = p.stat()
        except OSError:
            continue
        rows.append((p, st.st_size, st.st_mtime))
    return [(p, size) for p, size, _ in sorted(rows, key=lambda r: r[2], reverse=True)]


def job_logs(job_dir) -> list[Path]:
    """A job folder's logs and settings: its own text files (train.log, config.yaml, status.json, likeness.jsonl, a
    race's winner settings) and those of its sub-folders (race.json, soup.json, noise.json...). No images or states."""
    d = Path(job_dir)
    return sorted(p for pattern in ("*", "*/*") for p in d.glob(pattern)
                  if p.is_file() and p.suffix.lower() in LOG_SUFFIXES)


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
JOB_IMAGE_KINDS = {"probes": "probe", "samples": "sample"}  # the job folder's image folder -> one image's name


def job_images(job_dir, kind: str) -> list[Path]:
    """The images in a job folder's probes/ or samples/ folder (every step, and a race's configurations), sorted by
    name (so by configuration, then step)."""
    d = Path(job_dir) / kind if job_dir and kind in JOB_IMAGE_KINDS else None
    if d is None or not d.is_dir():
        return []
    return sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def size_text(n: int) -> str:
    return f"{n / 2**30:.1f} GB" if n >= 2**30 else f"{n / 2**20:.0f} MB"


@app.get("/rz-job-zip")
def _serve_job_zip(job: str, kind: str):
    """A zip of a job folder's (one with a status.json) LoRA files (kind "loras"), its logs and settings ("logs"),
    or its probe or sample images ("probes", "samples"), built now and deleted once sent. LoRAs and images are stored
    as they are (already compressed)."""
    import tempfile
    import zipfile

    from fastapi import HTTPException
    from fastapi.responses import FileResponse
    from starlette.background import BackgroundTask

    d = Path(job)
    if kind not in ("loras", "logs", *JOB_IMAGE_KINDS) or not (d / "status.json").is_file():
        raise HTTPException(status_code=404, detail="Not Found")
    files = [p for p, _ in job_loras(d)] if kind == "loras" else job_logs(d) if kind == "logs" else job_images(d, kind)
    if not files:
        raise HTTPException(status_code=404, detail="Nothing to pack")
    fd, tmp = tempfile.mkstemp(suffix=".zip")
    os.close(fd)
    try:
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED if kind == "logs" else zipfile.ZIP_STORED) as z:
            for p in files:
                z.write(p, p.relative_to(d).as_posix())
    except OSError as e:
        os.unlink(tmp)
        raise HTTPException(status_code=500, detail=str(e)) from e
    return FileResponse(tmp, filename=f"{d.name}_{kind}.zip", media_type="application/zip",
                        background=BackgroundTask(os.unlink, tmp))


class JobFiles:
    """A job folder's downloads: one LoRA picked in a dropdown, every LoRA as a zip, the logs and settings as a zip,
    and every probe image and every sample image, each as a zip (Monitor and Previous Runs)."""

    def __init__(self):
        self.job_dir: Path | None = None
        self.shown = None
        with ui.row().classes("w-full items-center gap-2"):
            self.pick = ui.select({}, label="LoRA file").classes("w-[30rem] max-w-full")
            self.one = ui.button("Download", icon="download", on_click=self.download_one, color=None) \
                .props("flat no-caps").classes("rz-file-btn") \
                .tooltip("Save the picked LoRA to this computer (useful when the WebUI runs remotely)")
            self.all = ui.button("All LoRAs (.zip)", icon="folder_zip", on_click=lambda: self.download_zip("loras"),
                                 color=None).props("flat no-caps").classes("rz-file-btn") \
                .tooltip("Every LoRA file in the job folder in one archive")
            self.logs = ui.button("Logs and settings (.zip)", icon="description",
                                  on_click=lambda: self.download_zip("logs"), color=None) \
                .props("flat no-caps").classes("rz-file-btn") \
                .tooltip("train.log, config.yaml, status.json, the likeness scores and the run's other text files in "
                         "one archive")
            self.images = {
                kind: ui.button(f"{noun.capitalize()} images (.zip)", icon=icon,
                                on_click=lambda _, k=kind: self.download_zip(k), color=None)
                .props("flat no-caps").classes("rz-file-btn")
                .tooltip(f"Every {noun} image of the run (every step, and each configuration of a race) in one "
                         "archive, as the job folder's " + kind + "/ holds them")
                for kind, noun, icon in (("probes", "probe", "insights"), ("samples", "sample", "photo_library"))}
        self.note = hint("")

    def show(self, job_dir) -> None:
        """List the files of this job folder (None: no run); cheap to call on every poll."""
        self.job_dir = Path(job_dir) if job_dir else None
        loras = job_loras(self.job_dir)
        has_logs = bool(self.job_dir and (self.job_dir / "status.json").is_file())
        images = {kind: len(job_images(self.job_dir, kind)) if has_logs else 0 for kind in JOB_IMAGE_KINDS}
        sig = (str(self.job_dir), [(p.name, size) for p, size in loras], has_logs, images)
        if sig == self.shown:
            return
        self.shown = sig
        options = {str(p): f"{p.name}  ({size_text(size)})" for p, size in loras}
        self.pick.set_options(options, value=self.pick.value if self.pick.value in options else
                              (next(iter(options)) if options else None))
        for btn in (self.one, self.all):
            btn.enabled = bool(loras)
        self.logs.enabled = has_logs
        for kind, btn in self.images.items():
            btn.enabled = images[kind] > 0
        self.note.text = (f"{len(loras)} LoRA file{'s' if len(loras) != 1 else ''}, "
                          f"{size_text(sum(s for _, s in loras))} in {self.job_dir}" if loras else
                          "No LoRA saved yet." if has_logs else "")

    def download_one(self):
        if self.pick.value:
            ui.download(lora_url(self.pick.value), Path(self.pick.value).name)

    def download_zip(self, kind: str):
        from urllib.parse import quote

        if self.job_dir is None:
            return
        n = len(self.shown[1]) if kind == "loras" else self.shown[3].get(kind, 0)
        if kind != "logs":
            what = "LoRA files" if kind == "loras" else f"{JOB_IMAGE_KINDS[kind]} images"
            ui.notify(f"Packing {n} {what}; the download starts when the archive is ready", type="info")
        ui.download(f"/rz-job-zip?job={quote(str(self.job_dir.resolve()))}&kind={kind}",
                    f"{self.job_dir.name}_{kind}.zip")


def history_job_dir(rec: dict, runs: list[dict]) -> Path | None:
    """The job folder of a Previous Runs record while it still holds that run's files: its status.json names the run
    (runs since v2.29.0), or for older runs no later run of the same job name was saved."""
    d = Path(rec.get("job_dir") or ROOT / "output" / (rec.get("job") or ""))
    try:
        status = json.loads((d / "status.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if status.get("run_id"):
        return d if status["run_id"] == rec["id"] else None
    later = any(r.get("job") == rec.get("job") and (r.get("started") or 0) > (rec.get("started") or 0) for r in runs)
    return None if later else d


def clip_or_still(path) -> Path:
    """A sample or probe picture's clip (the same name, .webp: a video model's render) when there is one."""
    clip = Path(path).with_suffix(".webp")
    return clip if Path(path).suffix.lower() == ".png" and clip.is_file() else Path(path)


def image_url(path) -> str:
    """The page's URL for an image file on this machine, served by one fixed route (/rz-image) and versioned by the
    file's modification time. NiceGUI's own route for a local file is shared by every element and page showing that
    file and removed with any one of them, so after a reload the old page's cleanup left thumbnails at 404."""
    from urllib.parse import quote

    p = Path(path).resolve()
    try:
        version = p.stat().st_mtime_ns
    except OSError:
        version = 0
    return f"/rz-image?path={quote(str(p))}&v={version}"


def list_dataset_images(folder: str) -> list[Path]:
    root = Path(folder)
    if not root.is_dir():
        return []
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS and not dataset_skipped(p, root))


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
            # (|white: thumbnails made before transparency was laid on white are made again once)
            key = hashlib.sha1(f"{img.resolve()}|{st.st_mtime_ns}|{st.st_size}|sq|white".encode()).hexdigest()[:16]
            t = tdir / f"{key}.jpg"
            if not t.exists():
                with Image.open(img) as im:
                    im = to_rgb(ImageOps.exif_transpose(im))
                    im = ImageOps.fit(im, (THUMB_SIZE, THUMB_SIZE), Image.Resampling.LANCZOS)  # square center crop
                    im.save(t, "JPEG", quality=85)
            out.append(t)
        except Exception:  # noqa: BLE001
            out.append(img)
    return out


DATASETS_ROOT = ROOT / "datasets"  # where Create dataset puts new datasets (git-ignored)
DATASETS_REGISTRY = ".rulz_datasets.json"  # in DATASETS_ROOT: dataset folders elsewhere that were added to the list
CAPTION_EXTS = {".txt", ".json"}
DATASET_FILE_EXTS = IMAGE_EXTS | CAPTION_EXTS | set(VIDEO_EXTS)  # what an upload or a folder copy takes
_RESERVED_NAMES = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}


def _same_path(a, b) -> bool:
    return os.path.normcase(os.path.abspath(str(a))) == os.path.normcase(os.path.abspath(str(b)))


def _inside(child, parent) -> bool:
    """True when `child` lies strictly inside the folder `parent`."""
    c, p = (os.path.normcase(os.path.abspath(str(x))) for x in (child, parent))
    return c != p and c.startswith(p.rstrip("\\/") + os.sep)


def dataset_name_error(name: str, root: Path | None = None) -> str | None:
    """Why a new dataset cannot take this name, or None when it can."""
    import re

    root = DATASETS_ROOT if root is None else Path(root)
    name = (name or "").strip()
    if not name:
        return "Give the dataset a name."
    if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 _.-]{0,63}", name) or name.endswith(".")
            or name.split(".")[0].lower() in _RESERVED_NAMES):
        return "Use letters, digits, spaces, - _ and . (starting with a letter or digit), up to 64 characters."
    if (root / name).exists():
        return f"There is already a dataset called {name}."
    return None


def create_dataset(name: str, root: Path | None = None) -> Path:
    """A new, empty dataset folder in the datasets folder."""
    root = DATASETS_ROOT if root is None else Path(root)
    err = dataset_name_error(name, root)
    if err:
        raise ValueError(err)
    path = root / name.strip()
    path.mkdir(parents=True)
    return path


def registered_datasets(root: Path | None = None) -> list[str]:
    """Dataset folders outside the datasets folder that were added to the list."""
    root = DATASETS_ROOT if root is None else Path(root)
    try:
        data = json.loads((root / DATASETS_REGISTRY).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [str(p) for p in (data.get("external") or []) if str(p).strip()] if isinstance(data, dict) else []


def _save_registry(paths: list[str], root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / DATASETS_REGISTRY).write_text(json.dumps({"external": paths}, indent=1), encoding="utf-8")


def register_dataset(path: str, root: Path | None = None) -> None:
    """Add a folder to the dataset list (one inside the datasets folder is listed anyway)."""
    root = DATASETS_ROOT if root is None else Path(root)
    if _same_path(path, root) or _inside(root, path):
        raise ValueError("That folder holds the datasets folder: pick one dataset inside it instead.")
    paths = registered_datasets(root)
    if _inside(path, root) or any(_same_path(path, p) for p in paths):
        return
    _save_registry(paths + [str(path)], root)


def unregister_dataset(path: str, root: Path | None = None) -> None:
    """Take a folder off the dataset list; its files stay where they are."""
    root = DATASETS_ROOT if root is None else Path(root)
    paths = registered_datasets(root)
    keep = [p for p in paths if not _same_path(p, path)]
    if len(keep) != len(paths):
        _save_registry(keep, root)


def known_datasets(extra=(), root: Path | None = None) -> list[str]:
    """The dataset folders to list: the sub-folders of the datasets folder (by name), then the folders added from
    elsewhere and `extra` (those a loaded config trains on), each once and only while it exists; paths inside the
    WebUI folder relative to it."""
    root = DATASETS_ROOT if root is None else Path(root)
    found: list[str] = []
    if root.is_dir():
        found += [str(p) for p in sorted(root.iterdir(), key=lambda p: p.name.lower())
                  if p.is_dir() and not p.name.startswith(".")]
    found += [str(p) for p in [*registered_datasets(root), *(extra or [])] if str(p).strip() and Path(p).is_dir()]
    out: list[str] = []
    for p in found:
        rel = _relative(p)
        if not any(_same_path(rel, q) for q in out):
            out.append(rel)
    return out


def upload_target(dest, name: str) -> Path | None:
    """Where an uploaded file goes in a dataset folder: its own file name (a same-named file is replaced), for images,
    video clips (PAN 2 and MiniMax H3 train them) and .txt / .json captions only; None for any other kind of file.
    The folder is created."""
    fname = Path(str(name).replace("\\", "/")).name  # the name only: never a folder from the browser
    if not fname or fname.startswith(".") or Path(fname).suffix.lower() not in DATASET_FILE_EXTS:
        return None
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    return dest / fname


def copy_into_dataset(src, dest) -> dict:
    """Copy the images, video clips and captions (.txt, .json) of a folder on this machine into a dataset, sub-folders included
    and kept; files the dataset already has stay as they are. Returns {"copied", "kept", "ignored"} counts."""
    src, dest = Path(src), Path(dest)
    if not src.is_dir():
        raise ValueError(f"Folder not found: {src}")
    if _same_path(src, dest) or _inside(dest, src) or _inside(src, dest):
        raise ValueError("Pick a folder that is not the dataset itself, nor inside it or around it.")
    out = {"copied": 0, "kept": 0, "ignored": 0}
    for p in sorted(src.rglob("*")):
        if not p.is_file() or ".rulz_cache" in p.parts:
            continue
        if p.suffix.lower() not in DATASET_FILE_EXTS:
            out["ignored"] += 1
            continue
        target = dest / p.relative_to(src)
        if target.exists():
            out["kept"] += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
        out["copied"] += 1
    return out


def images_text(n) -> str:
    """"1 image", "16 images" ("? images" when unknown)."""
    return f"{n} image" if n == 1 else f"{n} images"


def _captioned(img: Path) -> bool:
    return img.with_suffix(".txt").is_file() or img.with_suffix(".json").is_file()


def dataset_counts(paths) -> dict[str, tuple[int, int]]:
    """{path: (images, captioned)} for dataset folders that exist."""
    out = {}
    for p in paths:
        if Path(p).is_dir():
            imgs = list_dataset_images(p)
            out[p] = (len(imgs), sum(1 for i in imgs if _captioned(i)))
    return out


def dataset_cards(paths, n_thumbs: int = 4) -> list[dict]:
    """The Dataset tab's cards: name, path, image and caption counts and a few thumbnails spread over the dataset."""
    out = []
    for p in paths:
        imgs = list_dataset_images(p)
        pick = imgs[::max(1, len(imgs) // n_thumbs)][:n_thumbs]
        out.append({"path": p, "name": Path(p).name or p, "images": len(imgs),
                    "captioned": sum(1 for i in imgs if _captioned(i)),
                    "thumbs": [str(t) for t in make_thumbs(pick, Path(p))] if pick else []})
    return out


def datasets_summary(paths) -> dict | None:
    """dataset_summary over several dataset folders: images and captions added up, the median image size over all,
    a sample of (dataset, image) from each; "missing" lists folders that do not exist. None without any folder."""
    paths = [str(p) for p in paths or [] if str(p).strip()]
    if not paths:
        return None
    parts = [(p, dataset_summary(p)) for p in paths]
    found = [(p, x) for p, x in parts if x is not None]
    meds = sorted(x["median_mp"] for _, x in found if x["median_mp"])
    share = max(1, 8 // max(1, len(found)))
    return {"folder": "|".join(paths), "images": sum(x["images"] for _, x in found),
            "counts": {p: x["images"] for p, x in found},
            "outliers": len(set().union(*[lookfilter.outliers(p) for p, _ in found])),
            "scanned": any(lookfilter.load_scores(p, k) for p, _ in found for k in lookfilter.KINDS),
            "captioned": sum(x["captioned"] for _, x in found), "median_mp": meds[len(meds) // 2] if meds else None,
            "sample": [(p, img) for p, x in found for img in x["sample"][:share]][:8],
            "missing": [p for p, x in parts if x is None]}


def summary_thumbs(data: dict | None) -> list[Path]:
    """Thumbnails of a datasets_summary's sample, each in its own dataset's cache."""
    return [make_thumbs([img], Path(root))[0] for root, img in (data or {}).get("sample", [])]


def image_viewer():
    """Full-window image viewer dialog. Returns show(path, items=None): items, [(path, caption)], are the images its
    arrows (and the keyboard's left and right arrows) step through, the shown one among them."""
    state = {"items": [], "i": 0}
    with ui.dialog().props("maximized") as dlg, ui.card().tight().classes("w-full h-full bg-black"):
        with ui.row().classes("w-full items-center no-wrap gap-1 px-3 py-1"):
            title = ui.label("").classes("text-sm text-white truncate flex-1 min-w-0")
            back = ui.button(icon="chevron_left", on_click=lambda: step(-1)).props("flat dense round color=white") \
                .tooltip("Previous image")
            count = ui.label("").classes("text-xs text-grey-5 whitespace-nowrap")
            ahead = ui.button(icon="chevron_right", on_click=lambda: step(1)).props("flat dense round color=white") \
                .tooltip("Next image")
            ui.button(icon="close", on_click=dlg.close).props("flat dense round color=white")
        big = ui.image("").props("fit=contain").classes("w-full").style("height: calc(100vh - 48px)")
        big.on("click", dlg.close)

    def paint():
        path, caption = state["items"][state["i"]]
        title.text = caption or str(path)
        big.set_source(image_url(path))
        many = len(state["items"]) > 1
        back.visible = ahead.visible = count.visible = many
        count.text = f"{state['i'] + 1} / {len(state['items'])}"

    def step(delta: int):
        if len(state["items"]) > 1:
            state["i"] = (state["i"] + delta) % len(state["items"])
            paint()

    def key(e):
        if dlg.value and e.action.keydown:
            if e.key.arrow_left:
                step(-1)
            elif e.key.arrow_right:
                step(1)

    ui.keyboard(on_key=key, ignore=["input", "select", "button", "textarea"])

    def show(path, items=None):
        items = [(str(q), c) for q, c in (items or [(path, "")])]
        names = [q for q, _ in items]
        if str(path) not in names:
            items.insert(0, (str(path), ""))
            names.insert(0, str(path))
        state.update(items=items, i=names.index(str(path)))
        paint()
        dlg.open()

    return show


def short_scores(sc: dict | None) -> str:
    """"face 47% · body 17% · combined 38%": a sample's likeness for its thumbnail (normalized where it was recorded
    on the scale, the scores themselves otherwise)."""
    if not sc:
        return ""
    parts = []
    for label, nk, rk in (("face", "face_norm", "percent"), ("body", "body_norm", "body_percent"),
                          ("combined", "combined_norm", "combined_percent")):
        v = sc.get(nk)
        if v is None:
            v = sc.get(rk)
        if v is not None:
            parts.append(f"{label} {v:.0f}%")
        elif label == "face":
            parts.append("no face")
    return " · ".join(parts)


def stat_chips(box, items, sig_box: dict | None = None) -> None:
    """Fill `box` with "label value" chips, [(label, value, warn)]: easier to read than one long line of numbers, and
    they wrap on a phone. Redrawn only when they change (sig_box remembers the last ones)."""
    items = [(a, b, bool(w)) for a, b, w in items]
    if sig_box is not None:
        if sig_box.get("sig") == items:
            return
        sig_box["sig"] = items
    box.clear()
    with box:
        for label, value, warn in items:
            with ui.element("div").classes("rz-kv" + (" warn" if warn else "")):
                ui.label(label).classes("k")
                ui.label(value).classes("v")


class StepGallery:
    """A Monitor gallery of one kind of image (samples or likeness probes) by training step: arrows and a dropdown
    pick the step (the newest by default; it follows new steps while it shows the newest), a summary line for the
    step, and a grid of thumbnails, each with a title and its scores. A click opens the full-size viewer, whose
    arrows step through the step's images."""

    def __init__(self, empty: str, viewer):
        self.viewer = viewer
        self.empty = empty
        self.steps: dict[int, tuple[str, list]] = {}  # step -> (summary, [(group, path, title, scores)])
        self.step: int | None = None  # the step shown; None = the newest
        self._quiet = False
        with ui.row().classes("w-full items-center gap-1 no-wrap") as self.bar:
            self.older = ui.button(icon="chevron_left", on_click=lambda: self.move(1)) \
                .props("flat dense round").tooltip("The step before")
            self.pick = ui.select({}, on_change=self.picked).props("dense options-dense outlined") \
                .classes("rz-step-pick")
            self.newer = ui.button(icon="chevron_right", on_click=lambda: self.move(-1)) \
                .props("flat dense round").tooltip("The step after")
            self.latest = ui.button("Newest", icon="last_page", on_click=lambda: self.go(None), color=None) \
                .props("flat dense no-caps").tooltip("Show the newest images again (and follow new ones)")
        self.bar.visible = False
        self.summary = hint("").classes("w-full")
        self.summary.visible = False
        self.grid = ui.column().classes("w-full gap-2")
        with self.grid:
            hint(empty)

    def order(self) -> list[int]:
        return sorted(self.steps, reverse=True)

    def set(self, steps: dict[int, tuple[str, list]]) -> None:
        if steps == self.steps:
            return
        self.steps = steps
        if self.step not in steps:
            self.step = None
        self._quiet = True
        self.pick.options = {s: f"Step {s:,}" + (" (newest)" if i == 0 else "") for i, s in enumerate(self.order())}
        self.pick.update()
        self._quiet = False
        self.render()

    def shown(self) -> int | None:
        order = self.order()
        return None if not order else self.step if self.step is not None else order[0]

    def go(self, step: int | None) -> None:
        order = self.order()
        self.step = None if (step is None or (order and step == order[0])) else step
        self.render()

    def move(self, delta: int) -> None:
        order = self.order()
        s = self.shown()
        if s is not None:
            self.go(order[min(max(order.index(s) + delta, 0), len(order) - 1)])

    def picked(self, e) -> None:
        if not self._quiet and e.value is not None and e.value != self.shown():
            self.go(e.value)

    def render(self) -> None:
        order = self.order()
        s = self.shown()
        self.bar.visible = bool(order)
        self.grid.clear()
        if s is None:
            self.summary.visible = False
            with self.grid:
                hint(self.empty)
            return
        i = order.index(s)
        self._quiet = True
        self.pick.value = s
        self._quiet = False
        self.older.set_enabled(i < len(order) - 1)
        self.newer.set_enabled(i > 0)
        self.latest.visible = i > 0
        summary, items = self.steps[s]
        self.summary.text = summary
        self.summary.visible = bool(summary)
        view = [(q, " · ".join(x for x in (f"Step {s:,}", g, title, scores) if x)) for g, q, title, scores in items]
        groups: dict[str, list] = {}
        for item in items:
            groups.setdefault(item[0], []).append(item)
        named = any(groups)
        widest = max((len(m) for m in groups.values()), default=1)
        with self.grid:
            # named groups (a race: one per probe or prompt, its configurations side by side) share the rows, each
            # as wide as up to four of its pictures; one unnamed group fills the width
            with ui.element("div").classes("rz-groups" if named else "w-full") \
                    .style(f"--rz-group-min: {min(widest, 4) * 168}px" if named else ""):
                for group, members in groups.items():
                    with ui.element("div").classes("rz-group" if named else "w-full"):
                        if group:
                            ui.label(group).classes("rz-subhead w-full")
                        with ui.element("div").classes("rz-gallery"):
                            for _, q, title, scores in members:
                                with ui.column().classes("rz-shot gap-0"):
                                    ui.image(image_url(q)).props("ratio=1 fit=contain") \
                                        .classes("w-full rz-imgbg cursor-pointer") \
                                        .on("click", lambda _, q=q: self.viewer(Path(q), view))
                                    ui.label(title).classes("rz-shot-title")
                                    if scores:
                                        ui.label(scores).classes("rz-shot-sub")


def build_dataset_tab(f: Form):
    """The datasets as cards with a few thumbnails each, Create dataset and Add a folder; a dataset opened below to
    add images and captions (upload from this computer, or copy a folder on the machine running RulzTrainer), edit
    its captions and pick references."""
    picker: DatasetPicker = f.w["datasets"]
    state = {"page": 0, "images": [], "busy": False, "open": None, "sync": False}
    class LookState:
        """The look filter's state per kind (face, body); item access reads and writes the kind being shown."""

        def __init__(self):
            self.kind = "face"
            self.by_kind = {k: self.fresh() for k in lookfilter.KINDS}
            self.shared = {"busy": False, "progress": "", "excluded": 0, "all": False}

        @staticmethod
        def fresh():
            return {"baselines": [], "scores": {}, "marks": set(), "cutoff": None, "median": None}

        def _bucket(self, key):
            return self.shared if key in self.shared else self.by_kind[self.kind]

        def __getitem__(self, key):
            return self._bucket(key)[key]

        def __setitem__(self, key, value):
            self._bucket(key)[key] = value

        def update(self, **kw):
            for k, v in kw.items():
                self[k] = v

        def of(self, kind):
            return self.by_kind[kind]

    look = LookState()
    native = _native_dialogs
    show_full = image_viewer()
    root_label = _relative(str(DATASETS_ROOT))

    with ui.dialog() as create_dlg, ui.card().classes("w-[28rem] max-w-full"):
        ui.label("Create dataset").classes("rz-setting-title")
        name_in = ui.input("Name", placeholder="for example: anna full body").classes("w-full")
        name_err = ui.label("").classes("text-xs text-negative")
        hint(f"A new, empty folder in {root_label}/ on the machine running RulzTrainer.")
        with ui.row().classes("w-full justify-end gap-2"):
            ui.button("Cancel", on_click=create_dlg.close).props("flat")
            create_btn = ui.button("Create", icon="add").props("unelevated")

    with card():
        with ui.row().classes("w-full items-center gap-2"):
            ui.label("Datasets").classes("rz-setting-title")
            ui.space()
            ui.button("Create dataset", icon="add",
                      on_click=lambda: (name_err.set_text(""), create_dlg.open())).props("unelevated")
            add_btn = ui.button("Add a folder", icon="create_new_folder").props("flat") \
                .tooltip("A dataset folder you already have on the machine running RulzTrainer: it joins the list")
            ui.button(icon="refresh", on_click=lambda: refresh_cards()).props("flat round dense").tooltip("Refresh")
        hint(f"New datasets go in {root_label}/ on the machine running RulzTrainer. Click a dataset to add images and "
             "write captions. Tick the ones to train on with Train on this dataset, or on the Train tab (Dataset "
             "section), where each also gets its own resolutions and repeats.")
        cards = ui.element("div").classes("rz-ds-grid")

    editor = ui.column().classes("w-full gap-3")
    editor.visible = False
    with editor:
        with card():
            with ui.row().classes("w-full items-center gap-2"):
                ui.icon("photo_library").classes("text-xl")
                ed_title = ui.label("").classes("rz-setting-title")
                ed_path = ui.label("").classes("rz-hint")
                ui.space()
                use_box = ui.checkbox("Train on this dataset").tooltip("Ticks it on the Train tab (Dataset section)")
                copy_btn = ui.button("Copy from a folder", icon="drive_folder_upload").props("flat") \
                    .tooltip("Copy the images and captions of a folder on the machine running RulzTrainer into this "
                             "dataset (sub-folders kept; files it already has are left alone)")
                remove_btn = ui.button("Remove from list", icon="playlist_remove").props("flat") \
                    .tooltip("Takes this folder off the list; its files stay where they are")
                ui.button(icon="close", on_click=lambda: close_dataset()).props("flat round dense").tooltip("Close")
            with ui.row().classes("w-full items-center gap-2"):
                spinner = ui.spinner(size="sm")
                spinner.visible = False
                info = ui.label("").classes("text-sm")
            hint("Upload sends images and their .txt or .json captions from the computer this browser runs on to the "
                 "machine running RulzTrainer; on the same machine that is a plain copy. A file with the same name is "
                 "replaced. Click a picture to see it full size. Captions are saved to the image's .txt file as you "
                 "type. Face ref / Body ref pick the images that character LoRAs are scored against.")
            hint("Ideogram 4 was trained on structured JSON captions only: a same-name .json caption in its schema "
                 "(github.com/ideogram-oss/ideogram4, docs/prompting.md) trains it best and wins over the .txt. A "
                 "plain .txt caption is wrapped into the smallest valid JSON caption, and the trigger word goes into "
                 "its subject (object) element. Long JSON captions can pass 1024 tokens: raise Max caption tokens "
                 "(Train tab) to 2048, Ideogram's limit, so they are not cut. Each cached caption keeps 104 KB per "
                 "token (13 text encoder layers): about 4 MB for a short wrapped caption, 17-110 MB for JSON "
                 "captions of 165-1077 tokens like Ideogram's examples.")
        with section("Look consistency filter", "Find the pictures whose face drifts from the look you want",
                     "face_retouching_natural", open_=False):
            hint("Tick Baseline on a few pictures below that nail the look (3 is a good number), then Scan: every "
                 "picture's face is compared with theirs by the face model the likeness probes use, worst first. "
                 "0.45 and up is a solid match, 0.30 to 0.45 the same person drifting, under 0.30 likely off-look or "
                 "someone else. Auto-suggest marks the outliers (below the median minus 1.5 times the spread, never "
                 "under 0.25). Move marked out puts them and their captions in excluded_by_look/ inside the dataset, "
                 "which never trains; nothing is deleted, and Move all back restores them. The scores are saved with "
                 "the dataset, for the Train tab's Warm up look outliers. Body outliers: the same with the body (neck to "
                 "mid-thigh, cut out and turned upright, as the likeness probes score it) instead of the face, so a "
                 "different build, skin, hair or outfit than your body baselines shows; pictures without that part of "
                 "the body in the frame get no score. It runs on the Body score device (Identity similarity).")
            ui.toggle({"face": "Face outliers", "body": "Body outliers"}, value="face",
                                  on_change=lambda e: look_set_kind(e.value)).props("no-caps dense")
            with ui.row().classes("w-full items-center gap-2"):
                look_base_label = ui.label("").classes("text-sm")
                ui.space()
                look_scan_btn = ui.button("Scan", icon="search", on_click=lambda: look_scan()).props("unelevated")
                look_suggest_btn = ui.button("Auto-suggest", icon="auto_fix_high",
                                             on_click=lambda: look_suggest()).props("flat")
                ui.button("Clear marks", icon="clear_all", on_click=lambda: look_clear()).props("flat")
                look_move_btn = ui.button("Move marked out", icon="drive_file_move",
                                          on_click=lambda: look_move_dlg.open()).props("flat color=negative")
                look_back_btn = ui.button("Move all back", icon="undo", on_click=lambda: look_back()).props("flat")
            look_info = ui.label("").classes("text-sm")
            look_grid = ui.element("div").classes("rz-look-grid w-full")
            look_more = ui.button("", on_click=lambda: look_toggle_all()).props("flat dense")
            look_more.visible = False
        with ui.dialog() as look_move_dlg, ui.card().classes("w-[28rem] max-w-full"):
            look_move_title = ui.label("").classes("rz-setting-title")
            ui.label("They move with their captions into excluded_by_look/ inside the dataset, which never trains. "
                     "Nothing is deleted: Move all back restores them.").classes("rz-hint")
            with ui.row().classes("w-full justify-end gap-2"):
                ui.button("Cancel", on_click=look_move_dlg.close).props("flat")
                ui.button("Move them out", icon="drive_file_move", color="negative",
                          on_click=lambda: look_move()).props("unelevated")
        with ui.row().classes("w-full items-center gap-3"):
            upload = ui.upload(label="Upload images and captions (drop files here)", multiple=True, auto_upload=True,
                               on_multi_upload=lambda e: on_upload(e)) \
                .props(f"accept={','.join(sorted(DATASET_FILE_EXTS))} flat bordered color=transparent") \
                .classes("w-[36rem] max-w-full")
            ui.space()
            with ui.row().classes("items-center gap-1 no-wrap"):
                ui.button(icon="chevron_left", on_click=lambda: page(-1)).props("flat dense round")
                page_label = ui.label("").classes("text-sm")
                ui.button(icon="chevron_right", on_click=lambda: page(1)).props("flat dense round")
                columns = ui.select([2, 3, 4, 5, 6], value=4, label="Columns").classes("w-28 ml-2")
        grid = ui.grid(columns=4).classes("w-full gap-3")

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

    async def on_upload(e):
        """Every file of an upload is saved before the gallery refreshes: NiceGUI runs each on_upload handler as a
        background task and fires on_multi_upload at once, so saving there showed only the files done by then."""
        upload.reset()
        if state["open"] is None:
            return
        for file in e.files:
            target = upload_target(Path(state["open"]), file.name)
            if target is None:
                ui.notify(f"Skipped {Path(file.name).name}: not an image or a .txt / .json caption", type="warning")
                continue
            await file.save(target)  # streamed to disk
        await refresh()
        await refresh_cards()

    async def render():
        imgs = state["images"]
        n_pages = max(1, (len(imgs) + PAGE_SIZE - 1) // PAGE_SIZE)
        state["page"] = min(state["page"], n_pages - 1)
        page_label.text = f"page {state['page'] + 1} / {n_pages}"
        start = state["page"] * PAGE_SIZE
        page_imgs = imgs[start:start + PAGE_SIZE]
        spinner.visible = True
        thumbs = await run.io_bound(make_thumbs, page_imgs, Path(state["open"]))
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
                    ui.image(image_url(thumb)).props("fit=cover ratio=1").classes("w-full cursor-pointer") \
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
                            .classes("text-xs").tooltip("Use this image's body as a similarity reference")
                        ui.checkbox("Baseline", value=str(img) in look["baselines"],
                                    on_change=lambda e, p=img: look_toggle_base(p, e.value)).props("dense") \
                            .classes("text-xs").tooltip("A baseline of the look consistency filter: a picture that "
                                                        "nails the look")
                        for kind in lookfilter.KINDS:
                            st = look.of(kind)
                            sc = st["scores"].get(str(img), "unscanned")
                            if sc != "unscanned":
                                vd = lookfilter.verdict(sc, kind, st["median"], st["cutoff"])
                                word = "body" if kind == "body" else "look"
                                ui.badge(f"no {kind}" if sc is None else f"{word} {sc:.2f}", color=LOOK_COLORS[vd]) \
                                    .props("dense").tooltip(f"Look consistency ({kind}): {vd}")
                        status_label = ui.label("" if cap.is_file() else "no caption yet").classes("text-xs text-grey")
                    ui.textarea(value=text, placeholder="type a caption; it is saved to the .txt automatically",
                                on_change=lambda e, p=img, sl=status_label: save_caption(p, e.value, sl)) \
                        .props("dense outlined autogrow").classes("w-full px-2 pb-2 text-xs")

    # ---------------------------------------------------------------- look consistency filter
    def look_load(path: str):
        for kind in lookfilter.KINDS:
            st = look.by_kind[kind] = look.fresh()
            data = lookfilter.load_scores(path, kind)
            if data:
                st.update(baselines=[b for b in data["baselines"] if Path(b).is_file()], scores=data["scores"],
                          cutoff=data.get("cutoff"), median=data.get("median"))
        look.shared.update(all=False, progress="")
        look["excluded"] = len(lookfilter.excluded_images(path))

    async def look_set_kind(kind: str):
        look.kind = kind
        look["all"] = False
        await look_render()
        await render()

    def look_toggle_base(p: Path, on: bool):
        bases = [b for b in look["baselines"] if b != str(p)]
        look["baselines"] = bases + ([str(p)] if on else [])
        look["marks"].discard(str(p))
        look_status()

    def look_status():
        n = len(look["baselines"])
        what = "body" if look.kind == "body" else "look"
        look_base_label.text = (f"{n} {what} baseline{'s' if n != 1 else ''} picked" + (
            ": " + ", ".join(Path(b).name for b in look["baselines"]) if n else
            (" (tick Baseline on the pictures that show the body you want: neck to mid-thigh in the frame)"
             if look.kind == "body" else " (tick Baseline on the pictures that nail the look)")))
        look_scan_btn.enabled = n > 0 and not look["busy"]
        look_suggest_btn.enabled = look["cutoff"] is not None and not look["busy"]
        look_move_btn.text = f"Move marked out ({len(look['marks'])})"
        look_move_btn.enabled = bool(look["marks"]) and not look["busy"]
        look_back_btn.text = f"Move all back ({look['excluded']})"
        look_back_btn.visible = look["excluded"] > 0
        look_move_title.text = f"Move {len(look['marks'])} picture(s) out of the dataset?"
        if look["busy"]:
            look_info.text = look["progress"]
        elif look["scores"]:
            live = {str(p) for p in state["images"]}  # pictures moved out since the scan no longer count
            here = {p: v for p, v in look["scores"].items() if p in live}
            scored = [v for v in here.values() if v is not None]
            fence = (f"; fence {look['cutoff']:.2f} (median {look['median']:.2f})" if look["cutoff"] is not None else
                     f"; no fence (needs {lookfilter.MIN_FACES} scored pictures)")
            has = "a body" if look.kind == "body" else "a face"
            look_info.text = (f"{len(here)} pictures scanned, {len(scored)} with {has}" + fence +
                              f"; {len(lookfilter.suggest(here, look['baselines'], look.kind))} under it")
        else:
            look_info.text = "Not scanned yet."

    async def look_render():
        look_status()
        live = {str(p) for p in state["images"]}
        rows = sorted(((p, s) for p, s in look["scores"].items() if p in live),
                      key=lambda kv: (kv[1] is None, kv[1] if kv[1] is not None else 0.0))  # worst first, no face last
        shown = rows if look["all"] else rows[:LOOK_SHOWN]
        look_more.visible = len(rows) > LOOK_SHOWN
        look_more.text = (f"Show the worst {LOOK_SHOWN} only" if look["all"] else f"Show all {len(rows)}")
        thumbs = await run.io_bound(make_thumbs, [Path(p) for p, _ in shown], Path(state["open"])) if shown else []
        look_grid.clear()
        with look_grid:
            for (p, sc), thumb in zip(shown, thumbs):
                vd = lookfilter.verdict(sc, look.kind, look["median"], look["cutoff"])
                base = p in look["baselines"]
                none = "no body" if look.kind == "body" else "no face"
                with ui.card().tight().classes("rz-thumb rz-look-card"):
                    ui.image(image_url(thumb)).props("fit=cover ratio=1").classes("w-full cursor-pointer") \
                        .on("click", lambda _, q=Path(p): show_full(q)).tooltip("Click to view full size")
                    with ui.column().classes("gap-0 px-2 py-1 w-full min-w-0"):
                        ui.label(Path(p).name).classes("text-xs truncate w-full")
                        with ui.row().classes("items-center gap-1 no-wrap"):
                            ui.badge(none if sc is None else f"{sc:.2f}", color=LOOK_COLORS[vd]).props("dense")
                            if base:
                                ui.badge("baseline", color="primary").props("dense outline")
                            else:
                                ui.checkbox("move out", value=p in look["marks"],
                                            on_change=lambda e, q=p: look_mark(q, e.value)) \
                                    .props("dense").classes("text-xs")

    def look_mark(p: str, on: bool):
        (look["marks"].add if on else look["marks"].discard)(p)
        look_status()

    async def look_toggle_all():
        look["all"] = not look["all"]
        await look_render()

    async def look_scan():
        if look["busy"] or not look["baselines"] or state["open"] is None:
            return
        folder, files, bases = state["open"], [str(p) for p in state["images"]], list(look["baselines"])
        kind = look.kind
        look.update(busy=True, progress="Loading the body model..." if kind == "body" else "Loading the face model...")
        look_status()

        def progress(i, n):
            look["progress"] = f"Scanning {'bodies' if kind == 'body' else 'faces'}: {i} of {n}"

        def work():
            note = lambda t, f=None: look.__setitem__("progress", t)  # noqa: E731
            if kind == "body":
                device = str(f.w["body_device"].value or "cpu")
                embed = lookfilter.body_embedder("models", device, progress=note)
            else:
                embed = lookfilter.face_embedder(progress=note)
            scores = lookfilter.score_images(files, bases, embed, progress)
            lookfilter.save_scores(folder, bases, scores, kind)
            return scores

        timer = ui.timer(0.5, look_status)
        try:
            scores = await run.io_bound(work)
        except Exception as e:  # noqa: BLE001
            look["busy"] = False
            timer.cancel()
            look_status()
            ui.notify(f"Look scan failed: {e}", type="negative", multi_line=True)
            return
        timer.cancel()
        med, cut = lookfilter.fence(scores, kind)
        look.by_kind[kind].update(scores=scores, cutoff=cut, median=med, marks=set())
        look["busy"] = False
        await look_render()
        await render()
        ui.notify(f"Scanned {len(scores)} pictures; saved {lookfilter.scores_file(kind)} in the dataset",
                  type="positive")

    async def look_suggest():
        look["marks"] = lookfilter.suggest(look["scores"], look["baselines"], look.kind)
        await look_render()
        if not look["marks"]:
            ui.notify("Nothing is under the fence")

    async def look_clear():
        look["marks"] = set()
        await look_render()

    async def look_move():
        look_move_dlg.close()
        folder, marks = state["open"], sorted(look["marks"])
        moved, errors = await run.io_bound(lookfilter.move_out, folder, marks)
        for p in moved:  # a reference that left the dataset stops being one
            for key in ("face_refs", "body_refs"):
                toggle_ref(key, Path(p), False)
        look["marks"] -= set(moved)
        look["excluded"] = len(await run.io_bound(lookfilter.excluded_images, folder))
        await refresh()
        await refresh_cards()
        ui.notify(f"Moved {len(moved)} picture(s) to {lookfilter.EXCLUDED_DIR}/" +
                  (f"; {len(errors)} failed: {errors[0]}" if errors else ""),
                  type="warning" if errors else "positive", multi_line=True)

    async def look_back():
        folder = state["open"]
        moved, errors = await run.io_bound(lookfilter.move_back, folder)
        look["excluded"] = len(await run.io_bound(lookfilter.excluded_images, folder))
        await refresh()
        await refresh_cards()
        ui.notify(f"Moved {len(moved)} picture(s) back" + (f"; {len(errors)} failed: {errors[0]}" if errors else ""),
                  type="warning" if errors else "positive", multi_line=True)

    async def refresh():
        if state["busy"] or state["open"] is None:
            return
        state["busy"] = True
        try:
            state["images"] = await run.io_bound(list_dataset_images, state["open"])
            missing = sum(1 for p in state["images"] if not _captioned(p))
            info.text = (f"{images_text(len(state['images']))}, {missing} without caption" if state["images"]
                         else "No images yet: upload some, or copy a folder.")
            await render()
            await look_render()
        finally:
            state["busy"] = False

    async def page(delta: int):
        state["page"] = max(0, state["page"] + delta)
        await render()

    async def refresh_cards():
        paths = await run.io_bound(known_datasets, picker.selected)
        picker.refresh(paths)
        data = await run.io_bound(dataset_cards, paths)
        picker.set_counts({d["path"]: (d["images"], d["captioned"]) for d in data})
        chosen = {DatasetPicker.key(p) for p in picker.selected}
        cards.clear()
        with cards:
            if not data:
                ui.label("No datasets yet: Create dataset makes one.").classes("rz-hint")
            for d in data:
                opened = state["open"] is not None and _same_path(d["path"], state["open"])
                with ui.element("div").classes("rz-ds-card cursor-pointer" + (" rz-ds-open" if opened else "")) as c:
                    if d["thumbs"]:
                        with ui.element("div").classes("rz-ds-thumbs"):
                            for t in d["thumbs"]:
                                ui.image(image_url(t)).props("fit=cover ratio=1")
                    else:
                        ui.label("no images yet").classes("rz-ds-empty")
                    with ui.column().classes("gap-0 px-3 py-2 w-full min-w-0"):
                        with ui.row().classes("items-center gap-2 no-wrap w-full"):
                            ui.label(d["name"]).classes("font-medium truncate")
                            if DatasetPicker.key(d["path"]) in chosen:
                                ui.badge("training", color="red").props("dense")
                        ui.label(f"{images_text(d['images'])}, {d['captioned']} captioned") \
                            .classes("text-xs rz-hint")
                        ui.label(d["path"]).classes("text-xs rz-hint truncate w-full")
                c.on("click", lambda _, p=d["path"]: open_dataset(p))

    async def open_dataset(path: str):
        state.update(open=path, page=0, images=[])
        ed_title.text = Path(path).name or path
        ed_path.text = path
        state["sync"] = True
        try:
            use_box.value = any(_same_path(path, p) for p in picker.selected)
        finally:
            state["sync"] = False
        remove_btn.visible = any(_same_path(path, p) for p in registered_datasets())
        editor.visible = True
        await run.io_bound(look_load, path)
        await refresh()
        await refresh_cards()

    async def close_dataset():
        state["open"] = None
        editor.visible = False
        grid.clear()
        await refresh_cards()

    async def on_use(e):
        if state["sync"] or state["open"] is None:
            return
        rest = [p for p in picker.selected if not _same_path(p, state["open"])]
        picker.selected = rest + ([state["open"]] if e.value else [])
        await refresh_cards()

    async def do_create():
        err = dataset_name_error(name_in.value)
        if err:
            name_err.text = err
            return
        try:
            path = create_dataset(name_in.value)
        except (OSError, ValueError) as e:
            name_err.text = str(e)
            return
        create_dlg.close()
        name_in.value = ""
        rel = _relative(str(path))
        await refresh_cards()
        await open_dataset(rel)
        ui.notify(f"Created {rel}", type="positive")

    async def add_folder():
        picked = await pick_path("dir", "", native=native)
        if not picked:
            return
        try:
            register_dataset(_relative(picked))
        except ValueError as e:
            ui.notify(str(e), type="warning")
            return
        await refresh_cards()
        await open_dataset(_relative(picked))

    async def copy_folder():
        if state["open"] is None:
            return
        picked = await pick_path("dir", "", native=native)
        if not picked:
            return
        spinner.visible = True
        try:
            res = await run.io_bound(copy_into_dataset, picked, state["open"])
        except (OSError, ValueError) as e:
            ui.notify(f"Could not copy: {e}", type="negative")
            return
        finally:
            spinner.visible = False
        ui.notify(f"Copied {res['copied']} files"
                  + (f"; {res['kept']} the dataset already had were left alone" if res["kept"] else "")
                  + (f"; {res['ignored']} other files skipped" if res["ignored"] else ""), type="positive")
        await refresh()
        await refresh_cards()

    async def remove_from_list():
        if state["open"] is None:
            return
        gone = state["open"]
        unregister_dataset(gone)
        picker.selected = [p for p in picker.selected if not _same_path(p, gone)]
        await close_dataset()
        ui.notify(f"{gone} is off the list; its files are untouched")

    create_btn.on_click(do_create)
    name_in.on("keydown.enter", do_create)
    add_btn.on_click(add_folder)
    copy_btn.on_click(copy_folder)
    remove_btn.on_click(remove_from_list)
    use_box.on_value_change(on_use)
    columns.on_value_change(lambda e: render())
    ui.timer(0.1, refresh_cards, once=True)


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
        names = race_names(sd.parent if sd else None)
        grid.clear()
        if names:  # a race: the same generation of every configuration side by side
            grid.style("grid-template-columns: minmax(0, 1fr)")
            with grid:
                for step, rest, members in group_by_generation([str(p) for p in files[:120]], names):
                    ui.label(generation_label(step, rest)).classes("rz-subhead")
                    with ui.row().classes("w-full gap-3 no-wrap"):
                        for who, path in members:
                            with ui.card().tight().classes("rz-thumb").style("flex: 1 1 0; min-width: 0"):
                                ui.image(image_url(path)).props("fit=contain") \
                                    .classes("w-full h-64 rz-imgbg cursor-pointer") \
                                    .on("click", lambda _, q=Path(path): show_full(q)) \
                                    .tooltip("Click to view full size")
                                sc = scores.get(Path(path).name)
                                ui.label((who or "winner") + (score_caption(sc) if sc is not None else "")) \
                                    .classes("text-xs px-2 py-1")
            return
        with grid:
            for p in files[:120]:
                with ui.card().tight().classes("w-full rz-thumb"):
                    shown = clip_or_still(p)  # a video model's sample: its clip plays in place of the still
                    ui.image(image_url(shown)).props("fit=contain").classes("w-full h-64 rz-imgbg cursor-pointer") \
                        .on("click", lambda _, q=shown: show_full(q)).tooltip("Click to view full size")
                    sc = scores.get(p.name)
                    label = p.name + (score_caption(sc) if sc is not None else "")
                    ui.label(label).classes("text-xs px-2 py-1")

    columns.on_value_change(lambda e: (state.__setitem__("files", []), refresh()))
    ui.timer(3.0, while_connected(refresh))
    refresh()


MAP_HINTS = {
    "learned": "Green: the model got better on this part of the picture since the first capture. Magenta: it got "
               "worse. Read this one first; magenta after green means the LoRA is overwriting something it had right "
               "(an earlier save may be better).",
    "loss": "Where the model still misses the picture: red most, then yellow, blue least. Fine detail always looks "
            "warmer, so watch how it changes. A face that stays warm while the rest cools is slow likeness: check the "
            "face is large enough in the photos.",
    "effect": "Where the LoRA changes the prediction. For a person it should sit on the face and body; spreading onto "
              "a recurring background, jacket or prop means the LoRA is learning that too (vary it, caption it, or "
              "use the region-weighted loss).",
    "effect_off": "The same picture and noise with the trigger word taken out of the caption: where the LoRA still "
                  "changes the prediction. Little here means the LoRA is tied to its trigger; as much as with the "
                  "trigger means it acts whatever the caption says (it has bled into the model's idea of a person).",
    "attn": "How much attention each area gives the trigger word. It should sit on the subject; on a logo, watermark "
            "or object means the trigger is tied to that. The colours count from the quietest area up, so they show "
            "where the word gets more attention than elsewhere, not how much it gets in all.",
}
LEARNED_AS_HINTS = {
    "share": " Shown as a share of each part's own starting error: how far along each part is, whatever it started "
             "at. A part with little to learn (a plain wall, a plain sweater) shows here; by amount it stays pale.",
    "fixed": " Shown as a share of each part's own starting error on one scale for every run: full colour is half of "
             "the starting error gone (or come back). Two runs' maps compare by eye.",
}


def map_picture(data: dict, probe: dict):
    """A map picture as training saw it (probe_<n>.jpg), or None."""
    from PIL import Image

    try:
        with Image.open(Path(data["dir"]) / probe["image"]) as im:
            return im.convert("RGB")
    except (OSError, KeyError):
        return None


def map_data_url(img, quality: int = 88) -> str:
    import base64
    import io

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def build_maps_tab(f: Form):
    """Training maps: heatmaps over a few training pictures, capture by capture (rulztrainer/trainmaps.py)."""
    from rulztrainer import trainmaps as TM

    state = {"sig": None, "data": None, "view": None, "series": [], "top": 1.0, "pics": {}}
    with card():
        hint("Turn on Training maps (Train tab, Likeness section) before a run. A few training pictures are checked before "
             "training and about once an epoch at three fixed noise levels, so only the LoRA changes between "
             "captures. They only observe: training is the same with them on.")
        with fields():
            folder = ui.input("Job folder (output/<job>)", value="").classes("w-96 max-w-full")
            native = _native_dialogs

            async def browse():
                picked = await pick_path("dir", folder.value or str(ROOT / "output"), native=native)
                if picked:
                    follow.value = False
                    folder.value = _relative(picked)
                    refresh(force=True)

            ui.button(icon="folder_open", on_click=browse).props("flat dense round").tooltip(BROWSE_TIP[native])
            follow = ui.switch("Follow current job", value=True)
            info = ui.label("").classes("rz-hint")
        with fields():
            pick = ui.select({}, label="Picture").classes("w-72 max-w-full")
            kind = ui.select({k: v for k, v in TM.MAP_KINDS.items()}, value="learned", label="Map").classes("w-64")
            level = ui.select({"all": "All levels", **{f"{s:g}": TM.sigma_label(f"{s:g}") for s in TM.SIGMAS}},
                              value="all", label="Noise").classes("w-52")
            learned_as = ui.select(dict(TM.LEARNED_AS), value="amount", label="Learned, measured as").classes("w-72")
            learned_as.bind_visibility_from(kind, "value", backward=lambda v: v == "learned")
        with fields():
            play = ui.button(icon="play_arrow").props("flat dense round").tooltip("Play through the captures")
            at = ui.slider(min=0, max=0, step=1, value=0).classes("w-80 max-w-full")
            at_label = ui.label("").classes("text-sm whitespace-nowrap")
        with fields():
            smooth = ui.switch("Smooth", value=True)
            colour = ui.switch("Colour picture", value=False)
            ui.label("Map opacity").classes("text-sm")
            opacity = ui.slider(min=0, max=1, step=0.05, value=1.0).classes("w-40")
        meaning = ui.label("").classes("rz-hint")
    view = ui.image("").props("fit=contain").classes("w-full max-w-3xl rz-imgbg")
    view.set_visibility(False)

    def job_dir() -> Path | None:
        if follow.value:
            jd = _job_dir or (Path(f.w["output_dir"].value or "output") / (f.w["job_name"].value or "my_lora"))
            rel = _relative(str(jd))
            if folder.value != rel:
                folder.value = rel
        return Path(folder.value) if folder.value else None

    def kind_options(data: dict) -> dict:
        opts = {k: v for k, v in TM.MAP_KINDS.items()}
        words = data.get("words") or []
        if len(words) > 1:  # each word on its own besides all of them together
            opts["attn"] = "Trigger word: all"
            for w in words:
                opts[f"attn:{w}"] = f"Trigger word: {w}"
        return opts

    def rebuild():
        """The series for the chosen picture / map / noise level, and its one colour scale."""
        data = state["data"]
        if not data or not pick.value:
            state["series"], state["view"] = [], None
            return
        k, word = (kind.value or "learned"), ""
        if k.startswith("attn:"):
            k, word = "attn", k[5:]
        how = (learned_as.value or "amount") if k == "learned" else "amount"
        state["series"] = TM.series(data, pick.value, k, level.value or "all", word, how)
        state["top"] = TM.scale(state["series"], k, how)
        state["view"], state["how"] = k, how

    def draw():
        data, caps = state["data"], (state["data"] or {}).get("captures") or []
        if not data or not caps:
            view.set_source("")
            view.set_visibility(False)
            at_label.text = ""
            return
        i = max(0, min(int(at.value or 0), len(caps) - 1))
        at_label.text = f"{TM.capture_label(caps[i])} ({i + 1} of {len(caps)})"
        probe = next((p for p in data["probes"] if p["id"] == pick.value), None)
        if probe is None:
            return
        pic = state["pics"].get(probe["id"])
        if pic is None:
            pic = state["pics"][probe["id"]] = map_picture(data, probe)
        if pic is None:
            view.set_source("")
            view.set_visibility(False)
            info.text = f"{probe['image']} is missing from {data['dir']}"
            return
        a = state["series"][i] if i < len(state["series"]) else None
        meaning.text = MAP_HINTS.get(state["view"] or "", "") + LEARNED_AS_HINTS.get(state.get("how") or "", "")
        if a is None and state["view"] == "attn":
            meaning.text = "No trigger-word map for this picture: its caption doesn't name the trigger."
        elif a is None and state["view"] == "effect" and i == 0:
            meaning.text = "Before training the LoRA changes nothing yet; this map starts at the next capture."
        img = TM.render_overlay(pic, a, state["view"] or "loss", state["top"], float(opacity.value or 0),
                                bool(smooth.value), bool(colour.value))
        view.set_source(map_data_url(img))
        view.set_visibility(True)

    def refresh(force: bool = False):
        jd = job_dir()
        d = TM.maps_dir(jd) if jd else None
        sig = None
        if d is not None:
            sig = tuple((p.name, p.stat().st_mtime_ns) for p in (d / "probes.json", d / "maps.jsonl") if p.exists())
        if sig == state["sig"] and not force:
            return
        state["sig"] = sig
        was_last = bool(state["data"]) and int(at.value or 0) >= len(state["data"]["captures"]) - 1
        data = TM.read(jd) if jd else None
        if data is None or not data["probes"]:
            state["data"], state["series"] = None, []
            info.text = f"No training maps in {jd}" if jd else "No job selected"
            pick.set_options({})
            view.set_source("")
            view.set_visibility(False)
            at_label.text = ""
            return
        if state["data"] is None or state["data"].get("dir") != data["dir"]:
            state["pics"] = {}
        state["data"] = data
        info.text = f"{len(data['captures'])} capture(s) of {len(data['probes'])} picture(s) in {data['dir']}"
        pick.set_options({p["id"]: f"{int(p['id']) + 1}. {Path(p['source']).name}"
                          + (" (side-by-side pair)" if p.get("pair") else "") for p in data["probes"]})
        if pick.value not in {p["id"] for p in data["probes"]}:
            pick.value = data["probes"][0]["id"]
        opts = kind_options(data)
        kind.set_options(opts)
        if kind.value not in opts:
            kind.value = "learned"
        n = len(data["captures"])
        at.props["max"] = max(0, n - 1)  # (an observable dict: the browser gets the new end)
        if was_last or force or state["view"] is None:
            at.value = max(0, n - 1)  # auto-refresh: a new capture shows as it lands
        rebuild()
        draw()

    def advance():
        caps = (state["data"] or {}).get("captures") or []
        if not caps:
            return
        nxt = int(at.value or 0) + 1
        if nxt >= len(caps):
            nxt = 0
        at.value = nxt

    timer_play = ui.timer(0.8, advance, active=False)

    def toggle_play():
        timer_play.active = not timer_play.active
        play.props(f"icon={'pause' if timer_play.active else 'play_arrow'}")

    play.on_click(toggle_play)
    for wd in (pick, kind, level, learned_as):
        wd.on_value_change(lambda _: (rebuild(), draw()))
    for wd in (at, smooth, colour, opacity):
        wd.on_value_change(lambda _: draw())
    folder.on("keydown.enter", lambda _: (follow.set_value(False), refresh(force=True)))
    follow.on_value_change(lambda _: refresh(force=True))
    ui.timer(3.0, while_connected(refresh))
    refresh(force=True)


# ---------------------------------------------------------------- wizard

# names for the settings a racer may change (the rest show their config name)
RACE_LABELS = {
    "learning_rate": "Learning rate", "optimizer": "Optimizer", "weight_decay": "Weight decay",
    "lr_scheduler": "Learning rate schedule", "warmup_steps": "Warm-up steps", "lora_plus_ratio": "LoRA+ ratio",
    "automagic_min_lr": "Automagic min lr", "automagic_max_lr": "Automagic max lr",
    "automagic_polarity_history": "Automagic polarity history", "automagic_clip_threshold": "Automagic clip threshold",
    "automagic_groups": "Automagic lr groups", "automagic_sensitivity_quantile": "Automagic sensitivity quantile",
    "automagic_vote": "Automagic vote", "automagic_max_lr_change": "Automagic max lr change",
    "hull_rho": "Hull step", "hull_betas": "Hull betas", "hull_state_dtype": "Hull momentum dtype",
    "hull_clip": "Hull channel clip", "hull_factor_cap": "Hull factor cap", "hull_safety_cap": "Hull safety cap",
    "hull_gain_max": "Hull largest gain", "hull_ns_iters": "Hull matrix-root iterations",
    "hull_regions": "Hull body regions", "hull_gain_body": "Hull body gain",
    "hull_gain_background": "Hull background gain", "hull_ring_cells": "Hull outline width",
    "hull_face_safe": "Hull face-safe excess", "hull_retreat": "Hull excess retreat",
    "hull_damping": "Hull damping",
    "keel_gauge": "Keel gauge preconditioner", "keel_gauge_eps": "Keel gauge eps", "keel_cautious": "Keel cautious mask",
    "keel_pair_sigma": "Keel paired sigma", "keel_bands": "Keel noise bands", "keel_band_weights": "Keel band weights",
    "keel_consensus": "Keel consensus gate", "keel_consensus_power": "Keel consensus power",
    "keel_consensus_mode": "Keel consensus vote",
    "keel_lookahead": "Keel lookahead", "keel_lookahead_pullback": "Keel pull-back",
    "keel_lookahead_lr": "Keel pull lr factor", "keel_lookahead_grace": "Keel grace probes",
    "keel_face_gain": "Keel face gain", "keel_face_gain_max": "Keel face gain limit",
    "keel_face_gain_beta": "Keel face gain EMA", "keel_body_weight": "Keel body weight",
    "lora_ema": "Weight average", "timestep_sampling": "Timestep sampling", "loss_weighting": "Loss weighting",
    "loss_type": "Loss type",
    "region_weights": "Region-weighted loss", "face_loss_weight": "Face loss weight",
    "body_loss_weight": "Body loss weight", "background_loss_weight": "Background loss weight",
    "id_loss_weight": "Identity loss weight", "attn_localize": "Trigger attention on the subject",
    "attn_localize_weight": "Trigger attention weight", "id_align": "Identity alignment",
    "id_align_weight": "Identity alignment weight", "body_align": "Body alignment",
    "body_align_weight": "Body alignment weight", "likeness_curriculum": "Face/body curriculum",
    "likeness_curriculum_framing": "Curriculum: count lost framing",
    "training_maps_trigger_off": "Training maps: trigger-off pass",
    "likeness_lr_control": "Likeness lr control", "loss_watch": "Per-image loss watch",
    "region_watch": "Region watch", "region_class_weights": "Region class weights",
    "region_steer": "Region steering", "region_steer_regions": "Region steering: regions",
    "region_steer_cap": "Region steering: cap", "region_steer_ahead": "Region steering: ahead",
    "region_steer_boost": "Region steering: boost", "region_steer_prior": "Region steering: background prior",
    "region_steer_min": "Region steering: lowest weight", "region_steer_step": "Region steering: step",
    "region_steer_strength": "Region steering: strength", "region_steer_signal": "Region steering: reading",
    "region_steer_holdout": "Region steering: held-out gap", "region_sweep_every": "Region sweep every (epochs)",
    "region_steer_subject_ahead": "Region steering: subject ahead",
    "region_sweep_pictures": "Region sweep: pictures per sweep", "holdout_images": "Held-out photos",
    "holdout_track_best": "Held-out best checkpoint", "holdout_patience": "Held-out stop patience",
    "per_image_lr": "Per-image learning rate", "lora_rank": "LoRA rank", "lora_alpha": "LoRA alpha",
    "lora_targets": "LoRA layers", "lora_blocks": "LoRA blocks", "lora_dropout": "LoRA dropout",
    "network_type": "Adapter type", "lokr_factor": "LoKr factor", "oft_block_size": "OFT block size",
    "oft_constraint": "OFT constraint", "dora": "DoRA", "caption_dropout": "Caption dropout",
    "flip_augment": "Flip augment", "crop_augment": "Close-up crops", "crop_augment_weight": "Crop weight",
    "look_warmup": "Look warm-up", "selftrain": "Self-training", "selftrain_at": "Self-training at",
    "selftrain_images": "Self-training pictures", "selftrain_min_likeness": "Self-training likeness floor",
    "selftrain_share": "Self-training share", "selftrain_weight": "Self-training weight", "seed": "Seed",
    "copy_check": "Copy check (SSCD)", "copy_max": "Copy limit",
    "dop_weight": "DOP weight (LoRA Isolation)", "dop_every": "DOP every N steps (LoRA Isolation)",
    "int8_backward": "int8 backward", "nvfp4_backward": "nvfp4 backward (fp4 is faster on RTX 50)",
}


def near_duplicate_note(found: dict | None) -> str:
    """The Monitor's warning about training photos that nearly match each other; "" with none."""
    if not found:
        return ""
    shown = found.get("pairs") or []
    pairs = "; ".join(f"{a} and {b} ({score:.2f})" for a, b, score in shown)
    more = found.get("count", len(shown)) - len(shown)
    return (f"Near-duplicate photos above the copy limit {found.get('limit', 0):.2f}: {pairs}"
            + (f"; and {more} more" if more > 0 else "") + ". Consider removing one photo of each pair.")


def race_choice_options() -> dict[str, str]:
    """{setting: "group: name"} of everything a racer may change, for the wizard's picker."""
    from rulztrainer.race import RACE_KEYS

    return {k: f"{group}: {RACE_LABELS.get(k, k)}" for group, keys in RACE_KEYS.items() for k in keys}


def race_value_text(key: str, value) -> str:
    """A setting's value as a racer's text box shows it (the race rows' own notation)."""
    from rulztrainer import race

    return race.format_overrides({key: value}).split("=", 1)[1]


def race_review_line(cfg: TrainConfig) -> str:
    """The review's race row: the racers and what each changes, and how racers leave."""
    from rulztrainer import race

    rows = race.variant_configs(cfg)
    racers = "; ".join(f"{n}: {race.format_overrides(o) or 'the base setup'}" for n, _, o in rows)
    rules = []
    if cfg.race_follow:
        if cfg.race_pick:
            wait = (f"up to {cfg.race_pick_minutes:g} minutes, then the highest score leads"
                    if cfg.race_pick_minutes > 0 else "until you pick")
            return (f"{len(rows)} racers ({racers}); a relay where you pick on the Monitor, after each round, the racer "
                    f"everyone continues from (the run waits {wait}); the last round's pick is the run's LoRA")
        return (f"{len(rows)} racers ({racers}); a relay: after each round everyone continues from the round's "
                f"winner, and the last round's winner is the run's LoRA")
    if cfg.race_drop_lowest:
        rules.append("the lowest leaves after each round" + (f" (after {cfg.race_warmup_probes} warm-up rounds)"
                                                             if cfg.race_warmup_probes else "")
                     + (f" when a paired test on the probe seeds puts it below the leader with "
                        f"{cfg.race_drop_confidence:.0%} confidence (else after {cfg.race_max_close_rounds} rounds "
                        f"too close to call)" if cfg.race_drop_confidence > 0 else ""))
    else:
        rules.append("nobody is dropped for being behind")
    if cfg.race_patience_drop:
        rules.append(f"a racer leaves after {cfg.race_patience} probes without improvement"
                     + (" unless its likeness is rising again" if cfg.race_patience_rising else ""))
    if cfg.race_drop_drift > 0:
        rules.append(f"a racer whose probes leave their prompts' frame ({cfg.race_drop_drift:.0%} or more off-prompt, "
                     f"{cfg.race_drop_drift_probes} probes in a row) is dropped")
    if cfg.race_patience_drop or not cfg.race_drop_lowest:
        rules.append("the best score at any point wins")
    return f"{len(rows)} racers ({racers}); " + ", ".join(rules)


# Character / person: how much of the person the likeness covers. Face focus is the identity recipe as it is; the
# others move the scoring (face / body share of the combined score) and the loss weights (face region / person region)
# toward the face or toward the body.
CHARACTER_FOCUS = {
    "face_only": ("face", "Face only",
                  "The face alone. Probes render only the headshot (half the time), crops are face close-ups, and "
                  "the body is left to the prompt: no body probe, references or score."),
    "face_focus": ("face_retouching_natural", "Face focus",
                   "The face first, the body second. The face counts 3 against the person's 1.5 in the loss and 70% "
                   "of the likeness score (30% the body). Face and bust crops. The recipe as measured."),
    "face_body": ("accessibility_new", "Face and body",
                  "Both the same. Face and person each weigh 1.5 in the loss, and the likeness score is half face, "
                  "half body. Face and bust crops; the curriculum evens out whichever of the two lags."),
    "body_focus": ("directions_walk", "Body focus",
                   "The build first, the face second. The person counts 3 against the face's 1.5 in the loss and 70% "
                   "of the likeness score (30% the face)."),
    "body_only": ("emoji_people", "Body only",
                  "The build alone. Probes render only the full-body shot, there is no face reference or score, and "
                  "no close-up crops; the person counts 3 and the face 1 in the loss."),
}
FOCUS_NOTES = {
    "face_only": "face only: face 3 / person 1 / background 0.3, face crops, face probes only",
    "face_focus": "face focus: face 3 / person 1.5 / background 0.3, face and bust crops, score 70% face and 30% body",
    "face_body": "face and body: face 1.5 / person 1.5 / background 0.3, face and bust crops, score 50% face and "
                 "50% body",
    "body_focus": "body focus: face 1.5 / person 3 / background 0.3, face and bust crops, score 30% face and "
                  "70% body",
    "body_only": "body only: face 1 / person 3 / background 0.3, no crops, body probes only",
}


def focus_settings(focus: str, trigger: str = "") -> dict:
    """The settings a Character / person focus sets on top of the identity recipe: the probe prompts and references
    it uses (the ones it does not are emptied), the score weights, the crops, the loss weights and the curriculum."""
    face_p, body_p = PRESET_PROBE_PROMPTS["identity"]
    subject = trigger or "the character"
    out = {"probe_face_prompt": face_p.format(trigger=subject), "probe_body_prompt": body_p.format(trigger=subject),
           "crop_augment": "face+bust", "likeness_curriculum": True}
    weights = {"face_only": (1.0, 0.0, 3.0, 1.0), "face_focus": (0.7, 0.3, 3.0, 1.5),
               "face_body": (0.5, 0.5, 1.5, 1.5), "body_focus": (0.3, 0.7, 1.5, 3.0),
               "body_only": (0.0, 1.0, 1.0, 3.0)}[focus]
    out.update(face_weight=weights[0], body_weight=weights[1], face_loss_weight=weights[2],
               body_loss_weight=weights[3])
    if focus == "face_only":
        out.update(probe_body_prompt="", body_refs=[], crop_augment="face", likeness_curriculum=False)
    elif focus == "body_only":
        out.update(probe_face_prompt="", face_refs=[], crop_augment="none", likeness_curriculum=False)
    return out


def character_focus(cfg: TrainConfig) -> str:
    """Which focus a config is on: face only / body only when nothing scores the other, else by the weights of the
    combined score (equal: face and body)."""
    has_body = bool(cfg.probe_body_prompt.strip()) and cfg.body_weight > 0
    has_face = bool(cfg.probe_face_prompt.strip()) and cfg.face_weight > 0
    if not has_body:
        return "face_only"
    if not has_face:
        return "body_only"
    if abs(cfg.face_weight - cfg.body_weight) < 1e-6:
        return "face_body"
    return "face_focus" if cfg.face_weight > cfg.body_weight else "body_focus"


# Frozen DiT weights by family and quantization, in GB (what block swap moves into system RAM), and the memory a step
# needs besides them (activations, the LoRA, the VAE, the CUDA context).
MODEL_GB = {"moss_image": {"bf16": 10.0, "int8": 5.5, "nvfp4": 3.0}, "krea2": {"bf16": 25.0, "int8": 13.5, "nvfp4": 8.0},
            "fibo2": {"bf16": 16.0, "int8": 9.0, "nvfp4": 5.0},
            "qwen_image21": {"bf16": 14.2, "int8": 7.3, "nvfp4": 4.0},
            "flux3_klein": {"bf16": 9.7, "int8": 5.1, "nvfp4": 3.0}}  # Klein without its reference-image mode
# FLUX.2 Klein 9B: 9.08 B weights, bf16 16.9 GB; Fizgig holds it in about 9.6 GB fp8 and 5.6 GB 4-bit, the estimate here
MODEL_GB["flux2_klein"] = {"bf16": 16.9, "int8": 9.6, "nvfp4": 5.6}
# Ideogram 4: 9.29 B weights (the release's fp8 file, 9.29 GB): bf16 about 18.6 GB; Comfy-Org's int8_convrot file
# 9.58 GB, nvfp4_mixed 5.49 GB
MODEL_GB["ideogram4"] = {"bf16": 18.6, "int8": 9.6, "nvfp4": 5.5}
# MiniMax H3 pruned (Comfy-Org's file sizes: bf16 40.2 GB, int8_convrot 21.0 GB; nvfp4 is an estimate: the blocks at
# 4.5 bits per weight, 10.9 GB, and the bf16 token refiner, 1.5 GB)
MODEL_GB["minimax_h3"] = {"bf16": 40.2, "int8": 21.0, "nvfp4": 12.5}
# P-H3-Image: the same pruned transformer (PrunaAI/P-H3-Image-Comfy: bf16 40.5 GB, int8_convrot 21.2 GB)
MODEL_GB["p_h3_image"] = {"bf16": 40.5, "int8": 21.2, "nvfp4": 12.5}
# MiniMax-H3-Image: the same pruned transformer less the audio projections (Comfy-Org/MiniMax-H3-Image: bf16 40.2 GB,
# int8_convrot 21.0 GB; the 12.5 GB w4a8 file holds int8 weights once it is read)
MODEL_GB["minimax_h3_image"] = {"bf16": 40.2, "int8": 21.0, "nvfp4": 12.5}
# PAN 2: the diffusers folder's transformer is 67.6 GB (bf16 with fp32 norm projections); the converted files with
# those layers quantized too are 29.4 GB (PAN2_int8_convrot_full) and 16.6 GB (PAN2_nvfp4)
MODEL_GB["pan2"] = {"bf16": 67.6, "int8": 29.4, "nvfp4": 16.6}
STEP_HEADROOM_GB = 4.5


def block_swap_plan(cfg: TrainConfig, vram: float, quant: str) -> tuple[int, float]:
    """(blocks to swap, GB of weights they hold) so the model plus a step's memory fits in `vram` GB; 0 when it fits."""
    fam = cfg.arch_name()
    if quant == "auto":
        if cfg.model_source == "files" and fam != "flux2_klein":  # its float files are quantized for the memory
            quant = "bf16" if cfg.model_file_quant == "bf16" else "int8"
            quant = "nvfp4" if cfg.model_file_quant == "nvfp4_mixed" else quant  # Ideogram 4's 5.5 GB file
        else:
            quant = {"krea2": "int8" if vram >= 22 else "nvfp4", "fibo2": "int8" if vram < 22 else "bf16",
                     "qwen_image21": "bf16" if vram >= 24 else "int8" if vram >= 11 else "nvfp4",
                     "flux3_klein": "bf16" if vram >= 15 else "int8" if vram >= 9.5 else "nvfp4",
                     "flux2_klein": "bf16" if vram >= 30 else "int8" if vram >= 18 else "nvfp4",
                     "ideogram4": "bf16" if vram >= 30 else "int8" if vram >= 15 else "nvfp4"}.get(fam, "bf16")
    quant = {"int8_convrot": "int8", "none": "bf16"}.get(quant, quant)
    gb = MODEL_GB.get(fam, MODEL_GB["moss_image"]).get(quant, 10.0)
    blocks = get_arch(cfg).n_blocks or 28
    over = gb + STEP_HEADROOM_GB - vram
    if over <= 0:
        return 0, 0.0
    n = max(1, min(blocks - 2, math.ceil(blocks * over / gb)))
    return n, gb * n / blocks


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
    "Qwen-Image-2.1": ("", "Qwen-Image 2.1 (7B), text-to-image and editing in one model. Comfy-Org's single files "
                           "(int8, 7.3 GB, with the 9.4 GB Qwen3-VL-8B text encoder) train on one 16 GB card; its "
                           "LoRAs load in ComfyUI and diffusers, on Turbo too. Qwen Research License: non-commercial "
                           "unless Qwen grants more."),
    "Qwen-Image-2.1-Turbo": ("Not for training", "The 8-step distilled Qwen-Image 2.1. Train on Qwen-Image-2.1: its "
                                                 "LoRAs load onto Turbo, and the Turbo LoRA previews the same way."),
    "FLUX-3-Klein-Base": ("", "Black Forest Labs' FLUX 3 Image Klein (5.5B), the base model to train on. BFL's own "
                              "single files (int8, 5.8 GB, with the 4.9 GB Qwen3-VL-4B text encoder) train on one "
                              "16 GB card, bf16 too (9.7 GB). Gated: accept the FLUX Kommunity License on Hugging Face "
                              "first (non-commercial, plus commercial outputs under $5M revenue; read it before sharing "
                              "a LoRA). LoRAs load in ComfyUI and diffusers, on the distilled Klein too."),
    "FLUX-3-Klein": ("Not for training", "The 16-step distilled FLUX 3 Klein. Train on FLUX-3-Klein-Base and set this "
                                         "as the sampling model: a Base LoRA runs on it at 16 steps, CFG 1 (BFL: LoRA "
                                         "strength about 0.75)."),
    "FLUX-2-Klein-9B-Base": ("", "Black Forest Labs' FLUX.2 Klein 9B, the base model to train on: an 8.9 GB fp8 file "
                                 "(17 GB bf16) and the Qwen3-8B text encoder. Text-to-image LoRAs only here. Auto "
                                 "quantizes it for your GPU: nvfp4 below 18 GB, int8 up to 30. Gated: accept the FLUX "
                                 "Non-Commercial License on Hugging Face first (approved automatically; read it before "
                                 "sharing a LoRA). LoRAs load in ComfyUI and diffusers, on the distilled Klein too."),
    "FLUX-2-Klein-9B": ("Not for training", "The 4-step distilled FLUX.2 Klein 9B. Train on FLUX-2-Klein-9B-Base and set "
                                            "this as the sampling model: a Base LoRA runs on it at 4 steps, CFG 1."),
    "Ideogram-4": ("", "Ideogram 4 (9.3B), strong at typography and layout, trained on structured JSON captions. "
                       "Comfy-Org's single files (int8, 9.6 GB, with the 9.4 GB Qwen3-VL-8B text encoder) train on "
                       "one 16 GB card; its LoRAs load in ComfyUI and diffusers. Ideogram Non-Commercial Model "
                       "Agreement: non-commercial use only; a LoRA trained on it is a Model Derivative under the same "
                       "terms (share it with the agreement and its notice), and images made for revenue-generating "
                       "uses are not allowed either."),
    "MiniMax-H3-FL2VA": ("Experimental", "MiniMax H3 (33B video + audio), the FL2VA checkpoint (text / first-last "
                                         "frame to video): image LoRAs train as one-frame videos, as musubi-tuner "
                                         "does, for use in H3 video generation. Comfy-Org's pruned int8 file is "
                                         "21.0 GB: two 16 GB cards in model split, or one with block swap (64 GB of "
                                         "RAM advised); the Qwen3-VL-32B text encoder (27.1 GB int8, 15.7 GB NVFP4) "
                                         "caches the captions first. MiniMax H3 Community License: users in the "
                                         "USA, EU, UK and South Korea must apply to MiniMax first "
                                         "(platform.minimax.io/h3-license)."),
    "P-H3-Image": ("Experimental", "PrunaAI's P-H3-Image (33B): MiniMax H3 fine-tuned into an 8-step text-to-image "
                                   "model with no guidance, strong at photoreal and editorial looks. Trains on "
                                   "PrunaAI's ComfyUI files (int8 21.2 GB, the same size as H3's: two 16 GB cards in "
                                   "model split, or one with block swap) with a small Qwen3-VL-4B text encoder "
                                   "(5.1 GB), and ostris' P-H3-Image training adapter on by default. Text-to-image "
                                   "only here (no edit training). MiniMax H3 Community License: users in the USA, EU, "
                                   "UK and South Korea need a license from MiniMax first; products show 'Powered by "
                                   "MiniMax H3'."),
    "MiniMax-H3-Image": ("Experimental", "MiniMax's own image model (33B): the H3 transformer without its audio "
                                         "stream, made to render in 6 steps with no guidance, at 4 to 20 megapixels. "
                                         "Trains on Comfy-Org's files (int8 21.0 GB, the same size as H3's: two 16 GB "
                                         "cards in model split, or one with block swap) with H3's Qwen3-VL-32B text "
                                         "encoder (27.1 GB int8, 15.7 GB NVFP4; an H3 download is reused) and a small "
                                         "image VAE (1.2 GB), with ostris' MiniMax-H3-Image training adapter on by "
                                         "default. Text-to-image only here (no edit training). "
                                         "MiniMax-H3-Image Community License: it excludes the USA, EU, UK and South "
                                         "Korea unless MiniMax authorizes it (platform.minimax.io/license), which "
                                         "is also required above 1 million USD of annual revenue or to offer it as "
                                         "a hosted service."),
    "MiniMax-H3-Ref2VA": ("Not for training", "The reference-to-video H3 checkpoint (up to 9 images, 3 videos, 3 "
                                              "audio clips as references). Train on MiniMax-H3-FL2VA: this trainer "
                                              "trains text-only one-frame LoRAs, and a LoRA made on one H3 checkpoint "
                                              "degrades the other (diffusers' note)."),
}


def wizard_recommendations(cfg: TrainConfig, goal: str | None, gpus: list[dict], data: dict | None,
                           focus: str = "face_focus") -> dict[str, tuple]:
    """Recommended (value, reason) per setting key, from the goal (and a character's focus, a key of
    CHARACTER_FOCUS), the GPUs (nvidia-smi rows) and the dataset scan."""
    r: dict[str, tuple] = {}
    face_only, body_only = focus == "face_only", focus == "body_only"
    fibo = cfg.arch_name() == "fibo2"
    krea = cfg.arch_name() == "krea2"
    krea_turbo = krea and "krea-2-turbo" in (cfg.sample_model_path or "").lower()
    qwen = cfg.arch_name() == "qwen_image21"
    klein = cfg.arch_name() == "flux3_klein"
    klein_distilled = klein and get_arch(cfg).distilled_sample_model(cfg.sample_model_path or "")
    klein2 = cfg.arch_name() == "flux2_klein"
    klein2_distilled = klein2 and get_arch(cfg).distilled_sample_model(cfg.sample_model_path or "")
    ideo = cfg.arch_name() == "ideogram4"
    character = goal == "character"
    vram = min(g["total"] for g in gpus) / 1024 if gpus else None
    gpu_name = "your smallest GPU" if len(gpus) > 1 else "your GPU"
    # hardware
    if len(gpus) >= 2 and krea and vram is not None and vram < 22:
        r["multi_gpu"] = ("model_split", f"{len(gpus)} GPUs of {vram:.0f} GB: split, the 12B model trains in int8; a "
                                         "full copy on each card would only fit in nvfp4")
    elif len(gpus) >= 2 and (cfg.finetune or cfg.block_swap):
        r["multi_gpu"] = ("off", "a full fine-tune cannot run data parallel (its trained weights are not a LoRA to "
                                 "average)" if cfg.finetune else "block swap runs on one GPU")
    elif len(gpus) >= 2:
        r["multi_gpu"] = ("data_parallel", f"{len(gpus)} GPUs found: a model copy on each trains about 1.7-2x faster")
    if len(gpus) >= 2 and r["multi_gpu"][0] == "data_parallel":
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
        elif qwen:
            r["transformer_quant"] = ("auto", "Qwen-Image 2.1 picks for the memory it runs in: bf16 (14.2 GB) from "
                                              "24 GB, int8 (7.3 GB) from 11 GB, else nvfp4")
        elif klein:
            r["transformer_quant"] = ("auto", "FLUX 3 Klein picks for the memory it runs in: bf16 (9.7 GB; a 1 MP step "
                                              "peaked at 12.2 GB) on a 16 GB card, int8 (5.1 GB) from 10 GB, else "
                                              "nvfp4 (3 GB)")
        elif ideo:
            r["transformer_quant"] = ("auto", "Ideogram 4 (released in fp8) picks for the memory it runs in: bf16 "
                                              "(about 18.6 GB) on 32 GB cards, int8 (9.6 GB) on 16 GB ones, else "
                                              "nvfp4 (5.5 GB)")
        elif vram >= 22:
            r["transformer_quant"] = ("auto", f"{gpu_name} has {vram:.0f} GB, so the 10 GB bf16 model fits as it is")
        elif vram >= 11:
            r["transformer_quant"] = ("int8_convrot", f"{gpu_name} has {vram:.0f} GB: int8 halves the 10 GB model and "
                                                      "its W8A8 compute is as fast as bf16, at under 1% error")
        else:
            r["transformer_quant"] = ("nvfp4", f"{gpu_name} has {vram:.0f} GB: nvfp4 quarters the model so it fits "
                                               "(slower, fp4 weights are expanded every step)")
        te_gb = 9 if fibo or krea or klein else 17 if qwen else 19
        te_gb = 17 if ideo else te_gb  # Ideogram 4's Qwen3-VL-8B (bf16 17.5 GB)
        if vram >= te_gb + 4:
            r["text_encoder_quant"] = ("auto", f"the {te_gb} GB text encoder fits on {gpu_name} as it is")
        else:
            r["text_encoder_quant"] = ("int8", f"halves the {te_gb} GB text encoder so it fits on the GPU for caching "
                                               "instead of spilling into system RAM")
    if klein2:
        r["transformer_quant"] = ("auto", "FLUX.2 Klein (bf16 17 GB) picks for the memory it runs in: bf16 from 30 GB, "
                                          "int8 (9.6 GB) from 18 GB, else nvfp4 (5.6 GB)")
        r["text_encoder_quant"] = ("auto", "Qwen3-8B (bf16 15 GB) runs as it is from 19.5 GB, else int8 ConvRot (half)")
    r["gradient_checkpointing"] = (True, "saves most of the activation memory for a modest slowdown")
    r["checkpoint_keep_blocks"] = (-1, "measured at the start: as many blocks as fit skip the recompute")
    r["compile_model"] = ("auto", "measured 26% (MOSS-Image) and 12% (Krea 2) faster per step after a warm-up of "
                                  "under a minute; auto skips it for short test runs")
    if vram is not None and cfg.multi_gpu == "off" and not cfg.finetune:
        swap, saved = block_swap_plan(cfg, vram, r.get("transformer_quant", (cfg.transformer_quant,))[0])
        if swap:
            r["block_swap"] = (swap, f"the model and a step's memory do not fit {vram:.0f} GB: {swap} blocks' frozen "
                                     f"weights ({saved:.1f} GB) wait in system RAM and stream in as each block runs")
        else:
            r["block_swap"] = (0, f"the model fits {vram:.0f} GB as it is; swapping only slows the steps")
    # dataset
    target = 1.0 if vram is None or vram >= 12 else 0.5
    reason = ("1 MP (1024x1024) is the size the model was trained at" if target == 1.0
              else f"0.5 MP keeps memory low on a {vram:.0f} GB GPU")
    if data and data.get("median_mp") and data["median_mp"] < target * 0.9:
        med = data["median_mp"]
        target = max(0.25, int(med * 4) / 4)
        reason = f"half of your images are under {med:.2f} MP; training above their size only upscales them"
    if qwen and target >= 0.5:
        target, reason = 0.5, ("Fizgig's sweet spot for Qwen-Image 2.1: quicker than 1 MP, and a LoRA pulls less of the "
                               "model's sharpness toward the photos'")
    r["resolutions"] = ([target], reason)
    r["enable_buckets"] = (True, "keeps every image's aspect ratio instead of cropping it square")
    if character:
        r["flip_augment"] = (False, "faces are not symmetric (hair parting, moles, logos); mirrored copies blur them")
    elif goal == "style":
        r["flip_augment"] = (True, "a mirrored image has the same style, so this doubles the variety for free")
    else:
        r["flip_augment"] = (False, "only worth it if your subject looks the same mirrored (no text or logos)")
    ds_paths = [sp.path for sp in cfg.dataset_specs() if sp.path]
    if character and ds_paths:
        r["face_refs"] = ("\n".join(ds_paths), "every image with a face becomes a reference (a folder means all its "
                                               "images)")
    if character and body_only:
        r["face_refs"] = ("", "body only: nothing scores the face")
    if character and face_only:
        r["body_refs"] = ("", "face only: nothing scores the body")
    elif character and ds_paths and not cfg.body_refs:
        r["body_refs"] = ("\n".join(ds_paths), "every photo that shows the person from neck to mid-thigh becomes a "
                                               "reference; the others are skipped")
    if character:
        r["crop_augment"] = (("face", "close-ups teach the face's detail; bust crops would mostly teach the build")
                             if face_only else
                             ("none", "the crops are of the face and the bust; the build is learned from the whole "
                                      "photos") if body_only else
                             ("face+bust", "close-ups for the face, head-and-shoulders crops between them and the "
                                           "full photos"))
        n_out = (data or {}).get("outliers") or 0
        r["look_warmup"] = (bool(n_out), f"{n_out} pictures are under the look filter's fence: they stay in the "
                                         "dataset but start at a lower loss weight" if n_out else
                            "no outliers in the saved look scans (scan on the Dataset tab, Look consistency filter)")
        subject = cfg.trigger_word or any(sp.trigger for sp in cfg.dataset_specs())
        if subject and krea and cfg.network_type == "oft":
            r["isolation"] = (False, "with OFT on Krea 2 it added no likeness and went over the copy limit (one seed); "
                                     "turn it on if other people in your pictures pick up the subject's face")
        else:
            r["isolation"] = (bool(subject), "keeps the subject's face off other people (Fizgig: strangers reading as "
                                             "the subject 48 of 50 -> 0 of 50); about 1.5-2x the training time, and "
                                             "watch for memorised photos" if subject else
                              "needs a trigger word to isolate (Start page)")
    # training
    n = (data or {}).get("images") or 0
    if n:
        counts = (data or {}).get("counts") or {}
        reps = sum(counts.get(sp.path, 0) * sp.repeats for sp in cfg.dataset_specs() if sp.path in counts)
        items = reps or n  # pictures per epoch, repeats counted
        crops = {"face": 0.6, "face+bust": 1.2}.get(cfg.crop_augment, 0.0)  # extra items per picture (some are skipped)
        per_epoch = items * (1 + crops)
        steps = int(min(4000, max(1000, round(per_epoch * 100, -2))))
        r["steps"] = (steps, f"about 100 epochs: {n} images" + (f" x repeats = {items}" if items != n else "")
                             + (f", plus about {int(items * crops)} close-up crops" if crops else "")
                             + ("; the likeness probes can stop it earlier" if character else ""))
    else:
        r["steps"] = (2000, "a good start; check your photos (Photos page) for a number that fits it")
    if klein and n:
        r["steps"] = (int(min(4000, max(1000, round(per_epoch * 200, -2)))),
                      f"BFL's FLUX 3 Klein recipe: 4000 steps for 20 images, about 200 epochs ({n} images here)")
    if cfg.finetune:
        r["learning_rate"] = (1e-5, "a full fine-tune needs about a tenth of a LoRA's rate (Fizgig's measured 1e-5)")
        r["optimizer"] = ("adafactor", "Fizgig's measured optimizer for a fine-tune: a much smaller state than AdamW")
    else:
        r["optimizer"] = ("adamw8bit", "AdamW with 8-bit state: the same results as adamw with less memory")
        if cfg.optimizer not in AUTOMAGIC:  # it sets its own rate per layer type
            r["learning_rate"] = (1e-4, "the usual LoRA learning rate; go lower (5e-5) if samples look burnt or noisy")
            if qwen and cfg.network_type == "lora":
                r["learning_rate"] = ((1.5e-4, "Fizgig's Qwen-Image 2.1 style preset (rank 16)") if goal == "style" else
                                      (2e-4, "Fizgig's fastest-to-likeness Qwen-Image 2.1 recipe: rank 8 from 2e-4"))
            if qwen and cfg.network_type == "oft":
                r["learning_rate"] = (5e-5, "OFT on Qwen-Image 2.1 overtrained at 1e-4 with adamw8bit and with muon "
                                            "(copy over 0.6 by step 200, face likeness below the untrained model's); "
                                            "at 5e-5 it held the face and the full-body framing (one run each)")
            if klein and cfg.network_type == "lora":
                r["learning_rate"] = (1e-4, "BFL's FLUX 3 Klein recipe: 1e-4, constant after 200 warm-up steps")
                r["lr_scheduler"] = ("constant", "BFL's FLUX 3 Klein recipe")
                r["warmup_steps"] = (200, "BFL's FLUX 3 Klein recipe")
            if klein2 and cfg.network_type == "lora":
                r["learning_rate"] = (1e-4, "Fizgig's Klein 9B full-model recipe (\"Old Reliable\")")
        r["network_type"] = ("oft", "14 to 15 points over a plain LoRA on two seeds of one small set, face and body; "
                                    "loads in ComfyUI, not diffusers; not tested with the identity recipe's extras"
                             ) if character and krea else (
                             "lora", "reached likeness much sooner than LoKr in RulzTrainer's tests so far, and "
                                     "loads everywhere (ComfyUI and diffusers)")
        if cfg.network_type == "lokr":
            r["lokr_factor"] = (8, "Fizgig's measured default: lower is bigger files, nothing gained above 8")
        elif cfg.network_type == "lora":
            rank = 32 if goal in ("style", "character") else 16
            r["lora_rank"] = (rank, "styles carry a lot of varied detail" if goal == "style" else
                              "the wizard's default for a person: room for a face's detail" if goal == "character"
                              else "plenty for a concept; higher ranks mostly add file size")
            if qwen:
                rank = 16 if goal == "style" else 8
                r["lora_rank"] = (rank, "Fizgig's Qwen-Image 2.1 style preset" if goal == "style" else
                                  "Fizgig's Qwen-Image 2.1 recipe: rank 8 held skin detail best; 16 at the same rate "
                                  "overcooked")
            if klein:
                rank = 32
                r["lora_rank"] = (rank, "BFL's FLUX 3 Klein recipe: rank 32, alpha 32")
            if klein2:
                rank = 16
                r["lora_rank"] = (rank, "Fizgig's Klein 9B full-model recipe (\"Old Reliable\"): rank 16, alpha 16")
            r["lora_alpha"] = (float(rank), "alpha equal to rank keeps the adapter's scale at 1")
    if goal == "style":
        r["timestep_sampling"] = ("logsnr", "spends most steps at very high noise, where an art style is learned "
                                            "(the style-friendly sampler)")
    elif klein2:
        r["timestep_sampling"] = ("reference", "Fizgig's flux2_shift for Klein: logit-normal, shifted by the image size")
    elif not character:  # the identity recipe sets logit_normal itself
        r["timestep_sampling"] = ("logit_normal", "the default: most steps at middle noise, fewer at the extremes")
    r["batch_size"] = (1, "small batches with more steps beat large batches for LoRAs")
    if qwen and not cfg.finetune:
        r["adapter_mode"] = ("auto", "Fizgig's training adapter: without one Qwen-Image 2.1 LoRAs collapse into "
                                     "texture or wobble part-way (likeness 77 with it, 56 without)")
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
    if klein:
        every = 250
        r["save_every"] = (every, "BFL's FLUX 3 Klein recipe: a checkpoint every 250 steps to choose from")
    saves = total // max(1, int(cfg.save_every or every))  # with the save interval as it is now
    r["keep_checkpoints"] = ((0, f"{saves} small files at most: keep them all to compare") if saves <= 10 else
                             (8, f"{saves} checkpoints would pile up: delete all but the newest 8 as you go"))
    r["lora_ema"] = (0.98, "measured by Fizgig: 5 likeness points above no average late in a run, with half the swing "
                           "between checkpoints; longer averages (0.995) lag behind")
    r["sample_every"] = (every, "a sample set at every checkpoint shows which one to keep")
    fam = cfg.arch_name()
    if krea_turbo:
        r["sample_steps"] = (8, "Krea 2 Turbo is an 8-step distill")
        r["sample_cfg"] = (1.0, "Turbo samples without guidance")
    elif klein_distilled:
        r["sample_steps"] = (16, "the distilled FLUX 3 Klein (your sampling model) is a 16-step flow map")
        r["sample_cfg"] = (1.0, "the distilled Klein samples without guidance")
    elif klein2_distilled:
        r["sample_steps"] = (4, "the distilled FLUX.2 Klein (your sampling model) is a 4-step model")
        r["sample_cfg"] = (1.0, "the distilled Klein samples without guidance")
    elif klein2 and not cfg.turbo_lora:  # the turbo LoRA switched off
        r["sample_steps"] = (20, "BFL's default for Klein Base (the distilled Klein as the sampling model takes 4)")
        r["sample_cfg"] = (3.5, "BFL's guidance for Klein Base; it has no guidance embedding, CFG is all that steers it")
    elif klein and cfg.turbo_lora in ("auto", ""):  # no turbo LoRA for Klein
        r["sample_steps"] = (50, "BFL samples Klein Base at 50 steps (about 2.5 min per 1 MP preview on a 16 GB card; "
                                 "the distilled Klein as the sampling model takes 16 steps)")
        r["sample_cfg"] = (3.5, "BFL's guidance 3.5 for Klein Base (the same convention here)")
    elif cfg.samples_on_distilled_model() and fam in ARCH_TURBO:
        r["sample_steps"] = (ARCH_TURBO[fam]["steps"], "what the model's Turbo checkpoint is distilled for")
        r["sample_cfg"] = (1.0, "Turbo samples without guidance")
    elif cfg.turbo_lora == "auto" and fam in ARCH_TURBO:
        t = ARCH_TURBO[fam]
        r["sample_steps"] = (t["steps"], f"what {Path(t['file']).name}, the model's turbo LoRA, is made for")
        r["sample_cfg"] = (t["cfg"], "a turbo LoRA samples without guidance")
    elif cfg.turbo_lora:
        r["sample_steps"] = (8, "most turbo LoRAs are 8-step distills; match yours")
        r["sample_cfg"] = (1.0, "distilled turbo LoRAs sample without guidance")
    elif krea:
        r["sample_steps"] = (28, "Krea's pipeline samples Raw at 28 steps")
        r["sample_cfg"] = (5.5, "Krea's guidance 4.5 for Raw, in this trainer's convention (1 = off)")
    elif fibo:
        r["sample_steps"] = (30, "Bria samples dev at 50 steps; 20-30 is plenty for previews")
        r["sample_cfg"] = (5.0, "Bria's guidance scale for FIBO 2.0 dev")
    elif qwen:
        r["sample_steps"] = (25, "ComfyUI's Qwen-Image 2.1 template (the diffusers pipeline: 40)")
        r["sample_cfg"] = (1.0, "Qwen-Image 2.1 is meant to be sampled without guidance")
    elif ideo:
        r["sample_steps"] = (20, "Ideogram's V4_DEFAULT_20 preset (V4_QUALITY_48 for its best)")
        r["sample_cfg"] = (7.0, "Ideogram's guidance 7 (the last steps drop to 3 by themselves, as in its presets)")
    else:
        r["sample_steps"] = (20, "the reference MOSS-Image sampler")
        r["sample_cfg"] = (3.0, "the reference MOSS-Image guidance scale")
    if character:
        r["probe_every"] = (25, "frequent quick checks give the best-checkpoint and early-stop decisions enough points")
        if (cfg.turbo_lora and not (klein and cfg.turbo_lora == "auto")) or cfg.samples_on_distilled_model():
            r["probe_steps"] = (r["sample_steps"][0], "fast probes with the turbo model" if cfg.samples_on_distilled_model()
                                else "fast probes with the turbo LoRA")
            r["probe_cfg"] = (1.0, "turbo models sample without guidance")
        else:
            r["probe_steps"] = (r["sample_steps"][0], "without a turbo LoRA a probe needs the full sampler steps")
            r["probe_cfg"] = (r["sample_cfg"][0], "without a turbo LoRA a probe needs normal guidance")
        face_p, body_p = PRESET_PROBE_PROMPTS["identity"]
        subject = cfg.trigger_word or "the character"
        r["probe_face_prompt"] = (("", "body only: nothing scores the face") if body_only else
                                  (face_p.format(trigger=subject), "a plain headshot scores the face reliably"))
        r["probe_body_prompt"] = (("", "face only: each probe renders the headshot alone, in half the time")
                                  if face_only else
                                  (body_p.format(trigger=subject), "an upright full-body shot scores the build"))
    if cfg.arch_name() == "minimax_h3":
        h3_recommendations(cfg, r, goal, gpus, vram, character)
    elif cfg.arch_name() == "p_h3_image":
        ph3_recommendations(cfg, r, goal, gpus, vram, character)
    elif cfg.arch_name() == "minimax_h3_image":
        h3image_recommendations(cfg, r, goal, gpus, vram, character)
    return r


def h3image_recommendations(cfg: TrainConfig, r: dict, goal: str | None, gpus: list[dict], vram: float | None,
                            character: bool) -> None:
    """MiniMax-H3-Image: H3's recipe and memory plan (the same pruned 21 GB transformer and the same text encoder),
    with its own files, adapter and 6-step sampling (ai-toolkit's MiniMax-H3-Image defaults). Estimates, not
    measurements: nothing has been trained on it here."""
    h3_recommendations(cfg, r, goal, gpus, vram, character)
    if cfg.model_source == "files":
        r["model_file_quant"] = ("int8_convrot", "Comfy-Org's pruned int8 file (21.0 GB): the attention and "
                                                 "feed-forward layers the LoRA trains are the full model's (the 4-bit "
                                                 "w4a8 file is a smaller download that takes the same memory, at "
                                                 "lower precision)")
    r["resolutions"] = ([1.0], "1 MP: 1024 image tokens a step (a token covers 32 x 32 px). The model itself renders "
                               "at 4 MP and up, 4096 tokens a step: train higher if your photos are that large")
    r["timestep_sampling"] = ("reference", "only the 6 noise levels the model samples at, picked evenly, each "
                                           "through its own output head, as ai-toolkit trains MiniMax-H3-Image (the "
                                           "other choices train at noise levels its sampling never uses)")
    if not cfg.finetune:
        r["adapter_mode"] = ("auto", "ostris' MiniMax-H3-Image training adapter (what ai-toolkit trains it with): the "
                                     "model is step- and guidance-distilled, and a LoRA trained on it bare may drift")
    r["sample_steps"] = (6, "MiniMax-H3-Image's own 6-step schedule (one output head per step)")
    r["sample_cfg"] = (1.0, "MiniMax-H3-Image is guidance-distilled: no CFG")
    if character:
        r["probe_steps"] = (6, "the probes sample as the previews do")
        r["probe_cfg"] = (1.0, "MiniMax-H3-Image is guidance-distilled: no CFG")


def ph3_recommendations(cfg: TrainConfig, r: dict, goal: str | None, gpus: list[dict], vram: float | None,
                        character: bool) -> None:
    """P-H3-Image: H3's recipe and memory plan (the same pruned 21 GB transformer), with its own files, adapter and
    8-step sampling (ai-toolkit's P-H3-Image defaults; estimates, not measurements)."""
    h3_recommendations(cfg, r, goal, gpus, vram, character)
    if cfg.model_source == "files":
        r["model_file_quant"] = ("int8_convrot", "PrunaAI's int8 file (21.2 GB): already pruned, with Pruna's LoRA "
                                                 "merged in; the attention and feed-forward layers the LoRA trains "
                                                 "are the full model's")
        r["text_encoder_file_quant"] = ("int8_convrot", "the 5.1 GB int8 Qwen3-VL-4B (with its text connector): it "
                                                        "caches the captions first, then unloads")
    r["text_encoder_device"] = ("auto", "the small Qwen3-VL-4B fits one card")
    r["timestep_sampling"] = ("reference", "H3's own: one uniform base time per step on the shift-12 curve, as "
                                           "ai-toolkit trains P-H3-Image")
    if not cfg.finetune:
        r["adapter_mode"] = ("auto", "ostris' P-H3-Image training adapter (what ai-toolkit trains it with): the model "
                                     "is step- and guidance-distilled, and a LoRA trained on it bare drifts")
    r["sample_steps"] = (8, "P-H3-Image's own 8-step sigma grid (it is distilled for 8 steps)")
    r["sample_cfg"] = (1.0, "P-H3-Image is guidance-distilled: no CFG")
    if character:
        r["probe_steps"] = (8, "the probes sample as the previews do")
        r["probe_cfg"] = (1.0, "P-H3-Image is guidance-distilled: no CFG")


def h3_recommendations(cfg: TrainConfig, r: dict, goal: str | None, gpus: list[dict], vram: float | None,
                       character: bool) -> None:
    """MiniMax H3's own settings over the generic ones (musubi-tuner's and DiffSynth-Studio's H3 recipes; the memory
    figures are estimates from Comfy-Org's file sizes, not measurements)."""
    n_gpus = len(gpus)
    if cfg.model_source == "files":
        r["model_file_quant"] = ("pruned_int8_convrot", "Comfy-Org's pruned int8 file (21.0 GB): the 13B AdaLN "
                                                        "parameters are a small precomputed table, the attention and "
                                                        "feed-forward layers the LoRA trains are the full model's")
        r["text_encoder_file_quant"] = (("int8_convrot", "the 27.1 GB int8 Qwen3-VL-32B spreads over both cards for the "
                                                         "caption caching, then unloads") if n_gpus >= 2 else
                                        ("nvfp4_awq", "the 15.7 GB NVFP4 Qwen3-VL-32B: the smallest; what does not "
                                                      "fit the card runs from system RAM while the captions cache"))
    r["text_encoder_device"] = ("auto", "Qwen3-VL-32B is larger than one 16 GB card: auto spreads it over the GPUs "
                                        "(and system RAM)")
    r["transformer_quant"] = ("auto", "keeps the pruned int8 file as it is; a bf16 source is quantized for the GPUs "
                                      "it runs on (int8 from about 30 GB, else nvfp4)")
    if n_gpus >= 2 and not cfg.finetune:
        r["multi_gpu"] = ("model_split", f"{n_gpus} GPUs: the 21 GB int8 model split across them (about 10.5 GB "
                                         "each); a copy per card (data parallel) does not fit")
        r["block_swap"] = (0, "model split holds the whole model on the GPUs")
    elif vram is not None and not cfg.finetune:
        resident = max(1.0, vram - 4.5 - 1.6)  # the step's memory and the non-block weights (refiner, heads)
        swap = max(1, min(48, math.ceil(50 - resident / 0.387)))  # a pruned int8 block holds 0.387 GB
        r["multi_gpu"] = ("off", "one GPU found")
        r["block_swap"] = (swap, f"the 21 GB model does not fit {vram:.0f} GB: {swap} of the 50 blocks' frozen weights "
                                 f"({swap * 0.387:.1f} GB) wait in pinned system RAM and stream in as each runs "
                                 "(an estimate; musubi-tuner swaps up to 48)")
    r["resolutions"] = ([1.0], "1 MP: about H3's own 768 x 1344 canvas, and 1024 video tokens a step")
    if not cfg.finetune and cfg.network_type == "lora":
        rank = 32 if goal == "style" else 16
        r["lora_rank"] = (rank, "musubi-tuner's H3 examples train rank 16; DiffSynth-Studio's 32")
        r["lora_alpha"] = (float(rank), "alpha = rank: H3 LoRAs are saved without alpha (the layout ComfyUI, diffusers "
                                        "and musubi-tuner read at alpha = rank)")
        if cfg.optimizer != "automagic3":
            r["learning_rate"] = (1e-4, "musubi-tuner's and DiffSynth-Studio's H3 LoRA rate")
    r["timestep_sampling"] = ("reference", "H3's own: one uniform base time per step, the video sigma on the shift-12 "
                                           "curve (the audio sigma follows on shift 3), as the released schedulers "
                                           "and musubi-tuner train")
    if not cfg.finetune:
        r["adapter_mode"] = ("auto", "circlestone-labs' image training adapter: H3 is CFG-distilled, and musubi-tuner "
                                     "saw one-frame LoRAs trained without an adapter break up within about 50 steps")
    if cfg.turbo_lora != "auto":
        r["sample_steps"] = (30, "musubi-tuner's one-frame example (H3's pipeline: 49 evaluations)")
        r["sample_cfg"] = (1.0, "H3 is guidance-distilled: no CFG")
    if character:
        r["probe_steps"] = (r["sample_steps"][0], "the probes sample as the previews do")
        r["probe_cfg"] = (1.0, "H3 is guidance-distilled: no CFG")


def extra_review_rows(cfg: TrainConfig) -> list[tuple[str, str]]:
    """The review's rows about the adapter, memory, previews and the optional features that are on."""
    if cfg.finetune:
        adapter = "full fine-tune, extracting a LoRA of rank " + ", ".join(str(r) for r in cfg.finetune_extract_ranks)
    elif cfg.network_type == "oft":
        adapter = f"OFT, block {cfg.oft_block_size}" + (f", constraint {cfg.oft_constraint:g}"
                                                       if cfg.oft_constraint > 0 else "")
    elif cfg.network_type == "lokr":
        adapter = f"LoKr, factor {cfg.lokr_factor}" + (" + DoRA" if cfg.dora else "")
    else:
        adapter = f"LoRA, rank {cfg.lora_rank}, alpha {cfg.lora_alpha:g}" + (" + DoRA" if cfg.dora else "")
    rows = [("Adapter", adapter),
            ("Memory", f"gradient checkpointing {'on' if cfg.gradient_checkpointing else 'off'}; "
                       + (f"block swap {cfg.block_swap} blocks" if cfg.block_swap else "no block swap"))]
    if cfg.samples_on_distilled_model():
        turbo = ", on the Turbo model (no turbo LoRA on top)" if cfg.turbo_lora in ("", "auto") else             ", your turbo LoRA on the Turbo model (distils it twice)"
    elif cfg.turbo_lora == "auto" and cfg.arch_name() in ARCH_TURBO:
        turbo = ", with the model's turbo LoRA"
    elif cfg.turbo_lora and cfg.turbo_lora != "auto":
        turbo = ", with your turbo LoRA"
    else:
        turbo = ", plain model"
    if cfg.sample_lora == "auto" and cfg.arch_name() in ARCH_SAMPLE_LORA and cfg.samples_on_base_model():
        turbo += " + the SFT LoRA (Base previews with SFT's look)"
    elif cfg.sample_lora and cfg.sample_lora != "auto":
        turbo += f" + {Path(cfg.sample_lora).name}"
    rows.append(("Previews", f"{cfg.sample_steps} steps, CFG {cfg.sample_cfg:g}{turbo}"))
    options = []
    if cfg.timestep_sampling != "logit_normal":
        options.append(f"{cfg.timestep_sampling} timesteps")
    if cfg.isolation:
        options.append("LoRA Isolation")
    if cfg.attn_localize:
        options.append(f"trigger attention on the subject ({cfg.attn_localize_weight:g})")
    if cfg.id_align:
        options.append(f"identity alignment ({cfg.id_align_weight:g})")
    if cfg.body_align:
        options.append(f"body alignment ({cfg.body_align_weight:g})")
    if cfg.selftrain:
        options.append(f"self-training at {cfg.selftrain_at} of the steps (x{cfg.selftrain_weight:g})")
    if cfg.soup_files:
        options.append(f"no training: a seed soup of {len(cfg.soup_files)} LoRAs picked by the probes")
    if cfg.noise_files:
        options.append(f"no training: a noise check of the probes on {len(cfg.noise_files)} LoRA(s), "
                       f"{cfg.noise_seeds} seeds each")
    if cfg.reward_steps:
        options.append(f"then {cfg.reward_steps} likeness reward steps")
    if cfg.look_warmup:
        options.append("look warm-up")
    if cfg.lora_ema:
        options.append(f"weight average {cfg.lora_ema:g}")
    if cfg.keep_checkpoints:
        options.append(f"newest {cfg.keep_checkpoints} checkpoints kept")
    if options:
        rows.append(("Options", ", ".join(options)))
    if cfg.race:
        try:
            rows.append(("Race", race_review_line(cfg)))
        except ValueError as e:
            rows.append(("Race", f"cannot read the racers: {e}"))
    return rows


def datasets_line(cfg: TrainConfig, data: dict | None) -> str:
    """The review's dataset line: each dataset's name with its own repeats and resolutions, then the totals."""
    parts = []
    for sp in cfg.dataset_specs():
        if not sp.path:
            continue
        extra = ([f"x{sp.repeats}"] if sp.repeats != 1 else []) + \
            ([" + ".join(f"{mp:g} MP" for mp in sp.resolutions)] if sp.resolutions else [])
        parts.append((Path(sp.path).name or sp.path) + (f" ({', '.join(extra)})" if extra else ""))
    data = data or {}
    return (f"{', '.join(parts) or '(none picked)'}: {images_text(data.get('images', '?'))}, "
            f"{data.get('captioned', '?')} captioned")


# the usual learning rates, from the recipes the wizard recommends (LoRA: 5e-5 to 2e-4, full fine-tune: 1e-5); a third
# of the lowest to three times the highest counts as usual (after Fizgig's Start checks)
LR_USUAL = {"lora": (5e-5, 2e-4), "finetune": (1e-5, 1e-5)}


def start_warnings(cfg: TrainConfig) -> list[str]:
    """The usual mistakes, said before a run starts (never a block): images with no caption, captions that already
    hold the trigger word (RulzTrainer puts it at the start of every caption, so it would be there twice) and a
    learning rate well outside the model's usual range."""
    import re

    from rulztrainer.archs import get_arch

    out = []
    try:
        arch = get_arch(cfg)
    except (ValueError, NotImplementedError):
        arch = None
    missing = doubled = images = 0
    doubled_words = set()
    for sp in cfg.dataset_specs():
        if not sp.path or not Path(sp.path).is_dir():
            continue
        trigger = (sp.trigger or cfg.trigger_word or "").strip()
        pattern = re.compile(rf"(?<!\w){re.escape(trigger)}(?!\w)", re.IGNORECASE) if trigger else None
        for img in list_dataset_images(sp.path):
            images += 1
            cap = arch.caption_file(img, cfg) if arch is not None else img.with_suffix(cfg.caption_ext)
            if not cap.is_file():
                missing += not _captioned(img)
                continue
            if pattern is not None:
                try:
                    text = cap.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                if pattern.search(text):
                    doubled += 1
                    doubled_words.add(trigger)
    if missing:
        out.append(f"{missing} of {images} images have no caption: they train on "
                   + ("the trigger word alone." if cfg.trigger_word else "an empty prompt."))
    if doubled:
        out.append(f"{doubled} of {images} captions already hold the trigger word ({', '.join(sorted(doubled_words))}). "
                   "RulzTrainer puts it at the start of every caption, so those captions have it twice: take it "
                   "out of the caption files, or clear the trigger word.")
    kind = "finetune" if cfg.finetune else "lora" if cfg.network_type == "lora" else None
    if kind and cfg.optimizer not in AUTOMAGIC and cfg.learning_rate > 0:
        lo, hi = LR_USUAL[kind]
        if not lo / 3 <= cfg.learning_rate <= hi * 3:
            usual = f"{lo:g}" if lo == hi else f"{lo:g} to {hi:g}"
            out.append(f"Learning rate {cfg.learning_rate:g} is well outside the usual {usual} for "
                       f"{'a full fine-tune' if cfg.finetune else 'a LoRA'}: "
                       + ("expect burnt, noisy samples." if cfg.learning_rate > hi * 3 else
                          "expect it to learn very slowly."))
    return out


def review_issues(cfg: TrainConfig, goal: str | None, data: dict | None, n_gpus: int) -> list[tuple[str, str]]:
    """What the wizard's last step points out before starting: (error | warning | info, text)."""
    out: list[tuple[str, str]] = []
    if _running():
        out.append(("error", "A training job is already running. Stop it (Monitor tab) before starting another."))
    specs = [sp for sp in cfg.dataset_specs() if sp.path]
    missing = [sp.path for sp in specs if not Path(sp.path).is_dir()]
    if not specs:
        out.append(("error", "No dataset picked: pick one or more (Photos page), or create one on the Dataset tab."))
    elif missing:
        out.append(("error", f"Dataset folder not found: {', '.join(missing)}"))
    elif data is not None:
        if not data["images"]:
            out.append(("error", "The datasets have no images (png, jpg, webp or bmp)."))
        else:
            out += [("warning", w) for w in start_warnings(cfg)]
    try:
        cfg.validate()
    except ValueError as e:
        out.append(("error", str(e)))
    if cfg.model_source == "files":
        try:
            from rulztrainer.comfy_files import planned_downloads

            planned = planned_downloads(cfg)
        except ValueError as e:
            out.append(("error", str(e)))
            planned = []
        if planned:
            out.append(("info", "Downloads on start (cached after the first run): "
                                + ", ".join(file.rsplit("/", 1)[-1] for _, _, file in planned)))
        for key, label in (("transformer_file", "Diffusion model"), ("text_encoder_file", "Text encoder"),
                           ("vae_file", "VAE")):
            if getattr(cfg, key) and not Path(getattr(cfg, key)).is_file():
                out.append(("error", f"{label} file not found: {getattr(cfg, key)}"))
    if "edit-turbo" in cfg.model_path.lower():
        out.append(("warning", "FIBO-2.0-edit-turbo is a distilled model. Train on FIBO-2.0-dev: its LoRAs load "
                               "onto Edit Turbo."))
    elif "qwen-image-2.1-turbo" in cfg.model_path.lower():
        out.append(("warning", "Qwen-Image-2.1-Turbo is a distilled model. Train on Qwen-Image-2.1: its LoRAs load "
                               "onto Turbo, and the Turbo LoRA previews them the same way."))
    elif "krea-2-turbo" in cfg.model_path.lower():
        out.append(("warning", "Krea-2-Turbo is a distilled model. Train on Krea-2-Raw and set Turbo as the sampling "
                               "model: LoRAs trained on Raw are meant for Turbo."))
    elif cfg.model_source != "files" and cfg.arch_name() == "krea2":
        out.append(("info", "Krea 2 is gated: accept its license on Hugging Face and log in with hf auth login before "
                            "the first download." + (" Krea-2-Turbo (the sampling model) has a gate of its own: accept "
                                                     "its license as well." if "krea-2-turbo" in
                                                     (cfg.sample_model_path or "").lower() else "")))
    elif cfg.model_source != "files" and cfg.arch_name() == "fibo2":
        out.append(("info", "FIBO 2.0 is gated: accept its terms on Hugging Face and log in with hf auth login "
                            "before the first download."))
    elif cfg.arch_name() == "flux3_klein" and get_arch(cfg).distilled_sample_model(cfg.model_path):
        out.append(("warning", "FLUX-3-Klein is the distilled model (16 steps, samples only). Train on "
                               "FLUX-3-Klein-Base and set the distilled one as the sampling model."))
    elif cfg.arch_name() == "flux3_klein":
        out.append(("info", "Black Forest Labs' FLUX 3 Klein repos (the single files too) are gated: accept the FLUX "
                            "Kommunity License on huggingface.co/black-forest-labs/flux-3-image-klein-base and log in "
                            "with hf auth login before the first download. Captions should be full scene captions "
                            "(framing, subject, background, lighting): the trainer does not run BFL's prompt "
                            "upsampler."))
    elif cfg.arch_name() == "flux2_klein" and get_arch(cfg).distilled_sample_model(cfg.model_path):
        out.append(("warning", "FLUX-2-Klein-9B is the distilled model (4 steps, for samples). Train on "
                               "FLUX-2-Klein-9B-Base and set the distilled one as the sampling model."))
    elif cfg.arch_name() == "flux2_klein":
        out.append(("info", "Black Forest Labs' FLUX.2 Klein repos are gated: accept the FLUX Non-Commercial License on "
                            "huggingface.co/black-forest-labs/FLUX.2-klein-base-9b-fp8 (or -base-9B for bf16) and log "
                            "in with hf auth login before the first download. This trains text-to-image LoRAs, not "
                            "edit LoRAs."))
    if cfg.multi_gpu != "off" and n_gpus < 2:
        out.append(("error", "Multi-GPU is on, but only one GPU was found."))
    if goal == "character" or cfg.preset == "identity":
        if not (cfg.face_refs or cfg.body_refs):
            out.append(("warning", "No face or body references: likeness scores, the best checkpoint and early stop "
                                   "need them (Photos page, Advanced)."))
        elif character_focus(cfg) != "face_only" and not cfg.body_refs:
            out.append(("info", "This focus scores the body, but there are no body references: the body probe goes "
                                "unscored and the overall score is the face alone. Tick Body ref on full-length "
                                "photos, or take the recommendation (Photos page, Advanced)."))
        elif character_focus(cfg) != "body_only" and not cfg.face_refs:
            out.append(("info", "This focus scores the face, but there are no face references: the face probe goes "
                                "unscored (Photos page, Advanced)."))
        elif (cfg.probe_every > 0 and cfg.probe_steps <= 10
              and not (cfg.turbo_lora or cfg.samples_on_distilled_model())):
            out.append(("warning", f"Probes render in {cfg.probe_steps} steps without a turbo LoRA, so they come out "
                                   "unfinished and their likeness scores mean little. Set a turbo LoRA or raise the "
                                   "probe steps (Samples & probes page, Advanced)."))
    if cfg.sample_every <= 0 or not cfg.sample_prompts:
        out.append(("info", "No sample prompts (or sampling is off): there will be no preview images while training."))
    if cfg.save_every > cfg.steps:
        out.append(("info", "Save every is more than the number of steps, so only the final LoRA is saved."))
    return out


def suggested_prompts(goal: str | None, trigger: str, focus: str = "face_focus") -> list[str]:
    if goal == "character" and focus == "face_only":
        t = trigger or "the character"
        return [f"photo of {t} smiling in a sunny park, natural light",
                f"close-up portrait of {t}, studio lighting, plain grey background",
                f"photo of {t} laughing at a cafe table, waist up, warm evening light"]
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
    if isinstance(options, dict) and not isinstance(value, list) and value in options:
        return str(options[value])
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, list):
        return ", ".join(f"{v:g} MP" for v in value)
    if isinstance(value, float):
        return f"{value:g}"
    if value == "":
        return "empty"
    text = str(value)
    return text if len(text) <= 48 else text[:45] + "..."


# The wizard's character / person setup: the focus cards shown without Advanced, the basics it puts over the identity
# recipe, and the recommendations it takes by itself (nothing to choose: hardware, resolution, references, samples)
WIZARD_FOCUS = ("face_focus", "face_body", "face_only")
WIZARD_BASICS = {"network_type": "lora", "finetune": False, "lora_rank": 32, "lora_alpha": 32.0,
                 "optimizer": "adamw8bit", "learning_rate": 1e-4}
WIZARD_AUTO = ("multi_gpu", "dp_balance", "transformer_quant", "text_encoder_quant", "gradient_checkpointing",
               "checkpoint_keep_blocks", "block_swap", "compile_model", "resolutions", "enable_buckets",
               "flip_augment", "face_refs", "body_refs", "adapter_mode", "target_text_fusion", "target_attn_gate",
               "steps", "save_every", "keep_checkpoints", "lora_ema", "sample_every", "sample_model_path",
               "sample_steps", "sample_cfg", "probe_steps", "probe_cfg")

# How well tested an experimental setting is: (badge, what the badge means)
EVIDENCE = {
    "measured": ("Measured gain", "RulzTrainer measured a gain on more than one seed, on one set so far."),
    "worse": ("Looked worse here", "Measured here and scored below the plain setup, on one seed of one set so far."),
    "others": ("Tested by others", "Someone else (usually Fizgig's likeness tests) measured a gain; RulzTrainer has "
                                   "not repeated it yet."),
    "mixed": ("Small or mixed result", "Measured, but the effect was small, neutral or only seen in a short test."),
    "cost": ("Only speed tested", "Its speed and memory were measured, but not yet whether it improves the likeness."),
    "untested": ("Not tested yet", "Built and working, but nobody has measured whether it improves the likeness. "
                                   "A configuration race against the plain setup is the way to find out."),
    "tool": ("Comparison tool", "Not a setting that changes the LoRA itself: a way to test settings against each "
                                "other on your own photos."),
}
EXPERIMENT_GROUPS = ("Adapter and optimizer", "Likeness boosters", "How noise levels are trained",
                     "Compare setups")
# The experimental extras on the wizard's last page. key: the form key; off: its value with the extra off (set again
# when the user answers No); control: switch | select (options, or None for the form's own) | number | text; evidence:
# a key of EVIDENCE; what / tested / cost: the explanation dialog (from the README's measurements, with its numbers)
EXPERIMENTS = [
    {"key": "network_type", "group": "Adapter and optimizer", "title": "Adapter type", "off": "lora",
     "control": "select", "options": {"lora": "LoRA (default)", "lokr": "LoKr", "oft": "OFT"}, "evidence": "measured",
     "short": "LoKr or OFT instead of a plain LoRA.",
     "what": "A LoRA adds a small pair of matrices to each layer. LoKr uses a Kronecker pair instead (factor 8 by "
             "default); OFT rotates each layer's output channels in blocks, which keeps the base model's structure.",
     "tested": "OFT is the one confirmed gain in RulzTrainer's tests: 43.4 and 42.9 combined on two seeds, 14 to 15 "
               "points over its own base rows on face and body, on all 8 probe seeds, and still rising at step 2000. "
               "One small set of a generated character on Krea 2; the copy check crossed its limit once (0.59 at the "
               "last probe on one seed), so check the pictures. LoKr factor 8 tied plain here (29.5, one seed) with "
               "the lowest body score of the adapters, unlike Fizgig, who measured it as 'the highest likeness we've "
               "ever measured, with noticeably more natural skin than standard LoRA on the same data'.",
     "cost": "OFT: about 25% slower steps on Krea 2. LoKr: about 20% slower steps and files of about 400 MB on a 12B "
             "model. Neither loads in diffusers (both load in ComfyUI). Rank, alpha and LoRA+ do not apply. OFT "
             "cannot be souped or post-hoc averaged."},
    {"key": "dora", "group": "Adapter and optimizer", "title": "DoRA", "off": False, "control": "switch",
     "evidence": "mixed", "when": lambda wz: wz.w["network_type"].value != "oft",
     "short": "Trains each weight's size and direction separately.",
     "what": "Splits every layer's weight into a direction, which the LoRA or LoKr changes, and a magnitude that "
             "trains on its own (weight-decomposed adaptation).",
     "tested": "35.1 combined on one seed against about 31 for plain AdamW, a gap too small to count on one run (one "
               "small set on Krea 2). Speed: LoKr 8 with DoRA ran 4.9 s a step against 3.5 s without.",
     "cost": "Slower steps (about 30 to 40% on Krea 2). Saved as keys ComfyUI applies; diffusers cannot read them. "
             "Cannot be souped. Not with OFT."},
    {"key": "optimizer", "group": "Adapter and optimizer", "title": "Optimizer", "off": "adamw8bit",
     "control": "select", "options": {"adamw8bit": "adamw8bit (default)", "adamw": "adamw", "adafactor": "adafactor",
                                      "automagic3": "automagic3", "automagic4": "automagic4", "keel": "keel",
                                      "keel_fb": "keel_fb", "muon": "muon", "normuon": "normuon (untested)"},
     "evidence": "mixed",
     "short": "Muon or a self-tuning optimizer instead of adamw8bit.",
     "what": "adamw8bit is AdamW with its state in 8 bits (adamw keeps it in full precision; adafactor is lighter). "
             "muon steps each weight matrix as a whole, evening out the update across all its directions, at AdamW's "
             "learning rates; normuon (untested) is muon with each row's step evened out, OneTrainer's default "
             "Muon. automagic3 and automagic4 tune their own learning rate as they go; automagic4 adds "
             "smaller steps for the most sensitive weights. keel is automagic4 with extra machinery aimed at a "
             "character's identity, and keel_fb the same for the face and the body alike.",
     "tested": "keel against adamw8bit, two seeds on one small set on Krea 2 at gradient accumulation 1: a repeated "
               "face gain (FaceNet 52.7 and 54.3 against 44 to 49), no repeated body gain, and faces very alike "
               "across seeds. keel_fb tied keel on one seed. On a body-only set, single learning-rate groups beat "
               "by_type by 4.4 body points over 8 seeds, and Max lr at 1e-3, 5e-4 and 3e-4 tied at 1000 steps. "
               "Automagic was measured for learning rates and speed only, so adamw8bit stays the default. adafactor "
               "is Fizgig's pick for a full fine-tune, not tested for LoRAs. muon: Fizgig measured it level with "
               "AdamW on likeness (Qwen-Image 2.1, rank 8, 20 epochs) and made it every picture model's LoRA default "
               "for sharper fine detail by eye; a study of Muon on LoRA factors for language models found no gain "
               "over Adam. On Qwen-Image 2.1 (rank 32 at 1e-4, one run each on one set of 62 photos): keel led "
               "adamw8bit and muon by 12 to 15 combined points with the best FaceNet, but the highest copy score of "
               "the three (0.61); muon tied adamw8bit at about twice the time a step. normuon has never run in a real "
               "training here.",
     "cost": "automagic4 took 1.6 s a step against automagic3's 2.2 s. The learning-rate schedule, LoRA+ and the "
             "per-image learning rate do not apply to the automagic family. Keel's consensus gate wants gradient "
             "accumulation of 2 or more. muon keeps one full-precision number per weight (less than adamw, a "
             "little more than adamw8bit) and takes AdamW's learning rate as it is; normuon the same plus one "
             "number per row."},
    {"key": "oft_scaled_lr", "group": "Adapter and optimizer", "title": "Scaled OFT learning rate", "off": False,
     "control": "switch", "evidence": "untested", "when": lambda wz: wz.w["network_type"].value == "oft",
     "short": "Every OFT block size steps about as far as a block of 32.",
     "what": "OFT rotates each layer's outputs in blocks, and at one learning rate a bigger block moves further per "
             "step (a block of 64 about twice as far as one of 16). This gives each block size its own rate, "
             "sqrt(31 / (size - 1)) times yours, so block 32 is unchanged and a race of block sizes compares the "
             "sizes, not step sizes (OneTrainer's Scaled OFT, as a learning-rate factor).",
     "tested": "Never run in a real training here. The scaling is worked out from how OFT's blocks train, not "
               "measured.",
     "cost": "None in speed or memory. Layers whose outputs do not divide by the block size get smaller blocks and "
             "so a higher rate. Ignored by the self-tuning optimizers and hull."},
    {"key": "isolation", "group": "Likeness boosters", "title": "LoRA Isolation", "off": False, "control": "switch",
     "evidence": "mixed",
     "short": "Keeps the person's face off everyone else in the picture.",
     "what": "A character LoRA tends to put its subject's face on other people. Isolation also trains every photo "
             "without its trigger word, held to what the model draws without the LoRA; fuses the trigger into one word "
             "(you then prompt with it joined, for example gweltman); and adds pictures of the subject next to a "
             "generated stranger.",
     "tested": "Fizgig, on Krea 2: strangers read as the subject in 48 of 50 pictures without it and 0 of 50 with "
               "it, while the subject's own likeness held (0.77 to 0.78). In RulzTrainer the strangers test was not "
               "repeated. Its likeness was unsettled on one small set: 54.5 then 32.0 combined on two seeds, against "
               "about 31 plain. Copy reached 0.68, and headshots came back in a training photo's room on both seeds, "
               "so check for memorising.",
     "cost": "About 1.5x the training time over a run (Fizgig; about 2x per isolation step here). Needs a trigger "
             "word, and you must prompt with the joined word. Downloads a set of stranger pictures. Not with a full "
             "fine-tune or a parallel race."},
    {"key": "id_loss_weight", "group": "Likeness boosters", "title": "Direct identity loss", "off": 0.0,
     "control": "number", "step": 0.05, "precision": 2, "hint": "0 = off; try 0.2 (at most 0.5)",
     "evidence": "worse",
     "short": "Adds a face-recognition score to what training minimises.",
     "what": "Decodes the face the model predicts during training and adds a face-recognition term, so the LoRA is "
             "pulled straight toward your references. 0 = off; 0.2 is the tested weight.",
     "tested": "It hurt on a full run: weight 0.2 gave 27.5 combined against about 31 plain and 37.1 for its own "
               "base, on one seed of one small Krea 2 set. ArcFace, its training target, read 7.7 points over the "
               "held-out FaceNet, the body fell to 8.7, and the guard switched it off at step 2000. An earlier short "
               "test only showed the loss falling.",
     "cost": "Peak memory 8.0 GB against 6.05 GB and about 7% slower steps. With the ArcFace backend the probes' own "
             "scorer becomes the training target (the FaceNet backend avoids that). Needs face references. Not in a "
             "parallel race."},
    {"key": "attn_localize", "group": "Likeness boosters", "title": "Trigger attention on the subject", "off": False,
     "control": "switch", "evidence": "mixed",
     "short": "Ties the trigger word to the person, not the background.",
     "what": "Reads, inside the model, where the trigger word's attention lands in each training photo and pulls it "
             "onto the person (their face box and silhouette). Training-only: the LoRA file is the same.",
     "tested": "Speeds up the start: steps 200 to 800 averaged 34.5 and 28.2 combined against 19.5 to 24.1 plain, on "
               "two seeds of one small Krea 2 set. Late gains were small (39.0 and 33.6 against about 31), and its "
               "body gain did not repeat. Adding it to OFT gave no gain and raised copy to 0.49.",
     "cost": "Needs a trigger word. The tapped blocks run without torch.compile. Not in a parallel race or a full "
             "fine-tune."},
    {"key": "id_align", "group": "Likeness boosters", "title": "Identity alignment", "off": False,
     "control": "switch", "evidence": "mixed",
     "short": "Teaches the face's tokens to carry the person's identity.",
     "what": "A small helper network (thrown away afterwards) teaches the face's tokens inside the model to carry "
             "that photo's ArcFace identity, giving identity feedback at every noise level. Training-only.",
     "tested": "No gain: 29.3 combined and 45.0 FaceNet on one seed of one small Krea 2 set, below its own base "
               "(37.1) and about level with plain overall.",
     "cost": "Trains against ArcFace, so the FaceNet second scorer is switched on to keep the scores honest. First GPU "
             "only. Not in a full fine-tune or a parallel race."},
    {"key": "body_align", "group": "Likeness boosters", "title": "Body alignment", "off": False, "control": "switch",
     "evidence": "mixed", "when": lambda wz: wz.has_body(),
     "short": "The same for the build: the person's tokens carry their body.",
     "what": "Teaches the person's tokens inside the model to carry that photo's body (its DINOv2 body embedding, "
             "what the body probe scores), through a helper that is thrown away. Only photos showing the neck to "
             "mid-thigh count.",
     "tested": "No gain shown: 34.2 combined on one seed of one small Krea 2 set, body 19.1 against 18.2 for its own "
               "base.",
     "cost": "Needs photos that show the body, and person masks."},
    {"key": "selftrain", "group": "Likeness boosters", "title": "Self-training on its best pictures", "off": False,
     "control": "switch", "evidence": "mixed",
     "short": "Halfway through, adds its own best renders to the photos.",
     "what": "Halfway through, the LoRA draws about 30 pictures of the person in new scenes; the few that are clean, "
             "sharp and look more like them than the probes do train beside your photos for the rest of the run, at "
             "half weight and never more than a fifth of them.",
     "tested": "Barely tested: on one small Krea 2 set it kept 2 of 32 pictures at its bar of 60, so it changed "
               "little (34.4 combined, one seed). Research on models training on their own output (model collapse) "
               "warns that repeated rounds can amplify the model's own flaws; this runs one round, with a copy "
               "check.",
     "cost": "A few minutes. Can copy the model's own flaws, so judge the samples by eye. Not in a parallel race."},
    {"key": "reward_steps", "group": "Likeness boosters", "title": "Likeness reward fine-tuning", "off": 0,
     "control": "number", "precision": 0, "hint": "0 = off; try 100-200 steps", "evidence": "mixed",
     "short": "A second stage that rewards the most lifelike renders.",
     "what": "After the normal steps, the LoRA renders groups of pictures for new prompts, the probes rate how much "
             "each looks like the person, and the LoRA learns to make the better ones more likely (a Flow-GRPO "
             "style reinforcement stage). The number is how many reward steps; 0 = off.",
     "tested": "No visible gain: 150 steps went from 29.0 to 27.9 combined on one seed of one small Krea 2 set, with "
               "one probe after the stage. About 80 to 100 s a reward step on a 48 GB card, 3.5 hours for the stage; "
               "about 77 s on a 16 GB card with nvfp4.",
     "cost": "Slow (seconds to a minute or more a step). It can learn to please the scorer instead of the eye; the "
             "probes keep the best checkpoint and the first stage's LoRA is kept. One GPU, not with a full "
             "fine-tune."},
    {"key": "likeness_lr_control", "group": "Likeness boosters", "title": "Learning rate from the likeness trend",
     "off": False, "control": "switch", "evidence": "mixed",
     "short": "Raises the learning rate while likeness climbs, lowers it when it falls.",
     "what": "Scales the learning rate between 0.5x and 2x by the likeness probes' trend: up while the score rises, "
             "down when it falls.",
     "tested": "No gain: the rate climbed to 2x, and it read 30.4 combined on one seed of one small Krea 2 set, "
               "level with plain and below its own base.",
     "cost": "Needs the likeness probes. With automagic4 its own controller may pull the rate back."},
    {"key": "resolution_step_up", "group": "Likeness boosters", "title": "Resolution step-up", "off": False,
     "control": "switch", "evidence": "untested",
     "short": "Starts small and moves to a higher resolution when likeness stalls.",
     "what": "Trains at the lowest of the run's resolutions first and moves up when the likeness score stops "
             "improving (pick two or more resolutions under Advanced on the Photos page).",
     "tested": "Not settled: one run on one small Krea 2 set stalled at 0.25 MP until step 1600, so most of it was a "
               "plain run. Nothing clearly moved after 0.5 MP began, and steps took twice as long.",
     "cost": "Needs the likeness probes; it replaces the early stop. Not in a race."},
    {"key": "timestep_sampling", "group": "How noise levels are trained", "title": "Noise levels",
     "off": "logit_normal", "control": "select",
     "options": {"logit_normal": "logit_normal (default)", "identity": "identity", "content": "content"},
     "evidence": "mixed",
     "short": "Which noise levels the training steps practise on.",
     "what": "logit_normal favours the middle noise levels. identity centres them lower (a median of about 0.38), "
             "where a face's features are settled; content favours low noise, where fine detail is.",
     "tested": "Tied plain on one small Krea 2 set: identity 33.2 combined on one seed; content 34.9 and 27.3 on two "
               "seeds, a slow start and a late climb that did not repeat. A short 5-image test had identity at 73.9 "
               "against 74.2.",
     "cost": "None beyond the choice."},
    {"key": "region_steer", "group": "Likeness boosters", "title": "Region steering", "off": False,
     "control": "switch", "evidence": "mixed",
     "short": "Turns a picture's background or clothes down when they are learned faster than the person.",
     "what": "The region watch reads every training step's loss for the face, hair, skin, clothes and background. "
             "Region steering acts on it once an epoch, by rules that are each on their own setting on the Train "
             "tab: a cap on how far the background and clothes may be learned compared with the subject, a "
             "per-picture rule for pictures whose scenery runs ahead of the others', and, off unless asked for, a "
             "raise for a lagging face, hair or skin and a pull of the background towards the base model. What it "
             "takes from a region goes to the rest of that picture.",
     "tested": "Measured once, on a Qwen-Image 2.1 fine-tune (59 pictures, 2000 steps): at its defaults every reading "
               "stayed within about half a point of the plain run, on the training pictures and on held-out photos. "
               "It turned the clothes down on 49 pictures, typically to 0.82, and no picture triggered the "
               "per-picture rule. Read from the training steps the cap rule flickered at first; reading the sweeps "
               "steadied it. Four seeds of the plain run differ from each other by as much: no effect, and no "
               "benefit shown.",
     "cost": "No extra model calls, except one more forward pass a step while the background prior acts. Needs "
             "MediaPipe for the parts maps. Not in a configuration race, not with Hull's region gains."},
    {"key": "holdout_images", "group": "Likeness boosters", "title": "Held-out photos", "off": 0,
     "control": "number", "precision": 0, "hint": "0 = none; 2 or 3 of a set of 20 or more", "evidence": "tool",
     "short": "Keeps a few photos out of training to see what was learned of the person, not of the photos.",
     "what": "N photos, spread evenly through the set by file name, never train. Once an epoch the model is run on "
             "them at three fixed noise levels, and how much of their face, hair, skin, clothes and background error "
             "is gone is compared with the same reading on the training pictures. Learning that shows on unseen "
             "photos is the subject; the gap is memorising. The Train tab can also keep a checkpoint at the best "
             "held-out score, stop on it, and let region steering act on the gap.",
     "tested": "Measured on four Qwen-Image 2.1 fine-tune runs with 3 photos held out. The held-out reading peaked at "
               "step 944 of 2000 and then fell while the training pictures kept improving, in step with FaceNet on the "
               "probes; the combined likeness score chose step 1400. Runs that trained alike read within half a point "
               "of each other, where their probe scores differed by 10. The steering rule that acts on the gap "
               "changed nothing.",
     "cost": "The held-out photos do not train. 3 forward passes per held-out photo per sweep, plus the same per "
             "training picture swept. Needs at least 10 photos and MediaPipe. Not in a race, not with the "
             "resolution step-up."},
    {"key": "region_class_weights", "group": "Likeness boosters", "title": "Region class weights", "off": "",
     "control": "text", "hint": "empty = off; face=3, hair=1.5, skin=1.5, clothes=0.5, background=0.3",
     "evidence": "worse",
     "short": "One fixed loss weight each for the face, hair, skin, clothes and background.",
     "what": "Weights the training loss by what each part of a picture shows, for the whole run: the picture is "
             "split into face, hair, skin, clothes and background, and each takes the weight given here (1 when "
             "left out, 0 to mask it out), rescaled to mean 1 per picture. The plain alternative to region steering.",
     "tested": "Measured once, on a Qwen-Image 2.1 fine-tune with face 3, hair 1.5, skin 1.5, clothes 0.5, background "
               "0.3: the training faces were learned much further (47% of their error gone against 30%), but held-out "
               "faces ended worse (7% against 12%), the full-body probe lost its framing sooner and likeness was "
               "level. It memorised the training faces faster. Turning only the background down to 0.3 did the same "
               "in a milder form. Both ended well outside the spread of four seeds of the plain run.",
     "cost": "None per step. Needs MediaPipe for the parts maps. Not together with the region-weighted loss, so not "
             "with the identity recipe as it is, and not with Hull's region gains."},
    {"key": "clean_share", "group": "How noise levels are trained", "title": "Clean-end share", "off": "",
     "control": "text", "hint": "empty = off; Fizgig's: 60:7:30:8:3, or 11:7:40:8:3 with Muon", "evidence": "mixed",
     "short": "More steps at the clean end, where faces and fine detail settle.",
     "what": "Sets, by epoch, how many of the steps land at the clean, low-noise end of training, where faces and "
             "fine detail are refined. Empty = off. Fizgig's schedule was 60:7:30:8:3; since its presets train on "
             "Muon it is 11:7:40:8:3 (the Slide Muon button).",
     "tested": "Fizgig, on MOSS-Image: 60:7:30:8:3 was the best of eight schedules, still at 71% likeness at epoch "
               "60 and about 5 points over 60:7:30:15:2. With Muon it now uses 11:7:40:8:3, which it says stopped "
               "hair darkening late in a run; that one was tuned with a face boost RulzTrainer does not have, and "
               "is not measured here. On Krea 2 here 60:7:30:8:3 tied plain: 31.2 combined, one seed of one small "
               "set.",
     "cost": "None beyond the choice."},
    {"key": "timestep_buckets", "group": "How noise levels are trained", "title": "Even noise coverage", "off": 0,
     "control": "number", "precision": 0, "hint": "0 = off; 10-20 is the usual start", "evidence": "worse",
     "short": "Spreads the noise levels evenly instead of at random.",
     "what": "Every run of N steps covers the noise curve evenly instead of drawing each level at random, so no stretch "
             "of noise levels is skipped by chance. 0 = off; 10-20 is the usual start.",
     "tested": "Lost here: 16 buckets read 26.8 combined and 10 FaceNet points under its own base on all 8 seeds, on "
               "one seed of one small Krea 2 set. Unexplained and not repeated.",
     "cost": "None."},
    {"key": "loss_type", "group": "How noise levels are trained", "title": "Band loss", "off": "mse",
     "control": "select", "options": {"mse": "mse (default)", "band": "band"}, "evidence": "worse",
     "short": "Ignores the detail the noise has already buried.",
     "what": "Leaves out of the loss the fine-detail bands the noise has buried at each step, so the model is not "
             "asked to guess detail it cannot see.",
     "tested": "Poor at 1e-4: 19.0 combined and body 4.8 on one seed of one small Krea 2 set, still rising at step "
               "2000 (24), so it may want a higher learning rate.",
     "cost": "None."},
    {"key": "race", "group": "Compare setups", "title": "Configuration race", "off": False, "control": "switch",
     "evidence": "tool",
     "short": "Trains a few versions side by side; the likeness probes pick the winner.",
     "what": "Trains several versions of this setup on the same photos and probes, each changing only what you "
             "assign it (a learning rate, the adapter, a seed, any of the extras above), and lets the likeness "
             "probes drop the weakest. The way to find out whether an untested extra helps your person. The racers "
             "and the rules appear below once it is on.",
     "tested": "A tool rather than a result: the time is predictable, for example 10 racers take about 3.1x one run "
               "when the lowest is dropped every round.",
     "cost": "More training time (shown below). No resume and no resolution step-up in a race."},
]


class Wizard:
    """Guided setup on its own tab for a character / person LoRA: six short pages with the basics filled in, every
    other setting behind the Advanced switch, and the experimental extras offered on the last page, each with what it
    does and what testing found.

    Its controls are bound to the full form's widgets (Train tab), so both views edit the same settings and a run
    started from either is the same run."""

    STEPS = (("start", "Start", "flag"), ("model", "Model", "model_training"), ("dataset", "Photos", "photo_library"),
             ("training", "Training", "school"), ("samples", "Samples & probes", "image"),
             ("extras", "Extras & start", "rocket_launch"))

    def __init__(self, f: Form, gpus: list[str], start, goto, enqueue=None):
        self.f, self.w = f, f.w
        self.gpus = gpus  # torch device names for the device select
        self.start = start  # start_job(): starts training from the form, True when it started
        self.enqueue = enqueue  # queue_job(): adds the form's run to the training queue
        self.goto = goto  # goto(tab name)
        self.goal = "character"  # the wizard sets up identity runs (recommendations and review read it)
        self.focus = "face_focus"  # a key of CHARACTER_FOCUS; follows the form (character_focus)
        self.focus_stash: dict = {}  # references and probe prompts from before a face only / body only pick
        self.adv = False  # the Advanced switch: every setting on every page
        self.conditional: list[tuple] = []  # (element, predicate): shown only while the predicate holds
        self.watched: set[str] = set()  # form keys whose changes re-check the conditional elements
        self.smi: list[dict] = []  # nvidia-smi rows (memory per GPU)
        self.data: dict | None = None  # last dataset scan
        self.recs: dict[str, tuple] = {}
        self.rec_rows: list[tuple] = []
        self.focus_cards: dict[str, ui.card] = {}
        self.model_cards: dict[str, ui.card] = {}
        self.extras_cards: dict[bool, ui.card] = {}
        self.steps: dict[str, ui.step] = {}
        self.set_up = False  # the identity recipe and the wizard's basics were put in place this session
        # the race: how many racers, the settings they may differ in, and each racer's name and assigned values
        # (a value only once it is assigned: the others follow the base setup on the earlier pages)
        self.n_racers = 2
        self.race_keys: list[str] = ["learning_rate"]
        self.racer_vals: list[dict] = []
        self.racers_loaded = False  # the Train tab's race rows were read in once

    # ------------------------------------------------------------ building blocks

    def bind(self, widget, key: str):
        """Bind a wizard control to the form widget `key` and keep the recommendations current."""
        widget.bind_value(self.w[key], "value")
        widget.on_value_change(lambda _: self.refresh())
        return widget

    def show_when(self, element, when, adv: bool = False, show_if=None):
        """Show an element only while `when()` holds, the Advanced switch is on (adv) and show_if=(form key,
        predicate) holds for that widget's value."""
        conds = [c for c in (when, (lambda: self.adv) if adv else None) if c is not None]
        if show_if is not None:
            key, pred = show_if
            conds.append(lambda: pred(self.w[key].value))
            if key not in self.watched:
                self.watched.add(key)
                self.w[key].on_value_change(lambda _: self.update_visibility())
        if conds:
            pred_all = lambda: all(c() for c in conds)  # noqa: E731
            self.conditional.append((element, pred_all))
            element.visible = bool(pred_all())
        return element

    def update_visibility(self):
        for element, when in self.conditional:
            element.visible = bool(when())

    @contextmanager
    def setting(self, key: str | None, title: str, text: str, show_if=None, when=None, adv: bool = False):
        """One setting: title, explanation and recommendation on the left, the control (built in the block) on the
        right. show_if=(form key, predicate) shows it only while the predicate holds for that widget's value; when
        is a no-argument predicate re-read on every refresh (for conditions on more than one setting); adv shows it
        only with the Advanced switch on. Yields the setting's box."""
        box = ui.element("div").classes("rz-setting")
        self.show_when(box, when, adv, show_if)
        with box:
            with ui.column().classes("gap-1 min-w-0"):
                with ui.row().classes("items-center gap-2"):
                    ui.label(title).classes("rz-setting-title")
                    if adv:
                        ui.label("Advanced").classes("rz-badge muted")
                ui.label(text).classes("rz-hint")
                if key:
                    self.rec_row(key)
            with ui.column().classes("gap-2 w-full min-w-0 justify-center"):
                yield box

    def rec_row(self, key: str):
        with ui.row().classes("rz-rec items-start gap-2 no-wrap") as row:
            icon = ui.icon("lightbulb").classes("mt-[2px] shrink-0")
            label = ui.label("").classes("min-w-0")
            btn = ui.button("Use", on_click=lambda: self.apply([key]), color=None) \
                .props("outline dense size=sm no-caps").classes("shrink-0 px-2") \
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
                ui.button("Back", icon="arrow_back", on_click=lambda: self.move(-1)).props("flat")
            ui.space()
            ui.button("Next", on_click=lambda: self.move(1)).props("unelevated icon-right=arrow_forward")

    def racing(self) -> bool:
        return bool(self.w["race"].value)

    def move(self, d: int):
        names = [n for n, _, _ in self.STEPS]
        cur = self.stepper.value if isinstance(self.stepper.value, str) else names[0]
        i = names.index(cur) if cur in names else 0
        self.stepper.value = names[max(0, min(len(names) - 1, i + d))]

    def heading(self, title: str, text: str):
        ui.label(title).classes("rz-step-title")
        hint(text)

    # ------------------------------------------------------------ recommendations

    def is_character(self) -> bool:
        return True

    def has_face(self) -> bool:
        return self.focus != "body_only"

    def has_body(self) -> bool:
        return self.focus != "face_only"

    def is_lora(self) -> bool:
        return not self.w["finetune"].value and self.w["network_type"].value == "lora"

    def refresh(self):
        try:
            cfg = self.f.to_config()
        except (TypeError, ValueError):
            return  # a field is mid-edit (e.g. an empty number); keep the last recommendations
        if cfg.preset == "identity":
            self.focus = character_focus(cfg)
        self.mark_focus()
        self.update_visibility()
        if self.racing():
            self.estimate_race()
            self.check_race()
        self.recs = wizard_recommendations(cfg, self.goal, self.smi, self.data, self.focus)
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
        self.mark_extras()
        if hasattr(self, "refs_note"):
            n_face, n_body = len(cfg.face_refs) if self.has_face() else 0, len(cfg.body_refs) if self.has_body() else 0
            self.refs_note.text = ("Likeness references: your photos in " + ", ".join(
                Path(p).name for p in dict.fromkeys(cfg.face_refs + cfg.body_refs)) + ". The likeness checks compare "
                "with them (Advanced to change them)." if n_face or n_body else
                "Pick your photos above: they also become the likeness references the checks compare with.")

    def apply(self, keys: list[str]):
        for key in keys:
            rec = self.recs.get(key)
            if rec is not None:
                self.w[key].value = list(rec[0]) if isinstance(rec[0], list) else rec[0]
            self.refresh()  # later recommendations can depend on earlier ones (save every on steps)

    def set_up_identity(self, force: bool = False):
        """The identity recipe and the wizard's basics (WIZARD_BASICS, the experimental extras off, the
        recommendations that need no choosing: hardware, resolution, references, samples). Once per session unless
        forced (Reset), and only on a form without the identity preset, so a loaded identity config is kept."""
        if self.set_up and not force:
            return
        self.set_up = True
        if self.w["preset_toggle"].value and not force:
            self.refresh()
            return
        if not self.w["preset_toggle"].value:
            self.w["preset_toggle"].value = True  # applies the identity recipe (the Train tab's switch)
        for key, value in WIZARD_BASICS.items():
            self.w[key].value = value
        self.experiments_off()
        self.apply_focus()
        if not (self.w["sample_prompts"].value or "").strip():
            self.w["sample_prompts"].value = "\n".join(suggested_prompts(
                self.goal, (self.w["trigger_word"].value or "").strip(), self.focus))
        self.refresh()
        self.apply([k for k in WIZARD_AUTO if k in self.recs])
        ui.notify("Set up for a character / person LoRA with the recommended settings", type="info")

    def reset(self):
        self.set_up_identity(force=True)
        self.extras_answer(False, quiet=True)

    # ------------------------------------------------------------ steps

    def build(self):
        with ui.column().classes("rz-page gap-4"):
            with card():
                with ui.row().classes("w-full items-center gap-4"):
                    with ui.column().classes("gap-1 flex-1 min-w-[16rem]"):
                        ui.label("Character / person LoRA wizard").classes("rz-title")
                        hint("Six short pages with the basics already filled in for you. Advanced shows every "
                             "setting on every page; the last page offers the experimental extras. These are the "
                             "same settings as the Train tab's, which has every option.")
                    self.adv_switch = ui.switch("Advanced", value=self.adv, on_change=lambda e: self.set_adv(e.value)) \
                        .tooltip("Show every setting on every page (for experienced users)")
                    ui.button("Reset", icon="restart_alt", on_click=self.reset).props("flat") \
                        .tooltip("Put the recommended setup back: the identity recipe, rank 32, adamw8bit at 1e-4, "
                                 "experimental extras off")
                    ui.button("All settings", icon="tune", on_click=lambda: self.goto("Train")).props("outline")
            with ui.stepper(on_value_change=self.on_step).props("header-nav alternative-labels animated flat") \
                    .classes("rz-stepper w-full") as self.stepper:
                for name, title, icon in self.STEPS:
                    with ui.step(name, title=title, icon=icon).props(f"active-icon={icon}") as step:
                        self.steps[name] = step
                        getattr(self, f"step_{name}")()
        self.w["preset_toggle"].on_value_change(lambda _: self.refresh())
        self.w["model_preset"].on_value_change(lambda _: (self.mark_model(), self.refresh()))
        self.w["race"].on_value_change(lambda _: (self.mark_race(), self.refresh()))
        self.mark_model()
        self.mark_race()
        self.refresh()
        ui.timer(0.2, self.detect_gpus, once=True)

    def set_adv(self, on: bool):
        self.adv = bool(on)
        self.update_visibility()

    async def on_step(self, e):
        name = e.value if isinstance(e.value, str) else e.value.props.get("name")
        names = [s[0] for s in self.STEPS]
        idx = names.index(name) if name in names else 0
        for i, n in enumerate(names):
            self.steps[n].props(add="done") if i < idx else self.steps[n].props(remove="done")
        if name != "start":
            self.set_up_identity()
        if name == "dataset" and self.data is None and any(Path(p).is_dir() for p in self.w["datasets"].selected):
            await self.scan()
        if name == "extras":
            self.load_racers()
            self.rebuild_racers()  # the base setup may have changed on the earlier pages
            self.estimate_race()
            await self.render_review()

    # ---- 1 start
    def step_start(self):
        self.heading("Who are you training?", "A LoRA that draws one particular person or character whenever its "
                                               "trigger word is in the prompt. Give the run a name and a trigger "
                                               "word, and pick what the likeness should cover.")
        with self.setting(None, "Job name", "The name of this run. The LoRA files are saved in output/<job name>/."):
            self.bind(ui.input("Job name"), "job_name").classes("w-full")
        with self.setting(None, "Trigger word",
                          "A word you put in your prompts to call the person up, added to the front of every caption. "
                          "Pick one the model does not know, such as 'zxq woman'. Leave it empty if your captions "
                          "already have one."):
            self.bind(ui.input("Trigger word"), "trigger_word").classes("w-full")
            self.w["trigger_word"].on_value_change(lambda _: self.trigger_changed())
        ui.label("What should the likeness cover?").classes("rz-setting-title pt-2")
        with ui.element("div").classes("rz-choices"):
            for key, (icon, title, text) in CHARACTER_FOCUS.items():
                with ui.card().props("flat").classes("rz-choice") as c:
                    with ui.row().classes("w-full items-center gap-2 no-wrap"):
                        ui.icon(icon).classes("rz-choice-icon")
                        ui.label(title).classes("rz-setting-title")
                        if key == "face_focus":
                            ui.label("Recommended").classes("rz-badge")
                        ui.space()
                        ui.icon("check_circle").classes("rz-check")
                    ui.label(text).classes("rz-hint")
                c.on("click", lambda _, k=key: self.set_focus(k))
                self.show_when(c, lambda k=key: k in WIZARD_FOCUS or self.adv or k == self.focus)
                self.focus_cards[key] = c
        self.show_when(hint("Advanced also offers body focus and body only."), lambda: not self.adv)
        self.nav(back=False)

    def trigger_changed(self):
        """The probe prompts name the subject: follow a new trigger word while they are the recipe's own."""
        if not self.set_up or not self.w["preset_toggle"].value:
            return
        face_p, body_p = PRESET_PROBE_PROMPTS["identity"]
        for key, template in (("probe_face_prompt", face_p), ("probe_body_prompt", body_p)):
            cur = (self.w[key].value or "").strip()
            if cur and cur.startswith(template.split("{trigger}")[0]):
                self.w[key].value = template.format(trigger=(self.w["trigger_word"].value or "").strip()
                                                    or "the character")

    def set_focus(self, focus: str):
        """One of the five focuses: the probes, references, crops, loss weights and score weights for it."""
        self.set_up_identity()
        for key in ("face_refs", "body_refs", "probe_face_prompt", "probe_body_prompt"):
            if (self.w[key].value or "").strip():  # what the user has, kept for when the focus needs it again
                self.focus_stash[key] = self.w[key].value
        self.focus = focus
        self.apply_focus()
        self.refresh()

    def apply_focus(self):
        values = focus_settings(self.focus, (self.w["trigger_word"].value or "").strip())
        for key, needed in (("face_refs", self.focus != "body_only"), ("body_refs", self.focus != "face_only"),
                            ("probe_face_prompt", self.focus != "body_only"),
                            ("probe_body_prompt", self.focus != "face_only")):
            if needed and (self.focus_stash.get(key) or "").strip():
                values[key] = self.focus_stash[key]  # the user's own, from before it was emptied
        for key, value in values.items():
            self.w[key].value = "\n".join(value) if isinstance(value, list) else value

    def mark_focus(self):
        for key, c in self.focus_cards.items():
            c.classes(add="selected") if key == self.focus else c.classes(remove="selected")

    # ---- 2 model
    def step_model(self):
        self.heading("Which model?", "The base model the LoRA is made for: it only works with that model. The files "
                                     "download on the first run.")
        with ui.element("div").classes("rz-choices"):
            for name, (badge, text) in [*MODEL_NOTES.items(), (CUSTOM, ("", "Any local diffusers folder "
                                                                             "(transformer/, text_encoder/, vae/) or "
                                                                             "Hugging Face repo id."))]:
                with ui.card().props("flat").classes("rz-choice") as c:
                    with ui.row().classes("w-full items-center gap-2"):
                        ui.label(name).classes("rz-setting-title")
                        if badge:
                            ui.label(badge).classes("rz-badge" + ("" if badge == "Recommended" else " muted"))
                        ui.space()
                        ui.icon("check_circle").classes("rz-check")
                    ui.label(text).classes("rz-hint")
                c.on("click", lambda _, n=name: setattr(self.w["model_preset"], "value", n))
                if name == CUSTOM or badge == "Not for training":  # Advanced only (unless it is the one picked)
                    self.show_when(c, lambda n=name: self.adv or self.w["model_preset"].value == n)
                self.model_cards[name] = c
        with self.setting(None, "Custom model", "A local folder in diffusers layout, or a Hugging Face repo id "
                                                "such as OpenMOSS-Team/MOSS-Image-Base.",
                          show_if=("model_preset", lambda p: p == CUSTOM)):
            self.bind(ui.input("Path or repo id"), "model_path").classes("w-full")
        with self.setting(None, "Where the model comes from",
                          "Single files (the default) downloads Comfy-Org's ComfyUI files for the model: the diffusion "
                          "model and text encoder in the format you pick (int8 ConvRot: about half of bf16) and the "
                          "VAE, or uses files you already have. Hugging Face downloads the original diffusers repo "
                          "into models/ instead.", adv=True):
            self.bind(ui.radio([FILE_SOURCE, HF_SOURCE]), "model_source")
        with self.setting(None, "Download and check",
                          "Fetches what is missing and checks the files already there. Training would also do this on "
                          "start, but doing it now shows problems (gated repo, no login, disk space) before you start.",
                          adv=True, show_if=("model_source", lambda x: x == HF_SOURCE)):
            status = hint("")

            async def download():
                status.text = "Downloading / verifying... (about 30 GB on the first download)"
                status.text = await run.io_bound(download_model, self.f.to_config().model_path)

            ui.button("Download / verify", icon="download", on_click=download).props("outline")
        with self.setting(None, "File formats to download",
                          "int8 ConvRot is about half the size of bf16, trains as fast and loses under 1% accuracy (the "
                          "default). Krea 2 also offers fp8, Ideogram 4 fp8 and nvfp4.",
                          adv=True, show_if=("model_source", lambda x: x == FILE_SOURCE)):
            self.f.quant_copies["model_file_quant"] = self.bind(
                ui.select(file_quant_options(preset_arch("MOSS-Image-Base"), "model"), label="Diffusion model"),
                "model_file_quant").classes("w-full")
            self.f.quant_copies["text_encoder_file_quant"] = self.bind(
                ui.select(file_quant_options(preset_arch("MOSS-Image-Base"), "text"), label="Text encoder"),
                "text_encoder_file_quant").classes("w-full")
        with self.setting(None, "Download now",
                          "Training downloads the files on start; this does it now, so a problem (disk space, network) "
                          "shows before you start. Hugging Face's cache keeps them for later runs.",
                          adv=True, show_if=("model_source", lambda x: x == FILE_SOURCE)):
            files_status = hint("")

            async def download_now():
                files_status.text = "Downloading / checking the model files..."
                files_status.text = await run.io_bound(download_files, self.f.to_config())

            ui.button("Download now", icon="download", on_click=download_now).props("outline")
        for key, title, text, kind in (
                ("transformer_file", "Diffusion model file", "A ComfyUI models/diffusion_models file you already "
                                                             "have.", "file"),
                ("text_encoder_file", "Text encoder file", "Qwen3.5-9B for MOSS-Image, Qwen3-VL-4B for Krea 2, FIBO 2.0 "
                                                           "and FLUX 3 Klein, Qwen3-8B for FLUX.2 Klein 9B, "
                                                           "Qwen3-VL-8B for Qwen-Image 2.1 and "
                                                           "Ideogram 4, Qwen3-VL-32B (cut to 50 layers) for MiniMax "
                                                           "H3 and MiniMax-H3-Image (the same file), P-H3-Image's "
                                                           "Qwen3-VL-4B with its text connector for P-H3-Image.",
                 "file"),
                ("vae_file", "VAE file", "The Flux2 VAE for MOSS-Image, FIBO 2.0, both Kleins and Ideogram 4, the "
                                         "Qwen-Image VAE for Krea 2, its own for Qwen-Image 2.1, the video VAE for "
                                         "MiniMax H3 (its audio VAE is found next to it or downloaded), its own "
                                         "single-frame VAE for P-H3-Image, its own image VAE for MiniMax-H3-Image.",
                 "file"),
                ("tokenizer_path", "Tokenizer folder", "Left empty, the tokenizer is downloaded once (a few MB).",
                 "dir")):
            with self.setting(None, title, text + " Optional: left empty, it is downloaded.", adv=True,
                              show_if=("model_source", lambda x: x == FILE_SOURCE)):
                self.bind(path_field(title, kind, width="w-full"), key)
        self.nav()

    def mark_model(self):
        for name, c in self.model_cards.items():
            c.classes(add="selected") if name == self.w["model_preset"].value else c.classes(remove="selected")

    # ---- 3 photos
    def step_dataset(self):
        self.heading("Your photos", "15-50 sharp photos of the person, from different days, angles and distances, "
                                    "work best. Each can have a same-name .txt caption.")
        with self.setting(None, "Datasets", "The folders of photos to train on. Create them, upload photos and write "
                                            "captions on the Dataset tab."):
            picker = self.w["datasets"]

            def choices():
                return {p: Path(p).name or p for p in picker.paths()}

            chosen = ui.select(choices(), multiple=True, value=picker.selected, label="Datasets") \
                .props("use-chips").classes("w-full")
            chosen.bind_value(picker, "selected")
            chosen.on_value_change(lambda _: (self.forget_scan(), self.refresh(), self.fill_refs()))
            picker.listeners.append(lambda: chosen.set_options(choices(), value=picker.selected))
            with ui.row().classes("gap-2"):
                ui.button("Check my photos", icon="search", on_click=self.scan).props("unelevated")
                ui.button("Dataset tab", icon="photo_library", on_click=lambda: self.goto("Dataset")).props("flat")
        self.scan_box = ui.column().classes("w-full gap-3")
        with self.setting(None, "Captions",
                          "Optional. Describe what should stay changeable: clothes, background, pose, lighting. Do not "
                          "describe the face: what you leave out is what the trigger word learns."):
            ui.button("Edit captions", icon="edit_note", on_click=lambda: self.goto("Dataset")).props("outline")
        refs_note = hint("")
        self.refs_note = refs_note
        self.show_when(refs_note, lambda: not self.adv)
        with self.setting(None, "Ideogram 4 captions",
                          "Ideogram 4 was trained on structured JSON captions only. A same-name .json caption in its "
                          "schema (Ideogram's prompting guide) trains it best and wins over the .txt; a plain .txt "
                          "caption is wrapped into the smallest valid JSON caption, with the trigger word in its "
                          "subject element. JSON captions can be long: Max caption tokens 2048 keeps them whole.",
                          show_if=("model_preset",
                                   lambda p: preset_arch(p, self.w["model_path"].value) == "ideogram4")):
            pass
        adv_keys = ["resolutions", "enable_buckets", "flip_augment", "crop_augment", "face_refs", "body_refs",
                    "look_warmup"]
        self.show_when(ui.column().classes("w-full"), None, adv=True)  # keeps the bar below out of basic mode
        with ui.column().classes("w-full gap-4") as bar:
            self.apply_bar(adv_keys)
        self.show_when(bar, None, adv=True)
        with self.setting("face_refs", "Face references",
                          "Photos whose face is the identity to match. Samples and probes are compared with them "
                          "(ArcFace; 50%+ generally means the same person), which picks the best checkpoint and stops "
                          "early. A folder counts as all its photos; you can also tick Face ref on the Dataset tab.",
                          when=self.has_face, adv=True):
            self.bind(ui.textarea("Face references (one path per line)"), "face_refs").classes("w-full")
        with self.setting("body_refs", "Body references",
                          "Photos that show the person's build, from the neck to mid-thigh or more. A folder counts as "
                          "all its photos (those without a usable view of the body are skipped).",
                          when=self.has_body, adv=True):
            self.bind(ui.textarea("Body references (one path per line)"), "body_refs").classes("w-full")
        with self.setting("resolutions", "Training resolution",
                          "The image area training works at, in megapixels (1 MP = 1024x1024), with each photo's aspect "
                          "ratio kept. Higher learns finer detail but takes more memory and time.", adv=True):
            self.bind(ui.select(MP_OPTIONS, multiple=True, label="Resolutions").props("use-chips"), "resolutions") \
                .classes("w-full")
        with self.setting("enable_buckets", "Aspect-ratio buckets",
                          "Groups photos by shape and trains each at its own aspect ratio. Off, every photo is cropped "
                          "to a square.", adv=True):
            self.bind(ui.switch("Aspect-ratio buckets"), "enable_buckets")
        with self.setting("flip_augment", "Mirror images",
                          "Also trains a mirrored copy of every photo: more variety, but anything asymmetric (a hair "
                          "parting, a mole, a logo) gets blurred.", adv=True):
            self.bind(ui.switch("Horizontal flip"), "flip_augment")
        with self.setting("crop_augment", "Close-up crops",
                          "Adds crops of every photo as extra training images, so the model sees the face up close and "
                          "not only small in a full-length photo. Part of the measured recipe. Crops never upscale "
                          "past the source.", adv=True):
            self.bind(ui.select(["none", "face", "face+bust"], label="Crops"), "crop_augment").classes("w-full")
        with self.setting("look_warmup", "Warm up look outliers",
                          "Photos the Dataset tab's Look consistency filter puts under its fence (they do not look like "
                          "the rest) stay in but start at a lower loss weight and reach full weight after a few epochs.",
                          adv=True):
            self.bind(ui.switch("Warm up look outliers"), "look_warmup")
        self.nav()

    def fill_refs(self):
        """New datasets and no references of your own yet: the datasets become the references (the photos with a
        face, and those showing the body)."""
        if not self.set_up:
            return
        keys = [k for k in ("face_refs", "body_refs") if not (self.w[k].value or "").strip() and k in self.recs
                and self.recs[k][0]]
        self.apply(keys)

    def forget_scan(self):
        if self.data is not None and self.data.get("folder") != "|".join(self.w["datasets"].selected):
            self.data = None
            self.scan_box.clear()
            self.refresh()

    async def scan(self):
        paths = self.w["datasets"].selected
        self.scan_box.clear()
        with self.scan_box:
            ui.spinner(size="sm")
        self.data = await run.io_bound(datasets_summary, paths)
        thumbs = await run.io_bound(summary_thumbs, self.data) if self.data else []
        self.scan_box.clear()
        with self.scan_box:
            if self.data is None:
                ui.label("No dataset picked yet: pick one above, or create one on the Dataset tab.") \
                    .classes("rz-banner error")
            else:
                if self.data["missing"]:
                    ui.label(f"Folder not found: {', '.join(self.data['missing'])}").classes("rz-banner error")
                n, cap, mp = self.data["images"], self.data["captioned"], self.data["median_mp"]
                with ui.row().classes("w-full gap-3"):
                    for label, value, sub in (("Photos", str(n), "png, jpg, webp, bmp"),
                                              ("Captioned", f"{cap} / {n}", ".txt or .json next to the photo"),
                                              ("Typical size", f"{mp:.2f} MP" if mp else "-", "median image area")):
                        v, s = stat_tile(label)
                        v.text, s.text = value, sub
                if not n:
                    ui.label("No photos in these datasets yet. Add them on the Dataset tab (upload them, or copy a "
                             "folder), then check again.").classes("rz-banner warning")
                elif n < 10:
                    ui.label(f"Only {n} photos: 15 or more, from different days and angles, give a much steadier "
                             "likeness.").classes("rz-banner warning")
                elif cap < n:
                    ui.label(f"{n - cap} photos have no caption yet. They train on the trigger word alone, which works "
                             "for a quick character LoRA; captions give a more flexible one.") \
                        .classes("rz-banner info")
                if thumbs:
                    with ui.row().classes("w-full gap-2"):
                        for t in thumbs:
                            ui.image(image_url(t)).classes("w-24 h-24 rz-thumb")
                if n:
                    with ui.row().classes("w-full items-center gap-3 no-wrap"):
                        if self.data["outliers"]:
                            ui.label(f"The saved look scans put {self.data['outliers']} of {n} photos under the fence: "
                                     "they do not look like the rest. Review them on the Dataset tab (Look consistency "
                                     "filter).").classes("rz-banner warning")
                        elif self.data["scanned"]:
                            ui.label("The saved look scans found no outliers.").classes("rz-banner ok")
                        else:
                            ui.label("Tip: the Dataset tab's Look consistency filter finds photos whose face or body "
                                     "does not match the rest.").classes("rz-banner info")
                        ui.button("Dataset tab", icon="photo_library", on_click=lambda: self.goto("Dataset")) \
                            .props("outline dense no-caps").classes("shrink-0")
        self.refresh()
        self.apply([k for k in ("steps", "resolutions", "look_warmup") if k in self.recs and not self.adv])

    # ---- 4 training
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

    def step_training(self):
        self.heading("How to train", "How long and how strongly to train. The values here are a good start for most "
                                     "people; the likeness probes keep the best checkpoint either way.")
        with self.setting("steps", "Training steps",
                          "Each step trains on one photo. Too few and the LoRA is weak; too many and it starts copying "
                          "the photos. Checkpoints are saved along the way and the best one is kept, so a few too many "
                          "is safe."):
            self.bind(ui.number("Steps", min=1, precision=0), "steps").classes("w-full")
        with self.setting(None, "Learning rate",
                          "How big each update is. 1e-4 (0.0001) is the wizard's default. Too high gives burnt, noisy "
                          "samples; too low learns slowly."):
            self.bind(ui.number("Learning rate", format="%.1e", step=1e-5, min=0), "learning_rate").classes("w-full")
        with self.setting(None, "LoRA rank",
                          "The size of the LoRA. 32 (the default) has room for a face's detail; higher mostly adds file "
                          "size and copies the photos more easily. Alpha follows the rank unless you set it under "
                          "Advanced.", when=self.is_lora):
            rank = self.bind(ui.number("Rank", min=1, max=256, precision=0), "lora_rank").classes("w-full")
            rank.on_value_change(lambda e: self.follow_rank(e.value))
        with self.setting(None, "LoRA alpha",
                          "Scales the LoRA's effect (alpha / rank). Equal to the rank is the safe default.",
                          when=self.is_lora, adv=True):
            self.bind(ui.number("Alpha", min=0), "lora_alpha").classes("w-full")
        with self.setting("batch_size", "Batch size",
                          "Photos per step. Larger batches need more memory and, for LoRAs, rarely help.", adv=True):
            self.bind(ui.number("Batch size", min=1, precision=0), "batch_size").classes("w-full")
        is_krea = ("model_preset", lambda p: preset_arch(p, self.w["model_path"].value) == "krea2")
        with self.setting("target_text_fusion", "Train Krea's text fusion",
                          "Krea 2 mixes 12 layers of its text encoder in four text-fusion blocks before the main "
                          "model. The usual LoRA leaves them under-trained: boosting them on finished LoRAs lifted "
                          "likeness 2-7 points in Fizgig's tests. Adds about 8.9M LoRA weights at rank 16.",
                          show_if=is_krea, adv=True):
            self.bind(ui.switch("Text fusion"), "target_text_fusion")
        with self.setting("target_attn_gate", "Train the attention gate",
                          "Every Krea 2 attention has a sigmoid gate on its output. Krea's own LoRAs train it.",
                          show_if=is_krea, adv=True):
            self.bind(ui.switch("Attention gate"), "target_attn_gate")
        is_qwen = ("model_preset", lambda p: preset_arch(p, self.w["model_path"].value) in ("qwen_image21",
                                                                                              "minimax_h3",
                                                                                              "p_h3_image",
                                                                                              "minimax_h3_image"))
        with self.setting("adapter_mode", "Training adapter",
                          "A small LoRA that stays on while your LoRA trains and off for samples and probes, and is "
                          "never saved into it. Fizgig measured on Qwen-Image 2.1: without one, LoRAs collapse into "
                          "texture or wobble part-way; with it likeness reached 77 against 56. MiniMax H3's is "
                          "circlestone-labs' image training adapter; P-H3-Image's and MiniMax-H3-Image's are ostris' "
                          "(the ones ai-toolkit uses).",
                          show_if=is_qwen, adv=True):
            self.bind(ui.select(ADAPTER_MODES, label="Training adapter"), "adapter_mode").classes("w-full")
        hw = ui.column().classes("w-full gap-3 pt-2")
        self.show_when(hw, None, adv=True)
        with hw:
            ui.label("Your GPUs").classes("rz-setting-title")
            hint("Filled in from your GPUs when the wizard sets up the run; change them here if you need to.")
            self.gpu_box = ui.row().classes("w-full gap-3")
            with self.gpu_box:
                ui.spinner(size="sm")
            self.apply_bar(["multi_gpu", "dp_balance", "transformer_quant", "text_encoder_quant",
                            "gradient_checkpointing", "checkpoint_keep_blocks", "block_swap", "compile_model"])
        self.hardware_settings()
        self.nav()

    def follow_rank(self, rank):
        if not self.adv and rank:
            self.w["lora_alpha"].value = float(rank)

    def hardware_settings(self):
        with self.setting("multi_gpu", "Multi-GPU",
                          "Data parallel puts a full model copy on each GPU, each trains its own images, and the "
                          "results are averaged: about 1.7-2x the speed with two cards (each must fit the model). "
                          "Model split divides the model's blocks over the GPUs: half the memory each, not faster.",
                          adv=True):
            self.bind(ui.select(dict(self.w["multi_gpu"].options), label="Multi-GPU"), "multi_gpu").classes("w-full")
        with self.setting(None, "Training GPU", "The GPU that trains. Pick the one with the most memory (listed in "
                                                "CUDA order).", adv=True, show_if=("multi_gpu", lambda x: x == "off")):
            self.bind(ui.select(self.gpus, label="Training GPU"), "device").classes("w-full")
        with self.setting(None, "GPUs to use", "Comma-separated, the first is the main GPU (it also saves and "
                                               "samples).", adv=True, show_if=("multi_gpu", lambda x: x != "off")):
            self.bind(ui.input("Devices"), "devices").classes("w-full")
        with self.setting("dp_balance", "Balance unequal GPUs",
                          "With two different cards the faster one waits for the slower one every step. auto measures "
                          "them during the first steps and splits each step's images by speed (it needs a few images "
                          "per step: gradient accumulation 3 or more). grow lets the faster card train more images in "
                          "the same time instead.", adv=True, show_if=("multi_gpu", lambda x: x == "data_parallel")):
            self.bind(ui.select(dict(self.w["dp_balance"].options), label="Balance"), "dp_balance").classes("w-full")
        with self.setting("transformer_quant", "Model quantization",
                          "Stores the frozen base model in fewer bits to save memory; the LoRA itself always trains in "
                          "full precision. auto keeps the model as it is. int8_convrot halves it at under 1% error and "
                          "is as fast as bf16 or faster; nvfp4 quarters it but is slower.", adv=True):
            self.bind(ui.select(QUANTS, label="Diffusion model quant"), "transformer_quant").classes("w-full")
        with self.setting("text_encoder_quant", "Text encoder quantization",
                          "The text encoder only runs once before training, to turn your captions into embeddings, and "
                          "is unloaded afterwards. Quantizing it lets it fit on the GPU, so that step is quick.",
                          adv=True):
            self.bind(ui.select(QUANTS, label="Text encoder quant"), "text_encoder_quant").classes("w-full")
        with self.setting("gradient_checkpointing", "Gradient checkpointing",
                          "Recomputes intermediate results during the backward pass instead of keeping them in memory: "
                          "far less memory for some extra time per step.", adv=True):
            self.bind(ui.switch("Gradient checkpointing"), "gradient_checkpointing")
        with self.setting("checkpoint_keep_blocks", "Blocks that skip the recompute",
                          "Spends spare GPU memory on speed: these blocks keep their intermediate results instead of "
                          "recomputing them, about a third faster each. -1 measures your largest images at the start "
                          "and keeps as many as fit; 0 recomputes every block.",
                          adv=True, show_if=("gradient_checkpointing", bool)):
            self.bind(ui.number("Kept blocks (-1 = auto)", min=-1, precision=0), "checkpoint_keep_blocks") \
                .classes("w-full")
        with self.setting("block_swap", "Block swap",
                          "For a card the model does not fit: this many blocks keep their frozen weights in system RAM "
                          "and stream to the GPU as each one runs. Measured on Krea 2 int8 (16 GB card): 14 of 28 "
                          "blocks swapped cut the step's peak from 10.8 to 7.7 GB at about the same speed. Needs "
                          "gradient checkpointing and one GPU. 0 = off.", adv=True,
                          when=lambda: self.w["multi_gpu"].value == "off" and not self.w["finetune"].value):
            self.bind(ui.number("Blocks swapped (0 = off)", min=0, precision=0), "block_swap").classes("w-full")
        with self.setting("compile_model", "Compile the model",
                          "torch.compile fuses the small steps between the big matrix multiplies: measured 26% faster "
                          "per step on MOSS-Image and 12% on Krea 2, after a warm-up of under a minute. auto does it "
                          "for runs of 100 steps or more.", adv=True):
            self.bind(ui.select(COMPILE_CHOICES, label="Compile"), "compile_model").classes("w-full")

    # ---- 5 samples and probes
    def step_samples(self):
        self.heading("Samples and likeness checks", "Preview pictures show how training is going, and quick likeness "
                                                    "checks (probes) score each checkpoint against your photos, keep "
                                                    "the best one and stop when it no longer improves.")
        with self.setting(None, "Sample prompts",
                          "One prompt per line, drawn at every sample step with a fixed seed, so the pictures compare "
                          "over time. Use your trigger word and vary the scene to see that the person carries over to "
                          "new situations."):
            self.bind(ui.textarea("Sample prompts (one per line)"), "sample_prompts").classes("w-full")
            ui.button("Suggest prompts", icon="lightbulb", on_click=self.suggest).props("flat")
        with self.setting("sample_every", "Preview pictures every",
                          "Steps between sample sets (0 = never). Training pauses while they render; they show on the "
                          "Monitor and Samples tabs."):
            self.bind(ui.number("Sample every", min=0, precision=0, suffix="steps"), "sample_every").classes("w-full")
        with self.setting("save_every", "Save a checkpoint every",
                          "Steps between saved LoRAs. Several of them let you pick a different one afterwards; the best "
                          "by likeness is saved as <job>_best.safetensors anyway."):
            self.bind(ui.number("Save every", min=1, precision=0, suffix="steps"), "save_every").classes("w-full")
        with self.setting("probe_every", "Likeness check every",
                          "A quick, small headshot (and full-body shot) every N steps, scored against your photos. "
                          "Frequent checks pick the best checkpoint more precisely; each takes a few seconds."):
            self.bind(ui.number("Probe every", min=0, precision=0, suffix="steps"), "probe_every").classes("w-full")
        adv_keys = ["sample_model_path", "keep_checkpoints", "sample_steps", "sample_cfg", "probe_steps", "probe_cfg",
                    "probe_face_prompt", "probe_body_prompt"]
        with ui.column().classes("w-full") as bar:
            self.apply_bar(adv_keys)
        self.show_when(bar, None, adv=True)
        with self.setting(None, "Early stop", "Stop after this many likeness checks in a row without improvement "
                                              "(0 = never: run all the steps). The best checkpoint is kept either way.",
                          adv=True):
            self.bind(ui.number("Checks without improvement", min=0, precision=0), "likeness_patience") \
                .classes("w-full")
        with self.setting("keep_checkpoints", "Checkpoints to keep",
                          "Keeps only the newest N of the step checkpoints, deleting the oldest as new ones are saved. "
                          "The final file and the best-likeness file are never deleted. 0 keeps every one.", adv=True):
            self.bind(ui.number("Keep", min=0, precision=0), "keep_checkpoints").classes("w-full")
        with self.setting("probe_face_prompt", "Face probe prompt",
                          "A plain headshot of your character, rendered with a fixed seed. Empty = no face probe.",
                          when=self.has_face, adv=True):
            self.bind(ui.textarea("Face probe prompt").props("autogrow"), "probe_face_prompt").classes("w-full")
        with self.setting("probe_body_prompt", "Body probe prompt",
                          "A full-body, upright shot, for the body score. Empty = no body probe.",
                          when=self.has_body, adv=True):
            self.bind(ui.textarea("Body probe prompt").props("autogrow"), "probe_body_prompt").classes("w-full")
        is_krea = ("model_preset", lambda p: preset_arch(p, self.w["model_path"].value) == "krea2")
        with self.setting("sample_model_path", "Preview with Krea 2 Turbo",
                          "Renders previews and probes with Krea-2-Turbo carrying your LoRA, while training stays on "
                          "Raw: Turbo needs 8 steps without guidance where Raw needs 28 or more with it. Empty = "
                          "preview on Raw.", show_if=is_krea, adv=True):
            self.bind(ui.input("Sampling model (repo id or folder)"), "sample_model_path").classes("w-full")
        with self.setting(None, "Turbo LoRA",
                          "A step-distilled LoRA used only while rendering samples and probes, never in training: "
                          "previews take seconds. The model's own is downloaded (MOSS-Image: 12 steps at CFG 1; Krea 2, "
                          "Qwen-Image 2.1, Ideogram 4 and MiniMax H3: 8; FIBO 2.0: 4; the Kleins have none) and the "
                          "steps and guidance below follow it.", adv=True):
            self.bind(ui.select(TURBO_MODES, label="Turbo LoRA"), "turbo_mode").classes("w-full")
            self.bind(path_field("Turbo LoRA file", "file", width="w-full"), "turbo_lora") \
                .parent_slot.parent.bind_visibility_from(self.w["turbo_mode"], "value", lambda x: x == "file")
        with self.setting("sample_steps", "Sampling steps", "Denoising steps per preview picture. More is slower and a "
                                                            "little cleaner.", adv=True):
            self.bind(ui.number("Sample steps", min=1, precision=0), "sample_steps").classes("w-full")
        with self.setting("sample_cfg", "Guidance (CFG)", "How strictly previews follow the prompt.", adv=True):
            self.bind(ui.number("Sample CFG", min=0, step=0.5), "sample_cfg").classes("w-full")
        with self.setting("probe_steps", "Probe steps", "Denoising steps per likeness probe.", adv=True):
            self.bind(ui.number("Probe steps", min=1, precision=0), "probe_steps").classes("w-full")
        with self.setting("probe_cfg", "Probe guidance (CFG)", "Guidance for the likeness probes.", adv=True):
            self.bind(ui.number("Probe CFG", min=0, step=0.5), "probe_cfg").classes("w-full")
        self.nav()

    def suggest(self):
        replaced = bool((self.w["sample_prompts"].value or "").strip())
        self.w["sample_prompts"].value = "\n".join(suggested_prompts(self.goal, (self.w["trigger_word"].value or "")
                                                                     .strip(), self.focus))
        if replaced:
            ui.notify("Replaced the sample prompts with suggestions")

    # ---- 6 extras and start
    def step_extras(self):
        self.heading("Extras and start", "The run is ready. Experimental settings are optional extras that might help "
                                         "the likeness but are not part of the recipe yet; most have not been "
                                         "measured. You can skip them.")
        ui.label("Do you want to use any experimental settings?").classes("rz-setting-title")
        with ui.element("div").classes("rz-choices"):
            for answer, icon, title, text in (
                    (False, "verified", "No, keep it simple",
                     "Train with the recommended setup from the earlier pages. The best choice for a first LoRA."),
                    (True, "science", "Yes, show me",
                     "Pick extras to switch on. Each one says what it does and what testing found so far.")):
                with ui.card().props("flat").classes("rz-choice") as c:
                    with ui.row().classes("w-full items-center gap-2 no-wrap"):
                        ui.icon(icon).classes("rz-choice-icon")
                        ui.label(title).classes("rz-setting-title")
                        if not answer:
                            ui.label("Recommended").classes("rz-badge")
                        ui.space()
                        ui.icon("check_circle").classes("rz-check")
                    ui.label(text).classes("rz-hint")
                c.on("click", lambda _, a=answer: self.extras_answer(a))
                self.extras_cards[answer] = c
        self.extras_box = ui.column().classes("w-full gap-3")
        self.extras_box.visible = False
        with self.extras_box:
            self.build_experiments()
        ui.label("Review").classes("rz-step-title pt-4")
        self.review_box = ui.column().classes("w-full gap-3")
        with ui.row().classes("w-full items-center gap-3 pt-2"):
            self.cfg_path = ui.input("Save as", value="").classes("w-80 max-w-full")
            ui.button("Save config", icon="save", on_click=self.save).props("outline")
            ui.space()
            if self.enqueue is not None:
                ui.button("Add to queue", icon="playlist_add", on_click=self.enqueue).props("outline size=lg") \
                    .tooltip("Start this run after the current one (and any queued before it) is done")
            self.start_btn = ui.button("Start the race" if self.racing() else "Start training", icon="play_arrow",
                                       on_click=self.start).props("unelevated size=lg")
        with ui.stepper_navigation():
            ui.button("Back", icon="arrow_back", on_click=lambda: self.move(-1)).props("flat")

    def experiment_on(self, ex: dict) -> bool:
        return not _same(self.w[ex["key"]].value, ex["off"])

    def experiments_off(self):
        for ex in EXPERIMENTS:
            self.w[ex["key"]].value = ex["off"]

    def extras_answer(self, yes: bool, quiet: bool = False):
        self.extras_yes = yes
        if not yes:
            changed = [ex["title"] for ex in EXPERIMENTS if self.experiment_on(ex)]
            self.experiments_off()
            if changed and not quiet:
                ui.notify(f"Turned off: {', '.join(changed)}", type="info")
        self.mark_extras()
        self.refresh()
        background_tasks.create(self.render_review(), name="wizard review")

    def mark_extras(self):
        if not hasattr(self, "extras_box"):
            return
        yes = getattr(self, "extras_yes", None)
        if yes is None:  # not answered yet: yes when something experimental is already on (a loaded config)
            yes = any(self.experiment_on(ex) for ex in EXPERIMENTS)
        for answer, c in self.extras_cards.items():
            c.classes(add="selected") if answer == yes else c.classes(remove="selected")
        self.extras_box.visible = yes
        for ex, badge in getattr(self, "experiment_badges", []):
            on = self.experiment_on(ex)
            badge.text = "On" if on else "Off"
            badge.classes(replace="rz-badge" + ("" if on else " muted"))
        self.race_box.visible = self.racing()

    def build_experiments(self):
        self.experiment_badges: list[tuple] = []
        with ui.dialog() as self.info_dlg, ui.card().classes("w-[40rem] max-w-full gap-3"):
            self.info_title = ui.label("").classes("rz-step-title")
            self.info_body = ui.column().classes("w-full gap-3")
            with ui.row().classes("w-full justify-end"):
                ui.button("Close", on_click=self.info_dlg.close).props("flat")
        for group in EXPERIMENT_GROUPS:
            ui.label(group).classes("rz-setting-title pt-2")
            for ex in [e for e in EXPERIMENTS if e["group"] == group]:
                box = ui.element("div").classes("rz-setting rz-exp")
                if ex.get("when"):
                    self.show_when(box, lambda ex=ex: ex["when"](self))
                with box:
                    with ui.column().classes("gap-1 min-w-0"):
                        with ui.row().classes("items-center gap-2"):
                            ui.label(ex["title"]).classes("rz-setting-title")
                            badge = ui.label("Off").classes("rz-badge muted")
                            self.experiment_badges.append((ex, badge))
                            ui.label(EVIDENCE[ex["evidence"]][0]).classes(f"rz-evidence {ex['evidence']}") \
                                .tooltip(EVIDENCE[ex["evidence"]][1])
                        ui.label(ex["short"]).classes("rz-hint")
                    with ui.row().classes("w-full items-center gap-2 no-wrap"):
                        with ui.element("div").classes("flex-1 min-w-0"):
                            self.experiment_control(ex)
                        ui.button("What is this?", icon="help_outline", on_click=lambda _, e=ex: self.explain(e),
                                  color=None).props("flat no-caps dense").classes("rz-file-btn shrink-0")
        with ui.column().classes("w-full gap-3") as self.race_box:
            self.build_racers()
            self.build_rules()

    def experiment_control(self, ex: dict):
        key, kind = ex["key"], ex["control"]
        if kind == "switch":
            self.bind(ui.switch(), key)
        elif kind == "select":  # the form's own choices (a value set on the Train tab stays valid), named by the extra
            form = self.w[key].options
            options = {**(form if isinstance(form, dict) else {o: o for o in form}), **ex["options"]}
            self.bind(ui.select(options, label=ex["title"]), key).classes("w-full")
        elif kind == "text":
            self.bind(ui.input(ex["title"]), key).classes("w-full").tooltip(ex.get("hint", ""))
        else:  # a number: 0 is off
            self.bind(ui.number(ex["title"], min=0, step=ex.get("step", 1), precision=ex.get("precision")), key) \
                .classes("w-full").tooltip(ex.get("hint", ""))
        self.w[key].on_value_change(lambda _: background_tasks.create(self.render_review(), name="wizard review"))

    def explain(self, ex: dict):
        self.info_title.text = ex["title"]
        self.info_body.clear()
        with self.info_body:
            ui.label(EVIDENCE[ex["evidence"]][0]).classes(f"rz-evidence {ex['evidence']}")
            for heading, text in (("What it does", ex["what"]), ("Testing so far", ex["tested"]),
                                  ("Cost and limits", ex["cost"])):
                if text:
                    ui.label(heading).classes("rz-setting-title")
                    ui.label(text).classes("text-sm")
        self.info_dlg.open()

    # ---- the race (an extra): the racers and its rules
    def build_racers(self):
        ui.label("The racers").classes("rz-step-title pt-2")
        hint("Every racer starts from the setup on the earlier pages and changes only what you assign it below. They "
             "train in turn on the same photos and probes, and the likeness probes pick the winner.")
        with self.setting(None, "How many racers",
                          "Each racer trains up to the full steps, so every one adds to the time (shown below). Two to "
                          "four is a good race."):
            self.racer_count = ui.number("Racers", value=self.n_racers, min=2, max=8, precision=0) \
                .classes("w-full")
            self.racer_count.on_value_change(lambda e: self.set_racer_count(e.value))
        with self.setting(None, "What differs between them",
                          "Pick the settings the racers may change; each racer below gets a field for each, filled in "
                          "with the base value. Common picks: learning rate, LoRA rank, adapter type, or seed."):
            self.race_picker = ui.select(race_choice_options(), multiple=True, value=list(self.race_keys),
                                         label="Settings", with_input=True).props("use-chips").classes("w-full")
            self.race_picker.on_value_change(lambda e: self.set_race_keys(e.value))
        self.racer_grid = ui.element("div").classes("rz-racers w-full")
        self.race_check = ui.label("").classes("rz-banner info")

    def build_rules(self):
        ui.label("The race's rules").classes("rz-step-title pt-2")
        hint("When racers leave the race, and how the winner is chosen.")
        with self.setting(None, "Relay: everyone continues from the round's winner",
                          "Nobody is dropped. After each round every racer takes over the round winner's LoRA "
                          "(weights, optimizer state, weight average, likeness history) and trains on with its own "
                          "settings, seed, image order and noise. Every racer trains all the steps (one run's cost per "
                          "racer); the last round's winner is the run's LoRA. Racers must share the LoRA's shape: no "
                          "rank, alpha, layer or network type rows."):
            self.bind(ui.switch("Relay: continue from the round's winner"), "race_follow")
        with self.setting(None, "Pick the racer to continue from yourself",
                          "After each round the run waits while you look at the probe images, then everyone continues "
                          "from the racer you pick with its Continue from button on the Monitor, not from the highest "
                          "likeness score. The GPUs sit idle while it waits.", show_if=("race_follow", bool)):
            self.bind(ui.switch("Pick the leader after each round"), "race_pick")
        with self.setting(None, "Wait for the pick at most",
                          "Minutes a round waits for your pick; then the highest likeness score leads as usual. 0 waits "
                          "until you pick (a rented GPU bills all the while).", show_if=("race_pick", bool)):
            self.bind(ui.number("Minutes (0 = until you pick)", min=0, precision=0), "race_pick_minutes") \
                .classes("w-full")
        with self.setting(None, "Drop the lowest racer after each round",
                          "Every racer trains until its next probe, then the next one does; after a round the one with "
                          "the lowest likeness score leaves. Fast and cheap, but a racer that learns slowly and would "
                          "end higher can be cut early. Off: every racer can train all the steps (the steps must be a "
                          "multiple of the probe interval)."):
            self.bind(ui.switch("Drop the lowest after each round"), "race_drop_lowest")
        with self.setting(None, "Warm-up probe rounds",
                          "Rounds before the first racer is dropped: everyone gets a few probes to get going. 1-2 is "
                          "plenty.", show_if=("race_drop_lowest", bool)):
            self.bind(ui.number("Warm-up rounds", min=0, precision=0), "race_warmup_probes").classes("w-full")
        with self.setting(None, "Drop only when the probe seeds agree",
                          "The racers are probed on the same seeds, so each seed compares them like for like. On: the "
                          "lowest leaves only when its seeds put it below the round's leader with the confidence below "
                          "(0.9 is a good start); a round too close to call drops nobody. Use 2 or more probe seeds.",
                          show_if=("race_drop_lowest", bool), adv=True):
            self.bind(ui.switch("Drop only when the seeds agree"), "race_paired")
        with self.setting(None, "Confidence", "How sure the paired test must be that the lowest racer is behind.",
                          show_if=("race_paired", bool), adv=True):
            self.bind(ui.number("Confidence", min=0.5, max=0.99, step=0.05, precision=2),
                      "race_drop_confidence").classes("w-full")
        with self.setting(None, "Drop a racer when it stops improving",
                          "A racer leaves once its own probes have not improved for a number of probes in a row, "
                          "however it compares with the others. Good with slow starters. The winner is then the racer "
                          "whose score was highest at any point."):
            self.bind(ui.switch("Drop a racer when it stops improving"), "race_patience_drop")
        with self.setting(None, "After how many probes without improvement",
                          "3 gives a racer a fair chance; lower ends the race sooner.",
                          show_if=("race_patience_drop", bool)):
            self.bind(ui.number("Probes", min=1, precision=0), "race_patience").classes("w-full")
        with self.setting(None, "Keep it while the likeness is rising again",
                          "A racer out of patience is not dropped while its smoothed likeness is still going up from "
                          "its previous probe: a chance to come back after a dip.",
                          show_if=("race_patience_drop", bool), adv=True):
            self.bind(ui.switch("Keep a racer whose likeness is rising"), "race_patience_rising")
        with self.setting(None, "One racer per GPU at the same time",
                          "With data parallel on 2+ GPUs: each GPU trains its own racer at once instead of all of them "
                          "taking turns over every GPU. Not with the identity loss or self-training.",
                          show_if=("multi_gpu", lambda x: x == "data_parallel")):
            self.bind(ui.switch("One racer per GPU"), "race_parallel")
        self.race_estimate = ui.label("").classes("rz-banner info")
        for key in ("race_drop_lowest", "race_warmup_probes", "race_patience_drop", "race_patience", "race_follow",
                    "race_patience_rising"):
            self.w[key].on_value_change(lambda _: self.estimate_race())

    def load_racers(self):
        """Read the Train tab's race rows in once (a loaded config, or rows typed there)."""
        if self.racers_loaded:
            return
        self.racers_loaded = True
        rows = self.w["race_configs"].value or []
        if len(rows) < 2:
            return
        from rulztrainer import race

        self.n_racers = len(rows)
        keys = []
        for r in rows:
            keys += [k for k in race.row_overrides(r) if k not in keys and k in race.ALLOWED]
        self.race_keys = keys or self.race_keys
        self.racer_vals = [{"name": str(r.get("name") or f"r{i + 1}"),
                            **{k: v for k, v in race.row_overrides(r).items() if k in race.ALLOWED}}
                           for i, r in enumerate(rows)]
        self.racer_count.value = self.n_racers
        self.race_picker.value = list(self.race_keys)

    def set_racer_count(self, n):
        try:
            self.n_racers = max(2, min(8, int(n or 2)))
        except (TypeError, ValueError):
            return
        self.rebuild_racers()

    def set_race_keys(self, keys):
        self.race_keys = list(keys or [])
        self.rebuild_racers()

    def rebuild_racers(self):
        """One card per racer: its name and a field for every setting it may change (the base value until set)."""
        try:
            base = self.f.to_config()
        except (TypeError, ValueError):
            return
        while len(self.racer_vals) < self.n_racers:
            i = len(self.racer_vals)
            self.racer_vals.append({"name": "base" if i == 0 else f"r{i + 1}"})
        del self.racer_vals[self.n_racers:]
        self.racer_grid.clear()
        with self.racer_grid:
            for i, vals in enumerate(self.racer_vals):
                with ui.card().props("flat").classes("rz-racer"):
                    name = ui.input("Racer name", value=vals.get("name", f"r{i + 1}")).classes("w-full")
                    name.on_value_change(lambda e, v=vals: (v.__setitem__("name", (e.value or "").strip()),
                                                            self.write_race()))
                    for key in self.race_keys:
                        self.racer_field(key, vals, base)
                    if not self.race_keys:
                        ui.label("The base setup").classes("rz-hint")
        self.write_race()

    def racer_field(self, key: str, vals: dict, base: TrainConfig):
        """A racer's control for one setting: a switch, the Train tab's choices, or a text box (numbers, lists)."""
        from rulztrainer.race import _value

        label = RACE_LABELS.get(key, key)
        base_v = getattr(base, key)
        cur = vals.get(key, base_v)

        def set_v(value, v=vals):
            v[key] = value
            self.write_race()

        form = self.w.get(key)
        if isinstance(base_v, bool):
            ui.switch(label, value=bool(cur), on_change=lambda e: set_v(bool(e.value)))
        elif isinstance(form, ui.select) and not isinstance(base_v, list):
            ui.select(form.options, value=cur if cur in (form.options or {}) else base_v, label=label,
                      on_change=lambda e: set_v(e.value)).classes("w-full")
        else:
            ui.input(label, value=race_value_text(key, cur),
                     on_change=lambda e: set_v(_value(e.value or ""))).classes("w-full") \
                .tooltip(f"Base: {race_value_text(key, base_v)}")

    def write_race(self):
        """The racers into the Train tab's race rows (only what differs from the base), then check them."""
        try:
            base = self.f.to_config()
        except (TypeError, ValueError):
            return
        rows = []
        for i, vals in enumerate(self.racer_vals):
            over = {k: vals[k] for k in self.race_keys if k in vals and not _same(vals[k], getattr(base, k))}
            rows.append({"name": vals.get("name") or f"r{i + 1}", **over})
        self.w["race_configs"].value = rows
        self.check_race()

    def check_race(self):
        if not hasattr(self, "race_check"):
            return
        try:
            cfg = self.f.to_config()
            from rulztrainer import race

            race.validate(cfg)
        except (TypeError, ValueError) as e:
            self.race_check.text = f"Not ready: {e}"
            self.race_check.classes(replace="rz-banner warning")
            return
        same = len({race.format_overrides(r) for _, _, r in race.variant_configs(cfg)})
        dup = same < len(self.racer_vals)
        self.race_check.text = ("Two racers have the same settings: give each one something of its own." if dup else
                                race_review_line(cfg))
        self.race_check.classes(replace=f"rz-banner {'warning' if dup else 'info'}")

    def estimate_race(self):
        if not hasattr(self, "race_estimate"):
            return
        from rulztrainer import race

        try:
            cfg = self.f.to_config()
        except (TypeError, ValueError):
            return
        n = len(cfg.race_configs or [])
        if n < 2 or cfg.probe_every <= 0:
            self.race_estimate.text = ("Pick at least two racers." if n < 2 else
                                       "The race runs on the likeness probes: set a likeness check interval (Samples & probes page).")
            return
        est = race.estimate_steps(n, cfg.race_warmup_probes, cfg.probe_every, cfg.steps,
                                  cfg.race_drop_lowest and not cfg.race_follow)
        if cfg.race_follow:
            self.race_estimate.text = (
                f"{n} racers in a relay: {est['rounds']} rounds, after each one everyone continues from its winner; "
                f"{est['total']} training steps in all ({est['ratio']:.0f}x one run)"
                + (f". Steps must be a multiple of {cfg.probe_every}." if cfg.steps % cfg.probe_every else "."))
            return
        if cfg.race_drop_lowest:
            text = (f"{n} racers: {est['rounds']} rounds and {est['probes']} probes, about {est['total']} training "
                    f"steps in all ({est['ratio']:.1f}x one {cfg.steps}-step run)")
            text += (f". Steps must be at least {est['race_own_steps']} for the race to finish."
                     if cfg.steps < est["race_own_steps"] else ".")
        else:
            text = (f"{n} racers, nobody cut for being behind: at most {est['total']} training steps "
                    f"({est['ratio']:.0f}x one run)" + (f". Steps must be a multiple of {cfg.probe_every}."
                                                         if cfg.steps % cfg.probe_every else "."))
        if cfg.race_patience_drop:
            text += (f" A racer leaves after {cfg.race_patience} probes without improvement"
                     + (", unless its likeness is rising again." if cfg.race_patience_rising else "."))
        self.race_estimate.text = text

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
        ds_paths = [sp.path for sp in cfg.dataset_specs() if sp.path]
        if self.data is None or self.data.get("folder") != "|".join(ds_paths):
            self.data = await run.io_bound(datasets_summary, ds_paths)
        if not self.cfg_path.value:
            self.cfg_path.value = f"configs/{cfg.job_name}.yaml"
        issues = review_issues(cfg, self.goal, self.data, len(self.gpus))
        data = self.data or {}
        multi = {"off": f"single GPU ({cfg.device})", "data_parallel": f"data parallel on {cfg.devices}",
                 "model_split": f"model split over {cfg.devices}"}.get(cfg.multi_gpu, cfg.multi_gpu)
        model = (Path(cfg.transformer_file).name or "Hugging Face model") if cfg.model_source == "files" \
            else next((k for k, v in MODEL_PRESETS.items() if v == cfg.model_path), cfg.model_path)
        rows = [
            ("Job", f"{cfg.job_name}  ->  {_relative(str(cfg.job_dir()))}/"),
            ("Model", f"{model} ({'single files' if cfg.model_source == 'files' else 'Hugging Face'})"),
            ("GPUs", f"{multi}; model quant {cfg.transformer_quant}, text encoder quant {cfg.text_encoder_quant}"),
            ("Datasets" if len(ds_paths) > 1 else "Dataset", datasets_line(cfg, data)),
            ("Trigger word", cfg.trigger_word or "(none)"),
            ("Resolution", ", ".join(f"{mp:g} MP" for mp in cfg.resolutions)
             + (" (datasets with their own: those)" if any(sp.resolutions for sp in cfg.dataset_specs()) else "")),
            ("Training", f"{cfg.steps} steps, batch {cfg.batch_size}, lr {cfg.learning_rate:g}, {cfg.optimizer}"
                         + (f", LoRA rank {cfg.lora_rank} / alpha {cfg.lora_alpha:g}" if cfg.network_type == "lora"
                            and not cfg.finetune else "")),
            ("Experimental", ", ".join(ex["title"] for ex in EXPERIMENTS if self.experiment_on(ex)) or "none"),
            ("Saving", f"every {cfg.save_every} steps; samples every {cfg.sample_every} steps "
                       f"({len(cfg.sample_prompts)} prompts)"),
        ]
        rows[len(rows) - 1:len(rows) - 1] = extra_review_rows(cfg)  # before "Saving"
        if cfg.preset == "identity":
            focus = character_focus(cfg)
            refs = ([f"{len(cfg.face_refs)} face"] if focus != "body_only" else []) + \
                   ([f"{len(cfg.body_refs)} body"] if focus != "face_only" else [])
            rows.append(("Likeness", f"{CHARACTER_FOCUS[focus][1].lower()}: {' / '.join(refs)} reference entries, "
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

    def mark_race(self):
        if hasattr(self, "start_btn"):
            self.start_btn.text = "Start the race" if self.racing() else "Start training"


# ---------------------------------------------------------------- previous runs

HISTORY_COLORS = ["#FF3B30", "#60A5FA", "#34D399", "#F59E0B", "#C084FC", "#22D3EE", "#F472B6", "#A3E635", "#94A3B8",
                  "#FB923C"]
_WIDE = {"classes": "rz-wide-col", "headerClasses": "rz-wide-col"}  # hidden on a phone (the run's name has its date)
HISTORY_COLUMNS = [
    {"name": "run", "label": "Run", "field": "run", "align": "left", "sortable": True},
    {"name": "started", "label": "Started", "field": "started", "align": "left", "sortable": True, **_WIDE},
    {"name": "state", "label": "State", "field": "state", "align": "left", "sortable": True, **_WIDE},
    {"name": "steps", "label": "Steps", "field": "steps", "align": "right"},
    {"name": "adapter", "label": "Adapter", "field": "adapter", "align": "left"},
    {"name": "lr", "label": "LR", "field": "lr", "align": "right", **_WIDE},
    {"name": "face", "label": "Face", "field": "face", "align": "right", **_WIDE},
    {"name": "body", "label": "Body", "field": "body", "align": "right", **_WIDE},
    {"name": "combined", "label": "Combined", "field": "combined", "align": "right"},
    {"name": "best", "label": "Best", "field": "best", "align": "right"},
]
# the tooltip of a Previous Runs chart: its lines share their run's name (one legend entry per run), so it names each
# line by its id, "<run> · <metric>"
HISTORY_TOOLTIP = ("(ps) => { ps = [].concat(ps); if (!ps.length) return ''; "
                   "let s = 'Step ' + Math.round(ps[0].axisValue).toLocaleString('en-US'); "
                   "for (const p of ps) { const v = Array.isArray(p.value) ? p.value[1] : p.value; "
                   "s += '<br>' + p.marker + p.seriesId + ' <b>' + (v == null ? '-' : Number(v).toFixed(1) + '%') "
                   "+ '</b>'; } return s; }")


def history_label(run: dict) -> str:
    """A run's name in the table and the legends: the label it was given, else the job and when it started."""
    from datetime import datetime

    return run.get("label") or f"{run.get('job') or 'run'} {datetime.fromtimestamp(run.get('started') or 0):%b %d %H:%M}"


def history_rows(runs: list[dict]) -> list[dict]:
    """The Previous Runs table: one row per run, with its last smoothed probe scores and its best combined score."""
    from datetime import datetime

    last = lambda pts: "-" if not pts else f"{pts[-1][1]:.1f}%"  # noqa: E731
    rows = []
    for r in runs:
        cfg, probes = r.get("config") or {}, r.get("probes") or {}
        adapter = ("fine-tune" if cfg.get("finetune") else f"OFT b{cfg.get('oft_block_size', 32)}"
                   if cfg.get("network_type") == "oft" else f"LoKr f{cfg.get('lokr_factor', 8)}"
                   if cfg.get("network_type") == "lokr" else f"LoRA r{cfg.get('lora_rank', '?')}") \
            + (" + DoRA" if cfg.get("dora") else "")
        comb = probes.get("combined") or []
        rows.append({"id": r["id"], "run": history_label(r),
                     "started": datetime.fromtimestamp(r.get("started") or 0).strftime("%Y-%m-%d %H:%M"),
                     "state": r.get("state", ""), "steps": f"{r.get('step', 0):,} / {r.get('total', 0):,}",
                     "adapter": adapter, "lr": f"{cfg['learning_rate']:.1e}" if cfg.get("learning_rate") else "-",
                     "face": last(probes.get("face")), "body": last(probes.get("body")), "combined": last(comb),
                     "best": "-" if not comb else f"{max(v for _, v in comb):.1f}%"})
    return rows


COMMON_SCALE_KEYS = ("face_scale_span", "face_scale_curve", "body_scale_span", "body_scale_curve", "face_weight")


def rescaled_runs(runs: list[dict], scale: dict | None) -> tuple[list[dict], list[str]]:
    """(runs, the labels of those left on their own scale): with a common scale ({face_scale_span, face_scale_curve,
    body_scale_span, body_scale_curve, face_weight}) each run's probe series rebuilt from its raw scores
    (history.rescore: its own floors and references, these spans, curves and weights); a run without raw scores
    keeps its own. None: the runs as saved."""
    if not scale:
        return runs, []
    out, own = [], []
    fw = float(scale["face_weight"])
    for r in runs:
        series = history.rescore(r, float(scale["face_scale_span"]), float(scale["face_scale_curve"]),
                                 float(scale["body_scale_span"]), float(scale["body_scale_curve"]), fw, 1.0 - fw)
        if series is None:
            own.append(history_label(r))
            out.append(r)
        else:
            out.append({**r, "probes": series})
    return out, own


def history_series(runs: list[dict], part: str, metrics: list[str]) -> list[dict]:
    """ECharts line series overlaying the chosen runs: one colour per run, one line style per metric (face solid,
    body dashed, combined thick). part is "identity" (the sample images' means) or "probes". A run's lines share its
    name, so the legend has one entry per run (a click hides all of them); each line's id is "<run> · <metric>"."""
    styles = {"face": {}, "body": {"type": "dashed"}, "combined": {"width": 3}}
    out = []
    n_color = 0
    for r in runs:
        data = r.get(part) or {}
        racers = data.get("racers") if part == "probes" else None
        # a race: each configuration's own lines in its own colour ("<run> lora16"), else the run's
        groups = [(f"{history_label(r)} {name}", d) for name, d in racers.items()] if racers else \
            [(history_label(r), data)]
        for label, d in groups:
            color = HISTORY_COLORS[n_color % len(HISTORY_COLORS)]
            n_color += 1
            for metric in metrics:
                pts = d.get(metric) or []
                if not pts:
                    continue
                out.append({"name": label, "id": f"{label} · {metric}", "type": "line", "data": pts,
                            "showSymbol": part == "identity", "symbolSize": 6, "itemStyle": {"color": color},
                            "lineStyle": {"color": color, "width": 2, **styles[metric]}})
    return out


METRIC_LINE = {"face": "solid", "body": "dashed", "combined": "thick"}  # history_series' line style per metric


class MetricChip:
    """An on/off chip for one metric's lines on a Previous Runs chart, for every ticked run at once (`.value` like a
    checkbox's)."""

    def __init__(self, metric: str, on_change, whose: str = "ticked run"):
        self.value = True
        self._on_change = on_change
        self.button = ui.button(on_click=self.toggle, color=None).props("flat no-caps").classes("rz-metric-chip")
        with self.button:
            ui.html(f'<span class="rz-line rz-line-{METRIC_LINE[metric]}"></span>')
            ui.label(metric.capitalize())
        self.button.tooltip(f"Show or hide the {metric} line of every {whose}")
        self._paint()

    def _paint(self):
        if self.value:
            self.button.classes(add="rz-on")
        else:
            self.button.classes(remove="rz-on")

    def toggle(self):
        self.value = not self.value
        self._paint()
        self._on_change()


def metric_chips(metrics, on_change, title: str = "Show lines (all runs):",
                 whose: str = "ticked run") -> dict[str, MetricChip]:
    """A "Show lines (all runs)" row of MetricChips: {metric: chip}."""
    ui.label(title).classes("text-sm font-medium")
    return {m: MetricChip(m, on_change, whose) for m in metrics}


PROBE_SETTING_KEYS = ("probe_face_prompt", "probe_body_prompt", "probe_seed", "probe_size", "probe_steps", "probe_cfg",
                      "turbo_lora", "resolutions")


def probe_image_groups(runs: list[dict]) -> list[tuple[str, dict, list[dict]]]:
    """[(label, run, entries)] of the runs' kept probe images: one group per run, or per configuration of a race
    ("<run> lora16"; its winner after the race: "<run> winner")."""
    out = []
    for r in runs:
        entries = r.get("probe_images") or []
        racers = list(dict.fromkeys(e.get("racer", "") for e in entries))
        for racer in sorted(racers, key=lambda x: x == ""):  # the winner's after the race last
            mine = [e for e in entries if e.get("racer", "") == racer]
            label = history_label(r) + (f" {racer}" if racer else (" winner" if len(racers) > 1 else ""))
            out.append((label, r, mine))
    return out


def probe_image_steps(groups, kind: str) -> list[int]:
    """Every probe step any of the groups has an image of this kind at, ascending."""
    return sorted({e["step"] for _, _, entries in groups for e in entries if e["kind"] == kind})


def probe_images_at(entries: list[dict], step: int, kind: str) -> tuple[int | None, list[dict]]:
    """(the step shown, its images by seed): the group's probe at `step`, else its latest one before it (a run probed
    on another interval), else None."""
    steps = sorted({e["step"] for e in entries if e["kind"] == kind and e["step"] <= step})
    if not steps:
        return None, []
    at = steps[-1]
    return at, sorted((e for e in entries if e["kind"] == kind and e["step"] == at), key=lambda e: e["seed"])


def probe_settings_differences(runs: list[dict]) -> list[str]:
    """The probe settings that differ between the runs ("probe_size: 512 / 768"): their probes are not the same
    picture of a likeness then."""
    out = []
    for key in PROBE_SETTING_KEYS:
        vals = [(r.get("config") or {}).get(key) for r in runs]
        known = [v for v in vals if v is not None]
        if len({json.dumps(v, sort_keys=True) for v in known}) > 1:
            out.append(f"{key}: " + " / ".join(str(v) for v in vals))
    return out


def probe_caption(entry: dict) -> str:
    """"face 70% · body 40% · combined 61%": a kept probe image's scores (the ones it does not have left out)."""
    return " · ".join(f"{k} {entry[k]:.0f}%" for k in ("face", "body", "combined") if entry.get(k) is not None)


def probe_compare(groups, step: int, kind: str) -> tuple[list[str], list[tuple[str, dict, str, str]]]:
    """(the groups without a probe of this kind by `step`, [(heading, entry, group label, note)]): the groups' probe
    images at `step` (each group's latest before it when it has none there, noted "step N") under one heading per
    probe image ("Face probe 1"), the groups side by side under it, so the same test picture of every run is
    compared together."""
    missing, by_seed = [], {}
    for label, _, entries in groups:
        at, imgs = probe_images_at(entries, step, kind)
        if at is None:
            missing.append(label)
        for e in imgs:
            by_seed.setdefault(e["seed"], []).append((label, e, "" if at == step else f"its latest: step {at:,}"))
    return missing, [(f"{kind.capitalize()} probe {seed + 1}", e, label, note)
                     for seed in sorted(by_seed) for label, e, note in by_seed[seed]]


def build_history_tab(charts: list):
    """Previous Runs: the saved likeness history of every run (output/_history), any number of them ticked to overlay
    their charts."""
    state = {"runs": [], "view": [], "own": [], "sig": None, "picks": [], "captions": [], "shown": None}

    def selected_runs() -> list[dict]:
        ids = {row["id"] for row in table.selected}
        return [r for r in state["view"] if r["id"] in ids]

    def common_scale() -> dict | None:
        if not scale_on.value:
            return None
        vals = {k: w.value for k, w in scale_inputs.items()}
        if any(v is None for v in vals.values()):
            return None
        return {**vals, "face_weight": float(vals["face_weight"]) / 100.0}

    def apply_scale():
        """Rebuild the shown runs on the common scale (or as saved), the table and the charts with them."""
        state["view"], state["own"] = rescaled_runs(state["runs"], common_scale())
        keep = {row["id"] for row in table.selected}
        table.rows = history_rows(state["view"])
        table.selected = [row for row in table.rows if row["id"] in keep]
        table.update()
        own = [r for r in state["own"] if r in {history_label(x) for x in selected_runs()}]
        scale_note.text = (f"On their own scale (saved before the history kept raw probe scores): {', '.join(own)}"
                           if own and scale_on.value else "")
        draw()

    def draw():
        picked = selected_runs()
        for part, chart, boxes, empty in (("identity", id_chart, id_metrics, id_empty),
                                           ("probes", probe_chart_h, probe_metrics, probe_empty)):
            series = history_series(picked, part, [m for m, box in boxes.items() if box.value])
            chart.options["series"] = series
            chart.options["legend"]["data"] = list(dict.fromkeys(sr["name"] for sr in series))
            chart.update()
            empty.text = ("Tick one or more runs above to draw them here." if not picked else
                          "" if series else "The ticked runs have no scores of this kind.")
        compare.text = f"{len(picked)} run(s) overlaid" if picked else ""
        refresh_files()
        draw_probes()

    def draw_probes(*_):
        picked = selected_runs()
        groups = probe_image_groups(picked)
        kind = probe_kind.value
        steps = probe_image_steps(groups, kind)
        opts = {s: f"Step {s:,}" for s in steps}
        if probe_step.options != opts:
            probe_step.set_options(opts, value=probe_step.value if probe_step.value in opts else
                                   (steps[-1] if steps else None))
        step = probe_step.value
        sig = (tuple(r["id"] for r in picked), tuple(len(e) for _, _, e in groups), kind, step,
               tuple(state["picks"]))
        if sig == state["shown"]:
            return
        state["shown"] = sig
        diffs = probe_settings_differences(picked)
        probe_warn.text = ("Probe settings differ, so these probes are not the same test: " + "; ".join(diffs)) \
            if diffs else ""
        pick_label.text = f"{len(state['picks'])} of 2 picked" if state["picks"] else \
            "Tick any two images (any run, any step) to compare them large"
        compare_btn.enabled = len(state["picks"]) == 2
        probe_grid.clear()
        with probe_grid:
            if not picked:
                hint("Tick one or more runs above to see their probe images side by side.")
                return
            if not steps:
                hint("The ticked runs have no kept probe images (runs from before RulzTrainer kept them, unless their "
                     "job folder still holds them: Import old runs adds those).")
                return
            missing, items = probe_compare(groups, step, kind)
            if missing:
                hint(f"No {kind} probe by step {step:,}: {', '.join(missing)}")
            captions = {id(e): " · ".join(x for x in (heading, label, f"Step {e['step']:,}", probe_caption(e)) if x)
                        for heading, e, label, _ in items}
            view = [(history.HISTORY_DIR / e["file"], captions[id(e)]) for _, e, _, _ in items]
            for heading in dict.fromkeys(h for h, _, _, _ in items):
                ui.label(heading).classes("rz-subhead w-full")
                with ui.element("div").classes("rz-gallery"):
                    for _, e, label, note in (it for it in items if it[0] == heading):
                        path = history.HISTORY_DIR / e["file"]
                        picked_now = str(path) in state["picks"]
                        with ui.column().classes("rz-shot gap-0" + (" rz-picked" if picked_now else "")):
                            ui.image(image_url(path)).props("ratio=1 fit=contain") \
                                .classes("w-full rz-imgbg cursor-pointer") \
                                .on("click", lambda _, q=path: show_full(q, view))
                            ui.button(icon="check_box" if picked_now else "check_box_outline_blank", color=None,
                                      on_click=lambda _, q=str(path), c=captions[id(e)]: toggle_pick(q, c)) \
                                .props("flat dense round size=sm").classes("rz-shot-pick") \
                                .tooltip("Tick to compare with another image")
                            ui.label(label).classes("rz-shot-title")
                            ui.label(" · ".join(x for x in (probe_caption(e), note) if x)).classes("rz-shot-sub")

    def toggle_pick(path: str, caption: str):
        if path in state["picks"]:
            i = state["picks"].index(path)
            del state["picks"][i]
            del state["captions"][i]
        else:
            if len(state["picks"]) == 2:  # a third pick replaces the older one
                state["picks"].pop(0)
                state["captions"].pop(0)
            state["picks"].append(path)
            state["captions"].append(caption)
        state["shown"] = None
        if len(state["picks"]) == 2:  # before the redraw, which removes the button this click came from
            open_compare()
        draw_probes()

    def open_compare():
        if len(state["picks"]) != 2:
            return
        for img, cap, path, text in zip(cmp_imgs, cmp_caps, state["picks"], state["captions"]):
            img.set_source(image_url(path))
            cap.text = text
        cmp_dlg.open()

    def clear_picks():
        state["picks"].clear()
        state["captions"].clear()
        state["shown"] = None
        draw_probes()

    def reload(force: bool = False):
        sig = history.signature()
        if sig == state["sig"] and not force:
            return
        state["sig"] = sig
        state["runs"] = history.list_runs()
        count.text = (f"{len(state['runs'])} saved run(s)" if state["runs"] else
                      "No saved runs yet. A run is saved while it trains once it has likeness scores.")
        apply_scale()

    def export_runs():
        from urllib.parse import quote

        picked = selected_runs()
        if not picked:
            ui.notify("Tick the runs to export", type="warning")
            return
        missing = [history_label(r) for r in picked if not (r.get("probe_raw") or {}).get("records")]
        for r in picked:
            if (r.get("probe_raw") or {}).get("records"):
                ui.download(f"/rz-history-xlsx?run={quote(r['id'])}", f"{history_file_name(r)}_likeness.xlsx")
        if missing:
            ui.notify(f"No raw probe scores saved for {', '.join(missing)}: runs saved before this version only have "
                      "them if their job folder still holds the run (Import old runs adds them then)",
                      type="warning", multi_line=True)

    def import_old():
        n = history.import_jobs()
        ui.notify(f"Imported {n} earlier run(s) from the output folder" if n else
                  "Nothing new to import: every job folder with scores already has a saved run", type="info")
        reload(True)

    async def rename():
        picked = selected_runs()
        if len(picked) != 1:
            ui.notify("Tick one run to rename it", type="warning")
            return
        with ui.dialog() as dlg, ui.card().classes("w-96"):
            ui.label("Name this run").classes("rz-setting-title")
            name = ui.input("Name", value=picked[0].get("label") or "").classes("w-full") \
                .props('placeholder="empty = the job name and date"')
            with ui.row().classes("w-full justify-end gap-2"):
                ui.button("Cancel", on_click=lambda: dlg.submit(None)).props("flat")
                ui.button("Save", on_click=lambda: dlg.submit(name.value)).props("unelevated")
        result = await dlg
        if result is not None:
            history.set_label(picked[0]["id"], result)
            reload(True)

    async def delete():
        picked = selected_runs()
        if not picked:
            ui.notify("Tick the runs to remove", type="warning")
            return
        with ui.dialog() as dlg, ui.card():
            ui.label(f"Remove {len(picked)} run(s) from the history? The job folders and their files are not "
                     "touched.").classes("text-sm")
            with ui.row().classes("w-full justify-end gap-2"):
                ui.button("Cancel", on_click=lambda: dlg.submit(False)).props("flat")
                ui.button("Remove", on_click=lambda: dlg.submit(True)).props("unelevated color=negative")
        if await dlg:
            history.delete_runs([r["id"] for r in picked])
            reload(True)

    with card():
        with ui.row().classes("w-full items-center gap-4"):
            with ui.column().classes("gap-1 flex-1 min-w-[16rem]"):
                ui.label("Previous runs").classes("rz-title")
                hint("Every run with likeness scores is saved here while it trains. Tick runs to compare their "
                     "likeness charts and probe images, and see which setup learned the person faster or better "
                     "(0 = a stranger, 100 = your references).")
            ui.button("Import old runs", icon="move_to_inbox", on_click=import_old).props("outline") \
                .tooltip("Add the runs whose job folders are in the output folder but not in this list yet")
    with card():
        count = hint("")
        count.tooltip(f"Kept in {history.HISTORY_DIR}")
        table = ui.table(columns=HISTORY_COLUMNS, rows=[], row_key="id", selection="multiple",
                         pagination={"rowsPerPage": 10, "sortBy": "started", "descending": True},
                         on_select=lambda _: apply_scale()).props("flat dense").classes("w-full")
        with ui.row().classes("w-full items-center gap-2"):
            ui.button("Rename", icon="edit", on_click=rename).props("flat dense")
            ui.button("Export spreadsheet", icon="table_view", on_click=export_runs).props("flat dense") \
                .tooltip("The ticked runs' likeness spreadsheets (one file each), the same as the Monitor's: every "
                         "probe's face, body and combined score as formulas of the raw scores and the run's scale, "
                         "editable on the Scale sheet. Works after the job folder is gone or reused.")
            ui.button("Make report", icon="summarize", on_click=lambda: make_reports()).props("flat dense") \
                .tooltip("A summary picture and a PDF of each ticked run (settings, scores, probe renders through "
                         "the run, dataset examples, training maps), written into its job folder's report/ and "
                         "downloaded here. Needs the run's job folder to still hold its files.")
            ui.button("Remove from history", icon="delete", on_click=delete).props("flat dense color=negative")
            ui.button("Clear ticks", icon="deselect",
                      on_click=lambda: (setattr(table, "selected", []), table.update(), draw())).props("flat dense")
            ui.space()
            compare = hint("")
    async def make_reports():
        """The run report (runreport.py) of every ticked run whose job folder still holds its files."""
        from rulztrainer import runreport

        picked = selected_runs()
        if not picked:
            ui.notify("Tick a run first.", type="warning")
            return
        for rec in picked:
            job_dir = history_job_dir(rec, state["runs"])
            name = history_label(rec)
            if job_dir is None:
                ui.notify(f"{name}: its job folder is gone or holds a later run, so there is nothing to make a "
                          f"report from.", type="warning")
                continue
            ui.notify(f"Making the report of {name}...")
            try:
                done = await run.io_bound(runreport.generate, job_dir)
            except Exception as e:  # noqa: BLE001
                ui.notify(f"{name}: no report ({e})", type="negative")
                continue
            for path in done.values():
                ui.download(path)
            ui.notify(f"Report of {name} written to {next(iter(done.values())).parent}", type="positive")
        refresh_files()

    with section("Saved files", "Download a ticked run's LoRAs, logs, settings and images", "download"):
        files_run = ui.select({}, label="Run", on_change=lambda _: refresh_files()).classes("w-full max-w-[30rem]")
        files_note = hint("")
        with ui.column().classes("w-full gap-2") as files_box:
            run_files = JobFiles()

    def refresh_files():
        """The files of the run picked among the ticked ones, while its job folder still holds them."""
        picked = selected_runs()
        opts = {r["id"]: history_label(r) for r in picked}
        if files_run.options != opts:
            files_run.set_options(opts, value=files_run.value if files_run.value in opts else
                                  (next(iter(opts)) if opts else None))
        files_run.visible = len(opts) > 1
        rec = next((r for r in picked if r["id"] == files_run.value), None)
        job_dir = history_job_dir(rec, state["runs"]) if rec else None
        files_box.visible = job_dir is not None
        files_note.text = ("Tick a run above to download its files." if rec is None else
                           "" if job_dir else
                           f"Its job folder ({rec.get('job_dir') or rec.get('job')}) is gone or now holds a later run "
                           "of the same job name, so its files cannot be downloaded here.")
        run_files.show(job_dir)

    ui.timer(5.0, while_connected(refresh_files))  # a running run's new checkpoints
    with section("Identity similarity", "How much each run's samples looked like the person (the average of each "
                 "sample step)", "face"):
        with ui.row().classes("items-center gap-3"):
            id_metrics = metric_chips(("face", "body"), lambda: draw())
            id_empty = hint("")
        id_chart = RzChart(chart_options(_axis("step", min="dataMin"), _axis("similarity %", min=0), [],
                                           [])).classes("w-full h-72")
    with section("Likeness probes & controller", "How much each run's probe images looked like the person "
                 "(smoothed)", "show_chart"):
        with ui.row().classes("items-center gap-3"):
            probe_metrics = metric_chips(("face", "body", "combined"), lambda: draw())
            probe_empty = hint("")
        d = TrainConfig()
        with ui.row().classes("w-full items-center gap-3"):
            scale_on = ui.switch("Compare on one scale", value=False, on_change=lambda _: apply_scale()) \
                .tooltip("Rescore every ticked run's probes, and the table's scores, on the same face / body scale "
                         "and weights instead of each run's own. Each run keeps its own floor (its step-0 probe) and "
                         "references' level; only the span, curve and weights become common, so runs trained with "
                         "different scale settings compare fairly. Runs saved before v2.9.0 need their raw probe "
                         "scores (Import old runs adds them while the job folder still holds the run).")
            scale_inputs = {}
            with ui.row().classes("items-center gap-2") as scale_row:
                for key, label, value, mn, step in (
                        ("face_scale_span", "Face span", d.face_scale_span, 1, 0.05),
                        ("face_scale_curve", "Face curve", d.face_scale_curve, 1, 0.1),
                        ("body_scale_span", "Body span", d.body_scale_span, 1, 0.05),
                        ("body_scale_curve", "Body curve", d.body_scale_curve, 1, 0.1),
                        ("face_weight", "Face weight %", round(d.face_weight * 100), 0, 5)):
                    scale_inputs[key] = ui.number(label, value=value, min=mn, max=100 if key == "face_weight" else 10,
                                                  step=step, on_change=lambda _: apply_scale()) \
                        .props("dense").classes("w-28")
                ui.button("Recommended", icon="auto_fix_high", on_click=lambda: [
                    scale_inputs[k].set_value(v) for k, v in (("face_scale_span", 1.1), ("face_scale_curve", 3.0),
                                                             ("body_scale_span", 1.0), ("body_scale_curve", 1.0),
                                                             ("face_weight", 80))]).props("flat dense no-caps") \
                    .tooltip("Face 1.1 / 3, body 1 / 1, 80% face: the face lines spread further apart and the body, "
                             "which moves little between runs, no longer swings the combined score")
            scale_row.bind_visibility_from(scale_on, "value")
        scale_note = hint("", warn=True)
        probe_chart_h = RzChart(chart_options(_axis("step", min="dataMin"), _axis("likeness %", min=0),
                                                [], [])).classes("w-full h-72")
    with section("Probe images", "The ticked runs' probes at the same step, the same test picture side by side; "
                 "tick two images to compare them large", "compare"):
        with ui.row().classes("w-full items-center gap-2"):
            probe_kind = ui.toggle({"face": "Face", "body": "Body"}, value="face",
                                   on_change=lambda _: draw_probes()).props("dense no-caps")

            def step_by(delta: int):
                keys = list(probe_step.options)
                if probe_step.value in keys:
                    i = min(len(keys) - 1, max(0, keys.index(probe_step.value) + delta))
                    probe_step.value = keys[i]

            with ui.row().classes("items-center gap-1 no-wrap rz-grow-phone"):
                ui.button(icon="chevron_left", on_click=lambda: step_by(-1)).props("flat dense round") \
                    .tooltip("Earlier probe")
                probe_step = ui.select({}, on_change=lambda _: draw_probes()) \
                    .props("dense options-dense outlined").classes("rz-step-pick")
                ui.button(icon="chevron_right", on_click=lambda: step_by(1)).props("flat dense round") \
                    .tooltip("Later probe")
        with ui.row().classes("w-full items-center gap-2"):
            pick_label = hint("").classes("flex-1 min-w-[10rem]")
            compare_btn = ui.button("Compare", icon="compare", on_click=open_compare).props("outline dense no-caps")
            ui.button("Clear", icon="deselect", on_click=clear_picks).props("flat dense no-caps") \
                .tooltip("Untick the picked images")
        probe_warn = hint("", warn=True)
        probe_grid = ui.column().classes("w-full gap-2")

    async def score_folders_ui():
        from urllib.parse import quote

        from rulztrainer import score_folders as SF

        def lines(text):
            return [x.strip().strip('"') for x in (text or "").splitlines() if x.strip()]

        runs = [SF.parse_run(x) for x in lines(cmp_runs.value)]
        if not runs:
            ui.notify("List the runs to compare: name = folder, one per line", type="warning")
            return
        fw = float(cmp_fw.value if cmp_fw.value is not None else 70) / 100.0
        out = COMPARE_DIR / f"likeness_runs_{time.strftime('%Y%m%d_%H%M%S')}.xlsx"
        cmp_btn.disable()
        cmp_out.text = "Scoring..."
        _task_progress.update(text="Scoring sample folders...", frac=None, active=True)
        cmp_tick.active = True  # this tab has no progress bar of its own
        try:
            result = await run.io_bound(SF.score, lines(cmp_face.value), lines(cmp_body.value),
                                        (cmp_base.value or "").strip().strip('"') or None, runs, "models", "cuda:0",
                                        "cpu", True, "", set_progress)
            await run.io_bound(SF.export, result, out, fw)
        except (OSError, ValueError, RuntimeError) as e:
            cmp_out.text = f"Error: {e}"
            return
        finally:
            cmp_tick.active = False
            _task_progress.update(active=False)
            cmp_btn.enable()
        cmp_out.text = "\n".join(SF.summary_lines(result, fw)) + f"\n\nSaved {out}"
        ui.download(f"/rz-compare-xlsx?name={quote(out.name)}", out.name)

    with section("Score sample folders", "Compare finished runs from any trainer on the likeness scores, in one "
                 "spreadsheet", "folder_open", open_=False):
        hint("Each run is a folder of its sample pictures, sub-folders included. A file or folder name with \"face\" "
             "marks face pictures, \"body\" body pictures (neither: both scores); the step comes from the file name "
             "(\"step000200\", \"200.png\"). The baseline folder holds the base model's renders of the same prompts "
             "without a LoRA: the zero of the face score and of the anchored body score (4 to 8 body seeds).")
        with ui.row().classes("w-full gap-3"):
            cmp_face = ui.textarea("Face references (files or folders, one per line)").props("rows=3") \
                .classes("flex-1 min-w-[16rem]")
            cmp_body = ui.textarea("Body references (files or folders, one per line)").props("rows=3") \
                .classes("flex-1 min-w-[16rem]")
        cmp_base = ui.input("Baseline folder (the base model, no LoRA)").classes("w-full")
        cmp_runs = ui.textarea("Runs: name = folder, one per line",
                               placeholder="keel_fb = F:/results/rulz keel_fb\nai-toolkit = F:/results/ai toolkit") \
            .props("rows=4").classes("w-full")
        with ui.row().classes("items-center gap-3"):
            cmp_fw = ui.number("Face weight %", value=70, min=0, max=100, step=5).props("dense").classes("w-28")
            cmp_btn = ui.button("Score and export", icon="table_view", on_click=score_folders_ui).props("outline") \
                .tooltip("Scores every picture with the trainer's face scorers (ArcFace, FaceNet) and the anchored "
                         "body score, then downloads a spreadsheet: each run's late checkpoints, its best one, charts "
                         "and every raw score, as formulas you can rescale")
        cmp_out = ui.label("").classes("rz-mono whitespace-pre-wrap")

        def cmp_progress():
            frac = _task_progress.get("frac")
            cmp_out.text = (_task_progress.get("text") or "Scoring...") + (f" ({frac:.0%})" if frac else "")

        cmp_tick = ui.timer(0.5, cmp_progress, active=False)
    with ui.dialog().props("maximized") as cmp_dlg, ui.card().tight().classes("w-full h-full bg-black"):
        with ui.row().classes("w-full items-center justify-between px-3 py-1"):
            ui.label("Compare").classes("text-sm text-white")
            ui.button(icon="close", on_click=cmp_dlg.close).props("flat dense round color=white")
        with ui.row().classes("w-full no-wrap gap-2 px-2 rz-compare"):  # one above the other on a phone
            cmp_imgs, cmp_caps = [], []
            for _ in range(2):
                with ui.column().classes("items-center gap-1 rz-compare-half"):
                    cmp_imgs.append(ui.image("").props("fit=contain").classes("w-full rz-compare-img"))
                    cmp_caps.append(ui.label("").classes("text-xs text-white text-center"))
    show_full = image_viewer()
    for chart in (id_chart, probe_chart_h):
        chart.options["tooltip"][":formatter"] = HISTORY_TOOLTIP
    charts.extend([id_chart, probe_chart_h])
    try:
        history.import_jobs()  # the runs from before the history existed, once
    except Exception:  # noqa: BLE001
        pass
    reload(True)
    ui.timer(4.0, while_connected(reload))


# ---------------------------------------------------------------- page


# ---------------------------------------------------------------- Fleet: several vast.ai instances from this computer

_FLEET = None
_FLEET_LOCK = threading.Lock()
# an instance's state -> (what the card says, the dot's class)
FLEET_STATES = {"running": ("Training", "live"), "starting": ("Starting a run", "live"), "idle": ("Idle", "done"),
                "unreachable": ("Out of reach", "error"), "gone": ("Stopped or destroyed", ""),
                "off": ("Not watched", ""), "new": ("Not looked at yet", "")}
FLEET_RUN_COLUMNS = [
    {"name": "job", "label": "Run", "field": "job", "align": "left", "sortable": True},
    {"name": "state", "label": "State", "field": "state", "align": "left", "sortable": True},
    {"name": "where", "label": "Instance", "field": "where", "align": "left", "sortable": True, **_WIDE},
    {"name": "steps", "label": "Steps", "field": "steps", "align": "right"},
    {"name": "loss", "label": "Loss", "field": "loss", "align": "right", **_WIDE},
    {"name": "best", "label": "Best likeness", "field": "best", "align": "right", "sortable": True},
    {"name": "time", "label": "Time", "field": "time", "align": "right", **_WIDE},
    {"name": "cost", "label": "Cost", "field": "cost", "align": "right", **_WIDE},
    {"name": "ended", "label": "Last seen", "field": "ended", "align": "left", "sortable": True, **_WIDE},
    {"name": "group", "label": "Split race", "field": "group", "align": "left", "sortable": True, **_WIDE},
    {"name": "files", "label": "Downloaded", "field": "files", "align": "left", **_WIDE},
]
FLEET_SOURCES = {"form": "The Train tab's settings", "file": "A config file on this computer",
                 "run": "A past run's settings"}


def fleet():
    """rulz_fleet.py's Fleet for this WebUI: one for the whole process, as it holds the lock of the one that looks."""
    global _FLEET
    with _FLEET_LOCK:
        if _FLEET is None:
            vast = str(ROOT / "scripts" / "vast")
            if vast not in sys.path:
                sys.path.insert(0, vast)
            import rulz_fleet

            _FLEET = rulz_fleet.Fleet()
    return _FLEET


def fleet_loop() -> None:
    """Look at the fleet's instances while there are any (rulz_fleet's step: the queue moves on, due files come
    down, the watches are checked). A daemon or a wait started elsewhere may hold the looking instead; then this
    only waits for its turn."""
    while True:
        try:
            fleet().step()
        except Exception as e:  # noqa: BLE001  (the loop must outlive one bad look)
            print(f"[fleet] {e!r}", flush=True)
        time.sleep(2.0)


def fleet_num(value, digits: int = 1) -> str:
    return f"{value:,.{digits}f}" if isinstance(value, (int, float)) and not isinstance(value, bool) else "-"


def fleet_probe_text(probe: dict | None) -> str:
    """The last likeness probe in a line: "step 1,400: face 73.6 · body 48.2 · combined 47.0 · best 50.0 at 1,400"."""
    if not probe:
        return ""
    parts = [f"{name} {probe[key]:.1f}" for name, key in (("face", "face_ema"), ("body", "body_ema"),
                                                         ("combined", "combined_ema")) if probe.get(key) is not None]
    if probe.get("best") is not None:
        parts.append(f"best {probe['best']:.1f}" + (f" at {probe['best_step']:,}" if probe.get("best_step") else ""))
    return (f"step {probe['step']:,}: " if probe.get("step") is not None else "") + " · ".join(parts)


def fleet_job_line(inst: dict) -> str:
    """The run's numbers on an instance's card."""
    job, metrics = inst.get("job") or {}, inst.get("metrics") or {}
    if not job:
        return ""
    parts = [f"step {job.get('step') or 0:,} / {job.get('total') or 0:,}"]
    if job.get("avg_loss") is not None:
        parts.append(f"loss {job['avg_loss']:.4f}")
    if job.get("running"):
        if job.get("eta"):
            parts.append(f"{eta_text(job['eta'])} left")
        if metrics.get("cost") is not None:
            parts.append(f"${metrics['cost']:.2f} so far")
    else:
        parts.append("failed" if job.get("error") else "finished" if job.get("done") else "ended early")
    return " · ".join(parts)


def fleet_machine_line(inst: dict) -> str:
    parts = []
    if inst.get("disk_free_gb") is not None:
        parts.append(f"disk {inst['disk_free_gb']:.0f} GB free of {inst.get('disk_total_gb') or 0:.0f}")
    for gpu in inst.get("gpus") or []:
        parts.append(f"GPU {gpu.get('utilization', 0)}% · {gpu.get('temperature', 0)}°C · "
                     f"{gpu.get('memory_used_mb', 0) / 1024:.1f} / {gpu.get('memory_total_mb', 0) / 1024:.0f} GB")
    if inst.get("age") is not None:
        parts.append(f"seen {inst['age']} s ago" if inst["age"] < 120 else f"seen {eta_text(inst['age'])} ago")
    return " · ".join(parts)


def fleet_event_text(e: dict) -> str:
    """An event as a sentence for the Fleet tab's list."""
    kind, job, on = e.get("kind"), e.get("job"), f"instance {e.get('instance')}"
    if kind == "watch":
        text = f"{e.get('metric')} went {e.get('op')} {e.get('value'):g}: now {e.get('actual'):g}"
        text += f" ({job} on {on}" + (f", step {e['step']:,.0f})" if e.get("step") is not None else ")") \
            if job else f" ({on})"
        return text + (f" · {e['note']}" if e.get("note") else "")
    if kind == "run_done":
        best = f", best likeness {e['best']:.1f}" if isinstance(e.get("best"), (int, float)) else ""
        return f"{job} finished on {on}{best}"
    table = {"run_started": f"{job} started on {on}", "run_failed": f"{job} failed on {on}: {e.get('error')}",
             "run_ended": f"{job} ended early on {on} at step {e.get('step')}",
             "run_lost": f"{job} was lost with {on}", "dispatched": f"{job} sent to {on}",
             "dispatch_failed": f"{job} could not start on {on}: {e.get('error')}",
             "instance_down": f"{on} is out of reach: {e.get('error')}", "instance_up": f"{on} is back",
             "instance_gone": f"{on} is {e.get('status')}", "instance_added": f"{on} joined the fleet",
             "download_failed": f"{job}'s files could not be downloaded from {on}: {e.get('error')}",
             "queue_empty": "The fleet's queue is empty"}
    return table.get(kind, str(kind))


def fleet_run_rows(runs: list[dict]) -> list[dict]:
    """The past runs as table rows."""
    rows = []
    for r in runs:
        when = r.get("ended") or r.get("modified") or r.get("first_seen")
        rows.append({
            "key": r["key"], "job": r.get("job") or "", "state": r.get("state") or "",
            "where": f"{r.get('instance')} · {r.get('gpu') or '?'}",
            "steps": f"{r.get('step') or 0:,} / {r.get('total') or 0:,}",
            "loss": f"{r['avg_loss']:.4f}" if isinstance(r.get("avg_loss"), (int, float)) else "",
            "best": round(r["best"], 1) if isinstance(r.get("best"), (int, float)) else None,
            "time": eta_text(r.get("elapsed")) if r.get("elapsed") else "",
            "cost": f"${r['cost']:.2f}" if isinstance(r.get("cost"), (int, float)) else "",
            "ended": time.strftime("%Y-%m-%d %H:%M", time.localtime(when)) if when else "",
            "group": r.get("group") or "", "files": ", ".join(r.get("downloaded") or [])})
    return rows


def fleet_spark(series: dict | None) -> dict:
    """A small chart for a card or a past run: the loss, and the probes' combined likeness on a second axis."""
    series = series or {}
    probes = [[p["step"], p.get("combined_ema", p.get("combined"))] for p in series.get("probes") or []
              if p.get("step") is not None and p.get("combined_ema", p.get("combined")) is not None]
    axis = {"type": "value", "scale": True, "splitLine": {"show": False},
            "axisLabel": {"color": "#8E8E96", "fontSize": 10, "hideOverlap": True}}
    return {"backgroundColor": "transparent", "animation": False,
            "grid": {"left": 44, "right": 40 if probes else 12, "top": 22, "bottom": 22},
            "legend": {"top": 0, "right": 0, "itemWidth": 14, "itemHeight": 8,
                       "textStyle": {"color": "#8E8E96", "fontSize": 11}},
            "tooltip": {"trigger": "axis", "confine": True},
            "xAxis": {**axis, "min": 0}, "yAxis": [dict(axis), dict(axis)],
            "series": [{"name": "loss", "type": "line", "showSymbol": False, "data": series.get("losses") or [],
                        "lineStyle": {"width": 1.5, "color": "#FF3B30"}, "itemStyle": {"color": "#FF3B30"}},
                       {"name": "likeness", "type": "line", "yAxisIndex": 1, "data": probes, "symbolSize": 5,
                        "lineStyle": {"width": 1.5, "color": "#60A5FA"}, "itemStyle": {"color": "#60A5FA"}}]}


def build_fleet_tab(form_config=None):
    """Fleet: every vast.ai instance's run on one page, a queue whose next run goes to the first idle instance,
    the download rules, the watches, and every run on record. All of it is scripts/vast/rulz_fleet.py's Fleet;
    this page draws its state and calls it."""
    fl = fleet()
    import rulz_fleet

    state: dict = {"shape": None, "cards": {}, "seen": {}, "dl": {}, "accept": {}}

    async def act(func, *args, ok: str = "", **kw):
        """A fleet call off the page's thread (most of them go over ssh); a refusal becomes a notice."""
        try:
            result = await run.io_bound(func, *args, **kw)
        except (rulz_fleet.Refused, rulz_fleet.remote.Unreachable, SystemExit, OSError) as e:
            ui.notify(str(e) or "That did not work.", type="warning", multi_line=True)
            return None
        if ok:
            ui.notify(ok, type="positive")
        refresh(force=True)
        return True if result is None else result

    async def find():
        note = await run.io_bound(fl.discover)
        ui.notify(note or "Asked vast.ai for the running instances", type="warning" if note else "positive",
                  multi_line=True)
        refresh(force=True)

    def look_once():
        if not fl.hold():
            return "Another program (a daemon or a wait) is doing the looking: this page shows what it sees."
        fl.tick()
        return ""

    async def look_now():
        note = await run.io_bound(look_once)
        if note:
            ui.notify(note, type="info", multi_line=True)
        refresh(force=True)

    with card():
        with ui.row().classes("w-full items-center gap-3"):
            with ui.column().classes("gap-0"):
                ui.label("vast.ai fleet").classes("text-base font-medium")
                look_line = hint("")
            ui.space()
            ui.button("Find instances", icon="travel_explore", on_click=find, color=None) \
                .props("outline dense no-caps").tooltip("Ask vast.ai which instances are running: new RulzTrainer "
                                                        "instances join, stopped and destroyed ones are marked")
            ui.button("Look now", icon="refresh", on_click=look_now, color=None).props("flat dense no-caps")
        vast_note = ui.label("").classes("rz-banner warning")
        vast_note.visible = False
        hint("Your vast.ai instances from this computer, over ssh (the vastai CLI with your API key, and your ssh "
             "key on your vast.ai account). This WebUI looks at them every few seconds while it runs; closed, "
             "nothing moves the queue, downloads files or checks the watches unless "
             "python scripts/vast/rulz_fleet.py daemon runs instead. Nothing here rents, stops or destroys an "
             "instance.")
    grid = ui.element("div").classes("rz-fleet-grid w-full")
    no_instances = hint("No instances in the fleet yet. Rent one from your RulzTrainer template on vast.ai, then "
                        "Find instances.")

    # ------------------------------------------------------------ instances
    async def stop_run(iid: int, job: str):
        with ui.dialog() as dlg, ui.card().classes("w-[30rem] max-w-full"):
            ui.label(f"Stop {job} on instance {iid}?").classes("text-base font-medium")
            hint("The trainer saves the LoRA after the current step, then ends. The queue's next run for this "
                 "instance starts after it.")
            with ui.row().classes("w-full justify-end gap-2"):
                ui.button("Cancel", on_click=lambda: dlg.submit(False)).props("flat")
                ui.button("Stop the run", icon="stop", on_click=lambda: dlg.submit(True)).props("unelevated")
        go = await dlg
        dlg.delete()
        if go:
            await act(fl.stop_run, str(iid), ok=f"{job} is stopping")

    def make_card(inst: dict) -> dict:
        iid = inst["id"]
        w: dict = {}
        with ui.card().props("flat").classes("rz-card rz-fleet-card"):
            with ui.row().classes("w-full items-center gap-2 no-wrap"):
                w["dot"] = ui.element("span").classes("rz-dot")
                ui.label(inst.get("label") or f"Instance {iid}").classes("text-base font-medium")
                w["gpu"] = hint("")
                ui.space()
                w["state"] = ui.label("").classes("text-sm font-medium")
            w["error"] = ui.label("").classes("rz-banner warning")
            w["job"] = ui.label("").classes("text-sm font-medium")
            w["bar"] = ui.linear_progress(value=0, show_value=False, size="6px").props("rounded").classes("w-full")
            w["line"] = hint("")
            w["probe"] = hint("")
            w["chart"] = ui.echart(fleet_spark(None)).classes("w-full h-36")
            w["machine"] = hint("")
            w["theirs"] = hint("")
            with ui.row().classes("w-full items-center gap-2"):
                w["stop"] = ui.button("Stop run", icon="stop", color=None).props("outline dense no-caps")
                w["accept"] = ui.switch("Takes queued runs", value=inst.get("accept", True),
                                        on_change=lambda e, i=iid: act(fl.set_instance, i, accept=bool(e.value))
                                        if e.value != state["accept"].get(i, True) else None)
                ui.space()
                with ui.button(icon="more_vert", color=None).props("flat dense round"):
                    with ui.menu():
                        w["watch_item"] = ui.menu_item("")
                        ui.menu_item("Remove from the fleet", on_click=lambda i=iid: act(
                            fl.remove_instance, i, ok=f"Instance {i} removed; its runs stay on record"))
        return w

    def fill_card(w: dict, inst: dict) -> None:
        iid, job = inst["id"], inst.get("job") or {}
        text, dot = FLEET_STATES.get(inst["state"], (inst["state"], ""))
        w["dot"].classes(replace=f"rz-dot {dot}")
        w["state"].text = text
        price = f" · ${inst['dph']:.2f}/h" if isinstance(inst.get("dph"), (int, float)) else ""
        w["gpu"].text = f"{inst.get('gpu') or ''}{price}" + (f" · {iid}" if inst.get("label") else "")
        w["error"].text = inst.get("error") or ""
        w["error"].visible = inst["state"] == "unreachable" and bool(inst.get("error"))
        running = bool(job.get("running"))
        w["job"].text = (job.get("job") or "") + (f" · {job['stage']}" if running and job.get("stage") else "")
        w["job"].visible = bool(job)
        w["bar"].value = (job.get("step") or 0) / max(1, job.get("total") or 1)
        w["bar"].visible = running
        w["line"].text = fleet_job_line(inst) or ("Nothing has run here yet." if inst["state"] == "idle" else "")
        w["probe"].text = ("Last probe, " + fleet_probe_text(job.get("last_probe"))) if job.get("last_probe") else ""
        w["probe"].visible = bool(w["probe"].text)
        series = inst.get("series") or {}
        w["chart"].visible = bool(job) and len(series.get("losses") or []) > 1
        if w["chart"].visible:
            w["chart"].options.clear()
            w["chart"].options.update(fleet_spark(series))
            w["chart"].update()
        w["machine"].text = fleet_machine_line(inst)
        theirs = inst.get("queue") or []
        w["theirs"].text = ("Its own queue: " + ", ".join(e.get("job") or "?" for e in theirs)) if theirs else \
            ("Its WebUI is not running, so it cannot start a run." if inst.get("webui_running") is False else "")
        w["theirs"].visible = bool(w["theirs"].text)
        w["stop"].visible = running
        w["stop_target"] = (iid, job.get("job") or "")
        state["accept"][iid] = inst.get("accept", True)
        if w["accept"].value != inst.get("accept", True):
            w["accept"].value = inst.get("accept", True)
        w["accept"].visible = inst["state"] not in ("gone", "off")
        off = inst["state"] == "off"
        w["watch_item"].text = "Watch it again" if off else "Stop watching it"
        w["watch_target"] = (iid, off)

    def draw_instances(view: dict) -> None:
        instances = view["instances"]
        shape = [(i["id"], i.get("label")) for i in instances]
        if shape != state["shape"]:
            state["shape"], state["cards"] = shape, {}
            grid.clear()
            with grid:
                for inst in instances:
                    w = state["cards"][inst["id"]] = make_card(inst)
                    w["stop"].on_click(lambda _, w=w: stop_run(*w["stop_target"]))
                    w["watch_item"].on_click(lambda _, w=w: act(fl.set_instance, w["watch_target"][0],
                                                                enabled=w["watch_target"][1]))
        for inst in instances:
            fill_card(state["cards"][inst["id"]], inst)
        no_instances.visible = not instances

    # ------------------------------------------------------------ the queue
    with section("Fleet queue", "Runs waiting for an instance", "queue", open_=True) as queue_exp:
        with ui.row().classes("w-full items-center gap-2"):
            queue_state = ui.label("").classes("text-sm font-medium")
            ui.space()
            queue_pause = ui.button("", color=None, on_click=lambda: act(
                fl.queue_edit, "resume" if fl.read()["queue"]["paused"] else "pause")).props("outline dense no-caps")
            ui.button("Clear queue", icon="clear_all", color=None, on_click=lambda: act(fl.queue_edit, "clear")) \
                .props("flat dense no-caps")
        hint("The first waiting run goes to the first instance that has nothing running and nothing waiting in its "
             "own queue; when an instance finishes its run, it takes the next one. A run an instance refuses (its "
             "dataset is not there, a job folder of that name exists) is offered to the others.")
        queue_list = ui.column().classes("w-full gap-1")
        subhead("Add a run")
        with fields():
            q_job = ui.input("Job name").classes("w-56")
            q_source = ui.select(FLEET_SOURCES if form_config else {k: v for k, v in FLEET_SOURCES.items()
                                                                    if k != "form"},
                                 value="form" if form_config else "file", label="Settings from").classes("w-72")
            q_instance = ui.select({0: "The first idle instance"}, value=0, label="Send to").classes("w-56")
        with fields():
            q_file = path_field("Config file (YAML or JSON)", "file", "", "w-96",
                                [("config", "*.yaml *.yml *.json"), ("all files", "*.*")])
            q_run = ui.select({}, label="Past run", with_input=True).classes("w-96")
        q_file.parent_slot.parent.bind_visibility_from(q_source, "value", value="file")
        q_run.bind_visibility_from(q_source, "value", value="run")
        with fields():
            q_sets = ui.input("Changes", placeholder="steps=300 probe_every=50").classes("w-96") \
                .tooltip("KEY=VALUE pairs parted by spaces, applied on top of the settings")
            q_push = path_field("Dataset folder to send first (optional)", "dir", "", "w-96")
        hint("The settings are read on the instance: a dataset must be in its datasets folder (name a local folder "
             "to send first and the settings' paths into it are pointed there), and model files left empty are "
             "downloaded there.")
        with ui.row().classes("w-full items-center gap-3"):
            q_split = ui.switch("Split a race over the instances")
            q_parts = ui.number("Parts (0: one per instance)", value=0, min=0, max=32, precision=0).classes("w-56")
            q_parts.bind_visibility_from(q_split, "value")
            ui.space()
            q_add = ui.button("Add to the fleet queue", icon="playlist_add").props("unelevated no-caps")
        hint("A split race deals its configurations over the instances: each part trains on its own instance (a "
             "part with one configuration as a plain run, with more as a race among them), and the parts are "
             "compared by their best likeness under Past runs. A relay (race_follow) cannot be split.")

    def queue_config() -> tuple[dict | None, str]:
        """The settings the Add form names, and the job name they carry."""
        if q_source.value == "form":
            cfg = form_config() if form_config else None
            return (cfg.to_dict(), cfg.job_name) if cfg is not None else (None, "")
        if q_source.value == "file":
            if not q_file.value:
                raise rulz_fleet.Refused("Pick the config file.")
            data = rulz_fleet.load_config(q_file.value)
            return data, str(data.get("job_name") or "")
        rec = fl.load_run(q_run.value or "") or {}
        if not rec.get("config"):
            raise rulz_fleet.Refused("Pick a past run whose settings are on record.")
        return rec["config"], ""

    def queue_plan(job: str, config: dict, sets: list[str], instance, push: list[str], split: int):
        entries = fl.plan(job, config, sets=sets, instance=instance or None, push=push, split=split)
        checks = rulz_fleet.check_entries(fl, entries)
        fl.enqueue(entries=entries)
        return [c.get("note") for c in checks if c.get("note")]

    async def queue_add():
        try:
            config, name = queue_config()
        except (rulz_fleet.Refused, OSError, ValueError) as e:
            ui.notify(str(e), type="warning", multi_line=True)
            return
        if config is None:
            return  # the form said why
        split = 0
        if q_split.value:
            split = int(q_parts.value or 0) or len([i for i in fl.read()["instances"].values()
                                                    if i.get("enabled", True) and not i.get("gone")])
        q_add.disable()
        try:
            notes = await act(queue_plan, (q_job.value or name or "").strip(), config, (q_sets.value or "").split(),
                              q_instance.value, [q_push.value] if q_push.value else [], split)
        finally:
            q_add.enable()
        if notes is not None:
            ui.notify("Added to the fleet queue" + (": " + "; ".join(notes) if notes and notes is not True else ""),
                      type="positive", multi_line=True)

    q_add.on_click(queue_add)

    def draw_queue(view: dict) -> None:
        q = view["queue"]
        n = len(q["waiting"])
        queue_state.text = (f"{n} run{'s' if n != 1 else ''} waiting" if n else "Nothing queued") + \
            (" · paused" if q["paused"] else "")
        queue_exp.props["caption"] = queue_state.text
        queue_exp.update()
        queue_pause.text = "Resume queue" if q["paused"] else "Pause queue"
        queue_pause.props(f"icon={'play_arrow' if q['paused'] else 'pause'}")
        ids = {0: "The first idle instance", **{i["id"]: f"{i.get('label') or i['id']} · {i.get('gpu') or '?'}"
                                                for i in view["instances"] if i["state"] not in ("gone", "off")}}
        if ids != q_instance.options:
            q_instance.set_options(ids, value=q_instance.value if q_instance.value in ids else 0)
        queue_list.clear()
        with queue_list:
            for e in q["starting"]:
                hint(f"{e['job']}: being sent to instance {e.get('sent_to')}")
            for i, e in enumerate(q["waiting"]):
                what = ", ".join(x for x in (
                    f"{e['steps']:,} steps" if isinstance(e.get("steps"), int) else "",
                    f"race part of {e['group']}: {', '.join(e['race_rows'])}" if e.get("race_rows") else "",
                    f"only instance {e['instance']}" if e.get("instance") else "",
                    " ".join(e.get("set") or []),
                    f"refused by {', '.join(e['refused_by'])}" if e.get("refused_by") else "") if x)
                with ui.row().classes("w-full items-center gap-2 no-wrap rz-queue-row"):
                    ui.label(f"#{i + 1}").classes("rz-hint w-8")
                    ui.label(e["job"]).classes("text-sm font-medium")
                    hint(what)
                    ui.space()
                    ui.button(icon="arrow_upward", on_click=lambda _, k=e["id"]: act(fl.queue_edit, "move", k, -1)) \
                        .props("flat dense round size=sm").tooltip("Earlier")
                    ui.button(icon="arrow_downward", on_click=lambda _, k=e["id"]: act(fl.queue_edit, "move", k, 1)) \
                        .props("flat dense round size=sm").tooltip("Later")
                    ui.button(icon="close", on_click=lambda _, k=e["id"]: act(fl.queue_edit, "remove", k)) \
                        .props("flat dense round size=sm color=negative").tooltip("Remove from the queue")
            for r in q["recent"][:5]:
                when = time.strftime("%b %d %H:%M", time.localtime(r.get("time") or 0))
                hint(f"{r.get('job')}: " + (f"could not start on instance {r.get('instance')}: {r['error']}"
                                            if r.get("error") else f"sent to instance {r.get('instance')} {when}"),
                     warn=bool(r.get("error")))

    # ------------------------------------------------------------ downloads
    with section("Automatic downloads", "What comes down from the instances, each kind by its own rule",
                 "cloud_download", open_=False):
        hint(f"Files land in {fl.pulls} (one folder per instance and job) and are fetched only once at their "
             "size. An instance's disk goes with the instance: what is not downloaded when it is destroyed is gone.")
        dl_selects: dict = {}
        with fields():
            for kind, label in rulz_fleet.DOWNLOAD_KINDS.items():
                dl_selects[kind] = ui.select(rulz_fleet.DOWNLOAD_MODES, label=label, value="off",
                                             on_change=lambda e, k=kind: e.value != state["dl"].get(k) and
                                             fl.set_download({k: e.value})).classes("w-56")
            dl_every = ui.number("Seconds between downloads during a run", value=300, min=30, step=30, precision=0,
                                 on_change=lambda e: e.value and int(e.value) != state["dl"].get("every") and
                                 fl.set_download({}, every=int(e.value))).classes("w-72")
        hint("A LoRA is about 230 MB and a run keeps many: Every LoRA / checkpoint while it runs fills this "
             "computer's disk quickly. Best LoRA is the <job>_best file alone.")

    def draw_downloads(view: dict) -> None:
        settings = view["settings"]
        for kind, select in dl_selects.items():
            state["dl"][kind] = settings["download"].get(kind, "off")
            if select.value != state["dl"][kind]:
                select.value = state["dl"][kind]
        state["dl"]["every"] = settings["download_every"]
        if dl_every.value != settings["download_every"]:
            dl_every.value = settings["download_every"]

    # ------------------------------------------------------------ watches and events
    with section("Watches and events", "Get told when a number goes above or below a value", "notifications_active",
                 open_=False) as watch_exp:
        hint("A watch raises an event the moment a run's number is on the wrong side of a value: once per run, or "
             "with Repeat every time it comes back there. The numbers are the run's (while it trains), its last "
             "likeness probe's, and the machine's. An AI assistant sets the same watches from the command line "
             "(rulz_fleet.py watch add ...) and is woken by rulz_fleet.py wait.")
        with fields():
            w_metric = ui.select(list(rulz_fleet.METRICS), value="avg_loss", label="Number", with_input=True) \
                .classes("w-52")
            w_op = ui.select({"above": "goes above", "below": "goes below"}, value="above", label="When it") \
                .classes("w-40")
            w_value = ui.number("Value", value=None, format="%g").classes("w-36")
            w_job = ui.input("Only jobs named (optional, * allowed)").classes("w-64")
            w_note = ui.input("Note (optional)").classes("w-64")
            w_repeat = ui.checkbox("Repeat")

        async def watch_add():
            if w_value.value is None:
                ui.notify("Give the value.", type="warning")
                return
            await act(fl.add_watch, w_metric.value, w_op.value, float(w_value.value), job=w_job.value or "",
                      repeat=bool(w_repeat.value), note=w_note.value or "", by="webui", ok="Watch added")

        ui.button("Add watch", icon="add_alert", on_click=watch_add, color=None).props("outline dense no-caps")
        watch_list = ui.column().classes("w-full gap-1")
        subhead("What happened")
        event_list = ui.column().classes("w-full gap-0")

    def draw_watches(view: dict) -> None:
        watches = view["watches"]
        watch_exp.props["caption"] = (f"{len(watches)} watch{'es' if len(watches) != 1 else ''} set" if watches
                                      else "Get told when a number goes above or below a value")
        watch_exp.update()
        watch_list.clear()
        with watch_list:
            for watch in watches:
                scope = ", ".join(x for x in (f"jobs {watch['job']}" if watch.get("job") else "",
                                              f"instance {watch['instance']}" if watch.get("instance") else "",
                                              "repeats" if watch.get("repeat") else "",
                                              f"set by {watch['by']}" if watch.get("by") else "",
                                              watch.get("note") or "") if x)
                with ui.row().classes("w-full items-center gap-2 no-wrap rz-queue-row"):
                    ui.label(f"{watch['metric']} {watch['op']} {watch['value']:g}").classes("text-sm font-medium")
                    hint(scope)
                    ui.space()
                    hint(f"fired {watch.get('fired', 0)}x")
                    ui.button(icon="close", on_click=lambda _, k=watch["id"]: act(fl.remove_watch, k)) \
                        .props("flat dense round size=sm color=negative").tooltip("Remove this watch")

    def draw_events() -> None:
        events = fl.events(last=14)
        if not changed("events", events[-1]["seq"] if events else 0):
            return
        event_list.clear()
        with event_list:
            for e in reversed(events):
                bad = e["kind"] in ("watch", "run_failed", "run_lost", "dispatch_failed", "instance_down",
                                    "instance_gone", "download_failed")
                hint(time.strftime("%b %d %H:%M:%S  ", time.localtime(e.get("time") or 0)) + fleet_event_text(e),
                     warn=bad)
            if not events:
                hint("Nothing yet.")

    # ------------------------------------------------------------ past runs
    with section("Past runs", "Every run the fleet has seen, on any instance", "history", open_=True):
        with ui.row().classes("w-full items-center gap-3"):
            run_filter = ui.input("Filter", placeholder="job, instance, state, split race").classes("w-72") \
                .props("clearable")
            run_count = hint("")
        run_table = ui.table(columns=FLEET_RUN_COLUMNS, rows=[], row_key="key", selection="single",
                             pagination={"rowsPerPage": 12, "sortBy": "ended", "descending": True},
                             on_select=lambda _: show_run()).props("flat dense").classes("w-full")
        run_filter.bind_value_to(run_table, "filter")
        hint("Kept on this computer, so a run stays here after its instance is destroyed. Runs that were already "
             "on an instance when it joined are listed too. Tick a run for its numbers, charts and files.")
        with ui.column().classes("w-full gap-3") as run_detail:
            run_title = ui.label("").classes("text-base font-medium")
            run_facts = hint("")
            run_group = hint("")
            run_chart = ui.echart(fleet_spark(None)).classes("w-full h-56")
            run_settings = hint("")
            run_folder = hint("")
            run_images = ui.row().classes("w-full gap-2")
            with ui.row().classes("w-full items-center gap-2"):
                run_kinds = ui.select(rulz_fleet.DOWNLOAD_KINDS, value=["logs", "probes"], multiple=True,
                                      label="Download now").classes("w-72").props("use-chips")
                run_pull = ui.button("Download", icon="cloud_download", color=None).props("outline dense no-caps")
                ui.space()
                run_forget = ui.button("Forget this run", icon="delete_outline", color=None) \
                    .props("flat dense no-caps").tooltip("Drop it from this list. Downloaded files stay.")
        run_detail.visible = False

    def picked_run() -> dict | None:
        key = run_table.selected[0]["key"] if run_table.selected else None
        return fl.load_run(key) if key else None

    def show_run() -> None:
        rec = picked_run()
        run_detail.visible = rec is not None
        if rec is None:
            return
        cfg = rec.get("config") or {}
        run_title.text = f"{rec.get('job')} · {rec.get('state')}"
        facts = [f"instance {rec.get('instance')} ({rec.get('gpu') or '?'})",
                 f"step {rec.get('step') or 0:,} / {rec.get('total') or 0:,}"]
        if isinstance(rec.get("avg_loss"), (int, float)):
            facts.append(f"loss {rec['avg_loss']:.4f}")
        if isinstance(rec.get("best"), (int, float)):
            facts.append(f"best likeness {rec['best']:.1f}"
                         + (f" at step {rec['best_step']:,}" if rec.get("best_step") else ""))
        if rec.get("elapsed"):
            facts.append(eta_text(rec["elapsed"]))
        if isinstance(rec.get("cost"), (int, float)):
            facts.append(f"${rec['cost']:.2f}")
        if rec.get("error"):
            facts.append(f"error: {rec['error']}")
        run_facts.text = " · ".join(facts)
        run_group.text = ""
        if rec.get("group"):
            try:
                group = fl.group(rec["group"])
                run_group.text = f"Split race {rec['group']}: " + ", ".join(
                    f"{p['job']} {fleet_num(p.get('best'))}" for p in group["parts"]) + \
                    ("" if group["finished"] else " (still going)")
            except rulz_fleet.Refused:
                pass
        run_group.visible = bool(run_group.text)
        run_chart.options.clear()
        run_chart.options.update(fleet_spark(rec))
        run_chart.update()
        run_chart.visible = len(rec.get("losses") or []) > 1
        keys = [k for k in history.CONFIG_KEYS if k in cfg and cfg[k] not in (None, "", [], False)
                and not k.startswith("probe_") and k != "job_name"][:18]
        run_settings.text = ("Settings: " + ", ".join(f"{k} {cfg[k]}" for k in keys)) if keys else \
            "Its settings are not on record yet."
        folder = Path(rec["folder"]) if rec.get("folder") else None
        run_folder.text = (f"Files: {folder} ({rec.get('downloaded_mb', 0)} MB: "
                           f"{', '.join(rec.get('downloaded') or [])})" if folder else "No files downloaded.") + \
            (f" · last download failed: {rec['pull_error']}" if rec.get("pull_error") else "")
        run_images.clear()
        images = sorted((folder / "probes").glob("*.png"))[-8:] if folder and (folder / "probes").is_dir() else []
        with run_images:
            for img in images:
                ui.image(image_url(img)).classes("w-40 rounded").tooltip(img.name)
        run_forget.visible = rec.get("state") != "running"

    async def pull_run():
        rec = picked_run()
        if rec is None or not run_kinds.value:
            return
        run_pull.disable()
        try:
            result = await act(lambda: fl.pull(fl.instance(rec["instance"]), rec["key"], list(run_kinds.value)))
        finally:
            run_pull.enable()
        if isinstance(result, dict):
            ui.notify(result.get("error") or f"{result['fetched']} files, {result['mb']} MB",
                      type="warning" if result.get("error") else "positive", multi_line=True)
            show_run()

    async def forget_run():
        rec = picked_run()
        if rec is not None and await act(fl.forget_run, rec["key"], ok=f"{rec['job']} dropped from the list"):
            run_table.selected = []
            show_run()

    run_pull.on_click(pull_run)
    run_forget.on_click(forget_run)

    def draw_runs() -> None:
        if not changed("runs", fl.runs_signature()):  # the record files as they are on disk: no run is read for it
            return
        runs = fl.runs()
        keep = {row["key"] for row in run_table.selected}
        run_table.rows = fleet_run_rows(runs)
        run_table.selected = [row for row in run_table.rows if row["key"] in keep]
        run_table.update()
        run_count.text = f"{len(runs)} run{'s' if len(runs) != 1 else ''} on record"
        options = {r["key"]: f"{r.get('job')} · {r.get('instance')}" for r in runs if r.get("config")}
        if options != q_run.options:
            q_run.set_options(options, value=q_run.value if q_run.value in options else None)
        if keep:
            show_run()

    def changed(name: str, value) -> bool:
        """Whether a part of the page differs from what is drawn (and remember it as drawn)."""
        sig = json.dumps(value, sort_keys=True, default=str)
        if sig == state["seen"].get(name):
            return False
        state["seen"][name] = sig
        return True

    def refresh(force: bool = False) -> None:
        """Redraw the parts that changed since the last look at the fleet's files (every 3 s, and after an action)."""
        if force:
            state["seen"] = {}
        view = fl.overview()
        looked = view.get("looked")
        look_line.text = ("Nothing in the fleet to look at." if not view["instances"] and not view["queue"]["waiting"]
                          else f"Last look {looked} s ago" if looked is not None and looked < 600 else
                          "No look for a while: is this WebUI or a daemon doing the looking?") +             ("" if fl.ticker is not None or not view["looking"] else " · another program is doing the looking")
        vast_note.text = f"vast.ai could not be asked: {view.get('discover_error')}"
        vast_note.visible = bool(view.get("discover_error"))
        instances = view["instances"]
        if changed("instances", [{k: v for k, v in i.items() if k != "age"} for i in instances]):
            draw_instances(view)
        else:  # only the "seen 12 s ago" moved on
            for inst in instances:
                state["cards"][inst["id"]]["machine"].text = fleet_machine_line(inst)
        if changed("queue", [view["queue"], [(i["id"], i.get("label"), i.get("gpu"), i["state"]) for i in instances]]):
            draw_queue(view)
        if changed("settings", view["settings"]):
            draw_downloads(view)
        if changed("watches", view["watches"]):
            draw_watches(view)
        draw_events()
        draw_runs()

    refresh(force=True)
    ui.timer(3.0, while_connected(refresh))


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
    look = {"light": False}  # this browser's look, read from its localStorage once the page is connected
    form_state: dict = {"text": "", "warned": False}  # persist()'s last saved settings; "pristine": the defaults
    charts: list = []  # ECharts to recolor when the look changes

    def set_look(light: bool, save: bool = True) -> None:
        look["light"] = light
        ui.run_javascript(f"rzSetTheme({json.dumps('light' if light else 'dark')}, {json.dumps(save)})")
        for chart in charts:
            recolor_chart(chart, light)

    with ui.header().classes("rz-header"):
        with ui.row().classes("rz-header-row w-full items-center"):
            with ui.row().classes("rz-brand items-center no-wrap gap-2"):
                ui.element("img").props('src="/assets/logo_dark.png" alt="Rulz"').classes("rz-logo rz-dark-only")
                ui.element("img").props('src="/assets/logo.png" alt="Rulz"').classes("rz-logo rz-light-only")
                ui.label("TRAINER").classes("rz-wordmark")
                ui.label(f"v{__version__}").classes("rz-version")
            # the tabs in the order of the work: set a run up, watch it, look back. Thin rules part the three groups.
            with ui.tabs().props("inline-label no-caps align=left dense outside-arrows mobile-arrows")                     .classes("rz-tabs") as tabs:
                tab_wizard = ui.tab("Wizard", icon="auto_awesome")
                tab_train = ui.tab("Train", icon="tune")
                tab_dataset = ui.tab("Dataset", icon="photo_library")
                ui.element("span").classes("rz-tab-sep")
                tab_monitor = ui.tab("Monitor", icon="show_chart")
                tab_samples = ui.tab("Samples", icon="collections")
                tab_maps = ui.tab("Training maps", icon="layers")
                tab_history = ui.tab("Previous Runs", icon="history")
                # several vast.ai instances from this computer; an instance itself has no fleet to show
                tab_fleet = ui.tab("Fleet", icon="dns") if VAST_INSTANCE is None else None
                ui.element("span").classes("rz-tab-sep")
                tab_help = ui.tab("Help", icon="help_outline")
            with ui.row().classes("rz-header-tools items-center no-wrap gap-1"):
                # the machine: its free disk space, and which vast.ai instance this is on a rental
                with ui.row().classes("rz-chip items-center gap-2 no-wrap") as machine_chip:
                    machine_icon = ui.icon("storage")
                    machine_text = ui.label("").classes("rz-chip-text")
                    machine_tip = ui.tooltip("")
                # sun in the dark look (switch to light), moon in the light look; CSS picks the icon, so it is right
                # from the first paint
                with ui.button(on_click=lambda: set_look(not look["light"]), color=None)                         .props("flat round dense").classes("rz-theme-btn") as theme_btn:
                    ui.icon("light_mode").classes("rz-dark-only")
                    ui.icon("dark_mode").classes("rz-light-only")
                theme_btn.tooltip("Switch between the dark and the light look")
        # the run strip: what the trainer is doing, on every tab. A click opens the Monitor.
        with ui.element("div").classes("rz-run") as run_strip:
            with ui.row().classes("rz-run-row items-center no-wrap"):
                pill_dot = ui.element("span").classes("rz-dot")
                run_state = ui.label("Idle").classes("rz-run-state")
                run_job = ui.label("").classes("rz-run-job")
                run_step = ui.label("").classes("rz-run-step")
                run_bar = ui.linear_progress(value=0, show_value=False, size="6px").props("rounded")                     .classes("rz-run-bar")
                run_eta = ui.label("").classes("rz-run-step")
                run_note = ui.label("").classes("rz-run-note")
                ui.icon("chevron_right").classes("rz-run-go")
        run_strip.tooltip("Open the Monitor")

    def goto(name: str):
        tabs.value = name

    def form_config() -> TrainConfig | None:
        try:
            return validated_config(f.to_config())  # a race's own checks before a model loads
        except (TypeError, ValueError) as e:
            status_msg.text = f"A setting is invalid: {e}"
            ui.notify(status_msg.text, type="negative")
            return None

    def queue_job() -> bool:
        cfg = form_config()
        if cfg is None:
            return False
        ok, status_msg.text = queue_run(cfg)
        ui.notify(status_msg.text, type="positive" if ok else "warning")
        return ok

    async def start_job(confirm: bool = True) -> bool:
        """Start from the form; the usual mistakes (start_warnings) first ask Start anyway, unless the caller showed
        them already (the wizard's review)."""
        cfg = form_config()
        if cfg is None:
            return False
        warnings = await run.io_bound(start_warnings, cfg) if confirm else []
        if warnings:
            with ui.dialog() as dlg, ui.card().classes("w-[36rem] max-w-full"):
                ui.label("Before you start").classes("text-base font-medium")
                for w in warnings:
                    ui.label(w).classes("rz-banner warning")
                with ui.row().classes("w-full justify-end gap-2"):
                    ui.button("Cancel", on_click=lambda: dlg.submit(False)).props("flat")
                    ui.button("Start anyway", icon="play_arrow", on_click=lambda: dlg.submit(True)) \
                        .props("unelevated")
            go = await dlg
            dlg.delete()
            if not go:
                return False
        status_msg.text = start_training(cfg)
        started = status_msg.text.startswith("Started")
        ui.notify(status_msg.text, type="positive" if started else "warning")
        if started:
            tabs.value = tab_monitor
        return started

    run_strip.on("click", lambda: goto("Monitor"))
    machine_chip.on("click", lambda: goto("Monitor"))

    with ui.tab_panels(tabs, value=tab_monitor if job_running else tab_wizard).classes("w-full rz-panels") \
            .props("keep-alive"):
        # ------------------------------------------------------------ Train
        with ui.tab_panel(tab_train):
            with ui.row().classes("rz-page no-wrap items-start gap-6"):
                nav = ui.column().classes("rz-nav gt-sm")
                with ui.column().classes("gap-4 min-w-0 flex-1") as train_page:
                    sections = SectionNav()
                    with ui.row().classes("rz-intro w-full items-center gap-3 no-wrap"):
                        train_adv_switch = ui.switch("Advanced", value=False)
                        hint("Off: the everyday settings. On: every option, the Advanced panels and the experimental "
                             "settings and tools too (an experimental setting that is switched on always shows).")                             .classes("flex-1 min-w-0")
                    picker_row = ui.row().classes("w-full lt-md")

                    with sections.section("Basics", "Job", "Name, output folder, seed, preset and config file",
                                          "work_outline"):
                        with ui.row().classes("rz-intro w-full items-center gap-4"):
                            hint("New to this? The Wizard walks through the important settings with "
                                 "recommendations. Switch on Advanced above for every option, the "
                                 "experimental settings and the tools.").classes("flex-1 min-w-[16rem]")
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

                            def on_download_config():
                                try:
                                    cfg = f.to_config()
                                except (TypeError, ValueError) as e:
                                    status_msg.text = f"A setting is invalid: {e}"
                                    ui.notify(status_msg.text, type="negative")
                                    return
                                name = config_file_name(cfg.job_name)
                                ui.download.content(cfg.to_yaml(), name, "application/x-yaml")
                                status_msg.text = f"Downloaded {name}"

                            async def on_upload_config(e):
                                """Fill the form from a config file picked in the browser: it may be on another
                                computer than the WebUI. A file the form cannot take leaves the form as it was."""
                                upload_cfg.reset()
                                name = e.file.name
                                before = f.snapshot()
                                try:
                                    text = (await e.file.read()).decode("utf-8-sig")
                                    cfg, missing, unknown = TrainConfig.parse_report(text, name)
                                    f.from_config(cfg)
                                except Exception as err:  # noqa: BLE001
                                    f.restore(before)
                                    status_msg.text = f"Could not load {name}: {err}"
                                    ui.notify(status_msg.text, type="negative")
                                    return
                                status_msg.text = f"Loaded {name}"
                                if missing:
                                    status_msg.text += f". {len(missing)} newer options use their defaults"
                                if unknown:
                                    status_msg.text += f". Ignored options this version no longer has: {', '.join(unknown)}"
                                ui.notify(status_msg.text, type="positive")

                            upload_cfg = ui.upload(auto_upload=True, max_files=1, max_file_size=2_000_000,
                                                   on_upload=on_upload_config,
                                                   on_rejected=lambda: ui.notify("Pick one .yaml config file",
                                                                                 type="warning")) \
                                .props("accept=.yaml,.yml").classes("hidden")

                            async def on_reset_form():
                                with ui.dialog() as dlg, ui.card().classes("w-[28rem] max-w-full"):
                                    ui.label("Reset every setting to its default?").classes("text-base font-medium")
                                    ui.label("Config files you saved are not touched.").classes("rz-hint")
                                    with ui.row().classes("w-full justify-end gap-2"):
                                        ui.button("Cancel", on_click=lambda: dlg.submit(False)).props("flat")
                                        ui.button("Reset", icon="restart_alt", color="negative",
                                                  on_click=lambda: dlg.submit(True)).props("unelevated")
                                go = await dlg
                                dlg.delete()
                                if go:
                                    restore_page({**form_state["pristine"], "advanced": train_adv_switch.value})
                                    status_msg.text = "Settings reset to their defaults"

                            ui.button("Download", icon="file_download", on_click=on_download_config).props("flat") \
                                .tooltip("Download these settings as a config file to this device")
                            ui.button("Upload", icon="file_upload").props("flat") \
                                .on("click", js_handler=f"() => runMethod({upload_cfg.id}, 'pickFiles', [])") \
                                .tooltip("Load the settings from a config file on this device")
                            ui.button("Reset", icon="restart_alt", on_click=on_reset_form).props("flat") \
                                .tooltip("Put every setting back to its default")

                    # ---- Model
                    with sections.section("Basics", "Model", "Base model, model files and GPUs", "model_training"):
                        w["model_source"] = ui.radio([FILE_SOURCE, HF_SOURCE], value=FILE_SOURCE).props("inline")
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
                            fam0 = preset_arch("MOSS-Image-Base")
                            with fields():
                                w["model_file_quant"] = ui.select(file_quant_options(fam0, "model"),
                                                                  value=d.model_file_quant,
                                                                  label="Diffusion model download").classes("w-72") \
                                    .tooltip("The format of the diffusion model file downloaded from Comfy-Org for "
                                             "the model picked above, when no file is set below. int8 ConvRot is "
                                             "about half the size of bf16 and trains as fast.")
                                w["text_encoder_file_quant"] = ui.select(file_quant_options(fam0, "text"),
                                                                         value=d.text_encoder_file_quant,
                                                                         label="Text encoder download").classes("w-72")

                                async def on_download_files():
                                    files_status.text = "Downloading / checking the model files..."
                                    files_status.text = await run.io_bound(download_files, f.to_config())

                                ui.button("Download now", icon="download", on_click=on_download_files) \
                                    .props("outline").tooltip("Training downloads them on start anyway; this does it "
                                                              "now (Hugging Face's cache keeps them for later runs)")
                            files_status = hint("")
                            hint("The model picked above comes as single files from Comfy-Org's ComfyUI repackages "
                                 "(Comfy-Org/MOSS-Image, Comfy-Org/Krea-2, Comfy-Org/Fibo-2.0, Comfy-Org/Qwen-Image-2.1, "
                                 "Comfy-Org/Ideogram-4 (its int8 / bf16 text encoder from Qwen-Image 2.1's: the same "
                                 "model); FLUX 3 Klein: BFL's own black-forest-labs/flux-3-image-klein-comfy; "
                                 "FLUX.2 Klein: BFL's fp8 or bf16 file with Comfy-Org/Flux-2-Klein-9B's text encoder "
                                 "and VAE), "
                                 "downloaded on the first "
                                 "run. To use files you already have, set them below: anything left empty is "
                                 "downloaded.")
                            path_input(f, "transformer_file", "Diffusion model file (optional)", "file", "",
                                       "w-[40rem]").tooltip("Empty: downloaded in the format picked above.")
                            path_input(f, "text_encoder_file", "Text encoder file (optional)", "file", "",
                                       "w-[40rem]").tooltip("Empty: downloaded. Qwen3.5-9B for MOSS-Image; Qwen3-VL-4B "
                                                            "for Krea 2, FIBO 2.0 and FLUX 3 Klein; Qwen3-8B for "
                                                            "FLUX.2 Klein; Qwen3-VL-8B for "
                                                            "Qwen-Image 2.1 and Ideogram 4; Qwen3-VL-32B for MiniMax "
                                                            "H3 and MiniMax-H3-Image.")
                            path_input(f, "vae_file", "VAE file (optional)", "file", "", "w-[40rem]") \
                                .tooltip("Empty: downloaded. Flux2's for MOSS-Image, FIBO 2.0, both Kleins and "
                                         "Ideogram 4; Qwen-Image's for Krea 2; its own for Qwen-Image 2.1; the video "
                                         "VAE for MiniMax H3; its own image VAE for MiniMax-H3-Image.")
                            path_input(f, "tokenizer_path", "Tokenizer folder (optional)", "dir", "", "w-[40rem]") \
                                .tooltip("Empty: downloaded with the model.")
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
                            w["block_swap"] = ui.number("Block swap (blocks in RAM, 0 = off)", value=d.block_swap,
                                                        min=0, precision=0).classes("w-64") \
                                .bind_visibility_from(w["multi_gpu"], "value", lambda x: x == "off") \
                                .tooltip("For a card the model does not fit: this many DiT blocks keep their frozen "
                                         "weights in system RAM (pinned) and stream to the GPU as each block runs, "
                                         "the next one copied in while the current one computes. Saves about the "
                                         "blocks' share of the model in VRAM for some speed, depending on the PCIe "
                                         "link. Needs gradient checkpointing; single GPU only; turns torch.compile "
                                         "off unless the next switch is on.")
                            w["block_swap_compile"] = ui.switch("Compile with block swap (experimental)",
                                                                value=d.block_swap_compile) \
                                .bind_visibility_from(w["multi_gpu"], "value", lambda x: x == "off") \
                                .tooltip("torch.compile the blocks even with block swap: each block's forward is "
                                         "compiled and the swap's streaming runs outside it (OneTrainer compiles its "
                                         "offloaded blocks this way). Could bring compile's 12-26% to runs that swap; "
                                         "untested in RulzTrainer, so compare the step time and the compile report "
                                         "with a run without it.")
                            w["activation_offload"] = ui.switch("Activation offloading (experimental)",
                                                                value=d.activation_offload) \
                                .bind_visibility_from(w["gradient_checkpointing"], "value") \
                                .tooltip("Checkpointed blocks keep their inputs for the backward on the GPU: about "
                                         "tokens x hidden x 2 bytes each, 0.8 GB over H3's 50 blocks at 1 MP "
                                         "(estimate). This sends them to RAM during the forward and streams them "
                                         "back ahead of each block's recomputation, so more blocks can keep their "
                                         "activations or fewer need swapping, for extra PCIe traffic. Untested in "
                                         "RulzTrainer.")
                        with advanced():
                            with fields():
                                w["text_encoder_device"] = ui.select(["auto"] + gpus + ["cpu"], value="auto",
                                                                     label="Text encoder device (caching)") \
                                    .classes("w-56")
                                w["compile_model"] = ui.select(COMPILE_CHOICES, value=d.compile_model,
                                                               label="Compile model (torch.compile)").classes("w-72") \
                                    .tooltip("Fuses the work between the matrix multiplies into fewer GPU kernels. "
                                             "Measured at 1 MP on an RTX 5060 Ti: MOSS-Image 26% and Krea 2 12% faster "
                                             "per step, after a 25-40 s warm-up. auto compiles runs of 100+ steps "
                                             "(needs triton, installed with the requirements); a block that fails to "
                                             "compile then runs uncompiled.")
                                w["compile_cache"] = ui.input("Compile cache", value=d.compile_cache) \
                                    .classes("w-72").tooltip(
                                        "Keeps the compiled kernels in a file so another machine with the same GPU "
                                        "model, torch, triton, Python and operating system skips most of the "
                                        "compiling (a fresh vast.ai rental: put a file a rental made, its name ends "
                                        "in -linux.bin, in your model bucket's .rulz_compile folder). auto: "
                                        "<models folder>/.rulz_compile; a folder; or off.")
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
                                                                   value=d.attention_backend, label="Attention kernel") \
                                    .classes("w-72") \
                                    .tooltip("The attention kernel of the DiT blocks. cuDNN ran Krea 2 steps 15-22% "
                                             "faster on an RTX 5060 Ti at the same loss, so auto puts it first on "
                                             "Blackwell GPUs (RTX 50, B200) and lets PyTorch pick elsewhere. The other "
                                             "kernels stay enabled behind the chosen one: a call it cannot take runs on "
                                             "the next.")

                    # ---- Dataset
                    with sections.section("Basics", "Dataset", "Images, captions and resolution", "photo_library"):
                        subhead("Datasets to train on")
                        hint("Tick one or more. Each can train at its own resolutions (none picked: the Training "
                             "resolutions below) and come up several times per epoch (Repeats), to weight a small or "
                             "important dataset up. Create datasets and add images on the Dataset tab.")
                        w["datasets"] = DatasetPicker()
                        w["datasets"].refresh(known_datasets())
                        w["datasets"].value = [{"path": sp.path} for sp in d.dataset_specs() if Path(sp.path).is_dir()]
                        ds_native = _native_dialogs

                        async def refresh_datasets():
                            picker = w["datasets"]
                            picker.refresh(await run.io_bound(known_datasets, picker.selected))
                            picker.set_counts(await run.io_bound(dataset_counts, picker.paths()))

                        async def add_dataset_folder():
                            picked = await pick_path("dir", "", native=ds_native)
                            if picked:
                                try:
                                    register_dataset(_relative(picked))
                                except ValueError as e:
                                    ui.notify(str(e), type="warning")
                                    return
                                w["datasets"].selected = [*w["datasets"].selected, _relative(picked)]
                                await refresh_datasets()

                        with ui.row().classes("items-center gap-2"):
                            ui.button("Refresh list", icon="refresh", on_click=refresh_datasets).props("flat")
                            ui.button("Add a folder", icon="create_new_folder", on_click=add_dataset_folder) \
                                .props("flat").tooltip("A dataset folder you already have, here or elsewhere on "
                                                       "this machine: it joins the list")
                        ui.timer(0.2, refresh_datasets, once=True)
                        with fields():
                            w["trigger_word"] = ui.input("Trigger word").classes("w-64") \
                                .tooltip("Prepended to every caption.")

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
                                            ui.image(image_url(path)).classes("w-28 h-28 object-cover")
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
                                         "step-up, one after another from the lowest. A dataset with its own "
                                         "resolutions (above) trains at those instead.")
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
                        with fields() as video_row:
                            w["train_frames"] = ui.number("Frames per training clip", value=d.train_frames, min=1,
                                                          max=361, precision=0).classes("w-48") \
                                .tooltip("Video clip training. 1 trains on pictures only and leaves video files "
                                         "(.mp4, .mov, .webm, .mkv, .avi, .m4v) out. More trains every video file of "
                                         "the dataset as a clip of that many frames at 24 frames per second, with "
                                         "its caption file beside it, as for a picture; pictures in the same "
                                         "dataset still train as stills. PAN 2 rounds down to 4k + 1 (49 is 2 "
                                         "seconds), MiniMax H3 up to 17n + 5 (39, 56, ...). Face crops, region "
                                         "weights and the identity loss work on pictures only.")
                            w["train_video_mode"] = ui.select(
                                {"fps": "Real time from the start", "spread": "Spread over the whole video"},
                                value=d.train_video_mode, label="Clip frames").classes("w-64") \
                                .tooltip("Real time keeps motion at its speed: the clip is the first seconds of "
                                         "the video at 24 fps (a video too short for it is spread instead). Spread "
                                         "squeezes the whole video onto the clip's frames, which speeds motion up. "
                                         "Trim your videos to the part you want for real time.")
                            w["video_resolution"] = ui.number("Clip size (px, 0 = training resolution)",
                                                              value=d.video_resolution, min=0, step=64,
                                                              precision=0).classes("w-64") \
                                .tooltip("A clip's square-equivalent side (its aspect ratio kept). A clip costs "
                                         "about a picture per latent frame (PAN 2: one per 4 frames, MiniMax H3: 5 "
                                         "per 17), so 512 keeps a 2 second clip near a 4 MP picture's memory.")
                        video_row.bind_visibility_from(w["model_preset"], "value",
                                                       lambda p: preset_arch(p, w["model_path"].value) in VIDEO_ARCHS)
                        with advanced("Captions and cache"):
                            with fields():
                                w["caption_dropout"] = ui.number("Caption dropout", value=d.caption_dropout, min=0,
                                                                 max=0.5, step=0.01).classes("w-36")
                                w["caption_dropout_keep_trigger"] = ui.switch("Dropped captions keep the trigger word",
                                                                              value=d.caption_dropout_keep_trigger) \
                                    .tooltip("A dropped caption becomes the trigger word alone instead of the empty "
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
                                            ui.image(pth).props("fit=contain").classes("w-32 h-32 rz-imgbg")
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
                    with sections.section("Basics", "LoRA", "Adapter size, layers and blocks", "hub"):
                        with fields():
                            w["finetune"] = ui.switch("Fine-tune the whole model, then extract a LoRA",
                                                      value=d.finetune).tooltip(
                                "Trains the target layers' own weights instead of an adapter (no rank bottleneck), "
                                "by rotation: the model stays as loaded (int8 / nvfp4 / bf16), one window of layers "
                                "at a time is held in fp32 and trained for N epochs, then written back as a bf16 "
                                "delta in RAM and re-quantized. Saves extract a LoRA of the chosen rank from the "
                                "deltas, at cycle boundaries. Fizgig measured: an extracted LoRA beat one trained at "
                                "the same rank; rank 64 matched the full checkpoint. Needs a low learning rate "
                                "(1e-5) and a long run; not with a race, data parallel, torch.compile, the weight "
                                "average or the region prior.")
                        with fields() as ft_fields:
                            w["finetune_rotate_epochs"] = ui.number("Epochs per window", value=d.finetune_rotate_epochs,
                                                                    min=0.05, step=0.5).classes("w-40") \
                                .tooltip("Each window of layers trains this many epochs before the next one; a cycle "
                                         "trains every window once. Fizgig: 1.")
                            w["finetune_extract_ranks"] = ui.input("Extract LoRA rank(s)", value="64") \
                                .classes("w-44").tooltip("Every save extracts a LoRA per rank, comma separated "
                                                         "(64, 128). Rank 64 was indistinguishable from the full "
                                                         "checkpoint in Fizgig's tests.")
                            w["finetune_window_gb"] = ui.number("Window VRAM (GB, 0 = auto)", value=d.finetune_window_gb,
                                                                min=0, step=0.5).classes("w-48") \
                                .tooltip("What one window may hold: fp32 weights, gradients and optimizer state. "
                                         "Auto: the free VRAM after the model loads, less the activation reserve.")
                            w["finetune_reserve_gb"] = ui.number("Activation reserve (GB, 0 = auto)",
                                                                 value=d.finetune_reserve_gb, min=0, step=0.5) \
                                .classes("w-56").tooltip("VRAM kept out of the windows for activations and "
                                                         "fragmentation. Auto: 3 GB + 3 GB per megapixel.")
                            w["finetune_save_delta"] = ui.switch("Also save the deltas", value=d.finetune_save_delta) \
                                .tooltip("<job>_finetune.safetensors, 2 bytes per trained weight (MOSS ~10 GB): the "
                                         "fine-tune's state, to continue from or to extract at other ranks later "
                                         "(python -m rulztrainer.extract_finetune).")
                            w["finetune_anchor"] = ui.number("Pull toward the original (0 = off)",
                                                             value=d.finetune_anchor, min=0, step=0.005,
                                                             format="%.3f").classes("w-60") \
                                .tooltip("After each step the weights move this much x the learning rate back "
                                         "toward the original model's: bounded drift (Preserve and Personalize, ICLR "
                                         "2026), measured above an unheld fine-tune in likeness and prompt-following. "
                                         "It replaces weight decay, which would shrink the model's own weights toward "
                                         "zero. Keeps the window's original weights on the GPU (1-2 bytes each).")
                            w["finetune_resume"] = ui.input("Continue from deltas (optional)",
                                                            value=d.finetune_resume).classes("w-96 max-w-full")
                        ft_fields.bind_visibility_from(w["finetune"], "value")
                        ft_hint = hint("")
                        ft_hint.bind_visibility_from(w["finetune"], "value")

                        def on_finetune(e):
                            if not e.value:
                                ft_hint.text = ""
                                return
                            ft_hint.text = ("Rank, alpha, the adapter kind, LoRA+ and dropout do not apply to a "
                                            "fine-tune; the target layers and blocks do. A fine-tune wants a much "
                                            "lower learning rate than a LoRA.")
                            if float(w["learning_rate"].value or 0) > 5e-5:
                                w["learning_rate"].value = 1e-5
                                w["optimizer"].value = "adafactor"
                                ui.notify("Fine-tune: learning rate set to 1e-5 and optimizer to adafactor "
                                          "(Fizgig's measured settings)", type="info")

                        w["finetune"].on_value_change(on_finetune)
                        with fields():
                            w["network_type"] = ui.select({"lora": "LoRA", "lokr": "LoKr", "oft": "OFT"},
                                                          value=d.network_type, label="Adapter").classes("w-28") \
                                .tooltip("LoRA: a rank-r pair per layer (rank, alpha). LoKr: a Kronecker pair, w1 "
                                         "up to factor x factor and w2 the rest (about 1/64 of the weights at "
                                         "factor 8, Fizgig's measured default: its highest likeness, more natural "
                                         "skin, ~20% slower steps). OFT: rotates each layer's output channels in "
                                         "blocks instead of adding to the weights, which keeps the base model's "
                                         "structure; the OFT paper found it held a subject's identity better than "
                                         "LoRA and stayed stable in long runs. LoKr and OFT are LyCORIS files that "
                                         "load in ComfyUI (Krea 2 included); diffusers has no loader for them. Rank, "
                                         "alpha, LoRA+ and dropout are LoRA-only.")
                            w["lora_rank"] = ui.number("Rank", value=d.lora_rank, min=1, max=256, precision=0) \
                                .classes("w-28") \
                                .bind_visibility_from(w["network_type"], "value", lambda x: x == "lora")
                            w["lora_alpha"] = ui.number("Alpha", value=d.lora_alpha).classes("w-28") \
                                .bind_visibility_from(w["network_type"], "value", lambda x: x == "lora")
                            w["oft_block_size"] = ui.number("OFT block size", value=d.oft_block_size, min=2, max=256,
                                                            precision=0).classes("w-36") \
                                .bind_visibility_from(w["network_type"], "value", lambda x: x == "oft") \
                                .tooltip("Each layer's output channels are rotated in blocks of this size (the "
                                         "largest divisor of the layer's outputs up to it). Larger blocks rotate more "
                                         "channels together: more freedom and weights (32: about as many as a rank "
                                         "16 LoRA on a square layer).")
                            w["oft_constraint"] = ui.number("OFT constraint (0 = off)", value=d.oft_constraint, min=0,
                                                            step=0.01).classes("w-48") \
                                .bind_visibility_from(w["network_type"], "value", lambda x: x == "oft") \
                                .tooltip("COFT: caps how far each layer may rotate away from the base model (the "
                                         "norm of its rotation generator), saved as the file's alpha, which ComfyUI "
                                         "applies the same way. 0 = no cap.")
                            w["oft_scaled_lr"] = ui.checkbox("Scaled OFT lr (experimental)", value=d.oft_scaled_lr) \
                                .bind_visibility_from(w["network_type"], "value", lambda x: x == "oft") \
                                .tooltip("An lr per block size, so every block size moves about as far per step as "
                                         "a block of 32 (OneTrainer's Scaled OFT). Off, a block of 64 steps about "
                                         "twice as far as one of 16 at the same lr, which skews a race of block "
                                         "sizes. Block 32 is unchanged. Untested in RulzTrainer.")
                            w["lokr_factor"] = ui.number("LoKr factor", value=d.lokr_factor, min=1, max=64,
                                                         precision=0).classes("w-32") \
                                .bind_visibility_from(w["network_type"], "value", lambda x: x == "lokr") \
                                .tooltip("w1 is [a, b] with a and b the largest divisors of each layer's output / "
                                         "input size that are <= the factor; w2 holds the rest. Lower = more "
                                         "capacity and bigger files (8: ~400 MB on a 12B model, 16: ~100 MB). "
                                         "Fizgig found no quality gain above 8.")
                            w["dora"] = ui.switch("DoRA", value=d.dora) \
                                .bind_visibility_from(w["network_type"], "value", lambda x: x != "oft") \
                                .tooltip("Weight-decomposed LoRA / LoKr: each layer's weight is split into a "
                                         "direction (the adapter changes it) and a magnitude per input channel that "
                                         "trains on its own. Saved as dora_scale keys that ComfyUI applies; "
                                         "diffusers cannot read them. Each step also computes the layer's norm from "
                                         "its full weight: steps about 40% slower (Krea 2).")
                            w["lora_plus_ratio"] = ui.number("LoRA+ ratio (0 = off, 16 = paper)",
                                                             value=d.lora_plus_ratio, step=1).classes("w-60") \
                                .bind_visibility_from(w["network_type"], "value", lambda x: x == "lora") \
                                .tooltip("lora_up learns at lr x ratio (faster convergence); ignored by automagic3/4")
                            w["lora_save_dtype"] = ui.select(["bf16", "fp16", "fp32"], value=d.lora_save_dtype,
                                                             label="Save LoRA as").classes("w-36")
                        with fields():
                            w["adapter_mode"] = ui.select(ADAPTER_MODES, value="auto", label="Training adapter") \
                                .classes("w-96") \
                                .tooltip("A LoRA that stays on while your LoRA trains and off for samples and probes, "
                                         "never saved into it, so your LoRA works on the plain model. Qwen-Image "
                                         "2.1's is Fizgig's (downloaded, 80 MB): without one its LoRAs collapse "
                                         "into texture or wobble part-way (likeness 77 with it, 56 without). "
                                         "MiniMax H3's is circlestone-labs' image training adapter (0.62 GB), "
                                         "P-H3-Image's and MiniMax-H3-Image's are ostris' adapters for them "
                                         "(0.16 GB each). The other models have none of their own; a file "
                                         "works on any.")
                            path_input(f, "training_adapter", "Training adapter file", "file", "", "w-[30rem]")
                            f.w["training_adapter"].parent_slot.parent.bind_visibility_from(w["adapter_mode"], "value",
                                                                                            lambda x: x == "file")
                            w["training_adapter_strength"] = ui.number("Adapter strength",
                                                                       value=d.training_adapter_strength, step=0.05) \
                                .classes("w-40").bind_visibility_from(w["adapter_mode"], "value",
                                                                      lambda x: x != "none")
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
                                w["lora_blocks"] = ui.input("Blocks (empty = all)", value=d.lora_blocks) \
                                    .classes("w-[30rem] max-w-full") \
                                    .tooltip("e.g. 0-41, 21-41, 10-31,40 or even, or tick them below. Training only "
                                             "later blocks also skips the backward pass through earlier ones, unless "
                                             "LoRA layers sit before the blocks (Krea's text fusion, embedders, "
                                             "MOSS's text projection): their gradient passes through every block.")
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
                                                            value=d.save_state) \
                                    .tooltip("Lets a stopped or crashed run continue where it was. Each state "
                                             "takes about 3 times the size of the LoRA in fp32")
                                w["save_state_every"] = ui.number("State every N steps (0 = with each LoRA save)",
                                                                  value=d.save_state_every, precision=0) \
                                    .classes("w-80")
                                w["keep_states"] = ui.number("States to keep (rolling, 0 = all)", value=d.keep_states,
                                                             precision=0).classes("w-60")
                                w["oom_retries"] = ui.number("Out-of-memory retries (0 = off)", value=d.oom_retries,
                                                             min=0, precision=0).classes("w-60") \
                                    .tooltip("When a run ends because GPU or system memory ran out, the WebUI starts "
                                             "it again by itself, before the next queued run: from its newest "
                                             "training state, first on the same settings, then one step lighter "
                                             "(checkpointing, kept blocks, block swap) if it runs out again at about "
                                             "the same point. This many tries a run at most")
                            path_input(f, "resume_state", "Resume from training state folder (state_XXXXXX, optional)",
                                       "dir", "", "w-[40rem]")

                    # ---- Optimization
                    with sections.section("Basics", "Optimization", "Steps, learning rate, optimizer and timesteps",
                                          "speed"):
                        with fields():
                            w["steps"] = ui.number("Steps", value=d.steps, precision=0).classes("w-28")
                            w["batch_size"] = ui.number("Batch size", value=d.batch_size, precision=0).classes("w-28")
                            w["learning_rate"] = ui.number("Learning rate", value=d.learning_rate, format="%.2e",
                                                           step=1e-5).classes("w-36")
                            w["optimizer"] = ui.select(list(OPTIMIZERS), value="adamw8bit", label="Optimizer") \
                                .classes("w-40")
                            w["lr_scheduler"] = ui.select(["constant", "cosine", "linear"], value=d.lr_scheduler,
                                                          label="LR schedule").classes("w-36")
                            w["warmup_steps"] = ui.number("Warmup steps", value=d.warmup_steps, precision=0) \
                                .classes("w-32")
                        with fields().bind_visibility_from(w["optimizer"], "value", lambda x: x == "hull"):
                            w["hull_rho"] = ui.number("Hull step", value=d.hull_rho, min=0, step=0.0001,
                                                      format="%.6f").classes("w-40") \
                                .tooltip("How far each step moves a layer, as a share of that layer's own weight. "
                                         "Required: there is no default. Start at 0.0002: in races on one set it "
                                         "matched AdamW 8-bit, and 0.0001 scored clearly lower without copying less "
                                         "(docs/keel-test-results.md). Runs vary a lot, so watch the copy score. The "
                                         "learning-rate schedule acts on it.")
                            w["hull_regions"] = ui.select({"off": "Off", "parts": "Body parts"}, value=d.hull_regions,
                                                          label="Hull body regions").classes("w-40") \
                                .tooltip("Experimental: in two races on one set it did not improve the body, so leave "
                                         "it off unless you are testing it. Body parts finds skin, hair and clothes in "
                                         "each photo and pushes the update harder on body skin and the body's outline. "
                                         "The face keeps its own share. It scans the photos once, turns torch.compile "
                                         "off, and refuses flip augmentation and region-weighted loss.")
                            w["hull_gain_body"] = ui.number("Hull body gain", value=d.hull_gain_body, min=0.5, max=2,
                                                            step=0.1).classes("w-32") \
                                .bind_visibility_from(w["hull_regions"], "value", lambda x: x == "parts") \
                                .tooltip("How much harder body cells push the update. 1 changes nothing.")
                        with ui.column().classes("w-full gap-2").bind_visibility_from(
                                w["optimizer"], "value", lambda x: x in AUTOMAGIC):
                            hint("Automagic tunes its own learning rate (the learning rate above is its start); LR "
                                 "schedule and warmup are ignored. Automagic 4 also caps the step of the most "
                                 "sensitive weights and its rate holds on noise instead of drifting up. Keel is "
                                 "Automagic 4 plus the LoRA gauge preconditioner and the cautious mask below. "
                                 "Keel, from real tests: use one lr for every layer; the Max lr made no measurable "
                                 "difference from 3e-4 to 1e-3; and 1000 steps was not yet enough, so plan on more. "
                                 "The other Keel settings are not measured yet (docs/keel-test-results.md).")
                            with fields():
                                w["automagic_min_lr"] = ui.number("Min lr", value=d.automagic_min_lr, format="%.1e") \
                                    .classes("w-32")
                                w["automagic_max_lr"] = ui.number("Max lr", value=d.automagic_max_lr, format="%.1e") \
                                    .classes("w-32") \
                                    .tooltip("The ceiling of the self-tuned rate. Keel climbs to the default 1e-3 by "
                                             "about step 200. In a race over 8 seeds, 1e-3, 5e-4 and 3e-4 tied at 1000 "
                                             "steps on one body-only set, so keep the default or use 3e-4 for the "
                                             "gentlest setting at the same score.")
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
                                             "with a polarity history of 16. With Keel, use one lr: it beat by_type by "
                                             "4.4 body points over 8 seeds on one body-only set (p = 0.03).")
                            with fields().bind_visibility_from(w["optimizer"], "value",
                                                               lambda x: x in ("automagic4", *KEEL)):
                                w["automagic_sensitivity_quantile"] = ui.number("Sensitivity quantile",
                                                                                value=d.automagic_sensitivity_quantile,
                                                                                min=0.05, max=1, step=0.05) \
                                    .classes("w-40") \
                                    .tooltip("Weights whose gradient power is above this share of the LoRA's elements "
                                             "take a smaller, Newton-like step (0.9: the top 10%, on Krea 2 the "
                                             "attention values and keys of the early blocks and the timestep path).")
                                w["automagic_vote"] = ui.select({"graded": "graded (trend minus flips)",
                                                                 "extremes": "extremes (Automagic 3's rule)"},
                                                                value=d.automagic_vote, label="Vote").classes("w-56") \
                                    .tooltip("How the weights vote on the learning rate. graded: the mean update sign "
                                             "squared minus the sign-flip rate, zero on noise, up on a trend, down on "
                                             "overshoot (holds the rate on a noisy diffusion loss, climbs while there "
                                             "is signal, anneals after). extremes: only unanimous windows vote up and "
                                             "perfectly alternating ones down, as in Automagic 3.")
                                w["automagic_max_lr_change"] = ui.number("Max lr change / step",
                                                                         value=d.automagic_max_lr_change, min=0.001,
                                                                         max=1, step=0.01).classes("w-40") \
                                    .tooltip("The most the learning rate may move in one step (0.05 = about 5%); the "
                                             "signs that judge a step only arrive over the next polarity-history "
                                             "steps, so this bounds how far the rate can overshoot meanwhile.")
                            with fields().bind_visibility_from(w["optimizer"], "value", lambda x: x in KEEL):
                                w["keel_gauge"] = ui.switch("Gauge preconditioner", value=d.keel_gauge) \
                                    .tooltip("Each LoRA pair's gradient is scaled by the other factor's r x r Gram "
                                             "(Riemannian preconditioned LoRA), so the update does not depend on "
                                             "how the delta is split between lora_down and lora_up and neither "
                                             "factor takes it all. LoKr and OFT have no pairs: nothing happens.")
                                w["keel_gauge_eps"] = ui.number("Gauge eps", value=d.keel_gauge_eps, min=0.0001,
                                                                max=1, step=0.01, format="%.3f").classes("w-32") \
                                    .tooltip("The preconditioner's floor as a fraction of the larger Gram's mean "
                                             "diagonal: the still-small factor (lora_up starts at zero) is scaled "
                                             "by about the identity until it has grown. Raise it if the cautious "
                                             "mask's kept share in the log falls well under 1.")
                                w["keel_cautious"] = ui.switch("Cautious mask", value=d.keel_cautious) \
                                    .tooltip("Zero a preconditioned gradient element that points against the raw "
                                             "gradient (Cautious Optimizers, ICLR 2026), so the step never climbs "
                                             "the loss along a weight. The kept share is logged every 250 steps.")
                            with fields().bind_visibility_from(w["optimizer"], "value", lambda x: x in KEEL):
                                w["keel_pair_sigma"] = ui.switch("Paired sigma", value=d.keel_pair_sigma) \
                                    .tooltip("One noise level per step for all of its micro-batches (and the DOP "
                                             "pass), so the step's gradient sits in one noise band: what the band "
                                             "normalisation needs, and what makes the micro-batches comparable.")
                                w["keel_bands"] = ui.number("Noise bands", value=d.keel_bands, min=1, max=64,
                                                            precision=0).classes("w-32") \
                                    .tooltip("Bands over the noise level (logit sigma from -4 to 4, so sigma 0.02 to "
                                             "0.98 in 8 bands). Each weight tensor keeps the gradient power it sees "
                                             "per band, and a step's gradient enters the optimizer at the tensor's "
                                             "overall scale whatever its band, so the quiet bands (where the face "
                                             "forms) are not drowned by the loud ones. 1 = off. Needs paired sigma "
                                             "or one micro-batch per step; the per-band imbalance is logged every "
                                             "250 steps.")
                                w["keel_band_weights"] = ui.input("Band weights", value=d.keel_band_weights,
                                                                  placeholder="flat").classes("w-56") \
                                    .tooltip("A multiplier on the step's update per band: empty = 1 everywhere, "
                                             "identity = a hump over the face bands (sigma 0.12 to 0.5), subject = "
                                             "a plateau over the face's and the body's bands alike (sigma 0.18 to "
                                             "0.82, for keel_fb), or one number per band, comma-separated.")
                                w["keel_consensus"] = ui.number("Consensus gate", value=d.keel_consensus, min=0, max=1,
                                                                step=0.1, format="%.2f").classes("w-32") \
                                    .tooltip("Cross-photo consensus (the AND-mask, ICLR 2021): each weight's update "
                                             "is scaled by (1 - c) + c x (agreement share)^power, the share being "
                                             "how many of the step's gradients (micro-batches, the DOP pass) agree "
                                             "on its sign, so what every photo shares (the face) moves and what "
                                             "varies between them (outfit, background, light) moves less. 0 = off, "
                                             "1 = only unanimous elements move. Needs gradient accumulation of 2 or "
                                             "more; paired sigma keeps the noise level out of the disagreement.")
                                w["keel_consensus_power"] = ui.number("Consensus power", value=d.keel_consensus_power,
                                                                      min=0.1, max=8, step=0.5, format="%.1f") \
                                    .classes("w-32") \
                                    .tooltip("The exponent on the agreement share: higher is stricter on partial "
                                             "agreement (2: a 3-of-4 split counts as 0.25 agreement, before the "
                                             "floor).")
                                w["keel_consensus_mode"] = ui.select({"auto": "auto (sign for keel, magnitude for keel_fb)",
                                                                      "sign": "sign: one vote per gradient",
                                                                      "magnitude": "magnitude: votes weighted by size"},
                                                                     value=d.keel_consensus_mode,
                                                                     label="Consensus vote").classes("w-72") \
                                    .tooltip("How the step's gradients vote on an element. sign: each micro-batch "
                                             "one vote (1 byte per weight for the step). magnitude: each weighted by "
                                             "the size of its gradient (the sum's size over the sum of the sizes), so "
                                             "a photo that says little about a weight (a face crop, for the weights "
                                             "serving the body) barely votes instead of casting a random one (4 "
                                             "bytes per weight for the step).")
                                w["keel_lookahead"] = ui.switch("Lookahead", value=d.keel_lookahead) \
                                    .tooltip("Probe-gated lookahead: the weights that scored a new best likeness "
                                             "(by the noise band) become the anchor; after the grace probes without "
                                             "a gain the weights are pulled part of the way back toward the anchor "
                                             "and the learning rate is cut, so a run that wanders off its best is "
                                             "damped to it. Needs likeness probes (face refs). Not measured yet: leave it "
                                             "off, and race it as its own row.")
                                w["keel_lookahead_pullback"] = ui.number("Pull-back", value=d.keel_lookahead_pullback,
                                                                         min=0.05, max=1, step=0.05, format="%.2f") \
                                    .classes("w-32") \
                                    .tooltip("How far a pull moves the weights toward the anchor (1 = all the way).")
                                w["keel_lookahead_lr"] = ui.number("Pull lr factor", value=d.keel_lookahead_lr,
                                                                   min=0.1, max=1, step=0.05, format="%.2f") \
                                    .classes("w-32") \
                                    .tooltip("The learning-rate multiplier at each pull (within the Automagic rails).")
                                w["keel_lookahead_grace"] = ui.number("Grace probes", value=d.keel_lookahead_grace,
                                                                      min=0, max=20, precision=0).classes("w-32") \
                                    .tooltip("Probes without a gain before a pull; the count starts over after a "
                                             "pull or an accept.")
                                w["keel_face_gain"] = ui.number("Face gain", value=d.keel_face_gain, min=0, max=4,
                                                                step=0.25, format="%.2f").classes("w-32") \
                                    .tooltip("Face-token gain: each LoRA pair's step is scaled by (the share of its "
                                             "gradient that comes from the face's tokens / the median pair's share) "
                                             "to this power, so the weights serving the face move more and the "
                                             "background's less. Needs region maps with faces (face detection). "
                                             "0 = off; not measured yet, so race it at 0.5 and 1. keel_fb measures the "
                                             "body's tokens too (Body weight).")
                                w["keel_face_gain_max"] = ui.number("Face gain limit", value=d.keel_face_gain_max,
                                                                    min=1, max=10, step=0.5, format="%.1f") \
                                    .classes("w-32").tooltip("The gain stays within [1 / limit, limit].")
                                w["keel_face_gain_beta"] = ui.number("Face gain EMA", value=d.keel_face_gain_beta,
                                                                     min=0, max=0.99, step=0.05, format="%.2f") \
                                    .classes("w-32") \
                                    .tooltip("How much of the previous gain each step keeps (0.9: one batch without "
                                             "a face barely moves it).")
                            with fields().bind_visibility_from(w["optimizer"], "value", lambda x: x == "keel_fb"):
                                w["keel_body_weight"] = ui.number("Body weight", value=d.keel_body_weight, min=0, max=4,
                                                                  step=0.25, format="%.2f").classes("w-32") \
                                    .tooltip("keel_fb: the body tokens' weight against the face's in the token gain "
                                             "(the person's cells outside the face box, from the person masks). Each "
                                             "region's share is measured against its own median pair and the two are "
                                             "combined as a weighted geometric mean, so a step of person-only photos "
                                             "moves the gain by the body alone and face crops by the face alone. 1 = "
                                             "equal; 0 = face only. Needs Face gain above 0 (off by default: then "
                                             "keel_fb differs from keel by the consensus vote alone). Set the probes' "
                                             "face and body weights equal too. Not measured yet: race keel_fb against "
                                             "keel with gradient accumulation of 2 or more, or the two are alike.")
                        with advanced("Timesteps, loss weighting and gradients"):
                            with fields():
                                w["gradient_accumulation"] = ui.number("Grad accumulation",
                                                                       value=d.gradient_accumulation,
                                                                       precision=0).classes("w-40")
                                w["max_grad_norm"] = ui.number("Max grad norm (0 = off)", value=d.max_grad_norm) \
                                    .classes("w-44")
                                w["loss_weighting"] = ui.select(["none", "min_snr", "sigma_sqrt"], value="none",
                                                                label="Loss weighting").classes("w-40")
                                w["loss_type"] = ui.select(["mse", "band"], value=d.loss_type, label="Loss type")                                     .classes("w-32")                                     .tooltip("mse: the plain loss. band (experimental, from AI-Toolkit): splits each "
                                             "error into fine-to-coarse detail bands and drops the bands the noise "
                                             "has buried at that step, so very noisy steps stop pushing on details "
                                             "nobody could see. Near no noise it is the same as mse. Its loss value "
                                             "runs lower than mse (most at high noise): compare runs by samples and "
                                             "probes, not by the loss. A race can try both.")
                            with fields():
                                w["timestep_sampling"] = ui.select(["logit_normal", "sigmoid", "uniform", "identity",
                                                                    "content", "style", "logsnr", "reference"],
                                                                   value=d.timestep_sampling,
                                                                   label="Timestep sampling").classes("w-44") \
                                    .tooltip("identity: logit-normal mean -0.5 (detail/low noise). content: sigma = u^p "
                                             "toward low noise (ai-toolkit style). style: mirror toward high noise. "
                                             "reference: the model family's own training recipe (Bria's for FIBO 2.0, "
                                             "the MOSS-Image report's for MOSS). logsnr: the style-friendly sampler "
                                             "(log-SNR ~ N(mean, std), mean -6: mostly very noisy steps, where style is "
                                             "learned; for style LoRAs).")
                                w["logsnr_mean"] = ui.number("Log-SNR mean", value=d.logsnr_mean, step=0.5) \
                                    .classes("w-32").bind_visibility_from(w["timestep_sampling"], "value",
                                                                          lambda x: x == "logsnr")
                                w["logsnr_std"] = ui.number("Log-SNR std", value=d.logsnr_std, min=0.1, step=0.5) \
                                    .classes("w-32").bind_visibility_from(w["timestep_sampling"], "value",
                                                                          lambda x: x == "logsnr")
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
                                w["timestep_buckets"] = ui.number("Timestep buckets (0 = off)",
                                                                  value=d.timestep_buckets, min=0, precision=0) \
                                    .classes("w-48").tooltip(
                                        "Cuts the sampling curve above into this many equal slices and makes every "
                                        "that many training draws take one from each, so a short run or a small "
                                        "dataset covers the whole curve evenly instead of by luck. The curve's shape "
                                        "stays the same. 2 and more; 10 to 20 is a usual start (musubi-tuner's "
                                        "num_timestep_buckets). Not with the reference sampler.")
                            with fields():
                                w["clean_share"] = ui.input("Clean-end share (%, by epoch)", value=d.clean_share,
                                                            placeholder="empty = the sampler's own") \
                                    .classes("w-64").tooltip(
                                        "The share of training steps at the clean end (noise below half), where faces "
                                        "and fine detail are refined; the sampling curve above is moved to hit it. "
                                        "40 = 40% all run. 75:5:40 = 75% for 5 epochs, then 40%. 60:7:30:8:3 = "
                                        "Fizgig's MOSS-Image Slide: 60% until epoch 7, 30% at epoch 8, then 60% with "
                                        "every 3rd epoch at 30%, at half the learning rate. A full fine-tune counts "
                                        "rotations instead of epochs (Slide FT: 11:4:30:8:2). MiniMax-H3-Image "
                                        "trains at 6 fixed noise levels only: there it is the share of steps at the "
                                        "last one (0.232).")
                                ui.button("Slide", on_click=lambda: w["clean_share"].set_value(CLEAN_SLIDE)) \
                                    .props("flat dense").tooltip("Fizgig's MOSS-Image LoRA default until 8 November "
                                                                 "2026, 60:7:30:8:3 (measured on AdamW)")
                                ui.button("Slide Muon",
                                          on_click=lambda: w["clean_share"].set_value(CLEAN_SLIDE_MUON)) \
                                    .props("flat dense").tooltip(
                                        "Fizgig's MOSS-Image LoRA default since 10 November 2026, 11:7:40:8:3: 11% "
                                        "until epoch 7, 40% from epoch 8, every 3rd epoch at 11% at half the learning "
                                        "rate. Tuned for Muon, together with a face boost RulzTrainer does not have; "
                                        "not measured here")
                                ui.button("Slide FT", on_click=lambda: w["clean_share"].set_value(CLEAN_SLIDE_FT)) \
                                    .props("flat dense").tooltip("Fizgig's MOSS-Image full fine-tune default, "
                                                                 "11:4:30:8:2 (rotations)")
                                ui.button("H3 Image 40",
                                          on_click=lambda: w["clean_share"].set_value(CLEAN_H3_IMAGE)) \
                                    .props("flat dense").tooltip(
                                        "Fizgig's MiniMax-H3-Image presets: 40% of the steps at the model's last "
                                        "noise level (0.232), where skin, eyes and fine detail are made, and 12% on "
                                        "each of the other five (about 17% each without it). With timestep sampling "
                                        "reference")
                                clean_hint = hint("")

                                def clean_hint_update(_=None):
                                    clean_hint.text = clean_share_hint(f)

                                ui.timer(0.1, clean_hint_update, once=True)
                                for key in ("timestep_sampling", "logit_mean", "logit_std", "sigmoid_scale",
                                            "content_power", "sigma_min", "sigma_max", "timestep_shift",
                                            "clean_share"):
                                    w[key].on_value_change(clean_hint_update)

                    # ---- Saving & samples
                    with sections.section("Basics", "Saving & samples", "Checkpoints and preview images", "save"):
                        with fields():
                            w["run_report"] = ui.switch("Run report when training ends", value=d.run_report).tooltip(
                                "Writes a report of the run into its job folder (report/): a summary picture with the "
                                "settings, the scores, the probe renders at several points of the run with the best "
                                "checkpoint marked, examples of the dataset and the training maps; and a longer PDF with "
                                "every probe step, every changed setting, captions and the log's warnings. Takes about "
                                "half a minute at the end. Previous Runs makes the same for an earlier run.")
                            w["save_every"] = ui.number("Save every N steps", value=d.save_every, precision=0) \
                                .classes("w-44")
                            w["keep_checkpoints"] = ui.number("Checkpoints to keep (0 = all)",
                                                              value=d.keep_checkpoints, min=0, precision=0) \
                                .classes("w-56").tooltip(
                                    "Keeps only this many of the step checkpoints (the LoRA / LoKr files saved every "
                                    "N steps); each new one deletes the oldest. The final file, the best-likeness "
                                    "file and the race's dropped files are never deleted. 0 keeps every one.")
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
                            aspect0, mp0 = size_aspect(d.sample_width, d.sample_height)
                            w["sample_aspect"] = ui.select(SAMPLE_ASPECTS, value=aspect0, label="Aspect ratio") \
                                .classes("w-44")
                            w["sample_mp"] = ui.select(MP_OPTIONS, value=mp0, label="Resolution").classes("w-44")
                            size_label = ui.label("").classes("rz-hint self-center")

                            def show_size(_=None):
                                try:
                                    sw, sh = aspect_size(w["sample_aspect"].value, w["sample_mp"].value)
                                    size_label.text = f"= {sw} x {sh}"
                                except (TypeError, ValueError):
                                    size_label.text = ""

                            for key in ("sample_aspect", "sample_mp"):
                                w[key].on_value_change(show_size)
                            show_size()
                        with fields():
                            w["turbo_mode"] = ui.select(TURBO_MODES, value="auto", label="Turbo LoRA").classes("w-80") \
                                .tooltip("A step-distilled LoRA used only while rendering samples and probes, never "
                                         "in training. The model's own: MOSS-Image lodestones' 8-step DMD LoRA (12 "
                                         "steps, CFG 1, shift 5), Krea 2 and Qwen-Image 2.1 Comfy-Org's Turbo LoRAs "
                                         "(8 steps, CFG 1), FIBO 2.0 Comfy-Org's Turbo LoRA (4 steps, CFG 1), "
                                         "Ideogram 4 ostris' TurboTime LoRA (8 steps, CFG 1), MiniMax H3 "
                                         "lightx2v's FL2VA Turbo 8-step LoRA (8 steps, CFG 1).")
                            path_input(f, "turbo_lora", "Turbo / step-distill LoRA file", "file", "", "w-[30rem]")
                            f.w["turbo_lora"].parent_slot.parent.bind_visibility_from(w["turbo_mode"], "value",
                                                                                      lambda x: x == "file")
                            w["turbo_lora_strength"] = ui.number("Turbo LoRA strength", value=d.turbo_lora_strength,
                                                                 step=0.05).classes("w-44") \
                                .bind_visibility_from(w["turbo_mode"], "value", lambda x: x != "none")

                            def on_turbo_mode(e):
                                was = getattr(on_turbo_mode, "last", "auto") != "none"
                                on_turbo_mode.last = e.value
                                if f.loading:
                                    return
                                fam = preset_arch(w["model_preset"].value, w["model_path"].value)
                                f.follow_sample_defaults(fam, fam, was, e.value != "none")

                            w["turbo_mode"].on_value_change(on_turbo_mode)
                        with fields():
                            w["sample_steps"] = ui.number("Steps", value=d.sample_steps, precision=0).classes("w-28")
                            w["sample_cfg"] = ui.number("CFG", value=d.sample_cfg).classes("w-28")
                            w["sample_shift"] = ui.number("Shift", value=d.sample_shift).classes("w-28") \
                                .tooltip("MOSS-Image only (5); the other models take their shift from the image size "
                                         "(FLUX 3 Klein's is fixed at 6.93)")
                        hint("FIBO 2.0: with its turbo LoRA (Edit Turbo as a LoRA on dev) samples render 4 steps at "
                             "CFG 1; without it Bria samples dev at 50 steps / CFG 5 (20-30 is fine for previews). An "
                             "empty negative prompt means the null caption it was trained with. Prompts can be plain "
                             "sentences or VGL JSON captions.").bind_visibility_from(
                            w["model_preset"], "value", lambda p: is_fibo_preset(p, w["model_path"].value))
                        hint("Krea 2: with its turbo LoRA samples render 8 steps at CFG 1. Without it, Raw needs "
                             "many steps with guidance (Krea's pipeline: 28 steps at their guidance 4.5, CFG 5.5 "
                             "here).").bind_visibility_from(
                            w["model_preset"], "value", lambda p: preset_arch(p, w["model_path"].value) == "krea2")
                        hint("Qwen-Image 2.1: with its turbo LoRA samples render 8 steps at CFG 1 on the Turbo "
                             "checkpoint's own schedule. Without it, 25 steps at CFG 1 (ComfyUI's template; the model "
                             "is meant to be sampled without guidance).").bind_visibility_from(
                            w["model_preset"], "value",
                            lambda p: preset_arch(p, w["model_path"].value) == "qwen_image21")
                        hint("Ideogram 4: with ostris' TurboTime LoRA (community, 847 MB) samples render 8 steps at "
                             "CFG 1, no unconditional pass. Without it, Ideogram's V4_DEFAULT_20: 20 steps at CFG 7, "
                             "the last 2 at 3. Its guidance pass runs the image without text on the model you train "
                             "(Ideogram's pipeline uses a separate unconditional model, not loaded here), so full "
                             "samples are close to, not the same as, the official pipeline's. Prompts can be plain "
                             "sentences (wrapped into its JSON caption) or JSON captions, which it was trained on; an "
                             "empty negative prompt is the text-free pass.").bind_visibility_from(
                            w["model_preset"], "value",
                            lambda p: preset_arch(p, w["model_path"].value) == "ideogram4")
                        hint("MiniMax H3: samples are single-frame videos on the plain model (the training adapter "
                             "off): with lightx2v's 8-step turbo LoRA 8 steps at CFG 1, without it 30 steps at CFG 1 "
                             "(H3 is guidance-distilled). Its shifts are its own (12 video, 3 audio): Shift is "
                             "ignored. The turbo LoRA was distilled on video, not checked on stills."
                             ).bind_visibility_from(w["model_preset"], "value",
                                                    lambda p: preset_arch(p, w["model_path"].value) == "minimax_h3")
                        hint("PAN 2: samples are single-frame videos at the released settings, 50 steps at CFG 2.5 "
                             "with the model's structured negative prompt; Shift is ignored (its own is 9) and there "
                             "is no turbo LoRA. PAN 2 is a video model that was finished on video alone, so a still "
                             "shows what the LoRA learned rather than the model at its best. Single files are not "
                             "published: make them with scripts/pan2/convert_pan2.py and set all three."
                             ).bind_visibility_from(w["model_preset"], "value",
                                                    lambda p: preset_arch(p, w["model_path"].value) == "pan2")
                        hint("P-H3-Image: samples render on the plain model (the training adapter off) at its own "
                             "8-step grid, CFG 1; Shift is ignored. It is tuned for 4-8 MP images: 1 MP previews show "
                             "more artifacts in faces, hands and small details than the finished LoRA will at full "
                             "size.").bind_visibility_from(w["model_preset"], "value",
                                                           lambda p: preset_arch(p, w["model_path"].value) ==
                                                           "p_h3_image")
                        hint("MiniMax-H3-Image: samples render its own 6 steps at CFG 1, each step through the "
                             "output head made for it; Shift is ignored and no turbo LoRA applies. The model is made "
                             "for 4 to 20 MP: MiniMax's own pipeline renders anything smaller on a 4 MP canvas and "
                             "scales it down. Here a preview renders at the size you set, so set Resolution to 4 MP "
                             "to see the LoRA at the model's real quality (4096 tokens a step, about four times a "
                             "1 MP preview's time).").bind_visibility_from(
                            w["model_preset"], "value",
                            lambda p: preset_arch(p, w["model_path"].value) == "minimax_h3_image")
                        hint("MOSS-Image: lodestones' turbo LoRA renders 12 steps at CFG 1 on Base. Fizgig found it "
                             "can understate likeness: judge a run's likeness on full-quality samples (Turbo LoRA: "
                             "none, 20 steps, CFG 3). Training on Base, the samples also carry Kijai's SFT LoRA "
                             "(Sampling LoRA: auto, under Seeds, negative prompt and sampling models)."
                             ).bind_visibility_from(
                            w["model_preset"], "value", lambda p: preset_arch(p, w["model_path"].value) == "moss_image")
                        hint("FLUX 3 Klein: no turbo LoRA. Base samples 50 steps at CFG 3.5 (BFL's settings, about 2.5 "
                             "min per 1 MP preview on a 16 GB card). Faster previews: set the distilled "
                             "black-forest-labs/flux-3-image-klein (or its -comfy flux3-image-klein-*.safetensors) as "
                             "the sampling model below and sample 16 steps at CFG 1.").bind_visibility_from(
                            w["model_preset"], "value",
                            lambda p: preset_arch(p, w["model_path"].value) == "flux3_klein")
                        hint("FLUX.2 Klein 9B: Base samples with vafipas663's turbo LoRA (1.2 GB) at 4 steps, CFG 1. "
                             "Its author reports static noise above strength 0.5, sometimes needing as low as 0.1: "
                             "lower the strength if previews come out noisy. Without it, Base samples 20 steps at "
                             "CFG 3.5; the distilled "
                             "black-forest-labs/FLUX.2-klein-9B as the sampling model takes 4 steps at CFG 1 and "
                             "needs no turbo LoRA.").bind_visibility_from(
                            w["model_preset"], "value",
                            lambda p: preset_arch(p, w["model_path"].value) == "flux2_klein")
                        with advanced("Seeds, negative prompt and sampling models"):
                            with fields():
                                w["sample_seed"] = ui.number("Sample seed", value=d.sample_seed, precision=0) \
                                    .classes("w-32")
                                w["sample_walk_seed"] = ui.switch("Walk seed (+1 per prompt)",
                                                                  value=d.sample_walk_seed)
                                w["sample_before_training"] = ui.switch("Samples before training starts",
                                                                        value=d.sample_before_training)
                                w["sample_frames"] = ui.number("Frames per sample", value=d.sample_frames, min=1,
                                                               max=481, precision=0).classes("w-40") \
                                    .tooltip("PAN 2 and MiniMax H3 only. 1 renders a still. More renders a clip at "
                                             "24 frames per second, saved as an animated WebP beside its middle "
                                             "frame. PAN 2 rounds down to 4k + 1 (49 is 2 seconds), MiniMax H3 up to "
                                             "17n + 5 (39, 56, ...). A clip takes many times a still's time and "
                                             "memory: use a small size, about 0.35 MP.") \
                                    .bind_visibility_from(w["model_preset"], "value",
                                                          lambda p: preset_arch(p, w["model_path"].value)
                                                          in VIDEO_ARCHS)
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
                                w["sample_lora_mode"] = ui.select(SAMPLE_LORA_MODES, value="auto",
                                                                  label="Sampling LoRA").classes("w-96") \
                                    .tooltip("A second LoRA used only while rendering samples and probes, with the "
                                             "turbo LoRA, never in training. Auto: on MOSS-Image Base, Kijai's SFT "
                                             "LoRA (1.2 GB, SFT minus Base), so previews show the LoRA as it looks on "
                                             "SFT; nothing for the other models or a separate sampling model.")
                                path_input(f, "sample_lora", "Sampling LoRA file", "file", "", "w-[34rem]")
                                f.w["sample_lora"].parent_slot.parent.bind_visibility_from(
                                    w["sample_lora_mode"], "value", lambda x: x == "file")
                                w["sample_lora_strength"] = ui.number("Sampling LoRA strength",
                                                                      value=d.sample_lora_strength, step=0.05) \
                                    .classes("w-48") \
                                    .bind_visibility_from(w["sample_lora_mode"], "value", lambda x: x != "none")

                    def _begin(text):
                        _task_progress.update(text=text, frac=None, active=True)

                    def _end():
                        _task_progress.update(active=False)

                    # ---- Identity similarity
                    with sections.section("Likeness", "Identity similarity",
                                          "Face and body scoring for character LoRAs", "face"):
                        hint("Pick reference images on the Dataset tab (\"Face ref\" / \"Body ref\") or list paths "
                             "here. After each sampling pass, every sample's largest face is compared to the "
                             "reference faces with InsightFace (ArcFace; 50%+ generally means the same person), and "
                             "its body to the body references with the Body score below. Body references should "
                             "include three-quarter or full-length shots.")
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
                                                ui.image(crop).props("fit=contain").classes("w-32 h-32 rz-imgbg")
                                            ui.label(caption).classes("text-xs px-1")

                            ui.button("Preview reference faces", icon="face", on_click=on_face_preview) \
                                .props("outline")
                        face_info = hint("")
                        face_row = ui.row().classes("w-full gap-2")
                        w["body_refs"] = ui.textarea("Body reference images or folders (one path per line; "
                                                     "three-quarter or full-length shots)").classes("w-full")
                        with fields():
                            w["body_score"] = ui.select({"appearance": "appearance (DINOv2)",
                                                         "anchored": "anchored (SigLIP 2 + DINOv2)",
                                                         "proportions": "proportions (2D skeleton)",
                                                         "skeleton3d": "proportions 3D (skeleton + silhouette)",
                                                         "smplx": "SMPL-X body shape (NLF)"},
                                                        value=d.body_score, label="Body score").classes("w-64") \
                                .tooltip("appearance: a DINOv2 embedding of the body from the neck to mid-thigh, "
                                         "turned upright and cut out from the background, best match against the "
                                         "reference photos that show the same part of the body (build, skin, hair; "
                                         "clothing and nudity too, so vary the outfits). anchored: the same body "
                                         "window with its photographic look (colour grade, exposure, sharpness, "
                                         "grain) taken out, embedded by SigLIP 2 and DINOv2, scored as how far it "
                                         "has moved from the base model's people (the step-0 probe's body renders, "
                                         "8 seeds) toward the references: 0 = the base model, 100 = the references' "
                                         "mean. On a five-run comparison it told trained samples from the base model "
                                         "where the best match ranked the person's own photos below the base "
                                         "model's. proportions (2D): skeleton "
                                         "ratios on the flat picture, which shift with pose and camera angle more "
                                         "than people differ. proportions 3D: bone-length ratios from the pose "
                                         "model's 3D landmarks (the same bent or straight), plus waist, hip and thigh "
                                         "widths where the person stands upright facing the camera. SMPL-X: NLF fits "
                                         "the SMPL-X body model and measures it in its T-pose (bust, waist, hip, "
                                         "thigh, limb lengths over the height); needs photos down to the knees and "
                                         "NLF's file, downloaded by hand (noncommercial license). The last three "
                                         "ignore clothing and skin; each compares against the references' own spread.")
                            w["nlf_model"] = ui.input("NLF model (file or folder; empty = models/nlf)",
                                                      value=d.nlf_model).classes("w-72") \
                                .tooltip("SMPL-X needs NLF (noncommercial research license, not downloaded for you): "
                                         "nlf_l_multi_0.3.2.patch6.torchscript, 550 MB, from "
                                         "github.com/isarandi/nlf/releases (v0.3.2), into models/nlf/") \
                                .bind_visibility_from(w["body_score"], "value", lambda x: x == "smplx")
                            w["body_backend"] = ui.select(["mediapipe", "mediapipe+sam3"], value=d.body_backend,
                                                          label="Pose / mask backend").classes("w-48") \
                                .tooltip("mediapipe: pose landmarks (the person box, skeleton ratios, region masks). "
                                         "+sam3: a SAM 3 person mask for region masks and silhouette widths")
                            w["sam3_model"] = ui.input("SAM 3 model (repo id or folder)", value=d.sam3_model) \
                                .classes("w-64") \
                                .bind_visibility_from(w["body_backend"], "value", lambda x: x == "mediapipe+sam3")
                            w["body_device"] = ui.select(["cuda:0", "cuda:1", "cpu"], value=d.body_device,
                                                         label="SAM 3 / NLF device").classes("w-36") \
                                .tooltip("Where SAM 3 or NLF run while scoring (back on the CPU in between)")

                            def body_device_visible(_=None):
                                w["body_device"].set_visibility(w["body_backend"].value == "mediapipe+sam3"
                                                                or w["body_score"].value == "smplx")

                            w["body_backend"].on_value_change(body_device_visible)
                            w["body_score"].on_value_change(body_device_visible)
                            body_device_visible()

                            async def on_body_preview():
                                paths = [p.strip() for p in (w["body_refs"].value or "").splitlines() if p.strip()]
                                if not paths:
                                    body_info.text = "No body reference images listed."
                                    return
                                body_info.text = "Working..."
                                _begin("Preparing body models...")
                                try:
                                    body_info.text = await run.io_bound(body_preview, paths, w["body_backend"].value,
                                                                        w["sam3_model"].value, w["body_device"].value,
                                                                        w["body_score"].value,
                                                                        (w["nlf_model"].value or "").strip())
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
                    with sections.section("Likeness", "Region-weighted loss",
                                          "Face, person and background weights; identity loss", "filter_center_focus"):
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
                                            ui.image(pth).props("fit=contain").classes("w-32 h-32 rz-imgbg")
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
                        with advanced("Trigger attention and identity alignment"):
                            hint("Two training-only losses that read the model's insides; the saved LoRA is the same "
                                 "kind of file. Not yet measured here: race them against the run's own settings.")
                            with fields():
                                w["attn_localize"] = ui.switch("Keep the trigger's attention on the subject",
                                                               value=d.attn_localize).tooltip(
                                    "Attention localization: in a few attention layers, reads how much of each image "
                                    "token's attention to the caption goes to the trigger word, and penalizes the part "
                                    "of that map outside the person (the region maps' person mask and face box). Ties "
                                    "the identity to the trigger and keeps it off the background and other people. "
                                    "Needs a trigger word. FIBO 2.0 reads it in its injection blocks.")
                                w["attn_localize_weight"] = ui.number("Weight", value=d.attn_localize_weight, min=0,
                                                                      step=0.01).classes("w-28")
                                w["attn_localize_layers"] = ui.input("Blocks (empty = auto)",
                                                                     value=d.attn_localize_layers).classes("w-44") \
                                    .tooltip("Comma-separated block numbers to read it in. Auto: a third and half "
                                             "of the depth (FIBO 2.0: its injection blocks).")
                            with fields():
                                w["id_align"] = ui.switch("Align the face tokens with the photo's identity",
                                                          value=d.id_align).tooltip(
                                    "Identity feature alignment (REPA pointed at identity): after one block, the "
                                    "image tokens inside the face box are averaged and a small projector maps them "
                                    "onto that photo's ArcFace embedding. Identity feedback at every noise level, "
                                    "without the identity loss's decode. The projector is not saved.")
                                w["id_align_weight"] = ui.number("Weight", value=d.id_align_weight, min=0,
                                                                 step=0.01).classes("w-28")
                                w["id_align_layer"] = ui.number("Block (-1 = auto)", value=d.id_align_layer, min=-1,
                                                                precision=0).classes("w-36") \
                                    .tooltip("The block whose output is aligned (also for the body). Auto: a third of "
                                             "the depth.")
                            with fields():
                                w["body_align"] = ui.switch("Align the body tokens with the photo's build",
                                                            value=d.body_align).tooltip(
                                    "The same for the build: the person's tokens outside the face box (the region "
                                    "maps' person mask) are averaged and a second small projector maps them onto "
                                    "that photo's DINOv2 body-window embedding, the one the body probe scores with. "
                                    "Photos that do not show the neck to mid-thigh are left out.")
                                w["body_align_weight"] = ui.number("Weight", value=d.body_align_weight, min=0,
                                                                   step=0.01).classes("w-28")
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
                    with sections.section("Likeness", "Per-image loss watch", "Per-image verdicts and learning rate",
                                          "rule"):
                        hint("Tracks every image's loss normalised by its timestep bucket and, at each epoch boundary, "
                             "classes it as easy / healthy / suspect / stuck / exhausted. Per-image LR then throttles "
                             "stuck images (x0.5, then x0.25), eases exhausted ones (x0.6) and can exclude images "
                             "that stay stuck. Stuck means an outlier of the dataset, not just one of its harder "
                             "pictures. A true per-image learning rate needs batch size 1 (otherwise per-sample "
                             "loss weights). Not available with automagic3 or automagic4.")
                        with fields():
                            w["loss_watch"] = ui.switch("Loss watch (observe)", value=d.loss_watch)
                            w["per_image_lr"] = ui.switch("Per-image learning rate (act on verdicts)",
                                                          value=d.per_image_lr)
                            w["loss_watch_exclude_after"] = ui.number("Exclude after N epochs at lowest throttle "
                                                                      "(0 = never)", value=d.loss_watch_exclude_after,
                                                                      precision=0).classes("w-96 max-w-full")
                        with fields():
                            w["region_watch"] = ui.switch("Region watch (observe)", value=d.region_watch).tooltip(
                                "Splits every training step's loss by the picture's face, hair, skin, clothes and "
                                "background, and reports once an epoch how much of each region's starting error is "
                                "gone, and per picture whether its background or clothes are being learned faster "
                                "than its subject. Costs no extra model calls; changes nothing about training. Needs "
                                "MediaPipe for the parts maps (built once, before training). Not in a configuration "
                                "race. On its first real runs it added no time to a step. Its numbers jump from "
                                "epoch to epoch: read the sweeps before acting on them.")
                            w["region_class_weights"] = ui.input(
                                "Region class weights (experimental)", value=d.region_class_weights,
                                placeholder="face=3, hair=1.5, skin=1.5, clothes=0.5, background=0.3") \
                                .classes("w-[32rem] max-w-full").tooltip(
                                "One fixed loss weight per region for the whole run, from the same parts maps: face, "
                                "hair, skin, clothes, background. A region left out stays at 1, 0 masks one out, and "
                                "the weights are rescaled to mean 1 per picture. Empty = off. The plain alternative "
                                "to region steering, and the one to compare it against. Not together with the "
                                "region-weighted loss (face / person / background) or Hull's region gains. Measured "
                                "once on Qwen-Image 2.1 with the example weights: the training faces were learned "
                                "faster and held-out faces ended worse.")
                        with fields():
                            w["region_steer"] = ui.switch("Region steering (experimental)",
                                                          value=d.region_steer).tooltip(
                                "Acts on the region watch (and turns it on): once an epoch it moves loss weights a "
                                "step, by the rules beside it. Each rule is off at 0. What is taken from a region "
                                "goes to the rest of that picture. Works with every optimizer; not in a race. "
                                "Measured once on Qwen-Image 2.1 at its defaults: no difference from the plain run. "
                                "No benefit is claimed.")
                            w["region_steer_regions"] = ui.select(
                                {"background,clothes": "background and clothes", "background": "background only",
                                 "clothes": "clothes only"}, label="May be turned down",
                                value=d.region_steer_regions if d.region_steer_regions in (
                                    "background,clothes", "background", "clothes") else "background,clothes") \
                                .classes("w-56").bind_visibility_from(w["region_steer"], "value")
                        with fields().bind_visibility_from(w["region_steer"], "value"):
                            w["region_steer_cap"] = ui.number("Cap (x subject learned, 0 = off)",
                                                              value=d.region_steer_cap, min=0, max=2, step=0.05) \
                                .classes("w-60").tooltip(
                                "All pictures together: a steered region may be learned at most this share of what "
                                "the subject (face, hair, skin) has been learned. Past it the region is turned down "
                                "in every picture, further where it runs further ahead. Lower = stricter.")
                            w["region_steer_ahead"] = ui.number("Ahead (x median contrast, 0 = off)",
                                                                value=d.region_steer_ahead, min=0, max=0.95,
                                                                step=0.05).classes("w-64").tooltip(
                                "One picture at a time: its region is turned down when its contrast (the Region "
                                "watch table) is under this share of the median picture's, whatever the others do. "
                                "It reads one noisy training sample per picture per epoch.")
                            w["region_steer_boost"] = ui.number("Boost up to (0 = never raise)",
                                                                value=d.region_steer_boost, min=0, max=3, step=0.1) \
                                .classes("w-56").tooltip(
                                "All pictures together: a face, hair or skin that lags the subject as a whole by "
                                "more than a tenth is raised, up to this weight (1.5, say). Off by default: a "
                                "region that stays hard is often a bad picture, and raising it trains on that.")
                            w["region_steer_prior"] = ui.number("Background prior up to (0 = off)",
                                                                value=d.region_steer_prior, min=0, max=2, step=0.05) \
                                .classes("w-60").tooltip(
                                "While the background is past its cap, also pull the background's prediction "
                                "towards the base model's, with up to this weight. Needs the cap rule and the "
                                "background among the steered regions. One more forward pass a step while it acts.")
                        with fields().bind_visibility_from(w["region_steer"], "value"):
                            w["region_steer_min"] = ui.number("Lowest weight (0 = mask out)",
                                                              value=d.region_steer_min, min=0, max=0.95, step=0.05) \
                                .classes("w-52").tooltip(
                                "A steered region's weight never goes under this. 0 lets steering mask a region out "
                                "of a picture altogether: the model is then free to do anything there.")
                            w["region_steer_step"] = ui.number("Step per epoch", value=d.region_steer_step,
                                                               min=0.01, max=0.5, step=0.01).classes("w-40").tooltip(
                                "The most a weight moves in one epoch, as a share of itself. Back up half as fast.")
                            w["region_steer_strength"] = ui.number("Strength", value=d.region_steer_strength,
                                                                   min=0.1, max=4, step=0.1).classes("w-32").tooltip(
                                "How hard a rule pulls: the weight aimed for is (share within the limit) to the "
                                "power of 2 x this.")
                            w["region_steer_signal"] = ui.select(
                                {"steps": "training steps (free, noisy)", "sweep": "sweeps (fixed noise)"},
                                label="Per-picture reading", value=d.region_steer_signal if d.region_steer_signal in (
                                    "steps", "sweep") else "steps").classes("w-64").tooltip(
                                "What the cap rule's per-picture share and the ahead rule read: each picture's one "
                                "training step per epoch, or its fixed-noise sweep (set the sweeps below). A sweep is "
                                "a cleaner reading, as fresh as that picture's last sweep.")
                            w["region_steer_holdout"] = ui.number("Held-out gap over (0 = off)",
                                                                  value=d.region_steer_holdout, min=0, max=0.95,
                                                                  step=0.05).classes("w-56").tooltip(
                                "Needs held-out photos. A region learned this much further on the training pictures "
                                "than on the held-out ones (0.15 = 15 points), and no better on those for two sweeps, "
                                "is turned down in every picture. The one rule that may turn the face, hair or skin "
                                "down.")
                            w["region_steer_subject_ahead"] = ui.number(
                                "Subject ahead (x median left, 0 = off)", value=d.region_steer_subject_ahead, min=0,
                                max=0.95, step=0.05).classes("w-72").tooltip(
                                "Needs sweeps of the training pictures. One picture at a time: its face, hair or "
                                "skin is turned down in that picture when the sweeps show it with less than this "
                                "share of the median picture's error left (0.7, say): that one photo's face is "
                                "being learned far ahead of the others'. The only per-picture rule for the person "
                                "themselves.")
                        with fields():
                            w["region_sweep_every"] = ui.number("Region sweep every N epochs (0 = off)",
                                                                value=d.region_sweep_every, min=0, precision=0) \
                                .classes("w-72").tooltip(
                                "Experimental. Runs the model again on the training "
                                "pictures at three fixed noise levels with fixed noise, so a picture's learned share "
                                "is measured against its own first sweep: a clean reading where the region watch's "
                                "is a noisy one. 3 forward passes per picture per sweep, so sweeping every picture "
                                "every epoch about doubles the training time; the first sweep always takes all of "
                                "them. Not in a race, not with the resolution step-up.")
                            w["region_sweep_pictures"] = ui.number("Pictures per sweep (0 = all, -1 = held-out only)",
                                                                   value=d.region_sweep_pictures, min=-1,
                                                                   precision=0).classes("w-80").tooltip(
                                "How many training pictures a sweep takes: all of them, this many in turn, or none "
                                "(only the held-out photos, the cheapest).")
                            w["holdout_images"] = ui.number("Hold out N photos (0 = none)", value=d.holdout_images,
                                                            min=0, precision=0).classes("w-56").tooltip(
                                "Experimental. Keeps N photos (and every crop made from them) out of training, "
                                "spread evenly through the set by file name, and sweeps them: what the model learns "
                                "on photos it never trained on, it has learned of the subject and not of the photos. "
                                "The gap to the training pictures is memorising. Needs at least 10 photos, at most a "
                                "fifth held out. A held-out photo that nearly matches a training photo is not "
                                "unseen. Sweeps every epoch unless set otherwise.")
                            w["holdout_track_best"] = ui.switch("Keep the held-out best", value=d.holdout_track_best) \
                                .tooltip("Saves <job>_holdout_best.safetensors whenever the held-out photos' subject "
                                         "score (face, hair and skin together) reaches a new best.")
                            w["holdout_patience"] = ui.number("Stop after N sweeps without a better score (0 = never)",
                                                              value=d.holdout_patience, min=0, precision=0) \
                                .classes("w-96 max-w-full").tooltip(
                                "Ends training when the held-out subject score has not improved for this many sweeps "
                                "in a row. With two or three held-out photos that score rests on very few pictures.")
                        with fields():
                            w["look_warmup"] = ui.switch("Warm up look outliers", value=d.look_warmup).tooltip(
                                "Images the look consistency filter (Dataset tab) put under its fence, real but "
                                "unusual shots such as tight angles and profiles, start at a lower loss weight and "
                                "reach full weight after a few epochs, while the identity forms from the typical "
                                "images. Needs a look scan of the dataset; works without the loss watch.")
                            w["look_warmup_start"] = ui.number("Start weight", value=d.look_warmup_start, min=0,
                                                               max=1, step=0.05).classes("w-32") \
                                .bind_visibility_from(w["look_warmup"], "value")
                            w["look_warmup_epochs"] = ui.number("Full weight after (epochs)",
                                                                value=d.look_warmup_epochs, min=0, step=0.5) \
                                .classes("w-52").bind_visibility_from(w["look_warmup"], "value")
                        with advanced("Thresholds"):
                            with fields():
                                w["loss_watch_buckets"] = ui.number("Sigma buckets", value=d.loss_watch_buckets,
                                                                    precision=0).classes("w-32")
                                w["loss_watch_beta"] = ui.number("Residual EMA", value=d.loss_watch_beta, min=0,
                                                                 max=0.99, step=0.05).classes("w-32")
                                w["loss_watch_high"] = ui.number("Hard above median x", value=d.loss_watch_high,
                                                                 step=0.05).classes("w-44")
                                w["loss_watch_fence"] = ui.number("Stuck above median + x IQR",
                                                                  value=d.loss_watch_fence, min=0,
                                                                  step=0.25).classes("w-52").tooltip(
                                    "A hard picture that has stopped improving is only called stuck when it is "
                                    "also an outlier: above the median by this many times the spread of the "
                                    "middle half of the pictures. Late in a run few pictures still improve, so "
                                    "without it every hard picture ends up throttled. 0 = off.")
                                w["loss_watch_low"] = ui.number("Easy below median x", value=d.loss_watch_low,
                                                                step=0.05).classes("w-44")
                                w["loss_watch_improve_tol"] = ui.number("Improving = drop >",
                                                                        value=d.loss_watch_improve_tol,
                                                                        step=0.01).classes("w-40")
                                w["loss_watch_healthy_mult"] = ui.number("Healthy multiplier",
                                                                         value=d.loss_watch_healthy_mult,
                                                                         step=0.05).classes("w-40")

                    # ---- Training maps
                    with sections.section("Likeness", "Training maps", "Heatmaps of what training changes, picture "
                                          "by picture", "layers"):
                        hint("After Fizgig's training maps. A few training pictures (at least one per dataset) are "
                             "checked before training and about once an epoch at three fixed noise levels with fixed "
                             "noise, so only the LoRA changes between captures: what it has learned since the start, "
                             "what it still misses, where it acts, and where the trigger word's attention lands. "
                             "Observe only, a few seconds per capture; see them on the Training maps tab. Not in a "
                             "configuration race.")
                        with fields():
                            w["training_maps"] = ui.switch("Training maps", value=d.training_maps)
                            w["training_maps_images"] = ui.number("Pictures", value=d.training_maps_images, min=1,
                                                                  max=12, precision=0).classes("w-28") \
                                .bind_visibility_from(w["training_maps"], "value")
                            w["training_maps_every"] = ui.number("Every N steps (0 = each epoch)",
                                                                 value=d.training_maps_every, min=0,
                                                                 precision=0).classes("w-56") \
                                .bind_visibility_from(w["training_maps"], "value")
                            w["training_maps_trigger_off"] = ui.switch("Trigger-off pass",
                                                                       value=d.training_maps_trigger_off) \
                                .bind_visibility_from(w["training_maps"], "value").tooltip(
                                "Also maps each picture with the trigger word taken out of its caption: where the "
                                "LoRA changes the prediction although it was not asked for. The log gives one "
                                "number per capture, the bleed: how much the LoRA acts without the trigger word "
                                "against with it (near 0 = tied to its trigger, near 1 = acts the same either way). "
                                "Two more forward passes per picture and noise level. On its first real run, with long "
                                "captions that describe the person, it read 0.98-0.99 all run and told nothing.")

                    # ---- LoRA Isolation
                    with sections.section("Likeness", "LoRA Isolation",
                                          "Keep your subject(s) off everyone else in a picture", "group_off"):
                        hint("After Fizgig's LoRA Isolation. A character LoRA spreads its subject onto other people: "
                             "strangers in the background, a friend beside them. With it on: every caption that "
                             "names a subject also trains without the trigger, held to the model without the LoRA "
                             "(DOP, Ostris's AI-Toolkit); the trigger is joined to its class word (\"gwelt, a man\" "
                             "trains as \"gweltman\": prompt with that word); each person gets side-by-side "
                             "pictures of one of their photos next to a generated stranger; and several subjects "
                             "(one per dataset, each with its trigger on the Dataset rows above) balance. Fizgig "
                             "measured on Krea 2: strangers reading as the subject 48 of 50 -> 0 of 50, likeness "
                             "unchanged. A second person in the LoRA helps. About 1.5x the training time.")
                        with fields():
                            w["isolation"] = ui.switch("LoRA Isolation", value=d.isolation)
                        with fields() as iso_fields:
                            w["isolation_class"] = ui.input("Subject is a (one dataset)", value=d.isolation_class,
                                                            placeholder="from captions").classes("w-48") \
                                .tooltip("With a single dataset folder: what the subject is (man, woman, dog). "
                                         "With datasets: each row's 'Is a'.")
                            w["isolation_kind"] = ui.select({"auto": "auto", "person": "person",
                                                             "thing": "animal / thing", "style": "style"},
                                                            value=d.isolation_kind, label="Kind (one dataset)") \
                                .classes("w-44")
                            w["dop_weight"] = ui.number("DOP weight", value=d.dop_weight, min=0, step=0.5) \
                                .classes("w-32")
                            w["dop_every"] = ui.number("DOP every N steps", value=d.dop_every, min=1, precision=0) \
                                .classes("w-40")
                            w["isolation_pairs"] = ui.number("Side-by-sides per person", value=d.isolation_pairs,
                                                             min=0, precision=0).classes("w-48")
                            w["isolation_pair_weight"] = ui.number("Their weight", value=d.isolation_pair_weight,
                                                                   min=0, step=0.5).classes("w-32")
                            w["isolation_balance"] = ui.switch("Balance the subjects", value=d.isolation_balance) \
                                .tooltip("Smaller subject datasets repeat so each trains about as often as the "
                                         "largest.")
                        iso_fields.bind_visibility_from(w["isolation"], "value")

                    # ---- Likeness probes
                    with sections.section("Likeness", "Likeness probes & controller",
                                          "Best checkpoint, early stop and curriculum", "insights"):
                        hint("Renders a fixed-seed headshot and/or full-body probe every N steps (small, turbo LoRA), "
                             "scores it against the references, and lets a controller act on the smoothed scores: "
                             "keep the best checkpoint and stop early, nudge the learning rate, and weight the dataset "
                             "/ timesteps toward whichever of face or body is lagging. Works with any optimizer. Needs "
                             "face/body references (Identity similarity).")
                        with fields():
                            w["probe_every"] = ui.number("Probe every N steps (0 = off)", value=d.probe_every,
                                                         precision=0).classes("w-60")                                 .tooltip("Also probed once at the last step, so the final LoRA has its own score")
                            w["probe_at_start"] = ui.switch("Probe at step 0 (baseline)", value=d.probe_at_start)
                            w["probe_async"] = ui.switch("Render probes on the second GPU", value=d.probe_async) \
                                .tooltip("A copy of the model on the other GPU renders each probe while training goes "
                                         "on, instead of training pausing for it. The decisions (best checkpoint, "
                                         "early stop, lr control) land when the probe is scored, up to Probe every "
                                         "steps later; a best checkpoint still saves exactly the weights that were "
                                         "scored. Needs Multi-GPU off (data parallel already trains on both) and the "
                                         "model to fit the second GPU; not in a configuration race.")
                        w["probe_face_prompt"] = ui.input("Face probe prompt (headshot; empty = none)") \
                            .classes("w-full")
                        w["probe_body_prompt"] = ui.input("Body probe prompt (full body, upright; empty = none)") \
                            .classes("w-full")
                        # preset prompts for a character LoRA, and what each probe's renders should show (the
                        # prompt check, promptcheck.py)
                        expect_options = {"": "guess from the prompt", "none": "do not check",
                                          **promptcheck.EXPECT_LABELS}
                        with fields():
                            w["probe_subject"] = ui.input("Subject (for preset prompts)", value=d.probe_subject,
                                                          placeholder="a young woman with long dark brown hair") \
                                .classes("w-[26rem] max-w-full").tooltip(
                                "What the preset prompts call the person, next to the trigger word: a few words "
                                "such as 'a young woman with long dark brown hair'. Empty = a dataset's class (LoRA "
                                "Isolation), else 'a person'. Only the presets read it.")
                            face_preset = ui.select({p.key: p.label for p in promptcheck.presets_of("face")},
                                                    label="Face probe preset", clearable=True).classes("w-72") \
                                .tooltip("Writes a ready-made headshot prompt, with your trigger word and subject, "
                                         "into the face probe prompt above and sets what it should show. Pick one, "
                                         "then edit the prompt freely.")
                            body_preset = ui.select({p.key: p.label for p in promptcheck.presets_of("body")},
                                                    label="Body probe preset", clearable=True).classes("w-72") \
                                .tooltip("The same for the body probe prompt: full-body and waist-up prompts.")
                        with fields():
                            w["probe_face_expect"] = ui.select(expect_options, value=d.probe_face_expect,
                                                               label="Face probe should show").classes("w-72") \
                                .tooltip("The prompt check: every probe render is read for its framing (from the "
                                         "face's size and whether the hips are in the picture) and compared with "
                                         "the step-0 probe of the same seed. The log and the Monitor then say how "
                                         "many renders still keep their prompt's frame, and how many drifted (a "
                                         "full-body prompt coming back as a portrait). It only reports, unless the "
                                         "best guard below or the race's drift drop act on it. It sees framing and "
                                         "the number of faces, not clothing or setting.")
                            w["probe_body_expect"] = ui.select(expect_options, value=d.probe_body_expect,
                                                               label="Body probe should show").classes("w-72") \
                                .tooltip("Full body means the body at least down to the hips with a small face; "
                                         "the models draw 'full body' from head to thigh as often as head to toe. "
                                         "Guess from the prompt reads words such as full-body, waist-up, headshot "
                                         "and close-up.")
                            w["probe_checks"] = ui.select({p.key: p.label for p in promptcheck.PRESETS},
                                                          value=[], multiple=True,
                                                          label="Extra prompts to check (rendered, never scored)") \
                                .classes("w-[28rem] max-w-full").props("use-chips").tooltip(
                                "Preset prompts rendered at every probe and read by the prompt check, without "
                                "entering the likeness score (its scale stays as it is). Use them to watch poses "
                                "and framings the two likeness prompts do not cover. Each costs its seeds in "
                                "renders per probe. Prompts of your own go into the config file (probe_checks).")
                            w["probe_check_seeds"] = ui.number("Seeds per extra prompt", value=d.probe_check_seeds,
                                                               min=1, max=16, precision=0).classes("w-48")

                        def use_preset(e, key: str) -> None:
                            if not e.value:
                                return
                            w[f"probe_{key}_prompt"].value = promptcheck.build(
                                e.value, w["trigger_word"].value or "", w["probe_subject"].value or "")
                            w[f"probe_{key}_expect"].value = promptcheck.preset(e.value).expect

                        face_preset.on_value_change(lambda e: use_preset(e, "face"))
                        body_preset.on_value_change(lambda e: use_preset(e, "body"))
                        with fields():
                            w["probe_steps"] = ui.number("Probe steps", value=d.probe_steps, precision=0) \
                                .classes("w-32")
                            w["probe_cfg"] = ui.number("Probe CFG", value=d.probe_cfg, step=0.5).classes("w-32")
                            w["probe_size"] = ui.number("Probe size (px)", value=d.probe_size, step=64,
                                                        precision=0).classes("w-36") \
                                .tooltip("The face probe is size x size; the body probe is a 2:3 portrait with the "
                                         "same pixel count (1024: 832 x 1248), so it fits wherever samples of that "
                                         "size do")
                        subhead("Controller actions")
                        with fields():
                            w["likeness_normalize"] = ui.switch("Normalize face and body scores", value=d.likeness_normalize) \
                                .tooltip("Score each on one scale first: 0 = what a stranger scores (the step-0 probe), "
                                         "100 = what your references score against each other (both scales are "
                                         "stretched, see their span and curve). Raw, a body score "
                                         "moves ~10 points over a run and a face score ~60, so the combined score, "
                                         "the curriculum and the noise band would follow the face alone.")
                            w["face_scale_span"] = ui.number("Face scale span", value=d.face_scale_span, min=1,
                                                             step=0.05, format="%.2f").classes("w-36") \
                                .bind_visibility_from(w["likeness_normalize"], "value") \
                                .tooltip("An ArcFace cosine saturates: a decent likeness already matches your "
                                         "references, so the face read ~90 while the body read ~50 for pictures that "
                                         "looked equally close. The face's 100 sits this many times as far above a "
                                         "stranger as your references do (1 = the references themselves, the old "
                                         "scale).")
                            w["face_scale_curve"] = ui.number("Face scale curve", value=d.face_scale_curve, min=1,
                                                              step=0.1, format="%.2f").classes("w-36") \
                                .bind_visibility_from(w["likeness_normalize"], "value") \
                                .tooltip("The face score is its position on that scale raised to this power: above "
                                         "1 the face starts slower and leaves room to keep improving. 1 = linear "
                                         "(the old scale). With the defaults (1.35 / 1.5) your references' own face "
                                         "level reads about 64.")
                            w["body_scale_span"] = ui.number("Body scale span", value=d.body_scale_span, min=1,
                                                             step=0.05, format="%.2f").classes("w-36") \
                                .bind_visibility_from(w["likeness_normalize"], "value") \
                                .tooltip("The body's DINOv2 score saturates too: a probe that copies the photos' "
                                         "outfit and setting matches them better than your references, in their varied "
                                         "outfits, match each other, so the body read 100 while the face read 50-60. "
                                         "The body's 100 sits this many times as far above a stranger as your "
                                         "references do (1 = the references themselves, the scale of v2.1.3 and "
                                         "before).")
                            w["body_scale_curve"] = ui.number("Body scale curve", value=d.body_scale_curve, min=1,
                                                              step=0.1, format="%.2f").classes("w-36") \
                                .bind_visibility_from(w["likeness_normalize"], "value") \
                                .tooltip("The body score is its position on that scale raised to this power, as for "
                                         "the face. With the defaults (1.35 / 1.5) your references' own body level "
                                         "reads about 64, the same as their face.")
                            w["likeness_track_best"] = ui.switch("Keep best checkpoint (<job>_best.safetensors)",
                                                                 value=d.likeness_track_best) \
                                .tooltip("Saved whenever the smoothed combined score beats its best so far by more "
                                         "than the noise band.")
                            w["likeness_track_best_face"] = ui.switch("Keep best face checkpoint "
                                                                      "(<job>_best_face.safetensors)",
                                                                      value=d.likeness_track_best_face) \
                                .tooltip("A separate file at the best smoothed face score, whatever the body scores "
                                         "then (same rule as the overall best). Independent of the overall best checkpoint.")
                            w["likeness_track_best_body"] = ui.switch("Keep best body checkpoint "
                                                                      "(<job>_best_body.safetensors)",
                                                                      value=d.likeness_track_best_body) \
                                .tooltip("A separate file at the best smoothed body score, whatever the face scores "
                                         "then (same rule as the overall best). Independent of the overall best checkpoint.")
                            w["likeness_best_any_rise"] = ui.switch("New best on any rise",
                                                                    value=d.likeness_best_any_rise) \
                                .tooltip("Off (default): a best checkpoint (overall, face, body) is replaced only when "
                                         "the smoothed score beats it by more than the noise band, so the file "
                                         "doesn't churn on noise; the log can then show a probe slightly above "
                                         "'best' (e.g. 35.1 vs best 34.1 with a 1.0 band) without a new save. On: any "
                                         "higher smoothed score replaces it, so the file is always the highest score "
                                         "seen, even by a hair. Either way early stop and the resolution step-up "
                                         "only count a rise beyond the noise band as progress.")
                            w["likeness_framing"] = ui.switch("Lost body renders count as zero",
                                                              value=d.likeness_framing) \
                                .tooltip("On (default): a full-body probe render that no longer shows a body counts as "
                                         "zero in the body score, against how many showed one at the first probe, and "
                                         "one render's score stops at the top of the scale. Off: the body score is the "
                                         "average of the renders that still show a body, so the last few of them, or a "
                                         "score left over from when some did, can set a best checkpoint (in a test of "
                                         "nine runs that happened in eight). The 'Body probe framing' log line gives "
                                         "the count either way.")
                            w["likeness_best_min_kept"] = ui.number(
                                "No new best under this share on-prompt (0 = off)", value=d.likeness_best_min_kept,
                                min=0, max=1, step=0.05).classes("w-80").tooltip(
                                "Experimental, unit-tested only. Reads the prompt check: a probe whose renders keep "
                                "their prompt's frame in less than this share (in the prompt that drifted most, "
                                "against the step-0 probe) cannot set a new best checkpoint, overall, face or body. "
                                "It stops a best from resting on a body score taken from the two renders of eight "
                                "that still show a body. 0.5 = at least half the renders must still be on-prompt. "
                                "Needs a probe whose frame is known (Face / Body probe should show).")
                            w["likeness_patience"] = ui.number("Early stop after N probes w/o improvement (0 = never)",
                                                               value=d.likeness_patience, precision=0) \
                                .classes("w-96 max-w-full") \
                                .tooltip("Not used while Resolution step-up is on (its two counts replace it).")
                        with fields():
                            w["face_gold"] = ui.select({"auto": "auto (AdaFace, when training against ArcFace)",
                                                        "off": "off", "adaface": "AdaFace, always",
                                                        "facenet": "FaceNet, always"}, value=d.face_gold,
                                                       label="Gold face scorer").classes("w-72") \
                                .tooltip("A second face recognizer (AdaFace ViT, about 460 MB, downloaded once; or "
                                         "FaceNet, the one before 3.14.0, about 110 MB) scores every probe face too. Nothing trains on it, so it can judge a run that "
                                         "trains against the probes' own ArcFace: the reward stage, the identity loss "
                                         "on its arcface backend, identity alignment. Those can raise the ArcFace score "
                                         "without the face looking more like the person. auto turns it on for them "
                                         "(and for the probe noise check); the Monitor draws it as a dashed line.")
                            w["face_gold_select"] = ui.switch("Best checkpoint, early stop and race follow it",
                                                              value=d.face_gold_select) \
                                .bind_visibility_from(w["face_gold"], "value", backward=lambda m: m != "off") \
                                .tooltip("On (default): with the gold scorer on, its face score replaces ArcFace's in "
                                         "the combined score that keeps the best checkpoint, stops early and ranks a "
                                         "race. The lr control and the face/body curriculum keep ArcFace. Off: it is "
                                         "only logged and drawn next to ArcFace.")
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
                            w["likeness_curriculum_framing"] = ui.number(
                                "Count lost framing (0 = off)", value=d.likeness_curriculum_framing, min=0, max=1,
                                step=0.1).classes("w-56").bind_visibility_from(w["likeness_curriculum"], "value") \
                                .tooltip("Experimental, untested. A body probe render that shows no measurable body "
                                         "(a head-and-shoulders picture where a body was asked for) counts against "
                                         "the body for the curriculum, so a body probe turning into portraits pulls "
                                         "training towards the full-body photos. 1 = a probe with no body in any "
                                         "render reads as a body score of 0. The log reports the count either way.")
                        with advanced("Likeness-filtered self-training"):
                            hint("Partway through the steps the LoRA renders pictures of new prompts (Previews' steps "
                                 "and guidance, the largest training resolution). Those with one confident, uncut "
                                 "face, as sharp as your photos and not repeating each other, that score above both "
                                 "the floor and the probes' smoothed score, train beside the photos at a lower weight; "
                                 "each round replaces the last. Reward-ranked fine-tuning (RAFT) done inside the run: "
                                 "cheap, but it trains on the model's own output, so keep the share small and judge "
                                 "the samples by eye. The pictures and their scores are in <job>/selftrain.")
                            with fields():
                                w["selftrain"] = ui.switch("Self-train on its best pictures", value=d.selftrain)
                                w["selftrain_at"] = ui.input("At (fractions of the steps)", value=d.selftrain_at) \
                                    .classes("w-48").tooltip("e.g. 0.5, or 0.5, 0.75 for two rounds")
                                w["selftrain_images"] = ui.number("Pictures per round", value=d.selftrain_images,
                                                                  min=1, max=256, precision=0).classes("w-40")
                            with fields():
                                w["selftrain_min_likeness"] = ui.number("Likeness floor", value=d.selftrain_min_likeness,
                                                                        step=5).classes("w-36") \
                                    .tooltip("On the probes' scale (0 = a stranger, 100 = the references). A picture "
                                             "must also beat the probes' current smoothed score.")
                                w["selftrain_share"] = ui.number("Most, as a share of the photos",
                                                                 value=d.selftrain_share, min=0.01, max=1,
                                                                 step=0.05).classes("w-56")
                                w["selftrain_weight"] = ui.number("Loss weight", value=d.selftrain_weight, min=0.05,
                                                                  max=1, step=0.05).classes("w-32")
                            with fields():
                                w["copy_check"] = ui.select({"auto": "auto (with self-training)",
                                                             "on": "on (the probes too)", "off": "off"},
                                                            value=d.copy_check, label="Copy check (SSCD)") \
                                    .classes("w-56") \
                                    .tooltip("A copy detector (SSCD, about 99 MB, downloaded once) compares every "
                                             "self-training picture with your training photos and leaves out one that "
                                             "copies a photo (same pose, outfit and background): training on it "
                                             "teaches the photo again, not the person in a new scene. The probes log "
                                             "their closest photo too (Monitor: copy). auto: on with self-training; "
                                             "on: for any run with probes. Turn it on for races: in one the highest "
                                             "score was also the only run over the copy limit, memorising one photo.")
                                w["copy_max"] = ui.number("Copy limit", value=d.copy_max, min=0.05, max=1,
                                                          step=0.05).classes("w-32") \
                                    .bind_visibility_from(w["copy_check"], "value", backward=lambda m: m != "off") \
                                    .tooltip("SSCD cosine above which a picture copies a photo (0.5: a likely partial "
                                             "copy in diffusion memorization studies). Raised to the photos' own "
                                             "typical match among themselves + 0.1 when they already look alike "
                                             "(one shoot, one room).")
                            w["selftrain_prompts"] = ui.textarea("Self-training prompts (one per line; empty = the "
                                                                 "reward stage's varied close-up, upper-body and "
                                                                 "full-body prompts)").classes("w-full")
                        with advanced("Second stage: likeness reward fine-tuning"):
                            hint("After the steps, the LoRA renders groups of pictures of new prompts, the probes' "
                                 "scorers rate their likeness, and a policy-gradient update (Flow-GRPO) makes the "
                                 "better ones more likely and the worse ones less, held near the first stage's LoRA "
                                 "by a KL penalty. It optimizes what you are after directly, but it is slow and can "
                                 "learn what the scorer likes more than the person (keep the probes on and judge by "
                                 "eye). The probes run every Probe every reward steps and once more at the end, so the "
                                 "final LoRA's likeness is in the chart. A picture's face and body scores are capped "
                                 "far below a stranger's, so a body that differs from your body references only in "
                                 "clothing or framing can't outweigh the face; reward prompts like your body "
                                 "references (or Body weight 0) let the body count. Files: <job>_reward_<step> and "
                                 "<job>_reward.safetensors; <job>.safetensors stays the first stage's. Steps 0 with a LoRA in Resume from runs this stage alone.")
                            with fields():
                                w["reward_steps"] = ui.number("Reward steps (0 = off)", value=d.reward_steps, min=0,
                                                              precision=0).classes("w-44")
                                w["reward_lr"] = ui.number("Learning rate", value=d.reward_lr, min=0, format="%.1e",
                                                           step=1e-5).classes("w-36")
                                w["reward_group"] = ui.number("Images per step", value=d.reward_group, min=2,
                                                              max=16, precision=0).classes("w-36") \
                                    .tooltip("Pictures of one prompt per step; only the differences between them "
                                             "count, so 4 or more.")
                                w["reward_kl"] = ui.number("KL penalty", value=d.reward_kl, min=0, step=0.01) \
                                    .classes("w-32").tooltip("Keeps each step's moves near the first stage's LoRA: "
                                                             "higher is safer and slower.")
                            with fields():
                                w["reward_train_steps"] = ui.number("Trained denoising steps", value=d.reward_train_steps,
                                                                    min=1, precision=0).classes("w-52") \
                                    .tooltip("Of each picture's denoising steps, how many the update trains on.")
                                w["reward_sample_steps"] = ui.number("Denoising steps (0 = probe steps)",
                                                                     value=d.reward_sample_steps, min=0,
                                                                     precision=0).classes("w-60")
                                w["reward_noise"] = ui.number("Sampler noise", value=d.reward_noise, min=0.05, max=1.5,
                                                              step=0.05).classes("w-32") \
                                    .tooltip("The stochastic sampler's noise level; the pictures vary more above it.")
                                w["reward_size"] = ui.number("Size (0 = probe size)", value=d.reward_size, min=0,
                                                             step=64, precision=0).classes("w-44")
                            w["reward_prompts"] = ui.textarea("Reward prompts (one per line; empty = varied "
                                                              "close-up, upper-body and full-body prompts from the "
                                                              "trigger word)").classes("w-full")
                        with advanced("Probe scoring"):
                            with fields():
                                w["probe_frames"] = ui.number("Frames per probe", value=d.probe_frames, min=1,
                                                              max=481, precision=0).classes("w-40") \
                                    .tooltip("PAN 2 and MiniMax H3 only. 1 scores a still. More renders each probe "
                                             "as a clip (24 frames per second; PAN 2 4k + 1 frames, MiniMax H3 "
                                             "17n + 5) and scores its middle frame, so the score tells how the "
                                             "likeness holds in motion. Much slower.") \
                                    .bind_visibility_from(w["model_preset"], "value",
                                                          lambda p: preset_arch(p, w["model_path"].value)
                                                          in VIDEO_ARCHS)
                                w["probe_seeds"] = ui.number("Seeds per probe", value=d.probe_seeds, min=1, max=8,
                                                             precision=0).classes("w-36") \
                                    .tooltip("Use 8 for a race. Even then, gaps of about 3 body points did not hold up "
                                             "across seeds on a varied probe, and single runs differed by up to about "
                                             "15 points, so repeat a close race with a second training seed.")
                                w["probe_floor_seeds"] = ui.number("Step-0 seeds", value=d.probe_floor_seeds, min=1,
                                                                   max=16, precision=0).classes("w-32") \
                                    .tooltip("The step-0 probe (the base model: the 0 of the normalized scale) renders "
                                             "at least this many seeds, for a steadier zero; later probes render "
                                             "Seeds per probe. Costs a few extra renders once, at the start.")
                                w["probe_seed"] = ui.number("Probe seed", value=d.probe_seed, precision=0) \
                                    .classes("w-32")
                                w["probe_device"] = ui.input("Probe GPU", value=d.probe_device,
                                                             placeholder="auto").classes("w-32") \
                                    .bind_visibility_from(w["probe_async"], "value") \
                                    .tooltip("The GPU that renders the probes (e.g. cuda:1); empty: the first of "
                                             "Devices that is not the training GPU")
                                w["probe_ema"] = ui.number("Score smoothing (EMA)", value=d.probe_ema, min=0,
                                                           max=0.95, step=0.05).classes("w-48")
                                w["likeness_noise_band"] = ui.number("Noise band (% points)",
                                                                     value=d.likeness_noise_band, step=0.5) \
                                    .classes("w-48")

                    # ---- Configuration race
                    with sections.section("Tools", "Configuration race",
                                          "Several configurations, the probes pick the winner", "emoji_events"):
                        hint("Each configuration below trains until its next probe, then the next one does. Once all of "
                             "them have been probed (a round), the one with the lowest smoothed likeness score is "
                             "dropped; the last one left trains until Steps, which counts its own steps, so the race "
                             "adds the others' steps on top. They all start from the same LoRA initialisation and see "
                             "the same images and noise. A configuration changes only what its row lists; everything "
                             "else on this page is shared. Needs the likeness probes above. Relay instead: nobody is "
                             "dropped, and after each round every configuration continues from that round's winner "
                             "(Add seed racers for the best of several seeds at every probe).")
                        with fields():
                            w["race"] = ui.switch("Race these configurations", value=d.race)
                            w["race_follow"] = ui.switch("Relay: everyone continues from the round's winner",
                                                         value=d.race_follow) \
                                .tooltip("Nobody is dropped. After each round (every configuration probed once) the "
                                         "others take over the winner's LoRA: its weights, optimizer state, weight "
                                         "average and likeness history, keeping their own settings, seed, image order "
                                         "and noise. With seed racers that is the best of several random paths at "
                                         "every probe, instead of whole runs on different seeds. Every configuration "
                                         "trains all the steps (one run's cost each); the last round's winner is the "
                                         "run's LoRA, the others are saved as _final files, and the best checkpoint "
                                         "any of them reached is <job>_best. The rows must share the LoRA's shape (no "
                                         "rank, alpha, layer or network type rows); not with one configuration per GPU.")
                            w["race_pick"] = ui.switch("Pick the leader yourself after each round",
                                                       value=d.race_pick) \
                                .tooltip("Relay only. After each round the run waits while you look at the round's "
                                         "probe images; the Monitor's race table then has a Continue from button on "
                                         "each racer, and everyone continues from the one you pick instead of the "
                                         "highest likeness score. In the last round your pick is the run's LoRA. The "
                                         "GPUs sit idle while it waits.")
                            w["race_pick_minutes"] = ui.number("Wait for the pick (minutes, 0 = no limit)",
                                                               value=d.race_pick_minutes, min=0,
                                                               precision=0).classes("w-80") \
                                .tooltip("No pick by then: the highest likeness score leads, as without picking. A "
                                         "stop while it waits does the same and then stops.")
                            w["race_drop_lowest"] = ui.switch("Drop the lowest after each round",
                                                              value=d.race_drop_lowest) \
                                .tooltip("Off: nobody is dropped for being behind; every configuration trains all the "
                                         "steps (unless the patience drop below ends it) and the winner is the highest "
                                         "score any of them reached. Costs up to one full run per configuration.")
                            w["race_warmup_probes"] = ui.number("Warm-up probe rounds (nobody is dropped)",
                                                                value=d.race_warmup_probes, min=0, precision=0) \
                                .classes("w-72")
                        with fields():
                            w["race_paired"] = ui.switch("Drop only when the probe seeds agree",
                                                         value=d.race_drop_confidence > 0) \
                                .tooltip("The racers are probed on the same seeds, so each seed compares them like for "
                                         "like. On: the lowest leaves only when a paired test over its seeds (and "
                                         "those of the rounds since the last drop) puts it below the round's leader "
                                         "with this confidence; a round too close to call drops nobody. Use 2+ probe "
                                         "seeds. Off: the lowest smoothed score leaves, however small the gap.")
                            w["race_drop_confidence"] = ui.number("Confidence", value=d.race_drop_confidence or 0.9,
                                                                  min=0.5, max=0.99, step=0.05, precision=2) \
                                .classes("w-32")
                            w["race_max_close_rounds"] = ui.number("Close calls before it leaves anyway",
                                                                   value=d.race_max_close_rounds, min=0,
                                                                   precision=0).classes("w-64") \
                                .tooltip("Bounds the extra cost: the race keeps everyone at most this many rounds in a "
                                         "row, and the lowest leaves anyway once it has trained all the steps.")
                        with fields():
                            w["race_drop_drift"] = ui.number("Drop one that leaves the prompt: drift share (0 = off)",
                                                             value=d.race_drop_drift, min=0, max=1, step=0.05) \
                                .classes("w-96").tooltip(
                                "Experimental, unit-tested only. Reads the prompt check: a configuration whose probe "
                                "renders have left their prompt's frame (a full-body prompt coming back as a "
                                "portrait) in at least this share, in the prompt that drifted most and against the "
                                "step-0 probe, for the number of its own probes beside this, leaves the race. The "
                                "last one never leaves; when all of them are over it the least drifted stays. A "
                                "dropped one cannot win on an earlier best score. For racing seeds, turn Drop the "
                                "lowest off (score gaps between seeds are mostly noise) and set this to 0.5. It "
                                "sees framing only. Not in a relay race.")
                            w["race_drop_drift_probes"] = ui.number("Probes in a row", value=d.race_drop_drift_probes,
                                                                    min=1, precision=0).classes("w-40")
                        with fields():
                            w["race_patience_drop"] = ui.switch("Drop each one when it stops improving",
                                                                value=d.race_patience_drop) \
                                .tooltip("A configuration leaves once its own probes have not improved (by more than "
                                         "the noise band) for this many in a row; the others race on. A setting that "
                                         "learns slowly but ends higher is not cut for being behind. The winner is the "
                                         "configuration whose best smoothed score was the highest at any point: its "
                                         "best checkpoint becomes <job>_best.safetensors.")
                            w["race_patience"] = ui.number("Probes without improvement", value=d.race_patience,
                                                           min=1, precision=0).classes("w-56")
                            w["race_patience_rising"] = ui.switch("Keep one whose likeness is rising again",
                                                                  value=d.race_patience_rising) \
                                .tooltip("A configuration out of patience is not dropped while its smoothed likeness "
                                         "is still going up from its previous probe: a chance to come back after a "
                                         "dip. It leaves at the first probe that does not rise (unless it beats its "
                                         "best, which resets the count).")
                            w["race_parallel"] = ui.switch("One configuration per GPU at the same time",
                                                          value=d.race_parallel) \
                                .bind_visibility_from(w["multi_gpu"], "value", lambda x: x == "data_parallel") \
                                .tooltip("With data parallel on 2+ GPUs: instead of every configuration training over "
                                         "all the GPUs in turn, each GPU trains its own configuration at the same "
                                         "time. An odd one left over trains over the GPUs together when its step has "
                                         "batches enough (Gradient accumulation 2+) and that is faster, else on the "
                                         "first GPU. Every configuration then sees Batch size x Gradient accumulation "
                                         "images per step, the same for all. Not with the identity loss; no "
                                         "torch.compile.")
                        race_summary = hint("")

                        def race_rules_visible(_=None):
                            relay = bool(w["race_follow"].value)
                            w["race_pick"].set_visibility(relay)
                            w["race_pick_minutes"].set_visibility(relay and bool(w["race_pick"].value))
                            w["race_drop_lowest"].set_visibility(not relay)
                            drift_on = not relay and float(w["race_drop_drift"].value or 0) > 0
                            w["race_drop_drift"].set_visibility(not relay)
                            w["race_drop_drift_probes"].set_visibility(drift_on)
                            w["race_warmup_probes"].set_visibility(not relay and (bool(w["race_drop_lowest"].value)
                                                                                  or drift_on))
                            paired_on = not relay and bool(w["race_drop_lowest"].value)
                            w["race_paired"].set_visibility(paired_on)
                            w["race_drop_confidence"].set_visibility(paired_on and bool(w["race_paired"].value))
                            w["race_max_close_rounds"].set_visibility(paired_on and bool(w["race_paired"].value))
                            w["race_patience_drop"].set_visibility(not relay)
                            w["race_patience"].set_visibility(not relay and bool(w["race_patience_drop"].value))
                            w["race_patience_rising"].set_visibility(not relay
                                                                     and bool(w["race_patience_drop"].value))

                        for key in ("race_follow", "race_drop_lowest", "race_patience_drop", "race_paired",
                                    "race_drop_drift", "race_pick"):
                            w[key].on_value_change(race_rules_visible)
                        race_rules_visible()

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
                            lowest = bool(w["race_drop_lowest"].value) and not w["race_follow"].value
                            est = race.estimate_steps(len(rows), warmup, every, steps, lowest)
                            if w["race_follow"].value:
                                race_summary.text = (
                                    f"Relay of {len(rows)} configurations: {est['rounds']} rounds, after each one "
                                    f"everyone continues from "
                                    f"{'the racer you pick' if w['race_pick'].value else 'its winner'}; "
                                    f"{est['total']} training steps in all ({est['ratio']:.0f}x the run)."
                                    + (f" Steps must be a multiple of {every}." if steps % every else ""))
                                return
                            patience = (f" A configuration leaves after {int(w['race_patience'].value or 0)} probes "
                                        "without improvement" + (", unless its likeness is rising again"
                                                                 if w["race_patience_rising"].value else "")
                                        + "; the winner is the best score anyone reached."
                                        if w["race_patience_drop"].value else "")
                            if not lowest:
                                race_summary.text = (
                                    f"{len(rows)} configurations, nobody dropped for being behind: each trains up to "
                                    f"{steps} steps, at most {est['total']} training steps in all "
                                    f"({est['ratio']:.0f}x the run)." + patience
                                    + (f" Steps must be a multiple of {every}." if steps % every else ""))
                                return
                            race_summary.text = (
                                f"{len(rows)} configurations: {est['rounds']} rounds and {est['probes']} probes, about "
                                f"{est['total']} training steps in all ({est['ratio']:.1f}x the {steps}-step run)"
                                + (f". Steps must be at least {est['race_own_steps']} for the race to finish."
                                   if steps < est["race_own_steps"] else ".") + patience)

                        w["race_configs"] = RaceRows(on_change=race_summary_update)
                        w["race_configs"].value = d.race_configs
                        with ui.row().classes("items-center gap-2"):
                            ui.button("Add configuration", icon="add",
                                      on_click=lambda: w["race_configs"].add()).props("outline")
                            race_seeds = ui.input("Seeds", value="1-4").classes("w-32") \
                                .tooltip("Seeds for Add seed racers: 1, 2, 3 or a range 1-4")

                            def add_seed_racers():
                                from rulztrainer import race

                                try:
                                    seeds = race.parse_seeds(race_seeds.value)
                                    rows = w["race_configs"].value
                                except ValueError as e:
                                    ui.notify(str(e), type="warning")
                                    return
                                have = {r.get("seed") for r in rows}
                                new = [r for r in race.seed_rows(seeds) if r["seed"] not in have]
                                w["race_configs"].value = rows + new
                                w["race"].value = True
                                w["race_follow"].value = True
                                race_summary_update()
                                ui.notify(f"Added {len(new)} seed racers; relay on", type="positive")

                            ui.button("Add seed racers", icon="casino", on_click=add_seed_racers).props("outline") \
                                .tooltip("One racer per seed, the run's settings otherwise, and the relay on: after "
                                         "each probe round every seed continues from the one that probed best")
                        for key in ("steps", "probe_every", "race_warmup_probes", "race_drop_lowest",
                                    "race_patience_drop", "race_patience", "race_follow", "race_patience_rising",
                                    "race_pick"):
                            w[key].on_value_change(lambda _: race_summary_update())
                        with advanced("What a configuration may change"):
                            from rulztrainer.race import RACE_KEYS

                            for group, keys in RACE_KEYS.items():
                                ui.label(f"{group}: {', '.join(keys)}").classes("text-xs rz-mono")
                            hint('Lists in brackets (lora_targets=[q, k, v, o]); quote a value with a comma '
                                 '(lora_blocks="10-31,40"). Rank, layers and blocks may differ: each configuration '
                                 'keeps its own LoRA. A row with its own seed= starts from its own LoRA weights and '
                                 'sees its own image order and noise: seed twins for a seed soup.')

                    # ---- Seed soup
                    with sections.section("Tools", "Seed soup", "Average finished LoRAs of one subject into one",
                                          "blender"):
                        hint("Train the same subject two or three times with different seeds (or keep a race's "
                             "finalists: their _best files), then average what each one changes in the model and "
                             "save it as one LoRA: the model soup of Wortsman et al. (2022). What the runs agree on, "
                             "likely the likeness, stays; what each learned on its own fades. LoRA and LoKr files of "
                             "the same model; not OFT or DoRA. The average of LoRAs that disagree is weaker than "
                             "each, so it is tried at a few strengths. Untested on this trainer: compare by eye too.")
                        w["soup_files"] = ui.textarea("LoRA files (one per line)").props("autogrow") \
                            .classes("w-full")
                        with fields():
                            w["soup_rank"] = ui.number("Rank (0 = exact)", value=d.soup_rank, min=0,
                                                       precision=0).classes("w-40") \
                                .tooltip("The average of three rank-16 LoRAs has rank up to 48. 0 keeps it exactly "
                                         "(LoRA files only); a rank keeps the directions they share and drops the rest.")
                            w["soup_strengths"] = ui.input("Strengths to try", value=d.soup_strengths) \
                                .classes("w-48")
                        soup_info = ui.label("").classes("text-sm")

                        async def soup_now():
                            from rulztrainer.soup import describe, make_soup

                            files = [p.strip().strip('"') for p in (w["soup_files"].value or "").splitlines()
                                     if p.strip()]
                            cfg = f.to_config()
                            out = cfg.job_dir() / f"{cfg.job_name}_soup_plain.safetensors"
                            out.parent.mkdir(parents=True, exist_ok=True)
                            soup_info.text = f"Averaging {len(files)} files..."
                            try:
                                rep = await run.io_bound(make_soup, files, out, int(w["soup_rank"].value or 0))
                            except Exception as e:  # noqa: BLE001
                                soup_info.text = f"Could not make the soup: {e}"
                                return
                            soup_info.text = f"{describe(rep)}. Saved {rep['out']}"

                        with ui.row().classes("items-center gap-3"):
                            ui.button("Average now (strength 1, no probes)", icon="blender",
                                      on_click=soup_now).props("outline")
                            w["soup_run"] = ui.switch("Pick the soup with the probes when I press Start") \
                                .tooltip("Start then trains nothing: it probes each file alone (the best also at "
                                         "every strength), then adds the others best-first, keeping each only if the "
                                         "soup's probe score does not drop: the greedy soup. Saves "
                                         "<job>_soup.safetensors and soup/soup.json. Uses this page's model, "
                                         "references and probe settings; more probe seeds give a steadier choice.")

                    # ---- Post-hoc EMA
                    with sections.section("Tools", "Post-hoc EMA",
                                          "Average a finished run's checkpoints, choosing the decay afterwards",
                                          "timeline"):
                        hint("Weight average (lora_ema) smooths the LoRA while it trains, with a decay you choose "
                             "before the run. This makes the same kind of average after the run from the step "
                             "checkpoints it kept, so you can try a few decays on finished files and keep the one you "
                             "like (musubi-tuner's post-hoc EMA). Checkpoints of one run, in any order: they are put "
                             "in step order. Checkpoints to keep limits how far back it can reach. LoRA and LoKr "
                             "files; not OFT or DoRA.")
                        ema_files = ui.textarea("Checkpoints (one per line)").props("autogrow").classes("w-full")
                        with fields():
                            ema_mode = ui.select({"sigma_rel": "Power (EDM2)", "beta": "Decay per checkpoint"},
                                                 value="sigma_rel", label="Average").classes("w-56") \
                                .tooltip("Power weighs the checkpoints by their step whatever their number. A decay "
                                         "starts from the oldest file, so with few checkpoints it leans on it: 0.9 "
                                         "over 4 checkpoints leaves 73% on the oldest.")
                            ema_beta = ui.number("Decay", value=0.5, min=0, max=0.999, step=0.05).classes("w-28") \
                                .bind_visibility_from(ema_mode, "value", lambda x: x == "beta") \
                                .tooltip("Each newer checkpoint takes (1 - decay) of the average. 0.5 leans on the "
                                         "last two or three; 0.9 suits 20 or more checkpoints (over 10 the oldest "
                                         "still weighs about 39%).")
                            ema_beta2 = ui.number("End decay (optional)", value=None, min=0, max=0.999,
                                                  step=0.05).classes("w-44") \
                                .bind_visibility_from(ema_mode, "value", lambda x: x == "beta") \
                                .tooltip("Slides the decay in a straight line from Decay to this over the files; "
                                         "empty keeps it the same.")
                            ema_sigma = ui.number("sigma_rel", value=0.15, min=0.01, max=0.28, step=0.01) \
                                .classes("w-28").bind_visibility_from(ema_mode, "value", lambda x: x == "sigma_rel") \
                                .tooltip("The power-function average of Karras et al. (EDM2): every training step "
                                         "weighs by its step number to a power. 0.05 to 0.2; smaller leans harder "
                                         "on the last steps. 0.15 over 4 checkpoints: the newest 69%, the one before "
                                         "25%.")
                            ema_rank = ui.number("Rank (optional)", value=None, min=0, precision=0).classes("w-36") \
                                .tooltip("Empty: the files' own rank, which keeps nearly all of an average of one "
                                         "run's checkpoints. 0: exact, at the sum of their ranks (a larger file).")
                        ema_info = ui.label("").classes("text-sm")

                        def ema_fill():
                            cfg = f.to_config()
                            found = sorted(p for p in cfg.job_dir().glob(f"{cfg.job_name}_*.safetensors")
                                           if re.fullmatch(rf"{re.escape(cfg.job_name)}_\d+", p.stem))
                            ema_files.value = "\n".join(str(p) for p in found)
                            ema_info.text = (f"{len(found)} checkpoints of {cfg.job_name}" if found else
                                             f"No step checkpoints in {cfg.job_dir()}")

                        async def ema_now():
                            from rulztrainer.posthoc_ema import describe, make_ema

                            files = [p.strip().strip('"') for p in (ema_files.value or "").splitlines() if p.strip()]
                            cfg = f.to_config()
                            power = ema_mode.value == "sigma_rel"
                            tag = (f"sr{ema_sigma.value:g}" if power else f"b{ema_beta.value:g}"
                                   + (f"-{ema_beta2.value:g}" if ema_beta2.value not in (None, "") else ""))
                            out = cfg.job_dir() / f"{cfg.job_name}_posthoc_ema_{tag}.safetensors"
                            ema_info.text = f"Averaging {len(files)} checkpoints..."
                            try:
                                rep = await run.io_bound(
                                    make_ema, files, out, float(ema_beta.value or 0),
                                    None if power or ema_beta2.value in (None, "") else float(ema_beta2.value),
                                    float(ema_sigma.value) if power else None,
                                    None if ema_rank.value in (None, "") else int(ema_rank.value))
                            except Exception as e:  # noqa: BLE001
                                ema_info.text = f"Could not make the average: {e}"
                                return
                            ema_info.text = f"{describe(rep)}. Saved {rep['out']}"

                        with ui.row().classes("items-center gap-3"):
                            ui.button("Use this job's checkpoints", icon="playlist_add", on_click=ema_fill) \
                                .props("flat no-caps").tooltip("The <job>_<step> files in this job's output folder")
                            ui.button("Average now", icon="timeline", on_click=ema_now).props("outline")

                    # ---- Noise check
                    with sections.section("Tools", "Probe noise check",
                                          "How far the likeness probes move with their seeds", "query_stats"):
                        hint("A probe renders a few seeds and averages their scores, so part of every score is the "
                             "seeds' luck. This renders finished LoRAs (say a mid-run checkpoint and the best one of "
                             "a run) on many seeds, the same ones for each, and measures that luck: how far one seed "
                             "strays, how much of it two LoRAs share on the same seed, how many probe seeds it takes "
                             "to tell a 3, 5 or 8 point gap apart, and a noise band that fits your probe seeds. "
                             "Trains nothing; uses this page's model, references and probe settings. Writes "
                             "noise/noise.json.")
                        w["noise_files"] = ui.textarea("LoRA files (one per line; two from one run is best)") \
                            .props("autogrow").classes("w-full")
                        with ui.row().classes("items-center gap-3"):
                            w["noise_seeds"] = ui.number("Seeds per file", value=d.noise_seeds, min=3, max=64,
                                                         precision=0).classes("w-40")
                            w["noise_run"] = ui.switch("Run the noise check when I press Start") \
                                .tooltip("Start then trains nothing: it probes the base model (the scale's zero) and "
                                         "each file at this many seeds (face and body: two renders per seed).")

                sections.build(nav, picker_row)
                inline_descriptions(train_page)
                train_adv = TrainAdvanced(train_page, sections, w)
                train_adv_switch.on_value_change(lambda e: train_adv.apply(e.value))
                train_adv.apply(False)
                f.extras.update(cfg_path=cfg_path, race_seeds=race_seeds, ema_files=ema_files, ema_mode=ema_mode,
                                ema_beta=ema_beta, ema_beta2=ema_beta2, ema_sigma=ema_sigma, ema_rank=ema_rank)

        # ------------------------------------------------------------ Wizard (bound to the Train tab's widgets)
        with ui.tab_panel(tab_wizard):
            wizard = Wizard(f, gpus, lambda: start_job(confirm=False), goto, queue_job)
            wizard.build()  # its review lists the warnings
            wizard.adv_switch.bind_value(train_adv_switch, "value")  # one Advanced switch for both tabs
            f.extras["wizard_cfg_path"] = wizard.cfg_path

        # ------------------------------------------------------------ Dataset
        with ui.tab_panel(tab_dataset):
            with ui.column().classes("rz-page gap-4"):
                build_dataset_tab(f)

        # ------------------------------------------------------------ Monitor
        with ui.tab_panel(tab_monitor):
            show_full = image_viewer()
            with ui.column().classes("rz-page gap-4 rz-monitor"):
                with ui.element("div").classes("rz-tiles"):
                    tiles = {key: stat_tile(label) for key, label in (
                        ("state", "Status"), ("stage", "Stage"), ("step", "Step"), ("loss", "Loss"),
                        ("lr", "Learning rate"), ("eta", "Time left"))}
                job_bar = ui.linear_progress(value=0, show_value=False, size="6px").classes("w-full") \
                    .props("rounded")
                error_box = ui.label("").classes("rz-banner error")
                error_box.visible = False
                disk_box = ui.label("").classes("rz-banner warning")
                disk_box.visible = False
                bucket_box = ui.label("").classes("rz-banner info")
                bucket_box.visible = False
                # one part of the run at a time instead of one long page
                monitor_view = ViewSwitch(MONITOR_VIEWS, "overview")
                with ui.element("div").classes("rz-machine") as machine_row:
                    gpu_box = ui.element("div").classes("rz-machine-gpus")
                    sys_box = ui.element("div").classes("rz-machine-sys")
                gpu_widgets: list[dict] = []
                sys_widgets: dict = {}
                with section("Queue", "Runs that start one after another when the current one is done",
                             "queue", open_=False) as queue_exp:
                    with ui.row().classes("w-full items-center gap-2"):
                        queue_state = ui.label("").classes("text-sm font-medium")
                        ui.space()
                        queue_pause = ui.button("", on_click=lambda: (QUEUE.set_paused(
                            not QUEUE.snapshot()["paused"]), queue_draw(force=True)), color=None) \
                            .props("outline dense no-caps")
                        ui.button("Clear queue", icon="clear_all", on_click=lambda: (QUEUE.clear(),
                                                                                     queue_draw(force=True)),
                                  color=None).props("flat dense no-caps") \
                            .tooltip("Remove every waiting run (the running one goes on)")
                    hint("Add runs with Add to queue (bottom bar, or the wizard's last page): each starts when the "
                         "one before it is done, finished, stopped or failed. Kill pauses the queue. The queue lives "
                         "in output/_queue.json, but it is the WebUI that starts the next run, so keep the WebUI "
                         "running; a WebUI restarted with runs waiting and nothing running pauses the queue until you "
                         "Resume it.")
                    queue_list = ui.column().classes("w-full gap-1")
                    autostop_switch = None
                    if VAST_INSTANCE is not None and VAST_INSTANCE["can_stop"]:
                        # a rental bills while it sits idle: let it stop itself after the last run
                        def set_autostop(e):
                            if bool(e.value) != AUTOSTOP.armed:
                                AUTOSTOP.arm(_running()) if e.value else AUTOSTOP.disarm()

                        async def stop_now():
                            with ui.dialog() as dlg, ui.card().classes("w-[30rem] max-w-full"):
                                ui.label(f"Stop vast.ai instance {VAST_INSTANCE['id']} now?").classes("rz-setting-title")
                                ui.label("A running job ends without a final save, and this page goes away with the "
                                         "instance. The disk is kept (its storage charge goes on) and Start on "
                                         "vast.ai brings the instance back with everything on it.").classes("rz-hint")
                                with ui.row().classes("w-full justify-end gap-2"):
                                    ui.button("Keep it running", on_click=lambda: dlg.submit(False)).props("flat")
                                    ui.button("Stop the instance", icon="power_settings_new", color="negative",
                                              on_click=lambda: dlg.submit(True)).props("unelevated")
                            go = await dlg
                            dlg.delete()
                            if go:
                                AUTOSTOP.disarm("stopped from the Monitor")
                                ok, message = await run.io_bound(machine.stop_instance)
                                ui.notify(message, type="positive" if ok else "negative", timeout=0 if not ok else 8000)

                        with ui.column().classes("rz-intro w-full gap-2"):
                            with ui.row().classes("w-full items-center gap-3"):
                                autostop_switch = ui.switch("Stop this vast.ai instance when the runs are done",
                                                            value=AUTOSTOP.armed, on_change=set_autostop)
                                ui.space()
                                ui.button("Stop the instance now", icon="power_settings_new", on_click=stop_now,
                                          color=None).props("flat dense no-caps").classes("rz-danger")
                            hint(
                                f"{int(machine.STOP_GRACE_SECONDS // 60)} minutes after the last run ends (finished, "
                                "stopped or failed) with nothing left in the queue, the instance stops itself: the "
                                "GPU's billing ends, the disk and its storage charge stay, and Start on vast.ai brings "
                                "it back with every file. Runs waiting in a paused queue keep it running, and so does "
                                "starting a run in those minutes; Kill switches this off. It needs a run to have run "
                                "since you switched it on.")
                near_dup_label = ui.label("").classes("w-full text-sm text-orange-8")
                near_dup_label.visible = False
                with section("Configuration race", "Smoothed probe score per configuration; the lowest leaves after "
                             "every round", "emoji_events") as race_exp:
                    race_label = hint("")
                    race_table = ui.table(columns=[
                        {"name": "name", "label": "Configuration", "field": "name", "align": "left"},
                        {"name": "state", "label": "Status", "field": "state", "align": "left"},
                        {"name": "at", "label": "At step", "field": "at"},
                        {"name": "score", "label": "Score now", "field": "score"},
                        {"name": "face", "label": "Face now", "field": "face"},
                        {"name": "body", "label": "Body now", "field": "body"},
                        {"name": "drift", "label": "Off-prompt now", "field": "drift"},
                        {"name": "rounds", "label": "Score after each round", "field": "rounds", "align": "left"},
                        {"name": "settings", "label": "Settings", "field": "settings", "align": "left"},
                        {"name": "pick", "label": "", "field": "can_pick"},
                        {"name": "drop", "label": "", "field": "can_drop"}],
                        rows=[], row_key="name").classes("w-full")
                    race_table.add_slot("body-cell-pick", """
                        <q-td :props="props">
                            <q-btn v-if="props.row.can_pick" unelevated dense no-caps size="sm" color="primary"
                                   icon="call_split" label="Continue from"
                                   @click="() => $parent.$emit('pick', props.row.name)" />
                        </q-td>""")
                    race_table.add_slot("body-cell-drop", """
                        <q-td :props="props">
                            <q-btn v-if="props.row.can_drop" flat dense no-caps size="sm" color="negative"
                                   icon="close" label="Drop"
                                   @click="() => $parent.$emit('drop', props.row.name)" />
                        </q-td>""")

                    with ui.dialog() as drop_dlg, ui.card().classes("w-[26rem] max-w-full"):
                        drop_title = ui.label("").classes("rz-setting-title")
                        ui.label("It stops training now and its LoRA is saved as <job>_<name>_dropped; the race "
                                 "goes on without it. This cannot be undone.").classes("rz-hint")
                        with ui.row().classes("w-full justify-end gap-2"):
                            ui.button("Keep it", on_click=drop_dlg.close).props("flat")
                            drop_go = ui.button("Drop it", icon="close", color="negative").props("unelevated")
                    drop_state = {"name": ""}

                    def ask_drop(e):
                        drop_state["name"] = str(e.args)
                        drop_title.text = f"Drop {drop_state['name']} from the race?"
                        drop_dlg.open()

                    def do_drop():
                        drop_dlg.close()
                        if _job_dir is None or not _running():
                            ui.notify("No job is running.", type="warning")
                            return
                        request_race_drop(_job_dir, drop_state["name"])
                        ui.notify(f"{drop_state['name']} will be dropped after the current step.")

                    drop_go.on_click(do_drop)
                    race_table.on("drop", ask_drop)

                    def do_pick(e):
                        name = str(e.args)
                        if _job_dir is None or not _running():
                            ui.notify("No job is running.", type="warning")
                            return
                        request_race_pick(_job_dir, name)
                        ui.notify(f"Everyone continues from {name}.", type="positive")

                    race_table.on("pick", do_pick)
                race_exp.visible = False
                with section("Likeness probes & controller", "How much the probe images look like the person "
                             "(0 = a stranger, 100 = your references)", "insights") as probe_exp:
                    with ui.row().classes("w-full items-start gap-2"):
                        probe_chips = ui.row().classes("rz-kvs flex-1")
                        probe_sig: dict = {}

                        def export_likeness():
                            from urllib.parse import quote

                            if _job_dir is None:
                                ui.notify("No run to export yet.", type="warning")
                                return
                            ui.download(f"/rz-likeness-xlsx?job={quote(str(Path(_job_dir).resolve()))}",
                                        f"{Path(_job_dir).name}_likeness.xlsx")

                        ui.button("Export spreadsheet", icon="table_view", on_click=export_likeness, color=None) \
                            .props("flat no-caps").classes("rz-file-btn") \
                            .tooltip("Every probe's face, body and combined score so far, smoothed and per probe, as "
                                     "an Excel file: the scores are formulas of the raw ones and the run's scale "
                                     "(floors, references' level, span, curve, weights) on its own sheet, so you can "
                                     "rescore. In a race each configuration gets its own sheets.")
                    with ui.row().classes("items-center gap-3") as race_chips_row:
                        race_metrics = metric_chips(("combined", "face", "body"), lambda: draw_probe_chart(),
                                                    "Show lines (all configurations):", "configuration")
                    race_chips_row.visible = False
                    # the normalized scores (0 = a stranger, 100 = the references): smoothed lines, per-probe dots
                    probe_chart = RzChart(chart_options(
                        _axis("step", min="dataMin"), _axis("likeness % (normalized)", min=0, max=100),
                        *probe_chart_series([], []))).classes("w-full h-60")
                    charts.append(probe_chart)
                probe_exp.visible = False
                with section("Probe images", "What the likeness probes rendered at each probe step, with their scores; "
                             "click an image to see it full size", "face_retouching_natural") as probe_img_exp:
                    probe_gallery = StepGallery("No probe images yet.", show_full)
                probe_img_exp.visible = False
                with section("Sample images", "Your sample prompts at each sample step; click an image to see it full "
                             "size", "image") as sample_exp:
                    sample_gallery = StepGallery("No samples yet. They appear here every 'Sample every' steps.",
                                                 show_full)
                with section("Identity similarity", "How much the samples look like the person, per sample prompt",
                             "face") as face_exp:
                    face_chips = ui.row().classes("rz-kvs w-full")
                    face_sig: dict = {}
                    face_chart = RzChart(chart_options(_axis("step", min="dataMin"),
                                                         _axis("similarity %", min=0, max=100), [], [])) \
                        .classes("w-full h-60")
                    charts.append(face_chart)
                face_exp.visible = False
                with section("Loss", "Raw and smoothed training loss", "show_chart") as loss_exp:
                    with ui.row().classes("w-full items-center gap-4"):
                        ui.space()
                        ui.label("Smoothing").classes("rz-hint")
                        smoothing = ui.slider(min=0, max=0.98, step=0.02, value=0.8).classes("w-48")
                        ui.label().bind_text_from(smoothing, "value", lambda x: f"{x:.2f}").classes("rz-hint w-10")
                    loss_chart = RzChart(chart_options(
                        _axis("step", min="dataMin"),
                        [_axis("loss", scale=True),
                         _axis("id loss", scale=True, show=False, splitLine={"show": False})],
                        [_line("raw", "#52525B", lineStyle={"width": 1}),
                         _line("smoothed", "#FF3B30", lineStyle={"width": 2}),
                         _line("id loss (smoothed)", "#F4F4F5", yAxisIndex=1,
                               lineStyle={"width": 2, "type": "dashed"})],
                        ["raw", "smoothed"])).classes("w-full h-72")
                    charts.append(loss_chart)
                with section("Gradient norm", "How big each step's update was before clipping", "insights") \
                        as grad_card:
                    grad_label = ui.label("").classes("rz-hint")
                    hint("How big each step's update was before clipping (the loss chart's smoothing). A steady line "
                         "is healthy; spikes that keep growing, or a line that sits on the clip limit, mean the "
                         "learning rate is too high.")
                    grad_chart = RzChart(chart_options(
                        _axis("step", min="dataMin"), [_axis("gradient norm", scale=True)],
                        [_line("raw", "#52525B", lineStyle={"width": 1}),
                         _line("smoothed", "#FF3B30", lineStyle={"width": 2}),
                         _line("clip limit", "#F4F4F5", lineStyle={"width": 1, "type": "dashed"})],
                        ["raw", "smoothed"])).classes("w-full h-48")
                    charts.append(grad_chart)
                grad_card.visible = False
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
                with section("Region watch", "How far each part of the pictures has been learned", "grid_view") as rw_exp:
                    rw_label = hint("")
                    rs_label = hint("")
                    with ui.column().classes("w-full gap-1") as rs_box:
                        with ui.column().classes("w-full gap-1") as rs_gap_box:
                            hint("Gap between the training pictures and the held-out photos, sweep by sweep: how "
                                 "many points further each region has been learned on the pictures the model "
                                 "trains on than on the photos kept out of training. A line that keeps rising is "
                                 "that region being memorised rather than learned. The subject is face, hair and "
                                 "skin together. Click a name in the legend to hide or show its line.")
                            rs_gap_chart = RzChart(chart_options(
                                _axis("step", min="dataMin"),
                                _axis("training minus held-out (points)", scale=True), [], [])).classes("w-full h-72")
                            charts.append(rs_gap_chart)
                        hint("Each region on its own: how much of its starting error is gone (0% = as at the first "
                             "sweep), on the training pictures and on the held-out photos (grey, dashed).")
                        rs_region: dict = {}  # region -> (its cell, its chart)
                        with ui.element("div").classes("w-full grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3"):
                            for region in SWEEP_COLORS:
                                with ui.column().classes("w-full gap-0 min-w-0") as cell:
                                    ui.label(region.capitalize() + (" (face, hair and skin)" if region == "subject"
                                                                    else "")).classes("text-sm font-medium")
                                    chart = RzChart(chart_options(
                                        _axis("step", min="dataMin"), _axis("% gone", scale=True), [], [])) \
                                        .classes("w-full h-52")
                                    charts.append(chart)
                                rs_region[region] = (cell, chart)
                    rs_box.visible = False
                    rw_note = hint("Contrast: a picture's background (or clothes) loss against its subject's, "
                                   "relative to its own start. Under 1 = that region is being learned faster than "
                                   "the subject in this picture. One training sample per picture per epoch is a "
                                   "noisy reading, so read it as a trend.")
                    rw_note.visible = False
                    rw_table = ui.table(columns=[
                        {"name": "image", "label": "image", "field": "image", "align": "left", "sortable": True},
                        {"name": "background", "label": "background vs subject", "field": "background",
                         "sortable": True},
                        {"name": "clothes", "label": "clothes vs subject", "field": "clothes", "sortable": True},
                        {"name": "sweep_background", "label": "background (sweep)", "field": "sweep_background",
                         "sortable": True},
                        {"name": "sweep_clothes", "label": "clothes (sweep)", "field": "sweep_clothes",
                         "sortable": True},
                        {"name": "weights", "label": "steered weights", "field": "weights", "align": "left"},
                        {"name": "cells", "label": "cells: face / hair / skin / clothes / background",
                         "field": "cells", "align": "left"},
                        {"name": "seen", "label": "seen", "field": "seen", "sortable": True},
                    ], rows=[], row_key="path", pagination=15).props("flat dense").classes("w-full text-xs")
                    rw_table.visible = False  # (the per-picture table is the region watch's: a run with sweeps only
                    # has none)
                rw_exp.visible = False
                rw_state = {"sig": None, "sweep": None}
                with section("Saved files", "Download the run's LoRAs, logs and settings, probe and sample images",
                             "download") as files_exp:
                    job_files = JobFiles()
                    saved_label = ui.label("").classes("rz-mono whitespace-pre-wrap")
                with section("Log", "train.log as it is written", "terminal") as log_exp:
                    log_view = ui.log(max_lines=600).classes("w-full rz-log rz-log-tall rz-mono")
                # the views: each section into its view's column, in this order
                view_boxes: dict = {}
                for name, members, empty in (
                        ("overview", (machine_row, queue_exp, near_dup_label, race_exp, probe_exp, loss_exp), ""),
                        ("images", (probe_img_exp, sample_exp), ""),
                        ("details", (face_exp, grad_card, lw_exp, rw_exp),
                         "Nothing here for this run yet. This view fills in with what the run has switched on: the "
                         "samples' identity similarity, the gradient norm, the per-image loss watch and the region "
                         "watch."),
                        ("log", (files_exp, log_exp), "")):
                    with ui.column().classes("w-full gap-4 rz-view") as box:
                        note = hint(empty)
                    note.visible = False
                    for member in members:
                        member.move(box)
                    box.visible = name == "overview"
                    view_boxes[name] = (box, members, note if empty else None)

                def show_view(name: str) -> None:
                    for key, (box, _, _) in view_boxes.items():
                        box.visible = key == name
                    ui.run_javascript(f"try {{ localStorage.setItem('rulz.monitor', {json.dumps(name)}) }} catch (e) {{}}")

                monitor_view.on_change = show_view

                async def restore_view():
                    try:
                        name = await ui.run_javascript("localStorage.getItem('rulz.monitor')", timeout=3)
                    except Exception:  # noqa: BLE001  (storage blocked, slow client)
                        return
                    if name in MONITOR_VIEWS:
                        monitor_view.pick(name)

                ui.timer(0.05, restore_view, once=True)
            probe_view = {"probes": [], "names": []}

            def draw_probe_chart() -> None:
                """The likeness chart from the last poll's probes; in a race only the scores the chips leave on."""
                shown = [m for m, chip in race_metrics.items() if chip.value]
                series, legend = probe_chart_series(probe_view["probes"], probe_view["names"], shown)
                probe_chart.options["series"] = series
                probe_chart.options["legend"]["data"] = legend
                probe_chart.update()

            queue_shown = {"sig": None}

            def queue_draw(force: bool = False):
                q = QUEUE.snapshot()
                sig = json.dumps([[e["id"] for e in q["waiting"]], q["paused"], [r.get("started") or r.get("error")
                                                                                  for r in q["recent"][:5]]])
                if sig == queue_shown["sig"] and not force:
                    return
                queue_shown["sig"] = sig
                n = len(q["waiting"])
                queue_state.text = (f"{n} run{'s' if n != 1 else ''} waiting" if n else "Nothing queued") + \
                    (" · paused" if q["paused"] else "")
                queue_note.text = (f"Queue: {n} waiting" + (", paused" if q["paused"] else "")) if n else ""
                queue_pause.text = "Resume queue" if q["paused"] else "Pause queue"
                queue_exp.props["caption"] = (f"{n} waiting" + (", paused" if q["paused"] else "")) if n else \
                    "Nothing queued. Runs added with Add to queue start one after another"
                queue_exp.update()
                if n and queue_shown.get("n") != n:
                    queue_exp.value = True
                queue_shown["n"] = n
                queue_pause.props(f"icon={'play_arrow' if q['paused'] else 'pause'}")
                queue_list.clear()
                with queue_list:
                    for i, e in enumerate(q["waiting"]):
                        c = e.get("config") or {}
                        what = ", ".join(x for x in (
                            e.get("label") or "", f"{c.get('steps', 0):,} steps",
                            "race" if c.get("race") else "") if x)
                        with ui.row().classes("w-full items-center gap-2 no-wrap rz-queue-row"):
                            ui.label(f"#{i + 1}").classes("rz-hint w-8")
                            ui.label(e.get("job") or "(no job name)").classes("text-sm font-medium")
                            hint(what)
                            ui.space()
                            ui.button(icon="arrow_upward", on_click=lambda _, k=e["id"]: (QUEUE.move(k, -1),
                                                                                          queue_draw())) \
                                .props("flat dense round size=sm").tooltip("Earlier")
                            ui.button(icon="arrow_downward", on_click=lambda _, k=e["id"]: (QUEUE.move(k, 1),
                                                                                            queue_draw())) \
                                .props("flat dense round size=sm").tooltip("Later")
                            ui.button(icon="close", on_click=lambda _, k=e["id"]: (QUEUE.remove(k), queue_draw())) \
                                .props("flat dense round size=sm color=negative").tooltip("Remove from the queue")
                    for r in q["recent"][:5]:
                        when = time.strftime("%b %d %H:%M", time.localtime(r.get("started") or r.get("added") or 0))
                        text = (f"started {when}" if r.get("started") else f"could not start: {r.get('error')}")
                        hint(f"{r.get('job')}: {text}", warn=bool(r.get("error")))

            def autostop_draw():
                if autostop_switch is None:
                    return
                if autostop_switch.value != AUTOSTOP.armed:
                    autostop_switch.value = AUTOSTOP.armed
                if AUTOSTOP.seconds_left() is not None:  # counting down: the switch that cancels it in view
                    queue_exp.value = True

            ui.timer(2.0, while_connected(queue_draw))
            ui.timer(2.0, while_connected(autostop_draw))
            log_state = {"job": None, "n": 0, "console_shown": False}

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
                            with ui.column().classes("rz-tile rz-gpu gap-1"):
                                name = ui.label("").classes("text-sm font-semibold rz-strong")
                                bars = {}
                                for key, color in (("vram", "primary"), ("util", "grey-5")):
                                    with ui.row().classes("w-full items-center gap-2 no-wrap"):
                                        ui.label("VRAM" if key == "vram" else "Load").classes("rz-tile-sub w-12")
                                        bar = ui.linear_progress(value=0, show_value=False, size="6px",
                                                                 color=color).props("rounded") \
                                            .classes("flex-1")
                                        text = ui.label("").classes("rz-tile-sub w-28 text-right")
                                    bars[key] = (bar, text)
                            gpu_widgets.append({"name": name, **bars})
                for st, wdg in zip(stats, gpu_widgets):
                    wdg["name"].text = f"GPU {st['index']} · {st['name'].removeprefix('NVIDIA ')}"
                    wdg["vram"][0].value = st["used"] / max(1, st["total"])
                    wdg["vram"][1].text = f"{st['used'] / 1024:.1f} / {st['total'] / 1024:.1f} GB"
                    wdg["util"][0].value = st["util"] / 100
                    wdg["util"][1].text = f"{st['util']}%"

            def bar_tile(title: str):
                """A machine tile with one bar: (title label, bar, text)."""
                with ui.column().classes("rz-tile rz-gpu gap-1"):
                    name = ui.label(title).classes("text-sm font-semibold rz-strong")
                    with ui.row().classes("w-full items-center gap-2 no-wrap"):
                        bar = ui.linear_progress(value=0, show_value=False, size="6px", color="grey-5") \
                            .props("rounded").classes("flex-1")
                        text = ui.label("").classes("rz-tile-sub text-right")
                return name, bar, text

            machine_state = {"at": 0.0}

            def update_machine():
                """Disk and memory beside the GPUs (every 10 s), the low-disk warning and the model download."""
                if time.time() - machine_state["at"] < 10:
                    return
                machine_state["at"] = time.time()
                rows = machine.disks({"runs": ROOT / (w["output_dir"].value or "output"), "models": ROOT / "models"})
                mem = machine.memory()
                keys = [f"disk{i}" for i in range(len(rows))] + (["mem"] if mem else [])
                if list(sys_widgets) != keys:
                    sys_box.clear()
                    sys_widgets.clear()
                    with sys_box:
                        for key in keys:
                            sys_widgets[key] = bar_tile("")
                for i, r in enumerate(rows):
                    name, bar, text = sys_widgets[f"disk{i}"]
                    name.text = f"Disk · {r['label']}"
                    bar.value = r["used"] / max(1e-9, r["total"])
                    bar.props(f"color={ {'ok': 'grey-5', 'warning': 'warning', 'critical': 'negative'}[r['level']] }")
                    text.text = f"{r['free']:.0f} GB free of {r['total']:.0f}"
                if mem:
                    name, bar, text = sys_widgets["mem"]
                    name.text = "Memory"
                    bar.value = mem["used"] / max(1e-9, mem["total"])
                    text.text = f"{mem['used']:.0f} / {mem['total']:.0f} GB"
                low = [r for r in rows if r["level"] != "ok"]
                disk_box.visible = bool(low)
                if low:
                    worst = min(low, key=lambda r: r["free"])
                    disk_box.classes(replace=f"rz-banner {'error' if worst['level'] == 'critical' else 'warning'}")
                    disk_box.text = (f"Only {worst['free']:.1f} GB free on the drive with the {worst['label']} "
                                     f"({worst['path']}). A save that does not fit ends the run: download and remove "
                                     "old runs (Previous Runs), or keep fewer checkpoints (Train > Saving & samples).")
                dl = machine.bucket_download(ROOT / "models")
                bucket_box.visible = bool(dl) and dl["state"] != "done"
                if dl and dl["state"] == "running":
                    bucket_box.text = (f"Your model files are still downloading into models/bucket ({dl['gb']:.1f} GB "
                                       "so far). A run that needs a file that is not there yet fails at loading; the "
                                       "log is models/bucket_download.log.")
                elif dl and dl["state"] == "failed":
                    bucket_box.classes(replace="rz-banner warning")
                    bucket_box.text = ("The model download did not finish (models/bucket_download.log). Run it again "
                                       "in a terminal: bash scripts/vast/download_models.sh")

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

            def update_grad_chart(st: dict, start: int, a: float) -> None:
                """The gradient norm chart: one value per step beside the losses (None where it was not finite), the
                same x and smoothing as the loss chart, and the clip limit when max_grad_norm is on."""
                norms = st.get("grad_norms") or []
                if not any(v is not None for v in norms):  # an older run, or no step yet
                    grad_card.visible = False
                    return
                grad_card.visible = True
                start += len(st.get("losses") or []) - len(norms)  # (aligned at the end: the last step is the same)
                raw, sm, ema = [], [], None
                for i, v in enumerate(norms):
                    x = start + i + 1
                    raw.append([x, v])
                    if v is not None:
                        ema = v if ema is None else a * ema + (1 - a) * v
                        sm.append([x, ema])
                clip = float(st.get("grad_clip") or 0)
                clipped = sum(1 for v in norms[-100:] if v is not None and 0 < clip < v)
                bad = sum(1 for v in norms if v is None)
                grad_chart.options["series"][0]["data"] = raw
                grad_chart.options["series"][1]["data"] = sm
                grad_chart.options["series"][2]["data"] = [[raw[0][0], clip], [raw[-1][0], clip]] if clip > 0 else []
                grad_chart.options["legend"]["data"] = ["raw", "smoothed"] + (["clip limit"] if clip > 0 else [])
                grad_chart.update()
                now = st.get("grad_norm")
                parts = [f"now {now:.3g}" if now is not None else "now not finite"]
                if clip > 0:
                    parts.append(f"clipped in {clipped} of the last {min(100, len(norms))} steps")
                if bad:
                    parts.append(f"{bad} not finite")
                grad_label.text = " · ".join(parts)

            def view_notes():
                """A view whose sections are all hidden says so instead of showing an empty page."""
                for _, members, note in view_boxes.values():
                    if note is not None:
                        note.visible = not any(m.visible for m in members)

            def poll():
                update_gpus()
                update_machine()
                view_notes()
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
                show("step", f"{step:,}", f"of {total:,} · {step / total * 100:.0f}% done" if total else "")
                if loss is not None:
                    show("loss", f"{loss:.4f}", f"average {st.get('avg_loss') or 0:.4f}")
                show("lr", f"{st['lr']:.2e}" if st.get("lr") is not None else "-")
                show("eta", eta_text(eta) if running else "-")
                error_box.text = f"Error: {st['error']}" if st.get("error") else ""
                error_box.visible = bool(st.get("error"))
                job_bar.value = (step / total) if total else 0
                job_bar.props(f"color={'primary' if running else 'warning' if st.get('error') else 'positive' if st.get('done') else 'grey-6'}")
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
                    update_grad_chart(st, start, a)
                fscores = st.get("face_scores") or []
                if fscores:
                    face_exp.visible = True
                    by_prompt: dict[int, list] = {}
                    per_step: dict[int, dict[str, list]] = {}
                    normalized = any(sc.get("combined_norm") is not None for sc in fscores)  # else raw (older runs)
                    nz = lambda sc, nk, rk: sc.get(nk) if normalized else sc.get(rk)  # noqa: E731

                    for sc in fscores:
                        comb = nz(sc, "combined_norm", "combined_percent")
                        if comb is not None:
                            by_prompt.setdefault(sc["prompt_index"], []).append([sc["step"], round(comb, 1)])
                        d_ = per_step.setdefault(sc["step"], {"face": [], "body": [], "combined": []})
                        if nz(sc, "face_norm", "percent") is not None:
                            d_["face"].append(nz(sc, "face_norm", "percent"))
                        if nz(sc, "body_norm", "body_percent") is not None:
                            d_["body"].append(nz(sc, "body_norm", "body_percent"))
                        if comb is not None:
                            d_["combined"].append(comb)
                    series = []
                    for key, color, style in (("combined", "#F59E0B", {"width": 3}), ("face", "#FF3B30", {"width": 2}),
                                              ("body", "#F4F4F5", {"width": 2, "type": "dashed"})):
                        pts = [[stp, round(sum(v[key]) / len(v[key]), 1)] for stp, v in sorted(per_step.items()) if v[key]]
                        if pts:
                            series.append(_line(f"mean {key}", color, lineStyle=style, showSymbol=True, symbolSize=4,
                                                data=pts))
                    series += [_line(f"prompt {k + 1}", CHART_COLORS[(k + 3) % len(CHART_COLORS)],
                                     lineStyle={"width": 1, "opacity": 0.45}, data=v)
                               for k, v in sorted(by_prompt.items())]
                    ref_sim = st.get("face_ref_similarity")
                    if series and (normalized or ref_sim is not None):
                        xs = [pt[0] for v in by_prompt.values() for pt in v] or [0, 1]
                        level = 100.0 if normalized else ref_sim * 100  # normalized: the references are 100 as such
                        series.append(_line("your references" if normalized else "face reference self-similarity",
                                            "#34D399", lineStyle={"type": "dashed", "width": 1},
                                            data=[[min(xs), level], [max(xs), level]]))
                    face_chart.options["legend"]["data"] = [sr["name"] for sr in series]
                    face_chart.options["series"] = series
                    face_chart.update()
                    latest_step = max(sc["step"] for sc in fscores)
                    latest = [sc for sc in fscores if sc["step"] == latest_step]
                    fv = [nz(sc, "face_norm", "percent") for sc in latest if nz(sc, "face_norm", "percent") is not None]
                    bv = [nz(sc, "body_norm", "body_percent") for sc in latest
                          if nz(sc, "body_norm", "body_percent") is not None]
                    cv = [nz(sc, "combined_norm", "combined_percent") for sc in latest
                          if nz(sc, "combined_norm", "combined_percent") is not None]
                    chips = [("Step", f"{latest_step:,}", False)]
                    if cv:
                        chips.append(("Combined", f"{sum(cv) / len(cv):.1f}%", False))
                    chips.append(("Face", f"{sum(fv) / len(fv):.1f}% ({len(fv)} of {len(latest)} with a face)" if fv
                                  else f"no face in {len(latest)} sample(s)", not fv))
                    if st.get("body_refs"):
                        chips.append(("Body", f"{sum(bv) / len(bv):.1f}% ({len(bv)} of {len(latest)} with a person)"
                                      if bv else "no usable person", not bv))
                    if ref_sim is not None:
                        chips.append(("Face refs alike", f"{ref_sim * 100:.1f}%", False))
                    if st.get("body_ref_similarity") is not None:
                        chips.append(("Body refs alike", f"{st['body_ref_similarity'] * 100:.1f}%", False))
                    stat_chips(face_chips, chips, face_sig)
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
                rw = st.get("region_watch")
                rw_note.visible = rw_table.visible = bool(rw)
                if rw:
                    rw_exp.visible = True
                    sig = (rw.get("epoch"), rw.get("step"))
                    if sig != rw_state["sig"]:
                        rw_state["sig"] = sig
                        rw_label.text = f"Epoch {rw.get('epoch')} (step {rw.get('step')}): {region_watch_line(rw)}"
                        rw_table.rows = rw.get("rows") or []
                        rw_table.update()
                rs = st.get("region_sweep")
                if rs:
                    rw_exp.visible = True
                    rs_label.text = f"Sweep at step {rs.get('step')}: {rs.get('line') or ''}"
                    sig = (str(_job_dir), rs.get("step"), len(rs.get("history") or []))
                    if sig != rw_state["sweep"]:
                        rw_state["sweep"] = sig
                        history = sweep_history(rs, _job_dir)
                        series, legend = sweep_gap_series(history)
                        rs_gap_chart.options["series"] = series
                        rs_gap_chart.options["legend"]["data"] = legend
                        rs_gap_chart.update()
                        rs_gap_box.visible = any(len(sr["data"]) > 1 for sr in series)
                        drawn = False
                        for region, (cell, chart) in rs_region.items():
                            series, legend = sweep_region_series(history, region, rs.get("best"))
                            chart.options["series"] = series
                            chart.options["legend"]["data"] = legend
                            chart.update()
                            cell.visible = any(len(sr["data"]) > 1 for sr in series)
                            drawn = drawn or cell.visible
                        rs_box.visible = drawn
                else:
                    rs_box.visible = False
                dup_note = near_duplicate_note(st.get("near_duplicates"))
                near_dup_label.visible = bool(dup_note)
                near_dup_label.text = dup_note
                rc = st.get("race")
                race_exp.visible = bool(rc)
                if rc:
                    names, dropped = rc.get("names") or [], rc.get("dropped") or {}
                    f1 = lambda x: "n/a" if x is None else f"{x:.1f}%"  # noqa: E731

                    def part(kind, n):  # the normalized smoothed score
                        norm, _ = (rc.get(kind) or {}).get(n) or (None, None)
                        return "n/a" if norm is None else f1(norm)

                    rows = []
                    running_now = _running()
                    for n in names:
                        manual = n in (rc.get("manual") or [])
                        relay_done = rc.get("follow") and n in (rc.get("finished") or [])
                        state = ("winner" if rc.get("winner") == n else
                                 "finished (lost the last round)" if relay_done else
                                 f"dropped by you in round {dropped[n]}" if n in dropped and manual else
                                 f"out after round {dropped[n]} (its probes left their prompts' frame)"
                                 if n in dropped and n in (rc.get("drifted") or []) else
                                 f"out after round {dropped[n]}" if n in dropped
                                 else "training" if (n in (rc.get("lanes") or []) or (rc.get("active") == n
                                                                                       and not rc.get("parallel")))
                                 else "waiting its turn")
                        if rc.get("awaiting") and n in (rc.get("alive") or []):
                            state = "waiting for your pick" + (" (highest score)" if rc.get("awaiting") == n else "")
                        elif rc.get("follow") and rc.get("leader") == n and not rc.get("winner"):
                            state += " (led the last round)"
                        spared = ((rc.get("history") or [{}])[-1].get("spared") or {}).get(n)
                        if spared and n not in dropped:
                            state += f" (kept after {spared} probes without improvement: rising again)"
                        per_round = [(h["scores"].get(n), (h.get("steps") or {}).get(n))
                                     for h in rc.get("history") or [] if n in h["scores"]]
                        at = (rc.get("score_steps") or {}).get(n)
                        drift = (rc.get("prompt_drift") or {}).get(n)  # the prompt check's latest drift share
                        rows.append({"name": n, "state": state, "at": "-" if at is None else f"step {at}",
                                     "score": f1((rc.get("scores") or {}).get(n)),
                                     "face": part("face", n), "body": part("body", n),
                                     "drift": "-" if drift is None else f"{100 * drift:.0f}%",
                                     "rounds": " / ".join(f1(s) + ("" if st is None else f" @{st}")
                                                          for s, st in per_round) or "-",
                                     "settings": (rc.get("settings") or {}).get(n) or "the page's own settings",
                                     "can_pick": bool(rc.get("awaiting")) and n in (rc.get("alive") or [])
                                     and running_now,
                                     "can_drop": n not in dropped and not rc.get("winner") and running_now
                                     and not rc.get("awaiting")})
                    race_table.rows = rows
                    race_table.update()
                    est = rc.get("estimate") or {}
                    last_round = (rc.get("history") or [{}])[-1]
                    race_label.text = (f"Relay round {last_round.get('round')} is over: look at its probe images, then "
                                       f"press Continue from on the racer everyone should go on from"
                                       + (" (the last round: your pick is the run's LoRA)"
                                          if last_round.get("final") else "")
                                       + f". {rc.get('awaiting')} scored highest and leads if you do not pick in time."
                                       if rc.get("awaiting") else
                                       f"{rc.get('winner')} won the relay's last round; its LoRA is the run's"
                                       if rc.get("winner") and rc.get("follow") else
                                       f"{rc.get('winner')} won; it trains on to the end" if rc.get("winner") else
                                       f"Relay round {rc.get('round')}: everyone continues from "
                                       f"{rc.get('leader') or 'the round winner'} after each round, "
                                       f"{est.get('total')} steps in all" if rc.get("follow") else
                                       f"Round {rc.get('round')}: {len(rc.get('alive') or [])} of {len(names)} "
                                       f"configurations left, about {est.get('total')} steps in all")
                probes = st.get("probes") or []
                if probes:
                    probe_exp.visible = True
                    probe_view["probes"], probe_view["names"] = probes, (rc or {}).get("names") or []
                    race_chips_row.visible = bool(probe_view["names"])
                    draw_probe_chart()
                    lk = st.get("likeness") or {}
                    last = probes[-1]
                    f2 = lambda x: "n/a" if x is None else f"{x:.1f}%"  # noqa: E731
                    normed = last.get("face_norm_ema") is not None or last.get("body_norm_ema") is not None
                    fe = last.get("face_norm_ema") if normed else last.get("face_ema")
                    be = last.get("body_norm_ema") if normed else last.get("body_ema")
                    ge = last.get("gold_norm_ema") if normed else last.get("gold_ema")
                    since = lk.get("since_improve", 0)
                    chips = [("Step", f"{last['step']:,}", False), ("Combined", f2(last.get("combined_ema")), False),
                             ("Face", f2(fe), False)]
                    if ge is not None:
                        chips.append((f"{_gold_name(last)} face" + (" (in charge)" if last.get("gold_select")
                                                                      else ""), f2(ge),
                                      False))
                    chips += [("Body", f2(be), False),
                              ("Best", f"{f2(lk.get('best'))} at step {lk.get('best_step') or 0:,}"
                               if lk.get("best") is not None else "n/a", False),
                              ("Since best", f"{since} probe{'s' if since != 1 else ''}", False),
                              ("Learning rate", f"x{lk.get('lr_factor', 1.0):.2f}", False),
                              ("Face bias", f"{lk.get('face_bias', 0.0):+.2f}", False)]
                    if lk.get("stopped"):
                        chips.append(("Early stop", "stopped", True))
                    stat_chips(probe_chips, chips, probe_sig)
                    rnames = (rc or {}).get("names") or []
                    by_step = probe_steps(probes, rnames)
                    probe_img_exp.visible = bool(by_step)
                    probe_gallery.set(by_step)
                sample_gallery.set(sample_steps(st.get("samples") or [], fscores, (rc or {}).get("names") or []))
                job_files.show(_job_dir)
                states = list(st.get("states") or [])
                saved_label.text = "Training states:\n" + "\n".join(states) if states else ""
                update_log()

            ui.timer(2.0, while_connected(poll))

        # ------------------------------------------------------------ Samples
        with ui.tab_panel(tab_samples):
            with ui.column().classes("rz-page gap-4"):
                build_samples_tab(f)

        # ------------------------------------------------------------ Training maps
        with ui.tab_panel(tab_maps):
            with ui.column().classes("rz-page gap-4"):
                build_maps_tab(f)

        # ------------------------------------------------------------ Previous Runs
        with ui.tab_panel(tab_history):
            with ui.column().classes("rz-page gap-4 rz-history"):
                build_history_tab(charts)

        # ------------------------------------------------------------ Fleet
        if tab_fleet is not None:
            with ui.tab_panel(tab_fleet):
                with ui.column().classes("rz-page gap-4"):
                    build_fleet_tab(form_config)

        # ------------------------------------------------------------ Help
        with ui.tab_panel(tab_help):
            with ui.column().classes("rz-page"):
                with card():
                    ui.markdown(HELP).classes("rz-help")

    # ---------------------------------------------------------------- action bar (Train and Monitor tabs)
    with ui.footer().classes("rz-footer") as footer:
        with ui.row().classes("rz-footer-row w-full items-center gap-3"):
            with ui.column().classes("gap-1 min-w-[12rem] flex-1 rz-footer-msg"):
                status_msg = ui.label("").classes("rz-status-msg")
                with ui.row().classes("items-center gap-2 no-wrap") as task_row:
                    ui.spinner(size="sm")
                    task_text = ui.label("").classes("text-sm")
                    task_bar = ui.linear_progress(value=0, show_value=False, size="6px").classes("w-48") \
                        .props("rounded")
                task_row.visible = False
            with ui.row().classes("items-center gap-2 no-wrap rz-footer-btns"):
                queue_note = ui.label("").classes("rz-hint text-sm cursor-pointer")
                queue_note.on("click", lambda: goto("Monitor"))
                ui.button("Add to queue", icon="playlist_add", on_click=queue_job, color=None).props("outline") \
                    .classes("rz-icon-only") \
                    .tooltip("Start this run after the current one (and any queued before it) is done")
                start_btn = ui.button("Start training", icon="play_arrow", on_click=start_job).props("unelevated")
                stop_btn = ui.button("Stop", icon="stop",
                                     on_click=lambda: setattr(status_msg, "text", stop_training()), color=None) \
                    .props("outline").tooltip("Stop after the current step and save the LoRA")
                kill_btn = ui.button("Kill", icon="dangerous",
                                     on_click=lambda: setattr(status_msg, "text", kill_training())) \
                    .props("flat color=negative").classes("rz-icon-only rz-danger") \
                    .tooltip("End the process now, without a final save")
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

    strip_shown = {"title": "RulzTrainer", "running": None}

    def pill_tick():
        """The run strip (every tab), the action bar's buttons and the browser tab's title."""
        st = read_status() or {}
        running = _running()
        state = "live" if running else "error" if st.get("error") else "done" if st.get("done") else ""
        pill_dot.classes(replace=f"rz-dot {state}")
        job = _job_dir.name if _job_dir is not None and (running or st) else ""
        step, total = int(st.get("step") or 0), int(st.get("total") or 0)
        frac = min(1.0, step / total) if total else 0.0
        title = "RulzTrainer"
        if running:
            stage = str(st.get("stage") or "starting")
            ft = st.get("finetune") or {}
            ft_text = (f" · fine-tune window {ft['window']}/{ft['windows']}, cycle {ft['cycle']}"
                       if ft.get("windows") else "")
            run_state.text = stage[:1].upper() + stage[1:]
            run_step.text = f"step {step:,} / {total:,} · {frac * 100:.0f}%{ft_text}" if total else ""
            eta = st.get("eta")
            run_eta.text = f"{eta_text(eta)} left" if eta and stage == "training" else ""
            title = f"{frac * 100:.0f}% · {job} · RulzTrainer" if total else f"{run_state.text} · {job} · RulzTrainer"
        else:
            run_state.text = {"error": "Failed", "done": "Finished"}.get(state, "Stopped" if st else "Idle")
            run_step.text = (f"step {step:,} / {total:,}" if total else "") if st else \
                "No run yet: set one up in the Wizard or on the Train tab"
            run_eta.text = ""
            if state:
                title = f"{run_state.text} · {job} · RulzTrainer"
        run_job.text = job
        run_bar.props(f"color={'positive' if state == 'done' else 'warning' if state == 'error' else 'primary'}")
        run_bar.value = frac
        run_bar.visible = bool(total)
        left = AUTOSTOP.seconds_left()
        waiting = len(QUEUE.snapshot()["waiting"])
        notes = [f"{waiting} queued"] if waiting else []
        if left is not None:
            notes.append(f"this instance stops in {int(left) // 60}:{int(left) % 60:02d}")
        elif AUTOSTOP.armed:
            notes.append("the instance stops when the runs are done")
        run_note.text = " · ".join(notes)
        run_note.classes(add="rz-warn") if left is not None else run_note.classes(remove="rz-warn")
        if title != strip_shown["title"]:  # a background browser tab shows how far the run is
            strip_shown["title"] = title
            ui.page_title(title)
        if running != strip_shown["running"]:
            strip_shown["running"] = running
            start_btn.set_enabled(not running)
            stop_btn.set_enabled(running)
            kill_btn.set_enabled(running)

    def machine_tick():
        """The header's machine chip: the least free space of the drives the runs and the models are on."""
        rows = machine.disks({"runs": ROOT / (w["output_dir"].value or "output"), "models": ROOT / "models"})
        inst = VAST_INSTANCE
        if not rows:
            machine_chip.visible = inst is not None
            machine_text.text = "vast.ai" if inst else ""
            return
        low = min(rows, key=lambda r: r["free"])
        machine_text.text = ("vast.ai · " if inst else "") + f"{low['free']:.0f} GB free"
        machine_icon.props(f"name={'cloud' if inst else 'storage'}")
        machine_chip.classes(remove="ok warning critical", add=low["level"])
        lines = [f"{r['label']}: {r['free']:.1f} GB free of {r['total']:.0f} GB ({r['path']})" for r in rows]
        if low["level"] != "ok":
            lines.append("Low: a save that does not fit ends the run. Download and delete old runs (Previous Runs), "
                         "or keep fewer checkpoints.")
        if inst:
            lines.append(f"vast.ai instance {inst['id']}: it bills while it is running, with or without a job.")
        machine_tip.text = " | ".join(lines)

    ui.timer(0.5, while_connected(task_tick))
    ui.timer(2.0, while_connected(pill_tick))
    ui.timer(10.0, while_connected(machine_tick))
    pill_tick()
    machine_tick()

    tab_names = ("Wizard", "Train", "Dataset", "Monitor", "Samples", "Training maps", "Previous Runs", "Fleet",
                 "Help")

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

    def choose_look() -> None:
        """First visit in this browser (no saved look): pick Dark or Light. Closing the dialog keeps Dark."""
        picked = {"light": False, "saved": False}
        with ui.dialog() as dlg, ui.card().classes("rz-card gap-4").style("width: 520px; max-width: 94vw"):
            ui.label("Choose your look").classes("rz-step-title")
            ui.label("You can change it any time with the sun / moon button at the top right.").classes("rz-hint")
            cards = {}
            with ui.element("div").classes("rz-look-grid"):
                for light, name in ((False, "Dark"), (True, "Light")):
                    with ui.column().classes("rz-look") as c:
                        with ui.element("div").classes("rz-mini " + ("light" if light else "dark")):
                            ui.element("div").classes("bar text")
                            with ui.element("div").classes("panel"):
                                ui.element("div").classes("bar")
                                ui.element("div").classes("bar")
                            ui.element("div").classes("bar red")
                        with ui.row().classes("items-center w-full no-wrap"):
                            ui.label(name).classes("rz-look-name")
                            ui.space()
                            ui.icon("check_circle").classes("rz-check")
                    cards[light] = c
                    c.on("click", lambda _, lt=light: pick(lt))

            def pick(light: bool) -> None:
                picked["light"] = light
                for lt, c in cards.items():
                    c.classes(add="selected") if lt == light else c.classes(remove="selected")
                set_look(light, save=False)  # a live preview; saved on Continue

            def done() -> None:
                picked["saved"] = True
                set_look(picked["light"])
                dlg.close()

            def closed() -> None:
                if not picked["saved"]:  # dismissed without Continue: Dark, saved
                    set_look(False)

            with ui.row().classes("w-full justify-end"):
                ui.button("Continue", on_click=done).props("unelevated color=primary")
            dlg.on("hide", closed)
        cards[False].classes(add="selected")
        dlg.open()

    async def restore_look():
        """Charts follow the saved look; with no saved look yet, ask once."""
        try:
            saved = await ui.run_javascript(f"localStorage.getItem({json.dumps(THEME_KEY)})", timeout=3)
        except Exception:  # noqa: BLE001  (storage blocked, slow client): stay dark, don't ask
            return
        if saved in ("dark", "light"):
            look["light"] = saved == "light"
            for chart in charts:
                recolor_chart(chart, look["light"])
        else:
            choose_look()

    ui.timer(0.05, restore_look, once=True)

    def fit_charts(width) -> None:
        for chart in charts:
            chart._rz_width = int(width or 1280)
            chart.update()

    # charts fit the browser's width: sent once the page is connected, and again after the window is resized (a
    # phone turned on its side)
    ui.on("rz_width", lambda e: fit_charts(e.args))
    ui.timer(0.1, lambda: ui.run_javascript(
        "if (!window.rzWidthWatch) { let t; window.rzWidthWatch = true; addEventListener('resize', () => {"
        " clearTimeout(t); t = setTimeout(() => emitEvent('rz_width', innerWidth), 300); }); }"
        " emitEvent('rz_width', innerWidth);"), once=True)

    def page_state() -> dict:
        """What a reload keeps: the form and the page's own state around it."""
        snapshot = preset_state["snapshot"]
        return {"form": f.snapshot(form_state.get("form")), "advanced": bool(train_adv_switch.value),
                "wizard_set_up": wizard.set_up, "preset_note": preset_note.text,
                "preset_snapshot": snapshot.to_dict() if snapshot is not None else None}

    def restore_page(state: dict) -> None:
        wizard.set_up = False  # its follow-the-form reactions stay out of the restoring
        wizard.racers_loaded = False  # it reads the race rows in again
        f.restore(state.get("form") or {})
        train_adv_switch.value = bool(state.get("advanced"))
        wizard.set_up = bool(state.get("wizard_set_up"))  # false would put the recipe over the form on its first page
        preset_note.text = str(state.get("preset_note") or "")
        try:  # switching the identity preset off still brings back the settings from before it
            snapshot = state.get("preset_snapshot")
            preset_state["snapshot"] = TrainConfig.from_dict(snapshot) if snapshot else None
        except (TypeError, ValueError):
            preset_state["snapshot"] = None

    def persist() -> None:
        """Polled, so every input counts without a hook on each one, the rows that come and go included, and a page
        writes only what changed on it: an idle second page cannot overwrite the first page's settings."""
        state = page_state()
        text = json.dumps(state, indent=1, default=str)
        if text == form_state["text"]:
            return
        if save_form_state(text):
            form_state.update(text=text, form=state["form"], warned=False)
        elif not form_state["warned"]:
            form_state["warned"] = True
            ui.notify("Could not save the settings for the next reload; retrying", type="warning")

    form_state["pristine"] = page_state()  # what Reset puts back
    saved = load_form_state()
    job_config = _job_dir / "config.yaml" if job_running and _job_dir is not None else None
    if saved is not None:  # takes the place of the running job's settings: the form may hold the next run by now
        restore_page(saved)
        if job_config is not None:
            status_msg.text = f"Reconnected to the running job '{_job_dir.name}' (your settings on this page were kept)"
    elif job_config is not None and job_config.is_file():
        # reattached to a running job: show its settings instead of the defaults
        try:
            f.from_config(TrainConfig.load(job_config))
            status_msg.text = f"Reconnected to the running job '{_job_dir.name}' (settings loaded from its config)"
        except Exception as e:  # noqa: BLE001
            status_msg.text = f"Reconnected to the running job '{_job_dir.name}' (could not load its config: {e})"
    state = page_state()  # the baseline: nothing is written until something changes from this
    form_state.update(text=json.dumps(state, indent=1, default=str), form=state["form"])
    ui.timer(0.4, while_connected(persist))


HELP = """
## Finding your way around
The tabs follow the work, in three groups: **Wizard, Train, Dataset** set a run up, **Monitor, Samples, Training maps,
Previous Runs** watch it and look back, and **Help** is this page. The **run strip** under the tabs is on every tab: what
the trainer is doing, the job, the step, a progress bar, the time left and the number of queued runs. Click it for the
Monitor. The browser tab's title carries the progress too, so a tab in the background still tells you how far the run
is. The chip at the top right is the machine: the free space of the drive with your runs and models (amber under
15 GB, red under 5 GB), and the instance number on a vast.ai rental.

The **Monitor** shows one view at a time: **Overview** (GPUs, disk and memory, the queue, the race, the likeness chart
and the loss), **Images** (the probe images and your samples, a race's configurations side by side), **Details** (the
samples' identity similarity, gradient norm, per-image loss watch and region watch with the region sweeps' charts,
each when the run has it) and
**Log & files** (the downloads and train.log). The numbers above the views stay put.

Red marks the one main action of a view and whatever is switched on; every other button is plain text with a red icon.
**Start training** is greyed out while a run is on, **Stop** and **Kill** while none is.

## On a rented GPU (vast.ai)
- **Disk**: a rental's disk has a fixed size, and a save that does not fit ends the run. The chip at the top right and
  the Monitor's *Disk* tile show what is free; a banner on the Monitor warns under 15 GB. Download what you want to
  keep (Monitor > Log & files, or Previous Runs > Saved files) and remove old runs.
- **Stop the instance when the runs are done**: an instance bills while it sits idle. On an instance, the Monitor's
  *Queue* section has the switch *Stop this vast.ai instance when the runs are done*: three minutes after the last run
  ends (finished, stopped or failed) with nothing left in the queue, the instance stops itself. The GPU's billing ends,
  the disk and its storage charge stay, and *Start* on vast.ai brings it back with every file. The run strip counts
  the three minutes down; switching it off or starting a run cancels, and **Kill** switches it off. Runs waiting in
  a paused queue keep the instance running. *Stop the instance now* beside it does the same at once. It uses the key vast.ai gives every
  instance for itself (start, stop and destroy of that instance only); nothing here can rent or destroy anything.
- **Model download**: while the first-boot download of your model bucket is still going, the Monitor says so and how
  much has arrived.
- **Files**: Browse buttons open a browser of the instance's files, datasets upload from the Dataset tab, and every
  download button hands the file to your browser.
- The page reconnects by itself after a dropped connection; a run does not depend on the page being open, and
  neither do the queue and the stop switch.

## Fleet: several vast.ai instances
The **Fleet** tab (on your own computer, not on an instance) shows every vast.ai instance you have running on one
page, over ssh: it needs the `vastai` CLI with your API key and your ssh key on your vast.ai account. **Find
instances** asks vast.ai which are running. Each instance's card has its run, progress, loss, time left, cost so far,
the last likeness probe and a small chart.

**Fleet queue**: a run added here goes to the first instance with nothing running and nothing waiting, and an
instance that finishes takes the next one. The settings come from the Train tab, a config file or a past run; a
local dataset folder can be sent first. **Split a race over the instances** deals a race's configurations over the
instances, one part each; the parts are compared by best likeness under Past runs (a relay cannot be split).

**Automatic downloads**: logs, probe images, sample images, training maps, the best LoRA and every LoRA each have
their own rule: never, when the run ends, or while it runs. **Watches** raise an event when a number goes above or
below a value (loss, a probe score, free disk, ...). **Past runs** keeps every run the fleet has seen, also after its
instance is gone.

The WebUI does the looking while it runs. Closed, run `python scripts/vast/rulz_fleet.py daemon` instead, or nothing
moves the queue, downloads files or checks the watches. The same file is the command line for all of this
(`rulz_fleet.py --help`). Nothing here rents, stops or destroys an instance.

## Wizard
New to LoRA training? The **Wizard** tab sets up a character / person LoRA in six short pages: who you are training
(job name, trigger word, how much of the person counts), the model, your photos, the training, samples and likeness
checks, then the extras and the start. The basics are filled in for you (the identity recipe with LoRA rank 32,
adamw8bit at 1e-4, and recommendations from your GPUs and photos). **Advanced** at the top shows every setting on every
page. The last page asks whether you want experimental settings; each one has a *What is this?* button saying what it
does and what testing found so far. The wizard edits the same settings as the Train tab, so you can switch between the
two at any time; styles and concepts are set up on the Train tab.

## Quick start
1. **Datasets**: on the Dataset tab, *Create dataset*, open it and upload images with same-name `.txt` captions (or
   copy a folder on the machine running RulzTrainer); tick one or more for training. Optional trigger word is
   prepended.
2. **Model**: either a Hugging Face repo (Base / SFT, about 30 GB, stored under `models/`) or three single
   ComfyUI-style files: diffusion model, Qwen3.5-9B text encoder and Flux2 VAE.
3. Click **Start training** (the bar at the bottom of the Train and Monitor tabs). First run caches VAE latents and
   text embeddings; the text encoder is loaded once for caching and unloaded before the diffusion model is loaded.
   **Add to queue** instead starts the run after the current one (and any queued before it) is done; the Monitor's
   *Queue* section reorders, removes, pauses and resumes the waiting runs. Kill pauses the queue, Stop goes on to the
   next run. Keep the WebUI open: it is what starts the next run.
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
noisier gradients (cosine 0.93-0.99 to the exact ones); `bf16` stays exact. A configuration race can race the two
(*nvfp4_backward* in a row). *Attention kernel*: `auto` puts cuDNN's attention first on RTX 40 (Ada) and Blackwell GPUs
(RTX 50, B200), where it ran steps 5-22% faster at the same loss, and lets PyTorch pick elsewhere; `torch`, `cudnn`, `flash`, `efficient`, `math` put one first. The others stay enabled behind
it: a call it cannot take runs on the next.

## Datasets
The Dataset tab lists every dataset with a few thumbnails: the folders in `datasets/` (where *Create dataset* puts
new ones) and folders elsewhere added with *Add a folder*. Click one to open it. *Upload* sends images and their
`.txt` or `.json` captions from the computer the browser runs on to the machine running RulzTrainer (on the same
machine that is a plain copy; a file with the same name is replaced). *Copy from a folder* copies the images and
captions of a folder on that machine (sub-folders kept, files the dataset already has left alone). Each picture has
its caption and the Face ref / Body ref ticks, and *Train on this dataset* ticks it for training.
On the Train tab (Dataset section) tick the datasets a run trains on. Each can have its own resolutions (none
picked: the Training resolutions) and *Repeats*: how many times each of its images comes up per epoch, so a dataset
at 2 counts twice as much as one at 1. Each dataset caches its latents in its own `.rulz_cache`, so it trains again
without re-encoding whatever it is combined with. A config file lists them as `datasets:` entries (`path`,
`resolutions`, `repeats`); a single `dataset_dir` still works.

## Training resolutions
*Training resolutions* takes one or more megapixel values (1 MP = 1024x1024); each image is bucketed to every
selected area with its aspect ratio kept and latents are cached per resolution. Several values are mixed in every
epoch, or, with *Resolution step-up* (next to the resolutions), trained lowest first: when the probe likeness score
stops improving for *N probes* the next resolution starts, and at the highest resolution *N probes* without
improvement end training (two separate counts, default 1 each). Step-up needs probes (face/body references).

## Training states
*Save training state* (LoRA, Advanced; on by default) also writes `state_XXXXXX/` folders (LoRA in fp32 + optimizer +
scheduler + RNG + likeness controller). Point *Resume from training state folder* at one of them to continue where it
stopped, with the best score, patience and likeness scale it had. A run that fails on an error saves its LoRA and a
state first; a failed sample or probe is logged and training goes on.

*Out-of-memory retries* (default 3, 0 = off): a run that ends because GPU or system memory ran out is started again by
itself, before the next queued run. It carries on from its newest training state on the same settings first; if it
runs out again at about the same point, one step lighter (gradient checkpointing on, then no blocks keeping their
activations, then more block swap). Batch size, resolution and precision are never changed: when nothing lighter is
left the log says what to change. The failed try's logs are kept as train_oom1.log and console_oom1.log.

## Unequal GPUs
*Data parallel balance* (Model, next to the GPU settings): in data parallel every step waits for the slowest GPU.
auto (default) times each GPU during the first steps and splits a step's images by speed when that is at least 5%
faster per image, keeping the images per step the same (it needs several per step: gradient accumulation 3 or
more). grow lets the faster GPU train more images in the slower one's time, so each step holds more images: lower
the steps to match. The log says what was measured and chosen.

## Block swap
*Block swap* (Train tab, memory section; single GPU) keeps the frozen weights of that many DiT blocks in system RAM and
streams each block's weights to the GPU just before it runs, the next one copied in while the current one computes.
Measured on Krea 2 int8 with 14 of 28 blocks swapped: the training step's peak went from 10.8 to 7.7 GB at about the
same step time. It needs gradient checkpointing and turns torch.compile off, unless *Compile with block swap*
(experimental, untested) is on: then the swapped blocks compile their forward only and the swap runs outside it.

*Activation offloading* (next to gradient checkpointing; experimental, untested) sends each checkpointed block's
saved inputs to RAM during the forward and streams them back just before that block's recomputation: about 0.8 GB
less on H3 at 1 MP (estimate), for more kept blocks or less block swap, at the cost of PCIe traffic. Compare the step
time and peak memory with a run without it. The log ends every run with its peak RAM.

## Compile
*Compile model* (Model, Advanced; default auto): torch.compile fuses the small steps between the matrix multiplies
into fewer GPU kernels. Measured at 1 MP on an RTX 5060 Ti: MOSS-Image 26% and Krea 2 12% faster per step, after a
warm-up of under a minute. auto compiles runs of 100 steps or more (triton is installed with the requirements).
*Compile cache* (next to it; default auto): the compiled kernels are also saved to one file in
`models/.rulz_compile`, so a new machine with the same GPU model, torch, triton, Python and operating system skips
most of the compiling (Qwen-Image 2.1, to step 5: 82 -> 32 s on Windows, 52-58 -> 30-37 s on Linux). On vast.ai,
upload a file a rental made (its name ends in -linux.bin) to your model bucket's `.rulz_compile` folder; one from a
Windows PC does not help there.

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

## Optimizers
`adamw8bit` is the default (AdamW with 8-bit state); `adamw` keeps full-precision state and `adafactor` a smaller
one (the fine-tune choice). `automagic3` and `automagic4` tune the learning rate themselves from the weights' update
directions (the rate you set is the start; schedule, warm-up, LoRA+ and the per-image rate do not apply; *Automagic
lr groups* `by_type` gives the text-side, attention, feed-forward and other layers their own rate). Automagic 4 also
takes a smaller step for the most sensitive weights (*Sensitivity quantile*) and its rate holds on a noisy loss
instead of drifting up: ai-toolkit's v4 raised the rate 1.6% a step on gradient noise (doubling every 44 to 100
steps) until the LoRA turned to noise, which this port fixes (*Vote* graded, *Max lr change* 5% a step). On a real
Krea 2 LoRA upstream's rate ran from 1e-4 into the 1e-3 ceiling by step 100 and stayed there; this port's peaked at
6e-4 and eased back to 3e-4 by step 160. On a rank-16 LoRA its step is as fast as adamw8bit's (about 3x slower at
rank 64, 6x on a full fine-tune; Automagic 3's is about 10x slower than Automagic 4's) and its state is 1 byte per weight at the default *Polarity history* 8; which optimizer reaches likeness first is not
measured yet, so race them.

`muon` steps each weight matrix as a whole: its momentum is replaced by the nearest orthogonal matrix, so every
direction of the update gets about the same step and fine detail does not wait behind the broad shapes. Its step size
is matched to AdamW's, so AdamW's learning rates apply as they are (1e-4 for a LoRA), and the schedule, warm-up, LoRA+
and the per-image rate work as with AdamW. Fizgig trains every picture model's LoRAs on it since November 2026 (level
with AdamW on likeness in its test, sharper detail by eye); it is not raced here yet. One full-precision number of
state per weight. With *lora_targets* `embed`, `modulation` or `text_in` it warns: Muon's authors and OneTrainer keep
those layers on AdamW. `normuon` (experimental, never run in a real training here) is Muon with each row's step
evened out by a running mean of its size (NorMuon, OneTrainer's default Muon): race it before relying on it.

`keel` is Automagic 4 for character LoRAs (`docs/keel-optimizer.md`), with every Automagic 4 setting and two of
its own: the *Gauge preconditioner* scales each LoRA pair's gradient by the other factor's r x r Gram (Riemannian
preconditioned LoRA), so the update does not depend on how the delta is split between `lora_down` and `lora_up`
and neither factor takes it all (*Gauge eps* is its floor); the *Cautious mask* drops a preconditioned gradient
element that points against the raw gradient. LoKr and OFT have no pairs, so there Keel is Automagic 4. *Paired
sigma* gives all of a step's micro-batches one noise level, and the *Noise bands* keep, per weight tensor, the
gradient power seen in each band of the noise level, so a step's gradient enters at the tensor's overall scale
whatever its band and the quiet bands (where the face forms) take steps as large as the loud ones; *Band weights*
then emphasise bands on purpose (`identity`: a hump over sigma 0.12 to 0.5). The *Consensus gate* scales each
weight's update by how many of the step's micro-batches agree on its sign, so what every photo shares moves and
what varies between photos moves less (it needs gradient accumulation of 2 or more). *Lookahead* (off by default)
anchors the weights at each probe that sets a new best likeness and, after the grace probes without a gain, pulls
the weights part of the way back toward the anchor and cuts the learning rate. *Face gain* (off by default) scales
each LoRA pair's step by how much of its gradient comes from the face's tokens, against the median pair, so the
weights that serve the face move more and the background's less; it needs region maps with faces. *Consensus vote*
picks how the step's gradients vote: by sign (one vote each) or by magnitude (each weighted by its size, so a photo
that says little about a weight barely votes). Whether Keel beats adamw8bit on the likeness probes is not measured
yet: race them. Tested so far (`docs/keel-test-results.md`, one small body-only set, 8 seeds): one rate for every layer
beat *by_type* by 4.4 body points, *Max lr* at 1e-3, 5e-4 and 3e-4 tied at 1000 steps, and every run was still rising
at the end. Use one rate and plan on more than 1000 steps.

`keel_fb` is Keel for the face and the body alike, for a LoRA that should hold the build as firmly as the face: the
token gain measures the body's tokens too (the person outside the face box, from the person masks; *Body weight*
sets their weight against the face's, each region against its own median pair) and the consensus gate votes by
magnitude, since under the sign vote a face crop's tiny, random-signed gradient for a body-serving weight counts as
a full vote against it; `subject` *Band weights* are its plateau over the face's and the body's noise levels. The gain
is off by default in both, so with the defaults `keel_fb` differs from `keel` by the vote alone: set *Face gain* (0.5
or 1) to bring the body share in. Set the probes' face and body weights (`face_weight`, `body_weight`, 0.7 / 0.3 by
default) equal with it, since the lookahead and the best / stop decisions follow their combined score. Race it against
`keel`, with gradient accumulation of 2 or more: not measured yet.

## Adapters: LoRA, LoKr, OFT and DoRA
The *Adapter* field (Train tab, LoRA section) picks what each layer learns. **LoRA**: a rank-r pair. **LoKr**: a
Kronecker pair (factor 8 is Fizgig's best measured likeness). **OFT**: rotates each layer's output channels in blocks
(*OFT block size*, 32) instead of adding to the weights, which keeps the base model's structure; the OFT paper found
better subject identity than LoRA and stable long runs. *OFT constraint* > 0 caps how far a layer rotates.
*Scaled OFT lr* (experimental, untested) gives each block size its own learning rate so every size moves about as far
per step as a block of 32 (block 32 itself unchanged). **DoRA**
(a switch for LoRA and LoKr): a magnitude per input channel trains on its own while the adapter steers the
direction; steps are about 30 to 40% slower on Krea 2. OFT and DoRA files load in ComfyUI exactly as trained (not in diffusers). To
compare them, race rows such as `network_type=oft` or `dora=true` against a LoKr row.

## Trigger attention and identity alignment
Two training-only losses (Train tab, Region-weighted loss section; the Wizard's experimental extras). **Keep
the trigger's attention on the subject** reads, inside the model, where the trigger word's attention lands in each
picture and pulls it onto the person (their silhouette and face box), so the trigger is bound to them and not to the
background or other people; it needs a trigger word. **Align the face tokens with the photo's identity** teaches the
face's tokens after one block to carry that photo's identity (its ArcFace embedding) through a small helper network
that is thrown away. The LoRA file is the same either way. On one small Krea 2 set trigger attention sped up the start with little late gain, and identity alignment showed no gain.

## Likeness reward fine-tuning (second stage)
*Reward steps* (Likeness probes section, *Second stage*; or the Wizard's experimental extras) adds a stage after the steps:
the LoRA renders groups of 4 pictures of new prompts, the probes' scorers rate their likeness, and the better ones are
made more likely and the worse ones less (Flow-GRPO), held near the first stage's LoRA. It aims straight at likeness
but is slow and can learn what the scorer likes more than what you see: keep the probes on and look at the pictures.
The probes also score the stage's final LoRA when it ends, so its likeness shows next to the first stage's. Each
picture's face and body scores are capped far below a stranger's: a body that differs from your body references only
in clothing (a coat against unclothed references) can't outweigh the face. Reward prompts like your body references,
or Body weight 0, let the body count.
`<job>.safetensors` stays the first stage's; the stage saves `<job>_reward.safetensors`. Steps 0 with a LoRA in
*Resume from* runs the stage alone.

## Probe noise and fair race drops
A probe renders a few seeds and averages them, so part of every score is the seeds' luck. The **Probe noise check**
section (Train tab) measures it: give it two finished LoRAs of one run (a middle checkpoint and the best), switch it
on and press Start. It trains nothing; it renders both on many seeds, the same ones for each, and writes
`noise/noise.json` with how far one seed strays, how many probe seeds it takes to tell a 3, 5 or 8 point gap apart,
and a noise band that fits your probe settings. In a race, *Drop only when the probe seeds agree* (Configuration race)
drops the lowest racer only when its scores, seed by seed, sit below the leader's with the confidence you set (0.9 is a
good start); a round too close to call drops nobody, at most a few rounds in a row. Use 2 or more probe seeds with it.

## Gold face scorer (AdaFace)
The probes judge the face with ArcFace. The reward stage, the identity loss on its arcface backend and identity
alignment train against that same ArcFace, so they can raise its score without the face looking more like the person.
*Gold face scorer* (Likeness probes section) adds a second recognizer, AdaFace (FaceNet before 3.14.0, still a
choice), that nothing trains on: it scores the
same faces against the same references on its own scale, and (*Best checkpoint, early stop and race follow it*, on by
default) its score replaces ArcFace's in the decisions. *auto* turns it on for those runs and for the probe noise
check. The Monitor draws it as a dashed "face (AdaFace)" line; ArcFace pulling away from it is the warning sign, and the
reward stage logs a warning when ArcFace runs more than 10 points further ahead than when the stage began.

## Probes on the second GPU
With one GPU training and the other idle (Multi-GPU off, or block swap, which turns it off), *Render probes on the
second GPU* (Likeness probes section) starts a second process with a copy of the model there that renders the probes
while training goes on. Training pauses only to send the LoRA across (a fraction of a second) instead of for the whole
probe; the final probe renders on the training GPU. The probe's decisions (best checkpoint, early stop, lr control,
curriculum) land when it is scored, up to *Probe every* steps later; a best checkpoint still saves exactly the weights
that were scored. It pays off when one probe on the second GPU takes less time than *Probe every* training steps
(measured 10-24% faster on Qwen 2.1); with large probes every few steps training waits for them and it is slower. The
log shows each probe's time as "on cuda:0, in at step N", and the end of training how long training waited. Not used
in a configuration race or a fine-tune; if the second GPU runs out of memory, probes go back to the training GPU.

## Near-duplicate photos
Before training, unless *Copy check* is off, SSCD compares your training photos with each other (it is about 99 MB,
downloaded once) and names any two that match above *Copy limit*. Such a pair is nearly one picture shown twice, so it
is trained about twice as often as one photo and is the likeliest to be memorised. The Monitor shows the pairs, and
the log lists them. It only warns: to act on it, remove one photo of each pair and start again.

## Copy check (SSCD)
A LoRA trained on a few photos can learn to redraw one of them: same pose, same outfit, same background. *Copy check*
(Likeness-filtered self-training section) compares pictures with your training photos using SSCD, a copy detector
(a cosine of about 0 for an unrelated picture, above 0.9 for a flipped copy). Self-training leaves out a picture above
*Copy limit* (0.5, raised to the photos' own typical match among themselves + 0.1 when they already look alike, as in
one shoot in one room), shown as "copies a photo" in the round's log and in `selftrain.json`. Each probe logs its
closest training photo and score ("copy" in the Monitor's probe line, `copy` in `likeness.jsonl`); that decides
nothing. *auto* turns it on with self-training, *on* for any run with probes.

## Probe prompt presets and the prompt check
An overtrained LoRA stops following the prompt's framing: a full-body prompt comes back as a portrait, and the
likeness scores do not notice. The **prompt check** reads every probe render's frame (a close face, waist-up, or the
body down to the hips) from the face's size and the pose, and compares it with the step-0 probe of the same seed: a
render that showed its prompt's frame then and does not now has *drifted*. The log has a "Prompt check" line per
probe and the Monitor shows "on-prompt" beside the scores. *Face / Body probe should show* (under the probe prompts)
says what each probe asks for; empty guesses it from the prompt's words. The **preset** lists beside them write a
ready-made prompt with your trigger word and *Subject* into the field. *Extra prompts to check* renders more presets
at every probe without scoring them, to watch framings the two likeness prompts do not cover. It only reports unless
you set *No new best under this share on-prompt* (a drifted probe cannot set a best checkpoint) or, in a race, *Drop
one that leaves the prompt* (both experimental, unit-tested only). It sees framing and the number of faces, not
clothing or setting.

## Training maps
After Fizgig's training maps. With *Training maps* on (Likeness section), a few training pictures are checked before
training and about once an epoch at three fixed noise levels with fixed noise, so only the LoRA changes between
captures. The **Training maps** tab draws them over each picture: *Learned since the start* (green better, magenta
worse; read it first), *Not learned yet* (red misses most), *Where the LoRA acts* (should sit on the subject; on a
recurring background or prop the LoRA learns that too) and *Trigger word* (should sit on the subject, not on a logo,
watermark or object; coloured from the quietest area up). *Learned, measured as* shows the first map as the amount
of error gone, or as a share of each part's own starting error (how far along each part is, whatever it started at),
or as that share on one scale for every run (full colour = half gone), to compare two runs. They only observe, a few seconds per capture; not in a configuration race.

## Previous runs (run history)

The **Previous Runs** tab keeps the likeness history of every run, like TensorBoard, so runs can be compared after the
job folder has been reused. 
A run with likeness scores writes one small file in `output/_history/` while it trains (a
crash keeps what was saved) and again when it ends. It keeps what the Monitor's two likeness charts plot, on the run's
normalized scale: the **mean face and mean body** score of the sample images at each sampling step (Identity
similarity), and the smoothed **face, body and combined** score at each probe (Likeness probes & controller). Runs
from before the history existed are imported from their job folders when the tab first loads, or with *Import old runs*.

Tick any number of runs in the table (name, start, state, steps, adapter, learning rate, last and best scores) to
overlay them: one colour per run (one legend entry each), solid lines for the face, dashed for the body and thick for
the combined score, with chips to show only some of the three. *Rename* gives a run a name of your own and *Remove from history* deletes
its history file only (never the job folder). A run that is still training updates live.

*Score sample folders* compares finished runs from any trainer (ai-toolkit, Fizgig, OneTrainer, this one) on the
probe scores, in one spreadsheet. List each run as `name = folder` (its sample pictures, sub-folders included): a
file or folder name with "face" marks face pictures, "body" body pictures, and the step comes from the file name
("step000200", "200.png"). The baseline folder holds the base model's renders of the same prompts without a LoRA,
the zero of the face score and of the anchored body score (4 to 8 body seeds). The spreadsheet (also saved in
`output/_compare/`) has each run's late checkpoints, its best one, charts, every raw score and the scale as formulas
you can edit.

**Probe images** (below the charts) puts the ticked runs' likeness probes side by side: pick Face or Body and a probe
step (the arrows step through them). The images are grouped by test picture (*Face probe 1*...), every run, or each
configuration of a race, side by side under it with its face / body / combined scores. A run probed on another interval
shows its latest probe before that step, marked. Click an image to see it full size (the arrows step through the shown
images); tick the box in the corner of any two (any run, any step) to open them side by side, large, with their scores
(*Compare* opens them again, *Clear* starts over). When the ticked runs
used different probe settings (prompts, seed, size, steps, guidance, turbo LoRA or resolutions) a note says which:
their probes are not the same test then. The images are kept with the run, as small JPEG copies in
`output/_history/<run id>/`, so a later run with the same job name does not replace them; *Remove from history* deletes
them too. Runs saved before v2.4.0 get theirs from *Import old runs* while their job folder still holds that same run.

## Identity training (character LoRAs)
The **Identity training preset** switch (top of the Train tab, or *Character / person* in the Wizard) applies the
measured recipe below in one go and restores your previous settings when switched off; the CLI equivalent is
`--preset identity`.
In the Wizard, *Character / person* then asks how much of the person the likeness covers, and sets the loss weights
(face region / person region), the score weights, the crops and the probes for it:

| Focus | Loss face / person | Score face / body | Crops | Probes |
|---|---|---|---|---|
| Face only | 3 / 1 | 100% / 0 | face | face only (no body references) |
| Face focus (the recipe) | 3 / 1.5 | 70% / 30% | face + bust | both |
| Face and body | 1.5 / 1.5 | 50% / 50% | face + bust | both |
| Body focus | 1.5 / 3 | 30% / 70% | face + bust | both |
| Body only | 1 / 3 | 0 / 100% | none | body only (no face references) |

The curriculum is on for the three in the middle. The wizard also recommends the adapter (a LoRA for every model),
block swap for a card the model does not fit, LoRA Isolation, the look warm-up and how many checkpoints to keep.
Measured on a small test set (120 steps, same seeds): region-weighted loss gave the best face likeness (+2 points
over baseline), LoRA+ at ratio 4 a small gain (ratio 16 on a 2e-4 lr was unstable), `identity` timesteps were
neutral, and auto crops *hurt* when the crops had to be upscaled. Suggested recipe: *Region-weighted loss* on
(face 3 / person 1.5 / background 0.3), *LoRA+ ratio* 4 with `adamw8bit`, *Auto crops* only when your photos are
high-resolution (crops are never upscaled by default), timesteps left at `logit_normal`.
LoRA rules of thumb: LoRA lr is ~10x a full fine-tune lr and nearly rank-independent; keep MLP layers in the
targets; small batches with more steps beat large batches. *Dropped captions keep the trigger word* (Dataset
section, default on): caption dropout trains on the trigger alone instead of the empty prompt, so the trigger by
itself summons the subject.

**LoKr** (*Adapter* in the LoRA section): instead of a rank-r pair, each layer learns a Kronecker pair, a small w1
(up to factor x factor) and w2 holding the rest, so the update tiles the weight with structure rather than a thin
slice. Fizgig's measured default is factor 8 (its highest likeness and more natural skin on the same data; ~20%
slower steps; about 400 MB on a 12B model). Rank, alpha, LoRA+ and dropout don't apply. The file is LyCORIS
format: ComfyUI's LoRA loader takes it for MOSS-Image and Krea 2; diffusers cannot load LoKr. A race can pit LoRA
against LoKr: `network_type=lokr, lokr_factor=8` in one row.

**Probe scores.** The face score is ArcFace's best match against the references (a stranger ~10-25, the references
against each other ~60-80). The body score (*Body score*, Identity similarity section) is by default a DINOv2
embedding of the body from the neck to mid-thigh, turned upright, cut out from the background and without the head
(the face score covers that), best match against the reference photos that show the same part of the body
(three-quarter or full-length shots; close-ups are left out). Build, skin, hair and clothing count, so vary the
outfits among the references. *anchored* takes the same window with its colour grade, exposure and sharpness
normalised away, embeds it with SigLIP 2 and DINOv2, and scores how far it has moved from the base model's people
(the step-0 probe's body renders, 8 seeds) toward the references: 0 = the base model, 100 = the references' mean.
On a five-run comparison it told trained samples from the base model's in 96% of pairs, where the best match ranked
the person's own photos below the base model's; a probe in one outfit and scene stays well below 100, so compare
runs with each other. `proportions` is the older skeleton-ratio match (every human ~80, the references
~90-96), which shifts with pose and camera angle. *proportions 3D* measures bone-length ratios in 3D (the same bent
or straight) plus waist, hip and thigh widths where the person stands facing the camera; *SMPL-X body shape* fits
the SMPL-X body model with NLF and measures it in its T-pose (bust, waist, hip and thigh girths, limb lengths, over
the height). Neither counts clothing or skin. SMPL-X needs NLF's file, downloaded by hand (noncommercial license):
nlf_l_multi_0.3.2.patch6.torchscript from github.com/isarandi/nlf/releases into models/nlf/. With *Normalize face and body scores* (default on) each is first
mapped to 0 = a stranger's score (the step-0 probe, rendered with *Step-0 seeds* seeds, so keep *Probe at start*
on) and 100 = what a typical reference scores against the others (the median of each one's best match; faces cut
off by the frame left out), capped at 100: matching the references better than a typical one of them is no more
likeness. The combined score, best checkpoint, early stop, curriculum and race decisions use these; the Monitor
and Samples tab show only these (probe images carry their overall / face / body score), and in a race the same
generation of every configuration sits side by side in one row.

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
the race. With data parallel on 2+ GPUs, *One configuration per GPU at the same time* trains the configurations side
by side (an odd one left over splits over the GPUs, or runs on the first). The Monitor tab shows each configuration's score per round, with a *Drop* button on each one still in: it stops
that configuration after the current step, saves its LoRA as `_dropped` and the race goes on without it (one left wins).

*Drop each one when it stops improving* drops a configuration once its own probes have not improved for *Probes without
improvement* in a row, whatever the others score. *Keep one whose likeness is rising again* gives it a chance after a
dip: out of patience, it stays as long as each probe's smoothed likeness is higher than its previous probe's, and
leaves at the first one that is not (beating its best resets the count). The Monitor and log show who stayed.

*Relay: everyone continues from the round's winner* drops nobody: after each round the others take over the round
winner's LoRA, optimizer state, weight average and likeness history, and train the next round with their own settings,
seed, image order and noise. *Add seed racers* (seeds such as `1-4`) adds one `seed=` row per seed and turns it on: the
best of several seeds at every probe instead of separate runs. Every racer trains all the steps (N racers, N runs'
steps); the last round's winner is the run's LoRA, the others are `_final` files, and `<job>_best` is the best any of
them reached. Rows may not change the LoRA's shape (rank, alpha, layers, adapter); not with one configuration per GPU.

*Pick the leader yourself after each round* (relay only) lets you choose that winner. At the end of each round the run
waits; look at the round's probe images (Monitor > Images, each racer's carry its name), then press *Continue from* on
the racer everyone should go on from in the Monitor's race table. In the last round your pick is the run's LoRA. With
no pick within *Wait for the pick* minutes (0 = no limit) the highest likeness score leads as usual, and so does a Stop
while it waits. The GPUs sit idle while it waits, which a rented GPU bills for.

## LoRA Isolation
*LoRA Isolation* (Train tab) keeps your subject(s) off everyone else in a picture, after Fizgig's: every caption that
names a subject also trains without its trigger, held to the model without the LoRA (DOP); the trigger is joined to its
class word (`gwelt, a man` trains as `gweltman`: prompt with that word, the sample and probe prompts are rewritten);
each person gets side-by-side pictures of one of their photos next to a generated stranger, captioned stranger first;
and several subjects, one per dataset with its own *Trigger* on the Dataset rows, balance. *Is a* sets what a subject
is when its captions don't say; *Kind* makes it a person (fused trigger + pairs), an animal or thing (fused trigger) or
a style (trigger kept, no pairs). Fizgig measured on Krea 2: strangers reading as the subject 48 of 50 -> 0 of 50,
likeness unchanged. Training takes about 1.5-2x as long.

## Full fine-tune, then extract a LoRA
*Fine-tune the whole model, then extract a LoRA* (Train tab, LoRA section) trains the target layers' own weights
instead of an adapter, after Fizgig's fine-tune: the model stays as loaded (int8 / nvfp4 / bf16), the layers are
grouped into windows of what the card holds, and one window at a time is held in fp32 and trained for *Epochs per
window*, then written back as a bf16 delta in RAM and re-quantized; a cycle trains every window once (the log lists
the plan, the run strip shows the window). Every save extracts a LoRA of the rank(s) you set from the deltas, at the
end of a cycle, so each saved file has every layer trained alike; samples and probes render the fine-tuned model
itself. *Also save the deltas* writes `<job>_finetune.safetensors` (2 bytes per weight), to continue from (*Continue
from deltas*) or to extract at other ranks later with `python -m rulztrainer.extract_finetune`. Fizgig measured: a LoRA
extracted from a fine-tune beat one trained at the same rank, and rank 64 matched the full checkpoint. The switch sets
the learning rate to 1e-5 with adafactor: the weights themselves move, so a LoRA's rate overcooks. Not with a race,
data parallel, torch.compile, the weight average, training states or the region prior.

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

## Qwen-Image 2.1

Pick **Qwen-Image-2.1** as the model. By default it comes as Comfy-Org's single files (Comfy-Org/Qwen-Image-2.1,
int8 ConvRot): the 7B diffusion model (7.3 GB), the Qwen3-VL-8B text encoder (9.4 GB, used for caching and then
unloaded) and its RGBA VAE, so one 16 GB card trains it. Training images are bucketed on a 32 px grid (the model's
image tokens come in 2x2 groups).

The **training adapter** (LoRA, *Training adapter*, `auto`) is Fizgig's: a small LoRA that stays on while your LoRA
trains and off for samples and probes, and is never saved into your file. Fizgig measured that Qwen-Image 2.1 LoRAs
trained without one collapse into texture or wobble part-way, with nothing in the loss to warn you (likeness 77 with
it, 56 without). Fizgig's recipe: 0.5 MP, rank 8, learning rate 2e-4 (rank 16 for big or mixed datasets at half the
rate), EMA 0.98; the setup wizard suggests it.

Samples render with Comfy-Org's **Turbo LoRA** (8 steps, CFG 1, the Turbo checkpoint's own schedule); without it,
25 steps at CFG 1 (the model is meant to be sampled without guidance). LoRAs are saved in the diffusers / PEFT
format; ComfyUI's LoRA loader maps them (the feed-forward's gate and up halves onto its fused layer), and so does
diffusers:

```python
from diffusers import QwenImage21Pipeline
pipe = QwenImage21Pipeline.from_pretrained("Qwen/Qwen-Image-2.1-Turbo", torch_dtype=torch.bfloat16).to("cuda")
pipe.load_lora_weights("output/<job>/<job>.safetensors")
image = pipe("<trigger>, ...").images[0]
```

Only text-to-image LoRAs train for now (no edit LoRAs from before / after pairs). Qwen-Image 2.1 is under the Qwen
Research License: non-commercial use unless Qwen grants more, and a shared LoRA must say "Built with Qwen".

## FLUX.2 Klein 9B

Pick **FLUX-2-Klein-9B-Base** as the model (the 4-step distilled **FLUX-2-Klein-9B** is for samples). This trains
text-to-image LoRAs; the model's editing mode is not trained here. Black Forest Labs' repos are gated with automatic
approval: accept the FLUX Non-Commercial License on huggingface.co/black-forest-labs/FLUX.2-klein-base-9b-fp8 once and
log in (hf auth login). The single files are BFL's fp8 diffusion model (8.9 GB; bf16 is 17 GB) with
Comfy-Org/Flux-2-Klein-9B's Qwen3-8B text encoder and Flux2 VAE, which are not gated. There is no int8 release: the file
is read up to bf16 and Auto quantizes it for your GPU memory, nvfp4 below 18 GB, int8 from 18 GB, bf16 from 30 GB. The
latents are the ones MOSS-Image, FIBO 2.0 and FLUX 3 Klein cache, so a dataset cached for those is reused.

The wizard suggests Fizgig's full-model recipe: rank 16, alpha 16, learning rate 1e-4, and its flux2_shift noise levels
(Timestep sampling: reference). All 32 blocks are trainable: in Blocks, 0-7 are the double blocks and 8-31 the single
blocks. Fizgig finds style and composition mostly in blocks 0-9 and identity in 9-24. q, k and v are one fused layer, and
a single block fuses its attention and feed-forward projections, so those target groups overlap there.

Previews on Base use vafipas663's turbo LoRA (extracted from the distilled model, 1.2 GB) at 4 steps, CFG 1, strength 1.
Its author reports static noise above strength 0.5, sometimes needing as low as 0.1: lower it if previews are noisy. Turn it off
for Base's own 20 steps at CFG 3.5, or set the distilled black-forest-labs/FLUX.2-klein-9B as the sampling model, which
needs no turbo LoRA (4 steps, CFG 1). LoRAs are saved under BFL's module
names (`diffusion_model.<module>.lora_A / lora_B`, the alpha folded in), the layout ai-toolkit writes, which ComfyUI's
Load LoRA node reads for Base and the distilled model and diffusers' Flux2KleinPipeline converts. The FLUX
Non-Commercial License covers non-commercial use: read it before you share or sell a LoRA.

## FLUX 3 Klein

Pick **FLUX-3-Klein-Base** as the model (the 16-step distilled **FLUX-3-Klein** samples only). Black Forest Labs'
repos are gated with automatic approval: accept the FLUX Kommunity License on
huggingface.co/black-forest-labs/flux-3-image-klein-base once and log in (hf auth login); the single files come from
BFL's own black-forest-labs/flux-3-image-klein-comfy (int8 ConvRot by default: the 5.5B diffusion model, 5.8 GB, the
Qwen3-VL-4B text encoder, 4.9 GB, and the Flux2 VAE). One 16 GB card trains it in int8 or in bf16 (9.7 GB; a rank-32
step at 1 MP peaked at 12.2 GB). The Flux2 VAE latents are the ones MOSS-Image and FIBO 2.0 cache, so a dataset
cached for those is reused.

BFL's LoRA recipe, which the setup wizard suggests: about 20 images at 1 MP with a trigger word, rank 32 / alpha 32,
learning rate 1e-4 with 200 warm-up steps then constant, 4000 steps, a checkpoint every 250. Captions should be full
scene captions (framing, subject, background, lighting): the model is meant to be prompted through BFL's prompt
upsampler, which the trainer does not run.

There is no turbo LoRA: Base samples 50 steps at CFG 3.5 (slow previews). For quick previews set the distilled
black-forest-labs/flux-3-image-klein as the sampling model and sample 16 steps at CFG 1. LoRAs are saved under BFL's
module names (`diffusion_model.<module>.lora_A / lora_B`, the alpha folded in), which ComfyUI's Load LoRA node reads
for both Base and the distilled model, and which diffusers' modular pipeline loads:

```python
import torch
from diffusers import ModularPipeline
pipe = ModularPipeline.from_pretrained("black-forest-labs/flux-3-image-klein")  # or -klein-base: 50 steps, CFG 3.5
pipe.load_components(dtype=torch.bfloat16)
pipe.to("cuda")
pipe.load_lora_weights("output/<job>/<job>.safetensors")
image = pipe(prompt="<trigger>, <a full scene caption>", output="images")[0]
```

BFL suggests a LoRA strength of about 0.75 on the distilled model. The FLUX Kommunity License allows non-commercial
use, plus commercial outputs for companies under US$5M revenue; read it before you share or sell a LoRA.

## Ideogram 4

Pick **Ideogram-4** as the model. By default it comes as Comfy-Org's single files: the 9.3B diffusion model from
Comfy-Org/Ideogram-4 in int8 ConvRot (9.6 GB; fp8_scaled 9.3 GB and nvfp4_mixed 5.5 GB are the other formats), the
Flux2 VAE, and the Qwen3-VL-8B text encoder from Comfy-Org/Qwen-Image-2.1 (int8 ConvRot 9.4 GB: Ideogram 4 uses the
same model, used for caching and then unloaded), so one 16 GB card trains it. There is no bf16 release: the
Hugging Face source (ideogram-ai/ideogram-4-fp8, gated, or its unsloth/ideogram-4-fp8 mirror) is fp8 and is
dequantized on load, then quantized for your GPU. Buckets are on the 16 px grid.

**Captions**: Ideogram 4 was trained on structured JSON captions only (Ideogram's prompting guide has the schema). A
same-name `.json` caption is used as it is and trains best; a plain `.txt` caption is wrapped into the smallest valid
JSON caption, and the trigger word goes into its subject (object) element. Set *Max caption tokens* to 2048 for long
JSON captions. Sample prompts can be plain sentences (wrapped the same way) or JSON captions.

Samples render with ostris' community **TurboTime LoRA** (8 steps, CFG 1). Without it, Ideogram's V4_DEFAULT_20
preset: 20 steps at CFG 7 with the last 2 at 3. Ideogram's pipeline guides with a second, separately trained
*unconditional* model on the image alone; the trainer has one model, so its guidance pass runs the image without
text on the model you train: close to, not the same as, the official pipeline's samples.

LoRAs are saved with the checkpoint's own module names (`diffusion_model.layers.N.attention.qkv`, as ai-toolkit's):
ComfyUI loads them with *LoraLoaderModelOnly* on the conditional model (ideogram4_*, not ideogram4_unconditional_*),
and diffusers converts them itself:

```python
from diffusers import Ideogram4Pipeline
pipe = Ideogram4Pipeline.from_pretrained("ideogram-ai/ideogram-4-nf4", torch_dtype=torch.bfloat16).to("cuda")
pipe.load_lora_weights("output/<job>/<job>.safetensors")  # needs only the block layers (the default targets)
image = pipe(json_caption, num_inference_steps=20, guidance_schedule=[7.0] * 18 + [3.0] * 2, std=1.75).images[0]
```

Ideogram 4 is under the Ideogram Non-Commercial Model Agreement: non-commercial use only. A LoRA trained on it is a
Model Derivative under the same terms (share it with the agreement and its notice), and images made for
revenue-generating uses are not allowed either.

## MiniMax H3

MiniMax H3 is a 33B video + audio model. Pick **MiniMax-H3-FL2VA**: your pictures train as one-frame videos (one video
latent frame and a silent audio placeholder, the loss on the video alone), as musubi-tuner's one-frame mode does, so a
character or style LoRA trained on stills loads into H3 video generation (ComfyUI, diffusers, musubi-tuner). This is
experimental: how well a stills LoRA carries into motion is not known yet.

The single files come from Comfy-Org/MiniMax-H3: pick *pruned int8 ConvRot* (21.0 GB; the full int8 file is 34.0 GB
and is pruned on load anyway), the Qwen3-VL-32B text encoder cut to 50 layers (int8 27.1 GB or NVFP4 15.7 GB, used
for caching and then unloaded) and the video VAE (5.2 GB); the 0.6 GB audio VAE is fetched by itself for the silence
placeholder. Nothing fits one 16 GB card whole. Estimates, not measured: two 16 GB cards in **model split** (about
10.5 GB of weights each), or one card with **block swap** (about 25 of the 50 blocks in pinned system RAM; 64 GB of
RAM advised). The text encoder spreads over the GPUs and system RAM (*Text encoder device* auto).

H3 is CFG-distilled, so the **training adapter** is on by default: circlestone-labs' image training adapter, on while
your LoRA trains, off for samples and probes, never saved into the file (musubi-tuner saw one-frame LoRAs break up
within about 50 steps without an adapter). The setup wizard suggests the recipe musubi-tuner and DiffSynth-Studio use:
rank 16 (32 for styles), alpha = rank, learning rate 1e-4, adamw8bit, H3's own timesteps (*reference*: one base time,
shift 12 for video), 1 MP buckets on a 32 px grid, EMA. Samples render on the plain model with lightx2v's 8-step turbo
LoRA (CFG 1), or 30 steps without it.

LoRAs are saved as `diffusion_model.blocks.N.attn.qkv_proj.lora_A/B` without alpha (alpha = rank), the layout
ai-toolkit and diffusion-pipe write and ComfyUI's Load LoRA, musubi-tuner and diffusers read:

```python
from diffusers import ModularPipeline
pipe = ModularPipeline.from_pretrained("MiniMaxAI/MiniMax-H3")
pipe.load_components(dtype=torch.bfloat16)
pipe.load_lora_weights("output/<job>/<job>.safetensors")
```

MiniMax H3 Community License Agreement: users in the USA, EU, UK and South Korea must apply to MiniMax for a license
first (platform.minimax.io/h3-license).

## P-H3-Image

P-H3-Image is PrunaAI's MiniMax H3 fine-tune for still images: the same 33B transformer, made to render in 8 steps
with no guidance, with a small Qwen3-VL-4B text encoder and a VAE that decodes one frame. Pick **P-H3-Image**: it trains
text-to-image LoRAs (edit training with control images is not supported here) that load in ComfyUI on PrunaAI's files.
This is experimental: nothing has been measured on it yet.

The single files come from PrunaAI/P-H3-Image-Comfy: the transformer in *int8 ConvRot* (21.2 GB, already pruned, with
Pruna's LoRA merged in; bf16 40.5 GB), the text encoder with its text connector (int8 5.1 GB, bf16 9.1 GB) and the VAE
(5.6 GB). Memory is as for MiniMax H3: two 16 GB cards in **model split**, or one with **block swap**; the text encoder
fits one card.

The model is step- and guidance-distilled, so the **training adapter** is on by default: ostris' P-H3-Image adapter
(what ai-toolkit trains it with), on while your LoRA trains, off for samples and probes, never saved into the file.
Samples render on its own 8-step grid at CFG 1. It is tuned for 4-8 MP output; 1 MP previews show more artifacts than
the finished LoRA will at full size.

LoRAs are saved in H3's layout (`diffusion_model.blocks.N.attn.qkv_proj.lora_A/B`, alpha = rank), which ComfyUI's
Load LoRA reads on PrunaAI's P-H3-Image files.

P-H3-Image is under the MiniMax H3 Community License Agreement (its Qwen parts under Apache 2.0): users in the USA, EU,
UK and South Korea need a license from MiniMax first, and products built on it show "Powered by MiniMax H3".

## MiniMax-H3-Image

MiniMax-H3-Image is MiniMax's own image model: the H3 transformer (33B) without its audio stream, made to render in
6 steps with no guidance, at 4 to 20 megapixels (a token covers 32 x 32 px). Pick **MiniMax-H3-Image**: it trains
text-to-image LoRAs (edit training with reference images is not supported here) that load in ComfyUI on Comfy-Org's
files. This is experimental: nothing has been trained on it here yet.

The single files come from Comfy-Org/MiniMax-H3-Image: the transformer in *int8 ConvRot* (21.0 GB, already pruned),
*bf16* (40.2 GB) or *w4a8* (12.5 GB: ComfyUI's 4-bit file, read into the int8 weights it stands for, so it is a smaller
download that takes the same memory as int8), and the image VAE (1.2 GB). The text encoder is MiniMax H3's own
Qwen3-VL-32B file (int8 27.1 GB or NVFP4 15.7 GB), so a copy downloaded for H3 is used again. Memory is as for
MiniMax H3: two 16 GB cards in **model split**, or one with **block swap**.

The model has one output head for each of its 6 sampling steps, each made for the one noise level of its step.
*Reference* timesteps (the wizard's suggestion) train on exactly those 6 noise levels, picked evenly, each through
its own head, as ai-toolkit trains the model; timestep shift, the sigma range and clean share do not move them. The
other timestep choices train at noise levels in between, which sampling never uses. The model is step- and
guidance-distilled, so the **training adapter** is on by default: ostris' MiniMax-H3-Image adapter (0.16 GB,
downloaded), the one ai-toolkit trains it with. It is on while the LoRA trains and off for samples and probes.

Samples render its own 6 steps at CFG 1. MiniMax's pipeline renders anything under 4 MP on a 4 MP canvas and scales
it down; here a preview renders at the size you set, so set the sample resolution to 4 MP to judge the LoRA at the
model's real quality.

LoRAs are saved in H3's layout (`diffusion_model.blocks.N.attn.qkv_proj.lora_A/B`, alpha = rank), which ComfyUI's
Load LoRA reads on the MiniMax-H3-Image files.

MiniMax-H3-Image is under the MiniMax-H3-Image Community License Agreement: it excludes the USA, EU, UK and South
Korea unless MiniMax authorizes it (platform.minimax.io/license), which is also required above 1 million USD of
annual revenue or to offer the model as a hosted service. A LoRA is a Model Derivative: share it with a copy of the
agreement and its NOTICE file.
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

    app.on_startup(lambda: threading.Thread(target=queue_loop, name="rulz-queue", daemon=True).start())
    if VAST_INSTANCE is None:  # this computer looks after the vast.ai instances; an instance looks after itself
        app.on_startup(lambda: threading.Thread(target=fleet_loop, name="rulz-fleet", daemon=True).start())

    @ui.page("/", reconnect_timeout=PAGE_KEEP_SECONDS)
    def index(request: Request):
        # A real page function, not NiceGUI's script mode: script mode re-executes this file for every page load,
        # which reset the job globals, so a reload lost the running job.
        build_page(local=is_local_viewer(request))

    ui.run(host=a.host, port=port, title="RulzTrainer", show=a.open, reload=False, dark=True,
           favicon=ASSETS / "icon.png")
