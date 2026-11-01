"""RulzTrainer WebUI (NiceGUI). Run: python webui.py [--port 7870] [--open]

Training runs in a subprocess so Stop frees GPU memory cleanly; the UI polls
<output>/<job>/status.json and train.log. The running job (pid + job dir) is recorded in .webui_job.json, so a
reloaded page, a second browser or a restarted WebUI reattaches to it.
"""
from __future__ import annotations

import argparse
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

from nicegui import app, run, ui

from rulztrainer.archs import get_arch
from rulztrainer.config import (AUTOMAGIC, OPTIMIZERS, ARCH_DEFAULT_TARGETS, ARCH_SAMPLE_DEFAULTS, ARCH_TURBO, MEGAPIXEL_CHOICES,
                                MODEL_PRESETS,
                                PRESET_PROBE_PROMPTS, PRESETS, TrainConfig, apply_preset, mp_to_side, side_to_mp)
from rulztrainer import bodysim, facesim, history, lookfilter
from rulztrainer.train_queue import TrainQueue
from rulztrainer.dataset import skipped as dataset_skipped
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
    "text_in": (("qwen_image21", "flux3_klein"), "the text projection that brings Qwen3-VL's features into the model"),
    "embed": (("fibo2", "krea2", "qwen_image21", "flux3_klein"),
              "image / text / timestep embedders and the final layer"),
    "mode_blocks": (("flux3_klein",), "the 5 text and 5 image per-mode blocks that run before the 28 joint ones (on by "
                                      "default: the attention / feed-forward groups apply there too)"),
    "modulation": (("flux3_klein",), "the per-mode shift / scale / gate projections of the timestep"),
    "token_refiner": (("minimax_h3",), "the 2 token-refiner blocks over the text (musubi-tuner leaves them out; the "
                                       "training adapters train them)"),
}
N_BLOCKS = {"moss_image": 42, "fibo2": 44, "krea2": 28, "qwen_image21": 32, "flux3_klein": 28}
# Ideogram 4: 34 blocks; its per-block modulation, text projection and embedders are LoRA targets too
N_BLOCKS["ideogram4"] = 34
for _t in ("adaln", "text_in", "embed"):
    EXTRA_TARGETS[_t] = (EXTRA_TARGETS[_t][0] + ("ideogram4",), EXTRA_TARGETS[_t][1])
N_BLOCKS["minimax_h3"] = 50
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
ADAPTER_MODES = {"auto": "The model's training adapter (downloaded; Qwen-Image 2.1)", "none": "None",
                 "file": "A LoRA file"}
ADAPTER_MODES["auto"] = "The model's training adapter (downloaded; Qwen-Image 2.1, MiniMax H3)"
# MiniMax H3's Comfy-Org formats (h3/loading.py: the pruned files drop the ~13B AdaLN parameters)
FILE_QUANT_LABELS.update({"pruned_int8_convrot": "pruned int8 ConvRot (21.0 GB, for 16 GB cards)",
                          "pruned_bf16": "pruned bf16 (40.2 GB)", "nvfp4_awq": "NVFP4 AWQ (15.7 GB)"})
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
QUEUE = TrainQueue()  # runs waiting for the current one (output/_queue.json); queue_loop() starts them in turn
_start_lock = threading.RLock()  # the queue's thread and a page may start a run at the same moment


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
    with _start_lock:
        return _start_training(cfg)


def _start_training(cfg: TrainConfig) -> str:
    global _proc, _adopted, _job_dir
    reattach_job()
    if _running():
        return "A training job is already running."
    missing = [sp.path or "(none picked)" for sp in cfg.dataset_specs() if not sp.path or not Path(sp.path).is_dir()]
    if missing:
        return f"Dataset folder not found: {', '.join(missing)}"
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


def request_race_drop(job_dir: Path, name: str) -> None:
    """Ask the running trainer to drop a race configuration (it reads RACE_DROP after every step)."""
    with open(Path(job_dir) / "RACE_DROP", "a", encoding="utf-8") as f:
        f.write(name.strip() + "\n")


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
    if QUEUE.snapshot()["waiting"]:  # a kill usually means "stop everything": the queued runs wait for Resume
        QUEUE.set_paused(True)
        return "Process killed (no final save). The queue is paused: Resume it on the Monitor to go on."
    return "Process killed (no final save)."


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


def queue_tick() -> None:
    """One step of the queue: start the next waiting run when the job has ended."""
    reattach_job()
    done = QUEUE.advance(_running(), _start_queued)
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


def probe_chart_series(probes: list[dict], names: list[str]) -> tuple[list[dict], list[str]]:
    """(series, legend) of the Monitor's likeness chart. One run: the smoothed face, body and combined lines plus each
    probe's face and body dots. A race: the smoothed lines of every configuration, named after it ("lora16 face"),
    one colour per configuration and one line style per score (combined thick, face solid, body dashed); the
    winner's probes after the race are "winner"."""
    if not names:
        series = [_line("face", "#FF3B30", data=_probe_points(probes, "face_norm_ema", "face_ema")),
                  _line("body", "#F4F4F5", data=_probe_points(probes, "body_norm_ema", "body_ema")),
                  _line("combined", "#F59E0B", lineStyle={"width": 3}, data=_probe_points(probes, "combined_ema")),
                  {"name": "face (probe)", "type": "scatter", "symbolSize": 5,
                   "data": _probe_points(probes, "face_norm", "face"), "itemStyle": {"color": "#FB7185"}},
                  {"name": "body (probe)", "type": "scatter", "symbolSize": 5,
                   "data": _probe_points(probes, "body_norm", "body"), "itemStyle": {"color": "#A1A1AA"}}]
        return series, list(PROBE_DEFAULT_SERIES)
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
            series.append(_line(f"{label} {score}", color, lineStyle=dict(style),
                                data=_probe_points(by[name], key, fallback)))
            legend.append(f"{label} {score}")
    return series, legend


def probe_score_line(rec: dict | None) -> str:
    """"overall 38% | face 47% | body 17%": one probe's normalized scores (the scores themselves with normalization
    off)."""
    if not rec:
        return ""
    face, body, comb = rec.get("face_norm"), rec.get("body_norm"), rec.get("combined")
    if face is None and body is None and not rec.get("scale"):
        face, body = rec.get("face"), rec.get("body")
    parts = [f"{label} {v:.0f}%" for label, v in (("overall", comb), ("face", face), ("body", body)) if v is not None]
    return " | ".join(parts)


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
            self.follow_sample_defaults(self.target_family, family, self.turbo_on(), self.turbo_on())
            self.set_quant_options(family)
        self.target_family = family

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
                value = sel.value if sel.value in opts else "int8_convrot"
                if family == "minimax_h3" and kind == "model" and value == "int8_convrot":
                    value = "pruned_int8_convrot"  # H3: the pruned file (21 GB) instead of the full one (34 GB)
                sel.set_options(opts, value=value)

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
            compile_model=v("compile_model"), sample_seed=int(v("sample_seed")),
            sample_walk_seed=bool(v("sample_walk_seed")), sample_before_training=bool(v("sample_before_training")),
            steps=0 if v("soup_run") else int(v("steps")), batch_size=int(v("batch_size")),
            gradient_accumulation=int(v("gradient_accumulation")), learning_rate=float(v("learning_rate")),
            optimizer=v("optimizer"), lr_scheduler=v("lr_scheduler"), warmup_steps=int(v("warmup_steps")),
            max_grad_norm=float(v("max_grad_norm")), timestep_sampling=v("timestep_sampling"),
            automagic_min_lr=float(v("automagic_min_lr")), automagic_max_lr=float(v("automagic_max_lr")),
            automagic_polarity_history=int(v("automagic_polarity_history")), automagic_groups=v("automagic_groups"),
            automagic_clip_threshold=float(v("automagic_clip_threshold")),
            automagic_sensitivity_quantile=float(v("automagic_sensitivity_quantile")), automagic_vote=v("automagic_vote"),
            automagic_max_lr_change=float(v("automagic_max_lr_change")),
            timestep_shift=float(v("timestep_shift")), sigmoid_scale=float(v("sigmoid_scale")),
            logsnr_mean=float(v("logsnr_mean")), logsnr_std=float(v("logsnr_std")),
            block_swap=int(v("block_swap") or 0),
            clean_share=str(v("clean_share") or "").strip(),
            logit_mean=float(v("logit_mean")), logit_std=float(v("logit_std")), loss_weighting=v("loss_weighting"),
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
            sample_width=aspect_size(v("sample_aspect"), v("sample_mp"))[0],
            sample_height=aspect_size(v("sample_aspect"), v("sample_mp"))[1],
            sample_steps=int(v("sample_steps")), sample_cfg=float(v("sample_cfg")),
            sample_shift=float(v("sample_shift")), sample_negative=v("sample_negative") or "",
            turbo_lora={"auto": "auto", "none": ""}.get(v("turbo_mode"), s("turbo_lora")),
            turbo_lora_strength=float(v("turbo_lora_strength")),
            sample_model_path=s("sample_model_path"), sample_model_quant=v("sample_model_quant"),
            sample_lora=s("sample_lora"), sample_lora_strength=float(v("sample_lora_strength")),
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
            probe_seeds=int(v("probe_seeds")), probe_seed=int(v("probe_seed")), probe_size=int(v("probe_size")),
            probe_floor_seeds=int(v("probe_floor_seeds") or 4),
            probe_steps=int(v("probe_steps")), probe_cfg=float(v("probe_cfg")), probe_ema=float(v("probe_ema")),
            likeness_noise_band=float(v("likeness_noise_band")), likeness_track_best=bool(v("likeness_track_best")),
            likeness_track_best_face=bool(v("likeness_track_best_face")),
            likeness_track_best_body=bool(v("likeness_track_best_body")),
            likeness_best_any_rise=bool(v("likeness_best_any_rise")),
            likeness_normalize=bool(v("likeness_normalize")),
            face_scale_span=float(v("face_scale_span")), face_scale_curve=float(v("face_scale_curve")),
            body_scale_span=float(v("body_scale_span")), body_scale_curve=float(v("body_scale_curve")),
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
            race_configs=self.w["race_configs"].value,
            soup_files=([p.strip() for p in (v("soup_files") or "").splitlines() if p.strip()]
                        if v("soup_run") else []),
            soup_rank=int(v("soup_rank") or 0), soup_strengths=s("soup_strengths") or "1.0",
        )

    def from_config(self, cfg: TrainConfig) -> None:
        preset = next((k for k, v in MODEL_PRESETS.items() if v == cfg.model_path), CUSTOM)
        values = {**cfg.to_dict(), "model_preset": preset, "sample_prompts": "\n".join(cfg.sample_prompts),
                  "reward_prompts": "\n".join(cfg.reward_prompts),
                  "selftrain_prompts": "\n".join(cfg.selftrain_prompts),
                  "soup_files": "\n".join(cfg.soup_files), "soup_run": bool(cfg.soup_files),
                  "face_refs": "\n".join(cfg.face_refs), "body_refs": "\n".join(cfg.body_refs),
                  "model_source": FILE_SOURCE if cfg.model_source == "files" else HF_SOURCE}
        values["datasets"] = [{"path": sp.path, "resolutions": list(sp.resolutions), "repeats": sp.repeats}
                              for sp in cfg.dataset_specs() if sp.path]  # a single-folder config: that folder
        values["sample_aspect"], values["sample_mp"] = size_aspect(cfg.sample_width, cfg.sample_height)
        values["turbo_mode"] = ("auto" if cfg.turbo_lora == "auto" else "file" if cfg.turbo_lora else "none")
        values["finetune_extract_ranks"] = ", ".join(str(r) for r in cfg.finetune_extract_ranks)
        values["turbo_lora"] = "" if cfg.turbo_lora == "auto" else cfg.turbo_lora
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
  --rz-text: #111114; --rz-muted: #5F6068; --rz-ok: #16A34A;
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

/* header: logo, status pill, tabs */
.nicegui-header.rz-header {
  background: var(--rz-header-bg); border-bottom: 1px solid var(--rz-border); padding: 0; gap: 0;
  flex-direction: column; align-items: stretch;
}
.rz-header-row { max-width: 1440px; margin: 0 auto; padding: 12px 20px 2px; }
.rz-logo { height: 30px; width: auto; display: block; }
.rz-wordmark { letter-spacing: .34em; font-weight: 600; font-size: 12px; color: var(--rz-muted); padding-top: 7px; }
.rz-tagline { color: var(--rz-tagline); font-size: 13px; padding-top: 6px; margin-left: 8px; }
.rz-tabs { max-width: 1440px; margin: 0 auto; width: 100%; padding: 0 8px; }
.rz-tabs .q-tab { color: var(--rz-muted); min-height: 44px; padding: 0 14px; }
.rz-tabs .q-tab__label { font-weight: 500; font-size: 14px; }
.rz-tabs .q-tab__icon { font-size: 19px; }
.rz-tabs .q-tab--active { color: var(--rz-heading); }
.rz-tabs .q-tab--active .q-tab__icon { color: var(--rz-red-bright); }
.rz-tabs .q-tab__indicator { background: var(--rz-red); height: 3px; border-radius: 3px 3px 0 0; }
.rz-pill {
  border: 1px solid var(--rz-border-strong); border-radius: 999px; padding: 4px 12px; font-size: 13px;
  background: var(--rz-pill-bg); cursor: pointer; max-width: 320px;
}
.rz-pill:hover { border-color: var(--rz-hover-border); }
.rz-pill-text { color: var(--rz-text); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
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
.rz-section > .q-expansion-item__container > .q-item .q-item__label { font-size: 16px; font-weight: 600; color: var(--rz-heading); }
.rz-section > .q-expansion-item__container > .q-item .q-item__label--caption {
  font-size: 13px; color: var(--rz-muted); font-weight: 400; margin-top: 2px;
}
.rz-section .nicegui-expansion-content { padding: 2px 18px 20px; gap: 14px; }
.rz-advanced { border-top: 1px dashed var(--rz-border); }
.rz-advanced > .q-expansion-item__container > .q-item { padding: 10px 0 4px; min-height: 0; color: var(--rz-text-2); font-weight: 500; }
.rz-advanced > .q-expansion-item__container > .q-item .q-item__section--avatar { min-width: 0; padding-right: 8px; }
.rz-advanced > .q-expansion-item__container > .q-item .q-icon { color: var(--rz-muted); font-size: 18px; }
.rz-section .rz-advanced .nicegui-expansion-content { padding: 8px 0 6px; gap: 14px; }
.rz-title { font-size: 16px; font-weight: 600; color: var(--rz-heading); }
.rz-step-title { font-size: 20px; font-weight: 700; color: var(--rz-heading); }
.rz-subhead { font-size: 12px; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; color: var(--rz-muted); margin-top: 4px; }
.rz-hint { font-size: 13px; color: var(--rz-muted); line-height: 1.55; max-width: 980px; }
.rz-warn { color: var(--rz-warn); }
.rz-fields { gap: 14px 16px; align-items: center; flex-wrap: wrap; }

/* form fields and buttons */
.q-field--outlined .q-field__control { border-radius: 9px; }
.q-field--outlined.q-field--dark .q-field__control:before { border-color: var(--rz-border-strong); }
.q-field__label { color: var(--rz-label); }
.q-field--dark .q-field__native, .q-field--dark .q-field__input, .q-field--dark .q-select__dropdown-icon { color: #F4F4F5; }
.q-field--outlined:not(.q-field--dark) .q-field__control:before { border-color: var(--rz-border-strong); }
.q-btn { text-transform: none; font-weight: 600; letter-spacing: .01em; border-radius: 9px; }
.q-toggle__label, .q-checkbox__label, .q-radio__label { color: var(--rz-text-3); }
.q-tooltip { background: var(--rz-tooltip-bg); border: 1px solid var(--rz-border-strong); color: var(--rz-text); font-size: 13px; max-width: 440px; line-height: 1.45; }
.q-menu { background: var(--rz-menu-bg); border: 1px solid var(--rz-border-strong); }

/* Train tab: section list on the left */
.rz-nav { position: sticky; top: 108px; width: 210px; flex-shrink: 0; gap: 2px; }
.rz-nav-title { font-size: 12px; text-transform: uppercase; letter-spacing: .08em; color: var(--rz-muted); padding: 2px 12px 6px; }
.rz-nav-btn { width: 100%; color: var(--rz-text-2); font-weight: 500; }
.rz-nav-btn .q-btn__content { justify-content: flex-start; }
.rz-nav-btn .q-icon { color: var(--rz-red-bright); font-size: 18px; }

/* action bar */
.nicegui-footer.rz-footer {
  background: var(--rz-footer-bg); backdrop-filter: blur(6px); border-top: 1px solid var(--rz-border); padding: 0;
}
.rz-footer-row { max-width: 1440px; margin: 0 auto; padding: 10px 20px; }
.rz-status-msg {
  font-size: 13px; color: var(--rz-text-2); display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;
  overflow: hidden;
}

/* Monitor */
.rz-tile { background: var(--rz-surface); border: 1px solid var(--rz-border); border-radius: 12px; padding: 12px 16px; min-width: 150px; flex: 1 1 150px; }
.rz-tile-label { font-size: 11px; color: var(--rz-muted); text-transform: uppercase; letter-spacing: .08em; }
.rz-tile-value { font-size: 20px; font-weight: 600; color: var(--rz-heading); font-variant-numeric: tabular-nums; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 100%; }
.rz-tile-sub { font-size: 12px; color: var(--rz-muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 100%; }
.rz-banner { border-radius: 10px; padding: 10px 14px; font-size: 14px; width: 100%; }
.rz-banner.error { background: rgba(225, 6, 0, .12); border: 1px solid rgba(255, 59, 48, .45); color: var(--rz-err-text); }
.rz-banner.warning { background: rgba(245, 158, 11, .10); border: 1px solid rgba(245, 158, 11, .40); color: var(--rz-warn-text); }
.rz-banner.info { background: var(--rz-surface); border: 1px solid var(--rz-border-strong); color: var(--rz-text-2); }
.rz-banner.ok { background: rgba(34, 197, 94, .10); border: 1px solid rgba(34, 197, 94, .40); color: #86EFAC; }
.q-uploader { background: var(--rz-surface); border: 1px dashed var(--rz-border-strong); border-radius: 12px; }
.q-uploader__header { color: var(--rz-text); border-bottom: 1px solid var(--rz-border); }
.rz-log { background: var(--rz-deep); border: 1px solid var(--rz-border); border-radius: 10px; color: var(--rz-text-2); }
.rz-thumb { background: var(--rz-surface-2); border: 1px solid var(--rz-border); border-radius: 10px; overflow: hidden; }
.rz-thumb.rz-picked { border: 2px solid var(--rz-red, #E10600); }
.rz-ds-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 12px; width: 100%; }
.rz-look-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(130px, 1fr)); gap: 8px; }
.rz-ds-card { background: var(--rz-surface-2); border: 1px solid var(--rz-border); border-radius: 12px;
  overflow: hidden; transition: border-color .15s; min-width: 0; }
.rz-ds-card:hover { border-color: var(--rz-border-strong); }
.rz-ds-card.rz-ds-open { border-color: var(--rz-red); }
.rz-ds-thumbs { display: grid; grid-template-columns: repeat(4, 1fr); gap: 2px; background: var(--rz-thumb-bg); }
.rz-ds-empty { display: flex; align-items: center; justify-content: center; aspect-ratio: 4 / 1;
  color: var(--rz-muted); font-size: 13px; background: var(--rz-thumb-bg); }
.rz-mono { font-family: ui-monospace, "Cascadia Mono", Consolas, Menlo, monospace; font-size: 12px; color: var(--rz-text-2); }

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
.rz-rec { font-size: 13px; color: var(--rz-rec); margin-top: 2px; }
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
.rz-summary { display: grid; grid-template-columns: max-content minmax(0, 1fr); gap: 8px 24px; width: 100%; font-size: 14px; }
.rz-summary .k { color: var(--rz-muted); }
.rz-summary .v { color: var(--rz-text-3); overflow-wrap: anywhere; }
@media (max-width: 760px) {
  .rz-setting { grid-template-columns: 1fr; }
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


CLEAN_SLIDE = "60:7:30:15:2"  # Fizgig's MOSS-Image Slide (rulztrainer.clean_share.SLIDE)


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
    plan = (f"{first:g}% until epoch {until:g}, then {rest:g}%" if end <= until else
            f"{first:g}% until epoch {until:g}, sliding to {rest:g}% by epoch {end:g}")
    again = (f", and after that one epoch in {every} back at {first:g}%" if every > 1 else
             f", and after that {first:g}% again" if every == 1 else "")
    return text + " With the schedule: " + plan + again + "."


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


@app.get("/rz-history-xlsx")
def _serve_history_xlsx(run: str):
    """The likeness spreadsheet of a Previous Runs record, built from its saved raw probe scores."""
    import tempfile

    from fastapi import HTTPException
    from fastapi.responses import FileResponse

    from rulztrainer import likeness_export

    rec = next((r for r in history.list_runs() if r["id"] == run), None)
    if rec is None:
        raise HTTPException(status_code=404, detail="No such run in the history")
    out = Path(tempfile.gettempdir()) / f"{history_file_name(rec)}_likeness.xlsx"
    try:
        likeness_export.export_run(rec, out)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except (OSError, ValueError) as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    return FileResponse(out, filename=out.name,
                        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


def history_file_name(rec: dict) -> str:
    """A file name for a history run: its own name if it has one, else the job and its start."""
    started = time.strftime("%Y%m%d-%H%M", time.localtime(rec.get("started") or 0))
    base = rec.get("label") or f"{rec.get('job') or 'run'}_{started}"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._") or "run"


def lora_url(path) -> str:
    """The download URL of a LoRA a run saved (served by /rz-lora)."""
    from urllib.parse import quote

    return f"/rz-lora?path={quote(str(Path(path).resolve()))}"


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


DATASETS_ROOT = ROOT / "datasets"  # where Create dataset puts new datasets (git-ignored)
DATASETS_REGISTRY = ".rulz_datasets.json"  # in DATASETS_ROOT: dataset folders elsewhere that were added to the list
CAPTION_EXTS = {".txt", ".json"}
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
    """Where an uploaded file goes in a dataset folder: its own file name (a same-named file is replaced), for images
    and .txt / .json captions only; None for any other kind of file. The folder is created."""
    fname = Path(str(name).replace("\\", "/")).name  # the name only: never a folder from the browser
    if not fname or fname.startswith(".") or Path(fname).suffix.lower() not in IMAGE_EXTS | CAPTION_EXTS:
        return None
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    return dest / fname


def copy_into_dataset(src, dest) -> dict:
    """Copy the images and captions (.txt, .json) of a folder on this machine into a dataset, sub-folders included
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
        if p.suffix.lower() not in IMAGE_EXTS | CAPTION_EXTS:
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
    """Full-window image viewer dialog. Returns show(path) callable."""
    with ui.dialog().props("maximized") as dlg, ui.card().tight().classes("w-full h-full bg-black"):
        with ui.row().classes("w-full items-center justify-between px-3 py-1"):
            title = ui.label("").classes("text-sm text-white truncate")
            ui.button(icon="close", on_click=dlg.close).props("flat dense round color=white")
        big = ui.image("").props("fit=contain").classes("w-full").style("height: calc(100vh - 48px)")
        big.on("click", dlg.close)

    def show(path: Path):
        title.text = str(path)
        big.set_source(image_url(path))
        dlg.open()

    return show


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
                .props("accept=.png,.jpg,.jpeg,.webp,.bmp,.txt,.json flat bordered color=transparent") \
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
                    ui.image(image_url(p)).props("fit=contain").classes("w-full h-64 rz-imgbg cursor-pointer") \
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
                  "A specific person or character. Turns on the measured identity recipe: region-weighted loss, "
                  "close-up crops, and likeness probes that keep the best checkpoint and stop early. Then pick how "
                  "much of the person counts: the face, the body or both."),
    "style": ("palette", "Style",
              "An art style, medium, lighting or look that should carry over to any subject."),
    "concept": ("category", "Concept / object",
                "An object, outfit, pose or idea. Also the right pick when you are not sure."),
}

# The wizard's run modes: one run, or a configuration race of several racers built on the same base setup
RUN_MODES = {
    "single": ("play_circle", "One run",
               "Train the setup on these pages once."),
    "race": ("emoji_events", "Race several setups",
             "Train a few versions of this setup side by side, each changing only what you assign it (a learning "
             "rate, the adapter, a seed...), and let the likeness probes pick the winner."),
}

# names for the settings a racer may change (the rest show their config name)
RACE_LABELS = {
    "learning_rate": "Learning rate", "optimizer": "Optimizer", "weight_decay": "Weight decay",
    "lr_scheduler": "Learning rate schedule", "warmup_steps": "Warm-up steps", "lora_plus_ratio": "LoRA+ ratio",
    "automagic_min_lr": "Automagic min lr", "automagic_max_lr": "Automagic max lr",
    "automagic_polarity_history": "Automagic polarity history", "automagic_clip_threshold": "Automagic clip threshold",
    "automagic_groups": "Automagic lr groups", "automagic_sensitivity_quantile": "Automagic sensitivity quantile",
    "automagic_vote": "Automagic vote", "automagic_max_lr_change": "Automagic max lr change",
    "lora_ema": "Weight average", "timestep_sampling": "Timestep sampling", "loss_weighting": "Loss weighting",
    "region_weights": "Region-weighted loss", "face_loss_weight": "Face loss weight",
    "body_loss_weight": "Body loss weight", "background_loss_weight": "Background loss weight",
    "id_loss_weight": "Identity loss weight", "attn_localize": "Trigger attention on the subject",
    "attn_localize_weight": "Trigger attention weight", "id_align": "Identity alignment",
    "id_align_weight": "Identity alignment weight", "body_align": "Body alignment",
    "body_align_weight": "Body alignment weight", "likeness_curriculum": "Face/body curriculum",
    "likeness_lr_control": "Likeness lr control", "loss_watch": "Per-image loss watch",
    "per_image_lr": "Per-image learning rate", "lora_rank": "LoRA rank", "lora_alpha": "LoRA alpha",
    "lora_targets": "LoRA layers", "lora_blocks": "LoRA blocks", "lora_dropout": "LoRA dropout",
    "network_type": "Adapter type", "lokr_factor": "LoKr factor", "oft_block_size": "OFT block size",
    "oft_constraint": "OFT constraint", "dora": "DoRA", "caption_dropout": "Caption dropout",
    "flip_augment": "Flip augment", "crop_augment": "Close-up crops", "crop_augment_weight": "Crop weight",
    "look_warmup": "Look warm-up", "selftrain": "Self-training", "selftrain_at": "Self-training at",
    "selftrain_images": "Self-training pictures", "selftrain_min_likeness": "Self-training likeness floor",
    "selftrain_share": "Self-training share", "selftrain_weight": "Self-training weight", "seed": "Seed",
    "dop_weight": "DOP weight (LoRA Isolation)", "dop_every": "DOP every N steps (LoRA Isolation)",
}


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
        return (f"{len(rows)} racers ({racers}); a relay: after each round everyone continues from the round's "
                f"winner, and the last round's winner is the run's LoRA")
    if cfg.race_drop_lowest:
        rules.append("the lowest leaves after each round" + (f" (after {cfg.race_warmup_probes} warm-up rounds)"
                                                             if cfg.race_warmup_probes else ""))
    else:
        rules.append("nobody is dropped for being behind")
    if cfg.race_patience_drop:
        rules.append(f"a racer leaves after {cfg.race_patience} probes without improvement"
                     + (" unless its likeness is rising again" if cfg.race_patience_rising else ""))
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
# Ideogram 4: 9.29 B weights (the release's fp8 file, 9.29 GB): bf16 about 18.6 GB; Comfy-Org's int8_convrot file
# 9.58 GB, nvfp4_mixed 5.49 GB
MODEL_GB["ideogram4"] = {"bf16": 18.6, "int8": 9.6, "nvfp4": 5.5}
# MiniMax H3 pruned (Comfy-Org's file sizes: bf16 40.2 GB, int8_convrot 21.0 GB; nvfp4 is an estimate: the blocks at
# 4.5 bits per weight, 10.9 GB, and the bf16 token refiner, 1.5 GB)
MODEL_GB["minimax_h3"] = {"bf16": 40.2, "int8": 21.0, "nvfp4": 12.5}
STEP_HEADROOM_GB = 4.5


def block_swap_plan(cfg: TrainConfig, vram: float, quant: str) -> tuple[int, float]:
    """(blocks to swap, GB of weights they hold) so the model plus a step's memory fits in `vram` GB; 0 when it fits."""
    fam = cfg.arch_name()
    if quant == "auto":
        if cfg.model_source == "files":
            quant = "bf16" if cfg.model_file_quant == "bf16" else "int8"
            quant = "nvfp4" if cfg.model_file_quant == "nvfp4_mixed" else quant  # Ideogram 4's 5.5 GB file
        else:
            quant = {"krea2": "int8" if vram >= 22 else "nvfp4", "fibo2": "int8" if vram < 22 else "bf16",
                     "qwen_image21": "bf16" if vram >= 24 else "int8" if vram >= 11 else "nvfp4",
                     "flux3_klein": "bf16" if vram >= 15 else "int8" if vram >= 9.5 else "nvfp4",
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
        r["isolation"] = (bool(subject), "keeps the subject's face off other people (Fizgig: strangers reading as "
                                         "the subject 48 of 50 -> 0 of 50); about 1.5-2x the training time"
                          if subject else "needs a trigger word to isolate (Dataset step)")
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
        r["steps"] = (2000, "a good start; scan your dataset (Dataset step) for a number that fits it")
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
            if klein and cfg.network_type == "lora":
                r["learning_rate"] = (1e-4, "BFL's FLUX 3 Klein recipe: 1e-4, constant after 200 warm-up steps")
                r["lr_scheduler"] = ("constant", "BFL's FLUX 3 Klein recipe")
                r["warmup_steps"] = (200, "BFL's FLUX 3 Klein recipe")
        r["network_type"] = ("lora", "reached likeness much sooner than LoKr in RulzTrainer's tests so far, and "
                                     "loads everywhere (ComfyUI and diffusers)")
        if cfg.network_type == "lokr":
            r["lokr_factor"] = (8, "Fizgig's measured default: lower is bigger files, nothing gained above 8")
        elif cfg.network_type == "lora":
            rank = 32 if goal == "style" else 16
            r["lora_rank"] = (rank, "styles carry a lot of varied detail" if goal == "style"
                              else "plenty for a character or concept; higher ranks mostly add file size")
            if qwen:
                rank = 16 if goal == "style" else 8
                r["lora_rank"] = (rank, "Fizgig's Qwen-Image 2.1 style preset" if goal == "style" else
                                  "Fizgig's Qwen-Image 2.1 recipe: rank 8 held skin detail best; 16 at the same rate "
                                  "overcooked")
            if klein:
                rank = 32
                r["lora_rank"] = (rank, "BFL's FLUX 3 Klein recipe: rank 32, alpha 32")
            r["lora_alpha"] = (float(rank), "alpha equal to rank keeps the adapter's scale at 1")
    if goal == "style":
        r["timestep_sampling"] = ("logsnr", "spends most steps at very high noise, where an art style is learned "
                                            "(the style-friendly sampler)")
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
    return r


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


def review_issues(cfg: TrainConfig, goal: str | None, data: dict | None, n_gpus: int) -> list[tuple[str, str]]:
    """What the wizard's last step points out before starting: (error | warning | info, text)."""
    out: list[tuple[str, str]] = []
    if _running():
        out.append(("error", "A training job is already running. Stop it (Monitor tab) before starting another."))
    specs = [sp for sp in cfg.dataset_specs() if sp.path]
    missing = [sp.path for sp in specs if not Path(sp.path).is_dir()]
    if not specs:
        out.append(("error", "No dataset picked: pick one or more (Dataset step), or create one on the Dataset tab."))
    elif missing:
        out.append(("error", f"Dataset folder not found: {', '.join(missing)}"))
    elif data is not None:
        if not data["images"]:
            out.append(("error", "The datasets have no images (png, jpg, webp or bmp)."))
        elif data["captioned"] < data["images"] and not cfg.trigger_word:
            out.append(("warning", f"{data['images'] - data['captioned']} of {data['images']} images have no caption "
                                   "and there is no trigger word, so they train on an empty prompt."))
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
    if cfg.multi_gpu != "off" and n_gpus < 2:
        out.append(("error", "Multi-GPU is on, but only one GPU was found."))
    if goal == "character" or cfg.preset == "identity":
        if not (cfg.face_refs or cfg.body_refs):
            out.append(("warning", "No face or body references: likeness scores, the best checkpoint and early stop "
                                   "need them (Dataset step)."))
        elif character_focus(cfg) != "face_only" and not cfg.body_refs:
            out.append(("info", "This focus scores the body, but there are no body references: the body probe goes "
                                "unscored and the overall score is the face alone. Tick Body ref on full-length "
                                "photos, or take the recommendation (Dataset step)."))
        elif character_focus(cfg) != "body_only" and not cfg.face_refs:
            out.append(("info", "This focus scores the face, but there are no face references: the face probe goes "
                                "unscored (Dataset step)."))
        elif (cfg.probe_every > 0 and cfg.probe_steps <= 10
              and not (cfg.turbo_lora or cfg.samples_on_distilled_model())):
            out.append(("warning", f"Probes render in {cfg.probe_steps} steps without a turbo LoRA, so they come out "
                                   "unfinished and their likeness scores mean little. Set a turbo LoRA or raise the "
                                   "probe steps (Samples step)."))
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


class Wizard:
    """Guided setup on its own tab: one step per topic, each setting explained, with a recommended value.

    Its controls are bound to the full form's widgets (Train tab), so both views edit the same settings and a run
    started from either is the same run."""

    STEPS = (("goal", "Goal", "flag"), ("model", "Model", "model_training"), ("hardware", "Hardware", "memory"),
             ("dataset", "Dataset", "photo_library"), ("training", "Training", "school"),
             ("samples", "Samples", "image"), ("racers", "Racers", "groups"), ("rules", "Race rules", "emoji_events"),
             ("review", "Review & start", "rocket_launch"))
    RACE_STEPS = ("racers", "rules")  # shown only when the run is a race

    def __init__(self, f: Form, gpus: list[str], start, goto, enqueue=None):
        self.f, self.w = f, f.w
        self.gpus = gpus  # torch device names for the device select
        self.start = start  # start_job(): starts training from the form, True when it started
        self.enqueue = enqueue  # queue_job(): adds the form's run to the training queue
        self.goto = goto  # goto(tab name)
        self.goal: str | None = "character" if self.w["preset_toggle"].value else None
        self.focus = "face_focus"  # Character / person: a key of CHARACTER_FOCUS; follows the form (character_focus)
        self.focus_stash: dict = {}  # references and probe prompts from before a face only / body only pick
        self.conditional: list[tuple] = []  # (box, predicate): settings shown only while the predicate holds
        self.smi: list[dict] = []  # nvidia-smi rows (memory per GPU)
        self.data: dict | None = None  # last dataset scan
        self.recs: dict[str, tuple] = {}
        self.rec_rows: list[tuple] = []
        self.goal_cards: dict[str, ui.card] = {}
        self.focus_cards: dict[str, ui.card] = {}
        self.model_cards: dict[str, ui.card] = {}
        self.mode_cards: dict[str, ui.card] = {}
        self.steps: dict[str, ui.step] = {}
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

    @contextmanager
    def setting(self, key: str | None, title: str, text: str, show_if=None, when=None):
        """One setting: title, explanation and recommendation on the left, the control (built in the block) on the
        right. show_if=(form key, predicate) shows it only while the predicate holds for that widget's value; when
        is a no-argument predicate re-read on every refresh (for conditions on more than one setting). Yields the
        setting's box."""
        box = ui.element("div").classes("rz-setting")
        if when is not None:
            self.conditional.append((box, when))
            box.visible = bool(when())
        if show_if is not None:
            box.bind_visibility_from(self.w[show_if[0]], "value", show_if[1])
        with box:
            with ui.column().classes("gap-1 min-w-0"):
                ui.label(title).classes("rz-setting-title")
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
        """The next / previous step, skipping the race steps when the run is not a race."""
        names = [n for n, _, _ in self.STEPS if self.racing() or n not in self.RACE_STEPS]
        cur = self.stepper.value if isinstance(self.stepper.value, str) else names[0]
        i = names.index(cur) if cur in names else 0
        self.stepper.value = names[max(0, min(len(names) - 1, i + d))]

    def heading(self, title: str, text: str):
        ui.label(title).classes("rz-step-title")
        hint(text)

    # ------------------------------------------------------------ recommendations

    def is_character(self) -> bool:
        return self.goal == "character"

    def has_face(self) -> bool:
        return self.goal == "character" and self.focus != "body_only"

    def has_body(self) -> bool:
        return self.goal == "character" and self.focus != "face_only"

    def is_lora(self) -> bool:
        return not self.w["finetune"].value and self.w["network_type"].value == "lora"

    def refresh(self):
        try:
            cfg = self.f.to_config()
        except (TypeError, ValueError):
            return  # a field is mid-edit (e.g. an empty number); keep the last recommendations
        if self.goal == "character":
            self.focus = character_focus(cfg)
        self.mark_focus()
        for box, when in self.conditional:
            box.visible = bool(when())
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

    def apply(self, keys: list[str]):
        for key in keys:
            rec = self.recs.get(key)
            if rec is not None:
                self.w[key].value = list(rec[0]) if isinstance(rec[0], list) else rec[0]
            self.refresh()  # later recommendations can depend on earlier ones (save every on steps)

    # ------------------------------------------------------------ steps

    def build(self):
        names = [n for n, _, _ in self.STEPS]  # the race steps' headers, hidden while the run is not a race
        ui.add_css(",\n".join(f".rz-norace .q-stepper__header > .q-stepper__tab:nth-child({names.index(n) + 1})"
                             for n in self.RACE_STEPS) + " { display: none; }")
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
        self.w["race"].on_value_change(lambda _: (self.mark_race(), self.refresh()))
        self.mark_goal()
        self.mark_model()
        self.mark_race()
        self.refresh()
        ui.timer(0.2, self.detect_gpus, once=True)

    async def on_step(self, e):
        name = e.value if isinstance(e.value, str) else e.value.props.get("name")
        names = [s[0] for s in self.STEPS]
        idx = names.index(name) if name in names else 0
        for i, n in enumerate(names):
            self.steps[n].props(add="done") if i < idx else self.steps[n].props(remove="done")
        if name == "dataset" and self.data is None and any(Path(p).is_dir() for p in self.w["datasets"].selected):
            await self.scan()
        if name == "racers":
            self.load_racers()
            self.rebuild_racers()  # the base setup may have changed on the earlier pages
        if name == "rules":
            self.estimate_race()
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
        with ui.column().classes("w-full gap-2") as self.focus_box:
            ui.label("What should the likeness cover?").classes("rz-setting-title")
            with ui.element("div").classes("rz-choices"):
                for key, (icon, title, text) in CHARACTER_FOCUS.items():
                    with ui.card().props("flat").classes("rz-choice") as c:
                        with ui.row().classes("w-full items-center gap-2 no-wrap"):
                            ui.icon(icon).classes("rz-choice-icon")
                            ui.label(title).classes("rz-setting-title")
                            ui.space()
                            ui.icon("check_circle").classes("rz-check")
                        ui.label(text).classes("rz-hint")
                    c.on("click", lambda _, k=key: self.set_focus(k))
                    self.focus_cards[key] = c
        self.goal_note = ui.label("").classes("rz-banner info")
        with ui.column().classes("w-full gap-2") as self.mode_box:
            ui.label("One run, or a race?").classes("rz-setting-title")
            with ui.element("div").classes("rz-choices"):
                for key, (icon, title, text) in RUN_MODES.items():
                    with ui.card().props("flat").classes("rz-choice") as c:
                        with ui.row().classes("w-full items-center gap-2 no-wrap"):
                            ui.icon(icon).classes("rz-choice-icon")
                            ui.label(title).classes("rz-setting-title")
                            ui.space()
                            ui.icon("check_circle").classes("rz-check")
                        ui.label(text).classes("rz-hint")
                    c.on("click", lambda _, k=key: self.set_mode(k))
                    self.mode_cards[key] = c
            hint("A race needs the likeness probes, so it is for a character / person. The next pages set up the "
                 "base racer; then you pick the racers and the race's rules.")
        with self.setting(None, "Job name", "The name of this run. Checkpoints are saved as "
                                            "output/<job name>/<job name>_<step>.safetensors and the final LoRA as "
                                            "output/<job name>/<job name>.safetensors."):
            self.bind(ui.input("Job name"), "job_name").classes("w-full")
        self.nav(back=False)

    def set_goal(self, goal: str):
        focus = self.focus  # the recipe below is face focus; the pick made before is put back on top of it
        self.goal = goal
        want = goal == "character"
        if bool(self.w["preset_toggle"].value) != want:
            self.w["preset_toggle"].value = want  # applies the identity recipe, or restores the settings from before
        self.focus = focus
        if want and focus != "face_focus":
            self.apply_focus()
        self.mark_goal()
        self.refresh()

    def set_focus(self, focus: str):
        """One of the five focuses: the probes, references, crops, loss weights and score weights for it."""
        for key in ("face_refs", "body_refs", "probe_face_prompt", "probe_body_prompt"):
            if (self.w[key].value or "").strip():  # what the user has, kept for when the focus needs it again
                self.focus_stash[key] = self.w[key].value
        self.focus = focus
        self.apply_focus()
        self.mark_goal()
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
        self.mark_focus()
        if self.goal == "character":
            self.goal_note.text = ("Identity recipe, " + FOCUS_NOTES[self.focus] + "; LoRA+ 4 with adamw8bit, "
                                   "likeness probes every 25 steps with best checkpoint and early stop, per-image "
                                   "loss watch.")
        elif self.goal:
            self.goal_note.text = f"{GOALS[self.goal][1]}: general settings, tuned on the next pages."
        else:
            self.goal_note.text = "Pick one to get recommendations that fit it."

    def set_mode(self, mode: str):
        self.w["race"].value = mode == "race"  # mark_race follows through the switch

    def mark_race(self):
        """The run-mode cards, the race steps and the review's start button."""
        racing = self.racing()
        for key, c in self.mode_cards.items():
            c.classes(add="selected") if (key == "race") == racing else c.classes(remove="selected")
        if hasattr(self, "stepper"):  # a step's header is drawn by the stepper: hidden by its position (CSS)
            self.stepper.classes(remove="rz-norace") if racing else self.stepper.classes(add="rz-norace")
        if hasattr(self, "start_btn"):
            self.start_btn.text = "Start the race" if racing else "Start training"

    def mark_focus(self):
        """The focus cards."""
        character = self.goal == "character"
        if hasattr(self, "mode_box"):
            self.mode_box.visible = character
        if hasattr(self, "focus_box"):
            self.focus_box.visible = character
        for key, c in self.focus_cards.items():
            c.classes(add="selected") if key == self.focus else c.classes(remove="selected")

    # ---- 2 model
    def step_model(self):
        self.heading("Which model do you want to train on?", "The base model the LoRA is trained for.")
        with self.setting(None, "Where the model comes from",
                          "Single files (the default) downloads Comfy-Org's ComfyUI files for the model: the diffusion "
                          "model and text encoder in the format you pick (int8 ConvRot: about half of bf16) and the "
                          "VAE, or uses files you already have. Hugging Face downloads the original diffusers repo "
                          "into models/ instead."):
            self.bind(ui.radio([FILE_SOURCE, HF_SOURCE]), "model_source")
        with ui.column().classes("w-full gap-3 pt-4"):
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
            with ui.column().classes("w-full gap-0") \
                    .bind_visibility_from(self.w["model_source"], "value", lambda x: x == HF_SOURCE):
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
            with self.setting(None, "File formats to download",
                              "int8 ConvRot is about half the size of bf16, trains as fast and loses under 1% "
                              "accuracy (the default). Krea 2 also offers fp8, Ideogram 4 fp8 and nvfp4."):
                self.bind(ui.select(file_quant_options(preset_arch("MOSS-Image-Base"), "model"),
                                    label="Diffusion model"), "model_file_quant").classes("w-full")
                self.bind(ui.select(file_quant_options(preset_arch("MOSS-Image-Base"), "text"),
                                    label="Text encoder"), "text_encoder_file_quant").classes("w-full")
            with self.setting(None, "Download now",
                              "Training downloads the files on start; this does it now, so a problem (disk space, "
                              "network) shows before you start. Hugging Face's cache keeps them for later runs."):
                files_status = hint("")

                async def download_now():
                    files_status.text = "Downloading / checking the model files..."
                    files_status.text = await run.io_bound(download_files, self.f.to_config())

                ui.button("Download now", icon="download", on_click=download_now).props("outline")
            for key, title, text, kind in (
                    ("transformer_file", "Diffusion model file (optional)", "A ComfyUI models/diffusion_models file "
                                                                            "you already have.", "file"),
                    ("text_encoder_file", "Text encoder file (optional)", "Qwen3.5-9B for MOSS-Image, Qwen3-VL-4B for "
                                                                          "Krea 2, FIBO 2.0 and FLUX 3 Klein, "
                                                                          "Qwen3-VL-8B for Qwen-Image 2.1 and Ideogram 4, "
                                                                          "Qwen3-VL-32B (cut to 50 layers) for "
                                                                          "MiniMax H3.", "file"),
                    ("vae_file", "VAE file (optional)", "The Flux2 VAE for MOSS-Image, FIBO 2.0, FLUX 3 Klein and "
                                                        "Ideogram 4, the Qwen-Image VAE for Krea 2, its own for "
                                                        "Qwen-Image 2.1, the video VAE for MiniMax H3 (its audio "
                                                        "VAE is found next to it or downloaded).", "file"),
                    ("tokenizer_path", "Tokenizer folder (optional)", "Left empty, the tokenizer is downloaded once "
                                                                      "(a few MB).", "dir")):
                with self.setting(None, title, text + " Left empty, it is downloaded."):
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
                        "checkpoint_keep_blocks", "block_swap", "compile_model"])
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
        with self.setting("block_swap", "Block swap",
                          "For a card the model does not fit: this many blocks keep their frozen weights in system "
                          "RAM and stream to the GPU as each one runs, the next copied in while the current computes. "
                          "Measured on Krea 2 int8 (16 GB card): 14 of 28 blocks swapped cut the step's peak from "
                          "10.8 to 7.7 GB at about the same speed. Needs gradient checkpointing and one GPU; not "
                          "with a fine-tune. 0 = off.",
                          when=lambda: self.w["multi_gpu"].value == "off" and not self.w["finetune"].value):
            self.bind(ui.number("Blocks swapped (0 = off)", min=0, precision=0), "block_swap").classes("w-full")
        with self.setting("compile_model", "Compile the model",
                          "torch.compile turns the small steps between the big matrix multiplies into fewer, fused GPU "
                          "kernels: measured 26% faster per step on MOSS-Image and 12% on Krea 2, after a warm-up of "
                          "under a minute. auto does it for runs of 100 steps or more."):
            self.bind(ui.select(COMPILE_CHOICES, label="Compile"), "compile_model").classes("w-full")
        self.nav()

    # ---- 4 dataset
    def step_dataset(self):
        self.heading("Your images", "One or more datasets: folders of images, each with an optional same-name .txt "
                                    "caption. 15-50 sharp, varied images are typical for a character; styles and "
                                    "concepts like 20-100.")
        with self.setting(None, "Datasets", "Pick the datasets to train on. Create them, upload images and write "
                                            "captions on the Dataset tab. Each dataset's own resolutions and repeats "
                                            "are on the Train tab (Dataset section)."):
            picker = self.w["datasets"]

            def choices():
                return {p: Path(p).name or p for p in picker.paths()}

            chosen = ui.select(choices(), multiple=True, value=picker.selected, label="Datasets") \
                .props("use-chips").classes("w-full")
            chosen.bind_value(picker, "selected")
            chosen.on_value_change(lambda _: (self.forget_scan(), self.refresh()))
            picker.listeners.append(lambda: chosen.set_options(choices(), value=picker.selected))
            with ui.row().classes("gap-2"):
                ui.button("Scan datasets", icon="search", on_click=self.scan).props("unelevated")
                ui.button("Dataset tab", icon="photo_library", on_click=lambda: self.goto("Dataset")).props("flat")
        self.scan_box = ui.column().classes("w-full gap-3")
        self.apply_bar(["resolutions", "enable_buckets", "flip_augment", "crop_augment", "face_refs", "body_refs",
                        "look_warmup"])
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
        with self.setting("crop_augment", "Close-up crops",
                          "Adds crops of every image as extra training images, so the model sees the face up close "
                          "and not only small in a full-length photo: face close-ups, and with face+bust also head "
                          "and shoulders. Crops never upscale past the source.", when=self.is_character):
            self.bind(ui.select(["none", "face", "face+bust"], label="Crops"), "crop_augment").classes("w-full")
        with self.setting("face_refs", "Face references",
                          "Images whose face is the identity to match. Samples and probes are compared with them "
                          "(ArcFace; 50%+ generally means the same person), which picks the best checkpoint and "
                          "stops early. A folder counts as all its images; you can also tick Face ref on the Dataset "
                          "tab.", when=self.has_face):
            self.bind(ui.textarea("Face references (one path per line)"), "face_refs").classes("w-full")
        with self.setting("body_refs", "Body references",
                          "Photos that show the person's build, from the neck to mid-thigh or more. Probes and samples "
                          "are compared with them to score the body. A folder counts as all its images (those without "
                          "a usable view of the body are skipped); you can also tick Body ref on the Dataset tab.",
                          when=self.has_body):
            self.bind(ui.textarea("Body references (one path per line)"), "body_refs").classes("w-full")
        with self.setting("look_warmup", "Warm up look outliers",
                          "Pictures the Dataset tab's Look consistency filter puts under its fence (they do not look "
                          "like the rest) stay in the dataset but start at a lower loss weight and reach full weight "
                          "after a few epochs, while the identity forms from the typical pictures.",
                          when=self.is_character):
            self.bind(ui.switch("Warm up look outliers"), "look_warmup")
        with self.setting(None, "Ideogram 4 captions",
                          "Ideogram 4 was trained on structured JSON captions only. A same-name .json caption in its "
                          "schema (Ideogram's prompting guide) trains it best and wins over the .txt; a plain .txt "
                          "caption is wrapped into the smallest valid JSON caption, with the trigger word in its "
                          "subject element. JSON captions can be long: Max caption tokens 2048 keeps them whole.",
                          show_if=("model_preset",
                                   lambda p: preset_arch(p, self.w["model_path"].value) == "ideogram4")):
            pass
        self.nav()

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
                    for label, value, sub in (("Images", str(n), "png, jpg, webp, bmp"),
                                              ("Captioned", f"{cap} / {n}", ".txt or .json next to the image"),
                                              ("Typical size", f"{mp:.2f} MP" if mp else "-", "median image area")):
                        v, s = stat_tile(label)
                        v.text, s.text = value, sub
                if not n:
                    ui.label("No images in these datasets yet. Add them on the Dataset tab (upload them, or copy a "
                             "folder), then scan again.").classes("rz-banner warning")
                elif cap < n:
                    ui.label(f"{n - cap} images have no caption yet. They train on the trigger word alone, which "
                             "works for a quick character LoRA; captions give a more flexible one.") \
                        .classes("rz-banner warning")
                if thumbs:
                    with ui.row().classes("w-full gap-2"):
                        for t in thumbs:
                            ui.image(image_url(t)).classes("w-24 h-24 rz-thumb")
                if self.goal == "character" and n:
                    with ui.row().classes("w-full items-center gap-3 no-wrap"):
                        if self.data["outliers"]:
                            ui.label(f"The saved look scans put {self.data['outliers']} of {n} pictures under the "
                                     "fence: they do not look like the rest. Review them on the Dataset tab "
                                     "(Look consistency filter), or ease them in below.").classes("rz-banner warning")
                        elif self.data["scanned"]:
                            ui.label("The saved look scans found no outliers.").classes("rz-banner ok")
                        else:
                            ui.label("Not scanned for look outliers yet: the Dataset tab's Look consistency filter "
                                     "finds the pictures whose face or body does not match the rest.") \
                                .classes("rz-banner info")
                        ui.button("Dataset tab", icon="photo_library", on_click=lambda: self.goto("Dataset")) \
                            .props("outline dense no-caps").classes("shrink-0")
        self.refresh()

    # ---- 5 training
    def step_training(self):
        self.heading("How to train", "How long and how hard to train, and how big the LoRA is.")
        self.apply_bar(["steps", "learning_rate", "optimizer", "network_type", "lokr_factor", "lora_rank",
                        "lora_alpha", "timestep_sampling", "batch_size", "target_text_fusion", "target_attn_gate",
                        "region_weights", "isolation"])
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
                          "automagic3 tunes its own learning rate from the weights' update directions. automagic4 "
                          "adds a smaller step for the most sensitive weights (timestep embedders, modulation, "
                          "attention values) and a controller whose rate holds on noise, climbs on a trend and "
                          "comes down when it overshoots (ai-toolkit's v4 ran away; fixed here)."):
            self.bind(ui.select(list(OPTIMIZERS), label="Optimizer"), "optimizer") \
                .classes("w-full")
        with self.setting("network_type", "Adapter",
                          "LoRA: a rank-r pair of small matrices per layer. LoKr: a Kronecker pair, about 1/64 of the "
                          "weights at factor 8: Fizgig's measured best likeness and more natural skin, about 20% "
                          "slower steps. OFT: rotates each layer's output channels in blocks instead of adding to the "
                          "weights; the OFT paper found it held a subject's identity better than LoRA and stayed "
                          "stable in long runs (not yet measured here: race it). All load in ComfyUI; diffusers has "
                          "no LoKr or OFT loader.",
                          when=lambda: not self.w["finetune"].value):
            self.bind(ui.select({"lora": "LoRA", "lokr": "LoKr", "oft": "OFT"}, label="Adapter"), "network_type") \
                .classes("w-full")
        with self.setting(None, "DoRA",
                          "Splits each layer's weight into a direction, which the LoRA / LoKr changes, and a "
                          "magnitude that trains on its own (weight-decomposed adaptation). Saved as keys ComfyUI "
                          "applies; diffusers cannot read them. Steps are somewhat slower. Race it with and without.",
                          when=lambda: not self.w["finetune"].value and self.w["network_type"].value != "oft"):
            self.bind(ui.switch("DoRA"), "dora")
        with self.setting(None, "OFT block size",
                          "Each layer's output channels rotate in blocks of this size. Larger blocks rotate more "
                          "channels together: more freedom and more weights (32: about as many as a rank 16 LoRA).",
                          when=lambda: not self.w["finetune"].value and self.w["network_type"].value == "oft"):
            self.bind(ui.number("Block size", min=2, max=256, precision=0), "oft_block_size").classes("w-full")
        with self.setting("lokr_factor", "LoKr factor",
                          "How the Kronecker pair splits each layer: the first factor is at most this. Lower is more "
                          "capacity and bigger files (8: about 400 MB on a 12B model, 16: about 100 MB). Fizgig found "
                          "no gain above 8.", when=lambda: not self.w["finetune"].value
                          and self.w["network_type"].value == "lokr"):
            self.bind(ui.number("LoKr factor", min=1, max=64, precision=0), "lokr_factor").classes("w-full")
        with self.setting("lora_rank", "LoRA rank",
                          "The size of the adapter. Higher ranks can hold more detail but give bigger files and "
                          "memorise the training images more easily.", when=self.is_lora):
            self.bind(ui.number("Rank", min=1, max=256, precision=0), "lora_rank").classes("w-full")
        with self.setting("lora_alpha", "LoRA alpha",
                          "Scales the adapter's effect (alpha / rank). Keeping alpha equal to rank is the safe "
                          "default.", when=self.is_lora):
            self.bind(ui.number("Alpha", min=0), "lora_alpha").classes("w-full")
        with self.setting("timestep_sampling", "Noise levels (timestep sampling)",
                          "Which noise levels the steps train on. logit_normal (the default) favours the middle; "
                          "logsnr spends most steps at very high noise, where an art style is learned (use it for "
                          "styles). The Train tab has the rest: identity, content, style, sigmoid, uniform, "
                          "reference.", when=lambda: self.goal != "character"):
            self.bind(ui.select(["logit_normal", "sigmoid", "uniform", "identity", "content", "style", "logsnr",
                                 "reference"], label="Timestep sampling"), "timestep_sampling").classes("w-full")
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
        is_qwen = ("model_preset", lambda p: preset_arch(p, self.w["model_path"].value) in ("qwen_image21",
                                                                                              "minimax_h3"))
        with self.setting("adapter_mode", "Training adapter",
                          "A small LoRA that stays on while your LoRA trains and off for samples and probes, and is "
                          "never saved into it, so your LoRA works on the plain model. Fizgig measured on Qwen-Image "
                          "2.1: without one, LoRAs collapse into texture or wobble part-way with no warning in the "
                          "loss; with Fizgig's adapter (downloaded, 80 MB) likeness reached 77 against 56. MiniMax "
                          "H3's is circlestone-labs' image training adapter (0.62 GB): musubi-tuner saw one-frame H3 "
                          "LoRAs break up within about 50 steps without an adapter.",
                          show_if=is_qwen):
            self.bind(ui.select(ADAPTER_MODES, label="Training adapter"), "adapter_mode").classes("w-full")
        with self.setting("region_weights", "Region-weighted loss",
                          "Weights what the model learns by region: the face counts most, the rest of the person "
                          "next, the background least, so the LoRA spends its capacity on the identity.",
                          show_if=("preset_toggle", bool)):
            self.bind(ui.switch("Region-weighted loss"), "region_weights")
        with self.setting("isolation", "LoRA Isolation",
                          "A character LoRA tends to put its subject's face on everyone else. Isolation trains each "
                          "picture again without its trigger, held to what the model does without the LoRA; fuses "
                          "the trigger into one word (prompt with it joined, e.g. gweltman); adds pictures of the "
                          "subject next to a generated stranger; and balances several subjects. Needs a trigger "
                          "word. About 1.5-2x the training time.", when=self.is_character):
            self.bind(ui.switch("LoRA Isolation"), "isolation")
        with self.setting(None, "Trigger attention on the subject",
                          "Reads, inside the model, where the trigger word's attention lands in each training "
                          "picture and pulls it onto the person (their face box and silhouette), so the trigger is "
                          "bound to them and not to the background or other people. Training-only: the LoRA file is "
                          "the same. Not yet measured: race it.", when=self.is_character):
            self.bind(ui.switch("Keep the trigger's attention on the subject"), "attn_localize")
        with self.setting(None, "Identity alignment",
                          "Teaches the face's tokens inside the model to carry that photo's identity (its ArcFace "
                          "embedding), through a small helper network that is thrown away afterwards. Identity "
                          "feedback at every noise level. Training-only. Not yet measured: race it.",
                          when=self.is_character):
            self.bind(ui.switch("Align the face tokens with the identity"), "id_align")
        with self.setting(None, "Body alignment",
                          "The same for the build: teaches the person's tokens inside the model to carry that photo's "
                          "body (its DINOv2 body-window embedding, what the body probe scores), through a helper that "
                          "is thrown away. Only photos showing the neck to mid-thigh count. Not yet measured: race it.",
                          when=self.has_body):
            self.bind(ui.switch("Align the body tokens with the build"), "body_align")
        hint("Schedules, LoRA layers and blocks, the full fine-tune and per-image loss watch are on the Train tab.")
        self.nav()

    # ---- 6 samples
    def step_samples(self):
        self.heading("Checkpoints and preview images", "What gets saved along the way, and the pictures that show "
                                                       "how training is going.")
        # the sampling model first: the step and guidance recommendations depend on it
        self.apply_bar(["sample_model_path", "save_every", "keep_checkpoints", "lora_ema", "sample_every",
                        "sample_steps", "sample_cfg", "probe_every", "probe_steps", "probe_cfg", "probe_face_prompt",
                        "probe_body_prompt"])
        with self.setting("save_every", "Save a checkpoint every",
                          "Steps between saved LoRAs (output/<job>/<job>_<step>.safetensors). Several of them let you "
                          "pick the last one before it started to overfit."):
            self.bind(ui.number("Save every", min=1, precision=0, suffix="steps"), "save_every").classes("w-full")
        with self.setting("keep_checkpoints", "Checkpoints to keep",
                          "Keeps only the newest N of the step checkpoints, deleting the oldest as new ones are "
                          "saved. The final file and the best-likeness file are never deleted. 0 keeps every one."):
            self.bind(ui.number("Keep", min=0, precision=0), "keep_checkpoints").classes("w-full")
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
        with self.setting(None, "Turbo LoRA",
                          "A step-distilled LoRA used only while rendering samples and probes, never in training: "
                          "previews take seconds. The model's own is downloaded (MOSS-Image: 12 steps at CFG 1; "
                          "Krea 2, Qwen-Image 2.1, Ideogram 4 and MiniMax H3: 8; FIBO 2.0: 4; FLUX 3 Klein has "
                          "none) and the steps and guidance below follow it. "
                          "For MOSS-Image, Fizgig found it can understate likeness: judge likeness on full-quality "
                          "samples."):
            self.bind(ui.select(TURBO_MODES, label="Turbo LoRA"), "turbo_mode").classes("w-full")
            self.bind(path_field("Turbo LoRA file", "file", width="w-full"), "turbo_lora") \
                .parent_slot.parent.bind_visibility_from(self.w["turbo_mode"], "value", lambda x: x == "file")
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
                          "A plain headshot of your character, rendered with a fixed seed. Empty = no face "
                          "probe.", when=self.has_face):
            self.bind(ui.textarea("Face probe prompt").props("autogrow"), "probe_face_prompt").classes("w-full")
        with self.setting("probe_body_prompt", "Body probe prompt",
                          "A full-body, upright shot, for the body score. Empty = no body probe.",
                          when=self.has_body):
            self.bind(ui.textarea("Body probe prompt").props("autogrow"), "probe_body_prompt").classes("w-full")
        with self.setting("probe_steps", "Probe steps", "Denoising steps per probe.", show_if=is_identity):
            self.bind(ui.number("Probe steps", min=1, precision=0), "probe_steps").classes("w-full")
        with self.setting("probe_cfg", "Probe guidance (CFG)", "Guidance for the probes.", show_if=is_identity):
            self.bind(ui.number("Probe CFG", min=0, step=0.5), "probe_cfg").classes("w-full")
        with self.setting(None, "Self-training on its best pictures",
                          "Halfway through, the LoRA draws about 30 pictures of your person in new scenes; the few "
                          "that are clean, sharp and look more like them than the probes do train beside your photos "
                          "for the rest of the run, at half weight and never more than a fifth of them. Cheap (a few "
                          "minutes), but it can copy the model's own flaws: judge the samples by eye. Not yet "
                          "measured: race it.", show_if=is_identity):
            self.bind(ui.switch("Self-train on its best pictures"), "selftrain")
        with self.setting(None, "Second stage: likeness reward fine-tuning",
                          "After the steps, the LoRA renders groups of pictures of new prompts, the probes rate how "
                          "much each looks like your person, and the LoRA learns to make the better ones more likely. "
                          "It aims straight at likeness, but it is slow (a few seconds to a minute a step) and can "
                          "learn to please the scorer instead of the eye: the probes keep the best checkpoint, and "
                          "the first stage's LoRA is kept. 0 = off; try 100-200.", show_if=is_identity):
            self.bind(ui.number("Reward steps", min=0, precision=0), "reward_steps").classes("w-full")
        with self.setting(None, "Early stop", "Stop after this many probes in a row without improvement (0 = never: "
                                              "run all the steps). The best checkpoint is kept either way.",
                          show_if=is_identity):
            self.bind(ui.number("Probes without improvement", min=0, precision=0), "likeness_patience") \
                .classes("w-full")
        self.nav()

    def suggest(self):
        replaced = bool((self.w["sample_prompts"].value or "").strip())
        self.w["sample_prompts"].value = "\n".join(suggested_prompts(self.goal, (self.w["trigger_word"].value or "")
                                                                     .strip(), self.focus))
        if replaced:
            ui.notify("Replaced the sample prompts with suggestions")

    # ---- race: the racers
    def step_racers(self):
        self.heading("The racers", "Every racer starts from the setup on the earlier pages, the base racer, and "
                                   "changes only what you assign it below. They train in turn on the same images and "
                                   "probes, and the likeness probes pick the winner.")
        with self.setting(None, "How many racers",
                          "Each racer trains up to the full steps, so every one adds to the time (the rules page "
                          "shows the total). Two to four is a good race."):
            self.racer_count = ui.number("Racers", value=self.n_racers, min=2, max=8, precision=0) \
                .classes("w-full")
            self.racer_count.on_value_change(lambda e: self.set_racer_count(e.value))
        with self.setting(None, "What differs between them",
                          "Pick the settings the racers may change; each racer below gets a field for each, filled "
                          "in with the base value. Common picks: learning rate, LoRA rank, adapter type, or seed (the "
                          "same setup trained twice, for a seed soup). Everything else stays the same for all."):
            self.race_picker = ui.select(race_choice_options(), multiple=True, value=list(self.race_keys),
                                         label="Settings", with_input=True).props("use-chips").classes("w-full")
            self.race_picker.on_value_change(lambda e: self.set_race_keys(e.value))
        self.racer_grid = ui.element("div").classes("rz-racers w-full")
        self.race_check = ui.label("").classes("rz-banner info")
        self.nav()

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

    # ---- race: the rules
    def step_rules(self):
        self.heading("The race's rules", "When racers leave the race, and how the winner is chosen.")
        with self.setting(None, "Relay: everyone continues from the round's winner",
                          "Nobody is dropped. After each round every racer takes over the round winner's LoRA "
                          "(weights, optimizer state, weight average, likeness history) and trains on with its own "
                          "settings, seed, image order and noise. With racers that differ only in their seed, each "
                          "round tries a few random paths from the best LoRA so far and keeps the best one. Every "
                          "racer trains all the steps (one run's cost per racer); the last round's winner is the run's "
                          "LoRA. Racers must share the LoRA's shape: no rank, alpha, layer or network type rows."):
            self.bind(ui.switch("Relay: continue from the round's winner"), "race_follow")
        with self.setting(None, "Drop the lowest racer after each round",
                          "Every racer trains until its next probe, then the next one does; after a round the one "
                          "with the lowest likeness score leaves. Fast and cheap, but a racer that learns slowly and "
                          "would end higher can be cut early. Off: nobody is cut for being behind and every racer "
                          "can train all the steps (up to one full run each; the steps must be a multiple of the "
                          "probe interval)."):
            self.bind(ui.switch("Drop the lowest after each round"), "race_drop_lowest")
        with self.setting(None, "Warm-up probe rounds",
                          "Rounds before the first racer is dropped: everyone gets a few probes to get going. 1-2 is "
                          "plenty.", show_if=("race_drop_lowest", bool)):
            self.bind(ui.number("Warm-up rounds", min=0, precision=0), "race_warmup_probes").classes("w-full")
        with self.setting(None, "Drop a racer when it stops improving",
                          "A racer leaves once its own probes have not improved for a number of probes in a row, "
                          "however it compares with the others; the rest race on. Good with slow starters. With it, "
                          "or without dropping the lowest, the winner is the racer whose score was highest at any "
                          "point, and its best checkpoint becomes <job>_best.safetensors."):
            self.bind(ui.switch("Drop a racer when it stops improving"), "race_patience_drop")
        with self.setting(None, "After how many probes without improvement",
                          "Probes in a row without a gain bigger than the noise band. 3 gives a racer a fair chance; "
                          "lower ends the race sooner.", show_if=("race_patience_drop", bool)):
            self.bind(ui.number("Probes", min=1, precision=0), "race_patience").classes("w-full")
        with self.setting(None, "Keep it while the likeness is rising again",
                          "A configuration out of patience is not dropped while its smoothed likeness is still going "
                          "up from its previous probe: a chance to come back after a dip. It leaves at the first probe "
                          "that does not rise (unless it beats its best, which resets the count).",
                          show_if=("race_patience_drop", bool)):
            self.bind(ui.switch("Keep a racer whose likeness is rising"), "race_patience_rising")
        with self.setting(None, "One racer per GPU at the same time",
                          "With data parallel on 2+ GPUs: each GPU trains its own racer at once instead of all of "
                          "them taking turns over every GPU. Not with the identity loss or self-training.",
                          show_if=("multi_gpu", lambda x: x == "data_parallel")):
            self.bind(ui.switch("One racer per GPU"), "race_parallel")
        self.race_estimate = ui.label("").classes("rz-banner info")
        for key in ("race_drop_lowest", "race_warmup_probes", "race_patience_drop", "race_patience", "race_follow",
                    "race_patience_rising"):
            self.w[key].on_value_change(lambda _: self.estimate_race())
        self.nav()

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
                                       "The race runs on the likeness probes: set a probe interval (Samples page).")
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

    # ---- 7 review
    def step_review(self):
        self.heading("Review and start", "A last look at the run before it starts.")
        self.review_box = ui.column().classes("w-full gap-3")
        with ui.row().classes("w-full items-center gap-3 pt-2"):
            self.cfg_path = ui.input("Save as", value="").classes("w-80 max-w-full")
            ui.button("Save config", icon="save", on_click=self.save).props("outline")
            ui.space()
            ui.button("Adjust in all settings", icon="tune", on_click=lambda: self.goto("Train")).props("outline") \
                .tooltip("The Train tab has every setting, already filled in from these pages")
            if self.enqueue is not None:
                ui.button("Add to queue", icon="playlist_add", on_click=self.enqueue).props("outline size=lg") \
                    .tooltip("Start this run after the current one (and any queued before it) is done")
            self.start_btn = ui.button("Start the race" if self.racing() else "Start training", icon="play_arrow",
                                       on_click=self.start).props("unelevated size=lg")
        with ui.stepper_navigation():
            ui.button("Back", icon="arrow_back", on_click=lambda: self.move(-1)).props("flat")

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
            ("Goal", (GOALS[self.goal][1] + (f": {CHARACTER_FOCUS[self.focus][1].lower()}"
                                             if self.goal == "character" else "")) if self.goal
             else "not chosen (general settings)"),
            ("Job", f"{cfg.job_name}  ->  {_relative(str(cfg.job_dir()))}/"),
            ("Model", f"{model} ({'single files' if cfg.model_source == 'files' else 'Hugging Face'})"),
            ("GPUs", f"{multi}; model quant {cfg.transformer_quant}, text encoder quant {cfg.text_encoder_quant}"),
            ("Datasets" if len(ds_paths) > 1 else "Dataset", datasets_line(cfg, data)),
            ("Trigger word", cfg.trigger_word or "(none)"),
            ("Resolution", ", ".join(f"{mp:g} MP" for mp in cfg.resolutions)
             + (" (datasets with their own: those)" if any(sp.resolutions for sp in cfg.dataset_specs()) else "")),
            ("Training", f"{cfg.steps} steps, batch {cfg.batch_size}, lr {cfg.learning_rate:g}, {cfg.optimizer}"),
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


# ---------------------------------------------------------------- previous runs

HISTORY_COLORS = ["#FF3B30", "#60A5FA", "#34D399", "#F59E0B", "#C084FC", "#22D3EE", "#F472B6", "#A3E635", "#94A3B8",
                  "#FB923C"]
HISTORY_COLUMNS = [
    {"name": "run", "label": "Run", "field": "run", "align": "left", "sortable": True},
    {"name": "started", "label": "Started", "field": "started", "align": "left", "sortable": True},
    {"name": "state", "label": "State", "field": "state", "align": "left", "sortable": True},
    {"name": "steps", "label": "Steps", "field": "steps", "align": "right"},
    {"name": "adapter", "label": "Adapter", "field": "adapter", "align": "left"},
    {"name": "lr", "label": "LR", "field": "lr", "align": "right"},
    {"name": "face", "label": "Face", "field": "face", "align": "right"},
    {"name": "body", "label": "Body", "field": "body", "align": "right"},
    {"name": "combined", "label": "Combined", "field": "combined", "align": "right"},
    {"name": "best", "label": "Best", "field": "best", "align": "right"},
]


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
    body dashed, combined thick). part is "identity" (the sample images' means) or "probes"."""
    styles = {"face": {}, "body": {"type": "dashed"}, "combined": {"width": 3}}
    out = []
    n_color = 0
    for r in runs:
        data = r.get(part) or {}
        racers = data.get("racers") if part == "probes" else None
        # a race: each configuration's own lines in its own colour ("<run> lora16 - face"), else the run's
        groups = [(f"{history_label(r)} {name}", d) for name, d in racers.items()] if racers else \
            [(history_label(r), data)]
        for label, d in groups:
            color = HISTORY_COLORS[n_color % len(HISTORY_COLORS)]
            n_color += 1
            for metric in metrics:
                pts = d.get(metric) or []
                if not pts:
                    continue
                out.append({"name": f"{label} - {metric}", "type": "line", "data": pts,
                            "showSymbol": part == "identity", "symbolSize": 6, "itemStyle": {"color": color},
                            "lineStyle": {"color": color, "width": 2, **styles[metric]}})
    return out


METRIC_LINE = {"face": "solid", "body": "dashed", "combined": "thick"}  # history_series' line style per metric


class MetricChip:
    """An on/off chip for one metric's lines on a Previous Runs chart, for every ticked run at once (`.value` like a
    checkbox's)."""

    def __init__(self, metric: str, on_change):
        self.value = True
        self._on_change = on_change
        self.button = ui.button(on_click=self.toggle, color=None).props("flat no-caps").classes("rz-metric-chip")
        with self.button:
            ui.html(f'<span class="rz-line rz-line-{METRIC_LINE[metric]}"></span>')
            ui.label(metric.capitalize())
        self.button.tooltip(f"Show or hide the {metric} line of every ticked run")
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


def metric_chips(metrics, on_change) -> dict[str, MetricChip]:
    """A "Show lines (all runs)" row of MetricChips: {metric: chip}."""
    ui.label("Show lines (all runs):").classes("text-sm font-medium")
    return {m: MetricChip(m, on_change) for m in metrics}


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
    f2 = lambda x: "-" if x is None else f"{x:.0f}"  # noqa: E731
    return f"overall {f2(entry.get('combined'))} | face {f2(entry.get('face'))} | body {f2(entry.get('body'))}"


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
            chart.options["legend"]["data"] = [sr["name"] for sr in series]
            chart.update()
            empty.text = ("Tick one or more runs above to draw them here." if not picked else
                          "" if series else "The ticked runs have no scores of this kind.")
        compare.text = f"{len(picked)} run(s) overlaid" if picked else ""
        draw_probes()

    def draw_probes(*_):
        picked = selected_runs()
        groups = probe_image_groups(picked)
        kind = probe_kind.value
        steps = probe_image_steps(groups, kind)
        opts = {s: f"step {s:,}" for s in steps}
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
            "Pick any two images (any run, any step) to compare them large"
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
            for label, run, entries in groups:
                at, imgs = probe_images_at(entries, step, kind)
                with ui.row().classes("w-full items-start gap-2 no-wrap"):
                    with ui.column().classes("w-44 shrink-0 gap-0"):
                        ui.label(label).classes("text-sm font-medium break-words")
                        if at is None:
                            hint(f"no {kind} probe by step {step:,}")
                        elif at != step:
                            hint(f"its latest {kind} probe before: step {at:,}")
                    with ui.row().classes("gap-2"):
                        for e in imgs:
                            path = history.HISTORY_DIR / e["file"]
                            picked_now = str(path) in state["picks"]
                            with ui.card().tight().classes("rz-thumb" + (" rz-picked" if picked_now else "")):
                                ui.image(image_url(path)).props("fit=contain") \
                                    .classes("w-48 h-48 rz-imgbg cursor-pointer") \
                                    .on("click", lambda _, q=path: show_full(q))
                                with ui.row().classes("w-full items-center no-wrap px-1 gap-1"):
                                    ui.label(f"seed {e['seed'] + 1}  {probe_caption(e)}").classes("text-xs rz-hint")
                                    ui.space()
                                    ui.button(icon="check_box" if picked_now else "check_box_outline_blank",
                                              on_click=lambda _, q=str(path), c=f"{label}, step {e['step']:,}, "
                                              f"{kind} seed {e['seed'] + 1}: {probe_caption(e)}": toggle_pick(q, c)) \
                                        .props("flat dense round size=sm").tooltip("Pick to compare")

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
        count.text = (f"{len(state['runs'])} saved run(s) in {history.HISTORY_DIR}" if state["runs"] else
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
                hint("Every run with likeness scores is saved here while it trains, even when a later run reuses its "
                     "job name. Tick any number of runs to overlay the mean face and body of their samples and the "
                     "smoothed face, body and combined score of their probes, to see which setup learned the likeness "
                     "faster or higher. Scores are on the run's own normalized scale (0 = a stranger, 100 = the "
                     "references).")
            ui.button("Import old runs", icon="move_to_inbox", on_click=import_old).props("outline") \
                .tooltip("Add the runs whose job folders are in the output folder but not in this list yet")
    with card():
        count = hint("")
        table = ui.table(columns=HISTORY_COLUMNS, rows=[], row_key="id", selection="multiple",
                         pagination={"rowsPerPage": 10, "sortBy": "started", "descending": True},
                         on_select=lambda _: apply_scale()).props("flat dense").classes("w-full")
        with ui.row().classes("w-full items-center gap-2"):
            ui.button("Rename", icon="edit", on_click=rename).props("flat dense")
            ui.button("Export spreadsheet", icon="table_view", on_click=export_runs).props("flat dense") \
                .tooltip("The ticked runs' likeness spreadsheets (one file each), the same as the Monitor's: every "
                         "probe's face, body and combined score as formulas of the raw scores and the run's scale, "
                         "editable on the Scale sheet. Works after the job folder is gone or reused.")
            ui.button("Remove from history", icon="delete", on_click=delete).props("flat dense color=negative")
            ui.button("Clear ticks", icon="deselect",
                      on_click=lambda: (setattr(table, "selected", []), table.update(), draw())).props("flat dense")
            ui.space()
            compare = hint("")
    with section("Identity similarity", "Mean face and mean body score of the sample images", "face"):
        with ui.row().classes("items-center gap-3"):
            id_metrics = metric_chips(("face", "body"), lambda: draw())
            id_empty = hint("")
        id_chart = ui.echart(chart_options(_axis("step", min="dataMin"), _axis("similarity %", min=0), [],
                                           [])).classes("w-full h-72")
    with section("Likeness probes & controller", "Smoothed face, body and combined score of the probes", "show_chart"):
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
        probe_chart_h = ui.echart(chart_options(_axis("step", min="dataMin"), _axis("likeness %", min=0),
                                                [], [])).classes("w-full h-72")
    with section("Probe images", "The ticked runs' probes at the same step side by side; pick two to compare",
                 "compare"):
        with ui.row().classes("w-full items-center gap-2"):
            probe_kind = ui.toggle({"face": "Face", "body": "Body"}, value="face",
                                   on_change=lambda _: draw_probes()).props("dense no-caps")

            def step_by(delta: int):
                keys = list(probe_step.options)
                if probe_step.value in keys:
                    i = min(len(keys) - 1, max(0, keys.index(probe_step.value) + delta))
                    probe_step.value = keys[i]

            ui.button(icon="chevron_left", on_click=lambda: step_by(-1)).props("flat dense round") \
                .tooltip("Earlier probe")
            probe_step = ui.select({}, label="Probe step", on_change=lambda _: draw_probes()).classes("w-40")
            ui.button(icon="chevron_right", on_click=lambda: step_by(1)).props("flat dense round") \
                .tooltip("Later probe")
            ui.space()
            pick_label = hint("")
            compare_btn = ui.button("Compare", icon="compare", on_click=open_compare).props("outline dense no-caps")
            ui.button("Clear picks", icon="deselect", on_click=clear_picks).props("flat dense no-caps")
        probe_warn = hint("", warn=True)
        probe_grid = ui.column().classes("w-full gap-3")
    with ui.dialog().props("maximized") as cmp_dlg, ui.card().tight().classes("w-full h-full bg-black"):
        with ui.row().classes("w-full items-center justify-between px-3 py-1"):
            ui.label("Compare").classes("text-sm text-white")
            ui.button(icon="close", on_click=cmp_dlg.close).props("flat dense round color=white")
        with ui.row().classes("w-full no-wrap gap-2 px-2"):
            cmp_imgs, cmp_caps = [], []
            for _ in range(2):
                with ui.column().classes("w-1/2 items-center gap-1"):
                    cmp_imgs.append(ui.image("").props("fit=contain").classes("w-full")
                                    .style("height: calc(100vh - 96px)"))
                    cmp_caps.append(ui.label("").classes("text-xs text-white text-center"))
    show_full = image_viewer()
    charts.extend([id_chart, probe_chart_h])
    try:
        history.import_jobs()  # the runs from before the history existed, once
    except Exception:  # noqa: BLE001
        pass
    reload(True)
    ui.timer(4.0, reload)


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
    look = {"light": False}  # this browser's look, read from its localStorage once the page is connected
    charts: list = []  # ECharts to recolor when the look changes

    def set_look(light: bool, save: bool = True) -> None:
        look["light"] = light
        ui.run_javascript(f"rzSetTheme({json.dumps('light' if light else 'dark')}, {json.dumps(save)})")
        for chart in charts:
            recolor_chart(chart, light)

    with ui.header().classes("rz-header"):
        with ui.row().classes("rz-header-row w-full items-center no-wrap gap-2"):
            ui.element("img").props('src="/assets/logo_dark.png" alt="Rulz"').classes("rz-logo rz-dark-only")
            ui.element("img").props('src="/assets/logo.png" alt="Rulz"').classes("rz-logo rz-light-only")
            ui.label("TRAINER").classes("rz-wordmark")
            ui.label("LoRA training for MOSS-Image, Bria FIBO 2.0, Krea 2, Qwen-Image 2.1, FLUX 3 Klein, "
                     "Ideogram 4 and MiniMax H3") \
                .classes("rz-tagline gt-sm")
            ui.space()
            # sun in the dark look (switch to light), moon in the light look; CSS picks the icon, so it is right
            # from the first paint
            with ui.button(on_click=lambda: set_look(not look["light"]), color=None) \
                    .props("flat round dense").classes("rz-theme-btn") as theme_btn:
                ui.icon("light_mode").classes("rz-dark-only")
                ui.icon("dark_mode").classes("rz-light-only")
            theme_btn.tooltip("Switch between the dark and the light look")
            with ui.row().classes("rz-pill items-center gap-2 no-wrap") as pill:
                pill_dot = ui.element("span").classes("rz-dot")
                pill_text = ui.label("Idle").classes("rz-pill-text")
        with ui.tabs().props("inline-label no-caps align=left dense").classes("rz-tabs") as tabs:
            tab_wizard = ui.tab("Wizard", icon="auto_awesome")
            tab_train = ui.tab("Train", icon="tune")
            tab_dataset = ui.tab("Dataset", icon="photo_library")
            tab_monitor = ui.tab("Monitor", icon="show_chart")
            tab_samples = ui.tab("Samples", icon="collections")
            tab_history = ui.tab("Previous Runs", icon="history")
            tab_help = ui.tab("Help", icon="help_outline")

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

    def start_job() -> bool:
        cfg = form_config()
        if cfg is None:
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
                                 "model); FLUX 3 Klein: BFL's own black-forest-labs/flux-3-image-klein-comfy), "
                                 "downloaded on the first "
                                 "run. To use files you already have, set them below: anything left empty is "
                                 "downloaded.")
                            path_input(f, "transformer_file", "Diffusion model file (optional; empty = download)",
                                       "file", "", "w-[40rem]")
                            path_input(f, "text_encoder_file", "Text encoder file (optional; Qwen3.5-9B for MOSS-Image, "
                                                               "Qwen3-VL-4B for Krea 2, FIBO 2.0 and FLUX 3 Klein, "
                                                               "Qwen3-VL-8B for Qwen-Image 2.1 and Ideogram 4, "
                                                               "Qwen3-VL-32B for MiniMax H3)", "file", "",
                                       "w-[40rem]")
                            path_input(f, "vae_file", "VAE file (optional; Flux2 for MOSS-Image, FIBO 2.0, FLUX 3 "
                                                      "Klein and Ideogram 4, Qwen-Image for Krea 2, its own for "
                                                      "Qwen-Image 2.1, the video VAE for MiniMax H3)", "file", "",
                                       "w-[40rem]")
                            path_input(f, "tokenizer_path", "Tokenizer folder (optional; auto-downloads if empty)",
                                       "dir", "", "w-[40rem]")
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
                                         "off.")
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
                    with section("Dataset", "Images, captions and resolution", "photo_library") as sec:
                        nav_items.append(("Dataset", "photo_library", sec))
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
                    with section("LoRA", "Adapter size, layers and blocks", "hub") as sec:
                        nav_items.append(("LoRA", "hub", sec))
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
                                         "MiniMax H3's is circlestone-labs' image training adapter (0.62 GB). The "
                                         "other models have none of their own; a file works on any.")
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
                            w["optimizer"] = ui.select(list(OPTIMIZERS), value="adamw8bit", label="Optimizer") \
                                .classes("w-40")
                            w["lr_scheduler"] = ui.select(["constant", "cosine", "linear"], value=d.lr_scheduler,
                                                          label="LR schedule").classes("w-36")
                            w["warmup_steps"] = ui.number("Warmup steps", value=d.warmup_steps, precision=0) \
                                .classes("w-32")
                        with ui.column().classes("w-full gap-2").bind_visibility_from(
                                w["optimizer"], "value", lambda x: x in AUTOMAGIC):
                            hint("Automagic tunes its own learning rate (the learning rate above is its start); LR "
                                 "schedule and warmup are ignored. Automagic 4 also caps the step of the most "
                                 "sensitive weights and its rate holds on noise instead of drifting up.")
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
                            with fields().bind_visibility_from(w["optimizer"], "value", lambda x: x == "automagic4"):
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
                            with fields():
                                w["clean_share"] = ui.input("Clean-end share (%, by epoch)", value=d.clean_share,
                                                            placeholder="empty = the sampler's own") \
                                    .classes("w-64").tooltip(
                                        "The share of training steps at the clean end (noise below half), where faces "
                                        "and fine detail are refined; the sampling curve above is moved to hit it. "
                                        "40 = 40% all run. 75:5:40 = 75% for 5 epochs, then 40%. 60:7:30:15:2 = "
                                        "Fizgig's MOSS-Image Slide: 60% until epoch 7, down to 30% by epoch 15, and "
                                        "every 2nd epoch after that one epoch at 60% again.")
                                ui.button("Slide", on_click=lambda: w["clean_share"].set_value(CLEAN_SLIDE)) \
                                    .props("flat dense").tooltip("Fizgig's MOSS-Image default, 60:7:30:15:2")
                                clean_hint = hint("")

                                def clean_hint_update(_=None):
                                    clean_hint.text = clean_share_hint(f)

                                ui.timer(0.1, clean_hint_update, once=True)
                                for key in ("timestep_sampling", "logit_mean", "logit_std", "sigmoid_scale",
                                            "content_power", "sigma_min", "sigma_max", "timestep_shift",
                                            "clean_share"):
                                    w[key].on_value_change(clean_hint_update)

                    # ---- Saving & samples
                    with section("Saving & samples", "Checkpoints and preview images", "save") as sec:
                        nav_items.append(("Saving & samples", "save", sec))
                        with fields():
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
                        hint("MOSS-Image: lodestones' turbo LoRA renders 12 steps at CFG 1 on Base. Fizgig found it "
                             "can understate likeness: judge a run's likeness on full-quality samples (Turbo LoRA: "
                             "none, 20 steps, CFG 3).").bind_visibility_from(
                            w["model_preset"], "value", lambda p: preset_arch(p, w["model_path"].value) == "moss_image")
                        hint("FLUX 3 Klein: no turbo LoRA. Base samples 50 steps at CFG 3.5 (BFL's settings, about 2.5 "
                             "min per 1 MP preview on a 16 GB card). Faster previews: set the distilled "
                             "black-forest-labs/flux-3-image-klein (or its -comfy flux3-image-klein-*.safetensors) as "
                             "the sampling model below and sample 16 steps at CFG 1.").bind_visibility_from(
                            w["model_preset"], "value",
                            lambda p: preset_arch(p, w["model_path"].value) == "flux3_klein")
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
                                                         "proportions": "proportions (2D skeleton)",
                                                         "skeleton3d": "proportions 3D (skeleton + silhouette)",
                                                         "smplx": "SMPL-X body shape (NLF)"},
                                                        value=d.body_score, label="Body score").classes("w-64") \
                                .tooltip("appearance: a DINOv2 embedding of the body from the neck to mid-thigh, "
                                         "turned upright and cut out from the background, best match against the "
                                         "reference photos that show the same part of the body (build, skin, hair; "
                                         "clothing and nudity too, so vary the outfits). proportions (2D): skeleton "
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
                    with section("Per-image loss watch", "Per-image verdicts and learning rate", "rule",
                                 open_=False) as sec:
                        nav_items.append(("Loss watch", "rule", sec))
                        hint("Tracks every image's loss normalised by its timestep bucket and, at each epoch boundary, "
                             "classes it as easy / healthy / suspect / stuck / exhausted. Per-image LR then throttles "
                             "stuck images (x0.5, x0.25, x0.125), eases exhausted ones (x0.6) and can exclude images "
                             "that stay stuck. A true per-image learning rate needs batch size 1 (otherwise per-sample "
                             "loss weights). Not available with automagic3 or automagic4.")
                        with fields():
                            w["loss_watch"] = ui.switch("Loss watch (observe)", value=d.loss_watch)
                            w["per_image_lr"] = ui.switch("Per-image learning rate (act on verdicts)",
                                                          value=d.per_image_lr)
                            w["loss_watch_exclude_after"] = ui.number("Exclude after N epochs at lowest throttle "
                                                                      "(0 = never)", value=d.loss_watch_exclude_after,
                                                                      precision=0).classes("w-96 max-w-full")
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
                                w["loss_watch_low"] = ui.number("Easy below median x", value=d.loss_watch_low,
                                                                step=0.05).classes("w-44")
                                w["loss_watch_improve_tol"] = ui.number("Improving = drop >",
                                                                        value=d.loss_watch_improve_tol,
                                                                        step=0.01).classes("w-40")
                                w["loss_watch_healthy_mult"] = ui.number("Healthy multiplier",
                                                                         value=d.loss_watch_healthy_mult,
                                                                         step=0.05).classes("w-40")

                    # ---- LoRA Isolation
                    with section("LoRA Isolation", "Keep your subject(s) off everyone else in a picture",
                                 "group_off", open_=False) as sec:
                        nav_items.append(("LoRA Isolation", "group_off", sec))
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
                                                         precision=0).classes("w-60")                                 .tooltip("Also probed once at the last step, so the final LoRA has its own score")
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
                                w["probe_seeds"] = ui.number("Seeds per probe", value=d.probe_seeds, min=1, max=8,
                                                             precision=0).classes("w-36")
                                w["probe_floor_seeds"] = ui.number("Step-0 seeds", value=d.probe_floor_seeds, min=1,
                                                                   max=16, precision=0).classes("w-32") \
                                    .tooltip("The step-0 probe (the base model: the 0 of the normalized scale) renders "
                                             "at least this many seeds, for a steadier zero; later probes render "
                                             "Seeds per probe. Costs a few extra renders once, at the start.")
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
                            w["race_drop_lowest"] = ui.switch("Drop the lowest after each round",
                                                              value=d.race_drop_lowest) \
                                .tooltip("Off: nobody is dropped for being behind; every configuration trains all the "
                                         "steps (unless the patience drop below ends it) and the winner is the highest "
                                         "score any of them reached. Costs up to one full run per configuration.")
                            w["race_warmup_probes"] = ui.number("Warm-up probe rounds (nobody is dropped)",
                                                                value=d.race_warmup_probes, min=0, precision=0) \
                                .classes("w-72")
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
                            w["race_drop_lowest"].set_visibility(not relay)
                            w["race_warmup_probes"].set_visibility(not relay and bool(w["race_drop_lowest"].value))
                            w["race_patience_drop"].set_visibility(not relay)
                            w["race_patience"].set_visibility(not relay and bool(w["race_patience_drop"].value))
                            w["race_patience_rising"].set_visibility(not relay
                                                                     and bool(w["race_patience_drop"].value))

                        for key in ("race_follow", "race_drop_lowest", "race_patience_drop"):
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
                                    f"everyone continues from its winner; {est['total']} training steps in all "
                                    f"({est['ratio']:.0f}x the run)."
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
                                    "race_patience_drop", "race_patience", "race_follow", "race_patience_rising"):
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
                    with section("Seed soup", "Average finished LoRAs of one subject into one",
                                 "blender", open_=False) as sec:
                        nav_items.append(("Seed soup", "blender", sec))
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
            Wizard(f, gpus, start_job, goto, queue_job).build()

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
                    .props("rounded")
                error_box = ui.label("").classes("rz-banner error")
                error_box.visible = False
                gpu_box = ui.row().classes("w-full gap-3")
                gpu_widgets: list[dict] = []
                with section("Queue", "Runs that start one after another when the current one is done",
                             "queue") as queue_exp:
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
                    charts.append(loss_chart)
                with section("Identity similarity", "Face / body / combined score of every sample", "face") as face_exp:
                    face_label = hint("")
                    face_chart = ui.echart(chart_options(_axis("step", min="dataMin"),
                                                         _axis("similarity %", min=0, max=100), [], [])) \
                        .classes("w-full h-60")
                    charts.append(face_chart)
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
                        {"name": "at", "label": "At step", "field": "at"},
                        {"name": "score", "label": "Score now", "field": "score"},
                        {"name": "face", "label": "Face now", "field": "face"},
                        {"name": "body", "label": "Body now", "field": "body"},
                        {"name": "rounds", "label": "Score after each round", "field": "rounds", "align": "left"},
                        {"name": "settings", "label": "Settings", "field": "settings", "align": "left"},
                        {"name": "drop", "label": "", "field": "can_drop"}],
                        rows=[], row_key="name").classes("w-full")
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
                race_exp.visible = False
                with section("Likeness probes & controller", "Smoothed probe scores and controller state",
                             "insights") as probe_exp:
                    with ui.row().classes("w-full items-center gap-2"):
                        probe_label = hint("").classes("flex-1")

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
                    # the normalized scores (0 = a stranger, 100 = the references): smoothed lines, per-probe dots
                    probe_chart = ui.echart(chart_options(
                        _axis("step", min="dataMin"), _axis("likeness % (normalized)", min=0, max=100),
                        *probe_chart_series([], []))).classes("w-full h-60")
                    charts.append(probe_chart)
                    probe_row = ui.row().classes("w-full gap-2")
                probe_exp.visible = False
                shown_probes: list[str] = []
                with section("Latest samples", "Click an image to see it full size", "image"):
                    with ui.row().classes("w-full gap-2") as samples_row:
                        hint("No samples yet. They appear here every 'Sample every' steps.")
                with section("Log and saved files", "train.log, checkpoints and training states", "terminal"):
                    downloads_row = ui.row().classes("w-full gap-2 items-center")
                    saved_label = ui.label("").classes("rz-mono whitespace-pre-wrap")
                    log_view = ui.log(max_lines=600).classes("w-full h-80 rz-log rz-mono")
            shown_samples: list[str] = []
            shown_downloads: list = []

            def show_downloads(saved: list[str]) -> None:
                """A download button per saved LoRA file that is still there (rolling saves delete old ones)."""
                rows = []
                for p in saved:
                    try:
                        rows.append((p, Path(p).stat().st_size))
                    except OSError:
                        continue
                if rows == shown_downloads:
                    return
                shown_downloads[:] = rows
                downloads_row.clear()
                with downloads_row:
                    if rows:
                        ui.label("Download:").classes("rz-hint")
                    for p, size in rows:
                        ui.button(f"{Path(p).name} ({size / 2**20:.0f} MB)", icon="download",
                                  on_click=lambda _, q=p: ui.download(lora_url(q), Path(q).name), color=None) \
                            .props("flat no-caps").classes("rz-file-btn") \
                            .tooltip("Save this LoRA to this computer (useful when the WebUI runs remotely)")

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

            ui.timer(2.0, queue_draw)
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
                                name = ui.label("").classes("text-sm font-semibold rz-strong")
                                bars = {}
                                for key, color in (("vram", "primary"), ("util", "grey-5")):
                                    with ui.row().classes("w-full items-center gap-2 no-wrap"):
                                        ui.label("VRAM" if key == "vram" else "Load").classes("rz-tile-sub w-10")
                                        bar = ui.linear_progress(value=0, show_value=False, size="6px",
                                                                 color=color).props("rounded") \
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
                    normalized = any(sc.get("combined_norm") is not None for sc in fscores)  # else raw (older runs)
                    nz = lambda sc, nk, rk: sc.get(nk) if normalized else sc.get(rk)  # noqa: E731

                    for sc in fscores:
                        comb = nz(sc, "combined_norm", "combined_percent")
                        if comb is not None:
                            by_prompt.setdefault(sc["prompt_index"], []).append([sc["step"], round(comb, 1)])
                        d_ = per_step.setdefault(sc["step"], {"face": [], "body": []})
                        if nz(sc, "face_norm", "percent") is not None:
                            d_["face"].append(nz(sc, "face_norm", "percent"))
                        if nz(sc, "body_norm", "body_percent") is not None:
                            d_["body"].append(nz(sc, "body_norm", "body_percent"))
                    series = [{"name": f"combined p{k + 1}", "type": "line", "data": v}
                              for k, v in sorted(by_prompt.items())]
                    for key, name in (("face", "mean face"), ("body", "mean body")):
                        pts = [[stp, round(sum(v[key]) / len(v[key]), 1)] for stp, v in sorted(per_step.items()) if v[key]]
                        if pts:
                            series.append({"name": name, "type": "line", "data": pts, "lineStyle": {"type": "dotted"}})
                    ref_sim = st.get("face_ref_similarity")
                    if series and (normalized or ref_sim is not None):
                        xs = [pt[0] for v in by_prompt.values() for pt in v] or [0, 1]
                        level = 100.0 if normalized else ref_sim * 100  # normalized: the references are 100 as such
                        series.append({"name": "the references" if normalized else "face reference self-similarity",
                                       "type": "line", "showSymbol": False, "lineStyle": {"type": "dashed"},
                                       "data": [[min(xs), level], [max(xs), level]]})
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
                    parts = [f"Step {latest_step}" + (" (normalized: 0 = a stranger, 100 = the references):"
                                                      if normalized else ":")]
                    parts.append(f"face {sum(fv) / len(fv):.1f}% ({len(fv)}/{len(latest)} with a face)" if fv
                                 else f"face: none in {len(latest)} sample(s)")
                    if st.get("body_refs"):
                        parts.append(f"body {sum(bv) / len(bv):.1f}% ({len(bv)}/{len(latest)} with a person)" if bv
                                     else "body: no usable person")
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
                                 f"out after round {dropped[n]}" if n in dropped
                                 else "training" if (n in (rc.get("lanes") or []) or (rc.get("active") == n
                                                                                       and not rc.get("parallel")))
                                 else "waiting its turn")
                        if rc.get("follow") and rc.get("leader") == n and not rc.get("winner"):
                            state += " (led the last round)"
                        spared = ((rc.get("history") or [{}])[-1].get("spared") or {}).get(n)
                        if spared and n not in dropped:
                            state += f" (kept after {spared} probes without improvement: rising again)"
                        per_round = [(h["scores"].get(n), (h.get("steps") or {}).get(n))
                                     for h in rc.get("history") or [] if n in h["scores"]]
                        at = (rc.get("score_steps") or {}).get(n)
                        rows.append({"name": n, "state": state, "at": "-" if at is None else f"step {at}",
                                     "score": f1((rc.get("scores") or {}).get(n)),
                                     "face": part("face", n), "body": part("body", n),
                                     "rounds": " / ".join(f1(s) + ("" if st is None else f" @{st}")
                                                          for s, st in per_round) or "-",
                                     "settings": (rc.get("settings") or {}).get(n) or "the page's own settings",
                                     "can_drop": n not in dropped and not rc.get("winner") and running_now})
                    race_table.rows = rows
                    race_table.update()
                    est = rc.get("estimate") or {}
                    race_label.text = (f"{rc.get('winner')} won the relay's last round; its LoRA is the run's"
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
                    series, legend = probe_chart_series(probes, (rc or {}).get("names") or [])
                    probe_chart.options["series"] = series
                    probe_chart.options["legend"]["data"] = legend
                    probe_chart.options["grid"]["top"] = 60 if len(legend) > 5 else 40  # room for a race's legend
                    probe_chart.update()
                    lk = st.get("likeness") or {}
                    last = probes[-1]
                    f2 = lambda x: "n/a" if x is None else f"{x:.1f}%"  # noqa: E731
                    normed = last.get("face_norm_ema") is not None or last.get("body_norm_ema") is not None
                    fe = last.get("face_norm_ema") if normed else last.get("face_ema")
                    be = last.get("body_norm_ema") if normed else last.get("body_ema")
                    probe_label.text = (f"Probe @ step {last['step']}: face {f2(fe)} body {f2(be)}"
                                        f" combined {f2(last.get('combined_ema'))} | best {f2(lk.get('best'))} @ step {lk.get('best_step')}"
                                        f" | {lk.get('since_improve', 0)} probe(s) since improvement | lr x{lk.get('lr_factor', 1.0):.2f}"
                                        f" | face bias {lk.get('face_bias', 0.0):+.2f}" + (" | STOPPED" if lk.get("stopped") else ""))
                    rec_of = {Path(p).name: pr for pr in probes for p in (pr.get("images") or [])}
                    rnames = (rc or {}).get("names") or []
                    if rnames:  # a race: each probe image of every configuration side by side, newest step first
                        shown = group_by_generation([p for pr in probes[-60:] for p in (pr.get("images") or [])],
                                                    rnames)[:8]
                        sig = [(st_, r_, [(w_, p_, probe_score_line(rec_of.get(Path(p_).name))) for w_, p_ in m_])
                               for st_, r_, m_ in shown]
                    else:
                        sig = [(0, "", [("", p, probe_score_line(rec_of.get(Path(p).name)))])
                               for p in list(last.get("images") or [])[:6]]
                    if sig != shown_probes:
                        shown_probes[:] = sig
                        probe_row.clear()
                        with probe_row:
                            for st_, r_, members in sig:
                                if rnames:
                                    ui.label(generation_label(st_, r_)).classes("rz-subhead w-full")
                                for who, p, line in members:
                                    with ui.card().tight().classes("rz-thumb"):
                                        ui.image(image_url(p)).props("fit=contain") \
                                            .classes("w-40 h-40 rz-imgbg cursor-pointer") \
                                            .on("click", lambda _, q=p: show_full(Path(q)))
                                        ui.label(who or Path(p).name).classes("text-xs px-1")
                                        if line:
                                            ui.label(line).classes("text-xs px-1 pb-1 rz-hint")
                score_by_name = {Path(sc["image"]).name: sc for sc in fscores}
                rnames = (rc or {}).get("names") or []
                samples = (st.get("samples") or [])[:max(12, 6 * len(rnames))]
                sample_sig = [(p, score_caption(score_by_name[Path(p).name]) if Path(p).name in score_by_name else "")
                              for p in samples]
                if sample_sig != shown_samples:
                    shown_samples[:] = sample_sig
                    samples_row.clear()
                    cap_of = dict(sample_sig)
                    with samples_row:
                        if not sample_sig:
                            hint("No samples yet. They appear here every 'Sample every' steps.")
                        if rnames:  # a race: the same generation of every configuration side by side
                            for step_, rest_, members in group_by_generation(samples, rnames)[:6]:
                                ui.label(generation_label(step_, rest_)).classes("rz-subhead w-full")
                                for who, p in members:
                                    with ui.card().tight().classes("rz-thumb"):
                                        ui.image(image_url(p)).props("fit=contain") \
                                            .classes("w-48 h-48 rz-imgbg cursor-pointer") \
                                            .on("click", lambda _, q=p: show_full(Path(q)))
                                        ui.label((who or "winner") + cap_of.get(p, "")).classes("text-xs px-1")
                        else:
                            for p, cap_ in sample_sig:
                                with ui.card().tight().classes("rz-thumb"):
                                    ui.image(image_url(p)).props("fit=contain") \
                                        .classes("w-48 h-48 rz-imgbg cursor-pointer") \
                                        .on("click", lambda _, q=p: show_full(Path(q)))
                                    ui.label(Path(p).name + cap_).classes("text-xs px-1")
                saved = list(st.get("saved") or [])
                show_downloads(saved)
                states = list(st.get("states") or [])
                saved_label.text = ("Saved:\n" + "\n".join(saved) if saved else "No LoRA saved yet.") \
                    + ("\nStates:\n" + "\n".join(states) if states else "")
                update_log()

            ui.timer(2.0, poll)

        # ------------------------------------------------------------ Samples
        with ui.tab_panel(tab_samples):
            with ui.column().classes("rz-page gap-4"):
                build_samples_tab(f)

        # ------------------------------------------------------------ Previous Runs
        with ui.tab_panel(tab_history):
            with ui.column().classes("rz-page gap-4"):
                build_history_tab(charts)

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
                        .props("rounded")
                task_row.visible = False
            with ui.row().classes("items-center gap-2 no-wrap"):
                queue_note = ui.label("").classes("rz-hint text-sm cursor-pointer")
                queue_note.on("click", lambda: goto("Monitor"))
                ui.button("Add to queue", icon="playlist_add", on_click=queue_job, color=None).props("outline") \
                    .tooltip("Start this run after the current one (and any queued before it) is done")
                ui.button("Start training", icon="play_arrow", on_click=start_job).props("unelevated")
                ui.button("Stop", icon="stop", on_click=lambda: setattr(status_msg, "text", stop_training()),
                          color=None) \
                    .props("outline").tooltip("Stop after the current step and save the LoRA")
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
            ft = st.get("finetune") or {}
            ft_text = (f" · fine-tune window {ft['window']}/{ft['windows']}, cycle {ft['cycle']}"
                       if ft.get("windows") else "")
            pill_text.text = (f"Training · step {st.get('step', 0):,} / {st.get('total') or 0:,}{ft_text}"
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
noisier gradients (cosine 0.93-0.99 to the exact ones); `bf16` stays exact. *Attention kernel*: `auto` puts cuDNN's
attention on Blackwell GPUs (RTX 50, B200), where it ran Krea 2 steps 15-22% faster at the same loss, and lets
PyTorch pick elsewhere; `torch`, `cudnn`, `flash`, `efficient`, `math` put one first. The others stay enabled behind
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
Turn on *Save training state* (LoRA, Advanced) to also write `state_XXXXXX/` folders (LoRA in fp32 + optimizer +
scheduler + RNG). Point *Resume from training state folder* at one of them to continue exactly where it stopped.

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
same step time. It needs gradient checkpointing and turns torch.compile off.

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

## Adapters: LoRA, LoKr, OFT and DoRA
The *Adapter* field (Train tab, LoRA section) picks what each layer learns. **LoRA**: a rank-r pair. **LoKr**: a
Kronecker pair (factor 8 is Fizgig's best measured likeness). **OFT**: rotates each layer's output channels in blocks
(*OFT block size*, 32) instead of adding to the weights, which keeps the base model's structure; the OFT paper found
better subject identity than LoRA and stable long runs. *OFT constraint* > 0 caps how far a layer rotates. **DoRA**
(a switch for LoRA and LoKr): a magnitude per input channel trains on its own while the adapter steers the
direction; steps are about 40% slower on Krea 2. OFT and DoRA files load in ComfyUI exactly as trained (not in diffusers). To
compare them, race rows such as `network_type=oft` or `dora=true` against a LoKr row.

## Trigger attention and identity alignment
Two training-only losses (Train tab, Region-weighted loss section; the Wizard's Training step for a character). **Keep
the trigger's attention on the subject** reads, inside the model, where the trigger word's attention lands in each
picture and pulls it onto the person (their silhouette and face box), so the trigger is bound to them and not to the
background or other people; it needs a trigger word. **Align the face tokens with the photo's identity** teaches the
face's tokens after one block to carry that photo's identity (its ArcFace embedding) through a small helper network
that is thrown away. The LoRA file is the same either way. Not yet measured: race them.

## Likeness reward fine-tuning (second stage)
*Reward steps* (Likeness probes section, *Second stage*; or the Wizard's Samples step) adds a stage after the steps:
the LoRA renders groups of 4 pictures of new prompts, the probes' scorers rate their likeness, and the better ones are
made more likely and the worse ones less (Flow-GRPO), held near the first stage's LoRA. It aims straight at likeness
but is slow and can learn what the scorer likes more than what you see: keep the probes on and look at the pictures.
The probes also score the stage's final LoRA when it ends, so its likeness shows next to the first stage's. Each
picture's face and body scores are capped far below a stranger's: a body that differs from your body references only
in clothing (a coat against unclothed references) can't outweigh the face. Reward prompts like your body references,
or Body weight 0, let the body count.
`<job>.safetensors` stays the first stage's; the stage saves `<job>_reward.safetensors`. Steps 0 with a LoRA in
*Resume from* runs the stage alone.

## Previous runs (run history)

The **Previous Runs** tab keeps the likeness history of every run, like TensorBoard, so runs can be compared after the
job folder has been reused. 
A run with likeness scores writes one small file in `output/_history/` while it trains (a
crash keeps what was saved) and again when it ends. It keeps what the Monitor's two likeness charts plot, on the run's
normalized scale: the **mean face and mean body** score of the sample images at each sampling step (Identity
similarity), and the smoothed **face, body and combined** score at each probe (Likeness probes & controller). Runs
from before the history existed are imported from their job folders when the tab first loads, or with *Import old runs*.

Tick any number of runs in the table (name, start, state, steps, adapter, learning rate, last and best scores) to
overlay them: one colour per run, solid lines for the face, dashed for the body and thick for the combined score, with
checkboxes to show only some of the three. *Rename* gives a run a name of your own and *Remove from history* deletes
its history file only (never the job folder). A run that is still training updates live.

**Probe images** (below the charts) puts the ticked runs' likeness probes side by side: pick Face or Body and a probe
step (the arrows step through them), and each run, or each configuration of a race, shows its probe images of that
step in one row with their overall / face / body scores. A run probed on another interval shows its latest probe before
that step, marked. Click an image to see it full size; tick the box under any two (any run, any step) to open them
side by side, large, with their scores (*Compare* opens them again, *Clear picks* starts over). When the ticked runs
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
outfits among the references. `proportions` is the older skeleton-ratio match (every human ~80, the references
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
the plan, the status pill shows the window). Every save extracts a LoRA of the rank(s) you set from the deltas, at the
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

    @ui.page("/")
    def index(request: Request):
        # A real page function, not NiceGUI's script mode: script mode re-executes this file for every page load,
        # which reset the job globals, so a reload lost the running job.
        build_page(local=is_local_viewer(request))

    ui.run(host=a.host, port=port, title="RulzTrainer", show=a.open, reload=False, dark=True,
           favicon=ASSETS / "icon.png")
