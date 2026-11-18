"""Start and stop the "render on this PC" helper (python -m rulztrainer.renderhelper) for the WebUI's Fleet tab.

One helper per vast.ai instance, one GPU per helper. The helper is a child of the WebUI: its stdin is a pipe that
closes when the WebUI dies (the helper stops on stdin close), and stop_all() ends every helper when the WebUI exits.
No GPU is touched here: --list-gpus and the helper itself are run only when the user asks for them.

The command line and the status file are the contract's section 5 (apps/RulzTrainer/docs/remote-render-contract.md).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # apps/RulzTrainer: the helper runs with it as its working folder
STATUS_DIR = Path(os.environ.get("RULZ_VAST_HOME") or Path.home() / ".rulz_vast") / "render_helper"  # (as rulz_remote)
GPU_LIST_TTL = 60.0  # seconds a --list-gpus answer is reused
GPU_LIST_TIMEOUT = 90.0  # torch's first import can be slow
STALE_SECONDS = 30.0  # a running helper whose status file is older than this is said to be quiet

# state in the status file -> (short word, Material icon, level); level picks the theme colour (ok/warn/err/"")
STATES = {
    "connecting": ("Connecting", "sync", "warn"),
    "preparing": ("Preparing", "downloading", "warn"),
    "ready": ("Ready", "check_circle", "ok"),
    "idle": ("Ready", "check_circle", "ok"),
    "rendering": ("Rendering", "autorenew", "ok"),
    "not_ready": ("Not ready", "warning", "warn"),
    "stopped": ("Stopped", "stop_circle", ""),
    "error": ("Error", "error", "err"),
}
OFF = ("Stopped", "stop_circle", "")


class Refused(Exception):
    """The helper was not started, and the message says why in plain words."""


def parse_gpus(text: str) -> list[dict]:
    """The helper's --list-gpus output ([{index, name, total_gb, free_gb}]), tolerating log lines before the JSON."""
    start = text.find("[")
    if start < 0:
        raise ValueError("no GPU list in the helper's output")
    data = json.loads(text[start:text.rindex("]") + 1])
    out = []
    for g in data:
        out.append({"index": int(g["index"]), "name": str(g.get("name") or f"GPU {g['index']}"),
                    "total_gb": float(g.get("total_gb") or 0), "free_gb": float(g.get("free_gb") or 0)})
    return out


def gpu_label(g: dict) -> str:
    return f"{g['index']}: {g['name']} ({g['total_gb']:.0f} GB)" if g.get("total_gb") else f"{g['index']}: {g['name']}"


def status_view(raw: dict | None, running: bool, exit_code: int | None = None, now: float | None = None,
                stopped: bool = False) -> dict:
    """What the card shows: {word, icon, level, text}. A word and an icon always, never colour alone.

    raw is the status file's content (or None when there is none yet); running says whether the process is alive."""
    now = time.time() if now is None else now
    raw = raw or {}
    state = str(raw.get("state") or "")
    if not running:
        if stopped:  # the user turned it off: Windows' terminate() gives exit code 1, which is not an error
            word, icon, level = OFF
            return {"word": word, "icon": icon, "level": level, "text": ""}
        if exit_code not in (None, 0) and state != "stopped":
            word, icon, level = STATES["error"]
            msg = raw.get("message") or f"The helper ended (exit code {exit_code}). See its log."
            return {"word": word, "icon": icon, "level": level, "text": str(msg)}
        word, icon, level = OFF
        return {"word": word, "icon": icon, "level": level, "text": str(raw.get("message") or "")}
    word, icon, level = STATES.get(state, STATES["connecting"])
    parts: list[str] = []
    if raw.get("job"):
        parts.append(str(raw["job"]))
    done = raw.get("rounds_done")
    if isinstance(done, (int, float)) and done:
        parts.append(f"{int(done)} round{'s' if int(done) != 1 else ''} done")
    secs = raw.get("last_seconds")
    if isinstance(secs, (int, float)) and secs:
        parts.append(f"last {secs:.1f} s")
    behind = raw.get("behind")
    if isinstance(behind, (int, float)) and behind:
        parts.append(f"{int(behind)} behind")
    if raw.get("message"):
        parts.append(str(raw["message"]))
    stamp = raw.get("time")
    if isinstance(stamp, (int, float)) and now - stamp > STALE_SECONDS and state in ("rendering", "ready", "idle"):
        parts.append(f"no news for {int(now - stamp)} s")
        level = "warn"
    return {"word": word, "icon": icon, "level": level, "text": " · ".join(parts)}


def helper_command(instance, device_index: int, status_file: Path, sample_model: str = "", quant: str = "",
                   models_dir: str = "") -> list[str]:
    """The helper's command line (contract section 5)."""
    cmd = [sys.executable, "-m", "rulztrainer.renderhelper", "--instance", str(instance),
           "--device", f"cuda:{int(device_index)}", "--status-file", str(status_file)]
    if sample_model.strip():
        cmd += ["--sample-model", sample_model.strip()]
        if quant.strip() and quant.strip() != "auto":
            cmd += ["--sample-model-quant", quant.strip()]
    if models_dir.strip():
        cmd += ["--models-dir", models_dir.strip()]
    return cmd


class RenderHelpers:
    """The running helpers of this WebUI: instance id -> process."""

    def __init__(self, status_dir: Path | None = None, popen=subprocess.Popen, run=subprocess.run):
        self.dir = Path(status_dir) if status_dir else STATUS_DIR
        self._popen, self._run = popen, run
        self._procs: dict[str, dict] = {}  # str(instance) -> {proc, gpu, gpu_name, log}
        self._lock = threading.Lock()
        self._gpus: tuple[float, list[dict], str] | None = None

    # ------------------------------------------------------------ GPUs
    def cached_gpus(self) -> list[dict]:
        """The last listing's GPUs (empty before one): never runs anything."""
        with self._lock:
            return list(self._gpus[1]) if self._gpus else []

    def list_gpus(self, force: bool = False) -> tuple[list[dict], str]:
        """(GPUs, error). Asks the helper's CLI (so torch is only imported in a child) and reuses the answer."""
        with self._lock:
            if not force and self._gpus and time.time() - self._gpus[0] < GPU_LIST_TTL:
                return self._gpus[1], self._gpus[2]
        gpus: list[dict] = []
        error = ""
        try:
            r = self._run([sys.executable, "-m", "rulztrainer.renderhelper", "--list-gpus"], cwd=str(ROOT),
                          capture_output=True, text=True, encoding="utf-8", errors="replace",
                          timeout=GPU_LIST_TIMEOUT, **_quiet())
            if r.returncode != 0:
                tail = (r.stderr or r.stdout or "").strip().splitlines()
                error = tail[-1] if tail else "The GPU list did not work."
            else:
                gpus = parse_gpus(r.stdout)
                if not gpus:
                    error = "No GPU was found on this PC."
        except subprocess.TimeoutExpired:
            error = "Listing this PC's GPUs took too long."
        except (OSError, ValueError, KeyError) as e:
            error = f"Could not list this PC's GPUs: {e}"
        with self._lock:
            self._gpus = (time.time(), gpus, error)
        return gpus, error

    # ------------------------------------------------------------ processes
    def status_file(self, instance) -> Path:
        return self.dir / f"{instance}.json"

    def _alive(self, entry: dict | None) -> bool:
        return bool(entry) and entry["proc"].poll() is None

    def running(self, instance) -> bool:
        with self._lock:
            return self._alive(self._procs.get(str(instance)))

    def gpu_of(self, instance) -> int | None:
        with self._lock:
            e = self._procs.get(str(instance))
            return e["gpu"] if self._alive(e) else None

    def gpu_user(self, gpu: int, except_instance=None) -> str | None:
        """The instance whose running helper uses this GPU, if any."""
        with self._lock:
            for iid, e in self._procs.items():
                if iid != str(except_instance) and e["gpu"] == gpu and self._alive(e):
                    return iid
        return None

    def start(self, instance, gpu: dict, sample_model: str = "", quant: str = "", models_dir: str = "") -> list[str]:
        """Start the helper for an instance on a GPU ({index, name, ...}); returns its command line."""
        iid = str(instance)
        index = int(gpu["index"])
        if self.running(iid):
            raise Refused(f"The helper for instance {iid} is already running.")
        other = self.gpu_user(index, except_instance=iid)
        if other is not None:
            raise Refused(f"GPU {index} ({gpu.get('name', '')}) is already rendering for instance {other}. "
                          "Pick another GPU, or turn that one off first.")
        sample_model = (sample_model or "").strip()
        if sample_model and not Path(sample_model).expanduser().exists():
            raise Refused(f"The sampling model was not found: {sample_model}")
        self.dir.mkdir(parents=True, exist_ok=True)
        status = self.status_file(iid)
        try:
            status.unlink()  # a status from an earlier helper would pass for this one's
        except OSError:
            pass
        cmd = helper_command(iid, index, status, sample_model, quant, models_dir)
        log = open(self.dir / f"{iid}.log", "w", encoding="utf-8")
        try:
            proc = self._popen(cmd, cwd=str(ROOT), stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT,
                               **_quiet(group=True))
        except OSError as e:
            log.close()
            raise Refused(f"The helper did not start: {e}") from e
        with self._lock:
            old = self._procs.get(iid)
            self._procs[iid] = {"proc": proc, "gpu": index, "gpu_name": str(gpu.get("name") or ""), "log": log,
                                "stopped": False}
        if old:  # a crashed helper's handles
            _close(old["proc"].stdin, old.get("log"))
        return cmd

    def stop(self, instance, wait: float = 10.0) -> None:
        with self._lock:
            e = self._procs.get(str(instance))
        if e is None:
            return
        proc = e["proc"]
        e["stopped"] = True
        if proc.poll() is None:
            # stdin first: the helper stops cleanly on stdin close (releases its round, writes "stopped"). On Windows
            # terminate() is TerminateProcess, so its own signal handler never runs.
            _close(proc.stdin)
            try:
                proc.wait(timeout=wait)
            except subprocess.TimeoutExpired:
                try:
                    proc.terminate()
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        pass
                except OSError:
                    pass
            except OSError:
                pass
        _close(proc.stdin, e.get("log"))

    def stop_all(self) -> None:
        with self._lock:
            ids = list(self._procs)
        for iid in ids:
            self.stop(iid)

    # ------------------------------------------------------------ status
    def read_status(self, instance) -> dict | None:
        try:
            return json.loads(self.status_file(instance).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def view(self, instance) -> dict:
        """The status line of an instance's helper: {word, icon, level, text, gpu_name, running}."""
        iid = str(instance)
        with self._lock:
            e = self._procs.get(iid)
        running = self._alive(e)
        code = e["proc"].poll() if e else None
        v = status_view(self.read_status(iid) if e else None, running, code, stopped=bool(e and e.get("stopped")))
        v["running"] = running
        v["gpu_name"] = e["gpu_name"] if e else ""
        return v


def _close(*files) -> None:
    for f in files:
        try:
            if f:
                f.close()
        except OSError:
            pass


def _quiet(group: bool = False) -> dict:
    """No console window on Windows; with group, its own process group so Ctrl+C of the WebUI's console is not its."""
    if sys.platform == "win32":
        flags = subprocess.CREATE_NO_WINDOW | (subprocess.CREATE_NEW_PROCESS_GROUP if group else 0)
        return {"creationflags": flags}
    return {"start_new_session": True} if group else {}
