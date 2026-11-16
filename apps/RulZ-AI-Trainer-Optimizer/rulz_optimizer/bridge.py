"""Talk to a RulZ AI Trainer install: validate a config, queue it locally or on vast.ai, open the WebUI.

Every write goes through the trainer's own Python code, run with the trainer's own interpreter. All subprocesses use
list arguments (no shell), a timeout and captured output. Call these from a worker thread (nicegui run.io_bound),
never from the UI thread.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

IS_WINDOWS = sys.platform.startswith("win")
HELPER = Path(__file__).resolve().parent / "_trainer_helper.py"
WEBUI_PORT = 7870  # run_webui.bat / run_webui.sh default
VALIDATE_TIMEOUT = 120
QUEUE_TIMEOUT = 120
VAST_TIMEOUT = 300

WEBUI_NOTE = ("The local queue only advances while the RulZ AI Trainer WebUI runs (it starts the next run when "
              "nothing is running). Open the WebUI to start it.")
VAST_NOTE = ("The vast.ai queue starts the run on the first idle instance, but only while something is watching: "
             "the RulZ AI Trainer WebUI's Fleet tab, or `python scripts/vast/rulz_fleet.py daemon`.")


@dataclass
class TrainerInstall:
    root: Path
    python: Path
    version: str | None


def venv_python(root: Path) -> Path:
    win = root / "venv" / "Scripts" / "python.exe"
    nix = root / "venv" / "bin" / "python"
    if IS_WINDOWS:
        return win if win.exists() else nix
    return nix if nix.exists() else win


def is_trainer_root(root: Path) -> bool:
    return (root / "rulztrainer" / "config.py").is_file()


def read_version(root: Path) -> str | None:
    try:
        text = (root / "rulztrainer" / "__init__.py").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    m = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', text)
    return m.group(1) if m else None


def install_from(root: Path) -> TrainerInstall | None:
    root = Path(root).expanduser().resolve()  # absolute: the helper runs with cwd=root and also gets root
    if not is_trainer_root(root):
        return None
    py = venv_python(root)
    if not py.exists():
        py = Path(sys.executable)  # no venv yet: validation and queueing need only pyyaml, which this app has
    return TrainerInstall(root=root, python=py, version=read_version(root))


def find_trainer(hint: Path | None = None) -> TrainerInstall | None:
    """The settings path first, then a RulzTrainer folder next to this app (apps/RulzTrainer in the suite)."""
    candidates: list[Path] = []
    if hint:
        candidates.append(Path(hint))
    here = Path(__file__).resolve().parents[1]  # <app>
    candidates.append(here.parent / "RulzTrainer")
    for cand in candidates:
        inst = install_from(cand)
        if inst:
            return inst
    return None


def _run(args: list[str], cwd: Path, timeout: int) -> tuple[int, str]:
    """(exit code, stdout + stderr). Exit code -1: timed out, -2: could not start."""
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    try:
        done = subprocess.run(args, cwd=str(cwd), capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=timeout, env=env, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return -1, f"Timed out after {timeout} s."
    except OSError as e:
        return -2, f"Could not start {args[0]}: {e}"
    out = (done.stdout or "").strip()
    err = (done.stderr or "").strip()
    return done.returncode, "\n".join(x for x in (out, err) if x)


def validate(inst: TrainerInstall, cfg_path: Path) -> tuple[bool, str]:
    """TrainConfig.load(path).validate() in the trainer's own Python."""
    code, text = _run([str(inst.python), str(HELPER), "validate", str(inst.root), str(cfg_path)],
                      inst.root, VALIDATE_TIMEOUT)
    return code == 0, text or ("Valid." if code == 0 else f"Validation failed (exit {code}).")


def queue_local(inst: TrainerInstall, cfg_path: Path, label: str) -> tuple[bool, str]:
    """Validate, then TrainQueue(output/_queue.json).add(config, label) through the trainer's own code."""
    code, text = _run([str(inst.python), str(HELPER), "queue", str(inst.root), str(cfg_path), label],
                      inst.root, QUEUE_TIMEOUT)
    if code == 0:
        return True, f"{text}\n{WEBUI_NOTE}"
    return False, text or f"Queueing failed (exit {code})."


FLEET_EXIT = {1: "refused", 2: "timed out", 3: "unreachable (no instance reachable)", -1: "timed out"}


def fleet_command(inst: TrainerInstall, cfg_path: Path, job: str, datasets: list[Path]) -> list[str]:
    """rulz_fleet.py queue add ... --push FOLDER: the fleet copies the folders to the instance and points the
    config's dataset paths at the copy."""
    cmd = [str(inst.python), str(inst.root / "scripts" / "vast" / "rulz_fleet.py"), "queue", "add",
           "--job", job, "--config", str(cfg_path), "--label", "RulZ AI Trainer Optimizer"]
    for folder in datasets:
        cmd += ["--push", str(folder)]
    return cmd


def queue_vast(inst: TrainerInstall, cfg_path: Path, job: str, datasets: list[Path]) -> tuple[bool, str]:
    script = inst.root / "scripts" / "vast" / "rulz_fleet.py"
    if not script.is_file():
        return False, f"{script} not found: this RulZ AI Trainer has no vast.ai fleet tool."
    code, text = _run(fleet_command(inst, cfg_path, job, datasets), inst.root, VAST_TIMEOUT)
    if code == 0:
        return True, f"{_summary(text)}\n{VAST_NOTE}"
    reason = FLEET_EXIT.get(code, f"failed (exit {code})")
    return False, f"vast.ai queue {reason}: {text}".strip()


def _summary(text: str) -> str:
    """The fleet tool prints JSON; show which runs it queued, else the raw text."""
    try:
        data = json.loads(text[text.index("{"):text.rindex("}") + 1])
        queued = ", ".join(str(q.get("job")) for q in data.get("queued", []))
        if queued:
            return f"Queued on vast.ai: {queued}."
    except (ValueError, AttributeError):
        pass
    return text or "Queued on vast.ai."


def vast_home() -> Path:
    return Path(os.environ.get("RULZ_VAST_HOME") or Path.home() / ".rulz_vast")


def fleet_instances() -> list[dict]:
    """The instances the fleet knows (read-only, from ~/.rulz_vast/fleet/state.json): id, label, gpu, status, gone."""
    try:
        data = json.loads((vast_home() / "fleet" / "state.json").read_text(encoding="utf-8"))
        raw = data.get("instances", {})
    except (OSError, ValueError, AttributeError):
        return []
    if not isinstance(raw, dict):
        return []
    out = []
    for key, inst in raw.items():
        if not isinstance(inst, dict):
            continue
        out.append({"id": inst.get("id", key), "label": inst.get("label") or "", "gpu": inst.get("gpu") or "",
                    "status": inst.get("status") or "", "gone": bool(inst.get("gone")),
                    "enabled": inst.get("enabled", True)})
    return out


def webui_running(port: int = WEBUI_PORT) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=1.5):
            return True
    except Exception:  # noqa: BLE001
        return False


def open_webui(inst: TrainerInstall) -> str:
    """Start run_webui.bat / run_webui.sh unless the WebUI already answers. Returns a message for the user."""
    url = f"http://localhost:{WEBUI_PORT}"
    if webui_running():
        return f"The RulZ AI Trainer WebUI is already running: {url}"
    try:
        if IS_WINDOWS:
            bat = inst.root / "run_webui.bat"
            if not bat.is_file():
                return f"{bat} not found."
            subprocess.Popen([str(bat)], cwd=str(inst.root),
                             creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0))
        else:
            sh = inst.root / "run_webui.sh"
            if not sh.is_file():
                return f"{sh} not found."
            subprocess.Popen(["bash", str(sh)], cwd=str(inst.root), start_new_session=True,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
    except OSError as e:
        return f"Could not start the WebUI: {e}"
    return f"Starting the RulZ AI Trainer WebUI (it opens {url} in a moment)."
