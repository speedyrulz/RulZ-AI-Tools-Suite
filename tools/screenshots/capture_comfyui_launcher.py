"""Screenshots of RulZ ComfyUI Launcher with made-up demo data, in the dark and the light theme.

Run with any Python that has PySide6 (the app needs nothing else):

    python -B capture_comfyui_launcher.py --out R:\\out\\comfyui-launcher --readme R:\\out-readme\\comfyui-launcher

* The app runs from apps/RulZ-ComfyUI-Launcher in this repository.
* Nothing of the user's is touched. APPDATA points at a temporary folder, and every picture gets its own
  config file in it (demo configurations "Main", "Second" and "CPU test", Settings.theme "dark" or "light").
* Nothing starts wsl.exe or nvidia-smi: the GPU monitor, the port watcher, the distro list and the option
  catalog request are replaced with stand-ins that report three made-up cards and one distro. No server is
  started; the Log page is fed with made-up text.
* Qt renders at 100 % (QT_SCALE_FACTOR = 96 / system DPI), so the images have the same pixel size whatever the
  display scaling is.
* The main window and the Preferences dialog are grabbed with QWidget.grab().
* --out gets the catalog set, --readme (optional) the same files for the app README's docs folder:
  01-settings.png, 02-log.png, 03-preferences.png (dark) and 04-settings-light.png, 05-log-light.png,
  06-preferences-light.png.
"""
from __future__ import annotations

import argparse
import ctypes
import inspect
import json
import os
import shutil
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", "apps", "RulZ-ComfyUI-Launcher"))

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True, help="catalog set")
ap.add_argument("--readme", default="", help="README set (docs folder); skipped if empty")
ap.add_argument("--size", default="1280x860", help="window size in pixels")
args = ap.parse_args()
WIDTH, HEIGHT = (int(v) for v in args.size.lower().split("x"))

for d in (args.out, args.readme):
    if d:
        os.makedirs(d, exist_ok=True)

WORK = tempfile.mkdtemp(prefix="rulz-comfy-shots-")
os.environ["APPDATA"] = WORK
os.environ["QT_QPA_PLATFORM"] = os.environ.get("QT_QPA_PLATFORM", "windows")
sys.dont_write_bytecode = True
sys.path.insert(0, REPO)


def system_dpi() -> int:
    u = ctypes.windll.user32
    u.SetThreadDpiAwarenessContext.restype = ctypes.c_void_p
    u.SetThreadDpiAwarenessContext.argtypes = [ctypes.c_void_p]
    old = u.SetThreadDpiAwarenessContext(ctypes.c_void_p(-4))  # per-monitor v2, this thread only
    try:
        return u.GetDpiForSystem()
    finally:
        u.SetThreadDpiAwarenessContext(old)


if sys.platform == "win32" and system_dpi() != 96:
    os.environ.setdefault("QT_SCALE_FACTOR", f"{96 / system_dpi():.6f}")

from PySide6.QtCore import QEventLoop, QObject, QTimer, Signal  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from rulz_comfy_launcher import APP_NAME, __version__, wsl  # noqa: E402
from rulz_comfy_launcher.gpus import GpuInfo  # noqa: E402
from rulz_comfy_launcher.model import ConfigStore  # noqa: E402
from rulz_comfy_launcher.ui import main_window as mw  # noqa: E402
from rulz_comfy_launcher.ui import theme  # noqa: E402

# ---------------------------------------------------------------------------------------------------------
# made-up machine
# ---------------------------------------------------------------------------------------------------------
GPUS = [
    GpuInfo(0, "GPU-5a1e0c3d-0000-4000-8000-000000000001", "NVIDIA GeForce RTX 5090", "00000000:01:00.0",
            3120, 32607, 4, 41),
    GpuInfo(1, "GPU-5a1e0c3d-0000-4000-8000-000000000002", "NVIDIA GeForce RTX 4090", "00000000:41:00.0",
            14860, 24564, 63, 58),
    GpuInfo(2, "GPU-5a1e0c3d-0000-4000-8000-000000000003", "NVIDIA GeForce RTX 4060 Ti", "00000000:81:00.0",
            612, 16380, 0, 36),
]
DISTRO = "Ubuntu-24.04"


class FakeGpuMonitor(QObject):
    """Stands in for the nvidia-smi monitor: reports GPUS once and starts no process."""
    updated = Signal()

    def __init__(self, job=None, interval_seconds: int = 2, parent=None):
        super().__init__(parent)

    @property
    def available(self) -> bool:
        return True

    def gpus(self):
        return list(GPUS)

    def index_by_uuid(self):
        return {g.uuid: g.index for g in GPUS}

    def start(self):
        QTimer.singleShot(0, self.updated.emit)

    def stop(self):
        pass


class FakePortWatcher(QObject):
    """Stands in for the port probe: no port is ever open, nothing connects anywhere."""
    changed = Signal(int, bool)

    def __init__(self, interval_ms: int = 4000, parent=None):
        super().__init__(parent)

    def start(self):
        pass

    def stop(self):
        pass

    def set_ports(self, ports):
        pass

    def is_open(self, port):
        return False

    def forget(self, port):
        pass

    def check_now(self):
        pass


class FakeCommand(QObject):
    """Stands in for a wsl.exe run: answers at once with `out` (default: nothing) and starts no process."""
    finished = Signal(int, str, str)

    def __init__(self, code: int = 0, out: str = "", parent=None):
        super().__init__(parent)
        QTimer.singleShot(0, lambda: self.finished.emit(code, out, ""))


# Replace every route to wsl.exe and nvidia-smi before a window exists. The option catalog request answers
# "failed, not asked for by the user", so the built-in option list stays in use and no dialog opens.
mw.GpuMonitor = FakeGpuMonitor
mw.PortWatcher = FakePortWatcher
wsl.list_distros = lambda parent=None: FakeCommand(0, DISTRO + "\n", parent)
wsl.run_bash = lambda script, distro="", user="", timeout_ms=0, parent=None: FakeCommand(1, "", parent)
mw.wsl = wsl

# ---------------------------------------------------------------------------------------------------------
# made-up configurations and log
# ---------------------------------------------------------------------------------------------------------
PROFILES = [
    {
        "id": "demo-main", "name": "Main", "distro": DISTRO, "comfy_dir": "~/ComfyUI",
        "python": "venv/bin/python", "port": 8188, "listen": "127.0.0.1",
        "gpu_mode": "select", "gpus": [{"uuid": GPUS[0].uuid, "name": GPUS[0].name}],
        "options": {"--highvram": {"enabled": True, "value": ""},
                    "--use-sage-attention": {"enabled": True, "value": ""},
                    "--preview-method": {"enabled": True, "value": "auto"}},
        "env": [["CUDA_HOME", "/usr/local/cuda"], ["PATH", "$CUDA_HOME/bin:$PATH"]],
        "pre_commands": "source venv/bin/activate",
        "auto_open": True, "open_rulzflow": True, "checked": True,
    },
    {
        "id": "demo-second", "name": "Second", "distro": DISTRO, "comfy_dir": "~/ComfyUI",
        "python": "venv/bin/python", "port": 8189, "listen": "127.0.0.1",
        "gpu_mode": "select", "gpus": [{"uuid": GPUS[1].uuid, "name": GPUS[1].name}],
        "options": {"--lowvram": {"enabled": True, "value": ""}},
        "checked": True,
    },
    {
        "id": "demo-cpu", "name": "CPU test", "distro": DISTRO, "comfy_dir": "~/ComfyUI",
        "python": "venv/bin/python", "port": 8190, "listen": "127.0.0.1",
        "gpu_mode": "cpu", "gpus": [],
    },
]

LOG_TEXT = """\
GPU 0 (PCI bus order): NVIDIA GeForce RTX 5090
Checkpoint files will always be loaded safely.
Total VRAM 32607 MB, total RAM 65301 MB
pytorch version: 2.9.1+cu130
Set vram state to: HIGH_VRAM
Device: cuda:0 NVIDIA GeForce RTX 5090 : cudaMallocAsync
Using sage attention
Python version: 3.12.3 (main, Nov  6 2025, 13:44:16) [GCC 13.3.0]
ComfyUI version: 0.3.76
[Prompt Server] web root: /home/demo/ComfyUI/web
Import times for custom nodes:
   0.0 seconds: /home/demo/ComfyUI/custom_nodes/websocket_image_save.py
   0.3 seconds: /home/demo/ComfyUI/custom_nodes/ComfyUI-Manager
   0.6 seconds: /home/demo/ComfyUI/custom_nodes/RulZFlow
Starting server

To see the GUI go to: http://127.0.0.1:8188
got prompt
model weight dtype torch.bfloat16, manual cast: None
Requested to load Flux
loaded completely 30210.2 11350.07 True
100%|##########| 20/20 [00:09<00:00,  2.14it/s]
Prompt executed in 14.62 seconds
"""

OUT_NAMES = {
    "dark": ("01-settings.png", "02-log.png", "03-preferences.png"),
    "light": ("04-settings-light.png", "05-log-light.png", "06-preferences-light.png"),
}


def write_config(mode: str) -> str:
    folder = os.path.join(WORK, APP_NAME, mode)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "config.json")
    data = {"version": 1,
            "settings": {"theme": mode, "last_profile": "demo-main", "close_to_tray": False},
            "profiles": PROFILES}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return path


def apply_theme(app, mode: str) -> None:
    """The theme API is `apply_theme(app)` (reads the setting) or `apply_theme(app, mode)`; use what exists."""
    params = inspect.signature(theme.apply_theme).parameters
    if len(params) > 1:
        theme.apply_theme(app, mode)
    else:
        theme.apply_theme(app)


app = QApplication([sys.argv[0]])
app.setApplicationName(APP_NAME)
app.setApplicationVersion(__version__)


def pump(seconds: float) -> None:
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents(QEventLoop.AllEvents, 50)
        time.sleep(0.01)


def save(img, name: str) -> None:
    for folder in (args.out, args.readme):
        if folder:
            path = os.path.join(folder, name)
            img.save(path)
            print("saved", path, img.width(), "x", img.height(), flush=True)


def shoot(mode: str) -> None:
    store = ConfigStore(write_config(mode))
    store.load()
    assert store.path.is_relative_to(WORK), store.path  # never the user's own config
    if hasattr(store.settings, "theme"):
        assert store.settings.theme == mode, "the app did not read Settings.theme"
    else:
        print(f"note: this build has no Settings.theme yet, so the {mode} picture shows its only look",
              flush=True)
    apply_theme(app, mode)
    window = mw.MainWindow(store)
    window.resize(WIDTH, HEIGHT)
    window.show()
    pump(1.0)
    settings_png, log_png, prefs_png = OUT_NAMES[mode]

    # Settings page of "Main"
    window.tabs.setCurrentIndex(0)
    pump(0.5)
    save(window.grab().toImage(), settings_png)

    # Log page: made-up output of "Main"
    window._on_output("demo-main", LOG_TEXT)
    window._select_profile(store.find("demo-main"))
    window.tabs.setCurrentIndex(1)
    pump(0.8)
    save(window.grab().toImage(), log_png)

    # Preferences dialog on its own
    window.tabs.setCurrentIndex(0)
    dialog = mw.PreferencesDialog(store.settings, window)
    dialog.show()
    pump(0.6)
    save(dialog.grab().toImage(), prefs_png)
    dialog.close()

    window._closing_for_real = True
    window.close()
    window.deleteLater()
    pump(0.3)


try:
    for mode in ("dark", "light"):
        shoot(mode)
finally:
    shutil.rmtree(WORK, ignore_errors=True)
print("done", flush=True)
