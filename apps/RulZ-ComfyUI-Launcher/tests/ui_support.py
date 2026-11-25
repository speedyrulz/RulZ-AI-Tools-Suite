"""Helpers for tests that build the real main window offscreen without WSL, nvidia-smi or sockets."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QObject, Signal

from rulz_comfy_launcher.gpus import GpuInfo
from rulz_comfy_launcher.model import ConfigStore, Profile
from rulz_comfy_launcher.ui import main_window


class FakeGpuMonitor(QObject):
    updated = Signal()

    def __init__(self, job=None, interval_ms=0, parent=None):
        super().__init__(parent)
        self._gpus: list[GpuInfo] = list(getattr(type(self), "gpu_list", []))

    def start(self):
        pass

    def stop(self):
        pass

    def gpus(self):
        return list(self._gpus)

    def index_by_uuid(self):
        return {g.uuid: g.index for g in self._gpus}


class FakePortWatcher(QObject):
    changed = Signal(int, bool)
    open_ports: set[int] = set()

    def __init__(self, parent=None):
        super().__init__(parent)

    def set_ports(self, ports):
        pass

    def start(self):
        pass

    def stop(self):
        pass

    def forget(self, port):
        pass

    def is_open(self, port):
        return port in type(self).open_ports


class _NeverFinishes(QObject):
    finished = Signal(int, str, str)


def sample_gpus() -> list[GpuInfo]:
    return [
        GpuInfo(0, "GPU-aaaa", "NVIDIA GeForce RTX 5090", "0:1", 9216, 32768, 41, 56),
        GpuInfo(1, "GPU-bbbb", "NVIDIA GeForce RTX 4090", "0:2", 1024, 24576, 3, 38),
    ]


def make_window(tmp_path, monkeypatch, profiles=None, gpus=None, settings=None):
    """A real MainWindow on a temp config file. GPU monitor, port watcher and WSL are stubs."""
    monkeypatch.setattr(main_window, "GpuMonitor", FakeGpuMonitor)
    monkeypatch.setattr(main_window, "PortWatcher", FakePortWatcher)
    monkeypatch.setattr(main_window.wsl, "list_distros", lambda parent=None: _NeverFinishes(parent))
    FakeGpuMonitor.gpu_list = gpus or []
    FakePortWatcher.open_ports = set()
    store = ConfigStore(tmp_path / "config.json")
    store.load()
    store.profiles = profiles or [Profile(name="Main ComfyUI", port=8188)]
    for key, value in (settings or {}).items():
        setattr(store.settings, key, value)
    window = main_window.MainWindow(store)
    return window


def close_window(window):
    """Stop what the window started, without touching the config file or showing a question."""
    window._closing_for_real = True
    window._page_opener.stop()
    window._save_timer.stop()
    window.watcher.stop()
    window.gpu_monitor.stop()
    if window._tray is not None:
        window._tray.hide()
    window.close()
    window.deleteLater()
