"""Saved data: launch configurations ("profiles"), app settings, and the config file.

No Qt imports here, so this module can be unit-tested on its own.
"""
from __future__ import annotations

import copy
import json
import math
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from . import APP_NAME

CONFIG_VERSION = 1

GPU_AUTO = "auto"      # do not pass --cuda-device: ComfyUI sees every card
GPU_SELECT = "select"  # pass --cuda-device with the chosen cards
GPU_CPU = "cpu"        # pass --cpu
GPU_MODES = (GPU_AUTO, GPU_SELECT, GPU_CPU)

THEME_DARK = "dark"
THEME_LIGHT = "light"
THEMES = (THEME_DARK, THEME_LIGHT)


def default_config_dir() -> Path:
    base = os.environ.get("APPDATA") or str(Path.home())
    return Path(base) / APP_NAME


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


@dataclass
class OptionState:
    """One ComfyUI command line option as stored in a profile."""
    enabled: bool = False
    value: str = ""

    def to_dict(self) -> dict:
        return {"enabled": self.enabled, "value": self.value}

    @staticmethod
    def from_dict(data) -> "OptionState":
        if isinstance(data, bool):
            return OptionState(enabled=data)
        if not isinstance(data, dict):
            return OptionState()
        value = data.get("value", "")
        return OptionState(enabled=bool(data.get("enabled", False)),
                           value="" if value is None else str(value))


@dataclass
class GpuRef:
    """A GPU chosen in a profile. The UUID is what identifies the card; the name
    is only kept so the card can still be shown when it is not detected."""
    uuid: str
    name: str = ""

    def to_dict(self) -> dict:
        return {"uuid": self.uuid, "name": self.name}

    @staticmethod
    def from_dict(data) -> "GpuRef | None":
        if isinstance(data, str):
            return GpuRef(uuid=data) if data else None
        if isinstance(data, dict) and data.get("uuid"):
            return GpuRef(uuid=str(data["uuid"]), name=str(data.get("name") or ""))
        return None


@dataclass
class Profile:
    id: str = field(default_factory=_new_id)
    name: str = "ComfyUI"
    distro: str = ""                  # "" = the default WSL distro
    user: str = ""                    # "" = the distro's default user
    comfy_dir: str = "~/ComfyUI"
    python: str = "venv/bin/python"   # relative to comfy_dir, absolute, or a command name
    port: int = 8188
    listen: str = "127.0.0.1"
    gpu_mode: str = GPU_AUTO
    gpus: list[GpuRef] = field(default_factory=list)   # ordered: first = primary (cuda:0)
    options: dict[str, OptionState] = field(default_factory=dict)
    extra_args: str = ""
    env: list[list[str]] = field(default_factory=list)  # ordered [name, value] pairs
    pre_commands: str = ""            # bash lines run before ComfyUI starts
    auto_open: bool = False           # open the browser when the server is ready
    open_rulzflow: bool = False
    checked: bool = False             # part of the "start checked" set

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "distro": self.distro,
            "user": self.user,
            "comfy_dir": self.comfy_dir,
            "python": self.python,
            "port": self.port,
            "listen": self.listen,
            "gpu_mode": self.gpu_mode,
            "gpus": [g.to_dict() for g in self.gpus],
            "options": {k: v.to_dict() for k, v in self.options.items()
                        if v.enabled or v.value},
            "extra_args": self.extra_args,
            "env": [[k, v] for k, v in self.env],
            "pre_commands": self.pre_commands,
            "auto_open": self.auto_open,
            "open_rulzflow": self.open_rulzflow,
            "checked": self.checked,
        }

    @staticmethod
    def from_dict(data: dict) -> "Profile":
        p = Profile()
        if not isinstance(data, dict):
            return p
        p.id = str(data.get("id") or _new_id())
        p.name = str(data.get("name") or "ComfyUI")
        p.distro = str(data.get("distro") or "")
        p.user = str(data.get("user") or "")
        p.comfy_dir = str(data.get("comfy_dir") or "~/ComfyUI")
        p.python = str(data.get("python") or "venv/bin/python")
        try:
            p.port = int(data.get("port", 8188))
        except (TypeError, ValueError, OverflowError):
            p.port = 8188
        p.listen = str(data.get("listen") or "127.0.0.1")
        mode = data.get("gpu_mode", GPU_AUTO)
        p.gpu_mode = mode if mode in GPU_MODES else GPU_AUTO
        seen = set()
        gpus = data.get("gpus")
        for raw in gpus if isinstance(gpus, list) else []:
            ref = GpuRef.from_dict(raw)
            if ref and ref.uuid not in seen:
                seen.add(ref.uuid)
                p.gpus.append(ref)
        options = data.get("options") or {}
        if isinstance(options, dict):
            for key, raw in options.items():
                p.options[str(key)] = OptionState.from_dict(raw)
        p.extra_args = str(data.get("extra_args") or "")
        env = data.get("env")
        for pair in env if isinstance(env, list) else []:
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                p.env.append(["" if pair[0] is None else str(pair[0]),
                              "" if pair[1] is None else str(pair[1])])
        p.pre_commands = str(data.get("pre_commands") or "")
        p.auto_open = bool(data.get("auto_open", False))
        p.open_rulzflow = data.get("open_rulzflow") is True
        p.checked = bool(data.get("checked", False))
        return p

    def clone(self, name: str) -> "Profile":
        other = copy.deepcopy(self)
        other.id = _new_id()
        other.name = name
        return other

    def snapshot(self) -> "Profile":
        """An independent copy with the same id (used for a running instance)."""
        return copy.deepcopy(self)


@dataclass
class Settings:
    stagger_seconds: float = 3.0   # pause between instances when several start together
    stop_timeout: int = 10         # seconds to wait after Ctrl+C before forcing a stop
    log_max_lines: int = 5000
    last_profile: str = ""
    geometry: str = ""             # base64 of the saved window geometry
    splitter: list[int] = field(default_factory=list)
    close_to_tray: bool = False    # closing the window hides it to the tray instead of exiting
    theme: str = THEME_DARK        # "dark" (default) or "light"

    def to_dict(self) -> dict:
        return {
            "stagger_seconds": self.stagger_seconds,
            "stop_timeout": self.stop_timeout,
            "log_max_lines": self.log_max_lines,
            "close_to_tray": self.close_to_tray,
            "theme": self.theme,
            "last_profile": self.last_profile,
            "geometry": self.geometry,
            "splitter": list(self.splitter),
        }

    @staticmethod
    def from_dict(data) -> "Settings":
        s = Settings()
        if not isinstance(data, dict):
            return s

        def number(key, default, lo, hi, cast):
            try:
                value = float(data.get(key, default))
                if not math.isfinite(value):
                    return default
                return max(lo, min(hi, cast(value)))
            except (TypeError, ValueError, OverflowError):
                return default

        s.stagger_seconds = number("stagger_seconds", 3.0, 0.0, 120.0, float)
        s.stop_timeout = number("stop_timeout", 10, 1, 300, int)
        s.log_max_lines = number("log_max_lines", 5000, 200, 200000, int)
        s.close_to_tray = data.get("close_to_tray") is True
        theme = data.get("theme")
        s.theme = theme if theme in THEMES else THEME_DARK
        s.last_profile = str(data.get("last_profile") or "")
        s.geometry = str(data.get("geometry") or "")
        splitter = data.get("splitter") or []
        if isinstance(splitter, list):
            s.splitter = [int(x) for x in splitter
                          if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)]
        return s


class ConfigStore:
    """Loads and saves the config file. Saving is atomic (write, then rename)."""

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else default_config_dir() / "config.json"
        self.profiles: list[Profile] = []
        self.settings = Settings()
        self.load_error: str = ""

    # -- loading and saving -------------------------------------------------
    def load(self) -> None:
        self.profiles = []
        self.settings = Settings()
        self.load_error = ""
        if not self.path.exists():
            self.profiles.append(Profile())
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("the file does not hold a JSON object")
        except (OSError, ValueError) as exc:
            # Keep the unreadable file instead of overwriting it on the next save.
            backup = self.path.with_name(f"{self.path.stem}.unreadable-{int(time.time())}.json")
            try:
                os.replace(self.path, backup)
                self.load_error = (f"The config file could not be read ({exc}). "
                                   f"It was kept as {backup.name} and a new one was started.")
            except OSError:
                self.load_error = f"The config file could not be read ({exc})."
            self.profiles.append(Profile())
            return
        self.settings = Settings.from_dict(data.get("settings"))
        seen = set()
        for raw in data.get("profiles") or []:
            profile = Profile.from_dict(raw)
            if profile.id in seen:
                profile.id = _new_id()
            seen.add(profile.id)
            self.profiles.append(profile)
        if not self.profiles:
            self.profiles.append(Profile())

    def to_dict(self) -> dict:
        return {
            "version": CONFIG_VERSION,
            "settings": self.settings.to_dict(),
            "profiles": [p.to_dict() for p in self.profiles],
        }

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        text = json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
        with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text + "\n")
        os.replace(tmp, self.path)

    # -- profile helpers ----------------------------------------------------
    def find(self, profile_id: str) -> Profile | None:
        for profile in self.profiles:
            if profile.id == profile_id:
                return profile
        return None

    def unique_name(self, base: str) -> str:
        names = {p.name for p in self.profiles}
        if base not in names:
            return base
        n = 2
        while f"{base} ({n})" in names:
            n += 1
        return f"{base} ({n})"

    def free_port(self, start: int = 8188) -> int:
        used = {p.port for p in self.profiles}
        port = start
        while port in used and port < 65535:
            port += 1
        return port

    def add(self, profile: Profile, after: Profile | None = None) -> None:
        if after is not None and after in self.profiles:
            self.profiles.insert(self.profiles.index(after) + 1, profile)
        else:
            self.profiles.append(profile)

    def remove(self, profile: Profile) -> None:
        if profile in self.profiles:
            self.profiles.remove(profile)
