"""App settings: settings.json in %APPDATA%/RulZ-AI-Trainer-Optimizer (Windows) or ~/.config/rulz-ai-trainer-optimizer."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

IS_WINDOWS = sys.platform.startswith("win")
ADVISORS = ("claude", "codex", "grok", "agy", "none")


def config_dir() -> Path:
    override = os.environ.get("RULZ_OPT_CONFIG_DIR")
    if override:
        return Path(override).expanduser()
    if IS_WINDOWS:
        return Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")) / "RulZ-AI-Trainer-Optimizer"
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "rulz-ai-trainer-optimizer"


def settings_path() -> Path:
    return config_dir() / "settings.json"


@dataclass
class Settings:
    trainer_dir: str = ""  # the RulZ AI Trainer install (empty: look next to this app)
    advisor: str = "claude"  # claude | codex | grok | agy | none
    advisor_model: dict = field(default_factory=dict)  # provider -> model name ("" = the CLI's default)
    export_dir: str = ""  # empty: <trainer_dir>/configs, else ~/Documents/RulZ AI Trainer Optimizer
    run_roots: list = field(default_factory=list)  # extra folders searched for runs
    last: dict = field(default_factory=dict)  # last model, folders, goals
    dark_mode: str = "dark"  # dark | light
    look_chosen: bool = False
    host: str = "127.0.0.1"
    port: int = 8775

    def effective_export_dir(self) -> Path:
        if self.export_dir:
            return Path(self.export_dir).expanduser()
        if self.trainer_dir:
            return Path(self.trainer_dir).expanduser() / "configs"
        return Path.home() / "Documents" / "RulZ AI Trainer Optimizer"

    def model_for(self, provider: str) -> str:
        value = self.advisor_model.get(provider, "") if isinstance(self.advisor_model, dict) else ""
        return value if isinstance(value, str) else ""


_lock = threading.Lock()
_settings: Settings | None = None
_unread = False  # the file existed but could not be read at start: save() keeps it aside


def get() -> Settings:
    global _settings
    if _settings is None:
        _settings = _load()
    return _settings


def reset() -> None:
    """Forget the cached settings (tests, or after the config folder changed)."""
    global _settings, _unread
    _settings = None
    _unread = False


def _load() -> Settings:
    global _unread
    path = settings_path()
    if not path.exists():
        return Settings()
    text = None
    for attempt in range(3):  # a virus scanner may hold the file for a moment
        try:
            text = path.read_text(encoding="utf-8-sig")
            break
        except UnicodeDecodeError:
            text = ""  # not UTF-8: handled below as a damaged file (kept aside, defaults used)
            break
        except OSError:
            time.sleep(0.2 * (attempt + 1))
    if text is None:
        _unread = True
        return Settings()
    try:
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("not a JSON object")
    except ValueError:
        try:  # damaged: keep it aside so the next save does not destroy it
            path.replace(path.with_name(f"{path.name}.broken-{time.strftime('%Y%m%d-%H%M%S')}"))
        except OSError:
            pass
        return Settings()
    known = {f.name for f in fields(Settings)}
    defaults = Settings()
    values = {}
    for key, value in data.items():
        if key in known and isinstance(value, type(getattr(defaults, key))):  # a wrong type falls back
            values[key] = value
    result = Settings(**values)
    if result.advisor not in ADVISORS:
        result.advisor = defaults.advisor
    return result


def save() -> None:
    global _unread
    settings = get()
    path = settings_path()
    with _lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        if _unread and path.exists():
            try:
                path.replace(path.with_name(f"{path.name}.unread-{time.strftime('%Y%m%d-%H%M%S')}"))
            except OSError:
                pass
        _unread = False
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(settings), indent=2), encoding="utf-8")
        os.replace(tmp, path)


def remember(key: str, value) -> None:
    """Store one entry of `last` (last model, folders, goals) and save."""
    get().last[key] = value
    save()
