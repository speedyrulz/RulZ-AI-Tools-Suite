"""Persistent user settings (token, download folder, UI preferences)."""

from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from .platform_utils import IS_WINDOWS, config_dir, default_download_dir


@dataclass
class Settings:
    token: str = ''
    download_dir: str = ''
    dark_mode: str = 'dark'  # 'dark' | 'light' ('auto', from older versions, follows the system)
    look_chosen: bool = False  # False until the first-start "Choose your look" dialog was answered
    show_hidden: bool = True
    load_last_modified: bool = False  # slower: asks the Hub for the last commit of every file
    page_size: int = 100
    host: str = '127.0.0.1'
    port: int = 8765
    favorites: list[dict] = field(default_factory=list)  # [{"kind": "model", "id": "user/name"}]
    recent: list[dict] = field(default_factory=list)

    def effective_download_dir(self) -> Path:
        return Path(self.download_dir).expanduser() if self.download_dir else default_download_dir()

    def is_favorite(self, kind: str, repo_id: str) -> bool:
        return any(f.get('kind') == kind and f.get('id') == repo_id for f in self.favorites)

    def toggle_favorite(self, kind: str, repo_id: str) -> bool:
        if self.is_favorite(kind, repo_id):
            self.favorites = [f for f in self.favorites if not (f.get('kind') == kind and f.get('id') == repo_id)]
            save()
            return False
        self.favorites.append({'kind': kind, 'id': repo_id})
        save()
        return True

    def add_recent(self, kind: str, repo_id: str) -> None:
        entry = {'kind': kind, 'id': repo_id}
        self.recent = [entry] + [r for r in self.recent if r != entry]
        del self.recent[12:]
        save()


_lock = threading.Lock()
_settings: Settings | None = None


def settings_path() -> Path:
    return config_dir() / 'settings.json'


def get() -> Settings:
    global _settings
    if _settings is None:
        _settings = _load()
    return _settings


def _load() -> Settings:
    path = settings_path()
    if not path.exists():
        return Settings()
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return Settings()
    known = {f.name for f in fields(Settings)}
    return Settings(**{k: v for k, v in data.items() if k in known})


def save() -> None:
    settings = get()
    path = settings_path()
    with _lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(asdict(settings), indent=2), encoding='utf-8')
        if not IS_WINDOWS:
            os.chmod(tmp, 0o600)  # the file contains the HF token
        os.replace(tmp, path)
