"""Persistent user settings (token, download folder, UI preferences)."""

from __future__ import annotations

import json
import os
import threading
import time
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
    # vast.ai
    vast_api_key: str = ''  # empty: VAST_API_KEY or the key the vastai CLI saved
    ssh_key: str = ''  # empty: ssh's own defaults (~/.ssh/id_ed25519, id_rsa...)
    vast_conn: dict = field(default_factory=dict)  # the attached instance (see vast.Conn)
    vast_recent_dests: list[str] = field(default_factory=list)
    vast_layout: str = 'keep'

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
_unread = False  # the settings file existed but could not be read at start


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
    global _unread
    text = None
    for attempt in range(3):  # a virus scanner may hold the file for a moment
        try:
            text = path.read_text(encoding='utf-8-sig')  # -sig: a file once saved by Notepad starts with a BOM
            break
        except OSError:
            time.sleep(0.2 * (attempt + 1))
    if text is None:
        _unread = True  # save() keeps this file aside before writing over it
        return Settings()
    try:
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError('not a JSON object')
    except ValueError:
        # Damaged: start with defaults, but keep the file aside. The next save() would otherwise overwrite it,
        # and the token, keys and favorites in it would be gone for good.
        try:
            path.replace(path.with_name(f'{path.name}.broken-{time.strftime("%Y%m%d-%H%M%S")}'))
        except OSError:
            pass
        return Settings()
    known = {f.name: f for f in fields(Settings)}
    defaults = Settings()
    values = {}
    for k, v in data.items():
        if k in known and isinstance(v, type(getattr(defaults, k))):  # a wrong type falls back to the default
            values[k] = v
    return Settings(**values)


def save() -> None:
    global _unread
    settings = get()
    path = settings_path()
    with _lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        if _unread and path.exists():  # never read at start: keep it rather than overwrite it with defaults
            try:
                path.replace(path.with_name(f'{path.name}.unread-{time.strftime("%Y%m%d-%H%M%S")}'))
            except OSError:
                pass
        _unread = False
        tmp = path.with_suffix('.tmp')
        text = json.dumps(asdict(settings), indent=2)
        if IS_WINDOWS:
            tmp.write_text(text, encoding='utf-8')
        else:
            # Created private (the file holds the HF token and the vast.ai key): never readable by others, not
            # even for the moment between writing and a later chmod.
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                f.write(text)
            os.chmod(tmp, 0o600)  # an older temp file may have had wider permissions
        os.replace(tmp, path)
