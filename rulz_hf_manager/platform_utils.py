"""Platform helpers: Windows / WSL / Linux detection, local paths, opening URLs and folders."""

from __future__ import annotations

import functools
import os
import re
import shutil
import string
import subprocess
import sys
import webbrowser
from pathlib import Path

IS_WINDOWS = sys.platform.startswith('win')


@functools.cache
def is_wsl() -> bool:
    if IS_WINDOWS:
        return False
    if os.environ.get('WSL_DISTRO_NAME') or os.environ.get('WSL_INTEROP'):
        return True
    try:
        return 'microsoft' in Path('/proc/version').read_text(encoding='utf-8', errors='ignore').lower()
    except OSError:
        return False


def platform_label() -> str:
    if IS_WINDOWS:
        return 'Windows'
    if is_wsl():
        return f"WSL ({os.environ.get('WSL_DISTRO_NAME', 'Linux')})"
    if sys.platform == 'darwin':
        return 'macOS'
    return 'Linux'


@functools.cache
def windows_home_in_wsl() -> Path | None:
    """Return the Windows user profile folder as seen from WSL (e.g. /mnt/c/Users/me)."""
    if not is_wsl():
        return None
    try:
        out = subprocess.run(['cmd.exe', '/c', 'echo %USERPROFILE%'], capture_output=True, text=True,
                             timeout=5, cwd='/mnt/c' if Path('/mnt/c').exists() else None)
        win_path = out.stdout.strip().splitlines()[-1].strip() if out.stdout.strip() else ''
        if win_path and '%' not in win_path:
            p = Path(to_local_path(win_path))
            if p.exists():
                return p
    except Exception:
        pass
    users = Path('/mnt/c/Users')
    if users.exists():
        candidates = [p for p in users.iterdir()
                      if p.is_dir() and p.name not in {'Public', 'Default', 'Default User', 'All Users'}]
        if len(candidates) == 1:
            return candidates[0]
    return None


_WIN_PATH_RE = re.compile(r'^([a-zA-Z]):[\\/]*(.*)$')


def to_local_path(text: str) -> str:
    """Normalize a path typed or pasted by the user.

    Strips quotes, expands ``~`` and, when running inside WSL, converts Windows paths
    such as ``C:\\Users\\me\\model`` to ``/mnt/c/Users/me/model``.
    """
    text = text.strip().strip('"').strip("'")
    if not text:
        return text
    if not IS_WINDOWS:
        m = _WIN_PATH_RE.match(text)
        if m and is_wsl():
            drive, rest = m.group(1).lower(), m.group(2).replace('\\', '/')
            return f'/mnt/{drive}/{rest}'.rstrip('/') or f'/mnt/{drive}'
    return os.path.expanduser(text)


def default_download_dir() -> Path:
    win_home = windows_home_in_wsl()
    if win_home is not None:
        return win_home / 'Downloads' / 'HuggingFace'
    return Path.home() / 'Downloads' / 'HuggingFace'


def windows_drive_letters() -> list[str]:
    """The drive letters Windows has, from one bitmask call. Asking each letter with ``exists()`` instead waits on
    disconnected network drives and empty card readers, seconds per letter, while the file picker opens."""
    try:
        import ctypes  # noqa: PLC0415
        mask = ctypes.windll.kernel32.GetLogicalDrives()  # type: ignore[attr-defined]
    except (ImportError, AttributeError, OSError):
        mask = 0
    if not mask:
        return [c for c in string.ascii_uppercase if os.path.exists(f'{c}:\\')]
    return [c for i, c in enumerate(string.ascii_uppercase) if mask >> i & 1]


def quick_locations() -> list[tuple[str, str, Path]]:
    """(label, icon, path) shortcuts shown in the local file picker."""
    home = Path.home()
    locations: list[tuple[str, str, Path]] = [('Home', 'home', home)]
    win_home = windows_home_in_wsl()
    base = win_home or home
    for name, icon in (('Desktop', 'desktop_windows'), ('Downloads', 'download'), ('Documents', 'description')):
        p = base / name
        if p.exists():
            locations.append((name, icon, p))
    if win_home is not None:
        locations.append(('Windows home', 'window', win_home))
    if IS_WINDOWS:
        for letter in windows_drive_letters():
            locations.append((f'{letter}:', 'storage', Path(f'{letter}:\\')))
    else:
        locations.append(('/', 'storage', Path('/')))
        mnt = Path('/mnt')
        if is_wsl() and mnt.exists():
            for p in sorted(mnt.iterdir()):
                if len(p.name) == 1 and p.is_dir():
                    locations.append((f'{p.name.upper()}: drive', 'storage', p))
    return locations


def open_url(url: str) -> None:
    """Open a URL in the user's browser (handles WSL by delegating to Windows)."""
    if is_wsl():
        for cmd in (['wslview', url], ['explorer.exe', url], ['cmd.exe', '/c', 'start', '', url]):
            if shutil.which(cmd[0]):
                try:
                    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                     cwd='/mnt/c' if Path('/mnt/c').exists() else None)
                    return
                except OSError:
                    continue
    webbrowser.open(url)


def open_folder(path: str | Path) -> bool:
    """Reveal a local folder in the OS file manager. Returns False if not possible."""
    path = Path(path)
    if not path.exists():
        return False
    try:
        if IS_WINDOWS:
            os.startfile(str(path))  # type: ignore[attr-defined]  # noqa: S606
        elif is_wsl():
            win = subprocess.run(['wslpath', '-w', str(path)], capture_output=True, text=True, timeout=5).stdout.strip()
            subprocess.Popen(['explorer.exe', win or str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif sys.platform == 'darwin':
            subprocess.Popen(['open', str(path)])
        elif shutil.which('xdg-open'):
            subprocess.Popen(['xdg-open', str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            return False
        return True
    except Exception:
        return False


def config_dir() -> Path:
    override = os.environ.get('RULZ_HF_CONFIG_DIR')
    if override:
        return Path(override).expanduser()
    if IS_WINDOWS:
        return Path(os.environ.get('APPDATA', Path.home() / 'AppData' / 'Roaming')) / 'RulZ-HF-Manager'
    return Path(os.environ.get('XDG_CONFIG_HOME', Path.home() / '.config')) / 'rulz-hf-manager'
