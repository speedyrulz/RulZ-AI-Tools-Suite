"""Files written whole or not at all: a crash, a full disk or a killed process during a save leaves the previous file
(or none), never a torn one that loads as garbage. The data goes to <name>.tmp beside the target, which then takes the
target's place in one move."""
from __future__ import annotations

import contextlib
import os
import re
import time
import uuid
from pathlib import Path

from safetensors.torch import save_file

REPLACE_TRIES = 6  # on Windows a reader (the WebUI's poll, a scan) can briefly lock the target


def _replace(tmp: Path, path: Path) -> None:
    for attempt in range(REPLACE_TRIES):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == REPLACE_TRIES - 1:
                raise
            time.sleep(0.05 * (attempt + 1))


STALE_TMP_SECONDS = 600  # a temporary file this old, of a writer that is gone, was left by a killed save


def _sweep(path: Path) -> None:
    """Remove the temporary files killed writers left beside `path` (each writer names its own, so the next save no
    longer overwrites one): <name>.<pid>-<hex>.tmp of a process that has ended, and the old fixed <name>.tmp, once
    STALE_TMP_SECONDS old. A fine-tune's deltas file can leave several GB this way."""
    try:
        import psutil
    except ImportError:
        return
    pattern = re.compile(re.escape(path.name) + r"(?:\.(\d+)-[0-9a-f]{8})?\.tmp")
    try:
        siblings = [p for p in path.parent.iterdir() if p.name.startswith(path.name + ".") and p.name.endswith(".tmp")]
    except OSError:
        return
    now = time.time()
    for old in siblings:
        m = pattern.fullmatch(old.name)
        if not m or (m.group(1) and int(m.group(1)) == os.getpid()):
            continue
        with contextlib.suppress(OSError, ValueError):
            if now - old.stat().st_mtime < STALE_TMP_SECONDS:
                continue
            if m.group(1) and psutil.pid_exists(int(m.group(1))):
                continue
            old.unlink()


@contextlib.contextmanager
def writing(path):
    """Yields a temporary path beside `path`; what the block writes there becomes `path` when the block ends, and is
    removed if the block fails. The name is this writer's own (<name>.<pid>-<random>.tmp): two processes saving one
    file (the trainer and a WebUI preview writing faces.json) would otherwise move or delete each other's half."""
    path = Path(path)
    _sweep(path)
    tmp = path.with_name(f"{path.name}.{os.getpid()}-{uuid.uuid4().hex[:8]}.tmp")
    try:
        yield tmp
        _replace(tmp, path)
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)


def save_safetensors(tensors: dict, path, metadata: dict | None = None) -> None:
    with writing(path) as tmp:
        save_file(tensors, str(tmp), metadata=metadata)
