"""Files written whole or not at all: a crash, a full disk or a killed process during a save leaves the previous file
(or none), never a torn one that loads as garbage. The data goes to <name>.tmp beside the target, which then takes the
target's place in one move."""
from __future__ import annotations

import contextlib
import os
import time
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


@contextlib.contextmanager
def writing(path):
    """Yields a temporary path beside `path`; what the block writes there becomes `path` when the block ends, and is
    removed if the block fails."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    try:
        yield tmp
        _replace(tmp, path)
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)


def save_safetensors(tensors: dict, path, metadata: dict | None = None) -> None:
    with writing(path) as tmp:
        save_file(tensors, str(tmp), metadata=metadata)
