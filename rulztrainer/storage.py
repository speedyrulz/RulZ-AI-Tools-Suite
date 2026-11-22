"""Where RulZ AI Trainer keeps its Previous Runs history and its Fleet downloads, and how to move either one.

Two save folders can be changed by the user (the Previous Runs page and the Fleet page each have a "Change save folder"
button):

  * the history: default <install>/output/_history. The choice is kept per user in storage.json, in
    $RULZ_TRAINER_HOME (default ~/.rulz_trainer), key "history_dir" (not set = the default);
  * the Fleet's downloads: default ~/.rulz_vast/runs. The choice lives in the Fleet's state.json
    (settings["pulls_dir"], rulz_fleet.py) because the CLI and the daemon share it.

This module holds the pieces both share (standard library only): the settings file, the checks on a new folder, the
marker file that says "this is a RulZ save folder", and the journaled move. A move writes a "move" entry (from, to,
pid, started) before it touches anything and removes it when it is done, so a crash leaves a record that offers Resume
or Cancel on the next start.

The move: (1) os.rename when the target does not exist and the rename works (one atomic step, same volume); else
(2) merge-copy every file (skipping files already there at the same size and time, so a resumed copy continues),
verify every size, only then switch the setting, then delete the old files (what will not delete is reported and left),
then remove the journal. A failure before the switch leaves the setting on the old folder."""
from __future__ import annotations

import contextlib
import json
import os
import shutil
import threading
import time
import uuid
from pathlib import Path

MARKER_NAME = ".rulz_save_folder"
MARGIN_BYTES = 64 * 1024 * 1024  # free space wanted beyond the data itself, when copying to another volume
STILL_ACTIVE = 259


class StorageError(Exception):
    """A change or a move that cannot be carried out; the message says why and what state things are in."""


# ------------------------------------------------------------------ the per-user settings file

def home_dir() -> Path:
    return Path(os.environ.get("RULZ_TRAINER_HOME") or Path.home() / ".rulz_trainer")


def settings_path() -> Path:
    return home_dir() / "storage.json"


def read_settings() -> dict:
    try:
        data = json.loads(settings_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def write_json_atomic(path: Path, data) -> None:
    """Temp file next to it, then os.replace (tried again for a moment while Windows refuses it)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}-{uuid.uuid4().hex[:8]}.tmp")
    try:
        tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
        for attempt in range(20):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.05)
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)


def update_settings(change) -> dict:
    data = read_settings()
    change(data)
    write_json_atomic(settings_path(), data)
    return data


def history_dir_setting() -> Path | None:
    """The folder the user chose for the history, or None (use the default)."""
    value = read_settings().get("history_dir")
    return Path(value) if isinstance(value, str) and value else None


def history_move() -> dict | None:
    move = read_settings().get("move")
    return move if isinstance(move, dict) and move.get("from") and move.get("to") else None


def set_history_move(journal: dict) -> None:
    update_settings(lambda d: d.__setitem__("move", journal))


def clear_history_move() -> None:
    update_settings(lambda d: d.pop("move", None))


def set_history_dir(path: Path | None) -> None:
    def change(d):
        if path is None:
            d.pop("history_dir", None)
        else:
            d["history_dir"] = str(path)
    update_settings(change)


_ACTIVE: set[str] = set()  # the kinds of move this process is running now
_ACTIVE_LOCK = threading.Lock()


def begin_move(kind: str) -> bool:
    """Claim the right to move / resume / cancel the `kind` folder in this process; False when it is claimed."""
    with _ACTIVE_LOCK:
        if kind in _ACTIVE:
            return False
        _ACTIVE.add(kind)
        return True


def end_move(kind: str) -> None:
    with _ACTIVE_LOCK:
        _ACTIVE.discard(kind)


def moving_here(kind: str) -> bool:
    with _ACTIVE_LOCK:
        return kind in _ACTIVE


@contextlib.contextmanager
def move_guard(kind: str):
    """Only one move, resume or cancel of a folder at a time in this process (Resume and Cancel, or two moves,
    cannot overlap)."""
    if not begin_move(kind):
        raise StorageError(f"A change of the {kind} folder is already going.")
    try:
        yield
    finally:
        end_move(kind)


def journal_live(journal, kind: str) -> bool:
    """A move journal is owned by a running move: another live process, or this one while it is moving."""
    if not isinstance(journal, dict) or not pid_alive(journal.get("pid")):
        return False
    return int(journal["pid"]) != os.getpid() or moving_here(kind)


def history_move_active() -> bool:
    """A history move is going in a live process (writers wait for it: they skip a save, which the next one repeats)."""
    return journal_live(history_move(), "history")


def pid_alive(pid) -> bool:
    """Whether a process with this pid runs. (os.kill(pid, 0) on Windows would end the process.)"""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        kernel = ctypes.windll.kernel32
        handle = kernel.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == STILL_ACTIVE
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    return True


# ------------------------------------------------------------------ paths and the marker file

def norm(path) -> str:
    try:
        resolved = Path(path).expanduser().resolve()
    except (OSError, RuntimeError):
        resolved = Path(os.path.abspath(str(path)))
    return os.path.normcase(str(resolved))


def same_path(a, b) -> bool:
    """The same folder: equal resolved paths (case-insensitive on Windows), or os.path.samefile where both exist
    (a mapped drive, a UNC path, a subst drive or a junction to it)."""
    if norm(a) == norm(b):
        return True
    try:
        return Path(a).exists() and Path(b).exists() and os.path.samefile(a, b)
    except OSError:
        return False


def is_within(child, parent) -> bool:
    """child is parent or lies inside it (aliases of either included)."""
    c, p = norm(child), norm(parent)
    if c == p or c.startswith(p.rstrip("\\/") + os.sep):
        return True
    if Path(parent).exists():
        try:
            for ancestor in (Path(child).resolve(), *Path(child).resolve().parents):
                if ancestor.exists() and os.path.samefile(ancestor, parent):
                    return True
        except OSError:
            pass
    return False


def nearest_existing(path) -> Path:
    p = Path(path)
    for candidate in (p, *p.parents):
        if candidate.exists():
            return candidate
    return p


def read_marker(root) -> dict | None:
    try:
        data = json.loads((Path(root) / MARKER_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def write_marker(root, kind: str) -> None:
    """Mark a folder as a RulZ save folder of this kind ("history" or "downloads"); an existing marker is kept."""
    root = Path(root)
    if not root.is_dir():
        return
    old = read_marker(root)
    if old and old.get("kind") == kind:
        return
    with contextlib.suppress(OSError):
        write_json_atomic(root / MARKER_NAME, {"kind": kind, "created": time.strftime("%Y-%m-%d %H:%M:%S")})


def has_marker(root, kind: str) -> bool:
    m = read_marker(root)
    return bool(m) and m.get("kind") == kind


def is_network_path(path) -> bool:
    """A UNC path or a mapped network drive (Windows deletes from there for good, past the Recycle Bin)."""
    text = str(path)
    if text.startswith("\\\\") or text.startswith("//"):
        return True
    if os.name == "nt" and len(text) > 1 and text[1] == ":":
        try:
            import ctypes

            return ctypes.windll.kernel32.GetDriveTypeW(text[:3]) == 4  # DRIVE_REMOTE
        except (OSError, AttributeError):
            return False
    return False


def tree_stats(path) -> tuple[int, int]:
    """(files, bytes) under a folder, the marker file left out. A folder that cannot be read raises StorageError."""
    files = size = 0
    try:
        for here, _dirs, names in os.walk(path, onerror=_raise):
            for name in names:
                if name == MARKER_NAME and Path(here) == Path(path):
                    continue
                try:
                    size += os.path.getsize(os.path.join(here, name))
                    files += 1
                except OSError:
                    pass
    except OSError as e:
        raise StorageError(f"A folder in {path} cannot be read ({e}).") from e
    return files, size


def human_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def has_data(root) -> bool:
    root = Path(root)
    try:
        return root.is_dir() and any(p.name != MARKER_NAME for p in root.iterdir())
    except OSError:
        return False


# ------------------------------------------------------------------ checking a new folder

def validate_root(text, *, kind: str, old, default, trusted=(), output_dir=None) -> tuple[Path, list[str]]:
    """The folder the user typed, checked. Returns (path, warnings); raises StorageError with a plain message.

    old: the current root; default: the built-in root (always allowed, also when it holds unmarked data, as it is the
    app's own); trusted: more such roots; output_dir: for the history, the job folders' parent: a root inside it
    is refused (the job scanners would walk it) unless it is the default."""
    text = str(text or "").strip().strip('"')
    if not text:
        raise StorageError("Type the full path of a folder.")
    new = Path(text).expanduser()
    if not new.is_absolute():
        raise StorageError("Use the full path of the folder, for example D:\\RulZ\\history (relative paths are "
                           "not accepted).")
    new = Path(os.path.abspath(str(new)))
    is_default = same_path(new, default)
    if same_path(new, old):
        raise StorageError("That is the folder already in use.")
    if is_within(new, old):
        raise StorageError(f"That folder is inside the current save folder ({old}). Pick one outside it.")
    if is_within(old, new):
        raise StorageError(f"That folder contains the current save folder ({old}). Pick one that does not.")
    if kind == "history" and output_dir is not None and not is_default and is_within(new, output_dir):
        raise StorageError(f"A history folder cannot be inside {output_dir}: that is where training jobs are looked "
                           "for. Pick another place.")
    warnings: list[str] = []
    if new.exists():
        if not new.is_dir():
            raise StorageError("That path is a file, not a folder.")
        if has_data(new) and not has_marker(new, kind) \
                and not any(same_path(new, t) for t in (default, *trusted)):
            raise StorageError("That folder is not empty and was not made by RulZ AI Trainer as a "
                               f"{'history' if kind == 'history' else 'downloads'} folder. Pick an empty or new "
                               "folder (nothing there is touched).")
    probe_dir = nearest_existing(new)
    if not probe_dir.is_dir():
        raise StorageError("No part of that path exists as a folder.")
    probe = probe_dir / f".rulz_write_test_{uuid.uuid4().hex[:8]}"
    try:
        probe.write_text("x", encoding="utf-8")
        probe.unlink()
    except OSError as e:
        raise StorageError(f"That place cannot be written to ({e}).") from e
    if is_network_path(new):
        warnings.append("This is on a network share. Files deleted from it (Remove) skip the Recycle Bin and are "
                        "gone for good.")
    return new, warnings


def check_space(source_bytes: int, old, new) -> None:
    """Free space on the target volume covers the data (plus a margin) when the target is on another volume."""
    probe = nearest_existing(new)
    try:
        other_volume = os.stat(nearest_existing(old)).st_dev != os.stat(probe).st_dev
        free = shutil.disk_usage(probe).free
    except OSError:
        return
    if other_volume and free < source_bytes + MARGIN_BYTES:
        raise StorageError(f"Not enough free space there: {human_size(source_bytes)} to move, "
                           f"{human_size(free)} free.")


# ------------------------------------------------------------------ the move

PHASES_NO_CANCEL = ("renamed", "switched")  # once the data is only in the new folder, or the switch began


def _raise(error: OSError) -> None:
    raise error


def _walk(path):
    """os.walk that stops on a folder it cannot read (an unreadable folder must stop a move before the switch)."""
    try:
        yield from os.walk(path, onerror=_raise)
    except OSError as e:
        raise StorageError(f"A folder in {path} cannot be read ({e}). Nothing was changed.") from e


def _same_file(src: Path, dst: Path) -> bool:
    try:
        a, b = src.stat(), dst.stat()
    except OSError:
        return False
    return a.st_size == b.st_size and abs(a.st_mtime_ns - b.st_mtime_ns) <= 2_000_000_000


def _files(old: Path):
    """(relative path, source path) of every file under old, the old marker file left out."""
    for here, _dirs, names in _walk(old):
        rel = Path(here).relative_to(old)
        for name in names:
            if rel == Path(".") and name == MARKER_NAME:
                continue
            yield rel / name, Path(here) / name


def _conflicts(old: Path, new: Path) -> list[str]:
    """Files that exist in new and differ from old's copy (a copy would overwrite them)."""
    return [str(rel) for rel, src in _files(old) if (new / rel).is_file() and not _same_file(src, new / rel)]


def _copy_tree(old: Path, new: Path, progress=None, catch_up: bool = False) -> None:
    """Merge-copy old into new. A file already there at the same size and time is skipped (a resumed copy
    continues). catch_up: a file already in new is replaced only by a newer one from old."""
    new.mkdir(parents=True, exist_ok=True)
    for here, _dirs, _names in _walk(old):
        (new / Path(here).relative_to(old)).mkdir(parents=True, exist_ok=True)
    for rel, src in _files(old):
        dst = new / rel
        if dst.is_file():
            if _same_file(src, dst):
                continue
            if catch_up and src.stat().st_mtime_ns <= dst.stat().st_mtime_ns:
                continue
        tmp = dst.with_name(dst.name + ".rulzmove")
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)
        if progress:
            progress(src)


def _verify(old: Path, new: Path) -> None:
    bad = []
    for rel, src in _files(old):
        dst = new / rel
        try:
            if not dst.is_file() or dst.stat().st_size != src.stat().st_size:
                bad.append(str(rel))
        except OSError:
            bad.append(str(rel))
    if bad:
        raise StorageError(f"{len(bad)} file(s) did not copy correctly (first: {bad[0]}).")


def _delete_tree(root: Path) -> list[str]:
    """Delete everything inside root, then root itself; return what would not go (nothing is ignored)."""
    left: list[str] = []
    for here, dirs, names in os.walk(root, topdown=False):
        for name in names:
            p = Path(here) / name
            try:
                os.chmod(p, 0o666)
                p.unlink()
            except OSError as e:
                left.append(f"{p} ({e})")
        for name in dirs:
            p = Path(here) / name
            try:
                p.rmdir()
            except OSError:
                if not any(str(p) in item for item in left):
                    left.append(str(p))
    try:
        root.rmdir()
    except OSError:
        if not left:
            left.append(str(root))
    return left


def move_folder(old, new, *, kind: str, switch, set_move, clear_move, journal: dict | None = None,
                progress=None, check=None) -> dict:
    """Move a save folder's contents from old to new and call switch() once the data is safely there.

    switch(): make `new` the setting (and rewrite what points into old); it must be idempotent: it is run again by
    Resume. set_move(journal) / clear_move(): keep the journal where a restart finds it. journal: the leftover of an
    earlier attempt (Resume). check(): called right before the switch; it may raise StorageError to stop (the journal
    stays). The journal's "phase": "copying" (the setting is still old; Cancel is possible),
    "renamed" (the folder was renamed: the data is only in new), "switched" (written before switch() runs: only
    Resume can go on). Returns {"method": "rename"|"copy"|"nothing", "files", "bytes", "leftovers": [...]}.
    Raises StorageError before the switch with the setting still on `old` and the journal kept."""
    old, new = Path(old), Path(new)
    if journal is None:
        journal = {"from": str(old), "to": str(new), "kind": kind, "started": time.time(), "phase": "copying",
                   "target_was_empty": not has_data(new)}
    journal = {**journal, "pid": os.getpid()}
    journal.pop("failed", None)
    phase = journal.get("phase") or "copying"
    set_move(journal)
    result = {"method": "nothing", "files": 0, "bytes": 0, "leftovers": [], "path": new}

    def save(**fields) -> None:
        journal.update(fields)
        set_move(journal)

    try:
        if phase == "copying":
            if not old.is_dir():
                raise StorageError(f"The old folder {old} cannot be found; reconnect the drive and resume.")
            result["files"], result["bytes"] = tree_stats(old)
            if check is not None:
                check()  # before a rename, which cannot be taken back
            if not new.exists():
                try:
                    new.parent.mkdir(parents=True, exist_ok=True)
                    os.rename(old, new)
                    result["method"] = "rename"
                except OSError:
                    pass
            if result["method"] == "rename":
                save(phase="renamed")
            else:
                clash = _conflicts(old, new)
                if clash and not journal.get("target_was_empty"):
                    raise StorageError(f"{len(clash)} file(s) in {new} differ from the ones in {old} with the same "
                                       f"name (first: {clash[0]}). Nothing was copied. Pick another folder.")
                check_space(result["bytes"], old, new)
                _copy_tree(old, new, progress)
                _verify(old, new)
                result["method"] = "copy"
        elif old.is_dir():
            result["files"], result["bytes"] = tree_stats(old)
            result["method"] = "copy"
        new.mkdir(parents=True, exist_ok=True)
        write_marker(new, kind)
        if check is not None and phase == "copying" and result["method"] != "rename":
            check()
        save(phase="switched")  # from here on only Resume goes on: the switch may be half done
        switch()
        if old.is_dir():
            _copy_tree(old, new, progress, catch_up=True)  # anything written to old since the first pass
            _verify(old, new)
            result["leftovers"] = _delete_tree(old)
        clear_move()
    except (StorageError, OSError) as e:
        # the journal stays (Resume, or Cancel before the switch), but no process owns it: writers stop waiting
        with contextlib.suppress(Exception):
            set_move({**journal, "pid": 0, "failed": str(e)})
        if isinstance(e, StorageError):
            raise
        raise StorageError(f"The move stopped: {e}. Your data is safe in {old} or {new}. "
                           "Resume or cancel the move.") from e
    return result


def cancel_move(journal: dict, clear_move) -> dict:
    """Cancel an unfinished move before its switch: files it copied into a target that was empty or new are removed
    (only those that match the old folder's file by size and time; a .rulzmove temp file only when the old folder has
    that file); anything else is left and named. The setting stays on the old folder. Refused once the data is only in
    the new folder or the switch began: Resume finishes it."""
    old, new = Path(journal["from"]), Path(journal["to"])
    if journal.get("phase") in PHASES_NO_CANCEL:
        raise StorageError("The switch to the new folder has begun, so this move cannot be cancelled. Resume it to "
                           "finish.")
    if not old.is_dir():
        raise StorageError(f"The old folder {old} cannot be found; reconnect the drive and try again.")
    removed, kept = 0, []
    if new.is_dir() and journal.get("target_was_empty"):
        for here, dirs, names in os.walk(new, topdown=False):
            rel = Path(here).relative_to(new)
            for name in names:
                p = Path(here) / name
                if name == MARKER_NAME and rel == Path("."):
                    continue
                twin = old / rel / (name[:-len(".rulzmove")] if name.endswith(".rulzmove") else name)
                matches = twin.is_file() and (name.endswith(".rulzmove") or _same_file(twin, p))
                if matches:
                    with contextlib.suppress(OSError):
                        p.unlink()
                        removed += 1
                else:
                    kept.append(str(p))
            for name in dirs:
                with contextlib.suppress(OSError):
                    (Path(here) / name).rmdir()
        if not kept:
            with contextlib.suppress(OSError):
                (new / MARKER_NAME).unlink(missing_ok=True)
                new.rmdir()
    elif new.is_dir():
        kept.append(str(new))
    clear_move()
    return {"removed": removed, "kept": kept}


# ------------------------------------------------------------------ the Fleet's downloads folder, read from outside

def vast_home() -> Path:
    return Path(os.environ.get("RULZ_VAST_HOME") or Path.home() / ".rulz_vast")


def fleet_pull_settings(vast: Path | None = None) -> dict:
    """The download-folder settings in the Fleet's state.json (read directly, no Fleet object): pulls_dir (the
    chosen folder, or absent), pulls_roots (folders an earlier "start fresh" left behind), move (a journal)."""
    path = (Path(vast) if vast else vast_home()) / "fleet" / "state.json"
    try:
        settings = json.loads(path.read_text(encoding="utf-8")).get("settings")
    except (OSError, ValueError, AttributeError):
        return {}
    return settings if isinstance(settings, dict) else {}


def fleet_pull_roots(vast: Path | None = None) -> list[Path]:
    """Every folder that can hold downloaded runs: the chosen one first, then the default (processes of an older
    version keep writing there), then the marked roots left behind by a "start fresh"."""
    base = Path(vast) if vast else vast_home()
    settings = fleet_pull_settings(base)
    default = base / "runs"
    roots: list[Path] = []
    chosen = settings.get("pulls_dir")
    candidates = [Path(chosen)] if isinstance(chosen, str) and chosen else []
    candidates.append(default)
    candidates += [Path(p) for p in settings.get("pulls_roots") or [] if isinstance(p, str) and has_marker(p, "downloads")]
    for c in candidates:
        if not any(same_path(c, r) for r in roots):
            roots.append(c)
    return roots
