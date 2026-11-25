#!/usr/bin/env python3
"""Getting one local file or folder onto a vast.ai instance (docs/vast-send-contract.md, section 5).

Two routes, both ending in models/sent/<sha256[:12]>/<name> (a folder: models/sent/<tree digest12>/<name>/) with a
record on the instance, so a file that is already there at its size is skipped:

- Straight upload over ssh, resumable: `file-state` says how many bytes of <name>.part are there, the rest goes in 64 MB
  pieces (Remote.run, a shell command: the agent's program already takes stdin), a dropped connection asks again and
  goes on, `file-done` checks size and sha256 and renames. Folders: files of 64 MB or more go the same way one by one,
  the rest by rulz_remote.send (staged), then `file-done --tree`.
- Hugging Face bucket: the PC uploads to the bucket, asks huggingface.co for the signed address WITHOUT following its
  redirect (the token stays on this PC), and gives only that short-lived address to the instance (`file-fetch`, on ssh
  stdin), which downloads it in ranges over 16 connections.

Everything that talks to something takes its collaborator as an argument (remote, api, opener, sleep, sender) so the
tests need no ssh and no network. Nothing here puts a token or a signed address in a message, a file or a command
line. Standard library only; huggingface_hub is imported where it is used (a version with buckets is needed there)."""
from __future__ import annotations

import hashlib
import json
import os
import posixpath
import re
import shlex
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import rulz_remote as rr  # noqa: E402

PIECE = 64 * 1024 * 1024  # bytes per ssh piece
LARGE = 64 * 1024 * 1024  # files of this size or more of a folder go one by one
SENT = "models/sent"
BUCKET_PREFIX = "rulz-trainer"
MAX_FAILURES = 10  # consecutive dropped connections before a transfer gives up
PUT_TIMEOUT = 3600  # seconds one 64 MB piece may take
BATCH_FILES = 20  # small files per staged send: its shell command must stay short (ssh cuts one at ~8190 characters)
BATCH_BYTES = 64 * 1024 * 1024
BUCKET_BATCH_FILES = 16
HASH_BLOCK = 8 * 1024 * 1024
HOST_SUFFIXES = (".huggingface.co", ".hf.co")  # and huggingface.co itself: the hosts a signed address may name
BUCKET_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
_cache_lock = threading.Lock()


class FilesError(Exception):
    """A transfer that can't go on; the message is safe to show (no token, no address). kind: cancelled, ssh, checksum,
    local, remote, hf, expired."""

    def __init__(self, message: str, kind: str = "error"):
        super().__init__(message)
        self.kind = kind


class Cancelled(FilesError):
    def __init__(self, message: str = "Cancelled."):
        super().__init__(message, "cancelled")


class _Transient(Exception):
    """A call that did not get through (ssh dropped, hung, or an answer that was not one); asked again."""


# ---------------------------------------------------------------- names, digests

def scrub(text) -> str:
    """text without anything shaped like a token or an address."""
    text = re.sub(r"https?://\S+", "<address>", str(text))
    text = re.sub(r"hf_[A-Za-z0-9]{10,}", "<token>", text)
    return re.sub(r"(?i)bearer\s+\S+", "Bearer <token>", text)[:400]


def clean_name(name: str) -> str:
    """A basename safe on the instance: letters, digits and . _ + - only, no leading dot, at most 120 characters."""
    cleaned = re.sub(r"[^A-Za-z0-9._+-]+", "_", str(name)).lstrip(".")[:120]
    return cleaned or "file"


def sent_dest(digest12: str, name: str) -> str:
    return f"{SENT}/{digest12}/{clean_name(name)}"


def check_dest(dest: str) -> str:
    """dest must be a canonical posix path below models/sent/<12 hex>/ (nothing may leave models/sent)."""
    parts = dest.split("/") if isinstance(dest, str) else []
    if (len(parts) < 4 or parts[:2] != ["models", "sent"] or not re.fullmatch(r"[0-9a-f]{12}", parts[2])
            or any(p in ("", ".", "..") for p in parts) or any(c in dest for c in "\\\0\n\r")):
        raise FilesError("Not a path below models/sent.", "local")
    return dest


def cache_path(home=None) -> Path:
    return (Path(home) if home else rr.STATE) / "filehash.json"


def _read_cache(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_cache(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    os.replace(tmp, path)


def _check(cancel) -> None:
    if cancel is not None and cancel():
        raise Cancelled()


def digest(path, *, home=None, progress=None, cancel=None) -> str:
    """sha256 of a file, cached in filehash.json by (path, size, mtime_ns). progress(bytes done, size)."""
    p = Path(path)
    try:
        st = p.stat()
    except OSError as e:
        raise FilesError(f"Cannot read {p.name}: {e.strerror or e}", "local") from e
    key = str(p.resolve())
    cache = cache_path(home)
    with _cache_lock:
        hit = _read_cache(cache).get(key)
    if isinstance(hit, dict) and hit.get("size") == st.st_size and hit.get("mtime_ns") == st.st_mtime_ns \
            and re.fullmatch(r"[0-9a-f]{64}", str(hit.get("sha256"))):
        if progress:
            progress(st.st_size, st.st_size)
        return hit["sha256"]
    h, done = hashlib.sha256(), 0
    try:
        with open(p, "rb") as f:
            while True:
                _check(cancel)
                block = f.read(HASH_BLOCK)
                if not block:
                    break
                h.update(block)
                done += len(block)
                if progress:
                    progress(done, st.st_size)
    except OSError as e:
        raise FilesError(f"Cannot read {p.name}: {e.strerror or e}", "local") from e
    if done != st.st_size or p.stat().st_mtime_ns != st.st_mtime_ns:
        raise FilesError(f"{p.name} changed while it was read.", "local")
    result = h.hexdigest()
    with _cache_lock:
        data = _read_cache(cache)
        data[key] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": result}
        _write_cache(cache, data)
    return result


def tree_files(path) -> list[dict]:
    """The files of a folder, [{"rel" (posix), "path", "size"}] by rel; hidden files and folders are left out."""
    root = Path(path)
    rows = []
    for folder, dirs, names in os.walk(root):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for name in sorted(names):
            if name.startswith("."):
                continue
            f = Path(folder) / name
            try:
                rows.append({"rel": f.relative_to(root).as_posix(), "path": f, "size": f.stat().st_size})
            except OSError as e:
                raise FilesError(f"Cannot read {name}: {e.strerror or e}", "local") from e
    return sorted(rows, key=lambda r: r["rel"])


def tree_digest(path, *, home=None, progress=None, cancel=None) -> tuple[str, list[dict]]:
    """(digest of a folder: sha256 over its sorted "rel, size, sha256" lines, its files with "sha256" added).
    progress(bytes hashed so far, total bytes)."""
    files = tree_files(path)
    total, base = sum(f["size"] for f in files), 0
    lines = []
    for f in files:
        f["sha256"] = digest(f["path"], home=home, cancel=cancel,
                             progress=(lambda d, t, b=base: progress(b + d, total)) if progress else None)
        base += f["size"]
        lines.append(f"{f['rel']}\t{f['size']}\t{f['sha256']}")
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest(), files


def target_for(path, *, home=None, progress=None, cancel=None) -> dict:
    """What a file or folder becomes on the instance: {"kind": "file"|"folder", "name", "sha256" (a folder's: its tree
    digest), "d12", "dest", "size", "files": [{"rel", "path", "size", "sha256"}] (folders)}."""
    p = Path(path)
    if p.is_dir():
        sha, files = tree_digest(p, home=home, progress=progress, cancel=cancel)
        name = clean_name(p.name)
        return {"kind": "folder", "name": name, "sha256": sha, "d12": sha[:12], "dest": sent_dest(sha[:12], name),
                "size": sum(f["size"] for f in files), "files": files}
    if not p.is_file():
        raise FilesError(f"{p.name} was not found.", "local")
    sha = digest(p, home=home, progress=progress, cancel=cancel)
    name = clean_name(p.name)
    return {"kind": "file", "name": name, "sha256": sha, "d12": sha[:12], "dest": sent_dest(sha[:12], name),
            "size": p.stat().st_size, "files": []}


# ---------------------------------------------------------------- talking to the agent on the instance

def put_command(dest: str, offset: int) -> str:
    """The shell command for one piece (the piece is its stdin): refused (exit 1) unless <dest>.part is exactly `offset`
    bytes long, else appended. In braces, so a failed `cd` of Remote.command never lets it run elsewhere."""
    check_dest(dest)
    return (f"{{ P={shlex.quote(dest + '.part')}; D={shlex.quote(posixpath.dirname(dest))}; "
            f"mkdir -p \"$D\" && {{ [ -e \"$P\" ] || : > \"$P\"; }} && "
            f"[ \"$(stat -c %s \"$P\")\" = {int(offset)} ] && cat >> \"$P\"; }}")


def agent_call(remote, *args: str, config: bytes | None = None) -> tuple[int, dict | list | None]:
    """The agent's command: (exit status, its JSON answer). A dropped or hung ssh and an answer that is no JSON are
    _Transient; a refusal ("[refused] ...") is a FilesError. A command that failed but said something in JSON (a
    checksum mismatch) is returned with its status."""
    done = remote.agent(*args, config=config)
    code = done.returncode
    out = (done.stdout or b"").decode("utf-8", "replace").strip()
    err = (done.stderr or b"").decode("utf-8", "replace").strip()
    if code in (rr.SSH_FAILED, rr.TIMED_OUT):
        raise _Transient(f"ssh to the instance failed ({args[0]})")
    try:
        data = json.loads(out) if out else None
    except ValueError:
        data = None
    if code == 0 and data is not None:
        return code, data
    if code != 0 and data is not None:
        return code, data
    if "[refused]" in err:
        raise FilesError(scrub(err.split("[refused]", 1)[1].strip()), "remote")
    raise _Transient(f"{args[0]} gave no usable answer")


class _Retry:
    """The consecutive-failure budget of one transfer: MAX_FAILURES drops in a row end it; a success starts it again."""

    def __init__(self, sleep, cancel):
        self.failures, self.sleep, self.cancel = 0, sleep, cancel

    def failed(self, what: str) -> None:
        self.failures += 1
        if self.failures > MAX_FAILURES:
            raise FilesError(f"The instance could not be reached after {MAX_FAILURES} tries ({what}).", "ssh")
        _check(self.cancel)
        self.sleep(min(60, 2 ** self.failures))
        _check(self.cancel)

    def ok(self) -> None:
        self.failures = 0

    def ask(self, remote, *args: str, config: bytes | None = None):
        """agent_call until it gets through (an answer does not clear the count: only moving data does)."""
        while True:
            _check(self.cancel)
            try:
                result = agent_call(remote, *args, config=config)
            except _Transient as e:
                self.failed(str(e))
                continue
            return result


# ---------------------------------------------------------------- straight upload

def upload_file(remote, path, dest: str | None = None, *, sha256: str | None = None, progress=None, cancel=None,
                sleep=time.sleep, home=None, piece: int = PIECE) -> dict:
    """Put one local file at models/sent/<sha12>/<name> (or `dest`, below models/sent) on the instance, resumable.
    progress(bytes there, total) after every piece. Returns {"kind": "file", "dest", "size", "sha256", "skipped"}.
    FilesError: kind "ssh" after MAX_FAILURES drops in a row, "checksum" when the copy there differs twice, "local" when
    the file can't be read or changes, "cancelled"."""
    p = Path(path)
    if not p.is_file():
        raise FilesError(f"{p.name} was not found.", "local")
    st = p.stat()
    size = st.st_size
    sha = sha256 or digest(p, home=home, cancel=cancel)
    dest = check_dest(dest or sent_dest(sha[:12], p.name))
    retry = _Retry(sleep, cancel)

    def say(done: int) -> None:
        if progress:
            progress(done, size)

    for attempt in range(2):  # a checksum mismatch removes the part there: once more from the start
        _, state = retry.ask(remote, "file-state", dest, "--size", str(size))
        if state.get("done"):
            say(size)
            return {"kind": "file", "dest": dest, "size": size, "sha256": sha, "skipped": True}
        have = int(state.get("have") or 0)
        say(have)
        while have < size:
            _check(cancel)
            try:
                with open(p, "rb") as f:
                    f.seek(have)
                    data = f.read(min(piece, size - have))
            except OSError as e:
                raise FilesError(f"Cannot read {p.name}: {e.strerror or e}", "local") from e
            if not data:
                raise FilesError(f"{p.name} got shorter while it was sent.", "local")
            done = remote.run(put_command(dest, have), data=data, retry=False, timeout=PUT_TIMEOUT)
            if done.returncode == 0:
                have += len(data)
                retry.ok()
                say(have)
                continue
            retry.failed("a piece did not get through")  # the part there may hold some of it: ask how much
            _, state = retry.ask(remote, "file-state", dest, "--size", str(size))
            if state.get("done"):
                have = size
                break
            have = int(state.get("have") or 0)
            say(have)
        while size == 0:  # nothing to send, but file-done needs the (empty) part to exist
            done = remote.run(put_command(dest, 0), data=b"", retry=False, timeout=PUT_TIMEOUT)
            if done.returncode == 0:
                break
            retry.failed("the empty part could not be made")
        now = p.stat()
        if now.st_size != size or now.st_mtime_ns != st.st_mtime_ns:
            raise FilesError(f"{p.name} changed while it was sent.", "local")
        code, answer = retry.ask(remote, "file-done", dest, "--size", str(size), "--sha256", sha)
        if code == 0 and answer.get("state") == "done":
            say(size)
            return {"kind": "file", "dest": dest, "size": size, "sha256": sha, "skipped": False}
        if answer.get("state") != "mismatch":
            raise FilesError("The instance did not accept the file.", "remote")
    raise FilesError(f"{p.name} arrived changed twice (checksum differs).", "checksum")


def manifest_bytes(files: list[dict]) -> bytes:
    return json.dumps({"files": [[f["rel"], f["size"], f["sha256"]] for f in files]}).encode("utf-8")


def upload_tree(remote, path, *, progress=None, cancel=None, sleep=time.sleep, home=None, piece: int = PIECE,
                sender=None, large: int = LARGE, target: dict | None = None) -> dict:
    """Put a local folder at models/sent/<tree digest12>/<name>/ (hidden files and folders left out). Files of `large`
    bytes or more go through upload_file one by one; the rest through rulz_remote.send (staged, in batches by folder).
    A drop is asked about again (`file-state --tree` says which files are missing) up to MAX_FAILURES in a row.
    progress(bytes there, total). Returns {"kind": "folder", "dest", "size", "sha256" (tree digest), "files": count,
    "skipped"}."""
    sender = sender or rr.send
    plan = target or target_for(path, home=home, cancel=cancel)
    dest, files, total = plan["dest"], plan["files"], plan["size"]
    config = manifest_bytes(files)
    by_rel = {f["rel"]: f for f in files}
    retry = _Retry(sleep, cancel)
    skipped, clean_pass = True, False
    anomalies = 0

    def say(done: int) -> None:
        if progress:
            progress(min(done, total), total)

    while True:
        _, state = retry.ask(remote, "file-state", dest, "--tree", config=config)
        missing = [by_rel[r] for r in state.get("missing") or [] if r in by_rel]
        there = total - sum(f["size"] for f in missing)
        say(there)
        if not missing:
            break
        skipped = False
        if clean_pass:  # a pass without a drop, and still files missing: something there is not taking them
            anomalies += 1
            if anomalies > 3:
                raise FilesError("The folder keeps coming up incomplete on the instance.", "remote")
        clean_pass = False
        for f in (f for f in missing if f["size"] >= large):
            def sub(done, _t, base=there):
                say(base + done)
            upload_file(remote, f["path"], f"{dest}/{f['rel']}", sha256=f["sha256"], progress=sub, cancel=cancel,
                        sleep=sleep, home=home, piece=piece)
            there += f["size"]
            say(there)
        small = [f for f in missing if f["size"] < large]
        by_dir: dict[str, list[dict]] = {}
        for f in small:
            by_dir.setdefault(posixpath.dirname(f["rel"]), []).append(f)
        failed = False
        for folder, group in sorted(by_dir.items()):
            batch, batch_bytes = [], 0
            for f in group + [None]:
                if f is not None and len(batch) < BATCH_FILES and batch_bytes + f["size"] <= BATCH_BYTES:
                    batch.append(f)
                    batch_bytes += f["size"]
                    continue
                if batch:
                    _check(cancel)
                    sent_before, counted = there, [0]

                    def sent_bytes(n, _files, before=sent_before, counted=counted, room=batch_bytes):
                        counted[0] += n
                        say(before + min(counted[0], room))

                    try:
                        sender(remote, [(b["path"], posixpath.basename(b["rel"])) for b in batch],
                               posixpath.join(dest, folder) if folder else dest, hidden=True, staged=True,
                               progress=sent_bytes)
                        there += batch_bytes
                        say(there)
                        retry.ok()
                    except SystemExit:  # the copy dropped; the next pass asks what arrived
                        retry.failed("a copy of small files did not get through")
                        failed = True
                        break
                batch, batch_bytes = ([f], f["size"]) if f is not None else ([], 0)
            if failed:
                break
        clean_pass = not failed
    _, answer = retry.ask(remote, "file-done", dest, "--tree", "--sha256", plan["sha256"], config=config)
    if answer.get("state") != "done":
        raise FilesError("A file of the folder arrived changed (checksum differs).", "checksum")
    say(total)
    return {"kind": "folder", "dest": dest, "size": total, "sha256": plan["sha256"], "files": len(files),
            "skipped": skipped}


# ---------------------------------------------------------------- Hugging Face

def hf_token() -> str | None:
    """HF_TOKEN, else the token huggingface_hub has saved (None when there is none). Stays on this PC."""
    token = (os.environ.get("HF_TOKEN") or "").strip()
    if token:
        return token
    try:
        from huggingface_hub import get_token

        return get_token() or None
    except Exception:  # noqa: BLE001  (not installed, or no store)
        return None


def hf_api(token: str | None = None):
    """A huggingface_hub.HfApi that has buckets, else a FilesError saying what to install."""
    try:
        from huggingface_hub import HfApi
    except ImportError as e:
        raise FilesError("Buckets need huggingface_hub 1.0 or newer: pip install -U huggingface_hub", "hf") from e
    if not hasattr(HfApi, "create_bucket"):
        raise FilesError("This huggingface_hub has no bucket support: pip install -U 'huggingface_hub>=1.0'", "hf")
    return HfApi(token=token or hf_token())


def hf_login(token: str, *, login=None, api=None) -> dict:
    """Sign in with a token the user typed (huggingface_hub.login, saved by it). The caller clears its field at once;
    nothing here keeps the token. Returns {"ok": True, "user": name}."""
    token = (token or "").strip()
    if not token:
        raise FilesError("Paste a Hugging Face access token.", "hf")
    try:
        if login is None:
            from huggingface_hub import login as hf_login_fn

            login = hf_login_fn
        login(token=token, add_to_git_credential=False)
        who = (api or hf_api(token)).whoami(token=token)
    except FilesError:
        raise
    except Exception as e:  # noqa: BLE001
        raise FilesError("Hugging Face did not accept that token.", "hf") from e
    finally:
        token = ""
    return {"ok": True, "user": (who or {}).get("name") if isinstance(who, dict) else None}


def check_bucket_id(bucket_id: str) -> str:
    bucket_id = (bucket_id or "").strip()
    if not BUCKET_ID.match(bucket_id):
        raise FilesError("A bucket is named owner/name.", "hf")
    return bucket_id


def _not_found(e: Exception) -> bool:
    response = getattr(e, "response", None)
    return getattr(response, "status_code", None) == 404 or "NotFound" in type(e).__name__


def bucket_state(bucket_id: str, *, api=None) -> dict:
    """{"id", "exists", "private", "error"}: whether the bucket is there (error: why it could not be told)."""
    bucket_id = check_bucket_id(bucket_id)
    try:
        info = (api or hf_api()).bucket_info(bucket_id)
    except FilesError:
        raise
    except Exception as e:  # noqa: BLE001
        if _not_found(e):
            return {"id": bucket_id, "exists": False, "private": None, "error": None}
        return {"id": bucket_id, "exists": False, "private": None, "error": scrub(f"{type(e).__name__}: {e}")}
    return {"id": bucket_id, "exists": True, "private": getattr(info, "private", None), "error": None}


def buckets(*, api=None, namespace: str | None = None) -> list[dict]:
    """The account's buckets, [{"id", "private", "size"}]."""
    try:
        rows = list((api or hf_api()).list_buckets(namespace=namespace))
    except FilesError:
        raise
    except Exception as e:  # noqa: BLE001
        raise FilesError(scrub(f"Could not list the buckets ({type(e).__name__})."), "hf") from e
    return [{"id": getattr(r, "id", None) or getattr(r, "bucket_id", None), "private": getattr(r, "private", None),
             "size": getattr(r, "size", None)} for r in rows]


def create_bucket(bucket_id: str, private: bool = True, *, api=None) -> dict:
    bucket_id = check_bucket_id(bucket_id)
    try:
        (api or hf_api()).create_bucket(bucket_id, private=private, exist_ok=True)
    except FilesError:
        raise
    except Exception as e:  # noqa: BLE001
        raise FilesError(scrub(f"Could not create the bucket ({type(e).__name__})."), "hf") from e
    return {"id": bucket_id, "private": private}


def _bucket_sizes(api, bucket_id: str, prefix: str) -> dict[str, int]:
    """{bucket path: size} of what is under prefix there already."""
    try:
        return {getattr(e, "path", ""): getattr(e, "size", None) for e in api.list_bucket_tree(bucket_id, prefix=prefix,
                                                                                               recursive=True)
                if getattr(e, "size", None) is not None}
    except Exception as e:  # noqa: BLE001  (an empty or new bucket may answer 404 for a prefix)
        if _not_found(e):
            return {}
        raise FilesError(scrub(f"Could not look into the bucket ({type(e).__name__})."), "hf") from e


def upload_to_bucket(bucket_id: str, path, *, api=None, progress=None, cancel=None, home=None,
                     target: dict | None = None) -> dict:
    """Upload a file or folder to rulz-trainer/<sha12>/<name> of a bucket, skipping files already there at the same
    size. progress(bytes done, total) after each batch. Returns {"bucket", "kind", "root", "size", "sha256", "files":
    [{"rel", "bucket_path", "size", "sha256"}], "skipped"}: the manager gives these to fetch_from_bucket."""
    bucket_id = check_bucket_id(bucket_id)
    api = api or hf_api()
    plan = target or target_for(path, home=home, cancel=cancel)
    root = f"{BUCKET_PREFIX}/{plan['d12']}/{plan['name']}"
    if plan["kind"] == "file":
        items = [{"rel": "", "path": Path(path), "size": plan["size"], "sha256": plan["sha256"], "bucket_path": root}]
    else:
        items = [{"rel": f["rel"], "path": f["path"], "size": f["size"], "sha256": f["sha256"],
                  "bucket_path": f"{root}/{f['rel']}"} for f in plan["files"]]
    total = plan["size"]
    there = _bucket_sizes(api, bucket_id, f"{BUCKET_PREFIX}/{plan['d12']}/")
    todo = [i for i in items if there.get(i["bucket_path"]) != i["size"]]
    done = total - sum(i["size"] for i in todo)
    if progress:
        progress(done, total)
    batch: list[dict] = []
    for item in todo + [None]:
        if item is not None and len(batch) < BUCKET_BATCH_FILES:
            batch.append(item)
            continue
        if batch:
            _check(cancel)
            try:
                api.batch_bucket_files(bucket_id, add=[(str(b["path"]), b["bucket_path"]) for b in batch])
            except Exception as e:  # noqa: BLE001
                raise FilesError(scrub(f"The bucket upload failed ({type(e).__name__})."), "hf") from e
            done += sum(b["size"] for b in batch)
            if progress:
                progress(done, total)
        batch = [item] if item is not None else []
    return {"bucket": bucket_id, "kind": plan["kind"], "root": root, "size": total, "sha256": plan["sha256"],
            "files": [{k: i[k] for k in ("rel", "bucket_path", "size", "sha256")} for i in items], "skipped": not todo}


def host_allowed(url: str) -> bool:
    parts = urllib.parse.urlsplit(url)
    host = (parts.hostname or "").lower()
    return parts.scheme == "https" and (host == "huggingface.co" or host.endswith(HOST_SUFFIXES))


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def _open_no_redirect(request, timeout=30):
    return urllib.request.build_opener(_NoRedirect).open(request, timeout=timeout)


def signed_address(bucket_id: str, bucket_path: str, token: str | None, *, opener=None, host_ok=host_allowed) -> str:
    """The short-lived address huggingface.co redirects a bucket file's resolve URL to, asked for with the token and
    WITHOUT following the redirect: only the address is returned (and later sent to the instance), never the token."""
    url = f"https://huggingface.co/buckets/{bucket_id}/resolve/{urllib.parse.quote(bucket_path)}"
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"} if token else {})
    try:
        response = (opener or _open_no_redirect)(request, timeout=30)
    except urllib.error.HTTPError as e:
        where = e.headers.get("Location") if e.headers else None
        code = e.code
        e.close()
        if code in (301, 302, 303, 307, 308) and where:
            address = urllib.parse.urljoin(url, where)
            if not host_ok(address):
                raise FilesError("Hugging Face sent the file to a host that is not allowed.", "hf") from None
            return address
        if code in (401, 403):
            raise FilesError("Hugging Face refused the token for that bucket.", "hf") from None
        if code == 404:
            raise FilesError("That file is not in the bucket.", "hf") from None
        raise FilesError(f"Hugging Face answered HTTP {code}.", "hf") from None
    except (urllib.error.URLError, OSError) as e:
        raise FilesError(f"Hugging Face could not be reached ({type(e).__name__}).", "hf") from None
    try:
        response.close()
    except Exception:  # noqa: BLE001
        pass
    raise FilesError("Hugging Face answered with the file itself, not an address to fetch it from.", "hf")


def fetch_from_bucket(remote, bucket_id: str, bucket_path: str, dest: str, size: int, sha256: str, *,
                      token: str | None = None, opener=None, progress=None, cancel=None, sleep=time.sleep,
                      poll: float = 5.0, max_refresh: int = 8, host_ok=host_allowed) -> dict:
    """Have the instance download one bucket file into `dest` (below models/sent): ask huggingface.co for a signed
    address, give it to `file-fetch` on ssh stdin, poll `fetch-state`; an expired address (401/403 there) or an
    interrupted download gets a fresh address and goes on from the done ranges. progress(bytes, size). Returns
    {"dest", "size", "sha256"}. FilesError: kind "hf", "remote" (the download failed there: checksum, host), "ssh",
    "expired" after max_refresh fresh addresses, "cancelled"."""
    bucket_id = check_bucket_id(bucket_id)
    check_dest(dest)
    token = token if token is not None else hf_token()
    retry = _Retry(sleep, cancel)
    for _ in range(max_refresh + 1):
        address = signed_address(bucket_id, bucket_path, token, opener=opener, host_ok=host_ok)
        config = json.dumps({"url": address, "size": size, "sha256": sha256}).encode("utf-8")
        address = ""
        _, state = retry.ask(remote, "file-fetch", dest, config=config)
        config = b""
        while True:
            if state.get("state") == "done":
                if progress:
                    progress(size, size)
                return {"dest": dest, "size": size, "sha256": sha256}
            if state.get("state") in ("expired", "interrupted", "none"):
                break  # a fresh address, and the ranges already there are kept
            if state.get("state") == "failed":
                raise FilesError(scrub(state.get("error") or "The download on the instance failed."), "remote")
            if progress:
                progress(int(state.get("done") or 0), size)
            sleep(poll)
            _check(cancel)
            _, state = retry.ask(remote, "fetch-state", dest)
    raise FilesError("The download kept expiring: try again.", "expired")


def fetch_tree_from_bucket(remote, upload: dict, dest: str, *, token: str | None = None, opener=None, progress=None,
                           cancel=None, sleep=time.sleep, poll: float = 5.0, host_ok=host_allowed) -> dict:
    """The instance downloads a folder uploaded with upload_to_bucket (its result is `upload`) into `dest`: every file
    through fetch_from_bucket, then `file-done --tree`. progress(bytes, total)."""
    check_dest(dest)
    files, total = upload["files"], upload["size"]
    retry = _Retry(sleep, cancel)
    done = 0
    for f in files:
        def sub(d, _s, base=done):
            if progress:
                progress(base + d, total)
        fetch_from_bucket(remote, upload["bucket"], f["bucket_path"], f"{dest}/{f['rel']}", f["size"], f["sha256"],
                          token=token, opener=opener, progress=sub, cancel=cancel, sleep=sleep, poll=poll,
                          host_ok=host_ok)
        done += f["size"]
    manifest = manifest_bytes([{"rel": f["rel"], "size": f["size"], "sha256": f["sha256"]} for f in files])
    _, answer = retry.ask(remote, "file-done", dest, "--tree", "--sha256", upload["sha256"], config=manifest)
    if answer.get("state") != "done":
        raise FilesError("A file of the folder arrived changed (checksum differs).", "checksum")
    if progress:
        progress(total, total)
    return {"dest": dest, "size": total, "sha256": upload["sha256"]}
