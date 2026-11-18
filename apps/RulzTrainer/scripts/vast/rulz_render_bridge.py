#!/usr/bin/env python3
"""The instance's end of "render on this PC": a small file server the PC's helper (rulztrainer/renderhelper.py) talks to
over one long-lived ssh connection. Nothing is installed on the instance: the helper pipes this file to it, as
rulz_remote.py does with rulz_agent.py, so the instance's checkout may be of any age. Standard library only (it runs
on the instance's Python, and the tests run it on this PC).

Why a bridge and not scp or sftp: a round is a handful of small renames and files, and the one rule that matters is
that a rename either wins or loses atomically against the trainer's own rename (docs/remote-render-contract.md,
section 2). Over sftp every call would be a separate round trip with its own failure modes; here a call is a frame.

Start (python reads a script to EOF, so a bootstrap reads exactly one length-prefixed script, then stdin carries
frames; renderhelper.BOOTSTRAP is the one the helper sends):

    python3 -c 'import sys,struct;s=sys.stdin.buffer;exec(compile(s.read(struct.unpack(">I",s.read(4))[0]),"bridge","exec"))'

Frame, both directions: b"RZRB", 4-byte big-endian header length, header JSON (UTF-8), 8-byte big-endian body length,
body. A bad magic ends the bridge, and so does stdin's end. Every request has an "op" and gets one reply
{ok: true, ...} or {ok: false, error}; get's body is the file, put's body is the file to write.

    hello {protocol}                          -> {protocol, output_root, time}
    runs                                      -> {runs: [{job, run, has_prepare, ready, not_ready, rounds, claimed}]}
    get {path}                                -> body = the file
    put {path} + body                         -> writes it (whole file or nothing)
    write_json {path, data}                   -> writes it
    rename {src, dst}                         -> {renamed: bool}: False when src is gone or dst exists
    list {path}                               -> {entries: [names]}
    heartbeat {job, run_token, host, gpu}     -> writes helper.json with THIS machine's clock

A path is "<job>/remote_render/...", relative to the output folder, and must resolve (symlinks followed) to a place
inside that job's remote_render folder. The bridge never writes anywhere else: the trainer alone moves a finished
round's pictures to their real places."""
import ctypes
import errno
import json
import os
from pathlib import Path, PurePosixPath
import struct
import sys
import time

MAGIC = b"RZRB"
PROTOCOL = 1
ROOT = Path("output").resolve()  # the WebUI's output folder: rulz_agent.py's OUTPUT, relative to the install folder
JOB_FILE = Path(".webui_job.json")  # the WebUI's record of the job it started (rulz_agent.py's JOB_FILE)
MAX_HEADER = 16 * 1024 * 1024
MAX_BODY = 2 * 1024 ** 3
FOLDER = "remote_render"
ROUND_FOLDERS = ("rounds", "claimed", "done", "failed", "abandoned")


def read_exact(stream, size):
    chunks = []
    while size:
        chunk = stream.read(size)
        if not chunk:
            raise EOFError
        chunks.append(chunk)
        size -= len(chunk)
    return b"".join(chunks)


def read_json(p):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):  # missing, or torn mid-write
        return None


def pid_alive(pid, created=None):
    """The job's process runs. psutil when the instance's venv has it (as rulz_agent.live_process does: a pid only
    counts with its create time, pids get reused), else /proc on Linux. Where neither can tell (this PC in a test),
    a pid that is a number is taken as alive: os.kill(pid, 0) on Windows would press Ctrl+C."""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    try:
        import psutil
    except ImportError:
        psutil = None
    if psutil is not None:
        try:
            proc = psutil.Process(pid)
            if created is not None and abs(proc.create_time() - float(created)) > 1.0:
                return False
            return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
        except psutil.Error:
            return False
    if os.path.isdir("/proc/self"):
        try:
            with open("/proc/%d/stat" % pid, "rb") as f:
                state = f.read().rsplit(b")", 1)[1].split()[0]
        except (OSError, IndexError):
            return False
        return state != b"Z"
    return os.name == "nt"


def running_job():
    """The folder of the job the WebUI started while its process runs, else None (rulz_agent.running_dir). The only
    job a helper should render for: a finished run's folder keeps its remote_render files."""
    data = read_json(JOB_FILE) or {}
    folder = data.get("job_dir")
    if not folder or data.get("pid") is None or not pid_alive(data.get("pid"), data.get("create_time")):
        return None
    folder = Path(folder)
    return folder.resolve() if folder.is_dir() else None


def job_folder(job):
    """A job's folder: the running job's own (it may have an output folder of its own), else output/<job>."""
    if not isinstance(job, str) or not job or job in (".", "..") or any(c in job for c in "/\\\x00"):
        raise ValueError("not a job name")
    running = running_job()
    if running is not None and running.name == job:
        return running
    folder = ROOT / job
    real = folder.resolve()
    if real.parent != ROOT.resolve():
        raise ValueError("job folder is outside the output folder")
    return real


def remote_base(job, must_exist=True):
    """<job folder>/remote_render, refused when it is a link (a symlink, or a junction on Windows: it could point
    anywhere). The job folder is already resolved, so a base that resolves to somewhere else is a link."""
    base = job_folder(job) / FOLDER
    if base.resolve() != base:
        raise ValueError("remote_render is a link")
    if must_exist and not base.is_dir():
        raise ValueError("no remote_render folder")
    return base


def path(value, root_ok=False):
    """A request's path -> the real file path, or ValueError. "<job>/remote_render/<more>": no empty, "." or ".."
    parts, no backslashes, and the result with symlinks followed must stay inside the folder (a link inside it that
    points out is as refused as a ".." is)."""
    if not isinstance(value, str) or not value or value.startswith("/"):
        raise ValueError("path must be relative: <job>/remote_render/...")
    parts = value.split("/")
    if any(p in ("", ".", "..") or "\\" in p or "\x00" in p for p in parts):
        raise ValueError("bad path")
    if len(parts) < 2 or parts[1] != FOLDER or (len(parts) == 2 and not root_ok):
        raise ValueError("path must be inside a job's remote_render folder")
    base = remote_base(parts[0], must_exist=False)
    real_base = base.resolve()
    result = base.joinpath(*parts[2:]).resolve()
    if result != real_base and real_base not in result.parents:
        raise ValueError("path escapes remote_render")
    return result


def anchor_of(value):
    """The folder a write may not create: a round's own folder (<state>/<round>), so that a late write for a round the
    trainer has already taken back (renamed away) fails instead of bringing the round folder back to life, from which
    the helper's rename to done/ would then win with half a round. Anything else (prepare/, helper.json): the
    remote_render folder itself."""
    parts = value.split("/")
    if len(parts) > 2 and parts[2] in ROUND_FOLDERS:
        if len(parts) < 5:
            raise ValueError("a file goes inside a round folder")
        return "/".join(parts[:4])
    return "/".join(parts[:2])


def write_file(p, data, anchor):
    """Whole file or nothing: a reader never sees half of it. Folders are made below `anchor` only, one at a time
    (an anchor that has gone is an error, not something to make again)."""
    anchor = path(anchor, root_ok=True)
    if not anchor.is_dir():
        raise FileNotFoundError("the round is gone: %s" % anchor.name)
    below = p.parent.relative_to(anchor).parts
    folder = anchor
    for name in below:
        folder = folder / name
        try:
            os.mkdir(folder)
        except FileExistsError:
            pass
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, p)


def rename(src, dst):
    """Rename that never replaces: whichever of two competing renames (the trainer's and the helper's) gets there
    first wins, and the other is told False. renameat2(RENAME_NOREPLACE) on Linux, where a plain rename of a folder
    would replace an empty one."""
    if not os.path.lexists(src):
        return False
    dst.parent.mkdir(exist_ok=True)  # the state folder (claimed/, done/...): the remote_render folder is there, src is in it
    if os.name != "nt":
        try:
            fn = ctypes.CDLL(None, use_errno=True).renameat2
        except (AttributeError, OSError):  # an old libc
            fn = None
        if fn is not None:
            fn.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
            if fn(-100, os.fsencode(src), -100, os.fsencode(dst), 1) == 0:
                return True
            error = ctypes.get_errno()
            if error in (errno.ENOENT, errno.EEXIST, errno.ENOTEMPTY):
                return False
            if error not in (errno.ENOSYS, errno.EINVAL):  # (those: the file system has no such flag)
                raise OSError(error, os.strerror(error))
    if os.path.lexists(dst):
        return False
    try:
        os.rename(src, dst)  # Windows refuses an existing dst by itself
        return True
    except (FileNotFoundError, FileExistsError):
        return False


def names(folder):
    """The folders in a round folder (a "." name is a trainer's half-written one)."""
    try:
        return sorted(p.name for p in folder.iterdir() if p.is_dir() and not p.name.startswith("."))
    except OSError:
        return []


def run_entry(folder):
    base = folder / FOLDER
    if base.resolve() != base:
        return None
    run = read_json(base / "run.json")
    if not run:
        return None
    return {"job": folder.name, "run": run, "has_prepare": (base / "prepare" / "prepare.json").is_file(),
            "ready": read_json(base / "prepare" / "ready.json"), "not_ready": read_json(base / "prepare" / "not_ready.json"),
            "rounds": names(base / "rounds"), "claimed": names(base / "claimed")}


def dispatch(h, body):
    op = h.get("op")
    if op == "hello":
        if h.get("protocol") != PROTOCOL:
            raise ValueError("protocol mismatch: the bridge speaks %d" % PROTOCOL)
        return {"protocol": PROTOCOL, "output_root": str(ROOT), "time": time.time()}, b""
    if op == "runs":
        folder = running_job()  # only a job that is running: a dead run's folder is nobody's to render for
        entry = run_entry(folder) if folder is not None else None
        return {"runs": [entry] if entry else []}, b""
    if op == "heartbeat":
        base = remote_base(h["job"])
        write_file(base / "helper.json", json.dumps({"run_token": h["run_token"], "host": h["host"], "gpu": h["gpu"],
                                                     "time": time.time()}).encode("utf-8"), "%s/%s" % (h["job"], FOLDER))
        return {}, b""
    if op == "rename":
        src, dst = h.get("src"), h.get("dst")
        if not isinstance(src, str) or not isinstance(dst, str) or src.split("/")[0] != dst.split("/")[0]:
            raise ValueError("rename stays inside one job")
        return {"renamed": rename(path(src), path(dst))}, b""
    p = path(h.get("path"), root_ok=op == "list")
    anchor = anchor_of(h["path"]) if op in ("put", "write_json") else None
    if op == "get":
        return {}, p.read_bytes()
    if op == "put":
        write_file(p, body, anchor)
        return {}, b""
    if op == "write_json":
        write_file(p, json.dumps(h["data"]).encode("utf-8"), anchor)
        return {}, b""
    if op == "list":
        return {"entries": sorted(x.name for x in p.iterdir())}, b""
    raise ValueError("unknown operation: %s" % op)


def main():
    inp, out = sys.stdin.buffer, sys.stdout.buffer
    sys.stdout = sys.stderr  # a stray print must not land in the frames
    while True:
        try:
            if read_exact(inp, 4) != MAGIC:
                return
            size = struct.unpack(">I", read_exact(inp, 4))[0]
            if size > MAX_HEADER:
                return
            header = json.loads(read_exact(inp, size).decode("utf-8"))
            size = struct.unpack(">Q", read_exact(inp, 8))[0]
            if size > MAX_BODY:
                return
            body = read_exact(inp, size)
        except (EOFError, ValueError):
            return
        try:
            reply, data = dispatch(header, body)
            reply["ok"] = True
        except Exception as e:  # noqa: BLE001 - the helper is told, and the bridge stays up
            reply, data = {"ok": False, "error": "%s: %s" % (type(e).__name__, e)}, b""
        encoded = json.dumps(reply).encode("utf-8")
        out.write(MAGIC + struct.pack(">I", len(encoded)) + encoded + struct.pack(">Q", len(data)) + data)
        out.flush()


if __name__ == "__main__":
    main()
