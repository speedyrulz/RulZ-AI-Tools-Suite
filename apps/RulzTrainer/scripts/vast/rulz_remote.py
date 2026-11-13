#!/usr/bin/env python3
"""Reach a RulzTrainer on a vast.ai instance from this computer: read its runs, logs and probes, fetch their files,
start and stop runs. For a person at a terminal and for an AI assistant with a shell alike.

    python scripts/vast/rulz_remote.py connect              # find the running instance, test ssh, remember it
    python scripts/vast/rulz_remote.py status               # GPU, disk, the running job, the queue
    python scripts/vast/rulz_remote.py jobs                 # every run on the instance
    python scripts/vast/rulz_remote.py log my_lora -n 200   # also: --console, --grep REGEX
    python scripts/vast/rulz_remote.py probes my_lora       # the likeness probes as a table of scores
    python scripts/vast/rulz_remote.py pull my_lora --what logs --what probes
    python scripts/vast/rulz_remote.py start --job test1 --from my_lora --set steps=300 --dry-run
    python scripts/vast/rulz_remote.py wait test1           # blocks until the run ends
    python scripts/vast/rulz_remote.py ssh -- nvidia-smi    # any command in the install folder
    python scripts/vast/rulz_remote.py update --check       # the newest RulzTrainer; update installs it, restarts the WebUI

Needs: ssh on PATH, and the vastai CLI with an API key to find the instance (`vastai set api-key <key>`). Your account's
ssh key reaches every instance rented after it was added; connect --attach-key adds it to one rented before. The
standard library is enough here: the instance's side is rulz_agent.py, sent over ssh on every call, so nothing is
installed there and the instance's checkout may be of any age.

Reads change nothing on the instance. start queues a run in the WebUI's queue, so it shows on the Monitor, and stop
is the WebUI's Stop; update replaces the install's code with the newest release's and restarts the WebUI. This tool
never rents, stops or destroys an instance.

Exit status: 0 fine, 1 refused or the run failed, 2 wait timed out, 3 the instance can't be reached."""
from __future__ import annotations

import argparse
import base64
import errno
import hashlib
import io
import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
APP = HERE.parents[1]
STATE = Path(os.environ.get("RULZ_VAST_HOME") or Path.home() / ".rulz_vast")
CONNECTION = STATE / "connection.json"
PULLS = STATE / "runs"  # outside the checkout: pulled runs outlive a branch or a worktree
REMOTE_DIR = "/workspace/RulzTrainer"
PYTHON = '"$(test -x /venv/main/bin/python && echo /venv/main/bin/python || echo python3)"'
AGENT_COMMANDS = {"status", "jobs", "job", "log", "probes", "config", "start", "stop", "queue", "snapshot",
                  "record", "update", "webui"}  # rulz_agent.py's
CONFIG_SLOT = b'SENT_CONFIG = ""'  # the line of rulz_agent.py that takes a local config
SSH_FAILED = 255  # ssh's own exit status: it could not connect, or the connection dropped
SYNC_PATHS = ("rulztrainer", "webui.py", "scripts", "requirements.txt")  # what a run needs of the code
VASTAI_TIMEOUT = 120  # seconds for one vastai CLI call: a hung API request must not hold up every look after it
UPLOAD_MARK = ".rulz_upload"  # in a dataset sent by rulz_fleet.py: dataset_digest() of what was sent


class Unreachable(Exception):
    """No instance, or no way in."""


# ---------------------------------------------------------------- finding the instance


def vastai(*args: str) -> str:
    exe = shutil.which("vastai")
    if exe is None:
        raise Unreachable("The vastai CLI is not installed: pip install vastai && vastai set api-key <your key>")
    try:
        done = subprocess.run([exe, *args], capture_output=True, text=True, encoding="utf-8", errors="replace",
                              timeout=VASTAI_TIMEOUT)
    except subprocess.TimeoutExpired as e:
        raise Unreachable(f"vastai {args[0]} did not answer in {VASTAI_TIMEOUT} s") from e
    if done.returncode != 0:
        raise Unreachable(f"vastai {args[0]} failed: {(done.stderr or done.stdout).strip()[:300]}")
    return done.stdout


def list_instances() -> list[dict]:
    data = json.loads(vastai("show", "instances", "--raw"))
    return data.get("instances", []) if isinstance(data, dict) else data


def is_rulztrainer(inst: dict) -> bool:
    """Rented from a RulzTrainer template (its portal entry or repo variables) or from the image the Dockerfile builds."""
    marks = json.dumps(inst.get("extra_env") or "") + str(inst.get("image_uuid") or "")
    return "rulztrainer" in marks.lower() or "RULZ_" in marks


def pick_instance(instances: list[dict], wanted: int | None = None) -> dict:
    """The instance to use: the one named, else the only running RulzTrainer instance, else the only running one."""
    if wanted is not None:
        for inst in instances:
            if inst.get("id") == wanted:
                if inst.get("actual_status") != "running":
                    raise Unreachable(f"Instance {wanted} is {inst.get('actual_status')!r}, not running.")
                return inst
        raise Unreachable(f"No instance {wanted} on this account.")
    running = [i for i in instances if i.get("actual_status") == "running"]
    if not running:
        states = ", ".join(f"{i.get('id')}: {i.get('actual_status')}" for i in instances) or "none rented"
        raise Unreachable(f"No running instance ({states}).")
    pool = [i for i in running if is_rulztrainer(i)] or running
    if len(pool) > 1:
        listing = ", ".join(f"{i.get('id')} ({i.get('gpu_name')})" for i in pool)
        raise Unreachable(f"Several instances are running, pick one with --instance: {listing}")
    return pool[0]


def routes(inst: dict) -> list[dict]:
    """The ways in, best first: the instance's direct port, then vast's ssh proxy (which is not always up)."""
    found = []
    ports = (inst.get("ports") or {}).get("22/tcp") or []
    if inst.get("public_ipaddr") and ports:
        found.append({"route": "direct", "host": inst["public_ipaddr"].strip(), "port": int(ports[0]["HostPort"])})
    if inst.get("ssh_host") and inst.get("ssh_port"):
        found.append({"route": "proxy", "host": inst["ssh_host"], "port": int(inst["ssh_port"])})
    return found


# ---------------------------------------------------------------- ssh


def ssh_command(conn: dict, key: str | None = None, batch: bool = True) -> list[str]:
    """ssh with this tool's options. Each rental gets its own known-hosts file: vast hands the same host and port to
    new rentals with new host keys, and that must not read as an attack on the next one (or be ignored on this one)."""
    STATE.mkdir(parents=True, exist_ok=True)
    known = (STATE / f"known_hosts_{conn['id']}").as_posix()
    cmd = ["ssh", "-p", str(conn["port"]), "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=30",
           "-o", "ServerAliveCountMax=4", "-o", "LogLevel=ERROR", "-o", "StrictHostKeyChecking=accept-new",
           "-o", f"UserKnownHostsFile={known}"]
    if batch:
        cmd += ["-o", "BatchMode=yes"]
    if key or os.environ.get("RULZ_VAST_KEY"):
        cmd += ["-i", key or os.environ["RULZ_VAST_KEY"]]
    return cmd + [f"root@{conn['host']}"]


def reachable(conn: dict, key: str | None) -> bool:
    try:
        done = subprocess.run(ssh_command(conn, key) + ["echo rulz-ok"], capture_output=True, text=True, timeout=30,
                              stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return False
    return done.returncode == 0 and "rulz-ok" in done.stdout


def find_connection(wanted: int | None, key: str | None, remember: bool = True) -> dict:
    """Ask vast for the instance, try its routes, remember the one that works (remember=False only returns it:
    rulz_fleet.py keeps a route per instance of its own)."""
    inst = pick_instance(list_instances(), wanted)
    tried = routes(inst)
    if not tried:
        raise Unreachable(f"Instance {inst['id']} reports no ssh address yet: it may still be starting.")
    for route in tried:
        conn = {"id": inst["id"], "gpu": inst.get("gpu_name"), "dollars_per_hour": inst.get("dph_total"), **route}
        if reachable(conn, key):
            if remember:
                STATE.mkdir(parents=True, exist_ok=True)
                CONNECTION.write_text(json.dumps(conn, indent=1), encoding="utf-8")
            return conn
    raise Unreachable(f"Instance {inst['id']} refused ssh on {', '.join(r['route'] for r in tried)}. If it was rented "
                      "before your ssh key was added, run connect --attach-key; a new instance needs a minute first.")


def saved_connection(wanted: int | None) -> dict | None:
    try:
        conn = json.loads(CONNECTION.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return conn if wanted in (None, conn.get("id")) else None


def agent_program(config: bytes | None = None) -> bytes:
    """rulz_agent.py as it is piped to the instance, a local config written into it. A config can't go on the command
    line: Git for Windows' ssh cuts a command at about 8190 characters, with no error and exit status 0."""
    source = (HERE / "rulz_agent.py").read_bytes()
    if config is None:
        return source
    return source.replace(CONFIG_SLOT, b'SENT_CONFIG = "' + base64.b64encode(config) + b'"', 1)


class Remote:
    """One instance over ssh. The route is remembered between calls, so a call costs one ssh connection and no vast
    API request; when the instance has gone or moved, the first failure looks it up again."""

    def __init__(self, instance: int | None, key: str | None, folder: str, conn: dict | None = None,
                 remember: bool = True):
        self.instance, self.key, self.folder, self.remember = instance, key, folder, remember
        self.conn = conn or (saved_connection(instance) if remember else None) \
            or find_connection(instance, key, remember)

    def command(self, shell: str, in_folder: bool = True) -> list[str]:
        return ssh_command(self.conn, self.key) + [f"cd {shlex.quote(self.folder)} && {shell}" if in_folder else shell]

    def run(self, shell: str, data: bytes | None = None, in_folder: bool = True,
            retry: bool = True) -> subprocess.CompletedProcess:
        """The command over ssh. When ssh fails, the route is looked up again and the command runs once more, unless
        retry=False: a connection that dropped part way may have left the command running there (an update)."""
        done = subprocess.run(self.command(shell, in_folder), input=data, capture_output=True,
                              stdin=subprocess.DEVNULL if data is None else None)
        if done.returncode == SSH_FAILED and retry:
            self.conn = find_connection(self.instance, self.key, self.remember)
            done = subprocess.run(self.command(shell, in_folder), input=data, capture_output=True,
                                  stdin=subprocess.DEVNULL if data is None else None)
        return done

    def agent(self, *args: str, config: bytes | None = None, retry: bool = True) -> subprocess.CompletedProcess:
        """rulz_agent.py on the instance: piped in, so it is always this one."""
        return self.run(f"{PYTHON} - {shlex.join(args)}", agent_program(config), retry=retry)

    def agent_json(self, *args: str):
        done = self.agent(*args)
        if done.returncode != 0:
            raise SystemExit(done.stderr.decode("utf-8", "replace").strip() or f"{args[0]} failed")
        return json.loads(done.stdout)


def forward(done: subprocess.CompletedProcess) -> int:
    """Print a remote command's output as it is (bytes: the console's encoding must not get in the way)."""
    sys.stdout.buffer.write(done.stdout)
    sys.stdout.flush()
    sys.stderr.buffer.write(done.stderr)
    sys.stderr.flush()
    return done.returncode


# ---------------------------------------------------------------- commands


def cmd_connect(args) -> int:
    """Find the instance and test it; --wait waits for one that is still starting or installing the trainer."""
    deadline = time.time() + args.wait
    if args.attach_key:
        return attach_key(args)
    while True:
        try:
            conn = find_connection(args.instance, args.key)
            installed = subprocess.run(ssh_command(conn, args.key) + [f"test -f {shlex.quote(args.dir)}/webui.py"],
                                       capture_output=True, stdin=subprocess.DEVNULL).returncode == 0
            if installed:
                break
            problem = f"Instance {conn['id']} is up but {args.dir} has no RulzTrainer yet (first boot takes 3-5 minutes)."
        except Unreachable as e:
            problem = str(e)
        if time.time() >= deadline:
            print(problem, file=sys.stderr)
            return 3
        print(f"[waiting] {problem}", file=sys.stderr, flush=True)
        time.sleep(20)
    print(json.dumps({"connected": {k: conn.get(k) for k in ("id", "gpu", "dollars_per_hour", "route", "host", "port")},
                      "trainer": Remote(args.instance, args.key, args.dir).agent_json("status")}, indent=1))
    return 0


def local_public_key(key: str | None) -> str:
    candidates = [Path(key + ".pub")] if key else [Path.home() / ".ssh" / n
                                                   for n in ("id_ed25519.pub", "id_ecdsa.pub", "id_rsa.pub")]
    for path in candidates:
        if path.is_file():
            return path.read_text(encoding="utf-8").strip()
    raise Unreachable("No public ssh key found: name one with --key, or make one with ssh-keygen.")


def attach_key(args) -> int:
    inst = pick_instance(list_instances(), args.instance)
    print(vastai("attach", "ssh", str(inst["id"]), local_public_key(args.key)).strip())
    print("Attached. It reaches the instance within a few seconds: run connect again.")
    return 0


def pull_plan(files: list[dict], dest: Path) -> list[dict]:
    """The listed files that are not here yet at their size and time: a log that grew comes again, a probe image does
    not, and neither does a LoRA of the same size unless the run was made again (an --overwrite run of the same name
    writes files of the same size). tar keeps a file's time, so a fetched copy has the instance's; 2 s of slack for a
    FAT drive. A listing without "mtime" (an older agent) leaves the size alone to decide."""
    def here(entry: dict) -> bool:
        path = dest / entry["path"]
        if not path.is_file():
            return False
        st = path.stat()
        if st.st_size != entry["size"]:
            return False
        return entry.get("mtime") is None or abs(st.st_mtime - entry["mtime"]) <= 2

    return [f for f in files if not here(f)]


def extract(stream, dest: Path) -> None:
    """Unpack a tar stream from the instance. The data filter refuses paths that would land outside dest."""
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=stream, mode="r|") as tar:
        tar.extractall(dest, filter="data")


def fetch(remote: Remote, base: str, files: list[str], dest: Path) -> None:
    proc = subprocess.Popen(remote.command(f"tar -cf - -C {shlex.quote(base)} -T -", in_folder=False),
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def feed():  # from a thread: a long list would fill the pipe while tar waits for us to take its output
        proc.stdin.write(("\n".join(files) + "\n").encode("utf-8"))
        proc.stdin.close()

    threading.Thread(target=feed, daemon=True).start()
    extract(proc.stdout, dest)
    if proc.wait() > 1:  # 1: a file changed while it was read, which a log of a running job does
        raise SystemExit(f"tar on the instance ended with {proc.returncode}: the copy may be incomplete")


def cmd_pull(args) -> int:
    """Copy a job's files here. Files already here at the same size are skipped, so this can repeat during a run."""
    dest = Path(args.dest) if args.dest else PULLS / str(args.remote.conn["id"])
    extra = [a for what in args.what or [] for a in ("--what", what)]
    if args.step is not None:
        extra += ["--step", str(args.step)]
    listing = args.remote.agent_json("files", *([args.name] if args.name else []), *extra)
    todo = pull_plan(listing["files"], dest)
    size = sum(f["size"] for f in todo)
    if todo:
        fetch(args.remote, listing["base"], [f["path"] for f in todo], dest)
    print(json.dumps({"folder": str(dest), "fetched": len(todo), "mb": round(size / 1e6, 1),
                      "already_here": len(listing["files"]) - len(todo),
                      "files": [f["path"] for f in todo][:args.show]}, indent=1))
    return 0


def tar_info_for_instance(info: tarfile.TarInfo) -> tarfile.TarInfo:
    """Owner and mode for Linux: on Windows a file has neither, and scripts must stay runnable."""
    info.uid, info.gid, info.uname, info.gname = 0, 0, "root", "root"
    runnable = info.isdir() or info.name.endswith(".sh")
    info.mode = 0o755 if runnable else 0o644
    return info


def without_hidden(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
    """tar_info_for_instance, leaving out anything below the sent item whose name starts with "." (a hidden folder
    is skipped whole): a dataset's .rulz_cache of latents and text features, which can be gigabytes, and a NAS's
    .thumbnails. The instance encodes its own cache."""
    if any(part.startswith(".") for part in info.name.split("/")[1:]):
        return None
    return tar_info_for_instance(info)


def dataset_digest(path: Path) -> str:
    """What a dataset folder holds, as send(hidden=False) sends it: each file below it by path, size and time (hidden
    files and folders left out, as send leaves them out). rulz_fleet.py keeps it with the copy it sends, so a folder
    changed here since (captions fixed, pictures added) is not taken for the copy already on an instance."""
    root = Path(path)
    rows = []
    if root.is_file():
        st = root.stat()
        rows.append(f"{root.name}\t{st.st_size}\t{int(st.st_mtime)}")
    for folder, dirs, names in os.walk(root):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for name in sorted(names):
            if name.startswith("."):
                continue
            file = Path(folder) / name
            try:
                st = file.stat()
            except OSError:
                continue
            rows.append(f"{file.relative_to(root).as_posix()}\t{st.st_size}\t{int(st.st_mtime)}")
    return hashlib.sha1("\n".join(rows).encode("utf-8")).hexdigest()[:16]


def _pipe_gone(e: OSError) -> bool:
    """A write into ssh that failed because ssh (or the far end) went away, not because a file here could not be read:
    Windows reports a pipe with no reader as EINVAL, POSIX as EPIPE."""
    return isinstance(e, (BrokenPipeError, ConnectionResetError)) or (
        getattr(e, "filename", None) is None and e.errno in (errno.EPIPE, errno.EINVAL))


def send(remote: Remote, items: list[tuple[Path, str]], target: str, hidden: bool = True, *, staged: bool = False,
         marks: dict[str, str] | None = None) -> None:
    """Stream local paths, each with its name in the archive, into a folder on the instance; hidden=False leaves
    out hidden files and folders below each item. staged=True unpacks into a hidden folder beside them first and
    moves each item into place only once all of it is there, so a dropped connection leaves no half copy under the
    item's name for the next look to take as sent (stage folders left by a dropped copy go after 6 hours); marks
    puts a small UPLOAD_MARK file holding the given text into each named item.

    A file here that can't be read (moved, locked, shrunk while it was read) stops the copy with SystemExit: it used
    to be taken for the far end going away, which left ssh waiting for the rest of the archive."""
    quoted = shlex.quote(target)
    if staged:
        incoming = f"{target}/.rulz_incoming"
        stage = f"{incoming}/{uuid.uuid4().hex[:12]}"
        moves = " && ".join(f"rm -rf {shlex.quote(f'{target}/{name}')} && "
                            f"mv {shlex.quote(f'{stage}/{name}')} {shlex.quote(f'{target}/{name}')}"
                            for _, name in items) or "true"
        shell = (f"{{ find {shlex.quote(incoming)} -mindepth 1 -maxdepth 1 -mmin +360 -exec rm -rf {{}} + "
                 f"2>/dev/null || true; }} && mkdir -p {shlex.quote(stage)} && "
                 f"{{ tar -xf - -C {shlex.quote(stage)} && {moves}; }}; rc=$?; rm -rf {shlex.quote(stage)}; exit $rc")
    else:
        shell = f"mkdir -p {quoted} && tar -xf - -C {quoted}"
    proc = subprocess.Popen(remote.command(shell), stdin=subprocess.PIPE)
    keep = tar_info_for_instance if hidden else without_hidden
    try:
        with tarfile.open(fileobj=proc.stdin, mode="w|") as tar:
            for path, name in items:
                tar.add(path, arcname=name, filter=keep)
                mark = (marks or {}).get(name)
                if mark is not None:
                    data = mark.encode("utf-8")
                    info = tar_info_for_instance(tarfile.TarInfo(f"{name}/{UPLOAD_MARK}"))
                    info.size, info.mtime = len(data), int(time.time())
                    tar.addfile(info, io.BytesIO(data))
    except OSError as e:
        if not _pipe_gone(e):  # a file here, not the far end: stop rather than send part of it
            proc.kill()
            proc.wait()
            raise SystemExit(f"Could not read {getattr(e, 'filename', None) or 'a file'} here: {e}") from e
    finally:
        try:
            proc.stdin.close()
        except OSError:
            pass
    if proc.wait() != 0:
        raise SystemExit(f"tar on the instance ended with {proc.returncode}: the copy may be incomplete")


def cmd_push(args) -> int:
    """Copy a local file or folder to a folder on the instance, relative to the install: a dataset, say. Hidden files
    and folders inside it (its .rulz_cache) stay here unless --with-cache."""
    source = Path(args.source)
    if not source.exists():
        print(f"{source} does not exist", file=sys.stderr)
        return 1
    send(args.remote, [(source, source.resolve().name)], args.target, hidden=args.with_cache)
    print(f"{source} -> {args.target}/{source.resolve().name}")
    return 0


def code_files(app: Path = APP) -> list[str]:
    """The code a run needs, as git sees it here: uncommitted edits and new files included, ignored ones not."""
    out = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", *SYNC_PATHS],
                         cwd=app, capture_output=True, check=True).stdout
    return sorted(n for n in out.decode("utf-8").split("\0") if n and (app / n).is_file())


def cmd_sync(args) -> int:
    """Copy this checkout's code to the instance, to test a change before it is released. It overwrites the
    instance's files (files removed here stay there) and leaves its git checkout showing them as changed."""
    if args.remote.agent_json("status").get("job_running") and not args.force:
        print("A job is running: it would import a mix of old and new modules. Wait for it, or use --force.",
              file=sys.stderr)
        return 1
    files = code_files()
    send(args.remote, [(APP / n, n) for n in files], args.dir)
    print(f"Copied {len(files)} files to {args.dir}")
    if args.restart_webui:  # a running job is detached from the WebUI and survives this
        return forward(args.remote.run("supervisorctl restart rulztrainer", in_folder=False))
    print("The next run uses this code. The WebUI itself keeps its old code until --restart-webui.")
    return 0


def cmd_wait(args) -> int:
    """Block until a job ends, printing its progress. Exit 0 when it finished, 1 when it failed, 2 at --timeout, 3 when
    the instance stays out of reach for --max-down seconds. Survives dropped connections."""
    deadline = time.time() + args.timeout if args.timeout else None
    down_since, last = None, None
    while True:
        try:
            done = args.remote.agent("poll", *([args.name] if args.name else []))
        except Unreachable:
            done = None
        if done is None or done.returncode == SSH_FAILED:
            down_since = down_since or time.time()
            if time.time() - down_since > args.max_down:
                print(f"[unreachable] for {args.max_down} s", file=sys.stderr)
                return 3
        elif done.returncode != 0:
            return forward(done)
        else:
            down_since = None
            st = json.loads(done.stdout)
            line = f"{st['state']} step {st.get('step')}/{st.get('total')} {st.get('stage') or ''}".rstrip()
            if line != last:
                print(f"{time.strftime('%H:%M:%S')} {st['job']}: {line} loss {st.get('avg_loss')} eta {st.get('eta')}",
                      flush=True)
                last = line
            if st["state"] == "done":
                return 0
            if st["state"] in ("error", "ended"):
                print(json.dumps(st, indent=1), file=sys.stderr)
                return 1
            if args.step and (st.get("step") or 0) >= args.step:
                return 0
        if deadline is not None and time.time() > deadline:
            print("[timeout]", file=sys.stderr)
            return 2
        time.sleep(args.every)


def cmd_ssh(args) -> int:
    """A command in the install folder, or a terminal when none is given."""
    words = args.words[1:] if args.words[:1] == ["--"] else args.words
    if words:
        return forward(args.remote.run(" ".join(words)))
    return subprocess.call(ssh_command(args.remote.conn, args.key, batch=False))


def agent_args(argv: list[str]) -> tuple[list[str], bytes | None]:
    """A command's arguments for the agent, and the contents of a local config for start (see agent_program)."""
    out, config, items = [], None, iter(argv)
    for item in items:
        if item == "--config":
            config = Path(next(items)).read_bytes()
        else:
            out.append(item)
    return out, config


def global_options() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--instance", type=int, default=int(os.environ.get("RULZ_VAST_INSTANCE") or 0) or None,
                    help="instance id; default: the running RulzTrainer instance")
    ap.add_argument("--key", help="ssh private key; default: ssh's own choice, or RULZ_VAST_KEY")
    ap.add_argument("--dir", default=os.environ.get("RULZ_VAST_DIR") or REMOTE_DIR, help="the install on the instance")
    return ap


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(parents=[global_options()], description=__doc__.split("\n\n")[0],
                                 epilog=f"Run on the instance: {', '.join(sorted(AGENT_COMMANDS))}. "
                                        "<command> --help there lists their options.")
    sub = ap.add_subparsers(dest="command", required=True)

    def command(name, func, description=None):
        p = sub.add_parser(name, help=(description or func.__doc__ or "").split("\n")[0])
        p.set_defaults(func=func)
        return p

    p = command("connect", cmd_connect)
    p.add_argument("--wait", type=int, default=0, metavar="SECONDS", help="keep trying this long")
    p.add_argument("--attach-key", action="store_true", help="add your ssh key to the instance (rented before it)")
    p = command("pull", cmd_pull)
    p.add_argument("name", nargs="?", help="job name; default: the running or last job")
    p.add_argument("--what", action="append", choices=("logs", "probes", "samples", "maps", "loras", "best", "all"),
                   help="repeatable; default logs; maps: the training_maps folder")
    p.add_argument("--step", type=int, help="only this step's images and LoRA")
    p.add_argument("--dest", help="default: ~/.rulz_vast/runs/<instance>")
    p.add_argument("--show", type=int, default=20, help="how many fetched file names to list")
    p = command("push", cmd_push)
    p.add_argument("source")
    p.add_argument("target", help="folder on the instance, relative to the install: datasets, say")
    p.add_argument("--with-cache", action="store_true",
                   help="also send hidden files and folders, the dataset's .rulz_cache among them")
    p = command("sync", cmd_sync)
    p.add_argument("--force", action="store_true", help="even while a job runs")
    p.add_argument("--restart-webui", action="store_true", help="restart the WebUI on the new code")
    p = command("wait", cmd_wait)
    p.add_argument("name", nargs="?", help="job name; default: the running or last job")
    p.add_argument("--step", type=int, help="end when the job reaches this step instead")
    p.add_argument("--every", type=int, default=30, help="seconds between looks")
    p.add_argument("--timeout", type=int, default=0, help="give up after this many seconds")
    p.add_argument("--max-down", type=int, default=600, help="seconds the instance may stay out of reach")
    p = command("ssh", cmd_ssh)
    p.add_argument("words", nargs=argparse.REMAINDER, help="-- then the command")
    return ap


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    opts, rest = global_options().parse_known_args(argv)
    try:
        if rest and rest[0] in AGENT_COMMANDS:
            words, config = agent_args(rest)
            once = words[0] == "update" and "--check" not in words  # never run twice by a reconnect
            return forward(Remote(opts.instance, opts.key, opts.dir).agent(*words, config=config, retry=not once))
        args = parser().parse_args(argv)
        if args.func is not cmd_connect:
            args.remote = Remote(args.instance, args.key, args.dir)
        return args.func(args)
    except Unreachable as e:
        print(f"[unreachable] {e}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
