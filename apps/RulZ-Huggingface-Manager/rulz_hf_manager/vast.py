"""vast.ai side of the app: find your instances, attach to one over ssh, and send Hugging Face files to it.

The files never pass through this computer. The app plans the transfer here (which files, how big, where each one
goes), then hands the plan to ``remote_agent.py`` on the instance, which downloads straight from Hugging Face in a
background process that keeps going when the app closes.

Everything in here is blocking I/O: call it from ``nicegui.run.io_bound`` or a background job.
"""

from __future__ import annotations

import base64
import codecs
import fnmatch
import json
import os
import posixpath
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import config, hub
from .platform_utils import IS_WINDOWS, config_dir, is_wsl, windows_home_in_wsl

if TYPE_CHECKING:
    from .jobs import Job

VAST_API = os.environ.get('VAST_URL', 'https://console.vast.ai').rstrip('/') + '/api/v0'
AGENT = Path(__file__).with_name('remote_agent.py')
PAYLOAD_SLOT = "PAYLOAD = ''"
# The instance's Python: vast's own venv when there is one (it has pip), else the system python3.
REMOTE_PYTHON = ('P=/venv/main/bin/python; test -x "$P" || P="$(command -v python3 || command -v python)"; '
                 'exec "$P" -')
SSH_FAILED = 255
LAYOUTS = {
    'keep': 'Keep the repo\'s folders',
    'flat': 'Files only, no folders (e.g. into a ComfyUI models folder)',
    'repo': 'In a new folder named after the repo',
    'cache': 'Into the Hugging Face cache (HF_HOME), for from_pretrained()',
}


class VastError(Exception):
    """Something the user can act on: no key, no instance, ssh refused..."""


# --------------------------------------------------------------------------------------------------
# API key
# --------------------------------------------------------------------------------------------------

def _key_files() -> list[Path]:
    homes = [Path.home()]
    if is_wsl() and (win := windows_home_in_wsl()):
        homes.append(win)
    files = []
    for h in homes:
        files += [h / '.config' / 'vastai' / 'vast_api_key', h / '.vast_api_key']
    if xdg := os.environ.get('XDG_CONFIG_HOME'):
        files.insert(0, Path(xdg) / 'vastai' / 'vast_api_key')
    return files


def api_key() -> str | None:
    """The key from Settings, else VAST_API_KEY, else the one ``vastai set api-key`` saved."""
    if k := config.get().vast_api_key.strip():
        return k
    if k := os.environ.get('VAST_API_KEY', '').strip():
        return k
    for f in _key_files():
        try:
            k = f.read_text(encoding='utf-8').strip()
        except OSError:
            continue
        if k:
            return k
    return None


def key_source() -> str:
    if config.get().vast_api_key.strip():
        return 'saved in this app'
    if os.environ.get('VAST_API_KEY', '').strip():
        return 'VAST_API_KEY environment variable'
    for f in _key_files():
        if f.is_file() and f.read_text(encoding='utf-8', errors='replace').strip():
            return f'vastai CLI ({f})'
    return ''


# --------------------------------------------------------------------------------------------------
# Instances
# --------------------------------------------------------------------------------------------------

@dataclass
class Instance:
    id: int
    label: str = ''
    status: str = ''
    gpu: str = ''
    num_gpus: int = 1
    dollars_per_hour: float | None = None
    disk_gb: float | None = None
    disk_used_gb: float | None = None
    image: str = ''
    template: str = ''
    status_msg: str = ''
    routes: list[dict] = field(default_factory=list)

    @property
    def running(self) -> bool:
        return self.status == 'running'

    @property
    def title(self) -> str:
        gpu = f'{self.num_gpus}x {self.gpu}' if self.num_gpus and self.num_gpus > 1 else self.gpu
        return f'{self.label or "Instance"} {self.id} - {gpu}'.strip()


def routes(raw: dict) -> list[dict]:
    """Ways in, best first: the instance's direct port, then vast's ssh proxy (which is not always up)."""
    found = []
    ports = (raw.get('ports') or {}).get('22/tcp') or []
    if raw.get('public_ipaddr') and ports:
        found.append({'route': 'direct', 'host': str(raw['public_ipaddr']).strip(), 'port': int(ports[0]['HostPort'])})
    if raw.get('ssh_host') and raw.get('ssh_port'):
        found.append({'route': 'proxy', 'host': raw['ssh_host'], 'port': int(raw['ssh_port'])})
    return found


def to_instance(raw: dict) -> Instance:
    def num(v: Any) -> float | None:
        try:
            return float(v)
        except (TypeError, ValueError):
            return None
    return Instance(id=int(raw['id']), label=raw.get('label') or '', status=raw.get('actual_status') or '',
                    gpu=raw.get('gpu_name') or '', num_gpus=int(raw.get('num_gpus') or 1),
                    dollars_per_hour=num(raw.get('dph_total')), disk_gb=num(raw.get('disk_space')),
                    disk_used_gb=num(raw.get('disk_usage')), image=raw.get('image_uuid') or '',
                    template=raw.get('template_name') or '', status_msg=(raw.get('status_msg') or '').strip(),
                    routes=routes(raw))


def _api_get(path: str, key: str) -> Any:
    req = urllib.request.Request(f'{VAST_API}{path}', headers={'Authorization': f'Bearer {key}',
                                                               'Accept': 'application/json',
                                                               'User-Agent': 'rulz-hf-manager'})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            raise VastError('vast.ai refused the API key. Check it under Settings or at '
                            'https://cloud.vast.ai/manage-keys/') from e
        raise VastError(f'vast.ai answered {e.code} {e.reason}.') from e
    except urllib.error.URLError as e:
        raise VastError(f'Could not reach vast.ai: {e.reason}') from e


def list_instances() -> list[Instance]:
    key = api_key()
    if not key:
        raise VastError('No vast.ai API key. Add one on the vast.ai page (or run "vastai set api-key <key>").')
    data = _api_get('/instances/?owner=me', key)
    raw = data.get('instances', []) if isinstance(data, dict) else data
    return sorted((to_instance(r) for r in raw or []), key=lambda i: (not i.running, -i.id))


# --------------------------------------------------------------------------------------------------
# ssh
# --------------------------------------------------------------------------------------------------

@dataclass
class Conn:
    host: str
    port: int = 22
    user: str = 'root'
    route: str = 'manual'
    instance_id: int | None = None
    label: str = ''

    @property
    def key(self) -> str:
        return f'{self.instance_id or self.host.replace(":", "_")}'

    @property
    def title(self) -> str:
        return self.label or f'{self.user}@{self.host}:{self.port}'


_SSH_PORT_RE = re.compile(r'(?:^|\s)-p\s*(\d+)')
_SSH_DEST_RE = re.compile(r'(?:^|\s)(?:ssh://)?(?:([\w.\-]+)@)?([\w.\-]+\.[\w.\-]+|\d+\.\d+\.\d+\.\d+)(?::(\d+))?(?=\s|$)')


def parse_ssh_command(text: str) -> Conn:
    """Read what vast.ai's "Connect" box shows (``ssh -p 12345 root@ssh4.vast.ai -L 8080:localhost:8080``), or just
    ``root@1.2.3.4:41022`` / ``1.2.3.4 41022``."""
    text = (text or '').strip()
    if not text:
        raise ValueError('Paste the ssh command from the instance\'s Connect button on vast.ai.')
    body = re.sub(r'\s-[LRDWJ]\s*\S+', ' ', ' ' + text)  # drop port forwards and jumps
    body = re.sub(r'\s-[io]\s*\S+', ' ', body)
    port = None
    if m := _SSH_PORT_RE.search(body):
        port = int(m.group(1))
        body = body[:m.start()] + ' ' + body[m.end():]
    body = re.sub(r'(^|\s)ssh(\s|$)', ' ', body)
    m = _SSH_DEST_RE.search(body)
    if not m:
        raise ValueError('Could not find a host in that. Expected something like "ssh -p 12345 root@ssh4.vast.ai".')
    user, host, colon_port = m.group(1) or 'root', m.group(2), m.group(3)
    if colon_port:
        port = int(colon_port)
    if port is None:
        rest = body[m.end():].split()
        port = int(rest[0]) if rest and rest[0].isdigit() else 22
    return Conn(host=host, port=port, user=user, route='manual')


def ssh_exe() -> str:
    exe = shutil.which('ssh')
    if exe:
        return exe
    if IS_WINDOWS:
        candidate = Path(os.environ.get('SystemRoot', r'C:\Windows')) / 'System32' / 'OpenSSH' / 'ssh.exe'
        if candidate.exists():
            return str(candidate)
    raise VastError('No ssh program found. On Windows: Settings > System > Optional features > OpenSSH Client.')


def ssh_command(conn: Conn, batch: bool = True) -> list[str]:
    """ssh with this app's options. Each rental gets its own known-hosts file: vast hands the same host and port to new
    rentals with new host keys, and that must not read as an attack on the next one."""
    known = config_dir() / 'ssh' / f'known_hosts_{conn.key}'
    known.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ssh_exe(), '-p', str(conn.port), '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=20',
           '-o', 'ServerAliveCountMax=4', '-o', 'LogLevel=ERROR', '-o', 'StrictHostKeyChecking=accept-new',
           '-o', f'UserKnownHostsFile={known.as_posix()}', '-T']
    if batch:
        cmd += ['-o', 'BatchMode=yes']
    key = config.get().ssh_key.strip()
    if key:
        cmd += ['-i', os.path.expanduser(key), '-o', 'IdentitiesOnly=yes']
    return cmd + [f'{conn.user}@{conn.host}']


_NO_WINDOW = 0x08000000 if IS_WINDOWS else 0  # CREATE_NO_WINDOW: no console flashing up on every call


def _ssh_error(stderr: bytes | str, conn: Conn) -> str:
    text = stderr.decode('utf-8', 'replace') if isinstance(stderr, bytes) else stderr
    text = text.strip()
    low = text.lower()
    if 'permission denied' in low:
        return (f'{conn.title} refused your ssh key. Add your public key (e.g. ~/.ssh/id_ed25519.pub) under '
                'Keys on cloud.vast.ai; an instance rented before the key was added needs it added to the instance '
                '(its key button on vast.ai).')
    if 'connection refused' in low or 'timed out' in low or 'no route' in low or 'closed by remote' in low \
            or 'connection reset' in low:
        return f'Could not connect to {conn.title} ({text.splitlines()[-1] if text else "no answer"}).'
    return text.splitlines()[-1] if text else f'ssh to {conn.title} failed.'


def agent_program(payload: dict) -> bytes:
    """remote_agent.py as it goes down the pipe, the call's arguments written into it."""
    source = AGENT.read_text(encoding='utf-8')
    encoded = base64.b64encode(json.dumps(payload).encode('utf-8')).decode('ascii')
    if PAYLOAD_SLOT not in source:
        raise RuntimeError('remote_agent.py has no payload slot')
    return source.replace(PAYLOAD_SLOT, f"PAYLOAD = '{encoded}'", 1).encode('utf-8')


def runner_source() -> str:
    return base64.b64encode(AGENT.read_bytes()).decode('ascii')


def _last_json(text: str) -> dict:
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if line.startswith('{'):
            try:
                return json.loads(line)
            except ValueError:
                continue
    raise VastError('The instance gave no answer the app understands: ' + (text.strip()[-300:] or 'nothing'))


class Remote:
    """One attached instance over ssh."""

    def __init__(self, conn: Conn) -> None:
        self.conn = conn

    def _run(self, payload: dict, timeout: float = 90) -> subprocess.CompletedProcess:
        return subprocess.run(ssh_command(self.conn) + [REMOTE_PYTHON], input=agent_program(payload),
                              capture_output=True, timeout=timeout, creationflags=_NO_WINDOW)

    def call(self, cmd: str, timeout: float = 90, **args: Any) -> dict:
        payload = {'cmd': cmd, **args}
        try:
            done = self._run(payload, timeout)
        except subprocess.TimeoutExpired as e:
            raise VastError(f'{self.conn.title} did not answer within {int(timeout)} s.') from e
        if done.returncode == SSH_FAILED and self.conn.instance_id:
            # The instance may have moved (new port after a restart): ask vast for its current address once.
            fresh = reconnect(self.conn)
            if fresh:
                self.conn = fresh
                done = self._run(payload, timeout)
        if done.returncode == SSH_FAILED:
            raise VastError(_ssh_error(done.stderr, self.conn))
        stdout = done.stdout.decode('utf-8', 'replace')
        if done.returncode != 0 and not stdout.strip():
            err = done.stderr.decode('utf-8', 'replace').strip()
            raise VastError(f'The instance could not run the helper: {err[-400:] or f"exit {done.returncode}"}')
        result = _last_json(stdout)
        if isinstance(result, dict) and result.get('error') and cmd not in {'ls', 'df'}:
            raise VastError(result['error'])
        return result

    def stream(self, cmd: str, **args: Any) -> Iterator[dict]:
        """Run a call that prints a JSON line now and then (``watch``) and yield each line as it arrives."""
        # stderr goes to a temp file, not a pipe nobody reads while the call runs (a full pipe would stall ssh).
        with tempfile.TemporaryFile() as err:
            proc = subprocess.Popen(ssh_command(self.conn) + [REMOTE_PYTHON], stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE, stderr=err, creationflags=_NO_WINDOW)
            got = False
            try:
                assert proc.stdin and proc.stdout
                proc.stdin.write(agent_program({'cmd': cmd, **args}))
                proc.stdin.close()
                for line in proc.stdout:
                    line = line.strip()
                    if line.startswith(b'{'):
                        try:
                            data = json.loads(line)
                        except ValueError:
                            continue
                        got = True
                        yield data
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired as e:
                    raise VastError(f'{self.conn.title} stopped answering.') from e
                if proc.returncode == SSH_FAILED or (proc.returncode and not got):
                    # Nothing understandable came back (ssh failed, or the instance could not run the helper):
                    # a failure, so the caller backs off instead of reconnecting at once, forever.
                    err.seek(0)
                    text = err.read()
                    if proc.returncode == SSH_FAILED:
                        raise VastError(_ssh_error(text, self.conn))
                    raise VastError('The instance could not run the helper: '
                                    + (text.decode('utf-8', 'replace').strip()[-400:] or f'exit {proc.returncode}'))
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()

    def test(self) -> dict:
        return self.call('info', timeout=60)


def reconnect(conn: Conn) -> Conn | None:
    """Look the instance up on vast.ai again and return the first route that answers."""
    if not conn.instance_id:
        return None
    try:
        inst = next((i for i in list_instances() if i.id == conn.instance_id), None)
    except VastError:
        return None
    if not inst or not inst.running:
        return None
    for r in inst.routes:
        cand = Conn(host=r['host'], port=r['port'], route=r['route'], instance_id=inst.id, label=inst.title)
        if reachable(cand):
            save_attached(cand)
            return cand
    return None


def reachable(conn: Conn) -> bool:
    try:
        done = subprocess.run(ssh_command(conn) + ['echo rulz-ok'], capture_output=True, timeout=30,
                              stdin=subprocess.DEVNULL, creationflags=_NO_WINDOW)
    except (subprocess.TimeoutExpired, VastError):
        return False
    return done.returncode == 0 and b'rulz-ok' in done.stdout


def attach_instance(inst: Instance) -> tuple[Conn, dict]:
    """Try the instance's routes, remember the first that works, and return it with the instance's info."""
    if not inst.running:
        raise VastError(f'Instance {inst.id} is {inst.status or "not running"}: start it on vast.ai first.')
    if not inst.routes:
        raise VastError(f'Instance {inst.id} has no ssh address yet: it may still be starting. Try again in a minute.')
    errors = []
    for r in inst.routes:
        conn = Conn(host=r['host'], port=r['port'], route=r['route'], instance_id=inst.id, label=inst.title)
        try:
            info = Remote(conn).test()
        except VastError as e:
            errors.append(f'{r["route"]}: {e}')
            continue
        save_attached(conn)
        return conn, info
    raise VastError('Could not reach the instance.\n' + '\n'.join(errors))


def attach_manual(conn: Conn) -> tuple[Conn, dict]:
    info = Remote(conn).test()
    save_attached(conn)
    return conn, info


def attached() -> Conn | None:
    data = config.get().vast_conn
    if not data or not data.get('host'):
        return None
    known = {k: data[k] for k in ('host', 'port', 'user', 'route', 'instance_id', 'label') if k in data}
    return Conn(**known)


def save_attached(conn: Conn | None) -> None:
    config.get().vast_conn = asdict(conn) if conn else {}
    config.save()


def remote() -> Remote:
    conn = attached()
    if not conn:
        raise VastError('No vast.ai instance attached. Attach one on the vast.ai page first.')
    return Remote(conn)


# --------------------------------------------------------------------------------------------------
# Command prompt: one shell command at a time on the attached instance
# --------------------------------------------------------------------------------------------------

SHELL_MARK = '__RULZ_SHELL_DONE__'
# CSI (with private parameters like "\x1b[>4;1m"), OSC titles, charset picks, and the two-character escapes
# (ESC 7 / ESC 8 save and restore the cursor around apt's progress line, ESC M, ESC =, ...).
_ANSI_RE = re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b[()][A-Za-z0-9]'
                      r'|\x1b[78=>DEHMNOZc]')
# What can still grow into one of those when more output arrives (held back until it does).
_ANSI_START_RE = re.compile(r'\x1b(?:\[[0-?]*[ -/]*|\][^\x07\x1b]*\x1b?|[()])?')
_MARK_RE = re.compile(rf'^{SHELL_MARK} (\d+) (.*)$')


def shell_script(command: str, cwd: str) -> str:
    """The bash script one prompt line runs as: in ``cwd``, with vast's venv on PATH (a non-interactive shell does not
    get it from .bashrc), then a last line that reports the exit code and the folder the command ended in."""
    q = "'" + cwd.replace("'", "'\\''") + "'"
    return (f'cd {q} 2>/dev/null || cd ~\n'
            'if [ -x /venv/main/bin/python ]; then export VIRTUAL_ENV=/venv/main PATH="/venv/main/bin:$PATH"; fi\n'
            'export TERM=dumb PAGER=cat GIT_PAGER=cat SYSTEMD_PAGER=cat PYTHONUNBUFFERED=1 NO_COLOR=1\n'
            f'{command}\n'
            f'__rulz_rc=$?; printf "\\n{SHELL_MARK} %s %s\\n" "$__rulz_rc" "$PWD"\n')


def shell_ssh_command(conn: Conn, command: str, cwd: str) -> list[str]:
    """ssh with a forced tty, so stopping (closing the connection) hangs the command up instead of leaving it running,
    and programs flush their output line by line. Not an interactive shell: vast's .bashrc would start tmux."""
    encoded = base64.b64encode(shell_script(command, cwd).encode('utf-8')).decode('ascii')
    return ssh_command(conn) + ['-tt', f'bash -lc "$(echo {encoded} | base64 -d)"']


def clean_terminal_text(text: str) -> str:
    return _ANSI_RE.sub('', text).replace('\x07', '').replace('\x08', '')


def split_unfinished_escape(text: str) -> tuple[str, str]:
    """Cut off an escape sequence that a read split in two (``\\x1b[3`` now, ``2m`` with the next read), so it is
    cleaned as a whole next time instead of leaking ``2m`` into the output."""
    start = text.rfind('\x1b')
    if start < 0 or len(text) - start > 64 or not _ANSI_START_RE.fullmatch(text, start):
        return text, ''  # no escape at the end, or a finished one (or junk): nothing to wait for
    return text[:start], text[start:]


class TerminalDecoder:
    """Bytes from the terminal to clean text, chunk by chunk: a UTF-8 character or an escape sequence cut in two by
    a read is held back until the rest arrives."""

    def __init__(self) -> None:
        self._utf8 = codecs.getincrementaldecoder('utf-8')(errors='replace')
        self._held = ''

    def feed(self, chunk: bytes) -> str:
        text, self._held = split_unfinished_escape(self._held + self._utf8.decode(chunk))
        return clean_terminal_text(text)

    def flush(self) -> str:
        text, self._held = self._held + self._utf8.decode(b'', final=True), ''
        return clean_terminal_text(text)


def split_output(buffer: str) -> tuple[list[tuple[str, str]], str]:
    """Cut terminal output into ('line', text) for finished lines and ('cr', text) for a line that a carriage return
    redraws (progress bars); returns the pieces and the unfinished rest."""
    out: list[tuple[str, str]] = []
    buffer = buffer.replace('\r\n', '\n')
    while True:
        nl, cr = buffer.find('\n'), buffer.find('\r')
        if nl < 0 and cr < 0:
            return out, buffer
        if cr >= 0 and (nl < 0 or cr < nl):
            if cr == len(buffer) - 1:  # might be the start of \r\n: wait for more
                return out, buffer
            piece, buffer = buffer[:cr], buffer[cr + 1:]
            if piece:
                out.append(('cr', piece))
        else:
            out.append(('line', buffer[:nl]))
            buffer = buffer[nl + 1:]


class ShellRun:
    """A command running on the instance. ``events()`` yields ('line'|'cr', text) while it runs and finally
    ('done', ''); afterwards ``code`` and ``cwd`` hold its exit code and the folder it ended in."""

    def __init__(self, conn: Conn, command: str, cwd: str) -> None:
        self.conn = conn
        self.command = command
        self.cwd = cwd
        self.code: int | None = None
        self.stopped = False
        self.error = ''
        self.proc = subprocess.Popen(shell_ssh_command(conn, command, cwd), stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=_NO_WINDOW)

    def events(self) -> Iterator[tuple[str, str]]:
        assert self.proc.stdout
        rest = ''
        blank = 0  # empty lines held back: the last one comes from the end marker's own newline
        decoder = TerminalDecoder()
        while True:
            chunk = self.proc.stdout.read1(65536) if hasattr(self.proc.stdout, 'read1') else self.proc.stdout.read(1)
            pieces, rest = split_output(rest + (decoder.feed(chunk) if chunk else decoder.flush()))
            for kind, text in pieces:
                if kind == 'line' and (m := _MARK_RE.match(text.strip())):
                    self.code, self.cwd = int(m.group(1)), m.group(2) or self.cwd
                    blank = max(0, blank - 1)
                    continue
                if kind == 'line' and not text.strip():
                    blank += 1
                    continue
                for _ in range(blank):
                    yield 'line', ''
                blank = 0
                yield kind, text
            if not chunk:
                break
        for _ in range(blank):
            yield 'line', ''
        if rest.strip() and not _MARK_RE.match(rest.strip()):
            yield 'line', rest
        try:
            self.proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        if self.code is None and not self.stopped:
            if self.proc.returncode == SSH_FAILED:
                err = self.proc.stderr.read() if self.proc.stderr else b''
                self.error = _ssh_error(err, self.conn)
            else:
                self.code = self.proc.returncode
        yield 'done', ''

    def send(self, text: str) -> None:
        """Type a line into the running command (an answer to a y/n question, say)."""
        if self.proc.poll() is None and self.proc.stdin:
            try:
                self.proc.stdin.write(text.encode('utf-8') + b'\n')
                self.proc.stdin.flush()
            except OSError:
                pass

    def stop(self) -> None:
        """Close the connection: the command's terminal goes away and the command gets a hangup."""
        self.stopped = True
        if self.proc.poll() is None:
            if self.proc.stdin:
                try:
                    self.proc.stdin.write(b'\x03')  # Ctrl+C through the tty first
                    self.proc.stdin.flush()
                except OSError:
                    pass
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def run_shell(command: str, cwd: str = '/workspace') -> ShellRun:
    conn = attached()
    if not conn:
        raise VastError('No vast.ai instance attached. Attach one on the vast.ai page first.')
    return ShellRun(conn, command, cwd)


def ssh_login_command(conn: Conn) -> list[str]:
    """ssh for a person at a keyboard: a terminal, and ssh may ask questions (key passphrase)."""
    cmd = ssh_command(conn, batch=False)
    cmd.remove('-T')
    return cmd


def ssh_login_text(conn: Conn) -> str:
    key = config.get().ssh_key.strip()
    return f'ssh -p {conn.port}' + (f' -i "{key}"' if key else '') + f' {conn.user}@{conn.host}'


def open_terminal(conn: Conn | None = None) -> str:
    """Open a terminal window on this computer with an ssh session to the instance. Returns what was started."""
    conn = conn or attached()
    if not conn:
        raise VastError('No vast.ai instance attached. Attach one on the vast.ai page first.')
    cmd = ssh_login_command(conn)
    if IS_WINDOWS:
        subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_CONSOLE)  # type: ignore[attr-defined]
        return 'a console window'
    if is_wsl() and shutil.which('cmd.exe') and shutil.which('wsl.exe'):
        # A Windows terminal running this same ssh inside WSL (the WSL side has the keys and known hosts).
        subprocess.Popen(['cmd.exe', '/c', 'start', 'vast.ai', 'wsl.exe', '-d',
                          os.environ.get('WSL_DISTRO_NAME', ''), '--', *cmd], cwd='/mnt/c')
        return 'a Windows console running WSL'
    for term, args in (('x-terminal-emulator', ['-e']), ('gnome-terminal', ['--']), ('konsole', ['-e']),
                       ('xfce4-terminal', ['-x']), ('kitty', []), ('alacritty', ['-e']), ('xterm', ['-e'])):
        if shutil.which(term):
            subprocess.Popen([term, *args, *cmd], start_new_session=True)
            return term
    if sys.platform == 'darwin':
        script = ' '.join("'" + c.replace("'", "'\\''") + "'" for c in cmd)
        subprocess.Popen(['osascript', '-e', f'tell app "Terminal" to do script "{script}"'])
        return 'Terminal'
    raise VastError('No terminal program found here. Copy the ssh command and run it in a terminal.')


# --------------------------------------------------------------------------------------------------
# Planning a transfer (here, with the user's own Hugging Face access)
# --------------------------------------------------------------------------------------------------

@dataclass
class Source:
    """Something to send: a whole repo / bucket, or some files and folders of it."""
    kind: str
    repo_id: str
    revision: str | None = None
    paths: list[str] = field(default_factory=list)  # empty = everything
    dirs: list[str] = field(default_factory=list)  # which of ``paths`` are folders

    @property
    def whole(self) -> bool:
        return not self.paths

    @property
    def label(self) -> str:
        if self.whole:
            return f'{self.repo_id} (whole {hub.KIND_LABEL[self.kind].lower()})'
        if len(self.paths) == 1:
            return f'{self.repo_id}/{self.paths[0]}{"/" if self.paths[0] in self.dirs else ""}'
        return f'{self.repo_id} ({len(self.paths)} items)'

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_entries(cls, kind: str, repo_id: str, revision: str | None, entries: list[hub.Entry]) -> Source:
        return cls(kind, repo_id, revision, [e.path for e in entries], [e.path for e in entries if e.is_dir])


def source_from_text(text: str) -> Source:
    """A Source from a pasted id / huggingface.co link / hf:// path (a link to a file or folder sends just that)."""
    kind, repo_id, path = hub.parse_location(text)
    if kind is None:
        kind = hub.detect_kind(repo_id)
    revision = hub.parse_revision(text, kind)
    if not path:
        return Source(kind, repo_id, revision)
    is_dir = '/tree/' in text or text.rstrip().endswith('/')
    if kind == 'bucket' and not is_dir:
        is_dir = not posixpath.splitext(path)[1]
    return Source(kind, repo_id, revision, [path.strip('/')], [path.strip('/')] if is_dir else [])


@dataclass
class PlanItem:
    kind: str
    repo: str
    rev: str | None
    path: str
    size: int
    dest: str


@dataclass
class Plan:
    items: list[PlanItem]
    skipped_by_filter: int = 0

    @property
    def total(self) -> int:
        return sum(i.size for i in self.items)

    def collisions(self) -> list[str]:
        seen: dict[str, PlanItem] = {}
        clashes = []
        for i in self.items:
            if i.dest in seen and (seen[i.dest].repo, seen[i.dest].path) != (i.repo, i.path):
                clashes.append(i.dest)
            seen[i.dest] = i
        return sorted(set(clashes))


def _relative(path: str, source: Source) -> str:
    """A file's path relative to the folder holding the selected item it belongs to (selecting
    ``transformer/`` keeps ``transformer/...``; selecting ``a/b/model.safetensors`` gives ``model.safetensors``)."""
    for p in source.paths:
        if path == p or (p in source.dirs and path.startswith(p.rstrip('/') + '/')):
            parent = posixpath.dirname(p.rstrip('/'))
            return path[len(parent) + 1:] if parent else path
    return posixpath.basename(path)


def matches(path: str, patterns: list[str]) -> bool:
    name = posixpath.basename(path)
    return any(fnmatch.fnmatch(path, p) or fnmatch.fnmatch(name, p) for p in patterns)


def split_patterns(text: str | None) -> list[str]:
    return [p.strip() for p in re.split(r'[,;\s]+', text or '') if p.strip()]


def list_source_files(source: Source) -> list[tuple[str, int, str]]:
    """(path in repo, size, path relative to the selection) for every file the source covers."""
    fs = hub.open_fs(source.kind, source.repo_id, source.revision)
    if source.whole:
        files = fs.list_files('')
        return [(f.path, int(f.size or 0), f.path) for f in files]
    # Files picked in the browser come with sizes; a pasted link does not, and may even name a folder.
    unknown = [p for p in source.paths if p not in source.dirs]
    info = _paths_info(fs, unknown) if unknown else {}
    missing = [p for p in unknown if p not in info]
    if missing:
        raise VastError(f'Not found in {source.repo_id}: {", ".join(missing[:5])}')
    dirs = set(source.dirs) | {p for p, size in info.items() if size is None}
    if dirs != set(source.dirs):
        source = Source(source.kind, source.repo_id, source.revision, source.paths, sorted(dirs))
    out: list[tuple[str, int, str]] = []
    for p in source.paths:
        if p in dirs:
            out += [(f.path, int(f.size or 0), _relative(f.path, source)) for f in fs.list_files(p)]
        else:
            out.append((p, int(info.get(p) or 0), _relative(p, source)))
    return out


def _paths_info(fs: hub.RemoteFS, paths: list[str]) -> dict[str, int | None]:
    """Size of each path that exists (None for a folder)."""
    found: dict[str, int | None] = {}
    for i in range(0, len(paths), 500):
        chunk = paths[i:i + 500]
        if fs.is_bucket:
            infos = fs.api.get_bucket_paths_info(fs.repo_id, chunk)
        else:
            infos = fs.api.get_paths_info(fs.repo_id, chunk, repo_type=fs.kind, revision=fs.revision)
        for x in infos:
            is_dir = getattr(x, 'type', None) in {'directory', 'folder'} or type(x).__name__ in {'RepoFolder',
                                                                                              'BucketFolder'}
            found[x.path.rstrip('/')] = None if is_dir else int(getattr(x, 'size', 0) or 0)
    return found


def layout_dest(source: Source, rel: str, layout: str) -> str:
    if layout == 'flat':
        return posixpath.basename(rel)
    if layout == 'repo':
        return f'{source.repo_id.split("/")[-1]}/{rel}'
    return rel


def build_plan(sources: list[Source], layout: str, include: list[str] | None = None,
               exclude: list[str] | None = None, listed: dict[int, list[tuple[str, int, str]]] | None = None) -> Plan:
    """Turn sources into a list of files with their destination paths. ``listed`` reuses file lists already fetched
    (keyed by the source's position) so changing the layout or filter does not ask the Hub again."""
    if layout not in LAYOUTS:
        raise ValueError(f'Unknown layout {layout!r}')
    items: list[PlanItem] = []
    filtered = 0
    for n, src in enumerate(sources):
        if layout == 'cache' and src.kind == 'bucket':
            raise ValueError(f'{src.repo_id} is a bucket: buckets cannot go into the Hugging Face cache. '
                             'Pick another layout.')
        files = listed[n] if listed is not None and n in listed else list_source_files(src)
        for path, size, rel in files:
            if path.startswith('.cache/') or path in {'.gitattributes'} and layout == 'flat':
                filtered += 1
                continue
            if (include and not matches(path, include)) or (exclude and matches(path, exclude)):
                filtered += 1
                continue
            items.append(PlanItem(src.kind, src.repo_id, src.revision, path, size, layout_dest(src, rel, layout)))
    return Plan(items, filtered)


def start_transfer(plan: Plan, dest: str, layout: str, title: str, workers: int = 4) -> dict:
    """Hand the plan to the instance. Returns the job's first status (with its id)."""
    if not plan.items:
        raise VastError('Nothing to send: no file matches.')
    dest = dest.strip()
    if not dest:
        raise VastError('Choose where the files should go on the instance.')
    if clashes := plan.collisions():
        raise VastError(f'{len(clashes)} file(s) would land on the same name ({", ".join(clashes[:3])}...). '
                        'Pick "Keep the repo\'s folders" or "In a new folder named after the repo".')
    job = {'title': title, 'dest': dest, 'mode': layout, 'workers': workers,
           'items': [asdict(i) for i in plan.items]}
    token = hub.configured_token() or hub.external_token() or ''
    return remote().call('start', timeout=120, job=job, token=token, source=runner_source())


def retry_transfer(job_id: str) -> dict:
    token = hub.configured_token() or hub.external_token() or ''
    return remote().call('retry', id=job_id, token=token, source=runner_source())


def follow(job_id: str, job: Job | None = None, conn: Conn | None = None) -> dict:
    """Follow a transfer on the instance until it ends, reporting into ``job``. Reconnects when the connection drops;
    the transfer itself never depends on this."""
    rem = Remote(conn) if conn else remote()
    failures = 0
    while True:
        last: dict = {}
        try:
            for st in rem.stream('watch', id=job_id, interval=2, timeout=900):
                last = st
                failures = 0
                if job:
                    report(job, st)
                if st.get('state') in {'done', 'error', 'cancelled', 'stopped'}:
                    return st
            if not last:
                raise VastError('The instance sent no status.')
        except VastError:
            failures += 1
            if failures > 8:
                raise
            if job:
                job.set(detail=f'Connection lost, reconnecting ({failures})... the transfer keeps going')
            time.sleep(min(60, 5 * failures))
            if rem.conn.instance_id and (fresh := reconnect(rem.conn)):
                rem = Remote(fresh)
            continue
        if last.get('error') and not last.get('state'):
            raise VastError(last['error'])  # the helper itself refused (bad job id...): retrying will not help


def report(job: Job, st: dict) -> None:
    total = st.get('bytes_total') or 0
    done = (st.get('bytes_done') or 0) + (st.get('partial') or 0)
    progress = min(1.0, done / total) if total else None
    files = f'{st.get("files_done", 0)}/{st.get("files_total", 0)} files'
    size = f'{hub.human_size(done)} of {hub.human_size(total)}' if total else ''
    speed = st.get('speed') or 0
    detail = ' | '.join(x for x in (size, f'{hub.human_size(speed)}/s' if speed > 1024 else '') if x)
    state = st.get('state')
    message = {'installing': st.get('message') or 'Preparing the instance...', 'queued': 'Starting on the instance...',
               'starting': 'Starting on the instance...'}.get(state or '', files)
    job.set(progress=progress, message=message, detail=detail)


def describe_outcome(st: dict) -> str:
    state = st.get('state')
    n, total = st.get('files_done', 0), st.get('files_total', 0)
    skipped = st.get('skipped') or 0
    extra = f' ({skipped} already there)' if skipped else ''
    if state == 'done':
        return f'{n} file(s), {hub.human_size(st.get("bytes_total") or 0)} on the instance in {st.get("dest")}{extra}'
    if state == 'cancelled':
        raise VastError('Cancelled on the instance.')
    failed = st.get('failed') or []
    first = f'\nFirst: {failed[0]["file"]}: {failed[0]["error"]}' if failed else ''
    raise VastError((st.get('error') or f'Stopped after {n}/{total} files.') + first)


def remember_destination(path: str) -> None:
    s = config.get()
    path = path.strip()
    if not path:
        return
    s.vast_recent_dests = [path] + [p for p in s.vast_recent_dests if p != path]
    del s.vast_recent_dests[10:]
    config.save()

