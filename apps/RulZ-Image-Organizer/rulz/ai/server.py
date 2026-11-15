"""llama.cpp's llama-server for vision tagging, run with the Training Captioner's settings.

Used when RulZ shares the Training Captioner's server settings and the Captioner runs llama-server
itself ("managed" backend). When the Captioner's own servers are already running, RulZ uses them and
starts nothing: a second copy of the model would need its VRAM twice. Otherwise RulZ starts its own
llama-server(s) with the same settings on its own ports, so the two apps never fight over a port, and
stops them on request or when the app exits.

Plain worker threads call this, so there is no Qt here, and HTTP is the standard library only: the app
ships without `requests` and without Qt's network module.
"""
from __future__ import annotations

import atexit
import http.client
import os
import shlex
import socket
import subprocess
import threading
import time
from collections.abc import Callable

from . import winjob

CREATE_NO_WINDOW = 0x08000000
HOST = "127.0.0.1"
START_TIMEOUT = 900.0  # seconds for every server to load its model: big models from slow disks take minutes
POLL_INTERVAL = 0.25
# Windows takes 2 s to report a refused connection on this computer (it retries the SYN), while a server
# that listens accepts at once. Connecting with a short limit keeps the start loop and Cancel quick.
CONNECT_TIMEOUT = 0.5
LOADING_CONNECT_TIMEOUT = 0.25  # in the start loop, where a "no" only means asking again 0.25 s later
TAIL_LINES = 50   # output lines kept per server
ERROR_LINES = 12  # of those, shown when a server dies while loading
SETTINGS_HINT = "the Training Captioner's Settings ▸ Backend"
PORT_HINT = "'First port for RulZ's own llama-server' in Advanced AI settings"


class ServerError(RuntimeError):
    """User-facing message: llama-server can't be started or died."""


class Cancelled(RuntimeError):
    """cancelled() turned true (or stop() was called) while starting."""


def port_in_use(port: int, host: str = HOST) -> bool:
    """True when another socket already holds the port. A llama-server started there would fail to
    bind, and its health check would be answered by whatever holds the port instead."""
    with socket.socket() as sock:
        try:
            sock.bind((host, port))
        except OSError:
            return True
    return False


def build_plan(cap: dict, first_port: int) -> list[tuple[int, dict, list[str]]]:
    """(port, env_overrides, extra_cli_args) for each server, by the Captioner's rules: no GPUs picked
    -> one server; 'split' over several GPUs -> one server spanning them; else one server per GPU."""
    gpus = [str(g) for g in cap["gpus"]]
    base = int(first_port)
    if not gpus:
        return [(base, {}, [])]
    if cap["gpu_mode"] == "split" and len(gpus) > 1:
        args = ["--split-mode", "layer"]
        ts = str(cap["tensor_split"]).strip()
        if ts:
            args += ["--tensor-split", ts]
        return [(base, {"CUDA_VISIBLE_DEVICES": ",".join(gpus)}, args)]
    return [(base + i, {"CUDA_VISIBLE_DEVICES": g}, []) for i, g in enumerate(gpus)]


def build_command(cap: dict, port: int, plan_args: list[str], disable_thinking: bool) -> list[str]:
    """llama-server's command line for one server of the plan, as the Captioner builds it."""
    slots = max(1, int(cap["slots_per_server"]))
    cmd = [cap["llama_server"], "-m", cap["model"], "-ngl", str(cap["ngl"]),
           "-c", str(int(cap["ctx_size"]) * slots),  # the context is shared by the slots
           "-np", str(slots), "--host", HOST, "--port", str(port)]
    if cap["mmproj"]:
        cmd += ["--mmproj", cap["mmproj"]]
    if disable_thinking:
        cmd += ["--reasoning", "off"]
    else:
        cmd += ["--reasoning-budget", str(int(cap["thinking_budget"]))]
    extra = cap["extra_args"]
    try:
        # posix=False keeps Windows paths' backslashes; the quotes it leaves around a word go
        extra_args = [a.strip('"') for a in shlex.split(extra, posix=False)] if extra.strip() else []
    except ValueError as e:  # an unclosed quote
        raise ServerError(f"The extra llama-server arguments can't be read ({e}). "
                          f"Fix them in {SETTINGS_HINT}.") from e
    return cmd + list(plan_args) + extra_args


def server_env(env_over: dict) -> dict:
    """Environment for one llama-server. The Captioner's GPU list uses nvidia-smi's numbering (PCI bus
    order), but CUDA numbers the fastest card first unless CUDA_DEVICE_ORDER says otherwise."""
    env = dict(os.environ)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env.update(env_over)
    return env


def server_name(port: int, env_over: dict) -> str:
    gpu = env_over.get("CUDA_VISIBLE_DEVICES")
    return f":{port} GPU {gpu}" if gpu else f":{port}"


def _endpoint(port: int, env_over: dict, slots: int) -> dict:
    return {"url": f"http://{HOST}:{port}/v1", "model": "default", "slots": slots,
            "name": server_name(port, env_over)}


class _Connection(http.client.HTTPConnection):
    """http.client never looks at proxy settings, which is the point: Windows' system proxy caught
    requests to 127.0.0.1 in the Captioner. Connecting has its own limit; the reply gets `timeout`."""

    def __init__(self, host, port, timeout, connect_timeout):
        super().__init__(host, port, timeout=timeout)
        self.connect_timeout = connect_timeout

    def connect(self):
        self.sock = socket.create_connection((self.host, self.port), self.connect_timeout)
        self.sock.settimeout(self.timeout)


def healthy(port: int, timeout: float = 2.0, connect_timeout: float = CONNECT_TIMEOUT) -> bool:
    """GET http://127.0.0.1:{port}/health answers 200 (proxies ignored). llama-server answers 503 while
    it loads the model. Connecting is limited to `connect_timeout` (at most `timeout`)."""
    conn = _Connection(HOST, port, timeout, min(timeout, connect_timeout))
    try:
        conn.request("GET", "/health")
        resp = conn.getresponse()
        resp.read()
        return resp.status == 200
    except (OSError, http.client.HTTPException, ValueError, OverflowError):  # OverflowError: port > 65535
        return False
    finally:
        conn.close()


class _Server:
    """One llama-server this app started."""

    def __init__(self, popen: subprocess.Popen, port: int, name: str, job):
        self.popen, self.port, self.name, self.job = popen, port, name, job
        self.tail: list[str] = []  # its last output lines
        self.pump: threading.Thread | None = None

    def alive(self) -> bool:
        return self.popen.poll() is None


def _pump(srv: _Server, log: Callable[[str], None]):
    """Read a server's output into the log. It must keep reading whatever `log` does: a full pipe
    would freeze llama-server on its next line of output."""
    tag = f"[:{srv.port}]"
    try:
        for line in iter(srv.popen.stdout.readline, b""):
            text = line.decode("utf-8", "replace").rstrip()
            srv.tail.append(text)
            del srv.tail[:-TAIL_LINES]
            try:
                log(f"{tag} {text}")
            except Exception:  # noqa: BLE001  (a closed log window, say)
                pass
    except (OSError, ValueError):
        pass
    finally:
        try:
            srv.popen.stdout.close()
        except OSError:
            pass


def _kill(servers: list[_Server]):
    for srv in servers:
        if srv.job is not None and srv.job.terminate():
            continue  # the server and anything it started, at once
        if srv.alive() and os.name == "nt":
            try:  # not in a job: taskkill finds its child processes itself
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(srv.popen.pid)],
                               capture_output=True, creationflags=CREATE_NO_WINDOW)
            except OSError:
                srv.popen.kill()
        elif srv.alive():
            srv.popen.kill()
    deadline = time.monotonic() + 10
    for srv in servers:  # its port and VRAM are only free once it has exited; a restart needs both
        try:
            srv.popen.wait(timeout=max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass
        if srv.job is not None:
            # and so has anything it started: a .cmd wrapper exits before the server inside it
            srv.job.wait(max(0.1, deadline - time.monotonic()))
            srv.job.close()


class LlamaServers:
    """The llama-server(s) RulZ uses in "share the Training Captioner's settings" mode.

    Thread-safe: ensure() calls run one at a time (a second waits for the first, then reuses its
    servers), and stop() from any thread makes a waiting ensure() raise Cancelled."""

    def __init__(self):
        self._lock = threading.Lock()         # guards the fields below; never held during HTTP or waits
        self._ensure_lock = threading.Lock()  # one ensure() at a time, so two jobs can't start two sets
        self._servers: list[_Server] = []     # started by this object, loading or ready
        self._endpoints: list[dict] = []      # set once all of _servers answered healthy
        self._key = None                      # the commands _servers were started with
        self._gen = 0                         # bumped by stop(); an ensure() that sees it change gives up
        atexit.register(self.stop)

    def running(self) -> bool:
        """Servers this object started are all still alive (loading or ready)."""
        with self._lock:
            servers = list(self._servers)
        return bool(servers) and all(s.alive() for s in servers)

    def ensure(self, cap: dict, first_port: int, disable_thinking: bool,
               log: Callable[[str], None], cancelled: Callable[[], bool]) -> tuple[list[dict], str]:
        """Return (endpoints, source): the Captioner's running servers ('captioner'), or this object's
        own, reused or started now and ready ('own'). Raises ServerError with a message for the user,
        or Cancelled. On a failure or cancel, the servers this call started are killed."""
        log = log or (lambda line: None)
        cancelled = cancelled or (lambda: False)
        while not self._ensure_lock.acquire(timeout=POLL_INTERVAL):
            if cancelled():
                raise Cancelled("Cancelled")
        try:
            gen = self._gen  # a stop() from here on cancels this call

            def check():
                if self._gen != gen or cancelled():
                    raise Cancelled("Cancelled")

            slots = max(1, int(cap["slots_per_server"]))
            theirs = self._captioner_endpoints(cap, slots)
            if theirs:
                log("Using the Training Captioner's running llama-server "
                    f"({', '.join(ep['name'] for ep in theirs)}).")
                mine = self._take_all()
                if mine:  # the model would sit in VRAM twice
                    log("Stopping RulZ's own llama-server: the Training Captioner's is running.")
                    _kill(mine)
                return theirs, "captioner"
            check()

            plan = build_plan(cap, first_port)
            launches = [(port, over, build_command(cap, port, args, disable_thinking))
                        for port, over, args in plan]
            key = tuple((port, tuple(sorted(over.items())), tuple(cmd)) for port, over, cmd in launches)
            mine, why_not = self._reusable(key)
            if mine:
                return mine, "own"
            check()
            stale = self._take_all()
            if stale:
                log(f"Restarting RulZ's llama-server: {why_not}.")
                _kill(stale)
            check()
            self._validate(cap, [port for port, _, _ in launches], log)
            return self._start(launches, key, slots, gen, log, check), "own"
        finally:
            self._ensure_lock.release()

    def stop(self) -> None:
        """Kill the servers this object started and wait (up to 10 s) until they have exited."""
        _kill(self._take_all(cancel=True))

    # ------------------------------------------------------------------ helpers
    def _take_all(self, cancel: bool = False) -> list[_Server]:
        """Hand over every server this object started; whoever takes a server kills it, so no two
        threads ever close the same job handle. `cancel` also makes a running ensure() give up."""
        with self._lock:
            if cancel:
                self._gen += 1
            servers, self._servers, self._endpoints, self._key = self._servers, [], [], None
        return servers

    def _captioner_endpoints(self, cap: dict, slots: int) -> list[dict]:
        with self._lock:
            mine = {s.port for s in self._servers if s.alive()}  # first port may equal the Captioner's
        found = []
        for port, over, _ in build_plan(cap, cap["base_port"]):
            if port in mine or not 0 < port < 65536:
                continue
            # The Captioner's servers listen on 127.0.0.1, so a free port there means none is running:
            # skip the HTTP check and Windows' wait for a refused connection. When one holds the port,
            # connecting gets the full limit: a false "no" would load the model a second time.
            if port_in_use(port) and healthy(port, connect_timeout=2.0):
                found.append(_endpoint(port, over, slots))
        return found

    def _reusable(self, key) -> tuple[list[dict], str]:
        """(this object's endpoints, "") when its servers match `key`, are alive and answer, else
        ([], why not)."""
        with self._lock:
            servers, endpoints, same = list(self._servers), list(self._endpoints), self._key == key
        if not servers:
            return [], ""
        if not same or not endpoints:
            return [], "the Training Captioner's settings changed"
        if not all(s.alive() for s in servers):
            return [], "it stopped"
        # a generous limit: a server busy with a job still answers, and a false "no" would kill it mid-job
        if not all(healthy(s.port, timeout=5.0, connect_timeout=5.0) for s in servers):
            return [], "it stopped answering"
        return endpoints, ""

    @staticmethod
    def _validate(cap: dict, ports: list[int], log: Callable[[str], None]):
        exe, model, mmproj = cap["llama_server"], cap["model"], cap["mmproj"]
        if not exe.strip():
            raise ServerError(f"llama-server.exe isn't set. Set its path in {SETTINGS_HINT}.")
        if not os.path.isfile(exe):
            raise ServerError(f"llama-server.exe not found:\n{exe}\nFix its path in {SETTINGS_HINT}.")
        if not model.strip():
            raise ServerError(f"No model (.gguf) is set. Set one in {SETTINGS_HINT}.")
        if not os.path.isfile(model):
            raise ServerError(f"Model (.gguf) not found:\n{model}\nFix its path in {SETTINGS_HINT}.")
        if mmproj and not os.path.isfile(mmproj):
            raise ServerError(f"Vision projector (mmproj) not found:\n{mmproj}\n"
                              f"Fix its path in {SETTINGS_HINT}.")
        if not mmproj:
            log("note: no vision projector (mmproj) is set; without it most models can't see the images "
                f"and every image fails. Set it in {SETTINGS_HINT}.")
        span = f"{ports[0]}" if len(ports) == 1 else f"{ports[0]}-{ports[-1]}"
        if not all(0 < p < 65536 for p in ports):
            raise ServerError(f"RulZ's llama-server would need port {span}, past the last port (65535). "
                              f"Lower {PORT_HINT}.")
        busy = [p for p in ports if port_in_use(p)]
        if busy:
            raise ServerError(
                f"Port {', '.join(map(str, busy))} {'is' if len(busy) == 1 else 'are'} already in use by "
                "another program (possibly a llama-server that is still running). RulZ starts its own "
                f"llama-server on port {span}: close that program, or change {PORT_HINT}.")

    def _start(self, launches, key, slots: int, gen: int, log: Callable[[str], None],
               check: Callable[[], None]) -> list[dict]:
        started: list[_Server] = []
        try:
            for port, over, cmd in launches:
                name = server_name(port, over)
                log(f"[{name}] launching: {subprocess.list2cmdline(cmd)}")
                check()
                with self._lock:  # stop() can't miss a server launched while it runs
                    if self._gen != gen:
                        raise Cancelled("Cancelled")
                    try:
                        p = subprocess.Popen(cmd, env=server_env(over), stdin=subprocess.DEVNULL,
                                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                             creationflags=CREATE_NO_WINDOW if os.name == "nt" else 0,
                                             cwd=os.path.dirname(cmd[0]) or None)
                    except OSError as e:
                        raise ServerError(f"llama-server couldn't be started ({e}). "
                                          f"Check its path in {SETTINGS_HINT}.") from e
                    srv = _Server(p, port, name, winjob.adopt(p.pid))  # so it dies with the app
                    started.append(srv)
                    self._servers.append(srv)
                if srv.job is None:
                    log(f"[{name}] note: couldn't tie llama-server to the app; if the app crashes, "
                        "end it in Task Manager")
                srv.pump = threading.Thread(target=_pump, args=(srv, log), daemon=True,
                                            name=f"llama-server {name} output")
                srv.pump.start()
            self._wait(started, log, check)
            with self._lock:
                if self._gen != gen:  # stopped after the last health check: don't report ready
                    raise Cancelled("Cancelled")
                self._endpoints = [_endpoint(port, over, slots) for port, over, _ in launches]
                self._key = key
                return list(self._endpoints)
        except BaseException:
            with self._lock:  # what stop() already took, it kills itself
                mine = [s for s in started if s in self._servers]
                self._servers = [s for s in self._servers if s not in mine]
                if not self._servers:
                    self._endpoints, self._key = [], None
            _kill(mine)
            raise

    @staticmethod
    def _wait(servers: list[_Server], log: Callable[[str], None], check: Callable[[], None]):
        deadline = time.monotonic() + START_TIMEOUT
        pending = list(servers)
        while True:
            for srv in list(pending):
                check()
                if not srv.alive():
                    check()  # killed by stop(), not a crash
                    if srv.pump is not None:
                        srv.pump.join(2.0)  # its last lines say why
                    tail = "\n".join(srv.tail[-ERROR_LINES:]) or "(no output)"
                    raise ServerError(
                        f"llama-server ({srv.name}) exited while loading the model (code "
                        f"{srv.popen.returncode}). Check the model, mmproj and GPU settings in "
                        f"{SETTINGS_HINT}.\n\nIts last output:\n{tail}")
                if healthy(srv.port, connect_timeout=LOADING_CONNECT_TIMEOUT):
                    pending.remove(srv)
                    log(f"[{srv.name}] ready")
            if not pending:
                return
            check()
            if time.monotonic() > deadline:
                raise ServerError(f"llama-server didn't finish loading the model within "
                                  f"{START_TIMEOUT / 60:.0f} minutes. Check the model and GPU settings "
                                  f"in {SETTINGS_HINT}, and the log for what it was doing.")
            time.sleep(POLL_INTERVAL)
