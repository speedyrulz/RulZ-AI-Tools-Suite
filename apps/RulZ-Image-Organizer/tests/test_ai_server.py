"""rulz.ai.server: starting, reusing and stopping llama-server, with a fake llama-server (no GPU, no model)."""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading
import time

import pytest

from rulz.ai import server

NO_WINDOW = server.CREATE_NO_WINDOW if os.name == "nt" else 0
HINT = "the Training Captioner's Settings ▸ Backend"

# Answers /health like llama-server: 503 while "loading", then 200. Started with llama-server's arguments.
FAKE_SERVER = r'''
import argparse, http.server, os, sys, time
ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--host", default="127.0.0.1")
ap.add_argument("--fake-delay", type=float, default=0.0)    # seconds before it listens
ap.add_argument("--fake-loading", type=float, default=0.0)  # seconds /health answers 503 once listening
ap.add_argument("--fake-exit", type=int)                    # print a load log, then exit with this code
a, _ = ap.parse_known_args()
print("fake llama-server " + " ".join(sys.argv[1:]), flush=True)
print("env", os.environ.get("CUDA_VISIBLE_DEVICES"), os.environ.get("CUDA_DEVICE_ORDER"), flush=True)
if a.fake_exit is not None:
    for i in range(20):
        print(f"load line {i}", flush=True)
    print("error: failed to allocate CUDA buffer", flush=True)
    sys.exit(a.fake_exit)
time.sleep(a.fake_delay)
ready = time.monotonic() + a.fake_loading


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"  # keep-alive, like llama-server: the client closes first

    def do_GET(self):
        code = 404 if self.path != "/health" else 200 if time.monotonic() >= ready else 503
        body = b'{"status": "ok"}' if code == 200 else b'{"error": "Loading model"}'
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class Server(http.server.ThreadingHTTPServer):
    allow_reuse_address = False  # SO_REUSEADDR on Windows would let another socket share the port
    daemon_threads = True


srv = Server((a.host, a.port), Handler)
print(f"listening on {a.port}", flush=True)
srv.serve_forever()
'''


def make_cap(**over):
    cap = {"llama_server": r"C:\llama\llama-server.exe", "model": r"C:\models\model.gguf",
           "mmproj": r"C:\models\mmproj.gguf", "gpu_mode": "parallel", "gpus": [], "tensor_split": "",
           "ctx_size": 4096, "ngl": 99, "slots_per_server": 2, "extra_args": "", "base_port": 8090,
           "thinking_budget": -1}
    cap.update(over)
    return cap


def free_ports(n):
    """n different ports nothing listens on (each is held while the next is picked)."""
    socks = []
    try:
        for _ in range(n):
            s = socket.socket()
            s.bind(("127.0.0.1", 0))
            socks.append(s)
        return [s.getsockname()[1] for s in socks]
    finally:
        for s in socks:
            s.close()


def free_run(n):
    """The first of n free ports in a row (a server per GPU takes first_port, first_port + 1, ...)."""
    for _ in range(50):
        first = free_ports(1)[0]
        if first + n <= 65536 and not any(server.port_in_use(first + i) for i in range(1, n)):
            return first
    pytest.skip("no free ports in a row")


def wait_for(cond, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(0.02)
    return bool(cond())


def can_connect(port):
    try:
        socket.create_connection(("127.0.0.1", port), timeout=0.5).close()
        return True
    except OSError:
        return False


class Fake:
    """Files the Captioner's settings point at, and a Popen that runs FAKE_SERVER in place of the
    'llama-server.exe' (a real process, so job objects, output and kills are the real thing)."""

    def __init__(self, tmp_path, monkeypatch):
        self.exe = tmp_path / "llama-server.exe"
        self.model = tmp_path / "model.gguf"
        self.mmproj = tmp_path / "mmproj.gguf"
        for f in (self.exe, self.model, self.mmproj):
            f.write_bytes(b"fake")
        self.script = tmp_path / "fake_llama_server.py"
        self.script.write_text(FAKE_SERVER, encoding="utf-8")
        self.launched: list[subprocess.Popen] = []  # every 'llama-server.exe' started
        self.logs: list[str] = []
        self.managers: list[server.LlamaServers] = []
        self.others: list[subprocess.Popen] = []
        fake, real = self, subprocess.Popen

        class Popen(real):
            def __init__(self, args, *a, **k):
                ours = isinstance(args, list) and bool(args) and args[0] == str(fake.exe)
                if ours:
                    args = [sys.executable, "-u", str(fake.script)] + args[1:]
                super().__init__(args, *a, **k)
                if ours:
                    fake.launched.append(self)
        monkeypatch.setattr(server.subprocess, "Popen", Popen)

    def cap(self, **over):
        cap = make_cap(llama_server=str(self.exe), model=str(self.model), mmproj=str(self.mmproj),
                       base_port=free_ports(1)[0])  # nothing there: the Captioner isn't running
        cap.update(over)
        return cap

    def log(self, line):
        self.logs.append(line)

    def manager(self):
        m = server.LlamaServers()
        self.managers.append(m)
        return m

    def run_direct(self, port, *args, wait=True):
        """A fake server started by someone else (the Training Captioner)."""
        p = subprocess.Popen([sys.executable, "-u", str(self.script), "--port", str(port), *args],
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             creationflags=NO_WINDOW)
        self.others.append(p)
        assert wait_for(lambda: server.healthy(port) if wait else can_connect(port), 20), "fake never listened"
        return p

    def close(self):
        for m in self.managers:
            m.stop()
        for p in self.launched + self.others:
            if p.poll() is None:
                p.kill()
                p.wait(10)


@pytest.fixture
def fake(tmp_path, monkeypatch):
    f = Fake(tmp_path, monkeypatch)
    yield f
    f.close()


# ------------------------------------------------------------------ plan and command line
def test_build_plan_without_gpus_is_one_server():
    assert server.build_plan(make_cap(gpus=[]), 9000) == [(9000, {}, [])]


def test_build_plan_one_server_per_gpu():
    assert server.build_plan(make_cap(gpus=[0, 2]), 9000) == [
        (9000, {"CUDA_VISIBLE_DEVICES": "0"}, []), (9001, {"CUDA_VISIBLE_DEVICES": "2"}, [])]


def test_build_plan_split_is_one_server_over_the_gpus():
    assert server.build_plan(make_cap(gpus=[0, 1], gpu_mode="split", tensor_split=" 3,1 "), 9000) == [
        (9000, {"CUDA_VISIBLE_DEVICES": "0,1"}, ["--split-mode", "layer", "--tensor-split", "3,1"])]
    assert server.build_plan(make_cap(gpus=[0, 1], gpu_mode="split"), 9000) == [
        (9000, {"CUDA_VISIBLE_DEVICES": "0,1"}, ["--split-mode", "layer"])]
    # 'split' over a single GPU is just that GPU
    assert server.build_plan(make_cap(gpus=[1], gpu_mode="split", tensor_split="1"), 9000) == [
        (9000, {"CUDA_VISIBLE_DEVICES": "1"}, [])]


def test_build_command_matches_the_captioner():
    cap = make_cap(slots_per_server=3, thinking_budget=512,
                   extra_args=r'--flash-attn on --alias "my model" --lora "C:\My Loras\a.gguf"')
    assert server.build_command(cap, 9000, ["--split-mode", "layer"], disable_thinking=False) == [
        r"C:\llama\llama-server.exe", "-m", r"C:\models\model.gguf", "-ngl", "99", "-c", "12288",
        "-np", "3", "--host", "127.0.0.1", "--port", "9000", "--mmproj", r"C:\models\mmproj.gguf",
        "--reasoning-budget", "512", "--split-mode", "layer",
        "--flash-attn", "on", "--alias", "my model", "--lora", r"C:\My Loras\a.gguf"]
    cmd = server.build_command(make_cap(mmproj="", slots_per_server=0), 9001, [], disable_thinking=True)
    assert cmd == [r"C:\llama\llama-server.exe", "-m", r"C:\models\model.gguf", "-ngl", "99", "-c", "4096",
                   "-np", "1", "--host", "127.0.0.1", "--port", "9001", "--reasoning", "off"]


def test_unreadable_extra_args_say_where_to_fix_them():
    with pytest.raises(server.ServerError, match="extra llama-server arguments") as info:
        server.build_command(make_cap(extra_args='--alias "unclosed'), 9000, [], False)
    assert HINT in str(info.value)


def test_server_env_numbers_gpus_like_nvidia_smi(monkeypatch):
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "FASTEST_FIRST")
    env = server.server_env({"CUDA_VISIBLE_DEVICES": "1"})
    assert env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID" and env["CUDA_VISIBLE_DEVICES"] == "1"


# ------------------------------------------------------------------ health check
def test_healthy_ignores_proxies_and_waits_for_the_model(fake, monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid:3128")
    monkeypatch.setenv("http_proxy", "http://proxy.invalid:3128")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    up, loading, nothing = free_ports(3)
    fake.run_direct(up)
    assert server.healthy(up)
    fake.run_direct(loading, "--fake-loading", "60", wait=False)
    assert not server.healthy(loading)  # 503: still loading
    t = time.monotonic()
    assert not server.healthy(nothing)
    assert time.monotonic() - t < 1.5  # Windows takes 2 s to report the refused connection
    assert not server.healthy(70000)


# ------------------------------------------------------------------ what ensure() refuses
@pytest.mark.parametrize("key,value,want", [
    ("llama_server", "", "llama-server.exe isn't set"),
    ("llama_server", "missing", "llama-server.exe not found"),
    ("model", "", "No model (.gguf) is set"),
    ("model", "missing", "Model (.gguf) not found"),
    ("mmproj", "missing", "Vision projector (mmproj) not found"),
])
def test_missing_files_say_what_to_fix(fake, tmp_path, key, value, want):
    if value == "missing":
        value = str(tmp_path / "nowhere" / "file")
    with pytest.raises(server.ServerError) as info:
        fake.manager().ensure(fake.cap(**{key: value}), free_ports(1)[0], False, fake.log, lambda: False)
    assert want in str(info.value) and HINT in str(info.value)
    assert fake.launched == []


def test_busy_port_names_it_and_the_setting(fake):
    first = free_run(2)
    with socket.socket() as holder:
        holder.bind(("127.0.0.1", first + 1))  # the second GPU's port
        with pytest.raises(server.ServerError) as info:
            fake.manager().ensure(fake.cap(gpus=[0, 1]), first, False, fake.log, lambda: False)
    msg = str(info.value)
    assert f"Port {first + 1} is already in use" in msg and f"{first}-{first + 1}" in msg
    assert "'First port for RulZ's own llama-server' in Advanced AI settings" in msg
    assert fake.launched == []


# ------------------------------------------------------------------ the Captioner's servers
def test_uses_the_captioners_running_server_and_starts_nothing(fake):
    base, first = free_ports(2)
    theirs = fake.run_direct(base)
    m = fake.manager()
    eps, source = m.ensure(fake.cap(base_port=base, gpus=[0, 1]), first, False, fake.log, lambda: False)
    assert source == "captioner"
    # only the one that answers (the second GPU's server isn't running)
    assert eps == [{"url": f"http://127.0.0.1:{base}/v1", "model": "default", "slots": 2,
                    "name": f":{base} GPU 0"}]
    assert fake.launched == [] and not m.running()
    m.stop()
    assert theirs.poll() is None  # not ours to stop


# ------------------------------------------------------------------ RulZ's own servers
def test_starts_own_servers_reuses_restarts_and_stops_them(fake):
    first = free_run(2)
    m = fake.manager()
    cap = fake.cap(gpus=[0, 1], extra_args="--fake-loading 0.5")
    t = time.monotonic()
    eps, source = m.ensure(cap, first, True, fake.log, lambda: False)
    assert time.monotonic() - t >= 0.5  # waited out the 503s
    assert source == "own"
    assert eps == [
        {"url": f"http://127.0.0.1:{first}/v1", "model": "default", "slots": 2, "name": f":{first} GPU 0"},
        {"url": f"http://127.0.0.1:{first + 1}/v1", "model": "default", "slots": 2,
         "name": f":{first + 1} GPU 1"}]
    assert len(fake.launched) == 2 and m.running()
    assert server.healthy(first) and server.healthy(first + 1)
    # each server's output reaches the log under its port, with its GPU and the Captioner's command line
    assert wait_for(lambda: f"[:{first + 1}] env 1 PCI_BUS_ID" in fake.logs, 5)
    started = [line for line in fake.logs if line.startswith(f"[:{first}] fake llama-server ")]
    assert started and f"-np 2 --host 127.0.0.1 --port {first} --mmproj" in started[0]
    assert "--reasoning off" in started[0] and started[0].endswith("--fake-loading 0.5")
    assert f"[:{first} GPU 0] ready" in fake.logs and f"[:{first + 1} GPU 1] ready" in fake.logs
    if os.name == "nt":
        assert not any("couldn't tie" in line for line in fake.logs)

    # asked again: the same servers
    assert m.ensure(cap, first, True, fake.log, lambda: False) == (eps, "own")
    assert len(fake.launched) == 2

    # thinking turned on: another command line, so new servers on the same ports
    old = list(fake.launched)
    assert m.ensure(cap, first, False, fake.log, lambda: False) == (eps, "own")
    assert len(fake.launched) == 4 and all(p.poll() is not None for p in old)
    assert any("Restarting RulZ's llama-server: the Training Captioner's settings changed" in line
               for line in fake.logs)

    m.stop()
    assert not m.running()
    assert all(p.poll() is not None for p in fake.launched)
    # A process that just ended can hold its port a moment longer on Windows; the app allows the same settle.
    assert not server.port_in_use(first, settle=5.0) and not server.port_in_use(first + 1, settle=5.0)


def test_a_server_that_died_is_started_again(fake):
    first = free_ports(1)[0]
    m = fake.manager()
    cap = fake.cap(mmproj="")
    m.ensure(cap, first, False, fake.log, lambda: False)
    assert any(line.startswith("note: no vision projector (mmproj) is set") for line in fake.logs)
    fake.launched[0].kill()
    fake.launched[0].wait(10)
    assert not m.running()
    eps, source = m.ensure(cap, first, False, fake.log, lambda: False)
    assert source == "own" and len(fake.launched) == 2 and m.running()
    assert any(line == "Restarting RulZ's llama-server: it stopped." for line in fake.logs)


def test_exit_while_loading_shows_its_last_output(fake):
    m = fake.manager()
    with pytest.raises(server.ServerError) as info:
        m.ensure(fake.cap(extra_args="--fake-exit 3"), free_ports(1)[0], False, fake.log, lambda: False)
    msg = str(info.value)
    assert "exited while loading the model (code 3)" in msg and HINT in msg
    lines = msg.splitlines()
    assert lines[-1] == "error: failed to allocate CUDA buffer"
    assert "load line 9" in lines and "load line 8" not in lines  # the last 12 lines
    assert not m.running() and fake.launched[0].poll() == 3


def test_cancel_while_loading_kills_the_server(fake):
    m = fake.manager()
    t = time.monotonic()
    with pytest.raises(server.Cancelled):
        m.ensure(fake.cap(extra_args="--fake-delay 30"), free_ports(1)[0], False, fake.log,
                 lambda: bool(fake.launched) and time.monotonic() - t > 0.3)
    assert time.monotonic() - t < 5
    assert len(fake.launched) == 1 and fake.launched[0].poll() is not None
    assert not m.running()


def test_stop_while_loading_cancels_the_start(fake):
    m = fake.manager()
    got = {}

    def start():
        try:
            got["result"] = m.ensure(fake.cap(extra_args="--fake-delay 30"), free_ports(1)[0], False, fake.log,
                                     lambda: False)
        except BaseException as e:  # noqa: BLE001
            got["error"] = e
    th = threading.Thread(target=start, daemon=True)
    th.start()
    assert wait_for(lambda: fake.launched, 10)
    time.sleep(0.3)  # into the wait for /health
    t = time.monotonic()
    m.stop()
    th.join(5)
    assert not th.is_alive() and time.monotonic() - t < 3
    assert isinstance(got.get("error"), server.Cancelled)
    assert fake.launched[0].poll() is not None and not m.running()
