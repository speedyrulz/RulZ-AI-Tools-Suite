"""Render on this PC: a helper that renders a vast.ai run's samples and likeness probes on a GPU of this computer.

The trainer on the instance writes a "round" (a LoRA state and the pictures to render) into the run's remote_render
folder; this helper, connected out over ssh, claims the round, renders it with the trainer's own Renderer
(probeworker.py: the same code, so the pictures are the ones the instance would have made), and uploads the
pictures. The instance scores them and acts as it does for probe_async, up to one probe interval late. Training
never waits for this PC: the trainer takes a round back when its deadline passes or the heartbeat goes stale.
The rules are in docs/remote-render-contract.md (sections 4 and 5).

    python -m rulztrainer.renderhelper --instance 123456 --device cuda:1 [--sample-model PATH]
        [--sample-model-quant Q] [--models-dir DIR] [--status-file PATH]
    python -m rulztrainer.renderhelper --list-gpus      # JSON, loads no model

Two ssh connections, each used from one thread only: the main loop's (runs, heartbeat, every few seconds) and the
worker's (preparing, rounds: big transfers). One connection would let a 300 MB LoRA download hold the heartbeat back
until the trainer calls the helper gone. Windows OpenSSH has no ControlMaster, so each is its own ssh process; the
bridge (scripts/vast/rulz_render_bridge.py) is piped to the instance, nothing is installed there.

The helper's own folder (next to its status file) holds the text embeds and the rounds it works on: never this PC's
real text-embed cache, which a prepared copy of someone else's prompts must not be mixed into."""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("rulztrainer")

APP = Path(__file__).resolve().parent.parent
BRIDGE_FILE = APP / "scripts" / "vast" / "rulz_render_bridge.py"
STATE_DIR = Path(os.environ.get("RULZ_VAST_HOME") or Path.home() / ".rulz_vast") / "render_helper"  # (as rulz_remote)
# python reads a script to EOF, so this reads one length-prefixed script, runs it, and leaves stdin to its frames
BOOTSTRAP = ('import sys,struct;s=sys.stdin.buffer;'
             'exec(compile(s.read(struct.unpack(">I",s.read(4))[0]),"bridge","exec"))')
MAGIC = b"RZRB"
HEARTBEAT_EVERY = 5.0  # seconds; the trainer calls a helper gone after 30
TICK = 1.0  # seconds between looks at the instance
RETRY_PREPARE = 300.0  # seconds before a failed preparation is tried again (a model may have been put in place)
BACKOFF = (2.0, 5.0, 10.0, 20.0, 30.0)  # seconds between connection tries
VERSION_MISMATCH = "versions differ: update RulzTrainer on the instance or this PC"
STATES = ("connecting", "preparing", "ready", "rendering", "not_ready", "idle", "stopped", "error")


class BridgeError(Exception):
    """The bridge answered {ok: false}: its message says why."""


class Disconnected(Exception):
    """The ssh connection is gone (or spoke nonsense): the caller reconnects."""


class NotReady(Exception):
    """This helper cannot render this run; the message is written to not_ready.json."""

    def __init__(self, reason: str, retry: bool = True, delay: float | None = None):
        super().__init__(reason)
        self.reason, self.retry, self.delay = reason, retry, delay


# ---------------------------------------------------------------- the bridge's frames


class Bridge:
    """One connection to the instance's bridge, over a process's stdin and stdout. One thread at a time per bridge."""

    def __init__(self, proc, describe: str = ""):
        self.proc, self.describe = proc, describe
        self.lock = threading.Lock()
        self.errors: deque[str] = deque(maxlen=12)  # ssh's own words, for the message when it drops
        if getattr(proc, "stderr", None) is not None:
            threading.Thread(target=self._drain, daemon=True).start()

    def _drain(self) -> None:
        try:
            for line in iter(self.proc.stderr.readline, b""):
                self.errors.append(line.decode("utf-8", "replace").strip())
        except (OSError, ValueError):
            pass

    def _read(self, size: int) -> bytes:
        chunks = []
        while size:
            chunk = self.proc.stdout.read(size)
            if not chunk:
                tail = "; ".join(e for e in self.errors if e)
                raise Disconnected("the connection ended" + (f" ({tail})" if tail else ""))
            chunks.append(chunk)
            size -= len(chunk)
        return b"".join(chunks)

    def call(self, op: str, body: bytes = b"", **fields) -> tuple[dict, bytes]:
        """One request and its reply. BridgeError when the bridge refused, Disconnected when the line is gone."""
        header = json.dumps({"op": op, **fields}).encode("utf-8")
        with self.lock:
            try:
                self.proc.stdin.write(MAGIC + struct.pack(">I", len(header)) + header + struct.pack(">Q", len(body)) + body)
                self.proc.stdin.flush()
            except (OSError, ValueError) as e:  # (ValueError: write to a closed file)
                try:
                    self.proc.wait(timeout=1.5)  # ssh's own words (a refused key, no route) arrive just after it dies
                except (subprocess.TimeoutExpired, OSError):
                    pass
                time.sleep(0.2)
                tail = "; ".join(x for x in self.errors if x)
                raise Disconnected(f"could not send: {e}" + (f" ({tail})" if tail else "")) from e
            if self._read(4) != MAGIC:
                raise Disconnected("the instance sent something that is not the bridge's reply")
            reply = json.loads(self._read(struct.unpack(">I", self._read(4))[0]).decode("utf-8"))
            data = self._read(struct.unpack(">Q", self._read(8))[0])
        if not reply.get("ok"):
            raise BridgeError(str(reply.get("error") or f"{op} failed"))
        return reply, data

    def get(self, path: str) -> bytes:
        return self.call("get", path=path)[1]

    def get_json(self, path: str):
        return json.loads(self.get(path).decode("utf-8"))

    def put(self, path: str, data: bytes) -> None:
        self.call("put", data, path=path)

    def write_json(self, path: str, data) -> None:
        self.call("write_json", path=path, data=data)

    def rename(self, src: str, dst: str) -> bool:
        return bool(self.call("rename", src=src, dst=dst)[0].get("renamed"))

    def close(self) -> None:
        for f in (self.proc.stdin,):
            try:
                f.close()
            except (OSError, ValueError):
                pass
        try:
            if self.proc.poll() is None:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
        except OSError:
            pass


class SshConnector:
    """Opens bridges to one instance through scripts/vast/rulz_remote.py's routes: the saved one first, then the
    instance looked up again (its direct port, then vast's proxy)."""

    def __init__(self, instance: int | None, key: str | None = None):
        sys.path.insert(0, str(BRIDGE_FILE.parent))
        import rulz_remote  # noqa: PLC0415 - a script beside the bridge, not a package

        self.remote, self.instance, self.key = rulz_remote, instance, key
        self.conn = None
        self.lock = threading.Lock()  # the main loop and the worker both connect

    def _spawn(self, conn: dict) -> Bridge:
        shell = f"cd {self.remote.REMOTE_DIR} && {self.remote.PYTHON} -c {_quote(BOOTSTRAP)}"
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        proc = subprocess.Popen(self.remote.ssh_command(conn, self.key) + [shell], stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=flags)
        bridge = Bridge(proc, f"{conn.get('route')} {conn['host']}:{conn['port']}")
        script = BRIDGE_FILE.read_bytes()
        try:
            proc.stdin.write(struct.pack(">I", len(script)) + script)
            proc.stdin.flush()
            from . import remoterender

            bridge.call("hello", protocol=remoterender.PROTOCOL)
        except Exception:
            bridge.close()
            raise
        return bridge

    def __call__(self) -> Bridge:
        """A new bridge, or Disconnected (whatever went wrong: ssh, vast's CLI, its output)."""
        errors = []
        with self.lock:
            if self.conn is None:
                try:
                    self.conn = self.remote.saved_connection(self.instance)
                except Exception:  # noqa: BLE001 - a garbled saved file
                    self.conn = None
            for attempt in range(2):
                fresh = self.conn is None
                try:
                    if fresh:
                        self.conn = self.remote.find_connection(self.instance, self.key, remember=False)
                    return self._spawn(self.conn)
                except Exception as e:  # noqa: BLE001 - e.g. a ValueError from garbled vastai output
                    errors.append(str(e) if isinstance(e, (Disconnected, BridgeError)) else f"{type(e).__name__}: {e}")
                    self.conn = None  # the next try asks vast again
                    if fresh:
                        break
        raise Disconnected("; ".join(errors))


def _quote(text: str) -> str:
    import shlex

    return shlex.quote(text)


# ---------------------------------------------------------------- the status file


class Status:
    """The status file (contract section 5): rewritten on each change and on every touch, so a quiet helper shows."""

    def __init__(self, path: Path | None, instance, gpu: str):
        self.path, self.lock, self.closed = Path(path) if path else None, threading.Lock(), False
        self.data = {"state": "connecting", "instance": instance, "job": "", "gpu": gpu, "message": "", "rounds_done": 0,
                     "last_round": "", "last_seconds": 0.0, "behind": 0, "time": time.time(), "pid": os.getpid()}

    def update(self, **kw) -> None:
        with self.lock:
            if self.closed:
                return
            if "state" in kw and kw["state"] not in STATES:
                raise ValueError(kw["state"])
            self.data.update(kw)
            self._write()

    def touch(self) -> None:
        self.update()

    def close(self, **kw) -> None:
        """The last write ("stopped"); nothing writes after it."""
        self.update(**kw)
        with self.lock:
            self.closed = True

    def _write(self) -> None:
        self.data["time"] = time.time()
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(self.path.name + ".tmp")
            tmp.write_text(json.dumps(self.data), encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError:  # a reader has it open just now: the next touch writes it
            pass


# ---------------------------------------------------------------- what needs torch (patched in the tests)


def list_gpus() -> list[dict]:
    """[{index, name, total_gb, free_gb}] without creating a CUDA context on any card: names and sizes from the device
    properties, free memory from nvidia-smi (matched by PCI bus) when it is there."""
    import torch

    free: dict[str, float] = {}
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=pci.bus_id,memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout
        for line in out.splitlines():
            bus, _, mib = line.partition(",")
            free[bus.strip().lower()[-12:]] = float(mib) / 1024
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    gpus = []
    for i in range(torch.cuda.device_count() if torch.cuda.is_available() else 0):
        props = torch.cuda.get_device_properties(i)
        bus = getattr(props, "pci_bus_id", None)
        key = ("%04x:%02x:%02x.0" % (getattr(props, "pci_domain_id", 0), props.pci_bus_id, props.pci_device_id)
               if isinstance(bus, int) else "")
        gpus.append({"index": i, "name": props.name, "total_gb": round(props.total_memory / 1024 ** 3, 1),
                     "free_gb": round(free.get(key[-12:], 0.0), 1)})
    return gpus


def remote():
    """rulztrainer.remoterender: written with the trainer's side; the one place its names come in."""
    from . import remoterender

    return remoterender


def local_embed_path(cfg_dict: dict, prompt: str) -> Path:
    """Where the Renderer looks for a prompt's text embeds under this local config."""
    from .config import TrainConfig
    from .dataset import text_cache_path

    return text_cache_path(TrainConfig.from_dict(cfg_dict), prompt)


def ensure_hf_file(repo: str, file: str):
    """A Comfy-Org / Hugging Face file from this PC's Hugging Face cache, downloaded when it is not there."""
    from .downloads import ensure_hf_file as ensure

    return ensure(repo, file, file.rsplit("/", 1)[-1], None)


def cached_hf_file(repo: str, file: str):
    """The same file if it is already cached, else None (no download)."""
    from huggingface_hub import hf_hub_download

    try:
        return Path(hf_hub_download(repo, file, local_files_only=True))
    except Exception:  # noqa: BLE001 - not cached
        return None


def load_renderer(cfg_dict: dict, device: str):
    """(Renderer, {gpu, vram_gb}) on this PC's GPU; raises what the load raises (OOM included)."""
    import torch

    from .archs import get_arch
    from .config import TrainConfig
    from .probeworker import Renderer

    cfg = TrainConfig.from_dict({**cfg_dict, "device": device})
    model_dir = get_arch(cfg).resolve_for_config(cfg)  # as the trainer does before its own load
    torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(torch.device(device))
    return Renderer(cfg_dict, str(model_dir) if model_dir else None, device), {
        "gpu": props.name, "vram_gb": round(props.total_memory / 1024 ** 3, 1)}


def release_gpu() -> None:
    """Give the GPU's memory back. The caller has already dropped its reference to the Renderer: collecting while it
    still holds one frees nothing."""
    import gc

    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - nothing to free without torch
        pass


# ---------------------------------------------------------------- the helper


@dataclass
class Options:
    instance: str
    device: str = "cuda:0"
    sample_model: str = ""
    sample_quant: str = ""
    models_dir: str = ""
    status_file: Path | None = None
    key: str | None = None
    work_dir: Path | None = None


def find_local_file(name: str, size: int, folders: list[Path]) -> Path | None:
    """A file called `name` of exactly `size` bytes under one of the folders: a model that is on this PC under the
    name the instance knows it by (a file of the same name and another size is another model)."""
    for folder in folders:
        if not folder.is_dir():
            continue
        for here, _, names in os.walk(folder):  # (not rglob: a name may hold glob characters)
            if name in names:
                found = Path(here) / name
                try:
                    if found.stat().st_size == size:
                        return found
                except OSError:
                    continue
    return None


class Helper:
    """The state machine. tick() is one look at the instance (the main loop calls it every second); work (preparing
    for a run, rendering a round) runs in one worker thread at a time, so ticks and heartbeats go on meanwhile."""

    def __init__(self, opts: Options, connect, status: Status | None = None):
        self.o, self.connect = opts, connect
        self.host = socket.gethostname()
        self.status = status or Status(opts.status_file, opts.instance, opts.device)
        self.work_dir = Path(opts.work_dir) if opts.work_dir else (
            (Path(opts.status_file).parent if opts.status_file else STATE_DIR) / f"{opts.instance}_work")
        self.stop = threading.Event()
        self.hb: Bridge | None = None  # the main loop's connection
        self.wb: Bridge | None = None  # the worker's
        self.worker: threading.Thread | None = None
        self.renderer = None
        self.token: str | None = None  # the run this helper is prepared (or preparing) for
        self.prepared = False
        self.kinds: list[str] = []
        self.gpu_name = opts.device
        self.next_prepare = 0.0
        self.not_ready_reason = ""
        self.rounds_done = 0
        self.tries = 0
        self.next_connect = 0.0
        self.wtries, self.wnext = 0, 0.0  # the worker's own backoff
        self.last_beat = 0.0
        self.job = ""

    # ------------------------------------------------------------ the main loop's side
    def tick(self, now: float | None = None) -> None:
        """One look: connect if needed, find the running run, heartbeat, start work. Never raises for a dropped line."""
        now = time.time() if now is None else now
        if self.hb is None:
            if now < self.next_connect:
                self.status.touch()
                return
            try:
                self.hb = self.connect()
            except Disconnected as e:
                wait = BACKOFF[min(self.tries, len(BACKOFF) - 1)]
                self.tries += 1
                self.next_connect = now + wait
                self.status.update(state="connecting", message=f"Cannot reach instance {self.o.instance}: {e}. "
                                   f"Trying again in {wait:.0f} s")
                log.warning("Cannot reach instance %s: %s (again in %.0f s)", self.o.instance, e, wait)
                return
            log.info("Connected to instance %s", self.o.instance)
        try:
            self._look(now)
        except Disconnected as e:
            log.warning("Lost the connection to instance %s: %s", self.o.instance, e)
            self._drop_hb()
            self.status.update(state="connecting", message=f"The connection dropped: {e}")
        except BridgeError as e:
            log.warning("The instance refused: %s", e)
            self.status.update(state="error", message=str(e))

    def _drop_hb(self) -> None:
        if self.hb is not None:
            self.hb.close()
        self.hb = None
        self.next_connect = 0.0

    def _look(self, now: float) -> None:
        runs = self.hb.call("runs")[0].get("runs") or []
        self.tries = 0  # a call went through: only now is the connection good
        entry = runs[0] if runs else None
        if entry is None or not entry.get("run", {}).get("run_token"):
            if self.token is not None and not self._busy():
                self._reset(None)
            if not self._busy():
                self.status.update(state="idle", job="", behind=0,
                                   message="No run is rendering remotely on this instance yet.")
            else:
                self.status.touch()
            return
        run = entry["run"]
        job, token = entry["job"], run["run_token"]
        if token != self.token:
            if self._busy():  # still working for the old run: its end comes first
                self.status.touch()
                return
            self._reset(token)
        self.job = job
        if now - self.last_beat >= HEARTBEAT_EVERY or self.last_beat == 0.0:
            self.hb.call("heartbeat", job=job, run_token=token, host=self.host, gpu=self.gpu_name)
            self.last_beat = now
        mine = [r for r in entry.get("rounds") or [] if self._kind(r) in self.kinds]
        self.status.data["job"] = job
        if self._busy():
            self.status.update(behind=len(mine))
            return
        if not entry.get("has_prepare"):
            self.status.update(state="preparing", job=job, behind=0, message="Waiting for the run to prepare its files.")
        elif not self.prepared:
            if now < self.next_prepare:
                self.status.update(state="not_ready", job=job, message=self.not_ready_reason)
            else:
                self._start(self._prepare, entry)
        elif mine:
            self.status.update(job=job, behind=len(mine) - 1)
            self._start(self._render, entry, sorted(mine, key=lambda r: (self._kind(r) != "probe", r))[0])
        else:
            self.status.update(state="ready", job=job, behind=0, message="")

    @staticmethod
    def _kind(round_name: str) -> str:
        parts = round_name.split("_")
        return parts[1] if len(parts) > 2 else ""

    def _busy(self) -> bool:
        return self.worker is not None and self.worker.is_alive()

    def _start(self, fn, *args) -> None:
        self.worker = threading.Thread(target=self._guard, args=(fn, *args), daemon=True, name="render-worker")
        self.worker.start()

    def _guard(self, fn, *args) -> None:
        """The worker's body: nothing it raises may end the helper."""
        try:
            fn(*args)
        except Disconnected as e:
            log.warning("The worker's connection dropped: %s", e)
            self._drop_wb()
            self.status.update(state="connecting", message=f"The connection dropped: {e}")
        except Exception as e:  # noqa: BLE001
            log.error("Worker failed: %s", traceback.format_exc())
            self.status.update(state="error", message=f"{type(e).__name__}: {e}")

    def _drop_wb(self) -> None:
        if self.wb is not None:
            self.wb.close()
        self.wb = None

    def _reset(self, token: str | None) -> None:
        """A different run (or none): the old run's model goes, the next one is prepared from scratch."""
        self._free_renderer()
        self.token, self.prepared, self.kinds, self.next_prepare, self.not_ready_reason = token, False, [], 0.0, ""
        self.last_beat = 0.0
        self.rounds_done = 0
        self.status.update(rounds_done=0, last_round="", last_seconds=0.0)

    # ------------------------------------------------------------ the worker's side
    def _free_renderer(self) -> None:
        """Drop the model (the only reference), then collect."""
        had = self.renderer is not None
        self.renderer = None
        if had:
            release_gpu()

    def _bridge(self) -> Bridge:
        """The worker's connection, with the same backoff as the main loop's."""
        if self.wb is None:
            now = time.time()
            if now < self.wnext:
                raise Disconnected(f"waiting {self.wnext - now:.0f} s before trying again")
            try:
                self.wb = self.connect()
            except Disconnected:
                self.wnext = now + BACKOFF[min(self.wtries, len(BACKOFF) - 1)]
                self.wtries += 1
                raise
        return self.wb

    def _wok(self) -> None:
        self.wtries, self.wnext = 0, 0.0

    def _prepare(self, entry: dict) -> None:
        job, run = entry["job"], entry["run"]
        base = f"{job}/remote_render"
        self.status.update(state="preparing", job=job, message="Reading the run's settings.")
        t0 = time.time()
        try:
            try:
                ready = self._prepare_run(job, base, run)
                ready["load_seconds"] = round(time.time() - t0, 1)
                self._bridge().write_json(f"{base}/prepare/ready.json", ready)
            except (NotReady, Disconnected):
                raise
            except Exception as e:  # noqa: BLE001 - whatever it is, say so and wait: no retry every second
                log.error("Preparing failed: %s", traceback.format_exc())
                self._free_renderer()
                raise NotReady(f"Could not prepare: {type(e).__name__}: {e}"[:400], delay=60.0) from e
        except NotReady as e:
            self.prepared = False
            self.not_ready_reason = e.reason
            self.next_prepare = time.time() + (e.delay or RETRY_PREPARE) if e.retry else float("inf")
            log.warning("Not ready for %s: %s", job, e.reason)
            self.status.update(state="not_ready", message=e.reason)
            try:
                self._bridge().write_json(f"{base}/prepare/not_ready.json",
                                          {"run_token": run["run_token"], "reason": e.reason, "time": time.time()})
            except (BridgeError, Disconnected) as err:
                log.warning("Could not tell the instance why not: %s", err)
                if isinstance(err, Disconnected):
                    self._drop_wb()
            return
        self.prepared = True
        log.info("Ready to render %s on %s (%s) for kinds %s", job, self.gpu_name, self.host, ready["kinds"])
        self.status.update(state="ready", gpu=self.gpu_name, message="")

    def _prepare_run(self, job: str, base: str, run: dict) -> dict:
        shared = remote()
        w = self._bridge()
        prep = w.get_json(f"{base}/prepare/prepare.json")
        self._wok()
        if prep.get("run_token") != run["run_token"]:
            raise NotReady("the run's files are from an earlier run", retry=True)
        try:
            same = prep.get("protocol") == shared.PROTOCOL and prep.get("render_hash") == shared.render_hash(prep["family"])
        except Exception as e:  # noqa: BLE001 - e.g. a family this checkout has no folder for
            log.warning("render_hash failed: %s", e)
            same = False
        if not same:
            raise NotReady(VERSION_MISMATCH, retry=False)
        work = self.work_dir / str(run["run_token"])[:16]
        shutil.rmtree(self.work_dir, ignore_errors=True)  # an earlier run's embeds and rounds
        cfg = dict(prep["cfg"])
        override = bool(self.o.sample_model)
        # this PC's models folder: --models-dir, else the app's own models/ (a local run's default). Never the
        # instance's models_dir, which is a Linux path
        models = Path(self.o.models_dir).expanduser() if self.o.models_dir else APP / "models"
        for field, info in (prep.get("files") or {}).items():
            cfg[field] = self._resolve_file(field, info, models, override)
        cfg.update(cache_dir=str(work / "cache"), device=self.o.device, models_dir=str(models))
        if override:
            cfg["sample_model_path"] = self.o.sample_model
            cfg["sample_model_quant"] = self.o.sample_quant or "auto"
        self.status.update(message="Fetching the prompts' text embeds.")
        for prompt, rel in (prep.get("embeds") or {}).items():
            target = local_embed_path(cfg, prompt)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(w.get(f"{base}/prepare/{rel}"))
        self.status.update(message=f"Loading the model on {self.o.device}.")
        self._free_renderer()  # never a second model on top of the first
        failure = None
        try:
            self.renderer, info = load_renderer(cfg, self.o.device)
        except BaseException as e:  # noqa: BLE001 - an OOM comes as one of several kinds
            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                raise
            log.error("Renderer load failed: %s", traceback.format_exc())
            text = f"{type(e).__name__}: {e}".strip().splitlines()[0][:300]
            if "out of memory" in str(e).lower() or type(e).__name__ == "OutOfMemoryError":
                text = f"Out of GPU memory loading the model on {self.o.device} (no block swap here). {text}"
            failure = f"Could not load the model on {self.o.device}: {text}"
        if failure:  # (outside the except block: its traceback holds the half-loaded model)
            release_gpu()
            raise NotReady(failure)
        self.gpu_name = info.get("gpu") or self.o.device
        self.kinds = ["sample"] if override else ["sample", "probe"]
        return {"run_token": run["run_token"], "recipe_hash": prep.get("recipe_hash"), "render_hash": prep.get("render_hash"),
                "host": self.host, "gpu": self.gpu_name, "vram_gb": info.get("vram_gb"), "kinds": list(self.kinds),
                "time": time.time()}

    def _resolve_file(self, field: str, info: dict, models: Path, override: bool) -> str:
        """This PC's copy of one of the run's files. With provenance (repo and file: it was a Comfy-Org / Hugging Face
        download) from this PC's Hugging Face cache, downloading what is missing; without it, a file of the same name
        and size under the models folder. The text encoder is never downloaded here (the embeds replace it), and a
        local sampling model replaces the transformer. Anything else missing is listed in not_ready."""
        name, size = str(info.get("name") or Path(str(info.get("path") or "")).name), info.get("size")
        needless = field == "text_encoder_file" or (override and field in ("transformer_file", "sample_model_path"))
        found = None
        if info.get("repo") and info.get("file"):
            found = cached_hf_file(info["repo"], info["file"]) if needless else self._download(info)
        if found is None:
            found = find_local_file(name, int(size), [models]) if size is not None else None
        if found is not None and size is not None and found.stat().st_size != int(size):
            raise NotReady(f"{field}: {found} is {found.stat().st_size} bytes, the run's {name} is {size}")
        if found is None:
            if needless:
                return str(info.get("path") or name)  # not read here: the instance's own path stays in the config (a string: a Linux path)
            raise NotReady(f"Missing on this PC: {name} ({field}). Put it under {models}, or turn the helper off.")
        return str(found)

    def _download(self, info: dict):
        self.status.update(message=f"Fetching {info['file'].rsplit('/', 1)[-1]} from {info['repo']}.")
        try:
            return Path(ensure_hf_file(info["repo"], info["file"]))
        except Exception as e:  # noqa: BLE001 - offline, gated, disk
            raise NotReady(f"Could not fetch {info['file']} from {info['repo']}: {e}") from e

    def _render(self, entry: dict, name: str) -> None:
        job = entry["job"]
        base = f"{job}/remote_render"
        w = self._bridge()
        if not w.rename(f"{base}/rounds/{name}", f"{base}/claimed/{name}"):
            return  # the trainer took it back first
        claimed = f"{base}/claimed/{name}"
        self.status.update(state="rendering", job=job, message=f"Rendering {name}.")
        work = self.work_dir / str(self.token)[:16] / "rounds" / name
        try:
            started = time.time()
            rnd = w.get_json(f"{claimed}/round.json")
            if rnd.get("run_token") != self.token:
                raise RuntimeError("the round belongs to another run")
            payload = w.get(f"{claimed}/lora.safetensors")
            shared = remote()
            jobs = [shared.job_from_dict(d, work) for d in rnd["jobs"]]
            real = work.resolve()
            for j in jobs:
                if real not in Path(j.out_path).resolve().parents:
                    raise RuntimeError(f"a picture path leaves the work folder: {j.out_path}")
            for j in jobs:  # the render saves with Image.save, which makes no folders
                Path(j.out_path).parent.mkdir(parents=True, exist_ok=True)
            _, seconds = self.renderer.render(jobs, payload, rnd.get("negative_kind") or "probe")
            files = self._upload(w, claimed, jobs, work)
            w.write_json(f"{claimed}/done.json", {"seconds": round(seconds, 2), "files": files, "host": self.host,
                                                  "gpu": self.gpu_name})
            if w.rename(claimed, f"{base}/done/{name}"):
                self.rounds_done += 1
                log.info("Round %s: %d files in %.1f s (%.1f s with the transfers)", name, len(files), seconds,
                         time.time() - started)
                self.status.update(state="ready", message="", rounds_done=self.rounds_done, last_round=name,
                                   last_seconds=round(seconds, 2))
            else:
                log.warning("Round %s: the instance took it back first; its pictures are dropped", name)
                self.status.update(state="ready", message=f"{name} was taken back by the instance.")
        except Disconnected:
            raise
        except BridgeError as e:
            if "round is gone" not in str(e):
                raise
            # the bridge refuses writes into a round folder the instance has renamed away (its deadline passed)
            log.warning("Round %s: the instance took it back while it was sent; its pictures are dropped", name)
            self.status.update(state="ready", message=f"{name} was taken back by the instance.")
        except Exception as e:  # noqa: BLE001 - the instance is told, then renders the round itself
            log.error("Round %s failed: %s", name, traceback.format_exc())
            release_gpu()  # an OOM leaves cached blocks behind
            try:
                w.write_json(f"{claimed}/error.json", {"error": f"{type(e).__name__}: {e}",
                                                       "traceback": traceback.format_exc()})
                w.rename(claimed, f"{base}/failed/{name}")
            except (BridgeError, Disconnected) as err:
                log.warning("Could not report the failure: %s", err)
                if isinstance(err, Disconnected):
                    raise
            self.status.update(state="ready", message=f"{name} failed: {type(e).__name__}: {e}")
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def _upload(self, w: Bridge, claimed: str, jobs: list, work: Path) -> list[str]:
        """Each picture into claimed/<round>/out/ under its relative name; a clip's file first, then its still, as the
        trainer will move them."""
        sent = []
        for j in jobs:
            still = Path(j.out_path)
            files = ([still.with_suffix(".webp")] if getattr(j, "frames", 1) > 1 else []) + [still]
            for f in files:
                if not f.is_file():
                    if f == still:
                        raise FileNotFoundError(f"the render made no {f.name}")
                    continue
                rel = f.resolve().relative_to(work.resolve()).as_posix()
                w.put(f"{claimed}/out/{rel}", f.read_bytes())
                sent.append(rel)
        return sent

    # ------------------------------------------------------------ the whole run
    def run(self) -> None:
        """Ticks every second in a thread of its own, so the main thread stays free for signals (Windows does not
        interrupt a blocking pipe read for one), until stop is set."""
        loop = threading.Thread(target=self._loop, daemon=True, name="render-loop")
        loop.start()
        try:
            while not self.stop.wait(0.5):
                pass
        finally:
            self.shutdown(loop)

    def _loop(self) -> None:
        while not self.stop.is_set():
            try:
                self.tick()
            except Exception as e:  # noqa: BLE001 - a bug here must not end the helper
                log.error("Look failed: %s", traceback.format_exc())
                self.status.update(state="error", message=f"{type(e).__name__}: {e}")
            self.stop.wait(TICK)

    def shutdown(self, loop: threading.Thread | None = None) -> None:
        """Stop: close both connections (which unblocks a worker waiting on one), give a render a few seconds, and
        write "stopped". A render cannot be interrupted: the trainer's deadline takes its round back."""
        self.stop.set()
        self._drop_hb()
        self._drop_wb()
        for t in (loop, self.worker):
            if t is not None and t.is_alive():
                t.join(10)
        self.status.close(state="stopped", message="Stopped.")
        log.info("Stopped.")


# ---------------------------------------------------------------- command line


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="python -m rulztrainer.renderhelper", description=__doc__.split("\n\n")[0])
    p.add_argument("--instance", help="the vast.ai instance id")
    p.add_argument("--device", default="cuda:0", help="the GPU of this PC to render on, e.g. cuda:1")
    p.add_argument("--sample-model", default="", help="render samples (not probes) with this sampling model instead")
    p.add_argument("--sample-model-quant", default="", help="its quantization (auto, none, int8, int8_convrot, nvfp4)")
    p.add_argument("--models-dir", default="", help="where this PC keeps its models (default: this install's models/)")
    p.add_argument("--status-file", default="", help="default: ~/.rulz_vast/render_helper/<instance>.json")
    p.add_argument("--key", default=None, help="ssh key file")
    p.add_argument("--list-gpus", action="store_true", help="print this PC's GPUs as JSON and exit; loads no model")
    p.add_argument("--ignore-stdin", action="store_true", help="do not stop when stdin closes (the WebUI starts the "
                   "helper with a pipe and ends it by closing it)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.list_gpus:
        print(json.dumps(list_gpus()))
        return 0
    if not args.instance:
        print("--instance is required", file=sys.stderr)
        return 2
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S",
                        stream=sys.stdout)
    status_file = Path(args.status_file) if args.status_file else STATE_DIR / f"{args.instance}.json"
    opts = Options(instance=args.instance, device=args.device, sample_model=args.sample_model,
                   sample_quant=args.sample_model_quant, models_dir=args.models_dir, status_file=status_file,
                   key=args.key)
    helper = Helper(opts, SshConnector(int(args.instance) if str(args.instance).isdigit() else None, args.key))

    def stop(*_):
        helper.stop.set()

    for name in ("SIGTERM", "SIGINT", "SIGBREAK"):  # SIGBREAK: Ctrl+Break, Windows only
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), stop)
    if not args.ignore_stdin:
        def watch():  # the WebUI's end of our stdin closes when it dies: stop then
            try:
                while sys.stdin.buffer.read(4096):
                    pass
            except (OSError, ValueError):
                pass
            helper.stop.set()

        threading.Thread(target=watch, daemon=True).start()
    log.info("Render helper for instance %s on %s", args.instance, args.device)
    helper.run()
    if helper._busy():  # a render that would not stop: leave without waiting for the interpreter's teardown
        logging.shutdown()
        os._exit(0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
