"""Render on this PC with several GPUs: one worker process per GPU, a round's pictures dealt across them.

The helper (renderhelper.py) renders a round with one in-process Renderer when it has one GPU. With two or more (or with
the hidden `--workers process`) it uses this pool: one process per GPU, each loading the sampling model once, the
trainer's own probeworker process (`ProbeProcess`'s peer). A process and not a thread because the render path has
process-global state (attention backend, PyTorch's sdpa_kernel flags, quantization defaults, the LoRA round counter that
keys the DoRA cache, torch.cuda.set_device), and because a process gives all its VRAM back when it dies.

The rules (docs/remote-render-contract.md, section 5):
- Workers load one after another (a large transformer is read into system RAM once at a time). A GPU that fails to
  load is named and left out; the pool is usable when at least one loaded.
- A sample round's pictures are dealt in contiguous chunks sized by each GPU's measured speed. A probe round uses a
  FIXED map (equal chunks over the live GPUs in device order): pictures differ a little between card kinds, so the
  same picture should come from the same GPU every probe while the set of GPUs is unchanged.
- A worker that fails (an error, an exit, no answer within the cap) is dropped for the rest of the run and its chunk is
  dealt again, once, over the others after they answered their own. A second failure, or a re-deal that cannot finish
  before the round's deadline, fails the round (the instance then renders it itself).
- Nothing blocks: workers are polled, the helper's `stop` is honoured, and a worker is closed with a short deadline and
  then killed. On Windows every worker is put in a kill-on-close Job Object held by the helper, so none outlives it.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("rulztrainer")

CONNECT_TIMEOUT = 120.0  # seconds a worker process gets to start and connect back (a hung import never does)
LOAD_TIMEOUT = 900.0  # seconds a worker gets to load its model
POLL = 0.05  # seconds between looks at the workers
CLOSE_WAIT = 5.0  # seconds a worker gets to leave after "stop", then it is killed
HANG_FACTOR = 3.0  # a chunk may take this many times the seconds it is expected to
HANG_FLOOR = 120.0  # ... but never less than this, and never more than the round's deadline
DEFAULT_DEADLINE = 1800.0  # a round that does not say its deadline
EWMA = 0.5  # weight of the newest measurement in a GPU's seconds per unit

GPU_STATES = ("loading", "ready", "rendering", "dropped", "load_failed")


class Stopping(Exception):
    """The helper is stopping: the caller leaves (and the pool kills what is still running)."""


class RoundFailed(Exception):
    """The pool could not render the round; the message says why (it goes to the instance in error.json)."""


class WorkerFailed(Exception):
    """A worker failed in a way the pool already handled (kept for callers that want to name it)."""


# ---------------------------------------------------------------- what needs torch or a process (patched in the tests)


def device_info(device: str) -> dict:
    """{gpu, vram_gb} of one device from the device properties: no CUDA context is made in this (parent) process."""
    try:
        from .renderhelper import list_gpus

        index = int(str(device).rsplit(":", 1)[-1])
        for g in list_gpus():
            if g["index"] == index:
                return {"gpu": g["name"], "vram_gb": g["total_gb"]}
    except Exception:  # noqa: BLE001 - a name is only for show
        pass
    return {"gpu": str(device), "vram_gb": None}


def resolve_model_dir(cfg_dict: dict, device: str):
    """The model folder the workers load from, resolved once here as load_renderer does for the single GPU."""
    from .archs import get_arch
    from .config import TrainConfig

    cfg = TrainConfig.from_dict({**cfg_dict, "device": device})
    model_dir = get_arch(cfg).resolve_for_config(cfg)
    return str(model_dir) if model_dir else None


_JOB = None  # the Job Object handle: held for the life of the helper, closing it (at exit, however it exits) kills the workers


def adopt_process(proc) -> bool:
    """Windows: put the process in a Job Object that kills its members when the helper's last handle to it closes (the
    helper ended, crashed or was terminated). A no-op elsewhere and on any failure (never raises). True when adopted."""
    global _JOB
    import sys

    if sys.platform != "win32":
        return False
    try:
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)

        class Basic(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD)]

        class IoCounters(ctypes.Structure):
            _fields_ = [(n, ctypes.c_uint64) for n in ("ro", "wo", "oo", "rb", "wb", "ob")]

        class Extended(ctypes.Structure):
            _fields_ = [("Basic", Basic), ("Io", IoCounters), ("ProcessMemoryLimit", ctypes.c_size_t),
                        ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                        ("PeakJobMemoryUsed", ctypes.c_size_t)]

        k32.CreateJobObjectW.restype = wintypes.HANDLE
        k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        if _JOB is None:
            job = k32.CreateJobObjectW(None, None)
            if not job:
                return False
            info = Extended()
            info.Basic.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if not k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):
                k32.CloseHandle(job)
                return False
            _JOB = job
        return bool(k32.AssignProcessToJobObject(_JOB, wintypes.HANDLE(int(proc._handle))))
    except Exception as e:  # noqa: BLE001 - no job object is a lost safety net, not an error
        log.debug("Could not put the worker in a job object: %s", e)
        return False


_WORKER_CLASS = None


def real_worker_class():
    """The worker class over probeworker.ProbeProcess (imported here: probeworker imports torch, and the helper must not
    until it has a GPU job). It sends the round's LoRA bytes as they are (probeworker.py is part of the render hash, so
    it is not changed) and is read without ever blocking."""
    global _WORKER_CLASS
    if _WORKER_CLASS is not None:
        return _WORKER_CLASS
    from .probeworker import Pending, ProbeProcess, ProbeProcessError

    class PoolWorker(ProbeProcess):
        def __init__(self, cfg: dict, model_dir, device: str, log_path: Path, stop=None, on_start=None,
                     connect_timeout: float | None = None):
            """ProbeProcess.__init__'s steps (same Listener, authkey, key on stdin then closed), with what a pool needs:
            the process is in the Job Object right after Popen, `on_start(self)` runs then (so the pool can kill it while
            it connects), and the wait for its connection honours `stop` and times out."""
            import os
            import subprocess
            import sys
            import threading
            from multiprocessing.connection import Listener

            self.device, self.probes = device, True
            self.queue: list = []
            self._next_id = 1
            self.waited = self.rendered = self.analysed = 0.0
            self.count = self.analysis_count = 0
            self.analysis_wanted, self.analysis_error, self.error, self.free_since = False, None, None, None
            self.done: dict = {}
            self.memory = None
            self.conn = None
            self.log_path = log_path
            key = os.urandom(32)
            listener = Listener(("127.0.0.1", 0), authkey=key)
            root = str(Path(__file__).resolve().parent.parent)
            env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in (root, os.environ.get("PYTHONPATH")) if p)}
            self.log_file = open(log_path, "ab")
            try:
                self.proc = subprocess.Popen([sys.executable, "-m", "rulztrainer.probeworker",
                                              str(listener.address[1])], stdin=subprocess.PIPE, stdout=self.log_file,
                                             stderr=subprocess.STDOUT, env=env, cwd=root)
            except BaseException:
                listener.close()
                self.log_file.close()
                raise
            adopt_process(self.proc)
            if on_start:
                on_start(self)
            try:
                self.proc.stdin.write(key.hex().encode() + b"\n")
                self.proc.stdin.close()
                accepted: list = []
                def accept():
                    try:
                        accepted.append(listener.accept())
                    except OSError:  # the listener was closed under it (a stop, a timeout): nothing to accept
                        pass

                t = threading.Thread(target=accept, daemon=True)
                t.start()
                end = time.monotonic() + (CONNECT_TIMEOUT if connect_timeout is None else connect_timeout)
                while t.is_alive():
                    t.join(0.5)
                    if self.proc.poll() is not None:
                        raise ProbeProcessError(f"the probe process exited (code {self.proc.returncode}) before "
                                                f"connecting; see {log_path}")
                    if stop is not None and stop.is_set():
                        raise Stopping()
                    if time.monotonic() > end:
                        raise ProbeProcessError(f"the probe process did not connect within "
                                                f"{CONNECT_TIMEOUT if connect_timeout is None else connect_timeout:.0f} s; "
                                                f"see {log_path}")
                if not accepted:
                    raise ProbeProcessError(f"the probe process did not connect; see {log_path}")
            except BaseException:
                self.kill()
                self.release()
                raise
            finally:
                listener.close()
            self.conn = accepted[0]
            try:
                self._send(("setup", cfg, str(model_dir) if model_dir else None, device,
                            {"probes": True, "analysis": None}))
            except ProbeProcessError:
                self.kill()
                self.release()
                raise
            self.started = time.time()

        def poll_ready(self) -> bool:
            """True once its model has loaded (raises its error); False while it loads. Never waits."""
            if self.memory is not None:
                return True
            if self.error is not None:
                raise self.error
            if self._poll():
                self.wait_ready()  # a message is there: "ready" or "error"
                return True
            if self.proc.poll() is not None:
                raise ProbeProcessError(f"the probe process exited (code {self.proc.returncode}); see {self.log_path}")
            return False

        def submit_payload(self, jobs: list, payload: bytes, negative_kind: str) -> None:
            job_id = self._next_id
            self._next_id += 1
            self._send(("probe", job_id, jobs, payload, negative_kind))
            self.queue.append(Pending(0, jobs, {}, kind="probe", job_id=job_id))

        def poll_result(self):
            """(images, seconds) once the chunk is rendered; None while it renders; raises when the process failed."""
            p = self.queue[0] if self.queue else None
            if p is None:
                return None
            self._pump()
            if p.reply is None:
                if self.proc.poll() is not None:
                    raise ProbeProcessError(f"the probe process exited (code {self.proc.returncode}); see "
                                            f"{self.log_path}")
                return None
            self.queue.remove(p)
            _, images, seconds = p.reply
            return images, seconds

        def request_stop(self) -> None:
            try:
                self.conn.send(("stop",))
            except Exception:  # noqa: BLE001 - it may be gone already
                pass

        def exited(self) -> bool:
            return self.proc.poll() is not None

        def kill(self) -> None:
            try:
                if self.proc.poll() is None:
                    self.proc.kill()
                    self.proc.wait(timeout=5)
            except Exception:  # noqa: BLE001
                pass

        def release(self) -> None:
            for f in (getattr(self.conn, "close", None), self.log_file.close):  # (conn: not there while connecting)
                try:
                    if f:
                        f()
                except Exception:  # noqa: BLE001
                    pass

    _WORKER_CLASS = PoolWorker
    return PoolWorker


def spawn_worker(cfg: dict, model_dir, device: str, log_path: Path, stop=None, on_start=None):
    """A started worker process loading its model on `device` (returns once it has connected, not once it is ready).
    on_start(worker) is called as soon as the process exists (it is already in the Job Object), so it can be killed
    while it connects; `stop` ends the wait."""
    return real_worker_class()(cfg, model_dir, device, log_path, stop=stop, on_start=on_start)


# ---------------------------------------------------------------- the split


def job_cost(job) -> float:
    """What a picture costs to render: steps x width x height x frames (any of them missing counts 1). The cost of a
    clip is its frames; the fixed per-chunk work (reading the LoRA state, merging it in) is not subtracted."""
    cost = 1.0
    for name in ("steps", "width", "height", "frames"):
        try:
            cost *= max(1.0, float(getattr(job, name, 1) or 1))
        except (TypeError, ValueError):
            pass
    return cost


def fixed_sizes(n_jobs: int, n_gpus: int) -> list[int]:
    """The probe rounds' map: equal contiguous chunks over the GPUs in device order (the first n_jobs % n_gpus get one
    more). The same for every probe round while the set of GPUs and the number of pictures are unchanged."""
    if n_gpus <= 0:
        return []
    base, extra = divmod(max(0, n_jobs), n_gpus)
    return [base + (1 if i < extra else 0) for i in range(n_gpus)]


def plan_split(costs: list[float], rates: list[float | None]) -> list[int]:
    """How many of the jobs (a list in order, with these costs) each GPU takes, as contiguous chunks in device order.

    rates: each GPU's measured seconds per unit cost, None when it has none yet (it is given the mean of the measured
    ones; with none measured the split is equal). Shares are proportional to speed (1 / rate) of the total cost; a job
    goes to the GPU whose share holds the middle of the job (so remainders round to the nearest and costs weigh in).
    With fewer jobs than GPUs the fastest GPUs take one job each and the rest none; with at least as many, every GPU
    takes at least one."""
    k, n = len(rates), len(costs)
    if k == 0:
        return []
    if n == 0:
        return [0] * k
    measured = [r for r in rates if r and r > 0]
    mean = sum(measured) / len(measured) if measured else 1.0
    speeds = [1.0 / (r if r and r > 0 else mean) for r in rates]
    if n <= k:
        fastest = sorted(range(k), key=lambda i: (-speeds[i], i))[:n]
        return [1 if i in fastest else 0 for i in range(k)]
    total = float(sum(costs))
    scale = total / sum(speeds)
    edges, acc = [], 0.0
    for s in speeds:
        acc += s * scale
        edges.append(acc)
    sizes, gpu, before = [0] * k, 0, 0.0
    for c in costs:
        middle = before + c / 2
        while gpu < k - 1 and middle >= edges[gpu]:
            gpu += 1
        sizes[gpu] += 1
        before += c
    # with at least as many jobs as GPUs every GPU gets one: its speed keeps being measured, and the fixed cost of a
    # chunk (reading the LoRA state) cannot starve the slower card into never being timed again
    while 0 in sizes:
        sizes[sizes.index(max(sizes))] -= 1
        sizes[sizes.index(0)] += 1
    return sizes


# ---------------------------------------------------------------- the pool


@dataclass
class Slot:
    """One GPU of the pool."""
    device: str
    name: str = ""
    vram_gb: float | None = None
    state: str = "loading"
    worker: object = None
    renders: int = 0  # renders answered: the first is a warm-up and is not timed
    rates: dict = field(default_factory=dict)  # round kind -> seconds per unit cost (EWMA)
    error: str = ""
    chunk: list | None = None  # the jobs it is rendering now
    hang_at: float = 0.0

    def measure(self, kind: str, seconds: float, cost: float) -> None:
        self.renders += 1
        if self.renders == 1 or seconds <= 0 or cost <= 0:  # the first render of a worker warms up: not a speed
            return
        rate = seconds / cost
        old = self.rates.get(kind)
        self.rates[kind] = rate if old is None else EWMA * rate + (1 - EWMA) * old


def short_error(e: BaseException | str, limit: int = 220) -> str:
    """One line for a status message: the last line of a traceback, with an out-of-memory said plainly."""
    text = str(e).strip()
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    line = lines[-1] if lines else type(e).__name__ if isinstance(e, BaseException) else "failed"
    if "out of memory" in text.lower():
        line = f"Out of GPU memory. {line}"
    return line[:limit]


class RenderPool:
    """The workers of one helper. Everything here runs in the helper's worker thread (a round's polling included); the
    tick thread only calls kill_all() / close() when the run ends or the helper stops."""

    def __init__(self, devices: list[str], log_dir: Path, prefix: str, stop, on_change=None,
                 clock=time.monotonic):
        self.slots = [Slot(d) for d in devices]
        self.log_dir, self.prefix, self.stop, self.on_change, self.clock = Path(log_dir), prefix, stop, on_change, clock

    # ------------------------------------------------------------ state
    @property
    def live(self) -> list[Slot]:
        return [s for s in self.slots if s.state in ("ready", "rendering")]

    def snapshot(self) -> list[dict]:
        """The status file's `gpus`: [{device, name, state, sec_per_unit}]."""
        out = []
        for s in self.slots:
            rate = s.rates.get("sample") or s.rates.get("probe")
            out.append({"device": s.device, "name": s.name, "state": s.state,
                        "sec_per_unit": float(f"{rate:.4g}") if rate else None})
        return out

    def note(self) -> str:
        """A sentence naming every GPU that did not load or was dropped, and why; empty when there is none."""
        bad = [s for s in self.slots if s.state in ("dropped", "load_failed")]
        if not bad:
            return ""
        what = "; ".join(f"{s.device} {'did not load' if s.state == 'load_failed' else 'was dropped'}: {s.error}"
                         for s in bad)
        live = ", ".join(s.device for s in self.live) or "no GPU"
        return f"{what}. Rendering on {live}."

    def _changed(self) -> None:
        if self.on_change:
            try:
                self.on_change()
            except Exception:  # noqa: BLE001 - a status write never ends a render
                log.debug("pool on_change failed", exc_info=True)

    # ------------------------------------------------------------ loading
    def load_all(self, cfg: dict, model_dir) -> list[Slot]:
        """Load the workers one after another. Returns the slots that loaded; the others are "load_failed" (their error
        says why). Raises Stopping when the helper stops meanwhile."""
        for slot in self.slots:
            info = device_info(slot.device)
            slot.name, slot.vram_gb = info.get("gpu") or slot.device, info.get("vram_gb")
            slot.state = "loading"
            self._changed()
            try:
                self._load_one(slot, cfg, model_dir)
            except Stopping:
                self._discard(slot)
                raise
            except Exception as e:  # noqa: BLE001 - OOM, a missing file, a crashed process, the timeout
                log.error("Worker for %s did not load: %s", slot.device, e)
                slot.error = short_error(e)
                self._discard(slot)
                slot.state = "load_failed"
            else:
                slot.state = "ready"
            self._changed()
        return [s for s in self.slots if s.state == "ready"]

    def _load_one(self, slot: Slot, cfg: dict, model_dir) -> None:
        log_path = self.log_dir / f"{self.prefix}_{slot.device.replace(':', '')}.log"
        def started(w):  # the process exists: kill_all can reach it while it connects
            slot.worker = w

        slot.worker = spawn_worker(cfg, model_dir, slot.device, log_path, stop=self.stop, on_start=started)
        end = self.clock() + LOAD_TIMEOUT
        while not slot.worker.poll_ready():
            if self.stop.is_set():
                raise Stopping()
            if self.clock() > end:
                raise TimeoutError(f"the model did not load within {LOAD_TIMEOUT / 60:.0f} minutes")
            self.stop.wait(POLL)
        if self.stop.is_set():
            raise Stopping()

    # ------------------------------------------------------------ closing
    def _discard(self, slot: Slot) -> None:
        """Kill the slot's process and let go of it (a worker that is already gone is no error)."""
        w, slot.worker, slot.chunk = slot.worker, None, None
        if w is not None:
            for fn in (w.kill, w.release):
                try:
                    fn()
                except Exception:  # noqa: BLE001
                    pass

    def _drop(self, slot: Slot, why: str) -> None:
        log.warning("Dropped %s: %s", slot.device, why)
        slot.error, slot.state = why, "dropped"
        self._discard(slot)
        self._changed()

    def kill_all(self) -> None:
        """Kill every worker at once (the helper is stopping). Safe to call from any thread, and twice."""
        for s in self.slots:
            self._discard(s)

    def close(self, wait: float | None = None) -> None:
        """Stop every worker: "stop" to each, up to `wait` seconds (CLOSE_WAIT) for all of them to leave together, then
        kill the rest. Bounded, so the tick thread never waits longer."""
        wait = CLOSE_WAIT if wait is None else wait
        workers = [s.worker for s in self.slots if s.worker is not None]
        for w in workers:
            try:
                w.request_stop()
            except Exception:  # noqa: BLE001
                pass
        end = self.clock() + wait
        while self.clock() < end and not all(self._exited(w) for w in workers):
            time.sleep(min(0.05, max(0.0, end - self.clock())))
        self.kill_all()

    @staticmethod
    def _exited(w) -> bool:
        try:
            return bool(w.exited())
        except Exception:  # noqa: BLE001
            return True

    def drain(self) -> None:
        """Before a round's folder goes: every worker either answered or is killed, so none still writes into it and no
        job stays queued on a worker that goes on to the next round."""
        for s in self.slots:
            if s.chunk is not None:
                self._drop(s, "was still rendering when the round ended")

    # ------------------------------------------------------------ a round
    def _deal(self, jobs: list, slots: list[Slot], kind: str) -> list[tuple[Slot, list]]:
        if kind == "probe":
            sizes = fixed_sizes(len(jobs), len(slots))
        else:
            sizes = plan_split([job_cost(j) for j in jobs], [s.rates.get(kind) for s in slots])
        out, at = [], 0
        for s, n in zip(slots, sizes):
            if n:
                out.append((s, jobs[at:at + n]))
            at += n
        return out

    def _expected(self, slot: Slot, chunk: list, kind: str) -> float | None:
        rate = slot.rates.get(kind)
        return rate * sum(job_cost(j) for j in chunk) if rate else None

    def _send(self, chunks: list[tuple[Slot, list]], payload: bytes, negative_kind: str, kind: str,
              deadline_s: float, round_end: float | None = None, fail_fast: bool = False) -> list[list]:
        """Hand each worker its chunk, one after another (the LoRA bytes go to one worker at a time). The chunks of the
        workers that could not be reached are returned (fail_fast: the first one raises RoundFailed).

        A chunk's hang cap is its own cap (3x its expected seconds, at least HANG_FLOOR, at most deadline_s; deadline_s
        when it has no speed yet) and never later than the round's end (round_end, a clock time: the instance's deadline
        runs from the claim, so this is claim + deadline_s)."""
        lost = []
        round_end = self.clock() + deadline_s if round_end is None else round_end
        for slot, chunk in chunks:
            if self.stop.is_set():
                raise Stopping()
            try:
                slot.worker.submit_payload(chunk, payload, negative_kind)
            except Exception as e:  # noqa: BLE001 - a dead process
                lost.append(chunk)
                self._drop(slot, short_error(e))
                if fail_fast:
                    raise RoundFailed(f"a GPU failed again on the pictures dealt again ({self.note()})")
                continue
            slot.chunk, slot.state = chunk, "rendering"
            expected = self._expected(slot, chunk, kind)
            cap = deadline_s if expected is None else min(deadline_s, max(HANG_FLOOR, HANG_FACTOR * expected))
            now = self.clock()
            slot.hang_at = now + min(cap, round_end - now)
            self._changed()
        return lost

    def _collect(self, kind: str, fail_fast: bool = False) -> list[list]:
        """Poll the workers until each has answered or failed. The chunks of the ones that failed are returned; with
        fail_fast the first failure raises RoundFailed at once (the caller's drain kills the others)."""
        lost = []
        while True:
            busy = [s for s in self.slots if s.chunk is not None]
            if not busy:
                return lost
            if self.stop.is_set():
                raise Stopping()
            for s in busy:
                w, chunk = s.worker, s.chunk
                if w is None or chunk is None:  # killed by kill_all meanwhile (the helper is stopping)
                    continue
                failure = ""
                try:
                    result = w.poll_result()
                except Exception as e:  # noqa: BLE001 - the process said it failed, or it is gone
                    failure = short_error(e)
                    result = None
                if not failure and result is None and self.clock() > s.hang_at:
                    failure = "did not answer in time (killed)"
                if failure:
                    if s.chunk is None:  # (kill_all got there first)
                        continue
                    lost.append(chunk)
                    self._drop(s, failure)
                    if fail_fast:
                        raise RoundFailed(f"a GPU failed again on the pictures dealt again ({self.note()})")
                    continue
                if result is None:
                    continue
                _, seconds = result
                s.measure(kind, float(seconds), sum(job_cost(j) for j in chunk))
                s.chunk, s.state = None, "ready"
                self._changed()
            if any(s.chunk is not None for s in self.slots):
                self.stop.wait(POLL)

    def run_round(self, jobs: list, payload: bytes, kind: str, negative_kind: str = "probe",
                  deadline_s: float | None = None, claim: float | None = None) -> float:
        """Render the jobs (pictures land at their out_paths) across the live GPUs; the round's wall seconds.

        claim: the pool clock's time (time.monotonic) at which the helper claimed the round: the instance's deadline
        runs from there, so the LoRA fetch before this call already used some of it. Without it, from now.

        RoundFailed when it cannot be done: every GPU dropped; a chunk failed again after its re-deal (at once, the other
        re-dealt chunks are not waited for); or the re-deal cannot be shown to fit in the deadline. That last rule: never
        re-deal with no time left; with every chunk's finish predicted (from its GPU's measured speed) it must fit in
        the time left; with a prediction missing it is tried only while at least HANG_FLOOR seconds are left. Every worker
        has answered or been killed when this returns or raises."""
        start = self.clock()
        t0 = start if claim is None else claim
        deadline_s = float(deadline_s) if deadline_s else DEFAULT_DEADLINE
        round_end = t0 + deadline_s
        live = self.live
        if not live:
            raise RoundFailed("no GPU is left to render on")
        if round_end - start <= 0:
            raise RoundFailed("the round's deadline passed before rendering could start")
        try:
            lost = self._send(self._deal(jobs, live, kind), payload, negative_kind, kind, deadline_s, round_end)
            lost += self._collect(kind)
            if self.stop.is_set():
                raise Stopping()
            if lost:
                survivors = self.live
                if not survivors:
                    raise RoundFailed(f"every GPU failed ({self.note()})")
                again = [j for chunk in lost for j in chunk]
                chunks = self._deal(again, survivors, kind)
                left = round_end - self.clock()
                predicted = [self._expected(s, c, kind) for s, c in chunks]
                if left <= 0 or any(p is not None and p > left for p in predicted) or (
                        any(p is None for p in predicted) and left < HANG_FLOOR):
                    raise RoundFailed(f"a GPU failed and dealing its pictures again cannot be shown to fit in the "
                                      f"round's deadline ({self.note()})")
                self._send(chunks, payload, negative_kind, kind, deadline_s, round_end, fail_fast=True)
                self._collect(kind, fail_fast=True)
                if self.stop.is_set():
                    raise Stopping()
        finally:
            self.drain()
        return self.clock() - start
