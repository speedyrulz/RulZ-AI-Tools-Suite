"""Probes off the critical path (probe_async): a separate process holds a copy of the model on a second GPU and renders
each likeness probe while training goes on; the trainer scores the images and acts on them once they are in (up to
probe_every steps later).

A process, not a thread: in one process the training loop and the renders take turns at Python's interpreter lock,
and both launch thousands of small kernels, so each slowed the other more than the probe pause it saved (measured on
Qwen 2.1 int8, 4060 Ti + 5060 Ti: 8% slower with 512 px probes, 35% with 1024 px). The process gets the resolved
config once and, before each probe, the LoRA's state as CPU tensors (a few MB; Windows cannot share CUDA tensors
between processes). The trainer keeps that state, so a best checkpoint saves exactly the weights that were scored.
One probe renders at a time: when the next one is due before the last is in, training waits for it.

The same process can also run the region sweeps and the training maps (maps_async, sweep_async). They read the model the run
trains (the training base with the same quantization, the same LoRA slots and the training adapter, the sampling
LoRAs off), so a second model loads beside the sampling one when sample_model_path sets another, and one model serves
both when it does not. The pictures' paths and tensors go over once, with the setup; each job then carries the LoRA
state only. Jobs of every kind queue in one FIFO: the process answers them in the order they were sent, and
ProbeProcess hands each answer to its job.

Run as `python -m rulztrainer.probeworker <port>` by ProbeProcess, which writes the connection key on its stdin."""
from __future__ import annotations

import io
import logging
import os
import queue
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from multiprocessing.connection import Client, Listener
from pathlib import Path

import torch

from .multigpu import norm_device, parse_devices

log = logging.getLogger("rulztrainer")


def pick_device(probe_device: str, devices: str, train_device, n_gpus: int) -> tuple[str, str]:
    """(the GPU that renders the probes, "") or ("", why there is none): probe_device, else the first GPU of
    `devices` that is not the training one."""
    train = norm_device(train_device)
    if n_gpus < 2:
        return "", "only one GPU"
    for d in [probe_device] if probe_device else parse_devices(devices, train):
        dev = torch.device(d)
        if dev.type == "cuda" and dev.index is not None and dev.index < n_gpus and str(dev) != train:
            return str(dev), ""
    if probe_device:
        return "", f"probe_device {probe_device} is not a second GPU (training on {train})"
    return "", f"no GPU besides {train} in devices ({devices})"


def pack_tensors(data):
    """Data holding tensors, as bytes to send to the probe process (None stays None). A tensor sent as itself over the
    connection is not copied: torch's pickler sends a handle to shared memory that the other side fetches from this
    process's resource sharer, which checks the multiprocessing authkey. The probe process is a program of its own with
    another authkey, so the fetch is refused ("digest sent was rejected") and the process dies before loading."""
    if data is None:
        return None
    buf = io.BytesIO()
    torch.save(data, buf)
    return buf.getvalue()


def unpack_tensors(data):
    """pack_tensors' bytes back as the data (anything else is returned as it is)."""
    if isinstance(data, (bytes, bytearray)):
        return torch.load(io.BytesIO(data), weights_only=True)
    return data


class ProbeProcessError(RuntimeError):
    """The probe process failed (its traceback in the message)."""


ANALYSIS = ("sweep", "maps")  # the kinds of job besides the probes
_REPLY = {"probe": "done", "sample": "done", "sweep": "sweep_done", "maps": "maps_done"}  # the tag of each kind's answer


@dataclass
class Pending:
    step: int  # the training step whose weights it renders
    jobs: list  # a probe's RenderJobs, in order (a sweep's: the keys of its pictures)
    state: dict  # the LoRA state it renders (lora.lora_state), for a best save
    queued: float = field(default_factory=time.time)
    kind: str = "probe"  # probe | sample | sweep | maps
    extra: dict = field(default_factory=dict)  # what the trainer records it with (epoch; for maps, effect)
    reply: tuple | None = None  # the process's answer, once read


class ProbeProcess:
    """The trainer's side: starts the probe process, sends it probes and takes their images back."""

    def __init__(self, cfg: dict, model_dir, device: str, log_path: Path, probes: bool = True,
                 analysis: dict | None = None):
        """probes: it renders probes (loads the sampling model and a VAE). analysis: what it needs to run sweeps and
        maps ({"sweep": [pictures] | None, "maps": {...} | None}, see Renderer), or None for neither."""
        self.device = device
        self.probes = probes
        self.queue: list[Pending] = []  # the jobs sent and not yet taken, oldest first
        self.waited = 0.0  # seconds training spent waiting for a job to come in
        self.rendered = 0.0  # seconds the probe GPU spent rendering
        self.count = 0
        self.analysed = 0.0  # seconds it spent on sweeps and maps
        self.analysis_count = 0
        self.analysis_wanted = analysis is not None
        self.analysis_error: str | None = None  # why its analysis model did not load (known once it is ready)
        key = os.urandom(32)
        listener = Listener(("127.0.0.1", 0), authkey=key)
        root = str(Path(__file__).resolve().parent.parent)
        env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in (root, os.environ.get("PYTHONPATH")) if p)}
        self.log_path = log_path
        self.log_file = open(log_path, "ab")
        self.proc = subprocess.Popen([sys.executable, "-m", "rulztrainer.probeworker", str(listener.address[1])],
                                     stdin=subprocess.PIPE, stdout=self.log_file, stderr=subprocess.STDOUT, env=env,
                                     cwd=root)
        self.proc.stdin.write(key.hex().encode() + b"\n")
        self.proc.stdin.close()
        accepted = []
        t = threading.Thread(target=lambda: accepted.append(listener.accept()), daemon=True)
        t.start()
        try:
            while t.is_alive():
                t.join(0.5)
                if self.proc.poll() is not None:
                    raise ProbeProcessError(f"the probe process exited (code {self.proc.returncode}) before "
                                            f"connecting; see {log_path}")
        finally:
            listener.close()
        if not accepted:
            raise ProbeProcessError(f"the probe process did not connect; see {log_path}")
        self.conn = accepted[0]
        try:
            self._send(("setup", cfg, str(model_dir) if model_dir else None, device,
                        {"probes": probes, "analysis": pack_tensors(analysis)}))
        except ProbeProcessError:
            self.proc.kill()
            for f in (self.conn.close, self.log_file.close):
                try:
                    f()
                except Exception:  # noqa: BLE001
                    pass
            raise
        self.started = time.time()
        self.memory = None  # its GPU memory once its model has loaded (it loads while training starts)

    def _poll(self, timeout: float = 0) -> bool:
        """conn.poll with a broken connection (an OSError) as a ProbeProcessError, like the sends and reads."""
        try:
            return self.conn.poll(timeout)
        except (OSError, EOFError) as e:
            raise ProbeProcessError(f"the probe process closed its connection ({type(e).__name__}); see "
                                    f"{self.log_path}") from e

    def _recv(self):
        while not self._poll(0.5):
            if self.proc.poll() is not None:
                raise ProbeProcessError(f"the probe process exited (code {self.proc.returncode}); see "
                                        f"{self.log_path}")
        try:
            msg = self.conn.recv()
        except (EOFError, OSError) as e:
            raise ProbeProcessError(f"the probe process closed its connection ({type(e).__name__}); see "
                                    f"{self.log_path}") from e
        if msg[0] == "error":
            raise ProbeProcessError(msg[1])
        return msg

    def wait_ready(self) -> str:
        """Wait for the process's model(s) to load (raises its error); its GPU memory. A failed analysis model is not
        an error: probes work, and analysis_error says why sweeps and maps cannot."""
        if self.memory is None:
            msg = self._recv()
            self.memory = msg[1]
            self.analysis_error = msg[2] if len(msg) > 2 else None
            log.info("Probe process ready on %s in %.0fs (%s)", self.device, time.time() - self.started, self.memory)
        return self.memory

    @property
    def analysis_ok(self) -> bool:
        """Sweeps and maps can run here (known once it is ready: false before)."""
        return self.analysis_wanted and self.memory is not None and self.analysis_error is None

    @property
    def pending(self) -> Pending | None:
        """The probe that is out, if any."""
        return self.pending_of("probe")

    def pending_of(self, kind: str) -> Pending | None:
        return next((p for p in self.queue if p.kind == kind), None)

    def _pump(self, until: Pending | None = None) -> None:
        """Read the answers that are in, each into the job it belongs to (the process answers in the order it was
        sent them); with `until`, wait for that job's answer."""
        if self.memory is None:
            if until is None and not self._poll():
                return
            self.wait_ready()
        while True:
            waiting = next((p for p in self.queue if p.reply is None), None)
            if waiting is None or (until is None and not self._poll()):
                return
            if until is not None and until.reply is not None:
                return
            msg = self._recv()
            if msg[0] != _REPLY[waiting.kind] and not (msg[0] == "analysis_failed" and waiting.kind in ANALYSIS):
                raise ProbeProcessError(f"the probe process answered {msg[0]!r} to the {waiting.kind} of step "
                                        f"{waiting.step}")
            waiting.reply = msg

    def _in(self, p: Pending, wait: bool) -> bool:
        """The answer to `p` has been read (with wait, after waiting for it: that time counts as waited)."""
        if p.reply is None:
            t0 = time.time()
            try:
                self._pump(p if wait else None)
            finally:
                if wait:
                    self.waited += time.time() - t0
        return p.reply is not None

    def _send(self, msg: tuple) -> None:
        try:
            self.conn.send(msg)
        except (OSError, EOFError) as e:  # (BrokenPipeError is an OSError)
            raise ProbeProcessError(f"the probe process closed its connection ({type(e).__name__}); see "
                                    f"{self.log_path}") from e

    @staticmethod
    def _payload(state: dict) -> bytes:
        buf = io.BytesIO()
        torch.save(state, buf)
        return buf.getvalue()

    def submit(self, step: int, jobs: list, state: dict, kind: str = "probe") -> None:
        """Send the probe (kind "sample": the sample round) of `step` (its jobs, the LoRA state to render); the last of
        its kind must have been taken."""
        if self.pending_of(kind) is not None:
            raise RuntimeError(f"the {kind} of step {self.pending_of(kind).step} is still queued")
        self._send(("probe", jobs, self._payload(state), "sample" if kind == "sample" else "probe"))
        self.queue.append(Pending(step, jobs, state, kind=kind))

    def submit_analysis(self, kind: str, step: int, state: dict, keys: list | None = None,
                        extra: dict | None = None) -> None:
        """Send the sweep (over the pictures `keys`) or the maps of `step` to read at the LoRA `state`; one of each
        kind at a time. extra: what the trainer records it with (epoch; for maps, effect: read the LoRA's effect)."""
        if kind not in ANALYSIS:
            raise ValueError(kind)
        if self.pending_of(kind) is not None:
            raise RuntimeError(f"the {kind} of step {self.pending_of(kind).step} is still queued")
        extra = dict(extra or {})
        self._send((kind, self._payload(state), list(keys or []) if kind == "sweep" else bool(extra.get("effect"))))
        self.queue.append(Pending(step, list(keys or []), state, kind=kind, extra=extra))

    def ready(self) -> bool:
        p = self.pending
        return p is not None and self._in(p, False)

    def ready_analysis(self, kind: str) -> bool:
        p = self.pending_of(kind)
        return p is not None and self._in(p, False)

    def take(self, wait: bool, kind: str = "probe") -> tuple[Pending, list[str]] | None:
        """The queued probe and its image paths once rendered (removed from the queue); None with none queued, or
        while it renders unless wait (then blocks; the time counts as waited). The render's error is raised here."""
        p = self.pending_of(kind)
        if p is None or not self._in(p, wait):
            return None
        self.queue.remove(p)
        _, images, seconds = p.reply
        self.rendered += seconds
        self.count += 1
        return p, images

    def take_sample(self, wait: bool) -> tuple[Pending, list[str]] | None:
        """The queued sample round and its image paths: take() for kind "sample"."""
        return self.take(wait, "sample")

    def take_analysis(self, kind: str, wait: bool) -> tuple[Pending, object, str] | None:
        """The queued sweep or maps once the process is through with it: (job, its readings or maps, ""), or (job, None,
        why) when its computation failed (the same failure the code has here: it turns that analysis off, the process
        goes on). None with none queued, or while it runs unless wait. A process that failed raises here."""
        p = self.pending_of(kind)
        if p is None or not self._in(p, wait):
            return None
        self.queue.remove(p)
        if p.reply[0] == "analysis_failed":
            return p, None, p.reply[2]
        self.analysed += p.reply[2]
        self.analysis_count += 1
        return p, p.reply[1], ""

    def outstanding(self) -> list[Pending]:
        """The sample rounds, sweeps and maps still out (after a failure: the trainer takes the ones answered as they
        are and does the others again from their kept LoRA state; a probe is lost)."""
        return [p for p in self.queue if p.kind != "probe"]

    def close(self) -> None:
        """Let a job still running finish, then stop the process."""
        self.queue = []
        try:
            self.conn.send(("stop",))
            deadline = time.time() + 600
            while self.proc.poll() is None:  # read what it is still sending, or a blocked send never reaches "stop"
                if time.time() > deadline:
                    raise TimeoutError
                try:
                    if self.conn.poll(0.5):
                        self.conn.recv()
                except (EOFError, OSError):
                    self.proc.wait(timeout=max(0.0, deadline - time.time()))
                    break
        except Exception:  # noqa: BLE001
            self.proc.kill()
        finally:
            for f in (self.conn.close, self.log_file.close):
                try:
                    f()
                except Exception:  # noqa: BLE001
                    pass


# ------------------------------------------------------------ the probe process


class _Quiet:
    """A status that goes nowhere (the trainer shows the probe's progress itself)."""

    data: dict = {}

    def update(self, **kw):
        pass


class Renderer:
    """The probe process's models, on one GPU.

    probes: the sampling model (sample_model_path, else the training base) with the same LoRA slots and the sampling
    LoRAs, and a VAE: renders the likeness probes.
    analysis: the model the run trains, for the region sweeps and the training maps: the training base with the same
    quantization, the same LoRA slots, the training adapter if the run has one, and no sampling LoRAs. Without
    sample_model_path one model is both (the sampling LoRAs on for a render, off for an analysis; the training adapter
    follows that switch); with it a second model is loaded, and when that fails (out of memory on the second GPU)
    `analysis_error` says why and the probes carry on.

    What it cannot mirror of the training model: torch.compile, block swap, activation offload and gradient
    checkpointing (it only reads, eagerly, with the whole model on the GPU); the numbers they change are rounding."""

    @torch.no_grad()
    def __init__(self, cfg_dict: dict, model_dir: str | None, device: str, probes: bool = True,
                 analysis: dict | None = None):
        from . import attention
        from . import lora as lora_mod
        from . import quant as Q
        from .archs import detect_arch
        from .config import TrainConfig
        from .multigpu import gpu_memory_summary
        from .trainer import SampleSlot, Trainer

        torch.cuda.set_device(device)
        cfg = TrainConfig.from_dict({**cfg_dict, "device": device})  # the VAE and the base load onto this GPU
        Q.set_int8_defaults(cfg.int8_matmul, cfg.int8_backward)
        Q.set_nvfp4_default(cfg.nvfp4_matmul, cfg.nvfp4_backward)
        mdir = Path(model_dir) if model_dir else None
        tr = Trainer.__new__(Trainer)  # a shell for its sampling helpers; nothing trains here
        tr.cfg, tr.race_union_names, tr.compiled, tr.sample_graphs, tr.status = cfg, None, False, set(), _Quiet()
        tr.ft = None
        tr.arch = detect_arch(cfg, mdir)
        attention.set_backend(cfg.attention_backend, device)
        self.tr, self.mdir, self.device = tr, mdir, device
        self.slot = self.injected = None  # the rendering model
        self.model = self.train_injected = None  # the analysis model
        self.shared = probes and not cfg.sample_model_path  # one model for both
        self.extras = None  # whether the shared model's sampling LoRAs are on
        self.analysis_error: str | None = None
        self.sweep_pictures: dict[str, dict] = {}
        self.maps = None
        t0 = time.time()
        if probes:
            if cfg.sample_model_path:
                m = tr.arch.load_any_transformer(cfg.sample_model_path, device, cfg.sample_model_quant,
                                                 cfg.models_dir)
            else:
                m = tr.arch.get_transformer(cfg, mdir, device=device)
                Q.configure_int8(m, cfg.int8_matmul, cfg.int8_backward)
                Q.configure_nvfp4(m, cfg.nvfp4_matmul, cfg.nvfp4_backward)
            blocks = lora_mod.parse_blocks(cfg.lora_blocks, len(m.layers))
            self.injected = lora_mod.inject_lora(m, cfg.lora_rank, cfg.lora_alpha, cfg.lora_targets,
                                                 include_text_embedder=cfg.train_text_embedder, blocks=blocks,
                                                 names=tr._lora_names(m, blocks), kind=cfg.network_type,
                                                 factor=cfg.lokr_factor, block=cfg.oft_block_size,
                                                 constraint=cfg.oft_constraint, dora=cfg.dora)
            tr._attach_sampling_loras(m)
            vae = tr.arch.get_vae(cfg, mdir)
            self.slot = SampleSlot(m, device, vae)
        if analysis is not None:
            try:
                self._load_analysis(analysis)
            except Exception:  # noqa: BLE001
                if not probes:
                    raise
                self.analysis_error = traceback.format_exc()
                self.model = self.train_injected = self.maps = None
                self.sweep_pictures = {}
                import gc

                gc.collect()
                torch.cuda.empty_cache()
        if probes:
            self.slot.model.eval()
            self._set_extras(True)  # it renders: the sampling LoRAs stay on
        self.memory = gpu_memory_summary([device])
        log.info("Probe process models ready on %s in %.0fs (%s)%s", device, time.time() - t0, self.memory,
                 "" if self.analysis_error is None else "; the analysis model did not load")

    def _load_analysis(self, analysis: dict) -> None:
        """The training model (or the shared one with the training adapter on it) and the sweeps' and maps' pictures."""
        from safetensors.torch import load_file

        from . import lora as lora_mod
        from . import quant as Q
        from . import trainmaps

        tr, cfg = self.tr, self.tr.cfg
        if self.shared:
            m, inj = self.slot.model, self.injected
        else:
            m = tr.arch.get_transformer(cfg, self.mdir, device=self.device)
            Q.configure_int8(m, cfg.int8_matmul, cfg.int8_backward)
            Q.configure_nvfp4(m, cfg.nvfp4_matmul, cfg.nvfp4_backward)
            blocks = lora_mod.parse_blocks(cfg.lora_blocks, len(m.layers))
            inj = lora_mod.inject_lora(m, cfg.lora_rank, cfg.lora_alpha, cfg.lora_targets,
                                       include_text_embedder=cfg.train_text_embedder, dropout=cfg.lora_dropout,
                                       blocks=blocks, names=tr._lora_names(m, blocks), kind=cfg.network_type,
                                       factor=cfg.lokr_factor, block=cfg.oft_block_size,
                                       constraint=cfg.oft_constraint, dora=cfg.dora)
        tr._attach_training_adapter(m)  # on while the sampling LoRAs are off, as in training
        m.eval()
        arch = tr.arch

        def sweep_probe(d: dict) -> dict:
            def load(d=d):
                latents = load_file(d["latent_path"])["latent"][None].float()
                embeds, mask = arch.pad_embeds([load_file(d["text_path"])["embeds"]])
                return latents, embeds, mask

            return {"key": d["key"], "held": d["held"], "parts": d["parts"], "load": load}

        self.sweep_pictures = {d["key"]: sweep_probe(d) for d in (analysis.get("sweep") or [])}
        mp = analysis.get("maps")
        if mp:
            self.maps = trainmaps.TrainingMaps(".", mp["probes"], 1, sigmas=mp["sigmas"], seed=mp["seed"])
        self.model, self.train_injected = m, inj

    def _set_extras(self, on: bool) -> None:
        """The shared model's sampling LoRAs on (a render) or off (an analysis: the model the run trains)."""
        if self.shared and self.extras != on:
            from . import lora as lora_mod

            lora_mod.set_extras_enabled(self.slot.model, on)
            self.extras = on

    @torch.no_grad()
    def render(self, jobs: list, payload: bytes, negative_kind: str = "probe") -> tuple[list[str], float]:
        from . import lora as lora_mod
        from .trainer import _Board, load_text_embeds

        t0 = time.time()
        state = torch.load(io.BytesIO(payload), weights_only=True)
        self._set_extras(True)
        lora_mod.mirror_lora(lora_mod.lora_snapshots(state), self.injected)
        tr = self.tr
        self.slot.model.eval()  # mirror_lora may have handed it a new dropout module
        # the round's negative: the probes' by default (PAN 2's clip probes have one for their length, cached by
        # prepare_cache), the samples' for a remote-render sample round
        neg_e, neg_m = tr.arch.pad_embeds([load_text_embeds(tr.cfg, tr.negative_prompt(negative_kind))])
        with lora_mod.frozen_adapters():
            images = tr._render_round([self.slot], jobs, neg_e, neg_m, _Board(tr.status, ""))
        return images, time.time() - t0

    @torch.no_grad()
    def analyse(self, msg: tuple) -> tuple:
        """One sweep (msg: "sweep", the LoRA state, the keys of its pictures) or one capture of the maps (msg: "maps",
        the LoRA state, whether to read the LoRA's effect) at the state sent. The answer to send: the readings or maps
        and the seconds, or ("analysis_failed", ...) when the computation itself failed, as the same code failing in
        the trainer turns that analysis off. Running out of memory is not that: it is raised, and the trainer then
        takes the whole process as failed."""
        from . import lora as lora_mod
        from . import regionsweep

        kind, payload, arg = msg
        t0 = time.time()
        try:
            state = torch.load(io.BytesIO(payload), weights_only=True)
            self._set_extras(False)
            lora_mod.mirror_lora(lora_mod.lora_snapshots(state), self.train_injected)
            self.model.eval()  # mirror_lora may have handed it a new dropout module
            if kind == "sweep":
                result = regionsweep.forward_readings(self.model, self.device, [self.sweep_pictures[k] for k in arg])
            else:
                result = self.maps._run(self.model, self.train_injected, self.device, None, bool(arg))
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception as e:  # noqa: BLE001
            log.exception("The %s failed", kind)
            return "analysis_failed", kind, f"{type(e).__name__}: {str(e)[:300]}", time.time() - t0
        finally:
            torch.cuda.empty_cache()
        return f"{kind}_done", result, time.time() - t0


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S",
                        stream=sys.stdout)
    key = bytes.fromhex(sys.stdin.readline().strip())
    conn = Client(("127.0.0.1", int(sys.argv[1])), authkey=key)
    _, cfg, model_dir, device, *rest = conn.recv()
    opts = rest[0] if rest else {}
    try:
        r = Renderer(cfg, model_dir, device, probes=opts.get("probes", True),
                     analysis=unpack_tensors(opts.get("analysis")))
    except BaseException:  # noqa: BLE001
        conn.send(("error", traceback.format_exc()))
        return
    conn.send(("ready", r.memory, r.analysis_error))
    _serve(conn, r)


def _read_into(conn, q: "queue.Queue") -> None:
    """Receiver thread: put every incoming message into `q` at once ("stop" ends it; EOF or an error becomes "stop")."""
    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            q.put(("stop",))
            return
        q.put(msg)
        if msg and msg[0] == "stop":
            return


def _serve(conn, r) -> None:
    """Run the jobs the trainer sends, in order. A receiver thread reads each message as it arrives, so the trainer's
    send of the next job never waits on this thread while it is blocked sending a large reply (which the trainer reads
    only later): that was a deadlock. The thread starts only after the models have loaded (the caller sends "ready"
    first): on Windows a blocking read on an anonymous pipe in a thread stalled other threads' DLL loads. This
    connection is a socket, but the start stays after loading anyway."""
    q: queue.Queue = queue.Queue()
    threading.Thread(target=_read_into, args=(conn, q), daemon=True).start()
    while True:
        msg = q.get()
        if msg[0] == "stop":
            return
        try:
            if msg[0] == "probe":
                conn.send(("done", *r.render(msg[1], msg[2], msg[3] if len(msg) > 3 else "probe")))
            else:
                conn.send(r.analyse(msg))
        except Exception:  # noqa: BLE001
            log.exception("Probe process job failed")
            try:
                conn.send(("error", traceback.format_exc()))
            except (OSError, EOFError):
                return


if __name__ == "__main__":
    main()
