"""Probes off the critical path (probe_async): a separate process holds a copy of the model on a second GPU and renders
each likeness probe while training goes on; the trainer scores the images and acts on them once they are in (up to
probe_every steps later).

A process, not a thread: in one process the training loop and the renders take turns at Python's interpreter lock,
and both launch thousands of small kernels, so each slowed the other more than the probe pause it saved (measured on
Qwen 2.1 int8, 4060 Ti + 5060 Ti: 8% slower with 512 px probes, 35% with 1024 px). The process gets the resolved
config once and, before each probe, the LoRA's state as CPU tensors (a few MB; Windows cannot share CUDA tensors
between processes). The trainer keeps that state, so a best checkpoint saves exactly the weights that were scored.
One probe renders at a time: when the next one is due before the last is in, training waits for it.

Run as `python -m rulztrainer.probeworker <port>` by ProbeProcess, which writes the connection key on its stdin."""
from __future__ import annotations

import io
import logging
import os
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


class ProbeProcessError(RuntimeError):
    """The probe process failed (its traceback in the message)."""


@dataclass
class Pending:
    step: int  # the training step whose weights it renders
    jobs: list  # its RenderJobs, in order
    state: dict  # the LoRA state it renders (lora.lora_state), for a best save
    queued: float = field(default_factory=time.time)


class ProbeProcess:
    """The trainer's side: starts the probe process, sends it probes and takes their images back."""

    def __init__(self, cfg: dict, model_dir, device: str, log_path: Path):
        self.device = device
        self.pending: Pending | None = None
        self.waited = 0.0  # seconds training spent waiting for a probe to come in
        self.rendered = 0.0  # seconds the probe GPU spent rendering
        self.count = 0
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
        self.conn.send(("setup", cfg, str(model_dir) if model_dir else None, device))
        self.started = time.time()
        self.memory = None  # its GPU memory once its model has loaded (it loads while training starts)

    def _recv(self):
        while not self.conn.poll(0.5):
            if self.proc.poll() is not None:
                raise ProbeProcessError(f"the probe process exited (code {self.proc.returncode}); see "
                                        f"{self.log_path}")
        msg = self.conn.recv()
        if msg[0] == "error":
            raise ProbeProcessError(msg[1])
        return msg

    def wait_ready(self) -> str:
        """Wait for the process's model to load (raises its error); its GPU memory."""
        if self.memory is None:
            self.memory = self._recv()[1]
            log.info("Probe model ready on %s in %.0fs (%s)", self.device, time.time() - self.started, self.memory)
        return self.memory

    def _poll(self) -> bool:
        """A rendered probe is waiting to be read (the ready message that comes first is read on the way)."""
        if self.memory is None:
            if not self.conn.poll():
                return False
            self.wait_ready()
        return self.conn.poll()

    def submit(self, step: int, jobs: list, state: dict) -> None:
        """Send the probe of `step` (its jobs, the LoRA state to render); the last one must have been taken."""
        if self.pending is not None:
            raise RuntimeError(f"the probe of step {self.pending.step} is still queued")
        buf = io.BytesIO()
        torch.save(state, buf)
        self.conn.send(("probe", jobs, buf.getvalue()))
        self.pending = Pending(step, jobs, state)

    def ready(self) -> bool:
        return self.pending is not None and self._poll()

    def take(self, wait: bool) -> tuple[Pending, list[str]] | None:
        """The queued probe and its image paths once rendered (removed from the queue); None with none queued, or
        while it renders unless wait (then blocks; the time counts as waited). The render's error is raised here."""
        p = self.pending
        if p is None or not (wait or self._poll()):
            return None
        self.pending = None
        t0 = time.time()
        try:
            self.wait_ready()
            _, images, seconds = self._recv()
        finally:
            self.waited += time.time() - t0
        self.rendered += seconds
        self.count += 1
        return p, images

    def close(self) -> None:
        """Let a probe still rendering finish, then stop the process."""
        self.pending = None
        try:
            self.conn.send(("stop",))
            self.proc.wait(timeout=600)
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
    """The probe process's model copy: the sampling model (sample_model_path, else the training base) with the same
    LoRA slots and the sampling LoRAs, and a VAE, on one GPU."""

    @torch.no_grad()
    def __init__(self, cfg_dict: dict, model_dir: str | None, device: str):
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
        tr.arch = detect_arch(cfg, mdir)
        attention.set_backend(cfg.attention_backend, device)
        t0 = time.time()
        if cfg.sample_model_path:
            m = tr.arch.load_any_transformer(cfg.sample_model_path, device, cfg.sample_model_quant, cfg.models_dir)
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
        m.eval()
        lora_mod.set_extras_enabled(m, True)  # it only renders: the sampling LoRAs stay on
        vae = tr.arch.get_vae(cfg, mdir)
        self.tr, self.slot = tr, SampleSlot(m, device, vae)
        self.memory = gpu_memory_summary([device])
        log.info("Probe model ready on %s in %.0fs (%s)", device, time.time() - t0, self.memory)

    @torch.no_grad()
    def render(self, jobs: list, payload: bytes) -> tuple[list[str], float]:
        from . import lora as lora_mod
        from .trainer import _Board, load_text_embeds

        t0 = time.time()
        state = torch.load(io.BytesIO(payload), weights_only=True)
        lora_mod.mirror_lora(lora_mod.lora_snapshots(state), self.injected)
        tr = self.tr
        self.slot.model.eval()  # mirror_lora may have handed it a new dropout module
        neg_e, neg_m = tr.arch.pad_embeds([load_text_embeds(tr.cfg, tr.negative_prompt())])
        with lora_mod.frozen_adapters():
            images = tr._render_round([self.slot], jobs, neg_e, neg_m, _Board(tr.status, ""))
        return images, time.time() - t0


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S",
                        stream=sys.stdout)
    key = bytes.fromhex(sys.stdin.readline().strip())
    conn = Client(("127.0.0.1", int(sys.argv[1])), authkey=key)
    _, cfg, model_dir, device = conn.recv()
    try:
        r = Renderer(cfg, model_dir, device)
    except BaseException:  # noqa: BLE001
        conn.send(("error", traceback.format_exc()))
        return
    conn.send(("ready", r.memory))
    while True:
        try:
            msg = conn.recv()
        except EOFError:
            return
        if msg[0] == "stop":
            return
        try:
            conn.send(("done", *r.render(msg[1], msg[2])))
        except Exception:  # noqa: BLE001
            log.exception("Probe render failed")
            conn.send(("error", traceback.format_exc()))


if __name__ == "__main__":
    main()
