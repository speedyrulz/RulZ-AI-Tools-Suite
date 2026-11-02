"""LoRA training loop (flow matching) for the architectures in rulztrainer.archs: MOSS-Image and FIBO 2.0."""
from __future__ import annotations

import contextlib
import dataclasses
import gc
import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

from . import attention
from . import checkpointing as ckpt
from . import compiling
from . import history
from . import lora as lora_mod
from . import profiling
from . import race as race_mod
from .ema import LoraEMA
from . import quant as Q
from .config import AUTOMAGIC, KEEL, TrainConfig, keel_band_weights, keel_consensus_mode
from .config import side_to_mp
from .dataset import (BucketBatcher, augment_items, cache_latents, cache_text, describe_sides, dropout_text_path,
                      items_for_side, load_batch,
                      scan_dataset, text_cache_path, TEXT_BATCH)
from .archs import detect_arch, get_arch
from .idloss import effective_weight as idloss_weight
from .losses import elementwise_loss
from .race_parallel import RaceParallelMixin
from .multigpu import (Balancer, choose_counts, clone_to, compile_count, job_order, no_recompile_stance, norm_device,
                       run_queue, stance_available, step_seconds)
log = logging.getLogger("rulztrainer")
DP_COMPILE_QUIET_STEPS = 3  # data parallel + compile: serial steps without a new graph before going parallel
TEXT_TIMING_PROMPTS = 3 * TEXT_BATCH  # prompts (three batches) timed on the first copy before deciding on a second


@dataclass
class SampleSlot:
    """One GPU's share of a sampling round: the model copy, the VAE and the device they live on."""
    model: object
    device: str
    vae: object
    compiled: bool = False  # the model's blocks are torch.compiled (graph tracing must not overlap)


@dataclass
class RenderJob:
    prompt: str
    width: int
    height: int
    steps: int
    cfg_scale: float
    seed: int
    out_path: Path
    label: str  # e.g. "sample 2/4" or "probe face seed 1/2"
    kind: str = "sample"  # sample | face | body

    @property
    def shape_key(self):
        return (self.width, self.height, self.cfg_scale != 1.0)


class _Board:
    """Combines per-GPU progress into the single status 'stage' line the WebUI shows."""

    def __init__(self, status, title: str):
        import threading

        self.status, self.title, self.parts, self.lock = status, title, {}, threading.Lock()

    def set(self, device: str, text: str) -> None:
        with self.lock:
            self.parts[device] = text
            if len(self.parts) == 1 and not self.title:
                line = text
            else:
                line = " | ".join(([self.title] if self.title else []) +
                                  [f"{d}: {t}" for d, t in sorted(self.parts.items())])
            self.status.update(stage=line)


def setup_logging(job_dir: Path) -> None:
    job_dir.mkdir(parents=True, exist_ok=True)
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(job_dir / "train.log", encoding="utf-8")):
        h.setFormatter(fmt)
        log.addHandler(h)


def prune_checkpoints(job_dir: Path, stem: str, keep: int) -> list[Path]:
    """Keep the newest `keep` step checkpoints of `stem` (<stem>_<step>.safetensors and its extra ranks
    <stem>_<step>_r<rank>.safetensors) in a job folder and delete the older ones. The final file, <stem>_best and
    the dropped / winner files are never numbered, so they are never touched. keep <= 0 keeps everything. Returns the
    files deleted."""
    if keep <= 0:
        return []
    import re

    pat = re.compile(r"^" + re.escape(stem) + r"_(\d{6})(?:_r\d+)?\.safetensors$")
    by_step: dict[int, list[Path]] = {}
    for p in Path(job_dir).iterdir():
        m = pat.match(p.name)
        if m:
            by_step.setdefault(int(m.group(1)), []).append(p)
    gone = []
    for step in sorted(by_step)[: max(0, len(by_step) - keep)]:
        for p in by_step[step]:
            try:
                p.unlink()
                gone.append(p)
            except OSError as e:
                log.warning("Could not delete the old checkpoint %s: %s", p, e)
    return gone


def free_cuda():
    gc.collect()
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()


def unload_model(model) -> None:
    """Release a model fully, including accelerate dispatch hooks, so its VRAM is returned. Its tensors go to the
    meta device (dropped, not copied to RAM first: a 19 GB text encoder would otherwise pass through system memory);
    a model that cannot move there is moved to the CPU as before."""
    try:
        from accelerate.hooks import remove_hook_from_submodules

        remove_hook_from_submodules(model)
    except Exception:  # noqa: BLE001
        pass
    try:
        model.to("meta")
    except Exception:  # noqa: BLE001
        try:
            model.to("cpu")
        except Exception:  # noqa: BLE001
            pass
    del model
    free_cuda()


def _longest_text_path(items, pad_embeds) -> Path:
    """The cached caption with the most tokens, read from the file headers only (the family's pad_embeds, run on a
    meta tensor of that shape, says which dim is the sequence)."""
    from safetensors import safe_open

    def length(path) -> int:
        with safe_open(str(path), "pt") as f:
            shape = f.get_slice("embeds").get_shape()
        return pad_embeds([torch.empty(shape, device="meta")])[1].shape[1]

    return max({it.text_path for it in items}, key=length)


def load_text_embeds(cfg: TrainConfig, prompt: str) -> torch.Tensor:
    from safetensors.torch import load_file

    return load_file(str(text_cache_path(cfg, prompt)))["embeds"]


class Status:
    """Writes a small JSON file the WebUI polls."""

    def __init__(self, path: Path):
        import threading

        self.path = path
        self.lock = threading.RLock()  # GPU worker threads report progress concurrently
        self.data = {"stage": "starting", "step": 0, "total": 0, "loss": None, "avg_loss": None,
                     "elapsed": 0.0, "eta": None, "samples": [], "saved": [], "error": None, "done": False,
                     "losses": [], "states": []}
        self.write()

    def update(self, **kw):
        with self.lock:
            self.data.update(kw)
            self.write()

    def write(self):
        """Atomic replace with retries: on Windows a reader (the WebUI poll) can briefly lock the target."""
        with self.lock:
            self._write()

    def _write(self):
        payload = json.dumps(self.data)
        tmp = self.path.with_suffix(".tmp")
        for attempt in range(6):
            try:
                tmp.write_text(payload, encoding="utf-8")
                tmp.replace(self.path)
                return
            except OSError:
                time.sleep(0.02 * (attempt + 1))
        try:  # last resort: non-atomic write; a torn read is handled by the UI's JSON error check
            self.path.write_text(payload, encoding="utf-8")
        except OSError as e:
            log.debug("status write skipped: %s", e)


def _shift_logit(s: torch.Tensor, logit_shift: float) -> torch.Tensor:
    """Apply the curriculum shift in logit space to an arbitrary sigma sample (keeps it in (0, 1))."""
    if logit_shift == 0.0:
        return s
    s = s.clamp(1e-4, 1 - 1e-4)
    return torch.sigmoid(torch.log(s / (1 - s)) + logit_shift)


def sample_sigmas(cfg: TrainConfig, n: int, gen: torch.Generator, logit_shift: float = 0.0, tokens: int = 0,
                  arch=None) -> torch.Tensor:
    """tokens: the batch's image token count (H/16 x W/16), which the architectures' own recipes depend on."""
    mode = cfg.timestep_sampling
    if mode == "uniform":
        s = _shift_logit(torch.rand(n, generator=gen), logit_shift)
    elif mode == "reference":  # the model family's own training recipe (Bria's trainer / the MOSS-Image report)
        s = _shift_logit((arch or get_arch(cfg)).reference_sigmas(n, gen, tokens), logit_shift)
    elif mode == "logit_normal":
        s = torch.sigmoid(torch.randn(n, generator=gen) * cfg.logit_std + cfg.logit_mean + logit_shift)
    elif mode == "sigmoid":
        s = torch.sigmoid(torch.randn(n, generator=gen) * cfg.sigmoid_scale + logit_shift)
    elif mode == "logsnr":  # style-friendly (arXiv 2411.14793): log-SNR ~ N(mean, std); SNR = ((1 - s) / s)^2
        logsnr = torch.randn(n, generator=gen) * cfg.logsnr_std + cfg.logsnr_mean
        s = torch.sigmoid(-logsnr / 2 + logit_shift)
    elif mode == "identity":  # detail-heavy: logit-normal with mean -0.5 (median sigma ~0.38)
        s = torch.sigmoid(torch.randn(n, generator=gen) * 1.0 - 0.5 + logit_shift)
    elif mode == "content":  # ai-toolkit style cubic sampling toward low noise (structure/identity)
        s = _shift_logit(torch.rand(n, generator=gen) ** cfg.content_power, logit_shift)
    elif mode == "style":  # mirror: toward high noise
        s = _shift_logit(1 - torch.rand(n, generator=gen) ** cfg.content_power, logit_shift)
    elif mode == "shift":  # legacy alias: uniform + timestep_shift
        s = _shift_logit(torch.rand(n, generator=gen), logit_shift)
    else:
        raise ValueError(f"Unknown timestep_sampling {mode}")
    if cfg.timestep_shift != 1.0:
        s = cfg.timestep_shift * s / (1 + (cfg.timestep_shift - 1) * s)
    if cfg.sigma_min > 0.0 or cfg.sigma_max < 1.0:
        s = cfg.sigma_min + (cfg.sigma_max - cfg.sigma_min) * s
    # snap to the 1000-step training grid (sigma = k/1000, k in [1,1000])
    return (s * 1000).round().clamp(1, 1000) / 1000


def pairs_sigma(cfg: TrainConfig) -> bool:
    """Whether a step's micro-batches share one sigma (Keel's keel_pair_sigma)."""
    return cfg.optimizer.lower() in KEEL and bool(cfg.keel_pair_sigma)


def draw_sigmas(cfg: TrainConfig, n: int, gen: torch.Generator, shift: float, tokens: int, arch,
                step_sigma: torch.Tensor | None):
    """A micro-batch's sigmas: sample_sigmas, or the step's one sigma repeated when pairs_sigma(cfg) (drawn with the
    step's first micro-batch). Returns (sigma [n], the step's sigma to carry to the next micro-batch, or None)."""
    if step_sigma is not None:
        return step_sigma.expand(n).clone(), step_sigma
    sigma = sample_sigmas(cfg, n, gen, shift, tokens=tokens, arch=arch)
    if pairs_sigma(cfg):
        step_sigma = sigma[:1].clone()
        return step_sigma.expand(n).clone(), step_sigma
    return sigma, None


def loss_weight(cfg: TrainConfig, sigma: torch.Tensor) -> torch.Tensor:
    if cfg.loss_weighting == "none":
        return torch.ones_like(sigma)
    if cfg.loss_weighting == "sigma_sqrt":
        return (sigma ** -0.5).clamp(max=10.0) * 0.5
    if cfg.loss_weighting == "min_snr":
        snr = ((1 - sigma) / sigma) ** 2
        return (snr.clamp(max=5.0) / (snr + 1)).clamp(min=1e-3)
    raise ValueError(f"Unknown loss_weighting {cfg.loss_weighting}")


def build_optimizer(cfg: TrainConfig, params, pairs=None):
    """`params` may be a list of tensors or a list of param-group dicts (LoRA+). `pairs`: the (lora_down, lora_up)
    pairs among them (lora.lora_pairs), which Keel preconditions; the other optimizers ignore them."""
    name = cfg.optimizer.lower()
    if name == "adamw":
        return torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay, betas=(0.9, 0.99))
    if name == "adamw8bit":
        import bitsandbytes as bnb

        return bnb.optim.AdamW8bit(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay, betas=(0.9, 0.99))
    if name == "adafactor":
        from transformers.optimization import Adafactor

        return Adafactor(params, lr=cfg.learning_rate, scale_parameter=False, relative_step=False,
                         warmup_init=False, weight_decay=cfg.weight_decay)
    if name in AUTOMAGIC:
        if params and isinstance(params[0], dict):
            if {g.get("name") for g in params} <= {"lora", "lora_down", "lora_up"}:  # Automagic owns its lr: collapse
                if len(params) > 1:  # the LoRA+ groups
                    log.warning("lora_plus_ratio is ignored with %s (it adapts each group's lr itself)", name)
                params = [p for g in params for p in g["params"]]
            else:  # automagic_groups: by_type, each layer type adapts its own lr from the shared start
                params = [{"params": g["params"]} for g in params]
        kw = dict(lr=cfg.learning_rate, min_lr=cfg.automagic_min_lr, max_lr=cfg.automagic_max_lr,
                  weight_decay=cfg.weight_decay, polarity_history=cfg.automagic_polarity_history,
                  clip_threshold=cfg.automagic_clip_threshold, fused=False)
        if name in KEEL:
            from .optimizers.keel import Keel

            fb = name == "keel_fb"  # the face-and-body version: the body share and the magnitude vote
            opt = Keel(params, pairs=pairs, gauge=cfg.keel_gauge, gauge_eps=cfg.keel_gauge_eps,
                       cautious=cfg.keel_cautious, bands=cfg.keel_bands,
                       band_weights=keel_band_weights(cfg.keel_band_weights, cfg.keel_bands),
                       consensus=cfg.keel_consensus, consensus_power=cfg.keel_consensus_power,
                       consensus_mode=keel_consensus_mode(cfg.keel_consensus_mode, name),
                       lookahead=cfg.keel_lookahead, lookahead_pullback=cfg.keel_lookahead_pullback,
                       lookahead_lr=cfg.keel_lookahead_lr, lookahead_grace=cfg.keel_lookahead_grace,
                       face_gain=cfg.keel_face_gain, face_gain_max=cfg.keel_face_gain_max,
                       face_gain_beta=cfg.keel_face_gain_beta, body_weight=cfg.keel_body_weight if fb else 0.0,
                       sensitivity_quantile=cfg.automagic_sensitivity_quantile, vote=cfg.automagic_vote,
                       max_vote=cfg.automagic_max_lr_change, **kw)
            if fb and cfg.keel_lookahead and cfg.face_weight != cfg.body_weight:
                log.info("keel_fb: the lookahead follows the probes' combined score, weighted face %g / body %g "
                         "(face_weight, body_weight); set them equal for an equal focus", cfg.face_weight,
                         cfg.body_weight)
            return opt
        if name == "automagic4":
            from .optimizers.automagic4 import Automagic4

            return Automagic4(params, sensitivity_quantile=cfg.automagic_sensitivity_quantile, vote=cfg.automagic_vote,
                              max_vote=cfg.automagic_max_lr_change, **kw)
        from .optimizers.automagic3 import Automagic3

        return Automagic3(params, **kw)
    raise ValueError(f"Unknown optimizer {cfg.optimizer}")


def current_lr(optimizer, scheduler) -> float:
    """Adapted lr for self-tuning optimizers, scheduler lr otherwise."""
    if hasattr(optimizer, "get_avg_learning_rate"):
        return float(optimizer.get_avg_learning_rate())
    return float(scheduler.get_last_lr()[0])


def build_scheduler(cfg: TrainConfig, optimizer, offset: int = 0):
    """offset: the run's step this optimizer starts at (a fine-tune rebuilds its optimizer every window)."""
    warm = max(0, cfg.warmup_steps)
    if cfg.optimizer.lower() in AUTOMAGIC:  # self-tuning lr: no warmup/decay from outside
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)

    def fn(step):
        step += offset
        if step < warm:
            return (step + 1) / (warm + 1)
        p = (step - warm) / max(1, cfg.steps - warm)
        if cfg.lr_scheduler == "cosine":
            return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))
        if cfg.lr_scheduler == "linear":
            return max(0.0, 1 - p)
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


class Trainer(RaceParallelMixin):
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.job_dir = cfg.job_dir()
        setup_logging(self.job_dir)
        cfg.save(self.job_dir / "config.yaml")
        self.status = Status(self.job_dir / "status.json")
        self.run_id = history.new_run_id(cfg.job_name)  # this run's file in output/_history (Previous Runs tab)
        self.run_started = time.time()
        self.stop_file = self.job_dir / "STOP"
        if self.stop_file.exists():
            self.stop_file.unlink()
        self.device = torch.device(cfg.device)
        self.arch = get_arch(cfg)  # confirmed against the model's transformer config once the model dir is resolved
        self.vae = None
        # the sampling VAE waits in pinned RAM between rounds (_park_vae) instead of holding training memory:
        # vae_parked is its GPU while it is parked (None while it is on it), vae_park_rows its weights' pinned copies
        # [(module, name, is_buffer, CPU tensor)], made once (the VAE is frozen, nothing is copied back)
        self.vae_parked: str | None = None
        self.vae_park_rows: list | None = None
        self.kept: dict[int, list[int]] = {}  # id(model copy) -> blocks keeping their activations
        self.ema = None  # LoraEMA: the weight average saves, samples and probes use (lora_ema)
        self.resumed_ema = None  # its state from a resumed training state
        self.vaes: dict[str, object] = {}  # VAE copies on the other GPUs (sampling / caching side jobs)
        self.sample_graphs: set = set()  # (device, shape) pairs a compiled model has already traced for sampling
        self.face_scorer = None
        self.id_loss = None  # idloss.IdentityLoss when id_loss_weight > 0
        self.id_ref = None  # this step's reference embedding
        self.id_time, self.id_count = 0.0, 0  # identity-loss seconds and samples (overhead logging)
        self.body_scorer = None
        self.copy_checker = None  # copycheck.CopyChecker when copy_check is on
        self.controller = None
        self.batcher = None
        self.ts_logit_shift = 0.0
        self.item_face_frac: dict[str, float] = {}
        self.loss_watch = None
        self.curriculum_w: dict[str, float] | None = None
        self.all_items: list = []
        self.sides: list[int] = []
        self.res_stage = 0
        self.batch_iter = None
        self.face_detector = None
        self.face_analysis: dict[str, dict] = {}
        # attention localization / identity alignment (alignment.py): the trigger's token positions per cached text,
        # each photo's face embedding, the alignment's projector and its optimizer, and their values since the log
        self.stage_tag = ""  # "_reward" while the reward stage saves its files
        if cfg.reward_steps > 0 and not cfg.reward_prompts:
            from .reward import default_prompts

            cfg.reward_prompts = default_prompts(cfg.trigger_word, cfg.face_weight, cfg.body_weight)
        try:
            racing_selftrain = cfg.race and any(c.selftrain for _, c, _ in race_mod.variant_configs(cfg))
        except ValueError:  # bad race rows: validate() says so
            racing_selftrain = False
        if (cfg.selftrain or racing_selftrain) and not cfg.selftrain_prompts:
            from .reward import default_prompts

            cfg.selftrain_prompts = default_prompts(cfg.trigger_word, cfg.face_weight, cfg.body_weight)
        self.selftrain_steps: list[int] = []  # the steps a self-training round runs at (selftrain.py)
        self.trigger_pos: dict[str, list[int]] = {}
        self.face_embs: dict[str, torch.Tensor] = {}
        self.body_embs: dict[str, torch.Tensor] = {}
        self.align_layer: int | None = None
        self.id_aligner = None
        self.id_align_opt = None
        self.align_values: dict[str, list[float]] = {"attn": [], "id": [], "body": [], "pairs": []}
        if not (0.0 <= cfg.sigma_min < cfg.sigma_max <= 1.0):
            raise ValueError("sigma_min/sigma_max must satisfy 0 <= sigma_min < sigma_max <= 1")
        self.devices: list[str] = [cfg.device]
        self.dp = None  # DataParallelLoRA when multi_gpu == data_parallel
        self.balancer = None  # multigpu.Balancer: batches per model copy per step (dp_balance)
        self.balance_hinted = False  # the dp_balance suggestion was logged
        self.resumed_dp_counts = None  # its split from a resumed training state
        self.race = None  # race.RaceRun while a configuration race runs (and after, for its report)
        self.last_probe = None  # (face, body, images, per-seed scores) of the last probe
        self.probe_worker = None  # probeworker.ProbeProcess: probes render on a second GPU (probe_async)
        self.race_base = None  # the run's own config when racing (the configurations change parts of it)
        self.race_rows = None  # [(name, config, overrides)] of the race
        self.race_layouts = None  # [(blocks, LoRA module names)] per configuration
        self.race_union_names = None  # the LoRA wrappers: every configuration's modules together
        self.race_region_dirs: dict[str, Path | None] = {}  # configuration -> its own region-map folder
        self.race_id_loss = None  # the identity loss shared by the configurations that use it
        self.race_parallel = False  # race_parallel: one configuration per GPU at the same time (race_parallel.py)
        self.look_outliers: set[str] = set()  # look_warmup: the look filter's outlier images (_look_outlier_images)
        self.isolation = None  # LoRA Isolation's subjects (isolation.py) when it is on
        self.dop_values: list[float] = []  # DOP losses since the last log line
        self.ft = None  # finetune.FineTuner while a full fine-tune runs
        self.block_swap = None  # blockswap.BlockSwap when block_swap is on
        self.ft_rotate_steps = 1
        self.ft_pending_save = False
        self.ft_window = 0
        self.ft_rotated_at = -1  # the step of the last rotation
        self.compiled = False  # torch.compile active on the DiT blocks
        self.compile_dirs: list = []  # compile_cache folders (the first is written)
        self.dp_quiet_steps = 0  # consecutive data-parallel steps without a new compile
        self.replicas: list[tuple] = []  # (model, device) for every training copy, primary first
        if cfg.multi_gpu in ("data_parallel", "model_split"):
            from .multigpu import parse_devices

            self.devices = parse_devices(cfg.devices, cfg.device)
            if len(self.devices) < 2:
                log.warning("multi_gpu=%s needs at least two devices in 'devices'; running on %s only",
                            cfg.multi_gpu, self.devices[0])
                cfg.multi_gpu = "off"
            cfg.device = self.devices[0]
            self.device = torch.device(cfg.device)
        self.model_dir: Path | None = None

    def negative_prompt(self) -> str:
        """The sample / probe negative prompt: the config's, or the architecture's default (FIBO 2.0: the null caption)."""
        return self.cfg.sample_negative or self.arch.default_negative

    def side_devices(self) -> list[str]:
        """GPUs that side jobs (samples, probes, caching, SAM 3 masks) are split across."""
        cfg = self.cfg
        if cfg.parallel_jobs and cfg.multi_gpu != "off" and len(self.devices) > 1:
            return [norm_device(d) for d in self.devices]
        return [norm_device(self.device)]

    def _vae_on(self, device: str):
        """The VAE on `device` (the primary one, or a cached copy made on first use)."""
        device = norm_device(device)
        if self.vae is None:
            return None
        if getattr(self, "vae_parked", None) == device:
            self._unpark_vae()
        if norm_device(self.vae.device) == device:
            return self.vae
        if device not in self.vaes:
            self.vaes[device] = clone_to(self.vae, device).eval().requires_grad_(False)
        return self.vaes[device]

    @torch.no_grad()
    def _park_vae(self) -> None:
        """The sampling VAE to pinned RAM until the next render needs it (_vae_on brings it back): it is used only
        for samples, probes and other decodes, and on the GPU it would cost the training steps that memory. Not with
        the identity loss (its decoder may be this VAE, used every step) or a VAE that is not on a GPU."""
        from .blockswap import _set

        if getattr(self, "vae", None) is None or getattr(self, "vae_parked", None) is not None \
                or getattr(self, "id_loss", None) is not None:
            return
        home = norm_device(self.vae.device)
        if not home.startswith("cuda"):
            return
        if getattr(self, "vae_park_rows", None) is None:
            rows = []
            for mod in self.vae.modules():
                tensors = [(n, False, p) for n, p in mod.named_parameters(recurse=False) if p is not None]
                tensors += [(n, True, b) for n, b in mod.named_buffers(recurse=False) if b is not None]
                for name, is_buf, t in tensors:
                    cpu = t.detach().to("cpu")
                    try:
                        cpu = cpu.pin_memory()
                    except RuntimeError:  # no pinned memory left: pageable works, just slower to copy
                        pass
                    rows.append((mod, name, is_buf, cpu))
            self.vae_park_rows = rows
        for mod, name, is_buf, cpu in self.vae_park_rows:
            _set(mod, name, is_buf, cpu)  # the GPU copies are freed (the allocator reuses them for training)
        self.vae_parked = home

    def _home_vae(self):
        """The sampling VAE itself, on its own GPU (brought back from RAM if it is parked)."""
        self._unpark_vae()
        return self.vae

    @torch.no_grad()
    def _unpark_vae(self) -> None:
        from .blockswap import _set

        if getattr(self, "vae_parked", None) is None:
            return
        dev = torch.device(self.vae_parked)
        for mod, name, is_buf, cpu in self.vae_park_rows:
            _set(mod, name, is_buf, cpu.to(dev, non_blocking=cpu.is_pinned()))
        self.vae_parked = None

    # ------------------------------------------------------------ stages
    def _needs_face_analysis(self) -> bool:
        cfg = self.cfg
        return bool(cfg.wants_regions() or (cfg.crop_augment or "none") != "none" or cfg.likeness_curriculum
                    or (cfg.isolation and cfg.isolation_pairs > 0))  # pairs crop the subject around the face

    def _ensure_face_detector(self):
        """A FaceScorer used as a detector (no references needed); shared with the likeness scorer."""
        if self.face_detector is not None:
            return self.face_detector
        from . import facesim

        ok, msg = facesim.available()
        if not ok:
            log.warning("%s; face-based features (regions/crops/curriculum) disabled", msg)
            return None
        try:
            self.face_detector = facesim.FaceScorer(self.cfg.face_provider, self.cfg.face_det_size,
                                                    progress=lambda t, f: self.status.update(stage=t))
        except Exception as e:  # noqa: BLE001
            log.warning("Could not initialise the face detector (%s); face-based features disabled", e)
        return self.face_detector

    def analyze_dataset(self, items):
        """Face analysis cache, crop augmentation and region weight maps. Returns the (possibly extended) items."""
        from . import regions as R

        cfg = self.cfg
        if not self._needs_face_analysis():
            return items
        detector = self._ensure_face_detector()
        cache = cfg.resolved_cache_dir()
        self.status.update(stage="analysing faces in the dataset")
        self.face_analysis = R.analyze_faces(items, detector, cache / "faces.json",
                                             progress=lambda i, n: self.status.update(step=i, total=n),
                                             recache=cfg.recache, embeddings=cfg.id_align)
        if cfg.id_align:
            self.face_embs = {k: torch.tensor(r["emb"], dtype=torch.float32)
                              for k, r in self.face_analysis.items() if r.get("emb")}
            log.info("Identity alignment: %d of %d source images have a face embedding", len(self.face_embs),
                     len(self.face_analysis))
        if cfg.body_align:
            self._analyze_bodies(items, cache)
        n_face = sum(1 for r in self.face_analysis.values() if r.get("bbox"))
        log.info("Face analysis: %d of %d source images have a face", n_face, len(self.face_analysis))
        if (cfg.crop_augment or "none") != "none":
            items = augment_items(items, cfg, self.face_analysis)
        if cfg.wants_regions():
            person_fn = None
            holder = []
            if not cfg.wants_person_masks():
                pass  # identity loss / alignment only: face cells + landmarks are enough, no person masks
            elif cfg.body_backend == "mediapipe+sam3":
                try:
                    from .bodysim import SilhouetteScorer

                    sil = SilhouetteScorer(cfg.sam3_model, cfg.body_device,
                                           progress=lambda t, f: self.status.update(stage=t))
                    holder.append(sil)
                    person_fn = sil.person_mask
                except Exception as e:  # noqa: BLE001
                    log.warning("SAM 3 unavailable for person masks (%s); trying MediaPipe", e)
            if person_fn is None and cfg.wants_person_masks():
                try:
                    from .bodysim import BodyScorer, available

                    if available()[0]:
                        bs = BodyScorer(cfg.models_dir, progress=lambda t, f: self.status.update(stage=t))
                        holder.append(bs)

                        def person_fn(img, _bs=bs):
                            import tempfile

                            from .regions import person_mask_from_landmarks

                            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                                img.save(tf.name)
                                lm = _bs.landmarks(tf.name)
                            Path(tf.name).unlink(missing_ok=True)
                            if lm is None:
                                return None
                            return person_mask_from_landmarks(lm[0], lm[1], img.width, img.height)
                    else:
                        log.warning("No person-mask backend (mediapipe/SAM 3): body_loss_weight ignored")
                except Exception as e:  # noqa: BLE001
                    log.warning("MediaPipe unavailable for person masks (%s); body_loss_weight ignored", e)
            self.status.update(stage="building region weight maps")
            fns = None
            side = self.side_devices()
            sam_gpu = bool(holder and person_fn == getattr(holder[0], "person_mask", None)
                           and holder[0].device != "cpu")
            if sam_gpu and len(side) > 1:  # SAM 3: one copy per GPU
                for d in [d for d in side if d != norm_device(holder[0].device)]:
                    self.status.update(stage=f"copying SAM 3 to {d}")
                    holder.append(holder[0].clone(d))
                fns = [(h.person_mask, h.device) for h in holder]
            with contextlib.ExitStack() as stack:
                if sam_gpu:  # keep SAM 3 resident for the whole batch instead of moving it per image
                    for h in holder:
                        stack.enter_context(h.hold())
                for rcfg, ritems in self._region_jobs(items):
                    counts = R.cache_region_maps(ritems, rcfg, self.face_analysis, person_fn,
                                                 progress=lambda i, n: self.status.update(step=i, total=n),
                                                 person_mask_fns=fns)
                    log.info("Region maps ready (new this run): %s; weights face %.2g / body %.2g / background %.2g",
                             counts, rcfg.face_loss_weight, rcfg.body_loss_weight, rcfg.background_loss_weight)
            self.status.update(regions=counts)
            for obj in holder:
                unload_model(obj.model if hasattr(obj, "model") else obj)
            free_cuda()
        return items

    def _region_jobs(self, items) -> list[tuple]:
        """(config, items) per set of region weight maps to build: the run's own; in a race one per distinct set of
        region weights among the configurations that use maps (the run's own set in the usual folder, others in
        their own folder of the cache, remembered in race_region_dirs)."""
        if not self.race_rows:
            return [(self.cfg, items)]
        jobs, folders = [], {}
        own = race_mod.region_key(self.cfg)
        for name, vcfg, _ in self.race_rows:
            if not vcfg.wants_regions():
                continue
            key = race_mod.region_key(vcfg)
            if key not in folders:
                folder = None if key == own else self.cfg.resolved_cache_dir() / race_mod.region_dir(vcfg)
                folders[key] = folder
                if folder is not None:
                    folder.mkdir(parents=True, exist_ok=True)
                jobs.append((vcfg, items if folder is None else race_mod.variant_items(items, dataclasses.replace(
                    vcfg, flip_augment=True, crop_augment="face+bust"), folder)))
            self.race_region_dirs[name] = folders[key]
        return jobs

    def prepare_cache(self, items):
        cfg = self.cfg
        prompts = list(cfg.sample_prompts) + ([self.negative_prompt()] if cfg.sample_prompts else [])
        if cfg.probe_every > 0:
            prompts += [p for p in (cfg.probe_face_prompt, cfg.probe_body_prompt) if p] + [self.negative_prompt()]
        if cfg.reward_steps > 0:
            prompts += list(cfg.reward_prompts) + [self.negative_prompt()]
        if cfg.selftrain:
            prompts += list(cfg.selftrain_prompts) + [self.negative_prompt()]
        need_latents = cfg.recache or any(
            not it.latent_path.exists() or (it.latent_flip_path and not it.latent_flip_path.exists()) for it in items)
        need_text = cfg.recache or not dropout_text_path(cfg).exists() or any(
            not it.text_path.exists() or (it.dop_text_path is not None and not it.dop_text_path.exists())
            for it in items) or any(
            not text_cache_path(cfg, p).exists() for p in prompts)
        devices = self.side_devices()
        if need_latents:
            self.status.update(stage="caching latents")
            vae = self.arch.get_vae(cfg, self.model_dir)
            vaes = [vae]
            for d in devices:
                if d != norm_device(vae.device):
                    vaes.append(clone_to(vae, d).eval().requires_grad_(False))
            t0 = time.time()
            cache_latents(items, cfg, vae, progress=lambda i, n: self.status.update(step=i, total=n),
                          vaes=vaes if len(vaes) > 1 else None)
            log.info("Latent caching took %.1fs", time.time() - t0)
            for v in vaes:
                unload_model(v)
            del vaes, vae
            free_cuda()
        if need_text:
            self.status.update(stage="loading text encoder")
            t_load = time.time()
            te, tok = self.arch.get_text_encoder(cfg, self.model_dir)
            t_load = time.time() - t_load
            self.status.update(stage="caching text embeddings")
            t0 = time.time()
            encoders = self._extra_text_encoders(te, tok, items, prompts, devices, t_load)
            for subject in self.isolation or []:  # LoRA Isolation: does the fused word keep the class token whole?
                from .isolation import fusion_warning

                warn = fusion_warning(tok, subject)
                if warn:
                    log.warning("LoRA Isolation: %s", warn)
            cache_text(items, cfg, te, tok, extra_prompts=prompts,
                       progress=lambda i, n: self.status.update(step=i, total=n),
                       encoders=encoders if len(encoders) > 1 else None, encode_batch_fn=self.arch.encode_prompts)
            log.info("Text caching took %.1fs", time.time() - t0)
            for enc in encoders:
                unload_model(enc[0])
            del te, tok, encoders
            free_cuda()
            log.info("Text encoder unloaded after caching")

    def _extra_text_encoders(self, te, tok, items, prompts, devices, load_seconds: float) -> list[tuple]:
        """[(te, tokenizer, device), ...]: the loaded encoder plus a copy on every other GPU when it pays off.

        A copy only fits when the encoder sits wholly on one GPU (a quantized Qwen3.5-9B, ~10 GB; bf16 is 19 GB and
        is spread over several devices instead). Whether it pays off is measured: the first few prompts are encoded
        on the loaded copy, and a second copy is loaded only if the time it saves on the remaining prompts beats
        its load time (taken from how long the first copy took to load)."""
        import dataclasses

        from .dataset import text_jobs

        tensors = list(te.parameters()) + list(te.buffers())
        te_devs = {norm_device(t.device) for t in tensors if t.device.type != "meta"}
        primary = [(te, tok, next(iter(te_devs)) if len(te_devs) == 1 else "cpu")]
        if len(devices) < 2:
            return primary
        todo, _ = text_jobs(items, self.cfg, prompts)
        if len(te_devs) != 1 or not next(iter(te_devs)).startswith("cuda"):
            log.info("Text caching stays on one encoder: it is spread over %s (a quantized text encoder fits one GPU "
                     "per copy)", ", ".join(sorted(te_devs)))
            return primary
        if len(todo) <= TEXT_TIMING_PROMPTS * 2:
            return primary
        stamps = [(time.time(), 0)]  # (time, prompts done) after each batch
        cache_text(items, self.cfg, te, tok, extra_prompts=prompts, max_prompts=TEXT_TIMING_PROMPTS,
                   progress=lambda i, n: (stamps.append((time.time(), i)), self.status.update(step=i, total=len(todo))),
                   encode_batch_fn=self.arch.encode_prompts)
        (t0, _), (t1, n1), (t_end, n_end) = stamps[0], stamps[min(1, len(stamps) - 1)], stamps[-1]
        per_prompt = (t_end - t1) / max(1, n_end - n1)  # skips the first (warm-up) batch
        warm_up = max(0.0, t1 - t0 - per_prompt * n1)  # first-call cost, which a new copy pays again
        cost = load_seconds + warm_up
        remaining = len(todo) - TEXT_TIMING_PROMPTS
        saved = remaining * per_prompt * (1 - 1 / len(devices))
        if saved < cost * 1.2:
            log.info("Text caching stays on one encoder: %d prompts left at %.2fs each; a second copy would save "
                     "~%.0fs but costs ~%.0fs (load %.0fs + first-call warm-up %.0fs)", remaining, per_prompt, saved,
                     cost, load_seconds, warm_up)
            return primary
        log.info("Text caching: %d prompts left at %.2fs each; an encoder copy per GPU saves ~%.0fs (costs ~%.0fs)",
                 remaining, per_prompt, saved, cost)
        size = sum(t.numel() * t.element_size() for t in tensors)
        out = primary
        for d in devices:
            if d == primary[0][2]:
                continue
            free, _ = torch.cuda.mem_get_info(torch.device(d))
            if free < size * 1.15 + 1.5 * 2**30:
                log.info("Text caching: not enough free memory on %s for a second encoder (%.1f GB free, %.1f GB "
                         "needed)", d, free / 2**30, size / 2**30)
                continue
            self.status.update(stage=f"loading text encoder copy on {d}")
            te2, tok2 = self.arch.get_text_encoder(dataclasses.replace(self.cfg, text_encoder_device=d), self.model_dir)
            out.append((te2, tok2, d))
        return out

    def run(self):
        try:
            self._run()
        except Exception as e:  # noqa: BLE001
            log.exception("Training failed")
            self.status.update(stage="error", error=f"{type(e).__name__}: {e}", done=True)
            self._save_history("error")
            raise
        finally:
            self._close_probe_worker()

    def _run(self):
        cfg = self.cfg
        cfg.validate()
        torch.manual_seed(cfg.seed)
        rng = random.Random(cfg.seed)
        gen = torch.Generator("cpu").manual_seed(cfg.seed)
        from .comfy_files import resolve_downloads

        self.status.update(stage="model files")
        cfg = self.cfg = resolve_downloads(cfg, progress=lambda text, frac=None: self.status.update(stage=text))
        self.arch = get_arch(cfg)  # a downloaded single file names its own family
        self.status.update(stage="resolving model")
        Q.set_int8_defaults(cfg.int8_matmul, cfg.int8_backward)  # every QuantLinear built from here on
        Q.set_nvfp4_default(cfg.nvfp4_matmul, cfg.nvfp4_backward)
        self.model_dir = self.arch.resolve_for_config(cfg)
        self.arch = detect_arch(cfg, self.model_dir)
        cfg.model_arch = self.arch.name
        log.info("Model architecture: %s", self.arch.label)
        attention.set_backend(cfg.attention_backend, cfg.device)
        log.info("Attention backend: %s%s", attention.backend_name(),
                 " (auto: cuDNN on a Blackwell GPU)" if cfg.attention_backend == "auto"
                 and attention.backend_name() == "cudnn" else "")
        alloc = os.environ.get("PYTORCH_ALLOC_CONF") or os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
        if alloc:
            log.info("CUDA allocator: %s", alloc)
        if not cfg.lora_targets:
            cfg.lora_targets = list(self.arch.default_targets)
            log.info("LoRA targets: the %s default (%s)", self.arch.label, ", ".join(cfg.lora_targets))
        if cfg.race:  # the dataset is prepared for whatever any configuration of the race needs
            self.race_base = cfg
            cfg = self.cfg = self._race_prep_config(cfg)
        items = scan_dataset(cfg)
        log.info("Dataset: %d images in %s", len({it.image for it in items}),
                 ", ".join(spec.path for spec in cfg.dataset_specs()))
        items = self.analyze_dataset(items)
        self.look_outliers = self._look_outlier_images(cfg, items)
        if cfg.isolation:
            items = self._isolation_prepare(cfg, items)
        self.prepare_cache(items)

        self.status.update(stage="loading transformer", step=0, total=cfg.steps)
        # block swap: loaded in RAM, the swapped blocks' frozen weights stay there and the rest moves to the GPU
        model = self.arch.get_transformer(cfg, self.model_dir, device="cpu" if cfg.block_swap > 0 else None)
        model.gradient_checkpointing = cfg.gradient_checkpointing
        n_quant = Q.configure_int8(model, cfg.int8_matmul, cfg.int8_backward)
        n_fp4 = Q.configure_nvfp4(model, cfg.nvfp4_matmul, cfg.nvfp4_backward)
        if n_quant - n_fp4:
            log.info("%d quantized Linear layers: int8 matmul %s (backward %s)", n_quant - n_fp4,
                     "on" if cfg.int8_matmul else "off", cfg.int8_backward)
        if n_fp4:
            self._log_nvfp4(n_fp4)
        blocks = lora_mod.parse_blocks(cfg.lora_blocks, len(model.layers))
        if self.race_rows:  # wrappers over every configuration's modules
            blocks = self._race_lora_layout(model)
        unknown = [t for t in cfg.lora_targets if t not in self.arch.target_groups] if not self.race_rows else []
        if unknown:
            log.warning("LoRA targets %s do not exist for %s and are ignored", unknown, self.arch.label)
        if cfg.finetune:  # the targets' own weights train, by rotation (finetune.py); no adapter
            from .finetune import FineTuner

            injected = {}
            names = self._lora_names(model, blocks)
            if not names:
                raise ValueError("No layers selected to fine-tune: check lora_targets / lora_blocks")
            self.ft = FineTuner(model, names, self.devices[0], log)
            backprop = None  # a window spans every block: the backward runs through all of them
        else:
            injected = lora_mod.inject_lora(model, cfg.lora_rank, cfg.lora_alpha, cfg.lora_targets,
                                            include_text_embedder=cfg.train_text_embedder, dropout=cfg.lora_dropout,
                                            blocks=blocks, names=self._lora_names(model, blocks),
                                            kind=cfg.network_type, factor=cfg.lokr_factor,
                                            block=cfg.oft_block_size, constraint=cfg.oft_constraint, dora=cfg.dora)
            backprop = lora_mod.backprop_blocks(model, injected, blocks)  # blocks the backward runs from (None = all)
        if blocks is not None:
            log.info("LoRA blocks: %s (%d of %d); %s", lora_mod.format_blocks(blocks), len(blocks), len(model.layers),
                     f"backward skips blocks before {min(blocks) if blocks else 'n/a'}" if backprop is not None else
                     "LoRA before the blocks too (text fusion / projections), so the backward runs through every block")
        if not injected and not cfg.finetune:
            raise ValueError("No LoRA modules selected: check lora_targets / lora_blocks")
        self._attach_training_adapter(model, log_it=True)
        self.replicas = [(model, injected, self.devices[0])]
        if cfg.multi_gpu == "data_parallel":
            from .multigpu import DataParallelLoRA

            reps = []
            for dev in self.devices[1:]:
                self.status.update(stage=f"loading model replica on {dev}")
                m = self.arch.get_transformer(cfg, self.model_dir, device=dev)
                m.gradient_checkpointing = cfg.gradient_checkpointing
                Q.configure_int8(m, cfg.int8_matmul, cfg.int8_backward)
                Q.configure_nvfp4(m, cfg.nvfp4_matmul, cfg.nvfp4_backward)
                inj = lora_mod.inject_lora(m, cfg.lora_rank, cfg.lora_alpha, cfg.lora_targets,
                                           include_text_embedder=cfg.train_text_embedder, dropout=cfg.lora_dropout,
                                           blocks=blocks, names=self._lora_names(m, blocks),
                                           kind=cfg.network_type, factor=cfg.lokr_factor,
                                           block=cfg.oft_block_size, constraint=cfg.oft_constraint, dora=cfg.dora)
                self._attach_training_adapter(m)
                reps.append((m, inj, dev))
                self.replicas.append((m, inj, dev))
            self.dp = DataParallelLoRA(injected, reps)
            log.info("Data parallel on %s: %d images per step, LoRA grads averaged (dp_balance %s)",
                     ", ".join(self.devices), cfg.batch_size * cfg.gradient_accumulation * len(self.devices),
                     cfg.dp_balance)
            # measures in every mode (the log reports each GPU's speed); dp_balance off never changes the split
            self.balancer = Balancer(len(self.replicas), cfg.gradient_accumulation, cfg.dp_balance)
        elif cfg.multi_gpu == "model_split":
            from .multigpu import split_model

            split_model(model, self.devices, trained=backprop, plan=cfg.split_plan)
        if cfg.resume_from:
            n = lora_mod.load_lora_weights(injected, cfg.resume_from)
            log.info("Resumed %d LoRA modules from %s", n, cfg.resume_from)
        if cfg.block_swap > 0:
            self._setup_block_swap(cfg, model)
        if self.ft is not None:
            params, optimizer, scheduler = self._finetune_setup(cfg, items)
        elif self.race_rows is None:
            params = list(lora_mod.lora_parameters(injected))
            n_params = sum(p.numel() for p in params)
            log.info("%s: %d modules, %.2fM trainable params%s", lora_mod.adapter_label(
                cfg.network_type, cfg.lora_rank, cfg.lora_alpha, cfg.lokr_factor, cfg.oft_block_size, cfg.dora,
                cfg.oft_constraint), len(injected), n_params / 1e6,
                " (LoKr: w1 up to factor x factor per layer, w2 the rest)" if cfg.network_type == "lokr" else
                " (OFT: output channels rotated in blocks)" if cfg.network_type == "oft" else "")
            optimizer, scheduler = self._make_optimizer(cfg, injected)
        else:  # each configuration of the race builds its own on its first turn
            params, optimizer, scheduler = [], None, None
        step = 0
        if cfg.resume_state:
            step = self.load_state(cfg.resume_state, injected, optimizer, scheduler, gen, rng)
            log.info("Resumed training state from %s at step %d", cfg.resume_state, step)
        if cfg.lora_ema > 0 and self.ft is not None:
            log.info("Weight average (lora_ema) is not kept for a fine-tune; saves extract the deltas as they are")
        if cfg.lora_ema > 0 and self.race_rows is None and self.ft is None:
            per_step = cfg.batch_size * cfg.gradient_accumulation * len(self.replicas)
            self.ema = self._make_ema(cfg, injected)
            if self.resumed_ema is not None:
                self.ema.load_state_dict(self.resumed_ema)
                log.info("Weight average restored from the training state (%d images averaged)", self.ema.images)
            elif cfg.resume_state:
                log.info("Weight average starts from the resumed weights (the state has none)")
            log.info("Weight average (EMA) of the LoRA for saves, samples and probes: decay %g per image (%.4f per "
                     "step of %d images), %.0f MB", cfg.lora_ema, self.ema.steady_step_decay(per_step), per_step,
                     self.ema.nbytes() / 2**20)
        if self.balancer is not None and self.resumed_dp_counts and len(self.resumed_dp_counts) == self.balancer.n:
            self.balancer.counts, self.balancer.decided_at = list(self.resumed_dp_counts), step
            log.info("Data parallel balance from the training state: %s batches per step",
                     "+".join(map(str, self.balancer.counts)))
        self.injected = injected
        self.sample_model = None
        if cfg.sample_model_path and self.ft is not None:
            log.warning("sample_model_path is ignored for a fine-tune: samples and probes render the fine-tuned model "
                        "itself")
            cfg.sample_model_path = ""
        if not cfg.sample_model_path:
            for m, _, _ in self.replicas:
                self._attach_sampling_loras(m)
        self._check_sample_settings()
        if cfg.sample_prompts or cfg.probe_every > 0 or cfg.id_loss_weight > 0 or cfg.reward_steps > 0 \
                or cfg.selftrain or cfg.soup_files or cfg.noise_files:
            self.vae = self.arch.get_vae(cfg, self.model_dir)
            self.face_scorer = self._make_face_scorer()
            self.body_scorer = self._make_body_scorer()
            self.controller = self._make_controller(items)
        self.id_loss = self._make_id_loss()
        self._park_vae()  # to RAM until the first sample or probe (kept on the GPU with the identity loss)
        if torch.cuda.is_available():
            from .multigpu import gpu_memory_summary

            log.info("GPU memory after model load: %s", gpu_memory_summary(self.devices))
        side = self.side_devices()
        if len(side) > 1:
            if cfg.multi_gpu == "model_split" and not cfg.sample_model_path:
                log.info("Samples and probes render on the split model one at a time (it spans %s); set a separate "
                         "sampling model to give each GPU its own images", ", ".join(side))
            else:
                log.info("Samples and probes split across %s (each GPU renders the next image in the queue)",
                         ", ".join(side))

        self._setup_alignment(cfg, items)
        do_compile, why = compiling.decide(cfg.compile_model, cfg.steps - step, self.devices)
        if self.race_parallel and do_compile:
            do_compile, why = False, "race_parallel trains the model copies in threads at once"
        if self.ft is not None and do_compile:
            do_compile, why = False, "a fine-tune swaps its layers in and out of the model"
        if self.block_swap is not None and do_compile:
            do_compile, why = False, "block swap streams the blocks' weights in and out"
        if do_compile and any(c.optimizer.lower() in KEEL and c.keel_face_gain > 0 for c in
                              ([r[1] for r in self.race_rows] if self.race_rows is not None else [cfg])):
            do_compile, why = False, "Keel's face-token gain hooks the LoRA layers' outputs (keel_face_gain)"
        if why:
            log.info("torch.compile %s (%s)", "on" if do_compile else "off", why)
        if do_compile:
            self.compile_dirs = compiling.cache_dirs(cfg.compile_cache, cfg.models_dir)
            got = compiling.load_cache(self.compile_dirs, self.replicas[0][2])
            if got is not None:
                log.info("compile_cache: loaded %s", got)
            for m, _, _ in self.replicas:
                self.compiled = self.compile(m, fall_back=cfg.compile_model == "auto",
                                             shape_swaps=self.race_rows is not None) or self.compiled
            if self.compiled and self.dp is not None:
                log.info("Data parallel + compile: replicas run one after another until compilation settles "
                         "(%d steps without a new graph), then in parallel", DP_COMPILE_QUIET_STEPS)
        self.all_items = items
        self.copy_checker = self._make_copy_checker(items)
        self.sides = cfg.training_sides()
        if cfg.resolution_step_up:
            if self.controller is None:
                raise ValueError("resolution_step_up needs working likeness probes (face/body references and their "
                                 "scoring packages); the controller could not be created")
            if len(self.sides) < 2:
                log.warning("resolution_step_up with a single resolution (%s): nothing to step up to; the stop "
                            "patience still applies", describe_sides(cfg))
            self.controller.enter_stage(self.res_stage)
            log.info("Resolution step-up: %s; next resolution after %d probe(s) without improvement, stop after %d "
                     "at the highest", describe_sides(cfg), cfg.step_up_patience, cfg.step_up_stop_patience)
        elif len(self.sides) > 1:
            log.info("Training at %s mixed in every epoch", describe_sides(cfg))
        losses: list[float] = []
        id_losses: list[list] = []  # [step, mean id loss] for steps that had eligible samples
        n_dev = len(self.replicas)

        def bind(v):
            """The loop's per-configuration values from the race configuration that trains now."""
            nonlocal cfg, injected, params, optimizer, scheduler, gen, rng, step, losses, id_losses
            nonlocal true_per_image_lr, id_w
            st = v.state
            cfg, injected, params = v.cfg, st["injected"], st["params"]
            optimizer, scheduler, gen, rng = st["optimizer"], st["scheduler"], st["gen"], st["rng"]
            step, losses, id_losses = st["step"], st["losses"], st["id_losses"]
            true_per_image_lr, id_w = st["true_per_image_lr"], st["id_w"]

        if self.race_rows is None:
            self._enter_stage(self.res_stage, step, initial=True)
            self.loss_watch = self._make_loss_watch(items, optimizer)
            true_per_image_lr = (self.loss_watch is not None and cfg.per_image_lr and cfg.batch_size == 1
                                 and cfg.gradient_accumulation == 1 and len(self.replicas) == 1)
            id_w = idloss_weight(cfg.id_loss_weight)
        else:
            self.race_id_loss = self.id_loss
            self._race_setup(injected, items)
            if self.race_parallel:
                self._race_parallel_main(items, backprop)
                return
            bind(self.race.active)
        if self.dp is not None:
            self.dp.sync_weights()
        if cfg.selftrain:
            self._selftrain_setup(step)
        if cfg.soup_files:  # a greedy soup of finished LoRAs instead of training
            self._soup_stage(model)
            return
        if cfg.noise_files:  # the probes' own noise, measured on finished LoRAs instead of training
            self._noise_stage(model)
            return
        self._setup_probe_worker()
        if cfg.sample_prompts and cfg.sample_before_training and step == 0:
            self.sample_images(model, step)
        if self.controller is not None and cfg.probe_at_start and step == 0 and self.probe_worker is not None:
            self._probe_async_start(step, injected)  # the baseline renders on the probe GPU while training starts
        elif self.controller is not None and cfg.probe_at_start and step == 0:
            self.run_probe(model, step, optimizer, scheduler, injected)
            if self.race is not None:  # every configuration starts from the same base model: one baseline for all
                for v in self.race.variants[1:]:
                    v.state["controller"].update(0, *self.last_probe)
        model.train()
        if self.dp is not None:
            self.dp.train(True)
        self._setup_checkpointing(items, backprop)
        self.status.update(stage="training", step=step)
        start = time.time()
        start_step = step
        compute_done = 0  # optimizer steps of every configuration (a race trains several)
        step_times: list[float] = []  # seconds per optimizer step, training only (no saves, samples or probes)
        profile_at = profiling.requested_step()

        def work(job):
            """Forward + backward for one model copy on its device. Returns (loss value, per-image multipliers,
            (identity loss value, samples, seconds) or None)."""
            (rmodel, rinj, rdev), batch, latents, embeds, mask, sigma, noise, regions, cur_step, n_total, dop = job
            observe = getattr(optimizer, "observe", None)
            if observe is not None:  # Keel: this micro-batch's noise level (the CPU tensor: no sync); a replica's
                observe(sigma, hook=rmodel is model)  # gradient reaches take_replica, not the hook
            latents = latents.to(rdev, torch.float32)
            embeds, mask, sigma, noise = embeds.to(rdev), mask.to(rdev), sigma.to(rdev), noise.to(rdev)
            s = sigma.view(-1, 1, 1, 1)
            noisy = (1 - s) * latents + s * noise
            target = noise - latents
            noisy_in = noisy.to(torch.bfloat16)
            taps = getattr(rmodel, "_rz_taps", None)
            use_attn = bool(cfg.attn_localize and taps is not None and regions is not None)
            use_align = bool((cfg.id_align or cfg.body_align) and taps is not None and regions is not None
                             and rmodel is model and self.id_aligner is not None)
            face_sink = getattr(optimizer, "face_energy", None) if rmodel is model else None  # Keel's face gain
            use_body = bool(face_sink is not None and getattr(optimizer, "body_weight", 0.0) > 0)  # keel_fb
            use_face = bool(face_sink is not None and taps is not None and regions is not None
                            and any(rm.face_cells is not None or (use_body and rm.has_person) for rm in regions.meta))
            if use_attn or use_align or use_face:  # this forward records the trigger's attention / the face tokens'
                taps.begin([self.trigger_pos.get(p, []) for p in regions.text_paths] if use_attn  # states / layout
                           else [[] for _ in batch], attention=use_attn)
            if face_sink is not None:  # the face tap: this batch's face (and body) tokens per sample (disarmed
                hh, ww = regions.prior.shape[-2:] if use_face else (0, 0)  # without a face or a body)
                lora_mod.arm_face_tap(rinj, [lora_mod.face_token_mask(rm.face_cells, hh, ww) for rm in regions.meta]
                                      if use_face else None, lambda: taps.layout, face_sink,
                                      body_masks=[lora_mod.body_token_mask(rm, hh, ww) for rm in regions.meta]
                                      if use_face and use_body else None)
            pred = rmodel(noisy_in, sigma * 1000.0, embeds, mask).float()
            aux = None
            if use_attn or use_align:
                taps.stop()  # later forwards (the prior pass, the backward's recompute) record nothing
                aux = self._alignment_losses(cfg, taps, batch, regions, use_attn, use_align)
            elif use_face:
                taps.stop()
            sq = elementwise_loss(cfg.loss_type, pred, target, latents, noise, sigma)
            if regions is not None and cfg.region_weights:  # region-weighted loss: face > person > background (mean 1)
                sq = sq * regions.weight.to(rdev)
            per = sq.mean(dim=(1, 2, 3))
            if (regions is not None and cfg.region_weights and cfg.region_prior_weight > 0
                    and bool(regions.prior.any())):
                # prior preservation: keep background cells close to what the base model (LoRA off) predicts
                lora_mod.set_multiplier(rinj, 0.0)
                try:
                    with torch.no_grad():
                        base_pred = rmodel(noisy_in, sigma * 1000.0, embeds, mask).float()
                finally:  # an out-of-memory retry must not train with the LoRA switched off
                    lora_mod.set_multiplier(rinj, 1.0)
                pm = regions.prior.to(rdev).float()
                prior = ((pred - base_pred).pow(2) * pm).sum(dim=(1, 2, 3)) / (pm.sum(dim=(1, 2, 3)) * pred.shape[1]).clamp(min=1)
                per = per + cfg.region_prior_weight * prior
            weights = loss_weight(cfg, sigma)
            look = self._look_factors(batch, cur_step, cfg, self.batcher,
                                      cfg.batch_size * cfg.gradient_accumulation * len(self.replicas))
            if look is not None:  # look_warmup: the look filter's outliers ease in
                weights = weights * torch.tensor(look, device=weights.device, dtype=weights.dtype)
            mults: list[float] = []
            if self.loss_watch is not None:
                per_cpu, sig_cpu = per.detach().cpu().tolist(), sigma.detach().cpu().tolist()
                for it, lv, sv in zip(batch, per_cpu, sig_cpu):
                    self.loss_watch.observe(cur_step, it.key, float(lv), float(sv))
                    mults.append(self.loss_watch.multiplier(it.key))
                if cfg.per_image_lr and not true_per_image_lr:
                    weights = weights * torch.tensor(mults, device=weights.device, dtype=weights.dtype)
            if any(it.loss_scale != 1.0 for it in batch):  # LoRA Isolation's side-by-side pairs
                weights = weights * torch.tensor([it.loss_scale for it in batch], device=weights.device,
                                                 dtype=weights.dtype)
            loss = (per * weights).mean() / n_total  # the step's update is the mean over all of its batches
            report = loss.detach()  # the diffusion loss alone for the log and the chart
            if aux is not None:
                loss = loss + aux.to(loss.device) / n_total
            result = None
            if self.id_loss is not None and regions is not None and rmodel is model:  # primary replica only
                t_id = time.time()
                id_val, n_id = self.id_loss(noisy, pred, sigma, regions.meta, self.id_ref)
                if id_val is not None:
                    (loss + id_w * id_val / n_total).backward()
                    if str(rdev).startswith("cuda"):
                        torch.cuda.synchronize(rdev)
                    result = (report.item(), mults, (id_val.item(), n_id, time.time() - t_id))
            if result is None:
                loss.backward()
                result = (report.item(), mults, None)
            if use_attn or use_align or use_face:
                taps.end()
            if face_sink is not None:
                lora_mod.disarm_face_tap(rinj)  # the trigger-free pass and the next batch report nothing stale
            if dop is not None:  # LoRA Isolation: the trigger-free pass, after this batch's backward freed its graph
                del pred, sq, per, noisy, noisy_in
                if observe is not None:
                    observe(dop[3], hook=rmodel is model)
                self._dop_pass(rmodel, rinj, rdev, latents, len(batch), dop, n_total, cfg)
            return result

        def work_list(entry):
            """One model copy's batches of a step, one after another: [(work result, seconds, image tokens)]."""
            out = []
            for job in entry[1]:
                t0 = time.perf_counter()
                res = work(job)  # its .item() waits for the GPU, so this is the batch's own time
                out.append((res, time.perf_counter() - t0, job[2].shape[2] * job[2].shape[3]))
            return out

        def ft_rotate(at_step: int) -> None:
            """A fine-tune's next window: the trained one written back, the next one up, a fresh optimizer."""
            nonlocal params, optimizer, scheduler
            params, optimizer, scheduler = self._finetune_rotate(cfg, at_step, optimizer)

        retry_jobs = None  # (per_rep, step_images) of a step that ran out of memory, trained again as they were
        while step < cfg.steps or self._race_running():
            if (self.ft is not None and step > start_step and step % self.ft_rotate_steps == 0
                    and step != self.ft_rotated_at):  # (an out-of-memory retry of a step must not rotate twice)
                ft_rotate(step)
            t_step = time.perf_counter()
            prof = profiling.StepProfile(self.job_dir, step + 1, self.devices) if step + 1 == profile_at else None
            if prof is not None:
                prof.start()
            optimizer.zero_grad(set_to_none=True)
            if self.id_align_opt is not None:
                self.id_align_opt.zero_grad(set_to_none=True)
            accum_loss = 0.0
            step_images = 0
            step_lr_mult = 1.0
            step_id: list[tuple] = []
            if self.id_loss is not None:
                self.id_ref = self.id_loss.pick_ref()  # one reference face per step
            retries0 = self._alloc_retries() if any(self.kept.values()) else None
            # batches per model copy this step: gradient_accumulation each in lockstep, split by speed with dp_balance
            counts = self.balancer.counts if self.balancer is not None else [cfg.gradient_accumulation] * n_dev
            n_total = sum(counts)
            compiled_now = False
            if retry_jobs is not None:  # an out-of-memory retry trains the same batches with the same noise
                per_rep, step_images = retry_jobs
                retry_jobs = None
            else:
                per_rep: list[list] = [[] for _ in self.replicas]
                step_sigma = None  # keel_pair_sigma: the step's one sigma once drawn
                for r in job_order(counts):  # every batch and its noise drawn in the main thread (shared RNG), in
                    batch = next(self.batch_iter)  # lockstep order: copy 0, copy 1, copy 0, ...
                    step_images += len(batch)
                    latents, embeds, mask, regions = load_batch(batch, cfg, rng)
                    tok = latents.shape[2] * latents.shape[3]
                    shift = self.ts_logit_shift + self._clean_offset(
                        cfg, step, tok, self.batcher, cfg.batch_size * cfg.gradient_accumulation * n_dev)
                    sigma, step_sigma = draw_sigmas(cfg, latents.shape[0], gen, shift, tok, self.arch, step_sigma)
                    noise = torch.randn(latents.shape, generator=gen)
                    dop = self._dop_job(cfg, batch, latents, step + 1, gen, shift, tok, step_sigma)  # LoRA Isolation
                    per_rep[r].append((self.replicas[r], batch, latents, embeds, mask, sigma, noise, regions, step + 1,
                                       n_total, dop))
            try:
                entries = list(enumerate(per_rep))
                if self.dp is not None:
                    # torch.compile + threads: a replica tracing a graph breaks the other replica's forward, so steps
                    # that may compile run serially; parallel steps run under a stance that never compiles (an unseen
                    # shape then runs that block eagerly once instead of tracing)
                    serial = self.compiled and (self.dp_quiet_steps < DP_COMPILE_QUIET_STEPS or not stance_available())
                    n_before = compile_count()
                    if self.compiled and not serial:
                        with no_recompile_stance():
                            lists = self.dp.run(work_list, entries, serial=False)
                    else:
                        lists = self.dp.run(work_list, entries, serial=serial)
                    compiled_now = compile_count() != n_before
                    if compiled_now:
                        if not serial:
                            log.warning("Data parallel: a graph compiled during a parallel step; running replicas "
                                        "one after another again until compilation settles")
                        self.dp_quiet_steps = 0
                    elif self.compiled and self.dp_quiet_steps < DP_COMPILE_QUIET_STEPS:
                        self.dp_quiet_steps += 1
                        if self.dp_quiet_steps == DP_COMPILE_QUIET_STEPS:
                            if not stance_available():
                                log.warning("Data parallel + compile: this torch has no torch.compiler.set_stance, so "
                                            "replicas keep running one after another (no parallel speed-up)")
                            else:
                                log.info("Data parallel: compilation settled, replicas now run in parallel "
                                         "(new shapes run uncompiled instead of recompiling)")
                else:
                    lists = [work_list(entries[0])]
                timed = [(r, sec, tok) for r, lst in enumerate(lists) for _, sec, tok in lst]
                results = [res for lst in lists for res, _, _ in lst]
                accum_loss += sum(r[0] for r in results)
                step_id += [r[2] for r in results if r[2] is not None]
                if true_per_image_lr and results[0][1]:
                    step_lr_mult = results[0][1][0]
            except torch.OutOfMemoryError:
                # blocks keeping their activations ran out of memory (a larger bucket, the identity loss, another
                # process): keep fewer and redo the step; partial gradients of the failed step are dropped
                optimizer.zero_grad(set_to_none=True)
                if self.dp is not None:
                    self.dp.zero_grads()  # the replicas' too: reduce_grads never ran for this step
                if prof is not None:
                    prof.stop(None)
                if not self._shrink_kept_blocks():
                    raise
                retry_jobs = (per_rep, step_images)  # the same images, not the next ones from the iterator
                free_cuda()
                continue
            if self.dp is not None:  # (Keel scales each replica's sum by the step's band and counts its vote)
                self.dp.reduce_grads(take=getattr(optimizer, "take_replica", None))
            gather = getattr(optimizer, "gather", None)
            if gather is not None:  # Keel: the micro-batch sums join the replicas' on p.grad for the clip
                gather()
            if cfg.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
            step_lr_mult *= self._clean_lr_scale(  # clean_share: half the lr on the lower-share epochs
                cfg, step, self.batcher, cfg.batch_size * cfg.gradient_accumulation * n_dev)
            if step_lr_mult != 1.0:  # true per-image lr: scale this step only (the scheduler restores it)
                saved_lrs = [g["lr"] for g in optimizer.param_groups]
                for g in optimizer.param_groups:
                    g["lr"] = g["lr"] * step_lr_mult
            optimizer.step()
            if self.ft is not None and cfg.finetune_anchor > 0:  # bounded drift: toward the original weights
                self.ft.pull_to_base(current_lr(optimizer, scheduler) * cfg.finetune_anchor)
            if self.id_align_opt is not None and (cfg.id_align or cfg.body_align):
                self.id_align_opt.step()
            if step_lr_mult != 1.0:
                for g, lr0 in zip(optimizer.param_groups, saved_lrs):
                    g["lr"] = lr0
            if self.ema is not None:
                self.ema.update(step_images)
            scheduler.step()
            if retries0 is not None:
                self._ease_memory_pressure(retries0)
            if self.dp is not None:
                self.dp.sync_weights()
            step += 1
            compute_done += 1
            if prof is not None:
                prof.stop(time.perf_counter() - t_step)
            step_times.append(time.perf_counter() - t_step)
            if self.balancer is not None:
                if compute_done > 2 and not compiled_now:  # warm-up and compiling steps are not the GPUs' speed
                    self.balancer.observe(timed)
                self._balance(step)
            if self.compiled and compute_done == 5:
                compiling.report("after 5 steps")
                self._save_compile_cache()
            losses.append(accum_loss)
            avg = sum(losses[-50:]) / len(losses[-50:])
            elapsed = time.time() - start
            done_steps = max(1, compute_done)
            total_steps = self.race.estimate["total"] if self.race is not None else cfg.steps - start_step
            eta = elapsed / done_steps * max(0, total_steps - compute_done)
            lr_now = current_lr(optimizer, scheduler)
            id_extra = {}
            if step_id:
                n_id = sum(r[1] for r in step_id)
                id_now = sum(r[0] * r[1] for r in step_id) / n_id
                id_losses.append([step, id_now])
                self.id_time += sum(r[2] for r in step_id)
                self.id_count += n_id
                id_extra = {"id_losses": id_losses[-2000:], "id_loss": id_now}
            if step % 5 == 0 or compute_done == 1:
                id_s = ""
                if self.id_loss is not None and id_losses:
                    recent = [v for _, v in id_losses[-10:]]
                    id_s = (f" | id loss {sum(recent) / len(recent):.4f} ({self.id_count} samples, "
                            f"{self.id_time / max(1, self.id_count):.2f}s each incl. the step's backward, "
                            f"{self.id_time / done_steps:.2f}s/it")
                    if torch.cuda.is_available() and str(self.devices[0]).startswith("cuda"):
                        id_s += f", peak {torch.cuda.max_memory_allocated(self.devices[0]) / 1e9:.2f} GB"
                    id_s += ")"
                if self.dop_values:  # LoRA Isolation: the trigger-free passes since the last line
                    id_s += f" | dop {sum(self.dop_values) / len(self.dop_values):.4f}"
                    self.dop_values.clear()
                for key, label in (("attn", "trigger attention off the subject"), ("id", "identity alignment"),
                                   ("body", "body alignment")):
                    vals = self.align_values[key]
                    if vals:
                        id_s += f" | {label} {sum(vals) / len(vals):.3f}"
                        if key == "attn" and sum(self.align_values["pairs"]):
                            id_s += f" ({sum(self.align_values['pairs'])} Isolation pairs)"
                        vals.clear()
                self.align_values["pairs"].clear()
                last = step_times[-5:]
                who = f"[{self.race.active.name}] " if self._race_running() else ""
                log.info("%sstep %d/%d loss %.4f avg %.4f lr %.2e %.1fs/it (last %d steps %.2fs each)%s", who, step,
                         cfg.steps, accum_loss, avg, lr_now, elapsed / done_steps, len(last), sum(last) / len(last),
                         id_s)
            peak = {str(d): round(torch.cuda.max_memory_allocated(torch.device(d)) / 1e9, 2) for d in self.devices
                    if str(d).startswith("cuda")} if torch.cuda.is_available() else {}
            if self.race is not None:  # progress over every configuration's steps; the race table separately
                id_extra = {**id_extra, "race": self._race_status(), "race_step": step}
            self.status.update(step=compute_done if self.race is not None else step,
                               total=self.race.estimate["total"] if self.race is not None else cfg.steps,
                               loss=accum_loss, avg_loss=avg, elapsed=elapsed, eta=eta, lr=lr_now,
                               losses=losses[-2000:], step_times=[round(t, 3) for t in step_times[-2000:]],
                               peak_mem=peak, **id_extra)
            if self.loss_watch is not None and self.loss_watch.epoch_complete():
                summary = self.loss_watch.end_epoch(step)
                log.info("Loss watch epoch %d: %s (median residual %.3f, %d/%d active)", summary["epoch"],
                         summary["counts"], summary["median"] or 0.0, summary["active"], summary["total"])
                self.status.update(loss_watch=summary)
                self._refresh_batcher_weights()
            if cfg.save_every > 0 and step % cfg.save_every == 0 and step < cfg.steps:  # 0: the final save only
                if self.ft is not None and not self._finetune_cycle_done(step):
                    self.ft_pending_save = True  # saved at the end of the cycle: every window trained alike
                else:
                    self.save(injected, step)
            elif self.ft is not None and self.ft_pending_save and self._finetune_cycle_done(step) and step < cfg.steps:
                self.save(injected, step)
            state_every = cfg.save_state_every or cfg.save_every
            if (cfg.save_state and state_every > 0 and step % state_every == 0 and step < cfg.steps
                    and not self._race_running()):
                self.save_state(step, injected, optimizer, scheduler, gen, rng)  # (a race keeps no states)
            if cfg.sample_prompts and cfg.sample_every > 0 and step % cfg.sample_every == 0 and step < cfg.steps:
                self.sample_images(model, step)
                model.train()
            racing = self._race_running()
            final_too = self.race is not None and self.race.schedule.keeps_all  # scored at the end like the others
            last = step >= cfg.steps and not racing  # the final LoRA's own score, whatever probe_every
            rounds = self._selftrain_rounds(cfg)
            # probe_async: act on a probe the probe GPU has finished (a self-training round waits for it: its bar is
            # the latest score)
            stop = self.probe_worker is not None and self._probe_async_take(step, optimizer, scheduler,
                                                                             wait=step in rounds)
            if self.controller is not None and (step % cfg.probe_every == 0 and (step < cfg.steps or racing
                                                                                  or final_too) or last):
                if self.probe_worker is not None and not last:
                    stop = self._probe_async_take(step, optimizer, scheduler, wait=True) or stop
                if self.probe_worker is not None and not last:  # queued on the probe GPU; training goes on
                    if not stop:
                        self._probe_async_start(step, injected)
                        if step in rounds:
                            stop = self._probe_async_take(step, optimizer, scheduler, wait=True)
                else:  # (also when the probe process failed: from then on probes render here)
                    stop = self.run_probe(model, step, optimizer, scheduler, injected)
                    model.train()
                if racing:  # the race decides who goes on: record, maybe drop, and switch configuration
                    stop = False
                    self.race.active.state.update(step=step, losses=losses, id_losses=id_losses)
                    bind(self._race_after_probe())
            if stop and not last:
                log.info("Likeness controller: no improvement for %d probes%s, stopping at step %d",
                         cfg.step_up_stop_patience if cfg.resolution_step_up else cfg.likeness_patience,
                         " at the highest resolution" if cfg.resolution_step_up else "", step)
                break
            if step in rounds:  # in a race, the configuration training now: its own setting and its own steps
                self._self_train_round(model, step, rounds.index(step) + 1)
                model.train()
            if self._race_running() and self._race_manual_drops(step, losses, id_losses):
                bind(self.race.active)  # the configuration that trained was dropped: the next one takes over
                model.train()
            if self.stop_file.exists():
                log.info("Stop requested at step %d", step)
                break
        if self.probe_worker is not None:  # a probe still rendering when training stopped: its score still counts
            self._probe_async_take(step, optimizer, scheduler, wait=True)
        if self.compiled:
            compiling.report("end of training")
            self._save_compile_cache()
        if self._race_running():
            log.warning("Stopped before the race was decided; the final save is %s's (the others are in race.json "
                        "and their _dropped files)", self.race.active.name)
        if self.ft is not None:
            self.ft.deactivate()  # the last window's update into the deltas before the final extraction
        path = self.save(injected, step, final=True)
        if cfg.save_state and not self._race_running() and self.ft is None:
            self.save_state(step, injected, optimizer, scheduler, gen, rng)
        if cfg.sample_prompts:
            self.sample_images(model, step)
        if self.race is not None:
            self._race_decide()
            self._race_report()
        if cfg.reward_steps > 0 and not self.stop_file.exists():
            path = self._reward_stage(model, injected, step) or path
        self.status.update(stage="finished", done=True)
        self._save_history("finished" if step >= cfg.steps else "stopped")
        log.info("Done. Final LoRA: %s", path)

    # ------------------------------------------------------------ full fine-tune (see finetune.py)
    def _finetune_setup(self, cfg, items):
        """Plan the windows for this card, activate the first and build its optimizer."""
        from . import finetune as ft_mod
        from .config import side_to_mp

        ft = self.ft
        if cfg.finetune_resume:
            n = ft.load_delta(cfg.finetune_resume)
            log.info("Fine-tune continues from %s: %d layers' deltas", cfg.finetune_resume, n)
        dev = self.devices[0]
        free = torch.cuda.mem_get_info(torch.device(dev))[0] if str(dev).startswith("cuda") else 64 * 2**30
        max_mp = max((side_to_mp(s) for s in cfg.training_sides()), default=1.0)
        reserve = cfg.finetune_reserve_gb or ft_mod.default_reserve_gb(max_mp)
        budget = ft_mod.plan_budget(free, reserve, cfg.finetune_window_gb)
        bpp = ft_mod.BYTES_PER_PARAM.get(cfg.optimizer.lower(), 18) + (ft_mod.ANCHOR_BYTES if cfg.finetune_anchor > 0
                                                                        else 0)
        if cfg.optimizer.lower() in KEEL and cfg.keel_lookahead:
            bpp += 4  # the lookahead's fp32 anchor copy
        if cfg.optimizer.lower() in KEEL and cfg.keel_consensus > 0:  # the consensus vote's buffer for the step
            bpp += 4 if keel_consensus_mode(cfg.keel_consensus_mode, cfg.optimizer) == "magnitude" else 1
        ft.anchored = cfg.finetune_anchor > 0
        windows = ft.plan(budget, bpp)
        per_step = cfg.batch_size * cfg.gradient_accumulation
        self.ft_rotate_steps = ft_mod.epoch_steps(sum(max(1, int(it.repeats or 1)) for it in items), per_step,
                                                  cfg.finetune_rotate_epochs)
        self.ft_pending_save = False
        self.ft_window = 0
        log.info("Fine-tune: %.2fB weights in %d layers, %d window(s) of up to %.1f GiB (free %.1f GiB, %.1f GiB kept "
                 "for activations, %d bytes per weight with %s); each window trains %g epoch(s) = %d steps, a cycle "
                 "%d steps; saves extract rank %s", ft.n_params / 1e9, len(ft.slots), len(windows), budget / 2**30,
                 free / 2**30, reserve, bpp, cfg.optimizer, cfg.finetune_rotate_epochs, self.ft_rotate_steps,
                 self.ft_rotate_steps * len(windows), ", ".join(map(str, cfg.finetune_extract_ranks)))
        for i, names in enumerate(windows):
            log.info("  window %d: %s (%.2fB weights)", i + 1, ft_mod.describe_window(names), ft.window_params(i) / 1e9)
        if cfg.finetune_anchor > 0:
            log.info("Fine-tune: the weights decay toward the original model's (finetune_anchor %g x lr per step), "
                     "not toward zero", cfg.finetune_anchor)
        if cfg.learning_rate > 5e-5 and cfg.optimizer.lower() not in AUTOMAGIC:
            log.warning("Learning rate %g is high for a fine-tune (Fizgig measured 1e-5 on Krea 2; 1e-4 overcooks): "
                        "the weights themselves move, not an adapter", cfg.learning_rate)
        return self._finetune_window(cfg, 0, 0)

    def _finetune_window(self, cfg, i: int, step: int, prev=None):
        params = self.ft.activate(i)
        self.ft_window = i
        lr = cfg.learning_rate
        if prev is not None and hasattr(prev, "get_avg_learning_rate"):  # Automagic: the adapted lr carries over
            lr = float(prev.get_avg_learning_rate())
            log.info("Fine-tune: the next window starts at the adapted lr %.2e", lr)
        # the optimizer's weight decay would shrink the model's weights toward zero: off (finetune_anchor instead)
        optimizer = build_optimizer(dataclasses.replace(cfg, weight_decay=0.0, learning_rate=lr),
                                    [{"params": params, "lr": lr, "name": "finetune"}])
        if prev is not None and hasattr(optimizer, "inherit_reference"):  # Automagic 4: the window's first step trains
            optimizer.inherit_reference(prev)
        scheduler = build_scheduler(cfg, optimizer, offset=step)
        self.status.update(finetune={"window": i + 1, "windows": len(self.ft.windows), "cycle": self.ft.cycles + 1,
                                     "layers": len(self.ft.windows[i]), "rotate_steps": self.ft_rotate_steps})
        return params, optimizer, scheduler

    def _finetune_rotate(self, cfg, step: int, prev=None):
        ft = self.ft
        ft.deactivate()
        free_cuda()
        self.ft_rotated_at = step
        nxt = (self.ft_window + 1) % len(ft.windows)
        if nxt == 0:
            log.info("Fine-tune: cycle %d complete at step %d (every window trained)", ft.cycles, step)
        from .finetune import describe_window

        log.info("Fine-tune: window %d of %d (%s) from step %d", nxt + 1, len(ft.windows), describe_window(ft.windows[nxt]),
                 step)
        return self._finetune_window(cfg, nxt, step, prev)

    def _finetune_cycle_done(self, step: int) -> bool:
        """True at the steps that end a cycle (the last window's last step)."""
        n = len(self.ft.windows)
        return step % self.ft_rotate_steps == 0 and (step // self.ft_rotate_steps) % n == 0

    def _prune_saved(self, stem: str) -> None:
        """keep_checkpoints: delete the step checkpoints beyond the newest N (and forget them in the status)."""
        for p in prune_checkpoints(self.job_dir, stem, self.cfg.keep_checkpoints):
            log.info("Deleted the old checkpoint %s (keep_checkpoints %d)", p.name, self.cfg.keep_checkpoints)
            if str(p) in self.status.data["saved"]:
                self.status.data["saved"].remove(str(p))

    def _finetune_save(self, step: int, final: bool, label: str = "") -> Path:
        """Extract the deltas into a LoRA per rank (the usual file names; a second rank gets _r<rank>), and the deltas
        themselves when asked."""
        cfg = self.cfg
        ft = self.ft
        ranks = [int(r) for r in cfg.finetune_extract_ranks]
        meta = {"ss_base_model": cfg.model_path if cfg.model_source == "hf" else Path(cfg.transformer_file).name,
                "ss_steps": str(step), "modelspec.architecture": self.arch.modelspec, "rulztrainer_config": cfg.to_dict(),
                "rulztrainer_finetune": f"extracted at step {step}, cycle {ft.cycles}", "ss_blocks": cfg.lora_blocks or "all"}
        first = None
        t0 = time.time()
        for k, rank in enumerate(ranks):
            suffix = f"_r{rank}" if k else ""
            name = (f"{cfg.job_name}{label}{suffix}.safetensors" if final else
                    f"{cfg.job_name}{label}_{step:06d}{suffix}.safetensors")
            path = self.job_dir / name
            stubs = ft.extract(rank, device=self.devices[0])
            lora_mod.save_lora(stubs, path, metadata={**meta, "ss_network_dim": str(rank), "ss_network_alpha": str(rank)},
                               dtype=cfg.lora_save_dtype, fmt=self.arch.lora_format)
            log.info("Saved %s: rank %d LoRA extracted from %d fine-tuned layers (%.1fs)", path, rank, len(stubs),
                     time.time() - t0)
            if str(path) not in self.status.data["saved"]:
                self.status.data["saved"].append(str(path))
            first = first or path
        if not final:
            self._prune_saved(f"{cfg.job_name}{label}")
        if cfg.finetune_save_delta:
            dpath = self.job_dir / f"{cfg.job_name}{label}_finetune.safetensors"
            ft.save_delta(dpath, {**meta, "lora_format": self.arch.lora_format})
            log.info("Saved the fine-tune's deltas %s (%.1f GB)", dpath, dpath.stat().st_size / 1e9)
            if str(dpath) not in self.status.data["saved"]:
                self.status.data["saved"].append(str(dpath))
        self.ft_pending_save = False
        self.status.write()
        return first

    # ------------------------------------------------------------ saving
    @contextlib.contextmanager
    def _averaged(self, replicas: bool = False):
        """The weight average (lora_ema) in the LoRA while saving, sampling or probing; training goes on with the raw
        weights. replicas=True hands it to the data-parallel copies too (they render samples and probes) and the raw
        weights back afterwards. A no-op without an average, or inside another _averaged."""
        if self.ema is None or self.ema.applied_now:
            yield
            return
        try:
            with self.ema.applied():
                if replicas and self.dp is not None:
                    self.dp.sync_weights()
                yield
        finally:
            if replicas and self.dp is not None:
                self.dp.sync_weights()

    def save(self, injected, step, final=False) -> Path:
        cfg = self.cfg
        if getattr(self, "ft", None) is not None:
            return self._finetune_save(step, final)
        tag = self._file_tag()  # a race's training configuration while the race runs
        name = f"{cfg.job_name}{tag}.safetensors" if final else f"{cfg.job_name}{tag}_{step:06d}.safetensors"
        path = self.job_dir / name
        meta = {"ss_base_model": cfg.model_path if cfg.model_source == "hf" else Path(cfg.transformer_file).name,
                "ss_network_dim": str({"lokr": cfg.lokr_factor, "oft": cfg.oft_block_size}.get(cfg.network_type,
                                                                                               cfg.lora_rank)),
                "ss_network_alpha": str({"lokr": 1.0, "oft": cfg.oft_constraint}.get(cfg.network_type,
                                                                                      cfg.lora_alpha)),
                "ss_steps": str(step), "modelspec.architecture": self.arch.modelspec,
                "rulztrainer_config": cfg.to_dict(), "ss_blocks": cfg.lora_blocks or "all"}
        if self.ema is not None:
            meta["rulztrainer_ema"] = f"{cfg.lora_ema:g}"
        if self.race is not None:
            v = self.race.active
            meta["rulztrainer_race"] = f"{v.name}: {race_mod.format_overrides(v.overrides)}"
        with self._averaged():
            lora_mod.save_lora(injected, path, metadata=meta, dtype=cfg.lora_save_dtype, fmt=self.arch.lora_format)
        log.info("Saved %s (%s)", path, cfg.lora_save_dtype)
        self.status.data["saved"].append(str(path))
        if not final:
            self._prune_saved(f"{cfg.job_name}{tag}")
        self.status.write()
        return path

    def save_state(self, step, injected, optimizer, scheduler, gen, rng) -> Path:
        state_dir = self.job_dir / f"state_{step:06d}"
        state_dir.mkdir(exist_ok=True)
        lora_mod.save_lora(injected, state_dir / "lora.safetensors", dtype=torch.float32, fmt=self.arch.lora_format)
        torch.save({
            "step": step, "res_stage": self.res_stage, "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "torch_gen": gen.get_state(), "py_rng": rng.getstate(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "config": self.cfg.to_dict(),
            "ema": self.ema.state_dict() if self.ema is not None else None,  # lora.safetensors holds the raw weights
            "dp_counts": self.balancer.counts if self.balancer is not None else None,
        }, state_dir / "train_state.pt")
        log.info("Saved training state %s", state_dir)
        self.status.data["states"].append(str(state_dir))
        if self.cfg.keep_states > 0:
            import shutil

            states = sorted(p for p in self.job_dir.glob("state_*") if p.is_dir())
            for old in states[: max(0, len(states) - self.cfg.keep_states)]:
                shutil.rmtree(old, ignore_errors=True)
                log.info("Removed old training state %s", old)
                if str(old) in self.status.data["states"]:
                    self.status.data["states"].remove(str(old))
        self.status.write()
        return state_dir

    def _save_compile_cache(self) -> None:
        t0 = time.time()
        f = compiling.save_cache(self.compile_dirs, self.replicas[0][2])
        if f is not None:
            log.info("compile_cache: saved %s (%.1f MB, %.1fs)", f, f.stat().st_size / 2**20, time.time() - t0)

    def compile(self, model, fall_back: bool = False, shape_swaps: bool = False) -> bool:
        try:
            import triton  # noqa: F401
        except ImportError:
            log.warning("compile_model requested but triton is not installed; running without compile")
            return False
        try:
            compiling.prepare(fall_back, shape_swaps)  # raised recompile limits, dynamo warnings in train.log
            blocks = [b for b in self.arch.compile_blocks(model) if not getattr(b, "_rz_tapped", False)]
            for block in blocks:
                block.compile(backend="inductor", dynamic=True)
            log.info("Compiled %d DiT blocks with torch.compile (first steps will be slow)", len(blocks))
            return True
        except Exception as e:  # noqa: BLE001
            log.warning("torch.compile failed (%s); running without compile", e)
            return False

    def load_state(self, path, injected, optimizer, scheduler, gen, rng) -> int:
        state_dir = Path(path)
        if state_dir.is_file():
            state_dir = state_dir.parent
        n = lora_mod.load_lora_weights(injected, state_dir / "lora.safetensors")
        if n != len(injected):
            raise ValueError(f"State has {n} LoRA modules but the model has {len(injected)}; same rank/targets?")
        st = torch.load(state_dir / "train_state.pt", map_location="cpu", weights_only=False)
        saved_groups = len(st["optimizer"].get("param_groups", []))
        if saved_groups and saved_groups != len(optimizer.param_groups):
            saved_cfg = st.get("config", {})
            raise ValueError(f"Training state was saved with lora_plus_ratio={saved_cfg.get('lora_plus_ratio', 0)} "
                             f"and automagic_groups={saved_cfg.get('automagic_groups', 'single')} ({saved_groups} "
                             f"param groups); set the same values to resume")
        saved_opt = str((st.get("config") or {}).get("optimizer", "") or "").lower()
        if saved_opt and saved_opt != self.cfg.optimizer.lower():
            if saved_opt in AUTOMAGIC and self.cfg.optimizer.lower() in AUTOMAGIC:
                log.info("Training state was saved with %s; continuing with %s (the adapted lr and sign history carry "
                         "over%s)", saved_opt, self.cfg.optimizer,
                         "; the sensitivity reference is measured on the first step"
                         if self.cfg.optimizer.lower() in ("automagic4", *KEEL) and saved_opt == "automagic3" else "")
            else:
                raise ValueError(f"Training state was saved with optimizer {saved_opt} but the config says "
                                 f"{self.cfg.optimizer}; set optimizer: {saved_opt} to resume it (only Automagic 3, "
                                 f"4 and Keel states convert into each other)")
        optimizer.load_state_dict(st["optimizer"])
        scheduler.load_state_dict(st["scheduler"])
        gen.set_state(st["torch_gen"])
        rng.setstate(st["py_rng"])
        torch.set_rng_state(st["torch_rng"])
        if st.get("cuda_rng") is not None and torch.cuda.is_available():
            try:
                torch.cuda.set_rng_state_all(st["cuda_rng"])
            except Exception:  # noqa: BLE001
                pass
        self.res_stage = int(st.get("res_stage", 0))
        self.resumed_ema = st.get("ema")  # None in states saved without an average (or by an older version)
        self.resumed_dp_counts = st.get("dp_counts")
        return int(st["step"])

    # ------------------------------------------------------------ sampling
    def _lora_names(self, model, blocks) -> list[str]:
        if self.race_union_names is not None:  # a race: wrappers over every configuration's modules
            return list(self.race_union_names)
        return self.arch.lora_module_names(model, self.cfg.lora_targets, blocks, self.cfg.train_text_embedder)

    # ------------------------------------------------------------ selective checkpointing

    def _setup_checkpointing(self, items, blocks) -> None:
        """Pick the blocks that keep their activations (see rulztrainer.checkpointing)."""
        cfg = self.cfg
        n_blocks = len(self.replicas[0][0].layers)
        candidates = ckpt.trained_block_indices(n_blocks, blocks)
        if self.block_swap is not None:  # swapped blocks must stay checkpointed (or their weights stay on the GPU)
            swapped = set(self.block_swap.indices)
            candidates = [c for c in candidates if c not in swapped]
        self.kept = {}
        if not cfg.gradient_checkpointing or cfg.checkpoint_keep_blocks == 0 or not candidates:
            return
        if cfg.checkpoint_keep_blocks > 0:
            keep = ckpt.kept_indices(candidates, cfg.checkpoint_keep_blocks)
            for m, _, _ in self.replicas:
                self.kept[id(m)] = keep
                ckpt.set_kept_blocks(m, keep)
            log.info("Checkpointing: %d of %d blocks keep their activations (checkpoint_keep_blocks)", len(keep),
                     n_blocks)
            return
        if not torch.cuda.is_available() or not all(str(d).startswith("cuda") for d in self.devices):
            return
        self.status.update(stage="measuring activation memory")
        try:
            self._calibrate_checkpointing(items, candidates, n_blocks)
        except Exception as e:  # noqa: BLE001  (an OOM while measuring included: then nothing is kept)
            log.warning("Checkpointing calibration failed (%s: %s); every block is checkpointed",
                        type(e).__name__, str(e)[:200])
            for m, _, _ in self.replicas:
                ckpt.set_kept_blocks(m, [])
            self.kept = {}
        finally:
            for _, inj, _ in self.replicas:
                for p in lora_mod.lora_parameters(inj):
                    p.grad = None
            free_cuda()

    def _calibrate_checkpointing(self, items, candidates: list[int], n_blocks: int) -> None:
        """Measure one forward+backward on the largest bucket with every block checkpointed, then with a few blocks
        kept per GPU, and keep as many of the last trained blocks as the free memory allows."""
        cfg = self.cfg
        rng, gen = random.Random(0), torch.Generator("cpu").manual_seed(0)  # the run's own streams stay untouched
        big = max(items, key=lambda it: (it.width // 16) * (it.height // 16))
        # the worst case: the most image tokens with the longest caption (the blocks see text and image tokens)
        big = dataclasses.replace(big, text_path=_longest_text_path(items, self.arch.pad_embeds))
        latents, embeds, mask, _ = load_batch([big] * cfg.batch_size, dataclasses.replace(cfg, caption_dropout=0.0),
                                              rng)
        noise = torch.randn(latents.shape, generator=gen)
        sigma = torch.full((latents.shape[0],), 0.5)
        tokens = (big.width // 16) * (big.height // 16) + mask.shape[1]

        for m, inj, dev in self.replicas:
            dev = norm_device(dev)
            assignment = [norm_device(d) for d in getattr(m, "split_assignment", None) or [dev] * n_blocks]
            devices = list(dict.fromkeys(assignment))
            per_dev = {d: [i for i in candidates if assignment[i] == d] for d in devices}

            def measure(n_kept: int) -> dict[str, tuple[int, int]]:
                """Keep the last n_kept candidate blocks of every GPU; (peak reserved, peak allocated) per GPU."""
                ckpt.set_kept_blocks(m, [i for c in per_dev.values() for i in ckpt.kept_indices(c, n_kept)])
                free_cuda()  # an empty cache, so the reserved peak shows this pattern's fragmentation too
                for d in devices:
                    torch.cuda.synchronize(d)
                    torch.cuda.reset_peak_memory_stats(d)
                noisy = ((1 - 0.5) * latents + 0.5 * noise).to(dev, torch.bfloat16)
                pred = m(noisy, sigma.to(dev) * 1000.0, embeds.to(dev), mask.to(dev)).float()
                pred.pow(2).mean().backward()
                del pred
                for d in devices:
                    torch.cuda.synchronize(d)
                for p in lora_mod.lora_parameters(inj):
                    p.grad = None
                return {d: (torch.cuda.max_memory_reserved(d), torch.cuda.max_memory_allocated(d)) for d in devices}

            if self.compiled:
                measure(0)  # compile first: the peaks below are then the compiled blocks' own, not compilation's
            one, three = measure(1), measure(3)
            keep, notes = [], []
            for d in devices:
                n = len(per_dev[d])
                if not n:
                    continue
                kept_three = min(3, n)
                # per extra block: the allocated slope (clean); peak: reserved (includes fragmentation)
                per_block = ((three[d][1] - one[d][1]) / (kept_three - 1)) if kept_three > 1 else 0.0
                peak_one = one[d][0]
                free, total = torch.cuda.mem_get_info(d)
                budget = min(free + torch.cuda.memory_reserved(d), total) - ckpt.memory_margin(
                    total, identity_loss=self.id_loss is not None)
                k = ckpt.choose_kept_blocks(n, peak_one, per_block, budget)
                keep += ckpt.kept_indices(per_dev[d], k)
                peak_k = peak_one + max(0, k - 1) * per_block
                notes.append(f"{d}: {k} of {n} ({per_block / ckpt.GB:.2f} GB per extra block, peak "
                             f"{peak_k / ckpt.GB:.1f} of {total / ckpt.GB:.0f} GB, budget {budget / ckpt.GB:.1f})")
            self.kept[id(m)] = sorted(keep)
            ckpt.set_kept_blocks(m, keep)
            log.info("Checkpointing: %d of %d blocks keep their activations, measured at %d tokens (%s)", len(keep),
                     n_blocks, tokens, "; ".join(notes))

    def _alloc_retries(self) -> dict[str, int]:
        """cudaMalloc retries per training GPU so far: the allocator ran out, freed its cache and tried again (slow)."""
        return {d: torch.cuda.memory_stats(d).get("num_alloc_retries", 0) for d in self.devices
                if str(d).startswith("cuda")}

    def _ease_memory_pressure(self, before: dict[str, int]) -> None:
        """A step with allocator retries is slower than recomputing a block: give one kept block back on every GPU
        that retried (the earliest kept one there), until the steps run without retries."""
        after = self._alloc_retries()
        hit = {d for d, n in after.items() if n > before.get(d, 0)}
        if not hit:
            return
        for m, _, _ in self.replicas:
            keep = self.kept.get(id(m), [])
            assignment = [norm_device(d) for d in getattr(m, "split_assignment", None) or []]
            for d in hit:
                on_d = [i for i in keep if (assignment[i] if assignment else norm_device(self.devices[0])) == d]
                if on_d:
                    keep.remove(on_d[0])
            self.kept[id(m)] = keep
            ckpt.set_kept_blocks(m, keep)
        log.info("Checkpointing: memory allocator retries on %s; now %s", ", ".join(sorted(hit)),
                 ", ".join(f"{len(v)} kept" for v in self.kept.values()))

    # ------------------------------------------------------------ configuration race (see race.py)
    def _race_prep_config(self, cfg):
        """The race's configurations, and the config the dataset is prepared with: flips, crops, region maps, face
        analysis and the identity loss whenever any configuration needs them."""
        rows = race_mod.variant_configs(cfg)
        self.race_rows = rows
        crops = {(r[1].crop_augment or "none").lower() for r in rows}
        crop = "face+bust" if "face+bust" in crops else ("face" if "face" in crops else "none")
        prep = dataclasses.replace(cfg, flip_augment=any(r[1].flip_augment for r in rows), crop_augment=crop,
                                   region_weights=any(r[1].region_weights for r in rows),
                                   attn_localize=any(r[1].attn_localize for r in rows),
                                   id_align=any(r[1].id_align for r in rows),
                                   body_align=any(r[1].body_align for r in rows),
                                   selftrain=any(r[1].selftrain for r in rows),
                                   id_loss_weight=max(r[1].id_loss_weight for r in rows),
                                   likeness_curriculum=any(r[1].likeness_curriculum for r in rows))
        self.race_parallel = bool(cfg.race_parallel)
        if self.race_parallel:
            log.info("Race in parallel: one configuration per GPU at the same time; a lone one left over trains over "
                     "every GPU when that is faster, else on the first (no torch.compile, no identity loss)")
        est = race_mod.estimate_steps(len(rows), cfg.race_warmup_probes, cfg.probe_every, cfg.steps,
                                      cfg.race_drop_lowest and not cfg.race_follow)
        if cfg.race_follow:
            log.info("Relay race: %d configurations, a probe every %d steps; after each round everyone continues from "
                     "the round's winner (its LoRA, optimizer state, weight average and likeness history) with its "
                     "own settings, seed, image order and noise. %d training steps in all (%.0fx one %d-step run); "
                     "the last round's winner is the run's LoRA", len(rows), cfg.probe_every, est["total"],
                     est["ratio"], cfg.steps)
        elif cfg.race_drop_lowest:
            log.info("Configuration race: %d configurations, %d warm-up probe round(s), a probe every %d steps: %d "
                     "rounds and %d probes, about %d training steps in all (%.1fx one %d-step run)", len(rows),
                     cfg.race_warmup_probes, cfg.probe_every, est["rounds"], est["probes"], est["total"], est["ratio"],
                     cfg.steps)
        else:
            log.info("Configuration race without dropping the lowest: %d configurations, a probe every %d steps; each "
                     "trains up to %d steps of its own, at most %d training steps in all (%.1fx one run)", len(rows),
                     cfg.probe_every, cfg.steps, est["total"], est["ratio"])
        if cfg.race_patience_drop and not cfg.race_follow:
            log.info("Race: a configuration leaves after %d probes of its own without improvement; the winner is the "
                     "highest smoothed score any configuration reached", cfg.race_patience)
            if cfg.race_patience_rising:
                log.info("Race: one out of patience stays while its smoothed score is still rising from its last "
                         "probe")
        for name, _, over in rows:
            log.info("  %s: %s", name, race_mod.format_overrides(over) or "the run's own settings")
        return prep

    def _race_lora_layout(self, model) -> list[int] | None:
        """Every configuration's LoRA modules; the trainer's wrappers cover their union. Returns the union of their
        blocks (None = all)."""
        n = len(model.layers)
        layouts = []
        for name, vcfg, _ in self.race_rows:
            unknown = [t for t in vcfg.lora_targets if t not in self.arch.target_groups]
            if unknown:
                log.warning("Race configuration %s: LoRA targets %s do not exist for %s and are ignored", name,
                            unknown, self.arch.label)
            b = lora_mod.parse_blocks(vcfg.lora_blocks, n)
            names = self.arch.lora_module_names(model, vcfg.lora_targets, b, vcfg.train_text_embedder)
            if not names:
                raise ValueError(f"Race configuration {name}: no LoRA modules selected (check its targets / blocks)")
            layouts.append((b, names))
        order = {name: i for i, (name, _) in enumerate(model.named_modules())}
        self.race_union_names = sorted({nm for _, ns in layouts for nm in ns}, key=order.get)
        self.race_layouts = layouts
        return None if any(b is None for b, _ in layouts) else sorted({i for b, _ in layouts for i in b})

    def _race_setup(self, wrappers: dict, items) -> None:
        """Build the race's configurations (LoRA weights on the CPU, own controller and items) and start the first."""
        from .likeness import LikenessController

        if self.controller is None:
            raise ValueError("The configuration race needs working likeness probes (face/body references and their "
                             "scoring packages); the probe controller could not be created")
        base = self.race_base
        variants = []
        for i, ((name, vcfg, over), (_, names)) in enumerate(zip(self.race_rows, self.race_layouts)):
            v = race_mod.Variant(i, name, vcfg, over, names)
            v.init_lora(wrappers, vcfg.seed)  # the run's seed unless the row sets its own
            v.state["items"] = race_mod.variant_items(items, vcfg, self.race_region_dirs.get(name))
            v.state["controller"] = LikenessController(vcfg, **self._controller_ceilings())
            n_w = v.n_weights
            log.info("Race configuration %s: %d modules, %.2fM weights (%s), %d dataset items",
                     name, len(names), n_w / 1e6, v.label, len(v.state["items"]))
            variants.append(v)
        est = race_mod.estimate_steps(len(variants), base.race_warmup_probes, base.probe_every, base.steps,
                                      base.race_drop_lowest and not base.race_follow)
        sched = race_mod.RaceSchedule([v.name for v in variants], base.race_warmup_probes, base.race_drop_lowest,
                                      base.race_patience if base.race_patience_drop else 0, base.steps,
                                      follow=base.race_follow, rising=base.race_patience_rising,
                                      confidence=base.race_drop_confidence, max_close=base.race_max_close_rounds)
        self.race = race_mod.RaceRun(variants, sched, wrappers, base, est)
        if self.race_parallel:  # every configuration trains on its own parameters (race_parallel.py)
            self.race.active = variants[0]
        else:
            self._race_activate(variants[0])
        self._race_report()

    def _race_activate(self, v) -> None:
        """Swap configuration v in: its LoRA, settings and per-run state (built on its first turn)."""
        race_mod.swap_in(v, self.race.wrappers)
        if self.dp is not None:
            self.dp.sync_weights()  # replicas follow the wrappers' shapes, scale and on/off
        self.cfg = v.cfg
        race_mod.configure_backward(v.cfg, [m for m, _, _ in self.replicas])
        st = v.state
        if not v.started:
            self._race_start_variant(v)
        else:
            race_mod.onload_optimizer(st["optimizer"], v.offloaded)
            v.offloaded = []
            race_mod.move_average(st["ema"], to_cpu=False)
        self.ema, self.loss_watch, self.controller = st["ema"], st["loss_watch"], st["controller"]
        self.batcher, self.batch_iter = st["batcher"], st["batch_iter"]
        self.curriculum_w, self.ts_logit_shift = st["curriculum_w"], st["ts_logit_shift"]
        self.all_items, self.injected = st["items"], st["injected"]
        self.id_aligner, self.id_align_opt = st.get("id_aligner"), st.get("id_align_opt")
        for vals in self.align_values.values():  # the log reports this configuration's values only
            vals.clear()
        il = self.race_id_loss
        self.id_loss = il if il is not None and v.cfg.id_loss_weight > 0 and not st.get("id_vetoed") else None
        if self.id_loss is not None:  # the shared identity loss with this configuration's settings and counters
            il.sigma_max, il.every = float(v.cfg.id_loss_sigma_max), max(1, int(v.cfg.id_loss_every))
            il.margin, il.min_cells = max(0, int(v.cfg.id_loss_margin_cells)), max(1, int(v.cfg.id_loss_min_face_cells))
            il.random_ref = v.cfg.id_loss_random_ref
            il.rng = st.setdefault("id_rng", random.Random(v.cfg.seed))
            il.n_eligible = st.get("id_n_eligible", 0)
        if self.compiled:  # new guards (rank, scale, on/off): let dynamo compile this configuration's graphs
            self.dp_quiet_steps = 0
            self.sample_graphs = set()
        self.race.active = v
        self.status.update(race=self._race_status())

    def _race_start_variant(self, v) -> None:
        """A configuration's first turn: its optimizer, schedule, weight average, image queue and loss watch."""
        cfg, st = v.cfg, v.state
        st["injected"] = v.master if self.race_parallel else {name: self.race.wrappers[name] for name in v.names}
        st["params"] = list(lora_mod.lora_parameters(st["injected"]))
        st["optimizer"], st["scheduler"] = self._make_optimizer(cfg, st["injected"])
        st["ema"] = self._make_ema(cfg, st["injected"])
        st["rng"], st["gen"] = random.Random(cfg.seed), torch.Generator("cpu").manual_seed(cfg.seed)
        st["step"], st["losses"], st["id_losses"] = 0, [], []
        st["curriculum_w"], st["ts_logit_shift"] = None, 0.0
        self.all_items, self.curriculum_w, self.ts_logit_shift, self.loss_watch = st["items"], None, 0.0, None
        self._enter_stage(0, 0, initial=True)  # its own image queue, seeded like every other configuration's
        st["batcher"], st["batch_iter"] = self.batcher, self.batch_iter
        st["loss_watch"] = self._make_loss_watch(st["items"], st["optimizer"])
        st["true_per_image_lr"] = (st["loss_watch"] is not None and cfg.per_image_lr and cfg.batch_size == 1
                                   and cfg.gradient_accumulation == 1 and len(self.replicas) == 1)
        st["id_w"] = idloss_weight(cfg.id_loss_weight)
        st["id_aligner"], st["id_align_opt"] = self._make_id_aligner(cfg)
        self.id_aligner, self.id_align_opt = st["id_aligner"], st["id_align_opt"]
        v.started = True

    def _race_park(self, v) -> None:
        """Park the configuration that trained: its weights, optimizer state and weight average go to the CPU."""
        st = v.state
        st.update(curriculum_w=self.curriculum_w, ts_logit_shift=self.ts_logit_shift, batcher=self.batcher,
                  batch_iter=self.batch_iter, loss_watch=self.loss_watch)
        if self.id_loss is not None:
            st["id_n_eligible"] = self.id_loss.n_eligible
        race_mod.swap_out(v, self.race.wrappers)
        v.offloaded = race_mod.offload_optimizer(st["optimizer"])
        race_mod.move_average(st["ema"], to_cpu=True)

    def _race_after_probe(self):
        """After the training configuration's probe: record its smoothed combined score, drop the lowest when the
        round is complete, and switch to the configuration whose turn it is. Returns that configuration."""
        rr = self.race
        cur = rr.active
        rounds = len(rr.schedule.history)
        self._race_record(cur, self.controller)
        if len(rr.schedule.history) > rounds:
            log.info("Race round %s", race_mod.describe_round(rr.schedule.history[-1], rr.schedule))
        for i in rr.schedule.just_dropped:
            self._race_drop(rr.variants[i])
        nxt = rr.variants[rr.schedule.current]
        relay = rr.schedule.follow and len(rr.schedule.history) > rounds and not rr.schedule.done
        if nxt is not cur:
            if cur.alive:
                self._race_park(cur)
            if relay:
                self._race_follow(rr.variants[rr.schedule.leader])
            self._race_activate(nxt)
            free_cuda()
        if rr.schedule.done:
            self._race_finish()
        self._race_report()
        return nxt

    def _race_follow(self, win) -> None:
        """A relay race's round is over (everyone parked): every other configuration continues from the winner."""
        for v in self.race.variants:
            if v is not win and v.alive and v.started:
                for note in race_mod.follow_leader(win, v):
                    log.warning("Relay race: %s", note)
        log.info("Relay race: everyone continues from %s's LoRA (step %s)", win.name, win.state.get("step"))

    def _race_keeps_best(self, kind: str, score) -> bool:
        """A relay race shares one lineage, so its best checkpoints are shared too: a configuration's new best of its
        own is saved only when it beats the best saved by any of them (kind: "" combined, "face", "body")."""
        rr = self.race
        if rr is None or not rr.schedule.follow or not self._race_running() or score is None:
            return True
        prev = rr.best_saved.get(kind)
        if prev is not None and score <= prev:
            return False
        rr.best_saved[kind] = score
        return True

    def _race_record(self, v, controller) -> None:
        """Record a configuration's probe in the schedule: its smoothed score, its best and its probes since the best."""
        c = controller
        self.race.schedule.record(v.idx, c.combined_ema if c is not None else None,
                                  c.last_step if c is not None else v.state.get("step"),
                                  best=c.best if c is not None else None, best_step=c.best_step if c is not None else None,
                                  since_improve=c.since_improve if c is not None else None,
                                  seeds=getattr(c, "last_seeds", None))

    def _race_decide(self) -> None:
        """At the end of a race that keeps every configuration's chance (patience drops, or no lowest drops): the
        winner is the highest best score any configuration reached. Its best checkpoint becomes <job>_best and its
        settings <job>_winner.yaml."""
        import shutil

        rr = self.race
        if rr is None or not rr.schedule.keeps_all or not rr.schedule.done or rr.schedule.champion is not None:
            return
        sched = rr.schedule
        bests = {v.idx: v.state["controller"].best for v in rr.variants if v.state.get("controller") is not None}
        steps = {v.idx: v.state["controller"].best_step for v in rr.variants if v.state.get("controller") is not None}
        sched.best_step.update({i: s for i, s in steps.items() if bests.get(i) is not None})
        win = rr.variants[sched.decide(bests)]
        cfg = rr.base_cfg
        last = rr.variants[sched.alive[0]] if sched.alive else None
        best = self.job_dir / f"{cfg.job_name}_{win.name}_best.safetensors"
        if win is not last and best.exists():
            shutil.copyfile(best, self.job_dir / f"{cfg.job_name}_best.safetensors")
        dataclasses.replace(win.cfg, race=False, race_configs=[], race_warmup_probes=0, race_patience_drop=False,
                            race_drop_lowest=True).save(self.job_dir / f"{cfg.job_name}_winner.yaml")
        fmt = lambda b: "n/a" if b is None else f"{b:.1f}%"  # noqa: E731
        log.info("Race decided: %s wins with the best smoothed score %s at its step %s (%s); %s", win.name,
                 fmt(sched.best.get(win.idx)), sched.best_step.get(win.idx),
                 ", ".join(f"{v.name} {fmt(sched.best.get(v.idx))}" for v in rr.variants),
                 f"its best checkpoint is {cfg.job_name}_best.safetensors" + (
                     f" ({cfg.job_name}.safetensors is {last.name}'s, the last one racing)"
                     if last is not None and win is not last else ""))
        self._race_report()

    def _race_manual_drops(self, step: int, losses, id_losses) -> bool:
        """Drop the configurations the Monitor's Drop button asked for (RACE_DROP in the job folder, one name per
        line, read once and removed): each is saved like any dropped configuration and the race goes on without it.
        A dropped configuration that is training now hands over to the next one at once; one left is the winner.
        Returns True when the configuration that trains changed."""
        request = self.job_dir / "RACE_DROP"
        if not request.exists():
            return False
        try:
            names = [x.strip() for x in request.read_text(encoding="utf-8").splitlines() if x.strip()]
            request.unlink()
        except OSError:
            return False
        rr = self.race
        switched = False
        for name in names:
            v = next((x for x in rr.variants if x.name == name), None)
            if v is None or not v.alive or rr.schedule.done:
                log.warning("Race: cannot drop %r (not in the race any more)", name)
                continue
            was_active = v is rr.active
            if was_active:
                v.state.update(step=step, losses=losses, id_losses=id_losses)
            rr.schedule.drop(v.idx)
            log.info("Race: %s dropped by request at step %d (its own step count)", v.name, v.state.get("step", 0))
            self._race_drop(v)
            if was_active:
                self._race_activate(rr.variants[rr.schedule.current])  # the dropped one is gone: nothing to park
                switched = True
            if rr.schedule.done:
                self._race_finish()
                break
        if names:
            free_cuda()
            self._race_report()
        return switched

    def _race_drop(self, v) -> None:
        """Save a dropped configuration's LoRA (its weight average when it has one) and free its state."""
        cfg = self.race.base_cfg
        sched = self.race.schedule
        finished = v.idx in sched.finished
        path = self.job_dir / f"{cfg.job_name}_{v.name}_{'final' if finished else 'dropped'}.safetensors"
        meta = {"ss_network_dim": str({"lokr": v.factor, "oft": v.block}.get(v.kind, v.rank)),
                "ss_network_alpha": str({"lokr": 1.0, "oft": v.constraint}.get(v.kind, v.alpha)),
                "modelspec.architecture":
                self.arch.modelspec, "rulztrainer_race": f"{v.name}: {race_mod.format_overrides(v.overrides)}",
                "ss_steps": str(v.state.get("step", 0))}
        if v is self.race.active and not self.race_parallel:
            with self._averaged():
                lora_mod.save_lora(v.state["injected"], path, metadata=meta, dtype=cfg.lora_save_dtype,
                                   fmt=self.arch.lora_format)
        else:
            lora_mod.save_lora(v.view(use_average=True), path, metadata=meta, dtype=cfg.lora_save_dtype,
                               fmt=self.arch.lora_format)
        why = (" (by hand)" if v.idx in sched.manual else
               f" ({sched.patience} probes without improvement)" if v.idx in sched.stale else "")
        if finished:
            log.info("Race: %s finished its %d steps in round %d -> %s", v.name, v.state.get("step", 0),
                     sched.dropped[v.idx], path.name)
        else:
            log.info("Race: %s dropped after round %d%s -> %s", v.name, sched.dropped[v.idx], why, path.name)
        if str(path) not in self.status.data["saved"]:
            self.status.data["saved"].append(str(path))
        v.alive = False
        keep = {k: v.state[k] for k in ("step", "controller") if k in v.state}
        v.state.clear()
        v.state.update(keep)
        v.down, v.up = {}, {}
        if self.race_parallel:  # its own parameters (and their GPU memory) go with it
            v.weights, v.master = {}, {}
            free_cuda()

    def _race_finish(self) -> None:
        """One configuration left: it trains on as the run, with the run's own file names."""
        import shutil

        rr = self.race
        win = rr.active
        cfg = rr.base_cfg
        best = self.job_dir / f"{cfg.job_name}_{win.name}_best.safetensors"
        if best.exists():
            shutil.copyfile(best, self.job_dir / f"{cfg.job_name}_best.safetensors")
        win_cfg = dataclasses.replace(win.cfg, race=False, race_configs=[], race_warmup_probes=0,
                                      race_patience_drop=False, race_drop_lowest=True)
        win.cfg = win_cfg
        self.cfg = win_cfg
        if rr.schedule.keeps_all:
            log.info("Race: %s is the last one racing (%s); it trains on to step %d with the run's own file names. "
                     "The winner is decided at the end: the highest smoothed score any configuration reached",
                     win.name, race_mod.format_overrides(win.overrides) or "the run's own settings", cfg.steps)
            return
        win_cfg.save(self.job_dir / f"{cfg.job_name}_winner.yaml")
        if rr.schedule.follow:
            log.info("Relay race over: %s won the last round (%s); its LoRA is the run's, the others' are their "
                     "_final files, and the best checkpoint any of them reached is %s_best.safetensors", win.name,
                     race_mod.format_overrides(win.overrides) or "the run's own settings", cfg.job_name)
            return
        log.info("Race over: %s wins (%s); it trains on to step %d with the run's own file names (its settings: %s)",
                 win.name, race_mod.format_overrides(win.overrides) or "the run's own settings", cfg.steps,
                 f"{cfg.job_name}_winner.yaml")

    def _race_status(self) -> dict:
        rr = self.race
        out = rr.schedule.state()
        out["active"] = rr.active.name if rr.active is not None else None
        out["lanes"] = [v.name for v, _ in getattr(self, "rp_lanes", [])] if getattr(self, "race_parallel", False) else []
        out["parallel"] = getattr(self, "race_parallel", False)
        out["steps"] = {v.name: (v.state.get("step", 0) if (v is not rr.active or getattr(self, "race_parallel", False)) else None)
                        for v in rr.variants}
        out["scores"] = {v.name: (v.state["controller"].combined_ema if "controller" in v.state else None)
                         for v in rr.variants}
        out["score_steps"] = {v.name: (v.state["controller"].last_step if "controller" in v.state else None)
                              for v in rr.variants}  # the step each configuration's score is from
        # the smoothed face and body behind the combined score: [normalized, raw] (raw twice without normalization)
        out["face"], out["body"] = {}, {}
        for v in rr.variants:
            c = v.state.get("controller")
            if c is None:
                continue
            out["face"][v.name] = [c.norm("face", c.face_ema), c.face_ema]
            out["body"][v.name] = [c.norm("body", c.body_ema), c.body_ema]
        out["settings"] = {v.name: race_mod.format_overrides(v.overrides) for v in rr.variants}
        out["estimate"] = rr.estimate
        return out

    def _race_report(self) -> None:
        rr = self.race
        race_mod.write_report(self.job_dir / "race.json", rr.schedule, rr.variants, {"estimate": rr.estimate})
        self.status.update(race=self._race_status())

    def _race_running(self) -> bool:
        return self.race is not None and self.race.running

    def _file_tag(self) -> str:
        """While a race runs, files of the training configuration carry its name; the reward stage's carry _reward."""
        return (f"_{self.race.active.name}" if self._race_running() else "") + getattr(self, "stage_tag", "")

    def _best_tag(self) -> str:
        """Best checkpoints' tag: a relay race's racers share one lineage and so one <job>_best file."""
        if self._race_running() and self.race.schedule.follow:
            return getattr(self, "stage_tag", "")
        return self._file_tag()

    def _image_prefix(self) -> str:
        return f"{self.race.active.name}_" if self._race_running() else ""

    def _make_optimizer(self, cfg, injected):
        if cfg.optimizer.lower() in AUTOMAGIC and cfg.automagic_groups == "by_type":
            groups = lora_mod.lora_param_groups_by(injected, cfg.learning_rate, self.arch.param_group_of)
            log.info("%s: an adaptive lr per layer type (%s)", cfg.optimizer, ", ".join(
                f"{g['name']} {sum(p.numel() for p in g['params']) / 1e6:.1f}M" for g in groups))
        else:
            groups = lora_mod.lora_param_groups(injected, cfg.learning_rate, cfg.lora_plus_ratio)
            if len(groups) > 1:
                log.info("LoRA+: lora_up lr = %.2e (x%g), lora_down lr = %.2e", groups[1]["lr"], cfg.lora_plus_ratio,
                         groups[0]["lr"])
        optimizer = build_optimizer(cfg, groups, pairs=lora_mod.lora_pairs(injected))
        return optimizer, build_scheduler(cfg, optimizer)

    def _make_ema(self, cfg, injected):
        if cfg.lora_ema <= 0:
            return None
        per_step = cfg.batch_size * cfg.gradient_accumulation * (1 if self.race_parallel else len(self.replicas))
        return LoraEMA(injected, cfg.lora_ema, total_images=cfg.steps * per_step)

    def _balance(self, step: int) -> None:
        """dp_balance: take a new split of the batches when the measured speeds call for one, and once point out
        `grow` when auto cannot split the step (too few batches) but the GPUs differ enough for it to pay."""
        bal = self.balancer
        lock = [bal.base] * bal.n
        new = bal.update(step)
        if new is not None:
            log.info("Data parallel balance: %s batches per step on %s (measured %s s per batch of 1 MP), %.2f s per "
                     "image instead of %.2f in lockstep", "+".join(map(str, new)), ", ".join(self.devices),
                     " / ".join(f"{t:.2f}" for t in bal.meter.times()), bal.per_image(new), bal.per_image(lock))
            self.status.update(dp_counts=new)
        if not self.balance_hinted and bal.meter.ready():
            self.balance_hinted = True
            times = bal.meter.times()
            log.info("Data parallel: measured %s s per batch of 1 MP on %s", " / ".join(f"{t:.2f}" for t in times),
                     " / ".join(self.devices))
            grow = choose_counts(times, bal.base, "grow")
            if self.cfg.dp_balance == "auto" and bal.counts == lock and grow != lock:
                log.info("Data parallel: the GPUs take %s s per batch of 1 MP. dp_balance: grow would run %s batches "
                         "per step at %.2f s per image instead of %.2f (more images per step, so lower steps to "
                         "match), or gradient_accumulation 3 or more lets auto split the step",
                         " / ".join(f"{t:.2f}" for t in times), "+".join(map(str, grow)),
                         step_seconds(times, grow) / sum(grow), step_seconds(times, lock) / sum(lock))

    def _shrink_kept_blocks(self) -> bool:
        """After an out-of-memory step: halve the kept blocks of every model copy. False when none were kept."""
        if not any(self.kept.values()):
            return False
        for m, _, _ in self.replicas:
            keep = self.kept.get(id(m), [])
            keep = keep[len(keep) - len(keep) // 2:] if len(keep) > 1 else []
            self.kept[id(m)] = keep
            ckpt.set_kept_blocks(m, keep)
        log.warning("Out of GPU memory with blocks keeping their activations: now %s",
                    ", ".join(f"{len(v)} kept" for v in self.kept.values()))
        return True

    def _log_nvfp4(self, n: int) -> None:
        """How the nvfp4 layers compute on each training GPU, and a hint when another GPU has FP4 tensor cores."""
        if not self.cfg.nvfp4_matmul:
            log.info("%d nvfp4 layers: weights dequantized to bf16 (nvfp4_matmul off)", n)
            return
        native = [d for d in self.devices if Q.fp4_tensor_cores(d)]
        other = [d for d in self.devices if d not in native]
        if native:
            log.info("%d nvfp4 layers: native FP4 matmul (W4A4, %s backward) on %s", n,
                     "FP4" if self.cfg.nvfp4_backward == "fp4" else "exact bf16", ", ".join(native))
        if other:
            log.info("%d nvfp4 layers dequantize to bf16 on %s (no FP4 tensor cores)", n, ", ".join(other))
            capable = [f"cuda:{i} ({torch.cuda.get_device_name(i)})" for i in range(torch.cuda.device_count())
                       if Q.fp4_tensor_cores(f"cuda:{i}") and f"cuda:{i}" not in self.devices]
            if capable:
                log.info("Native FP4 is available on %s: train there for faster nvfp4 steps", ", ".join(capable))

    def _attach_training_adapter(self, model, log_it: bool = False) -> None:
        """Attach the training adapter (a LoRA on while training, off while sampling, never saved) to a model."""
        cfg = self.cfg
        if not cfg.training_adapter or cfg.training_adapter == "auto":
            return
        if self.ft is not None:
            if log_it:
                log.info("Training adapter %s is not used by a fine-tune (it trains the weights themselves)",
                         cfg.training_adapter)
            return
        n = lora_mod.attach_training_adapter(model, cfg.training_adapter, cfg.training_adapter_strength,
                                             rename=getattr(self.arch, "lora_module_name", None),
                                             convert=getattr(self.arch, "extra_lora_converter", None))
        if log_it:
            log.info("Training adapter %s: matched %d modules (strength %g; on while training, off for samples and "
                     "probes, not saved)", cfg.training_adapter, n, cfg.training_adapter_strength)
        if not n:
            raise ValueError(f"The training adapter {cfg.training_adapter} matches no layer of this {self.arch.label} "
                             "model: pick one made for it, or clear training_adapter")

    def _attach_sampling_loras(self, model) -> None:
        """Attach the sampling-only LoRAs (turbo + optional secondary) to a model."""
        cfg = self.cfg
        for label, path, strength in (("Turbo LoRA", cfg.turbo_lora, cfg.turbo_lora_strength),
                                      ("Sampling LoRA", cfg.sample_lora, cfg.sample_lora_strength)):
            if path:
                n = lora_mod.attach_extra_lora(model, path, strength,
                                               rename=getattr(self.arch, "lora_module_name", None),
                                               convert=getattr(self.arch, "extra_lora_converter", None))
                log.info("%s %s: matched %d modules (strength %g, sampling only)", label, path, n, strength)
                if label == "Turbo LoRA" and n:
                    model.turbo_lora_attached = True  # the Krea 2 sampler then uses the distilled schedule
                    if getattr(model, "is_distilled", False):
                        log.warning("Turbo LoRA %s is applied to a model that is already distilled (Turbo): it distils "
                                    "it twice, so samples and probes do not match how the LoRA is used; set "
                                    "turbo_lora to \"\" (auto picks none for a Turbo model)", path)

    def _check_sample_settings(self) -> None:
        """Warn about sample / probe settings that give unusable images: guidance on a distilled (turbo) sampler burns
        them, a few steps or no guidance without one leaves them unfinished."""
        cfg = self.cfg
        distilled = bool(cfg.turbo_lora) or cfg.samples_on_distilled_model()
        steps0, cfg0 = self.arch.default_sample_steps, self.arch.default_sample_cfg
        for what, steps, guidance, on in (("Samples", cfg.sample_steps, cfg.sample_cfg, bool(cfg.sample_prompts)),
                                          ("Probes", cfg.probe_steps, cfg.probe_cfg, cfg.probe_every > 0)):
            if not on:
                continue
            if distilled and guidance != 1.0:
                log.warning("%s use CFG %g with a turbo (distilled) sampler, which samples without guidance: set CFG "
                            "1 or the images come out overcooked", what, guidance)
            elif not distilled and (steps < 12 or guidance <= 1.0):
                log.warning("%s render %d steps at CFG %g without a turbo LoRA or turbo sampling model, so they come "
                            "out unfinished; %s samples at %d steps, CFG %g", what, steps, guidance, self.arch.label,
                            steps0, cfg0)

    def _get_sample_model(self):
        """Lazily load the separate sampling model (kept on CPU between samplings) with matching LoRA slots."""
        cfg = self.cfg
        if self.sample_model is None:
            log.info("Loading sampling model %s (quant=%s)", cfg.sample_model_path, cfg.sample_model_quant)
            m = self.arch.load_any_transformer(cfg.sample_model_path, "cpu", cfg.sample_model_quant, cfg.models_dir)
            blocks = lora_mod.parse_blocks(cfg.lora_blocks, len(m.layers))
            inj = lora_mod.inject_lora(m, cfg.lora_rank, cfg.lora_alpha, cfg.lora_targets,
                                       include_text_embedder=cfg.train_text_embedder, blocks=blocks,
                                       names=self._lora_names(m, blocks), kind=cfg.network_type,
                                       factor=cfg.lokr_factor, block=cfg.oft_block_size,
                                       constraint=cfg.oft_constraint, dora=cfg.dora)
            source = self.race.wrappers if self.race is not None else self.injected
            if set(inj) != set(source):
                raise ValueError("Sampling model has different LoRA targets than the training model")
            self._attach_sampling_loras(m)
            self.sample_model, self.sample_injected = m, inj
        # the current LoRA (in a race: the training configuration's, with every other layer switched off)
        lora_mod.mirror_lora(self.race.wrappers if self.race is not None else self.injected, self.sample_injected)
        return self.sample_model

    @contextlib.contextmanager
    def _sampling_slots(self, model, step):
        """Yield one SampleSlot per GPU that renders this round (several only with multi_gpu + parallel_jobs).

        Without a separate sampling model the training copies render: one per GPU in data parallel, the split model
        (one slot) in model_split. With sample_model_path every training copy is moved to the CPU and the sampling
        model gets a copy on each GPU instead."""
        cfg = self.cfg
        devices = self.side_devices()
        if not cfg.sample_model_path:
            if cfg.multi_gpu == "data_parallel" and len(devices) > 1:
                pairs = [(m, norm_device(d)) for m, _, d in self.replicas]
            else:
                pairs = [(model, norm_device(self.device))]
            modes = [m.training for m, _ in pairs]
            slots = [SampleSlot(m, d, self._vae_on(d), compiled=self.compiled) for m, d in pairs]
            try:
                yield slots
            finally:
                slots = None
                for (m, _), mode in zip(pairs, modes):
                    m.train(mode)
                self._park_vae()  # back to RAM until the next round
            return
        self.status.update(stage=f"swapping in sampling model (step {step})")
        train_models = [(m, d) for m, _, d in self.replicas] or [(model, str(self.device))]
        for m, _ in train_models:
            m.to("cpu")
        free_cuda()
        base = self._get_sample_model()  # on the CPU, with the current LoRA weights
        copies = []
        try:
            for d in devices[1:]:
                self.status.update(stage=f"copying sampling model to {d} (step {step})")
                copies.append(clone_to(base, d))
            base.to(devices[0])
            slots = [SampleSlot(base, devices[0], self._vae_on(devices[0]))]
            slots += [SampleSlot(c, d, self._vae_on(d)) for c, d in zip(copies, devices[1:])]
            yield slots
        finally:
            slots = None
            copies.clear()
            base.to("cpu")
            self._park_vae()
            free_cuda()
            for m, d in train_models:
                if getattr(m, "split_assignment", None):  # straight back to its GPUs (it may not fit one)
                    from .multigpu import apply_assignment

                    apply_assignment(m, m.split_assignment)
                elif self.block_swap is not None and m is self.block_swap.model:
                    self.block_swap.reoffload()  # all but the swapped blocks' weights back (it may not fit the GPU)
                else:
                    m.to(d)

    def _render_round(self, slots, jobs, neg_e, neg_m, board, on_done=None) -> list[str]:
        """Render every job, each GPU pulling the next one; returns the image paths in job order."""
        cfg = self.cfg

        def render(slot, job):
            emb = load_text_embeds(cfg, job.prompt)
            pos_e, pos_m = self.arch.pad_embeds([emb])
            log.info("%s on %s (seed %d, %d steps, %dx%d): %s", job.label[:1].upper() + job.label[1:], slot.device,
                     job.seed, job.steps, job.width, job.height, job.prompt[:80])
            board.set(slot.device, f"{job.label} | step 0/{job.steps}")

            def on_progress(k, total, _slot=slot, _job=job):
                board.set(_slot.device, f"{_job.label} | step {k}/{total}")

            lat = self.arch.sample(slot.model, pos_e, pos_m, neg_e, neg_m, width=job.width, height=job.height,
                                   steps=job.steps, cfg=job.cfg_scale, shift=cfg.sample_shift, seed=job.seed,
                                   skip_layer=cfg.sample_skip_layer, device=slot.device, progress=on_progress)
            img = self.arch.decode_latents(slot.vae, lat)[0]
            arr = ((img.permute(1, 2, 0).float().cpu().numpy() + 1) * 127.5).round().clip(0, 255).astype("uint8")
            Image.fromarray(arr).save(job.out_path)
            board.set(slot.device, f"{job.label} done")
            return str(job.out_path)

        if len(slots) > 1 and any(sl.compiled for sl in slots):
            # A compiled model traces new graphs for sampling shapes, and two threads tracing at once crash dynamo.
            # The first round of each shape runs serially (job i on GPU i % n); later rounds run in parallel under
            # a stance that never compiles.
            needed = {(sl.device, j.shape_key) for sl in slots for j in jobs}
            if not needed <= self.sample_graphs or not stance_available():
                res = run_queue(slots, jobs, render, serial=True, on_done=on_done)
                self.sample_graphs |= {(slots[i % len(slots)].device, j.shape_key) for i, j in enumerate(jobs)}
                return res
            with (no_recompile_stance() or contextlib.nullcontext()):
                return run_queue(slots, jobs, render, on_done=on_done)
        return run_queue(slots, jobs, render, on_done=on_done)

    @torch.no_grad()
    def _free_for_rendering(self) -> None:
        """Before samples / probes: drop the finished step's gradients (the next step clears them anyway) and hand the
        allocator's cached blocks back, so a large render finds the memory training freed."""
        for m, _, _ in self.replicas:
            for p in m.parameters():
                p.grad = None
        free_cuda()

    def sample_images(self, model, step):
        self._free_for_rendering()
        with self._averaged(replicas=True), lora_mod.frozen_adapters(), self._sampling_slots(model, step) as slots:
            return self._sample_with(slots, step)

    # ------------------------------------------------------------ per-image loss watch
    def _make_loss_watch(self, items, optimizer):
        cfg = self.cfg
        if not (cfg.loss_watch or cfg.per_image_lr):
            return None
        from .losswatch import LossWatch

        if cfg.per_image_lr and cfg.optimizer.lower() in AUTOMAGIC:
            log.warning("per_image_lr is not supported with %s (its step normalisation absorbs the scaling); "
                        "the loss watch will only observe", cfg.optimizer)
            cfg.per_image_lr = False
        watch = LossWatch(cfg, items, self.job_dir, cfg.resolved_cache_dir() / "excluded.json")
        mode = ("true per-image lr" if cfg.per_image_lr and cfg.batch_size == 1 and cfg.gradient_accumulation == 1
                else "per-sample loss weights" if cfg.per_image_lr else "observe only")
        log.info("Loss watch on: %d images, %d sigma buckets, %s; exclusion after %s", len(items),
                 cfg.loss_watch_buckets, mode, cfg.loss_watch_exclude_after or "never")
        if watch.active_count() < len(items):
            self._refresh_batcher_weights()
        return watch

    # ------------------------------------------------------------ training resolutions
    def _stage_label(self) -> str:
        side = self.sides[self.res_stage] if self.sides else self.cfg.resolution
        if self.cfg.resolution_step_up:
            return f"{side_to_mp(side):g} MP (stage {self.res_stage + 1}/{len(self.sides)})"
        return describe_sides(self.cfg)

    def _enter_stage(self, stage: int, step: int, initial: bool = False) -> None:
        """Build the batcher for a resolution stage (all resolutions at once unless resolution_step_up)."""
        cfg = self.cfg
        self.res_stage = max(0, min(stage, len(self.sides) - 1))
        if cfg.resolution_step_up:
            items = items_for_side(self.all_items, self.sides[self.res_stage])
            if not items:
                raise ValueError(f"No dataset items for resolution stage {self.res_stage}")
        else:
            items = self.all_items
        self.batcher = BucketBatcher(items, cfg.batch_size, cfg.seed + step)
        self.batch_iter = iter(self.batcher)
        self._refresh_batcher_weights()
        if self.balancer is not None and not initial:  # a new resolution changes the GPUs' speeds: measure again
            self.balancer.reset()
            self.status.update(dp_counts=self.balancer.counts)
        if cfg.resolution_step_up and not initial:
            log.info("Resolution step-up: now training at %s from step %d (%d items)", self._stage_label(), step,
                     len(items))
        self.status.update(resolution=self._stage_label(), res_stage=self.res_stage, res_stages=len(self.sides))

    # ------------------------------------------------------------ block swap (see blockswap.py)
    def _setup_block_swap(self, cfg, model) -> None:
        from .blockswap import BlockSwap

        n = len(model.layers)
        k = min(int(cfg.block_swap), n - 1)
        if k < cfg.block_swap:
            log.warning("block_swap %d is more than the %d blocks allow: %d swapped", cfg.block_swap, n, k)
        self.status.update(stage="setting up block swap")
        self.block_swap = BlockSwap(model, range(k), self.devices[0])
        model._rz_block_swap = self.block_swap  # set_extras_enabled resets its prefetch (rulztrainer.lora)
        log.info("Block swap: the frozen weights of blocks 0-%d (%d of %d) stay in RAM, %.1f GB pinned, and stream to "
                 "%s as each block runs (prefetching the next)", k - 1, k, n, self.block_swap.bytes / 1e9,
                 self.devices[0])

    # ------------------------------------------------------------ LoRA Isolation (see isolation.py)
    def _isolation_prepare(self, cfg, items):
        """Fused triggers, trigger-free DOP captions, balanced subjects and the side-by-side pairs; the sample and
        probe prompts read with the fused triggers too."""
        from . import isolation as iso
        from .dataset import Item, _hash, bucket_size

        specs = cfg.dataset_specs()
        subjects, by_ds = [], {}
        for spec in specs:
            root = str(Path(spec.path))
            trigger = spec.trigger or cfg.trigger_word
            if not trigger:
                log.info("LoRA Isolation: %s has no trigger word: it trains as it is, not as a subject", spec.path)
                continue
            caps = [it.caption for it in items if it.dataset == root and it.category == "image"]
            cls = spec.subject_class if cfg.datasets else cfg.isolation_class
            kind = spec.subject_kind if cfg.datasets else cfg.isolation_kind
            s = iso.resolve_subject(spec.path, trigger, caps, cls, kind)
            subjects.append(s)
            by_ds[root] = (s, spec)
        if not subjects:
            raise ValueError("LoRA Isolation needs a trigger word: set trigger_word, or a trigger on each subject "
                             "dataset")
        rules = [r for s in subjects for r in s.rules]
        dop_pairs = [s.dop_pair for s in subjects] + [(s.trigger, s.phrase) for s in subjects if s.rules]
        for it in items:
            it.caption = iso.fuse(it.caption, rules)
            it.text_path = text_cache_path(cfg, it.caption)
        if cfg.isolation_balance and len(by_ds) > 1:
            counts = {d: len({it.image for it in items if it.dataset == d and it.category == "image"}) for d in by_ds}
            reps = iso.balance_repeats(counts, {d: spec.repeats for d, (_, spec) in by_ds.items()})
            for it in items:
                if it.dataset in reps:
                    it.repeats = reps[it.dataset]
            log.info("LoRA Isolation: subjects balanced: %s", ", ".join(
                f"{by_ds[d][0].fused} {counts[d]} images x{r}" for d, r in reps.items()))
        n_pairs = 0
        people = None
        for d, (s, spec) in by_ds.items():
            if s.kind != "person" or cfg.isolation_pairs <= 0:
                continue
            if people is None:
                people = iso.strangers(cfg.models_dir)
            photos = sorted({it.image for it in items if it.dataset == d and it.category == "image"})
            cache = cfg.dataset_cache_dir(Path(d))
            pairs = iso.build_pairs(s, photos, cache / "isolation" / s.fused, people, cfg.isolation_pairs, cfg.seed,
                                    getattr(self, "face_analysis", None))
            reps = next((it.repeats for it in items if it.dataset == d), spec.repeats)
            tag = get_arch(cfg).latent_tag
            for png, caption in pairs:
                caption = iso.fuse(caption, rules)
                for side in cfg.dataset_sides(spec):
                    bw, bh = bucket_size(2 * iso.PANEL, iso.PANEL, cfg, side)
                    stem = _hash(f"{png.resolve()}|{png.stat().st_mtime_ns}") + f"_{bw}x{bh}"
                    items.append(Item(image=png, caption=caption, width=bw, height=bh,
                                      latent_path=cache / "latents" / f"{stem}{tag}.safetensors", latent_flip_path=None,
                                      text_path=text_cache_path(cfg, caption), category="pair", side=side,
                                      repeats=reps, dataset=d, loss_scale=cfg.isolation_pair_weight,
                                      pair_side=iso.pair_side(caption, s.fused)))
                    n_pairs += 1
        n_dop = 0
        for it in items:
            dc = iso.dop_caption(it.caption, dop_pairs)
            it.dop_caption = dc
            it.dop_text_path = text_cache_path(cfg, dc) if dc is not None else None
            n_dop += dc is not None
        cfg.sample_prompts = [iso.fuse(p, rules) for p in cfg.sample_prompts]
        cfg.probe_face_prompt = iso.fuse(cfg.probe_face_prompt, rules)
        cfg.probe_body_prompt = iso.fuse(cfg.probe_body_prompt, rules)
        cfg.reward_prompts = [iso.fuse(p, rules) for p in cfg.reward_prompts]
        cfg.selftrain_prompts = [iso.fuse(p, rules) for p in cfg.selftrain_prompts]
        self.isolation = subjects
        for s in subjects:
            log.info("LoRA Isolation: %s (%s) trains as '%s'%s; prompt with '%s'", s.trigger, s.kind, s.fused,
                     f", its trigger-free caption says '{s.phrase}'" if s.phrase else ", dropped from the trigger-free "
                     "caption", s.fused)
        log.info("LoRA Isolation: %d side-by-side pair item(s) at x%g, DOP on %d item(s) every %d step(s) at x%g",
                 n_pairs, cfg.isolation_pair_weight, n_dop, cfg.dop_every, cfg.dop_weight)
        return items

    def _dop_job(self, cfg, batch, latents, cur_step: int, gen, shift: float, tokens: int, step_sigma=None):
        """The step's trigger-free pass for a batch, drawn in the main thread with the run's random streams: (the
        batch rows with a DOP caption, their text features and mask, fresh sigmas and noise), or None. `step_sigma`:
        the step's shared sigma (keel_pair_sigma), used instead of fresh ones."""
        if not cfg.isolation or self.isolation is None or cfg.dop_weight <= 0 or cur_step % max(1, cfg.dop_every):
            return None
        sel = [i for i, it in enumerate(batch) if it.dop_text_path is not None]
        if not sel:
            return None
        from safetensors.torch import load_file
        emb, mask = self.arch.pad_embeds([load_file(str(batch[i].dop_text_path))["embeds"] for i in sel])
        sigma = (step_sigma.expand(len(sel)).clone() if step_sigma is not None
                 else sample_sigmas(cfg, len(sel), gen, shift, tokens=tokens, arch=self.arch))
        noise = torch.randn((len(sel), *latents.shape[1:]), generator=gen)
        return sel, emb, mask, sigma, noise

    def _dop_pass(self, rmodel, rinj, rdev, latents, n_batch: int, dop, n_total: int, cfg) -> None:
        """DOP (Differential Output Preservation, Ostris's AI-Toolkit): a fresh noising of the pictures with the
        trigger swapped for its class, held to the model's own prediction there with the LoRA at 0, so without the
        trigger the LoRA learns to change nothing. Backpropagated here."""
        sel, emb, mask, sigma, noise = dop
        lat = latents[sel]
        s = sigma.to(rdev).view(-1, 1, 1, 1)
        noisy = ((1 - s) * lat + s * noise.to(rdev, torch.float32)).to(torch.bfloat16)
        t = sigma.to(rdev) * 1000.0
        emb, mask = emb.to(rdev), mask.to(rdev)
        lora_mod.set_multiplier(rinj, 0.0)
        try:
            with torch.no_grad():
                target = rmodel(noisy, t, emb, mask).float()
        finally:
            lora_mod.set_multiplier(rinj, 1.0)
        pred = rmodel(noisy, t, emb, mask).float()
        loss = torch.nn.functional.mse_loss(pred, target)
        (loss * (cfg.dop_weight * len(sel) / max(1, n_batch) / n_total)).backward()
        self.dop_values.append(float(loss.detach()))

    def _epoch_images(self, batcher) -> int:
        """Images in one epoch of an image queue (repeats counted)."""
        cache = self.__dict__.setdefault("_epoch_sizes", {})
        size = cache.get(id(batcher))
        if size is None:
            size = cache[id(batcher)] = max(1, sum(max(1, int(it.repeats or 1)) for it in batcher.items))
        return size

    def _clean_schedule(self, cfg, batcher, images_per_step: int):
        """clean_share: (the configuration's schedule, the images in one of its epochs), or None when unset. One
        schedule per configuration (a race has several). A full fine-tune counts rotations: a cycle of every window."""
        if not cfg.clean_share or batcher is None:
            return None
        from .clean_share import CleanShare

        shares = self.__dict__.setdefault("_clean_shares", {})
        cs = shares.get(id(cfg))
        if cs is None:
            cs = shares[id(cfg)] = CleanShare(cfg, self.arch, log)
        ft = self.__dict__.get("ft")
        if ft is not None and ft.windows:
            cs.unit = "rotation"
            return cs, self.ft_rotate_steps * len(ft.windows) * images_per_step
        return cs, self._epoch_images(batcher)

    def _clean_offset(self, cfg, steps_done: int, tokens: int, batcher, images_per_step: int) -> float:
        """clean_share: the logit offset that puts the epoch's clean-end share of steps below sigma 0.5 (0 when
        unset)."""
        sched = self._clean_schedule(cfg, batcher, images_per_step)
        if sched is None:
            return 0.0
        cs, unit_images = sched
        return cs.offset(steps_done * images_per_step, unit_images, tokens)

    def _clean_lr_scale(self, cfg, steps_done: int, batcher, images_per_step: int) -> float:
        """clean_share: half the learning rate on the lower-share epochs after the slide, else 1."""
        sched = self._clean_schedule(cfg, batcher, images_per_step)
        if sched is None:
            return 1.0
        cs, unit_images = sched
        return cs.lr_scale(steps_done * images_per_step, unit_images)

    def _look_outlier_images(self, cfg, items) -> set[str]:
        """look_warmup: the images (as the items name them) that the look consistency filter's saved scan puts under
        its fence, in every dataset of the run."""
        race_rows = self.race_rows or []
        if not (cfg.look_warmup or any(r[1].look_warmup for r in race_rows)):
            return set()
        from . import lookfilter

        found, scanned = set(), 0
        for spec in cfg.dataset_specs():
            if any(lookfilter.load_scores(spec.path, k) is not None for k in lookfilter.KINDS):
                scanned += 1
            found |= lookfilter.outliers(spec.path)
        if not scanned:
            log.warning("look_warmup: no look consistency scan in the dataset folder(s) (look_scores.json / look_scores_body.json): "
                        "run the Look consistency filter on the Dataset tab first; warm-up off this run")
            return set()
        names = {str(it.image) for it in items if str(Path(it.image).resolve()) in found}
        log.info("look_warmup: %d look outlier image(s) start at a loss weight of x%g, full after %g epoch(s)",
                 len(names), cfg.look_warmup_start, cfg.look_warmup_epochs)
        return names

    def _look_factors(self, batch, cur_step: int, cfg, batcher, images_per_step: int) -> list[float] | None:
        """Each batch item's look warm-up loss weight (None: all 1). The epoch is the images trained so far over the
        images of one epoch (repeats counted) of the configuration's own image queue."""
        if not getattr(self, "look_outliers", None) or not cfg.look_warmup or batcher is None:
            return None
        size = self._epoch_images(batcher)
        from .lookfilter import warmup_factor

        f = warmup_factor((cur_step - 1) * images_per_step / size, cfg.look_warmup_start, cfg.look_warmup_epochs)
        if f >= 1.0:
            return None
        return [f if str(it.image) in self.look_outliers else 1.0 for it in batch]

    def _refresh_batcher_weights(self) -> None:
        """Combine curriculum weights with loss-watch exclusions into the batcher."""
        if self.batcher is None:
            return
        weights: dict[str, float] | None = dict(self.curriculum_w) if self.curriculum_w else None
        if (self.cfg.crop_augment or "none") != "none" and self.cfg.crop_augment_weight != 1.0:
            weights = weights or {}
            for it in self.batcher.items:
                if it.category not in ("image", "selftrain"):  # the crops (self-training has its own loss weight)
                    weights[it.key] = weights.get(it.key, 1.0) * self.cfg.crop_augment_weight
        if self.loss_watch is not None:
            mask = self.loss_watch.sampling_mask()
            if any(v == 0.0 for v in mask.values()):
                weights = {k: (weights.get(k, 1.0) if weights else 1.0) * m for k, m in mask.items()}
                if not any(weights.values()):
                    log.warning("Loss watch excluded every image; ignoring exclusions")
                    weights = dict(self.curriculum_w) if self.curriculum_w else None
        self.batcher.set_weights(weights)

    # ------------------------------------------------------------ likeness probes
    def _make_controller(self, items):
        cfg = self.cfg
        if cfg.probe_every <= 0:
            return None
        if self.face_scorer is None and self.body_scorer is None:
            log.warning("Likeness probes need face_refs and/or body_refs (with the scoring packages); probes disabled")
            return None
        if not (cfg.probe_face_prompt or cfg.probe_body_prompt):
            log.warning("Likeness probes need a face and/or body probe prompt; probes disabled")
            return None
        from .likeness import LikenessController, classify_counts, face_area_fractions

        ctrl = LikenessController(cfg, **self._controller_ceilings())
        if cfg.likeness_curriculum:
            if self.face_scorer is None:
                log.warning("Curriculum needs face_refs (face detector); curriculum disabled")
            else:
                self.status.update(stage="classifying dataset images (face area)")
                self.item_face_frac = face_area_fractions(
                    self.face_scorer, items, cfg.resolved_cache_dir() / "faces.json",
                    progress=lambda i, n: self.status.update(step=i, total=n), recache=cfg.recache)
                nf, nb = classify_counts(self.item_face_frac)
                log.info("Curriculum: %d face-heavy and %d body/other images in the dataset", nf, nb)
        log.info("Likeness controller on: probe every %d steps (%s), best=%s patience=%d lr_control=%s curriculum=%s",
                 cfg.probe_every, "+".join(k for k, p in (("face", cfg.probe_face_prompt), ("body", cfg.probe_body_prompt)) if p),
                 cfg.likeness_track_best, cfg.likeness_patience, cfg.likeness_lr_control, cfg.likeness_curriculum)
        return ctrl

    def _setup_alignment(self, cfg, items) -> None:
        """Attention localization / identity alignment: taps on every model copy, the trigger's token positions in
        every cached text, and the alignment's projector (a race: whenever any configuration uses them)."""
        from . import alignment as A

        rows = [r[1] for r in self.race_rows] if self.race_rows is not None else [cfg]
        attn, align = any(c.attn_localize for c in rows), any(c.id_align or c.body_align for c in rows)
        if not (attn or align):
            if (any(c.optimizer.lower() in KEEL and c.keel_face_gain > 0 for c in rows) and self.ft is None
                    and not self.race_parallel):
                for m, _, _ in self.replicas:
                    m._rz_taps = A.Taps()  # the token layout alone, for Keel's face tap (keel_face_gain)
            return
        if self.ft is not None:
            log.warning("Attention localization and identity alignment are adapter training losses; a fine-tune "
                        "trains without them")
            return
        if self.race_parallel:
            log.warning("The parallel race trains without attention localization and identity alignment; run the "
                        "race one configuration at a time to compare them")
            return
        model = self.replicas[0][0]
        n = len(model.layers)
        attn_layers = []
        if attn:
            attn_layers = A.parse_layers(cfg.attn_localize_layers, n) or A.auto_attn_layers(
                self.arch.name, n, getattr(model, "injection_layer_ids", ()))
        self.align_layer = (int(cfg.id_align_layer) if cfg.id_align_layer >= 0 else A.auto_hidden_layer(n)) \
            if align else None
        if self.align_layer is not None and not 0 <= self.align_layer < n:
            raise ValueError(f"id_align_layer {self.align_layer} is outside the model's {n} blocks")
        for m, _, _ in self.replicas:
            taps = A.Taps([self.align_layer] if align else [], attn_layers)
            m._rz_taps = taps
            for i in attn_layers:
                block = m.layers[i]
                module = A.attention_module(block)
                if module is None:
                    raise ValueError(f"block {i} has no attention that reads the caption token by token (FIBO 2.0: "
                                     f"use its injection blocks {list(getattr(m, 'injection_layer_ids', ()))})")
                module._rz_tap = A.AttnTap(taps, i)
                block._rz_tapped = True  # runs uncompiled: the tap is Python
        if attn:
            self.trigger_pos = self._trigger_positions(items, rows)
            n_pos = sum(1 for p in self.trigger_pos.values() if p)
            log.info("Attention localization: weight %g in blocks %s; the trigger's tokens found in %d of %d cached "
                     "texts", cfg.attn_localize_weight, ",".join(map(str, attn_layers)), n_pos, len(self.trigger_pos))
            if not n_pos:
                log.warning("Attention localization: the trigger word is in none of the captions; nothing to localize")
        if align:
            if self.race_rows is None:
                self.id_aligner, self.id_align_opt = self._make_id_aligner(cfg)
            dim = A.model_dim(model)
            for on, what, weight, out in ((any(c.id_align for c in rows), "face tokens", cfg.id_align_weight,
                                           "photo's ArcFace embedding", ),
                                          (any(c.body_align for c in rows), "person's tokens outside the face",
                                           cfg.body_align_weight, "photo's DINOv2 body-window embedding")):
                if on:
                    log.info("Alignment: weight %g, the %s after block %d mapped onto each %s by a projector that is "
                             "not saved", weight, what, self.align_layer, out)

    def _make_id_aligner(self, cfg):
        """(projectors, their AdamW) for face / body alignment, or (None, None)."""
        from . import alignment as A

        if not (cfg.id_align or cfg.body_align) or self.align_layer is None:
            return None, None
        proj = A.Aligners(A.model_dim(self.replicas[0][0]), cfg.id_align, cfg.body_align).to(self.devices[0])
        return proj, torch.optim.AdamW(proj.parameters(), lr=1e-4, weight_decay=0.0)

    def _analyze_bodies(self, items, cache) -> None:
        """Body alignment's targets: each source photo's DINOv2 body-window embedding (cached in bodies.json)."""
        from . import bodysim
        from . import regions as R

        ok, msg = bodysim.available()
        if not ok:
            log.warning("%s; body alignment has no targets", msg)
            return
        self.status.update(stage="embedding the photos' bodies")
        scorer = bodysim.AppearanceScorer(self.cfg.models_dir, self.cfg.body_device,
                                          progress=lambda t, f: self.status.update(stage=t))

        def embed(path):
            win = scorer.window(path)
            if win is None or win[1] < bodysim.WINDOW_MIN_SCORE_COVER:
                return None
            return scorer._embed(win[0])

        try:
            found = R.analyze_bodies(items, embed, cache / "bodies.json",
                                     progress=lambda i, n: self.status.update(step=i, total=n),
                                     recache=self.cfg.recache)
        finally:
            if scorer.model is not None:
                unload_model(scorer.model)
            del scorer
            free_cuda()
        self.body_embs = {k: torch.tensor(v, dtype=torch.float32) for k, v in found.items() if v}
        log.info("Body alignment: %d of %d source images show the body from the neck to mid-thigh", len(self.body_embs),
                 len(found))

    def _trigger_positions(self, items, rows) -> dict[str, list[int]]:
        """{cached text path: the trigger's token positions in it} for every caption training reads (each dataset's
        trigger, LoRA Isolation's fused words, caption dropout's prompt)."""
        from . import alignment as A
        from .dataset import dropout_prompt, text_cache_path

        cfg = self.cfg
        triggers = {c.trigger_word for c in rows} | {sp.trigger for sp in cfg.dataset_specs()}
        for s in self.isolation or []:
            triggers |= {getattr(s, "fused", ""), getattr(s, "trigger", "")}
        triggers = [t for t in triggers if t]
        captions = {str(text_cache_path(cfg, it.caption)): it.caption for it in items}
        for c in rows:
            captions[str(text_cache_path(c, dropout_prompt(c)))] = dropout_prompt(c)
        tok = self.arch.get_tokenizer(cfg, self.model_dir)
        out, mismatched = {}, 0
        for path, caption in captions.items():
            text, offsets, drop, length = self.arch.prompt_tokens(tok, caption)
            pos = A.token_positions(text, offsets, triggers, drop)
            if Path(path).exists():
                cached = self._cached_text_length(path)
                if cached is not None and cached != length:
                    mismatched += 1
                    pos = []  # the cache was made another way: positions would point at the wrong tokens
            out[path] = [p for p in pos if p < length]
        if mismatched:
            log.warning("Attention localization: %d cached texts do not have the length their caption tokenizes to "
                        "(recache to include them)", mismatched)
        return out

    def _cached_text_length(self, path) -> int | None:
        from safetensors import safe_open

        try:
            with safe_open(str(path), "pt") as f:
                shape = f.get_slice("embeds").get_shape()
        except Exception:  # noqa: BLE001
            return None
        return int(shape[1]) if self.arch.name == "fibo2" else int(shape[0])

    def _alignment_losses(self, cfg, taps, batch, regions, use_attn: bool, use_align: bool):
        """The weighted sum of this batch's attention localization and identity alignment losses (None when no
        sample had what they need)."""
        from . import alignment as A

        terms = []
        if use_attn:
            hh, ww = regions.prior.shape[-2:]  # a pair's subject is its own panel; the stranger's counts as leakage
            val, _ = A.localization_loss(taps, regions, [A.panel_mask(getattr(it, "pair_side", ""), hh, ww)
                                                         for it in batch])
            if val is not None:
                self.align_values["attn"].append(float(val.detach()))
                self.align_values["pairs"].append(sum(1 for it in batch if getattr(it, "pair_side", "")))
                terms.append(cfg.attn_localize_weight * val)
        if use_align:
            for key, on, proj, embs, weight in (("id", cfg.id_align, self.id_aligner.face, self.face_embs,
                                                 cfg.id_align_weight),
                                                ("body", cfg.body_align, self.id_aligner.body, self.body_embs,
                                                 cfg.body_align_weight)):
                if not on or proj is None:
                    continue
                targets = [embs.get(str(it.image.resolve())) for it in batch]
                val, _ = proj.loss(taps, self.align_layer, regions, targets)
                if val is not None:
                    self.align_values[key].append(float(val.detach()))
                    terms.append(weight * val)
        if not terms:
            return None
        return sum(t.to(terms[0].device) for t in terms[1:]) + terms[0] if len(terms) > 1 else terms[0]

    # ------------------------------------------------------------ likeness reward fine-tuning (see reward.py)
    def _reward_stage(self, model, injected, step0: int):
        """The optional second stage: reward_steps policy-gradient updates on the likeness of the LoRA's own pictures
        (Flow-GRPO). Starts from the LoRA the steps saved (their weight average), saves <job>_reward_<step> files and
        <job>_reward.safetensors, and keeps probing. Returns the final file."""
        from . import reward as RW
        from .likeness import probe_body_size

        cfg = self.cfg
        if self.face_scorer is None and self.body_scorer is None:
            log.warning("Reward stage skipped: no face or body scorer (references and the scoring packages)")
            return None
        if self.dp is not None:
            log.info("Reward stage: renders and trains on %s alone (the other GPUs wait)", self.devices[0])
        if self.ema is not None:  # start from the averaged LoRA: the one the steps saved
            with torch.no_grad():
                for p, s in zip(self.ema.params, self.ema.shadow):
                    p.copy_(s)
            self.ema = None
        params = list(lora_mod.lora_parameters(injected))
        ref = [p.detach().clone() for p in params]  # the KL reference: the LoRA as the stage starts
        opt = torch.optim.AdamW(params, lr=cfg.reward_lr, weight_decay=0.0)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
        dropouts = {name: m.dropout for name, m in injected.items()}
        for m in injected.values():
            m.dropout = torch.nn.Identity()  # the policy must be the same function when it renders and when it trains
        scale = self._likeness_scale()
        steps_t = cfg.reward_sample_steps or cfg.probe_steps
        dev = norm_device(self.device)
        skip = [cfg.sample_skip_layer] if self.arch.name == "moss_image" and cfg.sample_skip_layer >= 0 else None
        neg_row = load_text_embeds(cfg, self.negative_prompt())
        out_dir = self.job_dir / "reward"
        out_dir.mkdir(exist_ok=True)
        rng = random.Random(cfg.seed + 7)
        group = int(cfg.reward_group)
        total = step0 + cfg.reward_steps
        log.info("Reward stage: %d steps of %d images (%d denoising steps, guidance %g, noise %.2g), %d of them trained "
                 "per update, lr %.1e, KL %g, %d prompts; scored by %s", cfg.reward_steps, group, steps_t,
                 cfg.probe_cfg, cfg.reward_noise, min(cfg.reward_train_steps, steps_t - 1), cfg.reward_lr,
                 cfg.reward_kl, len(cfg.reward_prompts),
                 " + ".join(k for k, s in (("face", self.face_scorer), ("body", self.body_scorer)) if s is not None))
        self.stage_tag = "_reward"
        self._gold_gap0, self._gold_gap_warned = self._gold_gap(), False  # the gold check's starting point
        self.status.update(stage="reward fine-tuning", total=total)
        model.train()
        lora_mod.set_extras_enabled(model, True)  # the policy is the sampler the probes use: turbo LoRA included
        kept = ckpt.kept_blocks(model)  # measured for one training image; an update's guidance pass needs more:
        ckpt.set_kept_blocks(model, [])  # every block recomputes during the stage
        history = self.status.data.get("rewards", [])
        self._reward_body_seen = self._reward_body_low = 0  # for the clothing / framing mismatch warning
        s = probed = 0
        t0 = time.time()
        try:
            for s in range(1, cfg.reward_steps + 1):
                if self.stop_file.exists():
                    log.info("Reward stage stopped at step %d", s - 1)
                    s -= 1
                    break
                prompt = cfg.reward_prompts[(s - 1) % len(cfg.reward_prompts)]
                size = cfg.reward_size or cfg.probe_size
                w, h = probe_body_size(size) if RW.is_full_body(prompt) else (size, size)
                pos_e, pos_m = self.arch.pad_embeds([load_text_embeds(cfg, prompt)] * group)
                pos_e, pos_m = pos_e.to(dev), pos_m.to(dev)
                neg_e = neg_m = None
                if cfg.probe_cfg != 1.0:
                    neg_e, neg_m = self.arch.pad_embeds([neg_row] * group)
                    neg_e, neg_m = neg_e.to(dev), neg_m.to(dev)
                sigmas = self.arch.sample_sigmas(model, steps_t, h, w, cfg.sample_shift)
                sig = [float(x) for x in sigmas]
                seeds = [cfg.seed * 100_003 + s * 97 + k for k in range(group)]
                self.status.update(stage=f"reward fine-tuning {s}/{cfg.reward_steps}: rendering {group} images")
                with lora_mod.frozen_adapters():
                    xs, stds = RW.sample_group(model, sigmas, pos_e, pos_m, neg_e, neg_m, cfg_scale=cfg.probe_cfg,
                                               noise_level=cfg.reward_noise, seeds=seeds,
                                               shape=(model.latent_channels, h // 16, w // 16), device=dev, skip=skip)
                rewards = []
                vae = self._home_vae()  # parked again once the group is decoded
                for k in range(group):
                    img = self.arch.decode_latents(vae, xs[-1][k:k + 1])[0]
                    arr = ((img.permute(1, 2, 0).float().cpu().numpy() + 1) * 127.5).round().clip(0, 255)
                    path = out_dir / f"step{step0 + s:06d}_{k}.png"
                    Image.fromarray(arr.astype("uint8")).save(path)
                    rewards.append(self._reward_of(path, prompt, scale))
                vae = None
                self._park_vae()
                adv = RW.advantages(rewards)
                kl_val = None
                if adv is not None:
                    self.status.update(stage=f"reward fine-tuning {s}/{cfg.reward_steps}: updating")
                    idx = [j for j, sd in enumerate(stds) if sd > 0]
                    idx = sorted(rng.sample(idx, min(int(cfg.reward_train_steps), len(idx))))
                    opt.zero_grad(set_to_none=True)
                    adv_d, kls = adv.to(dev), []
                    for j in idx:
                        for k in range(group):  # one picture at a time: a training step's memory
                            one = slice(k, k + 1)
                            x_j, x_next = xs[j][one], xs[j + 1][one]
                            ne = None if neg_e is None else neg_e[one]
                            nm = None if neg_m is None else neg_m[one]
                            v = RW.velocity(model, x_j, sig[j], pos_e[one], pos_m[one], ne, nm, cfg.probe_cfg, skip)
                            mean, std = RW.sde_step_stats(x_j, v, sig[j], sig[j + 1], sig[1], cfg.reward_noise)
                            lp = RW.log_prob(x_next, mean, std)
                            with torch.no_grad():
                                self._swap_weights(params, ref)
                                try:
                                    v_ref = RW.velocity(model, x_j, sig[j], pos_e[one], pos_m[one], ne, nm,
                                                        cfg.probe_cfg, skip)
                                finally:
                                    self._swap_weights(params, ref)
                                mean_ref, _ = RW.sde_step_stats(x_j, v_ref, sig[j], sig[j + 1], sig[1],
                                                                cfg.reward_noise)
                            kl = RW.kl_to(mean, mean_ref, std)
                            ratio = torch.exp(lp - lp.detach())  # 1 on-policy; its gradient is the log-probability's
                            ((-(adv_d[k] * ratio) + cfg.reward_kl * kl).sum() / (len(idx) * group)).backward()
                            kls.append(float(kl.mean()))
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    opt.step()
                    sched.step()
                    kl_val = sum(kls) / len(kls)
                mean_r = sum(rewards) / len(rewards)
                history.append([step0 + s, round(mean_r * 100, 2)])
                log.info("Reward step %d/%d: likeness %s (mean %.1f) for \"%s\"%s | %.1fs/step", s, cfg.reward_steps,
                         " / ".join(f"{r * 100:.0f}" for r in rewards), mean_r * 100, prompt[:60],
                         f" | KL {kl_val:.4f}" if kl_val is not None else " | all alike: no update",
                         (time.time() - t0) / s)
                self.status.update(step=step0 + s, total=total, rewards=history[-2000:],
                                   stage=f"reward fine-tuning {s}/{cfg.reward_steps}")
                if cfg.save_every > 0 and s % cfg.save_every == 0 and s < cfg.reward_steps:
                    self.save(injected, step0 + s)
                if self.controller is not None and cfg.probe_every > 0 and s % cfg.probe_every == 0:
                    lora_mod.set_extras_enabled(model, False)
                    stop = self.run_probe(model, step0 + s, opt, sched, injected)
                    probed = s
                    model.train()
                    lora_mod.set_extras_enabled(model, True)
                    if stop:
                        log.info("Reward stage: the likeness probes stopped improving; ending at step %d", s)
                        break
        finally:
            lora_mod.set_extras_enabled(model, False)
            ckpt.set_kept_blocks(model, kept)
            for name, m in injected.items():
                m.dropout = dropouts[name]
        if s > probed and not self.stop_file.exists():
            self._reward_final_probe(model, step0 + s, opt, sched, injected)
        path = self.save(injected, step0 + s, final=True)
        self.stage_tag = ""
        log.info("Reward stage done: %s (the first stage's LoRA is %s.safetensors)", path.name, cfg.job_name)
        return path

    def _reward_final_probe(self, model, step: int, opt, sched, injected) -> None:
        """Probe the reward stage's final LoRA when its last step wasn't probed (probe_every not dividing reward_steps,
        or larger than it), so the stage's result has its likeness score, on the run's scale, next to the first
        stage's."""
        if self.controller is None:
            log.info("Reward stage: no likeness probes (probe_every 0 or no references), so no closing likeness check")
            return
        log.info("Reward stage: probing the final LoRA's likeness (step %d)", step)
        self.status.update(stage="reward fine-tuning: closing likeness probe")
        self.run_probe(model, step, opt, sched, injected)

    @staticmethod
    @torch.no_grad()
    def _swap_weights(params: list, other: list) -> None:
        """Exchange the adapter weights with another set in place (twice restores them)."""
        for i, p in enumerate(params):
            p.data, other[i] = other[i], p.data

    def _reward_of(self, path, prompt: str, scale) -> float:
        """A reward picture's likeness: on the probes' scale but not curved (0 = a stranger, 1 = the references;
        below 0 and above 1 still count, so a weak LoRA's pictures still differ, each part within reward.PART_MIN ..
        PART_MAX), combined with face_weight / body_weight. A missing face counts -1 (every reward prompt shows one),
        a missing body only in a full-body prompt."""
        from . import reward as RW
        from .likeness import combine

        cfg = self.cfg

        def linear(kind, value):
            if scale is None or not getattr(scale, "normalize", False):
                return value / 100.0
            lo, hi = scale.scale(kind)
            return (value - lo) / (hi - lo)

        face = body = None
        if self.face_scorer is not None and cfg.face_weight > 0:
            sc = self.face_scorer.score(path, 0, 0)
            face = -1.0 if sc.percent is None else RW.clip_part(linear("face", sc.percent))[0]
        if self.body_scorer is not None and cfg.body_weight > 0:
            pct, _ = self.body_scorer.score(path)
            if pct is not None:
                body, low = RW.clip_part(linear("body", pct))
                self._reward_body_seen = getattr(self, "_reward_body_seen", 0) + 1
                self._reward_body_low = getattr(self, "_reward_body_low", 0) + int(low)
                if (self._reward_body_seen == int(cfg.reward_group) * max(1, len(cfg.reward_prompts))
                        and self._reward_body_low * 2 >= self._reward_body_seen):
                    log.warning("Reward stage: %d of the first %d pictures' body scores sit far below a stranger's: "
                                "the reward prompts and the body references probably differ in clothing or "
                                "framing. Those body scores are capped, so the face decides; reward prompts like "
                                "the body references (or Body weight 0) make the body count again",
                                self._reward_body_low, self._reward_body_seen)
            elif RW.is_full_body(prompt):
                body = -1.0
        comb = combine(face, body, cfg.face_weight, cfg.body_weight)
        return -1.0 if comb is None else float(comb)

    # ------------------------------------------------------------ likeness-filtered self-training (see selftrain.py)
    def _selftrain_setup(self, step: int) -> None:
        """The rounds' steps; a resumed run trains on with the last round's pictures."""
        from . import selftrain as ST

        cfg = self.cfg
        if self.face_scorer is None and self.body_scorer is None:
            log.warning("Self-training off: no face or body scorer (references and the scoring packages)")
            return
        self.selftrain_steps = ST.round_steps(cfg.steps, cfg.selftrain_at)
        log.info("Self-training: round(s) at step %s, %d pictures each over %d prompts; kept above %g and the probes' "
                 "smoothed score, at most %d per round (%g%% of the real images), trained at x%g",
                 ", ".join(map(str, self.selftrain_steps)) or "(none: too few steps)", cfg.selftrain_images,
                 len(cfg.selftrain_prompts), cfg.selftrain_min_likeness,
                 ST.cap_count(self.all_items, cfg.selftrain_share), cfg.selftrain_share * 100, cfg.selftrain_weight)
        if step > 0:
            last = ST.last_kept(self.job_dir / "selftrain" / "selftrain.json", step)
            if last and last[1]:
                self._selftrain_add(last[1], step)
                log.info("Self-training: round %d's %d picture(s) train on from the resumed step %d", last[0],
                         len(last[1]), step)

    def _selftrain_rounds(self, cfg) -> list[int]:
        """The steps cfg's self-training rounds run at (none when it is off or nothing can score the pictures)."""
        if not cfg.selftrain or (self.face_scorer is None and self.body_scorer is None):
            return []
        from .selftrain import round_steps

        return round_steps(cfg.steps, cfg.selftrain_at)

    def _self_train_round(self, model, step: int, rnd: int) -> None:
        """Render selftrain_images pictures with the LoRA (its weight average, as the probes), score them like the
        probes, keep the best that pass the quality filter and the bar, and train on them in place of the last
        round's."""
        from . import selftrain as ST
        from .likeness import probe_body_size
        from .reward import is_full_body

        cfg = self.cfg
        t0 = time.time()
        out_dir = self.job_dir / "selftrain"
        out_dir.mkdir(exist_ok=True)
        cap = ST.cap_count(self.all_items, cfg.selftrain_share)
        if cap < 1:
            log.warning("Self-training round %d skipped: %g of the real images is less than one picture", rnd,
                        cfg.selftrain_share)
            return
        side = max(cfg.training_sides()) // 16 * 16  # the largest resolution: no item is trained upscaled
        jobs = []
        prompts = ST.plan(cfg.selftrain_prompts, cfg.selftrain_images)
        for i, prompt in enumerate(prompts):
            w, h = probe_body_size(side) if is_full_body(prompt) else (side, side)
            jobs.append(RenderJob(prompt, w, h, cfg.sample_steps, cfg.sample_cfg, ST.seed(cfg.seed, rnd, i),
                                  out_dir / f"{self._image_prefix()}round{rnd}_{i:02d}.png",
                                  f"self-training picture {i + 1}/{len(prompts)}", "selftrain"))
        log.info("Self-training round %d at step %d: rendering %d pictures", rnd, step, len(jobs))
        paths = self._selftrain_render(model, step, rnd, jobs)
        self.status.update(stage=f"self-training round {rnd}: scoring {len(paths)} pictures")
        scale = self._likeness_scale()
        cands = [self._selftrain_candidate(p, job.prompt, scale, side) for job, p in zip(jobs, paths)]
        # the candidates are scored with ArcFace: the bar is the probes' ArcFace combined score, gold face or not
        probe = getattr(self.controller, "arc_combined_ema", None) if self.controller is not None else None
        bar = max(cfg.selftrain_min_likeness, probe) if probe is not None else cfg.selftrain_min_likeness
        need_face = self.face_scorer is not None and cfg.face_weight > 0
        copy_max = self._selftrain_copy_limit()
        kept, reasons = ST.select(cands, need_face=need_face, bar=bar, cap=cap, copy_max=copy_max)
        ST.record(out_dir / f"{self._image_prefix()}selftrain.json", rnd, step, bar, cands, copy_max)
        self.status.update(stage=f"self-training round {rnd}: adding {len(kept)} pictures")
        n_items = self._selftrain_add([(c.path, c.prompt) for c in kept], step)
        scores = sorted((c.likeness for c in cands if c.likeness is not None), reverse=True)
        copies = sorted((c.copy for c in cands if c.copy is not None), reverse=True)
        log.info("Self-training round %d at step %d: kept %d of %d (bar %.1f: the floor %g%s; best %s%s); left out: "
                 "%s; %d training item(s) at x%g in place of the last round's (%.0fs)", rnd, step, len(kept),
                 len(cands), bar, cfg.selftrain_min_likeness,
                 f", the probes' smoothed {probe:.1f}" if probe is not None else "",
                 " / ".join(f"{s:.0f}" for s in scores[:5]) or "n/a",
                 f"; copy limit {copy_max:.2f}, closest {copies[0]:.2f}" if copy_max is not None and copies else "",
                 ST.describe(reasons), n_items, cfg.selftrain_weight, time.time() - t0)
        self.status.update(stage="training", selftrain={
            "round": rnd, "step": step, "rendered": len(cands), "kept": [c.path for c in kept], "bar": round(bar, 2),
            "left_out": dict(reasons), "copy_limit": None if copy_max is None else round(copy_max, 3)})
        free_cuda()

    def _selftrain_copy_limit(self) -> float | None:
        """The copy score above which a self-training picture copies a photo (copycheck.copy_limit), None when this
        configuration's copy check is off."""
        checker = getattr(self, "copy_checker", None)
        if checker is None or not self.cfg.copy_check_on():
            return None
        from .copycheck import copy_limit

        return copy_limit(self.cfg.copy_max, checker.baseline)

    @torch.no_grad()
    def _selftrain_render(self, model, step: int, rnd: int, jobs: list) -> list[str]:
        """Render a round's pictures with the weight average, as the probes do (on every sampling GPU)."""
        self._free_for_rendering()
        with self._averaged(replicas=True), lora_mod.frozen_adapters(), self._sampling_slots(model, step) as slots:
            for sl in slots:
                sl.model.eval()
                lora_mod.set_extras_enabled(sl.model, True)
            try:
                neg_e, neg_m = self.arch.pad_embeds([load_text_embeds(self.cfg, self.negative_prompt())])
                title = f"self-training round {rnd} on {len(slots)} GPUs" if len(slots) > 1 else ""
                return self._render_round(slots, jobs, neg_e, neg_m, _Board(self.status, title))
            finally:
                for sl in slots:
                    lora_mod.set_extras_enabled(sl.model, False)

    def _selftrain_candidate(self, path: str, prompt: str, scale, side: int = 512):
        """A rendered picture's faces, likeness on the probes' scale (None when the face, or a full-body prompt's
        body, is missing), sharpness, thumbnail and copy score (copy_check)."""
        from . import selftrain as ST
        from .facesim import cut_off, image_size
        from .likeness import combine
        from .reward import is_full_body

        cfg = self.cfg
        c = ST.Candidate(str(path), prompt, None)
        face = body = None
        missing = False
        box = None
        if self.face_scorer is not None:
            fs = self.face_scorer.faces(path)
            c.faces = len(fs)
            if fs:
                c.det = float(fs[0].det_score)
                c.cut_off = cut_off(fs[0].bbox, image_size(path))
                box = fs[0].bbox
            if cfg.face_weight > 0:
                face = self.face_scorer.score(path).percent
                missing = face is None
        if self.body_scorer is not None and cfg.body_weight > 0:
            body, _ = self.body_scorer.score(path)
            missing = missing or (body is None and is_full_body(prompt))
        if scale is not None and getattr(scale, "normalize", False):
            face, body = scale.norm("face", face), scale.norm("body", body)
        if not missing:
            c.likeness = combine(face, body, cfg.face_weight, cfg.body_weight)
        c.face_sharp = ST.big_face(box, image_size(path), side)  # large enough at the training resolution
        c.sharp, c.thumb = ST.sharpness(path, box if c.face_sharp else None, side), ST.thumb(path)
        if self._selftrain_copy_limit() is not None:
            c.copy, c.copy_of = self.copy_checker.score(path)
        return c

    def _selftrain_add(self, kept: list[tuple[str, str]], step: int) -> int:
        """Train on these (image, prompt) pairs in place of the last round's: one item per training resolution the
        picture is not upscaled to, at selftrain_weight; latents and region maps in <job>/selftrain/cache. Returns
        the items added."""
        from . import regions as R
        from .dataset import Item, bucket_size
        from .facesim import image_size

        cfg = self.cfg
        cache = self.job_dir / "selftrain" / "cache"
        cache.mkdir(parents=True, exist_ok=True)
        tag = self.arch.latent_tag
        new = []
        for image, prompt in kept:
            p = Path(image)
            w, h = image_size(p)
            for side in cfg.training_sides():
                if side * side > w * h * 1.05:  # an upscaled picture teaches blur
                    continue
                bw, bh = bucket_size(w, h, cfg, side)
                stem = f"{p.stem}_{bw}x{bh}"
                new.append(Item(image=p, caption=prompt, width=bw, height=bh,
                                latent_path=cache / f"{stem}{tag}.safetensors",
                                latent_flip_path=(cache / f"{stem}_flip{tag}.safetensors") if cfg.flip_augment
                                else None,
                                text_path=text_cache_path(cfg, prompt),
                                region_path=(cache / f"{stem}_regions.safetensors") if cfg.wants_regions() else None,
                                category="selftrain", side=side, repeats=1, dataset="selftrain",
                                loss_scale=cfg.selftrain_weight))
        if new:
            try:
                cache_latents(new, cfg, self._home_vae())
            finally:
                self._park_vae()
            if cfg.wants_regions():  # face cells and landmarks; the person-mask models were unloaded after the photos
                analysis = R.analyze_faces(new, self._ensure_face_detector(), cache / "faces.json")
                self.face_analysis.update(analysis)
                R.cache_region_maps(new, cfg, analysis)
            if self.trigger_pos:  # attention localization reads the trigger in their captions too
                self.trigger_pos.update(self._trigger_positions(new, [cfg]))
        self.all_items = [it for it in self.all_items if it.category != "selftrain"] + new
        self._enter_stage(self.res_stage, step, initial=True)
        if self._race_running():  # the configuration keeps its own items and image queue when it is parked
            self.race.active.state.update(items=self.all_items, batcher=self.batcher, batch_iter=self.batch_iter)
        return len(new)

    # ------------------------------------------------------------ seed soup (see soup.py)
    def _soup_stage(self, model) -> None:
        """A greedy soup of soup_files, scored by the likeness probes: each file alone (the best also at every soup
        strength, so a soup is never credited for a strength change), then best-first, each next file joins the soup
        when the soup's score at its best strength does not drop. Saves <job>_soup.safetensors and soup.json."""
        import json

        from . import soup as S
        from .config import soup_strength_list
        from .likeness import combine, probe_seed_count

        cfg = self.cfg
        if self.controller is None:
            raise ValueError("The soup is picked by the likeness probes, which could not be set up (references, "
                             "probe prompts and their scoring packages)")
        t0 = time.time()
        out_dir = self.job_dir / "soup"
        out_dir.mkdir(exist_ok=True)
        strengths = soup_strength_list(cfg.soup_strengths)
        names = [Path(p).name for p in cfg.soup_files]
        self.status.update(stage="seed soup: reading the LoRAs")
        ings = [S.read_ingredient(p) for p in cfg.soup_files]
        fmt = self.arch.lora_format
        log.info("Seed soup: %d LoRAs (%s), rank %s, strengths %s, %d probe seed(s) each", len(ings),
                 ", ".join(names), cfg.soup_rank or "exact", ", ".join(f"{s:g}" for s in strengths), cfg.probe_seeds)
        self._free_for_rendering()
        with lora_mod.frozen_adapters(), self._sampling_slots(model, 0) as slots:  # the base model: the scale's zero
            face, body, _, _ = self._probe_with(slots, 0, tag="soup_base")
        if self.controller.normalize:
            self.controller._set_floors(face, body)
            log.info("Likeness scale: %s (%d seeds)", self.controller.describe_scale(), probe_seed_count(cfg, 0))
        trials: list[dict] = []

        def score(members: list[int], strength: float, label: str) -> float | None:
            stubs, rep = S.cook([ings[i] for i in members], rank=cfg.soup_rank, strength=strength,
                                device=self.devices[0])
            path = S.save(stubs, out_dir / "candidate.safetensors", [cfg.soup_files[i] for i in members], rep, fmt)
            self.status.update(stage=f"seed soup: probing {label}")
            f, b, _ = self._soup_probe(model, path, f"soup{len(trials) + 1:02d}")
            comb = combine(self.controller.norm("face", f), self.controller.norm("body", b), cfg.face_weight,
                           cfg.body_weight)
            if rep["energy"] < 0.5 and not trials:
                log.warning("Seed soup: rank %d keeps only %.0f%% of the change (LoKr and high-rank files change "
                            "nearly every direction): a higher soup_rank, or 0 for LoRA files, keeps more",
                            cfg.soup_rank, rep["energy"] * 100)
            trials.append({"files": [names[i] for i in members], "strength": strength, "score": comb, "face": f,
                           "body": b, "agreement": rep["agreement"], "norm_ratio": rep["norm_ratio"],
                           "energy": rep["energy"]})
            fmt_s = "n/a" if comb is None else f"{comb:.1f}%"
            log.info("Seed soup: %s at strength %g -> %s%s", label, strength, fmt_s,
                     f" ({S.describe(rep)})" if len(members) > 1 else "")
            return comb

        def val(x):
            return float("-inf") if x is None else x

        alone = [score([i], 1.0, names[i]) for i in range(len(ings))]
        order = sorted(range(len(ings)), key=lambda i: -val(alone[i]))
        soup, best, best_s = [order[0]], alone[order[0]], 1.0
        for s in strengths:  # the best file alone at the other strengths: what a stronger soup must beat
            if s != 1.0:
                v = score([order[0]], s, names[order[0]])
                if val(v) > val(best):
                    best, best_s = v, s
        for j in order[1:]:
            cand = soup + [j]
            got = [(score(cand, s, " + ".join(names[i] for i in cand)), s) for s in strengths]
            v, s = max(got, key=lambda g: val(g[0]))
            if val(v) >= val(best):
                soup, best, best_s = cand, v, s
                log.info("Seed soup: %s joins (%.1f%% at strength %g)", names[j], val(v), s)
            else:
                log.info("Seed soup: %s left out (the soup with it scores %s, without %s)", names[j],
                         "n/a" if v is None else f"{v:.1f}%", "n/a" if best is None else f"{best:.1f}%")
        uniform = score(list(range(len(ings))), 1.0, "every file (plain average)") if len(soup) < len(ings) else None
        stubs, rep = S.cook([ings[i] for i in soup], rank=cfg.soup_rank, strength=best_s, device=self.devices[0])
        final = S.save(stubs, self.job_dir / f"{cfg.job_name}_soup.safetensors", [cfg.soup_files[i] for i in soup],
                       rep, fmt)
        (out_dir / "candidate.safetensors").unlink(missing_ok=True)
        (out_dir / "soup.json").write_text(json.dumps({
            "files": names, "alone": dict(zip(names, alone)), "soup": [names[i] for i in soup], "strength": best_s,
            "score": best, "uniform": uniform, "report": rep, "trials": trials}, indent=1), encoding="utf-8")
        what = ("only " + names[soup[0]] + ": no soup beat it" if len(soup) == 1
                else " + ".join(names[i] for i in soup))
        log.info("Seed soup done: %s at strength %g scores %s (the best single file %s%s) -> %s (%.0fs)", what, best_s,
                 "n/a" if best is None else f"{best:.1f}%", f"{val(alone[order[0]]):.1f}%",
                 "" if uniform is None else f", every file averaged {val(uniform):.1f}%", final.name, time.time() - t0)
        self.status.data["saved"].append(str(final))
        self.status.update(stage="finished", done=True, soup={"files": [names[i] for i in soup], "strength": best_s,
                                                              "score": best, "trials": trials})

    # ------------------------------------------------------------ noise check (see likeness_stats.py)
    def _noise_stage(self, model) -> None:
        """How far the probes move with their seeds: the base model (the normalized scale's zero), then every
        noise_files LoRA at noise_seeds seeds, all on the same seeds. Writes noise/noise.json
        (likeness_stats.noise_report) and logs the seeds a probe needs to tell gaps apart and a likeness_noise_band."""
        import json

        from . import likeness_stats as LS
        from .likeness import combine_seeds, mean_of, probe_seed_count

        cfg = self.cfg
        if self.controller is None:
            raise ValueError("The noise check measures the likeness probes, which could not be set up (references, "
                             "probe prompts and their scoring packages)")
        t0 = time.time()
        out_dir = self.job_dir / "noise"
        out_dir.mkdir(exist_ok=True)
        log.info("Noise check: %d LoRA(s), %d probe seeds each (seeds %d-%d)", len(cfg.noise_files), cfg.noise_seeds,
                 cfg.probe_seed, cfg.probe_seed + cfg.noise_seeds - 1)
        self._free_for_rendering()
        self.status.update(stage="noise check: probing the base model")
        with lora_mod.frozen_adapters(), self._sampling_slots(model, 0) as slots:  # the base model: the scale's zero
            face, body, _, seeds0 = self._probe_with(slots, 0, tag="noise_base")
        self.controller.has_gold = "gold" in seeds0
        if self.controller.normalize:
            self.controller._set_floors(face, body, mean_of(seeds0.get("gold")))
            log.info("Likeness scale: %s (%d seeds)", self.controller.describe_scale(), probe_seed_count(cfg, 0))
        files, raw = {}, {}
        for i, p in enumerate(cfg.noise_files):
            name = Path(p).name
            if name in files:
                name = f"{i + 1}:{name}"
            self.status.update(stage=f"noise check: probing {name} ({i + 1}/{len(cfg.noise_files)})")
            _, _, seeds = self._soup_probe(model, p, f"noise{i + 1:02d}", n_seeds=cfg.noise_seeds)
            per = self.controller.seed_scores(seeds)
            files[name] = {"face": per["face_norm_seeds"], "body": per["body_norm_seeds"],
                           "combined": combine_seeds(per["face_norm_seeds"], per["body_norm_seeds"], cfg.face_weight,
                                                     cfg.body_weight)}
            raw[name] = {"face": per["face_seeds"], "body": per["body_seeds"]}
            if "gold_norm_seeds" in per:  # the held-out FaceNet face, alone and combined with the body
                files[name]["gold"] = per["gold_norm_seeds"]
                files[name]["combined_gold"] = combine_seeds(per["gold_norm_seeds"], per["body_norm_seeds"],
                                                             cfg.face_weight, cfg.body_weight)
                raw[name]["gold"] = per["gold_seeds"]
            m, sd = LS.mean_sd(files[name]["combined"])
            log.info("Noise check: %s scores %s (seeds spread %s)", name, "n/a" if m is None else f"{m:.1f}%",
                     "n/a" if sd is None else f"{sd:.1f} points")
        rep = LS.noise_report(files, cfg.probe_seeds, cfg.probe_ema)
        rep.update(seeds=list(range(cfg.probe_seed, cfg.probe_seed + cfg.noise_seeds)), per_seed=files, raw=raw,
                   scale=self.controller.scale_record() if self.controller.normalize else {})
        (out_dir / "noise.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
        for line in LS.describe(rep):
            log.info("Noise check: %s", line)
        log.info("Noise check done in %.0fs -> %s", time.time() - t0, out_dir / "noise.json")
        comb = rep["kinds"]["combined"]
        noise = {"sigma": comb.get("sigma"), "sd_diff": comb.get("sd_diff"), "rho": comb.get("rho"),
                 "seeds_needed": comb.get("seeds_needed"), "suggested_noise_band": rep.get("suggested_noise_band")}
        for kind in ("face", "gold"):
            if kind in rep["kinds"]:
                noise[f"{kind}_sd_diff"] = rep["kinds"][kind].get("sd_diff")
        self.status.update(stage="finished", done=True, noise=noise)

    @torch.no_grad()
    def _soup_probe(self, model, path, tag: str, n_seeds: int | None = None):
        """Probe the model with a LoRA file attached as an extra (the training adapter is untouched and still
        zero), then take it off again. Returns the raw (face, body, per-seed scores); n_seeds overrides probe_seeds."""
        from .lora import LoRALinear

        models = [m for m, _, _ in self.replicas] or [model]
        sizes = {id(mod): (len(mod.extras), len(mod.extra_biases)) for m in models for mod in m.modules()
                 if isinstance(mod, LoRALinear)}
        try:
            for m in models:
                lora_mod.attach_extra_lora(m, str(path), 1.0, rename=getattr(self.arch, "lora_module_name", None),
                                           convert=getattr(self.arch, "extra_lora_converter", None))
            self._free_for_rendering()
            with lora_mod.frozen_adapters(), self._sampling_slots(model, 0) as slots:
                face, body, _, seeds = self._probe_with(slots, 0, tag=tag, n_seeds=n_seeds)
        finally:
            for m in models:
                for mod in m.modules():
                    if isinstance(mod, LoRALinear):
                        n_e, n_b = sizes.get(id(mod), (0, 0))
                        del mod.extras[n_e:]
                        del mod.extra_biases[n_b:]
        return face, body, seeds

    def _likeness_scale(self):
        """The scale the samples' scores are normalized on: the probe controller's (its floors from the step-0
        probe), else one with the references' ceilings and default floors; None with likeness_normalize off."""
        if not self.cfg.likeness_normalize:
            return None
        if self.controller is not None:
            return self.controller
        if getattr(self, "_fallback_scale", None) is None:
            from .likeness import LikenessController

            self._fallback_scale = LikenessController(self.cfg, **self._controller_ceilings())
        return self._fallback_scale

    def _controller_ceilings(self) -> dict:
        """The references' own self-similarity per scorer (raw percent), the 100 of the normalized scale."""
        out = {}
        for key, scorer in (("face_ceiling", self.face_scorer), ("body_ceiling", self.body_scorer)):
            sim = getattr(scorer, "ref_self_similarity", None) if scorer is not None else None
            if sim is not None:
                out[key] = float(sim) * 100.0
        sim = getattr(self.face_scorer, "gold_self_similarity", None) if self.face_scorer is not None else None
        if sim is not None:
            out["gold_ceiling"] = float(sim) * 100.0
        return out

    @torch.no_grad()
    def _probe_with(self, slots, step, tag: str = "", n_seeds: int | None = None):
        """Render and score one probe: (face, body, images, seeds), face / body the mean raw score over the seeds that
        were scored, seeds each seed's own ({"face": [...], "body": [...]}, seed order, None = no face / body found;
        "gold" too with the held-out recognizer). n_seeds overrides probe_seeds (the noise check)."""
        for sl in slots:
            sl.model.eval()
            lora_mod.set_extras_enabled(sl.model, True)
        neg_e, neg_m = self.arch.pad_embeds([load_text_embeds(self.cfg, self.negative_prompt())])
        jobs = self._probe_jobs(step, tag, n_seeds, len(slots))
        title = f"probes at step {step} on {len(slots)} GPUs" if len(slots) > 1 and len(jobs) > 1 else ""
        board = _Board(self.status, title)
        try:
            images = self._render_round(slots, jobs, neg_e, neg_m, board)
        finally:
            for sl in slots:
                lora_mod.set_extras_enabled(sl.model, False)
        face, body, seeds = self._score_probe(jobs, images, step)
        return face, body, images, seeds

    def _probe_jobs(self, step: int, tag: str = "", n_seeds: int | None = None, n_slots: int = 1) -> list:
        """The probe's RenderJobs: each enabled kind (face / body) with a scorer, one per seed."""
        from .likeness import probe_body_size, probe_seed_count

        cfg = self.cfg
        out_dir = self.job_dir / "probes"
        out_dir.mkdir(exist_ok=True)
        jobs = []
        for kind, prompt in (("face", cfg.probe_face_prompt), ("body", cfg.probe_body_prompt)):
            if not prompt:
                continue
            if kind == "face" and self.face_scorer is None:
                continue
            if kind == "body" and self.body_scorer is None:
                continue
            w, h = (cfg.probe_size, cfg.probe_size) if kind == "face" else probe_body_size(cfg.probe_size)
            n = n_seeds or probe_seed_count(cfg, step)  # more at step 0: the normalized scale's zero
            for k in range(n):
                jobs.append(RenderJob(prompt, w, h, cfg.probe_steps, cfg.probe_cfg, cfg.probe_seed + k,
                                      out_dir / f"{self._image_prefix()}{tag or f'step{step:06d}'}_{kind}_{k}.png",
                                      f"probe {kind} seed {k + 1}/{n} at step {step}" if n_slots == 1
                                      else f"probe {kind} seed {k + 1}/{n}", kind))
        return jobs

    def _score_probe(self, jobs: list, images: list[str], step: int):
        """Score a probe's rendered images: (face, body, seeds) as _probe_with returns them."""
        seeds: dict[str, list] = {"face": [], "body": []}
        gold = self.face_scorer is not None and getattr(self.face_scorer, "gold", None) is not None
        if gold:
            seeds["gold"] = []
        self.status.update(stage=f"scoring probes (step {step})")
        for job, p in zip(jobs, images):  # scoring stays in this thread: the scorers are not shared across threads
            if job.kind == "face":
                sc = self.face_scorer.score(p, step, 0, gold=True) if gold else self.face_scorer.score(p, step, 0)
                seeds["face"].append(sc.percent)
                if gold:
                    seeds["gold"].append(sc.gold_percent)
            else:
                seeds["body"].append(self.body_scorer.score(p)[0])
        face_scores = [v for v in seeds["face"] if v is not None]
        body_scores = [v for v in seeds["body"] if v is not None]
        face = sum(face_scores) / len(face_scores) if face_scores else None
        body = sum(body_scores) / len(body_scores) if body_scores else None
        return face, body, seeds

    def run_probe(self, model, step, optimizer, scheduler, injected, ema=None, sync_replicas: bool = True) -> bool:
        """Render + score probes, update the controller, apply its actions. Returns True to stop training.

        With a weight average the probes score it, so a best checkpoint is exactly the weights that were scored; `ema`
        is that average when it is not the trainer's own (the parallel race's configurations). A Keel optimizer's
        probe-gated lookahead then hears the decision, with the raw weights back in place; the data-parallel copies
        follow a pull when sync_replicas. With probe_async it renders here too (training waits for it anyway, and
        the training GPU is idle) while the probe GPU finishes the probe it has, which is acted on first."""
        if self.probe_worker is not None:
            t0 = time.time()
            self._free_for_rendering()
            with self._averaged(replicas=True), lora_mod.frozen_adapters(), \
                    self._sampling_slots(model, step) as slots:
                scored = lora_mod.lora_state(injected) if hasattr(optimizer, "on_probe") else None  # before a pull
                face, body, images, seeds = self._probe_with(slots, step)
            took = f"{time.time() - t0:.1f}s"
            stop = self.probe_worker is not None and self._probe_async_take(step, optimizer, scheduler, wait=True)
            with self._averaged(replicas=True):
                stop = self._probe_finish(step, face, body, images, seeds, took, optimizer, scheduler,
                                          injected) or stop
            self._keel_probe(optimizer, ema, sync_replicas, scored)
            return stop
        self._free_for_rendering()
        with self._averaged(replicas=True):
            stop = self._run_probe(model, step, optimizer, scheduler, injected)
        self._keel_probe(optimizer, ema, sync_replicas)
        return stop

    def _keel_probe(self, optimizer, ema=None, sync_replicas: bool = True, scored: dict | None = None) -> None:
        """Keel's probe-gated lookahead (keel_lookahead): the controller's smoothed decision score and its noise band
        to the optimizer, which anchors the weights that scored or pulls the weights back toward the anchor. The
        weights that scored are the average's shadow (`ema`, the trainer's own when None) or the raw weights, or
        `scored`: a lora_state of them (an async probe's, sent to the probe GPU up to probe_every steps earlier)."""
        on_probe = getattr(optimizer, "on_probe", None)
        if on_probe is None or self.controller is None:
            return
        weights = None
        if scored is not None:
            weights = {p: d["params"][part] for name, d in scored.items()
                       for part, p in self.injected[name].adapter_params().items()}
        ema = self.ema if ema is None else ema
        if on_probe(self.controller.combined_ema, self.cfg.likeness_noise_band, ema, weights) == "pull":
            if sync_replicas and self.dp is not None:
                self.dp.sync_weights()

    def _run_probe(self, model, step, optimizer, scheduler, injected) -> bool:
        t0 = time.time()
        with lora_mod.frozen_adapters(), self._sampling_slots(model, step) as slots:
            face, body, images, seeds = self._probe_with(slots, step)
        return self._probe_finish(step, face, body, images, seeds, f"{time.time() - t0:.1f}s", optimizer, scheduler,
                                  injected)

    def _probe_finish(self, step, face, body, images, seeds, took: str, optimizer, scheduler, injected,
                      now: int | None = None) -> bool:
        """Act on a scored probe: the controller's update, its record and log line, best saves (from `injected`, the
        weights that were scored) and actions. now: the training step it is acted on at (an async probe lands
        later than the step it scored). Returns True to stop training."""
        from .likeness import append_record, apply_lr_ratio, curriculum_weights, timestep_logit_shift

        cfg = self.cfg
        now = step if now is None else now
        self.last_probe = (face, body, images, seeds)
        rec, actions = self.controller.update(step, face, body, images, seeds)
        copy_note = self._copy_probe(rec, images)
        append_record(self.job_dir, rec)
        fmt = lambda v: "n/a" if v is None else f"{v:.1f}%"  # noqa: E731
        who = f"[{self.race.active.name}] " if self._race_running() else ""
        if self.controller.normalize and self.controller.n_probes == 1:
            log.info("Likeness scale: %s", self.controller.describe_scale())
        if self.controller.normalize:
            log.info("%sProbe step %d: face %s body %s (normalized %s / %s, combined %s) | ema face %s body %s "
                     "(normalized %s / %s, combined %s) | best %s @%s | lr x%.2f | face bias %+.2f (%s)", who, step,
                     fmt(face), fmt(body), fmt(rec.face_norm), fmt(rec.body_norm), fmt(rec.combined),
                     fmt(rec.face_ema), fmt(rec.body_ema), fmt(rec.face_norm_ema), fmt(rec.body_norm_ema),
                     fmt(rec.combined_ema), fmt(rec.best), rec.best_step, rec.lr_factor, rec.face_bias, took)
        else:
            log.info("%sProbe step %d: face %s body %s combined %s | ema face %s body %s combined %s | best %s @%s "
                     "| lr x%.2f | face bias %+.2f (%s)", who, step, fmt(face), fmt(body), fmt(rec.combined),
                     fmt(rec.face_ema), fmt(rec.body_ema), fmt(rec.combined_ema), fmt(rec.best), rec.best_step,
                     rec.lr_factor, rec.face_bias, took)
        if self.controller.has_gold:
            arc, gold = rec.face_norm_ema if self.controller.normalize else rec.face_ema, rec.gold_norm_ema
            log.info("%sGold face (FaceNet) step %d: %s (normalized %s) | ema %s vs ArcFace %s%s%s", who, step,
                     fmt(rec.gold), fmt(rec.gold_norm), fmt(gold), fmt(arc),
                     f" (ArcFace {arc - gold:+.1f})" if arc is not None and gold is not None else "",
                     " | the decisions follow the gold face" if rec.gold_select else "")
            self._check_gold_gap(step)
        if copy_note:
            log.info("%sCopy check step %d: %s", who, step, copy_note)
        if "step_up" in actions and step < cfg.steps:  # not at the end, nor in the reward stage
            log.info("Resolution step-up: stage %d stalled for %d probe(s) (stage best %s)", rec.stage + 1,
                     rec.stage_since_improve, fmt(rec.stage_best))
            self._enter_stage(actions["step_up"], now)
        if actions.get("save_best") and not self._race_keeps_best("", rec.best):
            actions["save_best"] = False  # a relay racer's best of its own, below the one already saved this round
        for kind in ("face", "body"):
            if actions.get(f"save_best_{kind}") and not self._race_keeps_best(kind,
                                                                              self.controller.part_best[kind][0]):
                actions[f"save_best_{kind}"] = False
        if actions.get("save_best") and getattr(self, "ft", None) is not None:
            path = self._finetune_save(step, True, label="_best")
        elif actions.get("save_best"):
            path = self.job_dir / f"{cfg.job_name}{self._best_tag()}_best.safetensors"
            lora_mod.save_lora(injected, path, metadata={"ss_steps": str(step), "likeness_best": f"{rec.best:.2f}",
                                                         "modelspec.architecture": self.arch.modelspec},
                               dtype=cfg.lora_save_dtype, fmt=self.arch.lora_format)
        if actions.get("save_best"):
            log.info("New best likeness %.1f%% at step %d -> %s", rec.best, step, path)
            if str(path) not in self.status.data["saved"]:
                self.status.data["saved"].append(str(path))
        for kind in ("face", "body"):
            if actions.get(f"save_best_{kind}"):
                score = self.controller.part_best[kind][0]
                if getattr(self, "ft", None) is not None:
                    path = self._finetune_save(step, True, label=f"_best_{kind}")
                else:
                    path = self.job_dir / f"{cfg.job_name}{self._best_tag()}_best_{kind}.safetensors"
                    lora_mod.save_lora(injected, path, metadata={"ss_steps": str(step),
                                                                 f"likeness_best_{kind}": f"{score:.2f}",
                                                                 "modelspec.architecture": self.arch.modelspec},
                                       dtype=cfg.lora_save_dtype, fmt=self.arch.lora_format)
                log.info("New best %s likeness %.1f%% at step %d -> %s", kind, score, step, path)
                if str(path) not in self.status.data["saved"]:
                    self.status.data["saved"].append(str(path))
        if actions.get("id_loss_veto") and self.id_loss is not None:
            self.id_loss = None
            if self._race_running():
                self.race.active.state["id_vetoed"] = True
            log.warning("Identity loss vetoed at step %d: the smoothed %sface score %s fell more than %.1f points "
                        "below its best %s for two probes in a row; the identity loss is off for the rest of the run",
                        step, "gold " if rec.gold_select else "",
                        fmt(rec.gold_norm_ema if rec.gold_select else rec.face_norm_ema
                            if self.controller.normalize else rec.face_ema),
                        2 * cfg.likeness_noise_band, fmt(self.controller.face_best))
            self.status.update(id_loss_vetoed=step)
        if "lr_ratio" in actions:
            apply_lr_ratio(optimizer, scheduler, actions["lr_ratio"])
            log.info("Likeness lr control: factor now x%.2f (lr %.2e)", rec.lr_factor, current_lr(optimizer, scheduler))
        if "face_bias" in actions and self.batcher is not None:
            bias = actions["face_bias"]
            if self.item_face_frac:
                self.curriculum_w = curriculum_weights(self.batcher.items, self.item_face_frac, bias)
                self._refresh_batcher_weights()
            self.ts_logit_shift = timestep_logit_shift(bias)
            log.info("Curriculum: face bias %+.2f -> %s images favoured, timestep logit shift %+.2f", bias,
                     "face-heavy" if bias > 0 else "body/other", self.ts_logit_shift)
        probes = self.status.data.get("probes", [])
        from dataclasses import asdict

        probes.append(asdict(rec))
        self.status.update(probes=probes[-500:], likeness=self.controller.state(), stage="training")
        self._save_history()
        return bool(actions.get("stop"))

    # ------------------------------------------------------------ probes on a second GPU (probe_async; probeworker.py)
    def _probe_async_device(self) -> str:
        """The GPU that renders the probes while training goes on, or "" (the reason logged) when probe_async cannot
        run here."""
        cfg = self.cfg
        if not cfg.probe_async or self.controller is None or cfg.probe_every <= 0:
            return ""
        from .probeworker import pick_device

        why = ("a race switches configurations at each probe" if self.race_rows is not None else
               "a fine-tune has no LoRA to copy" if self.ft is not None else
               f"multi_gpu {cfg.multi_gpu} uses both GPUs already" if cfg.multi_gpu != "off" else "")
        dev = ""
        if not why:
            dev, why = pick_device(cfg.probe_device, cfg.devices, self.device,
                                   torch.cuda.device_count() if torch.cuda.is_available() else 0)
        if why:
            log.info("Probes render on the training GPU between steps (probe_async not used: %s)", why)
        return dev

    def _setup_probe_worker(self) -> None:
        """probe_async: start the probe process, which loads a copy of the sampling model (sample_model_path, else the
        training base) with the same LoRA slots and the sampling LoRAs on the probe GPU."""
        from .probeworker import ProbeProcess, ProbeProcessError

        dev = self._probe_async_device()
        if not dev:
            return
        self.status.update(stage=f"loading the probe model on {dev}")
        t0 = time.time()
        try:
            w = ProbeProcess(self.cfg.to_dict(), self.model_dir, dev, self.job_dir / "probe_worker.log")
        except ProbeProcessError as e:
            log.warning("probe_async: the probe process on %s failed to start, so probes render on the training GPU "
                        "between steps: %s", dev, str(e).strip().splitlines()[-1])
            return
        self.probe_worker = w
        log.info("Probes render on %s in a separate process while training goes on (probe_async; started in %.0fs, "
                 "its model loads while training starts): each probe is acted on once it is scored, up to "
                 "probe_every steps later; the final one renders here", dev, time.time() - t0)

    @torch.no_grad()
    def _probe_async_start(self, step: int, injected) -> None:
        """Send the LoRA (the weight average with lora_ema) to the probe process and queue its probe there."""
        with self._averaged():
            state = lora_mod.lora_state(injected)
        self.probe_worker.submit(step, self._probe_jobs(step), state)
        self.status.update(stage=f"training; probe of step {step} rendering on {self.probe_worker.device}")

    def _probe_async_take(self, now: int, optimizer, scheduler, wait: bool = False) -> bool:
        """Score and act on the probe the probe process has rendered, if any (wait: until it is in). True to stop.
        Best saves write the weights that were scored (swapped in for the moment), not training's newer ones."""
        from .probeworker import ProbeProcessError

        w = self.probe_worker
        if wait and w.pending is not None and not w.ready():
            self.status.update(stage=f"waiting for the probe of step {w.pending.step} on {w.device}")
        lost = w.pending.step if w.pending is not None else None
        try:
            got = w.take(wait)
        except ProbeProcessError as e:
            log.warning("probe_async: the probe process on %s failed (the probe of step %s is lost), so probes render "
                        "on the training GPU from here: %s", w.device, lost, str(e).strip().splitlines()[-1])
            self._close_probe_worker()
            return False
        if got is None:
            return False
        p, images = got
        face, body, seeds = self._score_probe(p.jobs, images, p.step)
        took = f"{time.time() - p.queued:.1f}s on {w.device}" + (f", in at step {now}" if now != p.step else "")
        live = lora_mod.lora_state(self.injected)
        lora_mod.mirror_lora(lora_mod.lora_snapshots(p.state), self.injected)
        try:
            stop = self._probe_finish(p.step, face, body, images, seeds, took, optimizer, scheduler, self.injected,
                                      now=now)
        finally:
            lora_mod.mirror_lora(lora_mod.lora_snapshots(live), self.injected)
        self._keel_probe(optimizer, scored=p.state)  # the lookahead anchors the weights that scored, pulls the live ones
        return stop

    def _close_probe_worker(self) -> None:
        """Stop the probe process (also after an error), freeing the probe GPU."""
        w, self.probe_worker = self.probe_worker, None
        if w is None:
            return
        if w.count:
            log.info("probe_async: %d probes rendered on %s in %.0fs; training waited %.0fs for them", w.count,
                     w.device, w.rendered, w.waited)
        w.close()

    def _copy_probe(self, rec, images) -> str:
        """The probe's highest SSCD match among the training photos (copy_check), into its record; the log line
        ("" without the copy check)."""
        checker = getattr(self, "copy_checker", None)
        if checker is None or not images:
            return ""
        from .copycheck import copy_limit

        score, photo, render = checker.closest(images)
        if score is None:
            return ""
        rec.copy, rec.copy_of, rec.copy_baseline = round(score, 4), photo, checker.baseline
        limit, base = copy_limit(self.cfg.copy_max, checker.baseline), checker.baseline
        return (f"closest training photo {score:.2f} ({Path(render).name} to {Path(photo).name}) | the photos among "
                f"themselves {'n/a' if base is None else f'{base:.2f}'} | "
                + (f"above the copy limit {limit:.2f}: it may copy that photo" if score > limit
                   else f"copy limit {limit:.2f}"))

    def _make_copy_checker(self, items):
        """The SSCD copy check (copycheck.py) when the run, or any race configuration, wants it and something renders
        pictures to check (probes or self-training): the training photos described once, on the face scorer's
        device."""
        cfg = self.cfg
        cfgs = [cfg] + [c for _, c, _ in (getattr(self, "race_rows", None) or [])]
        if not any(c.copy_check_on() for c in cfgs) or cfg.noise_files or cfg.soup_files:
            return None
        if self.controller is None and not any(c.selftrain for c in cfgs):
            return None
        if self.face_scorer is None and self.body_scorer is None:
            return None
        photos = [str(it.image) for it in items if it.category == "image"]
        if not photos:
            return None
        from . import copycheck, facesim

        index = facesim.cuda_index(cfg.face_provider)
        dev = f"cuda:{index}" if index is not None and torch.cuda.is_available() else "cpu"
        self.status.update(stage="loading the copy check (SSCD)")
        t0 = time.time()
        try:
            checker = copycheck.CopyChecker(cfg.models_dir, dev, progress=lambda t, f: self.status.update(stage=t))
            checker.set_photos(photos)
        except Exception as e:  # noqa: BLE001
            log.warning("Could not load the copy check (%s); off", e)
            return None
        base = checker.baseline
        gate = any(c.selftrain and c.copy_check_on() for c in cfgs)
        log.info("Copy check on: SSCD described the %d training photos in %.0fs (their typical best match among "
                 "themselves %s); %sthe probes log their closest photo", len(checker.photos), time.time() - t0,
                 "n/a" if base is None else f"{base:.2f}",
                 f"self-training leaves out pictures above {copycheck.copy_limit(cfg.copy_max, base):.2f}; "
                 if gate else "")
        self.status.update(copy_baseline=base)
        return checker

    def _gold_gap(self) -> float | None:
        """ArcFace's smoothed normalized face minus the gold face's (None without both)."""
        st = self.controller.state() if self.controller is not None else {}
        arc, gold = st.get("face_norm_ema"), st.get("gold_norm_ema")
        return None if arc is None or gold is None else arc - gold

    def _check_gold_gap(self, step: int) -> None:
        """In the reward stage (it trains against ArcFace): warn once when ArcFace's smoothed face has run ahead of the
        gold face by more than max(10, 3 noise bands) points beyond the gap the stage started with, the sign that
        the reward games ArcFace rather than improving the likeness."""
        if getattr(self, "stage_tag", "") != "_reward" or getattr(self, "_gold_gap_warned", False):
            return
        gap, base = self._gold_gap(), getattr(self, "_gold_gap0", None)
        if gap is None or base is None:
            return
        limit = max(10.0, 3 * self.cfg.likeness_noise_band)
        if gap - base > limit:
            self._gold_gap_warned = True
            log.warning("Reward stage, step %d: ArcFace's face score has run %.1f points further ahead of the gold "
                        "(FaceNet) face than when the stage started (%+.1f -> %+.1f): the reward may be gaming "
                        "ArcFace rather than improving the likeness. %s", step, gap - base, base, gap,
                        "The best checkpoint follows the gold face" if self.controller.gold_select else
                        "Compare the probes by eye; face_gold_select makes the best checkpoint follow the gold face")
            self.status.update(gold_gap_warning=step)

    def _save_history(self, state: str = "running") -> None:
        """Keep this run's likeness scores in output/_history (never fails a run)."""
        try:
            history.save_run(self.cfg, self.run_id, self.run_started, self.status.data, state)
        except Exception as e:  # noqa: BLE001
            log.debug("Run history not saved: %s", e)

    @torch.no_grad()
    def _sample_with(self, slots, step):
        cfg = self.cfg
        for sl in slots:
            sl.model.eval()
            lora_mod.set_extras_enabled(sl.model, True)
        self.status.update(stage=f"sampling (step {step})")
        neg_e, neg_m = self.arch.pad_embeds([load_text_embeds(cfg, self.negative_prompt())])
        out_dir = self.job_dir / "samples"
        out_dir.mkdir(exist_ok=True)
        n_prompts = len(cfg.sample_prompts)
        jobs = [RenderJob(prompt, cfg.sample_width, cfg.sample_height, cfg.sample_steps, cfg.sample_cfg,
                          cfg.sample_seed + (i if cfg.sample_walk_seed else 0),
                          out_dir / f"{self._image_prefix()}step{step:06d}_{i}.png",
                          f"sampling {i + 1}/{n_prompts} at step {step}" if len(slots) == 1
                          else f"sample {i + 1}/{n_prompts}")
                for i, prompt in enumerate(cfg.sample_prompts)]
        title = f"sampling at step {step} on {len(slots)} GPUs" if len(slots) > 1 and n_prompts > 1 else ""
        board = _Board(self.status, title)
        t0 = time.time()
        try:
            paths = self._render_round(slots, jobs, neg_e, neg_m, board,
                                       on_done=lambda i, job, p: log.info("Sample saved %s", p))
        finally:
            for sl in slots:
                lora_mod.set_extras_enabled(sl.model, False)
        log.info("Sampling took %.1fs for %d image(s) on %s", time.time() - t0, len(paths),
                 ", ".join(sorted({sl.device for sl in slots})))
        self.status.data["samples"] = paths + self.status.data["samples"]
        if (self.face_scorer is not None or self.body_scorer is not None) and paths:
            self.status.update(stage=f"similarity scoring (step {step})")
            try:
                from .facesim import score_samples
                from dataclasses import asdict

                scores = score_samples(self.face_scorer, paths, step, self.job_dir, self.body_scorer,
                                       cfg.face_weight, cfg.body_weight, scale=self._likeness_scale())
                self.status.data["face_scores"] = self.status.data.get("face_scores", []) + [asdict(s) for s in scores]
                self._save_history()
            except Exception as e:  # noqa: BLE001
                log.warning("Similarity scoring failed: %s", e)
        self.status.update(stage="training")
        free_cuda()

    def _make_face_scorer(self):
        cfg = self.cfg
        if not cfg.face_refs:
            return None
        from . import facesim

        ok, msg = facesim.available()
        if not ok:
            log.warning("%s; face scoring disabled", msg)
            return None
        try:
            scorer = self.face_detector or facesim.FaceScorer(cfg.face_provider, cfg.face_det_size,
                                                               progress=lambda t, f: self.status.update(stage=t))
            self.face_detector = scorer
            scorer.set_references(cfg.face_refs)
        except Exception as e:  # noqa: BLE001
            log.warning("Could not initialise face scoring (%s); disabled", e)
            return None
        if not scorer.refs:
            log.warning("No faces found in any face reference image; face scoring disabled")
            return None
        self.status.update(face_ref_similarity=scorer.ref_self_similarity, face_refs=len(scorer.refs))
        self._setup_gold(scorer)
        return scorer

    def _setup_gold(self, scorer) -> None:
        """The held-out gold face recognizer (face_gold, likeness.py): FaceNet on the face scorer when the run, or any
        race configuration, trains against ArcFace (or face_gold is facenet, or the noise check runs), on the face
        scorer's device. Never when a configuration trains on FaceNet itself."""
        cfgs = [self.cfg] + [c for _, c, _ in (getattr(self, "race_rows", None) or [])]
        if not any(c.gold_backend() for c in cfgs):
            scorer.gold = None
            return
        if any(c.trains_on_facenet() for c in cfgs):
            log.warning("Gold face scorer off: a race configuration trains on FaceNet (id_loss_backend facenet), so "
                        "it is no held-out judge; the probes judge with ArcFace alone")
            scorer.gold = None
            return
        from . import facesim, idloss

        cfg = self.cfg
        self.status.update(stage="loading the gold face scorer (FaceNet)")
        index = facesim.cuda_index(cfg.face_provider)
        dev = f"cuda:{index}" if index is not None and torch.cuda.is_available() else "cpu"
        try:
            emb = idloss.facenet_embedder(cfg.models_dir, progress=lambda t, f: self.status.update(stage=t))
            scorer.set_gold(emb.to(dev))
        except Exception as e:  # noqa: BLE001
            log.warning("Could not load the gold face scorer (%s); the probes judge with ArcFace alone", e)
            scorer.gold = scorer.gold_refs = None
            return
        if scorer.gold_refs is None:
            log.warning("Gold face scorer: no reference face with landmarks; the probes judge with ArcFace alone")
            scorer.gold = None
            return
        why = ("the run trains against ArcFace" if any(c.trains_on_arcface() for c in cfgs)
               else "the noise check measures it" if cfg.noise_files else "face_gold facenet")
        log.info("Gold face scorer on (%s): FaceNet (VGGFace2), which nothing trains on, scores every probe face "
                 "too; %s", why, "the best checkpoint, early stop and race ranking follow it, the lr control and "
                 "curriculum keep ArcFace" if cfg.face_gold_select else "logged next to ArcFace (face_gold_select off)")
        self.status.update(face_gold="facenet", face_gold_similarity=scorer.gold_self_similarity)

    def _make_id_loss(self):
        """The direct identity loss (experimental), or None when it is off or cannot run."""
        cfg = self.cfg
        if cfg.id_loss_weight <= 0:
            return None
        if self.face_scorer is None:
            log.warning("Identity loss needs face_refs (and insightface) for the reference identity; disabled")
            return None
        from . import idloss

        dev = self.devices[0]
        vae = self.arch.grad_vae(self._vae_on(dev))
        self.status.update(stage=f"loading the identity loss ({cfg.id_loss_backend})")
        try:
            loss = idloss.build(cfg, self.face_scorer, lambda z: self.arch.decode_latents_grad(vae, z), dev,
                                progress=lambda t, f: self.status.update(stage=t))
        except Exception as e:  # noqa: BLE001
            log.warning("Could not initialise the identity loss (%s); disabled", e)
            return None
        self.id_ref = loss.pick_ref()
        log.info("Identity loss on: backend %s, weight %g (applied %g), sigma <= %g, every %d eligible sample(s), "
                 "faces >= %d cells, margin %d cells, %d reference(s)%s", cfg.id_loss_backend, cfg.id_loss_weight,
                 idloss.effective_weight(cfg.id_loss_weight), cfg.id_loss_sigma_max, cfg.id_loss_every,
                 cfg.id_loss_min_face_cells, cfg.id_loss_margin_cells, len(loss.refs),
                 ", random reference per step" if cfg.id_loss_random_ref else ", mean reference")
        if cfg.id_loss_backend == "arcface" and self.controller is not None:
            log.info("Identity loss uses the probe's own face network (arcface): the face probe score is now the "
                     "training target; check the samples by eye")
        return loss

    def _make_body_scorer(self):
        cfg = self.cfg
        if not cfg.body_refs:
            return None
        from . import bodysim

        ok, msg = bodysim.available()
        if not ok:
            log.warning("%s; body scoring disabled", msg)
            return None
        try:
            if cfg.body_score == "appearance":
                scorer = bodysim.AppearanceScorer(cfg.models_dir, cfg.body_device,
                                                  progress=lambda t, f: self.status.update(stage=t))
            elif cfg.body_score == "skeleton3d":
                from .bodyshape import Skeleton3DScorer

                scorer = Skeleton3DScorer(cfg.models_dir, progress=lambda t, f: self.status.update(stage=t))
            elif cfg.body_score == "smplx":
                from .bodyshape import SMPLXScorer

                scorer = SMPLXScorer(cfg.nlf_model, cfg.models_dir, cfg.body_device,
                                     progress=lambda t, f: self.status.update(stage=t))
            else:
                sil = None
                if cfg.body_backend == "mediapipe+sam3":
                    try:
                        sil = bodysim.SilhouetteScorer(cfg.sam3_model, cfg.body_device,
                                                       progress=lambda t, f: self.status.update(stage=t))
                    except Exception as e:  # noqa: BLE001
                        log.warning("SAM 3 unavailable (%s); body scoring uses MediaPipe only. For facebook/sam3 "
                                    "accept the license on huggingface.co and run 'hf auth login'.", e)
                scorer = bodysim.BodyScorer(cfg.models_dir, silhouette=sil,
                                            progress=lambda t, f: self.status.update(stage=t))
            if hasattr(scorer, "hold"):  # SMPL-X: keep NLF on the GPU for the whole reference batch
                with scorer.hold():
                    scorer.set_references(cfg.body_refs)
            else:
                scorer.set_references(cfg.body_refs)
        except Exception as e:  # noqa: BLE001
            log.warning("Could not initialise body scoring (%s); disabled", e)
            return None
        if not scorer.ready:
            log.warning("No usable %s in any body reference image; body scoring disabled",
                        {"appearance": "person", "smplx": "body down to the knees"}.get(cfg.body_score, "pose"))
            return None
        self.status.update(body_ref_similarity=scorer.ref_self_similarity, body_refs=scorer.ref_count)
        return scorer
