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

from . import checkpointing as ckpt
from . import compiling
from . import lora as lora_mod
from . import profiling
from . import race as race_mod
from .ema import LoraEMA
from . import quant as Q
from .config import TrainConfig
from .config import side_to_mp
from .dataset import (BucketBatcher, augment_items, cache_latents, cache_text, describe_sides, empty_text_path,
                      items_for_side, load_batch,
                      scan_dataset, text_cache_path)
from .archs import detect_arch, get_arch
from .idloss import effective_weight as idloss_weight
from .multigpu import (Balancer, choose_counts, clone_to, compile_count, job_order, no_recompile_stance, norm_device,
                       run_queue, stance_available, step_seconds)
log = logging.getLogger("rulztrainer")
DP_COMPILE_QUIET_STEPS = 3  # data parallel + compile: serial steps without a new graph before going parallel
TEXT_TIMING_PROMPTS = 4  # prompts encoded on the first copy to time it before deciding on a second copy


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


def free_cuda():
    gc.collect()
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()


def unload_model(model) -> None:
    """Release a model fully, including accelerate dispatch hooks, so its VRAM is returned."""
    try:
        from accelerate.hooks import remove_hook_from_submodules

        remove_hook_from_submodules(model)
    except Exception:  # noqa: BLE001
        pass
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


def loss_weight(cfg: TrainConfig, sigma: torch.Tensor) -> torch.Tensor:
    if cfg.loss_weighting == "none":
        return torch.ones_like(sigma)
    if cfg.loss_weighting == "sigma_sqrt":
        return (sigma ** -0.5).clamp(max=10.0) * 0.5
    if cfg.loss_weighting == "min_snr":
        snr = ((1 - sigma) / sigma) ** 2
        return (snr.clamp(max=5.0) / (snr + 1)).clamp(min=1e-3)
    raise ValueError(f"Unknown loss_weighting {cfg.loss_weighting}")


def build_optimizer(cfg: TrainConfig, params):
    """`params` may be a list of tensors or a list of param-group dicts (LoRA+)."""
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
    if name == "automagic3":
        from .optimizers.automagic3 import Automagic3

        if params and isinstance(params[0], dict):
            if {g.get("name") for g in params} <= {"lora", "lora_down", "lora_up"}:  # Automagic owns its lr: collapse
                if len(params) > 1:  # the LoRA+ groups
                    log.warning("lora_plus_ratio is ignored with automagic3 (it adapts each group's lr itself)")
                params = [p for g in params for p in g["params"]]
            else:  # automagic_groups: by_type, each layer type adapts its own lr from the shared start
                params = [{"params": g["params"]} for g in params]
        return Automagic3(params, lr=cfg.learning_rate, min_lr=cfg.automagic_min_lr, max_lr=cfg.automagic_max_lr,
                          weight_decay=cfg.weight_decay, polarity_history=cfg.automagic_polarity_history,
                          clip_threshold=cfg.automagic_clip_threshold, fused=False)
    raise ValueError(f"Unknown optimizer {cfg.optimizer}")


def current_lr(optimizer, scheduler) -> float:
    """Adapted lr for self-tuning optimizers, scheduler lr otherwise."""
    if hasattr(optimizer, "get_avg_learning_rate"):
        return float(optimizer.get_avg_learning_rate())
    return float(scheduler.get_last_lr()[0])


def build_scheduler(cfg: TrainConfig, optimizer):
    warm = max(0, cfg.warmup_steps)
    if cfg.optimizer.lower() == "automagic3":  # self-tuning lr: no warmup/decay from outside
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)

    def fn(step):
        if step < warm:
            return (step + 1) / (warm + 1)
        p = (step - warm) / max(1, cfg.steps - warm)
        if cfg.lr_scheduler == "cosine":
            return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))
        if cfg.lr_scheduler == "linear":
            return max(0.0, 1 - p)
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.job_dir = cfg.job_dir()
        setup_logging(self.job_dir)
        cfg.save(self.job_dir / "config.yaml")
        self.status = Status(self.job_dir / "status.json")
        self.stop_file = self.job_dir / "STOP"
        if self.stop_file.exists():
            self.stop_file.unlink()
        self.device = torch.device(cfg.device)
        self.arch = get_arch(cfg)  # confirmed against the model's transformer config once the model dir is resolved
        self.vae = None
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
        if not (0.0 <= cfg.sigma_min < cfg.sigma_max <= 1.0):
            raise ValueError("sigma_min/sigma_max must satisfy 0 <= sigma_min < sigma_max <= 1")
        self.devices: list[str] = [cfg.device]
        self.dp = None  # DataParallelLoRA when multi_gpu == data_parallel
        self.balancer = None  # multigpu.Balancer: batches per model copy per step (dp_balance)
        self.balance_hinted = False  # the dp_balance suggestion was logged
        self.resumed_dp_counts = None  # its split from a resumed training state
        self.race = None  # race.RaceRun while a configuration race runs (and after, for its report)
        self.last_probe = None  # (face, body, images) of the last probe
        self.race_base = None  # the run's own config when racing (the configurations change parts of it)
        self.race_rows = None  # [(name, config, overrides)] of the race
        self.race_layouts = None  # [(blocks, LoRA module names)] per configuration
        self.race_union_names = None  # the LoRA wrappers: every configuration's modules together
        self.race_region_dirs: dict[str, Path | None] = {}  # configuration -> its own region-map folder
        self.race_id_loss = None  # the identity loss shared by the configurations that use it
        self.compiled = False  # torch.compile active on the DiT blocks
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
        if norm_device(self.vae.device) == device:
            return self.vae
        if device not in self.vaes:
            self.vaes[device] = clone_to(self.vae, device).eval().requires_grad_(False)
        return self.vaes[device]

    # ------------------------------------------------------------ stages
    def _needs_face_analysis(self) -> bool:
        cfg = self.cfg
        return bool(cfg.wants_regions() or (cfg.crop_augment or "none") != "none" or cfg.likeness_curriculum)

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
                                             recache=cfg.recache)
        n_face = sum(1 for r in self.face_analysis.values() if r.get("bbox"))
        log.info("Face analysis: %d of %d source images have a face", n_face, len(self.face_analysis))
        if (cfg.crop_augment or "none") != "none":
            items = augment_items(items, cfg, self.face_analysis)
        if cfg.wants_regions():
            person_fn = None
            holder = []
            if not cfg.region_weights:
                pass  # identity loss only: face cells + landmarks are enough, no person masks
            elif cfg.body_backend == "mediapipe+sam3":
                try:
                    from .bodysim import SilhouetteScorer

                    sil = SilhouetteScorer(cfg.sam3_model, cfg.body_device,
                                           progress=lambda t, f: self.status.update(stage=t))
                    holder.append(sil)
                    person_fn = sil.person_mask
                except Exception as e:  # noqa: BLE001
                    log.warning("SAM 3 unavailable for person masks (%s); trying MediaPipe", e)
            if person_fn is None and cfg.region_weights:
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
        need_latents = cfg.recache or any(
            not it.latent_path.exists() or (it.latent_flip_path and not it.latent_flip_path.exists()) for it in items)
        need_text = cfg.recache or not empty_text_path(cfg).exists() or any(
            not it.text_path.exists() for it in items) or any(
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
            cache_text(items, cfg, te, tok, extra_prompts=prompts,
                       progress=lambda i, n: self.status.update(step=i, total=n),
                       encoders=encoders if len(encoders) > 1 else None, encode_fn=self.arch.encode_prompt)
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
        stamps = [time.time()]
        cache_text(items, self.cfg, te, tok, extra_prompts=prompts, max_prompts=TEXT_TIMING_PROMPTS,
                   progress=lambda i, n: (stamps.append(time.time()), self.status.update(step=i, total=len(todo))),
                   encode_fn=self.arch.encode_prompt)
        per_prompt = (stamps[-1] - stamps[1]) / max(1, len(stamps) - 2)  # skips the first (warm-up) prompt
        warm_up = max(0.0, stamps[1] - stamps[0] - per_prompt)  # first-call cost, which a new copy pays again
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
            raise

    def _run(self):
        cfg = self.cfg
        cfg.validate()
        torch.manual_seed(cfg.seed)
        rng = random.Random(cfg.seed)
        gen = torch.Generator("cpu").manual_seed(cfg.seed)
        self.status.update(stage="resolving model")
        Q.set_int8_defaults(cfg.int8_matmul, cfg.int8_backward)  # every QuantLinear built from here on
        Q.set_nvfp4_default(cfg.nvfp4_matmul, cfg.nvfp4_backward)
        self.model_dir = self.arch.resolve_for_config(cfg)
        self.arch = detect_arch(cfg, self.model_dir)
        cfg.model_arch = self.arch.name
        log.info("Model architecture: %s", self.arch.label)
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
        log.info("Dataset: %d images in %s", len(items), cfg.dataset_dir)
        items = self.analyze_dataset(items)
        self.prepare_cache(items)

        self.status.update(stage="loading transformer", step=0, total=cfg.steps)
        model = self.arch.get_transformer(cfg, self.model_dir)
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
        injected = lora_mod.inject_lora(model, cfg.lora_rank, cfg.lora_alpha, cfg.lora_targets,
                                        include_text_embedder=cfg.train_text_embedder, dropout=cfg.lora_dropout,
                                        blocks=blocks, names=self._lora_names(model, blocks))
        backprop = lora_mod.backprop_blocks(model, injected, blocks)  # blocks the backward runs from (None = all)
        if blocks is not None:
            log.info("LoRA blocks: %s (%d of %d); %s", lora_mod.format_blocks(blocks), len(blocks), len(model.layers),
                     f"backward skips blocks before {min(blocks) if blocks else 'n/a'}" if backprop is not None else
                     "LoRA before the blocks too (text fusion / projections), so the backward runs through every block")
        if not injected:
            raise ValueError("No LoRA modules selected: check lora_targets / lora_blocks")
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
                                           blocks=blocks, names=self._lora_names(m, blocks))
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
        if self.race_rows is None:
            params = list(lora_mod.lora_parameters(injected))
            n_params = sum(p.numel() for p in params)
            log.info("LoRA: %d modules, %.2fM trainable params, rank %d alpha %g", len(injected), n_params / 1e6,
                     cfg.lora_rank, cfg.lora_alpha)
            optimizer, scheduler = self._make_optimizer(cfg, injected)
        else:  # each configuration of the race builds its own on its first turn
            params, optimizer, scheduler = [], None, None
        step = 0
        if cfg.resume_state:
            step = self.load_state(cfg.resume_state, injected, optimizer, scheduler, gen, rng)
            log.info("Resumed training state from %s at step %d", cfg.resume_state, step)
        if cfg.lora_ema > 0 and self.race_rows is None:
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
        if not cfg.sample_model_path:
            for m, _, _ in self.replicas:
                self._attach_sampling_loras(m)
        self._check_sample_settings()
        if cfg.sample_prompts or cfg.probe_every > 0 or cfg.id_loss_weight > 0:
            self.vae = self.arch.get_vae(cfg, self.model_dir)
            self.face_scorer = self._make_face_scorer()
            self.body_scorer = self._make_body_scorer()
            self.controller = self._make_controller(items)
        self.id_loss = self._make_id_loss()
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

        do_compile, why = compiling.decide(cfg.compile_model, cfg.steps - step, self.devices)
        if why:
            log.info("torch.compile %s (%s)", "on" if do_compile else "off", why)
        if do_compile:
            for m, _, _ in self.replicas:
                self.compiled = self.compile(m, fall_back=cfg.compile_model == "auto") or self.compiled
            if self.compiled and self.dp is not None:
                log.info("Data parallel + compile: replicas run one after another until compilation settles "
                         "(%d steps without a new graph), then in parallel", DP_COMPILE_QUIET_STEPS)
        self.all_items = items
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
            bind(self.race.active)
        if self.dp is not None:
            self.dp.sync_weights()
        if cfg.sample_prompts and cfg.sample_before_training and step == 0:
            self.sample_images(model, step)
        if self.controller is not None and cfg.probe_at_start and step == 0:
            self.run_probe(model, step, optimizer, scheduler, injected)
            if self.race is not None:  # every configuration starts from the same base model: one baseline for all
                face, body, images = self.last_probe
                for v in self.race.variants[1:]:
                    v.state["controller"].update(0, face, body, images)
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
            (rmodel, rinj, rdev), batch, latents, embeds, mask, sigma, noise, regions, cur_step, n_total = job
            latents = latents.to(rdev, torch.float32)
            embeds, mask, sigma, noise = embeds.to(rdev), mask.to(rdev), sigma.to(rdev), noise.to(rdev)
            s = sigma.view(-1, 1, 1, 1)
            noisy = (1 - s) * latents + s * noise
            target = noise - latents
            noisy_in = noisy.to(torch.bfloat16)
            pred = rmodel(noisy_in, sigma * 1000.0, embeds, mask).float()
            sq = (pred - target).pow(2)
            if regions is not None and cfg.region_weights:  # region-weighted loss: face > person > background (mean 1)
                sq = sq * regions.weight.to(rdev)
            per = sq.mean(dim=(1, 2, 3))
            if (regions is not None and cfg.region_weights and cfg.region_prior_weight > 0
                    and bool(regions.prior.any())):
                # prior preservation: keep background cells close to what the base model (LoRA off) predicts
                lora_mod.set_multiplier(rinj, 0.0)
                with torch.no_grad():
                    base_pred = rmodel(noisy_in, sigma * 1000.0, embeds, mask).float()
                lora_mod.set_multiplier(rinj, 1.0)
                pm = regions.prior.to(rdev).float()
                prior = ((pred - base_pred).pow(2) * pm).sum(dim=(1, 2, 3)) / (pm.sum(dim=(1, 2, 3)) * pred.shape[1]).clamp(min=1)
                per = per + cfg.region_prior_weight * prior
            weights = loss_weight(cfg, sigma)
            mults: list[float] = []
            if self.loss_watch is not None:
                per_cpu, sig_cpu = per.detach().cpu().tolist(), sigma.detach().cpu().tolist()
                for it, lv, sv in zip(batch, per_cpu, sig_cpu):
                    self.loss_watch.observe(cur_step, it.key, float(lv), float(sv))
                    mults.append(self.loss_watch.multiplier(it.key))
                if cfg.per_image_lr and not true_per_image_lr:
                    weights = weights * torch.tensor(mults, device=weights.device, dtype=weights.dtype)
            loss = (per * weights).mean() / n_total  # the step's update is the mean over all of its batches
            if self.id_loss is not None and regions is not None and rmodel is model:  # primary replica only
                t_id = time.time()
                id_val, n_id = self.id_loss(noisy, pred, sigma, regions.meta, self.id_ref)
                if id_val is not None:
                    (loss + id_w * id_val / n_total).backward()
                    if str(rdev).startswith("cuda"):
                        torch.cuda.synchronize(rdev)
                    return loss.item(), mults, (id_val.item(), n_id, time.time() - t_id)
            loss.backward()
            return loss.item(), mults, None

        def work_list(entry):
            """One model copy's batches of a step, one after another: [(work result, seconds, image tokens)]."""
            out = []
            for job in entry[1]:
                t0 = time.perf_counter()
                res = work(job)  # its .item() waits for the GPU, so this is the batch's own time
                out.append((res, time.perf_counter() - t0, job[2].shape[2] * job[2].shape[3]))
            return out

        while step < cfg.steps or self._race_running():
            t_step = time.perf_counter()
            prof = profiling.StepProfile(self.job_dir, step + 1, self.devices) if step + 1 == profile_at else None
            if prof is not None:
                prof.start()
            optimizer.zero_grad(set_to_none=True)
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
            try:
                per_rep: list[list] = [[] for _ in self.replicas]
                for r in job_order(counts):  # every batch and its noise drawn in the main thread (shared RNG), in
                    batch = next(self.batch_iter)  # lockstep order: copy 0, copy 1, copy 0, ...
                    step_images += len(batch)
                    latents, embeds, mask, regions = load_batch(batch, cfg, rng)
                    sigma = sample_sigmas(cfg, latents.shape[0], gen, self.ts_logit_shift,
                                          tokens=latents.shape[2] * latents.shape[3], arch=self.arch)
                    noise = torch.randn(latents.shape, generator=gen)
                    per_rep[r].append((self.replicas[r], batch, latents, embeds, mask, sigma, noise, regions, step + 1,
                                       n_total))
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
                free_cuda()
                continue
            if self.dp is not None:
                self.dp.reduce_grads()
            if cfg.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
            if step_lr_mult != 1.0:  # true per-image lr: scale this step only (the scheduler restores it)
                saved_lrs = [g["lr"] for g in optimizer.param_groups]
                for g in optimizer.param_groups:
                    g["lr"] = g["lr"] * step_lr_mult
            optimizer.step()
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
            if step % cfg.save_every == 0 and step < cfg.steps:
                self.save(injected, step)
            state_every = cfg.save_state_every or cfg.save_every
            if cfg.save_state and step % state_every == 0 and step < cfg.steps and not self._race_running():
                self.save_state(step, injected, optimizer, scheduler, gen, rng)  # (a race keeps no states)
            if cfg.sample_prompts and cfg.sample_every > 0 and step % cfg.sample_every == 0 and step < cfg.steps:
                self.sample_images(model, step)
                model.train()
            racing = self._race_running()
            if self.controller is not None and step % cfg.probe_every == 0 and (step < cfg.steps or racing):
                stop = self.run_probe(model, step, optimizer, scheduler, injected)
                model.train()
                if racing:  # the race decides who goes on: record, maybe drop, and switch configuration
                    stop = False
                    self.race.active.state.update(step=step, losses=losses, id_losses=id_losses)
                    bind(self._race_after_probe())
                if stop:
                    log.info("Likeness controller: no improvement for %d probes%s, stopping at step %d",
                             cfg.step_up_stop_patience if cfg.resolution_step_up else cfg.likeness_patience,
                             " at the highest resolution" if cfg.resolution_step_up else "", step)
                    break
            if self.stop_file.exists():
                log.info("Stop requested at step %d", step)
                break
        if self.compiled:
            compiling.report("end of training")
        if self._race_running():
            log.warning("Stopped before the race was decided; the final save is %s's (the others are in race.json "
                        "and their _dropped files)", self.race.active.name)
        path = self.save(injected, step, final=True)
        if cfg.save_state and not self._race_running():
            self.save_state(step, injected, optimizer, scheduler, gen, rng)
        if cfg.sample_prompts:
            self.sample_images(model, step)
        if self.race is not None:
            self._race_report()
        self.status.update(stage="finished", done=True)
        log.info("Done. Final LoRA: %s", path)

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
        tag = self._file_tag()  # a race's training configuration while the race runs
        name = f"{cfg.job_name}{tag}.safetensors" if final else f"{cfg.job_name}{tag}_{step:06d}.safetensors"
        path = self.job_dir / name
        meta = {"ss_base_model": cfg.model_path if cfg.model_source == "hf" else Path(cfg.transformer_file).name,
                "ss_network_dim": str(cfg.lora_rank), "ss_network_alpha": str(cfg.lora_alpha),
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

    def compile(self, model, fall_back: bool = False) -> bool:
        try:
            import triton  # noqa: F401
        except ImportError:
            log.warning("compile_model requested but triton is not installed; running without compile")
            return False
        try:
            compiling.prepare(fall_back)  # raised recompile limits, dynamo warnings in train.log
            blocks = self.arch.compile_blocks(model)
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
                                   id_loss_weight=max(r[1].id_loss_weight for r in rows),
                                   likeness_curriculum=any(r[1].likeness_curriculum for r in rows))
        est = race_mod.estimate_steps(len(rows), cfg.race_warmup_probes, cfg.probe_every, cfg.steps)
        log.info("Configuration race: %d configurations, %d warm-up probe round(s), a probe every %d steps: %d rounds "
                 "and %d probes, about %d training steps in all (%.1fx one %d-step run)", len(rows),
                 cfg.race_warmup_probes, cfg.probe_every, est["rounds"], est["probes"], est["total"], est["ratio"],
                 cfg.steps)
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
            v.init_lora(wrappers, base.seed)
            v.state["items"] = race_mod.variant_items(items, vcfg, self.race_region_dirs.get(name))
            v.state["controller"] = LikenessController(vcfg)
            n_w = sum(v.down[m].numel() + v.up[m].numel() for m in names)
            log.info("Race configuration %s: %d LoRA modules, %.2fM weights (rank %d, alpha %g), %d dataset items",
                     name, len(names), n_w / 1e6, v.rank, v.alpha, len(v.state["items"]))
            variants.append(v)
        est = race_mod.estimate_steps(len(variants), base.race_warmup_probes, base.probe_every, base.steps)
        self.race = race_mod.RaceRun(variants, race_mod.RaceSchedule([v.name for v in variants],
                                                                     base.race_warmup_probes), wrappers, base, est)
        self._race_activate(variants[0])
        self._race_report()

    def _race_activate(self, v) -> None:
        """Swap configuration v in: its LoRA, settings and per-run state (built on its first turn)."""
        race_mod.swap_in(v, self.race.wrappers)
        if self.dp is not None:
            self.dp.sync_weights()  # replicas follow the wrappers' shapes, scale and on/off
        self.cfg = v.cfg
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
        st["injected"] = {name: self.race.wrappers[name] for name in v.names}
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
        score = self.controller.combined_ema if self.controller is not None else None
        rounds = len(rr.schedule.history)
        dropped = rr.schedule.record(cur.idx, score)
        if len(rr.schedule.history) > rounds:
            entry = rr.schedule.history[-1]
            fmt = ", ".join(f"{n} {'n/a' if s is None else f'{s:.1f}%'}" for n, s in entry["scores"].items())
            log.info("Race round %d: %s%s", entry["round"], fmt,
                     f" -> {entry['dropped']} is out" if entry["dropped"] else " (warm-up: nobody is dropped)")
        if dropped is not None:
            self._race_drop(rr.variants[dropped])
        nxt = rr.variants[rr.schedule.current]
        if nxt is not cur:
            if cur.alive:
                self._race_park(cur)
            self._race_activate(nxt)
            free_cuda()
        if rr.schedule.done:
            self._race_finish()
        self._race_report()
        return nxt

    def _race_drop(self, v) -> None:
        """Save a dropped configuration's LoRA (its weight average when it has one) and free its state."""
        cfg = self.race.base_cfg
        path = self.job_dir / f"{cfg.job_name}_{v.name}_dropped.safetensors"
        meta = {"ss_network_dim": str(v.rank), "ss_network_alpha": str(v.alpha), "modelspec.architecture":
                self.arch.modelspec, "rulztrainer_race": f"{v.name}: {race_mod.format_overrides(v.overrides)}",
                "ss_steps": str(v.state.get("step", 0))}
        if v is self.race.active:
            with self._averaged():
                lora_mod.save_lora(v.state["injected"], path, metadata=meta, dtype=cfg.lora_save_dtype,
                                   fmt=self.arch.lora_format)
        else:
            lora_mod.save_lora(v.view(use_average=True), path, metadata=meta, dtype=cfg.lora_save_dtype,
                               fmt=self.arch.lora_format)
        log.info("Race: %s dropped after round %d -> %s", v.name, self.race.schedule.dropped[v.idx], path.name)
        if str(path) not in self.status.data["saved"]:
            self.status.data["saved"].append(str(path))
        v.alive = False
        keep = {k: v.state[k] for k in ("step", "controller") if k in v.state}
        v.state.clear()
        v.state.update(keep)
        v.down, v.up = {}, {}

    def _race_finish(self) -> None:
        """One configuration left: it trains on as the run, with the run's own file names."""
        import shutil

        rr = self.race
        win = rr.active
        cfg = rr.base_cfg
        best = self.job_dir / f"{cfg.job_name}_{win.name}_best.safetensors"
        if best.exists():
            shutil.copyfile(best, self.job_dir / f"{cfg.job_name}_best.safetensors")
        win_cfg = dataclasses.replace(win.cfg, race=False, race_configs=[], race_warmup_probes=0)
        win.cfg = win_cfg
        self.cfg = win_cfg
        win_cfg.save(self.job_dir / f"{cfg.job_name}_winner.yaml")
        log.info("Race over: %s wins (%s); it trains on to step %d with the run's own file names (its settings: %s)",
                 win.name, race_mod.format_overrides(win.overrides) or "the run's own settings", cfg.steps,
                 f"{cfg.job_name}_winner.yaml")

    def _race_status(self) -> dict:
        rr = self.race
        out = rr.schedule.state()
        out["active"] = rr.active.name if rr.active is not None else None
        out["steps"] = {v.name: (v.state.get("step", 0) if v is not rr.active else None) for v in rr.variants}
        out["scores"] = {v.name: (v.state["controller"].combined_ema if "controller" in v.state else None)
                         for v in rr.variants}
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
        """While a race runs, files of the training configuration carry its name."""
        return f"_{self.race.active.name}" if self._race_running() else ""

    def _image_prefix(self) -> str:
        return f"{self.race.active.name}_" if self._race_running() else ""

    def _make_optimizer(self, cfg, injected):
        if cfg.optimizer.lower() == "automagic3" and cfg.automagic_groups == "by_type":
            groups = lora_mod.lora_param_groups_by(injected, cfg.learning_rate, self.arch.param_group_of)
            log.info("Automagic3: an adaptive lr per layer type (%s)", ", ".join(
                f"{g['name']} {sum(p.numel() for p in g['params']) / 1e6:.1f}M" for g in groups))
        else:
            groups = lora_mod.lora_param_groups(injected, cfg.learning_rate, cfg.lora_plus_ratio)
            if len(groups) > 1:
                log.info("LoRA+: lora_up lr = %.2e (x%g), lora_down lr = %.2e", groups[1]["lr"], cfg.lora_plus_ratio,
                         groups[0]["lr"])
        optimizer = build_optimizer(cfg, groups)
        return optimizer, build_scheduler(cfg, optimizer)

    def _make_ema(self, cfg, injected):
        if cfg.lora_ema <= 0:
            return None
        per_step = cfg.batch_size * cfg.gradient_accumulation * len(self.replicas)
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

    def _attach_sampling_loras(self, model) -> None:
        """Attach the sampling-only LoRAs (turbo + optional secondary) to a model."""
        cfg = self.cfg
        for label, path, strength in (("Turbo LoRA", cfg.turbo_lora, cfg.turbo_lora_strength),
                                      ("Sampling LoRA", cfg.sample_lora, cfg.sample_lora_strength)):
            if path:
                n = lora_mod.attach_extra_lora(model, path, strength,
                                               rename=getattr(self.arch, "lora_module_name", None))
                log.info("%s %s: matched %d modules (strength %g, sampling only)", label, path, n, strength)
                if label == "Turbo LoRA" and n:
                    model.turbo_lora_attached = True  # the Krea 2 sampler then uses the distilled schedule

    def _check_sample_settings(self) -> None:
        """Warn about sample / probe settings that give unusable images: guidance on a distilled (turbo) sampler burns
        them, a few steps or no guidance without one leaves them unfinished."""
        cfg = self.cfg
        distilled = bool(cfg.turbo_lora) or "turbo" in (cfg.sample_model_path or "").lower()
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
                                       names=self._lora_names(m, blocks))
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
                for (m, _), mode in zip(pairs, modes):
                    m.train(mode)
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
            free_cuda()
            for m, d in train_models:
                if getattr(m, "split_assignment", None):  # straight back to its GPUs (it may not fit one)
                    from .multigpu import apply_assignment

                    apply_assignment(m, m.split_assignment)
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
        with self._averaged(replicas=True), self._sampling_slots(model, step) as slots:
            return self._sample_with(slots, step)

    # ------------------------------------------------------------ per-image loss watch
    def _make_loss_watch(self, items, optimizer):
        cfg = self.cfg
        if not (cfg.loss_watch or cfg.per_image_lr):
            return None
        from .losswatch import LossWatch

        if cfg.per_image_lr and cfg.optimizer.lower() == "automagic3":
            log.warning("per_image_lr is not supported with automagic3 (its step normalisation absorbs the scaling); "
                        "the loss watch will only observe")
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

    def _refresh_batcher_weights(self) -> None:
        """Combine curriculum weights with loss-watch exclusions into the batcher."""
        if self.batcher is None:
            return
        weights: dict[str, float] | None = dict(self.curriculum_w) if self.curriculum_w else None
        if (self.cfg.crop_augment or "none") != "none" and self.cfg.crop_augment_weight != 1.0:
            weights = weights or {}
            for it in self.batcher.items:
                if it.category != "image":
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

        ctrl = LikenessController(cfg)
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

    @torch.no_grad()
    def _probe_with(self, slots, step):
        from .likeness import probe_body_size

        cfg = self.cfg
        for sl in slots:
            sl.model.eval()
            lora_mod.set_extras_enabled(sl.model, True)
        neg_e, neg_m = self.arch.pad_embeds([load_text_embeds(cfg, self.negative_prompt())])
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
            n_seeds = max(1, cfg.probe_seeds)
            for k in range(n_seeds):
                jobs.append(RenderJob(prompt, w, h, cfg.probe_steps, cfg.probe_cfg, cfg.probe_seed + k,
                                      out_dir / f"{self._image_prefix()}step{step:06d}_{kind}_{k}.png",
                                      f"probe {kind} seed {k + 1}/{n_seeds} at step {step}" if len(slots) == 1
                                      else f"probe {kind} seed {k + 1}/{n_seeds}", kind))
        title = f"probes at step {step} on {len(slots)} GPUs" if len(slots) > 1 and len(jobs) > 1 else ""
        board = _Board(self.status, title)
        try:
            images = self._render_round(slots, jobs, neg_e, neg_m, board)
        finally:
            for sl in slots:
                lora_mod.set_extras_enabled(sl.model, False)
        face_scores, body_scores = [], []
        self.status.update(stage=f"scoring probes (step {step})")
        for job, p in zip(jobs, images):  # scoring stays in this thread: the scorers are not shared across threads
            if job.kind == "face":
                sc = self.face_scorer.score(p, step, 0)
                if sc.percent is not None:
                    face_scores.append(sc.percent)
            else:
                pct, _ = self.body_scorer.score(p)
                if pct is not None:
                    body_scores.append(pct)
        face = sum(face_scores) / len(face_scores) if face_scores else None
        body = sum(body_scores) / len(body_scores) if body_scores else None
        return face, body, images

    def run_probe(self, model, step, optimizer, scheduler, injected) -> bool:
        """Render + score probes, update the controller, apply its actions. Returns True to stop training.

        With a weight average the probes score it, so a best checkpoint is exactly the weights that were scored."""
        self._free_for_rendering()
        with self._averaged(replicas=True):
            return self._run_probe(model, step, optimizer, scheduler, injected)

    def _run_probe(self, model, step, optimizer, scheduler, injected) -> bool:
        from .likeness import append_record, apply_lr_ratio, curriculum_weights, timestep_logit_shift

        cfg = self.cfg
        t0 = time.time()
        with self._sampling_slots(model, step) as slots:
            face, body, images = self._probe_with(slots, step)
        self.last_probe = (face, body, images)
        rec, actions = self.controller.update(step, face, body, images)
        append_record(self.job_dir, rec)
        fmt = lambda v: "n/a" if v is None else f"{v:.1f}%"  # noqa: E731
        who = f"[{self.race.active.name}] " if self._race_running() else ""
        log.info("%sProbe step %d: face %s body %s combined %s | ema face %s body %s combined %s | best %s @%s "
                 "| lr x%.2f | face bias %+.2f (%.1fs)", who, step, fmt(face), fmt(body), fmt(rec.combined),
                 fmt(rec.face_ema), fmt(rec.body_ema), fmt(rec.combined_ema), fmt(rec.best), rec.best_step,
                 rec.lr_factor, rec.face_bias, time.time() - t0)
        if "step_up" in actions:
            log.info("Resolution step-up: stage %d stalled for %d probe(s) (stage best %s)", rec.stage + 1,
                     rec.stage_since_improve, fmt(rec.stage_best))
            self._enter_stage(actions["step_up"], step)
        if actions.get("save_best"):
            path = self.job_dir / f"{cfg.job_name}{self._file_tag()}_best.safetensors"
            lora_mod.save_lora(injected, path, metadata={"ss_steps": str(step), "likeness_best": f"{rec.best:.2f}",
                                                         "modelspec.architecture": self.arch.modelspec},
                               dtype=cfg.lora_save_dtype, fmt=self.arch.lora_format)
            log.info("New best likeness %.1f%% at step %d -> %s", rec.best, step, path)
            if str(path) not in self.status.data["saved"]:
                self.status.data["saved"].append(str(path))
        if actions.get("id_loss_veto") and self.id_loss is not None:
            self.id_loss = None
            if self._race_running():
                self.race.active.state["id_vetoed"] = True
            log.warning("Identity loss vetoed at step %d: the smoothed face score %s fell more than %.1f points below "
                        "its best %s for two probes in a row; the identity loss is off for the rest of the run", step,
                        fmt(rec.face_ema), 2 * cfg.likeness_noise_band, fmt(self.controller.face_best))
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
        return bool(actions.get("stop"))

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
                                       cfg.face_weight, cfg.body_weight)
                self.status.data["face_scores"] = self.status.data.get("face_scores", []) + [asdict(s) for s in scores]
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
        return scorer

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
            sil = None
            if cfg.body_backend == "mediapipe+sam3":
                try:
                    sil = bodysim.SilhouetteScorer(cfg.sam3_model, cfg.body_device,
                                                   progress=lambda t, f: self.status.update(stage=t))
                except Exception as e:  # noqa: BLE001
                    log.warning("SAM 3 unavailable (%s); body scoring uses MediaPipe only. For facebook/sam3 accept "
                                "the license on huggingface.co and run 'hf auth login'.", e)
            scorer = bodysim.BodyScorer(cfg.models_dir, silhouette=sil,
                                        progress=lambda t, f: self.status.update(stage=t))
            scorer.set_references(cfg.body_refs)
        except Exception as e:  # noqa: BLE001
            log.warning("Could not initialise body scoring (%s); disabled", e)
            return None
        if not scorer.profile:
            log.warning("No usable pose in any body reference image; body scoring disabled")
            return None
        self.status.update(body_ref_similarity=scorer.ref_self_similarity, body_refs=scorer.ref_count)
        return scorer
