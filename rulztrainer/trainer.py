"""LoRA training loop for MOSS-Image (flow matching)."""
from __future__ import annotations

import gc
import json
import logging
import math
import random
import sys
import time
from pathlib import Path

import torch
from PIL import Image

from . import lora as lora_mod
from . import quant as Q
from .config import TrainConfig
from .config import side_to_mp
from .dataset import (BucketBatcher, augment_items, cache_latents, cache_text, describe_sides, empty_text_path,
                      items_for_side, load_batch,
                      scan_dataset, text_cache_path)
from .moss import loading
from .multigpu import compile_count, no_recompile_stance, stance_available
from .sampler import sample

log = logging.getLogger("rulztrainer")
DP_COMPILE_QUIET_STEPS = 3  # data parallel + compile: serial steps without a new graph before going parallel


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


class Status:
    """Writes a small JSON file the WebUI polls."""

    def __init__(self, path: Path):
        self.path = path
        self.data = {"stage": "starting", "step": 0, "total": 0, "loss": None, "avg_loss": None,
                     "elapsed": 0.0, "eta": None, "samples": [], "saved": [], "error": None, "done": False,
                     "losses": [], "states": []}
        self.write()

    def update(self, **kw):
        self.data.update(kw)
        self.write()

    def write(self):
        """Atomic replace with retries: on Windows a reader (the WebUI poll) can briefly lock the target."""
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


def sample_sigmas(cfg: TrainConfig, n: int, gen: torch.Generator, logit_shift: float = 0.0) -> torch.Tensor:
    mode = cfg.timestep_sampling
    if mode == "uniform":
        s = _shift_logit(torch.rand(n, generator=gen), logit_shift)
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

        if params and isinstance(params[0], dict):  # Automagic owns its lr: collapse LoRA+ groups
            if len(params) > 1:
                log.warning("lora_plus_ratio is ignored with automagic3 (it adapts each group's lr itself)")
            params = [p for g in params for p in g["params"]]
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
        self.vae = None
        self.face_scorer = None
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

    # ------------------------------------------------------------ stages
    def _needs_face_analysis(self) -> bool:
        cfg = self.cfg
        return bool(cfg.region_weights or (cfg.crop_augment or "none") != "none" or cfg.likeness_curriculum)

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
        if cfg.region_weights:
            person_fn = None
            holder = []
            if cfg.body_backend == "mediapipe+sam3":
                try:
                    from .bodysim import SilhouetteScorer

                    sil = SilhouetteScorer(cfg.sam3_model, cfg.body_device,
                                           progress=lambda t, f: self.status.update(stage=t))
                    holder.append(sil)
                    person_fn = sil.person_mask
                except Exception as e:  # noqa: BLE001
                    log.warning("SAM 3 unavailable for person masks (%s); trying MediaPipe", e)
            if person_fn is None:
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
            counts = R.cache_region_maps(items, cfg, self.face_analysis, person_fn,
                                         progress=lambda i, n: self.status.update(step=i, total=n))
            log.info("Region maps ready (new this run): %s; weights face %.2g / body %.2g / background %.2g",
                     counts, cfg.face_loss_weight, cfg.body_loss_weight, cfg.background_loss_weight)
            self.status.update(regions=counts)
            for obj in holder:
                unload_model(obj.model if hasattr(obj, "model") else obj)
            free_cuda()
        return items

    def prepare_cache(self, items):
        cfg = self.cfg
        prompts = list(cfg.sample_prompts) + ([cfg.sample_negative] if cfg.sample_prompts else [])
        if cfg.probe_every > 0:
            prompts += [p for p in (cfg.probe_face_prompt, cfg.probe_body_prompt) if p] + [cfg.sample_negative]
        need_latents = cfg.recache or any(
            not it.latent_path.exists() or (it.latent_flip_path and not it.latent_flip_path.exists()) for it in items)
        need_text = cfg.recache or not empty_text_path(cfg).exists() or any(
            not it.text_path.exists() for it in items) or any(
            not text_cache_path(cfg, p).exists() for p in prompts)
        if need_latents:
            self.status.update(stage="caching latents")
            vae = loading.get_vae(cfg, self.model_dir)
            cache_latents(items, cfg, vae, progress=lambda i, n: self.status.update(step=i, total=n))
            unload_model(vae)
        if need_text:
            self.status.update(stage="loading text encoder")
            te, tok = loading.get_text_encoder(cfg, self.model_dir)
            self.status.update(stage="caching text embeddings")
            cache_text(items, cfg, te, tok, extra_prompts=prompts,
                       progress=lambda i, n: self.status.update(step=i, total=n))
            unload_model(te)
            del tok
            log.info("Text encoder unloaded after caching")

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
        self.model_dir = loading.resolve_for_config(cfg)
        items = scan_dataset(cfg)
        log.info("Dataset: %d images in %s", len(items), cfg.dataset_dir)
        items = self.analyze_dataset(items)
        self.prepare_cache(items)

        self.status.update(stage="loading transformer", step=0, total=cfg.steps)
        model = loading.get_transformer(cfg, self.model_dir)
        model.gradient_checkpointing = cfg.gradient_checkpointing
        n_w8a8 = Q.configure_int8(model, cfg.int8_matmul, cfg.int8_backward)
        if n_w8a8:
            log.info("%d quantized Linear layers: int8 matmul %s (backward %s)", n_w8a8,
                     "on" if cfg.int8_matmul else "off", cfg.int8_backward)
        blocks = lora_mod.parse_blocks(cfg.lora_blocks, len(model.layers))
        injected = lora_mod.inject_lora(model, cfg.lora_rank, cfg.lora_alpha, cfg.lora_targets,
                                        include_text_embedder=cfg.train_text_embedder, dropout=cfg.lora_dropout,
                                        blocks=blocks)
        if blocks is not None:
            log.info("LoRA blocks: %s (%d of %d); backward skips blocks before %s", lora_mod.format_blocks(blocks),
                     len(blocks), len(model.layers), min(blocks) if blocks else "n/a")
        if not injected:
            raise ValueError("No LoRA modules selected: check lora_targets / lora_blocks")
        self.replicas = [(model, injected, self.devices[0])]
        if cfg.multi_gpu == "data_parallel":
            from .multigpu import DataParallelLoRA

            reps = []
            for dev in self.devices[1:]:
                self.status.update(stage=f"loading model replica on {dev}")
                m = loading.get_transformer(cfg, self.model_dir, device=dev)
                m.gradient_checkpointing = cfg.gradient_checkpointing
                Q.configure_int8(m, cfg.int8_matmul, cfg.int8_backward)
                inj = lora_mod.inject_lora(m, cfg.lora_rank, cfg.lora_alpha, cfg.lora_targets,
                                           include_text_embedder=cfg.train_text_embedder, dropout=cfg.lora_dropout,
                                           blocks=blocks)
                reps.append((m, inj, dev))
                self.replicas.append((m, inj, dev))
            self.dp = DataParallelLoRA(injected, reps)
            log.info("Data parallel on %s: %d images per step, LoRA grads averaged", ", ".join(self.devices),
                     cfg.batch_size * len(self.devices))
        elif cfg.multi_gpu == "model_split":
            from .multigpu import split_model

            split_model(model, self.devices, trained=blocks, plan=cfg.split_plan)
        if cfg.resume_from:
            n = lora_mod.load_lora_weights(injected, cfg.resume_from)
            log.info("Resumed %d LoRA modules from %s", n, cfg.resume_from)
        params = list(lora_mod.lora_parameters(injected))
        n_params = sum(p.numel() for p in params)
        log.info("LoRA: %d modules, %.2fM trainable params, rank %d alpha %g", len(injected), n_params / 1e6,
                 cfg.lora_rank, cfg.lora_alpha)
        groups = lora_mod.lora_param_groups(injected, cfg.learning_rate, cfg.lora_plus_ratio)
        if len(groups) > 1:
            log.info("LoRA+: lora_up lr = %.2e (x%g), lora_down lr = %.2e", groups[1]["lr"], cfg.lora_plus_ratio,
                     groups[0]["lr"])
        optimizer = build_optimizer(cfg, groups)
        scheduler = build_scheduler(cfg, optimizer)
        step = 0
        if cfg.resume_state:
            step = self.load_state(cfg.resume_state, injected, optimizer, scheduler, gen, rng)
            log.info("Resumed training state from %s at step %d", cfg.resume_state, step)
        self.injected = injected
        self.sample_model = None
        if not cfg.sample_model_path:
            self._attach_sampling_loras(model)
        if cfg.sample_prompts or cfg.probe_every > 0:
            self.vae = loading.get_vae(cfg, self.model_dir)
            self.face_scorer = self._make_face_scorer()
            self.body_scorer = self._make_body_scorer()
            self.controller = self._make_controller(items)
        if torch.cuda.is_available():
            from .multigpu import gpu_memory_summary

            log.info("GPU memory after model load: %s", gpu_memory_summary(self.devices))

        if cfg.compile_model:
            for m, _, _ in self.replicas:
                self.compiled = self.compile(m) or self.compiled
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
        self._enter_stage(self.res_stage, step, initial=True)
        self.loss_watch = self._make_loss_watch(items, optimizer)
        true_per_image_lr = (self.loss_watch is not None and cfg.per_image_lr and cfg.batch_size == 1
                             and cfg.gradient_accumulation == 1 and len(self.replicas) == 1)
        n_dev = len(self.replicas)
        if self.dp is not None:
            self.dp.sync_weights()
        if cfg.sample_prompts and cfg.sample_before_training and step == 0:
            self.sample_images(model, step)
        if self.controller is not None and cfg.probe_at_start and step == 0:
            self.run_probe(model, step, optimizer, scheduler, injected)
        model.train()
        if self.dp is not None:
            self.dp.train(True)
        self.status.update(stage="training", step=step)
        start = time.time()
        start_step = step
        losses: list[float] = []
        def work(job):
            """Forward + backward for one model copy on its device. Returns (loss value, per-image multipliers)."""
            (rmodel, rinj, rdev), batch, latents, embeds, mask, sigma, noise, regions, cur_step = job
            latents = latents.to(rdev, torch.float32)
            embeds, mask, sigma, noise = embeds.to(rdev), mask.to(rdev), sigma.to(rdev), noise.to(rdev)
            s = sigma.view(-1, 1, 1, 1)
            noisy = (1 - s) * latents + s * noise
            target = noise - latents
            noisy_in = noisy.to(torch.bfloat16)
            pred = rmodel(noisy_in, sigma * 1000.0, embeds, mask).float()
            sq = (pred - target).pow(2)
            if regions is not None:  # region-weighted loss: face > person > background (map has mean 1)
                sq = sq * regions.weight.to(rdev)
            per = sq.mean(dim=(1, 2, 3))
            if regions is not None and cfg.region_prior_weight > 0 and bool(regions.prior.any()):
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
            loss = (per * weights).mean() / (cfg.gradient_accumulation * n_dev)
            loss.backward()
            return loss.item(), mults

        while step < cfg.steps:
            optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0
            step_lr_mult = 1.0
            for _ in range(cfg.gradient_accumulation):
                jobs = []
                for rep in self.replicas:  # draw every copy's batch and noise in the main thread (shared RNG)
                    batch = next(self.batch_iter)
                    latents, embeds, mask, regions = load_batch(batch, cfg, rng)
                    sigma = sample_sigmas(cfg, latents.shape[0], gen, self.ts_logit_shift)
                    noise = torch.randn(latents.shape, generator=gen)
                    jobs.append((rep, batch, latents, embeds, mask, sigma, noise, regions, step + 1))
                if self.dp is not None:
                    # torch.compile + threads: a replica tracing a graph breaks the other replica's forward, so steps
                    # that may compile run serially; parallel steps run under a stance that never compiles (an unseen
                    # shape then runs that block eagerly once instead of tracing)
                    serial = self.compiled and (self.dp_quiet_steps < DP_COMPILE_QUIET_STEPS or not stance_available())
                    n_before = compile_count()
                    if self.compiled and not serial:
                        with no_recompile_stance():
                            results = self.dp.run(work, jobs, serial=False)
                    else:
                        results = self.dp.run(work, jobs, serial=serial)
                    if compile_count() != n_before:
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
                    results = [work(jobs[0])]
                accum_loss += sum(r[0] for r in results)
                if true_per_image_lr and results[0][1]:
                    step_lr_mult = results[0][1][0]
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
            scheduler.step()
            if self.dp is not None:
                self.dp.sync_weights()
            step += 1
            losses.append(accum_loss)
            avg = sum(losses[-50:]) / len(losses[-50:])
            elapsed = time.time() - start
            done_steps = max(1, step - start_step)
            eta = elapsed / done_steps * (cfg.steps - step)
            lr_now = current_lr(optimizer, scheduler)
            if step % 5 == 0 or step == start_step + 1:
                log.info("step %d/%d loss %.4f avg %.4f lr %.2e %.1fs/it", step, cfg.steps, accum_loss, avg,
                         lr_now, elapsed / done_steps)
            self.status.update(step=step, total=cfg.steps, loss=accum_loss, avg_loss=avg, elapsed=elapsed, eta=eta,
                               lr=lr_now, losses=losses[-2000:])
            if self.loss_watch is not None and self.loss_watch.epoch_complete():
                summary = self.loss_watch.end_epoch(step)
                log.info("Loss watch epoch %d: %s (median residual %.3f, %d/%d active)", summary["epoch"],
                         summary["counts"], summary["median"] or 0.0, summary["active"], summary["total"])
                self.status.update(loss_watch=summary)
                self._refresh_batcher_weights()
            if step % cfg.save_every == 0 and step < cfg.steps:
                self.save(injected, step)
            state_every = cfg.save_state_every or cfg.save_every
            if cfg.save_state and step % state_every == 0 and step < cfg.steps:
                self.save_state(step, injected, optimizer, scheduler, gen, rng)
            if cfg.sample_prompts and cfg.sample_every > 0 and step % cfg.sample_every == 0 and step < cfg.steps:
                self.sample_images(model, step)
                model.train()
            if self.controller is not None and step % cfg.probe_every == 0 and step < cfg.steps:
                stop = self.run_probe(model, step, optimizer, scheduler, injected)
                model.train()
                if stop:
                    log.info("Likeness controller: no improvement for %d probes%s, stopping at step %d",
                             cfg.step_up_stop_patience if cfg.resolution_step_up else cfg.likeness_patience,
                             " at the highest resolution" if cfg.resolution_step_up else "", step)
                    break
            if self.stop_file.exists():
                log.info("Stop requested at step %d", step)
                break
        path = self.save(injected, step, final=True)
        if cfg.save_state:
            self.save_state(step, injected, optimizer, scheduler, gen, rng)
        if cfg.sample_prompts:
            self.sample_images(model, step)
        self.status.update(stage="finished", done=True)
        log.info("Done. Final LoRA: %s", path)

    # ------------------------------------------------------------ saving
    def save(self, injected, step, final=False) -> Path:
        cfg = self.cfg
        name = f"{cfg.job_name}.safetensors" if final else f"{cfg.job_name}_{step:06d}.safetensors"
        path = self.job_dir / name
        meta = {"ss_base_model": cfg.model_path if cfg.model_source == "hf" else Path(cfg.transformer_file).name,
                "ss_network_dim": str(cfg.lora_rank), "ss_network_alpha": str(cfg.lora_alpha),
                "ss_steps": str(step), "modelspec.architecture": "moss-image", "rulztrainer_config": cfg.to_dict(),
                "ss_blocks": cfg.lora_blocks or "all"}
        lora_mod.save_lora(injected, path, metadata=meta, dtype=cfg.lora_save_dtype)
        log.info("Saved %s (%s)", path, cfg.lora_save_dtype)
        self.status.data["saved"].append(str(path))
        self.status.write()
        return path

    def save_state(self, step, injected, optimizer, scheduler, gen, rng) -> Path:
        state_dir = self.job_dir / f"state_{step:06d}"
        state_dir.mkdir(exist_ok=True)
        lora_mod.save_lora(injected, state_dir / "lora.safetensors", dtype=torch.float32)
        torch.save({
            "step": step, "res_stage": self.res_stage, "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "torch_gen": gen.get_state(), "py_rng": rng.getstate(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "config": self.cfg.to_dict(),
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

    def compile(self, model) -> bool:
        try:
            import triton  # noqa: F401
        except ImportError:
            log.warning("compile_model requested but triton is not installed; running without compile")
            return False
        try:
            # Python float attributes (LoRA scale / multiplier) would otherwise be traced as 0-d CPU tensors, and the
            # CPU ops on them make Inductor emit C++ that needs MSVC on Windows ("InvalidCxxCompiler: cl is not found")
            torch._dynamo.config.specialize_float = True
            for block in (*model.layers, *model.text_embedder.blocks):
                block.compile(backend="inductor", dynamic=True)
            log.info("Compiled %d DiT blocks with torch.compile (first steps will be slow)", len(model.layers) + 2)
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
            raise ValueError(f"Training state was saved with lora_plus_ratio="
                             f"{st.get('config', {}).get('lora_plus_ratio', 0)} ({saved_groups} param groups); "
                             f"set the same value to resume")
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
        return int(st["step"])

    # ------------------------------------------------------------ sampling
    def _attach_sampling_loras(self, model) -> None:
        """Attach the sampling-only LoRAs (turbo + optional secondary) to a model."""
        cfg = self.cfg
        for label, path, strength in (("Turbo LoRA", cfg.turbo_lora, cfg.turbo_lora_strength),
                                      ("Sampling LoRA", cfg.sample_lora, cfg.sample_lora_strength)):
            if path:
                n = lora_mod.attach_extra_lora(model, path, strength)
                log.info("%s %s: matched %d modules (strength %g, sampling only)", label, path, n, strength)

    def _get_sample_model(self):
        """Lazily load the separate sampling model (kept on CPU between samplings) with matching LoRA slots."""
        cfg = self.cfg
        if self.sample_model is None:
            log.info("Loading sampling model %s (quant=%s)", cfg.sample_model_path, cfg.sample_model_quant)
            m = loading.load_any_transformer(cfg.sample_model_path, "cpu", cfg.sample_model_quant, cfg.models_dir)
            inj = lora_mod.inject_lora(m, cfg.lora_rank, cfg.lora_alpha, cfg.lora_targets,
                                       include_text_embedder=cfg.train_text_embedder,
                                       blocks=lora_mod.parse_blocks(cfg.lora_blocks, len(m.layers)))
            if set(inj) != set(self.injected):
                raise ValueError("Sampling model has different LoRA targets than the training model")
            self._attach_sampling_loras(m)
            self.sample_model, self.sample_injected = m, inj
        with torch.no_grad():  # copy the current LoRA weights
            for name, src in self.injected.items():
                dst = self.sample_injected[name]
                dst.lora_down.weight.copy_(src.lora_down.weight.detach().to(dst.lora_down.weight.device))
                dst.lora_up.weight.copy_(src.lora_up.weight.detach().to(dst.lora_up.weight.device))
                dst.alpha, dst.scale = src.alpha, src.scale
        return self.sample_model

    def _with_sampling_model(self, model, fn, step):
        """Run fn(model_for_sampling) on the training model or, if configured, the separate sampling model."""
        cfg = self.cfg
        if not cfg.sample_model_path:
            return fn(model)
        self.status.update(stage=f"swapping in sampling model (step {step})")
        model.to("cpu")
        free_cuda()
        sample_model = self._get_sample_model().to(self.device)
        try:
            return fn(sample_model)
        finally:
            sample_model.to("cpu")
            free_cuda()
            model.to(self.device)
            if getattr(model, "split_assignment", None):
                from .multigpu import apply_assignment

                apply_assignment(model, model.split_assignment)

    @torch.no_grad()
    def sample_images(self, model, step):
        return self._with_sampling_model(model, lambda m: self._sample_with(m, step), step)

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

    def _render(self, model, prompt, width, height, steps, cfg_scale, seed, out_path, neg_e, neg_m, stage):
        from safetensors.torch import load_file

        cfg = self.cfg
        emb = load_file(str(text_cache_path(cfg, prompt)))["embeds"]
        pos_e, pos_m = loading.pad_embeds([emb])
        self.status.update(stage=stage)
        lat = sample(model, pos_e, pos_m, neg_e, neg_m, width=width, height=height, steps=steps, cfg=cfg_scale,
                     shift=cfg.sample_shift, seed=seed, skip_layer=cfg.sample_skip_layer, device=self.device)
        img = loading.decode_latents(self.vae, lat)[0]
        arr = ((img.permute(1, 2, 0).float().cpu().numpy() + 1) * 127.5).round().clip(0, 255).astype("uint8")
        Image.fromarray(arr).save(out_path)
        return str(out_path)

    @torch.no_grad()
    def _probe_with(self, model, step):
        from safetensors.torch import load_file

        from .likeness import probe_body_size

        cfg = self.cfg
        model.eval()
        lora_mod.set_extras_enabled(model, True)
        neg = load_file(str(text_cache_path(cfg, cfg.sample_negative)))["embeds"]
        neg_e, neg_m = loading.pad_embeds([neg])
        out_dir = self.job_dir / "probes"
        out_dir.mkdir(exist_ok=True)
        face_scores, body_scores, images = [], [], []
        try:
            for kind, prompt in (("face", cfg.probe_face_prompt), ("body", cfg.probe_body_prompt)):
                if not prompt:
                    continue
                if kind == "face" and self.face_scorer is None:
                    continue
                if kind == "body" and self.body_scorer is None:
                    continue
                w, h = (cfg.probe_size, cfg.probe_size) if kind == "face" else probe_body_size(cfg.probe_size)
                for s in range(max(1, cfg.probe_seeds)):
                    seed = cfg.probe_seed + s
                    p = self._render(model, prompt, w, h, cfg.probe_steps, cfg.probe_cfg, seed,
                                     out_dir / f"step{step:06d}_{kind}_{s}.png", neg_e, neg_m,
                                     f"probe {kind} seed {s + 1}/{cfg.probe_seeds} at step {step}")
                    images.append(p)
                    if kind == "face":
                        sc = self.face_scorer.score(p, step, 0)
                        if sc.percent is not None:
                            face_scores.append(sc.percent)
                    else:
                        pct, _ = self.body_scorer.score(p)
                        if pct is not None:
                            body_scores.append(pct)
        finally:
            lora_mod.set_extras_enabled(model, False)
        face = sum(face_scores) / len(face_scores) if face_scores else None
        body = sum(body_scores) / len(body_scores) if body_scores else None
        return face, body, images

    def run_probe(self, model, step, optimizer, scheduler, injected) -> bool:
        """Render + score probes, update the controller, apply its actions. Returns True to stop training."""
        from .likeness import append_record, apply_lr_ratio, curriculum_weights, timestep_logit_shift

        cfg = self.cfg
        t0 = time.time()
        face, body, images = self._with_sampling_model(model, lambda m: self._probe_with(m, step), step)
        rec, actions = self.controller.update(step, face, body, images)
        append_record(self.job_dir, rec)
        fmt = lambda v: "n/a" if v is None else f"{v:.1f}%"  # noqa: E731
        log.info("Probe step %d: face %s body %s combined %s | ema face %s body %s combined %s | best %s @%s "
                 "| lr x%.2f | face bias %+.2f (%.1fs)", step, fmt(face), fmt(body), fmt(rec.combined),
                 fmt(rec.face_ema), fmt(rec.body_ema), fmt(rec.combined_ema), fmt(rec.best), rec.best_step,
                 rec.lr_factor, rec.face_bias, time.time() - t0)
        if "step_up" in actions:
            log.info("Resolution step-up: stage %d stalled for %d probe(s) (stage best %s)", rec.stage + 1,
                     rec.stage_since_improve, fmt(rec.stage_best))
            self._enter_stage(actions["step_up"], step)
        if actions.get("save_best"):
            path = self.job_dir / f"{cfg.job_name}_best.safetensors"
            lora_mod.save_lora(injected, path, metadata={"ss_steps": str(step), "likeness_best": f"{rec.best:.2f}",
                                                         "modelspec.architecture": "moss-image"},
                               dtype=cfg.lora_save_dtype)
            log.info("New best likeness %.1f%% at step %d -> %s", rec.best, step, path)
            if str(path) not in self.status.data["saved"]:
                self.status.data["saved"].append(str(path))
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
    def _sample_with(self, model, step):
        from safetensors.torch import load_file

        cfg = self.cfg
        model.eval()
        lora_mod.set_extras_enabled(model, True)
        self.status.update(stage=f"sampling (step {step})")
        neg = load_file(str(text_cache_path(cfg, cfg.sample_negative)))["embeds"]
        neg_e, neg_m = loading.pad_embeds([neg])
        out_dir = self.job_dir / "samples"
        out_dir.mkdir(exist_ok=True)
        paths = []
        try:
            n_prompts = len(cfg.sample_prompts)
            for i, prompt in enumerate(cfg.sample_prompts):
                seed = cfg.sample_seed + (i if cfg.sample_walk_seed else 0)
                log.info("Sampling %d/%d (seed %d, %d steps): %s", i + 1, n_prompts, seed, cfg.sample_steps,
                         prompt[:80])
                stage = f"sampling {i + 1}/{n_prompts} at step {step}"
                self.status.update(stage=stage + " | step 0/%d" % cfg.sample_steps)

                def on_progress(k, n, _stage=stage):
                    self.status.update(stage=f"{_stage} | step {k}/{n}")

                emb = load_file(str(text_cache_path(cfg, prompt)))["embeds"]
                pos_e, pos_m = loading.pad_embeds([emb])
                lat = sample(model, pos_e, pos_m, neg_e, neg_m, width=cfg.sample_width, height=cfg.sample_height,
                             steps=cfg.sample_steps, cfg=cfg.sample_cfg, shift=cfg.sample_shift,
                             seed=seed, skip_layer=cfg.sample_skip_layer, device=self.device, progress=on_progress)
                img = loading.decode_latents(self.vae, lat)[0]
                arr = ((img.permute(1, 2, 0).float().cpu().numpy() + 1) * 127.5).round().clip(0, 255).astype("uint8")
                p = out_dir / f"step{step:06d}_{i}.png"
                Image.fromarray(arr).save(p)
                paths.append(str(p))
                log.info("Sample saved %s", p)
        finally:
            lora_mod.set_extras_enabled(model, False)
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
