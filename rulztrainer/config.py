"""Training configuration (dataclass + YAML round trip)."""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from pathlib import Path

import yaml

MODEL_PRESETS = {
    "MOSS-Image-Base": "OpenMOSS-Team/MOSS-Image-Base",
    "MOSS-Image-SFT": "OpenMOSS-Team/MOSS-Image-SFT",
    "FIBO-2.0-dev": "briaai/Fibo-2.0-dev",
    "FIBO-2.0-edit-turbo": "briaai/Fibo-2.0-edit-turbo",
    "Krea-2-Raw": "krea/Krea-2-Raw",
    "Krea-2-Turbo": "krea/Krea-2-Turbo",
}
ARCH_NAMES = ("moss_image", "fibo2", "krea2")
# LoRA target groups each family trains when lora_targets is empty. Krea 2 adds the attention output gate and its
# text-fusion stack, which Fizgig measured to be under-trained by the attention + feed-forward recipe (scaling a
# finished LoRA's text-fusion blocks x3 lifted likeness 2-7 points on several Krea 2 LoRAs).
_CORE_TARGETS = ("q", "k", "v", "o", "ffn_up_gate", "ffn_down")
ARCH_DEFAULT_TARGETS = {"moss_image": _CORE_TARGETS, "fibo2": _CORE_TARGETS,
                        "krea2": _CORE_TARGETS + ("attn_gate", "text_fusion")}
# (sample steps, sample CFG) per family, in this trainer's guidance convention (uncond + cfg * (cond - uncond), 1 = off):
# MOSS-Image's reference CFG 3; Bria's FIBO 2.0 pipeline, 30 steps at 5; Krea's pipeline, 28 steps at its
# guidance_scale 4.5, which computes cond + g * (cond - uncond): cfg 5.5 here
ARCH_SAMPLE_DEFAULTS = {"moss_image": (20, 3.0), "fibo2": (30, 5.0), "krea2": (28, 5.5)}


# Named presets: values applied over a config (UI toggle, --preset on the CLI, or `preset:` in YAML).
PRESETS: dict[str, dict] = {
    "identity": {  # measured recipe for character LoRAs (see README "Identity training")
        "region_weights": True, "face_loss_weight": 3.0, "body_loss_weight": 1.5, "background_loss_weight": 0.3,
        "face_margin": 0.25, "region_feather_cells": 1, "region_prior_weight": 0.0,
        "lora_plus_ratio": 4.0, "optimizer": "adamw8bit", "learning_rate": 1e-4,
        "timestep_sampling": "logit_normal", "sigma_min": 0.0, "sigma_max": 1.0,
        "crop_augment": "face+bust", "crop_augment_min_px": 0,
        "body_backend": "mediapipe+sam3",
        "batch_size": 1, "gradient_accumulation": 1, "lora_rank": 16, "lora_alpha": 16.0,
        "probe_every": 25, "probe_seeds": 2, "probe_at_start": True, "probe_steps": 8, "probe_cfg": 1.0,
        "likeness_track_best": True, "likeness_patience": 5, "likeness_lr_control": False,
        "likeness_curriculum": True, "likeness_curriculum_strength": 0.5,
        "loss_watch": True, "per_image_lr": True,
    },
}
PRESET_PROBE_PROMPTS = {
    "identity": ("close-up portrait photo of {trigger}, neutral background, soft light",
                 "full body photo of {trigger} standing straight facing the camera, arms at sides, plain background, "
                 "whole body visible head to feet"),
}


MEGAPIXEL_CHOICES = [round(0.25 * i, 2) for i in range(1, 17)]  # 0.25 .. 4.0 MP


def mp_to_side(mp: float) -> int:
    """Megapixels -> square-equivalent side in pixels (1 MP = 1024x1024)."""
    return int(round(math.sqrt(float(mp) * 1024 * 1024)))


def side_to_mp(side: int) -> float:
    return round(side * side / (1024 * 1024), 2)


def apply_preset(data: dict, name: str) -> dict:
    """Return a copy of a config dict with the preset applied. Probe prompts are filled in only when empty."""
    if name not in PRESETS:
        raise ValueError(f"Unknown preset {name!r}; available: {sorted(PRESETS)}")
    out = dict(data)
    out.update(PRESETS[name])
    face_p, body_p = PRESET_PROBE_PROMPTS.get(name, ("", ""))
    subject = out.get("trigger_word") or "the character"
    if not out.get("probe_face_prompt"):
        out["probe_face_prompt"] = face_p.format(trigger=subject)
    if not out.get("probe_body_prompt"):
        out["probe_body_prompt"] = body_p.format(trigger=subject)
    out["preset"] = name
    return out


@dataclass
class TrainConfig:
    # --- job ---
    preset: str = ""  # name of a preset that was applied (identity); informational, see apply_preset
    job_name: str = "my_lora"
    output_dir: str = "output"
    seed: int = 42

    # --- model ---
    model_arch: str = "auto"  # auto | moss_image | fibo2 | krea2 (auto: the model's transformer config, or the path)
    model_source: str = "hf"  # hf: repo id / diffusers dir | files: single ComfyUI-style safetensors files
    model_path: str = "OpenMOSS-Team/MOSS-Image-Base"  # HF repo id or local dir with transformer/ text_encoder/ vae/
    models_dir: str = "models"  # where HF repos are downloaded
    transformer_file: str = ""  # model_source=files: ComfyUI diffusion_models/*.safetensors (bf16, fp8, int8, nvfp4)
    text_encoder_file: str = ""  # model_source=files: ComfyUI text_encoders/*.safetensors (Qwen3.5-9B)
    vae_file: str = ""  # model_source=files: ComfyUI vae/*.safetensors (Flux2 VAE)
    tokenizer_path: str = ""  # model_source=files: dir with tokenizer_config.json; empty = auto-download
    device: str = "cuda:0"
    multi_gpu: str = "off"  # off | data_parallel (model copy per GPU, faster) | model_split (blocks across GPUs, memory only)
    devices: str = "cuda:0,cuda:1"  # GPUs for multi_gpu; the first is the primary
    split_plan: str = ""  # model_split: manual block plan e.g. "cuda:0=0-25;cuda:1=26-41"; empty = auto-balance
    parallel_jobs: bool = True  # multi_gpu: split samples, probes, latent/text caching and SAM 3 masks across the GPUs
    dp_balance: str = "auto"  # data_parallel batches per GPU and step by measured speed: off (lockstep) | auto (same
    #                           images per step, split by speed) | grow (faster GPUs fill the slowest one's time)
    text_encoder_device: str = "auto"  # "auto" splits across GPUs/CPU; or "cuda:0", "cuda:1", "cpu"
    transformer_quant: str = "auto"  # auto (keep file format) | none | int8 | int8_convrot | nvfp4
    text_encoder_quant: str = "auto"  # auto | none | int8 | int8_convrot | nvfp4
    gradient_checkpointing: bool = True
    checkpoint_keep_blocks: int = -1  # with checkpointing: blocks that keep their activations instead (no recompute);
    #                                   -1 = as many as the free GPU memory allows (measured), 0 = none, N = the last N
    int8_matmul: bool = True  # int8/int8_convrot layers: W8A8 (int8 activations x int8 weight via torch._int_mm)
    int8_backward: str = "bf16"  # bf16 (exact: grad = grad_out @ W in bf16) | int8 (also quantizes grad_out)
    nvfp4_matmul: bool = True  # nvfp4 layers on Blackwell GPUs (RTX 50): activations to NVFP4, FP4 tensor-core matmul
    #                            (W4A4, backward exact in bf16); off, or on older GPUs: weight dequantized to bf16
    nvfp4_backward: str = "bf16"  # native nvfp4 layers' grad_x: bf16 (exact) | fp4 (FP4 tensor cores, faster, approx.)
    attention_backend: str = "auto"  # DiT attention kernel: auto (cuDNN on RTX 50 / Blackwell, else PyTorch's pick) |
    #   torch (PyTorch picks) | cudnn | flash | efficient | math

    # --- dataset ---
    dataset_dir: str = "datasets/my_dataset"
    resolution: int = 1024  # target total pixels = resolution^2, aspect preserved, buckets at multiples of 16
    resolutions: list[float] = field(default_factory=list)  # megapixels to train at (0.25..4 in 0.25 steps); [] = resolution
    resolution_step_up: bool = False  # train the lowest resolution first and step up when the probe score stalls
    step_up_patience: int = 1  # probes without improvement before moving to the next resolution
    step_up_stop_patience: int = 1  # probes without improvement at the highest resolution before stopping
    enable_buckets: bool = True
    min_bucket_side: int = 512
    max_bucket_side: int = 2048
    flip_augment: bool = False
    caption_ext: str = ".txt"
    trigger_word: str = ""
    caption_dropout: float = 0.05
    caption_dropout_keep_trigger: bool = True  # a dropped caption keeps the trigger word (else the empty prompt)
    max_token_length: int = 1024
    cache_dir: str = ""  # defaults to <dataset_dir>/.rulz_cache
    recache: bool = False

    # --- lora ---
    lora_rank: int = 16
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    lora_targets: list[str] = field(default_factory=list)  # target groups; [] = the family's (ARCH_DEFAULT_TARGETS)
    lora_blocks: str = ""  # DiT blocks to train, e.g. "0-41" (all), "21-41", "10-31,40", "even"; empty = all
    train_text_embedder: bool = False  # also LoRA the 2-block text projection inside the DiT
    resume_from: str = ""  # path to a previously exported LoRA .safetensors (weights only)
    lora_save_dtype: str = "bf16"  # bf16 | fp16 | fp32 for the exported LoRA files
    lora_ema: float = 0.98  # weight average (EMA) decay per image for saves, samples and probes; 0 = off (see ema.py)
    save_state: bool = False  # also save optimizer/scheduler/RNG state (resumable); final state always saved
    save_state_every: int = 0  # steps between state saves; 0 = same as save_every
    keep_states: int = 3  # rolling number of state folders to keep; 0 = keep all
    resume_state: str = ""  # path to a state_XXXXXX directory to resume training from
    compile_model: bool | str = "auto"  # torch.compile the DiT blocks: true | false | auto (runs of 100+ steps on a GPU
    #                                     with triton; measured 26% faster per step on MOSS-Image, 12% on Krea 2)

    # --- optimization ---
    steps: int = 2000
    batch_size: int = 1
    gradient_accumulation: int = 1
    learning_rate: float = 1e-4
    optimizer: str = "adamw"  # adamw | adamw8bit | adafactor | automagic3
    weight_decay: float = 0.01
    automagic_min_lr: float = 1e-8  # automagic3: hard floor for the adapted lr
    automagic_max_lr: float = 1e-3  # automagic3: hard ceiling for the adapted lr
    automagic_polarity_history: int = 8  # automagic3: sign-history window (2-64)
    automagic_clip_threshold: float = 1.0  # automagic3: update trust region
    automagic_groups: str = "single"  # automagic3: single (one adaptive lr) | by_type (text / attention / feed-forward /
    #                                   other layers each adapt their own; Fizgig's setup for Krea 2's text fusion)
    lr_scheduler: str = "constant"  # constant | cosine | linear
    warmup_steps: int = 50
    max_grad_norm: float = 1.0
    timestep_sampling: str = "logit_normal"  # logit_normal | sigmoid | uniform
    logit_mean: float = 0.0
    logit_std: float = 1.0
    sigmoid_scale: float = 1.0  # sigmoid mode: sigma = sigmoid(randn * sigmoid_scale)
    timestep_shift: float = 1.0  # sigma shift applied to sampled sigmas (training_shift is 1.0 upstream)
    loss_weighting: str = "none"  # none | min_snr (sigma-based) | sigma_sqrt

    # --- saving / sampling ---
    save_every: int = 250
    sample_every: int = 250
    sample_prompts: list[str] = field(default_factory=list)
    sample_width: int = 1024
    sample_height: int = 1024
    sample_steps: int = 20
    sample_cfg: float = 3.0
    sample_shift: float = 5.0
    sample_seed: int = 42
    sample_walk_seed: bool = True  # seed + i for the i-th sample prompt; off = same seed for every prompt
    sample_before_training: bool = False  # generate samples at step 0 before the first training step
    sample_negative: str = ""
    sample_skip_layer: int = 22  # negative-branch skip layer; -1 disables
    turbo_lora: str = ""  # optional ComfyUI-format LoRA (e.g. an 8-step distill) applied only while sampling
    turbo_lora_strength: float = 1.0
    sample_lora: str = ""  # optional second ComfyUI-format LoRA applied only while sampling (with the turbo LoRA)
    sample_lora_strength: float = 1.0
    # --- face similarity (character LoRAs) ---
    face_refs: list[str] = field(default_factory=list)  # dataset images whose largest face is the reference identity
    face_provider: str = "cpu"  # cpu | cuda (onnxruntime provider for InsightFace)
    face_det_size: int = 640
    body_refs: list[str] = field(default_factory=list)  # dataset images (full body, neutral pose) for proportions
    body_score: str = "appearance"  # appearance (DINOv2 of the person crop, best match against the references) |
    #   proportions (skeleton ratios from body_backend, + SAM 3 silhouette widths with mediapipe+sam3)
    body_backend: str = "mediapipe"  # mediapipe (skeleton ratios, region masks) | mediapipe+sam3 (adds SAM 3 masks)
    sam3_model: str = "facebook/sam3"  # HF repo id or local transformers-format directory
    body_device: str = "cuda:0"  # where SAM 3 runs while scoring (moved back to CPU afterwards)
    face_weight: float = 0.7  # weights for the combined score
    body_weight: float = 0.3
    # --- per-image loss watch ---
    loss_watch: bool = False  # track every image's timestep-normalised loss and give epoch verdicts
    per_image_lr: bool = False  # act on verdicts: stuck x0.5/0.25/0.125, exhausted x0.6 (not with automagic3)
    loss_watch_buckets: int = 40  # sigma buckets for loss normalisation
    loss_watch_beta: float = 0.9  # per-image residual EMA
    loss_watch_high: float = 1.3  # residual above median x this = hard
    loss_watch_low: float = 0.7  # residual below median x this = easy / learned
    loss_watch_improve_tol: float = 0.03  # relative epoch-to-epoch drop that counts as improving
    loss_watch_healthy_mult: float = 1.0  # lr multiplier for the healthy cohort (Fizgig uses 1.1)
    loss_watch_exclude_after: int = 0  # epochs at the lowest throttle before an image is excluded; 0 = never
    # --- identity training: region-weighted loss ---
    region_weights: bool = False  # weight the loss by region: face > person > background (needs insightface)
    face_loss_weight: float = 3.0
    body_loss_weight: float = 1.5
    background_loss_weight: float = 0.3
    face_margin: float = 0.25  # face box dilation as a fraction of its size
    region_feather_cells: int = 1  # 3x3 averaging passes on the cell map (0 = hard edges)
    region_prior_weight: float = 0.0  # background cells regularised toward the base model's prediction (0 = off)
    # --- identity training: timesteps ---
    content_power: float = 3.0  # timestep_sampling=content/style: sigma = u^p (content) or 1 - u^p (style)
    sigma_min: float = 0.0  # training sigma range (rescaled after timestep_shift; 0..1)
    sigma_max: float = 1.0
    # --- identity training: LoRA+ ---
    lora_plus_ratio: float = 0.0  # lr of lora_up = lr x ratio (0/1 = off, 16 = LoRA+ paper); not with automagic3
    network_type: str = "lora"  # lora | lokr (LyCORIS Kronecker factors: w1 up to lokr_factor square, w2 the rest)
    lokr_factor: int = 8  # LoKr: w1 is [a, b] with a, b the largest divisors of out / in <= this (8: Fizgig's default)
    # --- identity training: automatic crops ---
    crop_augment: str = "none"  # none | face | face+bust: add face / bust crops of each image as extra items
    crop_augment_min_px: int = 0  # min crop side in source px; 0 = the training resolution (never upscale crops)
    crop_face_scale: float = 2.5  # face crop side = scale x max(face w, h)
    crop_bust_scale: float = 5.0  # bust crop side = scale x max(face w, h), centred below the face
    crop_skip_face_frac: float = 0.15  # already a close-up: no face crop
    crop_skip_cover: float = 0.70  # crop covering more of the source than this is redundant
    crop_face_caption: str = "close-up portrait photo of {caption}"
    crop_bust_caption: str = "head and shoulders photo of {caption}"
    crop_augment_weight: float = 1.0  # sampling weight of crop items
    # --- identity training: direct identity loss (experimental; needs face_refs) ---
    id_loss_weight: float = 0.0  # 0 = off; applied weight is capped at 0.5. Loss = w * (1 - cos(face emb, ref))
    id_loss_sigma_max: float = 0.35  # only low-noise samples, where the predicted x0 shows a recognisable face
    id_loss_every: int = 1  # use every k-th eligible sample
    id_loss_backend: str = "arcface"  # arcface (= the probe metric, onnx2torch) | facenet (independent of the probe)
    id_loss_margin_cells: int = 2  # latent cells of context decoded around the face box
    id_loss_min_face_cells: int = 4  # skip faces smaller than this many latent cells (16 px each) on a side
    id_loss_random_ref: bool = True  # one random reference face per step (else the mean reference embedding)
    # --- likeness probes & controller (character LoRAs) ---
    probe_every: int = 0  # steps between probe renders; 0 = off (needs face_refs and/or body_refs)
    probe_at_start: bool = True  # also probe at step 0 for a baseline
    probe_face_prompt: str = ""  # headshot prompt; empty = no face probe
    probe_body_prompt: str = ""  # full-body prompt; empty = no body probe
    probe_seeds: int = 1  # seeds per probe (scores averaged)
    probe_seed: int = 1234
    probe_size: int = 512  # face probe is square; body probe is 2:3 portrait with the same pixel count
    probe_steps: int = 8
    probe_cfg: float = 1.0
    probe_ema: float = 0.5  # smoothing of probe scores (weight on the previous value)
    likeness_noise_band: float = 1.0  # percent points a change must exceed to count
    likeness_normalize: bool = True  # score face and body on one scale (0 = a stranger, 100 = the references) first
    likeness_track_best: bool = True  # keep <job>_best.safetensors at the best smoothed score
    likeness_patience: int = 0  # stop after N probes without improvement; 0 = never
    likeness_lr_control: bool = False  # bounded lr multiplier driven by the probe trend
    likeness_lr_min: float = 0.5
    likeness_lr_max: float = 2.0
    likeness_lr_step: float = 0.1  # relative change per decision
    likeness_curriculum: bool = False  # face/body dataset weighting + timestep bias from the score gap
    likeness_curriculum_strength: float = 1.0
    # configuration race (see race.py): the configurations train in turn, are probed, and the lowest is dropped
    race: bool = False
    race_configs: list[dict] = field(default_factory=list)  # [{"name": ..., <setting>: <value>, ...}], 2 or more
    race_warmup_probes: int = 0  # probe rounds before the first configuration is dropped
    sample_model_path: str = ""  # optional different diffusion model for sampling only: .safetensors, diffusers dir or repo id
    sample_model_quant: str = "auto"  # auto | none | int8 | int8_convrot | nvfp4

    def training_sides(self) -> list[int]:
        """Training resolutions as square-equivalent side lengths (area = side^2), ascending and unique."""
        if not self.resolutions:
            return [int(self.resolution)]
        sides = sorted({mp_to_side(mp) for mp in self.resolutions})
        return sides

    def arch_name(self) -> str:
        """The architecture this config trains: model_arch when set, else guessed from the model path or files."""
        if self.model_arch and self.model_arch != "auto":
            return self.model_arch
        hint = " ".join([self.model_path or "", self.transformer_file or ""]).lower()
        return "fibo2" if "fibo" in hint else "krea2" if "krea" in hint else "moss_image"

    def resolved_lora_targets(self) -> list[str]:
        """lora_targets, or the model family's default groups when it is empty."""
        return list(self.lora_targets) or list(ARCH_DEFAULT_TARGETS[self.arch_name()])

    def validate(self) -> None:
        if self.model_arch not in ("auto", *ARCH_NAMES):
            raise ValueError(f"model_arch must be auto or one of {ARCH_NAMES}, got {self.model_arch!r}")
        for mp in self.resolutions:
            if not (0.0 < float(mp) <= 16.0):
                raise ValueError(f"resolutions must be megapixel values in (0, 16], got {mp!r}")
        if self.id_loss_backend not in ("arcface", "facenet"):
            raise ValueError(f"id_loss_backend must be arcface or facenet, got {self.id_loss_backend!r}")
        if self.id_loss_every < 1:
            raise ValueError("id_loss_every must be at least 1")
        if not 0.0 <= self.lora_ema < 1.0:
            raise ValueError(f"lora_ema must be 0 (off) or a decay below 1, got {self.lora_ema!r}")
        if self.dp_balance not in ("off", "auto", "grow"):
            raise ValueError(f"dp_balance must be off, auto or grow, got {self.dp_balance!r}")
        if self.automagic_groups not in ("single", "by_type"):
            raise ValueError(f"automagic_groups must be single or by_type, got {self.automagic_groups!r}")
        if self.compile_model not in (True, False, "auto"):
            raise ValueError(f"compile_model must be true, false or auto, got {self.compile_model!r}")
        if self.race:
            from . import race

            race.validate(self)
        if self.resolution_step_up:
            if self.step_up_patience < 1 or self.step_up_stop_patience < 1:
                raise ValueError("step_up_patience and step_up_stop_patience must be at least 1")
            if self.probe_every <= 0 or not (self.face_refs or self.body_refs):
                raise ValueError("resolution_step_up needs likeness probes: set probe_every > 0 and face_refs "
                                 "and/or body_refs (the step-up decision is the probe likeness score)")

    def wants_regions(self) -> bool:
        """Region maps (face cells + landmarks) are needed by the region-weighted loss and the identity loss."""
        return bool(self.region_weights or self.id_loss_weight > 0)

    def resolved_cache_dir(self) -> Path:
        return Path(self.cache_dir) if self.cache_dir else Path(self.dataset_dir) / ".rulz_cache"

    def job_dir(self) -> Path:
        return Path(self.output_dir) / self.job_name

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TrainConfig":
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**data)

    @classmethod
    def load(cls, path: str | Path) -> "TrainConfig":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f) or {})

    @classmethod
    def load_report(cls, path: str | Path) -> tuple["TrainConfig", list[str], list[str]]:
        """Lenient load for configs saved by an older version.

        Returns (config, missing, unknown): `missing` are options added since the file was saved (they take their
        defaults), `unknown` are keys this version no longer has (ignored instead of refusing the file)."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{path} is not a RulzTrainer config (expected a mapping of options)")
        known = [f.name for f in dataclasses.fields(cls)]
        unknown = sorted(k for k in data if k not in known)
        missing = [k for k in known if k not in data]
        return cls(**{k: v for k, v in data.items() if k in known}), missing, unknown

    @staticmethod
    def append_missing(path: str | Path, cfg: "TrainConfig", missing: list[str]) -> None:
        """Add the missing options (with their values in `cfg`) to the end of an existing config file.

        Appends instead of rewriting, so the file's own values, order and comments stay exactly as they were."""
        if not missing:
            return
        values = cfg.to_dict()
        block = yaml.safe_dump({k: values[k] for k in missing}, sort_keys=False, allow_unicode=True)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        sep = "" if not text or text.endswith("\n") else "\n"
        with open(path, "a", encoding="utf-8", newline="\n") as f:
            f.write(f"{sep}# Options added in a newer RulzTrainer version (defaults; edit as needed):\n{block}")

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False, allow_unicode=True)
