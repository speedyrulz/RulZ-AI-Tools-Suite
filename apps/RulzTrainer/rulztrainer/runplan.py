"""Checks before a run starts, shared by the WebUI and the vast.ai agent (scripts/vast/rulz_agent.py): what the run's
adapter holds in GPU memory (for the block swap plan), and whether its job name is free (one name, one run).

Memory: the plan (webui.block_swap_plan) adds a flat allowance for a step's memory to the frozen model's size. That
allowance was set with the default run in mind: a rank-16 LoRA on the family's default layers, AdamW (fp32 state) and
the weight average on. A bigger adapter holds more: Fizgig counted LoKr factor 8 on Krea 2 at about 200M parameters, ~3 GB
with optimizer state and its average, against ~0.5 GB for its rank-16 LoRA, and planning without it put a 16 GB card
past its memory. adapter_extra_gb() is the part beyond what the allowance already covers.

These are estimates from counting tensors, not measurements:
- The wrapped Linears' shapes come from the family's transformer class built with its default arguments (the
  reference checkpoint's config) on the meta device, so no weights are read or allocated, and the family's own
  Arch.lora_module_names picks the layers exactly as the trainer does. A checkpoint with another config than the
  reference one (none is known to be trained here) would be counted at the reference shapes.
- The adapter is fp32 (lora.LoRALinear): 4 bytes of weight and 4 of gradient per parameter, plus the optimizer's
  per-parameter state (OPTIMIZER_STATE_BYTES, read from each optimizer's code) and 4 for the weight average's fp32
  shadow on the same device (ema.LoraEMA) when lora_ema is on.
- Activations, the frozen weights and the DoRA norm's working memory are not counted here.
Anything that cannot be counted (an unknown family, a model class that fails to build) gives 0, so the plan stays as
it was.

Job names: two runs sharing a job name write into one folder, the later one's checkpoints, states, samples and probe
records mixed with (or written over) the earlier one's. A new run is refused a folder that already holds such output
(holds_output) and told the next free name (next_free_name); carrying the same run on from one of its own states is
not a new run (carries_on), and a folder whose run saved nothing (only its config, logs and step-0 pictures) is free."""
from __future__ import annotations

import functools
import importlib
import os
import re
from pathlib import Path

# Klein 4B is the 9B's class at its own size (3072 wide, 24 heads, 5 double + 20 single blocks, 7680 text features)
KLEIN_4B_CONFIG = dict(context_in_dim=7680, hidden_size=3072, num_heads=24, depth=5, depth_single_blocks=20)
# The transformer class of each family: (module under rulztrainer, class[, constructor arguments]). The H3 image variants train the same
# transformer blocks as MiniMax H3 (their differences are the audio stream and the AdaLN tables, which hold no LoRA).
FAMILY_MODEL = {"moss_image": ("moss", "MOSSImageTransformer"), "fibo2": ("fibo", "FiboTransformer"),
                "krea2": ("krea2", "Krea2Transformer"), "qwen_image21": ("qwen21", "QwenImage21Transformer"),
                "flux3_klein": ("flux3", "Flux3KleinTransformer"), "flux2_klein": ("flux2", "Flux2KleinTransformer"),
                "flux2_klein4b": ("flux2", "Flux2KleinTransformer", KLEIN_4B_CONFIG),
                "ideogram4": ("ideogram4", "Ideogram4Transformer"), "minimax_h3": ("h3", "MiniMaxH3Transformer"),
                "p_h3_image": ("h3", "MiniMaxH3Transformer"), "minimax_h3_image": ("h3", "MiniMaxH3Transformer"),
                "pan2": ("pan2", "PAN2Transformer"), "sdxl": ("sdxl", "SdxlUNet"),
                "zimage": ("zimage", "ZImageTransformer")}

# Optimizer state per trainable fp32 parameter, in bytes, from the buffers each one keeps (trainer.build_optimizer):
#   adamw      torch AdamW: exp_avg + exp_avg_sq in fp32                                                    8
#   adamw8bit  bitsandbytes AdamW8bit: two 8-bit moments (its block scales, a few % more, are left out)     2
#   adafactor  transformers Adafactor without beta1: factored row + column second moments for a matrix,
#              a vanishing share of its elements                                                            0
#   automagic3 / automagic4 / keel / keel_fb / keel3 / keel3_fb: the sign history, polarity_history bits per element
#              (H / 8 bytes), plus factored row / column moments                                     history / 8
#   keel / keel_fb / keel3 / keel3_fb with keel_lookahead: plus an fp32 anchor copy of the weights                             +4
#   hull       fp32 momentum "m" per element (per-channel "v" left out)                                     4
#   muon / normuon: fp32 momentum buffer (normuon's per-row second moment left out)                         4
OPTIMIZER_STATE_BYTES = {"adamw": 8.0, "adamw8bit": 2.0, "adafactor": 0.0, "hull": 4.0, "muon": 4.0, "normuon": 4.0}
_AUTOMAGIC = ("automagic3", "automagic4", "keel", "keel_fb", "keel3", "keel3_fb")
WEIGHT_BYTES, GRAD_BYTES, EMA_BYTES = 4.0, 4.0, 4.0  # fp32 adapter weight, its gradient, the average's fp32 shadow

# The run the flat step allowance (webui.STEP_HEADROOM_GB) was set for: TrainConfig's defaults
BASE_RANK, BASE_OPTIMIZER, BASE_EMA = 16, "adamw", True


def optimizer_state_bytes(optimizer: str, polarity_history: int = 8, lookahead: bool = False) -> float:
    """Bytes of optimizer state per trainable parameter (OPTIMIZER_STATE_BYTES; unknown names count as AdamW's 8)."""
    name = str(optimizer or "adamw").lower()
    if name in _AUTOMAGIC:
        state = max(1, int(polarity_history or 8)) / 8.0
        return state + (4.0 if lookahead and name in ("keel", "keel_fb", "keel3", "keel3_fb") else 0.0)
    return OPTIMIZER_STATE_BYTES.get(name, 8.0)


def bytes_per_param(optimizer: str, ema: bool, polarity_history: int = 8, lookahead: bool = False) -> float:
    """Bytes the whole run holds per trainable parameter: weight + gradient + optimizer state (+ the average)."""
    return WEIGHT_BYTES + GRAD_BYTES + optimizer_state_bytes(optimizer, polarity_history, lookahead) + \
        (EMA_BYTES if ema else 0.0)


def adapter_params(shapes, kind: str = "lora", rank: int = 16, factor: int = 8, block: int = 32,
                   dora: bool = False) -> int:
    """Trainable parameters of an adapter over Linears of (out, in) shapes, as lora.LoRALinear.adapter_shapes builds
    them: LoRA rank * (in + out); LoKr the two Kronecker factors a*b + c*d (a, b the largest divisors of out / in up
    to the factor); OFT out * block size (the block size the largest divisor of out up to `block`); DoRA adds a
    magnitude per input on LoRA and LoKr."""
    from .lora import factorization

    total = 0
    for out_f, in_f in shapes:
        if kind == "lokr":
            a, c = factorization(out_f, factor)
            b, d = factorization(in_f, factor)
            total += a * b + c * d
        elif kind == "oft":
            size, _ = factorization(out_f, block)
            total += out_f * size
        else:
            total += int(rank) * (in_f + out_f)
        if dora and kind in ("lora", "lokr"):
            total += in_f
    return total


@functools.lru_cache(maxsize=64)
def target_shapes(family: str, targets: tuple[str, ...], blocks_spec: str = "",
                  text_embedder: bool = False) -> tuple[tuple[int, int], ...]:
    """(out, in) of every Linear the trainer would wrap for these targets / lora_blocks, from the family's model
    class built on the meta device (no weights); () when it cannot be built."""
    try:
        import torch

        from .archs import _ARCHS
        from .lora import parse_blocks

        module, cls, *kwargs = FAMILY_MODEL[family]
        model_cls = getattr(importlib.import_module(f".{module}.model", __package__), cls)
        with torch.device("meta"):
            model = model_cls(**(kwargs[0] if kwargs else {}))
        arch = _ARCHS[family]()
        blocks = parse_blocks(blocks_spec, len(model.layers))
        names = arch.lora_module_names(model, list(targets), blocks, text_embedder)
        mods = dict(model.named_modules())
        return tuple((int(mods[n].out_features), int(mods[n].in_features)) for n in names)
    except Exception:  # an unknown family, a class that needs arguments: no estimate
        return ()


def adapter_gb(cfg, overrides: dict | None = None) -> float:
    """GB the run's adapter holds on the GPU (weights, gradients, optimizer state, the average); 0 when unknown or for
    a fine-tune (no adapter). overrides: TrainConfig fields to read instead of cfg's (the wizard's picks). A race
    counts its largest configuration: one configuration is on the GPU at a time (race.py parks the others on the
    CPU), and a row may change the adapter, its size and the optimizer; a row's own value beats an override."""
    o = overrides or {}
    if o.get("race", getattr(cfg, "race", False)) and getattr(cfg, "race_configs", None):
        try:
            from . import race

            rows = race.variant_configs(cfg)
        except Exception:  # rows the race itself will refuse: count the form's adapter
            rows = []
        if rows:
            return max(_adapter_gb(row_cfg, {k: v for k, v in o.items() if k not in over})
                       for _, row_cfg, over in rows)
    return _adapter_gb(cfg, o)


def _adapter_gb(cfg, o: dict) -> float:
    """adapter_gb for one configuration."""

    def get(key, default=None):
        return o[key] if key in o else getattr(cfg, key, default)

    if get("finetune", False):
        return 0.0
    try:
        from .config import ARCH_DEFAULT_TARGETS

        fam = cfg.arch_name()
        targets = tuple(get("lora_targets") or ARCH_DEFAULT_TARGETS[fam])
    except Exception:
        return 0.0
    shapes = target_shapes(fam, targets, str(get("lora_blocks", "") or ""), bool(get("train_text_embedder", False)))
    if not shapes:
        return 0.0
    kind = str(get("network_type", "lora") or "lora")
    n = adapter_params(shapes, kind, int(get("lora_rank", 16) or 16), int(get("lokr_factor", 8) or 8),
                       int(get("oft_block_size", 32) or 32), bool(get("dora", False)))
    opt = str(get("optimizer", "adamw") or "adamw").lower()
    if get("isolation", False) and float(get("dop_weight", 0) or 0) > 0 and opt in _AUTOMAGIC:
        opt = "adamw"  # (config.effective_optimizer: isolation trains an Automagic pick with AdamW)
    per = bytes_per_param(opt, float(get("lora_ema", 0) or 0) > 0,
                          int(get("automagic_polarity_history", 8) or 8), bool(get("keel_lookahead", False)))
    return n * per / 1e9


def base_adapter_gb(family: str) -> float:
    """GB the default run's adapter holds on this family (what the flat allowance already covers); 0 when unknown."""
    try:
        from .config import ARCH_DEFAULT_TARGETS

        shapes = target_shapes(family, tuple(ARCH_DEFAULT_TARGETS[family]))
    except Exception:
        return 0.0
    return adapter_params(shapes, "lora", BASE_RANK) * bytes_per_param(BASE_OPTIMIZER, BASE_EMA) / 1e9


def adapter_extra_gb(cfg, overrides: dict | None = None) -> float:
    """GB this run's adapter holds beyond what the default run's does (never below 0); 0 when it cannot be counted."""
    try:
        run = adapter_gb(cfg, overrides)
        if run <= 0:
            return 0.0
        return max(0.0, run - base_adapter_gb(cfg.arch_name()))
    except Exception:
        return 0.0


# ---------------------------------------------------------------- one name, one run

# What a run writes that a later run in the same folder would mix with or write over (trainer.py): its LoRA files
# (checkpoints, final, best, race and soup files, fine-tune deltas: *.safetensors at the top), its training states
# (state_XXXXXX folders), and its sample, probe and training-map pictures. Everything else (the config, train.log,
# console.log, status.json, a STOP file, per-image loss logs, the probes' records) is what a run that never got
# anywhere leaves behind, and the next run writes it afresh or adds to it. So are the pictures of step 0 (BASELINE in
# their names): the sample and the likeness probe a run makes before its first training step (sample_before_training,
# probe_at_start, both on by default). A run stopped in its first minutes to fix a setting has those and nothing
# else; the next run under the name writes the same files again, and likeness_export keeps the later of two step-0
# records.
PICTURE_DIRS = ("samples", "probes", "training_maps")
PICTURE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".mp4", ".webm")
BASELINE = "step000000"


def check_state_paths(cfg, must_exist: bool = True) -> None:
    """Reject misplaced or incomplete training states before a run is started. must_exist=False (validate(): a queued
    run may resume from a state a run ahead of it writes) lets a resume folder that does not exist yet pass."""
    def is_state(path):
        return path.is_dir() and (path / "train_state.pt").is_file() and (path / "lora.safetensors").is_file()

    output = cfg.job_dir()
    if is_state(output):
        raise ValueError(f"Output folder {output} is a saved training state; put that folder in resume_state and "
                         "choose an output directory and job name for the run's parent folder.")
    if not cfg.resume_state:
        return
    state = Path(cfg.resume_state)
    if not must_exist and not state.exists():
        return
    if state.is_file():
        state = state.parent
    if is_state(state):
        return
    children = [p for p in state.glob("state_*") if is_state(p)] if state.is_dir() else []
    if children:
        newest = max(children, key=lambda p: p.name)  # state_<step, 6 digits>: the highest step
        raise ValueError(f"Resume folder {state} contains saved states; select the newest state folder: {newest}")
    raise ValueError(f"Resume folder {state} is not a saved training state (needs train_state.pt and lora.safetensors).")


def holds_output(folder) -> bool:
    """Does this job folder hold an earlier run's output (see PICTURE_DIRS above)? False for a missing folder or one
    with only the leftovers of a run that never trained far enough to save or show anything (its config, logs and
    step-0 pictures)."""
    folder = Path(folder)
    try:
        entries = list(folder.iterdir())
    except OSError:  # no such folder (or not one): nothing to mix with
        return False
    for p in entries:
        name = p.name
        if p.is_dir():
            if name.startswith("state_"):
                return True
            if name in PICTURE_DIRS and any(q.suffix.lower() in PICTURE_SUFFIXES and BASELINE not in q.name
                                            for q in p.rglob("*") if q.is_file()):
                return True
        elif name.lower().endswith(".safetensors"):
            return True
    return False


def carries_on(job_dir, resume_state: str) -> bool:
    """Does a run with this resume_state carry on a run of this job folder (the state folder is inside it)? A file
    inside the state folder names that folder, as it does for the trainer (Trainer.load_state)."""
    if not resume_state:
        return False
    try:
        state = Path(resume_state).resolve()
        if state.is_file():
            state = state.parent
        return state.parent == Path(job_dir).resolve()
    except (OSError, RuntimeError, ValueError):
        return False


def name_key(name: str) -> str:
    """A job name as its folder is told apart on this system: on Windows `Portrait` and `portrait` are one folder."""
    return os.path.normcase(str(name))


def name_in(name: str, names) -> bool:
    """Is this job name one of `names`, as folders go on this system (name_key)?"""
    return name_key(name) in {name_key(n) for n in names}


def next_free_name(name: str, busy, output_dir) -> str:
    """The next job name after `name` that no running or waiting run uses (busy) and whose folder under output_dir
    holds no output: name_2, name_3, ...; a name that already ends in _N counts on from N (run_7 -> run_8, keeping
    leading zeros: run_007 -> run_008)."""
    m = re.fullmatch(r"(.*)_(\d+)", name)
    stem, n, width = (m.group(1), int(m.group(2)), len(m.group(2))) if m else (name, 1, 1)
    busy = {name_key(b) for b in busy}
    while True:
        n += 1
        candidate = f"{stem}_{n:0{width}d}"
        if name_key(candidate) not in busy and not holds_output(Path(output_dir) / candidate):
            return candidate


def job_name_taken(job_name: str, output_dir, resume_state: str = "", busy=()) -> str | None:
    """None when a new run may use this job name; else the next free name (next_free_name). The name is taken when
    its folder holds an earlier run's output, unless the run carries that run on from one of its own states. (A name
    a running or waiting run uses is the caller's check: its message differs.)"""
    job_dir = Path(output_dir) / job_name
    if not holds_output(job_dir) or carries_on(job_dir, resume_state):
        return None
    return next_free_name(job_name, set(busy) | {job_name}, output_dir)


def name_taken_message(job_name: str, job_dir, free: str) -> str:
    """The refusal, in the words of the WebUI's other start checks."""
    return (f"Job name '{job_name}' already has an earlier run's files in {job_dir} (LoRAs, training states, samples "
            f"or probes), and a new run there would mix with them or write over them. Give this run another job name "
            f"('{free}' is free), or carry that run on from one of its state folders (Resume from training state). "
            f"A run that starts from one of its LoRA files, a seed soup and a probe noise check are runs of their "
            f"own: they need their own name too.")
