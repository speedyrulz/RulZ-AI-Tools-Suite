"""Post-hoc EMA: a weight average of one run's saved checkpoints, made after the run (musubi-tuner's
lora_post_hoc_ema.py).

`lora_ema` averages the weights while training, with a decay chosen before the run. This makes the same kind of
average afterwards from the step checkpoints the run kept, so the decay can be tried at several values on finished
files. It is a weighted seed soup (soup.cook) with exponential-moving-average weights over the checkpoints in step
order:

- beta (and beta2): a plain EMA over the files, oldest first: avg = beta x avg + (1 - beta) x next. With beta2 the
  decay slides in a straight line from beta to beta2 over the files (musubi-tuner's --beta / --beta2).
- sigma_rel: the power-function EMA of Karras et al. (EDM2, 2024), musubi-tuner's --sigma_rel. Its averaging profile
  weighs training step t by t^gamma, gamma from the relative width sigma_rel (0.05 to 0.2 is the paper's range; smaller
  leans harder on the last steps). Each checkpoint stands for the steps since the one before it, so uneven save
  intervals are weighed by the steps they cover.

Unlike musubi-tuner, which averages the up and down factors as they are, this averages each layer's weight change
(up @ down x scale, as soup.py does): the factors of two checkpoints can drift apart while their product stays close,
and an average of factors is not the average of the changes. The result is saved at the inputs' rank (an SVD of the
average), which keeps nearly all of it for checkpoints of one run (the report's "kept" share), or exactly with rank 0.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from .soup import cook, read_ingredient

log = logging.getLogger("rulztrainer")

STEP_RE = re.compile(r"_(\d+)$")


def sigma_rel_to_gamma(sigma_rel: float) -> float:
    """EDM2's power-function exponent for a relative width: the largest real root of
    gamma^3 + 7 gamma^2 + (16 - t) gamma + (12 - t) = 0, t = sigma_rel^-2."""
    if not 0 < sigma_rel < 0.288:  # (gamma 0, every step weighed the same, is sigma_rel 0.2887)
        raise ValueError("sigma_rel must be above 0 and below 0.288 (0.05 to 0.2 is usual)")
    t = sigma_rel ** -2
    roots = np.roots([1.0, 7.0, 16.0 - t, 12.0 - t])
    gamma = max(float(r.real) for r in roots if abs(r.imag) < 1e-9)
    return max(gamma, 0.0)


def ema_weights(steps: list[int], beta: float = 0.95, beta2: float | None = None,
                sigma_rel: float | None = None) -> list[float]:
    """Each checkpoint's weight in the average (summing to 1), the checkpoints in step order."""
    n = len(steps)
    if n == 0:
        raise ValueError("no checkpoints")
    if sigma_rel is not None:
        gamma = sigma_rel_to_gamma(sigma_rel)
        last = float(steps[-1])
        w, prev = [], 0
        for s in steps:  # the profile t^gamma times the steps this checkpoint stands for (normalized: no overflow)
            w.append((s / last) ** gamma * max(s - prev, 1))
            prev = s
    else:
        b2 = beta if beta2 is None else beta2
        for b in (beta, b2):
            if not 0 <= b < 1:
                raise ValueError("beta and beta2 must be at least 0 and below 1")
        k = n - 1  # the files merged into the first one
        decays = [beta + (b2 - beta) * (j / (k - 1) if k > 1 else 0.0) for j in range(k)]
        w = [1.0] * n
        for j, b in enumerate(decays):  # avg = b * avg + (1 - b) * file j + 1
            w = [x * b for x in w[: j + 1]] + [1 - b] + w[j + 2:]
    total = sum(w)
    return [x / total for x in w]


def checkpoint_step(path) -> int | None:
    """The step a LoRA file was saved at: its ss_steps metadata (RulzTrainer and kohya trainers write it), else the
    _NNNNNN its name ends in."""
    try:
        with safe_open(str(path), "pt") as f:
            meta = f.metadata() or {}
        if str(meta.get("ss_steps", "")).isdigit():
            return int(meta["ss_steps"])
    except Exception:  # noqa: BLE001
        pass
    m = STEP_RE.search(Path(path).stem)
    return int(m.group(1)) if m else None


def order_checkpoints(paths: list) -> list[tuple[Path, int]]:
    """(path, step) oldest first: by step where every file has one, else by modification time (steps then count 1,
    2, ...). Two files of the same step are refused (two runs mixed up, or a copy)."""
    paths = [Path(p) for p in paths]
    for p in paths:
        if not p.is_file():
            raise ValueError(f"not a file: {p}")
    steps = [checkpoint_step(p) for p in paths]
    if all(s is not None and s > 0 for s in steps):
        pairs = sorted(zip(paths, steps), key=lambda x: x[1])
        dup = [s for i, (_, s) in enumerate(pairs) if i and s == pairs[i - 1][1]]
        if dup:
            raise ValueError(f"two checkpoints of step {dup[0]}: post-hoc EMA averages one run's checkpoints")
        return pairs
    log.warning("Post-hoc EMA: not every file names its step; ordered by modification time, one step apart")
    ordered = sorted(paths, key=lambda p: p.stat().st_mtime)
    return [(p, i + 1) for i, p in enumerate(ordered)]


def _rank_of(ingredients: list[dict]) -> int:
    ranks = [e[1].shape[0] for ing in ingredients for e in ing.values() if e[0] == "lora"]
    return max(ranks) if ranks else 32  # (LoKr files: a change of no fixed rank)


def make_ema(paths: list, out, beta: float = 0.95, beta2: float | None = None, sigma_rel: float | None = None,
             rank: int | None = None, device="cpu", fmt: str | None = None) -> dict:
    """Read one run's checkpoints, average them with EMA weights and save one LoRA. rank None: the inputs' rank; 0:
    exact (the factors side by side). Returns the report (soup.cook's, plus the steps, weights and "out")."""
    from safetensors.torch import load_file

    from .lora import save_lora

    if len(paths) < 2:
        raise ValueError("post-hoc EMA needs 2 or more checkpoints of one run")
    pairs = order_checkpoints(paths)
    files, steps = [p for p, _ in pairs], [s for _, s in pairs]
    weights = ema_weights(steps, beta, beta2, sigma_rel)
    ingredients = [read_ingredient(p) for p in files]
    rank = _rank_of(ingredients) if rank is None else int(rank)
    stubs, report = cook(ingredients, weights, rank, 1.0, device)
    if not stubs:
        raise ValueError("the average has no layers (the checkpoints' changes were empty)")
    if fmt is None:
        fmt = "peft" if any(k.endswith(".lora_A.weight") for k in load_file(str(files[-1]))) else "comfy"
    how = (f"sigma_rel {sigma_rel:g}" if sigma_rel is not None
           else f"beta {beta:g}" + (f" to {beta2:g}" if beta2 is not None and beta2 != beta else ""))
    with safe_open(str(files[-1]), "pt") as f:  # the newest file's description (base model, config, steps)
        meta = dict(f.metadata() or {})
    ranks = sorted({s.rank for s in stubs.values()})
    meta.update({"rulztrainer_posthoc_ema": f"{how}: " + " + ".join(
                     f"{p.name} x{w:.3g}" for p, w in zip(files, weights)),
                 "ss_network_dim": str(ranks[-1]), "ss_network_alpha": str(ranks[-1])})
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_lora(stubs, out, metadata=meta, dtype=torch.bfloat16, fmt=fmt)
    report.update(out=str(out), steps=steps, files=[str(p) for p in files], how=how,
                  effective_step=sum(s * w for s, w in zip(steps, weights)))
    return report


def describe(report: dict) -> str:
    rank = "exact" if report["rank"] <= 0 else f"rank {report['rank']}"
    newest = report["weights"][-1]
    return (f"{report['ingredients']} checkpoints (steps {report['steps'][0]}-{report['steps'][-1]}), {report['how']}: "
            f"the newest weighs {newest:.0%}, the average sits near step {report['effective_step']:.0f}; "
            f"{report['layers']} layers, {rank}, {report['energy']:.0%} of the average kept")
