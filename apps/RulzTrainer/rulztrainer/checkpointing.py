"""Selective gradient checkpointing: blocks that fit in spare GPU memory keep their activations.

With gradient checkpointing every DiT block's forward runs twice per step (once in the forward, once recomputed in the
backward). A block flagged `keep_activations` skips the checkpoint, so its forward runs once and its activations stay
in memory until the backward reaches it. The block runners of every model family check the flag (MOSS `_run_layer`,
FIBO / Krea 2 `_run`).

`checkpoint_keep_blocks: -1` (auto) measures the peak memory of one forward+backward on the largest training bucket
with one and with three blocks kept and keeps as many of the last trained blocks as the free memory allows
(Trainer._calibrate_checkpointing, choose_kept_blocks); a step that still runs out keeps fewer from then on. N keeps the
last N trained blocks; 0 checkpoints every block.
"""
from __future__ import annotations

from typing import Iterable

from torch import nn

GB = 1024 ** 3


def set_kept_blocks(model: nn.Module, indices: Iterable[int]) -> int:
    """Flag model.layers[i] for i in `indices` to keep its activations, clear the flag on every other block.
    Returns the number of kept blocks."""
    keep = set(int(i) for i in indices)
    for i, block in enumerate(model.layers):
        block.keep_activations = i in keep
    return sum(1 for b in model.layers if getattr(b, "keep_activations", False))


def kept_blocks(model: nn.Module) -> list[int]:
    return [i for i, b in enumerate(model.layers) if getattr(b, "keep_activations", False)]


def trained_block_indices(n_blocks: int, blocks: list[int] | None) -> list[int]:
    """Blocks that run a backward pass: every block from the first one with a LoRA on (blocks before it have neither
    trainable parameters nor inputs that need a gradient, so checkpointing them saves nothing)."""
    if blocks is None:
        return list(range(n_blocks))
    if not blocks:
        return []
    return list(range(min(blocks), n_blocks))


def kept_indices(candidates: list[int], k: int) -> list[int]:
    """The last k of the candidate blocks (k <= 0: none)."""
    if k <= 0:
        return []
    return list(candidates[-k:])


def choose_kept_blocks(n_candidates: int, peak_one: float, per_block: float, budget: float) -> int:
    """How many of the last blocks fit (all in bytes). Keeping the very last block costs next to nothing: its backward
    recomputes it anyway when checkpointed. Every further kept block adds its full activations, since they are all
    alive when the backward starts. So the peak with k >= 1 kept is peak_one + (k - 1) * per_block, where per_block is
    the measured slope between keeping one and keeping three blocks."""
    if n_candidates <= 0 or peak_one > budget:
        return 0
    if per_block <= 0:  # a further kept block measured as free: keep them all
        return n_candidates
    return min(n_candidates, 1 + int((budget - peak_one) // per_block))


def memory_margin(total: float, identity_loss: bool = False) -> float:
    """Memory left unused below the card's total: max(1.5 GB, 8%), plus 1.5 GB more while the identity loss runs (its
    VAE decode with gradients happens inside the step, after the calibration's measurement)."""
    return max(1.5 * GB, 0.08 * total) + (1.5 * GB if identity_loss else 0.0)
