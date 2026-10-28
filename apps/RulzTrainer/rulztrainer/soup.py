"""Seed soup: average the weight changes of several LoRAs of one subject (different seeds, or a race's finalists) and
save the average as one LoRA.

It is the "model soup" of Wortsman et al. (ICML 2022): models fine-tuned from the same starting point sit in one
low-loss basin, so averaging their weights keeps what they agree on and loses what each learned on its own; their
greedy soup (add an ingredient only when the held-out score does not drop) beat the best single model on ImageNet.
Here the starting point is the base model and each LoRA is a change to it.

What is averaged is each layer's weight change, delta = up @ down x scale (LoKr: kron(w1, w2)), never the factors:
two runs can reach the same change with entirely different up / down, so averaging those would be meaningless. The
average of k rank-r changes has rank up to k x r; it is saved either exactly (rank 0, LoRA ingredients: the factors
side by side) or as its best rank-r approximation (an SVD, finetune.lowrank_factors), which keeps the directions the
ingredients share and drops the ones only one of them has. Where they disagree the average is smaller than each of
them, so a soup can act like a lower LoRA strength: `strength` scales it back, and the greedy soup tries a few.

OFT and DoRA files change the weights through the base weights themselves, so they cannot be averaged from the files
alone: not supported.
"""
from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import load_file

log = logging.getLogger("rulztrainer")


def read_ingredient(path) -> dict[str, tuple]:
    """{module name: (kind, first, second, scale)} of a LoRA or LoKr file (ComfyUI, PEFT or LyCORIS keys)."""
    from .lora import parse_comfy_lora, read_peft_metadata

    state = load_file(str(path))
    if any(k.endswith((".oft_blocks", ".dora_scale")) for k in state):
        raise ValueError(f"{Path(path).name}: OFT and DoRA change the weights through the base weights, so they "
                         "cannot be averaged from the files alone; soup LoRA or LoKr files")
    peft = read_peft_metadata(path)
    entries = parse_comfy_lora(state, default_alpha=peft.get("lora_alpha"), alpha_pattern=peft.get("alpha_pattern"))
    if not entries:
        raise ValueError(f"{Path(path).name}: no LoRA or LoKr layers found")
    return entries


def delta_of(entry: tuple, device="cpu") -> torch.Tensor:
    """A layer's weight change [out, in] in fp32, as ComfyUI applies it."""
    kind, first, second, scale = entry
    first, second = first.to(device, torch.float32), second.to(device, torch.float32)
    if kind == "lokr":
        return torch.kron(first, second) * scale
    return (second @ first) * scale


def is_lora(entry: tuple) -> bool:
    return entry[0] == "lora"


def normalize_weights(n: int, weights=None) -> list[float]:
    w = [1.0] * n if not weights else [float(x) for x in weights]
    if len(w) != n or any(x < 0 for x in w) or sum(w) <= 0:
        raise ValueError(f"soup weights: {n} values of 0 or more, not all 0")
    total = sum(w)
    return [x / total for x in w]


def cook(ingredients: list[dict], weights=None, rank: int = 32, strength: float = 1.0, device="cpu",
         min_norm: float = 1e-8) -> tuple[dict, dict]:
    """The soup of these ingredients (read_ingredient's): ({name: stub} for lora.save_lora, report). rank 0 = exact
    (LoRA ingredients only: their factors side by side, rank = the sum of theirs). The report: layers, the mean cosine
    between every two ingredients' changes ("agreement", 1 = the same direction), the soup's size against the
    ingredients' average size ("norm_ratio", below 1 where they disagree), and the share of the average kept at this
    rank ("energy", 1 = all)."""
    from .finetune import lowrank_factors

    w = normalize_weights(len(ingredients), weights)
    names = sorted(set().union(*ingredients))
    exact = int(rank) <= 0
    if exact and not all(is_lora(e) for ing in ingredients for e in ing.values()):
        raise ValueError("an exact soup (rank 0) needs LoRA ingredients: a LoKr change is not low-rank; give a rank")
    stubs, cos_sum, cos_n = {}, 0.0, 0
    soup_sq = ing_sq = kept_sq = 0.0
    for name in names:
        deltas = [(wi, ing[name]) for wi, ing in zip(w, ingredients) if name in ing]  # a missing layer counts 0
        dense = [delta_of(e, device) for _, e in deltas]
        avg = sum(wi * d for (wi, _), d in zip(deltas, dense))
        norm = float(avg.norm())
        if norm < min_norm:
            continue
        flat = [d.flatten() for d in dense]
        for i in range(len(flat)):
            for j in range(i + 1, len(flat)):
                cos_sum += float(torch.nn.functional.cosine_similarity(flat[i], flat[j], dim=0))
                cos_n += 1
        soup_sq += norm ** 2
        ing_sq += (sum(float(d.norm()) for d in dense) / len(dense)) ** 2
        if exact:  # sum_i w_i s_i up_i @ down_i = [up_1 .. up_k] @ [w_1 s_1 down_1; ..; w_k s_k down_k]
            up = torch.cat([e[2].float() for _, e in deltas], 1)
            down = torch.cat([e[1].float() * (wi * e[3]) for wi, e in deltas], 0)
            kept_sq += norm ** 2
        else:
            down, up = lowrank_factors(avg, rank)
            kept_sq += float((up.to(device) @ down.to(device)).norm()) ** 2
        r = down.shape[0]
        parts = {"lora_down": (down * strength).contiguous().cpu(), "lora_up": up.contiguous().cpu()}
        stubs[name] = SimpleNamespace(kind="lora", alpha=float(r), rank=r, factor=8, trainable=True,
                                      adapter_params=lambda parts=parts: parts)
        del dense, avg
    report = {"layers": len(stubs), "ingredients": len(ingredients), "weights": w, "rank": int(rank),
              "strength": strength, "agreement": cos_sum / cos_n if cos_n else 1.0,
              "norm_ratio": (soup_sq / ing_sq) ** 0.5 if ing_sq else 1.0,
              "energy": kept_sq / soup_sq if soup_sq else 1.0}
    return stubs, report


def save(stubs: dict, out, sources: list, report: dict, fmt: str = "comfy") -> Path:
    from .lora import save_lora

    if not stubs:
        raise ValueError("the soup has no layers (every ingredient's changes cancelled out or were empty)")
    ranks = sorted({s.rank for s in stubs.values()})
    meta = {"rulztrainer_soup": " + ".join(f"{Path(p).name} x{w:.3g}" for p, w in zip(sources, report["weights"])),
            "rulztrainer_soup_report": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in report.items()},
            "ss_network_dim": str(ranks[-1]), "ss_network_alpha": str(ranks[-1])}
    out = Path(out)
    save_lora(stubs, out, metadata=meta, dtype=torch.bfloat16, fmt=fmt)
    return out


def describe(report: dict) -> str:
    rank = "exact" if report["rank"] <= 0 else f"rank {report['rank']}"
    return (f"{report['ingredients']} ingredients, {report['layers']} layers, {rank}, strength {report['strength']:g}: "
            f"agreement {report['agreement']:.2f} (cosine between their changes), the average is "
            f"{report['norm_ratio']:.0%} of their size, {report['energy']:.0%} of it kept")


def make_soup(paths: list, out, rank: int = 32, weights=None, strength: float = 1.0, device="cpu",
              fmt: str | None = None) -> dict:
    """Read the files, cook, save; returns the report (with "out")."""
    if len(paths) < 2:
        raise ValueError("a soup needs 2 or more LoRA files")
    ingredients = [read_ingredient(p) for p in paths]
    if fmt is None:
        fmt = "peft" if any(any(k.endswith(".lora_A.weight") for k in load_file(str(p))) for p in paths) else "comfy"
    stubs, report = cook(ingredients, weights, rank, strength, device)
    report["out"] = str(save(stubs, out, paths, report, fmt))
    return report
