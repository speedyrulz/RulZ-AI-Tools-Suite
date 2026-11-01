"""Single-file downloads: Comfy-Org's ComfyUI repackages for the model a config trains (model_source "files" with a
file left empty), in the quant picked for it, the family's turbo LoRA (turbo_lora "auto") and its training adapter
(training_adapter "auto").

    planned_downloads(cfg)  what a run would fetch: [(field, repo, file)]
    resolve_downloads(cfg)  a copy of cfg with those fields set to the local files, downloading what is missing
                            (Hugging Face's cache: a second run, or another model sharing a file, reuses it)
"""
from __future__ import annotations

import dataclasses
import logging

from .config import (ARCH_TRAINING_ADAPTER, ARCH_TURBO, COMFY_DIFFUSION_FILES, COMFY_DIFFUSION_QUANTS, COMFY_REPOS,
                     COMFY_TEXT_ENCODER_FILES, COMFY_TEXT_ENCODER_QUANTS, COMFY_VAE_FILES, TrainConfig)

log = logging.getLogger("rulztrainer")


def planned_downloads(cfg: TrainConfig) -> list[tuple[str, str, str]]:
    """(config field, repo, file) for every file the config leaves to a download."""
    fam = cfg.arch_name()
    out = []
    if cfg.model_source == "files":
        repo = COMFY_REPOS[fam]
        if not cfg.transformer_file:
            pattern = COMFY_DIFFUSION_FILES.get(cfg.model_path)
            if pattern is None:
                raise ValueError(f"No single-file download for {cfg.model_path}: pick one of the listed models, set the "
                                 "diffusion model file, or use the Hugging Face source")
            if cfg.model_file_quant not in COMFY_DIFFUSION_QUANTS[fam]:
                raise ValueError(f"model_file_quant {cfg.model_file_quant!r}: {COMFY_REPOS[fam]} has "
                                 f"{', '.join(COMFY_DIFFUSION_QUANTS[fam])}")
            from .config import COMFY_QUANT_NAMES

            quant = COMFY_QUANT_NAMES.get(fam, {}).get(cfg.model_file_quant, cfg.model_file_quant)
            out.append(("transformer_file", repo, pattern.format(quant=quant)))
        if not cfg.text_encoder_file:
            if cfg.text_encoder_file_quant not in COMFY_TEXT_ENCODER_QUANTS[fam]:
                raise ValueError(f"text_encoder_file_quant {cfg.text_encoder_file_quant!r}: {COMFY_REPOS[fam]} has "
                                 f"{', '.join(COMFY_TEXT_ENCODER_QUANTS[fam])}")
            out.append(("text_encoder_file", repo, COMFY_TEXT_ENCODER_FILES[fam].format(quant=cfg.text_encoder_file_quant)))
        if not cfg.vae_file:
            out.append(("vae_file", repo, COMFY_VAE_FILES[fam]))
    if cfg.turbo_lora == "auto" and fam in ARCH_TURBO:
        out.append(("turbo_lora", ARCH_TURBO[fam]["repo"], ARCH_TURBO[fam]["file"]))
    if cfg.training_adapter == "auto" and fam in ARCH_TRAINING_ADAPTER and not cfg.finetune:
        out.append(("training_adapter", ARCH_TRAINING_ADAPTER[fam]["repo"], ARCH_TRAINING_ADAPTER[fam]["file"]))
    return out


def resolve_downloads(cfg: TrainConfig, progress=None) -> TrainConfig:
    """cfg with every planned file downloaded (or found in the cache) and its field set to the local path."""
    from .downloads import ensure_hf_file

    changes = {}
    for field, repo, file in planned_downloads(cfg):
        try:
            path = ensure_hf_file(repo, file, file.rsplit("/", 1)[-1], progress)
        except Exception as e:  # noqa: BLE001
            from .flux3.loading import gated_error, is_gated_error

            if is_gated_error(e):  # a gated repo (Black Forest Labs'): say what to accept, not just "401"
                raise gated_error(repo, e) from e
            raise
        log.info("%s: %s (%s)", field, path, repo)
        changes[field] = str(path)
    if cfg.turbo_lora == "auto" and "turbo_lora" not in changes:
        changes["turbo_lora"] = ""  # a family without a turbo LoRA
    if cfg.training_adapter == "auto" and "training_adapter" not in changes:
        changes["training_adapter"] = ""  # a family without a training adapter (or a fine-tune)
    return dataclasses.replace(cfg, **changes) if changes else cfg
