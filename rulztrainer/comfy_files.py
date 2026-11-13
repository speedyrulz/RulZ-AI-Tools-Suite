"""Single-file downloads: Comfy-Org's ComfyUI repackages for the model a config trains (model_source "files" with a
file left empty), in the quant picked for it, the family's turbo LoRA (turbo_lora "auto"; none when samples render a
Turbo checkpoint), its sampling LoRA (sample_lora "auto": MOSS-Image Base's SFT LoRA) and its training adapter
(training_adapter "auto").

    planned_downloads(cfg)  what a run would fetch: [(field, repo, file)]
    resolve_downloads(cfg)  a copy of cfg with those fields set to the local files, downloading what is missing
                            (Hugging Face's cache: a second run, or another model sharing a file, reuses it)
"""
from __future__ import annotations

import dataclasses
import logging

from .config import (ARCH_SAMPLE_LORA, ARCH_TRAINING_ADAPTER, ARCH_TURBO, COMFY_DIFFUSION_FILES, COMFY_DIFFUSION_QUANTS, COMFY_REPOS,
                     COMFY_TEXT_ENCODER_FILES, COMFY_TEXT_ENCODER_QUANTS, COMFY_VAE_FILES, TrainConfig)
from .config import COMFY_DEFAULT_QUANT, COMFY_TEXT_ENCODER_REPOS, training_adapter_for

log = logging.getLogger("rulztrainer")


def _file_quant(fam: str, quant: str) -> str:
    """The format to fetch: the config's default names the family's own default where it has no int8 file."""
    return COMFY_DEFAULT_QUANT.get(fam, quant) if quant == "int8_convrot" else quant


def planned_downloads(cfg: TrainConfig) -> list[tuple[str, str, str]]:
    """(config field, repo, file) for every file the config leaves to a download."""
    fam = cfg.arch_name()
    out = []
    if cfg.model_source == "files" and fam not in COMFY_REPOS:
        # a family nobody publishes single files of (PAN 2): every file has to be set
        missing = [f for f in ("transformer_file", "text_encoder_file", "vae_file") if not getattr(cfg, f)]
        if missing:
            raise ValueError(f"No single-file download exists for {fam}: set {', '.join(missing)} (for PAN 2, the "
                             "files scripts/pan2/convert_pan2.py makes), or use the Hugging Face source")
    elif cfg.model_source == "files":
        repo = COMFY_REPOS[fam]
        if not cfg.transformer_file:
            pattern = COMFY_DIFFUSION_FILES.get(cfg.model_path)
            if pattern is None:
                raise ValueError(f"No single-file download for {cfg.model_path}: pick one of the listed models, set the "
                                 "diffusion model file, or use the Hugging Face source")
            model_quant = _file_quant(fam, cfg.model_file_quant)
            if model_quant not in COMFY_DIFFUSION_QUANTS[fam]:
                raise ValueError(f"model_file_quant {cfg.model_file_quant!r}: {COMFY_REPOS[fam]} has "
                                 f"{', '.join(COMFY_DIFFUSION_QUANTS[fam])}")
            from .config import COMFY_QUANT_NAMES

            quant = COMFY_QUANT_NAMES.get(fam, {}).get(model_quant, model_quant)
            if isinstance(pattern, dict):  # one repo per format (FLUX.2 Klein's)
                out.append(("transformer_file", *pattern[quant]))
            else:
                out.append(("transformer_file", repo, pattern.format(quant=quant)))
        if not cfg.text_encoder_file:
            te_quant = _file_quant(fam, cfg.text_encoder_file_quant)
            if te_quant not in COMFY_TEXT_ENCODER_QUANTS[fam]:
                raise ValueError(f"text_encoder_file_quant {cfg.text_encoder_file_quant!r}: {COMFY_REPOS[fam]} has "
                                 f"{', '.join(COMFY_TEXT_ENCODER_QUANTS[fam])}")
            te_repo = COMFY_TEXT_ENCODER_REPOS.get(fam, {}).get(te_quant, repo)
            te_file = COMFY_TEXT_ENCODER_FILES[fam]
            out.append(("text_encoder_file", te_repo,
                        te_file[te_quant] if isinstance(te_file, dict) else te_file.format(quant=te_quant)))
        if not cfg.vae_file:
            out.append(("vae_file", repo, COMFY_VAE_FILES[fam]))
    if cfg.turbo_lora == "auto" and fam in ARCH_TURBO and not cfg.samples_on_distilled_model():
        out.append(("turbo_lora", ARCH_TURBO[fam]["repo"], ARCH_TURBO[fam]["file"]))
    if cfg.sample_lora == "auto" and fam in ARCH_SAMPLE_LORA and cfg.samples_on_base_model():
        out.append(("sample_lora", ARCH_SAMPLE_LORA[fam]["repo"], ARCH_SAMPLE_LORA[fam]["file"]))
    if cfg.training_adapter == "auto" and fam in ARCH_TRAINING_ADAPTER and not cfg.finetune:
        out.append(("training_adapter", *training_adapter_for(fam, cfg.model_path)))
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
        if cfg.samples_on_distilled_model():
            log.info("turbo_lora auto: none, the sampling model is already a distilled (Turbo) checkpoint")
        changes["turbo_lora"] = ""  # a family without a turbo LoRA, or a Turbo model that needs none
    if cfg.sample_lora == "auto" and "sample_lora" not in changes:
        changes["sample_lora"] = ""  # samples render another model than the family's base checkpoint (SFT, Turbo)
    if cfg.training_adapter == "auto" and "training_adapter" not in changes:
        changes["training_adapter"] = ""  # a family without a training adapter (or a fine-tune)
    return dataclasses.replace(cfg, **changes) if changes else cfg
