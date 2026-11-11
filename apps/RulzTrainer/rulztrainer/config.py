"""Training configuration (dataclass + YAML round trip)."""
from __future__ import annotations

import dataclasses
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from . import promptcheck as _promptcheck

MODEL_PRESETS = {
    "MOSS-Image-Base": "OpenMOSS-Team/MOSS-Image-Base",
    "MOSS-Image-SFT": "OpenMOSS-Team/MOSS-Image-SFT",
    "FIBO-2.0-dev": "briaai/Fibo-2.0-dev",
    "FIBO-2.0-edit-turbo": "briaai/Fibo-2.0-edit-turbo",
    "Krea-2-Raw": "krea/Krea-2-Raw",
    "Krea-2-Turbo": "krea/Krea-2-Turbo",
    "Qwen-Image-2.1": "Qwen/Qwen-Image-2.1",
    "Qwen-Image-2.1-Turbo": "Qwen/Qwen-Image-2.1-Turbo",
    "FLUX-3-Klein-Base": "black-forest-labs/flux-3-image-klein-base",
    "FLUX-3-Klein": "black-forest-labs/flux-3-image-klein",
    "FLUX-2-Klein-9B-Base": "black-forest-labs/FLUX.2-klein-base-9B",
    "FLUX-2-Klein-9B": "black-forest-labs/FLUX.2-klein-9B",
    "MiniMax-H3-FL2VA": "MiniMaxAI/MiniMax-H3",  # one repository holds both: transformer/ (FL2VA), transformer_ref/
    "MiniMax-H3-Ref2VA": "MiniMaxAI/MiniMax-H3#ref2va",
}
ARCH_NAMES = ("moss_image", "fibo2", "krea2", "qwen_image21", "flux3_klein", "flux2_klein", "minimax_h3")
OPTIMIZERS = ("adamw", "adamw8bit", "adafactor", "automagic3", "automagic4", "keel", "keel_fb", "hull", "muon",
              "normuon")
# Layer groups Muon's authors, Moonshot and OneTrainer keep on AdamW (embedders, modulation, text projection, final
# layer): RulzTrainer's muon / normuon orthogonalize them too and warn when they are trained.
MUON_EDGE_TARGETS = ("embed", "modulation", "text_in")
AUTOMAGIC = ("automagic3", "automagic4", "keel", "keel_fb")  # self-tuning lr: no schedule, no LoRA+, no per-image lr
KEEL = ("keel", "keel_fb")  # Keel and its face-and-body version: one class (optimizers/keel.py), keel_fb with the
#                             body share in the token gain and the magnitude vote in the consensus gate
KEEL_CONSENSUS_MODES = ("auto", "sign", "magnitude")
KEEL_BAND_LOGIT_RANGE = 4.0  # Keel's noise bands span logit(sigma) in [-range, range] (sigma 0.018 to 0.982)


def keel_band_edges(bands: int) -> list[float]:
    """The sigma edges of Keel's noise bands: bands + 1 values from 0 to 1 (equal bands in logit sigma over
    [-KEEL_BAND_LOGIT_RANGE, +range], the ends open)."""
    r = KEEL_BAND_LOGIT_RANGE
    inner = [1.0 / (1.0 + math.exp(-(-r + 2.0 * r * k / bands))) for k in range(1, bands)]
    return [0.0] + inner + [1.0]


def keel_band_weights(spec: str, bands: int) -> list[float] | None:
    """keel_band_weights resolved to one multiplier per band: None (flat: "" or a single band), the identity hump
    (0.5 + 0.8 exp(-((c + 0.7) / 1.5)^2) at each band's logit centre c: about 1.3 over sigma 0.12 to 0.5, 0.5 at the
    ends), the subject plateau (0.6 + 0.7 exp(-(c / 2.2)^4): about 1.2 to 1.3 over sigma 0.18 to 0.82, where the
    face's detail and the body's silhouette form alike, 0.6 at the ends), or the listed numbers. Raises ValueError on
    a bad spec."""
    s = (spec or "").strip().lower()
    if bands <= 1 or s in ("", "flat"):
        return None
    if s in ("identity", "subject"):
        r = KEEL_BAND_LOGIT_RANGE
        centres = [-r + 2.0 * r * (k + 0.5) / bands for k in range(bands)]
        if s == "identity":
            return [round(0.5 + 0.8 * math.exp(-((c + 0.7) / 1.5) ** 2), 3) for c in centres]
        return [round(0.6 + 0.7 * math.exp(-((c / 2.2) ** 4)), 3) for c in centres]
    try:
        vals = [float(x) for x in s.replace(";", ",").split(",") if x.strip()]
    except ValueError:
        raise ValueError(f"keel_band_weights must be empty, identity, subject or {bands} comma-separated numbers, "
                         f"got {spec!r}")
    if len(vals) != bands:
        raise ValueError(f"keel_band_weights lists {len(vals)} numbers for keel_bands {bands}")
    if any(not v > 0.0 for v in vals):
        raise ValueError("keel_band_weights must all be above 0")
    return vals


def keel_consensus_mode(spec: str, optimizer: str) -> str:
    """keel_consensus_mode resolved for an optimizer: auto = sign for keel, magnitude for keel_fb. Raises ValueError
    on a bad spec."""
    s = (spec or "auto").strip().lower()
    if s not in KEEL_CONSENSUS_MODES:
        raise ValueError(f"keel_consensus_mode must be auto, sign or magnitude, got {spec!r}")
    if s == "auto":
        return "magnitude" if (optimizer or "").lower() == "keel_fb" else "sign"
    return s


def family_named_in(text: str) -> str | None:
    """The model family a path, repo id or file name names (krea/Krea-2-Turbo: krea2), or None when it names none."""
    hint = (text or "").lower()
    if "qwen-image" in hint or "qwen_image" in hint or "qwenimage" in hint:
        return "qwen_image21"
    if any(s in hint for s in ("flux-3", "flux3", "flux_3", "flux.3")):
        return "flux3_klein"
    if any(s in hint for s in ("flux-2", "flux2", "flux_2", "flux.2")):
        return "flux2_klein"
    if "p-h3" in hint or "p_h3" in hint:  # PrunaAI/P-H3-Image, p-h3-image_bf16.safetensors (before "minimax")
        return "p_h3_image"
    # MiniMaxAI/MiniMax-H3-Image, minimax_h3_image_pruned_bf16.safetensors (before "minimax"), and ostris' versioned
    # adapter for it, minimax_h3_image_training_adapter_v1.safetensors. Not the unversioned adapter of that name:
    # circlestone-labs' MiniMax-H3-Image-Training-Adapter is an image adapter for H3 itself
    if "minimax-h3-image" in hint or "minimax_h3_image" in hint:
        if "adapter" not in hint or "minimax_h3_image_training_adapter_v" in hint:
            return "minimax_h3_image"
    if re.search(r"(^|[^a-z0-9])pan[-_ ]?2", hint):  # IFM/PAN2, PAN2_nvfp4.safetensors, pan2_int8.safetensors
        return "pan2"
    for word, family in (("ideogram", "ideogram4"), ("minimax", "minimax_h3"), ("fibo", "fibo2"), ("krea", "krea2"),
                         ("moss", "moss_image")):
        if word in hint:
            return family
    return None


def sampling_file_family(key: str, value: str) -> str | None:
    """The family a sampling model or sampling-only LoRA setting names: a LoRA by its file name only (a folder of the
    user's, /data/krea_stuff/moss_lora, says nothing; a LoRA of another model is stopped when it matches no layer),
    the sampling model by its name, else its whole path or repo id (a diffusers folder krea__Krea-2-Turbo/transformer).
    None for "", auto, or a name without a family."""
    if value in ("", "auto"):
        return None
    name = family_named_in(Path(value).name)
    return name or (family_named_in(value) if key == "sample_model_path" else None)


# LoRA target groups each family trains when lora_targets is empty. Krea 2 adds the attention output gate and its
# text-fusion stack, which Fizgig measured to be under-trained by the attention + feed-forward recipe (scaling a
# finished LoRA's text-fusion blocks x3 lifted likeness 2-7 points on several Krea 2 LoRAs).
_CORE_TARGETS = ("q", "k", "v", "o", "ffn_up_gate", "ffn_down")
ARCH_DEFAULT_TARGETS = {"moss_image": _CORE_TARGETS, "fibo2": _CORE_TARGETS,
                        "krea2": _CORE_TARGETS + ("attn_gate", "text_fusion"), "qwen_image21": _CORE_TARGETS,
                        "flux3_klein": _CORE_TARGETS + ("mode_blocks",),  # Klein: every block's attention + MLP
                        "flux2_klein": _CORE_TARGETS,  # every block's attention + MLP, as ai-toolkit and Fizgig train
                        "minimax_h3": _CORE_TARGETS}  # H3: attn.qkv_proj, out_proj, mlp.fc1, fc2 (musubi-tuner's)
# (sample steps, sample CFG) per family at full quality, in this trainer's guidance convention (uncond + cfg * (cond -
# uncond), 1 = off): MOSS-Image's reference CFG 3; Bria's FIBO 2.0 pipeline, 30 steps at 5; Krea's pipeline, 28 steps
# at its guidance_scale 4.5, which computes cond + g * (cond - uncond): cfg 5.5 here; Qwen-Image 2.1 is meant to be
# sampled without guidance (ComfyUI's template: 25 steps at CFG 1; the diffusers pipeline 40)
ARCH_FULL_SAMPLE = {"moss_image": (20, 3.0), "fibo2": (30, 5.0), "krea2": (28, 5.5), "qwen_image21": (25, 1.0),
                    "flux3_klein": (50, 3.5)}  # FLUX 3 Klein Base: BFL's 50 steps at guidance 3.5 (diffusers' CFG)
# MiniMax H3 is guidance-distilled (no CFG); 30 steps as musubi-tuner's one-frame example (the pipeline: 49)
ARCH_FULL_SAMPLE["minimax_h3"] = (30, 1.0)
ARCH_FULL_SAMPLE["flux2_klein"] = (20, 3.5)  # FLUX.2 Klein Base: BFL's defaults (no guidance embedding, CFG only)
# Each family's turbo LoRA (turbo_lora: auto downloads it) and the sampling it is made for. The shift only matters for
# MOSS-Image: the Krea 2 and FIBO 2.0 samplers take theirs from the image size.
#   MOSS-Image: lodestones' 8-step DMD LoRA distilled from Base (Fizgig's pick), at 12 steps (Fizgig: much better than
#               the card's 8), CFG 1, shift 5; it suits Base, not SFT, and can understate likeness (Fizgig, 20 Oct 2026)
#   Krea 2:     Comfy-Org's Krea 2 Turbo LoRA (rank 64), 8 steps, CFG 1 (ComfyUI's Krea 2 Turbo workflow)
#   FIBO 2.0:   Comfy-Org's FIBO 2.0 Turbo LoRA (Edit Turbo as a LoRA on dev), 4 steps, CFG 1 (Bria's Edit Turbo card)
#   Qwen-Image 2.1: Comfy-Org's Turbo LoRA (extracted from Qwen-Image-2.1-Turbo), 8 steps, CFG 1 (its model card), with
#               the Turbo checkpoint's own sigmas
ARCH_TURBO = {
    "moss_image": {"repo": "lodestones/Kromoss", "file": "8-steps-dmd-lora.safetensors", "strength": 1.0,
                   "steps": 12, "cfg": 1.0, "shift": 5.0},
    "krea2": {"repo": "Comfy-Org/Krea-2", "file": "loras/krea2_turbo_lora_rank_64_bf16.safetensors", "strength": 1.0,
              "steps": 8, "cfg": 1.0, "shift": 5.0},
    "fibo2": {"repo": "Comfy-Org/Fibo-2.0", "file": "loras/fibo_2.0_turbo_lora_avg_rank_270_bf16.safetensors",
              "strength": 1.0, "steps": 4, "cfg": 1.0, "shift": 5.0},
    "qwen_image21": {"repo": "Comfy-Org/Qwen-Image-2.1",
                     "file": "loras/qwen_image_2.1_turbo_lora_avg_rank_178_bf16.safetensors", "strength": 1.0,
                     "steps": 8, "cfg": 1.0, "shift": 5.0},
    # MiniMax H3: lightx2v's FL2VA Turbo 8-step v1.0 (ComfyUI layout, blocks.N.attn.qkv_proj..., the port's names),
    # distilled at video shift 12 / audio 3 on 544p video: 8 steps at CFG 1 on the base schedule (its model card).
    # Untested on single frames
    "minimax_h3": {"repo": "Comfy-Org/MiniMax-H3",
                   "file": "loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors", "strength": 1.0,
                   "steps": 8, "cfg": 1.0, "shift": 12.0},
    # FLUX.2 Klein 9B: vafipas663's rank 256 LoRA, Klein 9B extracted against Base (kohya names). 4 steps at CFG 1 on
    # the distilled model's schedule. Its card reports static noise above strength 0.5, sometimes needing as low as
    # 0.1; strength 1 is the project owner's choice. Not yet tried on real weights here
    "flux2_klein": {"repo": "vafipas663/flux2-klein-base-9b-distill-lora",
                    "file": "flux-2-klein-9b_extracted_lora_rank_256-bf16.safetensors", "strength": 1.0,
                    "steps": 4, "cfg": 1.0, "shift": 2.02},
}
# Each family's training adapter (training_adapter: auto downloads it): a LoRA that is on while the LoRA trains and off
# while it samples, and never saved into it, so the trained LoRA works on the plain model. Fizgig measured on
# Qwen-Image 2.1: a LoRA trained without one collapses into texture or wobbles part-way, with no warning in the loss;
# with Fizgig's adapter likeness reached 77 against 56 without (and 76 with SimpleTuner's).
ARCH_TRAINING_ADAPTER = {
    "qwen_image21": {"repo": "ShootTheSound/Fizgig-Qwen-Image-2.1-Training-Adapter",
                     "file": "fizgig_qwen_image_2.1_training_adapter_v2.safetensors"},
    # MiniMax H3 (CFG-distilled: musubi-tuner saw one-frame LoRAs break up within ~50 steps without an adapter or a
    # guidance loss): circlestone-labs' image training adapter (rank 64, made for image-only training on FL2VA,
    # 0.62 GB), the first of the three musubi-tuner tested; ostris' v2 (rank 32) is the alternative
    "minimax_h3": {"repo": "circlestone-labs/MiniMax-H3-Image-Training-Adapter",
                   "file": "minimax_h3_image_training_adapter.safetensors"},
}
# Each family's sampling LoRA (sample_lora: auto downloads it), applied with the turbo LoRA while samples render the
# family's base checkpoint, never in training. MOSS-Image: Kijai's SFT LoRA (SFT minus Base, rank 512, blocks 3, 4
# and 21-41 with their norms), so samples of a LoRA trained on Base show it as it looks on SFT. Kijai's card uses it
# at -0.5 on SFT to soften SFT's banding; on Base, +1 adds SFT's look
ARCH_SAMPLE_LORA = {
    "moss_image": {"repo": "Kijai/MOSS-image_experimental",
                   "file": "loras/moss_image_sft_lora_blocks_21_to_49_rank_512_bf16.safetensors", "strength": 1.0},
}
# (sample steps, sample CFG) per family by default: the turbo LoRA's
ARCH_SAMPLE_DEFAULTS = {fam: (t["steps"], t["cfg"]) for fam, t in ARCH_TURBO.items()}
# FLUX 3 Klein has no turbo LoRA here: on 31 Oct 2026 neither BFL nor Comfy-Org publishes one, and the community's
# need their own ComfyUI nodes (NoiseToNova's PDD, terminusresearch's flow-map extract) or call themselves a quick
# experiment (Muinez's 4-step, shift 3). It samples at full quality, or with BFL's distilled Klein as the sampling
# model (sample_model_path: 16 steps, CFG 1); a family without one is not in ARCH_SAMPLE_DEFAULTS (ARCH_FULL_SAMPLE's)

# Single files (model_source "files") come from Comfy-Org's ComfyUI repackages when a file is left empty: the diffusion
# model of the picked model in model_file_quant, the family's text encoder in text_encoder_file_quant, its VAE
COMFY_REPOS = {"moss_image": "Comfy-Org/MOSS-Image", "krea2": "Comfy-Org/Krea-2", "fibo2": "Comfy-Org/Fibo-2.0",
               "qwen_image21": "Comfy-Org/Qwen-Image-2.1",
               "flux3_klein": "black-forest-labs/flux-3-image-klein-comfy"}  # BFL's own ComfyUI files (gated)
# a quant's spelling in a repo's file names where it is not the config's (BFL's files: int8-convrot)
COMFY_QUANT_NAMES = {"flux3_klein": {"int8_convrot": "int8-convrot"}}
COMFY_REPOS["minimax_h3"] = "Comfy-Org/MiniMax-H3"
COMFY_DIFFUSION_FILES = {
    "OpenMOSS-Team/MOSS-Image-Base": "diffusion_models/moss_image_base_{quant}.safetensors",
    "OpenMOSS-Team/MOSS-Image-SFT": "diffusion_models/moss_image_sft_{quant}.safetensors",
    "krea/Krea-2-Raw": "diffusion_models/krea2_raw_{quant}.safetensors",
    "krea/Krea-2-Turbo": "diffusion_models/krea2_turbo_{quant}.safetensors",
    "briaai/Fibo-2.0-dev": "diffusion_models/fibo2_dev_{quant}.safetensors",
    "briaai/Fibo-2.0-edit-turbo": "diffusion_models/fibo2_edit_turbo_{quant}.safetensors",
    "Qwen/Qwen-Image-2.1": "diffusion_models/qwen_image_2.1_{quant}.safetensors",
    "Qwen/Qwen-Image-2.1-Turbo": "diffusion_models/qwen_image_2.1_turbo_{quant}.safetensors",
    "black-forest-labs/flux-3-image-klein-base": "diffusion_models/flux3-image-klein-base-{quant}.safetensors",
    "black-forest-labs/flux-3-image-klein": "diffusion_models/flux3-image-klein-{quant}.safetensors",
    "MiniMaxAI/MiniMax-H3": "diffusion_models/minimax_h3_fl2va_{quant}.safetensors",
    "MiniMaxAI/MiniMax-H3#ref2va": "diffusion_models/minimax_h3_ref2va_{quant}.safetensors",
}
COMFY_DIFFUSION_QUANTS = {"moss_image": ("int8_convrot", "bf16"), "krea2": ("int8_convrot", "fp8_scaled", "bf16"),
                          "fibo2": ("int8_convrot", "bf16"), "qwen_image21": ("int8_convrot", "bf16"),
                          "flux3_klein": ("int8_convrot", "nvfp4", "bf16")}
COMFY_TEXT_ENCODER_FILES = {"moss_image": "text_encoders/qwen3.5_9b_{quant}.safetensors",
                            "krea2": "text_encoders/qwen3vl_4b_{quant}.safetensors",
                            "fibo2": "text_encoders/qwen3vl_4b_{quant}.safetensors",
                            "qwen_image21": "text_encoders/qwen3vl_8b_{quant}.safetensors",
                            "flux3_klein": "text_encoders/qwen3vl_4b_{quant}.safetensors"}
COMFY_TEXT_ENCODER_QUANTS = {"moss_image": ("int8_convrot", "bf16"), "krea2": ("int8_convrot", "fp8_scaled", "bf16"),
                             "fibo2": ("int8_convrot", "bf16"), "qwen_image21": ("int8_convrot", "bf16"),
                             "flux3_klein": ("int8_convrot", "bf16")}
COMFY_VAE_FILES = {"moss_image": "vae/flux2-vae_small.safetensors", "krea2": "vae/qwen_image_vae.safetensors",
                   "fibo2": "vae/flux2-vae.safetensors", "qwen_image21": "vae/qwen_image_2.1_vae_bf16.safetensors",
                   "flux3_klein": "vae/flux2-vae.safetensors"}
# A text encoder file another family's repackage holds (family -> quant -> repo; the family's own COMFY_REPOS repo
# otherwise)
COMFY_TEXT_ENCODER_REPOS: dict[str, dict[str, str]] = {}
# FLUX.2 Klein 9B: no int8 release. BFL's own fp8 and bf16 files of the diffusion model live in one gated repo per
# format ({quant: (repo, file)}); Comfy-Org's repo holds the text encoder and the VAE, ungated. An fp8 file is read up
# to bf16 and quantized for the memory it runs in (transformer_quant), as any float file.
COMFY_REPOS["flux2_klein"] = "Comfy-Org/Flux-2-Klein-9B"
_BFL = "black-forest-labs/FLUX.2-klein"
COMFY_DIFFUSION_FILES["black-forest-labs/FLUX.2-klein-base-9B"] = {
    "fp8_scaled": (f"{_BFL}-base-9b-fp8", "flux-2-klein-base-9b-fp8.safetensors"),
    "bf16": (f"{_BFL}-base-9B", "flux-2-klein-base-9b.safetensors")}
COMFY_DIFFUSION_FILES["black-forest-labs/FLUX.2-klein-9B"] = {
    "fp8_scaled": (f"{_BFL}-9b-fp8", "flux-2-klein-9b-fp8.safetensors"),
    "bf16": (f"{_BFL}-9B", "flux-2-klein-9b.safetensors")}
COMFY_DIFFUSION_QUANTS["flux2_klein"] = ("fp8_scaled", "bf16")
COMFY_TEXT_ENCODER_FILES["flux2_klein"] = {"fp8_scaled": "split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors",
                                           "bf16": "split_files/text_encoders/qwen_3_8b.safetensors"}
COMFY_TEXT_ENCODER_QUANTS["flux2_klein"] = ("fp8_scaled", "bf16")
COMFY_VAE_FILES["flux2_klein"] = "split_files/vae/flux2-vae.safetensors"
# The format the config's default (int8_convrot) means for a family that has no such file
COMFY_DEFAULT_QUANT = {"flux2_klein": "fp8_scaled"}

# Ideogram 4 (ideogram-ai/ideogram-4-fp8, gated; the unsloth/ideogram-4-fp8 mirror is not): no bf16 release, the fp8
# weights are the source. Full sampling is Ideogram's V4_DEFAULT_20 preset, 20 steps at guidance 7 (its g * cond +
# (1 - g) * uncond is uncond + cfg * (cond - uncond) with cfg = g), the last 2 at 3; the turbo LoRA is ostris'
# TurboTime (rank 128, 847 MB; "good quality in as little as 2 steps with no CFG and no unconditional model", its
# samples at 8 steps), 8 steps at CFG 1. Comfy-Org's repackage: diffusion models in int8_convrot (9.58 GB), fp8_scaled
# (9.28 GB) and nvfp4_mixed (5.49 GB), its Flux2 VAE (0.34 GB); its Qwen3-VL-8B text encoder is the same model as
# Qwen-Image 2.1's, whose int8_convrot (9.35 GB) and bf16 (17.5 GB) files come from Comfy-Org/Qwen-Image-2.1, the
# fp8_scaled (10.6 GB) and nvfp4 (6.3 GB) ones from Comfy-Org/Ideogram-4.
MODEL_PRESETS["Ideogram-4"] = "ideogram-ai/ideogram-4-fp8"
ARCH_NAMES += ("ideogram4",)
ARCH_DEFAULT_TARGETS["ideogram4"] = _CORE_TARGETS
ARCH_FULL_SAMPLE["ideogram4"] = (20, 7.0)
ARCH_TURBO["ideogram4"] = {"repo": "ostris/ideogram_4_turbotime_lora", "file": "ideogram_4_turbotime_v1.safetensors",
                           "strength": 1.0, "steps": 8, "cfg": 1.0, "shift": 5.0}
ARCH_SAMPLE_DEFAULTS["ideogram4"] = (ARCH_TURBO["ideogram4"]["steps"], ARCH_TURBO["ideogram4"]["cfg"])
COMFY_REPOS["ideogram4"] = "Comfy-Org/Ideogram-4"
COMFY_DIFFUSION_FILES["ideogram-ai/ideogram-4-fp8"] = "diffusion_models/ideogram4_{quant}.safetensors"
COMFY_DIFFUSION_FILES["unsloth/ideogram-4-fp8"] = "diffusion_models/ideogram4_{quant}.safetensors"
COMFY_DIFFUSION_QUANTS["ideogram4"] = ("int8_convrot", "fp8_scaled", "nvfp4_mixed")
COMFY_TEXT_ENCODER_FILES["ideogram4"] = "text_encoders/qwen3vl_8b_{quant}.safetensors"
COMFY_TEXT_ENCODER_QUANTS["ideogram4"] = ("int8_convrot", "bf16", "fp8_scaled", "nvfp4")
COMFY_TEXT_ENCODER_REPOS["ideogram4"] = {"int8_convrot": "Comfy-Org/Qwen-Image-2.1", "bf16": "Comfy-Org/Qwen-Image-2.1"}
COMFY_VAE_FILES["ideogram4"] = "vae/flux2-vae.safetensors"
# MiniMax H3 (Comfy-Org/MiniMax-H3, sizes from the Hub): the pruned files replace the ~13B AdaLN parameters by a
# precomputed curve table: pruned_int8_convrot 21.0 GB (the one for 16 GB cards), pruned_bf16 40.2 GB, int8_convrot
# 34.0 GB, bf16 66.3 GB (full files are pruned on load, h3/loading.py); Qwen3-VL-32B cut to 50 layers: int8_convrot
# 27.1 GB, nvfp4_awq 15.7 GB, bf16 51.5 GB; the video VAE fp16 5.2 GB (+ the audio VAE, 0.6 GB, fetched by h3/loading)
COMFY_DIFFUSION_QUANTS["minimax_h3"] = ("pruned_int8_convrot", "pruned_bf16", "int8_convrot", "bf16")
COMFY_TEXT_ENCODER_FILES["minimax_h3"] = "text_encoders/qwen3vl_32b_minimax_h3_{quant}.safetensors"
COMFY_TEXT_ENCODER_QUANTS["minimax_h3"] = ("int8_convrot", "nvfp4_awq", "bf16")
COMFY_VAE_FILES["minimax_h3"] = "vae/minimax_h3_video_vae_fp16.safetensors"

# P-H3-Image (PrunaAI): MiniMax H3 fine-tuned into an 8-step, guidance-free text-to-image model (rulztrainer.ph3).
# PrunaAI/P-H3-Image-Comfy (sizes from the Hub): the pruned transformer with Pruna's LoRA merged, int8_convrot 21.2 GB,
# bf16 40.5 GB (w4a8, 13.4 GB, reads as a file you set: decoded to int8, rulztrainer.quant); the Qwen3-VL-4B text
# encoder with its text connector, int8_convrot 5.1 GB, bf16 9.1 GB; the single-frame VAE, bf16 5.6 GB (its int8 file,
# 3.1 GB, is read too).
# Training adapter: ostris' P-H3-Image adapter (rank 16, 0.16 GB), what ai-toolkit trains P-H3-Image with by default:
# the model is step- and guidance-distilled, so a LoRA trained on it bare drifts the way H3's do
MODEL_PRESETS["P-H3-Image"] = "PrunaAI/P-H3-Image"
ARCH_NAMES += ("p_h3_image",)
ARCH_DEFAULT_TARGETS["p_h3_image"] = _CORE_TARGETS
ARCH_FULL_SAMPLE["p_h3_image"] = (8, 1.0)  # the checkpoint's own 8-sigma grid, no guidance (no turbo LoRA needed)
ARCH_TRAINING_ADAPTER["p_h3_image"] = {"repo": "ostris/minimax_h3_training_adapter",
                                       "file": "p_h3_image_training_adapter_v1.safetensors"}
COMFY_REPOS["p_h3_image"] = "PrunaAI/P-H3-Image-Comfy"
COMFY_DIFFUSION_FILES["PrunaAI/P-H3-Image"] = "diffusion_models/p-h3-image_{quant}.safetensors"
COMFY_DIFFUSION_QUANTS["p_h3_image"] = ("int8_convrot", "bf16")
COMFY_TEXT_ENCODER_FILES["p_h3_image"] = "text_encoders/p-h3-image_text_encoder_{quant}.safetensors"
COMFY_TEXT_ENCODER_QUANTS["p_h3_image"] = ("int8_convrot", "bf16")
COMFY_VAE_FILES["p_h3_image"] = "vae/p-h3-image_vae_bf16.safetensors"

# MiniMax-H3-Image (MiniMaxAI, November 2026): H3's transformer without the audio stream, distilled to 6 steps with one
# output head per step and no guidance, with its own image VAE (rulztrainer.h3image). Comfy-Org/MiniMax-H3-Image
# (sizes from the Hub): the pruned transformer as int8_convrot 21.0 GB, bf16 40.2 GB or w4a8 12.5 GB (ComfyUI's 4-bit
# format, decoded on read to the int8 weight it stands for: a smaller download, int8's memory); the image VAE fp16
# 1.2 GB. The text encoder is MiniMax H3's, the same files byte for byte (the Hub's sha256), so they come from
# Comfy-Org/MiniMax-H3 and a copy already downloaded for H3 is reused.
# Training adapter: ostris' MiniMax-H3-Image adapter (rank 16, 0.16 GB, the port's names, no alpha), what ai-toolkit
# trains the model with by default since its commit 5b1670a (10 Nov 2026): the model is step- and guidance-distilled.
# ai-toolkit trains it on the 6 sampling sigmas only (archs.MiniMaxH3ImageArch.reference_sigmas)
MODEL_PRESETS["MiniMax-H3-Image"] = "MiniMaxAI/MiniMax-H3-Image"
ARCH_NAMES += ("minimax_h3_image",)
ARCH_DEFAULT_TARGETS["minimax_h3_image"] = _CORE_TARGETS
ARCH_FULL_SAMPLE["minimax_h3_image"] = (6, 1.0)  # the checkpoint's own 6 sigmas and heads, no guidance
ARCH_TRAINING_ADAPTER["minimax_h3_image"] = {"repo": "ostris/minimax_h3_training_adapter",
                                             "file": "minimax_h3_image_training_adapter_v1.safetensors"}
COMFY_REPOS["minimax_h3_image"] = "Comfy-Org/MiniMax-H3-Image"
COMFY_DIFFUSION_FILES["MiniMaxAI/MiniMax-H3-Image"] = "diffusion_models/minimax_h3_image_{quant}.safetensors"
COMFY_DIFFUSION_QUANTS["minimax_h3_image"] = ("int8_convrot", "bf16", "w4a8")
COMFY_QUANT_NAMES["minimax_h3_image"] = {"int8_convrot": "pruned_int8_convrot", "bf16": "pruned_bf16",
                                         "w4a8": "pruned_w4a8"}  # every file of it is the pruned-AdaLN one
COMFY_TEXT_ENCODER_FILES["minimax_h3_image"] = COMFY_TEXT_ENCODER_FILES["minimax_h3"]
COMFY_TEXT_ENCODER_QUANTS["minimax_h3_image"] = COMFY_TEXT_ENCODER_QUANTS["minimax_h3"]
COMFY_TEXT_ENCODER_REPOS["minimax_h3_image"] = {q: COMFY_REPOS["minimax_h3"]
                                                for q in COMFY_TEXT_ENCODER_QUANTS["minimax_h3"]}
COMFY_VAE_FILES["minimax_h3_image"] = "vae/minimax_h3_image_vae_fp16.safetensors"

# PAN 2 (IFM, November 2026): the 29B video model, trained here on stills as one-frame videos (rulztrainer.pan2), the
# way IFM's own trainer mixes images in. IFM/PAN2 is a diffusers folder of about 81 GB (transformer 63, Qwen3.5-9B
# text encoder 15, Wan 2.2 VAE 2.7). Nobody publishes single files of it: scripts/pan2/convert_pan2.py makes them
# (int8_convrot 39.8 GB with the fp32 norm layers kept or 27.4 GB without, nvfp4 15.5 GB; text encoder 8.4 / 4.6 GB;
# VAE fp16 1.3 GB), so model_source "files" needs all three files set (COMFY_REPOS has no entry: nothing to fetch).
# Sampling: the released 50 steps at CFG 2.5 on shift 9, with its structured negative prompt. No turbo LoRA and no
# training adapter exist for it (the model is not distilled).
MODEL_PRESETS["PAN-2"] = "IFM/PAN2"
ARCH_NAMES += ("pan2",)
ARCH_DEFAULT_TARGETS["pan2"] = _CORE_TARGETS
ARCH_FULL_SAMPLE["pan2"] = (50, 2.5)


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
# the probe prompts a preset fills in when the config has none: (face, body) templates with {trigger}, taken from the
# probe prompt presets (promptcheck.PRESETS, where the Train tab's preset lists come from too)
PRESET_PROBE_PROMPTS = {
    "identity": (_promptcheck.preset(_promptcheck.CLASSIC["face"]).template,
                 _promptcheck.preset(_promptcheck.CLASSIC["body"]).template),
}


MEGAPIXEL_CHOICES = [round(0.25 * i, 2) for i in range(1, 17)]  # 0.25 .. 4.0 MP


def mp_to_side(mp: float) -> int:
    """Megapixels -> square-equivalent side in pixels (1 MP = 1024x1024)."""
    return int(round(math.sqrt(float(mp) * 1024 * 1024)))


def side_to_mp(side: int) -> float:
    return round(side * side / (1024 * 1024), 2)


DATASET_KEYS = ("path", "resolutions", "repeats", "trigger", "class", "kind")
# the values the trainer acts on (trainer.build_scheduler, sample_sigmas, loss_weight): anything else is refused at
# start, not taken as constant / failing at step 1 after the caching
LR_SCHEDULERS = ("constant", "cosine", "linear")
TIMESTEP_SAMPLINGS = ("logit_normal", "sigmoid", "uniform", "identity", "content", "style", "logsnr", "reference",
                      "shift")
LOSS_WEIGHTINGS = ("none", "min_snr", "sigma_sqrt")



BODY_SCORES = ("appearance", "anchored", "proportions", "skeleton3d", "smplx")

@dataclass(frozen=True)
class DatasetSpec:
    """One dataset of a run: its folder (sub-folders included), its own training resolutions in megapixels (empty:
    the run's) and how many times each of its images comes up per epoch."""
    path: str
    resolutions: tuple = ()
    repeats: int = 1
    trigger: str = ""  # this dataset's trigger word (empty: the run's trigger_word)
    subject_class: str = ""  # LoRA Isolation: what the subject is ("man", "a woman"); empty: read from the captions
    subject_kind: str = "auto"  # LoRA Isolation: auto | person | thing | style


def soup_strength_list(text: str) -> list[float]:
    """soup_strengths ("1.0, 1.15, 1.3") as numbers, each above 0; 1.0 always among them."""
    vals = [float(x) for x in str(text or "").replace(";", ",").split(",") if x.strip()]
    if any(v <= 0 for v in vals):
        raise ValueError("soup_strengths must all be above 0")
    return sorted(set(vals) | {1.0})


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
    model_arch: str = "auto"  # auto | moss_image | fibo2 | krea2 | qwen_image21 (auto: the model's transformer config,
    #                           or the path) | flux3_klein
    model_source: str = "files"  # files: single ComfyUI-style safetensors files | hf: repo id / diffusers dir
    model_path: str = "OpenMOSS-Team/MOSS-Image-Base"  # HF repo id or local dir with transformer/ text_encoder/ vae/
    models_dir: str = "models"  # where HF repos are downloaded
    transformer_file: str = ""  # model_source=files: ComfyUI diffusion_models/*.safetensors (bf16, fp8, int8, nvfp4)
    text_encoder_file: str = ""  # model_source=files: ComfyUI text_encoders/*.safetensors (Qwen3.5-9B)
    vae_file: str = ""  # model_source=files: ComfyUI vae/*.safetensors (Flux2 VAE)
    tokenizer_path: str = ""  # model_source=files: dir with tokenizer_config.json; empty = auto-download
    # model_source=files, a file left empty: downloaded from Comfy-Org's ComfyUI repackage of the model (COMFY_*), in
    # this format: int8_convrot (default; half of bf16), bf16, fp8_scaled (Krea 2 only)
    model_file_quant: str = "int8_convrot"
    text_encoder_file_quant: str = "int8_convrot"
    device: str = "cuda:0"
    multi_gpu: str = "off"  # off | data_parallel (model copy per GPU, faster) | model_split (blocks across GPUs, memory only)
    devices: str = "cuda:0,cuda:1"  # GPUs for multi_gpu; the first is the primary
    block_swap: int = 0  # DiT blocks whose frozen weights stay in RAM and stream to the GPU as they run (blockswap.py)
    # torch.compile with block swap (experimental, untested): each block's forward is compiled and the swap hooks run
    # outside it, as OneTrainer does with its offloading. Off: block swap turns compile off.
    block_swap_compile: bool = False
    # checkpointed blocks' saved inputs wait in pinned RAM until the backward (experimental, untested; actoffload.py):
    # frees about tokens x hidden x 2 bytes per checkpointed block for more kept blocks or less block swap
    activation_offload: bool = False
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
    attention_backend: str = "auto"  # DiT attention kernel: auto (cuDNN on RTX 40 / 50 and B200, else PyTorch's pick) |
    #   torch (PyTorch picks) | cudnn | flash | efficient | math

    # --- dataset ---
    dataset_dir: str = "datasets/my_dataset"  # the dataset folder when `datasets` lists none (single-dataset configs)
    # several datasets: [{"path": folder, "resolutions": [MP, ...] (empty: the run's), "repeats": N (times each image
    # comes up per epoch, default 1)}, ...]; empty = dataset_dir alone
    datasets: list[dict] = field(default_factory=list)
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
    cache_dir: str = ""  # defaults to <dataset>/.rulz_cache per dataset (the run's own files: the first dataset's)
    recache: bool = False

    # --- lora ---
    lora_rank: int = 16
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    lora_targets: list[str] = field(default_factory=list)  # target groups; [] = the family's (ARCH_DEFAULT_TARGETS)
    lora_blocks: str = ""  # DiT blocks to train, e.g. "0-41" (all), "21-41", "10-31,40", "even"; empty = all
    train_text_embedder: bool = False  # also LoRA the 2-block text projection inside the DiT
    resume_from: str = ""  # path to a previously exported LoRA .safetensors (weights only)
    # --- LoRA Isolation (isolation.py): keep the subject(s) off everyone else in a picture ---
    isolation: bool = False  # DOP + fused triggers + side-by-side pairs with strangers + balanced subjects
    isolation_class: str = ""  # what a single dataset_dir subject is (datasets: each entry's "class"); empty: captions
    isolation_kind: str = "auto"  # likewise its kind (datasets: each entry's "kind"): auto | person | thing | style
    dop_weight: float = 2.0  # DOP: weight of the trigger-free pass, held to the model without the LoRA
    dop_every: int = 2  # DOP: a trigger-free pass every N steps (Fizgig: every 2nd at weight 2)
    isolation_pairs: int = 16  # side-by-side pictures per person subject (a training photo next to a stranger)
    isolation_pair_weight: float = 2.0  # their loss weight
    isolation_balance: bool = True  # repeat smaller subject datasets so each trains about as often as the largest
    # --- full fine-tune (finetune.py): the model's own Linears by rotation, saved as extracted LoRAs ---
    finetune: bool = False  # fine-tune the targets' weights themselves instead of a LoRA (lora_rank etc. unused)
    finetune_rotate_epochs: float = 1.0  # epochs each window trains before the next (a cycle trains every window)
    finetune_window_gb: float = 0.0  # VRAM a window may hold (fp32 weights, grads, optimizer state); 0 = auto
    finetune_reserve_gb: float = 0.0  # VRAM kept free of the windows for activations; 0 = 3 GB + 3 GB per megapixel
    finetune_extract_ranks: list[int] = field(default_factory=lambda: [64])  # the LoRA rank(s) each save extracts
    finetune_save_delta: bool = False  # also save the deltas (<job>_finetune.safetensors, 2 bytes per weight)
    finetune_resume: str = ""  # a delta file to continue a fine-tune from
    finetune_anchor: float = 0.01  # a fine-tune's weights decay toward the original model's (x lr per step; 0 = off).
    #                                 Replaces weight_decay, which would shrink the model's own weights toward zero
    lora_save_dtype: str = "bf16"  # bf16 | fp16 | fp32 for the exported LoRA files
    lora_ema: float = 0.98  # weight average (EMA) decay per image for saves, samples and probes; 0 = off (see ema.py)
    save_state: bool = True  # also save optimizer/scheduler/RNG state (resumable), and the final state with it
    save_state_every: int = 0  # steps between state saves; 0 = same as save_every
    keep_states: int = 1  # rolling number of state folders to keep; 0 = keep all
    resume_state: str = ""  # path to a state_XXXXXX directory to resume training from
    compile_model: bool | str = "auto"  # torch.compile the DiT blocks: true | false | auto (runs of 100+ steps on a GPU
    #                                     with triton; measured 26% faster per step on MOSS-Image, 12% on Krea 2)
    compile_cache: str = "auto"  # compiled kernels kept across machines: auto (<models_dir>/.rulz_compile, also read
    #                              from models/bucket/.rulz_compile) | a folder | off. Saves a fresh vast.ai rental's
    #                              cold compile (one file per torch / triton / GPU model / Python / OS)

    # --- optimization ---
    steps: int = 2000
    batch_size: int = 1
    gradient_accumulation: int = 1
    learning_rate: float = 1e-4
    optimizer: str = "adamw"  # adamw | adamw8bit | adafactor | automagic3 | automagic4 | keel (Automagic 4 for
    #                           character LoRAs: docs/keel-optimizer.md) | keel_fb (Keel for the face and the body
    #                           alike: the body share in the token gain, the magnitude vote in the consensus gate)
    #                           | hull | muon (orthogonalized momentum, step matched to AdamW's: AdamW's rates apply)
    #                           | normuon (muon with each row's step evened out; experimental, untested)
    weight_decay: float = 0.01
    automagic_min_lr: float = 1e-8  # automagic3/4: hard floor for the adapted lr
    automagic_max_lr: float = 1e-3  # automagic3/4: hard ceiling for the adapted lr (about 10x the start lr)
    automagic_polarity_history: int = 8  # automagic3/4: sign-history window (2-64)
    automagic_clip_threshold: float = 1.0  # automagic3/4: update trust region
    automagic_groups: str = "single"  # automagic3/4: single (one adaptive lr) | by_type (text / attention / feed-forward
    #                                   / other layers each adapt their own; Fizgig's setup for Krea 2's text fusion)
    automagic_sensitivity_quantile: float = 0.9  # automagic4: weights more sensitive (gradient power) than this share
    #                                              of the LoRA's elements take a smaller step, (0, 1]
    automagic_vote: str = "graded"  # automagic4: graded (mean update sign squared minus the flip rate, zero on noise;
    #                                 raises the lr on a trend, lowers it on overshoot) | extremes (Automagic 3's rule,
    #                                 a third of the gain)
    automagic_max_lr_change: float = 0.05  # automagic4: the lr moves at most this fraction per step (0.05 = 5%): bounds
    #                                        the climb over the sign window's lag
    keel_gauge: bool = True  # keel: the r x r gauge preconditioner on each LoRA pair's gradient (lora_up by the down's
    #                          Gram, lora_down by the up's), so neither factor takes the whole update
    keel_gauge_eps: float = 0.01  # keel: the preconditioner's floor as a fraction of the larger Gram's mean diagonal
    keel_cautious: bool = True  # keel: zero a preconditioned gradient element that points against the raw gradient
    keel_pair_sigma: bool = True  # keel: one sigma per step for all of its micro-batches (and the DOP pass), so the
    #                               step's gradient sits in one noise band
    keel_bands: int = 8  # keel: noise bands over logit(sigma) in [-4, 4]; a step's gradient enters the second moment
    #                      at the tensor's overall scale whatever its band, so the quiet bands are not drowned; 1 = off
    keel_band_weights: str = ""  # keel: a multiplier on the step's update per band: "" = 1 everywhere, identity = a
    #                              hump over the face bands (sigma 0.12 to 0.5), subject = a plateau over the face's
    #                              and the body's bands alike (sigma 0.18 to 0.82), or keel_bands comma-separated numbers
    keel_consensus: float = 0.5  # keel: the cross-photo consensus gate, 0 = off .. 1 = a hard AND-mask: the update is
    #                              scaled by (1 - c) + c * (agreement share)^power, the share being how many of the
    #                              step's gradients (micro-batches, the DOP pass) agree on the element's sign
    keel_consensus_power: float = 2.0  # keel: the exponent on the agreement share
    keel_consensus_mode: str = "auto"  # keel: how the step's gradients vote on an element: sign (one vote each; 1
    #                                   byte per weight for the step) | magnitude (each weighted by its size, |sum g|
    #                                   / sum |g|, so a photo that says little about a weight barely votes; 4 bytes
    #                                   per weight) | auto (sign for keel, magnitude for keel_fb)
    keel_lookahead: bool = False  # keel: probe-gated lookahead: the weights that beat the anchor's smoothed likeness
    #                               score by the noise band become the anchor, and after keel_lookahead_grace probes
    #                               without a gain the weights are pulled back toward it (needs likeness probes)
    keel_lookahead_pullback: float = 0.5  # keel: how far a pull moves the weights toward the anchor (1 = all the way)
    keel_lookahead_lr: float = 0.7  # keel: the learning-rate multiplier at each pull (within the Automagic rails)
    keel_lookahead_grace: int = 2  # keel: probes without a gain before a pull (the count starts over after one)
    keel_face_gain: float = 0.0  # keel: face-token gain: a LoRA pair's step is scaled by (the share of its gradient
    #                              from the face's tokens / the median pair's share) ** this; needs region maps with
    #                              faces (face detection); 0 = off
    keel_face_gain_max: float = 2.0  # keel: the gain stays within [1 / max, max]
    keel_face_gain_beta: float = 0.9  # keel: EMA of the gain over steps (one batch without a face barely moves it)
    hull_rho: float = 0.0  # hull (docs/hull-optimizer.md): the step, a relative change of each layer's weight per step;
    #                        required, and the trainer's schedule acts on it (hull is not self-tuning)
    hull_betas: list[float] = field(default_factory=lambda: [0.9, 0.99])  # hull: momentum, second moment
    hull_state_dtype: str = "fp32"  # hull: the momentum's dtype; bf16 is ignored with a warning (trainer.build_optimizer)
    hull_clip: float = 5.0  # hull: per-channel RMS clip in the balanced frame (it replaces max_grad_norm)
    hull_factor_cap: float = 0.2  # hull: a step moves a factor by at most this share of its balanced norm
    hull_safety_cap: float = 0.0  # hull: the realised step's cap in steps of rho; 0 = 2 x max(1, hull_gain_max)
    hull_gain_max: float = 2.0  # hull: the largest region gain, and with it the safety cap
    hull_regions: str = "off"  # hull: off | parts (body cells get hull_gain_body in the update: design sections 5.1-5.3)
    hull_gain_body: float = 1.5  # hull_regions parts: body skin and the silhouette ring, flat across noise levels
    hull_gain_background: float = 1.0  # hull_regions parts: at most 1; face, text, hair and clothing stay at 1
    hull_ring_cells: int = 1  # hull_regions parts: the silhouette ring's half width in 16 px cells
    hull_face_safe: bool = True  # hull_regions parts: the gained excess cannot undo the face's part of a step (C3)
    hull_retreat: float = 0.5  # hull_regions parts: a tensor whose gains oppose its gradient for a probe interval
    #                            has its excess multiplied by this; it never rises
    hull_ns_iters: int = 30  # hull: Newton-Schulz iterations per matrix root
    hull_damping: float = 1e-6  # hull: the Grams' damping, as a fraction of their mean eigenvalue
    keel_body_weight: float = 1.0  # keel_fb: the body tokens' weight against the face's in the token gain (the
    #                                 person's cells outside the face box: needs person masks); 1 = equal, 0 = face only
    lr_scheduler: str = "constant"  # constant | cosine | linear
    warmup_steps: int = 50
    max_grad_norm: float = 1.0
    timestep_sampling: str = "logit_normal"  # logit_normal | sigmoid | uniform
    logit_mean: float = 0.0
    logit_std: float = 1.0
    sigmoid_scale: float = 1.0  # sigmoid mode: sigma = sigmoid(randn * sigmoid_scale)
    logsnr_mean: float = -6.0  # logsnr mode (style-friendly, arXiv 2411.14793): log-SNR ~ N(mean, std), very noisy
    logsnr_std: float = 2.0  #   steps where style is learned; the paper recommends -6 and 2 or 3
    timestep_shift: float = 1.0  # sigma shift applied to sampled sigmas (training_shift is 1.0 upstream)
    timestep_buckets: int = 0  # 0 = off; N (2+): every N sigma draws take one from each of N equal slices of the
    #                            sampling curve (shape unchanged), so small runs cover it evenly (musubi-tuner)
    clean_share: str = ""  # share of steps below sigma 0.5 by epoch (clean_share.py): "" = the sampler's own,
    #                        "40", "75:5:40", or Fizgig's MOSS Slide "60:7:30:15:2"
    loss_weighting: str = "none"  # none | min_snr (sigma-based) | sigma_sqrt
    loss_type: str = "mse"  # mse | band (experimental, AI-Toolkit: frequency bands the noise has buried drop out)

    # --- saving / sampling ---
    save_every: int = 250
    keep_checkpoints: int = 0  # rolling number of the step checkpoints (LoRA / LoKr files) to keep; 0 = keep all
    sample_every: int = 250
    sample_prompts: list[str] = field(default_factory=list)
    sample_width: int = 1024
    sample_height: int = 1024
    sample_steps: int = 12  # the default MOSS-Image turbo LoRA's (ARCH_TURBO); each family has its own
    sample_cfg: float = 1.0
    sample_shift: float = 5.0
    sample_seed: int = 42
    sample_walk_seed: bool = True  # seed + i for the i-th sample prompt; off = same seed for every prompt
    sample_before_training: bool = False  # generate samples at step 0 before the first training step
    sample_negative: str = ""
    sample_skip_layer: int = 22  # negative-branch skip layer; -1 disables
    turbo_lora: str = "auto"  # a LoRA (e.g. an 8-step distill) applied only while sampling: auto = the model family's
    #                            turbo LoRA (ARCH_TURBO, downloaded), "" = none, or a .safetensors path
    turbo_lora_strength: float = 1.0
    sample_lora: str = "auto"  # a second ComfyUI-format LoRA applied only while sampling (with the turbo LoRA): auto =
    #   the family's (ARCH_SAMPLE_LORA, downloaded) while samples render its base checkpoint (MOSS-Image Base: the
    #   SFT LoRA), "" = none, or a .safetensors path
    sample_lora_strength: float = 1.0
    training_adapter: str = "auto"  # a LoRA on while training and off while sampling, never saved (ARCH_TRAINING_ADAPTER;
    #   Qwen-Image 2.1: Fizgig's): auto = the model family's (downloaded; the others have none), "" = none, or a path
    training_adapter_strength: float = 1.0
    # --- face similarity (character LoRAs) ---
    face_refs: list[str] = field(default_factory=list)  # dataset images whose largest face is the reference identity
    face_provider: str = "cpu"  # cpu | cuda (onnxruntime provider for InsightFace)
    face_det_size: int = 640
    body_refs: list[str] = field(default_factory=list)  # dataset images (full body, neutral pose) for proportions
    body_score: str = "appearance"  # appearance (DINOv2 of the person crop, best match against the references) |
    #   anchored (SigLIP 2 + DINOv2 of the body window with its photographic look taken out: 0 = the step-0 probe's
    #   base-model renders, 100 = the references' mean; bodyanchor.py) |
    #   proportions (2D skeleton ratios from body_backend, + SAM 3 silhouette widths with mediapipe+sam3) |
    #   skeleton3d (3D bone ratios, + silhouette widths where upright and frontal; bodyshape.py) |
    #   smplx (NLF's SMPL-X body shape measured in the T-pose; its model is downloaded by hand, see nlf_model)
    nlf_model: str = ""  # body_score smplx: NLF's TorchScript file or its folder; empty = models/nlf/
    body_backend: str = "mediapipe+sam3"  # mediapipe+sam3 (SAM 3 person masks too) | mediapipe (skeleton only)
    sam3_model: str = "facebook/sam3"  # HF repo id or local transformers-format directory
    body_device: str = "cuda:0"  # where SAM 3 runs while scoring (moved back to CPU afterwards)
    face_weight: float = 0.7  # weights for the combined score
    body_weight: float = 0.3
    # --- per-image loss watch ---
    loss_watch: bool = False  # track every image's timestep-normalised loss and give epoch verdicts
    per_image_lr: bool = False  # act on verdicts: stuck x0.5/0.25/0.125, exhausted x0.6 (not with automagic3/4)
    loss_watch_buckets: int = 40  # sigma buckets for loss normalisation
    loss_watch_beta: float = 0.9  # per-image residual EMA
    loss_watch_high: float = 1.3  # residual above median x this = hard
    loss_watch_low: float = 0.7  # residual below median x this = easy / learned
    loss_watch_improve_tol: float = 0.03  # relative epoch-to-epoch drop that counts as improving
    loss_watch_healthy_mult: float = 1.0  # lr multiplier for the healthy cohort (Fizgig uses 1.1)
    loss_watch_exclude_after: int = 0  # epochs at the lowest throttle before an image is excluded; 0 = never
    # --- region watch (regionwatch.py): how far the face, hair, skin, clothes and background have been learned ---
    region_watch: bool = False  # observe only: the training steps' own losses split by region (needs mediapipe; not
    #                             in a race)
    # --- acting on the parts maps: experimental, never run in a real training (regionwatch.py) ---
    region_class_weights: str = ""  # one fixed loss weight per region, "face=3,hair=1.5,skin=1.5,clothes=0.5,
    #                                 background=0.3" (left out = 1; mean 1 per sample); "" = off. Not with
    #                                 region_weights or hull_regions parts
    region_steer: bool = False  # region steering: the region watch (turned on with it) moves loss weights once an
    #                             epoch by the rules below, each off at 0. Not in a race
    region_steer_regions: str = "background,clothes"  # what the cap and ahead rules may turn down
    region_steer_cap: float = 0.6  # cohort rule: a steered region's learned share, at most this x the subject's
    region_steer_ahead: float = 0.7  # per-picture rule: turned down when its contrast is under this x the median
    region_steer_boost: float = 0.0  # cohort rule: a lagging face / hair / skin may be raised up to this (1.5, say)
    region_steer_prior: float = 0.0  # cohort rule: while the background is past its cap, pull its prediction towards
    #                                  the base model's with up to this weight (one more forward a step while it acts)
    region_steer_min: float = 0.3  # a region's lowest loss weight (0 = it may be masked out altogether)
    region_steer_step: float = 0.2  # the most a weight moves per epoch, as a share of itself (back up half as fast)
    region_steer_strength: float = 1.0  # how hard a rule pulls: the weight aimed for is (share within limit) ^ (2 x this)
    region_steer_signal: str = "steps"  # what the cap rule's per-picture share and the ahead rule read: steps (the
    #                                     training steps' own, noisy) | sweep (the fixed-noise sweeps')
    region_steer_subject_ahead: float = 0.0  # per-picture rule, needs sweeps of the training pictures: a picture's
    #                                          face / hair / skin with less than this x the median picture's error
    #                                          left (0.7, say) is turned down in that picture; 0 = off
    region_steer_holdout: float = 0.0  # cohort rule, needs holdout_images: a region learned this much further on the
    #                                    training pictures than on the held-out ones (0.15, say), and no longer
    #                                    improving on those, is turned down in every picture; 0 = off
    # --- region sweep (regionsweep.py): the loss per region at fixed noise, and photos kept out of training.
    #     Experimental, never run in a real training; 3 forward passes per picture per sweep ---
    region_sweep_every: int = 0  # epochs between sweeps; 0 = off (with holdout_images: every epoch)
    region_sweep_pictures: int = 0  # training pictures per sweep: 0 = all, K = K in turn, -1 = none (held-out only)
    holdout_images: int = 0  # photos kept out of training and swept: what is learned of the subject, not of the photos
    holdout_track_best: bool = False  # keep <job>_holdout_best.safetensors at the best held-out subject score
    holdout_patience: int = 0  # stop after this many sweeps without a better held-out subject score; 0 = never
    # --- training maps (trainmaps.py): heatmaps over a few training pictures, before training and about each epoch ---
    training_maps: bool = False  # observe only: nothing about training changes
    training_maps_images: int = 3  # pictures (at least one per dataset, so more with several datasets)
    training_maps_every: int = 0  # steps between captures; 0 = one epoch's worth of steps
    training_maps_trigger_off: bool = False  # also map each picture with the trigger word taken out of its caption:
    #                                          what the LoRA changes unasked ("bleed"; 2 more forwards per noise level)
    # --- look consistency filter (Dataset tab): ease the scan's outliers in ---
    look_warmup: bool = False  # images under the look filter's fence (look_scores.json) start at a lower loss weight
    look_warmup_start: float = 0.4  # their loss weight at the start, rising linearly to 1
    look_warmup_epochs: float = 4.0  # epochs (of their dataset) to reach full weight
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
    lora_plus_ratio: float = 0.0  # lr of lora_up = lr x ratio (0/1 = off, 16 = LoRA+ paper); not with automagic3/4
    network_type: str = "lora"  # lora | lokr (LyCORIS Kronecker factors: w1 up to lokr_factor square, w2 the rest) |
    #                             oft (orthogonal fine-tuning: each layer's output channels rotated in blocks)
    lokr_factor: int = 8  # LoKr: w1 is [a, b] with a, b the largest divisors of out / in <= this (8: Fizgig's default)
    oft_block_size: int = 32  # OFT: rotation block size (the largest divisor of each layer's outputs <= this)
    oft_constraint: float = 0.0  # OFT: cap on the rotation's size (COFT; the file's .alpha); 0 = none
    # OFT: an lr per block size, sqrt(31 / (block - 1)) x learning_rate, so a block of any size moves about as far per
    # step as a block of 32 (OneTrainer's Scaled OFT, as an lr factor: the saved file is unchanged). Off: one lr, and
    # bigger blocks step further (64: about 2x 16's). Experimental; not with the self-tuning optimizers or hull.
    oft_scaled_lr: bool = False
    dora: bool = False  # DoRA on a LoRA or LoKr: the weight split into a trained magnitude and a direction
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
    # --- trigger attention and identity alignment: training-only losses read inside the model (alignment.py) ---
    attn_localize: bool = False  # keep the trigger word's attention on the subject (person mask + face box)
    attn_localize_weight: float = 0.05
    attn_localize_layers: str = ""  # blocks it is read in, e.g. "14,21"; "" = auto (FIBO: its injection blocks)
    id_align: bool = False  # align the face tokens' states after one block to the photo's ArcFace embedding
    id_align_weight: float = 0.05
    id_align_layer: int = -1  # that block; -1 = auto (a third of the depth)
    body_align: bool = False  # the same for the build: the person's tokens -> the photo's DINOv2 body-window embedding
    body_align_weight: float = 0.05
    # --- likeness reward fine-tuning: an optional second stage after the steps (reward.py, after Flow-GRPO) ---
    reward_steps: int = 0  # optimizer steps of the stage (0 = off); it starts from the LoRA the steps saved
    reward_lr: float = 2e-5
    reward_group: int = 4  # images per prompt and step; their scores count relative to each other
    reward_prompts: list[str] = field(default_factory=list)  # empty: varied prompts built from the trigger word
    reward_sample_steps: int = 0  # denoising steps per image; 0 = probe_steps (guidance: probe_cfg)
    reward_train_steps: int = 4  # of those, the steps each update trains on (picked at random)
    reward_noise: float = 0.7  # the stochastic sampler's noise level (Flow-GRPO's)
    reward_kl: float = 0.04  # KL penalty to the LoRA as the stage started
    reward_size: int = 0  # image size (square; full-body prompts 2:3); 0 = probe_size
    # --- likeness-filtered self-training: the LoRA's own best pictures train beside the photos (selftrain.py) ---
    selftrain: bool = False
    selftrain_at: str = "0.5"  # fractions of the steps a round runs at, e.g. "0.5,0.75" (each replaces the last)
    selftrain_images: int = 32  # pictures rendered per round (sample_steps / sample_cfg, the largest resolution)
    selftrain_min_likeness: float = 60.0  # a kept picture scores at least this and the probes' smoothed score
    selftrain_share: float = 0.2  # at most this share of the real images (repeats counted) per round
    selftrain_weight: float = 0.5  # their loss weight
    selftrain_prompts: list[str] = field(default_factory=list)  # empty: the reward stage's varied prompts
    # --- copy check: SSCD similarity of renders to the training photos (copycheck.py) ---
    copy_check: str = "auto"  # auto = on with selftrain (its gate); on = with any probes too (logged); off
    copy_max: float = 0.5  # selftrain leaves out a picture above max(this, the photos' own best match + 0.1)
    # --- seed soup: average LoRAs of one subject, picked by the probes (soup.py; steps 0) ---
    soup_files: list[str] = field(default_factory=list)  # 2+ LoRA / LoKr files: a greedy soup instead of training
    soup_rank: int = 32  # the soup's rank (0 = exact, LoRA files only)
    soup_strengths: str = "1.0, 1.15, 1.3"  # strengths each candidate soup is probed at (an average is weaker)
    # --- noise check: how far the probes move with their seeds (likeness_stats.py; steps 0) ---
    noise_files: list[str] = field(default_factory=list)  # 1+ LoRA files probed at noise_seeds seeds instead of training
    noise_seeds: int = 16  # seeds per file (the same ones for every file: paired)
    # --- likeness probes & controller (character LoRAs) ---
    probe_every: int = 0  # steps between probe renders; 0 = off (needs face_refs and/or body_refs)
    probe_at_start: bool = True  # also probe at step 0 for a baseline
    probe_face_prompt: str = ""  # headshot prompt; empty = no face probe
    probe_body_prompt: str = ""  # full-body prompt; empty = no body probe
    probe_seeds: int = 1  # seeds per probe (scores averaged)
    probe_floor_seeds: int = 4  # the step-0 probe (the normalized scale's zero) renders at least this many seeds
    probe_seed: int = 1234
    probe_size: int = 512  # face probe is square; body probe is 2:3 portrait with the same pixel count
    probe_steps: int = 8
    probe_cfg: float = 1.0
    probe_ema: float = 0.5  # smoothing of probe scores (weight on the previous value)
    probe_async: bool = False  # render probes on a second GPU while training goes on (multi_gpu off, no race)
    probe_device: str = ""  # probe_async's GPU; empty = the first of devices that is not the training GPU
    # the prompt check (promptcheck.py): does a probe render still show the frame its prompt asks for? Observe only
    # unless likeness_best_min_kept or race_drop_drift act on it
    probe_subject: str = ""  # what the preset prompts call the person ("a young woman with long dark brown hair");
    #                          empty = a dataset's subject_class, else "a person"
    probe_face_expect: str = ""  # the face probe's frame: "" = guessed from the prompt | none | headshot | upper | full
    probe_body_expect: str = ""  # the body probe's, the same way
    probe_checks: list = field(default_factory=list)  # extra prompts rendered and checked at every probe, never
    #                 scored: a preset key (promptcheck.PRESETS) or {"name", "prompt", "expect", "kind": face | body}
    probe_check_seeds: int = 4  # seeds per extra prompt (the first of the probe seeds)
    likeness_noise_band: float = 1.0  # percent points a change must exceed to count
    likeness_normalize: bool = True  # score face and body on one scale (0 = a stranger, 100 = the references) first
    face_scale_span: float = 1.35  # the face's 100 sits this many times as far above a stranger as the references do
    face_scale_curve: float = 1.5  # the face score is its position on that scale raised to this power (1 = linear)
    body_scale_span: float = 1.35  # the same for the body: its 100 this many times as far above a stranger
    body_scale_curve: float = 1.5  # and its curve (1 and 1 = the linear body scale of v2.1.3 and before)
    #   (both unused by body_score anchored: its scale is linear from the step-0 probe to the references' mean)
    likeness_track_best: bool = True  # keep <job>_best.safetensors at the best smoothed score
    likeness_track_best_face: bool = False  # also keep <job>_best_face.safetensors at the best smoothed face score
    likeness_track_best_body: bool = False  # and <job>_best_body.safetensors at the best smoothed body score
    # a best checkpoint (overall, face, body) is replaced by any higher smoothed score, not only one that beats it by
    # the noise band; whether the run still improves (early stop, step-up) still needs the band
    likeness_best_any_rise: bool = False
    # a body probe render that no longer shows a body (the scorer finds none: a head-and-shoulders picture where a body
    # was asked for) counts as zero in the body score, against the share that showed one at the first probe, and one
    # render's score stops at the top of the scale. Off: the body score is the mean of the renders that still show a
    # body, so the last few of them, or a score left over from when some did, can set a best checkpoint (the
    # behaviour before 3.4.0; it happened in 8 of 9 test runs)
    likeness_framing: bool = True
    # when the training ends, write a report of it into the job folder's report/: a summary picture (settings, scores,
    # probe renders through the run with the best checkpoint marked, dataset examples, training maps) and a longer
    # PDF (runreport.py; needs matplotlib). Any earlier run: python -m rulztrainer.runreport output/<job>, or the
    # Previous Runs tab
    run_report: bool = False
    # experimental, unit-tested only: a probe whose renders keep their prompt's frame in less than this share (the
    # prompt check's kept share, against the step-0 probe) cannot set a new best checkpoint. 0 = off
    likeness_best_min_kept: float = 0.0
    # a second, held-out face recognizer on the probes (likeness.py): auto = FaceNet whenever the run trains against
    # ArcFace (reward_steps, id_loss_backend arcface, id_align), off, or facenet (always). Nothing ever trains on it,
    # so with face_gold_select it, not ArcFace, picks the best checkpoint, stops early and ranks a race
    face_gold: str = "auto"
    face_gold_select: bool = True
    likeness_patience: int = 0  # stop after N probes without improvement; 0 = never
    likeness_lr_control: bool = False  # bounded lr multiplier driven by the probe trend
    likeness_lr_min: float = 0.5
    likeness_lr_max: float = 2.0
    likeness_lr_step: float = 0.1  # relative change per decision
    likeness_curriculum: bool = False  # face/body dataset weighting + timestep bias from the score gap
    likeness_curriculum_strength: float = 1.0
    likeness_curriculum_framing: float = 0.0  # experimental, untested: a body probe render with no measurable body
    #                                           counts against the body for the curriculum, by this much (0 = off,
    #                                           1 = a probe where no render shows a body reads as a body score of 0)
    # configuration race (see race.py): the configurations train in turn, are probed, and the lowest is dropped
    race: bool = False
    race_configs: list[dict] = field(default_factory=list)  # [{"name": ..., <setting>: <value>, ...}], 2 or more
    race_prep_regions: bool = False  # set by the trainer on a race's preparation config: some configuration needs maps
    race_warmup_probes: int = 0  # probe rounds before the first configuration is dropped
    race_drop_lowest: bool = True  # after each round the lowest-scoring configuration leaves
    race_drop_confidence: float = 0.0  # > 0 (0.9): the lowest leaves only when a paired test on the shared probe
    #                                    seeds puts it below the leader with this confidence (race.py)
    race_max_close_rounds: int = 3  # with race_drop_confidence: rounds in a row too close to call before the lowest
    #                                 leaves anyway
    race_patience_drop: bool = False  # a configuration leaves after race_patience probes of its own without
    #                                   improvement; the winner is the best score anyone reached
    race_patience: int = 3
    race_patience_rising: bool = False  # with race_patience_drop: one out of patience stays while its smoothed score
    #                                     still rises from its previous probe (a chance to come back after a dip)
    # experimental, unit-tested only: a configuration whose probes have drifted from their prompts' frame (the prompt
    # check's drift share at or above this) for race_drop_drift_probes of its own probes in a row leaves the race.
    # 0 = off. The last one never leaves; when all of them are over it, the least drifted stays
    race_drop_drift: float = 0.0
    race_drop_drift_probes: int = 2
    race_follow: bool = False  # a relay: nobody is dropped; after each round everyone continues from its winner's
    #                            LoRA, optimizer state and likeness history (seed rows: the best of several paths)
    race_parallel: bool = False  # data_parallel race: one configuration per GPU at the same time (an odd one out
    #                              is split over the GPUs when its step has batches enough, else it runs on the first)
    sample_model_path: str = ""  # optional different diffusion model for sampling only: .safetensors, diffusers dir or repo id
    sample_model_quant: str = "auto"  # auto | none | int8 | int8_convrot | nvfp4

    def dataset_specs(self) -> list[DatasetSpec]:
        """The run's datasets: the `datasets` entries, else dataset_dir alone at the run's resolutions, once per epoch."""
        if not self.datasets:
            return [DatasetSpec(str(self.dataset_dir or "").strip())]
        out = []
        for d in self.datasets:
            if not isinstance(d, dict):
                d = {"path": str(d)}
            reps = d.get("repeats", 1)
            out.append(DatasetSpec(str(d.get("path") or "").strip(),
                                   tuple(float(x) for x in (d.get("resolutions") or [])),
                                   1 if reps is None else int(reps), str(d.get("trigger") or "").strip(),
                                   str(d.get("class") or "").strip(), str(d.get("kind") or "auto").strip() or "auto"))
        return out

    def run_sides(self) -> list[int]:
        """The run's own training resolutions as square-equivalent side lengths (area = side^2), ascending, unique."""
        if not self.resolutions:
            return [int(self.resolution)]
        return sorted({mp_to_side(mp) for mp in self.resolutions})

    def dataset_sides(self, spec: DatasetSpec) -> list[int]:
        """A dataset's training resolutions as sides: its own when it has any, else the run's."""
        return sorted({mp_to_side(mp) for mp in spec.resolutions}) if spec.resolutions else self.run_sides()

    def training_sides(self) -> list[int]:
        """Every resolution the run trains at, all datasets together, as square-equivalent side lengths (area =
        side^2), ascending and unique (the stages of resolution_step_up)."""
        return sorted({s for spec in self.dataset_specs() for s in self.dataset_sides(spec)})

    def arch_name(self) -> str:
        """The architecture this config trains: model_arch when set, else guessed from the model path or files."""
        if self.model_arch and self.model_arch != "auto":
            return self.model_arch
        return family_named_in(" ".join([self.model_path or "", self.transformer_file or ""])) or "moss_image"

    def samples_on_distilled_model(self) -> bool:
        """The model that renders samples and probes is a distilled checkpoint: sample_model_path, else the trained
        model (its file when it has one), by the family's naming rule (Arch.distilled_sample_model: a "turbo" name such
        as Krea-2-Turbo or Qwen-Image-2.1-Turbo; Klein without "base"). A turbo LoRA on top would distil it twice."""
        from .archs import get_arch

        if self.sample_model_path:
            name = self.sample_model_path
        else:
            name = self.transformer_file if self.model_source == "files" and self.transformer_file else self.model_path
        return get_arch(self).distilled_sample_model(name or "")

    def samples_on_base_model(self) -> bool:
        """Samples and probes render MOSS-Image Base (the trained model, no other sampling model), by name: what the
        family's sampling LoRA (ARCH_SAMPLE_LORA, Base -> SFT) is made for."""
        if self.arch_name() != "moss_image" or self.sample_model_path or self.finetune:
            return False
        name = (self.transformer_file if self.model_source == "files" and self.transformer_file
                else self.model_path).lower()
        return "base" in name and "sft" not in name

    def foreign_sampling_files(self) -> list[str]:
        """The sampling model and sampling-only LoRA files whose names belong to another model family than the one
        trained (a Krea 2 run's krea/Krea-2-Turbo left in sample_model_path after switching to MOSS-Image): as
        "field: value" lines."""
        family = self.arch_name()
        out = []
        for key in ("sample_model_path", "turbo_lora", "sample_lora", "training_adapter"):
            value = getattr(self, key) or ""
            named = sampling_file_family(key, value)
            if named is not None and named != family:
                out.append(f"{key}: {value} (a {named} file)")
        return out

    def trains_on_arcface(self) -> bool:
        """Something optimizes the probes' own ArcFace: the reward stage, the identity loss on its arcface backend, or
        identity alignment (it pulls the face tokens toward the photos' ArcFace embeddings)."""
        return bool(self.reward_steps > 0 or self.id_align
                    or (self.id_loss_weight > 0 and self.id_loss_backend == "arcface"))

    def trains_on_facenet(self) -> bool:
        return bool(self.id_loss_weight > 0 and self.id_loss_backend == "facenet")

    def gold_backend(self) -> str:
        """The held-out face recognizer the probes add ("" = none): face_gold facenet always; auto when the run trains
        against ArcFace and not against FaceNet, and for the noise check (it measures both recognizers)."""
        mode = str(self.face_gold or "auto").lower()
        if mode == "facenet":
            return "facenet"
        if mode == "auto" and (self.trains_on_arcface() or self.noise_files) and not self.trains_on_facenet():
            return "facenet"
        return ""

    def __post_init__(self) -> None:
        # YAML 1.1 and race rows read a bare on / off as true / false: the switches with an "off" mode take it back
        for name in ("face_gold", "copy_check"):
            if isinstance(getattr(self, name), bool):
                setattr(self, name, "on" if getattr(self, name) else "off")

    def copy_check_on(self) -> bool:
        """The SSCD copy check runs: copy_check on, or auto with self-training (it gates the kept pictures)."""
        mode = str(self.copy_check or "auto").lower()
        return mode == "on" or (mode == "auto" and bool(self.selftrain))

    def resolved_lora_targets(self) -> list[str]:
        """lora_targets, or the model family's default groups when it is empty."""
        return list(self.lora_targets) or list(ARCH_DEFAULT_TARGETS[self.arch_name()])

    @property
    def clip_norm(self) -> float:
        """The gradient norm clip in force: hull only measures the norm (a clip on factor gradients is gauge-dependent;
        it clips per channel itself)."""
        return 0.0 if self.optimizer.lower() == "hull" else self.max_grad_norm

    def validate_hull(self) -> None:
        if not self.hull_rho > 0:
            raise ValueError("optimizer hull needs hull_rho above 0 (a relative change of a layer's weight per step: "
                             "docs/hull-optimizer.md section 4.8)")
        if len(self.hull_betas) != 2 or not 0 <= self.hull_betas[0] < 1 or not 0 < self.hull_betas[1] < 1:
            raise ValueError("hull_betas must be [momentum in 0..1, second moment in 0..1]")
        if self.hull_state_dtype not in ("bf16", "fp32"):
            raise ValueError(f"hull_state_dtype must be bf16 or fp32, got {self.hull_state_dtype!r}")
        if self.hull_clip <= 0 or self.hull_factor_cap <= 0 or self.hull_safety_cap < 0 or self.hull_gain_max < 1:
            raise ValueError("hull_clip and hull_factor_cap must be above 0, hull_safety_cap at least 0 and "
                             "hull_gain_max at least 1")
        if self.hull_ns_iters < 1 or self.hull_damping <= 0:
            raise ValueError("hull_ns_iters must be at least 1 and hull_damping above 0")
        if self.network_type != "lora" or self.dora or self.finetune:
            raise ValueError("optimizer hull trains LoRA pairs: no LoKr, OFT, DoRA or fine-tune")
        if self.likeness_lr_control or self.likeness_curriculum:
            raise ValueError("optimizer hull: nothing driven by a score may raise its step, so likeness_lr_control and "
                             "likeness_curriculum are refused")
        if self.loss_watch_healthy_mult > 1:
            raise ValueError("optimizer hull: per-step multipliers stay at or below 1 (loss_watch_healthy_mult)")
        if self.id_loss_weight > 0 and self.id_loss_backend == "facenet":
            raise ValueError("optimizer hull keeps FaceNet as a held-out judge: id_loss_backend facenet trains on it")
        if self.reward_steps > 0 or self.selftrain:
            raise ValueError("optimizer hull refuses reward_steps and selftrain: a judge's score must not train")
        if self.race and (self.face_gold != "facenet" or self.copy_check != "on"):
            raise ValueError("optimizer hull in a race needs face_gold: facenet and copy_check: on (the guards that "
                             "decide whether a row stays)")
        if self.hull_regions not in ("off", "parts"):
            raise ValueError(f"hull_regions must be off or parts, got {self.hull_regions!r}")
        if not 0.5 <= self.hull_gain_body <= self.hull_gain_max or not 0.5 <= self.hull_gain_background <= 1.0:
            raise ValueError("hull_gain_body must be in 0.5..hull_gain_max and hull_gain_background in 0.5..1")
        if self.hull_ring_cells < 0 or not 0 < self.hull_retreat < 1:
            raise ValueError("hull_ring_cells must be at least 0 and hull_retreat between 0 and 1")
        if self.hull_regions == "parts":
            if self.flip_augment:
                raise ValueError("hull_regions parts refuses flip_augment: a mirrored mole or tattoo is another body")
            if self.region_weights:
                raise ValueError("hull_regions parts refuses region_weights: the body's emphasis would count twice")
            if self.multi_gpu == "data_parallel" or self.race_parallel:
                raise ValueError("hull_regions parts runs on the primary model copy only: multi_gpu data_parallel and "
                                 "race_parallel would dilute or skip its gains")

    def validate_region_actions(self) -> None:
        """region_class_weights and region steering: raises ValueError with what to change."""
        from .regionwatch import parse_class_weights

        if parse_class_weights(self.region_class_weights):
            if self.region_weights:
                raise ValueError("region_class_weights and region_weights are two fixed weightings of the same loss: "
                                 "use one")
            if self.wants_parts():
                raise ValueError("region_class_weights refuses hull_regions parts: Hull's gains are its only weighting")
        if self.region_sweep_every < 0 or self.region_sweep_pictures < -1 or self.holdout_images < 0 \
                or self.holdout_patience < 0:
            raise ValueError("region_sweep_every, holdout_images and holdout_patience must be 0 or more, "
                             "region_sweep_pictures -1 or more")
        if (self.holdout_track_best or self.holdout_patience > 0) and self.holdout_images <= 0:
            raise ValueError("holdout_track_best and holdout_patience need holdout_images above 0")
        if self.region_sweep_pictures < 0 and self.holdout_images <= 0:
            raise ValueError("region_sweep_pictures -1 sweeps the held-out pictures only: set holdout_images")
        if self.wants_region_sweep() and self.resolution_step_up:
            raise ValueError("region sweeps and held-out pictures are measured against a first sweep at one "
                             "resolution: not with resolution_step_up")
        if self.region_steer_signal not in ("steps", "sweep"):
            raise ValueError(f"region_steer_signal must be steps or sweep, got {self.region_steer_signal!r}")
        if not self.region_steer:
            return
        self.region_steer_list()
        if self.region_steer_signal == "sweep" and (not self.wants_region_sweep() or self.region_sweep_pictures < 0):
            raise ValueError("region_steer_signal sweep reads the sweeps of the training pictures: set "
                             "region_sweep_every (and region_sweep_pictures 0 or more)")
        if not 0.0 <= self.region_steer_holdout < 1.0 or not 0.0 <= self.region_steer_subject_ahead < 1.0:
            raise ValueError("region_steer_holdout and region_steer_subject_ahead must be 0 (off) to under 1")
        if self.region_steer_subject_ahead > 0 and (not self.wants_region_sweep() or self.region_sweep_pictures < 0):
            raise ValueError("region_steer_subject_ahead reads the sweeps of the training pictures: set "
                             "region_sweep_every (and region_sweep_pictures 0 or more)")
        if self.region_steer_holdout > 0 and (self.holdout_images <= 0 or self.region_sweep_pictures < 0):
            raise ValueError("region_steer_holdout compares training pictures with held-out ones: set holdout_images "
                             "and sweep the training pictures too (region_sweep_pictures 0 or more)")
        if self.wants_parts():
            raise ValueError("region_steer refuses hull_regions parts: Hull's region gains are its only weighting")
        if not 0.0 <= self.region_steer_cap <= 2.0 or not 0.0 <= self.region_steer_ahead < 1.0:
            raise ValueError("region_steer_cap must be 0 (off) to 2 and region_steer_ahead 0 (off) to under 1")
        if self.region_steer_boost != 0 and not 1.0 < self.region_steer_boost <= 3.0:
            raise ValueError("region_steer_boost must be 0 (off) or above 1 and at most 3")
        if not 0.0 <= self.region_steer_prior <= 2.0:
            raise ValueError("region_steer_prior must be 0 (off) to 2")
        if self.region_steer_prior > 0 and (self.region_steer_cap <= 0
                                            or "background" not in self.region_steer_list()):
            raise ValueError("region_steer_prior follows the cap rule on the background: set region_steer_cap above 0 "
                             "and keep background in region_steer_regions")
        if self.region_steer_prior > 0 and self.finetune:
            raise ValueError("region_steer_prior needs an adapter to switch off: not with a full fine-tune")
        if not (self.region_steer_cap > 0 or self.region_steer_ahead > 0 or self.region_steer_boost > 0
                or self.region_steer_holdout > 0 or self.region_steer_subject_ahead > 0):
            raise ValueError("region_steer has every rule at 0: set region_steer_cap, region_steer_ahead, "
                             "region_steer_boost, region_steer_holdout or region_steer_subject_ahead")
        if not 0.0 <= self.region_steer_min < 1.0:
            raise ValueError("region_steer_min must be 0 to under 1")
        if not 0.0 < self.region_steer_step <= 0.5 or not 0.0 < self.region_steer_strength <= 4.0:
            raise ValueError("region_steer_step must be above 0 and at most 0.5, region_steer_strength above 0 and at "
                             "most 4")

    def validate_prompt_check(self) -> None:
        """The prompt check's settings (promptcheck.py): the probes' expected frames, the extra prompts, and the two
        settings that act on it."""
        allowed = ("", "none", *_promptcheck.EXPECTS)
        for key in ("probe_face_expect", "probe_body_expect"):
            if str(getattr(self, key) or "").strip().lower() not in allowed:
                raise ValueError(f"{key} must be empty (guess from the prompt), none, or one of "
                                 f"{', '.join(_promptcheck.EXPECTS)}, got {getattr(self, key)!r}")
        if not 1 <= int(self.probe_check_seeds) <= 16:
            raise ValueError("probe_check_seeds must be 1 to 16")
        if not 0.0 <= self.likeness_best_min_kept <= 1.0:
            raise ValueError("likeness_best_min_kept must be 0 (off) to 1")
        if not 0.0 <= self.race_drop_drift <= 1.0:
            raise ValueError("race_drop_drift must be 0 (off) to 1")
        if int(self.race_drop_drift_probes) < 1:
            raise ValueError("race_drop_drift_probes must be 1 or more")
        _promptcheck.normalize_checks(self.probe_checks, self.trigger_word, _promptcheck.subject_of(self))
        probing = self.probe_every > 0 and bool(self.face_refs or self.body_refs)
        if self.probe_checks and not probing:
            raise ValueError("probe_checks are rendered with the likeness probes and read by their scorers: set "
                             "probe_every > 0 and face_refs and/or body_refs")
        acting = [k for k, on in (("likeness_best_min_kept", self.likeness_best_min_kept > 0),
                                  ("race_drop_drift", self.race and self.race_drop_drift > 0)) if on]
        if acting and not (probing and _promptcheck.wanted(self)):
            raise ValueError(f"{' and '.join(acting)} read the prompt check, which has nothing to check here: it "
                             f"needs likeness probes (probe_every, face_refs and/or body_refs) and a probe prompt "
                             f"whose frame is known (set probe_face_expect / probe_body_expect, or add probe_checks)")

    def validate(self) -> None:
        if self.optimizer.lower() == "hull":
            self.validate_hull()
        self.validate_region_actions()
        if self.body_score not in BODY_SCORES:
            raise ValueError(f"body_score must be one of {BODY_SCORES}, got {self.body_score!r}")
        if min(self.face_scale_span, self.face_scale_curve, self.body_scale_span, self.body_scale_curve) < 1.0:
            raise ValueError("face_scale_span / _curve and body_scale_span / _curve must be 1 or more (1 and 1 = the "
                             "plain linear scale)")
        if self.model_arch not in ("auto", *ARCH_NAMES):
            raise ValueError(f"model_arch must be auto or one of {ARCH_NAMES}, got {self.model_arch!r}")
        if self.training_maps and not 1 <= self.training_maps_images <= 12:
            raise ValueError("training_maps_images must be 1 to 12 (each picture adds 3-9 forwards per capture)")
        if self.training_maps_every < 0:
            raise ValueError("training_maps_every must be 0 (once an epoch) or a number of steps")
        if not 0.0 <= self.likeness_curriculum_framing <= 1.0:
            raise ValueError("likeness_curriculum_framing must be 0 (off) to 1")
        self.validate_prompt_check()
        foreign = self.foreign_sampling_files()
        if foreign:
            raise ValueError(f"This run trains {self.arch_name()}, but these settings name another model's files: "
                             f"{'; '.join(foreign)}. Clear them (or set auto) after switching models")
        if self.loss_type not in ("mse", "band"):
            raise ValueError(f"loss_type must be mse or band, got {self.loss_type!r}")
        for key, allowed in (("lr_scheduler", LR_SCHEDULERS), ("timestep_sampling", TIMESTEP_SAMPLINGS),
                             ("loss_weighting", LOSS_WEIGHTINGS)):
            if getattr(self, key) not in allowed:
                raise ValueError(f"{key} must be one of {', '.join(allowed)}, got {getattr(self, key)!r}")
        if int(self.batch_size) < 1 or int(self.gradient_accumulation) < 1:
            raise ValueError("batch_size and gradient_accumulation must be 1 or more (gradient_accumulation 0 would "
                             "train no image and save the LoRA untrained)")
        if not 0.0 <= float(self.caption_dropout) <= 1.0:
            raise ValueError(f"caption_dropout is a share of the steps, 0 to 1, got {self.caption_dropout!r}")
        if self.int8_backward not in ("bf16", "int8"):
            raise ValueError(f"int8_backward must be bf16 or int8, got {self.int8_backward!r}")
        if self.nvfp4_backward not in ("bf16", "fp4"):
            raise ValueError(f"nvfp4_backward must be bf16 or fp4, got {self.nvfp4_backward!r}")
        for mp in self.resolutions:
            if not (0.0 < float(mp) <= 16.0):
                raise ValueError(f"resolutions must be megapixel values in (0, 16], got {mp!r}")
        self.validate_datasets()
        from .clean_share import parse as parse_clean_share

        parse_clean_share(self.clean_share)
        if self.finetune:
            self.validate_finetune()
        if self.finetune_anchor < 0:
            raise ValueError("finetune_anchor must be 0 (off) or more")
        if self.reward_steps < 0:
            raise ValueError("reward_steps must be 0 (off) or more")
        if self.reward_steps > 0:
            if not (self.face_refs or self.body_refs):
                raise ValueError("reward_steps scores its pictures against face_refs / body_refs: set references")
            if self.race:
                raise ValueError("reward_steps does not follow a race: race first, then run the stage on the "
                                 "winner's file (resume_from, steps 0)")
            if self.finetune:
                raise ValueError("reward_steps fine-tunes an adapter; a full fine-tune has none")
            if self.reward_group < 2 or self.reward_train_steps < 1 or self.reward_noise <= 0 or self.reward_lr <= 0 \
                    or self.reward_kl < 0:
                raise ValueError("reward_group must be 2 or more, reward_train_steps 1 or more, reward_noise and "
                                 "reward_lr above 0 and reward_kl 0 or more")
            if self.steps == 0 and not self.resume_from:
                raise ValueError("steps 0 runs only the reward stage: give it a trained LoRA (resume_from)")
        if self.soup_files:
            if len(self.soup_files) < 2:
                raise ValueError("soup_files needs 2 or more LoRA files")
            if not (self.face_refs or self.body_refs) or self.probe_every <= 0 \
                    or not (self.probe_face_prompt or self.probe_body_prompt):
                raise ValueError("soup_files picks the soup with the likeness probes: set face_refs / body_refs, "
                                 "probe_every and a probe prompt")
            if self.steps != 0 or self.resume_from or self.race or self.finetune or self.sample_model_path:
                raise ValueError("soup_files runs alone: steps 0, no resume_from, race, full fine-tune or separate "
                                 "sampling model")
            soup_strength_list(self.soup_strengths)  # raises on a bad list
        if self.noise_files:
            if not (self.face_refs or self.body_refs) or self.probe_every <= 0 \
                    or not (self.probe_face_prompt or self.probe_body_prompt):
                raise ValueError("noise_files measures the likeness probes: set face_refs / body_refs, probe_every "
                                 "and a probe prompt")
            if self.steps != 0 or self.resume_from or self.race or self.finetune or self.sample_model_path \
                    or self.soup_files:
                raise ValueError("noise_files runs alone: steps 0, no resume_from, race, seed soup, full fine-tune "
                                 "or separate sampling model")
            if not 3 <= self.noise_seeds <= 64:
                raise ValueError("noise_seeds must be 3 to 64")
        if self.selftrain:
            if not (self.face_refs or self.body_refs):
                raise ValueError("selftrain keeps the pictures that look most like face_refs / body_refs: set "
                                 "references")
            if self.selftrain_images < 1 or not 0.0 < self.selftrain_share <= 1.0 or self.selftrain_weight <= 0:
                raise ValueError("selftrain_images must be 1 or more, selftrain_share above 0 and at most 1, "
                                 "selftrain_weight above 0")
            from .selftrain import round_steps

            round_steps(max(2, self.steps), self.selftrain_at)  # raises on a bad fraction
        if self.attn_localize_weight < 0 or self.id_align_weight < 0 or self.body_align_weight < 0:
            raise ValueError("attn_localize_weight, id_align_weight and body_align_weight must be 0 or more")
        if self.attn_localize and not (self.trigger_word or any(sp.trigger for sp in self.dataset_specs())):
            raise ValueError("attn_localize needs a trigger word (trigger_word, or a dataset's own trigger): it keeps "
                             "that word's attention on the subject")
        from .lora import KINDS as NETWORK_KINDS

        if self.network_type not in NETWORK_KINDS:
            raise ValueError(f"network_type must be one of {NETWORK_KINDS}, got {self.network_type!r}")
        if self.oft_block_size < 2 or self.oft_constraint < 0:
            raise ValueError("oft_block_size must be 2 or more and oft_constraint 0 (off) or more")
        if self.dora and self.network_type == "oft":
            raise ValueError("dora works on a LoRA or a LoKr; OFT rotates the weights instead (turn one off)")
        if self.dora and self.finetune:
            raise ValueError("dora is an adapter option; a fine-tune trains the weights themselves (turn one off)")
        if self.keep_checkpoints < 0:
            raise ValueError("keep_checkpoints must be 0 (keep all) or the number of checkpoints to keep")
        if self.block_swap:
            if self.block_swap < 0:
                raise ValueError("block_swap must be 0 (off) or a number of blocks")
            if self.multi_gpu in ("model_split", "data_parallel"):
                raise ValueError("block_swap runs on one GPU: set multi_gpu to off (model_split spreads the blocks "
                                 "over GPUs instead)")
            if self.finetune:
                raise ValueError("block_swap streams frozen weights; a fine-tune trains them: turn one off")
            if not self.gradient_checkpointing:
                raise ValueError("block_swap needs gradient_checkpointing (without it every swapped block's weights "
                                 "stay on the GPU until the backward)")
        if self.activation_offload and not self.gradient_checkpointing:
            raise ValueError("activation_offload moves checkpointed blocks' inputs to RAM: it needs "
                             "gradient_checkpointing")
        if self.timestep_sampling == "logsnr" and self.logsnr_std <= 0:
            raise ValueError("logsnr_std must be above 0")
        if self.timestep_buckets < 0 or self.timestep_buckets == 1:
            raise ValueError("timestep_buckets must be 0 (off) or 2 and more")
        if self.isolation:
            if self.finetune:
                raise ValueError("isolation and finetune cannot be combined (DOP holds the LoRA to the model without "
                                 "it; a fine-tune has no LoRA to switch off)")
            if self.race and self.race_parallel:
                raise ValueError("isolation does not run in a race_parallel race: turn race_parallel off")
            if self.dop_every < 1 or self.dop_weight < 0 or self.isolation_pairs < 0:
                raise ValueError("isolation: dop_every must be 1 or more, dop_weight and isolation_pairs 0 or more")
            for spec in self.dataset_specs():
                kind = spec.subject_kind if self.datasets else self.isolation_kind
                if (kind or "auto") not in ("auto", "person", "thing", "style"):
                    raise ValueError(f"{spec.path}: subject kind must be auto, person, thing or style")
        if self.id_loss_backend not in ("arcface", "facenet"):
            raise ValueError(f"id_loss_backend must be arcface or facenet, got {self.id_loss_backend!r}")
        if self.id_loss_every < 1:
            raise ValueError("id_loss_every must be at least 1")
        if str(self.face_gold).lower() not in ("auto", "off", "facenet"):
            raise ValueError(f"face_gold must be auto, off or facenet, got {self.face_gold!r}")
        if str(self.face_gold).lower() == "facenet" and self.trains_on_facenet():
            raise ValueError("face_gold facenet: the identity loss trains on FaceNet (id_loss_backend facenet), so it "
                             "is no held-out judge; use face_gold off or id_loss_backend arcface")
        if str(self.copy_check).lower() not in ("auto", "on", "off"):
            raise ValueError(f"copy_check must be auto, on or off, got {self.copy_check!r}")
        if not 0.0 < self.copy_max <= 1.0:
            raise ValueError(f"copy_max is a cosine above 0 and at most 1, got {self.copy_max!r}")
        if self.probe_device and not self.probe_device.strip().startswith("cuda"):
            raise ValueError(f"probe_device is a CUDA GPU such as cuda:1, got {self.probe_device!r}")
        if not 0.0 <= self.lora_ema < 1.0:
            raise ValueError(f"lora_ema must be 0 (off) or a decay below 1, got {self.lora_ema!r}")
        if self.dp_balance not in ("off", "auto", "grow"):
            raise ValueError(f"dp_balance must be off, auto or grow, got {self.dp_balance!r}")
        if self.optimizer.lower() not in OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {OPTIMIZERS}, got {self.optimizer!r}")
        if self.automagic_groups not in ("single", "by_type"):
            raise ValueError(f"automagic_groups must be single or by_type, got {self.automagic_groups!r}")
        if self.automagic_min_lr > self.automagic_max_lr:
            raise ValueError("automagic_min_lr must not be above automagic_max_lr")
        if not 2 <= self.automagic_polarity_history <= 64:
            raise ValueError("automagic_polarity_history must be between 2 and 64")
        if self.automagic_clip_threshold <= 0:
            raise ValueError("automagic_clip_threshold must be above 0")
        if not 0.0 < self.automagic_sensitivity_quantile <= 1.0:
            raise ValueError("automagic_sensitivity_quantile must be above 0 and at most 1")
        if self.automagic_vote not in ("graded", "extremes"):
            raise ValueError(f"automagic_vote must be graded or extremes, got {self.automagic_vote!r}")
        if self.automagic_max_lr_change <= 0:
            raise ValueError("automagic_max_lr_change must be above 0")
        if self.keel_gauge_eps <= 0:
            raise ValueError("keel_gauge_eps must be above 0")
        if not 1 <= self.keel_bands <= 64:
            raise ValueError("keel_bands must be between 1 and 64")
        keel_band_weights(self.keel_band_weights, self.keel_bands)  # raises on a bad spec
        if not 0 <= self.keel_consensus <= 1:
            raise ValueError("keel_consensus must be between 0 and 1")
        if self.keel_consensus_power <= 0:
            raise ValueError("keel_consensus_power must be above 0")
        keel_consensus_mode(self.keel_consensus_mode, self.optimizer)  # raises on a bad spec
        if not 0 < self.keel_lookahead_pullback <= 1:
            raise ValueError("keel_lookahead_pullback must be above 0 and at most 1")
        if not 0 < self.keel_lookahead_lr <= 1:
            raise ValueError("keel_lookahead_lr must be above 0 and at most 1")
        if self.keel_lookahead_grace < 0:
            raise ValueError("keel_lookahead_grace must be 0 or more")
        if self.keel_face_gain < 0:
            raise ValueError("keel_face_gain must be 0 (off) or above")
        if self.keel_face_gain_max < 1:
            raise ValueError("keel_face_gain_max must be 1 or more")
        if not 0 <= self.keel_face_gain_beta < 1:
            raise ValueError("keel_face_gain_beta must be 0 or more and below 1")
        if self.keel_body_weight < 0:
            raise ValueError("keel_body_weight must be 0 (face only) or above")
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

    def validate_finetune(self) -> None:
        """What a full fine-tune cannot combine with (finetune.py)."""
        if self.race:
            raise ValueError("finetune and race cannot be combined (race LoRA configurations, then fine-tune the "
                             "winner's settings)")
        if self.multi_gpu == "data_parallel":
            raise ValueError("finetune does not run data parallel (its trained weights are not a LoRA to average): "
                             "set multi_gpu to off or model_split")
        if self.resume_from:
            raise ValueError("finetune has no LoRA to resume from (resume_from); continue a fine-tune with "
                             "finetune_resume (its delta file)")
        if self.region_prior_weight > 0:
            raise ValueError("finetune cannot use region_prior_weight (there is no base model to compare with)")
        ranks = [int(r) for r in (self.finetune_extract_ranks or [])]
        if not ranks or any(r < 1 for r in ranks):
            raise ValueError("finetune_extract_ranks needs one or more ranks of 1 or more (64 is Fizgig's measured "
                             "match for the full checkpoint)")
        if self.finetune_rotate_epochs <= 0:
            raise ValueError("finetune_rotate_epochs must be above 0")

    def validate_datasets(self) -> None:
        """Every dataset has a path, at least one repeat and megapixel resolutions; none is listed twice or lies
        inside another (whose scan includes its sub-folders, so its images would train twice)."""
        import os

        for d in self.datasets:
            if not isinstance(d, (dict, str)):
                raise ValueError(f"datasets entries are mappings of {', '.join(DATASET_KEYS)}, got {d!r}")
            bad = sorted(set(d) - set(DATASET_KEYS)) if isinstance(d, dict) else []
            if bad:
                raise ValueError(f"datasets entries take {', '.join(DATASET_KEYS)}; unknown: {', '.join(bad)}")
        seen: dict[str, str] = {}
        for spec in self.dataset_specs():
            if not spec.path:
                raise ValueError("No dataset: pick at least one (datasets), or set dataset_dir")
            if spec.repeats < 1:
                raise ValueError(f"Dataset {spec.path}: repeats must be at least 1, got {spec.repeats}")
            for mp in spec.resolutions:
                if not (0.0 < mp <= 16.0):
                    raise ValueError(f"Dataset {spec.path}: resolutions must be megapixel values in (0, 16], "
                                     f"got {mp!r}")
            key = os.path.normcase(os.path.abspath(spec.path))
            if key in seen:
                raise ValueError(f"Dataset {spec.path} is listed twice")
            seen[key] = spec.path
        for a in seen:
            for b in seen:
                if a != b and a.startswith(b.rstrip("\\/") + os.sep):
                    raise ValueError(f"Dataset {seen[a]} lies inside dataset {seen[b]}, whose scan includes its "
                                     "sub-folders: its images would train twice")

    def wants_regions(self) -> bool:
        """Region maps (face cells, landmarks, the subject's cells) are needed by the region-weighted loss, the identity
        loss, attention localization, identity alignment and Keel's face-token gain."""
        return bool(self.region_weights or self.id_loss_weight > 0 or self.attn_localize or self.id_align
                    or self.body_align or (self.optimizer.lower() in KEEL and self.keel_face_gain > 0)
                    or self.wants_part_maps() or self.race_prep_regions)

    def wants_parts(self) -> bool:
        """Region maps carry the parser's parts tensor: Hull's region gains."""
        return self.optimizer.lower() == "hull" and self.hull_regions == "parts"

    def wants_part_maps(self) -> bool:
        """Region maps carry the parser's parts tensor: Hull's region gains, or the region watch (which, unlike
        wants_parts, changes nothing about the training step)."""
        return (self.wants_parts() or self.wants_region_watch() or self.wants_region_sweep()
                or bool(str(self.region_class_weights or "").strip()))

    def wants_region_sweep(self) -> bool:
        """Fixed-noise sweeps run: asked for, or needed by held-out pictures."""
        return self.region_sweep_every > 0 or self.holdout_images > 0

    def sweep_every_epochs(self) -> int:
        return max(1, int(self.region_sweep_every))

    def wants_region_watch(self) -> bool:
        """The region watch runs: asked for, or needed by region steering."""
        return bool(self.region_watch or self.region_steer)

    def region_steer_list(self) -> tuple[str, ...]:
        """region_steer_regions as names, in a fixed order. Raises ValueError on an unknown one or none at all."""
        names = [n.strip().lower() for n in str(self.region_steer_regions or "").split(",") if n.strip()]
        if not names or any(n not in ("background", "clothes") for n in names):
            raise ValueError(f"region_steer_regions must list background, clothes or both, got "
                             f"{self.region_steer_regions!r}")
        return tuple(n for n in ("background", "clothes") if n in names)

    def region_steer_settings(self) -> dict | None:
        """The region watch's steering settings, or None with steering off."""
        if not self.region_steer:
            return None
        return {"regions": self.region_steer_list(), "cap": self.region_steer_cap, "ahead": self.region_steer_ahead,
                "boost": self.region_steer_boost, "prior": self.region_steer_prior,
                "holdout": self.region_steer_holdout, "signal": self.region_steer_signal,
                "subject_ahead": self.region_steer_subject_ahead,
                "min": self.region_steer_min, "step": self.region_steer_step, "strength": self.region_steer_strength}

    def wants_person_masks(self) -> bool:
        """Person masks in the region maps: the region-weighted loss's person weight, attention localization's
        subject, body alignment, the face-and-body Keel's token gain (its body share)."""
        return bool(self.region_weights or self.attn_localize or self.body_align
                    or (self.optimizer.lower() == "keel_fb" and self.keel_face_gain > 0 and self.keel_body_weight > 0))

    def resolved_cache_dir(self) -> Path:
        """The run's cache (text embeddings, face analysis, loss-watch exclusions): cache_dir, else the first
        dataset's .rulz_cache."""
        return Path(self.cache_dir) if self.cache_dir else Path(self.dataset_specs()[0].path) / ".rulz_cache"

    def dataset_cache_dir(self, path: str | Path) -> Path:
        """A dataset's own cache for its images' latents and region maps: cache_dir, else <dataset>/.rulz_cache, so a
        dataset's latents are reused whichever datasets it trains with."""
        return Path(self.cache_dir) if self.cache_dir else Path(path) / ".rulz_cache"

    def job_dir(self) -> Path:
        return Path(self.output_dir) / self.job_name

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def numbers(cls, data: dict) -> dict:
        """`data` with the number options that arrived as text turned back into numbers. YAML reads `1e-4` and
        `1e-08` (an exponent without a decimal point) as text, in a hand-written file and in a config that was sent
        as JSON alike, and a text learning rate then fails in a comparison with a message that names no setting.
        Text that is no number is left for validate to refuse."""
        kinds = {f.name: str(f.type) for f in dataclasses.fields(cls)}
        out = dict(data)
        for key, value in data.items():
            if not isinstance(value, str) or kinds.get(key) not in ("float", "int"):
                continue
            try:
                number = float(value.strip())
            except ValueError:
                continue
            if kinds[key] == "int":
                if number != int(number):
                    continue
                number = int(number)
            out[key] = number
        return out

    @classmethod
    def from_dict(cls, data: dict) -> "TrainConfig":
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**cls.numbers(data))

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
            return cls.parse_report(f.read(), str(path))

    @classmethod
    def parse_report(cls, text: str, source: str = "This file") -> tuple["TrainConfig", list[str], list[str]]:
        """load_report for YAML text that is not a file on this computer (a config uploaded in the browser)."""
        data = yaml.safe_load(text) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{source} is not a RulzTrainer config (expected a mapping of options)")
        known = [f.name for f in dataclasses.fields(cls)]
        unknown = sorted(k for k in data if k not in known)
        missing = [k for k in known if k not in data]
        return cls(**cls.numbers({k: v for k, v in data.items() if k in known})), missing, unknown

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

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_yaml())
