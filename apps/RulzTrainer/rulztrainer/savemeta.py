"""Public training metadata for exported weights: the run's settings without its private parts (folders and file
paths, prompts, captions, trigger and class words, anything named like a token, key or URL), plus the standard kohya
ss_* training keys. A shared LoRA file then says how it was trained, not where or on what."""
import os
import re

from . import __version__
from .config import effective_optimizer


# fields dropped whatever their value (free text naming the subject, or pointing at the owner's files)
_PRIVATE = {"path", "trigger", "trigger_word", "class", "subject_class", "isolation_class", "probe_subject",
            "sample_negative", "resume_from", "finetune_resume", "compile_cache", "face_refs", "body_refs", "nlf_model"}
_SUFFIXES = ("_dir", "_path", "_file", "_files", "_state", "_folder")
# words that mark free text or secrets: a field named with one is dropped when it holds text (caption_dropout,
# max_token_length and other numbers and switches stay)
_TEXT_WORDS = ("prompt", "caption", "token", "key", "secret", "password", "url")


def _is_path(value) -> bool:
    """A string that looks like a filesystem path (an HF repo id "org/name" does not, unless it exists here)."""
    if not isinstance(value, str) or not value:
        return False
    if "\\" in value or re.match(r"^[A-Za-z]:", value) or value.startswith(("/", "~", "./", "../")):
        return True
    if "/" in value:
        try:
            return os.path.exists(value)  # a relative folder such as models/sam3
        except (OSError, ValueError):
            return True
    return False


def _is_text(value) -> bool:
    return isinstance(value, str) or (isinstance(value, (list, tuple, dict)) and len(value) > 0)


def _private(key, value) -> bool:
    key = str(key).lower()
    if key in _PRIVATE or key.endswith(_SUFFIXES) or _is_path(value):
        return True
    return any(w in key for w in _TEXT_WORDS) and _is_text(value)


def _clean(value):
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items() if not _private(k, v)}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value if not _is_path(v)]
    return value


def public_config(cfg) -> dict:
    """cfg.to_dict() without the private fields and path-like values, at any depth (a `datasets` entry keeps its
    resolutions, repeats and kind but not its folder)."""
    return _clean(cfg.to_dict())


def public_overrides(overrides: dict) -> dict:
    """A race configuration's overrides with the same private parts left out."""
    return _clean(dict(overrides))


def base_model_name(cfg) -> str:
    """ss_base_model: the HF repo id, or only the file or folder name of a local model."""
    if cfg.model_source == "hf" and not _is_path(cfg.model_path):
        return str(cfg.model_path)
    local = str(cfg.model_path if cfg.model_source == "hf" else cfg.transformer_file).rstrip("/\\")
    return re.split(r"[\\/]", local)[-1]  # (either separator, on any OS)


def kohya_training_keys(cfg, *, epoch=None, num_epochs=None, num_train_images=None, num_reg_images=None) -> dict:
    """The standard kohya training fields that are known, as strings (never a path)."""
    sides = cfg.run_sides()
    resolution = (sides[0], sides[0]) if len(sides) == 1 else tuple(sides)
    optimizer, lr = effective_optimizer(cfg)[:2]
    if optimizer == "hull":  # Hull's step is hull_rho; its learning_rate field is not the rate
        lr = getattr(cfg, "hull_rho", None)
    values = {"ss_seed": cfg.seed, "ss_learning_rate": lr,
              "ss_lr_scheduler": cfg.lr_scheduler, "ss_optimizer": optimizer,
              "ss_num_epochs": num_epochs, "ss_epoch": epoch, "ss_max_train_steps": cfg.steps,
              "ss_batch_size_per_device": cfg.batch_size, "ss_gradient_accumulation_steps": cfg.gradient_accumulation,
              "ss_resolution": str(resolution), "ss_num_train_images": num_train_images,
              "ss_num_reg_images": num_reg_images, "ss_training_comment": f"RulZ Trainer {__version__}"}
    return {k: str(v) for k, v in values.items() if v is not None}
