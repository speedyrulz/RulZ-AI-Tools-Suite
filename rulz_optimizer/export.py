"""Build and atomically write trainer configuration files."""
import os
from pathlib import Path
import tempfile
import yaml
from .recommend import Recommendation, coerce
from .trainer_keys import KEYS, MODEL_KEYS


def build_config(rec: Recommendation, job_name: str, base_config: dict | None = None, extra: dict | None = None) -> dict:
    base = base_config or {}
    cfg = {k: v for k, v in base.items() if k in MODEL_KEYS} if base.get('model_arch', 'auto') in ('auto', rec.model_id) else {}
    if cfg.get('model_arch') == 'auto':
        cfg.pop('model_arch')
    preserved = {k: v for k, v in cfg.items() if k in MODEL_KEYS}
    cfg.update(rec.config())
    cfg.update(preserved)
    cfg.update({k: v for k, v in (extra or {}).items() if k in KEYS})
    cfg['job_name'] = job_name
    return {k: coerce(k, v) for k, v in cfg.items()}


def to_yaml(cfg: dict, header: str) -> str:
    comments = ''.join('# ' + line.lstrip('# ').rstrip() + '\n' for line in header.splitlines())
    return comments + yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False)


def write(cfg: dict, path: Path, header: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='\n', dir=path.parent,
                                         delete=False, suffix='.tmp') as handle:
            temp = handle.name
            handle.write(to_yaml(cfg, header))
        os.replace(temp, path)
    finally:
        if temp and os.path.exists(temp):
            os.unlink(temp)
    return path


def load_trainer_config(path: Path) -> dict:
    try:
        value = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
        return value if isinstance(value, dict) else {}
    except (OSError, yaml.YAMLError, UnicodeError):
        return {}
