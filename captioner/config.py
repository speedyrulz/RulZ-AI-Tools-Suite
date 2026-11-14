r"""Settings and saved system prompts, stored as JSON in %APPDATA%\RulzCaptioner.

Set the RULZ_CAPTIONER_DIR environment variable to use a different folder (portable installs, testing).
"""
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}

APP_DIR = (Path(os.environ["RULZ_CAPTIONER_DIR"]) if os.environ.get("RULZ_CAPTIONER_DIR")
           else Path(os.environ.get("APPDATA", str(Path.home()))) / "RulzCaptioner")
SETTINGS_FILE = APP_DIR / "settings.json"
PROMPTS_FILE = APP_DIR / "prompts.json"

DEFAULT_SETTINGS = {
    # backend: "managed" (app launches llama-server) or "external" (OpenAI-compatible URLs)
    "backend": "managed",
    "llama_server": "",
    "model": "",
    "mmproj": "",
    "gpu_mode": "parallel",  # "parallel" = one server per GPU, "split" = one server across GPUs
    "gpus": [],  # selected GPU indices
    "tensor_split": "",  # e.g. "1,1" or "16,12"; blank = even
    "ctx_size": 4096,
    "ngl": 99,
    "slots_per_server": 1,
    "extra_args": "",
    "base_port": 8090,
    "external_urls": "http://127.0.0.1:1234/v1",
    "external_model": "",
    "api_key": "",
    "temperature": 0.4,
    "max_tokens": 400,
    "max_image_side": 1024,
    "skip_existing": True,
    "prefix": "",
    "suffix": "",
    "single_line": True,
    "disable_thinking": True,
    "thinking_budget": 1024,  # max thinking tokens when thinking is enabled; -1 = unlimited
    "thumb_size": 180,
    "last_dir": "",
    "last_prompt": "Natural language",
    "theme": "",  # "dark" or "light"; blank until the first-run "Choose your look" prompt is answered
}

DEFAULT_PROMPTS = [
    {
        "name": "Natural language",
        "system": "You are an expert image captioner creating training data for text-to-image models. "
                  "Describe only what is visible. Do not mention that this is an image or photo. "
                  "Output a single paragraph with no preamble, lists or markdown.",
        "user": "Write a detailed caption for this image.",
    },
    {
        "name": "Short caption",
        "system": "You write concise captions for image datasets. Output one sentence under 30 words, "
                  "no preamble.",
        "user": "Caption this image.",
    },
    {
        "name": "Booru tags",
        "system": "You tag images for diffusion model training. Output only comma-separated lowercase "
                  "tags describing subject, attributes, clothing, pose, setting, style and composition. "
                  "No sentences, no preamble.",
        "user": "Tag this image.",
    },
    {
        "name": "Character / person LoRA",
        "system": "You caption images for training a character LoRA. Describe clothing, expression, pose, "
                  "background and lighting in one paragraph. Do not describe permanent features of the "
                  "subject such as hair colour or face shape. No preamble.",
        "user": "Write the training caption for this image.",
    },
]


MISSING = object()  # _load(): there is no file
BAD = object()      # _load(): the file can't be used (set aside, or protected)
_protected = set()  # files not to save over during this run: unreadable (locked?) or not set aside
_copied = set()     # files already copied aside during this run (settings are loaded twice at start)


def _set_aside(path, copy=False):
    """Move (or copy) a file that can't be used, or not all of it, to <name>.bad-<time>.json, so saving
    the defaults doesn't destroy it: a hand edit gone wrong is easier to repair than to retype. Never
    over an earlier copy. When it fails, the file is protected instead: _save() leaves it alone."""
    if copy and path in _copied:
        return True
    try:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        for n in range(1, 100):
            dest = path.with_name(f"{path.stem}.bad-{stamp}{f'-{n}' if n > 1 else ''}{path.suffix}")
            if not dest.exists():
                break
        else:
            raise FileExistsError(dest)
        if copy:
            shutil.copy2(path, dest)
            _copied.add(path)
        else:
            os.rename(path, dest)  # unlike os.replace, never overwrites (on Windows)
        return True
    except OSError:
        _protected.add(path)
        return False


def _load(path):
    """The file's JSON, MISSING when there is no file, or BAD when it can't be used. A broken file is
    set aside; one that can't be read at all (locked by a virus scanner or a backup, say) is likely
    fine, so it is only protected from being saved over during this run."""
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            return json.load(f)
    except FileNotFoundError:
        return MISSING
    except ValueError:  # not JSON, or not text
        _set_aside(path)
        return BAD
    except OSError:
        _protected.add(path)
        return BAD


def _save(path, data):
    """Write `data` as JSON. Returns False, writing nothing, for a file that is protected during this
    run (see _load): it may hold settings worth keeping."""
    if path in _protected:
        return False
    APP_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)
    return True


def _fits(value, default):
    """True when a stored setting has the type its default has (bool is not taken for a number)."""
    if isinstance(default, bool) or isinstance(value, bool):
        return isinstance(value, bool) and isinstance(default, bool)
    if isinstance(default, float):
        return isinstance(value, (int, float))
    if isinstance(default, list):
        return isinstance(value, list) and all(isinstance(x, int) and not isinstance(x, bool) for x in value)
    return isinstance(value, type(default))


def load_settings():
    """Defaults, overlaid with the saved settings that have the right type. A value of the wrong type
    (a hand edit, a file from another program) stopped the app from starting at all."""
    s = dict(DEFAULT_SETTINGS)
    saved = _load(SETTINGS_FILE)
    if isinstance(saved, dict):
        rejected = False
        for k, v in saved.items():
            if k not in DEFAULT_SETTINGS or _fits(v, DEFAULT_SETTINGS[k]):
                s[k] = v
            else:
                rejected = True
        if rejected:  # the next save drops those values: keep a copy of the file as it was
            _set_aside(SETTINGS_FILE, copy=True)
    elif saved is not MISSING and saved is not BAD:
        _set_aside(SETTINGS_FILE)  # valid JSON, but not settings (a list, null…)
    return s


def save_settings(s):
    return _save(SETTINGS_FILE, s)


def _prompt(x):
    """A saved prompt as {name, system, user}, or None when it isn't one."""
    if not isinstance(x, dict) or not isinstance(x.get("name"), str) or not x["name"].strip():
        return None
    out = {"name": x["name"]}
    for k in ("system", "user"):
        v = x.get(k, "")
        if not isinstance(v, str):
            return None
        out[k] = v
    return out


def load_prompts():
    """Saved prompts, or the built-in ones. A prompts file that can't be used is set aside instead of
    being replaced by the defaults, which lost every prompt the user had saved."""
    saved = _load(PROMPTS_FILE)
    if isinstance(saved, list):
        p = [q for q in map(_prompt, saved) if q]
        if p:
            # some entries aren't prompts: keep the whole file as a copy, then save the good ones (once,
            # instead of a new copy at every start)
            if len(p) < len(saved) and _set_aside(PROMPTS_FILE, copy=True):
                _save(PROMPTS_FILE, p)
            return p
        if saved:  # entries, none of them a prompt
            _set_aside(PROMPTS_FILE)
    elif saved is not MISSING and saved is not BAD:
        _set_aside(PROMPTS_FILE)  # valid JSON, but not a list
    p = [dict(x) for x in DEFAULT_PROMPTS]
    if not PROMPTS_FILE.exists():  # never save over a file that couldn't be set aside
        _save(PROMPTS_FILE, p)
    return p


def save_prompts(p):
    return _save(PROMPTS_FILE, p)


_gpus = None


def detect_gpus():
    """Return [(index, name, mem_mib)] using nvidia-smi; empty if unavailable. Asked once per run
    (nvidia-smi takes about 0.3 s, and Settings asked every time it opened); an empty answer isn't
    kept, so a driver that wasn't ready yet is asked again."""
    global _gpus
    if not _gpus:
        _gpus = _query_gpus()
    return list(_gpus)


def _query_gpus():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, creationflags=0x08000000,
        ).stdout
    except Exception:
        return []
    gpus = []
    for line in out.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 3 and parts[0].isdigit():
            try:
                mem = int(float(parts[2]))
            except ValueError:  # "[N/A]" on some cards: Settings then wouldn't open at all
                mem = 0
            gpus.append((int(parts[0]), parts[1], mem))
    return gpus
