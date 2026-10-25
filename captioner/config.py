"""Settings and saved system prompts, stored as JSON under %APPDATA%."""
import json
import os
import subprocess
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}

APP_DIR = Path(os.environ.get("APPDATA", str(Path.home()))) / "RulzCaptioner"
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


def _load(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _save(path, data):
    APP_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def load_settings():
    s = dict(DEFAULT_SETTINGS)
    s.update(_load(SETTINGS_FILE, {}))
    return s


def save_settings(s):
    _save(SETTINGS_FILE, s)


def load_prompts():
    p = _load(PROMPTS_FILE, None)
    if not isinstance(p, list) or not p:
        p = [dict(x) for x in DEFAULT_PROMPTS]
        _save(PROMPTS_FILE, p)
    return p


def save_prompts(p):
    _save(PROMPTS_FILE, p)


def detect_gpus():
    """Return [(index, name, mem_mib)] using nvidia-smi; empty if unavailable."""
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
            gpus.append((int(parts[0]), parts[1], int(float(parts[2]))))
    return gpus
