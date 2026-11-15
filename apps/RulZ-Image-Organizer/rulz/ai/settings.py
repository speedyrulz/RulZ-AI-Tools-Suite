"""AI settings: the Organizer's own (ai_settings.json next to the library), its built-in prompts, and the
Rulz Training Captioner's settings for the "share the Captioner's server" mode.

Everything here is plain Python (no Qt), so worker threads can read it.
"""
from __future__ import annotations

import ipaddress
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from .. import config

MODES = ("off", "own", "captioner")

# ---- built-in prompts -------------------------------------------------------------------------------
# The replies are read by code: tagging expects a comma-separated list, Look for expects YES or NO first.
# Advanced AI settings can change these; a blank value means "use the built-in prompt", so improved
# prompts in a later version reach everyone who never edited them.
TAG_SYSTEM = (
    "You tag photos so they can be found later in a personal photo library. List everything that is "
    "clearly visible: people (how many, apparent age group, clothing and its colours, accessories, "
    "hairstyle), what they are doing, animals, objects, food, vehicles, text or signs, the place (indoors "
    "or outdoors, the kind of room, landscape or building), weather, season, time of day, the event or "
    "occasion, the main colours, and the kind of picture (selfie, portrait, group photo, screenshot, "
    "document, drawing). Do not guess names or identities, and leave out anything you can't see.\n\n"
    "Reply with only a comma-separated list of short lowercase tags (one to four words each), most "
    "important first. No sentences, no numbering, no preamble."
)
TAG_USER = "Tag this photo."
LOOK_SYSTEM = (
    "You check photos for one thing at a time. Look carefully at the whole photo, including the background "
    "and small details. Answer YES only if the thing you are asked about is clearly visible in this photo. "
    "If it is missing, unclear or only implied, answer NO.\n\n"
    "Reply with exactly one word: YES or NO."
)
LOOK_USER = "Does this photo show: {query}?\nReply YES or NO."
QUERY_PLACEHOLDER = "{query}"

PROMPT_DEFAULTS = {"tag_system": TAG_SYSTEM, "tag_user": TAG_USER, "look_system": LOOK_SYSTEM, "look_user": LOOK_USER}

DEFAULTS = {
    "mode": "off",                         # "off", "own" (urls/model below) or "captioner" (its settings)
    "urls": "http://127.0.0.1:11434/v1",   # Ollama's OpenAI-compatible address; one per line for several
    "model": "",
    "api_key": "",
    "concurrency": 1,                      # requests at a time per URL (own mode)
    "temperature": 0.1,
    "tag_max_tokens": 400,
    "look_max_tokens": 32,
    "max_tags": 40,
    "max_image_side": 1024,
    "timeout": 300,                        # seconds to wait for one reply
    "disable_thinking": True,
    "managed_port": 8190,                  # first port for RulZ's own llama-server (Captioner managed mode)
    "tag_system": "", "tag_user": "", "look_system": "", "look_user": "",
}

# Bounds for the numbers (a hand edit can't make a request that never ends or asks for nothing).
LIMITS = {
    "concurrency": (1, 16), "temperature": (0.0, 2.0), "tag_max_tokens": (16, 8192), "look_max_tokens": (1, 2048),
    "max_tags": (1, 200), "max_image_side": (256, 4096), "timeout": (10, 3600), "managed_port": (1024, 65000),
}


def settings_file() -> Path:
    return config.data_dir() / "ai_settings.json"


def _fits(value, default) -> bool:
    if isinstance(default, bool) or isinstance(value, bool):
        return isinstance(value, bool) and isinstance(default, bool)
    if isinstance(default, float):
        return isinstance(value, (int, float))
    if isinstance(default, list):
        return isinstance(value, list) and all(isinstance(x, int) and not isinstance(x, bool) for x in value)
    return isinstance(value, type(default))


def _clamp(key: str, value):
    lo, hi = LIMITS[key]
    value = min(max(value, lo), hi)
    return float(value) if isinstance(DEFAULTS[key], float) else int(value)


def _set_aside(path: Path) -> None:
    """Keep a broken settings file as <name>.bad-<time>.json instead of losing a hand edit to the defaults."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for n in range(1, 100):
        dest = path.with_name(f"{path.stem}.bad-{stamp}{f'-{n}' if n > 1 else ''}{path.suffix}")
        if not dest.exists():
            try:
                os.rename(path, dest)
            except OSError:
                pass
            return


def load() -> dict:
    """Defaults overlaid with the saved values that have the right type (others are dropped)."""
    s = dict(DEFAULTS)
    path = settings_file()
    try:
        with open(path, encoding="utf-8-sig") as f:
            saved = json.load(f)
    except FileNotFoundError:
        return s
    except ValueError:
        _set_aside(path)
        return s
    except OSError:  # locked by another program: use the defaults this time, don't touch the file
        return s
    if not isinstance(saved, dict):
        _set_aside(path)
        return s
    for k, v in saved.items():
        if k in DEFAULTS and _fits(v, DEFAULTS[k]):
            s[k] = _clamp(k, v) if k in LIMITS else v
    if s["mode"] not in MODES:
        s["mode"] = "off"
    return s


def save(s: dict) -> None:
    """Write the settings (only known keys; prompts equal to the built-in ones are saved blank)."""
    out = {k: s.get(k, v) for k, v in DEFAULTS.items()}
    for k, default in PROMPT_DEFAULTS.items():
        if out[k].strip() == default.strip():
            out[k] = ""
    path = settings_file()
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def prompt(s: dict, key: str) -> str:
    """The prompt in use: the saved one, or the built-in one when none is saved."""
    return s.get(key, "").strip() or PROMPT_DEFAULTS[key]


def look_user_prompt(s: dict, query: str) -> str:
    text = prompt(s, "look_user")
    if QUERY_PLACEHOLDER in text:
        return text.replace(QUERY_PLACEHOLDER, query)
    return f"{text}\n{query}"  # an edited prompt without the placeholder still names what to look for


# ---- the Training Captioner's settings ----------------------------------------------------------------
CAPTIONER_DEFAULTS = {
    "backend": "managed", "llama_server": "", "model": "", "mmproj": "", "gpu_mode": "parallel", "gpus": [],
    "tensor_split": "", "ctx_size": 4096, "ngl": 99, "slots_per_server": 1, "extra_args": "", "base_port": 8090,
    "external_urls": "http://127.0.0.1:1234/v1", "external_model": "", "api_key": "",
    "disable_thinking": True, "thinking_budget": 1024,
}


def captioner_settings_file() -> Path:
    """Where the Training Captioner keeps its settings (its RULZ_CAPTIONER_DIR override included)."""
    if os.environ.get("RULZ_CAPTIONER_DIR"):
        return Path(os.environ["RULZ_CAPTIONER_DIR"]) / "settings.json"
    return Path(os.environ.get("APPDATA") or Path.home()) / "RulzCaptioner" / "settings.json"


def read_captioner() -> tuple[dict | None, str | None]:
    """(the Captioner's settings with its defaults filled in, None) or (None, why they can't be used).
    Read only: this app never writes the Captioner's files."""
    path = captioner_settings_file()
    try:
        with open(path, encoding="utf-8-sig") as f:
            saved = json.load(f)
    except FileNotFoundError:
        return None, (f"The Training Captioner's settings weren't found ({path}). Open the Rulz Training "
                      "Captioner, set up its server in Settings and press OK, then try again.")
    except ValueError:
        return None, f"The Training Captioner's settings file can't be read ({path}): it isn't valid JSON."
    except OSError as e:
        return None, f"The Training Captioner's settings file can't be opened ({path}): {e.strerror or e}"
    if not isinstance(saved, dict):
        return None, f"The Training Captioner's settings file has no settings in it ({path})."
    cap = dict(CAPTIONER_DEFAULTS)
    for k, v in saved.items():
        if k in CAPTIONER_DEFAULTS and _fits(v, CAPTIONER_DEFAULTS[k]):
            cap[k] = v
    cap["slots_per_server"] = max(1, min(int(cap["slots_per_server"]), 16))
    if cap["backend"] not in ("managed", "external"):
        cap["backend"] = "managed"
    return cap, None


# ---- what to connect to ------------------------------------------------------------------------------
def endpoints_from_urls(urls: str, model: str, slots: int) -> list[dict]:
    """OpenAI-compatible base URLs (one per line or comma-separated) as endpoints, each ending in /v1.
    The same rules as the Training Captioner, so a URL that works there works here."""
    out, seen = [], set()
    for line in urls.replace(",", "\n").splitlines():
        u = line.strip().rstrip("/")
        if u.endswith("/chat/completions"):  # the full request URL pasted from a server's docs
            u = u[:-len("/chat/completions")].rstrip("/")
        if not u:
            continue
        if "://" not in u:
            u = "http://" + u
        if not u.endswith("/v1"):
            u += "/v1"
        if u in seen:
            continue
        seen.add(u)
        out.append({"url": u, "model": model or "default", "slots": max(1, int(slots)), "name": u})
    return out


def is_loopback(url: str) -> bool:
    """True for a server on this computer (localhost, 127.x.x.x, ::1)."""
    host = (urlsplit(url).hostname or "").lower()
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@dataclass
class Connection:
    """Where requests go, resolved from the settings when a job starts."""
    source: str                     # "own" or "captioner"
    kind: str                       # "external" (URLs) or "managed" (llama-server from the Captioner's settings)
    endpoints: list[dict] = field(default_factory=list)  # external only; managed ones come from the launcher
    api_key: str = ""
    model: str = ""                 # for messages and the record of what tagged a photo
    captioner: dict | None = None   # the Captioner's settings (managed)
    problem: str | None = None      # why nothing can be sent; None when it's ready

    @property
    def remote(self) -> bool:
        """Some endpoint isn't on this PC: photos would be sent over the network."""
        return any(not is_loopback(ep["url"]) for ep in self.endpoints)


def resolve(s: dict) -> Connection:
    mode = s.get("mode", "off")
    if mode == "own":
        model = s["model"].strip()
        eps = endpoints_from_urls(s["urls"], model, s["concurrency"])
        conn = Connection("own", "external", eps, s["api_key"].strip(), model)
        if not eps:
            conn.problem = "Type the address of your AI server in Library ▸ AI tagging and search."
        elif not model:
            conn.problem = ("Choose a model in Library ▸ AI tagging and search (for Ollama, a vision model such "
                            "as qwen2.5vl:7b; Load models lists what the server has).")
        return conn
    if mode == "captioner":
        cap, problem = read_captioner()
        if cap is None:
            return Connection("captioner", "external", problem=problem)
        if cap["backend"] == "external":
            model = cap["external_model"].strip()
            eps = endpoints_from_urls(cap["external_urls"], model, cap["slots_per_server"])
            conn = Connection("captioner", "external", eps, cap["api_key"].strip(), model or "default", cap)
            if not eps:
                conn.problem = ("The Training Captioner has no server address set (its Settings ▸ Backend ▸ "
                                "Endpoint URLs).")
            return conn
        model = os.path.basename(cap["model"]) if cap["model"] else ""
        conn = Connection("captioner", "managed", [], "", model or "llama-server", cap)
        if not cap["model"]:
            conn.problem = "The Training Captioner has no model set (its Settings ▸ Backend ▸ Model)."
        return conn
    return Connection("own", "external", problem="AI tagging and search is off. Turn it on in Library ▸ AI "
                                                 "tagging and search.")


def describe(conn: Connection) -> str:
    """One line for the Library page: what the AI requests go to."""
    if conn.kind == "managed":
        cap = conn.captioner or {}
        return (f"Training Captioner's llama.cpp setup · model {conn.model}. RulZ uses the Captioner's servers "
                f"while they run (port {cap.get('base_port', '?')}), or starts its own copy.")
    urls = ", ".join(ep["url"] for ep in conn.endpoints) or "no address"
    return f"{urls} · model {conn.model or '(none)'}"
