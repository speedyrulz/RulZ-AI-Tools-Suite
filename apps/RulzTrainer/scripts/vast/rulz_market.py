#!/usr/bin/env python3
"""Find, rent, start, stop and destroy vast.ai instances from this computer: the REST client, the search page's
filters, offers, the account's credit and burn rate, RulZ Trainer templates, and the records that keep a rent
from being lost or doubled. For the WebUI's Rent GPUs tab and the send dialog (docs/vast-send-contract.md, 2 and 3).

    python scripts/vast/rulz_market.py search --set gpu_ram=[24,null] --limit 10   # offers, cheapest first
    python scripts/vast/rulz_market.py account        # credit, what is billed per hour, how long it lasts
    python scripts/vast/rulz_market.py templates      # the account's RulZ Trainer templates (no environment kept)
    python scripts/vast/rulz_market.py instances      # the account's instances
    python scripts/vast/rulz_market.py rentals        # what this app rented, and rents still being settled

The command line only reads. Renting, starting, stopping and destroying are Market methods that the WebUI calls from
a click whose dialog shows the price and the effect. The money rules are in the contract, section 2; the short
version: a rent is sent once and never retried, and an answer that is not clear is searched for by label afterwards
instead of being sent again.

Every key, template environment value and error text from vast.ai that could carry one stays out of exceptions,
state files and output. Standard library only."""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import re
import shlex
import shutil
import socket
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _path in (str(HERE), str(HERE.parents[1])):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from rulztrainer.train_queue import QueueLock  # noqa: E402  (standard library only)

STATE = Path(os.environ.get("RULZ_VAST_HOME") or Path.home() / ".rulz_vast")
BASE = "https://console.vast.ai"
READ_TIMEOUT = 20.0
RENT_TIMEOUT = 30.0
BACKOFF = (1, 2, 4, 8)  # seconds a read waits after a 429, one try per entry
PAGE_SIZE = 25  # /api/v1/instances/ pages
MAX_PAGES = 200
SEARCH_LIMIT = 1000
RAM_TOLERANCE = 0.98  # a card sold as N GB reports a little under N * 1000 MB (or N * 1024): see conv "gb"
PRICE_RISE_REFUSE = 1.005  # the offer's price may be 0.5% above the one shown before the rent is refused
PRICE_RISE_REPORT = 1.02  # a contract 2% above the shown price is reported on the rental
# seconds after a rent whose answer was lost at which the account is searched for its label: ten minutes in all
CHECK_OFFSETS = (5, 15, 30, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600)
STOP_EVERY = 120.0  # seconds between looks at an instance that was told to stop
STOP_LIMIT = 600.0  # and the time after which it still running is a loud alert
TEMPLATE_BRANCH = "app/rulz-trainer"  # the suite's branch whose provision.sh a current template runs
SSH_KEY_NAMES = ("id_ed25519.pub", "id_ecdsa.pub", "id_rsa.pub")
AFTER_RUN = ("leave", "stop", "destroy")
STOPPED_STATUSES = ("exited", "stopped", "offline", "inactive")  # what vast.ai reports for a resting instance

# What create_template.sh makes: used for a rent without a template of the account's own.
_BUILTIN_BRANCH = TEMPLATE_BRANCH
_BUILTIN_REPO = "https://github.com/speedyrulz/RulZ-AI-Tools-Suite"
BUILTIN_IMAGE = "vastai/pytorch"
BUILTIN_TAG = "cuda-12.8.1-auto"
BUILTIN_ENV = {
    "-p 1111:1111": "1", "-p 7870:7870": "1", "-p 8080:8080": "1", "-p 22:22": "1",
    "OPEN_BUTTON_PORT": "1111",
    "PROVISIONING_SCRIPT": f"https://raw.githubusercontent.com/speedyrulz/RulZ-AI-Tools-Suite/{_BUILTIN_BRANCH}"
                           "/scripts/vast/provision.sh",
    "RULZ_REPO": _BUILTIN_REPO + ".git", "RULZ_BRANCH": _BUILTIN_BRANCH,
    "PORTAL_CONFIG": "localhost:1111:11111:/:Instance Portal|localhost:7870:17870:/:RulzTrainer|"
                     "localhost:8080:18080:/:Jupyter|localhost:8080:8080:/terminals/1:Jupyter Terminal",
}
BUILTIN_ONSTART = "/opt/instance-tools/bin/entrypoint.sh"
BUILTIN_RUNTYPE = "jupyter_direc ssh_direc ssh_proxy"
BUILTIN = {"id": None, "hash_id": None, "name": "RulZ AI Trainer (built in)", "image": BUILTIN_IMAGE,
           "tag": BUILTIN_TAG, "disk": 120, "created": 0, "current": True, "builtin": True, "cuda": 12.8}


class MarketError(Exception):
    """A vast.ai call that did not work, in words fit for the screen (never a key or an environment value).

    `kind`: no-key, auth, rate, client, server, network, unreadable (Api); busy, gone, price, bid, ssh, reserved,
    refused, pending, bad (Market). `definite` is True when vast.ai certainly did not act on the request, so a
    rent that raised it was not made. `data` carries extras (the new price, the intent's label)."""

    def __init__(self, message: str, kind: str = "client", definite: bool = False, status: int | None = None,
                 **data):
        super().__init__(message)
        self.message = message
        self.kind = kind
        self.definite = definite
        self.status = status
        self.data = data


# ---------------------------------------------------------------- small helpers


def num(v):
    """A finite number from a number or a numeric string, else None (booleans are not numbers)."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v) if math.isfinite(v) else None
    if isinstance(v, str) and v.strip():
        try:
            f = float(v)
        except ValueError:
            return None
        return f if math.isfinite(f) else None
    return None


def _clean_number(v: float):
    """6 decimals, and an int when it is a whole number (so a body reads `23520`, not `23520.0`)."""
    v = round(v, 6)
    return int(v) if v == int(v) else v


def _pos_int(v) -> int | None:
    n = num(v)
    return int(n) if n is not None and n > 0 and n == int(n) else None


def _text(v, limit: int = 200) -> str:
    return v.strip()[:limit] if isinstance(v, str) else ""


def api_key() -> str | None:
    """VAST_API_KEY, else the vastai CLI's key file, else the old one; None when there is none."""
    key = (os.environ.get("VAST_API_KEY") or "").strip()
    if key:
        return key
    for path in (Path.home() / ".config" / "vastai" / "vast_api_key", Path.home() / ".vast_api_key"):
        try:
            key = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if key:
            return key
    return None


_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_URL_QUERY = re.compile(r"https?://[^\s?#]*\?\S*")
_PAIR = re.compile(r"\b([A-Za-z_][\w.-]*)=(\S+)")
_TOKEN = re.compile(r"[A-Za-z0-9_-]{24,}")


def scrub_text(text, limit: int = 200) -> str:
    """Text from vast.ai that is shown or stored: emails, urls with a query string, KEY=VALUE values and long
    token-like strings are hidden, and the length is capped."""
    if not isinstance(text, str):
        return ""
    text = _PAIR.sub(lambda m: m.group(1) + "=[hidden]", text)
    text = _EMAIL.sub("[email]", text)
    text = _URL_QUERY.sub("[link]", text)
    text = _TOKEN.sub("[hidden]", text)
    return text.strip()[:limit]


def lasts(credit, per_hour) -> float | None:
    """Hours the credit pays for at this hourly burn; None when nothing is billed or the credit is unknown."""
    c, p = num(credit), num(per_hour)
    if c is None or p is None or p <= 0:
        return None
    return max(0.0, c) / p


def _save_json(path: Path, data) -> None:
    """Atomic and fsynced: the rent intent must be on disk before the PUT that could make it matter."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}-{uuid.uuid4().hex[:8]}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(data, indent=1))
            fh.flush()
            os.fsync(fh.fileno())
        for attempt in range(20):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.05)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def _load_json(path: Path, default, strict: bool = False):
    """A state file's content. Lenient (reads of preferences and caches): anything unreadable is `default`.
    Strict (everything that is written back, and rentals.json always): only a missing file starts empty. A file
    another program holds open for a moment (a scan, a sharing violation) is tried again a few times; one that
    stays unreadable or is not what it should be raises MarketError kind `state`, so that it is never overwritten
    with an empty record."""
    last = ""
    for attempt in range(5):
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return default
        except OSError as e:
            last = type(e).__name__
            if attempt < 4:
                time.sleep(0.05)
                continue
            break
        try:
            data = json.loads(text)
        except ValueError:
            last = "its content is not valid JSON"
        else:
            if isinstance(data, type(default)):
                return data
            last = "its content is not what was expected"
        break
    if not strict:
        return default
    raise MarketError(f"{path.name} could not be read ({last}), so it is left alone. Look at the file in "
                      f"{path.parent}.", "state", False)


# ---------------------------------------------------------------- the client


def _urlopen(method: str, url: str, headers: dict, data: bytes | None, timeout: float) -> tuple[int, str]:
    """The real opener: (status, text); an HTTP error status is an answer, not an exception."""
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310  (https only, see Api.call)
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        try:
            text = e.read().decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            text = ""
        return e.code, text


@dataclass
class Reply:
    status: int
    json: object
    parsed: bool


class Api:
    """The only thing that talks HTTP. `opener(method, url, headers, data, timeout) -> (status, text)` is replaced in
    tests; `key` is read lazily from api_key() when not given. Errors are rewritten to fixed texts: a urllib
    error can carry the URL and headers, and vast.ai's own message is scrubbed of the key before it is shown."""

    def __init__(self, opener=None, key: str | None = None, sleep=time.sleep, base: str = BASE):
        self.opener = opener or _urlopen
        self._key = key
        self.sleep = sleep
        self.base = base.rstrip("/")

    @property
    def key(self) -> str | None:
        return self._key or api_key()

    def _scrub(self, text: str) -> str:
        key = self.key
        return text.replace(key, "***") if key else text

    def _message(self, body) -> str:
        if isinstance(body, dict):
            for name in ("msg", "error", "detail", "message"):
                v = body.get(name)
                if isinstance(v, str) and v.strip():
                    return scrub_text(self._scrub(v.strip()))
        return ""

    def call(self, method: str, path: str, body=None, retry429: bool = True, timeout: float = READ_TIMEOUT) -> Reply:
        key = self.key
        if not key:
            raise MarketError("Add your vast.ai API key first.", "no-key", True)
        headers = {"Authorization": f"Bearer {key}", "Accept": "application/json", "User-Agent": "rulz-trainer"}
        data = None
        if body is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(body).encode("utf-8")
        url = self.base + path
        attempt = 0
        while True:
            try:
                status, text = self.opener(method, url, headers, data, timeout)
            except (TimeoutError, socket.timeout):
                raise MarketError("vast.ai did not answer in time.", "network", False) from None
            except Exception:  # noqa: BLE001  (its text may carry the address or the headers)
                raise MarketError("Could not reach vast.ai.", "network", False) from None
            try:
                parsed_json, parsed = json.loads(text), True
            except (TypeError, ValueError):
                parsed_json, parsed = None, False
            clear = parsed and isinstance(parsed_json, dict)  # a refusal is clear only with vast.ai's own JSON
            if status == 429:
                if retry429 and attempt < len(BACKOFF):
                    self.sleep(BACKOFF[attempt])
                    attempt += 1
                    continue
                raise MarketError("vast.ai says there are too many requests right now. Try again in a minute.",
                                  "rate", clear, 429)
            if status in (401, 403):
                raise MarketError("vast.ai refused the API key. Check it on the vast.ai page.", "auth", clear, status)
            if status >= 500:
                raise MarketError(f"vast.ai had an error ({status}).", "server", False, status)
            if status >= 400:
                msg = self._message(parsed_json)
                raise MarketError(f"vast.ai said no: {msg}" if msg else f"vast.ai answered {status}.", "client",
                                  clear, status)
            return Reply(status, parsed_json, parsed)

    def _readable(self, reply: Reply):
        if not reply.parsed:
            raise MarketError("vast.ai sent an answer the app could not read.", "unreadable", False)
        return reply.json

    def get(self, path: str):
        return self._readable(self.call("GET", path))

    def current_user(self) -> dict:
        data = self.get("/api/v0/users/current/")
        return data if isinstance(data, dict) else {}

    def search(self, body: dict) -> list[dict]:
        """POST /bundles/: the raw offer rows."""
        data = self._readable(self.call("POST", "/api/v0/bundles/", body))
        rows = data.get("offers") if isinstance(data, dict) else data
        if isinstance(rows, dict):
            rows = [rows]
        return [r for r in rows if isinstance(r, dict)] if isinstance(rows, list) else []

    def find_offer(self, offer_id: int, base_body: dict, attempts: int = 2) -> dict | None:
        """One offer as it is now. `id` finds nothing in a search; `ask_contract_id` does. vast.ai leaves some
        matching offers out of a given answer, so it is asked twice before the offer counts as gone."""
        body = dict(base_body)
        body["ask_contract_id"] = {"eq": offer_id}
        for _ in range(attempts):
            for row in self.search(body):
                if _pos_int(row.get("id")) == offer_id or _pos_int(row.get("ask_contract_id")) == offer_id:
                    return row
        return None

    def show_instance(self, instance_id: int) -> dict | None:
        """One instance's raw row (it carries the environment: parse it with parse_instance); None when gone."""
        try:
            data = self.get(f"/api/v0/instances/{int(instance_id)}/")
        except MarketError as e:
            if e.status == 404:
                return None
            raise
        rows = data.get("instances") if isinstance(data, dict) else None
        row = rows[0] if isinstance(rows, list) and rows else rows
        if not isinstance(row, dict):
            return None
        return row if _pos_int(row.get("id")) == int(instance_id) else None

    def instances_raw(self) -> list[dict]:
        """Every instance of the account, raw (so with `extra_env`: in memory only), all pages of /api/v1/instances/
        by 25 with next_token. Raises unless every page was read."""
        found: dict[int, dict] = {}
        token = ""
        for _ in range(MAX_PAGES):
            path = f"/api/v1/instances/?limit={PAGE_SIZE}"
            if token:
                path += "&after_token=" + urllib.parse.quote(token, safe="")
            data = self.get(path)
            rows = data if isinstance(data, list) else data.get("instances") if isinstance(data, dict) else None
            if not isinstance(rows, list):
                raise MarketError("vast.ai sent an instance list the app could not read.", "unreadable", False)
            for row in rows:
                iid = _pos_int(row.get("id")) if isinstance(row, dict) else None
                if not iid:  # a row that cannot be read might be an instance that bills
                    raise MarketError("vast.ai sent an instance without a usable id, so the list is not trusted.",
                                      "unreadable", False)
                found[iid] = row
            nxt = _text(data.get("next_token"), 4000) if isinstance(data, dict) else ""
            if not nxt:
                total = num(data.get("total_instances")) if isinstance(data, dict) else None
                if total is not None and int(total) != len(found):
                    raise MarketError("vast.ai's instance count does not match the list it sent, so the list is "
                                      "not trusted.", "unreadable", False)
                return list(found.values())
            if nxt == token:
                break
            token = nxt
        raise MarketError("vast.ai kept sending more pages of instances; the list may be incomplete.",
                          "unreadable", False)

    def put_instance(self, instance_id: int, body: dict, path: str | None = None) -> None:
        reply = self.call("PUT", path or f"/api/v0/instances/{int(instance_id)}/", body)
        self._require_ok(reply)

    def delete_instance(self, instance_id: int) -> None:
        self._require_ok(self.call("DELETE", f"/api/v0/instances/{int(instance_id)}/"))

    def _require_ok(self, reply: Reply) -> None:
        """`success: true` in a JSON body is the only yes. `success: false` is a refusal; anything else is unclear
        (kind `unclear`, not definite): the caller reads the instance again or the user looks on vast.ai."""
        if isinstance(reply.json, dict):
            if reply.json.get("success") is True:
                return
            if reply.json.get("success") is False:
                raise MarketError(f"vast.ai refused: {self._message(reply.json) or 'no reason given'}.", "refused",
                                  True)
        raise MarketError("vast.ai did not say whether that worked. Check the instance on vast.ai.", "unclear",
                          False)

    def rent(self, offer_id: int, body: dict) -> int:
        """PUT /asks/{offer}/, sent once whatever happens (a 429 included). Returns the new contract id. Raises with
        `definite` True only when vast.ai certainly did not rent (a 4xx answer or `success: false`)."""
        reply = self.call("PUT", f"/api/v0/asks/{int(offer_id)}/", body, retry429=False, timeout=RENT_TIMEOUT)
        if not reply.parsed or not isinstance(reply.json, dict):
            raise MarketError("vast.ai answered the rent request with something unreadable.", "unreadable", False)
        if reply.json.get("success") is False:
            raise MarketError(f"vast.ai refused the rent: {self._message(reply.json) or 'no reason given'}.",
                              "refused", True)
        contract = _pos_int(reply.json.get("new_contract"))
        if reply.json.get("success") is not True or contract is None:
            raise MarketError("vast.ai answered the rent request without a contract number.", "unreadable", False)
        return contract

    def ssh_keys(self) -> list[str]:
        data = self.get("/api/v0/ssh/")
        rows = data if isinstance(data, list) else data.get("keys") or data.get("ssh_keys") if isinstance(data, dict) \
            else None
        if not isinstance(rows, list):
            raise MarketError("vast.ai sent a key list the app could not read.", "unreadable", False)
        out = []
        for row in rows:
            v = row if isinstance(row, str) else (row.get("key") or row.get("public_key")) if isinstance(row, dict) \
                else None
            if isinstance(v, str) and v.strip():
                out.append(v.strip())
        return out

    def add_ssh_key(self, public_key: str) -> None:
        self.call("POST", "/api/v0/ssh/", {"ssh_key": public_key.strip()})

    def templates_raw(self, creator_id: int) -> list[dict]:
        """The account's own templates, raw: `env` holds secrets, so templates() reads it and drops it at once."""
        flt = urllib.parse.quote(json.dumps({"creator_id": {"eq": creator_id}}, separators=(",", ":")), safe="")
        data = self.get(f"/api/v0/template/?select_filters={flt}")
        rows = data.get("templates") if isinstance(data, dict) else data
        return [r for r in rows if isinstance(r, dict)] if isinstance(rows, list) else []


# ---------------------------------------------------------------- the search page's filters


@dataclass
class Filter:
    """One control of vast.ai's search page, with what a screen needs to draw it.

    kind: `range` (value [min|None, max|None]; lo / hi / step are the slider's bounds), `min` (one number or
    None: a minimum), `switch` (bool), `choice` (one of `options`, a list of (value, label)), `multi` (a list of
    values, from `options` when given), `text`. `field` is the API field and `conv` how the unit on screen becomes
    the API's: gb (MB, with the card tolerance), pct (percent -> fraction), days (-> seconds), per_tb (-> $/GB)."""

    key: str
    group: str
    label: str
    kind: str
    unit: str = ""
    lo: float | None = None
    hi: float | None = None
    step: float | None = None
    default: object = None
    hint: str = ""
    options: list = field(default_factory=list)
    field: str = ""
    conv: str = ""

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


GROUPS = ("Rental, GPU and region", "Disk", "Reliability", "Duration", "Machine options", "Price", "GPU", "Machine", "More")
NUM_GPUS_OPTIONS = [("any", "Any"), ("1", "1"), ("2", "2"), ("4", "4"), ("8", "8"), ("9+", "9+")]
TYPE_OPTIONS = [("ask", "On-demand"), ("bid", "Interruptible"), ("reserved", "Reserved")]
# key -> (label, API order); the site's nine, the ninth (video RAM per dollar) is sorted here
SORTS = {
    "auto": ("Auto", ("score", "desc")),
    "dlperf_per_dollar": ("DLPerf per $/hr", ("dlperf_per_dphtotal", "desc")),
    "dlperf": ("DLPerf", ("dlperf", "desc")),
    "tflops_per_dollar": ("TFLOPS per $/hr", ("flops_per_dphtotal", "desc")),
    "tflops": ("TFLOPS", ("total_flops", "desc")),
    "price_up": ("Price, low to high", ("dph_total", "asc")),
    "price_down": ("Price, high to low", ("dph_total", "desc")),
    "vram": ("Video RAM", ("gpu_total_ram", "desc")),
    "vram_per_dollar": ("Video RAM per $/hr", ("dph_total", "asc")),
}
SORT_OPTIONS = [(k, v[0]) for k, v in SORTS.items()]

# region -> country codes (the site's regions; a country is in one)
_COUNTRY_ROWS = """
US United States|na CA Canada|na MX Mexico|na CR Costa Rica|na PA Panama|na GT Guatemala|na DO Dominican Republic|na
PR Puerto Rico|na BR Brazil|sa AR Argentina|sa CL Chile|sa CO Colombia|sa PE Peru|sa UY Uruguay|sa EC Ecuador|sa
PY Paraguay|sa BO Bolivia|sa VE Venezuela|sa
GB United Kingdom|eu IE Ireland|eu FR France|eu DE Germany|eu NL Netherlands|eu BE Belgium|eu LU Luxembourg|eu
CH Switzerland|eu AT Austria|eu IT Italy|eu ES Spain|eu PT Portugal|eu SE Sweden|eu NO Norway|eu DK Denmark|eu
FI Finland|eu IS Iceland|eu PL Poland|eu CZ Czechia|eu SK Slovakia|eu HU Hungary|eu RO Romania|eu BG Bulgaria|eu
GR Greece|eu HR Croatia|eu SI Slovenia|eu RS Serbia|eu EE Estonia|eu LV Latvia|eu LT Lithuania|eu UA Ukraine|eu
MD Moldova|eu BY Belarus|eu RU Russia|eu CY Cyprus|eu MT Malta|eu AL Albania|eu BA Bosnia and Herzegovina|eu
MK North Macedonia|eu
TR Turkey|as IL Israel|as AE United Arab Emirates|as SA Saudi Arabia|as QA Qatar|as KW Kuwait|as IN India|as
PK Pakistan|as BD Bangladesh|as LK Sri Lanka|as NP Nepal|as CN China|as HK Hong Kong|as TW Taiwan|as JP Japan|as
KR South Korea|as MN Mongolia|as KZ Kazakhstan|as UZ Uzbekistan|as GE Georgia|as AM Armenia|as AZ Azerbaijan|as
TH Thailand|as VN Vietnam|as MY Malaysia|as SG Singapore|as ID Indonesia|as PH Philippines|as KH Cambodia|as
ZA South Africa|af EG Egypt|af MA Morocco|af DZ Algeria|af TN Tunisia|af NG Nigeria|af KE Kenya|af GH Ghana|af
ET Ethiopia|af AU Australia|oc NZ New Zealand|oc
"""
REGION_NAMES = {"earth": "Planet Earth", "na": "North America", "sa": "South America", "eu": "Europe", "as": "Asia",
                "af": "Africa", "oc": "Oceania"}
COUNTRY_NAMES: dict[str, str] = {}
REGION_OF: dict[str, str] = {}
for _row in re.finditer(r"([A-Z]{2}) ([^|]+)\|(\w\w)", _COUNTRY_ROWS):
    COUNTRY_NAMES[_row.group(1)] = _row.group(2).strip()
    REGION_OF[_row.group(1)] = _row.group(3)
REGIONS = {r: sorted(c for c, rr in REGION_OF.items() if rr == r) for r in REGION_NAMES if r != "earth"}
REGION_OPTIONS = list(REGION_NAMES.items())
FALLBACK_GPUS = ["RTX 5090", "RTX 5080", "RTX 4090", "RTX 4080", "RTX 3090", "RTX 3090 Ti", "RTX 3080", "RTX A6000",
                 "RTX A5000", "RTX 6000Ada", "RTX PRO 6000 WS", "L40S", "L40", "A100 PCIE", "A100 SXM4", "A100X",
                 "H100 SXM", "H100 NVL", "H200", "B200", "A40", "A10", "Tesla V100", "Q RTX 8000"]


def _f(key, group, label, kind, *rest, **kw) -> Filter:
    return Filter(key, group, label, kind, *rest, **kw)


def _rng(key, group, label, field_, unit, lo, hi, step, conv="", hint="", default=None) -> Filter:
    return Filter(key, group, label, "range", unit, lo, hi, step, default if default is not None else [None, None],
                  hint, [], field_, conv)


FILTERS: list[Filter] = [
    _f("type", "Rental, GPU and region", "Rental type", "choice", default="ask", options=TYPE_OPTIONS, field="type",
       hint="Reserved offers can only be looked at here; rent them on vast.ai."),
    _f("num_gpus", "Rental, GPU and region", "Number of GPUs", "choice", default="1", options=NUM_GPUS_OPTIONS, field="num_gpus"),
    _f("gpu_name", "Rental, GPU and region", "GPU model", "multi", field="gpu_name", default=[], hint="Any when none is picked."),
    _f("region", "Rental, GPU and region", "Region", "choice", default="earth", options=REGION_OPTIONS, field="geolocation"),
    _f("countries", "Rental, GPU and region", "Countries", "multi", default=[], field="geolocation",
       hint="Single countries; they take the place of the region."),
    _f("sort", "Rental, GPU and region", "Sort by", "choice", default="dlperf_per_dollar", options=SORT_OPTIONS),
    _f("disk", "Disk", "Container size", "min", "GB", 10, 4000, 10, 200, field="disk_space",
       hint="Also the disk the prices include."),
    _f("reliability", "Reliability", "Reliability", "min", "%", 0, 100, 1, 98, field="reliability2", conv="pct"),
    _f("duration", "Duration", "Rentable for at least", "min", "days", 0, 365, 1, None, field="duration", conv="days"),
    _f("secure_cloud", "Machine options", "Secure cloud only", "switch", default=False, field="datacenter"),
    _f("unverified", "Machine options", "Unverified machines", "switch", default=False, field="verified"),
    _f("incompatible", "Machine options", "Incompatible machines", "switch", default=False,
       hint="Machines the RulZ AI Trainer image cannot run on."),
    _f("unavailable", "Machine options", "Unavailable offers", "switch", default=False, field="rentable"),
    _f("static_ip", "Machine options", "Static IP", "switch", default=False, field="static_ip"),
    _f("min_cuda", "Machine options", "Minimum CUDA", "min", "", 10, 14, 0.1, None, field="cuda_max_good"),
    _f("min_driver", "Machine options", "Minimum driver version", "text", default="", field="driver_version"),
    _f("ubuntu_version", "Machine options", "Ubuntu version", "text", default="", field="ubuntu_version"),
    _rng("dph", "Price", "Price", "dph_total", "$/hour", 0, 20, 0.01),
    _rng("tflops_price", "Price", "Price per TFLOPS", "", "$/TFLOPS/hour", 0, 5, 0.01, hint="Worked out here."),
    _rng("up_cost", "Price", "Upload price", "inet_up_cost", "$/TB", 0, 100, 0.5, conv="per_tb"),
    _rng("down_cost", "Price", "Download price", "inet_down_cost", "$/TB", 0, 100, 0.5, conv="per_tb"),
    _rng("gpu_count", "GPU", "GPU count", "num_gpus", "", 1, 16, 1),
    _f("total_flops", "GPU", "Total TFLOPS (at least)", "min", "TFLOPS", 0, 2000, 1, None, field="total_flops"),
    _rng("gpu_ram", "GPU", "RAM per GPU", "gpu_ram", "GB", 0, 200, 1, "gb", default=[24, None]),
    _rng("gpu_total_ram", "GPU", "Total GPU RAM", "gpu_total_ram", "GB", 0, 2000, 1, "gb"),
    _rng("gpu_mem_bw", "GPU", "GPU RAM bandwidth", "gpu_mem_bw", "GB/s", 0, 10000, 10),
    _rng("pcie_bw", "GPU", "PCIe bandwidth", "pcie_bw", "GB/s", 0, 100, 1),
    _rng("bw_nvlink", "GPU", "NVLink bandwidth", "bw_nvlink", "GB/s", 0, 2000, 10),
    _rng("dlperf", "GPU", "DLPerf", "dlperf", "", 0, 2000, 1),
    _rng("cpu_cores", "Machine", "CPU cores", "cpu_cores_effective", "", 0, 512, 1),
    _rng("cpu_ram", "Machine", "CPU RAM", "cpu_ram", "GB", 0, 4096, 1, "gb"),
    _rng("cpu_ghz", "Machine", "CPU speed", "cpu_ghz", "GHz", 0, 8, 0.1),
    _rng("disk_bw", "Machine", "Disk bandwidth", "disk_bw", "MB/s", 0, 20000, 10),
    _rng("inet_up", "Machine", "Internet upload", "inet_up", "Mbps", 0, 100000, 10),
    _rng("inet_down", "Machine", "Internet download", "inet_down", "Mbps", 0, 100000, 10, default=[500, None]),
    _rng("direct_port_count", "Machine", "Open ports", "direct_port_count", "", 0, 1000, 1),
    _rng("pci_gen", "More", "PCIe generation", "pci_gen", "", 1, 6, 0.5),
    _rng("gpu_max_power", "More", "GPU max power", "gpu_max_power", "W", 0, 2000, 10),
    _f("has_avx", "More", "Has AVX", "switch", default=False, field="has_avx"),
    _rng("gpu_frac", "More", "GPU fraction", "gpu_frac", "", 0, 1, 0.05),
    _f("machine_id", "More", "Machine id", "text", default="", field="machine_id"),
    _f("host_id", "More", "Host id", "text", default="", field="host_id"),
]
FILTER_BY_KEY = {f.key: f for f in FILTERS}
REQUIRED = {"type", "num_gpus", "region", "sort", "disk"}  # a value that is cleared falls back to the default
_NAME_RE = re.compile(r"^[A-Za-z0-9 _.+()/-]{1,60}$")
_TEXT_RE = re.compile(r"^[A-Za-z0-9 _.+-]{0,40}$")


def _copy_default(f: Filter):
    return list(f.default) if isinstance(f.default, list) else f.default


def default_filters() -> dict:
    """On-demand, 1 GPU, 200 GB, 98%, verified, 24 GB+ per GPU, 500 Mbps down, DLPerf per $/hr."""
    return {f.key: _copy_default(f) for f in FILTERS}


def _clean_bound(v, f: Filter):
    n = num(v)
    if n is None:
        return None
    if f.lo is not None:
        n = max(n, f.lo)
    if f.hi is not None:
        n = min(n, f.hi)
    return _clean_number(n)


def clean_filters(raw) -> dict:
    """Every filter, every value forced into range; unknown keys are dropped and a bad value is the default."""
    raw = raw if isinstance(raw, dict) else {}
    out = default_filters()
    for f in FILTERS:
        if f.key not in raw:
            continue
        v = raw[f.key]
        if f.kind == "range":
            pair = v if isinstance(v, (list, tuple)) and len(v) == 2 else None
            if pair is None:
                continue
            lo, hi = _clean_bound(pair[0], f), _clean_bound(pair[1], f)
            lo = None if lo is not None and lo == f.lo else lo  # a bound on the slider's own end limits nothing
            hi = None if hi is not None and hi == f.hi else hi
            if lo is not None and hi is not None and lo > hi:
                lo, hi = hi, lo
            out[f.key] = [lo, hi]
        elif f.kind == "min":
            n = None if v is None or v == "" else _clean_bound(v, f)
            if n is None and f.key in REQUIRED:
                n = f.default
            out[f.key] = n
        elif f.kind == "switch":
            out[f.key] = v is True or v == 1 or (isinstance(v, str) and v.lower() in ("1", "true", "yes", "on"))
        elif f.kind == "choice":
            allowed = {str(o[0]) for o in f.options}
            out[f.key] = str(v) if str(v) in allowed else f.default
        elif f.kind == "multi":
            out[f.key] = _clean_multi(f, v)
        elif f.kind == "text":
            s = v.strip() if isinstance(v, str) else (str(int(v)) if isinstance(v, (int, float))
                                                       and not isinstance(v, bool) else "")
            if f.key in ("machine_id", "host_id"):
                ok = s.isdigit()
            elif f.key == "min_driver":
                ok = bool(re.fullmatch(r"\d{1,3}(\.\d{1,3}){0,2}", s))
            else:
                ok = bool(_TEXT_RE.match(s))
            out[f.key] = s if (ok or not s) and len(s) <= 40 else ""
    return out


def _clean_multi(f: Filter, v) -> list:
    if not isinstance(v, (list, tuple)):
        return []
    seen, out = set(), []
    for item in v:
        if not isinstance(item, str):
            continue
        s = item.strip()
        if f.key == "countries":
            s = s.upper()
            ok = bool(re.fullmatch(r"[A-Z]{2}", s))
        else:
            ok = bool(_NAME_RE.match(s))
        if ok and s not in seen:
            seen.add(s)
            out.append(s)
    return out[:80]


def _conv(v: float, conv: str, high: bool):
    """The screen's unit to the API's. A card sold as N GB reports a little under N * 1000 MB or N * 1024: a minimum
    of N GB is sent as N * 1000 * 0.98, a maximum as N * 1024."""
    if conv == "gb":
        return _clean_number(v * 1024 if high else v * 1000 * RAM_TOLERANCE)
    if conv == "pct":
        return _clean_number(v / 100)
    if conv == "days":
        return _clean_number(v * 86400)
    if conv == "per_tb":
        return _clean_number(v / 1000)
    return _clean_number(v)


def _merge(body: dict, field_: str, op: str, value) -> None:
    """Add one operator to a field, keeping the stricter bound when two controls ask about the same field."""
    cur = body.setdefault(field_, {})
    if op in ("gte", "lte") and op in cur:
        value = max(cur[op], value) if op == "gte" else min(cur[op], value)
    cur[op] = value


def _tag_cuda(tag: str) -> float | None:
    m = re.search(r"cuda-?(\d+)\.(\d+)", tag or "")
    return float(f"{m.group(1)}.{m.group(2)}") if m else None


def _template_image(template: dict | None) -> tuple[str, float | None]:
    t = template or BUILTIN
    image = _text(t.get("image")) or BUILTIN_IMAGE
    tag = _text(t.get("tag"))
    return (f"{image}:{tag}" if tag else image), (num(t.get("cuda")) or _tag_cuda(tag))


def _gpu_count_bounds(f: dict) -> tuple[int | None, int | None]:
    lo, hi = None, None
    choice = f["num_gpus"]
    if choice == "9+":
        lo = 9
    elif choice != "any":
        lo = hi = int(choice)
    r = f["gpu_count"]
    if r[0] is not None:
        lo = int(max(lo or 0, r[0]))
    if r[1] is not None:
        hi = int(min(hi if hi is not None else r[1], r[1]))
    return lo, hi


def _driver_number(text: str) -> int:
    """vast.ai compares driver versions as numbers: 535.86.05 is 535086005 (each part padded to 3 digits)."""
    parts = (text.split(".") + ["0", "0"])[:3]
    return int("".join(p.zfill(3) for p in parts))


def search_body(filters, template: dict | None = None) -> dict:
    """The POST /bundles/ body for these filters (see FILTERS for what each control becomes)."""
    f = clean_filters(filters)
    disk = f["disk"]
    body: dict = {"limit": SEARCH_LIMIT, "type": f["type"], "allocated_storage": disk,
                  "order": [list(SORTS[f["sort"]][1])], "external": {"eq": False}, "disk_space": {"gte": disk}}
    for fl in FILTERS:
        if fl.kind == "range" and fl.field and fl.key != "gpu_count":
            lo, hi = f[fl.key]
            if lo is not None:
                _merge(body, fl.field, "gte", _conv(lo, fl.conv, False))
            if hi is not None:
                _merge(body, fl.field, "lte", _conv(hi, fl.conv, True))
        elif fl.kind == "min" and fl.field and fl.key != "disk" and f[fl.key] is not None:
            _merge(body, fl.field, "gte", _conv(f[fl.key], fl.conv, False))
    lo, hi = _gpu_count_bounds(f)
    if lo is not None and lo == hi:
        body["num_gpus"] = {"eq": lo}
    else:
        if lo is not None:
            _merge(body, "num_gpus", "gte", lo)
        if hi is not None:
            _merge(body, "num_gpus", "lte", hi)
    if f["gpu_name"]:
        body["gpu_name"] = {"in": list(f["gpu_name"])}
    codes = f["countries"] or ([] if f["region"] == "earth" else REGIONS.get(f["region"], []))
    if codes:
        body["geolocation"] = {"in": list(codes)}
    if f["secure_cloud"]:
        body["datacenter"] = {"eq": True}
    if not f["unverified"]:
        body["verified"] = {"eq": True}
    if not f["unavailable"]:
        body["rentable"] = {"eq": True}
    if f["static_ip"]:
        body["static_ip"] = {"eq": True}
    if f["has_avx"]:
        body["has_avx"] = {"eq": True}
    if f["min_driver"]:
        body["driver_version"] = {"gte": _driver_number(f["min_driver"])}
    if f["ubuntu_version"]:
        body["ubuntu_version"] = {"eq": f["ubuntu_version"]}
    for key in ("machine_id", "host_id"):
        if f[key]:
            body[key] = {"eq": int(f[key])}
    if not f["incompatible"]:
        image, cuda = _template_image(template)
        if cuda:
            _merge(body, "cuda_max_good", "gte", cuda)
        body["image_tag"] = {"eq": image}
        _merge(body, "direct_port_count", "gte", 4)
        body["cpu_arch"] = {"in": ["amd64"]}
    return body


# ---------------------------------------------------------------- offers


@dataclass
class Offer:
    id: int
    machine_id: int = 0
    host_id: int = 0
    gpu_name: str = ""
    num_gpus: int = 1
    gpu_ram_mb: float = 0
    gpu_total_ram_mb: float = 0
    dph_total: float = 0
    dph_base: float | None = None
    storage_dph: float | None = None
    min_bid: float | None = None
    dlperf: float | None = None
    dlperf_per_dollar: float | None = None
    tflops: float | None = None
    tflops_per_dollar: float | None = None
    reliability: float | None = None
    inet_up: float | None = None
    inet_down: float | None = None
    inet_up_per_gb: float | None = None
    inet_down_per_gb: float | None = None
    cpu_name: str = ""
    cpu_cores: float | None = None
    cpu_cores_total: float | None = None
    cpu_ram_mb: float | None = None
    cpu_ghz: float | None = None
    disk_bw: float | None = None
    disk_space: float | None = None
    disk_name: str = ""
    pcie_bw: float | None = None
    pci_gen: float | None = None
    gpu_mem_bw: float | None = None
    bw_nvlink: float | None = None
    cuda: float | None = None
    driver: str = ""
    os_version: str = ""
    geolocation: str = ""
    country: str = ""
    verified: bool = False
    datacenter: bool = False
    static_ip: bool = False
    ports: int | None = None
    duration_days: float | None = None
    mobo: str = ""
    rentable: bool = True
    type: str = "ask"
    ask_contract_id: int | None = None

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _nonneg(v) -> float | None:
    n = num(v)
    return n if n is not None and n >= 0 else None


def parse_offer(raw, type: str = "ask") -> Offer | None:  # noqa: A002
    """An Offer from one search row; None for a row that cannot be rented from (no id, no price, no GPU)."""
    if not isinstance(raw, dict):
        return None
    oid = _pos_int(raw.get("id")) or _pos_int(raw.get("ask_contract_id"))
    dph = _nonneg(raw.get("dph_total"))
    gpus = _pos_int(raw.get("num_gpus"))
    ram = num(raw.get("gpu_ram"))
    if oid is None or dph is None or gpus is None or ram is None or ram <= 0:
        return None
    total_ram = _nonneg(raw.get("gpu_total_ram")) or ram * gpus
    tflops = _nonneg(raw.get("total_flops"))
    per_dollar = _nonneg(raw.get("flops_per_dphtotal"))
    if per_dollar is None and tflops and dph > 0:
        per_dollar = tflops / dph
    rel = num(raw.get("reliability2"))
    if rel is None:
        rel = num(raw.get("reliability"))
    geo = _text(raw.get("geolocation"), 80)
    dur = _nonneg(raw.get("duration"))
    hosting = raw.get("hosting_type")
    return Offer(
        id=oid, machine_id=_pos_int(raw.get("machine_id")) or 0, host_id=_pos_int(raw.get("host_id")) or 0,
        gpu_name=_text(raw.get("gpu_name"), 80), num_gpus=gpus, gpu_ram_mb=ram, gpu_total_ram_mb=total_ram,
        dph_total=dph, dph_base=_nonneg(raw.get("dph_base")), storage_dph=_nonneg(raw.get("storage_total_cost")),
        min_bid=_nonneg(raw.get("min_bid")), dlperf=_nonneg(raw.get("dlperf")),
        dlperf_per_dollar=_nonneg(raw.get("dlperf_per_dphtotal")), tflops=tflops, tflops_per_dollar=per_dollar,
        reliability=None if rel is None else min(1.0, max(0.0, rel)),
        inet_up=_nonneg(raw.get("inet_up")), inet_down=_nonneg(raw.get("inet_down")),
        inet_up_per_gb=_nonneg(raw.get("inet_up_cost")), inet_down_per_gb=_nonneg(raw.get("inet_down_cost")),
        cpu_name=_text(raw.get("cpu_name"), 80), cpu_cores=_nonneg(raw.get("cpu_cores_effective")),
        cpu_cores_total=_nonneg(raw.get("cpu_cores")), cpu_ram_mb=_nonneg(raw.get("cpu_ram")),
        cpu_ghz=_nonneg(raw.get("cpu_ghz")), disk_bw=_nonneg(raw.get("disk_bw")),
        disk_space=_nonneg(raw.get("disk_space")), disk_name=_text(raw.get("disk_name"), 80),
        pcie_bw=_nonneg(raw.get("pcie_bw")), pci_gen=_nonneg(raw.get("pci_gen")),
        gpu_mem_bw=_nonneg(raw.get("gpu_mem_bw")), bw_nvlink=_nonneg(raw.get("bw_nvlink")),
        cuda=_nonneg(raw.get("cuda_max_good")), driver=_text(raw.get("driver_version"), 40),
        os_version=_text(raw.get("os_version"), 40), geolocation=geo,
        country=geo.rsplit(",", 1)[-1].strip() if "," in geo else geo[-2:] if len(geo) == 2 else "",
        verified=raw.get("verification") == "verified", datacenter=hosting == 1 or raw.get("datacenter") is True,
        static_ip=raw.get("static_ip") is True, ports=_pos_int(raw.get("direct_port_count")),
        duration_days=None if not dur else dur / 86400, mobo=_text(raw.get("mobo_name"), 80),
        rentable=raw.get("rentable") is not False, type=type, ask_contract_id=_pos_int(raw.get("ask_contract_id")))


def client_side(offers: list[Offer], filters) -> list[Offer]:
    """What the API cannot do: the price per TFLOPS range, and the sort by video RAM per dollar."""
    f = clean_filters(filters)
    lo, hi = f["tflops_price"]
    out = list(offers)
    if lo is not None or hi is not None:
        keep = []
        for o in out:
            if not o.tflops or o.tflops <= 0:
                continue
            price = o.dph_total / o.tflops
            if (lo is None or price >= lo) and (hi is None or price <= hi):
                keep.append(o)
        out = keep
    if f["sort"] == "vram_per_dollar":
        out.sort(key=lambda o: o.gpu_total_ram_mb / o.dph_total if o.dph_total > 0 else float("inf"), reverse=True)
    return out


def search(api: Api, filters, template: dict | None = None) -> list[Offer]:
    """Offers for these filters, parsed, unusable rows dropped, client-side rules applied."""
    f = clean_filters(filters)
    offers = [o for o in (parse_offer(r, f["type"]) for r in api.search(search_body(f, template))) if o]
    return client_side(offers, f)


# ---------------------------------------------------------------- the catalog of GPU models and countries


def _market_file(home: Path | None) -> Path:
    return (home or STATE) / "market.json"


def catalog(api: Api, home: Path | None = None, now=time.time, refresh: bool = False) -> dict:
    """GPU models and countries seen in a wide live search, cached a day in market.json, with the built-in lists
    when the search fails. {"gpus": [...], "countries": [{"code", "name", "region"}], "live": bool}."""
    path = _market_file(home)
    data = _load_json(path, {})
    cached = data.get("catalog") if isinstance(data.get("catalog"), dict) else None
    if cached and not refresh and now() - (num(cached.get("at")) or 0) < 86400:
        return _catalog_out(cached.get("gpus"), cached.get("countries"), True)
    try:
        rows = api.search({"limit": SEARCH_LIMIT, "type": "ask", "external": {"eq": False},
                           "rentable": {"eq": True}, "order": [["score", "desc"]]})
    except MarketError:
        return _catalog_out(cached.get("gpus") if cached else None, cached.get("countries") if cached else None,
                            bool(cached))
    gpus = sorted({_text(r.get("gpu_name"), 60) for r in rows if _text(r.get("gpu_name"), 60)})
    codes = set()
    for r in rows:
        geo = _text(r.get("geolocation"), 80)
        code = geo.rsplit(",", 1)[-1].strip() if "," in geo else ""
        if re.fullmatch(r"[A-Z]{2}", code):
            codes.add(code)
    if gpus:
        try:
            with QueueLock(path.with_name(path.name + ".lock")):
                data = _load_json(path, {}, strict=True)  # a file that cannot be read is not rewritten
                data["catalog"] = {"at": now(), "gpus": gpus, "countries": sorted(codes)}
                _save_json(path, data)
        except (MarketError, OSError):
            pass  # the cache is a convenience
    return _catalog_out(gpus, sorted(codes), bool(gpus))


def _catalog_out(gpus, codes, live: bool) -> dict:
    gpus = [g for g in (gpus or []) if isinstance(g, str)] or list(FALLBACK_GPUS)
    codes = [c for c in (codes or []) if isinstance(c, str) and len(c) == 2] or sorted(COUNTRY_NAMES)
    return {"gpus": gpus, "live": live,
            "countries": [{"code": c, "name": COUNTRY_NAMES.get(c, c), "region": REGION_OF.get(c, "")}
                          for c in sorted(codes, key=lambda c: COUNTRY_NAMES.get(c, c))]}


def gpu_names(api: Api, home: Path | None = None, now=time.time) -> list[str]:
    return catalog(api, home, now)["gpus"]


def country_list(api: Api, home: Path | None = None, now=time.time) -> list[dict]:
    return catalog(api, home, now)["countries"]


# ---------------------------------------------------------------- instances and the account


_ON_THE_WAY = ("loading", "created", "scheduling", "connecting", "")


def _env_names_and_values(raw_env) -> dict:
    """The env of an instance (a list of [name, value] pairs, or a dict) or a template (a string like
    "-p 1:1 -e NAME=value") as {name: value}. In memory only, for the is-RulZ verdict."""
    out: dict = {}
    if isinstance(raw_env, dict):
        out = {str(k): str(v) for k, v in raw_env.items()}
    elif isinstance(raw_env, list):
        for pair in raw_env:
            if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                out[str(pair[0])] = str(pair[1])
    elif isinstance(raw_env, str):
        try:
            tokens = shlex.split(raw_env)
        except ValueError:
            tokens = raw_env.split()
        i = 0
        while i < len(tokens):
            if tokens[i] == "-e" and i + 1 < len(tokens) and "=" in tokens[i + 1]:
                name, _, value = tokens[i + 1].partition("=")
                out[name] = value
                i += 2
            else:
                i += 1
    return out


def _script_verdict(env: dict) -> tuple[bool, bool]:
    """(sets RulZ Trainer up, runs the suite's current branch): RULZ_* names, or a PROVISIONING_SCRIPT with
    `rulz` and `provision.sh` in it."""
    script = env.get("PROVISIONING_SCRIPT", "").lower()
    rulz = any(k.startswith("RULZ_") for k in env) or ("rulz" in script and "provision.sh" in script)
    return rulz, bool(script) and TEMPLATE_BRANCH in script


def _is_rulz_instance(raw: dict) -> bool:
    env = _env_names_and_values(raw.get("extra_env"))
    rulz, _ = _script_verdict(env)
    image = (str(raw.get("image_uuid") or "") + str(raw.get("image") or "")).lower()
    return rulz or "rulztrainer" in image


def _status(raw: dict) -> tuple[str, str]:
    actual = _text(raw.get("actual_status"), 40).lower()
    intended = _text(raw.get("intended_status"), 40).lower()
    return actual, intended


def parse_instance(raw) -> dict | None:
    """One instance as the app uses it. The environment is read for `is_rulz` and dropped here.

    {id, label, status, actual_status, intended_status, status_msg, gpu_name, num_gpus, gpu_ram_mb, dph_total,
    storage_per_hour, disk_gb, billing, per_hour, image, started, geolocation, min_bid, is_rulz, ssh_host,
    ssh_port, public_ip}. `per_hour` is what is billed now: the full price while running or on its way to it,
    else the disk."""
    if not isinstance(raw, dict):
        return None
    iid = _pos_int(raw.get("id"))
    if iid is None:
        return None
    actual, intended = _status(raw)
    disk = max(0.0, num(raw.get("disk_space")) or 0.0)
    total = num(raw.get("storage_total_cost"))
    per_gb_month = num(raw.get("storage_cost"))
    if total is not None and total >= 0:
        storage = total
    elif per_gb_month is not None and per_gb_month >= 0 and disk > 0:
        storage = per_gb_month * disk / 730
    else:
        storage = 0.0
    dph = _nonneg(raw.get("dph_total")) or 0.0
    billing = actual == "running" or (intended == "running" and actual in _ON_THE_WAY)
    start = num(raw.get("start_date"))
    return {
        "id": iid, "label": _text(raw.get("label"), 120),
        "status": actual or ("loading" if intended == "running" else intended or "unknown"),
        "actual_status": actual, "intended_status": intended, "status_msg": scrub_text(raw.get("status_msg"), 400),
        "gpu_name": _text(raw.get("gpu_name"), 80), "num_gpus": max(1, _pos_int(raw.get("num_gpus")) or 1),
        "gpu_ram_mb": _nonneg(raw.get("gpu_ram")) or 0, "dph_total": dph, "storage_per_hour": storage,
        "disk_gb": disk, "billing": billing, "per_hour": dph if billing else storage,
        "image": _text(raw.get("image_uuid") or raw.get("image"), 200),
        "started": start if start and start > 0 else None, "geolocation": _text(raw.get("geolocation"), 80),
        "min_bid": _nonneg(raw.get("min_bid")), "is_rulz": _is_rulz_instance(raw),
        "ssh_host": _text(raw.get("ssh_host"), 253), "ssh_port": _pos_int(raw.get("ssh_port")),
        "public_ip": _text(raw.get("public_ipaddr"), 100),
    }


def account(api: Api, now=time.time) -> dict:
    """{"credit", "instances": [parsed], "burn": {"per_hour", "running", "stopped", "gpus", "disks",
    "running_per_hour", "stopped_per_hour"}, "hours_left", "empty_at"}.

    credit is users/current's `credit` plus `balance` when that is positive. `burn.running` / `stopped` count the
    instances billed at the full price / for their disk only, `gpus` the GPUs of the first, `disks` the GB held by
    all. hours_left and empty_at (epoch seconds) are None when nothing is billed. Traffic is never included."""
    user = api.current_user()
    credit = num(user.get("credit"))
    balance = num(user.get("balance"))
    if balance is not None and balance > 0:
        credit = (credit or 0.0) + balance
    insts = [i for i in (parse_instance(r) for r in api.instances_raw()) if i]
    run = [i for i in insts if i["billing"]]
    rest = [i for i in insts if not i["billing"]]
    run_cost = sum(i["per_hour"] for i in run)
    rest_cost = sum(i["per_hour"] for i in rest)
    per_hour = run_cost + rest_cost
    hours = lasts(credit, per_hour)
    return {"credit": credit, "instances": insts,
            "burn": {"per_hour": per_hour, "running": len(run), "stopped": len(rest),
                     "gpus": sum(i["num_gpus"] for i in run), "disks": sum(i["disk_gb"] for i in insts),
                     "running_per_hour": run_cost, "stopped_per_hour": rest_cost},
            "hours_left": hours, "empty_at": None if hours is None else now() + hours * 3600}


# ---------------------------------------------------------------- templates


def templates(api: Api) -> list[dict]:
    """The account's own templates that set RulZ Trainer up, newest first, one per hash_id:
    {id, hash_id, name, image, tag, disk, created, current, cuda}. Nothing of `env` is kept: its values are
    read for the verdicts and dropped. `current` = its provisioning script is the suite's app/rulz-trainer branch."""
    uid = _pos_int(api.current_user().get("id"))
    if uid is None:
        return []
    found, seen = [], set()
    for raw in api.templates_raw(uid):
        env = _env_names_and_values(raw.get("env"))
        rulz, current = _script_verdict(env)
        env.clear()
        image = _text(raw.get("image"), 200)
        if not rulz and "rulztrainer" not in image.lower():
            continue
        hash_id = _text(raw.get("hash_id"), 80)
        if not hash_id or hash_id in seen:
            continue
        seen.add(hash_id)
        tag = _text(raw.get("tag"), 120)
        found.append({"id": _pos_int(raw.get("id")), "hash_id": hash_id, "name": _text(raw.get("name"), 120),
                      "image": image, "tag": tag, "disk": num(raw.get("recommended_disk_space")),
                      "created": num(raw.get("created_at")) or 0, "current": current, "cuda": _tag_cuda(tag)})
    found.sort(key=lambda t: t["created"], reverse=True)
    return found


def default_template(items: list[dict]) -> dict:
    """The newest current template, else the built-in one."""
    return next((t for t in items if t.get("current")), dict(BUILTIN))


# ---------------------------------------------------------------- local keys


def local_public_keys(ssh_dir: Path | None = None) -> list[str]:
    """Public keys on this PC, the key RULZ_VAST_KEY names first, then ~/.ssh's ed25519, ecdsa, rsa."""
    paths = []
    own = os.environ.get("RULZ_VAST_KEY")
    if own:
        paths.append(Path(own + ".pub"))
    base = ssh_dir or Path.home() / ".ssh"
    paths += [base / n for n in SSH_KEY_NAMES]
    keys = []
    for p in paths:
        try:
            text = p.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if text.startswith(("ssh-", "ecdsa-", "sk-")) and text not in keys:
            keys.append(text.splitlines()[0])
    return keys


def _key_material(key: str) -> str:
    return " ".join(key.split()[:2])


# ---------------------------------------------------------------- rentals


def _builtin_body(disk: int, label: str, bid) -> dict:
    return {"image": f"{BUILTIN_IMAGE}:{BUILTIN_TAG}", "env": dict(BUILTIN_ENV), "onstart": BUILTIN_ONSTART,
            "runtype": BUILTIN_RUNTYPE, "disk": disk, "label": label, "target_state": "running",
            "cancel_unavail": True, **({} if bid is None else {"price": bid})}


def _try_lock(path: Path):
    """The OS lock on `path` if it is free right now (an open file that holds it), else None. The OS drops it when
    the holder dies, so a crash cannot leave it stuck."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fh = open(path, "a+b")  # noqa: SIM115
    except OSError:
        return None
    try:
        fh.seek(0)
        if sys.platform == "win32":
            import msvcrt
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return None
    return fh


class Market:
    """Everything that bills, with the records that make it safe. `home` is where rentals.json and market.json
    live (remote.STATE by default); `api` an Api; `now` and `sleep` are replaced by tests. `which` finds ssh and
    `ssh_dir` holds the local public keys (also replaced by tests)."""

    def __init__(self, home: Path | None = None, api: Api | None = None, now=time.time, sleep=time.sleep,
                 which=shutil.which, ssh_dir: Path | None = None):
        self.home = Path(home) if home else STATE
        self.api = api or Api()
        self.now = now
        self.sleep = sleep
        self.which = which
        self.ssh_dir = ssh_dir
        self.lock = QueueLock(self.home / "rentals.json.lock")
        self.prefs_lock = QueueLock(self.home / "market.json.lock")
        self._sending: set[str] = set()  # labels of rents this process is sending right now
        self._busy = threading.Lock()

    # -- files

    @property
    def path(self) -> Path:
        return self.home / "rentals.json"

    def _load(self) -> dict:
        data = _load_json(self.path, {}, strict=True)
        for name in ("rentals", "intents"):
            if not isinstance(data.get(name), list):
                data[name] = []
        return data

    def _save(self, data: dict) -> None:
        _save_json(self.path, {"rentals": data["rentals"], "intents": data["intents"]})

    def rentals(self) -> list[dict]:
        with self.lock:
            return [dict(r) for r in self._load()["rentals"]]

    def intents(self) -> list[dict]:
        with self.lock:
            return [dict(i) for i in self._load()["intents"]]

    def search(self, filters=None, template: dict | None = None) -> list[Offer]:
        return search(self.api, self.prefs()["filters"] if filters is None else filters, template)

    def account(self) -> dict:
        data = account(self.api, self.now)
        ours = {r["id"] for r in self.rentals() if not r.get("destroyed")}
        for i in data["instances"]:
            i["ours"] = i["id"] in ours
        return data

    def templates(self) -> list[dict]:
        return templates(self.api)

    def instances(self) -> list[dict]:
        """The parsed v1 list with a fresh `is_rulz`, `status`, `per_hour` and `ours` (rented by this app)."""
        ours = {r["id"] for r in self.rentals() if not r.get("destroyed")}
        out = [i for i in (parse_instance(r) for r in self.api.instances_raw()) if i]
        for i in out:
            i["ours"] = i["id"] in ours
        return out

    # -- ssh

    def ssh_key_state(self) -> str:
        """`missing` (no public key on this PC), `on_account` or `not_on_account`."""
        keys = local_public_keys(self.ssh_dir)
        if not keys:
            return "missing"
        have = {_key_material(k) for k in self.api.ssh_keys()}
        return "on_account" if any(_key_material(k) in have for k in keys) else "not_on_account"

    def add_ssh_key(self) -> dict:
        """Puts this PC's public key on the account (it changes the account: only from a click)."""
        keys = local_public_keys(self.ssh_dir)
        if not keys:
            raise MarketError("There is no ssh public key on this PC (~/.ssh/id_ed25519.pub). Make one with "
                              "ssh-keygen first.", "ssh", True)
        if self.ssh_key_state() == "on_account":
            return {"added": False}
        self.api.add_ssh_key(keys[0])
        return {"added": True}

    def reachable_before_rent(self) -> None:
        """Amendment 8: ssh on PATH, a local public key, and that key on the account; else refused with the fix."""
        if not self.which("ssh"):
            raise MarketError("ssh is not installed on this PC, so a rented instance could not be reached. "
                              "Install the OpenSSH client first.", "ssh", True)
        try:
            state = self.ssh_key_state()
        except MarketError as e:
            raise MarketError(f"Could not check your ssh key on vast.ai: {e.message}", "ssh", True) from None
        if state == "missing":
            raise MarketError("There is no ssh public key on this PC. Make one with ssh-keygen, then add it to "
                              "your vast.ai account.", "ssh", True)
        if state != "on_account":
            raise MarketError("Your ssh key is not on your vast.ai account yet, so a rented instance could not be "
                              "reached. Add it first (Add ssh key).", "ssh", True)

    # -- renting

    def rent(self, offer, filters=None, template: dict | None = None, disk: int | None = None,
             label_note: str = "", bid: float | None = None, send_id: str | None = None) -> dict:
        """Rent an offer, once. Returns the rental record. Raises MarketError: `definite` True = nothing was
        rented; kind `pending` (definite False) = the answer was lost, an intent is checking the account for the
        label (see settle(), check_intent()), and no other rent is allowed until it is settled.

        The offer is looked up again right before the PUT (by ask_contract_id, with the same allocated_storage):
        gone -> refused; its price more than 0.5% above the one shown -> refused with the new price; for an
        interruptible rent the bid must reach the fresh min_bid."""
        o = offer if isinstance(offer, Offer) else parse_offer(offer, clean_filters(filters)["type"])
        if o is None:
            raise MarketError("That offer is not valid.", "bad", True)
        f = clean_filters(filters)
        kind = o.type if o.type in ("ask", "bid", "reserved") else f["type"]
        if kind == "reserved":
            raise MarketError("Reserved offers cannot be rented here. Rent them on vast.ai.", "reserved", True)
        disk = int(disk or f["disk"])
        if kind == "bid":
            if num(bid) is None or num(bid) <= 0:
                raise MarketError("An interruptible rent needs a bid.", "bid", True)
            bid = float(bid)
        else:
            bid = None
        shown = o.dph_total if kind == "ask" else bid + (o.storage_dph or 0.0)
        if not shown or shown <= 0:
            raise MarketError("The price shown is not valid.", "bad", True)
        if not self._busy.acquire(blocking=False):
            raise MarketError("Another rent is being sent. Wait for it to finish.", "busy", True)
        held = _try_lock(self.home / "rent.lock")  # held for the whole rent, so no other process settles it
        try:
            if held is None:
                raise MarketError("Another rent is being sent (by another window of this app). Wait for it to "
                                  "finish.", "busy", True)
            return self._rent_locked(o, kind, f, template, disk, label_note, bid, send_id, shown)
        finally:
            if held is not None:
                held.close()
            self._busy.release()

    def _rent_locked(self, o: Offer, kind: str, f: dict, template, disk: int, note: str, bid, send_id, shown) -> dict:
        with self.lock:
            if self._load()["intents"]:
                raise MarketError("Another rent is still open. Wait until it is settled (or dismiss it once you have "
                                  "checked vast.ai) before renting again.", "busy", True)
        self.reachable_before_rent()
        base = {"limit": 10, "type": kind, "allocated_storage": disk, "external": {"eq": False},
                "order": [["dph_total", "asc"]]}
        fresh_raw = self.api.find_offer(o.id, base)
        fresh = parse_offer(fresh_raw, kind) if fresh_raw else None
        if fresh is None or not fresh.rentable:
            raise MarketError("That offer is gone. Search again.", "gone", True)
        if kind == "ask":
            if fresh.dph_total > shown * PRICE_RISE_REFUSE:
                raise MarketError(f"The price rose from ${shown:.3f} to ${fresh.dph_total:.3f} per hour, so nothing "
                                  "was rented. Search again to see the current price.", "price", True,
                                  new_price=fresh.dph_total)
        else:
            if fresh.min_bid is None or fresh.storage_dph is None:
                raise MarketError("vast.ai's offer does not show its minimum bid or its disk price, so the cost "
                                  "cannot be checked and nothing was rented.", "price", True)
            if bid < fresh.min_bid:
                raise MarketError(f"The minimum bid is now ${fresh.min_bid:.3f} per hour; yours is ${bid:.3f}, so "
                                  "nothing was rented.", "bid", True, min_bid=fresh.min_bid)
            if bid + fresh.storage_dph > shown * PRICE_RISE_REFUSE:
                raise MarketError("The disk price rose, so nothing was rented. Search again.", "price", True,
                                  new_price=bid + fresh.storage_dph)
        label = "rulz-trainer-" + uuid.uuid4().hex[:8]
        if template and template.get("hash_id"):
            body = {"template_hash_id": template["hash_id"], "disk": disk, "label": label,
                    "target_state": "running", "cancel_unavail": True}
            if bid is not None:
                body["price"] = bid
        else:
            body = _builtin_body(disk, label, bid)
        intent = {"label": label, "offer": o.id, "price": shown, "send": send_id, "created": self.now(),
                  "disk": disk, "bid": bid, "note": _text(note, 200), "gpu_name": o.gpu_name, "num_gpus": o.num_gpus,
                  "template": (template or {}).get("hash_id"), "type": kind, "state": "sending", "done": 0,
                  "since": None, "message": ""}
        with self.lock:
            data = self._load()
            if data["intents"]:
                raise MarketError("Another rent is still open.", "busy", True)
            data["intents"].append(intent)
            self._save(data)  # fsynced before the PUT
        self._sending.add(label)
        try:
            contract = self.api.rent(o.id, body)
        except MarketError as e:
            self._sending.discard(label)
            if e.definite:
                self._drop_intent(label)
                raise
            self._set_intent(label, state="checking", since=self.now(), done=0,
                             message="The rent request got no clear answer. It may have been rented: the app is "
                                     "checking your vast.ai instances for it. Do not rent again yet.")
            raise MarketError("The rent request got no clear answer, so it may have been made. The app is checking "
                              "your vast.ai instances for it; nothing else can be rented until that is settled.",
                              "pending", False, label=label) from None
        self._sending.discard(label)
        return self._adopt(intent, contract, None)

    def _drop_intent(self, label: str) -> None:
        with self.lock:
            data = self._load()
            data["intents"] = [i for i in data["intents"] if i.get("label") != label]
            self._save(data)

    def _set_intent(self, label: str, **change) -> None:
        with self.lock:
            data = self._load()
            for i in data["intents"]:
                if i.get("label") == label:
                    i.update(change)
            self._save(data)

    def _adopt(self, intent: dict, contract_id: int, found: dict | None) -> dict:
        """The rental exists: read back its real price, record it, close the intent."""
        real = found
        if real is None:
            try:
                real = parse_instance(self.api.show_instance(contract_id))
            except MarketError:
                real = None
        real_price = real["dph_total"] if real and real["dph_total"] > 0 else None
        shown = intent["price"]
        rental = {"id": contract_id, "label": intent["label"], "offer": intent["offer"], "price": shown,
                  "real_price": real_price, "price_over": bool(real_price and real_price > shown * PRICE_RISE_REPORT),
                  "bid": intent.get("bid"), "type": intent.get("type", "ask"), "disk": intent["disk"],
                  "gpu_name": intent.get("gpu_name", ""), "num_gpus": intent.get("num_gpus", 1),
                  "template": intent.get("template"), "send": intent.get("send"), "note": intent.get("note", ""),
                  "created": intent["created"], "adopted": self.now(), "destroyed": None}
        if rental["price_over"]:
            rental["note"] = (rental["note"] + " " if rental["note"] else "") + \
                f"The contract costs ${real_price:.3f} per hour, more than the ${shown:.3f} shown."
        with self.lock:
            data = self._load()
            data["intents"] = [i for i in data["intents"] if i.get("label") != intent["label"]]
            data["rentals"] = [r for r in data["rentals"] if r.get("id") != contract_id] + [rental]
            self._save(data)
        return dict(rental)

    def settle(self) -> list[dict]:
        """One step of settling the rents whose answer was lost: an intent left by a crash (state `sending` that
        this process is not sending) becomes `checking`; a `checking` one whose time has come searches every page
        of the account for its label. Found -> adopted. Not found after the 13 checks over 10 minutes ->
        `unresolved`. Returns what changed: [{"label", "state": "found"|"unresolved", "rental": dict|None}].
        A search that fails counts as no check."""
        changed = []
        in_flight = self._rent_in_flight()
        for intent in self.intents():
            label = intent["label"]
            if intent["state"] == "sending" and not in_flight and label not in self._sending:
                self._set_intent(label, state="checking", since=self.now(), done=0)
                intent = dict(intent, state="checking", since=self.now(), done=0)
            if intent["state"] != "checking":
                continue
            done = int(intent.get("done") or 0)
            if done >= len(CHECK_OFFSETS) or self.now() < (intent.get("since") or 0) + CHECK_OFFSETS[done]:
                continue
            try:
                hit = self._find_label(label)
            except MarketError:
                continue
            if hit:
                changed.append({"label": label, "state": "found", "rental": self._adopt(intent, hit["id"], hit)})
                continue
            done += 1
            if done >= len(CHECK_OFFSETS):
                self._set_intent(label, state="unresolved", done=done, message=(
                    f"No instance labelled {label} was found in 10 minutes, but one may still exist. Look on "
                    "vast.ai's site, then check again or dismiss this."))
                changed.append({"label": label, "state": "unresolved", "rental": None})
            else:
                self._set_intent(label, done=done)
        return changed

    def _rent_in_flight(self) -> bool:
        """True while some process (this one included) is inside rent(): its rent.lock cannot be taken."""
        held = _try_lock(self.home / "rent.lock")
        if held is None:
            return True
        held.close()
        return False

    def _find_label(self, label: str) -> dict | None:
        for raw in self.api.instances_raw():
            if isinstance(raw, dict) and raw.get("label") == label:
                return parse_instance(raw)
        return None

    def check_intent(self, label: str) -> dict:
        """Check again now (the Check again button). {"found": bool, "rental": dict|None}; raises when the account
        cannot be read."""
        intent = next((i for i in self.intents() if i["label"] == label), None)
        if intent is None:
            return {"found": False, "rental": None}
        if intent["state"] == "sending" and (label in self._sending or self._rent_in_flight()):
            raise MarketError("The rent request is still being sent.", "busy", True)
        try:
            hit = self._find_label(label)
        except MarketError as e:
            raise MarketError(f"Could not read your vast.ai instances. {e.message}", e.kind, False) from None
        if hit:
            return {"found": True, "rental": self._adopt(intent, hit["id"], hit)}
        if intent["state"] == "unresolved":
            self._set_intent(label, message=f"Still no instance labelled {label}. Look on vast.ai's site, then "
                                            "dismiss this if there is none.")
        return {"found": False, "rental": None}

    def dismiss_intent(self, label: str) -> None:
        """Drop an intent: only once its checks are over (`unresolved`), or it is older than ten minutes and has
        been checked at least once."""
        intent = next((i for i in self.intents() if i["label"] == label), None)
        if intent is None:
            return
        old = self.now() - intent["created"] >= CHECK_OFFSETS[-1] and int(intent.get("done") or 0) > 0
        if intent["state"] != "unresolved" and not old:
            raise MarketError("The app is still checking for this rent. It can be dismissed after the checks end.",
                              "busy", True)
        self._drop_intent(label)

    # -- start, stop, destroy, bid

    def _instance(self, instance_id: int) -> dict:
        iid = _pos_int(instance_id)
        if iid is None:
            raise MarketError("That instance id is not valid.", "bad", True)
        inst = parse_instance(self.api.show_instance(iid))
        if inst is None:
            raise MarketError(f"Instance {iid} is not on your vast.ai account (any more).", "gone", True)
        return inst

    def start(self, instance_id: int) -> dict:
        """Start a stopped instance (it bills again: only from a click showing the price). {"id", "started"}."""
        inst = self._instance(instance_id)
        if inst["actual_status"] == "running":
            return {"id": inst["id"], "started": False, "note": "It is already running."}
        self.api.put_instance(inst["id"], {"state": "running"})
        return {"id": inst["id"], "started": True}

    def stop(self, instance_id: int) -> dict:
        """Re-read the instance, then stop that id only. Returns after the request; follow it with watch_stop().
        {"id", "ok", "requested_at"}."""
        inst = self._instance(instance_id)
        self.api.put_instance(inst["id"], {"state": "stopped"})
        return {"id": inst["id"], "ok": True, "requested_at": self.now()}

    def stop_state(self, instance_id: int) -> dict:
        """One look at a stop: {"stopped": bool, "status", "known": bool}. Stopped only when vast.ai reports a
        known resting status (exited, stopped, offline, inactive) and the intended status is not running, or the
        instance is gone. A look that failed, or an answer with no actual status, is unknown (not stopped)."""
        try:
            inst = parse_instance(self.api.show_instance(int(instance_id)))
        except MarketError:
            return {"stopped": False, "status": "unknown", "known": False}
        if inst is None:
            return {"stopped": True, "status": "gone", "known": True}
        actual, intended = inst["actual_status"], inst["intended_status"]
        if not actual:
            return {"stopped": False, "status": intended or "unknown", "known": False}
        return {"stopped": actual in STOPPED_STATUSES and intended != "running", "status": actual, "known": True}

    def watch_stop(self, instance_id: int, every: float = STOP_EVERY, limit: float = STOP_LIMIT) -> dict:
        """Follow a stop until vast.ai no longer reports the instance running (a look every `every` seconds, for
        `limit`). {"stopped": True, "status", "waited"} or {"stopped": False, "alert": text, ...}: still running
        after ten minutes is a loud alert, because it still bills. A look that fails counts as not stopped."""
        start = self.now()
        while True:
            self.sleep(every)
            state = self.stop_state(instance_id)
            status = state["status"]
            if state["stopped"]:
                return {"stopped": True, "status": status, "waited": self.now() - start}
            if self.now() - start >= limit:
                return {"stopped": False, "status": status, "waited": self.now() - start,
                        "alert": f"Instance {instance_id} was told to stop but vast.ai still reports it "
                                 f"{'running' if status == 'running' else 'as ' + status} after "
                                 f"{int(limit // 60)} minutes. It is still billing: stop it on vast.ai now."}

    def stop_confirmed(self, instance_id: int, every: float = STOP_EVERY, limit: float = STOP_LIMIT) -> dict:
        """stop() then watch_stop(): {"id", "ok", "requested_at", "stopped", "status", "waited", "alert"?}."""
        done = self.stop(instance_id)
        return {**done, **self.watch_stop(done["id"], every, limit)}

    def destroy(self, instance_id: int, require_rental: bool = False) -> dict:
        """Destroy an instance and its disk for good. `require_rental` (the after-run action): only an instance
        in rentals.json whose label is still the recorded one. A click on Destroy with its confirmation passes
        False and may destroy any instance of the account. {"id", "ok"}."""
        inst = self._instance(instance_id)
        rec = next((r for r in self.rentals() if r["id"] == inst["id"] and not r.get("destroyed")), None)
        if require_rental and (rec is None or rec["label"] != inst["label"]):
            raise MarketError(f"Instance {inst['id']} was not rented by this app (or its label changed), so it is "
                              "not destroyed.", "refused", True)
        try:
            self.api.delete_instance(inst["id"])
        except MarketError as e:
            if e.definite:
                raise
            try:  # no clear answer: it counts as destroyed only when the instance is gone
                gone = self.api.show_instance(inst["id"]) is None
            except MarketError:
                gone = False
            if not gone:
                raise
        if rec is not None:
            with self.lock:
                data = self._load()
                for r in data["rentals"]:
                    if r["id"] == inst["id"]:
                        r["destroyed"] = self.now()
                self._save(data)
        return {"id": inst["id"], "ok": True}

    def change_bid(self, instance_id: int, price: float) -> dict:
        """Change an interruptible instance's bid (per machine, the GPU part); below its min_bid is refused."""
        inst = self._instance(instance_id)
        p = num(price)
        if p is None or p <= 0:
            raise MarketError("The bid is not valid.", "bid", True)
        if inst["min_bid"] is not None and p < inst["min_bid"]:
            raise MarketError(f"The minimum bid is ${inst['min_bid']:.3f} per hour.", "bid", True,
                              min_bid=inst["min_bid"])
        self.api.put_instance(inst["id"], {"client_id": "me", "price": p},
                              path=f"/api/v0/instances/bid_price/{inst['id']}/")
        return {"id": inst["id"], "price": p}

    # -- preferences

    def prefs(self) -> dict:
        """{"filters", "template", "after", "idle_minutes", "bucket"}: what the Rent GPUs tab and the send dialog
        remember (market.json). Always complete and valid."""
        return _prefs_out(_load_json(_market_file(self.home), {}))

    def set_prefs(self, **change) -> dict:
        """Merge changes (filters, template, after, idle_minutes, bucket); unknown names are dropped, bad values are
        refused with MarketError(kind `bad`)."""
        with self.prefs_lock:
            data = _load_json(_market_file(self.home), {}, strict=True)
            cur = _prefs_out(data)
            for key, value in change.items():
                if key == "filters":
                    cur["filters"] = clean_filters(value)
                elif key == "template":
                    cur["template"] = _text(value, 80) or None
                elif key == "after":
                    if value not in AFTER_RUN:
                        raise MarketError("After the run: leave, stop or destroy.", "bad", True)
                    cur["after"] = value
                elif key == "idle_minutes":
                    n = num(value)
                    if n is None:
                        raise MarketError("Idle minutes must be a number.", "bad", True)
                    cur["idle_minutes"] = int(min(1440, max(10, n)))
                elif key == "bucket":
                    cur["bucket"] = _text(value, 200)
            data["prefs"] = cur
            _save_json(_market_file(self.home), data)
            return cur


def _prefs_out(data: dict) -> dict:
    p = data.get("prefs") if isinstance(data.get("prefs"), dict) else {}
    idle = num(p.get("idle_minutes"))
    return {"filters": clean_filters(p.get("filters")), "template": _text(p.get("template"), 80) or None,
            "after": p.get("after") if p.get("after") in AFTER_RUN else "stop",
            "idle_minutes": int(min(1440, max(10, idle))) if idle is not None else 60,
            "bucket": _text(p.get("bucket"), 200)}


# ---------------------------------------------------------------- for rulz_remote.list_instances


def remote_instances(api: Api | None = None) -> list[dict]:
    """The REST list in the shape `vastai show instances --raw` has: raw rows, with `extra_env` as a dict (the CLI's
    list of pairs read into one). In memory only: the environment may hold secrets, and the callers only use it to
    recognise a RulZ Trainer instance."""
    out = []
    for row in (api or Api()).instances_raw():
        row = dict(row)
        env = row.get("extra_env")
        if not isinstance(env, dict):
            row["extra_env"] = _env_names_and_values(env)
        out.append(row)
    return out


# ---------------------------------------------------------------- command line (reads only)


def _cli(argv=None) -> int:
    ap = argparse.ArgumentParser(description="vast.ai market, read only")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search")
    s.add_argument("--set", action="append", default=[], metavar="KEY=JSON", help="a filter, e.g. gpu_ram=[24,null]")
    s.add_argument("--limit", type=int, default=20)
    for name in ("account", "templates", "instances", "rentals"):
        sub.add_parser(name)
    args = ap.parse_args(argv)
    market = Market()
    try:
        if args.cmd == "search":
            f = market.prefs()["filters"]
            for item in args.set:
                key, _, raw = item.partition("=")
                try:
                    f[key] = json.loads(raw)
                except ValueError:
                    f[key] = raw
            out = [o.to_dict() for o in market.search(f)[:args.limit]]
        elif args.cmd == "account":
            out = market.account()
        elif args.cmd == "templates":
            out = market.templates()
        elif args.cmd == "instances":
            out = market.instances()
        else:
            out = {"rentals": market.rentals(), "intents": market.intents()}
    except MarketError as e:
        print(json.dumps({"error": e.message, "kind": e.kind}))
        return 1
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
