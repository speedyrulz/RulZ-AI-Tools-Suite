"""NiceGUI parts of renting a vast.ai GPU and sending a run there (docs/vast-send-contract.md, section 8).

`AccountWatch` (credit and burn, polled in the background), `MarketView` (the filters, the offers, the rent dialog),
`SendDialog` (what goes up and what happens after the run), `SendView` and the sends card (the progress of a send).
Everything here is drawn from `scripts/vast/rulz_market.py` and `rulz_send.py`; this module never imports webui.py:
the page's helpers (card, hint, eta_text, ...) come in as a `kit` object, and the services (Market, SendManager) from
`SERVICES`, which a test or a preview replaces with fakes.

Nothing here talks to vast.ai on the page's thread: every call that reaches the network, ssh or the disk goes
through `run.io_bound`. A failure is shown as a sentence, never a traceback. Only the red button of the rent dialog
and the send dialog rent or start anything; a search, Enter in a field and a restart of the page never do.
"""
from __future__ import annotations

import asyncio
import functools
import importlib
import sys
import threading
import time
from pathlib import Path

from nicegui import background_tasks, run, ui

VAST_DIR = Path(__file__).resolve().parents[1] / "scripts" / "vast"
MANAGE_KEYS_URL = "https://cloud.vast.ai/manage-keys/"
CREATE_URL = "https://cloud.vast.ai/create/"
INSTANCES_URL = "https://cloud.vast.ai/instances/"
SET_KEY_COMMAND = "vastai set api-key"
TAB = "Rent GPUs"
LOCAL_ONLY = "Only from a browser on this computer."  # what bills is done only from a page opened here
RUN_ON_KEY = "rulz.runon"
WARN_HOURS, CRITICAL_HOURS = 6.0, 1.0  # the credit chip turns amber under 6 h, red under 1 h
PAGE_SIZE = 50
MODELS_GB = 60  # what the models take on a disk when the family's size is not known
SPARE_GB = 20
RELIABILITY_WARN = 0.95
SHORT_DAYS = 3.0
ASSUMED_UPLOAD_MBPS = 100.0  # this PC's upload speed is not known: the estimate says what it assumes
STEP_ORDER = ("rent", "boot", "reach", "setup", "update", "files", "datasets", "check", "queue", "run", "download",
              "after")
BEFORE_QUEUE = STEP_ORDER[:STEP_ORDER.index("queue")]
ENDED_STATES = ("done", "failed", "cancelled")
HOW_LABELS = {"upload": "Upload to the instance", "bucket": "Via a Hugging Face bucket",
              "standard": "Use the standard file"}


# ---------------------------------------------------------------- the services (replaced by fakes in tests)


class Services:
    """Market and SendManager, built once on first use. A missing module (or one that fails to import) leaves its
    part off and says why in `errors`; nothing here ever stops the WebUI from starting."""

    def __init__(self):
        self._lock = threading.RLock()
        self.market = None
        self.manager = None
        self.key_override = None  # a test or a preview: True / False instead of looking for the key file
        self.fleet_getter = None  # webui's fleet(): the SendManager works with the same Fleet the page shows
        self.errors: dict[str, str] = {}

    def install(self, market=None, manager=None, has_key=None, fleet=None) -> None:
        with self._lock:
            if market is not None:
                self.market = market
            if manager is not None:
                self.manager = manager
            if has_key is not None:
                self.key_override = has_key
            if fleet is not None:
                self.fleet_getter = fleet

    def module(self, name: str):
        """rulz_market / rulz_send / rulz_files / rulz_plan, or None (the reason is in `errors`)."""
        with self._lock:
            if str(VAST_DIR) not in sys.path:
                sys.path.insert(0, str(VAST_DIR))
            try:
                mod = importlib.import_module(name)
            except Exception as e:  # noqa: BLE001  (missing, or its own import broke)
                self.errors[name] = f"{name} could not be loaded ({type(e).__name__})."
                return None
            self.errors.pop(name, None)
            return mod

    def get_market(self):
        with self._lock:
            if self.market is None:
                mod = self.module("rulz_market")
                if mod is not None:
                    try:
                        self.market = mod.Market()
                    except Exception as e:  # noqa: BLE001
                        self.errors["rulz_market"] = f"The vast.ai client could not start ({type(e).__name__})."
            return self.market

    def get_manager(self):
        with self._lock:
            if self.manager is None:
                mod = self.module("rulz_send")
                market = self.get_market()
                if mod is not None and market is not None:
                    try:
                        fleet = self.fleet_getter() if self.fleet_getter else None
                        self.manager = mod.SendManager(fleet=fleet, market=market)
                    except Exception as e:  # noqa: BLE001
                        self.errors["rulz_send"] = f"The send manager could not start ({type(e).__name__})."
            return self.manager

    def has_key(self) -> bool:
        if self.key_override is not None:
            return bool(self.key_override)
        mod = self.module("rulz_market")
        try:
            return bool(mod and mod.api_key())
        except Exception:  # noqa: BLE001
            return False

    def available(self) -> bool:
        return self.get_market() is not None

    def why_not(self) -> str:
        self.get_market()
        return self.errors.get("rulz_market") or "Renting is not available in this install."


SERVICES = Services()


def market_module():
    return SERVICES.module("rulz_market")


def notify_page(text: str, **kw) -> None:
    """A notification on the page, when the page is still there (a dialog that was closed, or a browser that went
    away, must not make the finishing task raise)."""
    try:
        ui.notify(text, **kw)
    except Exception:  # noqa: BLE001
        pass


def alive(*elements) -> bool:
    """False once one of the elements is gone (its dialog deleted, its page closed): a task that awaited something
    checks this before it touches the screen again."""
    for el in elements:
        if el is None:
            continue
        if getattr(el, "is_deleted", False):
            return False
    return True


def accepts(fn, name: str) -> bool:
    """Whether a callable takes this keyword (the manager's create() learns `shown`)."""
    import inspect

    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return name in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def instance_changed(shown: dict, fresh) -> str:
    """What changed in an instance since it was shown (a sentence), or "" when nothing that matters did: a start is
    asked again when its status or its price moved."""
    if not fresh:
        return "That instance is not on your account any more."
    if (fresh.get("status") or "") != (shown.get("status") or ""):
        return f"Instance {shown.get('id')} is now {fresh.get('status') or 'unknown'}, not {shown.get('status')}."
    a, b = num(shown.get("dph_total")), num(fresh.get("dph_total"))
    if a is not None and b is not None and abs(a - b) > 0.0005:
        return f"The price changed from {per_hour(a)} to {per_hour(b)}."
    return ""


def sentence(e: BaseException, fallback: str = "That did not work.") -> str:
    """An error as one sentence for the screen: Market's and Send's own messages, else the fallback (never a
    traceback, never an exception's repr)."""
    message = getattr(e, "message", None)
    if isinstance(message, str) and message.strip():
        return message.strip()
    text = str(e).strip()
    return text if isinstance(e, (OSError, ValueError)) and text and len(text) < 240 else fallback


# ---------------------------------------------------------------- formatting


def num(v):
    """A finite number from a number or numeric string, else None (booleans are not numbers)."""
    if isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f and f not in (float("inf"), float("-inf")) else None


def price(x) -> str:
    """$0.53, $0.534, $11.44: two decimals, a third only when the price has one."""
    n = num(x)
    if n is None:
        return "-"
    text = f"{n:.3f}"
    return "$" + (text[:-1] if text.endswith("0") else text)


def per_hour(x) -> str:
    return f"{price(x)}/hour"


def hours_text(hours) -> str:
    """14 h, 40 min, 3 d: how long the credit lasts."""
    h = num(hours)
    if h is None:
        return ""
    if h < 1:
        return f"{max(1, round(h * 60))} min"
    if h < 100:
        return f"{h:.0f} h"
    return f"{h / 24:.0f} d"


def when_text(epoch, now=None) -> str:
    """A local date and time: "Wed 25 Nov, 03:12"."""
    e = num(epoch)
    return "" if e is None else time.strftime("%a %d %b, %H:%M", time.localtime(e))


def size_text(n) -> str:
    v = num(n)
    if v is None:
        return "-"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(v) < 1024 or unit == "TB":
            return f"{v:.0f} B" if unit == "B" else f"{v:.1f} {unit}"
        v /= 1024
    return "-"


def duration_text(seconds) -> str:
    s = num(seconds)
    if s is None or s <= 0:
        return "-"
    if s < 90:
        return "under 2 min"
    m = round(s / 60)
    return f"{m} min" if m < 60 else f"{m // 60} h {m % 60:02d} min"


def lasts(credit, per_hour_cost):
    """Hours the credit pays for at this hourly burn; None when nothing bills or the credit is unknown."""
    c, p = num(credit), num(per_hour_cost)
    if c is None or p is None or p <= 0:
        return None
    return max(0.0, c) / p


def level_of(hours) -> str:
    h = num(hours)
    if h is None:
        return "ok"
    return "critical" if h < CRITICAL_HOURS else "warning" if h < WARN_HOURS else "ok"


def chip_view(snapshot: dict) -> dict:
    """The header chip from AccountWatch's snapshot: {"billed", "text", "level", "icon", "tip": [lines]}.
    `billed` is False when nothing costs anything now (the chip then shows only on the Rent GPUs tab, with the credit
    alone). Level: amber under 6 h, red under 1 h; an alert (a stop that did not happen) is red whatever the credit."""
    acct = snapshot.get("account")
    alerts = snapshot.get("alerts") or []
    if not acct:
        err = snapshot.get("error") or ""
        return {"billed": bool(alerts), "text": "vast.ai: not reachable" if err else "vast.ai", "icon": "cloud_off",
                "level": "critical" if alerts else "ok", "tip": [err] if err else [], "credit": None, "hours": None}
    burn = acct.get("burn") or {}
    cost = num(burn.get("per_hour")) or 0.0
    credit = acct.get("credit")
    hours = acct.get("hours_left")
    billed = cost > 0
    text = price(credit) if credit is not None else "credit unknown"
    if billed and hours is not None:
        text += f" · {hours_text(hours)} left"
    level = level_of(hours) if billed else "ok"
    if alerts:
        level = "critical"
    tip = []
    run_n, rest_n = int(burn.get("running") or 0), int(burn.get("stopped") or 0)
    if run_n:
        tip.append(f"{per_hour(burn.get('running_per_hour'))} for {run_n} running instance{'s' if run_n != 1 else ''}"
                   f" ({int(burn.get('gpus') or 0)} GPU{'s' if burn.get('gpus') != 1 else ''})")
    if rest_n:
        tip.append(f"{per_hour(burn.get('stopped_per_hour'))} for the disk{'s' if rest_n != 1 else ''} of {rest_n} "
                   f"stopped instance{'s' if rest_n != 1 else ''}")
    if billed and hours is not None and acct.get("empty_at"):
        tip.append(f"The credit runs out about {when_text(acct['empty_at'])}.")
    if not billed:
        tip.append("Nothing is billed now.")
    tip.append("Traffic (upload and download) is not included.")
    for a in alerts:
        tip.insert(0, a.get("text", ""))
    icon = "error" if alerts else "warning" if level in ("warning", "critical") else "account_balance_wallet"
    return {"billed": billed or bool(alerts), "text": text, "level": level, "icon": icon, "tip": tip,
            "credit": credit, "hours": hours}


def credit_lines(snapshot: dict) -> list[str]:
    """The credit strip of the Rent GPUs tab: sentences."""
    acct = snapshot.get("account")
    if not acct:
        return [snapshot.get("error") or "The account has not been read yet."]
    burn = acct.get("burn") or {}
    cost = num(burn.get("per_hour")) or 0.0
    lines = []
    if cost > 0:
        parts = []
        if burn.get("running"):
            parts.append(f"{per_hour(burn.get('running_per_hour'))} for {burn['running']} running "
                         f"instance{'s' if burn['running'] != 1 else ''}")
        if burn.get("stopped"):
            parts.append(f"{per_hour(burn.get('stopped_per_hour'))} for the disks of {burn['stopped']} stopped "
                         f"instance{'s' if burn['stopped'] != 1 else ''}")
        lines.append("Billing " + per_hour(cost) + " now: " + " and ".join(parts) + ".")
        if acct.get("hours_left") is not None:
            lines.append(f"The credit lasts about {hours_text(acct['hours_left'])}, until "
                         f"{when_text(acct.get('empty_at'))}. Traffic is not included.")
    else:
        lines.append("Nothing is billed now: no instance is running and none keeps a disk.")
    return lines


# ---------------------------------------------------------------- filters and offers


def is_set(f, value) -> bool:
    """Whether a filter's value limits the search (the count the groups show). Type and sort always have a value, so
    they never count; "any" GPUs and Planet Earth limit nothing."""
    key = getattr(f, "key", "")
    if key in ("type", "sort"):
        return False
    kind = getattr(f, "kind", "")
    if kind == "range":
        return isinstance(value, (list, tuple)) and any(v is not None for v in value)
    if kind == "min":
        return value is not None and value != ""
    if kind == "switch":
        return bool(value)
    if kind == "choice":
        return value not in (None, "", "any", "earth")
    if kind in ("multi",):
        return bool(value)
    return bool(str(value or "").strip())


def filter_counts(filters: dict, spec=None) -> dict[str, int]:
    """{group: how many of its filters are set}, every group present."""
    spec = spec if spec is not None else list(market_module().FILTERS)
    out: dict[str, int] = {}
    for f in spec:
        out.setdefault(f.group, 0)
        if f.key in filters and is_set(f, filters[f.key]):
            out[f.group] += 1
    return out


def group_label(group: str, count: int) -> str:
    return group + (f" · {count} set" if count else "")


def _get(o, key, default=None):
    if isinstance(o, dict):
        return o.get(key, default)
    return getattr(o, key, default)


def can_rent(o) -> tuple[bool, str]:
    """Whether a row has a Rent button, else why not: reserved rows only link to vast.ai."""
    if _get(o, "type") == "reserved":
        return False, "Reserved: rent it on vast.ai"
    if _get(o, "rentable") is False:
        return False, "Not available now"
    return True, ""


def _n(v, digits=0, unit=""):
    n = num(v)
    return "-" if n is None else f"{n:,.{digits}f}{unit}"


def offer_row(o) -> dict:
    """One offer as the results table shows it (strings, so the table needs no formatting of its own)."""
    gpus = int(num(_get(o, "num_gpus")) or 1)
    ok, why = can_rent(o)
    up, down = num(_get(o, "inet_up_per_gb")), num(_get(o, "inet_down_per_gb"))
    cores = num(_get(o, "cpu_cores"))
    cpu = (_get(o, "cpu_name") or "").strip()
    rel = num(_get(o, "reliability"))
    dur = num(_get(o, "duration_days"))
    where = "Datacenter" if _get(o, "datacenter") else "Verified" if _get(o, "verified") else "Unverified"
    loc = (_get(o, "geolocation") or "").strip()
    return {
        "id": _get(o, "id"), "gpu": f"{gpus}x {_get(o, 'gpu_name') or 'GPU'}",
        "vram": _n((num(_get(o, "gpu_total_ram_mb")) or 0) / 1024, 0, " GB"), "tflops": _n(_get(o, "tflops"), 1),
        "dlperf": _n(_get(o, "dlperf"), 1), "dlperf_dollar": _n(_get(o, "dlperf_per_dollar"), 1),
        "price": price(_get(o, "dph_total")) if _get(o, "dph_total") is not None else "-",
        "up": "-" if up is None else f"${up * 1000:,.0f}/TB", "down": "-" if down is None else f"${down * 1000:,.0f}/TB",
        "cpu": ((cpu + " ") if cpu else "") + (f"{cores:.1f}".rstrip("0").rstrip(".") + " cores" if cores else ""),
        "ram": _n((num(_get(o, "cpu_ram_mb")) or 0) / 1024, 0, " GB") if num(_get(o, "cpu_ram_mb")) else "-",
        "disk_bw": _n(_get(o, "disk_bw"), 0, " MB/s"), "disk": _n(_get(o, "disk_space"), 0, " GB"),
        "net": f"{_n(_get(o, 'inet_up'), 0)} / {_n(_get(o, 'inet_down'), 0)} Mbps",
        "ports": _n(_get(o, "ports")), "cuda": _n(_get(o, "cuda"), 1),
        "reliability": "-" if rel is None else f"{rel * 100:.1f}%",
        "duration": "-" if dur is None else f"{dur:.0f} d" if dur >= 1 else f"{dur * 24:.0f} h",
        "location": loc or "-", "where": where, "where_icon": "verified" if _get(o, "verified") or
        _get(o, "datacenter") else "help_outline", "machine": _get(o, "machine_id") or "-",
        "can_rent": ok, "reason": why, "type": _get(o, "type") or "ask"}


def offer_price(o, bid=None) -> float | None:
    """What the offer costs per hour with the disk: the asking price, or bid plus the disk for an interruptible one."""
    if _get(o, "type") == "bid":
        b = num(bid)
        return None if b is None else b + (num(_get(o, "storage_dph")) or 0.0)
    return num(_get(o, "dph_total"))


def disk_warning(disk_gb, upload_bytes=0, models_gb=MODELS_GB) -> str:
    """The container is smaller than the models, what goes up and the room to work (20 GB): the run would not fit."""
    need = (num(models_gb) or MODELS_GB) + (num(upload_bytes) or 0) / 1e9 + SPARE_GB
    d = num(disk_gb)
    if d is not None and d < need:
        return (f"A {d:.0f} GB disk is small for this: the models, what you send and some room to work need about "
                f"{need:.0f} GB. Raise the container size filter and search again.")
    return ""


def rent_warnings(o, disk_gb, bid=None, upload_bytes=0, models_gb=MODELS_GB) -> list[str]:
    out = []
    w = disk_warning(disk_gb, upload_bytes, models_gb)
    if w:
        out.append(w)
    rel = num(_get(o, "reliability"))
    if rel is not None and rel < RELIABILITY_WARN:
        out.append(f"This host's reliability is {rel * 100:.1f}%: it drops out more often than most.")
    dur = num(_get(o, "duration_days"))
    if dur is not None and dur < SHORT_DAYS:
        out.append(f"This machine can be rented for {dur:.1f} day{'s' if dur != 1 else ''} at most; the host may "
                   "take it back after that.")
    if _get(o, "type") == "bid":
        minimum = num(_get(o, "min_bid"))
        out.append("Interruptible: when another renter bids more, vast.ai pauses this instance and the run stops "
                   "until you start it again. The disk and its files stay, and keep billing.")
        b = num(bid)
        if minimum is not None and b is not None and b < minimum:
            out.append(f"The bid is under the minimum of {price(minimum)}/hour.")
        elif minimum is not None and b is not None and b < minimum * 1.1:
            out.append("The bid is close to the minimum, so a small rise in the market pauses the run.")
    return out


def upload_seconds(nbytes, mbps=ASSUMED_UPLOAD_MBPS) -> float | None:
    b, m = num(nbytes), num(mbps)
    if b is None or m is None or m <= 0:
        return None
    return b * 8 / (m * 1e6)


def upload_text(nbytes, mbps=ASSUMED_UPLOAD_MBPS) -> str:
    s = upload_seconds(nbytes, mbps)
    if s is None:
        return ""
    return f"{size_text(nbytes)} to send: about {duration_text(s)} at {mbps:g} Mbit/s (your connection's real speed decides)."


def cost_lines(credit, burn_per_hour, add_per_hour, run_hours=None) -> list[tuple[str, str]]:
    """The send dialog's cost block as (label, text) rows: the instance's price, the credit, how long it lasts with
    this instance on top of what already bills, and the run's estimated cost when its time is known."""
    rows = []
    add = num(add_per_hour) or 0.0
    rows.append(("Instance", per_hour(add) if add > 0 else "already billing"))
    if num(credit) is not None:
        rows.append(("Credit", price(credit)))
        before, after = lasts(credit, burn_per_hour), lasts(credit, (num(burn_per_hour) or 0.0) + add)
        if after is not None:
            now = f" (now {hours_text(before)})" if before is not None and add > 0 else ""
            rows.append(("Credit lasts", f"about {hours_text(after)} with this instance{now}"))
    rh = num(run_hours)
    if rh is not None and rh > 0 and add > 0:
        rows.append(("Estimated run cost", f"about {price(rh * add)} for {hours_text(rh)} of GPU time"))
    return rows


# ---------------------------------------------------------------- the send dialog's logic


def file_options(entry: dict) -> list[tuple[str, str]]:
    """The choices for one local model file: both uploads, and the standard file when there is one."""
    out = [("upload", HOW_LABELS["upload"]), ("bucket", HOW_LABELS["bucket"])]
    if entry.get("standard"):
        out.append(("standard", HOW_LABELS["standard"]))
    return out


def default_choice(entry: dict) -> str | None:
    """`standard` when the instance can download the family's own file, else undecided: the user must choose."""
    return "standard" if entry.get("standard") else None


def standard_text(entry: dict) -> str:
    s = entry.get("standard")
    if not isinstance(s, dict):
        return ""
    return f"{s.get('file') or ''} from {s.get('repo') or 'the model hub'}".strip()


def _how(choice) -> str | None:
    """The route of one file's choice: {"how": ...} (SendManager's) or the bare word (the dialog's picks)."""
    return choice.get("how") if isinstance(choice, dict) else choice


def needs_bucket(choices: dict) -> bool:
    return any(_how(c) == "bucket" for c in choices.values())


def send_state(plan, choices: dict, *, bucket: str = "", bucket_ok: bool = False, target=None, after: str = "stop",
               destroy_allowed: bool = True, busy: bool = False) -> tuple[bool, list[str]]:
    """Whether the send dialog's red button is on, and what is missing (every reason, in the order they appear):
    a blocker in the plan, a model file with no choice, a bucket route without a confirmed bucket, no target."""
    why: list[str] = []
    if plan is None:
        return False, ["Looking at what the run needs from this PC."]
    for b in getattr(plan, "blockers", None) or []:
        where = b.get("path") or b.get("key") or ""
        why.append(f"{b.get('reason', 'A problem')}: {where}".strip(": "))
    for entry in getattr(plan, "files", None) or []:
        how = _how((choices or {}).get(entry["path"]))
        if how not in ("upload", "bucket", "standard") or (how == "standard" and not entry.get("standard")):
            why.append(f"Choose how to send {Path(entry['path']).name}.")
    if needs_bucket(choices or {}) and not (bucket and bucket_ok):
        why.append("Confirm the Hugging Face bucket.")
    if target is None:
        why.append("Pick an instance, or rent a new one.")
    if after == "destroy" and not destroy_allowed:
        why.append("Destroy is only for instances this app rented.")
    if busy:
        why.append("Sending.")
    return not why, why


def choices_for(plan, picks: dict, bucket: str) -> dict:
    """SendManager.create's `choices`: {path: {"how", "bucket"?}} from the dialog's picks."""
    out = {}
    for entry in getattr(plan, "files", None) or []:
        how = picks.get(entry["path"])
        if how in ("upload", "bucket", "standard"):
            out[entry["path"]] = {"how": how, **({"bucket": bucket} if how == "bucket" else {})}
    return out


def after_options(*, ours: bool) -> list[tuple[str, str]]:
    """After the run: Stop, Destroy and Leave running for an instance this app rents; for another one Destroy is not
    offered."""
    out = [("stop", "Stop it")]
    if ours:
        out.append(("destroy", "Destroy it"))
    out.append(("leave", "Leave it running"))
    return out


def after_default(prefs_after: str, *, ours: bool) -> str:
    """The saved choice (stop by default); an instance this app did not rent defaults to leaving it running."""
    if not ours:
        return "leave"
    return prefs_after if prefs_after in ("stop", "destroy", "leave") else "stop"


def after_text(after: str, disk_per_hour=None, gpu_per_hour=None) -> str:
    disk = f" and costs {per_hour(disk_per_hour)}" if num(disk_per_hour) else ""
    if after == "stop":
        return (f"Stops billing for the GPU once every result is downloaded; the disk stays{disk}. Start it again on "
                "the Rent GPUs tab.")
    if after == "destroy":
        return ("Deletes the instance and its disk for good, but only after every file of the run is downloaded here "
                "(resume states are not downloaded). When anything is missing or unclear it stops instead and says "
                "why.")
    gpu = f" {per_hour(gpu_per_hour)}" if num(gpu_per_hour) else ""
    return f"The instance keeps billing{gpu} until you stop it."


def instance_label(row: dict) -> str:
    gpus = int(num(row.get("num_gpus")) or 1)
    return f"{gpus}x {row.get('gpu_name') or 'GPU'}"


def instance_state_word(row: dict) -> tuple[str, str, str]:
    """(icon, word, level) of an account instance."""
    status = (row.get("status") or "unknown").lower()
    if status == "running":
        return "play_circle", "Running", "ok"
    if status in ("loading", "created", "scheduling", "connecting"):
        return "hourglass_top", status.capitalize(), "warn"
    if status in ("stopped", "exited", "offline"):
        return "pause_circle", "Stopped" if status != "exited" else "Exited", "muted"
    return "help_outline", status.capitalize() or "Unknown", "muted"


def _is_stopped(row: dict) -> bool:
    return (row.get("status") or "").lower() in ("stopped", "exited")


def _is_up(row: dict) -> bool:
    return (row.get("status") or "").lower() in ("running", "loading", "created", "scheduling", "connecting")


# ---------------------------------------------------------------- send step view (logic)

STEP_LOOK = {"waiting": ("schedule", "Waiting", "muted"), "going": ("autorenew", "Going", "warn"),
             "done": ("check_circle", "Done", "ok"), "failed": ("error", "Failed", "err"),
             "skipped": ("remove_circle_outline", "Skipped", "muted")}
STATE_LOOK = {"going": ("autorenew", "Sending", "warn"), "waiting": ("schedule", "Waiting", "muted"),
              "running": ("play_circle", "Running", "ok"), "done": ("check_circle", "Done", "ok"),
              "failed": ("error", "Failed", "err"), "cancelled": ("cancel", "Cancelled", "muted"),
              "interrupted": ("warning", "Interrupted", "warn")}


def countdown_text(seconds) -> str:
    s = num(seconds)
    if s is None:
        return ""
    s = max(0, int(s))
    return f"{s // 60}:{s % 60:02d}"


def send_can_cancel(send: dict) -> bool:
    """Cancel works until the run is queued; after that only through the Fleet (Stop)."""
    if send.get("state") in ENDED_STATES:
        return False
    for step in send.get("steps") or []:
        if step.get("key") in ("queue", "run", "download", "after") and step.get("state") in ("going", "done", "failed"):
            return False
    return True


def send_signature(send: dict) -> str:
    """What decides whether a send's view must be rebuilt (a new list of steps), not just repainted."""
    return ",".join(str(s.get("key")) for s in send.get("steps") or [])


def step_line(step: dict) -> str:
    """A step's second line: its own text, with the percentage when it has one."""
    text = (step.get("text") or "").strip()
    frac = num(step.get("frac"))
    if frac is not None and step.get("state") == "going" and "%" not in text:
        text = (text + " " if text else "") + f"{frac * 100:.0f}%"
    return text


# ---------------------------------------------------------------- the account, watched in the background


class AccountWatch:
    """The account's credit and instances, read once a minute in a background thread and shared by every page (a page
    only reads `snapshot()`; it never waits on vast.ai). Runs only when an API key is there. `alerts` are the loud
    ones: a stop that vast.ai still reports as running ten minutes later."""

    EVERY = 60.0

    def __init__(self, services: Services = SERVICES, now=time.time):
        self.services = services
        self.now = now
        self._lock = threading.Lock()
        self.data: dict = {"account": None, "error": "", "at": 0.0, "key": None}
        self.alerts: list[dict] = []
        self.stopping: dict[int, float] = {}
        self._force = threading.Event()
        self._thread: threading.Thread | None = None

    def poll(self) -> dict:
        """One look, now (blocking: for the loop and for run.io_bound). Never raises."""
        market = self.services.get_market()
        if market is None or not self.services.has_key():
            with self._lock:
                self.data = {"account": None, "error": "", "at": self.now(), "key": False}
            return self.snapshot()
        try:
            acct, error = market.account(), ""
        except Exception as e:  # noqa: BLE001  (a sentence, not a traceback)
            acct, error = None, sentence(e, "vast.ai did not answer.")
        with self._lock:
            old = self.data.get("account")
            self.data = {"account": acct or old, "error": error, "at": self.now(), "key": True}
        return self.snapshot()

    def snapshot(self) -> dict:
        with self._lock:
            return {**self.data, "alerts": [dict(a) for a in self.alerts], "stopping": dict(self.stopping)}

    def request(self) -> None:
        """Look again at the next beat (after a click that changed the account)."""
        self._force.set()

    def start(self, enabled: bool = True) -> None:
        if not enabled or (self._thread is not None and self._thread.is_alive()):
            return
        self._thread = threading.Thread(target=self._loop, name="rulz-vast-account", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while True:
            try:
                due = self.now() - self.data.get("at", 0.0) >= self.EVERY
                if self._force.is_set() or (due and self.services.has_key()):
                    self._force.clear()
                    self.poll()
                elif due:  # no key: look again for one every minute, without calling vast.ai
                    self.poll()
            except Exception as e:  # noqa: BLE001  (the loop outlives one bad look)
                print(f"[vast] {e!r}", flush=True)
            self._force.wait(5.0)

    # -- stops that must be seen through

    def follow_stop(self, market, instance_id: int) -> dict:
        """After a stop was asked: wait until vast.ai no longer reports the instance running (Market.watch_stop,
        blocking: run it with run.io_bound). Still running after the limit becomes an alert."""
        with self._lock:
            self.stopping[int(instance_id)] = self.now()
        try:
            result = market.watch_stop(int(instance_id))
        except Exception as e:  # noqa: BLE001
            result = {"stopped": False, "alert": f"Could not check that instance {instance_id} stopped "
                                                 f"({sentence(e, 'vast.ai did not answer')}). Look on vast.ai: a "
                                                 "running instance still bills."}
        with self._lock:
            self.stopping.pop(int(instance_id), None)
            if not result.get("stopped"):
                self.alerts = [a for a in self.alerts if a.get("id") != int(instance_id)]
                self.alerts.append({"id": int(instance_id), "text": result.get("alert") or f"Instance {instance_id} "
                                                                                           "may still be running.",
                                    "at": self.now()})
        self.request()
        return result

    def dismiss_alert(self, instance_id) -> None:
        with self._lock:
            self.alerts = [a for a in self.alerts if a.get("id") != instance_id]


WATCH = AccountWatch()


async def confirm(title: str, lines: list[str], yes: str, icon: str | None = None, danger: bool = False) -> bool:
    """A dialog that says what happens (price, loss) and asks. Cancel is the quiet button, the answer is the one
    button that is red."""
    with ui.dialog() as dlg, ui.card().classes("w-[34rem] max-w-full gap-3"):
        ui.label(title).classes("text-base font-medium")
        for line in lines:
            ui.label(line).classes("rz-hint")
        with ui.row().classes("w-full justify-end gap-2"):
            ui.button("Cancel", on_click=lambda: dlg.submit(False), color=None).props("flat no-caps")
            if danger:
                ui.button(yes, icon=icon, on_click=lambda: dlg.submit(True), color=None).props("flat no-caps") \
                    .classes("rz-danger")
            else:
                ui.button(yes, icon=icon, on_click=lambda: dlg.submit(True)).props("unelevated no-caps")
    go = await dlg
    dlg.delete()
    return bool(go)


def status_line(icon: str, word: str, text: str = "", level: str = "muted"):
    """A status is always a word and an icon (colour alone says nothing): icon, word, then the text."""
    with ui.row().classes("items-center gap-1 no-wrap rz-vast-status"):
        ui.icon(icon).classes(f"rz-rs-icon {level}")
        ui.label(word).classes(f"rz-rs-word {level}")
        if text:
            ui.label(text).classes("rz-hint")


# ---------------------------------------------------------------- one send's steps


class SendView:
    """A send as rows: each step with its icon, word, text and bar, repainted in place by `update(send)` (a poll every
    2 s must not rebuild buttons under the pointer). Buttons: Cancel (before the run is queued), Retry (failed),
    Keep running with the countdown, Resume queue, Dismiss (ended). `extra` adds the caller's own (Close)."""

    def __init__(self, vast: "VastUI", parent, send: dict, extra=None, dismissible: bool = False, on_gone=None):
        self.vast = vast
        self.id = send.get("id")
        self.dismissible = dismissible
        self.on_gone = on_gone
        self.sig = send_signature(send)
        self.warn_sig = None
        self.rows: dict = {}
        self.send = send
        with parent:
            self.root = ui.column().classes("w-full gap-2 rz-vast-send")
        with self.root:
            with ui.row().classes("w-full items-center gap-2 no-wrap"):
                self.state_icon = ui.icon("schedule").classes("rz-rs-icon")
                self.title = ui.label("").classes("text-base font-medium")
                self.state_word = ui.label("").classes("rz-rs-word")
                ui.space()
                self.remaining = ui.label("").classes("rz-hint")
            self.meta = ui.label("").classes("rz-hint")
            self.steps_box = ui.element("div").classes("rz-vast-steps w-full")
            self.warn_box = ui.column().classes("w-full gap-1")
            self.error = ui.label("").classes("rz-banner error")
            self.result = ui.label("").classes("rz-banner info")
            self.countdown = ui.label("").classes("rz-banner warning")
            with ui.row().classes("w-full items-center gap-2") as self.buttons:
                self.keep_btn = ui.button("Keep running", icon="play_circle", on_click=self.keep, color=None) \
                    .props("outline no-caps dense")
                self.resume_btn = ui.button("Resume the queue", icon="play_arrow", on_click=self.resume, color=None) \
                    .props("outline no-caps dense")
                self.retry_btn = ui.button("Retry", icon="replay", on_click=self.retry, color=None) \
                    .props("outline no-caps dense")
                self.cancel_btn = ui.button("Cancel the send", icon="close", on_click=self.cancel, color=None) \
                    .props("flat no-caps dense")
                self.dismiss_btn = ui.button("Dismiss", icon="done", on_click=self.dismiss, color=None) \
                    .props("flat no-caps dense")
                if extra is not None:
                    extra()
        self.build_steps(send)
        self.update(send)

    def build_steps(self, send: dict) -> None:
        self.steps_box.clear()
        self.rows = {}
        with self.steps_box:
            for step in send.get("steps") or []:
                key = step.get("key")
                with ui.element("div").classes("rz-vast-step"):
                    icon = ui.icon("schedule").classes("rz-rs-icon")
                    word = ui.label("").classes("rz-rs-word")
                    with ui.column().classes("gap-0 min-w-0"):
                        label = ui.label(step.get("label") or str(key)).classes("rz-setting-title")
                        text = ui.label("").classes("rz-hint")
                        bar = ui.linear_progress(value=0, show_value=False, size="5px").props("rounded") \
                            .classes("w-full")
                self.rows[key] = {"icon": icon, "word": word, "label": label, "text": text, "bar": bar}

    def update(self, send: dict) -> None:
        self.send = send
        if send_signature(send) != self.sig:
            self.sig = send_signature(send)
            self.build_steps(send)
        state = send.get("state") or "waiting"
        icon, word, level = STATE_LOOK.get(state, STATE_LOOK["waiting"])
        self.state_icon.props(f"name={icon}")
        self.state_icon.classes(replace=f"rz-rs-icon {level}")
        self.state_word.text = word
        self.state_word.classes(replace=f"rz-rs-word {level}")
        self.title.text = send.get("job") or "Run"
        bits = []
        if send.get("instance"):
            bits.append(f"instance {send['instance']}")
        if num(send.get("per_hour")):
            bits.append(per_hour(send["per_hour"]))
        bits.append({"stop": "stops after the run", "destroy": "is destroyed after the run",
                     "leave": "stays running after the run"}.get(send.get("after"), ""))
        self.meta.text = " · ".join(b for b in bits if b)
        for step in send.get("steps") or []:
            row = self.rows.get(step.get("key"))
            if row is None:
                continue
            s_icon, s_word, s_level = STEP_LOOK.get(step.get("state"), STEP_LOOK["waiting"])
            row["icon"].props(f"name={s_icon}")
            row["icon"].classes(replace=f"rz-rs-icon {s_level}")
            row["word"].text = s_word
            row["word"].classes(replace=f"rz-rs-word {s_level}")
            row["label"].text = step.get("label") or str(step.get("key"))
            row["text"].text = step_line(step)
            row["text"].visible = bool(row["text"].text)
            frac = num(step.get("frac"))
            row["bar"].visible = step.get("state") == "going" and frac is not None
            row["bar"].value = max(0.0, min(1.0, frac or 0.0))
        warnings = [str(w) for w in send.get("warnings") or []]
        if tuple(warnings) != self.warn_sig:
            self.warn_sig = tuple(warnings)
            self.warn_box.clear()
            with self.warn_box:
                for w in warnings:
                    ui.label(w).classes("rz-banner warning")
        self.error.text = str(send.get("error") or "")
        self.error.visible = bool(self.error.text)
        self.result.text = str(send.get("result") or "")
        self.result.visible = bool(self.result.text)
        left = send.get("countdown")
        act = send.get("action")
        self.countdown.visible = left is not None and act in ("stop", "destroy")
        if self.countdown.visible:
            self.countdown.text = (f"The instance is {'destroyed' if act == 'destroy' else 'stopped'} in "
                                   f"{countdown_text(left)}. Keep running cancels that.")
        self.keep_btn.visible = self.countdown.visible
        self.resume_btn.visible = any("paused" in w.lower() for w in warnings) and state not in ENDED_STATES \
            and self.vast.kit.resume_queue is not None
        self.retry_btn.visible = state == "failed"
        self.cancel_btn.visible = send_can_cancel(send)
        self.dismiss_btn.visible = self.dismissible and state in ENDED_STATES
        started = num(send.get("created"))
        self.remaining.text = f"started {when_text(started)}" if started else ""

    # -- the buttons (each a call off the page's thread)

    async def _call(self, name: str, *args, ok: str = "", **kw) -> bool:
        manager = SERVICES.get_manager()
        if manager is None:
            notify_page("Sending is not available in this install.", type="warning")
            return False
        try:
            await run.io_bound(functools.partial(getattr(manager, name), self.id, *args, **kw))
        except Exception as e:  # noqa: BLE001
            notify_page(sentence(e), type="warning", multi_line=True)
            return False
        if ok:
            notify_page(ok, type="positive")
        return True

    async def retry(self) -> None:
        await self._call("retry", ok="Trying again from the step that failed.")

    async def keep(self) -> None:
        if await self._call("keep", ok="The instance stays running. It keeps billing until you stop it."):
            WATCH.request()

    async def resume(self) -> None:
        try:
            await run.io_bound(self.vast.kit.resume_queue)
        except Exception as e:  # noqa: BLE001
            notify_page(sentence(e), type="warning", multi_line=True)
            return
        notify_page("The Fleet queue is running again.", type="positive")

    async def dismiss(self) -> None:
        if await self._call("dismiss") and self.on_gone:
            self.on_gone(self.id)

    async def cancel(self) -> None:
        rented = bool(self.send.get("rented") or self.send.get("started"))
        with ui.dialog() as dlg, ui.card().classes("w-[34rem] max-w-full gap-3"):
            ui.label(f"Cancel sending {self.send.get('job') or 'this run'}?").classes("text-base font-medium")
            ui.label("What is already on the instance stays there, so a new send does not start over.") \
                .classes("rz-hint")
            stop = None
            if rented:
                stop = ui.switch("Stop the instance too (this app " + ("rented" if self.send.get("rented") else "started")
                                 + " it for this send)", value=True)
                ui.label("A running instance bills until it is stopped.").classes("rz-hint")
            with ui.row().classes("w-full justify-end gap-2"):
                ui.button("Keep sending", on_click=lambda: dlg.submit(False), color=None).props("flat no-caps")
                ui.button("Cancel the send", icon="close", on_click=lambda: dlg.submit(True)).props("unelevated no-caps")
        go = await dlg
        action = bool(stop.value) if stop is not None else False
        dlg.delete()
        if not go:
            return
        if await self._call("cancel", instance_action="stop" if action else "keep", ok="Cancelled."):
            WATCH.request()


# ---------------------------------------------------------------- the offers (filters, results)

COLUMNS = [  # (name, label, field, align, wide): `wide` ones are hidden on a phone; Rent comes first, in reach
    ("rent", "", "rent", "left", False), ("gpu", "GPU", "gpu", "left", False), ("vram", "VRAM", "vram", "right", False),
    ("tflops", "TFLOPS", "tflops", "right", True), ("dlperf", "DLPerf", "dlperf", "right", True),
    ("dlperf_dollar", "DLPerf/$/h", "dlperf_dollar", "right", False), ("price", "$/hour", "price", "right", False),
    ("up", "Upload", "up", "right", True), ("down", "Download", "down", "right", True),
    ("cpu", "CPU", "cpu", "left", True), ("ram", "RAM", "ram", "right", True),
    ("disk_bw", "Disk speed", "disk_bw", "right", True), ("disk", "Disk", "disk", "right", True),
    ("net", "Net up / down", "net", "right", True), ("ports", "Ports", "ports", "right", True),
    ("cuda", "CUDA", "cuda", "right", True), ("reliability", "Reliability", "reliability", "right", False),
    ("duration", "Max duration", "duration", "right", True), ("location", "Location", "location", "left", True),
    ("where", "Host", "where", "left", True), ("machine", "Machine", "machine", "right", True)]
RENT_SLOT = """
<q-td :props="props">
  <q-btn v-if="props.row.can_rent" dense outline no-caps icon="shopping_cart" label="Rent"
         @click="() => $parent.$emit('rent', props.row.id)" />
  <a v-else-if="props.row.type === 'reserved'" href="https://cloud.vast.ai/create/" target="_blank" rel="noopener"
     class="rz-vast-link">Open on vast.ai</a>
  <span v-else class="rz-hint">{{ props.row.reason }}</span>
</q-td>"""
WHERE_SLOT = """
<q-td :props="props">
  <q-icon :name="props.row.where_icon" size="16px" class="q-mr-xs" />{{ props.row.where }}
</q-td>"""


def with_price(offer, new_price: float, bid=None):
    """The offer as it is priced now (vast.ai said the price moved): the asking price, or for a bid the disk part."""
    import dataclasses

    if _get(offer, "type") == "bid" and num(bid) is not None:
        storage = max(0.0, new_price - float(bid))
        change = {"storage_dph": storage}
    else:
        change = {"dph_total": new_price}
    if isinstance(offer, dict):
        return {**offer, **change}
    return dataclasses.replace(offer, **change)


class MarketView:
    """The filters (every control of vast.ai's own search page, in collapsible groups) and the offers. Built lazily on
    first show. In `pick` mode (the send dialog's "Rent a new one") the Rent button of a row opens the rent dialog in
    its pick form and hands the choice to `on_pick` instead of renting."""

    def __init__(self, vast: "VastUI", pick: bool = False, on_pick=None, upload_bytes: int = 0):
        self.vast = vast
        self.pick = pick
        self.on_pick = on_pick
        self.upload_bytes = upload_bytes
        self.mk = market_module()
        self.widgets: dict = {}
        self.group_boxes: dict = {}
        self.applying = False
        self.offers: list = []
        self.shown = 0
        self.searched = False
        self.searching = False
        self.catalog_loaded = False
        self.root = None
        self.searched_with: dict | None = None  # the filters and template the shown offers were searched with

    # -- building

    def build(self, parent) -> None:
        mk = self.mk
        with parent:
            self.root = ui.element("div").classes("rz-vast-layout")
        with self.root:
            with ui.column().classes("rz-vast-filters"):
                with ui.row().classes("w-full items-center gap-2 no-wrap"):
                    ui.label("Filters").classes("rz-title")
                    ui.space()
                    ui.button("Reset filters", icon="restart_alt", on_click=self.reset, color=None) \
                        .props("flat dense no-caps")
                for group in mk.GROUPS:
                    items = [f for f in mk.FILTERS if f.group == group and f.key != "sort"]
                    if not items:
                        continue
                    exp = ui.expansion(group, value=group == "Rental, GPU and region").props("dense").classes("rz-vast-group w-full")
                    self.group_boxes[group] = exp
                    with exp:
                        with ui.column().classes("w-full gap-2 pb-2"):
                            for f in items:
                                self.control(f)
            with ui.column().classes("gap-3 min-w-0"):
                with ui.row().classes("w-full items-center gap-3"):
                    ui.button("Search", icon="search", on_click=self.search).props("unelevated no-caps")
                    self.sort = ui.select(dict(mk.SORT_OPTIONS), value="dlperf_per_dollar", label="Sort by") \
                        .props("dense outlined").classes("w-56")
                    self.sort.on_value_change(self.on_sort)
                    self.spinner = ui.spinner(size="sm")
                    self.spinner.visible = False
                    self.count = ui.label("").classes("rz-hint")
                    ui.space()
                self.message = ui.label("").classes("rz-banner warning")
                self.message.visible = False
                self.stale = ui.label("Filters changed: search again to apply them.").classes("rz-hint rz-warn")
                self.stale.visible = False
                with ui.element("div").classes("rz-vast-scroll w-full"):
                    self.table = ui.table(columns=[
                        {"name": n, "label": label, "field": field_, "align": align,
                         **({"classes": "rz-vast-wide", "headerClasses": "rz-vast-wide"} if wide else {})}
                        for n, label, field_, align, wide in COLUMNS], rows=[], row_key="id",
                        pagination={"rowsPerPage": 0}).props("dense flat square hide-pagination") \
                        .classes("rz-vast-table w-full")
                    self.table.add_slot("body-cell-rent", RENT_SLOT)
                    self.table.add_slot("body-cell-where", WHERE_SLOT)
                    self.table.on("rent", self.on_rent)
                    self.table.props('no-data-label="Press Search to look at offers."')
                self.more = ui.button("Show more", icon="expand_more", on_click=self.show_more, color=None) \
                    .props("outline no-caps")
                self.more.visible = False
        self.apply(self.mk.default_filters())
        self.update_counts()
        background_tasks.create(self.load_prefs(), name="vast filters")
        background_tasks.create(self.load_catalog(), name="vast catalog")

    def control(self, f) -> None:
        w = self.widgets
        enter = self.search
        if f.kind == "range":
            lo = f.lo if f.lo is not None else 0
            hi = f.hi if f.hi is not None else ""
            ui.label(f.label + (f" ({f.unit})" if f.unit else "")).classes("rz-setting-title")
            with ui.row().classes("w-full gap-2 no-wrap"):
                a = ui.number(label="min", placeholder=f"{lo:g}", step=f.step or 1).props("dense outlined clearable") \
                    .classes("flex-1 min-w-0")
                b = ui.number(label="max", placeholder=f"{hi:g}" if hi != "" else "", step=f.step or 1) \
                    .props("dense outlined clearable").classes("flex-1 min-w-0")
            for el in (a, b):
                el.on("keydown.enter", enter)
                el.on_value_change(self.on_change)
            w[f.key] = (a, b)
        elif f.kind == "min":
            lo = f.lo if f.lo is not None else 0
            e = ui.number(label=f.label + (f" ({f.unit})" if f.unit else ""), placeholder=f"{lo:g}",
                          step=f.step or 1).props("dense outlined clearable").classes("w-full")
            e.on("keydown.enter", enter)
            e.on_value_change(self.on_change)
            w[f.key] = e
        elif f.kind == "switch":
            e = ui.switch(f.label)
            e.on_value_change(self.on_change)
            w[f.key] = e
        elif f.kind == "choice":
            e = ui.select({str(v): label for v, label in f.options}, value=str(f.default), label=f.label) \
                .props("dense outlined").classes("w-full")
            e.on_value_change(self.on_change)
            w[f.key] = e
        elif f.kind == "multi":
            options = self.multi_options(f)
            e = ui.select(options, value=[], multiple=True, with_input=True, label=f.label) \
                .props("dense outlined use-chips").classes("w-full")
            e.on_value_change(self.on_change)
            w[f.key] = e
        else:
            e = ui.input(f.label).props("dense outlined clearable").classes("w-full")
            e.on("keydown.enter", enter)
            e.on_value_change(self.on_change)
            w[f.key] = e
        if f.hint and f.kind != "range":
            ui.label(f.hint).classes("rz-hint text-xs")

    async def load_prefs(self) -> None:
        """The remembered filters (market.json) replace the defaults once they are read."""
        market = self.vast.services.get_market()
        if market is None:
            return
        try:
            prefs = await run.io_bound(market.prefs)
        except Exception:  # noqa: BLE001  (the defaults stay)
            return
        if not alive(self.root):
            return
        self.apply(prefs.get("filters") or self.mk.default_filters())
        self.update_counts()

    def multi_options(self, f) -> dict:
        mk = self.mk
        if f.key == "gpu_name":
            return {g: g for g in mk.FALLBACK_GPUS}
        if f.key == "countries":
            return {c: n for c, n in sorted(mk.COUNTRY_NAMES.items(), key=lambda kv: kv[1])}
        return {str(o): str(o) for o in f.options}

    async def load_catalog(self) -> None:
        """GPU models and countries seen on vast.ai (cached a day); the built-in lists stand in until then."""
        market = self.vast.services.get_market()
        if market is None or self.catalog_loaded:
            return
        try:
            data = await run.io_bound(self.vast.catalog)
        except Exception:  # noqa: BLE001  (the built-in lists stay)
            return
        if not alive(self.root):
            return
        self.catalog_loaded = True
        gpus, countries = data.get("gpus") or [], data.get("countries") or []
        for key, options in (("gpu_name", {g: g for g in gpus}),
                             ("countries", {c["code"]: c["name"] for c in countries})):
            sel = self.widgets.get(key)
            if sel is None or not options:
                continue
            keep = {v: v for v in (sel.value or []) if v not in options}
            if key == "countries":
                keep = {v: self.mk.COUNTRY_NAMES.get(v, v) for v in keep}
            sel.set_options({**options, **keep})

    # -- values

    def read(self) -> dict:
        out = {}
        for f in self.mk.FILTERS:
            w = self.widgets.get(f.key)
            if f.key == "sort":
                out["sort"] = self.sort.value
            elif w is None:
                continue
            elif f.kind == "range":
                out[f.key] = [num(w[0].value), num(w[1].value)]
            elif f.kind == "min":
                out[f.key] = num(w.value)
            elif f.kind == "switch":
                out[f.key] = bool(w.value)
            elif f.kind == "multi":
                out[f.key] = list(w.value or [])
            else:
                out[f.key] = w.value if w.value is not None else ""
        return out

    def apply(self, filters: dict) -> None:
        """Put filter values on the controls (a saved set, the defaults, or what the search cleaned them to)."""
        self.applying = True
        try:
            for f in self.mk.FILTERS:
                v = filters.get(f.key, f.default)
                w = self.widgets.get(f.key)
                if f.key == "sort":
                    self.sort.value = str(v)
                elif w is None:
                    continue
                elif f.kind == "range":
                    pair = v if isinstance(v, (list, tuple)) and len(v) == 2 else [None, None]
                    w[0].value, w[1].value = pair[0], pair[1]
                elif f.kind in ("min",):
                    w.value = v
                elif f.kind == "switch":
                    w.value = bool(v)
                elif f.kind == "choice":
                    w.value = str(v)
                elif f.kind == "multi":
                    have = set((w.options if isinstance(w.options, dict) else {}))
                    extra = [x for x in v if x not in have]
                    if extra:
                        w.set_options({**(w.options if isinstance(w.options, dict) else {}), **{x: x for x in extra}})
                    w.value = list(v)
                else:
                    w.value = v or ""
        finally:
            self.applying = False

    def on_change(self, _=None) -> None:
        if not self.applying:
            self.update_counts()
            self.paint_stale()

    def cleaned(self) -> dict:
        try:
            return self.mk.clean_filters(self.read())
        except Exception:  # noqa: BLE001
            return self.read()

    def paint_stale(self) -> None:
        """The offers belong to the filters they were searched with: say so when the controls moved on since."""
        self.stale.visible = self.searched_with is not None and self.cleaned() != self.searched_with["filters"]

    def update_counts(self) -> None:
        counts = filter_counts(self.read(), self.mk.FILTERS)
        for group, exp in self.group_boxes.items():
            exp.props["label"] = group_label(group, counts.get(group, 0))
            exp.update()

    async def reset(self) -> None:
        self.apply(self.mk.default_filters())
        self.update_counts()
        market = self.vast.services.get_market()
        if market is not None:
            try:
                await run.io_bound(functools.partial(market.set_prefs, filters=self.read()))
            except Exception:  # noqa: BLE001  (not remembered; the controls are reset all the same)
                pass
        notify_page("The filters are back to the defaults. Search to see offers.", type="info")

    # -- searching (never rents)

    async def on_sort(self, _=None) -> None:
        if not self.applying and self.searched:
            await self.search()

    async def search(self, _=None) -> None:
        market = self.vast.services.get_market()
        if market is None or self.searching:
            return
        filters = self.read()
        try:
            filters = self.mk.clean_filters(filters)
        except Exception:  # noqa: BLE001
            pass
        self.apply(filters)
        self.update_counts()
        self.searching = True
        self.spinner.visible = True
        self.message.visible = False
        try:
            template = await self.vast.search_template()
            await run.io_bound(functools.partial(market.set_prefs, filters=filters))
            offers = await run.io_bound(market.search, filters, template)
        except Exception as e:  # noqa: BLE001
            if alive(self.root):
                self.message.text = sentence(e, "The search did not work. Try again in a moment.")
                self.message.visible = True
            return
        finally:
            self.searching = False
            if alive(self.root):
                self.spinner.visible = False
        if not alive(self.root):
            return
        self.searched = True
        self.searched_with = {"filters": filters, "template": template}  # what Rent uses, not what the controls say now
        self.stale.visible = False
        self.offers = list(offers)
        self.shown = min(PAGE_SIZE, len(self.offers))
        self.paint()

    def paint(self) -> None:
        total = len(self.offers)
        self.table.rows = [offer_row(o) for o in self.offers[:self.shown]]
        self.table.update()
        self.table.props('no-data-label="No offers match these filters. Loosen one and search again."')
        self.count.text = (f"{total} offer{'s' if total != 1 else ''}"
                           + (f", showing {self.shown}" if self.shown < total else "")) if total else "No offers."
        self.more.visible = self.shown < total

    def show_more(self) -> None:
        self.shown = min(len(self.offers), self.shown + PAGE_SIZE)
        self.paint()

    # -- renting

    async def on_rent(self, e) -> None:
        oid = e.args
        offer = next((o for o in self.offers if _get(o, "id") == oid), None)
        if offer is None or not can_rent(offer)[0] or self.searched_with is None:
            return
        result = await self.vast.open_rent(offer, dict(self.searched_with["filters"]), pick=self.pick,
                                           upload_bytes=self.upload_bytes, template=self.searched_with["template"])
        if result and self.pick and self.on_pick and alive(self.root):
            self.on_pick(result)


# ---------------------------------------------------------------- the page's side: chip, tab, run-on, rent dialog


class VastUI:
    """Everything one page has of this: the credit chip, the Rent GPUs tab, the Run on control, the rent and send
    dialogs, the sends card. `kit` holds the page's own helpers (see webui.build_page): card, hint, eta_text,
    while_connected, goto, current_tab, resume_queue, root, version, step_seconds."""

    def __init__(self, kit, services: Services = SERVICES, watch: AccountWatch = WATCH):
        self.kit = kit
        self.services = services
        self.watch = watch
        self.run_on = "pc"
        self.controls: list = []
        self.listeners: list = []
        self.chip = None
        self.tab = {"built": False, "sig": None, "intents_at": 0.0, "intents": [], "intents_sig": None}
        self._templates = None
        self.template_error = ""
        self.sending_busy = False  # a send dialog is open or opening on this page
        self.rent_busy = False  # a rent dialog is open or opening on this page
        self.local = bool(getattr(kit, "local", True))  # False: opened from another device (LAN, Tailscale)
        self.notices: list[str] = []  # stops the send manager could not confirm (its warnings, the Fleet's alerts)
        self.dismissed: set = set()  # notices this page's viewer dismissed
        self._notice_at = 0.0

    # -- blocking helpers (run with run.io_bound)

    def catalog(self) -> dict:
        market = self.services.get_market()
        fn = getattr(market, "catalog", None)
        if callable(fn):
            return fn()
        return market_module().catalog(market.api, market.home)

    async def manager(self):
        m = self.services.manager
        return m if m is not None else await run.io_bound(self.services.get_manager)

    def sending_ok(self) -> bool:
        return self.services.manager is not None or self.services.module("rulz_send") is not None

    async def templates(self, force: bool = False) -> list:
        market = self.services.get_market()
        if market is None:
            return []
        if self._templates is None or force:
            try:
                self._templates = await run.io_bound(market.templates)
                self.template_error = ""
            except Exception as e:  # noqa: BLE001
                self.template_error = sentence(e, "The templates could not be read.")
                return []
        return self._templates

    async def search_template(self):
        """The template whose image the "incompatible machines" filter reads (the saved one, else the default)."""
        market = self.services.get_market()
        items = await self.templates()
        try:
            prefs = await run.io_bound(market.prefs)
        except Exception:  # noqa: BLE001
            prefs = {}
        return self.pick_template(items, prefs.get("template"))

    def pick_template(self, items: list, saved) -> dict:
        found = next((t for t in items if saved and saved in (t.get("hash_id"), str(t.get("id")))), None)
        return found or market_module().default_template(items)

    # -- the header chip

    def build_chip(self):
        """The credit chip, hidden until something bills (and shown whenever the Rent GPUs tab is open)."""
        with ui.row().classes("rz-chip items-center gap-2 no-wrap") as chip:
            icon = ui.icon("account_balance_wallet")
            text = ui.label("").classes("rz-chip-text")
            tip = ui.tooltip("").style("white-space: pre-line")
        chip.on("click", lambda: self.kit.goto(TAB))
        chip.visible = False
        self.chip = {"row": chip, "icon": icon, "text": text, "tip": tip, "level": "ok"}
        return chip

    def paint_chip(self, snap: dict) -> None:
        if self.chip is None or not alive(self.chip["row"]):
            return
        view = chip_view(self.merged(snap))
        on_tab = self.kit.current_tab() == TAB
        self.chip["row"].visible = bool(view["billed"] or (on_tab and snap.get("key") is not False))
        self.chip["text"].text = view["text"]
        self.chip["icon"].props(f"name={view['icon']}")
        if view["level"] != self.chip["level"]:
            self.chip["level"] = view["level"]
            self.chip["row"].classes(remove="ok warning critical", add=view["level"])
        self.chip["tip"].text = "\n".join(t for t in view["tip"] if t)

    # -- the Run on control

    def build_run_on(self):
        """"This PC | vast.ai", one value for the page (every control on it follows), remembered per browser."""
        if not self.sending_ok():
            return None
        with ui.row().classes("rz-views items-center no-wrap") as row:
            pc = ui.button("This PC", icon="computer", on_click=lambda: self.set_run_on("pc"), color=None) \
                .props("flat no-caps dense").tooltip("Train on this computer's GPUs")
            cloud = ui.button("vast.ai", icon="cloud", on_click=lambda: self.set_run_on("vast"), color=None) \
                .props("flat no-caps dense").tooltip("Send the run to a vast.ai instance and train there")
        self.controls.append((row, pc, cloud))
        self.paint_run_on()
        return row

    def paint_run_on(self) -> None:
        for _, pc, cloud in self.controls:
            for btn, mine in ((pc, "pc"), (cloud, "vast")):
                btn.classes(add="rz-on") if self.run_on == mine else btn.classes(remove="rz-on")

    def set_run_on(self, value: str, save: bool = True) -> None:
        if value not in ("pc", "vast") or value == self.run_on:
            return
        self.run_on = value
        self.paint_run_on()
        if save:
            ui.run_javascript(f"try {{ localStorage.setItem({RUN_ON_KEY!r}, {value!r}) }} catch (e) {{}}")
        for fn in list(self.listeners):
            fn(value)

    def on_run_on(self, fn) -> None:
        self.listeners.append(fn)

    async def restore_run_on(self) -> None:
        if not self.controls:
            return
        try:
            value = await ui.run_javascript(f"localStorage.getItem({RUN_ON_KEY!r})", timeout=3)
        except Exception:  # noqa: BLE001  (storage blocked, slow client)
            return
        if value in ("pc", "vast"):
            self.set_run_on(value, save=False)

    # -- the timer

    def guard_local(self) -> bool:
        """Renting, starting, destroying and adding a key bill or change the account: only from a browser on this
        computer (checked here as well as on the buttons)."""
        if self.local:
            return True
        notify_page(LOCAL_ONLY, type="warning")
        return False

    def collect_notices(self) -> list[str]:
        """(blocking) A stop that is not confirmed: a stop warning on a send, or a send_alert in the Fleet's events in
        the last hour."""
        out: list[str] = []
        manager = self.services.manager
        if manager is not None:
            for send in manager.sends() or []:
                for w in send.get("warnings") or []:
                    low = str(w).lower()
                    if "stop" in low and "bill" in low and str(w) not in out:
                        out.append(str(w))
        getter = self.services.fleet_getter
        if getter is not None:
            try:
                for e in getter().events(kinds=("send_alert",)):
                    if time.time() - (num(e.get("time")) or 0) < 3600:
                        text = str(e.get("error") or e.get("text") or "A send needs a look.")
                        if text not in out:
                            out.append(text)
            except Exception:  # noqa: BLE001  (an unreadable event log shows nothing)
                pass
        return out

    def merged(self, snap: dict) -> dict:
        """The watch's snapshot with the send manager's unconfirmed stops as alerts too (minus those dismissed)."""
        have = {a.get("id") for a in snap.get("alerts") or []}
        extra = [{"id": "n:" + t, "text": t} for t in self.notices if "n:" + t not in self.dismissed
                 and "n:" + t not in have]
        return {**snap, "alerts": list(snap.get("alerts") or []) + extra}

    async def tick(self) -> None:
        if time.time() - self._notice_at >= 20.0:
            self._notice_at = time.time()
            try:
                self.notices = await run.io_bound(self.collect_notices)
            except Exception:  # noqa: BLE001
                pass
        snap = self.watch.snapshot()
        self.paint_chip(snap)
        if self.tab["built"] and self.kit.current_tab() == TAB:
            await self.paint_tab(snap)

    # -- the Rent GPUs tab

    def build_tab(self) -> None:
        """The tab's body is drawn when the tab is first shown (the filters are ~40 controls)."""
        self.tab_root = ui.column().classes("w-full gap-4")
        with self.tab_root:
            ui.label("Opening vast.ai ...").classes("rz-hint")

    def tab_shown(self) -> None:
        if self.tab["built"]:
            self.watch.request()
            return
        self.tab["built"] = True
        self.tab_root.clear()
        with self.tab_root:
            if not self.services.available():
                ui.label(self.services.why_not()).classes("rz-banner error")
                return
            self.alert_box = ui.column().classes("w-full gap-2")
            self.nokey = self.kit.card()
            with self.nokey:
                self.build_nokey()
            self.credit_card = self.kit.card()
            with self.credit_card:
                self.build_credit()
            self.intent_box = ui.column().classes("w-full gap-2")
            self.inst_card = self.kit.card()
            with self.inst_card:
                ui.label("Your instances").classes("text-base font-medium")
                self.inst_note = self.kit.hint("")
                self.inst_box = ui.column().classes("w-full gap-0")
            self.market_holder = ui.column().classes("w-full gap-3")
            self.market_view = MarketView(self)
            self.market_view.build(self.market_holder)
        background_tasks.create(self.paint_tab(self.watch.snapshot()), name="vast tab")
        self.watch.request()

    def build_nokey(self) -> None:
        ui.label("Connect your vast.ai account").classes("text-base font-medium")
        self.kit.hint("This page needs your vast.ai API key to look at your account and to rent. Make a key at "
                      f"{MANAGE_KEYS_URL}, then run this once in a terminal on this PC. The key is stored by the "
                      "vastai tool; this page never asks for it.")
        with ui.row().classes("items-center gap-2"):
            ui.link("Open vast.ai API keys", MANAGE_KEYS_URL, new_tab=True)
        with ui.row().classes("items-center gap-2 no-wrap"):
            ui.label(f"{SET_KEY_COMMAND} <your key>").classes("rz-mono rz-vast-code")
            ui.button("Copy", icon="content_copy", color=None,
                      on_click=lambda: ui.run_javascript(
                          f"navigator.clipboard && navigator.clipboard.writeText({SET_KEY_COMMAND + ' '!r})")) \
                .props("flat dense no-caps")
        with ui.row().classes("items-center gap-2"):
            ui.button("Check again", icon="refresh", on_click=self.check_key, color=None).props("outline no-caps dense")
            self.nokey_note = self.kit.hint("")

    async def check_key(self) -> None:
        snap = await run.io_bound(self.watch.poll)
        self.nokey_note.text = "" if snap.get("key") else "No API key found yet."
        await self.paint_tab(snap)

    def build_credit(self) -> None:
        with ui.row().classes("w-full items-start gap-4"):
            with ui.column().classes("gap-0"):
                ui.label("vast.ai credit").classes("rz-tile-label")
                self.credit_value = ui.label("-").classes("rz-tile-value")
            with ui.column().classes("gap-0 flex-1 min-w-0"):
                self.credit_lines = ui.column().classes("gap-0")
            ui.button("Refresh", icon="refresh", on_click=self.refresh_account, color=None) \
                .props("flat dense no-caps")
        self.credit_error = ui.label("").classes("rz-banner warning")
        self.credit_error.visible = False

    async def refresh_account(self) -> None:
        snap = await run.io_bound(self.watch.poll)
        await self.paint_tab(snap)

    async def paint_tab(self, snap: dict) -> None:
        """The credit, the alerts, the open rents and the instances, from the shared snapshot (no call to vast.ai)."""
        if not self.tab["built"] or not hasattr(self, "credit_card") or not alive(self.tab_root):
            return
        snap = self.merged(snap)
        key = snap.get("key")
        self.nokey.visible = key is False
        self.credit_card.visible = key is not False
        self.inst_card.visible = key is not False
        self.market_holder.visible = key is not False  # offers need the key too
        if key is False:
            self.alert_box.clear()
            return
        acct = snap.get("account")
        credit = acct.get("credit") if acct else None
        self.credit_value.text = price(credit) if credit is not None else "-"
        sig = ("credit", tuple(credit_lines(snap)), snap.get("error"))
        if sig != self.tab.get("credit_sig"):
            self.tab["credit_sig"] = sig
            self.credit_lines.clear()
            with self.credit_lines:
                for line in credit_lines(snap):
                    ui.label(line).classes("rz-hint")
        self.credit_error.text = snap.get("error") or ""
        self.credit_error.visible = bool(snap.get("error")) and bool(acct)
        alerts = snap.get("alerts") or []
        asig = ("alerts", tuple((a["id"], a["text"]) for a in alerts))
        if asig != self.tab.get("alert_sig"):
            self.tab["alert_sig"] = asig
            self.alert_box.clear()
            with self.alert_box:
                for a in alerts:
                    with ui.row().classes("rz-banner error items-center gap-2 no-wrap"):
                        ui.icon("error").classes("rz-rs-icon err")
                        ui.label(a["text"]).classes("flex-1")
                        ui.button("Dismiss", on_click=lambda _, i=a["id"]: self.dismiss_alert(i), color=None) \
                            .props("flat dense no-caps")
        rows = list((acct or {}).get("instances") or [])
        stopping = snap.get("stopping") or {}
        isig = ("inst", tuple((r.get("id"), r.get("status"), round(num(r.get("per_hour")) or 0, 4), r.get("label"),
                              bool(r.get("ours")), r.get("id") in stopping) for r in rows))
        if isig != self.tab.get("sig"):
            self.tab["sig"] = isig
            self.draw_instances(rows, stopping)
        await self.paint_intents()

    def dismiss_alert(self, instance_id) -> None:
        if isinstance(instance_id, str) and instance_id.startswith("n:"):
            self.dismissed.add(instance_id)
        else:
            self.watch.dismiss_alert(instance_id)
        self.tab["alert_sig"] = None
        background_tasks.create(self.paint_tab(self.watch.snapshot()), name="vast alerts")

    async def paint_intents(self) -> None:
        """Rents whose answer was lost, shown with Check again and Dismiss (they also stop any new rent)."""
        market = self.services.get_market()
        if market is None or not hasattr(self, "intent_box"):
            return
        if time.time() - self.tab["intents_at"] >= 4.0:
            self.tab["intents_at"] = time.time()
            try:
                self.tab["intents"] = await run.io_bound(market.intents)
            except Exception:  # noqa: BLE001
                pass
            if not alive(self.intent_box):
                return
        intents = self.tab["intents"]
        sig = tuple((i.get("label"), i.get("state"), i.get("message")) for i in intents)
        if sig == self.tab["intents_sig"]:
            return
        self.tab["intents_sig"] = sig
        self.intent_box.clear()
        with self.intent_box:
            for i in intents:
                self.draw_intent(i)

    def draw_intent(self, intent: dict, after=None) -> None:
        with ui.column().classes("rz-banner warning gap-2"):
            status_line("warning", {"unresolved": "Not found", "checking": "Checking", "sending": "Sending"}.get(
                intent.get("state"), "Open"), f"Rent {intent.get('label')}: {intent.get('gpu_name') or 'GPU'}", "warn")
            ui.label(intent.get("message") or "The rent request is still being sent.").classes("rz-hint")
            with ui.row().classes("gap-2"):
                ui.button("Check again", icon="refresh", color=None,
                          on_click=lambda _, i=intent: self.check_intent(i, after))                     .props("outline dense no-caps").classes("rz-vast-btn")
                ui.button("Dismiss", icon="close", color=None,
                          on_click=lambda _, i=intent: self.dismiss_intent(i, after))                     .props("flat dense no-caps").classes("rz-vast-btn")

    async def check_intent(self, intent: dict, after=None) -> None:
        market = self.services.get_market()
        try:
            found = await run.io_bound(market.check_intent, intent["label"])
        except Exception as e:  # noqa: BLE001
            notify_page(sentence(e), type="warning", multi_line=True)
            return
        if found.get("found"):
            notify_page(f"Found it: instance {found['rental']['id']} is yours.", type="positive")
            self.watch.request()
        else:
            notify_page("No instance with that label yet.", type="info")
        self.tab["intents_at"] = 0.0
        self.tab["intents_sig"] = None
        if after:
            await after()
        elif self.tab["built"]:
            await self.paint_intents()

    async def dismiss_intent(self, intent: dict, after=None) -> None:
        market = self.services.get_market()
        try:
            await run.io_bound(market.dismiss_intent, intent["label"])
        except Exception as e:  # noqa: BLE001
            notify_page(sentence(e), type="warning", multi_line=True)
            return
        self.tab["intents_at"] = 0.0
        self.tab["intents_sig"] = None
        if after:
            await after()
        elif self.tab["built"]:
            await self.paint_intents()

    # -- your instances

    def draw_instances(self, rows: list, stopping: dict) -> None:
        self.inst_box.clear()
        self.inst_note.text = ("" if rows else "No instances on your account.")
        self.inst_note.visible = not rows
        with self.inst_box:
            for row in sorted(rows, key=lambda r: (not _is_up(r), r.get("id") or 0)):
                with ui.element("div").classes("rz-vast-inst"):
                    icon, word, level = instance_state_word(row)
                    with ui.element("div").classes("rz-vast-inst-state"):
                        status_line(icon, word, "", level)
                    with ui.column().classes("gap-0 min-w-0"):
                        with ui.row().classes("items-center gap-2"):
                            ui.label(f"#{row['id']} · {instance_label(row)}").classes("rz-setting-title")
                            if row.get("ours"):
                                ui.label("Rented by this app").classes("rz-badge muted")
                            elif row.get("is_rulz"):
                                ui.label("RulZ AI Trainer").classes("rz-badge muted")
                        bits = [per_hour(row.get("per_hour")) + (" (disk only)" if not _is_up(row) else "")]
                        if row.get("label"):
                            bits.append(row["label"])
                        if row.get("geolocation"):
                            bits.append(row["geolocation"])
                        ui.label(" · ".join(bits)).classes("rz-hint")
                        if row.get("id") in stopping and _is_up(row):
                            ui.label("A stop was asked and vast.ai still reports it running: it still bills. "
                                     "This page keeps checking.").classes("rz-hint rz-warn")
                    with ui.row().classes("items-center gap-1 rz-vast-inst-btns"):
                        if _is_stopped(row):
                            self.local_only(ui.button("Start", icon="play_arrow", color=None,
                                                      on_click=lambda _, r=row: self.start_instance(r))
                                            .props("outline dense no-caps"))
                        if _is_up(row):
                            ui.button("Stop", icon="stop", color=None,
                                      on_click=lambda _, r=row: self.stop_instance(r)).props("outline dense no-caps")
                        self.local_only(ui.button("Destroy", icon="delete_forever", color=None,
                                                  on_click=lambda _, r=row: self.destroy_instance(r))
                                        .props("flat dense no-caps").classes("rz-danger"))
                        ui.link("Open on vast.ai", INSTANCES_URL, new_tab=True).classes("rz-vast-link")

    def local_only(self, button):
        """A button that bills or changes the account: off, with the reason, on a page opened from another device."""
        if not self.local:
            button.set_enabled(False)
            button.tooltip(LOCAL_ONLY)
        return button

    def credit_after(self, extra_per_hour: float) -> str:
        snap = self.watch.snapshot()
        acct = snap.get("account") or {}
        burn = (acct.get("burn") or {}).get("per_hour") or 0.0
        h = lasts(acct.get("credit"), burn + extra_per_hour)
        return f" Your credit then lasts about {hours_text(h)}." if h is not None else ""

    async def start_instance(self, row: dict) -> None:
        if not self.guard_local():
            return
        extra = max(0.0, (num(row.get("dph_total")) or 0.0) - (num(row.get("storage_per_hour")) or 0.0))
        if not await confirm(f"Start instance {row['id']} ({instance_label(row)})?",
                             [f"It bills {per_hour(row.get('dph_total'))} from the moment it starts. It keeps billing until you stop "
                              "or destroy it, with or without a run on it." + self.credit_after(extra),
                              "Everything on its disk is still there."], "Start it", "play_arrow"):
            return
        try:  # what is billed is what was shown: read it again right before the start
            fresh = await run.io_bound(self.market_instance, row["id"])
        except Exception as e:  # noqa: BLE001
            notify_page(sentence(e, "The instance could not be read again, so nothing was started."), type="warning",
                        multi_line=True)
            return
        change = instance_changed(row, fresh)
        if change:
            notify_page(f"{change} Nothing was started: look at it and click Start again.", type="warning",
                        multi_line=True)
            self.watch.request()
            snap = await run.io_bound(self.watch.poll)
            await self.paint_tab(snap)
            return
        if await self.market_call("start", row["id"], ok=f"Starting instance {row['id']}."):
            manager = self.services.manager  # a stop asked for earlier must not stop this start again
            if manager is not None and hasattr(manager, "clear_stop"):
                await run.io_bound(manager.clear_stop, row["id"])

    async def stop_instance(self, row: dict) -> None:
        disk = num(row.get("storage_per_hour"))
        if not await confirm(f"Stop instance {row['id']} ({instance_label(row)})?",
                             ["Stops billing for the GPU; the disk stays"
                              + (f" and costs {per_hour(disk)}." if disk else "."),
                              "Anything running on it ends now: save or download what you need first."],
                             "Stop it", "stop"):
            return
        market = self.services.get_market()
        if await self.market_call("stop", row["id"], ok=f"Stop asked for instance {row['id']}."):
            background_tasks.create(self.follow_stop(market, row["id"]), name="vast stop watch")

    async def follow_stop(self, market, instance_id: int) -> None:
        """A stop is followed until vast.ai no longer reports the instance running: by the send manager (it goes on
        after a restart of this WebUI) and by this page's own watch (the loud alert on the tab and the chip)."""
        manager = self.services.manager
        if manager is not None and hasattr(manager, "follow_stop"):
            try:
                await run.io_bound(manager.follow_stop, instance_id)  # looks every 2 min, send_alert after 10
                return
            except Exception:  # noqa: BLE001  (then the page's own watch follows it)
                pass
        await run.io_bound(self.watch.follow_stop, market, instance_id)

    async def destroy_instance(self, row: dict) -> None:
        if not self.guard_local():
            return
        label = f", label {row['label']}" if row.get("label") else ""
        if not await confirm(f"Destroy instance {row['id']} ({instance_label(row)}{label})?",
                             ["Every dataset, model and run on its disk is deleted for good. This cannot be undone.",
                              "The instance stops billing at once."], "Destroy instance " + str(row["id"]),
                             "delete_forever", danger=True):
            return
        await self.market_call("destroy", row["id"], ok=f"Instance {row['id']} destroyed.")

    async def market_call(self, name: str, *args, ok: str = "") -> bool:
        market = self.services.get_market()
        try:
            await run.io_bound(getattr(market, name), *args)
        except Exception as e:  # noqa: BLE001
            notify_page(sentence(e), type="warning", multi_line=True)
            return False
        if ok:
            notify_page(ok, type="positive")
        self.watch.request()
        snap = await run.io_bound(self.watch.poll)
        self.paint_chip(snap)
        if self.tab["built"]:
            await self.paint_tab(snap)
        return True

    def market_instance(self, instance_id: int):
        """(blocking) The instance as vast.ai lists it now, or None."""
        market = self.services.get_market()
        return next((i for i in market.instances() if i.get("id") == instance_id), None)

    # -- the rent dialog (second step)

    async def open_rent(self, *args, **kw):
        """One rent dialog per page: a second click while one is open or opening does nothing."""
        if self.rent_busy:
            return None
        self.rent_busy = True
        try:
            return await self._open_rent(*args, **kw)
        finally:
            self.rent_busy = False

    async def _open_rent(self, offer, filters: dict, pick: bool = False, upload_bytes: int = 0,
                         models_gb: float = MODELS_GB, template=None):
        """Price per hour and per day, template, bid, what the credit lasts with it, the ssh key, warnings, and one red
        button: "Rent for $0.53/hour" (in the send dialog "Use this offer": the rent itself happens with the send)."""
        mk, market = market_module(), self.services.get_market()
        kind = _get(offer, "type") or "ask"
        disk = int(num(filters.get("disk")) or 200)
        st = {"offer": offer, "ssh": None, "templates": [], "intents": [], "busy": False, "price_note": ""}
        min_bid = num(_get(offer, "min_bid"))
        with ui.dialog().props("persistent") as dlg, ui.card().classes("w-[40rem] max-w-full gap-3") \
                .style("max-width: 96vw !important"):
            ui.label(f"{'Use' if pick else 'Rent'} {_get(offer, 'num_gpus') or 1}x {_get(offer, 'gpu_name')}") \
                .classes("rz-step-title")
            ui.label(" · ".join(x for x in (f"machine {_get(offer, 'machine_id')}", _get(offer, "geolocation"),
                                            "interruptible" if kind == "bid" else "on-demand") if x)).classes("rz-hint")
            if not pick:
                ui.label("It keeps billing until you stop or destroy it.").classes("rz-hint")
            summary = ui.element("div").classes("rz-summary")
            bid_input = None
            if kind == "bid":
                start = round(min_bid * 1.2, 3) if min_bid else None
                bid_input = ui.number("Bid per hour for the GPU ($)", value=start, step=0.01, min=0, format="%.3f") \
                    .props("dense outlined").classes("w-64")
                if min_bid:
                    ui.label(f"The minimum bid now is {price(min_bid)}/hour. The disk adds "
                             f"{per_hour(_get(offer, 'storage_dph'))}.").classes("rz-hint")
                bid_total = ui.label("").classes("rz-hint")
            template_select = ui.select({}, label="Template").props("dense outlined").classes("w-full")
            template_hint = ui.label("Loading templates ...").classes("rz-hint")
            lasts_line = ui.label("").classes("rz-hint")
            with ui.row().classes("items-center gap-2 no-wrap"):
                ssh_box = ui.column().classes("gap-0 flex-1 min-w-0")
                ssh_btn = ui.button("Add this PC's ssh key to your vast.ai account", icon="vpn_key", color=None,
                                    on_click=lambda: add_key()).props("outline dense no-caps")
            ssh_btn.visible = False
            ssh_btn.set_enabled(self.local)
            warn_box = ui.column().classes("w-full gap-1")
            intent_box = ui.column().classes("w-full gap-2")
            error = ui.label("").classes("rz-banner error")
            error.visible = False
            why_label = ui.label("").classes("rz-hint rz-vast-why")
            with ui.row().classes("w-full items-center justify-end gap-2"):
                spin = ui.spinner(size="sm")
                spin.visible = False
                cancel_btn = ui.button("Cancel", on_click=lambda: dlg.submit(None), color=None).props("flat no-caps")
                go = ui.button("", icon="shopping_cart_checkout", on_click=lambda: do()).props("unelevated no-caps")

        def bid_value():
            return num(bid_input.value) if bid_input is not None else None

        def total():
            return offer_price(st["offer"], bid_value())

        def snapshot_burn():
            acct = self.watch.snapshot().get("account") or {}
            return acct.get("credit"), (acct.get("burn") or {}).get("per_hour") or 0.0

        def paint() -> None:
            if not alive(dlg):
                return
            p = total()
            summary.clear()
            with summary:
                rows = [("Price", f"{per_hour(p)} with a {disk} GB disk" if p is not None else "set a bid"),
                        ("Per day", f"${p * 24:,.2f}" if p is not None else "-"),
                        ("Reliability", f"{(num(_get(offer, 'reliability')) or 0) * 100:.1f}%"),
                        ("Network", f"{_n(_get(offer, 'inet_up'))} up / {_n(_get(offer, 'inet_down'))} down Mbps"),
                        ("Traffic", f"{offer_row(offer)['up']} up, {offer_row(offer)['down']} down")]
                for k, v in rows:
                    ui.label(k).classes("k")
                    ui.label(v).classes("v")
            credit, burn = snapshot_burn()
            h = lasts(credit, burn + (p or 0.0)) if credit is not None else None
            lasts_line.text = (f"Your {price(credit)} credit lasts about {hours_text(h)} with this instance on top of "
                               f"what bills now.") if h is not None else ""
            warn_box.clear()
            with warn_box:
                for w in rent_warnings(st["offer"], disk, bid_value(), upload_bytes, models_gb):
                    ui.label(w).classes("rz-banner warning")
            key = st["ssh"]
            ssh_box.clear()
            with ssh_box:
                if key == "on_account":
                    status_line("check_circle", "Key ready", "Your ssh key is on your vast.ai account.", "ok")
                elif key == "not_on_account":
                    status_line("warning", "Key missing", "This PC's ssh key is not on your vast.ai account, so a "
                                                          "rented instance could not be reached.", "warn")
                elif key == "missing":
                    status_line("error", "No key", "There is no ssh key on this PC. Run ssh-keygen, then add it "
                                                   "here.", "err")
                else:
                    status_line("hourglass_top", "Checking", "Looking at your ssh key.", "muted")
            ssh_btn.visible = key == "not_on_account"
            intents = st["intents"]
            intent_box.clear()
            with intent_box:
                for i in intents:
                    self.draw_intent(i, after=refresh_intents)
            blocked = []
            if not pick and not self.local:
                blocked.append(LOCAL_ONLY)
            if bid_input is not None and p is not None:
                bid_total.text = (f"Total: bid {price(bid_value())} + disk {price(_get(st['offer'], 'storage_dph') or 0)}"
                                  f" = {per_hour(p)}")
            if intents:
                blocked.append("Another rent is still open: check it again or dismiss it first.")
            if not pick and key in ("missing", "not_on_account"):
                blocked.append("The ssh key is not ready.")
            if kind == "bid" and (p is None or (min_bid is not None and bid_value() is not None
                                                and bid_value() < min_bid)):
                blocked.append("The bid is under the minimum.")
            go.text = "Use this offer" if pick else f"Rent for {per_hour(p)}" if p is not None else "Rent"
            go.set_enabled(not st["busy"] and not blocked)
            why_label.text = ("The rent request is being sent. Closing this does not stop it." if st["busy"]
                              else " ".join(blocked))
            why_label.visible = bool(blocked) or st["busy"]
            error.visible = bool(error.text)

        async def refresh_intents() -> None:
            try:
                st["intents"] = await run.io_bound(market.intents)
            except Exception:  # noqa: BLE001
                pass
            paint()  # (does nothing once the dialog is gone)

        async def load() -> None:
            results = await asyncio.gather(self.templates(), run.io_bound(market.ssh_key_state),
                                           run.io_bound(market.intents), run.io_bound(market.prefs),
                                           return_exceptions=True)
            if not alive(dlg):
                return
            items = results[0] if isinstance(results[0], list) else []
            st["templates"] = items
            st["ssh"] = results[1] if isinstance(results[1], str) else None
            if isinstance(results[1], BaseException):
                st["ssh"] = None
                error.text = sentence(results[1], "Your ssh key could not be checked.")
            st["intents"] = results[2] if isinstance(results[2], list) else []
            prefs = results[3] if isinstance(results[3], dict) else {}
            options = {(t.get("hash_id") or "builtin"): t.get("name", "Template") + ("" if t.get("current")
                                                                                       else " (older setup)")
                       for t in items}
            builtin = mk.BUILTIN
            if not options:
                options = {"builtin": builtin["name"]}
            chosen = self.pick_template(items, prefs.get("template"))
            if template and any(t.get("hash_id") == template.get("hash_id") for t in items):
                chosen = template  # the template the search was made with
            template_select.set_options(options)
            template_select.value = chosen.get("hash_id") or "builtin"
            template_hint.text = (self.template_error or ("Older setups are listed too; the newest one that sets up "
                                                          "RulZ AI Trainer is picked." if items else
                                                          "None of your templates set up RulZ AI Trainer: the built-in "
                                                          "setup is used."))
            paint()

        async def add_key() -> None:
            if not self.guard_local():
                return
            if not await confirm("Add this PC's ssh key to your vast.ai account?",
                                 ["This changes your vast.ai account: the public key from this PC's ~/.ssh folder is "
                                  "added, so this app can reach the instances you rent. The private key never "
                                  "leaves this PC."], "Add the key", "vpn_key"):
                return
            try:
                await run.io_bound(market.add_ssh_key)
                st["ssh"] = await run.io_bound(market.ssh_key_state)
                notify_page("Your ssh key is on your vast.ai account.", type="positive")
            except Exception as e:  # noqa: BLE001
                if alive(dlg):
                    error.text = sentence(e)
            paint()

        def chosen_template():
            key = template_select.value
            return next((t for t in st["templates"] if (t.get("hash_id") or "builtin") == key), mk.BUILTIN)

        async def do() -> None:
            if st["busy"]:
                return
            error.text = ""
            tmpl, bid = chosen_template(), bid_value()
            if pick:
                dlg.submit({"offer": st["offer"], "filters": filters, "template": tmpl, "disk": disk, "bid": bid})
                return
            if not self.guard_local():
                return
            st["busy"] = True
            spin.visible = True
            cancel_btn.text = "Close"  # the rent request is on its way: closing does not stop it
            why_label.text = "The rent request is being sent. Closing this does not stop it."
            why_label.visible = True
            paint()
            try:
                rental = await run.io_bound(functools.partial(
                    market.rent, st["offer"], filters, tmpl, disk, "", bid, None))
            except Exception as e:  # noqa: BLE001
                if not alive(dlg):  # closed meanwhile: the answer goes to the page as a notification
                    notify_page(sentence(e), type="warning", multi_line=True)
                    return
                kind_ = getattr(e, "kind", "")
                data = getattr(e, "data", {}) or {}
                if kind_ == "price" and num(data.get("new_price")) is not None:
                    st["offer"] = with_price(st["offer"], float(data["new_price"]), bid)
                    error.text = (f"{sentence(e)} The button shows the new price: click it again to rent at "
                                  f"{per_hour(data['new_price'])}.")
                elif kind_ == "bid" and num(data.get("min_bid")) is not None and bid_input is not None:
                    bid_input.value = float(data["min_bid"])
                    error.text = sentence(e) + " The bid is raised to the minimum: check it, then rent again."
                else:
                    error.text = sentence(e)
                if kind_ == "pending":
                    await refresh_intents()
            else:
                self.watch.request()
                note = (rental or {}).get("note") or ""
                notify_page(f"Rented instance {rental.get('id')}. It is starting; it shows under Your instances."
                            + (f" {note}" if note else ""), type="warning" if note else "positive", multi_line=True)
                st["busy"] = False
                if alive(dlg):
                    dlg.submit({"rental": rental})
                return
            finally:
                if alive(spin):
                    spin.visible = False
            st["busy"] = False
            cancel_btn.text = "Cancel"
            paint()

        if bid_input is not None:
            bid_input.on_value_change(lambda _: paint())
        paint()
        dlg.open()
        background_tasks.create(load(), name="vast rent dialog")
        result = await dlg
        dlg.delete()
        return result

    # -- sending

    async def open_send(self, config: dict, job: str, warnings: list | None = None, root=None,
                        held: bool = False) -> None:
        """The send dialog for this run (see SendDialog). Returns when it is closed; a send that was made goes on in
        the background (its progress is on the Fleet tab). One per page: a second click while one is open or opening
        does nothing (`held`: the caller already took that place with `sending_busy`)."""
        if not held:
            if self.sending_busy:
                return
            self.sending_busy = True
        try:
            dialog = SendDialog(self, config, job, Path(root or self.kit.root), list(warnings or []))
            await dialog.run()
        finally:
            if not held:
                self.sending_busy = False

    def build_sends_card(self) -> None:
        """The Fleet tab's list of sends: a card, hidden while there are none, with each send's steps, the countdown
        with Keep running, and Dismiss for ended ones."""
        if not self.sending_ok():
            return
        views: dict = {}
        state = {"busy": False}
        with self.kit.card() as card:
            with ui.row().classes("w-full items-center gap-2"):
                ui.label("Sends to vast.ai").classes("text-base font-medium")
                ui.space()
            box = ui.column().classes("w-full gap-4")
        card.visible = False

        def gone(send_id) -> None:
            view = views.pop(send_id, None)
            if view is not None:
                view.root.delete()
            card.visible = bool(views)

        async def poll() -> None:
            manager = self.services.manager
            if manager is None or state["busy"]:
                return
            state["busy"] = True
            try:
                sends = await run.io_bound(manager.sends)
            except Exception:  # noqa: BLE001  (the next beat looks again)
                return
            finally:
                state["busy"] = False
            if not alive(card):
                return
            sends = list(sends or [])[:10]
            ids = [s.get("id") for s in sends]
            for stale in [i for i in views if i not in ids]:
                gone(stale)
            for send in sends:
                view = views.get(send.get("id"))
                if view is None:
                    view = SendView(self, box, send, dismissible=True, on_gone=gone)
                    views[send.get("id")] = view
                else:
                    view.update(send)
            card.visible = bool(views)

        ui.timer(2.0, self.kit.while_connected(poll))


# ---------------------------------------------------------------- the send dialog


def instance_row_text(row: dict) -> list[str]:
    """What the send dialog says about an instance: price, what runs there, free disk, version."""
    bits = [instance_label(row), per_hour(row.get("per_hour") if row.get("per_hour") is not None
                                          else row.get("dph_total"))]
    job = row.get("job")
    name = job.get("job") if isinstance(job, dict) else job
    if name:
        bits.append(f"running {name}")
    elif num(row.get("waiting")):
        bits.append(f"{int(row['waiting'])} waiting")
    else:
        bits.append("idle")
    if num(row.get("free_gb")) is not None:
        bits.append(f"{row['free_gb']:.0f} GB free")
    if row.get("version"):
        bits.append(f"RulZ AI Trainer {row['version']}")
    return bits


class SendDialog:
    """Where to run, what goes up, what happens after, what it costs, and one red button; then the send's progress.
    The steps are sections of one scrolling dialog. The red button is on only when every local model file has a
    choice, a bucket route has a confirmed bucket, and the plan has no blocker."""

    def __init__(self, vast: VastUI, config: dict, job: str, root: Path, warnings: list):
        self.vast = vast
        self.kit = vast.kit
        self.config, self.job, self.root, self.warnings = config, job, root, warnings
        self.plan = None
        self.plan_gpus = 1
        self.instances: list = []
        self.target = None  # {"kind": "instance", "row": row} | {"kind": "rent", "pick": {...}}
        self.picks: dict = {}
        self.prefs: dict = {}
        self.bucket = ""
        self.bucket_ok = False
        self.bucket_info = {"token": None, "buckets": [], "loaded": False}
        self.after = "stop"
        self.after_touched = False
        self.idle = 60
        self.busy = False
        self.view: SendView | None = None
        self.send_id = None
        self.timer = None
        self.closed = False

    # -- building

    async def run(self) -> None:
        self.build()
        self.dlg.open()
        background_tasks.create(self.load(), name="vast send dialog")
        await self.dlg
        self.closed = True  # what is still in flight must not touch the closed dialog
        if self.timer is not None:
            self.timer.cancel()
        self.dlg.delete()

    def live(self) -> bool:
        """The dialog is still open on a page that is still there."""
        return not self.closed and alive(self.dlg)

    def build(self) -> None:
        with ui.dialog().props("persistent") as self.dlg, ui.card().classes("rz-vast-dialog gap-3") \
                .style("width: 60rem; max-width: 96vw !important"):
            with ui.row().classes("w-full items-center gap-2 no-wrap"):
                ui.label(f"Send {self.job} to vast.ai").classes("rz-step-title")
                ui.space()
                self.credit_chip = ui.label("").classes("rz-chip rz-chip-text")
            self.form = ui.column().classes("w-full gap-3")
            self.progress = ui.column().classes("w-full gap-3")
            self.progress.visible = False
        with self.form:
            for w in self.warnings:
                ui.label(w).classes("rz-banner warning")
            if self.warnings:
                self.kit.hint("These do not stop the send; the run is not started on this PC's GPUs.")
            self.blocker_box = ui.column().classes("w-full gap-1")
            # 1. where
            self.section("1. Where to run", "cloud")
            self.where_box = ui.column().classes("w-full gap-2")
            with ui.row().classes("items-center gap-2"):
                self.rent_btn = self.vast.local_only(ui.button("Rent a new one", icon="add_shopping_cart",
                                                               on_click=self.rent_new, color=None)
                                                     .props("outline no-caps"))
                self.where_note = self.kit.hint("")
            # 2. what goes up
            self.section("2. What goes up", "cloud_upload")
            self.up_box = ui.column().classes("w-full gap-3")
            self.bucket_box = ui.column().classes("w-full gap-2 rz-vast-bucket")
            self.bucket_box.visible = False
            self.build_bucket()
            self.estimate = self.kit.hint("")
            self.changes_box = ui.column().classes("w-full gap-0")
            # 3. after the run
            self.section("3. After the run", "power_settings_new")
            self.after_box = ui.column().classes("w-full gap-2")
            # 4. cost
            self.section("4. Cost", "payments")
            self.cost_box = ui.element("div").classes("rz-summary")
            self.reasons = ui.column().classes("w-full gap-0")
            self.bg_note = ui.label("").classes("rz-hint")
            self.bg_note.visible = False
            self.error = ui.label("").classes("rz-banner error")
            self.error.visible = False
            with ui.row().classes("w-full items-center justify-end gap-2"):
                self.spin = ui.spinner(size="sm")
                self.spin.visible = False
                self.cancel_btn = ui.button("Cancel", on_click=lambda: self.dlg.submit(None), color=None) \
                    .props("flat no-caps")
                self.go_btn = ui.button("Send", icon="cloud_upload", on_click=self.go).props("unelevated no-caps")
        self.go_btn.set_enabled(False)
        self.paint_where()
        self.paint_after()
        self.refresh()

    def section(self, title: str, icon: str) -> None:
        with ui.row().classes("w-full items-center gap-2 rz-vast-sect"):
            ui.icon(icon).classes("rz-rs-icon")
            ui.label(title).classes("rz-subhead")

    # -- loading (everything that touches the disk or the network is off the page's thread)

    async def load(self) -> None:
        manager = await self.vast.manager()
        market = self.vast.services.get_market()
        if not self.live():
            return
        if manager is None or market is None:
            self.error.text = "Sending is not available in this install."
            self.error.visible = True
            return
        results = await asyncio.gather(
            run.io_bound(functools.partial(manager.preview, self.config, self.root, 1)),
            run.io_bound(manager.instances), run.io_bound(market.prefs), return_exceptions=True)
        if not self.live():
            return
        plan, insts, prefs = results
        self.prefs = prefs if isinstance(prefs, dict) else {}
        self.bucket = self.prefs.get("bucket") or ""
        self.after = after_default(self.prefs.get("after") or "stop", ours=True)
        self.idle = int(self.prefs.get("idle_minutes") or 60)
        if isinstance(plan, BaseException):
            self.error.text = sentence(plan, "The run's files could not be looked at.")
            self.error.visible = True
        else:
            self.plan = plan
            self.picks = {e["path"]: default_choice(e) for e in plan.files}
            self.paint_up()
        if isinstance(insts, BaseException):
            self.where_note.text = sentence(insts, "Your instances could not be listed.")
        else:
            self.instances = list(insts or [])
        self.paint_where()
        self.paint_after()
        self.refresh()

    # -- 1. where

    def selected_id(self):
        return self.target["row"]["id"] if self.target and self.target["kind"] == "instance" else None

    def paint_where(self) -> None:
        self.where_box.clear()
        with self.where_box:
            if self.target and self.target["kind"] == "rent":
                p = self.target["pick"]
                o = p["offer"]
                with ui.element("div").classes("rz-vast-pick on"):
                    ui.icon("radio_button_checked").classes("rz-rs-icon ok")
                    with ui.column().classes("gap-0 min-w-0 flex-1"):
                        ui.label(f"New rental: {_get(o, 'num_gpus')}x {_get(o, 'gpu_name')} · "
                                 f"{per_hour(offer_price(o, p.get('bid')))}").classes("rz-setting-title")
                        ui.label(f"{'Interruptible' if _get(o, 'type') == 'bid' else 'On-demand'} · "
                                 f"{p['disk']} GB disk · rented when you press the red button").classes("rz-hint")
                    ui.button("Change", color=None, on_click=self.rent_new).props("flat dense no-caps")
            for row in self.instances:
                ok = bool(row.get("selectable"))
                starts = ok and _is_stopped(row) and not self.vast.local  # starting one bills: only from this computer
                ok = ok and not starts
                on = self.selected_id() == row.get("id")
                with ui.element("div").classes("rz-vast-pick" + (" on" if on else "") + ("" if ok else " off")) as el:
                    ui.icon("radio_button_checked" if on else "radio_button_unchecked" if ok else "block") \
                        .classes("rz-rs-icon " + ("ok" if on else "muted"))
                    with ui.column().classes("gap-0 min-w-0 flex-1"):
                        ui.label(f"#{row['id']} · " + " · ".join(instance_row_text(row))).classes("rz-setting-title")
                        why = LOCAL_ONLY if starts else row.get("reason") or ("" if ok else "Not available")
                        if _is_stopped(row) and ok:
                            why = why or f"Stopped: starting it bills {per_hour(row.get('dph_total'))}."
                        if why:
                            ui.label(why).classes("rz-hint")
                if ok:
                    el.on("click", lambda _, r=row: self.pick_instance(r))
            if not self.instances and not self.target:
                ui.label("No instance of yours can take this run yet. Rent a new one.").classes("rz-hint")

    def pick_instance(self, row: dict) -> None:
        self.target = {"kind": "instance", "row": row}
        gpus = int(num(row.get("num_gpus")) or 1)
        if gpus != self.plan_gpus:
            background_tasks.create(self.replan(gpus), name="vast replan")
        if not self.after_touched:
            self.after = after_default(self.prefs.get("after") or "stop", ours=bool(row.get("ours")))
        self.paint_where()
        self.paint_after()
        self.refresh()

    async def replan(self, gpus: int) -> None:
        """The settings that change for the instance (the device names, multi-GPU) depend on its GPU count."""
        manager = await self.vast.manager()
        try:
            plan = await run.io_bound(functools.partial(manager.preview, self.config, self.root, gpus))
        except Exception:  # noqa: BLE001  (the first plan stands)
            return
        if not self.live():
            return
        self.plan_gpus = gpus
        keep = dict(self.picks)
        self.plan = plan
        self.picks = {e["path"]: keep.get(e["path"]) or default_choice(e) for e in plan.files}
        self.paint_up()
        self.refresh()

    async def rent_new(self) -> None:
        if not self.vast.guard_local():
            return
        uploads = sum(e["bytes"] for e in (self.plan.files if self.plan else [])) + \
            sum(d["bytes"] for d in (self.plan.datasets if self.plan else []))
        chosen: dict = {}
        with ui.dialog().props("maximized persistent") as mdlg, ui.card().classes("w-full gap-3 rz-vast-max"):
            with ui.row().classes("w-full items-center gap-2 no-wrap"):
                ui.label("Rent a new instance").classes("rz-step-title")
                ui.label("Pick an offer; it is rented when you press the red button of the send dialog.") \
                    .classes("rz-hint")
                ui.space()
                ui.label(self.credit_chip.text).classes("rz-chip rz-chip-text")
                ui.button("Close", icon="close", on_click=lambda: mdlg.submit(False), color=None) \
                    .props("outline no-caps")
            holder = ui.column().classes("w-full gap-3")
        view = MarketView(self.vast, pick=True, upload_bytes=uploads,
                          on_pick=lambda result: (chosen.update(result), mdlg.submit(True)))
        view.build(holder)
        mdlg.open()
        await mdlg
        mdlg.delete()
        if chosen and self.live():
            self.target = {"kind": "rent", "pick": dict(chosen)}
            self.after = after_default(self.prefs.get("after") or "stop", ours=True) if not self.after_touched \
                else self.after
            gpus = int(num(_get(chosen["offer"], "num_gpus")) or 1)
            if gpus != self.plan_gpus:
                await self.replan(gpus)
            self.paint_where()
            self.paint_after()
            self.refresh()

    # -- 2. what goes up

    def paint_up(self) -> None:
        plan = self.plan
        self.up_box.clear()
        self.changes_box.clear()
        self.blocker_box.clear()
        if plan is None:
            return
        with self.blocker_box:
            for b in plan.blockers:
                ui.label(f"{b.get('reason', 'A problem')}: {b.get('path') or b.get('key') or ''} "
                         f"({b.get('key')})").classes("rz-banner error")
        with self.up_box:
            if plan.datasets:
                ui.label("Datasets").classes("rz-setting-title")
                for d in plan.datasets:
                    files = d.get("files")
                    count = len(files) if isinstance(files, list) else files
                    ui.label(f"{d.get('name')} · {count} file{'s' if count != 1 else ''} · "
                             f"{size_text(d.get('bytes'))}").classes("rz-hint")
            if plan.files:
                ui.label("Model files that are only on this PC").classes("rz-setting-title")
                self.kit.hint("Choose for each: upload it to the instance, send it through your own Hugging Face "
                              "bucket, or use the standard file the instance downloads itself.")
            for e in plan.files:
                with ui.element("div").classes("rz-vast-file"):
                    with ui.column().classes("gap-0 min-w-0"):
                        ui.label(Path(e["path"]).name).classes("rz-setting-title")
                        ui.label(f"{e['path']} · {size_text(e.get('bytes'))}"
                                 + (" · folder" if e.get("kind") == "folder" else "")).classes("rz-hint rz-vast-path")
                        if e.get("standard"):
                            ui.label("Standard file: " + standard_text(e)).classes("rz-hint")
                    sel = ui.select({k: label for k, label in file_options(e)}, value=self.picks.get(e["path"]),
                                    label="How to send").props("dense outlined").classes("rz-vast-how")
                    sel.on_value_change(lambda ev, p=e["path"]: self.pick_how(p, ev.value))
            if not plan.datasets and not plan.files:
                ui.label("Nothing has to go up: the instance has everything this run reads.").classes("rz-hint")
        with self.changes_box:
            if plan.changes:
                ui.label("Settings that change for the instance").classes("rz-setting-title")
                for c in plan.changes:
                    ui.label(f"{c.get('key')}: {c.get('old') or '(empty)'} becomes {c.get('new') or '(empty)'} "
                             f"({c.get('reason')})").classes("rz-hint")

    def pick_how(self, path: str, how) -> None:
        self.picks[path] = how
        self.refresh()
        if how == "bucket":
            background_tasks.create(self.load_bucket_info(), name="vast bucket info")

    # the Hugging Face bucket

    def build_bucket(self) -> None:
        with self.bucket_box:
            ui.label("Hugging Face bucket").classes("rz-setting-title")
            self.kit.hint("The bucket holds the files you chose to send that way; the instance downloads them from "
                          "there. Your Hugging Face token stays on this PC.")
            self.hf_note = self.kit.hint("")
            with ui.row().classes("items-center gap-2 no-wrap") as self.token_row:
                self.token_input = ui.input("Hugging Face access token (write access)", password=True,
                                            password_toggle_button=True).props("dense outlined").classes("w-80")
                ui.button("Sign in", icon="login", on_click=self.sign_in, color=None).props("outline dense no-caps")
            self.token_row.visible = False
            with ui.row().classes("items-center gap-2 w-full"):
                self.bucket_select = ui.select({}, label="Bucket (owner/name)", with_input=True,
                                               new_value_mode="add-unique").props("dense outlined") \
                    .classes("w-96 max-w-full")
                self.bucket_select.on_value_change(self.bucket_changed)
                ui.button("Use this bucket", icon="check", on_click=self.confirm_bucket, color=None) \
                    .props("outline dense no-caps")
            with ui.row().classes("items-center gap-2"):
                self.create_btn = ui.button("Create it (private)", icon="add", on_click=self.create_bucket,
                                            color=None).props("outline dense no-caps")
                self.create_btn.visible = False
                self.save_default = ui.switch("Save as my default bucket", value=True)
            self.bucket_status = ui.label("").classes("rz-hint")

    async def load_bucket_info(self) -> None:
        files = self.vast.services.module("rulz_files")
        if files is None or self.bucket_info["loaded"]:
            return
        self.bucket_info["loaded"] = True
        token = await run.io_bound(files.hf_token)
        if not self.live():
            return
        self.bucket_info["token"] = bool(token)
        self.token_row.visible = not token
        self.hf_note.text = "" if token else "Not signed in to Hugging Face. Paste an access token to continue."
        options = {}
        if token:
            try:
                rows = await run.io_bound(files.buckets)
                options = {r["id"]: r["id"] for r in rows if r.get("id")}
            except Exception as e:  # noqa: BLE001
                if self.live():
                    self.hf_note.text = sentence(e, "Your buckets could not be listed.")
            if not self.live():
                return
        if self.bucket and self.bucket not in options:
            options[self.bucket] = self.bucket
        self.bucket_select.set_options(options)
        if self.bucket:
            self.bucket_select.value = self.bucket
        saved = self.prefs.get("bucket") or ""
        self.save_default.value = not saved
        self.bucket_status.text = (f"Your default bucket is {saved}. Press Use this bucket to confirm it, or pick "
                                   "another." if saved else "No default bucket yet: pick one of your buckets or type "
                                   "owner/name, then create it if it does not exist.")

    def bucket_changed(self, e) -> None:
        self.bucket = e.value or ""
        self.bucket_ok = False
        self.create_btn.visible = False
        self.refresh()

    async def sign_in(self) -> None:
        if not self.vast.guard_local():  # signing in changes what this PC may do with the account's files
            return
        files = self.vast.services.module("rulz_files")
        token, self.token_input.value = self.token_input.value, ""  # the field is cleared at once
        try:
            who = await run.io_bound(functools.partial(files.hf_login, token))
        except Exception as e:  # noqa: BLE001
            if self.live():
                self.hf_note.text = sentence(e, "Hugging Face did not accept that token.")
            return
        finally:
            token = ""
        if not self.live():
            return
        self.bucket_info["loaded"] = False
        notify_page(f"Signed in to Hugging Face{' as ' + who['user'] if who.get('user') else ''}.", type="positive")
        await self.load_bucket_info()

    async def confirm_bucket(self) -> None:
        files = self.vast.services.module("rulz_files")
        bucket = (self.bucket or "").strip()
        if not bucket:
            self.bucket_status.text = "Pick a bucket or type owner/name first."
            return
        try:
            state = await run.io_bound(functools.partial(files.bucket_state, bucket))
        except Exception as e:  # noqa: BLE001
            if self.live():
                self.bucket_status.text = sentence(e, "That bucket name is not valid.")
            return
        if not self.live():
            return
        if state.get("error"):
            self.bucket_status.text = f"Hugging Face could not tell if {bucket} exists ({state['error']})."
            return
        if not state.get("exists"):
            self.bucket_status.text = f"{bucket} does not exist yet."
            self.create_btn.visible = True
            return
        self.bucket_ok = True
        warn = " It is a public bucket: anyone with the link can download your model files." \
            if state.get("private") is False else ""
        self.bucket_status.text = f"Using {bucket}.{warn}"
        if self.save_default.value:
            try:
                await run.io_bound(functools.partial(self.vast.services.get_market().set_prefs, bucket=bucket))
                self.prefs["bucket"] = bucket
            except Exception:  # noqa: BLE001  (not remembered; this send still uses it)
                pass
        if self.live():
            self.refresh()

    async def create_bucket(self) -> None:
        if not self.vast.guard_local():
            return
        files = self.vast.services.module("rulz_files")
        try:
            await run.io_bound(functools.partial(files.create_bucket, self.bucket, True))
        except Exception as e:  # noqa: BLE001
            if self.live():
                self.bucket_status.text = sentence(e, "The bucket could not be created.")
            return
        if not self.live():
            return
        self.create_btn.visible = False
        notify_page(f"Created the private bucket {self.bucket}.", type="positive")
        await self.confirm_bucket()

    # -- 3. after the run

    def ours(self) -> bool:
        if self.target is None:
            return True
        return self.target["kind"] == "rent" or bool(self.target["row"].get("ours"))

    def paint_after(self) -> None:
        self.after_box.clear()
        ours = self.ours()
        if self.after == "destroy" and not ours:
            self.after = "leave"
        if self.after == "destroy" and not self.vast.local:
            self.after = "stop"
        with self.after_box:
            radio = ui.radio({k: label for k, label in after_options(ours=ours)}, value=self.after) \
                .props("inline")
            radio.on_value_change(self.set_after)
            gpu_ph, disk_ph = self.prices()
            self.after_label = self.kit.hint(after_text(self.after, disk_ph, gpu_ph))
            if not ours and self.target is not None:
                self.kit.hint("This instance was not rented by this app, so it is left running unless you pick Stop; "
                              "Destroy is not offered.")
            with ui.row().classes("items-center gap-2") as self.idle_row:
                self.idle_input = ui.number("Stop it after this many idle minutes", value=self.idle, min=10, max=1440,
                                            step=10, precision=0).props("dense outlined").classes("w-72")
                self.idle_input.on_value_change(lambda e: setattr(self, "idle", int(e.value or 60)))
                self.kit.hint("A safety stop that runs on the instance itself, in case this PC is off.")
            self.idle_row.visible = self.after != "leave"
            row = self.target["row"] if self.target and self.target["kind"] == "instance" else None
            if row and row.get("version") and row["version"] != getattr(self.kit, "version", row["version"]):
                self.update_sw = ui.switch(f"Update RulZ AI Trainer on the instance first ({row['version']} there, "
                                           f"{self.kit.version} here)", value=False)
            else:
                self.update_sw = None

    def set_after(self, e) -> None:
        if e.value == "destroy" and not self.vast.local:  # destroying is only chosen from a browser on this PC
            notify_page(LOCAL_ONLY, type="warning")
            e.sender.value = self.after
            return
        self.after, self.after_touched = e.value, True
        gpu_ph, disk_ph = self.prices()
        self.after_label.text = after_text(self.after, disk_ph, gpu_ph)
        self.idle_row.visible = self.after != "leave"
        self.refresh()

    # -- 4. cost and the button

    def prices(self) -> tuple[float | None, float | None]:
        """(what the instance bills per hour while it runs, what its disk bills once stopped)."""
        t = self.target
        if t is None:
            return None, None
        if t["kind"] == "rent":
            o, p = t["pick"]["offer"], t["pick"]
            return offer_price(o, p.get("bid")), num(_get(o, "storage_dph"))
        row = t["row"]
        return num(row.get("dph_total")), num(row.get("storage_per_hour"))

    def extra_per_hour(self) -> float:
        """What this send adds to the hourly burn: all of it for a new rental, the GPU part for a stopped instance
        (its disk bills already), nothing for one that is running."""
        t = self.target
        if t is None:
            return 0.0
        gpu, disk = self.prices()
        if t["kind"] == "rent":
            return gpu or 0.0
        return max(0.0, (gpu or 0.0) - (disk or 0.0)) if _is_stopped(t["row"]) else 0.0

    def button_text(self) -> str:
        t = self.target
        if t is None:
            return "Send"
        if t["kind"] == "rent":
            return f"Rent for {per_hour(self.prices()[0])} and send"
        if _is_stopped(t["row"]):
            return f"Start for {per_hour(t['row'].get('dph_total'))} and send"
        return "Send"

    def refresh(self) -> None:
        snap = self.vast.watch.snapshot()
        self.credit_chip.text = chip_view(snap)["text"] if snap.get("account") else "credit unknown"
        ready, why = send_state(self.plan, self.picks, bucket=self.bucket, bucket_ok=self.bucket_ok,
                                target=self.target, after=self.after, destroy_allowed=self.ours(), busy=self.busy)
        self.bucket_box.visible = needs_bucket(self.picks)
        if self.bucket_box.visible and not self.bucket_info["loaded"]:
            background_tasks.create(self.load_bucket_info(), name="vast bucket info")
        if self.plan is not None:
            up = sum(d["bytes"] for d in self.plan.datasets) + sum(
                e["bytes"] for e in self.plan.files if self.picks.get(e["path"]) in ("upload", "bucket"))
            mbps = ASSUMED_UPLOAD_MBPS
            if self.target and self.target["kind"] == "rent":
                mbps = min(mbps, num(_get(self.target["pick"]["offer"], "inet_down")) or mbps)
            self.estimate.text = upload_text(up, mbps) if up else ""
        credit = (snap.get("account") or {}).get("credit")
        burn = ((snap.get("account") or {}).get("burn") or {}).get("per_hour") or 0.0
        run_hours = None
        step_fn = getattr(self.kit, "step_seconds", None)
        if step_fn:
            try:
                seconds = step_fn(self.config)
                steps = num(self.config.get("steps"))
                run_hours = seconds * steps / 3600 if seconds and steps else None
            except Exception:  # noqa: BLE001
                run_hours = None
        self.cost_box.clear()
        with self.cost_box:
            for k, v in cost_lines(credit, burn, self.extra_per_hour(), run_hours):
                ui.label(k).classes("k")
                ui.label(v).classes("v")
        self.reasons.clear()
        with self.reasons:
            for line in ([] if ready else [w for w in why if w != "Sending."]):
                ui.label(line).classes("rz-hint rz-vast-why")
        self.go_btn.text = self.button_text()
        self.go_btn.set_enabled(ready)

    # -- sending

    async def go(self) -> None:
        if self.busy:
            return
        t = self.target
        manager = await self.vast.manager()
        if not self.live() or manager is None or t is None:
            return
        starts = t["kind"] == "instance" and _is_stopped(t["row"])
        if (t["kind"] == "rent" or starts) and not self.vast.guard_local():
            self.error.text = LOCAL_ONLY
            self.error.visible = True
            return
        self.busy = True
        self.spin.visible = True
        self.error.visible = False
        self.cancel_btn.text = "Close"  # once it is on its way, Close does not stop it
        self.bg_note.text = "The send goes on in the background when you close this."
        self.bg_note.visible = True
        self.refresh()
        shown = None
        if starts:  # what bills is what was shown: read the instance again right before the start
            try:
                fresh = await run.io_bound(manager.instances)
            except Exception as e:  # noqa: BLE001
                fresh, why = None, sentence(e, "The instance could not be read again.")
            else:
                why = ""
            if not self.live():
                return
            row = next((r for r in fresh or [] if r.get("id") == t["row"]["id"]), None)
            change = why or instance_changed(t["row"], row)
            if change:
                if fresh is not None:
                    self.instances = list(fresh)
                    self.target = {"kind": "instance", "row": row} if row and row.get("selectable") else None
                self.paint_where()
                self.paint_after()
                self.error.text = f"{change} Nothing was started: check it and press the button again."
                self.error.visible = True
                self.busy = False
                self.spin.visible = False
                self.cancel_btn.text = "Cancel"
                self.bg_note.visible = False
                self.refresh()
                return
            shown = num(row.get("dph_total"))
        kw = {"choices": choices_for(self.plan, self.picks, self.bucket), "after": self.after,
              "idle_minutes": self.idle, "update": bool(self.update_sw.value) if self.update_sw else False}
        if t["kind"] == "rent":
            p = t["pick"]
            kw["rent"] = {"offer": p["offer"], "filters": p["filters"], "template": p["template"], "disk": p["disk"],
                          "bid": p.get("bid")}
        else:
            kw["instance"] = t["row"]["id"]
            kw["start"] = _is_stopped(t["row"])
            if shown is not None and accepts(manager.create, "shown"):
                kw["shown"] = shown  # the manager refuses a start whose price rose more than 0.5%
        try:
            await run.io_bound(functools.partial(self.vast.services.get_market().set_prefs, after=self.after,
                                                 idle_minutes=self.idle))
        except Exception:  # noqa: BLE001  (not remembered)
            pass
        send, pending = None, ""
        try:
            send = await run.io_bound(functools.partial(manager.create, self.job, self.config, self.root, **kw))
        except Exception as e:  # noqa: BLE001
            if getattr(e, "kind", "") == "pending":
                pending = sentence(e)
            else:
                text = sentence(e, "The send could not be made.")
                if not self.live():  # closed meanwhile: the result goes to the page
                    notify_page(f"{self.job}: {text}", type="warning", multi_line=True)
                    return
                self.error.text = text
                self.error.visible = True
                self.busy = False
                self.spin.visible = False
                self.cancel_btn.text = "Cancel"
                self.bg_note.visible = False
                self.refresh()
                return
        self.vast.watch.request()
        if not self.live():
            notify_page(pending or f"{self.job} was sent to vast.ai. It goes on in the background: see the Fleet tab.",
                        type="warning" if pending else "positive", multi_line=True)
            return
        self.show_progress(send, pending)

    def show_progress(self, send, pending: str = "") -> None:
        self.form.visible = False
        self.progress.visible = True
        self.send_id = (send or {}).get("id")
        placeholder = send or {"id": None, "job": self.job, "state": "waiting", "steps": [], "warnings": [pending]
                               if pending else []}

        def buttons() -> None:
            ui.button("Close", icon="close", on_click=lambda: self.dlg.submit(True), color=None) \
                .props("outline no-caps dense").tooltip("The send goes on in the background; its progress is on the "
                                                       "Fleet tab.")
            ui.button("Show on the Fleet tab", icon="dns", color=None,
                      on_click=lambda: (self.dlg.submit(True), self.kit.goto("Fleet"))).props("flat no-caps dense")

        self.view = SendView(self.vast, self.progress, placeholder, extra=buttons)
        self.timer = ui.timer(2.0, self.poll)

    async def poll(self) -> None:
        manager = self.vast.services.manager
        if manager is None or not self.live():
            return
        try:
            if self.send_id is None:  # the rent's answer was lost: the send waits for it; find it by its job
                found = [s for s in await run.io_bound(manager.sends) if s.get("job") == self.job]
                send = found[0] if found else None
                if send is not None:
                    self.send_id = send.get("id")
            else:
                send = await run.io_bound(manager.get, self.send_id)
        except Exception:  # noqa: BLE001
            return
        if send and self.live():
            self.view.update(send)


# ---------------------------------------------------------------- the background loops


def sends_loop(services: Services = SERVICES, every: float = 3.0, sleep=time.sleep, rounds: int | None = None) -> None:
    """The SendManager's beat (settle open rents, move the sends on): every 3 s beside the fleet's loop. `step()` does
    nothing unless this process holds the sends (another WebUI or a daemon may). A missing module only slows the beat."""
    n = 0
    while rounds is None or n < rounds:
        n += 1
        wait = every
        try:
            manager = services.get_manager()
            if manager is None:
                wait = 30.0
            else:
                hold = getattr(manager, "hold", None)
                if callable(hold):
                    hold()
                manager.step()
        except Exception as e:  # noqa: BLE001  (the loop must outlive one bad beat)
            print(f"[vast sends] {type(e).__name__}: {e}", flush=True)
        sleep(wait)


def start_loops(enabled: bool = True) -> None:
    """The account watch and the sends loop, as daemon threads (webui's app.on_startup; not on a vast.ai instance)."""
    if not enabled:
        return
    try:
        WATCH.start(True)
        threading.Thread(target=sends_loop, name="rulz-vast-sends", daemon=True).start()
    except Exception as e:  # noqa: BLE001  (this must never stop the WebUI from starting)
        print(f"[vast] could not start: {type(e).__name__}: {e}", flush=True)
