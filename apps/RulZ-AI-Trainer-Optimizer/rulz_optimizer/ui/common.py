"""Shared UI pieces: the page frame and tabs, the per-page state, confidence chips, safe file serving, helpers."""

from __future__ import annotations

import hashlib
import logging
import subprocess
import sys
import threading
import urllib.parse
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Callable, Iterator

from nicegui import app, run, ui
from starlette.responses import FileResponse, Response

from .. import APP_NAME, __version__, bridge, settings
from .theme import LOGO_DARK, LOGO_LIGHT, PageLook

log = logging.getLogger("rulz_optimizer")

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}

CONF_HELP = {
    "CONF": "Confirmed by several sources: two or more independent sources agree, or numbers / an A-B comparison were shown.",
    "MEAS": "Measured in our runs: seen in our own RulZ AI Trainer runs.",
    "DOC": "Documented: from the official model card or the trainer's documentation.",
    "REP": "Reported once: one source says so.",
    "DISP": "Disputed: sources disagree. See the reason for both sides.",
    "GUESS": "Guess, no evidence: a reasoned default. Treat it as a guess.",
}
# The word and the icon a chip shows, so the rating never depends on colour alone.
CONF_LOOK = {
    "CONF": ("Confirmed", "check_circle"),
    "MEAS": ("Measured", "straighten"),
    "DOC": ("Documented", "menu_book"),
    "REP": ("Reported", "chat_bubble_outline"),
    "DISP": ("Disputed", "report_problem"),
    "GUESS": ("Guess", "help_outline"),
}

# ---------------------------------------------------------------------------------------------------------------------
# Knowledge (loaded once, shared by every tab of the browser)
# ---------------------------------------------------------------------------------------------------------------------

_knowledge_lock = threading.Lock()
_knowledge: tuple[Any, str | None] | None = None
_knowledge_problems: list[str] = []
WIZARD_SOURCE = "RulZ AI Trainer setup wizard (webui.py)"


def get_knowledge(reload: bool = False) -> tuple[Any, str | None]:
    """(Knowledge or None, error text or None). A missing or broken knowledge folder gives (None, reason)."""
    global _knowledge
    with _knowledge_lock:
        if _knowledge is None or reload:
            try:
                from .. import knowledge
                root = knowledge.default_dir()
                if not Path(root).is_dir():
                    _knowledge = (None, f"Knowledge base not found (looked in {root}).")
                else:
                    loaded = knowledge.load()
                    _knowledge = (loaded, None)
                    from .. import trainer_keys
                    _knowledge_problems[:] = knowledge.validate(loaded, set(trainer_keys.KEYS))
                    for problem in _knowledge_problems:
                        log.warning("knowledge: %s", problem)
            except Exception as e:  # noqa: BLE001  (KnowledgeError names the file and key)
                _knowledge = (None, f"Knowledge base not found or invalid: {e}")
        return _knowledge


def knowledge_problems() -> list[str]:
    """What knowledge.validate() found in the loaded knowledge base (empty when it is clean)."""
    return list(_knowledge_problems)


def arch_ids() -> list[str]:
    try:
        from .. import trainer_keys
        return list(trainer_keys.ARCH_NAMES)
    except Exception:  # noqa: BLE001
        return []


# ---------------------------------------------------------------------------------------------------------------------
# Per-page state
# ---------------------------------------------------------------------------------------------------------------------

@dataclass
class AppState:
    """What the tabs of one browser page share, so the Ask AI tab can use the recommendation and the run analysis."""

    model_id: str | None = None
    scan: Any = None
    rec: Any = None
    goals: Any = None
    run: Any = None
    findings: list = field(default_factory=list)
    proposals: list = field(default_factory=list)
    folders: list[Path] = field(default_factory=list)
    tabs: Any = None
    look: Any = None  # the PageLook of this browser tab (charts follow it)
    on_context_change: list[Callable[[], None]] = field(default_factory=list)

    def changed(self) -> None:
        for fn in self.on_context_change:
            try:
                fn()
            except Exception:  # noqa: BLE001
                log.exception("context listener failed")

    def goto(self, tab: str) -> None:
        if self.tabs is not None:
            self.tabs.set_value(tab)


# ---------------------------------------------------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------------------------------------------------

def conf_chip(conf: str | None) -> ui.element:
    conf = (conf or "GUESS").upper()
    word, icon = CONF_LOOK.get(conf, (conf.capitalize(), "help_outline"))
    chip = ui.element("span").classes(f"rz-conf {conf}").props(f'aria-label="{CONF_HELP.get(conf, word)}"')
    with chip:
        ui.icon(icon)
        ui.label(word)
        ui.tooltip(CONF_HELP.get(conf, word))
    return chip


# Buttons: one primary (red) per view, secondary (panel-2 with a border) for the other actions, ghost for quiet ones.
def _button(text: str, icon: str | None, on_click: Callable | None, kind: str, small: bool) -> ui.button:
    btn = ui.button(text, icon=icon, on_click=on_click, color=None).props("unelevated no-caps" if kind == "primary"
                                                                          else "flat no-caps")
    btn.classes(f"rz-{kind}" + (" rz-sm" if small else ""))
    return btn


def primary_button(text: str, icon: str | None = None, on_click: Callable | None = None) -> ui.button:
    return _button(text, icon, on_click, "primary", False)


def secondary_button(text: str, icon: str | None = None, on_click: Callable | None = None,
                     small: bool = False) -> ui.button:
    return _button(text, icon, on_click, "secondary", small)


def ghost_button(text: str, icon: str | None = None, on_click: Callable | None = None,
                 small: bool = False) -> ui.button:
    return _button(text, icon, on_click, "ghost", small)


def promote(btn: ui.button, primary_btn: bool) -> None:
    """Make `btn` the view's one primary button, or hand the red back (it becomes a secondary one)."""
    if primary_btn:
        btn.props(add="unelevated", remove="flat")
        btn.classes(add="rz-primary", remove="rz-secondary")
    else:
        btn.props(add="flat", remove="unelevated")
        btn.classes(add="rz-secondary", remove="rz-primary")


def clean_path(text: str) -> str:
    return (text or "").strip().strip('"').strip("'")


def parse_folders(text: str) -> list[Path]:
    return [Path(clean_path(line)).expanduser() for line in (text or "").splitlines() if clean_path(line)]


def get_trainer() -> bridge.TrainerInstall | None:
    hint = settings.get().trainer_dir
    return bridge.find_trainer(Path(hint) if hint else None)


def export_dir() -> Path:
    """Where configs are saved: the setting, else the found trainer's configs folder, else Documents."""
    s = settings.get()
    if s.export_dir:
        return s.effective_export_dir()
    trainer = get_trainer()
    return trainer.root / "configs" if trainer else s.effective_export_dir()


def fmt_num(v: Any, digits: int = 3) -> str:
    if v is None:
        return "-"
    try:
        return f"{float(v):.{digits}f}"
    except (TypeError, ValueError):
        return str(v)


def fmt_value(v: Any) -> str:
    text = str(v)
    return text if len(text) <= 160 else text[:157] + "..."


def detect_vram_gb() -> float | None:
    """Total memory of the largest NVIDIA GPU, from nvidia-smi (None when there is none)."""
    try:
        done = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=10, stdin=subprocess.DEVNULL)
        values = [float(x) for x in done.stdout.split() if x.strip().replace(".", "", 1).isdigit()]
        return round(max(values) / 1024, 1) if values else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def export_header(model_name: str, trainer_version: str | None) -> str:
    return "\n".join([
        f"made by {APP_NAME} {__version__}",
        f"model: {model_name}",
        f"RulZ AI Trainer version: {trainer_version or 'unknown'}",
        f"date: {date.today().isoformat()}",
        "Load it in the RulZ AI Trainer WebUI with 'Config file' -> Load.",
    ])


def banner(text: str, icon: str = "info", kind: str = "info") -> ui.row:
    row = ui.row().classes(f"rz-banner {kind} items-center gap-2 w-full no-wrap")
    with row:
        ui.icon(icon)
        ui.label(text).classes("min-w-0")
    return row


class Output:
    """A message area under a button row: the last result of an action, in plain text."""

    def __init__(self) -> None:
        self.label = ui.label("").classes("rz-pre w-full")
        self.label.set_visibility(False)

    def show(self, text: str, ok: bool | None = None) -> None:
        self.label.set_text(text)
        self.label.classes(remove="rz-ok rz-err rz-warn")
        if ok is True:
            self.label.classes(add="rz-ok")
        elif ok is False:
            self.label.classes(add="rz-err")
        self.label.set_visibility(bool(text))


async def io(fn: Callable, *args: Any, **kwargs: Any) -> Any:
    """Run a blocking function off the UI thread."""
    return await run.io_bound(fn, *args, **kwargs)


# ---------------------------------------------------------------------------------------------------------------------
# Files: only the known run folders are ever served
# ---------------------------------------------------------------------------------------------------------------------

_run_roots: dict[str, Path] = {}


def run_token(run_path: Path) -> str:
    resolved = Path(run_path).resolve()
    token = hashlib.sha1(str(resolved).lower().encode("utf-8")).hexdigest()[:12]
    _run_roots[token] = resolved
    return token


def probe_url(run_path: Path, image: Path) -> str | None:
    """URL for an image inside a registered run folder, else None."""
    token = run_token(run_path)
    try:
        rel = Path(image).resolve().relative_to(_run_roots[token])
    except ValueError:
        return None
    if rel.suffix.lower() not in IMAGE_SUFFIXES:
        return None
    return f"/run-file/{token}/{urllib.parse.quote(rel.as_posix())}"  # '#', '?', '%' in names stay in the path


def resolve_run_file(token: str, rel: str) -> Path | None:
    """The file for a /run-file/ URL, or None unless it is an image inside a registered run folder."""
    root = _run_roots.get(token)
    if root is None:
        return None
    try:
        target = (root / rel).resolve()
        target.relative_to(root)
    except (ValueError, OSError):
        return None
    if target.suffix.lower() not in IMAGE_SUFFIXES or not target.is_file():
        return None
    return target


_routes_done = False


def register_routes() -> None:
    global _routes_done
    if _routes_done:
        return
    _routes_done = True

    @app.get("/run-file/{token}/{rel:path}")
    def run_file(token: str, rel: str):
        target = resolve_run_file(token, rel)
        if target is None:
            return Response(status_code=404)
        return FileResponse(target)


# ---------------------------------------------------------------------------------------------------------------------
# Page frame
# ---------------------------------------------------------------------------------------------------------------------

TABS = (("recommend", "Recommend", "tune"), ("runs", "Runs", "show_chart"), ("advisor", "Ask AI", "psychology"),
        ("knowledge", "Knowledge", "menu_book"), ("settings", "Settings", "settings"))


@contextmanager
def frame() -> Iterator[PageLook]:
    look = PageLook()
    ui.page_title(APP_NAME)
    with ui.header().classes("rz-header"):
        with ui.row().classes("rz-header-row items-center gap-3 no-wrap"):
            ui.element("img").props(f'src="{LOGO_DARK}" alt="RulZ"').classes("rz-logo rz-dark-only")
            ui.element("img").props(f'src="{LOGO_LIGHT}" alt="RulZ"').classes("rz-logo rz-light-only")
            ui.label("AI Trainer Optimizer").classes("rz-wordmark gt-xs")
            ui.space()
            ui.label(f"v{__version__}").classes("text-xs rulz-muted tabular-nums")
            with ui.button(on_click=look.toggle, color=None).props("flat round").classes("rz-icon-btn"):
                ui.icon("light_mode").classes("rz-dark-only")
                ui.icon("dark_mode").classes("rz-light-only")
                ui.tooltip("Switch between the dark and the light look")
    with ui.column().classes("w-full gap-3 mx-auto p-8").style("max-width: 1280px"):
        yield look


def build_main_page() -> None:
    from . import advisor_page, knowledge_page, recommend_page, runs_page, settings_page

    state = AppState()
    with frame() as look:
        state.look = look
        if not settings.get().look_chosen:
            from .theme import choose_look
            ui.timer(0.3, lambda: choose_look(look), once=True)
        with ui.tabs().classes("rz-tabs w-full").props("align=left dense no-caps") as tabs:
            for key, label, icon in TABS:
                ui.tab(key, label=label, icon=icon)
        state.tabs = tabs
        with ui.tab_panels(tabs, value="recommend", animated=False).classes("w-full").style("background: transparent"):
            builders = {"recommend": recommend_page.build, "runs": runs_page.build, "advisor": advisor_page.build,
                        "knowledge": knowledge_page.build, "settings": settings_page.build}
            for key, _label, _icon in TABS:
                with ui.tab_panel(key).classes("p-0 pt-3 gap-3"):
                    builders[key](state)


def section(title: str, icon: str, subtitle: str = "") -> ui.card:
    """A titled card; use as a context manager."""
    card = ui.card().classes("rulz-card w-full gap-3 p-3.5")
    with card:
        with ui.row().classes("items-center gap-3 no-wrap w-full"):
            ui.icon(icon).classes("rz-sec-icon")
            with ui.column().classes("gap-0 min-w-0"):
                ui.label(title).classes("rz-title")
                if subtitle:
                    ui.label(subtitle).classes("rz-tagline")
    return card


def settings_table(rows: list[tuple[str, str, str, str, str]]) -> None:
    """key / value / why / confidence / sources rows as a table made of plain elements (no HTML injection)."""
    with ui.element("table").classes("rz-set-table"):
        with ui.element("thead"):
            with ui.element("tr"):
                for head in ("Setting", "Value", "Why", "Confidence", "Sources"):
                    with ui.element("th"):
                        ui.label(head)
        with ui.element("tbody"):
            for key, value, why, conf, sources in rows:
                with ui.element("tr"):
                    for css, text in (("key", key), ("val", value), ("why", why)):
                        with ui.element("td").classes(css):
                            ui.label(text)
                    with ui.element("td"):
                        conf_chip(conf)
                    with ui.element("td").classes("src"):
                        ui.label(sources)


def python_is_windows() -> bool:
    return sys.platform.startswith("win")
