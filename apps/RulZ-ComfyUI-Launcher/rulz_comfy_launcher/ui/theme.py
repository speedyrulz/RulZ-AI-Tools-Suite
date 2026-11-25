"""The RulZ AI Tools look: dark (default) and light, applied live, with the logo and icons.

Every colour is a token of the RulZ AI Tools design guide (name and exact value). Nothing
here keeps a colour in a module constant that was read at import: code asks the active
palette (``colour(token)``, ``tone_colour(tone)``, ``C``) when it paints or styles, so a
switch with ``apply_theme(app, mode)`` reaches every widget. Sizes are in points
(14 px = 10.5 pt).
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from PySide6.QtCore import QByteArray, QEvent, QObject, QStandardPaths, Qt
from PySide6.QtGui import QColor, QFont, QFontDatabase, QIcon, QPainter, QPalette, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QAbstractButton, QApplication, QDialogButtonBox, QWidget

from ..instance import State

ASSETS = Path(__file__).resolve().parent.parent / "assets"

DARK = "dark"
LIGHT = "light"
MODES = (DARK, LIGHT)

# name -> (dark, light), exactly as in the design guide's tokens.json.
TOKENS: dict[str, tuple[str, str]] = {
    "bg": ("#050505", "#ffffff"),
    "panel": ("#0c0c0e", "#f5f5f7"),
    "panel-2": ("#141417", "#ebebef"),
    "hover": ("#1c1c21", "#e2e2e7"),
    "border": ("#232328", "#d9d9de"),
    "border-strong": ("#34343b", "#c4c4cb"),
    "text": ("#f2f2f3", "#111114"),
    "body": ("#d4d4d8", "#2a2a30"),
    "muted": ("#a3a3ab", "#5f6068"),
    "faint": ("#82828c", "#68696f"),
    "accent": ("#ed000d", "#ed000d"),
    "accent-hover": ("#ff2633", "#c8000b"),
    "accent-soft": ("rgba(237, 0, 13, 0.13)", "rgba(237, 0, 13, 0.08)"),
    "accent-fg": ("#ffffff", "#ffffff"),
    "accent-text": ("#ff6b72", "#c8000b"),
    "link": ("#ff5961", "#d0000b"),
    "danger": ("#ff5a5f", "#d4161c"),
    "danger-text": ("#ff8a8e", "#b4000a"),
    "danger-bg": ("#3a0c0f", "#fdecec"),
    "danger-border": ("#6b1418", "#f5b8bb"),
    "danger-hover": ("#4a1014", "#fbd8da"),
    "warn": ("#fbbf24", "#a16207"),
    "warn-text": ("#fcd679", "#8a5300"),
    "ok": ("#4ade80", "#15803d"),
    "banner-bg": ("rgba(42, 7, 9, 0.8)", "#fdecec"),
    "banner-text": ("#ffb3b6", "#9f000a"),
    "code-bg": ("#0a0a0c", "#f5f5f7"),
    "sunken": ("rgba(0, 0, 0, 0.4)", "rgba(0, 0, 0, 0.03)"),
    "subtle": ("rgba(255, 255, 255, 0.06)", "rgba(0, 0, 0, 0.04)"),
    "subtle-border": ("rgba(255, 255, 255, 0.08)", "rgba(0, 0, 0, 0.08)"),
    "track": ("rgba(255, 255, 255, 0.08)", "rgba(0, 0, 0, 0.08)"),
    "media-bg": ("#000000", "#ececef"),
    "scrim": ("rgba(0, 0, 0, 0.7)", "rgba(0, 0, 0, 0.35)"),
    "shadow": ("rgba(0, 0, 0, 0.9)", "rgba(0, 0, 0, 0.18)"),
}

# Sizes from the guide (px): fixed control heights, radii.
CONTROL_SM, CONTROL_MD, CONTROL_LG = 32, 36, 44
RADIUS_SM, RADIUS_MD, RADIUS_LG, RADIUS_XL = 4, 6, 8, 12
DISABLED_OPACITY = 0.45

# What an instance is doing, as a tone (not a colour) so the colour always comes from the
# active palette. A status always has a word and a shape as well as a colour.
TONE_STOPPED = "stopped"      # faint ring
TONE_BUSY = "busy"            # warn (Queued, Starting, Stopping)
TONE_OK = "ok"                # ok (Running)
TONE_FAILED = "failed"        # danger (Failed)
TONE_EXTERNAL = "external"    # warn triangle: informational, the guide has no token for it
STATE_TONES = {
    State.STOPPED: TONE_STOPPED,
    State.QUEUED: TONE_BUSY,
    State.STARTING: TONE_BUSY,
    State.RUNNING: TONE_OK,
    State.STOPPING: TONE_BUSY,
    State.FAILED: TONE_FAILED,
}
_TONE_TOKEN = {TONE_STOPPED: "muted", TONE_BUSY: "warn", TONE_OK: "ok", TONE_FAILED: "danger",
               TONE_EXTERNAL: "warn"}
_TONE_TEXT_TOKEN = {TONE_STOPPED: "muted", TONE_BUSY: "warn", TONE_OK: "ok", TONE_FAILED: "danger-text",
                    TONE_EXTERNAL: "warn"}
_TONE_ICON = {TONE_STOPPED: "circle", TONE_BUSY: "loader-circle", TONE_OK: "circle-check",
              TONE_FAILED: "circle-x", TONE_EXTERNAL: "triangle-alert"}

_mode = DARK
C: dict[str, str] = {}      # the active palette: token name -> value as written in the guide


def mode() -> str:
    return _mode


def is_dark() -> bool:
    return _mode == DARK


def palette_for(name: str) -> dict[str, str]:
    index = 1 if name == LIGHT else 0
    return {token: values[index] for token, values in TOKENS.items()}


def _use(name: str) -> None:
    global _mode
    _mode = LIGHT if name == LIGHT else DARK
    C.clear()
    C.update(palette_for(_mode))


_use(DARK)


# -- reading colours ----------------------------------------------------------
_RGBA = re.compile(r"rgba\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*([\d.]+)\s*\)")


def qcolor(value: str) -> QColor:
    """A QColor from a token value ('#rrggbb' or the guide's 'rgba(r, g, b, a)')."""
    match = _RGBA.fullmatch(value.strip())
    if match:
        r, g, b, a = match.groups()
        return QColor(int(r), int(g), int(b), round(float(a) * 255))
    return QColor(value)


def colour(token: str) -> QColor:
    return qcolor(C[token])


def qss(value: str) -> str:
    """A token value as a style sheet colour (Qt wants the alpha as 0 to 255)."""
    match = _RGBA.fullmatch(value.strip())
    if match:
        r, g, b, a = match.groups()
        return f"rgba({r}, {g}, {b}, {round(float(a) * 255)})"
    return value


def faded(token: str, opacity: float = DISABLED_OPACITY) -> str:
    """A token at part opacity, as a style sheet colour (the guide's 45% disabled state)."""
    c = colour(token)
    return f"rgba({c.red()}, {c.green()}, {c.blue()}, {round(c.alphaF() * opacity * 255)})"


def tone_token(tone: str) -> str:
    return _TONE_TOKEN[tone]


def tone_colour(tone: str) -> QColor:
    return colour(_TONE_TOKEN[tone])


def tone_text_colour(tone: str) -> QColor:
    return colour(_TONE_TEXT_TOKEN[tone])


def tone_icon(tone: str) -> str:
    return _TONE_ICON[tone]


# -- fonts ----------------------------------------------------------------------
def ui_family() -> str:
    """Segoe UI Variable Text (Windows 11), else Segoe UI."""
    return "Segoe UI Variable Text" if "Segoe UI Variable Text" in QFontDatabase.families() else "Segoe UI"


def display_family() -> str:
    """Titles: Segoe UI Variable Display, else Segoe UI."""
    return "Segoe UI Variable Display" if "Segoe UI Variable Display" in QFontDatabase.families() else "Segoe UI"


def mono_font(size: float = 9) -> QFont:
    """Cascadia Code, 12 px (9 pt) by default, as the guide's code style."""
    families = set(QFontDatabase.families())
    for family in ("Cascadia Code", "Cascadia Mono", "Consolas", "Courier New"):
        if family in families:
            font = QFont(family)
            break
    else:
        font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
    font.setPointSizeF(size)
    font.setStyleHint(QFont.StyleHint.Monospace)
    return font


def mono_family() -> str:
    return mono_font().family()


# -- Lucide icons ---------------------------------------------------------------
# Outline icons at the default 2 px stroke (24 unit grid), drawn in the colour asked for.
_LUCIDE = {
    "sun": '<circle cx="12" cy="12" r="4"/><path d="M12 2v2"/><path d="M12 20v2"/>'
           '<path d="m4.93 4.93 1.41 1.41"/><path d="m17.66 17.66 1.41 1.41"/><path d="M2 12h2"/>'
           '<path d="M20 12h2"/><path d="m6.34 17.66-1.41 1.41"/><path d="m19.07 4.93-1.41 1.41"/>',
    "moon": '<path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/>',
    "circle": '<circle cx="12" cy="12" r="9"/>',
    "circle-check": '<circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/>',
    "circle-x": '<circle cx="12" cy="12" r="10"/><path d="m15 9-6 6"/><path d="m9 9 6 6"/>',
    "triangle-alert": '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/>'
                      '<path d="M12 9v4"/><path d="M12 17h.01"/>',
    "loader-circle": '<path d="M21 12a9 9 0 1 1-6.219-8.56"/>',
    "chevron-up": '<path d="m18 15-6-6-6 6"/>',
    "chevron-down": '<path d="m6 9 6 6 6-6"/>',
    "arrow-up": '<path d="m5 12 7-7 7 7"/><path d="M12 19V5"/>',
    "arrow-down": '<path d="M12 5v14"/><path d="m19 12-7 7-7-7"/>',
    "check": '<path d="M20 6 9 17l-5-5"/>',
}


@lru_cache(maxsize=256)
def lucide_pixmap(name: str, color: str, size: int = 16, scale: int = 2, stroke: float = 2.0) -> QPixmap:
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="{color}" '
           f'stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round">{_LUCIDE[name]}</svg>')
    renderer = QSvgRenderer(QByteArray(svg.encode("utf-8")))
    pixmap = QPixmap(size * scale, size * scale)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    renderer.render(painter)
    painter.end()
    pixmap.setDevicePixelRatio(scale)
    return pixmap


def icon(name: str, token: str = "muted", size: int = 16) -> QIcon:
    return QIcon(lucide_pixmap(name, C[token], size))


_ICON_PROP = "_rulzIcon"


def set_icon(widget, name: str, token: str = "muted", size: int = 16) -> None:
    """Give a button a Lucide icon in a palette colour; ``retheme`` redraws it on a switch."""
    widget.setProperty(_ICON_PROP, f"{name}|{token}|{size}")
    widget.setIcon(icon(name, token, size))


def retheme(root: QWidget) -> None:
    """Redraw what a style sheet cannot: button icons, and anything with a ``retheme`` method."""
    for widget in [root, *root.findChildren(QWidget)]:
        spec = widget.property(_ICON_PROP)
        if spec:
            name, token, size = spec.split("|")
            widget.setIcon(icon(name, token, int(size)))
        hook = getattr(widget, "retheme", None)
        if callable(hook) and widget is not root:
            hook()
        widget.update()


def set_role(widget, **properties) -> None:
    """Set style properties (primary, danger, flat, controlSize ...) and restyle the widget."""
    for key, value in properties.items():
        widget.setProperty(key, value)
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
    widget.update()


# -- images the style sheet needs as files --------------------------------------
def _image_file(name: str, color: str, size: int) -> str:
    base = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.TempLocation)
    folder = Path(base) / "rulz-comfyui-launcher"
    path = folder / f"{name}-{color.lstrip('#')}-{size}.png"
    try:
        folder.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            stroke = 3.0 if name == "check" else 2.0   # the checkbox tick uses stroke 3
            lucide_pixmap(name, color, size, 2, stroke).save(str(path), "PNG")
    except OSError:
        pass
    return path.as_posix()


# -- logo and icon --------------------------------------------------------------
def app_icon() -> QIcon:
    """The RulZ app icon (window, taskbar and notification area)."""
    path = ASSETS / "icon.ico"
    return QIcon(str(path)) if path.exists() else QIcon()


def logo_file(mode_name: str | None = None) -> Path:
    """White lettering on the dark theme, black lettering on the light one."""
    return ASSETS / ("logo_full_black.png" if (mode_name or _mode) == LIGHT else "logo_full_white.png")


def logo_pixmap(width: int, scale: float = 2.0) -> QPixmap:
    """The full RulZ logo for the active theme, `width` wide. Empty if the file is missing."""
    pixmap = QPixmap(str(logo_file()))
    if pixmap.isNull():
        return pixmap
    pixmap = pixmap.scaledToWidth(round(width * scale), Qt.TransformationMode.SmoothTransformation)
    pixmap.setDevicePixelRatio(scale)
    return pixmap


# -- applying -------------------------------------------------------------------
def make_palette() -> QPalette:
    palette = QPalette()
    roles = QPalette.ColorRole
    for role, token in (
        (roles.Window, "bg"), (roles.WindowText, "text"), (roles.Base, "panel"),
        (roles.AlternateBase, "panel-2"), (roles.ToolTipBase, "panel-2"), (roles.ToolTipText, "text"),
        (roles.Text, "text"), (roles.PlaceholderText, "faint"), (roles.Button, "panel-2"),
        (roles.ButtonText, "text"), (roles.BrightText, "accent-fg"), (roles.Link, "link"),
        (roles.Highlight, "accent"), (roles.HighlightedText, "accent-fg"), (roles.Mid, "border"),
        (roles.Dark, "border-strong"),
    ):
        palette.setColor(role, colour(token))
    disabled = QPalette.ColorGroup.Disabled
    for role in (roles.WindowText, roles.Text, roles.ButtonText):
        palette.setColor(disabled, role, QColor(*_rgba(faded("text"))))
    return palette


def _rgba(text: str) -> tuple[int, int, int, int]:
    r, g, b, a = (int(x) for x in re.findall(r"\d+", text)[:4])
    return r, g, b, a


class _TabFocusOnly(QObject):
    """Qt cannot tell a keyboard focus from a click, so a clicked button would keep its ring.
    Buttons take focus from Tab only; the ring then always means the keyboard is there.
    The accepting button of a dialog (OK, Yes) is its primary button unless it was set otherwise."""

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.Polish and isinstance(obj, QAbstractButton):
            if obj.focusPolicy() in (Qt.FocusPolicy.StrongFocus, Qt.FocusPolicy.ClickFocus):
                obj.setFocusPolicy(Qt.FocusPolicy.TabFocus)
            box = obj.parentWidget()
            if isinstance(box, QDialogButtonBox) and obj.property("primary") is None                     and obj.property("danger") is None                     and box.buttonRole(obj) in (QDialogButtonBox.ButtonRole.AcceptRole,
                                                QDialogButtonBox.ButtonRole.YesRole):
                set_role(obj, primary=True)   # the one primary button of a dialog
        return False


def _set_color_scheme(app: QApplication, dark: bool) -> None:
    """The Windows title bar (and native dialogs) follow the app's own theme, not the Windows setting."""
    try:
        app.styleHints().setColorScheme(Qt.ColorScheme.Dark if dark else Qt.ColorScheme.Light)
    except (AttributeError, RuntimeError):
        pass


def apply_theme(app: QApplication, mode_name: str | None = None) -> None:
    """Apply a theme to the whole app. Call it again at runtime to switch live."""
    _use(mode_name if mode_name in MODES else _mode)
    _set_color_scheme(app, _mode == DARK)
    if app.property("_rulzTabFocus") is None:
        app._rulz_tab_focus = _TabFocusOnly(app)
        app.installEventFilter(app._rulz_tab_focus)
        app.setProperty("_rulzTabFocus", True)
    app.setStyle("Fusion")
    font = QFont(ui_family())
    font.setPointSizeF(10.5)
    app.setFont(font)
    app.setPalette(make_palette())
    app.setStyleSheet(stylesheet())
    for widget in app.topLevelWidgets():
        retheme(widget)


def ensure_theme(app: QApplication, mode_name: str) -> None:
    """Apply a theme unless that one is already on (what a window needs before it builds itself)."""
    if mode_name not in MODES:
        mode_name = DARK
    if mode_name != _mode or not app.styleSheet():
        apply_theme(app, mode_name)


def stylesheet() -> str:
    t = {key.replace("-", "_"): qss(value) for key, value in C.items()}
    t.update(
        off_text=faded("text"), off_muted=faded("muted"), off_panel2=faded("panel-2"),
        off_border=faded("border"), off_accent=faded("accent"), off_fg=faded("accent-fg"),
        off_danger_bg=faded("danger-bg"), off_danger_border=faded("danger-border"),
        off_danger_text=faded("danger-text"),
        display=display_family(),
        check=_image_file("check", C["accent-fg"], 12),
        up=_image_file("chevron-up", C["muted"], 10), down=_image_file("chevron-down", C["muted"], 10),
        combo_down=_image_file("chevron-down", C["muted"], 14),
    )
    return _STYLE % t


# Heights: a QSS min-height is the inside of the box, so a 1 px border takes 2 px more and a
# focused control (2 px border) drops the padding/min-height by the same amount.
_STYLE = """
QWidget { color: %(text)s; }
QMainWindow { background: %(bg)s; }
QDialog, QMessageBox { background: %(panel)s; }
QToolTip { background: %(panel_2)s; color: %(text)s; border: 1px solid %(border)s; border-radius: 8px;
           padding: 4px 8px; font-size: 9.75pt; }

/* ---- layout surfaces -------------------------------------------------------- */
QWidget#sidebar { background: %(panel)s; border-right: 1px solid %(border)s; }
QWidget#sidebar QLabel { background: transparent; }
QFrame#card { background: %(panel)s; border: 1px solid %(border)s; border-radius: 12px; }
QFrame#card QLabel { background: transparent; }
QWidget#page { background: %(bg)s; }

/* ---- text ------------------------------------------------------------------- */
QLabel#brandSub { color: %(muted)s; font-family: "%(display)s"; font-size: 9.75pt; font-weight: 600;
                  letter-spacing: 2px; }
QLabel#sideTitle { color: %(muted)s; font-size: 9.75pt; font-weight: 600; }
QLabel#cardTitle { font-family: "%(display)s"; font-size: 11.25pt; font-weight: 600; }
QLabel#pageTitle { font-family: "%(display)s"; font-size: 21pt; font-weight: 700; letter-spacing: -0.5px; }
QLabel#rowName { font-family: "%(display)s"; font-size: 11.25pt; font-weight: 600; }
QLabel#muted, QLabel[muted="true"] { color: %(muted)s; }
QLabel#hint { color: %(faint)s; font-size: 9pt; }
QLabel#error { color: %(danger_text)s; }
QLabel#warning { color: %(warn_text)s; }
QLabel#notice { color: %(warn_text)s; font-size: 9.75pt; }

/* ---- status pill (a word, a shape and a colour) ------------------------------ */
QFrame#pill { background: %(subtle)s; border: 1px solid %(subtle_border)s; border-radius: 12px; }
QFrame#pill QLabel { background: transparent; font-size: 8.625pt; font-weight: 500; }
QFrame#pill[tone="stopped"] QLabel#pillText { color: %(muted)s; }
QFrame#pill[tone="busy"] QLabel#pillText, QFrame#pill[tone="external"] QLabel#pillText { color: %(warn)s; }
QFrame#pill[tone="ok"] QLabel#pillText { color: %(ok)s; }
QFrame#pill[tone="failed"] QLabel#pillText { color: %(danger_text)s; }

/* ---- buttons: 36 px, radius 8; small 32 px, radius 6; large 44 px ------------ */
QPushButton { background: %(panel_2)s; color: %(text)s; border: 1px solid %(border)s; border-radius: 8px;
              padding: 0 16px; min-height: 34px; font-size: 10.5pt; font-weight: 500; }
QPushButton:hover { background: %(hover)s; border-color: %(border_strong)s; }
QPushButton:pressed { background: %(hover)s; border-color: %(border_strong)s; }
QPushButton:focus { border: 2px solid %(accent)s; padding: 0 15px; min-height: 32px; }
QPushButton:disabled { background: %(off_panel2)s; color: %(off_text)s; border-color: %(off_border)s; }
QPushButton[controlSize="sm"] { border-radius: 6px; padding: 0 12px; min-height: 30px; font-size: 9.75pt; }
QPushButton[controlSize="sm"]:focus { padding: 0 11px; min-height: 28px; }
QPushButton[controlSize="lg"] { padding: 0 24px; min-height: 42px; font-size: 11.25pt; font-weight: 600; }
QPushButton[controlSize="lg"]:focus { padding: 0 23px; min-height: 40px; }

QPushButton[primary="true"] { background: %(accent)s; border-color: %(accent)s; color: %(accent_fg)s; font-weight: 600; }
QPushButton[primary="true"]:hover, QPushButton[primary="true"]:pressed { background: %(accent_hover)s; border-color: %(accent_hover)s; }
QPushButton[primary="true"]:focus { border: 2px solid %(accent_fg)s; }
QPushButton[primary="true"]:disabled { background: %(off_accent)s; border-color: %(off_accent)s; color: %(off_fg)s; }

QPushButton[danger="true"] { background: %(danger_bg)s; border-color: %(danger_border)s; color: %(danger_text)s; }
QPushButton[danger="true"]:hover, QPushButton[danger="true"]:pressed { background: %(danger_hover)s; border-color: %(danger_border)s; }
QPushButton[danger="true"]:disabled { background: %(off_danger_bg)s; border-color: %(off_danger_border)s; color: %(off_danger_text)s; }

QPushButton[flat="true"] { background: transparent; border: 1px solid transparent; color: %(muted)s; }
QPushButton[flat="true"]:hover, QPushButton[flat="true"]:pressed { background: %(hover)s; color: %(text)s; border-color: transparent; }
QPushButton[flat="true"]:focus { border: 2px solid %(accent)s; }
QPushButton[flat="true"]:disabled { background: transparent; color: %(off_muted)s; border-color: transparent; }

QToolButton { background: %(panel_2)s; color: %(text)s; border: 1px solid %(border)s; border-radius: 6px;
              padding: 0 12px; min-height: 30px; font-size: 9.75pt; font-weight: 500; }
QToolButton:hover, QToolButton:pressed { background: %(hover)s; border-color: %(border_strong)s; }
QToolButton:focus { border: 2px solid %(accent)s; padding: 0 11px; min-height: 28px; }
QToolButton:disabled { background: %(off_panel2)s; color: %(off_text)s; border-color: %(off_border)s; }
QToolButton[iconOnly="true"] { padding: 0; min-width: 30px; max-width: 30px; background: transparent;
                               border-color: transparent; }
QToolButton[iconOnly="true"]:hover { background: %(hover)s; border-color: transparent; }
QToolButton[iconOnly="true"]:focus { border: 2px solid %(accent)s; padding: 0; min-width: 28px; max-width: 28px; }
QToolButton#sectionHeader { background: transparent; border: 1px solid transparent; border-radius: 6px; text-align: left;
                            font-size: 10.5pt; font-weight: 600; padding: 0 4px; min-height: 30px; }
QToolButton#sectionHeader:hover { background: %(panel_2)s; border-color: transparent; }
QToolButton#sectionHeader:focus { border: 2px solid %(accent)s; padding: 0 3px; min-height: 28px; }

/* ---- fields: radius 8 ---------------------------------------------------------- */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background: %(panel)s; border: 1px solid %(border)s; border-radius: 8px; padding: 0 10px; min-height: 34px;
    font-size: 10.5pt; selection-background-color: %(accent)s; selection-color: %(accent_fg)s; }
QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover, QComboBox:hover { border-color: %(border_strong)s; }
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border: 2px solid %(accent)s; padding: 0 9px; min-height: 32px; }
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {
    color: %(off_text)s; background: %(panel)s; border-color: %(off_border)s; }
QLineEdit[invalid="true"] { border-color: %(danger)s; }
QTableWidget QLineEdit, QComboBox QLineEdit, QSpinBox QLineEdit, QDoubleSpinBox QLineEdit {
    background: transparent; border: none; border-radius: 0; padding: 0; min-height: 0; }
QComboBox { padding-right: 4px; }
QComboBox::drop-down { border: none; width: 28px; }
QComboBox::down-arrow { image: url("%(combo_down)s"); width: 14px; height: 14px; }
QComboBox QAbstractItemView { background: %(panel_2)s; border: 1px solid %(border)s; border-radius: 8px; padding: 4px;
                              outline: 0; selection-background-color: %(hover)s; selection-color: %(text)s; }
QSpinBox, QDoubleSpinBox { padding-right: 24px; }
QSpinBox::up-button, QDoubleSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::down-button {
    subcontrol-origin: border; width: 22px; background: transparent; border: none; }
QSpinBox::up-button, QDoubleSpinBox::up-button { subcontrol-position: top right; }
QSpinBox::down-button, QDoubleSpinBox::down-button { subcontrol-position: bottom right; }
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover { background: %(hover)s; }
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow { image: url("%(up)s"); width: 10px; height: 10px; }
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow { image: url("%(down)s"); width: 10px; height: 10px; }

/* Large areas mark focus with a 1 px border-strong, not the ring. */
QPlainTextEdit { background: %(panel)s; border: 1px solid %(border)s; border-radius: 8px; padding: 4px;
                 font-size: 9pt; selection-background-color: %(accent)s; selection-color: %(accent_fg)s; }
QPlainTextEdit:focus { border: 1px solid %(border_strong)s; }
QPlainTextEdit[readOnly="true"] { background: %(code_bg)s; }
QPlainTextEdit#logView { background: %(sunken)s; padding: 6px; }
QPlainTextEdit#logView:focus { background: %(sunken)s; }
QPlainTextEdit:disabled { color: %(off_text)s; }

/* ---- checkboxes (18 px, radius 4) and radio buttons ------------------------------ */
QCheckBox, QRadioButton { spacing: 12px; background: transparent; }
QCheckBox:disabled, QRadioButton:disabled { color: %(off_text)s; }
QCheckBox::indicator { width: 16px; height: 16px; border: 1px solid %(border_strong)s; border-radius: 4px;
                       background: %(panel_2)s; }
QCheckBox::indicator:hover { border-color: %(muted)s; }
QCheckBox::indicator:focus { width: 14px; height: 14px; border: 2px solid %(accent)s; }
QCheckBox::indicator:checked { background: %(accent)s; border-color: %(accent)s; image: url("%(check)s"); }
QCheckBox::indicator:checked:focus { width: 14px; height: 14px; border: 2px solid %(text)s; }
QCheckBox::indicator:disabled { border-color: %(off_border)s; background: %(off_panel2)s; }
QCheckBox::indicator:checked:disabled { background: %(off_accent)s; border-color: %(off_accent)s; }
QRadioButton::indicator { width: 16px; height: 16px; border: 1px solid %(border_strong)s; border-radius: 9px;
                          background: %(panel_2)s; }
QRadioButton::indicator:hover { border-color: %(muted)s; }
QRadioButton::indicator:focus { width: 14px; height: 14px; border: 2px solid %(accent)s; }
QRadioButton::indicator:checked { width: 8px; height: 8px; border: 5px solid %(accent)s; border-radius: 9px; background: %(accent_fg)s; }
QRadioButton::indicator:checked:focus { width: 6px; height: 6px; border: 6px solid %(accent)s; border-radius: 9px; background: %(accent_fg)s; }
QRadioButton::indicator:disabled { border-color: %(off_border)s; background: %(off_panel2)s; }
QRadioButton::indicator:checked:disabled { border-color: %(off_accent)s; background: %(off_fg)s; }

/* ---- lists, tabs, tables -------------------------------------------------------------- */
QListWidget { background: transparent; border: 1px solid transparent; border-radius: 8px; outline: none; }
QListWidget:focus { border: 1px solid %(border_strong)s; }
QListWidget::item { border-radius: 8px; margin: 2px 6px; }
QListWidget::item:hover { background: %(hover)s; }
QListWidget::item:selected { background: %(accent_soft)s; }

QTabWidget::pane { border: none; }
QTabBar { background: transparent; }
QTabBar::tab { background: transparent; color: %(muted)s; padding: 0 16px; min-height: 34px;
               border: none; border-bottom: 2px solid transparent; font-size: 10.5pt; font-weight: 500; }
QTabBar::tab:hover { color: %(text)s; }
QTabBar::tab:selected { color: %(text)s; border-bottom: 2px solid %(accent)s; font-weight: 600; }
QTabBar::tab:focus { background: %(subtle)s; border-radius: 6px; }

QTableWidget { background: %(panel)s; alternate-background-color: %(subtle)s; border: 1px solid %(border)s;
               border-radius: 8px; gridline-color: %(border)s; outline: 0; font-size: 9pt;
               selection-background-color: %(accent_soft)s; selection-color: %(text)s; }
QTableWidget:focus { border: 1px solid %(border_strong)s; }
QHeaderView::section { background: %(panel_2)s; color: %(muted)s; border: none; border-bottom: 1px solid %(border)s;
                       padding: 6px 10px; font-size: 9.75pt; font-weight: 500; }

QScrollArea { border: none; background: transparent; }
QScrollBar:vertical { background: transparent; width: 12px; margin: 0; }
QScrollBar::handle:vertical { background: %(border_strong)s; border-radius: 4px; min-height: 30px; margin: 2px; }
QScrollBar::handle:vertical:hover { background: %(muted)s; }
QScrollBar:horizontal { background: transparent; height: 12px; margin: 0; }
QScrollBar::handle:horizontal { background: %(border_strong)s; border-radius: 4px; min-width: 30px; margin: 2px; }
QScrollBar::handle:horizontal:hover { background: %(muted)s; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }

QProgressBar { background: %(track)s; border: none; border-radius: 3px; max-height: 6px; min-height: 6px; }
QProgressBar::chunk { background: %(accent)s; border-radius: 3px; }

QSplitter::handle { background: %(border)s; width: 1px; }
QStatusBar { background: %(bg)s; border-top: 1px solid %(border)s; color: %(muted)s; font-size: 9.75pt; }
QStatusBar::item { border: none; }
QStatusBar QLabel { color: %(muted)s; }

/* ---- menus: radius 8, items radius 6 ---------------------------------------------------- */
QMenu { background: %(panel_2)s; border: 1px solid %(border)s; border-radius: 8px; padding: 4px; font-size: 9.75pt; }
QMenu::item { padding: 7px 24px 7px 12px; border-radius: 6px; }
QMenu::item:selected { background: %(hover)s; }
QMenu::item:disabled { color: %(off_text)s; }
QMenu::separator { height: 1px; background: %(border)s; margin: 4px 8px; }
QMenu::indicator { width: 14px; height: 14px; left: 6px; border: 1px solid %(border_strong)s; border-radius: 4px;
                   background: %(panel)s; }
QMenu::indicator:checked { background: %(accent)s; border-color: %(accent)s; image: url("%(check)s"); }

QDialogButtonBox QPushButton { min-width: 72px; }
"""
