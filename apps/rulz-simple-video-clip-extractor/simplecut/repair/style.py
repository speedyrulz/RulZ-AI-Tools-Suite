"""Style for the Distortion repair window: Distortion Repair's widget names in the app's theme.

The app-wide stylesheet (theme.stylesheet) already styles buttons, inputs and sliders; this
adds the window's own pieces (segmented buttons, correction panels, tool panes) in the current
dark or light colours. Built when the window opens, so it follows a theme switch.
"""
from __future__ import annotations

from .. import theme


def _rgba(hex_color: str, alpha: int) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"


def stylesheet() -> str:
    t = theme
    return f"""
QWidget#repairTopBar, QWidget#repairBottomBar {{ background: {t.BAR}; }}
QWidget#repairTopBar {{ border-bottom: 2px solid {t.RED}; }}
QWidget#repairBottomBar {{ border-top: 1px solid {t.BORDER}; }}

QToolButton {{ background: transparent; border: none; padding: 3px; border-radius: 3px; }}
QToolButton:hover {{ background: {t.RAISED_HOVER}; }}
QToolButton:checked {{ background: {t.RED}; color: #ffffff; }}
QToolButton:disabled {{ color: {t.DIS_TEXT}; }}
QToolButton[segment="true"] {{
    background: {t.RAISED}; border: 1px solid {t.BORDER}; border-radius: 0; min-height: 22px; padding: 2px 6px;
}}
QToolButton[segment="true"]:hover {{ background: {t.RAISED_HOVER}; border-color: {t.RED}; }}
QToolButton[segment="true"]:checked {{ background: {t.RED}; border-color: {t.RED_HOVER}; color: #ffffff; }}
QToolButton[segment="true"]:disabled {{ background: {t.DIS_BG}; border-color: {t.DIS_BORDER}; color: {t.DIS_TEXT}; }}
QToolButton[segFirst="true"] {{ border-top-left-radius: 4px; border-bottom-left-radius: 4px; }}
QToolButton[segLast="true"] {{ border-top-right-radius: 4px; border-bottom-right-radius: 4px; }}
QToolButton#dockToggle {{ background: {t.PANEL}; border-left: 1px solid {t.BORDER}; border-radius: 0; color: {t.MUTED}; }}

QWidget#dockContents, QScrollArea#settingsScroll, QWidget#settingsContents {{ background: {t.PANEL}; }}
QWidget#dockTitle {{ background: {t.BG}; border-bottom: 1px solid {t.BORDER}; }}
QToolButton#resetAllButton {{ color: {t.MUTED}; font-size: 8pt; }}
QToolButton#resetAllButton:hover {{ color: {t.TEXT}; }}
QWidget#panelHeader {{ background: {t.RAISED}; }}
QLabel#panelTitle {{ color: {t.ACCENT_TEXT}; font-weight: bold; }}
QLabel#panelTitle[active="false"] {{ color: {t.TEXT}; font-weight: normal; }}
QFrame#panelSeparator {{ background: {t.BORDER}; max-height: 1px; min-height: 1px; }}
QWidget#infoBox {{ background: {t.BG}; border-left: 3px solid {t.RED}; }}
QLabel#bubble {{
    background: #000000; color: #ffffff; border: 1px solid {t.RED}; border-radius: 3px; padding: 0 4px; font-size: 8pt;
}}

QWidget#toolPane {{ background: {t.PANEL}; border-top: 1px solid {t.RED}; }}
QFrame#toolPaneDivider {{ background: {t.BORDER}; }}
QPushButton[toolPane="true"] {{ min-width: 90px; }}
QPushButton#applyButton {{ background: {t.RED}; border-color: {t.RED_HOVER}; color: #ffffff; font-weight: bold; }}
QPushButton#applyButton:hover {{ background: {t.RED_HOVER}; }}
QWidget#floatingOptions {{ background: {_rgba(t.PANEL, 240)}; border: 1px solid {t.RED}; border-radius: 4px; }}
QLabel#floatingTitle {{ font-weight: bold; color: {t.ACCENT_TEXT}; }}
QLabel#sizeLabel {{ color: {t.MUTED}; }}
"""
