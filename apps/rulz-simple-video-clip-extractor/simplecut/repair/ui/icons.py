"""Application icons, drawn as small SVGs (24×24, ``currentColor`` strokes).

From RulZ Image Distortion Repair; the colours follow the Clip Extractor's dark/light theme.
"""
from __future__ import annotations

from functools import lru_cache

from PySide6.QtCore import QByteArray, QRectF, QSize, Qt
from PySide6.QtGui import QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

S = 'fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"'
F = 'fill="currentColor" stroke="none"'

_SVG = {
    "folder": f'<path {S} d="M3 6.5h6l2 2h10v10H3z"/>',
    "prev": f'<path {S} d="M19 12H5m5-5-5 5 5 5"/>',
    "next": f'<path {S} d="M5 12h14m-5-5 5 5-5 5"/>',
    "compare": f'<rect {S} x="4" y="5" width="16" height="14"/><path {S} d="M8 19 20 7M12 19l8-8M16 19l4-4M4 13l8-8M4 9l4-4"/>',
    "sidebyside": f'<rect {S} x="2.5" y="6" width="9" height="12"/><rect {S} x="12.5" y="6" width="9" height="12"/>'
                  f'<path {S} d="M13 15l5-5M15 18l6-6"/>',
    "fit": f'<path {S} d="M12 3v18M3 12h18M12 3l-3 3m3-3 3 3M12 21l-3-3m3 3 3-3M3 12l3-3m-3 3 3 3M21 12l-3-3m3 3-3 3"/>',
    "flip_h": f'<path {S} d="M12 3v18"/><path {S} d="M10 7 4 17h6z"/><path {F} d="M14 7l6 10h-6z"/>',
    "flip_v": f'<path {S} d="M3 12h18"/><path {S} d="M7 10 17 4v6z"/><path {F} d="M7 14l10 6v-6z"/>',
    "rot_left": f'<rect {S} x="9" y="11" width="10" height="8"/><path {S} d="M5 14V9a4 4 0 0 1 4-4h5m-3-3 3 3-3 3"/>',
    "rot_right": f'<rect {S} x="5" y="11" width="10" height="8"/><path {S} d="M19 14V9a4 4 0 0 0-4-4h-5m3-3-3 3 3 3"/>',
    "grid": f'<rect {S} x="4" y="5" width="16" height="14"/><path {S} d="M4 9.7h16M4 14.3h16M9.3 5v14M14.7 5v14"/>',
    "guides": f'<path {S} d="M6 4v16M4 6h16" stroke-dasharray="2 2"/><path {S} d="M6 4v3M4 6h3"/>',
    "guides_edit": f'<rect {S} x="4" y="4" width="13" height="16"/><path {S} d="M8 8h5M8 12h5M8 16h3"/><path {S} d="M15 17l5-5 1.5 1.5-5 5H15z"/>',
    "fullscreen": f'<path {S} d="M4 9V4h5M15 4h5v5M20 15v5h-5M9 20H4v-5M4 4l5 5M20 4l-5 5M20 20l-5-5M4 20l5-5"/>',
    "reset": f'<path {S} d="M7 9H4V6"/><path {S} d="M4.5 9A8 8 0 1 1 6 16"/>',
    "trash": f'<path {S} d="M5 7h14M10 7V4h4v3M7 7l1 13h8l1-13"/>',
    "plus_small": f'<path {S} d="M7 12h10M12 7v10"/>',
    "minus_small": f'<path {S} d="M7 12h10"/>',
    "check": '<path fill="none" stroke="white" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" d="M5 12.5l4.5 4.5L19 7.5"/>',
    "ok_circle": f'<circle {S} cx="12" cy="12" r="9"/><path {S} d="M7.5 12.5l3 3 6-6"/>',
    # correction tools
    "distortion_manual": f'<circle {S} cx="12" cy="12" r="9"/><path {S} d="M9 6c-1.5 4-1.5 8 0 12M15 6c1.5 4 1.5 8 0 12M12 5v14"/>',
    "volume_hv": f'<circle {S} cx="12" cy="8" r="2.4"/><path {S} d="M8.5 19v-4a3.5 3.5 0 0 1 7 0v4"/>'
                 f'<path {S} d="M2.5 12h3m-1.5-1.5L2.5 12 4 13.5M21.5 12h-3m1.5-1.5 1.5 1.5-1.5 1.5M12 2v1.5"/>',
    "volume_diag": f'<circle {S} cx="12" cy="9" r="2.4"/><path {S} d="M8.5 20v-4a3.5 3.5 0 0 1 7 0v4"/>'
                   f'<path {S} d="M3 3h4M3 3v4M21 3h-4M21 3v4M3 3l3 3M21 3l-3 3"/>',
    "persp_verticals": f'<path {S} d="M8 5v14M16 5v14"/><circle {F} cx="8" cy="5" r="1.8"/><circle {F} cx="8" cy="19" r="1.8"/>'
                       f'<circle {F} cx="16" cy="5" r="1.8"/><circle {F} cx="16" cy="19" r="1.8"/>',
    "persp_horizontals": f'<path {S} d="M5 8h14M5 16h14"/><circle {F} cx="5" cy="8" r="1.8"/><circle {F} cx="19" cy="8" r="1.8"/>'
                         f'<circle {F} cx="5" cy="16" r="1.8"/><circle {F} cx="19" cy="16" r="1.8"/>',
    "persp_rect": f'<rect {S} x="6" y="6" width="12" height="12"/><circle {F} cx="6" cy="6" r="1.8"/><circle {F} cx="18" cy="6" r="1.8"/>'
                  f'<circle {F} cx="18" cy="18" r="1.8"/><circle {F} cx="6" cy="18" r="1.8"/>',
    "persp_eight": f'<circle {S} cx="7" cy="5" r="1.6"/><circle {S} cx="17" cy="5" r="1.6"/><circle {S} cx="4" cy="12" r="1.6"/>'
                   f'<circle {S} cx="20" cy="12" r="1.6"/><circle {S} cx="7" cy="19" r="1.6"/><circle {S} cx="17" cy="19" r="1.6"/>'
                   f'<path {S} d="M12 9v6M9 12h6"/>',
    "grid4": f'<rect {S} x="4" y="4" width="16" height="16"/><path {S} d="M12 4v16M4 12h16"/>',
    "grid8": f'<rect {S} x="4" y="4" width="16" height="16"/><path {S} d="M9.3 4v16M14.7 4v16M4 9.3h16M4 14.7h16"/>',
    "grid16": f'<rect {S} x="4" y="4" width="16" height="16"/><path {S} d="M8 4v16M12 4v16M16 4v16M4 8h16M4 12h16M4 16h16"/>',
    "grid32": f'<rect {F} x="4" y="4" width="16" height="16" opacity="0.85"/>',
    "grid_custom": f'<rect {S} x="4" y="4" width="13" height="13"/><path {S} d="M10.5 4v13M4 10.5h13"/><circle {S} cx="17.5" cy="17.5" r="3"/><path {S} d="M17.5 16v3M16 17.5h3"/>',
    "horizon_h": f'<path {S} d="M3 12h18"/><rect {S} x="9" y="9.5" width="6" height="5" rx="2.5"/>',
    "horizon_v": f'<path {S} d="M12 3v18M7 9h10"/><rect {S} x="9" y="7" width="6" height="4" rx="2"/>',
    "crop": f'<path {S} d="M7 3v14h14M3 7h14v14"/>',
    "tiltshift": f'<path {S} d="M3 9h18M3 15h18" stroke-dasharray="1.5 2"/><circle {F} cx="12" cy="12" r="1.6"/>',
    "move": f'<path {S} d="M12 3v18M3 12h18M12 3l-2.5 2.5M12 3l2.5 2.5M12 21l-2.5-2.5M12 21l2.5-2.5M3 12l2.5-2.5M3 12l2.5 2.5M21 12l-2.5-2.5M21 12l-2.5 2.5"/>',
    "rotate": f'<path {S} d="M19 12a7 7 0 1 1-2-5"/><path {S} d="M18 3v4h-4"/>',
    "scale": f'<path {S} d="M5 19 19 5M13 5h6v6M11 19H5v-6"/>',
    "warp_free": f'<path {S} d="M4 5c5 2 11 2 16 0M4 12c5 2 11 2 16 0M4 19c5 2 11 2 16 0M5 4c-2 5-2 11 0 16M12 4c-2 5-2 11 0 16M19 4c-2 5-2 11 0 16"/>',
    "warp_persp": f'<path {S} d="M7 5h10l3 14H4z"/><path {S} d="M12 5v14M5.5 12h13"/>',
    "warp_anamorph": f'<rect {S} x="4" y="5" width="16" height="14"/><path {S} d="M9 5c-2 5-2 9 0 14M15 5c2 5 2 9 0 14"/>',
    "image_browser": f'<rect {S} x="3" y="4" width="18" height="16"/><path {S} d="M3 16l5-5 4 4 3-3 6 6"/><circle {S} cx="15.5" cy="8.5" r="1.5"/>',
    "recent": f'<circle {S} cx="12" cy="12" r="8.5"/><path {S} d="M12 7v5l3.5 2"/>',
    "sort": f'<path {S} d="M8 5v14m-3-3 3 3 3-3M16 19V5m-3 3 3-3 3 3"/>',
    "color": f'<rect {F} x="4" y="4" width="16" height="16"/>',
    "settings": f'<circle {S} cx="12" cy="12" r="3"/><path {S} d="M12 2v3M12 19v3M2 12h3M19 12h3M4.9 4.9l2.1 2.1M17 17l2.1 2.1M4.9 19.1 7 17M17 7l2.1-2.1"/>',
    "play": f'<path {F} d="M8 5.5v13l10.5-6.5z"/>',
    "pause": f'<rect {F} x="7" y="5.5" width="3.5" height="13" rx="0.8"/><rect {F} x="13.5" y="5.5" width="3.5" height="13" rx="0.8"/>',
    "step_back": f'<path {F} d="M17 6v12l-8-6z"/><rect {F} x="6" y="6" width="2" height="12"/>',
    "step_fwd": f'<path {F} d="M7 6v12l8-6z"/><rect {F} x="16" y="6" width="2" height="12"/>',
    "loop": f'<path {S} d="M5 12V10a3 3 0 0 1 3-3h10m-3-3 3 3-3 3M19 12v2a3 3 0 0 1-3 3H6m3 3-3-3 3-3"/>',
    "volume": f'<path {F} d="M4 9.5h3.5L12 6v12l-4.5-3.5H4z"/><path {S} d="M15 9a4 4 0 0 1 0 6M17.5 6.5a7.5 7.5 0 0 1 0 11"/>',
    "muted": f'<path {F} d="M4 9.5h3.5L12 6v12l-4.5-3.5H4z"/><path {S} d="M15.5 9.5l5 5M20.5 9.5l-5 5"/>',
    "video": f'<rect {S} x="3" y="6" width="13" height="12" rx="1.5"/><path {S} d="M16 10.5l5-3v9l-5-3z"/>',
    "app":'<rect x="1" y="1" width="22" height="22" rx="4" fill="#2d4a5c"/>'
           '<path fill="none" stroke="#cfe6f5" stroke-width="1.6" stroke-linejoin="round" d="M5 18 9 6h6l4 12z"/>'
           '<path fill="none" stroke="#8cbcda" stroke-width="1.6" d="M9 6v12M15 6v12"/>',
}

# (normal, hover, checked, disabled) per theme. Checked buttons sit on the red accent: white.
COLORS = {"dark": ("#d6d6d6", "#ffffff", "#ffffff", "#5a5a5a"),
          "light": ("#3a3a40", "#111114", "#ffffff", "#a3a3ab")}
NORMAL = COLORS["dark"][0]


def _mode() -> str:
    from ... import theme
    return theme.mode()


def svg_bytes(name: str, color: str = NORMAL) -> bytes:
    body = _SVG[name].replace("currentColor", color)
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">{body}</svg>'.encode()


def pixmap(name: str, size: int = 24, color: str | None = None, dpr: float = 2.0) -> QPixmap:
    if color is None:
        color = COLORS.get(_mode(), COLORS["dark"])[0]
    r = QSvgRenderer(QByteArray(svg_bytes(name, color)))
    pm = QPixmap(int(size * dpr), int(size * dpr))
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    r.render(p, QRectF(0, 0, size * dpr, size * dpr))
    p.end()
    pm.setDevicePixelRatio(dpr)
    return pm


def icon(name: str, size: int = 24) -> QIcon:
    return _icon(name, size, _mode())


@lru_cache(maxsize=None)
def _icon(name: str, size: int, mode: str) -> QIcon:
    normal, hover, checked, disabled = COLORS.get(mode, COLORS["dark"])
    ic = QIcon()
    ic.addPixmap(pixmap(name, size, normal), QIcon.Normal, QIcon.Off)
    ic.addPixmap(pixmap(name, size, checked), QIcon.Normal, QIcon.On)
    ic.addPixmap(pixmap(name, size, hover), QIcon.Active, QIcon.Off)
    ic.addPixmap(pixmap(name, size, checked), QIcon.Active, QIcon.On)
    ic.addPixmap(pixmap(name, size, disabled), QIcon.Disabled, QIcon.Off)
    ic.addPixmap(pixmap(name, size, disabled), QIcon.Disabled, QIcon.On)
    return ic


def icon_size(px: int = 18) -> QSize:
    return QSize(px, px)


def write_asset(name: str, path: str, color: str = NORMAL) -> str:
    with open(path, "wb") as f:
        f.write(svg_bytes(name, color))
    return path.replace("\\", "/")
