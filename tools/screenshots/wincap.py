"""Tiny ctypes helpers to capture a single window with PrintWindow(PW_RENDERFULLCONTENT).

PrintWindow asks the window (via DWM) to render itself into our bitmap, so whatever happens to be on the
desktop or on top of the window is never captured.  Works for Win32/WinForms/Qt windows.

    from wincap import find_windows, capture
    for hwnd, title, cls in find_windows(pid): ...
    w, h, bgra = capture(hwnd, "client")   # or "frame" (visible window incl. title bar)
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes

user32 = ctypes.WinDLL("user32", use_last_error=True)
gdi32 = ctypes.WinDLL("gdi32")
dwmapi = ctypes.WinDLL("dwmapi")

HWND, HDC, HBITMAP, HGDIOBJ = wintypes.HWND, wintypes.HDC, wintypes.HBITMAP, wintypes.HGDIOBJ
user32.GetDC.restype = HDC
user32.GetDC.argtypes = [HWND]
user32.ReleaseDC.argtypes = [HWND, HDC]
user32.PrintWindow.argtypes = [HWND, HDC, wintypes.UINT]
user32.GetWindowRect.argtypes = [HWND, ctypes.POINTER(wintypes.RECT)]
user32.GetClientRect.argtypes = [HWND, ctypes.POINTER(wintypes.RECT)]
user32.ClientToScreen.argtypes = [HWND, ctypes.POINTER(wintypes.POINT)]
user32.GetWindowThreadProcessId.argtypes = [HWND, ctypes.POINTER(wintypes.DWORD)]
user32.IsWindowVisible.argtypes = [HWND]
user32.GetWindowTextW.argtypes = [HWND, wintypes.LPWSTR, ctypes.c_int]
user32.GetClassNameW.argtypes = [HWND, wintypes.LPWSTR, ctypes.c_int]
user32.PostMessageW.argtypes = [HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
user32.SetWindowPos.argtypes = [HWND, HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.UINT]
user32.GetWindow.restype = HWND
user32.GetWindow.argtypes = [HWND, wintypes.UINT]
gdi32.CreateCompatibleDC.restype = HDC
gdi32.CreateCompatibleDC.argtypes = [HDC]
gdi32.CreateCompatibleBitmap.restype = HBITMAP
gdi32.CreateCompatibleBitmap.argtypes = [HDC, ctypes.c_int, ctypes.c_int]
gdi32.SelectObject.restype = HGDIOBJ
gdi32.SelectObject.argtypes = [HDC, HGDIOBJ]
gdi32.DeleteObject.argtypes = [HGDIOBJ]
gdi32.DeleteDC.argtypes = [HDC]
gdi32.GetDIBits.argtypes = [HDC, HBITMAP, wintypes.UINT, wintypes.UINT, ctypes.c_void_p, ctypes.c_void_p,
                            wintypes.UINT]
dwmapi.DwmGetWindowAttribute.argtypes = [HWND, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD]

PW_RENDERFULLCONTENT = 2
DWMWA_EXTENDED_FRAME_BOUNDS = 9
WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, HWND, wintypes.LPARAM)
user32.EnumWindows.argtypes = [WNDENUMPROC, wintypes.LPARAM]
user32.EnumChildWindows.argtypes = [HWND, WNDENUMPROC, wintypes.LPARAM]
user32.MapVirtualKeyW.argtypes = [wintypes.UINT, wintypes.UINT]


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [("biSize", wintypes.DWORD), ("biWidth", ctypes.c_long), ("biHeight", ctypes.c_long),
                ("biPlanes", wintypes.WORD), ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
                ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", ctypes.c_long),
                ("biYPelsPerMeter", ctypes.c_long), ("biClrUsed", wintypes.DWORD),
                ("biClrImportant", wintypes.DWORD)]


def find_windows(pid: int, visible_only: bool = True):
    """Top-level windows of a process: [(hwnd, title, class_name)]."""
    found = []

    def cb(hwnd, _):
        p = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(p))
        if p.value == pid and (not visible_only or user32.IsWindowVisible(hwnd)):
            t = ctypes.create_unicode_buffer(512)
            c = ctypes.create_unicode_buffer(256)
            user32.GetWindowTextW(hwnd, t, 512)
            user32.GetClassNameW(hwnd, c, 256)
            found.append((hwnd, t.value, c.value))
        return True

    user32.EnumWindows(WNDENUMPROC(cb), 0)
    return found


def child_windows(hwnd):
    """All descendant windows: [(hwnd, class_name, (l, t, r, b))]."""
    found = []

    def cb(child, _):
        c = ctypes.create_unicode_buffer(256)
        user32.GetClassNameW(child, c, 256)
        found.append((child, c.value, window_rect(child)))
        return True

    user32.EnumChildWindows(hwnd, WNDENUMPROC(cb), 0)
    return found


def post_key(hwnd, vk):
    """Post a key press (WM_KEYDOWN + WM_KEYUP) to a window; it does not need to be in the foreground."""
    scan = user32.MapVirtualKeyW(vk, 0)
    user32.PostMessageW(hwnd, 0x0100, vk, 1 | (scan << 16))
    user32.PostMessageW(hwnd, 0x0101, vk, 1 | (scan << 16) | (0xC0 << 24))


def window_rect(hwnd):
    r = wintypes.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(r))
    return r.left, r.top, r.right, r.bottom


def frame_rect(hwnd):
    r = wintypes.RECT()
    if dwmapi.DwmGetWindowAttribute(hwnd, DWMWA_EXTENDED_FRAME_BOUNDS, ctypes.byref(r), ctypes.sizeof(r)) == 0:
        return r.left, r.top, r.right, r.bottom
    return window_rect(hwnd)


def client_rect(hwnd):
    """Client area in screen coordinates."""
    r = wintypes.RECT()
    user32.GetClientRect(hwnd, ctypes.byref(r))
    pt = wintypes.POINT(0, 0)
    user32.ClientToScreen(hwnd, ctypes.byref(pt))
    return pt.x, pt.y, pt.x + r.right, pt.y + r.bottom


def capture(hwnd, area: str = "client"):
    """Render the window with PrintWindow and crop it. Returns (width, height, BGRA bytes)."""
    wl, wt, wr, wb = window_rect(hwnd)
    w, h = wr - wl, wb - wt
    screen = user32.GetDC(None)
    dc = gdi32.CreateCompatibleDC(screen)
    bmp = gdi32.CreateCompatibleBitmap(screen, w, h)
    old = gdi32.SelectObject(dc, bmp)
    try:
        if not user32.PrintWindow(hwnd, dc, PW_RENDERFULLCONTENT):
            raise OSError("PrintWindow failed")
        bi = BITMAPINFOHEADER(ctypes.sizeof(BITMAPINFOHEADER), w, -h, 1, 32, 0, 0, 0, 0, 0, 0)
        buf = ctypes.create_string_buffer(w * h * 4)
        gdi32.GetDIBits(dc, bmp, 0, h, buf, ctypes.byref(bi), 0)
    finally:
        gdi32.SelectObject(dc, old)
        gdi32.DeleteObject(bmp)
        gdi32.DeleteDC(dc)
        user32.ReleaseDC(None, screen)
    l, t, r, b = client_rect(hwnd) if area == "client" else frame_rect(hwnd)
    x0, y0 = max(0, l - wl), max(0, t - wt)
    cw, ch = min(w - x0, r - l), min(h - y0, b - t)
    raw = buf.raw
    rows = [raw[(y0 + y) * w * 4 + x0 * 4:(y0 + y) * w * 4 + (x0 + cw) * 4] for y in range(ch)]
    return cw, ch, b"".join(rows)


def save_png(path, w, h, bgra):
    """Save BGRA bytes with Pillow (alpha dropped: PrintWindow leaves it undefined)."""
    from PIL import Image

    Image.frombuffer("RGBA", (w, h), bgra, "raw", "BGRA", 0, 1).convert("RGB").save(path, optimize=True)


def to_qimage(w, h, bgra):
    from PySide6.QtGui import QImage

    img = QImage(bgra, w, h, w * 4, QImage.Format_RGB32)
    return img.copy()  # detach from the Python buffer
