# -*- mode: python ; coding: utf-8 -*-
# PyInstaller build file. Build with:  python -m PyInstaller --noconfirm "RulZ ComfyUI Launcher.spec"

a = Analysis(
    ["main.pyw"],
    pathex=[],
    binaries=[],
    datas=[
        ("rulz_comfy_launcher/builtin_catalog.json", "rulz_comfy_launcher"),
        ("rulz_comfy_launcher/scripts/introspect_args.py", "rulz_comfy_launcher/scripts"),
    ],
    hiddenimports=[],
    hookspath=[],
    runtime_hooks=[],
    # Large packages that may be installed next to PySide6 but are never used here.
    excludes=["torch", "numpy", "scipy", "matplotlib", "PIL", "tkinter", "pandas", "cv2",
              "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtQml",
              "PySide6.QtQuick", "PySide6.QtPdf", "PySide6.QtMultimedia"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="RulZ ComfyUI Launcher",
    debug=False,
    strip=False,
    upx=False,
    console=False,
    icon="assets/app.ico",
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="RulZ ComfyUI Launcher",
)
