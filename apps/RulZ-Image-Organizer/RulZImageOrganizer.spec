# PyInstaller spec – build with:  .\build.ps1
# -*- mode: python ; coding: utf-8 -*-

a = Analysis(
    ["run.py"],
    pathex=[],
    binaries=[],
    datas=[("models/*.onnx", "models"), ("assets/icon.ico", "assets"), ("assets/icon.png", "assets"),
           ("assets/logo_full_white.png", "assets"), ("assets/logo_full.png", "assets")],
    hiddenimports=["pillow_heif"],
    excludes=[
        "tkinter", "matplotlib", "scipy", "pandas", "IPython", "pytest",
        "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtWebEngineQuick",
        "PySide6.Qt3DCore", "PySide6.QtQuick", "PySide6.QtQml", "PySide6.QtMultimedia",
        "PySide6.QtCharts", "PySide6.QtDataVisualization", "PySide6.QtPdf", "PySide6.QtBluetooth",
        "PySide6.QtSql", "PySide6.QtTest", "PySide6.QtDesigner",
    ],
    noarchive=False,
)
# Drop large DLLs this app never loads (video decoding, software OpenGL, Qt Quick/QML/PDF).
_UNUSED = ("opencv_videoio_ffmpeg", "opengl32sw", "qt6quick", "qt6qml", "qt6pdf", "qt6virtualkeyboard",
           "qt6opengl", "qt6network")
a.binaries = [b for b in a.binaries if not any(u in b[0].lower().replace("\\", "/").split("/")[-1] for u in _UNUSED)]
a.datas = [d for d in a.datas if "/translations/" not in d[0].replace("\\", "/").lower()]
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="RulZ Image Organizer",
    icon="assets/icon.ico",
    console=False,
    upx=False,
    version=None,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="RulZ Image Organizer")
