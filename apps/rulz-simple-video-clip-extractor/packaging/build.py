"""Build the portable Windows release zip.

    .venv\\Scripts\\python -m pip install pyinstaller
    .venv\\Scripts\\python packaging\\build.py

Output: dist\\RulZ-Simple-Video-Clip-Extractor-v<version>-win64.zip
"""
from __future__ import annotations

import os
import re
import shutil
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(ROOT, "packaging")
EXE_NAME = "RulZ Clip Extractor"


def version() -> str:
    text = open(os.path.join(ROOT, "simplecut", "__init__.py"), encoding="utf-8").read()
    return re.search(r'__version__\s*=\s*"([^"]+)"', text).group(1)


def main():
    import PyInstaller.__main__

    ver = version()
    os.chdir(ROOT)
    for d in ("build", os.path.join("dist", EXE_NAME)):
        shutil.rmtree(d, ignore_errors=True)

    sep = os.pathsep
    PyInstaller.__main__.run([
        "main.py",
        "--name", EXE_NAME,
        "--noconfirm",
        "--windowed",
        "--icon", os.path.join("simplecut", "assets", "icon.ico"),
        "--add-data", f"{os.path.join('simplecut', 'assets')}{sep}{'simplecut/assets'}",
        # Qt modules the app never uses; keeps the download smaller.
        "--exclude-module", "PySide6.QtWebEngineCore",
        "--exclude-module", "PySide6.QtWebEngineWidgets",
        "--exclude-module", "PySide6.Qt3DCore",
        "--exclude-module", "PySide6.QtQuick",
        "--exclude-module", "PySide6.QtQml",
        "--exclude-module", "PySide6.QtPdf",
        "--exclude-module", "tkinter",
    ])

    out = os.path.join(ROOT, "dist", EXE_NAME)
    shutil.copy(os.path.join(ROOT, "LICENSE"), os.path.join(out, "LICENSE.txt"))
    shutil.copy(os.path.join(ROOT, "simplecut", "assets", "LOGO-NOTICE.md"), os.path.join(out, "LOGO-NOTICE.md"))
    shutil.copy(os.path.join(PKG, "THIRD-PARTY-NOTICES.txt"), out)
    shutil.copy(os.path.join(PKG, "README.txt"), out)

    top = f"RulZ-Simple-Video-Clip-Extractor-v{ver}"
    zip_path = os.path.join(ROOT, "dist", f"{top}-win64.zip")
    if os.path.exists(zip_path):
        os.remove(zip_path)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for dirpath, _dirs, files in os.walk(out):
            for f in files:
                full = os.path.join(dirpath, f)
                z.write(full, os.path.join(top, os.path.relpath(full, out)))
    print(f"\nBuilt {zip_path} ({os.path.getsize(zip_path) / 1e6:.1f} MB)")


if __name__ == "__main__":
    sys.exit(main())
