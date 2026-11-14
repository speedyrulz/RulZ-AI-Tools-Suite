"""Write assets/app.ico (used for the packaged .exe) from the icon the app draws itself.

Run from the project folder:  python tools/make_icon.py
"""
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QBuffer, QByteArray, QIODevice  # noqa: E402
from PySide6.QtGui import QGuiApplication  # noqa: E402

from rulz_comfy_launcher.ui import theme  # noqa: E402

SIZES = (16, 24, 32, 48, 64, 128, 256)


def png_bytes(icon, size: int) -> bytes:
    data = QByteArray()
    buffer = QBuffer(data)
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    icon.pixmap(size, size).toImage().save(buffer, "PNG")
    buffer.close()
    return data.data()


def main() -> None:
    app = QGuiApplication.instance() or QGuiApplication(sys.argv)
    icon = theme.make_icon()
    images = [(size, png_bytes(icon, size)) for size in SIZES]

    # An .ico file is a small table followed by the images; PNG images are allowed.
    header = struct.pack("<HHH", 0, 1, len(images))
    offset = len(header) + 16 * len(images)
    table = b""
    for size, data in images:
        table += struct.pack("<BBBBHHII", size % 256, size % 256, 0, 0, 1, 32, len(data), offset)
        offset += len(data)

    target = ROOT / "assets" / "app.ico"
    target.parent.mkdir(exist_ok=True)
    target.write_bytes(header + table + b"".join(data for _size, data in images))
    print(f"wrote {target} ({target.stat().st_size} bytes)")
    del app


if __name__ == "__main__":
    main()
