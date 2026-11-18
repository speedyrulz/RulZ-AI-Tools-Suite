"""Caption files as text: whatever an editor saved them as.

A caption written by PowerShell's ">" or Notepad's "Unicode" is UTF-16 with a BOM, one saved with a BOM is UTF-8 with a
leading U+FEFF, and an older tool may have left Windows-1252. read_caption_text() reads all of them (never raises for a
file that can be read): BOM first, then strict UTF-8, then Windows-1252. Captions are always written as plain UTF-8."""
from __future__ import annotations

import codecs
from pathlib import Path

_BOMS = (  # longest first: the UTF-32 LE mark starts with the UTF-16 LE one
    (codecs.BOM_UTF32_LE, "utf-32"),
    (codecs.BOM_UTF32_BE, "utf-32"),
    (codecs.BOM_UTF8, "utf-8-sig"),
    (codecs.BOM_UTF16_LE, "utf-16"),
    (codecs.BOM_UTF16_BE, "utf-16"),
)


def decode_caption(raw: bytes) -> str:
    """The text of a caption file's bytes."""
    for bom, codec in _BOMS:
        if raw.startswith(bom):
            try:
                return raw.decode(codec)
            except UnicodeDecodeError:
                break  # a mark that is no mark: read it as plain text below
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace")  # the five bytes cp1252 leaves undefined become U+FFFD


def read_caption_text(path: str | Path) -> str:
    """The text of a caption file (OSError only when the file cannot be read at all)."""
    return decode_caption(Path(path).read_bytes())
