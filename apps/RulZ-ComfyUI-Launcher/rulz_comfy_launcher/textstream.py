"""Turn raw process output into simple display steps.

ComfyUI prints progress bars by sending a carriage return and writing the line
again, and some tools add ANSI colour codes. ``TextStream`` removes the codes
and reports what a terminal would do, as a list of steps:

    ("text", "abc")   write text at the end of the current line
    ("newline",)      start a new line
    ("restart",)      the next text replaces the current line (carriage return)

No Qt imports here, so this module can be unit-tested on its own.
"""
from __future__ import annotations

import codecs
import re

# CSI sequences (colours, cursor moves), OSC sequences (window titles), escapes with
# intermediate bytes (character sets), and two-byte escapes (save cursor, reset, ...).
# An OSC sequence never spans lines: one without an end stops at the line break.
_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]"
                   r"|\x1b\][^\x07\x1b\r\n]*(?:\x07|\x1b\\)?"
                   r"|\x1b[ -/]+[0-~]"
                   r"|\x1b[0-~]")
# An escape sequence that is cut off at the end of a chunk. The length limit keeps
# a broken sequence from holding text back for ever.
_ANSI_PARTIAL = re.compile(r"\x1b(?:\[[0-9;?]{0,64}[ -/]{0,8}|\][^\x07\x1b\r\n]{0,256}\x1b?|[ -/]{1,8})?$")
_OTHER_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_BREAKS = re.compile(r"(\r\n|\n|\r)")


class TextStream:
    def __init__(self):
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._held = ""           # text kept back because an escape sequence may continue
        self._pending_cr = False  # a carriage return was seen and nothing followed it yet

    def decode(self, data: bytes) -> str:
        """Bytes to text. Handles multi-byte characters split across chunks."""
        return self._decoder.decode(data)

    def feed(self, text: str) -> list[tuple]:
        """Clean one chunk of text and return the display steps for it."""
        text = self._held + text
        self._held = ""
        partial = _ANSI_PARTIAL.search(text)
        if partial and partial.group(0):
            self._held = partial.group(0)
            text = text[: partial.start()]
        text = _ANSI.sub("", text)
        text = _OTHER_CONTROL.sub("", text)

        steps: list[tuple] = []
        for part in _BREAKS.split(text):
            if not part:
                continue
            if part in ("\n", "\r\n"):
                self._pending_cr = False
                steps.append(("newline",))
            elif part == "\r":
                self._pending_cr = True
            else:
                if self._pending_cr:
                    self._pending_cr = False
                    steps.append(("restart",))
                steps.append(("text", part))
        return steps


class LineSplitter:
    """Collects finished lines from a stream, for pattern matching on whole lines."""

    def __init__(self, limit: int = 4000):
        self._buffer = ""
        self._limit = limit

    def feed(self, text: str) -> list[str]:
        self._buffer += text
        parts = re.split(r"\r\n|\n|\r", self._buffer)
        self._buffer = parts.pop()
        if len(self._buffer) > self._limit:
            self._buffer = self._buffer[-self._limit:]
        return [p for p in parts if p]
