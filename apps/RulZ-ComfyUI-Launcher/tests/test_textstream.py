from __future__ import annotations

import pytest

from rulz_comfy_launcher.textstream import LineSplitter, TextStream


def render(chunks, stream=None):
    """Apply the steps like a terminal would and return the lines."""
    stream = stream or TextStream()
    lines = [""]
    for chunk in chunks:
        for step in stream.feed(chunk):
            if step[0] == "text":
                lines[-1] += step[1]
            elif step[0] == "newline":
                lines.append("")
            elif step[0] == "restart":
                lines[-1] = ""
    return lines


def test_plain_text_and_newlines():
    ts = TextStream()
    assert ts.feed("abc\ndef\r\n") == [("text", "abc"), ("newline",), ("text", "def"), ("newline",)]
    assert ts.feed("") == []


def test_carriage_return_restarts_line():
    ts = TextStream()
    assert ts.feed("10%\r20%\r") == [("text", "10%"), ("restart",), ("text", "20%")]
    # The pending carriage return carries over to the next chunk.
    assert ts.feed("30%") == [("restart",), ("text", "30%")]


def test_carriage_return_before_newline_is_just_newline():
    ts = TextStream()
    assert ts.feed("abc\r") == [("text", "abc")]
    assert ts.feed("\n") == [("newline",)]
    assert ts.feed("x") == [("text", "x")]


def test_cr_crlf():
    assert TextStream().feed("a\r\r\nb") == [("text", "a"), ("newline",), ("text", "b")]


def test_progress_bar_render():
    assert render(["Loading\n", " 10%|#  |\r", " 50%|## |\r 100%|###|", "\ndone\n"]) == \
        ["Loading", " 100%|###|", "done", ""]


@pytest.mark.parametrize("raw, clean", [
    ("\x1b[31mred\x1b[0m", "red"),
    ("\x1b[1;32;40mbold\x1b[m", "bold"),
    ("\x1b[?25lhidden\x1b[?25h", "hidden"),
    ("\x1b[2K\x1b[1Gline", "line"),
    ("\x1b]0;window title\x07text", "text"),
    ("\x1b]0;window title\x1b\\text", "text"),
    ("a\x1bMb", "ab"),
    ("a\x1b7b\x1b8c", "abc"),
    ("a\x1b=b\x1b>c\x1bcd", "abcd"),
    ("a\x1b(Bb", "ab"),
    ("bell\x07 back\x08 nul\x00 del\x7f", "bell back nul del"),
    ("tab\tkept", "tab\tkept"),
])
def test_ansi_and_control_codes_removed(raw, clean):
    assert render([raw]) == [clean]


@pytest.mark.parametrize("split_at", range(1, 9))
def test_csi_split_across_chunks(split_at):
    data = "ab\x1b[1;31mcd\x1b[0m\n"
    assert render([data[:split_at], data[split_at:]]) == ["abcd", ""]


def test_csi_split_into_single_characters():
    data = "x\x1b[38;5;196my\x1b]0;t\x1b\\z\n"
    assert render(list(data)) == ["xyz", ""]


def test_osc_split_across_chunks():
    ts = TextStream()
    assert ts.feed("a\x1b]0;tit") == [("text", "a")]
    assert ts.feed("le\x07b") == [("text", "b")]


def test_osc_st_split_between_escape_and_backslash():
    ts = TextStream()
    assert ts.feed("a\x1b]0;title\x1b") == [("text", "a")]
    assert ts.feed("\\b") == [("text", "b")]


def test_unterminated_osc_does_not_swallow_lines():
    ts = TextStream()
    steps = ts.feed("\x1b]0;title\nhello\n")
    assert ("text", "hello") in steps


def test_decode_split_multibyte_utf8():
    ts = TextStream()
    data = "Grüße € \U0001F600\n".encode("utf-8")
    out = "".join(ts.decode(data[i:i + 1]) for i in range(len(data)))
    assert out == "Grüße € \U0001F600\n"


def test_decode_invalid_bytes_replaced():
    assert TextStream().decode(b"a\xffb") == "a�b"


def test_decode_then_feed():
    ts = TextStream()
    data = "\x1b[32m✓ ok\x1b[0m\n".encode("utf-8")
    steps = []
    for i in range(len(data)):
        steps += ts.feed(ts.decode(data[i:i + 1]))
    text = "".join(s[1] for s in steps if s[0] == "text")
    assert text == "✓ ok" and steps[-1] == ("newline",)


# -- LineSplitter --------------------------------------------------------------
def test_line_splitter_basic():
    ls = LineSplitter()
    assert ls.feed("abc") == []
    assert ls.feed("def\nghi\r\njkl\rmno") == ["abcdef", "ghi", "jkl"]
    assert ls.feed("\n") == ["mno"]


def test_line_splitter_crlf_across_chunks_gives_no_empty_line():
    ls = LineSplitter()
    assert ls.feed("one\r") == ["one"]
    assert ls.feed("\ntwo\n") == ["two"]


def test_line_splitter_skips_empty_lines():
    assert LineSplitter().feed("\n\na\n\n") == ["a"]


def test_line_splitter_limit():
    ls = LineSplitter(limit=10)
    assert ls.feed("x" * 25) == []
    assert ls.feed("END\n") == ["x" * 10 + "END"]


def test_line_splitter_pid_line():
    from rulz_comfy_launcher.command import PID_PATTERN
    ls = LineSplitter()
    lines = ls.feed("[launcher] pi") + ls.feed("d=4242\r\n")
    assert lines == ["[launcher] pid=4242"]
    assert PID_PATTERN.match(lines[0]).group(1) == "4242"
