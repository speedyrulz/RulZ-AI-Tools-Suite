"""Caption files read whatever they were saved as (rulztrainer.caption_text): UTF-8 with and without BOM, UTF-16 and
UTF-32 with a BOM, Windows-1252; and the dataset reader uses it."""
import codecs
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rulztrainer.caption_text import decode_caption, read_caption_text  # noqa: E402

TEXT = "a woman in a café, “smiling” — naïve"


def test_every_encoding_reads_the_same_text(tmp_path):
    files = {
        "utf8": TEXT.encode("utf-8"),
        "utf8bom": codecs.BOM_UTF8 + TEXT.encode("utf-8"),
        "utf16le": codecs.BOM_UTF16_LE + TEXT.encode("utf-16-le"),
        "utf16be": codecs.BOM_UTF16_BE + TEXT.encode("utf-16-be"),
        "utf32le": codecs.BOM_UTF32_LE + TEXT.encode("utf-32-le"),
        "utf32be": codecs.BOM_UTF32_BE + TEXT.encode("utf-32-be"),
    }
    for name, raw in files.items():
        p = tmp_path / f"{name}.txt"
        p.write_bytes(raw)
        assert read_caption_text(p) == TEXT, name


def test_cp1252_and_edge_cases(tmp_path):
    p = tmp_path / "old.txt"
    p.write_bytes("café “x”".encode("cp1252"))  # not valid UTF-8
    assert read_caption_text(p) == "café “x”"
    p.write_bytes(b"\x81\x8d plain")  # bytes cp1252 leaves undefined: still no exception
    assert read_caption_text(p).endswith("plain")
    p.write_bytes(b"")
    assert read_caption_text(p) == ""
    assert decode_caption(b"plain ascii") == "plain ascii"
    assert decode_caption(codecs.BOM_UTF8) == ""


def test_dataset_reader_uses_it(tmp_path):
    from PIL import Image

    from rulztrainer.config import TrainConfig
    from rulztrainer.dataset import read_caption

    img = tmp_path / "a.png"
    Image.new("RGB", (8, 8)).save(img)
    cfg = TrainConfig(dataset_dir=str(tmp_path))
    (tmp_path / "a.txt").write_bytes(codecs.BOM_UTF16_LE + "x café".encode("utf-16-le"))
    assert read_caption(img, cfg, "") == "x café"
    (tmp_path / "a.txt").write_bytes(codecs.BOM_UTF8 + b"plain")
    assert read_caption(img, cfg, "") == "plain"
