"""4.7.0 review fixes: a capped dataset's bucket that draws no place, PNG text-chunk orientation, the public metadata
keeping numeric settings and dropping race overrides' private parts, ss_base_model of a local folder."""
import json
import random
import zlib
from pathlib import Path

from PIL import Image

from rulztrainer.config import TrainConfig
from rulztrainer.dataset import BucketBatcher, Item, png_has_exif_chunk
from rulztrainer.savemeta import base_model_name, public_config, public_overrides


def test_resample_with_no_places_is_empty():
    b = BucketBatcher.__new__(BucketBatcher)
    b.rng = random.Random(0)
    items = [object(), object()]
    assert b._resample(items, [1.0, 2.0], 0) == []


def _chunk(kind: bytes, data: bytes) -> bytes:
    return len(data).to_bytes(4, "big") + kind + data + zlib.crc32(kind + data).to_bytes(4, "big")


def _png_with_text(tmp_path: Path, keyword: bytes) -> Path:
    plain = tmp_path / "plain.png"
    Image.new("RGB", (8, 4)).save(plain)
    raw = plain.read_bytes()
    at = 8 + 25  # signature + IHDR chunk
    out = tmp_path / "text.png"
    out.write_bytes(raw[:at] + _chunk(b"zTXt", keyword + b"\0\0" + zlib.compress(b"x")) + raw[at:])
    return out


def test_raw_exif_profile_text_chunk_counts(tmp_path):
    assert png_has_exif_chunk(_png_with_text(tmp_path, b"Raw profile type exif"))
    assert png_has_exif_chunk(_png_with_text(tmp_path, b"XML:com.adobe.xmp"))
    assert not png_has_exif_chunk(_png_with_text(tmp_path, b"Comment"))


def test_numeric_settings_named_like_text_stay(tmp_path):
    cfg = TrainConfig()
    cfg.caption_dropout = 0.1
    cfg.max_token_length = 77
    cfg.trigger_word = "ohwx"
    cfg.dataset_dir = str(tmp_path / "secret_folder")
    pub = public_config(cfg)
    assert pub["caption_dropout"] == 0.1 and pub["max_token_length"] == 77
    text = json.dumps(pub)
    assert "secret_folder" not in text and "ohwx" not in text


def test_relative_local_folder_dropped(tmp_path, monkeypatch):
    (tmp_path / "models" / "sam3").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    cfg = TrainConfig()
    cfg.sam3_model = "models/sam3"
    assert "sam3_model" not in public_config(cfg)
    cfg.sam3_model = "facebook/sam3"  # an HF repo id stays
    assert public_config(cfg)["sam3_model"] == "facebook/sam3"


def test_race_overrides_lose_private_parts():
    out = public_overrides({"learning_rate": 1e-4, "trigger_word": "ohwx", "model_path": "D:\\models\\moss",
                            "lora_rank": 32})
    assert out == {"learning_rate": 1e-4, "lora_rank": 32}


def test_base_model_of_a_local_folder_is_its_name():
    cfg = TrainConfig()
    cfg.model_source = "hf"
    cfg.model_path = "D:\\models\\MOSS-Image"
    assert base_model_name(cfg) == "MOSS-Image"
    cfg.model_path = "org/repo"
    assert base_model_name(cfg) == "org/repo"
