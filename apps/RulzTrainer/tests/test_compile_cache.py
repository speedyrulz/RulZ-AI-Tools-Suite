"""compile_cache: the compiled kernels saved to one file per torch / triton / GPU model / Python / operating system and
loaded on another machine."""
import logging
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rulztrainer import compiling as C  # noqa: E402


def test_the_file_name_names_the_python_version_and_operating_system(monkeypatch):
    pytest.importorskip("triton")
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda dev: "NVIDIA GeForce RTX 5060 Ti")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda dev: (12, 0))
    names = {}
    for platform in ("win32", "linux"):
        monkeypatch.setattr(sys, "platform", platform)
        names[platform] = C.cache_name("cuda:0")
    py = "py%d%d" % sys.version_info[:2]
    assert names["win32"].endswith(f"-NVIDIA-GeForce-RTX-5060-Ti-sm120-{py}-windows.bin")
    assert names["linux"].endswith(f"-NVIDIA-GeForce-RTX-5060-Ti-sm120-{py}-linux.bin")
    assert "+" not in names["linux"]


def test_a_file_for_another_machine_is_skipped_and_named_in_the_log(tmp_path, monkeypatch, caplog):
    loaded = []
    monkeypatch.setattr(C, "cache_name", lambda device: "k-linux.bin")
    monkeypatch.setattr(torch.compiler, "load_cache_artifacts", lambda b: loaded.append(b))
    bucket = tmp_path / "bucket"
    bucket.mkdir()
    (bucket / "k-windows.bin").write_bytes(b"windows kernels")
    with caplog.at_level(logging.INFO, logger="rulztrainer"):
        assert C.load_cache([tmp_path / "own", bucket], "cuda:0") is None
    assert loaded == [] and "k-windows.bin" in caplog.text and "k-linux.bin" in caplog.text


def test_cache_folders():
    assert C.cache_dirs("auto", "models") == [Path("models/.rulz_compile"), Path("models/bucket/.rulz_compile")]
    assert C.cache_dirs("off", "models") == [] and C.cache_dirs("", "models") == []
    assert C.cache_dirs("D:/cache", "models") == [Path("D:/cache")]


def test_saved_artifacts_are_loaded_from_the_first_folder_that_has_them(tmp_path, monkeypatch):
    loaded = []
    monkeypatch.setattr(C, "cache_name", lambda device: "k.bin")
    monkeypatch.setattr(torch.compiler, "save_cache_artifacts", lambda: (b"kernels", None))
    monkeypatch.setattr(torch.compiler, "load_cache_artifacts", lambda b: loaded.append(b))
    own, bucket = tmp_path / "own", tmp_path / "bucket"
    assert C.load_cache([own, bucket], "cuda:0") is None and loaded == []
    assert C.save_cache([bucket], "cuda:0") == bucket / "k.bin"
    assert C.load_cache([own, bucket], "cuda:0") == bucket / "k.bin" and loaded == [b"kernels"]
    assert C.save_cache([], "cuda:0") is None


def test_a_cache_torch_refuses_or_cannot_write_never_stops_the_run(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "cache_name", lambda device: "k.bin")
    (tmp_path / "k.bin").write_bytes(b"other torch")

    def refuse(b):
        raise RuntimeError("version mismatch")

    monkeypatch.setattr(torch.compiler, "load_cache_artifacts", refuse)
    assert C.load_cache([tmp_path], "cuda:0") is None
    loaded = []
    bucket = tmp_path / "bucket"
    bucket.mkdir()
    (bucket / "k.bin").write_bytes(b"good")
    monkeypatch.setattr(torch.compiler, "load_cache_artifacts",
                        lambda b: refuse(b) if b == b"other torch" else loaded.append(b))
    assert C.load_cache([tmp_path, bucket], "cuda:0") == bucket / "k.bin" and loaded == [b"good"]
    monkeypatch.setattr(torch.compiler, "save_cache_artifacts", lambda: None)  # nothing compiled
    assert C.save_cache([tmp_path / "new"], "cuda:0") is None
    monkeypatch.setattr(torch.compiler, "save_cache_artifacts", refuse)
    assert C.save_cache([tmp_path / "new"], "cuda:0") is None
