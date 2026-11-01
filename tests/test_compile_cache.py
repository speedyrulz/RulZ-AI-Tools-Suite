"""compile_cache: the compiled kernels saved to one file per torch / triton / GPU model and loaded on another machine."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rulztrainer import compiling as C  # noqa: E402


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
    monkeypatch.setattr(torch.compiler, "save_cache_artifacts", lambda: None)  # nothing compiled
    assert C.save_cache([tmp_path / "new"], "cuda:0") is None
    monkeypatch.setattr(torch.compiler, "save_cache_artifacts", refuse)
    assert C.save_cache([tmp_path / "new"], "cuda:0") is None
