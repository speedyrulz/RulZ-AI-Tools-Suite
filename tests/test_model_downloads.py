"""Model folders count as downloaded only when whole: an interrupted Hugging Face download (its configs and shard
indexes come first) is resumed, not loaded with shards missing; an interrupted InsightFace extraction is redone."""
import json
import sys
import types
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import downloads as D  # noqa: E402
from rulztrainer.moss import loading as moss_loading  # noqa: E402


def _component(folder: Path, shards: list[str], present: list[str]) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.json").write_text("{}", encoding="utf-8")
    if shards:
        index = {"weight_map": {f"w{i}": s for i, s in enumerate(shards)}}
        (folder / "diffusion_pytorch_model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    for name in present:
        (folder / name).write_bytes(b"x")
    return folder


def test_component_ready_needs_every_shard(tmp_path):
    shards = ["a-00001.safetensors", "a-00002.safetensors"]
    assert not D.component_ready(tmp_path / "nothing")
    assert not D.component_ready(_component(tmp_path / "partial", shards, shards[:1]))
    assert D.component_ready(_component(tmp_path / "whole", shards, shards))
    assert not D.component_ready(_component(tmp_path / "single_missing", [], []))
    assert D.component_ready(_component(tmp_path / "single", [], ["model.safetensors"]))


def test_an_interrupted_download_is_resumed(tmp_path, monkeypatch):
    import huggingface_hub

    local = tmp_path / "org__model"
    shards = ["a-00001.safetensors", "a-00002.safetensors"]
    _component(local / "transformer", shards, shards[:1])  # the configs and the first shard: then it stopped
    calls = []

    def fake_snapshot(repo, local_dir, **kw):
        calls.append(repo)
        _component(Path(local_dir) / "transformer", shards, shards)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
    assert moss_loading.resolve_model_dir("org/model", str(tmp_path)) == local.resolve()
    assert calls == ["org/model"]
    moss_loading.resolve_model_dir("org/model", str(tmp_path))  # whole now: no second download
    assert calls == ["org/model"]


def test_insightface_pack_is_extracted_whole_or_not_at_all(tmp_path, monkeypatch):
    zip_src = tmp_path / "src.zip"
    with zipfile.ZipFile(zip_src, "w") as zf:
        zf.writestr("det_10g.onnx", b"d")
        zf.writestr("w600k_r50.onnx", b"r")
    storage = types.SimpleNamespace(model_zoo_download_url=lambda name: name)
    monkeypatch.setitem(sys.modules, "insightface", types.ModuleType("insightface"))
    monkeypatch.setitem(sys.modules, "insightface.utils", types.SimpleNamespace(storage=storage))
    monkeypatch.setattr(D, "download_url", lambda url, dest, label, progress=None: dest.write_bytes(
        zip_src.read_bytes()) or dest)
    root = tmp_path / "if"
    left = root / "models" / "buffalo_l.extracting"  # what an interrupted extraction leaves: never the pack itself
    left.mkdir(parents=True)
    (left / "det_10g.onnx").write_bytes(b"half")
    pack = D.ensure_insightface_pack("buffalo_l", str(root))
    assert sorted(p.name for p in pack.glob("*.onnx")) == ["det_10g.onnx", "w600k_r50.onnx"]
    assert not left.exists() and (pack / "det_10g.onnx").read_bytes() == b"d"


def test_snapshot_byte_progress_is_reported_when_stderr_is_no_terminal(capsys):
    """huggingface_hub builds its bars with disable=None, which tqdm turns off without a terminal (train.log): the
    log then sat at the opening 0%. The bars are built here the way snapshot_download builds them (1.x / 2.x)."""
    try:
        from huggingface_hub.utils.tqdm import _create_progress_bar
    except ImportError:  # older huggingface_hub: the plain constructor with the same auto-detect
        def _create_progress_bar(*, cls, log_level, name=None, **kw):
            return cls(disable=None, **kw)

    seen = []
    cls = D._reporting_tqdm("the transformer", lambda text, frac: seen.append((text, frac)))
    transfer = _create_progress_bar(cls=cls, log_level=20, name="t", desc="Downloading bytes", total=0, unit="B")
    disk = _create_progress_bar(cls=cls, log_level=20, name="s", desc="Reconstructing (incomplete total...)",
                                total=0, unit="B")
    transfer.total = disk.total = 1_000_000_000
    for _ in range(100):  # 100 chunks of 10 MB
        transfer.update(10_000_000)
        disk.update(10_000_000)
    transfer.close()
    disk.close()
    fracs = [f for _, f in seen]
    assert fracs and fracs[-1] == 1.0 and fracs == sorted(fracs)
    assert len(fracs) <= 51  # a line per 2% (or 15 s), not one per chunk
    assert all("1.00 GB" in t for t, _ in seen)  # only the disk bar: the network bar's total is a guess
    assert capsys.readouterr().err == ""  # the bars themselves draw nothing


def test_small_files_and_file_counters_stay_quiet():
    seen = []
    cls = D._reporting_tqdm("x", lambda text, frac: seen.append(frac))
    cfg = cls(total=5_000, unit="B", desc="Reconstructing")
    cfg.update(5_000)
    files = cls(total=3, desc="Fetching 3 files")
    files.update(1)
    assert seen == []
