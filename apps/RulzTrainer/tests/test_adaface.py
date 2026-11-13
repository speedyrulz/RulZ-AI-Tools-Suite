"""AdaFace ViT (the default gold face recognizer): the weights file's layout, the pinned download, the gold embedder
by name and its display names. The real weights (460 MB) are not downloaded here; they were checked against Fizgig's
network code on 2,940 saved probe renders (cosine 1.0)."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import adaface, idloss  # noqa: E402


def _small_file(path: Path, prefix="model.net."):
    from safetensors.torch import save_file

    net = adaface.AdaFaceViT()
    torch.manual_seed(0)
    params = dict(net.named_parameters())
    state = {prefix + k: ((torch.randn_like(v) * 0.02) if k in params and v.dim() > 1 else v.clone()).contiguous()
             for k, v in net.state_dict().items() if v.is_floating_point()}  # batch-norm statistics as they are
    save_file(state, str(path))
    return path


def test_the_weights_file_loads_and_embeds(tmp_path):
    net = adaface.load(_small_file(tmp_path / "w.safetensors"))
    assert not net.training
    emb = idloss.Embedder(adaface.FlipSum(net), 112, 1.0, "adaface")
    x = torch.rand(2, 3, 112, 112) * 2 - 1
    with torch.no_grad():
        e = emb(x)
    assert e.shape == (2, 512) and torch.allclose(e.norm(dim=-1), torch.ones(2), atol=1e-4)
    with torch.no_grad():  # the mirror image gives the same embedding (the face and its flip are summed)
        assert torch.allclose(emb(x.flip(3)), e, atol=1e-4)


def test_a_file_with_other_keys_is_refused(tmp_path):
    with pytest.raises(RuntimeError, match="don't fit"):
        adaface.load(_small_file(tmp_path / "w.safetensors", prefix="other."))


def test_the_download_is_pinned_and_checked(tmp_path, monkeypatch):
    from rulztrainer import downloads

    assert adaface.REVISION in adaface.URL and len(adaface.SHA256) == 64
    got = []

    def fake(url, dest, label, progress=None):
        got.append(url)
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(b"not the weights")
        return dest

    monkeypatch.setattr(downloads, "download_url", fake)
    with pytest.raises(RuntimeError, match="SHA-256"):
        adaface.ensure_weights(tmp_path)
    assert got == [adaface.URL] and not (tmp_path / "adaface" / adaface.LOCAL_NAME).exists()
    # a file already in place is used as it is (no download)
    (tmp_path / "adaface").mkdir(exist_ok=True)
    (tmp_path / "adaface" / adaface.LOCAL_NAME).write_bytes(b"x")
    assert adaface.ensure_weights(tmp_path) == tmp_path / "adaface" / adaface.LOCAL_NAME and len(got) == 1


def test_the_gold_embedder_by_name(monkeypatch):
    monkeypatch.setattr(idloss, "adaface_embedder", lambda *a, **k: "ada")
    monkeypatch.setattr(idloss, "facenet_embedder", lambda *a, **k: "fn")
    assert idloss.gold_embedder("adaface") == "ada" and idloss.gold_embedder("facenet") == "fn"
    with pytest.raises(ValueError):
        idloss.gold_embedder("arcface")
    # a probe record from before 3.14.0 names no recognizer: it was FaceNet
    assert idloss.gold_label("adaface") == "AdaFace" and idloss.gold_label("") == "FaceNet"
    assert idloss.gold_label(None) == "FaceNet" and idloss.gold_label("facenet") == "FaceNet"
