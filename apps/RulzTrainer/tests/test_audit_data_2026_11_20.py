"""Regression tests of the 2026-11-20 data-side audit: clip rotation, cache_latents, SDXL text cache key, the ready
checks of the loaders, look filter."""
import shutil
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import clips, dataset  # noqa: E402
from rulztrainer.config import TrainConfig  # noqa: E402

FFMPEG = shutil.which("ffmpeg")


# ---------------------------------------------------------------- B13 clip rotation


def _marker_clip(tmp_path: Path, rotation: int) -> Path:
    """320x180 clip with a red block top-left, a display-matrix rotation of `rotation` (counter-clockwise)."""
    pat = np.full((180, 320, 3), 40, np.uint8)
    pat[0:60, 0:100] = (255, 0, 0)
    Image.fromarray(pat).save(tmp_path / "pat.png")
    base, out = tmp_path / "base.mp4", tmp_path / f"r{rotation}.mp4"
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-loop", "1", "-i", str(tmp_path / "pat.png"), "-t", "1",
                    "-r", "10", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(base)], check=True)
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-display_rotation", str(rotation), "-i", str(base),
                    "-c", "copy", str(out)], check=True)
    return out


def _red_centre(img) -> tuple[float, float]:
    a = np.asarray(img.convert("RGB"))
    ys, xs = np.where((a[..., 0] > 200) & (a[..., 1] < 60))
    return float(ys.mean()), float(xs.mean())


@pytest.mark.skipif(FFMPEG is None, reason="ffmpeg is not installed")
@pytest.mark.parametrize("rotation", [90, 270])
def test_display_matrix_rotation_is_applied_like_ffmpeg_does(tmp_path, rotation):
    pytest.importorskip("av")
    clip = _marker_clip(tmp_path, rotation)
    png = tmp_path / "auto.png"
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(clip), "-frames:v", "1", str(png)], check=True)
    want = Image.open(png)
    w, h, _, _ = clips.video_info(clip)
    assert (w, h) == want.size == (180, 320)  # the size of the upright picture
    got = clips.read_frames(clip, 3, 10.0)[0]
    assert got.size == want.size
    wy, wx = _red_centre(want)
    gy, gx = _red_centre(got)
    assert abs(wy - gy) < 8 and abs(wx - gx) < 8  # the marker sits in the same corner


@pytest.mark.skipif(FFMPEG is None, reason="ffmpeg is not installed")
def test_load_clip_fit_matches_the_old_hold_every_frame_path(tmp_path):
    """P3: fitting each frame as it is decoded gives exactly the tensor the full-size path gave."""
    pytest.importorskip("av")
    from PIL import ImageOps

    clip = _marker_clip(tmp_path, 90)
    cfg = TrainConfig()
    got = dataset.load_clip_fit(clip, 4, 96, 160, cfg, 10.0)
    pics = clips.read_frames(clip, 4, 10.0, cfg.train_video_mode)
    arr = np.stack([np.asarray(ImageOps.fit(dataset.to_rgb(im), (96, 160), method=Image.Resampling.LANCZOS,
                                            centering=(0.5, 0.5))) for im in pics])
    want = torch.from_numpy(arr).permute(3, 0, 1, 2).float() / 127.5 - 1.0
    assert got.shape == want.shape and torch.equal(got, want)


# ---------------------------------------------------------------- B14 / P2 cache_latents


class _Vae:
    device = "cpu"


def _arch(encodes):
    def encode_images(v, t):
        encodes.append(1)
        return torch.zeros(1, 4, 2, 2)

    return types.SimpleNamespace(encode_images=encode_images, video_fps=24.0)


def _item(tmp_path, name, image, flip=True):
    d = tmp_path / "lat"
    d.mkdir(exist_ok=True)
    return dataset.Item(image=image, caption="c", width=16, height=16, latent_path=d / f"{name}.safetensors",
                        latent_flip_path=(d / f"{name}_flip.safetensors") if flip else None,
                        text_path=tmp_path / "t.safetensors")


def test_a_truncated_picture_is_named_in_the_error(tmp_path, monkeypatch):
    good = tmp_path / "ok.png"
    Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)).save(good)
    bad = tmp_path / "cut.png"
    bad.write_bytes(good.read_bytes()[:200])  # header intact, data cut off
    monkeypatch.setattr(dataset, "get_arch", lambda cfg: _arch([]))
    with pytest.raises(Exception) as e:
        dataset.cache_latents([_item(tmp_path, "a", bad)], TrainConfig(flip_augment=True), _Vae())
    assert "cut.png" in str(e.value) and "remove or replace" in str(e.value)


def test_a_missing_flip_latent_alone_does_not_re_encode_the_plain_one(tmp_path, monkeypatch):
    from safetensors.torch import load_file

    from rulztrainer.atomic import save_safetensors

    img = tmp_path / "ok.png"
    Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)).save(img)
    it = _item(tmp_path, "a", img)
    save_safetensors({"latent": torch.ones(4, 2, 2)}, it.latent_path)
    encodes = []
    monkeypatch.setattr(dataset, "get_arch", lambda cfg: _arch(encodes))
    dataset.cache_latents([it], TrainConfig(flip_augment=True), _Vae())
    assert len(encodes) == 1  # the flip only
    assert torch.equal(load_file(str(it.latent_path))["latent"], torch.ones(4, 2, 2))  # untouched
    assert it.latent_flip_path.exists()


# ---------------------------------------------------------------- B15 SDXL text cache key


def test_sdxl_text_cache_follows_the_checkpoint_others_are_unchanged(tmp_path):
    a, b = tmp_path / "pony.safetensors", tmp_path / "illustrious.safetensors"
    a.write_bytes(b"x" * 10)
    b.write_bytes(b"x" * 10)

    def cfg(f, arch="sdxl"):
        return TrainConfig(model_arch=arch, model_source="files", transformer_file=str(f),
                           cache_dir=str(tmp_path / "c"))

    pa, pb = dataset.text_cache_path(cfg(a), "hello"), dataset.text_cache_path(cfg(b), "hello")
    assert pa != pb and pa == dataset.text_cache_path(cfg(a), "hello")
    a.write_bytes(b"x" * 11)  # same name, different file
    assert dataset.text_cache_path(cfg(a), "hello") != pa
    hf = dataset.text_cache_path(TrainConfig(model_arch="sdxl", cache_dir=str(tmp_path / "c")), "hello")
    assert hf.name.endswith("_sdxl_wo.safetensors")  # a Hugging Face folder keeps its name
    assert dataset.text_cache_path(cfg(a, "moss_image"), "hello") == dataset.text_cache_path(cfg(b, "moss_image"),
                                                                                           "hello")


# ---------------------------------------------------------------- B16 ready checks of the loaders


@pytest.mark.parametrize("mod,dirs", [("moss", ("transformer", "text_encoder", "vae")),
                                      ("ideogram4", ("transformer", "text_encoder", "vae")),
                                      ("ph3", ("transformer", "text_connector", "text_encoder", "vae"))])
def test_a_download_missing_a_component_is_resumed(tmp_path, monkeypatch, mod, dirs):
    import importlib

    loading = importlib.import_module(f"rulztrainer.{mod}.loading")
    models = tmp_path / "models"
    local = models / "org__model"
    for d in dirs:
        (local / d).mkdir(parents=True)
        (local / d / "config.json").write_text("{}")
        (local / d / "model.safetensors").write_bytes(b"0")
    assert loading.resolve_model_dir("org/model", str(models)) == local.resolve()  # all there: no download
    shutil.rmtree(local / dirs[-1])  # the last component never arrived
    called = []
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: called.append(1))
    loading.resolve_model_dir("org/model", str(models))
    assert called


# ---------------------------------------------------------------- B17 / B18 look filter


def test_a_caption_shared_by_two_images_stays_with_the_one_left_behind(tmp_path):
    from rulztrainer import lookfilter

    (tmp_path / "a.png").write_bytes(b"p")
    (tmp_path / "a.jpg").write_bytes(b"j")
    (tmp_path / "a.txt").write_text("cap")
    moved, errors = lookfilter.move_out(tmp_path, [tmp_path / "a.png"])
    assert moved and not errors
    assert (tmp_path / "a.txt").read_text() == "cap"  # a.jpg keeps its caption
    assert (tmp_path / "excluded_by_look" / "a.txt").read_text() == "cap"
    lookfilter.move_out(tmp_path, [tmp_path / "a.jpg"])  # the last one takes it along
    assert not (tmp_path / "a.txt").exists()
    assert (tmp_path / "excluded_by_look" / "a.txt").read_text() == "cap"
    assert not (tmp_path / "excluded_by_look" / "a_2.txt").exists()
    lookfilter.move_back(tmp_path)
    assert (tmp_path / "a.txt").read_text() == "cap"
    assert (tmp_path / "a.png").exists() and (tmp_path / "a.jpg").exists()


def test_save_scores_writes_atomically(tmp_path, monkeypatch):
    from rulztrainer import lookfilter

    base = tmp_path / "a.png"
    base.write_bytes(b"p")
    lookfilter.save_scores(tmp_path, [base], {base: 0.5})
    good = (tmp_path / "look_scores.json").read_text(encoding="utf-8")
    real = Path.write_text

    def failing(self, *a, **k):
        if self.name.endswith(".tmp"):
            real(self, "{torn", encoding="utf-8")
            raise OSError("disk full")
        return real(self, *a, **k)

    monkeypatch.setattr(Path, "write_text", failing)
    with pytest.raises(OSError):
        lookfilter.save_scores(tmp_path, [base], {base: 0.9})
    monkeypatch.undo()
    assert (tmp_path / "look_scores.json").read_text(encoding="utf-8") == good
    assert not list(tmp_path.glob("*.tmp"))
