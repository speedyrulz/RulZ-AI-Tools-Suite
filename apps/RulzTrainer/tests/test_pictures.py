"""Transparency laid on white (rulztrainer.pictures): RGBA, LA and palette transparency, plain pictures untouched, the
training loader, a transparent picture's latents named apart from those cached before; and telemetry off on import."""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rulztrainer.dataset import latent_key, load_image_fit, upright_header  # noqa: E402
from rulztrainer.pictures import file_has_alpha, has_alpha, to_rgb  # noqa: E402
from rulztrainer.regions import _file_key  # noqa: E402


def _cutout(mode: str = "RGBA") -> Image.Image:
    """A red square on fully transparent pixels that store black (what a cut-out PNG often holds)."""
    a = np.zeros((32, 32, 4), dtype=np.uint8)
    a[8:24, 8:24] = (255, 0, 0, 255)
    im = Image.fromarray(a, "RGBA")
    return im if mode == "RGBA" else im.convert(mode)


def test_transparent_pixels_become_white():
    out = np.asarray(to_rgb(_cutout()))
    assert out.shape == (32, 32, 3)
    assert (out[0, 0] == 255).all() and tuple(out[16, 16]) == (255, 0, 0)
    assert tuple(np.asarray(_cutout().convert("RGB"))[0, 0]) == (0, 0, 0)  # what convert("RGB") gave
    la = Image.fromarray(np.dstack([np.full((8, 8), 40, np.uint8), np.zeros((8, 8), np.uint8)]), "LA")
    assert (np.asarray(to_rgb(la)) == 255).all()


def test_half_transparent_pixels_blend_on_white():
    a = np.zeros((4, 4, 4), dtype=np.uint8)
    a[...] = (0, 0, 0, 128)
    v = np.asarray(to_rgb(Image.fromarray(a, "RGBA")))[0, 0]
    assert 120 <= int(v[0]) <= 130  # half black on white: mid grey


def test_palette_transparency_and_plain_pictures():
    pal = Image.new("P", (4, 4), 0)
    pal.putpalette([0, 0, 0, 255, 0, 0] + [0] * 762)
    pal.putpixel((1, 1), 1)
    pal.info["transparency"] = 0
    assert has_alpha(pal)
    out = np.asarray(to_rgb(pal))
    assert tuple(out[0, 0]) == (255, 255, 255) and tuple(out[1, 1]) == (255, 0, 0)
    rgb = Image.new("RGB", (4, 4), (10, 20, 30))
    assert to_rgb(rgb) is rgb and not has_alpha(rgb)
    grey = Image.new("L", (4, 4), 70)
    assert to_rgb(grey).mode == "RGB" and tuple(np.asarray(to_rgb(grey))[0, 0]) == (70, 70, 70)


def test_training_loader_and_latent_names(tmp_path):
    png, jpg = tmp_path / "cut.png", tmp_path / "photo.jpg"
    _cutout().save(png)
    Image.new("RGB", (32, 32), (0, 128, 0)).save(jpg)
    img = np.asarray(load_image_fit(png, 16, 16))
    assert img.shape == (16, 16, 3) and (img[0, 0] >= 250).all()
    assert file_has_alpha(png) and not file_has_alpha(jpg)
    assert upright_header(png) == (32, 32, True) and upright_header(jpg) == (32, 32, False)
    # a transparent picture's latents get a new name, so latents cached on its black are encoded again once
    assert latent_key(png) != _file_key(png) and latent_key(png, False) == _file_key(png)
    assert latent_key(jpg) == _file_key(jpg)


def _env_after_import(extra: dict) -> list[str]:
    env = {k: v for k, v in os.environ.items() if k not in ("ORT_DISABLE_TELEMETRY", "HF_HUB_DISABLE_TELEMETRY")}
    env.update(extra)
    code = ("import os, rulztrainer; print(os.environ['ORT_DISABLE_TELEMETRY'], "
            "os.environ['HF_HUB_DISABLE_TELEMETRY'])")
    out = subprocess.run([sys.executable, "-c", code], env=env, cwd=ROOT, capture_output=True, text=True, check=True)
    return out.stdout.split()


def test_telemetry_off_on_import_and_a_set_value_wins():
    assert _env_after_import({}) == ["1", "1"]
    assert _env_after_import({"ORT_DISABLE_TELEMETRY": "0"}) == ["0", "1"]
    from rulztrainer import onnxruntime_telemetry_off

    onnxruntime_telemetry_off()  # never raises, with or without onnxruntime
