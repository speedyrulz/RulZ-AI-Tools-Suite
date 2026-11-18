"""Transparency laid on white (rulztrainer.pictures): RGBA, LA and palette transparency, plain pictures untouched, the
training loader, a transparent picture's latents named apart from those cached before; and telemetry off on import."""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
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


# colour profiles and 16-bit grey

def _wide_profile_bytes() -> bytes:
    """A wide-gamut RGB profile built offline: sRGB's, with Display P3 primaries written into its colorant tags and a
    name that no longer says sRGB."""
    import struct

    from PIL import ImageCms

    data = bytearray(ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes())
    p3 = {b"rXYZ": (0.5151, 0.2412, -0.0011), b"gXYZ": (0.2920, 0.6922, 0.0419), b"bXYZ": (0.1571, 0.0666, 0.7841)}
    for i in range(struct.unpack(">I", data[128:132])[0]):
        sig, off, size = struct.unpack(">4sII", data[132 + 12 * i:144 + 12 * i])
        if sig in p3:
            data[off + 8:off + 20] = struct.pack(">iii", *(round(v * 65536) for v in p3[sig]))
        elif sig == b"desc":
            body = bytes(data[off:off + size])
            for old, new in ((b"sRGB", b"Disp"), ("sRGB".encode("utf-16-be"), "Disp".encode("utf-16-be"))):
                at = body.find(old)
                if at > 0:
                    data[off + at:off + at + len(old)] = new
    return bytes(data)


def _profile_picture(path: Path, profile: str | None, mode: str = "RGB") -> None:
    """A saturated gradient PNG with a profile embedded: "srgb", "wide" (see above) or none."""
    from PIL import ImageCms

    g = np.linspace(0, 255, 32, dtype=np.uint8)
    a = np.stack([np.tile(g, (32, 1)), np.tile(g[::-1], (32, 1)), np.full((32, 32), 90, np.uint8)], axis=2)
    im = Image.fromarray(a, "RGB")
    if mode == "RGBA":
        im.putalpha(Image.new("L", im.size, 200))
    icc = {None: None, "srgb": ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes(),
           "wide": _wide_profile_bytes()}[profile]
    im.save(path, **({"icc_profile": icc} if icc else {}))


def test_non_srgb_profile_is_converted(tmp_path):
    from rulztrainer.pictures import colour_tag, file_colour_tag

    p = tmp_path / "wide.png"
    _profile_picture(p, "wide")
    with Image.open(p) as im:
        raw = np.asarray(im.convert("RGB")).astype(int)
        assert colour_tag(im) == "|icc"
        out = to_rgb(im)
    arr = np.asarray(out).astype(int)
    assert out.mode == "RGB" and "icc_profile" not in out.info
    assert np.abs(arr - raw).max() > 10  # the wide-gamut values moved to where sRGB shows them
    assert file_colour_tag(p) == "|icc"
    assert latent_key(p) == _file_key(p) + "|icc"
    assert np.asarray(load_image_fit(p, 32, 32)).astype(int).tolist() == arr.tolist()


def test_rgba_profile_keeps_alpha(tmp_path):
    from rulztrainer.pictures import _to_srgb

    p = tmp_path / "widea.png"
    _profile_picture(p, "wide", "RGBA")
    with Image.open(p) as im:
        assert im.mode == "RGBA"
        out = _to_srgb(im)
        white = to_rgb(im)
    assert out.mode == "RGBA" and "icc_profile" not in out.info and (np.asarray(out)[..., 3] == 200).all()
    assert white.mode == "RGB"


def test_srgb_or_no_profile_is_untouched(tmp_path):
    from rulztrainer.pictures import colour_tag

    for name, prof in (("none", None), ("srgb", "srgb")):
        p = tmp_path / f"{name}.png"
        _profile_picture(p, prof)
        with Image.open(p) as im:
            assert colour_tag(im) == ""
            before = np.asarray(im).copy()
            out = to_rgb(im)
            assert out is im and np.array_equal(np.asarray(out), before)
        assert latent_key(p) == _file_key(p)  # latents cached before stay valid


def test_unreadable_profile_leaves_the_picture(tmp_path):
    p = tmp_path / "bad.png"
    Image.new("RGB", (8, 8), (200, 10, 10)).save(p, icc_profile=b"not a profile at all" * 20)
    with Image.open(p) as im:
        out = to_rgb(im)
        assert out is im and tuple(np.asarray(out)[0, 0]) == (200, 10, 10)
    assert latent_key(p) == _file_key(p)


def test_cmyk_without_a_profile_converts_as_before():
    out = to_rgb(Image.new("CMYK", (4, 4), (0, 255, 255, 0)))
    assert out.mode == "RGB" and tuple(np.asarray(out)[0, 0]) == tuple(np.asarray(Image.new("CMYK", (1, 1), (0, 255, 255, 0)).convert("RGB"))[0, 0])


def test_16_bit_grey_png_scales_over_the_full_range(tmp_path):
    from rulztrainer.pictures import colour_tag

    v = np.tile(np.linspace(0, 65535, 256, dtype=np.uint16), (4, 1))
    p = tmp_path / "grey16.png"
    Image.fromarray(v).save(p)
    with Image.open(p) as im:
        assert im.mode in ("I;16", "I;16L", "I;16B", "I")
        assert colour_tag(im) == "|grey16"
        out = to_rgb(im)
    arr = np.asarray(out)
    assert out.mode == "RGB" and arr[0, 0, 0] == 0 and arr[0, -1, 0] == 255
    assert (arr[..., 0] == arr[..., 1]).all() and (arr[..., 1] == arr[..., 2]).all()
    assert np.array_equal(arr[0, :, 0], np.rint(v[0] / 257.0).astype(np.uint8))
    assert 120 <= int(arr[0, 128, 0]) <= 136  # the middle stays mid grey, not white
    assert latent_key(p) == _file_key(p) + "|grey16"


def test_every_16_bit_mode_and_32_bit_int():
    v = np.array([[0, 257, 32768, 65535]], dtype=np.uint16)
    want = [0, 1, 128, 255]
    for mode, dt in (("I;16", "<u2"), ("I;16L", "<u2"), ("I;16B", ">u2")):
        im = Image.frombytes(mode, (4, 1), v.astype(dt).tobytes())
        assert [int(x) for x in np.asarray(to_rgb(im))[0, :, 0]] == want, mode
    im32 = Image.frombytes("I", (4, 1), np.array([[0, 257, 32768, 65535]], dtype="=i4").tobytes())
    assert [int(x) for x in np.asarray(to_rgb(im32))[0, :, 0]] == want  # within the 16-bit range
    big = Image.frombytes("I", (3, 1), np.array([[0, 1 << 30, (1 << 31) - 1]], dtype="=i4").tobytes())
    assert [int(x) for x in np.asarray(to_rgb(big))[0, :, 0]] == [0, 128, 255]
    assert tuple(np.asarray(to_rgb(Image.new("L", (2, 2), 70)))[0, 0]) == (70, 70, 70)  # other modes as before


@pytest.mark.parametrize("desc,converted", [("sRGB IEC61966-2.1", False), ("Linear sRGB", True), ("scRGB", True),
                                            ("Display P3", True)])
def test_only_a_real_srgb_name_skips_conversion(monkeypatch, desc, converted):
    from types import SimpleNamespace

    from rulztrainer import pictures as P

    fake = SimpleNamespace(profile=SimpleNamespace(xcolor_space="RGB ", profile_description=desc))
    monkeypatch.setattr(P.ImageCms, "ImageCmsProfile", lambda f: fake)
    monkeypatch.setattr(P.ImageCms, "buildTransform", lambda *a, **k: "transform")
    P._srgb_transform.cache_clear()
    try:
        assert (P._srgb_transform(desc.encode(), "RGB") == "transform") is converted
    finally:
        P._srgb_transform.cache_clear()
