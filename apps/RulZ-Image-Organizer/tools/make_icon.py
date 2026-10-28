"""Generate logo and icon assets from assets/logo_source.png (the RulZ logo on white).

Outputs (all transparent PNGs unless noted):
  logo_full.png        R mark + "RulZ" wordmark, original red/black (for light backgrounds)
  logo_full_white.png  same, black ink turned white (for dark backgrounds)
  logo_mark.png        the R mark alone, original colors
  icon.png / icon.ico  app icon: R mark (red + white) on a black rounded square
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ASSETS = Path(__file__).resolve().parent.parent / "assets"
RED = (238, 1, 13)


def unwhite(img: Image.Image) -> np.ndarray:
    """Turn a logo on a white background into RGBA with real transparency."""
    a = np.asarray(img.convert("RGB")).astype(np.float32)
    alpha = 1.0 - a.min(axis=2) / 255.0
    alpha = np.clip((alpha - 0.03) / 0.97, 0, 1)  # drop the off-white paper tone
    safe = np.maximum(alpha, 1e-3)[..., None]
    rgb = np.clip((a - (1 - alpha[..., None]) * 255) / safe, 0, 255)
    return np.dstack([rgb, alpha * 255]).astype(np.uint8)


def recolor_black(rgba: np.ndarray, color) -> np.ndarray:
    out = rgba.copy()
    is_black = out[..., 0] < 120  # the red ink has a high red channel
    out[is_black, :3] = color
    return out


def crop(rgba: np.ndarray, pad: int = 0) -> np.ndarray:
    ys, xs = np.nonzero(rgba[..., 3] > 8)
    return rgba[max(0, ys.min() - pad):ys.max() + 1 + pad, max(0, xs.min() - pad):xs.max() + 1 + pad]


def save(arr: np.ndarray, name: str, height: int | None = None) -> Image.Image:
    im = Image.fromarray(arr, "RGBA")
    if height:
        im = im.resize((round(im.width * height / im.height), height), Image.Resampling.LANCZOS)
    im.save(ASSETS / name)
    return im


def main():
    src = Image.open(ASSETS / "logo_source.png")
    rgba = crop(unwhite(src), pad=4)
    save(rgba, "logo_full.png", 160)
    save(recolor_black(rgba, (255, 255, 255)), "logo_full_white.png", 160)

    # The R mark is everything left of the gap before the wordmark.
    ink_cols = (rgba[..., 3] > 8).any(axis=0)
    x = int(np.argmax(ink_cols))
    while ink_cols[x]:
        x += 1
    mark = crop(rgba[:, :x])
    save(mark, "logo_mark.png")

    # App icon: black rounded square, red/white R centered.
    S = 1024
    icon = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    ImageDraw.Draw(icon).rounded_rectangle((24, 24, S - 24, S - 24), radius=220, fill=(14, 14, 16, 255))
    ImageDraw.Draw(icon).rounded_rectangle((24, 24, S - 24, S - 24), radius=220, outline=(*RED, 255), width=18)
    m = Image.fromarray(recolor_black(mark, (255, 255, 255)), "RGBA")
    target = int(S * 0.62)
    scale = target / max(m.size)
    m = m.resize((round(m.width * scale), round(m.height * scale)), Image.Resampling.LANCZOS)
    icon.alpha_composite(m, ((S - m.width) // 2 + 10, (S - m.height) // 2))
    icon.resize((256, 256), Image.Resampling.LANCZOS).save(ASSETS / "icon.png")
    icon.save(ASSETS / "icon.ico", sizes=[(16, 16), (20, 20), (24, 24), (32, 32), (40, 40), (48, 48), (64, 64),
                                          (128, 128), (256, 256)])
    print("wrote", ASSETS)


if __name__ == "__main__":
    main()
