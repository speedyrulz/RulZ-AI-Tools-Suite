"""Procedurally generated demo images for the RulZ AI Tools Suite screenshots.

Nothing here comes from the user's machine: every picture is drawn from random numbers (landscapes,
oceans, dunes, aurora, bokeh and geometric abstracts), so the screenshots are safe for a public repo.

    python make_demo_images.py OUT_DIR [--count 48] [--seed 7] [--captions] [--exif]
                               [--names img|dataset|titles] [--size 1600]

--captions writes a same-name .txt caption next to every image (Training Captioner).
--exif     stamps an EXIF "date taken" spread over several months (Image Organizer's timeline).
Needs Pillow + numpy (the Image Organizer venv has both).
"""
from __future__ import annotations

import argparse
import math
import os
from datetime import datetime, timedelta

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# --------------------------------------------------------------------------- palettes
SKIES = {
    "dawn": [(38, 48, 112), (178, 110, 168), (255, 186, 150)],
    "midday": [(46, 104, 190), (126, 180, 232), (214, 232, 246)],
    "golden hour": [(62, 82, 156), (232, 140, 92), (255, 214, 128)],
    "dusk": [(22, 18, 58), (118, 46, 108), (250, 118, 78)],
    "night": [(4, 6, 22), (12, 22, 58), (34, 56, 98)],
    "pastel morning": [(96, 168, 196), (214, 196, 214), (252, 214, 186)],
    "stormy": [(48, 56, 70), (98, 108, 122), (170, 170, 168)],
}
NEAR = {  # foreground (closest ridge) colour per sky
    "dawn": (28, 22, 54), "midday": (34, 70, 52), "golden hour": (52, 30, 40), "dusk": (18, 10, 30),
    "night": (6, 8, 18), "pastel morning": (60, 76, 104), "stormy": (26, 30, 36),
}


def mix(a, b, t):
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


def vgrad(h, w, stops, y0=0.0, y1=1.0):
    """Vertical multi-stop gradient as float32 HxWx3."""
    t = np.clip((np.linspace(0, 1, h) - y0) / max(1e-6, y1 - y0), 0, 1)
    stops = np.asarray(stops, np.float32)
    n = len(stops) - 1
    pos = t * n
    i = np.minimum(pos.astype(int), n - 1)
    f = (pos - i)[:, None]
    rows = stops[i] * (1 - f) + stops[i + 1] * f
    return np.repeat(rows[:, None, :], w, axis=1)


def noise1d(rng, w, octaves=6, base=4, persistence=0.55):
    out = np.zeros(w, np.float32)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        n = base * 2 ** o + 1
        pts = rng.uniform(-1, 1, n)
        x = np.linspace(0, n - 1, w)
        xi = np.floor(x).astype(int).clip(0, n - 2)
        f = x - xi
        f = f * f * (3 - 2 * f)
        out += amp * (pts[xi] * (1 - f) + pts[xi + 1] * f)
        total += amp
        amp *= persistence
    return out / total


def blur(arr, radius):
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return np.asarray(img.filter(ImageFilter.GaussianBlur(radius)), np.float32)


def blend(base, color, alpha):
    a = alpha[..., None] if alpha.ndim == 2 else alpha
    return base * (1 - a) + np.asarray(color, np.float32) * a


# --------------------------------------------------------------------------- scene parts
def add_sun(img, cx, cy, r, color, glow=3.5, strength=0.9):
    h, w, _ = img.shape
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    halo = np.exp(-np.maximum(d - r, 0) / (r * glow)) * strength
    img = blend(img, color, np.clip(halo * 0.75, 0, 1))
    disc = np.clip((r - d) / 2.0 + 0.5, 0, 1)
    return blend(img, mix(color, (255, 255, 255), 0.55), disc)


def add_stars(img, rng, density, horizon):
    h, w, _ = img.shape
    n = int(w * h * density)
    ys = rng.integers(0, int(h * horizon), n)
    xs = rng.integers(0, w, n)
    b = rng.uniform(0.25, 1.0, n) ** 2
    layer = np.zeros((h, w), np.float32)
    layer[ys, xs] = b
    big = rng.random(n) < 0.04
    for y, x in zip(ys[big], xs[big]):
        layer[max(0, y - 1):y + 2, max(0, x - 1):x + 2] = np.maximum(layer[max(0, y - 1):y + 2, max(0, x - 1):x + 2], 0.5)
    layer = np.maximum(layer, blur(layer[..., None].repeat(3, 2) * 255, 1.2)[..., 0] / 255 * 1.5)
    return blend(img, (255, 250, 235), np.clip(layer, 0, 1))


def add_clouds(img, rng, top, bottom, color, coverage=0.5, softness=10):
    h, w, _ = img.shape

    def layer(rows, cols):  # stretched value noise: long horizontal wisps
        small = rng.random((rows, cols)).astype(np.float32)
        im = Image.fromarray((small * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)
        return np.asarray(im, np.float32) / 255

    n = 0.55 * layer(7, 5) + 0.3 * layer(16, 9) + 0.15 * layer(40, 22)
    yy = np.linspace(0, 1, h)[:, None]
    band = np.clip((yy - top) / 0.08, 0, 1) * np.clip((bottom - yy) / 0.12, 0, 1)
    a = np.clip((n - (1 - coverage)) * 2.2, 0, 1) * band
    a = blur(np.repeat(a[..., None] * 255, 3, 2), softness)[..., 0] / 255
    shade = np.clip(1.1 - a * 0.35, 0.75, 1.0)[..., None]  # slightly darker cores
    return blend(img, np.asarray(color, np.float32) * shade, a * 0.7)


def ridge_layer(img, rng, base_y, amp, color, rough=0.55, octaves=7, top_light=None):
    h, w, _ = img.shape
    line = base_y * h + noise1d(rng, w, octaves, int(rng.integers(2, 6)), rough) * amp * h
    yy = np.arange(h)[:, None]
    edge = np.clip(yy - line[None, :] + 0.5, 0, 1)
    shade = np.clip((yy - line[None, :]) / (h * 0.35), 0, 1)
    c0 = np.asarray(color, np.float32)
    c1 = c0 * 0.72
    fill = c0 * (1 - shade[..., None]) + c1 * shade[..., None]
    if top_light is not None:  # rim light on the ridge line
        rim = np.exp(-np.maximum(yy - line[None, :], 0) / (h * 0.01)) * edge
        fill = blend(fill, top_light, rim * 0.35)
    return img * (1 - edge[..., None]) + fill * edge[..., None], line


def add_pines(img, rng, ground, count, color, height=(0.08, 0.22), side=None):
    h, w, _ = img.shape
    layer = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(layer)
    for _ in range(count):
        if side == "left":
            x = rng.uniform(-0.02, 0.3) * w
        elif side == "right":
            x = rng.uniform(0.7, 1.02) * w
        else:
            x = rng.uniform(-0.02, 1.02) * w
        gy = ground[int(np.clip(x, 0, w - 1))]
        th = rng.uniform(*height) * h
        tw = th * rng.uniform(0.28, 0.38)
        tiers = int(rng.integers(4, 7))
        for t in range(tiers):
            ty = gy - th * (t / tiers)
            ww = tw * (1 - t / tiers) * 0.9 + tw * 0.1
            d.polygon([(x - ww / 2, ty), (x + ww / 2, ty), (x, ty - th / tiers * 1.8)], fill=255)
        d.rectangle([x - tw * 0.05, gy - th * 0.05, x + tw * 0.05, gy + 4], fill=255)
    a = np.asarray(layer.filter(ImageFilter.GaussianBlur(0.6)), np.float32) / 255
    return blend(img, color, a)


def add_reflection(img, rng, water_y, tint, darken=0.72):
    h, w, _ = img.shape
    y0 = int(water_y * h)
    src = img[:y0][::-1]
    out = img.copy()
    rows = min(h - y0, src.shape[0])
    ys = np.arange(rows)
    shift = (np.sin(ys * 0.35 + rng.uniform(0, 6)) * (1 + ys / rows * 6)).astype(int)
    for i in range(rows):
        out[y0 + i] = np.roll(src[i], shift[i], axis=0)
    water = out[y0:y0 + rows]
    fade = np.linspace(0, 1, rows)[:, None, None]
    water = water * darken + np.asarray(tint, np.float32) * (1 - darken)
    water = water * (1 - fade * 0.35) + np.asarray(tint, np.float32) * fade * 0.35
    # thin ripple highlights
    rip = (rng.random((rows, w)) > 0.985).astype(np.float32)
    rip = blur(np.repeat(rip[..., None] * 255, 3, 2), 1.5)[..., 0] / 255 * (1 - fade[..., 0])
    water = blend(water, (255, 255, 255), np.clip(rip * 0.5, 0, 1))
    out[y0:y0 + rows] = water
    out[y0:y0 + 2] = out[y0:y0 + 2] * 0.85 + 255 * 0.15
    return out


def finish(img, rng, grain=4.0, vignette=0.28):
    h, w, _ = img.shape
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
    img = img * (1 - vignette * np.clip(d - 0.55, 0, 1)[..., None])
    img = img + rng.normal(0, grain, img.shape)
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


# --------------------------------------------------------------------------- scenes
def scene_mountains(rng, w, h):
    sky_name = rng.choice(["dawn", "midday", "golden hour", "dusk", "night", "pastel morning", "stormy"],
                          p=[0.17, 0.17, 0.2, 0.18, 0.1, 0.12, 0.06])
    stops = SKIES[sky_name]
    img = vgrad(h, w, stops, 0, 0.72)
    words = []
    lake = rng.random() < 0.5
    horizon = rng.uniform(0.5, 0.62)
    if sky_name == "night":
        img = add_stars(img, rng, 0.0009, horizon + 0.05)
        mx, my, mr = rng.uniform(0.2, 0.8) * w, rng.uniform(0.12, 0.3) * h, h * 0.035
        img = add_sun(img, mx, my, mr, (220, 228, 255), glow=2.5, strength=0.5)
        words.append("a starry night sky with a bright full moon")
    elif sky_name != "stormy":
        sx, sy = rng.uniform(0.2, 0.8) * w, (horizon - rng.uniform(0.02, 0.2)) * h
        sun_c = mix(stops[-1], (255, 240, 200), 0.5)
        img = add_sun(img, sx, sy, h * rng.uniform(0.035, 0.06), sun_c)
        words.append(f"a {sky_name} sky with a glowing sun low over the peaks")
    else:
        words.append("a heavy overcast sky")
    if sky_name in ("midday", "pastel morning", "stormy", "golden hour") or rng.random() < 0.3:
        cc = (250, 250, 252) if sky_name in ("midday", "stormy") else mix(stops[-1], (255, 255, 255), 0.4)
        img = add_clouds(img, rng, 0.02, horizon - 0.08, cc, rng.uniform(0.35, 0.6))
        words.append("soft drifting clouds")
    n = int(rng.integers(3, 6))
    near = NEAR[sky_name]
    far = mix(stops[-1], stops[1], 0.35)
    ground = None
    water_y = horizon + rng.uniform(0.1, 0.16) if lake else None
    for i in range(n):
        t = (i + 1) / n
        col = mix(far, near, t ** 0.8)
        base = horizon - 0.18 + i * (0.42 if lake else 0.5) / n
        amp = 0.16 * (1 - i / (n + 1)) + 0.04
        img, ground = ridge_layer(img, rng, base, amp, col, rough=0.5 + 0.08 * i,
                                  top_light=stops[-1] if i < 2 else None)
        if lake and base > water_y - 0.1:
            break
    words.insert(1, f"{n} layered mountain ridges fading into the haze")
    if lake:
        img = add_reflection(img, rng, water_y, mix(stops[1], (10, 20, 40), 0.5))
        words.append("a still lake mirroring the ridges in the foreground")
        if rng.random() < 0.7:
            ground = np.full(w, water_y * h + 2)
            img = add_pines(img, rng, ground, int(rng.integers(6, 14)), mix(near, (0, 0, 0), 0.4),
                            side=rng.choice(["left", "right"]))
            words.append("dark pine silhouettes along the shore")
    elif rng.random() < 0.6:
        img = add_pines(img, rng, ground, int(rng.integers(10, 26)), mix(near, (0, 0, 0), 0.5))
        words.append("a pine forest silhouetted on the nearest slope")
    return img, "A mountain landscape, " + ", ".join(words) + ", digital painting."


def scene_ocean(rng, w, h):
    sky_name = rng.choice(["golden hour", "dusk", "midday", "pastel morning"])
    stops = SKIES[sky_name]
    hz = rng.uniform(0.52, 0.64)
    img = vgrad(h, w, stops, 0, hz)
    sx = rng.uniform(0.3, 0.7) * w
    sy = (hz - rng.uniform(0.03, 0.14)) * h
    sun_c = mix(stops[-1], (255, 236, 190), 0.5)
    img = add_sun(img, sx, sy, h * 0.045, sun_c, glow=4)
    if rng.random() < 0.6:
        img = add_clouds(img, rng, 0.03, hz - 0.06, mix(stops[-1], (255, 255, 255), 0.3), 0.45, 14)
    sea_top = mix(stops[-1], (20, 60, 100), 0.55)
    sea_bot = (8, 26, 48)
    y0 = int(hz * h)
    img[y0:] = vgrad(h - y0, w, [sea_top, sea_bot])
    # sun glitter path
    yy, xx = np.mgrid[y0:h, 0:w]
    spread = (yy - y0) / (h - y0) * w * 0.12 + w * 0.01
    glit = np.exp(-((xx - sx) / spread) ** 2) * (rng.random((h - y0, w)) > 0.82)
    glit = blur(np.repeat(glit[..., None] * 255, 3, 2), 1.2)[..., 0] / 255
    img[y0:] = blend(img[y0:], sun_c, np.clip(glit * 1.6, 0, 1))
    waves = (np.sin(xx * 0.02 + yy * 0.9) * 0.5 + 0.5) ** 8 * ((yy - y0) / (h - y0))
    img[y0:] = blend(img[y0:], (230, 240, 250), waves * 0.12)
    words = [f"a calm ocean at {sky_name}", "the sun hanging just above a clean horizon",
             "a shimmering path of light across gentle waves"]
    if rng.random() < 0.5:  # a headland on one side
        side = rng.choice([-1, 1])
        line = hz * h - (np.clip((np.linspace(-1, 1, w) * side - 0.35) * 2.2, 0, 1) ** 0.6) * h * 0.18 \
            + noise1d(rng, w, 6, 6) * h * 0.015
        yyy = np.arange(h)[:, None]
        m = np.clip(yyy - line[None, :], 0, 1) * (yyy < hz * h + 2)
        img = blend(img, mix(NEAR[sky_name], (0, 0, 0), 0.3), m)
        words.append("a dark rocky headland on the " + ("right" if side > 0 else "left"))
    return img, "A seascape, " + ", ".join(words) + ", digital painting."


def scene_dunes(rng, w, h):
    stops = [(62, 120, 196), (150, 196, 230), (246, 222, 190)]
    img = vgrad(h, w, stops, 0, 0.55)
    img = add_sun(img, rng.uniform(0.6, 0.9) * w, rng.uniform(0.1, 0.25) * h, h * 0.04, (255, 246, 220), 3, 0.6)
    base = np.array([224, 146, 84], np.float32)
    yy = np.arange(h)[:, None]
    for i in range(5):
        t = i / 4
        x = np.linspace(0, 1, w)
        line = (0.45 + t * 0.12) * h + np.sin(x * rng.uniform(4, 9) + rng.uniform(0, 6)) * h * 0.04 * (1 + t) \
            + noise1d(rng, w, 4, 3) * h * 0.03
        edge = np.clip(yy - line[None, :] + 0.5, 0, 1)
        slope = np.gradient(line)[None, :]
        light = np.clip(0.75 + slope * 0.08, 0.45, 1.15)
        col = mix((236, 186, 150), tuple(base), t)
        fill = np.asarray(col, np.float32)[None, None, :] * light[..., None]
        fill = fill * (1 - np.clip((yy - line[None, :]) / (h * 0.4), 0, 0.35)[..., None])
        img = img * (1 - edge[..., None]) + fill * edge[..., None]
    return img, ("A desert landscape with rolling sand dunes under a clear blue sky, a pale sun high on the right, "
                 "sharp ridgelines with warm light on one face and cool shadow on the other.")


def scene_aurora(rng, w, h):
    img = vgrad(h, w, [(2, 4, 16), (8, 18, 44), (18, 40, 70)], 0, 0.75)
    img = add_stars(img, rng, 0.0011, 0.8)
    yy, xx = np.mgrid[0:h, 0:w]
    hues = [(60, 255, 160), (90, 230, 200), (180, 90, 255)]
    words = []
    for k in range(int(rng.integers(2, 4))):
        c = hues[k % len(hues)]
        path = (0.2 + 0.18 * k) * h + noise1d(rng, w, 4, 2) * h * 0.18
        d = (yy - path[None, :])
        curtain = np.exp(-np.maximum(d, 0) / (h * 0.02)) * np.exp(-np.maximum(-d, 0) / (h * 0.16))
        rays = 0.6 + 0.4 * np.sin(xx * rng.uniform(0.05, 0.12) + noise1d(rng, w, 3, 8)[None, :] * 6) ** 2
        img = img + np.asarray(c, np.float32) * (curtain * rays * 0.55)[..., None]
    words.append("green and violet aurora curtains rippling across a starry sky")
    img, ground = ridge_layer(img, rng, 0.72, 0.1, (16, 22, 34), top_light=(160, 200, 220))
    img, ground = ridge_layer(img, rng, 0.84, 0.05, (8, 10, 16))
    img = add_pines(img, rng, ground, int(rng.integers(14, 30)), (2, 3, 6), height=(0.06, 0.16))
    words.append("snowy ridges and a line of black pines in the foreground")
    return img, "A night landscape, " + ", ".join(words) + ", digital painting."


def scene_hills(rng, w, h):
    stops = SKIES["midday"]
    img = vgrad(h, w, stops, 0, 0.6)
    img = add_clouds(img, rng, 0.0, 0.5, (252, 252, 255), rng.uniform(0.4, 0.6), 8)
    greens = [(150, 190, 110), (110, 170, 80), (80, 150, 70), (60, 125, 55), (46, 104, 44)]
    if rng.random() < 0.4:
        greens[1] = (214, 190, 90)  # a field of ripe wheat
    ground = None
    yy = np.arange(h)[:, None]
    for i, c in enumerate(greens):
        x = np.linspace(0, 1, w)
        line = (0.5 + i * 0.08) * h + np.sin(x * rng.uniform(2, 5) + rng.uniform(0, 6)) * h * 0.05 \
            + noise1d(rng, w, 3, 2) * h * 0.04
        edge = np.clip(yy - line[None, :] + 0.5, 0, 1)
        shade = np.clip((yy - line[None, :]) / (h * 0.25), 0, 1)[..., None]
        fill = np.asarray(c, np.float32) * (1.08 - shade * 0.25)
        img = img * (1 - edge[..., None]) + fill * edge[..., None]
        if i == 2:
            ground = line
    # round trees on the middle hill
    layer = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(layer)
    for _ in range(int(rng.integers(5, 12))):
        x = rng.uniform(0.05, 0.95) * w
        gy = ground[int(x)]
        r = rng.uniform(0.018, 0.035) * h
        d.rectangle([x - r * 0.12, gy - r * 1.4, x + r * 0.12, gy + 2], fill=255)
        d.ellipse([x - r, gy - r * 3.0, x + r, gy - r * 1.0], fill=255)
    a = np.asarray(layer.filter(ImageFilter.GaussianBlur(0.8)), np.float32) / 255
    img = blend(img, (34, 70, 38), a)
    # a scatter of wildflowers in the foreground
    n = int(w * h * 0.0006)
    fx, fy = rng.integers(0, w, n), rng.integers(int(h * 0.82), h, n)
    cols = np.asarray([(250, 240, 120), (255, 255, 255), (240, 120, 160), (200, 160, 250)], np.float32)
    img[fy, fx] = cols[rng.integers(0, len(cols), n)]
    return img, ("A countryside scene with rolling green hills under a bright blue sky with fluffy white clouds, "
                 "round trees along a ridge, wildflowers scattered across the meadow in the foreground.")


def scene_bokeh(rng, w, h):
    pal = rng.choice(["warm", "cool", "neon"])
    stops = {"warm": [(30, 10, 20), (70, 24, 30)], "cool": [(6, 14, 30), (16, 40, 70)],
             "neon": [(14, 6, 30), (40, 10, 60)]}[pal]
    cols = {"warm": [(255, 180, 90), (255, 120, 80), (255, 220, 160)],
            "cool": [(120, 200, 255), (80, 140, 255), (200, 240, 255)],
            "neon": [(255, 60, 160), (80, 220, 255), (200, 100, 255)]}[pal]
    img = vgrad(h, w, stops)
    layer = Image.new("RGB", (w, h), (0, 0, 0))
    d = ImageDraw.Draw(layer)
    for _ in range(int(rng.integers(40, 80))):
        r = rng.uniform(0.015, 0.08) * h
        x, y = rng.uniform(0, w), rng.uniform(0, h)
        c = cols[int(rng.integers(0, 3))]
        a = rng.uniform(0.15, 0.6)
        d.ellipse([x - r, y - r, x + r, y + r], fill=tuple(int(v * a) for v in c))
    lay = np.asarray(layer.filter(ImageFilter.GaussianBlur(3)), np.float32)
    img = 255 - (255 - img) * (255 - lay) / 255  # screen blend
    return img, f"An abstract bokeh background, soft out-of-focus {pal} light orbs of different sizes over a dark gradient background."


def scene_geometric(rng, w, h):
    palettes = [[(241, 233, 218), (230, 57, 70), (29, 53, 87), (69, 123, 157), (244, 162, 97)],
                [(250, 244, 230), (38, 70, 83), (42, 157, 143), (233, 196, 106), (231, 111, 81)],
                [(20, 20, 24), (237, 0, 13), (240, 240, 240), (120, 120, 128), (255, 180, 0)]]
    pal = palettes[int(rng.integers(0, len(palettes)))]
    img = Image.new("RGB", (w, h), pal[0])
    d = ImageDraw.Draw(img)
    for _ in range(int(rng.integers(8, 16))):
        c = pal[int(rng.integers(1, len(pal)))]
        kind = rng.choice(["circle", "rect", "arc", "tri"])
        x, y = rng.uniform(0, w), rng.uniform(0, h)
        s = rng.uniform(0.08, 0.35) * h
        if kind == "circle":
            d.ellipse([x - s, y - s, x + s, y + s], fill=c)
        elif kind == "rect":
            d.rectangle([x - s, y - s * 0.6, x + s, y + s * 0.6], fill=c)
        elif kind == "arc":
            d.pieslice([x - s, y - s, x + s, y + s], int(rng.integers(0, 4)) * 90, int(rng.integers(0, 4)) * 90 + 180, fill=c)
        else:
            d.polygon([(x, y - s), (x + s, y + s * 0.8), (x - s, y + s * 0.8)], fill=c)
    arr = np.asarray(img, np.float32)
    return arr, "An abstract geometric composition of flat circles, half-discs, rectangles and triangles in a bold limited palette on a plain background."


SCENES = [(scene_mountains, 0.38), (scene_ocean, 0.15), (scene_hills, 0.12), (scene_aurora, 0.08),
          (scene_dunes, 0.08), (scene_bokeh, 0.09), (scene_geometric, 0.1)]
TITLES = {
    scene_hills: ["Meadow", "Hillside"],
    scene_mountains: ["Ridge Lines", "Alpine Haze", "Lakeside", "Mountain Light", "Peaks"],
    scene_ocean: ["Horizon", "Sea Glow", "Coastline"],
    scene_aurora: ["Aurora", "Northern Lights"],
    scene_dunes: ["Dunes", "Desert"],
    scene_bokeh: ["Bokeh", "City Lights"],
    scene_geometric: ["Shapes", "Composition"],
}


def month_dates(start, counts, rng):
    """One date per image: counts[k] images in the k-th month after start (YYYY-MM), sorted."""
    y, m = (int(v) for v in start.split("-"))
    out = []
    for n in counts:
        days = sorted(rng.uniform(1, 28, n))
        out += [datetime(y, m, 1) + timedelta(days=float(d) - 1, hours=float(rng.uniform(8, 19))) for d in days]
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def generate(out, count, seed, captions, exif, names, size, start="2026-03", month_counts=None):
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(seed)
    funcs, weights = zip(*SCENES)
    weights = np.asarray(weights) / sum(weights)
    counts = month_counts or [6] * (count // 6) + ([count % 6] if count % 6 else [])
    dates = month_dates(start, counts, np.random.default_rng(seed + 1000))
    count = min(count, len(dates))
    written = []
    for i in range(count):
        f = funcs[int(rng.choice(len(funcs), p=weights))]
        shape = rng.choice(["land", "land", "land", "portrait", "square", "wide"])
        w, h = {"land": (size, int(size * 2 / 3)), "portrait": (int(size * 2 / 3), size),
                "square": (int(size * 0.8), int(size * 0.8)), "wide": (size, int(size * 9 / 16))}[shape]
        arr, caption = f(rng, w, h)
        img = finish(arr, rng)
        if names == "dataset":
            stem = f"landscape_{i + 1:03d}"
        elif names == "titles":
            stem = f"{TITLES[f][int(rng.integers(0, len(TITLES[f])))]} {i + 1:02d}"
        else:
            stem = f"IMG_{4012 + i * 7 + int(rng.integers(0, 6))}"
        path = os.path.join(out, stem + ".jpg")
        kw = {"quality": 90, "optimize": True}
        if exif:
            when = dates[i]
            ex = Image.Exif()
            stamp = when.strftime("%Y:%m:%d %H:%M:%S")
            ex[306] = stamp
            ex[0x8769] = {36867: stamp, 36868: stamp}
            ex[271] = "RulZ Demo"
            ex[272] = "Procedural Camera"
            kw["exif"] = ex
        img.save(path, "JPEG", **kw)
        if exif:
            t = when.timestamp()
            os.utime(path, (t, t))
        if captions:
            with open(os.path.join(out, stem + ".txt"), "w", encoding="utf-8", newline="\n") as fh:
                fh.write(caption)
        written.append(path)
        print(path)
    return written


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--count", type=int, default=48)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--captions", action="store_true")
    ap.add_argument("--exif", action="store_true")
    ap.add_argument("--names", choices=["img", "dataset", "titles"], default="img")
    ap.add_argument("--size", type=int, default=1600)
    ap.add_argument("--start", default="2026-03", help="first month (YYYY-MM) for --exif dates")
    ap.add_argument("--month-counts", default="", help="images per month, oldest first, e.g. 4,6,6,12")
    a = ap.parse_args()
    mc = [int(v) for v in a.month_counts.split(",") if v.strip()] or None
    generate(a.out, a.count, a.seed, a.captions, a.exif, a.names, a.size, a.start, mc)
