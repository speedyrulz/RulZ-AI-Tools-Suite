"""Copy reviewed captures into catalog/<id>/screenshots/, optimized.

    python finalize.py R:\\out  "C:\\coding\\RulZ AI Tools Suite\\catalog"

Every PNG in OUT/<id>/ whose name starts with two digits is re-saved with Pillow (RGB, optimize=True),
downscaled to 1920 px wide if larger; anything still over 1.5 MB is written as a quality-90 JPEG instead.
"""
from __future__ import annotations

import os
import re
import sys

from PIL import Image

LIMIT = 1_500_000
src_root, catalog = sys.argv[1], sys.argv[2]
for app_id in sorted(os.listdir(src_root)):
    src = os.path.join(src_root, app_id)
    dst = os.path.join(catalog, app_id, "screenshots")
    if not os.path.isdir(src) or not os.path.isdir(dst):
        continue
    for name in sorted(os.listdir(src)):
        if not re.match(r"\d\d-.*\.png$", name):
            continue
        im = Image.open(os.path.join(src, name)).convert("RGB")
        if im.width > 1920:
            im = im.resize((1920, round(im.height * 1920 / im.width)), Image.LANCZOS)
        out = os.path.join(dst, name)
        im.save(out, optimize=True)
        if os.path.getsize(out) > LIMIT:
            os.remove(out)
            out = out[:-4] + ".jpg"
            im.save(out, quality=90, optimize=True, progressive=True)
        print(f"{app_id}/{os.path.basename(out)}  {im.width}x{im.height}  {os.path.getsize(out) / 1e6:.2f} MB")
