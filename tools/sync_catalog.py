"""Refreshes catalog/<app>/ from each app's repository (needs Pillow).

For every app it copies README.md, the images the README references (into catalog/<app>/repo/, keeping their
paths), and the app icon as icon.png (256 px) and icon.ico. Screenshots in catalog/<app>/screenshots/ are left
alone. The source is the app's submodule in apps/ (run `git submodule update --init` first, or
`git submodule update --remote` for the newest commits); without it, the file comes from GitHub.

    python tools/sync_catalog.py            # all apps
    python tools/sync_catalog.py llm-suite  # just one
"""
import io
import re
import shutil
import sys
import urllib.request
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
CATALOG = ROOT / "catalog"

# id: (GitHub repo = submodule folder in apps/, branch, icon png, icon ico or None)
APPS = {
    "rulz-trainer": ("speedyrulz/RulzTrainer", "main", "assets/icon.png", None),
    "image-organizer": ("speedyrulz/RulZ-Image-Organizer", "main", "assets/icon.png", "assets/icon.ico"),
    "image-tournament": ("speedyrulz/RulZ-Image-Tournament", "main", "assets/app.png", "assets/app.ico"),
    "llm-suite": ("speedyrulz/rulz-llm-suite", "master", "build/icon.png", "build/icon.ico"),
    "coding-agent": ("speedyrulz/rulz-coding-agent", "master", "build/icon.png", "build/icon.ico"),
    "training-captioner": ("speedyrulz/Rulz-Training-Captioner", "main", "assets/icon.png", "assets/icon.ico"),
    "clip-extractor": ("speedyrulz/rulz-simple-video-clip-extractor", "main", "simplecut/assets/icon256.png",
                       "simplecut/assets/icon.ico"),
}

IMAGE_REFS = [
    re.compile(r"!\[[^\]]*\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)"),
    re.compile(r"<img[^>]+src=\"([^\"]+)\"", re.I),
    re.compile(r"<source[^>]+srcset=\"([^\"]+)\"", re.I),
]


class Source:
    def __init__(self, repo, branch):
        sub = ROOT / "apps" / repo.split("/")[1]
        self.local = sub if (sub / "README.md").exists() else None
        self.repo, self.branch = repo, branch

    def read(self, rel):
        if self.local:
            p = self.local / rel
            return p.read_bytes() if p.exists() else None
        url = f"https://raw.githubusercontent.com/{self.repo}/{self.branch}/{rel}"
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                return r.read()
        except Exception:
            return None

    def __str__(self):
        return f"apps/{self.local.name}" if self.local else f"github.com/{self.repo}@{self.branch}"


def sync(app_id):
    repo, branch, icon_png, icon_ico = APPS[app_id]
    src = Source(repo, branch)
    out = CATALOG / app_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "screenshots").mkdir(exist_ok=True)

    readme = src.read("README.md")
    if readme is None:
        raise SystemExit(f"{app_id}: README.md not found in {src}")
    (out / "README.md").write_bytes(readme)

    repo_dir = out / "repo"
    shutil.rmtree(repo_dir, ignore_errors=True)
    text = readme.decode("utf-8")
    refs = {m.group(1) for pattern in IMAGE_REFS for m in pattern.finditer(text)}
    for ref in sorted(refs):
        if re.match(r"^[a-z]+:", ref, re.I) or ref.startswith("#"):
            continue
        rel = ref.lstrip("./")
        data = src.read(rel)
        if data is None:
            print(f"  {app_id}: README image {ref} not found")
            continue
        dest = repo_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)

    png = src.read(icon_png)
    icon = Image.open(io.BytesIO(png)).convert("RGBA")
    if icon.size != (256, 256):
        icon = icon.resize((256, 256), Image.LANCZOS)
    icon.save(out / "icon.png")
    ico = src.read(icon_ico) if icon_ico else None
    if ico:
        (out / "icon.ico").write_bytes(ico)
    else:
        icon.save(out / "icon.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    print(f"{app_id}: synced from {src} ({len(refs)} README images)")


if __name__ == "__main__":
    for app_id in sys.argv[1:] or APPS:
        sync(app_id)
