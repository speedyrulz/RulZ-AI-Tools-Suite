"""Refreshes catalog/<app>/ from each app's repository (needs Pillow).

For every app it copies README.md, the images the README references (into catalog/<app>/repo/, keeping their
paths), and the app icon as icon.png (256 px) and icon.ico. Screenshots in catalog/<app>/screenshots/ are left
alone. Sources are the local checkouts below when they exist, otherwise the GitHub repositories.

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

# id: (local checkout, GitHub repo, branch, icon png, icon ico or None)
APPS = {
    "rulz-trainer": (r"C:\ai\RulzTrainer", "speedyrulz/RulzTrainer", "main", "assets/icon.png", None),
    "image-organizer": (r"C:\coding\RulZ Image Organizer", "speedyrulz/RulZ-Image-Organizer", "main",
                        "assets/icon.png", "assets/icon.ico"),
    "image-tournament": (r"C:\coding\RulZ Image Tournament", "speedyrulz/RulZ-Image-Tournament", "main",
                         "assets/app.png", "assets/app.ico"),
    "llm-suite": (r"C:\coding\RulZ LLM Suite", "speedyrulz/rulz-llm-suite", "master", "build/icon.png", "build/icon.ico"),
    "training-captioner": (r"C:\coding\Rulz Training Captioner", None, "main", "assets/icon.png", "assets/icon.ico"),
    "clip-extractor": (r"C:\simplest-video-edit", "speedyrulz/rulz-simple-video-clip-extractor", "main",
                       "simplecut/assets/icon256.png", "simplecut/assets/icon.ico"),
}

IMAGE_REFS = [
    re.compile(r"!\[[^\]]*\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)"),
    re.compile(r"<img[^>]+src=\"([^\"]+)\"", re.I),
    re.compile(r"<source[^>]+srcset=\"([^\"]+)\"", re.I),
]


class Source:
    def __init__(self, local, repo, branch):
        self.local = Path(local) if local and Path(local).exists() else None
        self.repo, self.branch = repo, branch
        if not self.local and not repo:
            raise SystemExit(f"No source for {local}: not on this PC and not on GitHub")

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
        return str(self.local) if self.local else f"github.com/{self.repo}@{self.branch}"


def sync(app_id):
    local, repo, branch, icon_png, icon_ico = APPS[app_id]
    src = Source(local, repo, branch)
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
