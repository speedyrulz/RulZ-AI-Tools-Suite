"""Pure URL / srcset / file-type helpers (no I/O, unit-tested)."""

from __future__ import annotations

import re
from pathlib import PurePosixPath, PureWindowsPath
from urllib.parse import unquote, urljoin, urlsplit, urlunsplit

TRACKING_PARAMS = {"fbclid", "gclid", "dclid", "msclkid", "mc_cid", "mc_eid", "igshid", "ref_src", "_ga", "yclid"}

# Links with these extensions are files, not pages, so the crawler never opens them.
NON_PAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".svg", ".ico", ".avif", ".heic", ".tif", ".tiff",
    ".pdf", ".zip", ".rar", ".7z", ".gz", ".tar", ".exe", ".msi", ".dmg", ".apk", ".iso",
    ".mp4", ".webm", ".mov", ".avi", ".mkv", ".mp3", ".wav", ".ogg", ".flac",
    ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".csv", ".txt", ".xml", ".json", ".rss",
    ".css", ".js", ".woff", ".woff2", ".ttf", ".eot",
}


def _is_tracking(key: str) -> bool:
    k = key.lower()
    return k.startswith("utm_") or k in TRACKING_PARAMS


def _clean_netloc(scheme: str, host: str, port: int | None) -> str:
    default = {"http": 80, "https": 443}[scheme]
    return host if port in (None, default) else f"{host}:{port}"


def normalize_page_url(url: str, base: str | None = None) -> str | None:
    """Absolute http(s) page URL without fragment, credentials or tracking params; None if not crawlable.
    Query parts are kept verbatim (never re-encoded) so they still match what the server expects."""
    try:
        p = urlsplit(urljoin(base, url.strip()) if base else url.strip())
        scheme = p.scheme.lower()
        if scheme not in ("http", "https") or not p.hostname:
            return None
        netloc = _clean_netloc(scheme, p.hostname.lower(), p.port)
    except ValueError:
        return None
    query = "&".join(q for q in p.query.split("&") if q and not _is_tracking(q.split("=", 1)[0]))
    return urlunsplit((scheme, netloc, p.path or "/", query, ""))


def normalize_image_url(url: str, base: str | None = None) -> str | None:
    """Absolute http(s) or data: image URL with the fragment dropped. The query is untouched
    (CDN signatures live there)."""
    url = (url or "").strip()
    if not url:
        return None
    if url[:5].lower() == "data:":
        return url
    try:
        p = urlsplit(urljoin(base, url) if base else url)
        scheme = p.scheme.lower()
        if scheme not in ("http", "https") or not p.hostname:
            return None
        netloc = _clean_netloc(scheme, p.hostname.lower(), p.port)
    except ValueError:
        return None
    return urlunsplit((scheme, netloc, p.path or "/", p.query, ""))


def site_key(url: str) -> str:
    """Host used for the same-site check; `www.example.com` and `example.com` are the same site."""
    host = (urlsplit(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def looks_like_page(url: str) -> bool:
    return PurePosixPath(urlsplit(url).path.lower()).suffix not in NON_PAGE_EXTS


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".avif", ".heic", ".tif", ".tiff"}

# Query keys that only ask a CDN to shrink or re-encode the image.
_RESIZE_PARAMS = {
    "w", "h", "width", "height", "resize", "fit", "crop", "dpr", "q", "quality", "size",
    "maxwidth", "maxheight", "mw", "mh", "scale", "thumbnail", "imwidth",
}
_WIKIMEDIA_THUMB = re.compile(r"^https?://(?:upload|thumb)\.wikimedia\.org/(.+?)/thumb/(.+)/[^/]+$")
_WP_SIZE = re.compile(r"-(?:\d{2,5}x\d{2,5}|scaled)(?=\.\w{3,4}$)", re.IGNORECASE)
_SHOPIFY_SIZE = re.compile(
    r"_(?:\d+x\d*|\d*x\d+|pico|icon|thumb|small|compact|medium|large|grande|master)(?:_crop_\w+)?(?:@\dx)?(?=\.\w{3,4}$)",
    re.IGNORECASE,
)
_GOOGLE_SIZE_SEG = re.compile(r"/[swh]\d+(?:-[a-z0-9-]+)?/(?=[^/]+$)")
_GOOGLE_SIZE_TAIL = re.compile(r"=[swh]\d+[^/]*$")


def looks_like_image_file(url: str) -> bool:
    return PurePosixPath(urlsplit(url).path.lower()).suffix in IMAGE_EXTS


def original_candidates(url: str) -> list[str]:
    """Best guesses at the full-size original of a resized image URL, most likely first (empty if the
    URL doesn't look resized). Each guess is only tried; the fetcher falls back to `url` itself."""
    if url[:5].lower() == "data:":
        return []
    p = urlsplit(url)
    host = (p.hostname or "").lower()
    out: list[str] = []

    def add(u: str) -> None:
        if u != url and u not in out:
            out.append(u)

    if m := _WIKIMEDIA_THUMB.match(urlunsplit((p.scheme, p.netloc, p.path, "", ""))):
        add(f"https://upload.wikimedia.org/{m.group(1)}/{m.group(2)}")
    if host.endswith(("googleusercontent.com", "blogspot.com", "bp.blogspot.com")):
        path = _GOOGLE_SIZE_SEG.sub("/s0/", p.path)
        path = _GOOGLE_SIZE_TAIL.sub("=s0", path)
        add(urlunsplit((p.scheme, p.netloc, path, p.query, "")))
    if "shopify" in host or "/cdn/shop/" in p.path:
        path = _SHOPIFY_SIZE.sub("", p.path)
        query = "&".join(q for q in p.query.split("&") if q and q.split("=", 1)[0].lower() not in _RESIZE_PARAMS)
        add(urlunsplit((p.scheme, p.netloc, path, query, "")))
    if _WP_SIZE.search(p.path):
        add(urlunsplit((p.scheme, p.netloc, _WP_SIZE.sub("", p.path), p.query, "")))
    if p.query:
        parts = [q for q in p.query.split("&") if q]
        kept = [q for q in parts if q.split("=", 1)[0].lower() not in _RESIZE_PARAMS]
        if len(kept) != len(parts):
            add(urlunsplit((p.scheme, p.netloc, p.path, "&".join(kept), "")))
    return out


_SRCSET_ENTRY = re.compile(r"\s*(\S+?)(?:\s+(\d+(?:\.\d+)?)([wxh]))?\s*(?:,\s*|$)")


def parse_srcset(srcset: str) -> list[tuple[str, float]]:
    """[(url, score)]. A `w` descriptor scores its width; an `x` descriptor scores 1000 px per x,
    so mixed sets still rank sensibly. No descriptor counts as 1x."""
    out: list[tuple[str, float]] = []
    for m in _SRCSET_ENTRY.finditer(srcset or ""):
        url, value, unit = m.group(1), m.group(2), m.group(3)
        if not url:
            continue
        if unit == "w":
            score = float(value)
        elif unit == "x":
            score = float(value) * 1000
        else:
            score = 1000.0
        out.append((url, score))
    return out


def best_from_srcsets(srcsets: list[str]) -> str | None:
    """The largest candidate across several srcset strings (an <img> plus its <picture> sources)."""
    best: tuple[str, float] | None = None
    for s in srcsets:
        for url, score in parse_srcset(s):
            if best is None or score > best[1]:
                best = (url, score)
    return best[0] if best else None


_CSS_URL = re.compile(r"""url\(\s*(['"]?)(.*?)\1\s*\)""")


def css_urls(value: str) -> list[str]:
    return [m.group(2) for m in _CSS_URL.finditer(value or "") if m.group(2)]


# Formats Pillow (the engine) can decode. SVG and ICO are deliberately not here.
def sniff_image(data: bytes, heif_ok: bool = False) -> tuple[str, str] | None:
    """(extension, mime) from the file's leading bytes, or None if the engine can't decode it."""
    if data[:3] == b"\xff\xd8\xff":
        return ".jpg", "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png", "image/png"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return ".gif", "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp", "image/webp"
    if data[:2] == b"BM":
        return ".bmp", "image/bmp"
    if heif_ok and data[4:8] == b"ftyp":
        brand = data[8:12]
        if brand in (b"avif", b"avis"):
            return ".avif", "image/avif"
        if brand in (b"heic", b"heix", b"mif1", b"msf1", b"heim", b"heis"):
            return ".heic", "image/heic"
    return None


def file_stem(url: str, max_len: int = 40) -> str:
    """Safe filename stem from the URL's last path segment: [a-z0-9_-] only, never empty."""
    if url[:5].lower() == "data:":
        return "inline"
    if re.match(r"^[a-z][a-z0-9+.-]*://", url, re.IGNORECASE):
        name = PurePosixPath(unquote(urlsplit(url).path)).stem.lower()
    else:  # a local file path (folder scans)
        name = PureWindowsPath(url).stem.lower()
    name = re.sub(r"[^a-z0-9_-]+", "-", name).strip("-_")
    return (name[:max_len].rstrip("-_")) or "image"


def slug(text: str, max_len: int = 40) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:max_len].rstrip("-") or "x"
