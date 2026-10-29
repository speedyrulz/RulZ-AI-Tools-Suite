"""Page loading with Playwright: render, scroll for lazy images, then collect image candidates and links."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

from playwright.async_api import BrowserContext
from playwright.async_api import Error as PlaywrightError

from .urls import (
    best_from_srcsets,
    css_urls,
    looks_like_image_file,
    normalize_image_url,
    normalize_page_url,
    original_candidates,
)

log = logging.getLogger("crawler_app.crawler")

# Scrolls one viewport at a time so lazy loaders fire; stops after the bottom is reached twice
# (the second time lets infinite-scroll pages append one more batch), then returns to the top.
SCROLL_JS = """async (maxSteps) => {
  let atBottom = 0;
  for (let i = 0; i < maxSteps && atBottom < 2; i++) {
    window.scrollBy(0, window.innerHeight);
    await new Promise(r => setTimeout(r, 250));
    const bottom = window.scrollY + window.innerHeight >= document.documentElement.scrollHeight - 2;
    atBottom = bottom ? atBottom + 1 : 0;
  }
  window.scrollTo(0, 0);
}"""

EXTRACT_JS = r"""() => {
  const imgs = [...document.images].map(i => {
    const pic = i.parentElement && i.parentElement.tagName === 'PICTURE'
      ? [...i.parentElement.querySelectorAll('source')]
          .map(s => s.getAttribute('srcset') || s.getAttribute('data-srcset')).filter(Boolean)
      : [];
    return {
      src: i.getAttribute('src'),
      current: i.currentSrc || null,
      srcset: i.getAttribute('srcset'),
      dsrc: i.getAttribute('data-src') || i.getAttribute('data-lazy-src') || i.getAttribute('data-original')
            || i.getAttribute('data-lazy') || null,
      dsrcset: i.getAttribute('data-srcset') || i.getAttribute('data-lazy-srcset') || null,
      nw: i.naturalWidth, nh: i.naturalHeight, pic,
      link: i.closest('a[href]') ? i.closest('a[href]').href : null,
    };
  });
  const bgs = [];
  for (const el of document.querySelectorAll('body *')) {
    const bg = getComputedStyle(el).backgroundImage;
    if (bg && bg !== 'none') bgs.push(bg);
  }
  const og = [...document.querySelectorAll('meta[property="og:image"], meta[name="og:image"]')]
    .map(m => m.getAttribute('content')).filter(Boolean);
  const links = [...document.querySelectorAll('a[href]')].map(a => a.href);
  return {base: document.baseURI, imgs, bgs, og, links};
}"""


class PageError(Exception):
    """The page could not be loaded. Fails only that page."""


@dataclass(frozen=True)
class ImageCandidate:
    url: str  # absolute http(s) or data: URL
    page: str  # page it was found on (used as Referer)
    short_edge: int | None = None  # rendered natural size if known, for skipping tiny images early
    fallbacks: tuple[str, ...] = ()  # tried in order if `url` fails; `url` may be a guessed original

    @property
    def urls(self) -> tuple[str, ...]:
        return (self.url, *self.fallbacks)


@dataclass
class PageResult:
    url: str  # final URL after redirects
    images: list[ImageCandidate]
    links: list[str]  # normalized absolute page URLs, not yet filtered for site/robots


def _pick_img(i: dict, base: str) -> tuple[str | None, int | None]:
    """Largest srcset candidate, else the lazy-load attribute, else what the browser shows.
    The natural size is only a reliable hint when the shown image is the one we picked."""
    sets = [s for s in (i.get("dsrcset"), i.get("srcset"), *i.get("pic", [])) if s]
    best = best_from_srcsets(sets)
    if best:
        return normalize_image_url(best, base), None
    if i.get("dsrc"):
        return normalize_image_url(i["dsrc"], base), None
    url = normalize_image_url(i.get("current") or i.get("src") or "", base)
    hint = min(i["nw"], i["nh"]) if i.get("nw") and i.get("nh") else None
    return url, hint


def extract_candidates(
    raw: dict, page_url: str, prefer_originals: bool = True
) -> tuple[list[ImageCandidate], list[str]]:
    """With prefer_originals, each image is first tried at its full-size original: the image file its
    <a> links to, or the URL with the site's resize markers removed (see urls.original_candidates)."""
    base = raw.get("base") or page_url
    seen: set[str] = set()
    images: list[ImageCandidate] = []

    def add(url: str | None, hint: int | None = None, link: str | None = None) -> None:
        if not url or url in seen:
            return
        guesses: list[str] = []
        if prefer_originals:
            # Known resize patterns first: a linked ".jpg" can also be a page (e.g. Wikipedia's File: pages).
            guesses = original_candidates(url)
            if link and link != url and link not in guesses and looks_like_image_file(link):
                guesses.append(link)
        if guesses:
            # The displayed size says nothing about the original's, so don't skip on it.
            cand = ImageCandidate(guesses[0], page_url, None, (*guesses[1:], url))
        else:
            cand = ImageCandidate(url, page_url, hint)
        if cand.url in seen:
            return
        seen.update(cand.urls)
        images.append(cand)

    for i in raw.get("imgs", []):
        url, hint = _pick_img(i, base)
        add(url, hint, normalize_image_url(i["link"], base) if i.get("link") else None)
    for bg in raw.get("bgs", []):
        for u in css_urls(bg):
            add(normalize_image_url(u, base))
    for u in raw.get("og", []):
        add(normalize_image_url(u, base))

    links: list[str] = []
    seen_links: set[str] = set()
    for href in raw.get("links", []):
        u = normalize_page_url(href, base)
        if u and u not in seen_links:
            seen_links.add(u)
            links.append(u)
    return images, links


async def load_page(
    context: BrowserContext, url: str, timeout_s: float, max_scroll_steps: int, prefer_originals: bool = True
) -> PageResult:
    page = await context.new_page()
    try:
        try:
            resp = await page.goto(url, wait_until="domcontentloaded", timeout=timeout_s * 1000)
        except PlaywrightError as e:
            raise PageError(f"load failed: {str(e).splitlines()[0]}") from e
        if resp is not None:
            if resp.status >= 400:
                raise PageError(f"HTTP {resp.status}")
            ctype = (await resp.all_headers()).get("content-type", "")
            if ctype and "html" not in ctype:
                raise PageError(f"not a web page ({ctype.split(';')[0]})")
        try:
            await page.evaluate(SCROLL_JS, max_scroll_steps)
            try:
                await page.wait_for_load_state("networkidle", timeout=5000)
            except PlaywrightError:
                pass  # pages with constant background traffic never go idle; carry on
            raw = await page.evaluate(EXTRACT_JS)
        except PlaywrightError as e:
            raise PageError(f"page script failed: {str(e).splitlines()[0]}") from e
        final = normalize_page_url(page.url) or url
        images, links = extract_candidates(raw, final, prefer_originals)
        return PageResult(final, images, links)
    finally:
        await page.close()


class Robots:
    """robots.txt per origin, fetched through the run's browser context. Missing or broken
    robots.txt means everything is allowed."""

    def __init__(self, context: BrowserContext, timeout_s: float):
        self._context = context
        self._timeout_ms = timeout_s * 1000
        self._parsers: dict[str, RobotFileParser | None] = {}

    async def _parser(self, url: str) -> RobotFileParser | None:
        p = urlsplit(url)
        origin = f"{p.scheme}://{p.netloc}"
        if origin not in self._parsers:
            parser: RobotFileParser | None = None
            try:
                resp = await self._context.request.get(f"{origin}/robots.txt", timeout=self._timeout_ms)
                if resp.ok:
                    parser = RobotFileParser()
                    parser.parse((await resp.text()).splitlines())
                await resp.dispose()
            except PlaywrightError as e:
                log.info("robots.txt unavailable for %s: %s", origin, str(e).splitlines()[0])
            self._parsers[origin] = parser
        return self._parsers[origin]

    async def allowed(self, url: str) -> bool:
        parser = await self._parser(url)
        return parser is None or parser.can_fetch("*", url)

    async def crawl_delay(self, url: str) -> float:
        parser = await self._parser(url)
        delay = parser.crawl_delay("*") if parser else None
        return float(delay) if delay else 0.0
