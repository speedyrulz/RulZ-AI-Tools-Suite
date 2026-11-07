"""Page loading with Playwright: render, scroll for lazy images, then collect image candidates and links."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable
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

# Scrolls one viewport at a time so lazy loaders fire. At the bottom it gives infinite-scroll pages
# up to 600 ms to append more; when the page stops growing it stops, then returns to the top.
SCROLL_JS = """async (maxSteps) => {
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const height = () => document.documentElement.scrollHeight;
  for (let i = 0; i < maxSteps; i++) {
    window.scrollBy(0, window.innerHeight);
    await sleep(150);
    if (window.scrollY + window.innerHeight < height() - 2) continue;
    const before = height();
    let grew = false;
    for (let t = 0; t < 6 && !grew; t++) {
      await sleep(100);
      grew = height() > before;
    }
    if (!grew) break;
  }
  window.scrollTo(0, 0);
}"""

# Waits (up to `ms`) until every <img> has loaded or failed, instead of for network idle, which pages
# with analytics or long polling never reach.
IMAGES_DONE_JS = """async (ms) => {
  const end = Date.now() + ms;
  while (Date.now() < end) {
    if ([...document.images].every(i => i.complete)) return true;
    await new Promise(r => setTimeout(r, 100));
  }
  return false;
}"""
IMAGES_WAIT_MS = 1500

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
    """`urls` is tried in order: guessed full-size originals first (`guesses` of them), then the address
    the page uses, then possibly the copy the browser actually showed, as a backup."""

    url: str  # absolute http(s) or data: URL
    page: str  # page it was found on (used as Referer)
    short_edge: int | None = None  # rendered natural size if known, for skipping tiny images early
    fallbacks: tuple[str, ...] = ()  # tried in order if `url` fails
    guesses: int = 0  # how many of `urls`, from the front, are guessed originals the page never used

    @property
    def urls(self) -> tuple[str, ...]:
        return (self.url, *self.fallbacks)

    @property
    def ids(self) -> tuple[str, ...]:
        """The guesses and the page's own address: what makes two candidates the same image. The backup
        is left out, since unrelated images can share one (a lazy loader's placeholder)."""
        return self.urls[: self.guesses + 1]


@dataclass
class PageResult:
    url: str  # final URL after redirects
    images: list[ImageCandidate]
    links: list[str]  # normalized absolute page URLs, not yet filtered for site/robots


def _pick_img(i: dict, base: str) -> tuple[str | None, int | None, str | None]:
    """(url, size hint, backup): the largest srcset candidate, else the lazy-load attribute, else what
    the browser shows. The natural size is only a reliable hint when the shown image is the one we
    picked. When it isn't, the shown copy is the backup if the browser really loaded it (bigger than a
    1x1 placeholder), in case the picked URL turns out to be broken or in a format we can't read."""
    shown = normalize_image_url(i.get("current") or i.get("src") or "", base)
    sets = [s for s in (i.get("dsrcset"), i.get("srcset"), *i.get("pic", [])) if s]
    picked = best_from_srcsets(sets) or i.get("dsrc")
    url = normalize_image_url(picked, base) if picked else None
    if not url:
        hint = min(i["nw"], i["nh"]) if i.get("nw") and i.get("nh") else None
        return shown, hint, None
    loaded = (i.get("nw") or 0) > 1 and (i.get("nh") or 0) > 1
    backup = shown if shown and shown != url and loaded and not shown.startswith("data:") else None
    return url, None, backup


def extract_candidates(
    raw: dict, page_url: str, prefer_originals: bool = True
) -> tuple[list[ImageCandidate], list[str]]:
    """With prefer_originals, each image is first tried at its full-size original: the image file its
    <a> links to, or the URL with the site's resize markers removed (see urls.original_candidates)."""
    base = raw.get("base") or page_url
    seen: set[str] = set()
    images: list[ImageCandidate] = []

    def add(url: str | None, hint: int | None = None, link: str | None = None, backup: str | None = None) -> None:
        if not url or url in seen:
            return
        guesses: list[str] = []
        if prefer_originals:
            # Known resize patterns first: a linked ".jpg" can also be a page (e.g. Wikipedia's File: pages).
            guesses = original_candidates(url)
            if link and link != url and link not in guesses and looks_like_image_file(link):
                guesses.append(link)
        backups = (backup,) if backup and backup != url and backup not in guesses else ()
        if guesses:
            # The displayed size says nothing about the original's, so don't skip on it.
            cand = ImageCandidate(guesses[0], page_url, None, (*guesses[1:], url, *backups), len(guesses))
        else:
            cand = ImageCandidate(url, page_url, hint, backups)
        if cand.url in seen:
            return
        seen.update(cand.ids)
        images.append(cand)

    for i in raw.get("imgs", []):
        url, hint, backup = _pick_img(i, base)
        add(url, hint, normalize_image_url(i["link"], base) if i.get("link") else None, backup)
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
            await page.evaluate(IMAGES_DONE_JS, IMAGES_WAIT_MS)
            raw = await page.evaluate(EXTRACT_JS)
        except PlaywrightError as e:
            raise PageError(f"page script failed: {str(e).splitlines()[0]}") from e
        final = normalize_page_url(page.url) or url
        images, links = extract_candidates(raw, final, prefer_originals)
        return PageResult(final, images, links)
    finally:
        await page.close()


class Robots:
    """robots.txt per origin, fetched through the run's browser context. A missing robots.txt (4xx) or
    one that can't be fetched means everything is allowed. A server error (5xx) means nothing is, as
    RFC 9309 asks; `on_note` is told so once per site."""

    def __init__(self, context: BrowserContext, timeout_s: float, on_note: Callable[[str], None] | None = None):
        self._context = context
        self._timeout_ms = timeout_s * 1000
        self._on_note = on_note
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
                elif resp.status >= 500:
                    parser = RobotFileParser()
                    parser.disallow_all = True
                    if self._on_note:
                        self._on_note(
                            f"robots.txt on {origin} answered HTTP {resp.status}, so the site counts as closed to "
                            "crawlers; turn off Respect robots.txt to crawl it anyway"
                        )
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
