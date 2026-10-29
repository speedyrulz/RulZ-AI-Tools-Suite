r"""Catalog screenshots of RulZ Image Crawler (folder scan of procedurally drawn demo pictures).

    python -B capture_image_crawler.py --url http://127.0.0.1:8011/ --folder R:\Pictures\Wallpapers ^
        --real "<the folder R: is SUBSTed to>" --out R:\out\rulz-image-crawler

Needs the app running (with a model server) and Playwright's Chromium, i.e. the app's own venv. The app resolves
R: to the real folder it is SUBSTed to, so --real is rewritten back to R:\ on the page before each capture (and
checked), keeping the Windows user name out of the pictures.
"""
from __future__ import annotations

import argparse

from playwright.sync_api import sync_playwright

FIX = """([r, s]) => {
  const w = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT)
  let n
  while ((n = w.nextNode())) if (n.textContent.includes(r)) n.textContent = n.textContent.split(r).join(s)
}"""

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8011/")
ap.add_argument("--folder", required=True)
ap.add_argument("--real", required=True)
ap.add_argument("--prompt", default="landscapes with mountains")
ap.add_argument("--out", required=True)
a = ap.parse_args()
real = a.real.rstrip("\\") + "\\"

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1600, "height": 960})

    def shot(name: str) -> None:
        page.evaluate(FIX, [real, "R:\\"])
        assert real not in page.inner_text("body"), "the real folder is still on the page"
        page.wait_for_timeout(500)
        page.screenshot(path=f"{a.out}/{name}")

    page.goto(a.url)
    page.wait_for_selector("#modelLine")
    page.check("input[type=radio][value=folder]")
    page.fill("#folderPath", a.folder)
    page.fill("#prompt", a.prompt)
    page.click("#genBtn")
    page.wait_for_function("document.querySelector('#rubric').value.length > 20", timeout=120000)
    shot("02-checklist.png")
    page.click("#startBtn")
    page.wait_for_function("!document.querySelector('#newBtn').classList.contains('hidden')", timeout=600000)
    page.wait_for_timeout(1500)
    page.evaluate("scrollTo(0, document.querySelector('#results').getBoundingClientRect().top + scrollY - 20)")
    shot("01-matches.png")
    page.click("#themeBtn")
    page.evaluate("scrollTo(0, document.querySelector('#progress').getBoundingClientRect().top + scrollY - 20)")
    shot("03-light-mode.png")
    browser.close()
