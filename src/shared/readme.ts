// README clean-up for display inside the launcher, which shows the app's name and logo itself.

const LOGO_ALT = /alt="Rul[zZ]"/

/**
 * Drops the RulZ logo images and the leading "# App name" heading, which would repeat the page header.
 * Everything else is left as written.
 */
export function prepareReadme(markdown: string): string {
  let md = markdown.replace(/\r\n/g, '\n')
  // <p align="center"><picture>…<img alt="RulZ">…</picture></p> and bare <picture> logo blocks.
  md = md.replace(/<p[^>]*>\s*<picture>[\s\S]*?<\/picture>\s*<\/p>\n?/gi, (block) => (LOGO_ALT.test(block) ? '' : block))
  md = md.replace(/<picture>[\s\S]*?<\/picture>\n?/gi, (block) => (LOGO_ALT.test(block) ? '' : block))
  md = md.replace(/^!\[Rul[zZ]\]\([^)]*\)\s*\n/gm, '')
  md = md.replace(/^\s*#\s+[^\n]+\n/, '')
  return md.trimStart()
}

/** Resolves a README-relative URL against a base; absolute URLs and #anchors are returned unchanged. */
export function resolveReadmeUrl(url: string, base: string): string {
  if (/^[a-z][a-z0-9+.-]*:/i.test(url) || url.startsWith('#') || url.startsWith('//')) return url
  try {
    return new URL(url.replace(/^\.\//, ''), base).toString()
  } catch {
    return url
  }
}

/** GitHub's heading anchors: lower case, punctuation dropped, spaces to dashes, duplicates numbered. */
export function createSlugger(): (text: string) => string {
  const seen = new Map<string, number>()
  return (text) => {
    const base = text
      .toLowerCase()
      .trim()
      .replace(/[^\p{L}\p{N}\s_-]/gu, '')
      .replace(/\s/g, '-')
    const n = seen.get(base) ?? 0
    seen.set(base, n + 1)
    return n ? `${base}-${n}` : base
  }
}
