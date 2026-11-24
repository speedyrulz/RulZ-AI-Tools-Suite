// Pure text handling for PDF reading: page ranges, turning pdf.js text items into lines, searching, and the
// layout of a tool result. Nothing here touches pdf.js or the disk, so it runs in tests as it is.

/** The part of a pdf.js text item the line builder uses. */
export interface TextItemLike {
  str?: string
  /** [a, b, c, d, e, f]: e and f are where the text starts, d is about its height. */
  transform?: number[]
  width?: number
  height?: number
  hasEOL?: boolean
}

/** Why a read failed, so the caller can say it plainly. */
export class PdfProblem extends Error {
  constructor(
    readonly code: 'password' | 'damaged' | 'range' | 'timeout' | 'crash' | 'stopped' | 'busy' | 'unavailable',
    message: string
  ) {
    super(message)
    this.name = 'PdfProblem'
  }
}

export const NO_TEXT_PAGE = '(no readable text on this page; it may be a scan)'
export const FAILED_PAGE = '(this page could not be read; the file may be damaged)'

/** Pages are written out at about this many characters per tool result, cut at page boundaries. */
export const RESULT_BUDGET = 20_000
/** Longest search text: it is repeated in every snippet. */
export const MAX_QUERY_CHARS = 200
/** Characters of context on each side of a search match. */
const SNIPPET_SIDE = 70
const SNIPPETS_PER_PAGE = 3
/** Pages listed in a search result before it says there are more. */
export const SEARCH_PAGES_SHOWN = 30

const n = (v: number): string => v.toLocaleString('en-US')

// ---------- page ranges ----------

/** Turns "3", "2-5", "1,4-6" or "7-" into ascending, unique page numbers. A plain number is accepted too. */
export function parsePageRange(spec: unknown, pageCount: number): number[] {
  if (spec === undefined || spec === null || spec === '') return Array.from({ length: pageCount }, (_, i) => i + 1)
  const text = typeof spec === 'number' ? String(spec) : typeof spec === 'string' ? spec.trim() : ''
  if (!text) throw new PdfProblem('range', 'pages must be text such as "3", "2-5" or "1,4-6".')
  const wanted = new Set<number>()
  for (const part of text.split(',')) {
    const m = /^\s*(\d+)\s*(?:(-)\s*(\d*))?\s*$/.exec(part)
    if (!m) throw new PdfProblem('range', `"${part.trim()}" is not a page or page range. Use "3", "2-5" or "1,4-6".`)
    const from = Number(m[1])
    const to = m[2] ? (m[3] ? Number(m[3]) : pageCount) : from
    if (from < 1) throw new PdfProblem('range', 'Pages are numbered from 1.')
    if (to < from) throw new PdfProblem('range', `"${part.trim()}" runs backwards; write the lower page first.`)
    if (from > pageCount) throw new PdfProblem('range', `This PDF has ${pageCount} page${pageCount === 1 ? '' : 's'}; page ${from} does not exist.`)
    if (to > pageCount) throw new PdfProblem('range', `This PDF has ${pageCount} page${pageCount === 1 ? '' : 's'}; "${part.trim()}" goes past the end.`)
    for (let p = from; p <= to; p++) wanted.add(p)
  }
  return [...wanted].sort((a, b) => a - b)
}

/** Page numbers as a short range text, the form `pages` accepts: [1,2,3,7] -> "1-3,7". */
export function formatPageList(pages: number[]): string {
  const out: string[] = []
  for (let i = 0; i < pages.length; ) {
    let j = i
    while (j + 1 < pages.length && pages[j + 1] === pages[j] + 1) j++
    out.push(j > i ? `${pages[i]}-${pages[j]}` : String(pages[i]))
    i = j + 1
  }
  return out.join(',')
}

// ---------- text items to lines ----------

// Zero-width joiners stay: Persian, Arabic and Indic text needs them.
const clean = (s: string): string => s.replace(/\u0000/g, '').replace(/[\u00a0\u2007\u202f]/g, ' ').replace(/[\u200b\u2060\ufeff\u00ad]/g, '')

/**
 * Joins pdf.js text items into lines. A line break comes from the item's own end-of-line mark or from the text moving
 * to another baseline; a space from a visible gap between two items on one line. No columns or tables are rebuilt:
 * text comes in the order the file draws it.
 */
export function itemsToText(items: TextItemLike[]): string {
  let out = ''
  let prev: { y: number; end: number; size: number } | null = null
  const endsWithBreak = (): boolean => out === '' || out.endsWith('\n')
  for (const it of items) {
    if (typeof it.str !== 'string') continue
    const str = clean(it.str)
    const t = it.transform
    const x = t && Number.isFinite(t[4]) ? t[4] : null
    const y = t && Number.isFinite(t[5]) ? t[5] : null
    const size = Math.abs(it.height || (t && t[3]) || 0) || 10
    if (str.length && prev && x !== null && y !== null) {
      const dy = Math.abs(y - prev.y)
      if (dy > Math.max(prev.size, size) * 0.5) {
        // A new baseline. A gap of more than about a line and a half is a paragraph break.
        out = out.replace(/ +$/, '')
        if (!endsWithBreak()) out += dy > Math.max(prev.size, size) * 1.9 ? '\n\n' : '\n'
        else if (dy > Math.max(prev.size, size) * 1.9 && !out.endsWith('\n\n')) out += '\n'
      } else if (!endsWithBreak() && !/\s$/.test(out) && !/^\s/.test(str) && x - prev.end > Math.max(prev.size, size) * 0.15) {
        out += ' '
      }
    }
    out += str
    if (str.length && x !== null && y !== null) prev = { y, end: x + (it.width || 0), size }
    if (it.hasEOL) {
      out = out.replace(/ +$/, '')
      if (!endsWithBreak()) out += '\n'
    }
  }
  return out
    .split('\n')
    .map((l) => l.replace(/[ \t]+$/, ''))
    .join('\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim()
}

// ---------- search ----------

export interface PageMatches {
  page: number
  count: number
  snippets: string[]
}

const flat = (s: string): string => s.replace(/\s+/g, ' ').trim()

/** Case-insensitive plain-text matches of `query` on one page (whitespace in the text and the query is one space). */
export function matchesOnPage(page: number, text: string, query: string): PageMatches | null {
  const q = flat(query).toLowerCase()
  if (!q) return null
  const body = flat(text)
  const lower = body.toLowerCase()
  const snippets: string[] = []
  let count = 0
  let lastEnd = -1
  for (let at = lower.indexOf(q); at !== -1; at = lower.indexOf(q, at + q.length)) {
    count++
    if (snippets.length >= SNIPPETS_PER_PAGE || at < lastEnd) continue
    const from = Math.max(0, at - SNIPPET_SIDE)
    const to = Math.min(body.length, at + q.length + SNIPPET_SIDE)
    lastEnd = to
    snippets.push(`${from > 0 ? '…' : ''}${body.slice(from, to)}${to < body.length ? '…' : ''}`)
  }
  return count ? { page, count, snippets } : null
}

// ---------- result layout ----------

export interface PdfInfo {
  name: string
  pageCount: number
  title?: string
}

export interface PageText {
  page: number
  text: string
  failed?: boolean
}

const header = (info: PdfInfo): string =>
  `${info.name}: ${info.pageCount} page${info.pageCount === 1 ? '' : 's'}${info.title ? `, title "${info.title}"` : ''}`

const pageBody = (p: PageText): string => (p.failed ? FAILED_PAGE : p.text.trim() ? p.text : NO_TEXT_PAGE)

export interface RenderedPages {
  text: string
  /** The call that continues where this result stopped, or null when everything asked for is shown. */
  next: { pages: string; offset?: number } | null
  /** Every page that was read has no text at all. */
  noText: boolean
}

/**
 * Lays out pages read from a PDF: a header, then "--- Page N ---" sections, cut at page boundaries once `budget`
 * characters are used. `asked` is every page the call asked for (ascending); `pages` the ones read, which may stop
 * early. A first page longer than the budget is cut inside the page and continues at `offset`, which counts from the
 * start of the first page asked for, so the pages after it stay in the next call.
 */
export function renderPages(info: PdfInfo, asked: number[], pages: PageText[], opts: { offset?: number; budget?: number } = {}): RenderedPages {
  const budget = opts.budget ?? RESULT_BUDGET
  const head = header(info)
  if (!pages.length) return { text: `${head}\nNo pages were read.`, next: null, noText: false }
  if (pages.every((p) => !p.failed && !p.text.trim())) {
    const range = pages.length === 1 ? `page ${pages[0].page}` : `${pages.length} pages (${formatPageList(pages.map((p) => p.page))})`
    const rest = asked.filter((p) => !pages.some((q) => q.page === p))
    return {
      text: `${head}\nNo readable text on ${range}: it looks like scanned images. This tool reads text only, with no OCR.`,
      next: rest.length ? { pages: formatPageList(rest) } : null,
      noText: !rest.length
    }
  }
  const parts: string[] = [head]
  let used = head.length
  let shown = 0
  let next: RenderedPages['next'] = null
  for (let i = 0; i < pages.length; i++) {
    const p = pages[i]
    const body = pageBody(p)
    const start = i === 0 ? Math.min(opts.offset ?? 0, body.length) : 0
    const piece = body.slice(start)
    const label = `\n\n--- Page ${p.page} ---\n`
    if (shown > 0 && used + label.length + piece.length > budget) {
      next = { pages: formatPageList(asked.filter((a) => a >= p.page)) }
      break
    }
    if (shown === 0 && label.length + piece.length > budget) {
      // One page longer than a result: show a slice and say where it goes on.
      const end = start + Math.max(1000, budget - used - label.length - 200)
      const cut = end < body.length ? sliceAtBreak(body, start, end) : body.length
      parts.push(`${label}${body.slice(start, cut)}`)
      if (cut < body.length) {
        const left = formatPageList(asked.filter((a) => a >= p.page))
        parts.push(`\n\n(Page ${p.page} continues: characters ${n(start)}-${n(cut)} of ${n(body.length)} shown. Call again with pages "${left}" and offset ${cut}.)`)
        return { text: parts.join(''), next: { pages: left, offset: cut }, noText: false }
      }
      used += label.length + body.length - start
      shown++
      continue
    }
    parts.push(`${label}${piece}`)
    used += label.length + piece.length
    shown++
  }
  if (!next) {
    const rest = asked.filter((a) => a > pages[pages.length - 1].page)
    if (rest.length) next = { pages: formatPageList(rest) }
  }
  if (next) parts.push(`\n\n(More pages follow. Call again with pages "${next.pages}".)`)
  return { text: parts.join(''), next, noText: false }
}

/** Ends a slice at a line or word break near `end`, so a cut does not split a word. */
function sliceAtBreak(body: string, start: number, end: number): number {
  const window = body.slice(Math.max(start, end - 300), end)
  const nl = window.lastIndexOf('\n')
  if (nl > 0) return end - window.length + nl + 1
  const sp = window.lastIndexOf(' ')
  return sp > 0 ? end - window.length + sp + 1 : end
}

/** A search result: pages with matches and short snippets. */
export function renderSearch(info: PdfInfo, query: string, scope: string, hits: PageMatches[], pagesSearched: number): string {
  const head = header(info)
  const where = scope ? `pages ${scope}` : 'all pages'
  if (!hits.length) return `${head}\nNo matches for "${query}" in ${where} (${n(pagesSearched)} page${pagesSearched === 1 ? '' : 's'} searched).`
  const total = hits.reduce((s, h) => s + h.count, 0)
  const lines: string[] = []
  let used = head.length
  for (const h of hits.slice(0, SEARCH_PAGES_SHOWN)) {
    const line = `Page ${h.page} (${h.count} match${h.count === 1 ? '' : 'es'}):\n` + h.snippets.map((s) => `  ${s}`).join('\n')
    if (lines.length && used + line.length + 2 > RESULT_BUDGET) break
    lines.push(line)
    used += line.length + 2
  }
  const left = hits.length - lines.length
  const more = left
    ? `\n\n(${left} more page${left === 1 ? ' has' : 's have'} matches, from page ${hits[lines.length].page} on. Search again with pages "${hits[lines.length].page}-${hits[hits.length - 1].page}", or for something more specific.)`
    : ''
  return `${head}\n${n(total)} match${total === 1 ? '' : 'es'} for "${query}" on ${hits.length} page${hits.length === 1 ? '' : 's'} (${where}).\n\n${lines.join('\n\n')}${more}\nRead a page with pages "N".`
}

// ---------- attachments ----------

/** Characters of PDF text a chat attachment may hold. */
export const MAX_ATTACH_CHARS = 400_000

export interface AttachmentText {
  text: string
  pagesIncluded: number
  /** The text stops before the end of the PDF. */
  cut: boolean
  /** No page has readable text. */
  noText: boolean
}

/** The text of an attached PDF: "--- Page N ---" sections up to the character cap, cut at a page boundary. */
export function attachmentText(info: PdfInfo, pages: PageText[], cap: number = MAX_ATTACH_CHARS): AttachmentText {
  if (!pages.length || pages.every((p) => !p.failed && !p.text.trim())) return { text: '', pagesIncluded: 0, cut: false, noText: true }
  const parts: string[] = []
  let used = 0
  let included = 0
  let cutInside = false
  for (const p of pages) {
    const section = `--- Page ${p.page} ---\n${pageBody(p)}`
    const sep = parts.length ? 2 : 0
    if (included > 0 && used + sep + section.length > cap) break
    if (included === 0 && section.length > cap) {
      parts.push(section.slice(0, cap))
      cutInside = true
      included++
      break
    }
    parts.push(section)
    used += sep + section.length
    included++
  }
  const last = pages[included - 1].page
  const cut = cutInside || included < pages.length || last < info.pageCount
  let text = parts.join('\n\n')
  if (cut) {
    text += `\n\n[The text stops${cutInside ? ' inside' : ' after'} page ${last} of ${info.pageCount}: a chat attachment holds at most ${n(cap)} characters of PDF text, so the rest of this PDF is not included.]`
  }
  return { text: `${header(info)}\n\n${text}`, pagesIncluded: included, cut, noText: false }
}
