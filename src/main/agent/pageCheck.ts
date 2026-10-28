// check_page: opens a page of the owner's app in a hidden browser window and reports errors and
// what it shows. This file holds the parts that don't need Electron (which page, and the report).
import { promises as fs } from 'node:fs'
import { pathToFileURL, fileURLToPath } from 'node:url'
import { resolveInProject } from './pathGuard'

export interface PageTarget {
  /** Address loaded in the hidden window. */
  url: string
  /** Pages may only move to addresses that start with this (same server, or the project folder). */
  allowedPrefix: string
}

export interface PageMessage {
  level: 'error' | 'warning'
  message: string
  source?: string
  line?: number
}

export interface PageReport {
  url: string
  title: string
  /** HTTP status of the page itself (web servers only). */
  status?: number
  /** Why the page didn't load at all. */
  loadError?: string
  /** The page froze or its window crashed. */
  crashed?: string
  messages: PageMessage[]
  failedRequests: { url: string; problem: string }[]
  text: string
  controls: string[]
  /** Images, canvases, videos and drawings on the page. */
  visuals: number
}

const LOCAL_HOST = /^(localhost|127\.0\.0\.1|\[::1\]|0\.0\.0\.0)$/i

/** Works out which page to open: a local web address, or an HTML file in the project. */
export async function pageTarget(root: string, input: unknown): Promise<PageTarget> {
  const raw = typeof input === 'string' ? input.trim().replace(/^["']|["']$/g, '') : ''
  if (!raw) throw new Error('Give the page to check: a local address like "http://localhost:5173" or an HTML file in the project like "index.html".')
  if (/^https?:\/\//i.test(raw)) {
    let url: URL
    try {
      url = new URL(raw)
    } catch {
      throw new Error(`"${raw}" is not a valid web address.`)
    }
    if (!LOCAL_HOST.test(url.hostname)) {
      throw new Error(`check_page only opens the owner's own app: addresses on this computer (localhost) or HTML files in the project. "${raw}" is on the internet.`)
    }
    if (url.hostname === '0.0.0.0') url.hostname = 'localhost'
    return { url: url.href, allowedPrefix: `${url.origin}/` }
  }
  const path = /^file:/i.test(raw) ? fileURLToPath(raw) : raw.replace(/[?#].*$/, '')
  const target = await resolveInProject(root, path)
  const st = await fs.stat(target.abs).catch(() => null)
  if (!st?.isFile()) throw new Error(`"${target.rel || raw}" is not a file in the project. Give an HTML file like "index.html", or a local address like "http://localhost:5173".`)
  return { url: pathToFileURL(target.abs).href, allowedPrefix: pathToFileURL(root).href.replace(/\/?$/, '/') }
}

const clip = (s: string, n: number): string => (s.length > n ? `${s.slice(0, n)}…` : s)
const MAX_TEXT = 3000
const MAX_LINES = 15

function where(m: PageMessage): string {
  if (!m.source) return ''
  const file = m.source.split(/[\\/]/).pop()?.replace(/[?#].*$/, '') || m.source
  return ` (${file}${m.line ? `:${m.line}` : ''})`
}

/** True when the report shows something is wrong with the page. */
export function pageHasProblems(r: PageReport): boolean {
  return !!(r.loadError || r.crashed || (r.status && r.status >= 400) || r.failedRequests.length || r.messages.some((m) => m.level === 'error'))
}

/** The report as text for the model. */
export function formatPageReport(r: PageReport): string {
  const lines: string[] = []
  if (r.loadError) {
    lines.push(`The page did not load: ${r.loadError}`)
    if (/ERR_CONNECTION_REFUSED/i.test(r.loadError)) lines.push('Nothing is answering at that address. Start the server first (start_background_command) and check its output for the address it uses.')
    return lines.join('\n')
  }
  lines.push(`Opened ${r.url}${r.status ? ` (status ${r.status})` : ''}. Title: ${r.title ? `"${clip(r.title, 120)}"` : '(none)'}`)
  if (r.crashed) lines.push(`Problem: ${r.crashed}`)

  const errors = r.messages.filter((m) => m.level === 'error')
  const warnings = r.messages.filter((m) => m.level === 'warning')
  if (errors.length) {
    lines.push(`\nErrors in the page (${errors.length}):`)
    for (const m of errors.slice(0, MAX_LINES)) lines.push(`- ${clip(m.message, 500)}${where(m)}`)
    if (errors.length > MAX_LINES) lines.push(`- …and ${errors.length - MAX_LINES} more`)
  } else lines.push('No errors in the page.')
  if (warnings.length) {
    lines.push(`\nWarnings (${warnings.length}):`)
    for (const m of warnings.slice(0, 5)) lines.push(`- ${clip(m.message, 300)}${where(m)}`)
  }
  if (r.failedRequests.length) {
    lines.push(`\nFiles or requests that failed to load (${r.failedRequests.length}):`)
    for (const f of r.failedRequests.slice(0, MAX_LINES)) lines.push(`- ${clip(f.url, 200)}: ${f.problem}`)
  }

  const text = r.text.replace(/\n{3,}/g, '\n\n').trim()
  if (text) lines.push(`\nWhat the page shows (its text):\n${clip(text, MAX_TEXT)}`)
  if (r.controls.length) lines.push(`\nButtons and fields: ${r.controls.slice(0, 40).join(', ')}${r.controls.length > 40 ? ', …' : ''}`)
  if (r.visuals) lines.push(`Images and drawings on the page: ${r.visuals}`)
  if (!text && !r.visuals && !r.controls.length) lines.push('\nThe page looks empty: no text, buttons or images are visible. Something probably went wrong (check the errors above, or the script that builds the page).')
  return lines.join('\n')
}

/**
 * Runs inside the page to describe what it shows. Kept as a string so it doesn't depend on how
 * the bundler transforms functions.
 */
export const PAGE_SNAPSHOT_SCRIPT = `(() => {
  const visible = (el) => { const r = el.getBoundingClientRect(); const s = getComputedStyle(el); return r.width > 0 && r.height > 0 && s.visibility !== 'hidden' && s.display !== 'none' }
  const label = (el) => (el.getAttribute('aria-label') || el.innerText || el.value || el.getAttribute('placeholder') || el.getAttribute('title') || el.getAttribute('name') || '').trim().replace(/\\s+/g, ' ').slice(0, 60)
  const controls = []
  for (const el of document.querySelectorAll('button, a[href], input, select, textarea, [role=button]')) {
    if (!visible(el) || controls.length >= 60) continue
    const tag = el.tagName.toLowerCase()
    const kind = tag === 'input' ? 'field (' + (el.type || 'text') + ')' : tag === 'a' ? 'link' : tag === 'select' ? 'list' : tag === 'textarea' ? 'text box' : 'button'
    const name = label(el)
    controls.push(name ? kind + ' "' + name + '"' : kind)
  }
  const visuals = [...document.querySelectorAll('img, canvas, svg, video')].filter(visible).length
  return { title: document.title || '', text: document.body ? document.body.innerText.slice(0, 20000) : '', controls, visuals }
})()`
