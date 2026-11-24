// Reads text out of a PDF with pdf.js. It runs inside the PDF worker thread (or, in tests, in-process); the
// main thread never parses a file. pdf.js is used for text only: no rendering, no scripts, no fonts, no network.
import { existsSync } from 'node:fs'
import { itemsToText, matchesOnPage, PdfProblem, parsePageRange, type PageMatches, type PageText, type TextItemLike } from './text'

/** What to read. `pages` is the text the tool was given ("2-5"), checked against the real page count here. */
export type PdfJob =
  | { type: 'pages'; pages: string | null; /** Stop reading once this many characters are in hand (the page that crosses it is kept). */ maxChars: number }
  | { type: 'search'; pages: string | null; query: string }

export interface PdfJobResult {
  pageCount: number
  title: string
  /** The pages that were read, in order (a 'pages' job). */
  pages?: PageText[]
  /** Pages with matches (a 'search' job), and how many pages were searched. */
  hits?: PageMatches[]
  searched?: number
}

/** A page longer than this is cut: it is already far more than any chat can use. */
const MAX_PAGE_CHARS = 2_000_000

// The few browser classes pdf.js touches while loading in Node. Rendering is never used, so empty stand-ins do.
class StubMatrix {}
class StubPath {}

type PdfjsModule = typeof import('pdfjs-dist/legacy/build/pdf.mjs')
let loaded: Promise<PdfjsModule> | null = null

/**
 * Loads pdf.js once. Its Node path reaches for the optional canvas package; that package is not shipped and must
 * not load, so the lookup is answered with an empty module (pdf.js treats that as "no canvas" and goes on). Its fake worker is the
 * pdf.js worker module, handed over in a global because there is no separate worker file to open.
 */
export function loadPdfjs(): Promise<PdfjsModule> {
  loaded ??= (async () => {
    const g = globalThis as Record<string, unknown>
    g.DOMMatrix ??= StubMatrix
    g.Path2D ??= StubPath
    const real = process.getBuiltinModule
    process.getBuiltinModule = ((id: string) => {
      const mod = real.call(process, id as 'module') as typeof import('node:module')
      if (id !== 'module' && id !== 'node:module') return mod
      return {
        ...mod,
        createRequire: (from: string) => {
          const req = mod.createRequire(from)
          return Object.assign((request: string) => {
            // An empty module: pdf.js finds no canvas classes in it and uses the stand-ins above, without a warning.
            if (request.includes('@napi-rs/canvas')) return {}
            return req(request)
          }, req)
        }
      }
    }) as typeof process.getBuiltinModule
    try {
      const pdfjs = await import('pdfjs-dist/legacy/build/pdf.mjs')
      g.pdfjsWorker = await import('pdfjs-dist/legacy/build/pdf.worker.mjs')
      return pdfjs
    } finally {
      process.getBuiltinModule = real
    }
  })().catch((e) => {
    loaded = null
    throw e
  })
  return loaded
}

/** pdf.js errors by name, so a password or a damaged file gets its own message. */
function problemOf(err: unknown): PdfProblem {
  const name = (err as { name?: string })?.name ?? ''
  if (name === 'PasswordException') return new PdfProblem('password', 'This PDF is password-protected, so it cannot be read.')
  if (name === 'InvalidPDFException' || name === 'FormatError' || name === 'MissingPDFException' || name === 'UnknownErrorException')
    return new PdfProblem('damaged', 'This PDF is damaged or is not a readable PDF file.')
  return new PdfProblem('damaged', `This PDF could not be read: ${(err as Error)?.message ?? String(err)}`)
}

const cleanTitle = (t: unknown): string => (typeof t === 'string' ? t.replace(/[\u0000-\u001f]/g, ' ').replace(/\s+/g, ' ').trim().slice(0, 200) : '')

/**
 * Runs a read job on the bytes of a PDF. `cmapDir` holds pdf.js's character maps, which
 * some Chinese, Japanese and Korean files need; without it those files read with gaps but the read does not fail.
 */
export async function runPdfJob(data: Uint8Array, job: PdfJob, opts: { cmapDir?: string } = {}): Promise<PdfJobResult> {
  const pdfjs = await loadPdfjs()
  // pdf.js takes the folder as a URL and refuses one that does not end in "/", a Windows path included.
  const cmaps = opts.cmapDir && existsSync(opts.cmapDir) ? { cMapUrl: `${opts.cmapDir.replace(/[\\/]+$/, '')}/`, cMapPacked: true } : {}
  const task = pdfjs.getDocument({
    data,
    ...cmaps,
    // Nothing the file contains may run or fetch anything: no scripts (pdf.js 6 has no eval path), no font faces, no
    // forms, no WebAssembly, no URLs (the bytes are passed in).
    disableFontFace: true,
    useWasm: false,
    useSystemFonts: false,
    enableXfa: false,
    useWorkerFetch: false,
    isOffscreenCanvasSupported: false,
    verbosity: 0
  })
  let doc: Awaited<typeof task.promise>
  try {
    doc = await task.promise
  } catch (e) {
    await task.destroy().catch(() => undefined)
    throw problemOf(e)
  }
  try {
    const pageCount = doc.numPages
    let title = ''
    try {
      title = cleanTitle(((await doc.getMetadata()).info as { Title?: unknown } | undefined)?.Title)
    } catch {
      // A broken info dictionary only costs the title.
    }
    const wanted = parsePageRange(job.pages, pageCount)
    const readPage = async (n: number): Promise<PageText> => {
      try {
        const page = await doc.getPage(n)
        const content = await page.getTextContent()
        page.cleanup()
        const text = itemsToText(content.items as TextItemLike[])
        if (text.length <= MAX_PAGE_CHARS) return { page: n, text }
        return { page: n, text: `${text.slice(0, MAX_PAGE_CHARS)}\n[This page has more text than can be read (${MAX_PAGE_CHARS.toLocaleString('en-US')} characters); the rest is left out.]` }
      } catch {
        return { page: n, text: '', failed: true }
      }
    }
    if (job.type === 'pages') {
      const pages: PageText[] = []
      let chars = 0
      for (const n of wanted) {
        const p = await readPage(n)
        pages.push(p)
        chars += p.text.length
        if (chars > job.maxChars) break
      }
      return { pageCount, title, pages }
    }
    const hits: PageMatches[] = []
    for (const n of wanted) {
      const p = await readPage(n)
      const m = matchesOnPage(n, p.text, job.query)
      if (m) hits.push(m)
    }
    return { pageCount, title, hits, searched: wanted.length }
  } finally {
    await task.destroy().catch(() => undefined)
  }
}
