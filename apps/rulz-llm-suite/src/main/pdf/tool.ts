// The read_pdf tool's work and the attachment read, both on top of a PdfReader. fileTools.ts checks and expands the
// path first; this file takes the path as it is.
import { promises as fs } from 'node:fs'
import { basename, join, parse, resolve } from 'node:path'
import { MAX_PDF_ATTACH_BYTES, type PdfAttachmentResult } from '@shared/pdf'
import type { PdfReader } from './reader'
import {
  attachmentText,
  MAX_ATTACH_CHARS,
  MAX_QUERY_CHARS,
  parsePageRange,
  PdfProblem,
  renderPages,
  renderSearch,
  RESULT_BUDGET
} from './text'

export const MAX_PDF_BYTES = 200 * 1024 * 1024

const mb = (bytes: number): string => `${Math.round(bytes / 1024 / 1024)} MB`

/** PDFs start with "%PDF-", within the first 1024 bytes (some files have a little junk in front). */
export function hasPdfMagic(bytes: Uint8Array): boolean {
  const head = bytes.subarray(0, 1024)
  for (let i = 0; i + 5 <= head.length; i++) {
    if (head[i] === 0x25 && head[i + 1] === 0x50 && head[i + 2] === 0x44 && head[i + 3] === 0x46 && head[i + 4] === 0x2d) return true
  }
  return false
}

type Out = { text: string; isError: boolean }
const fail = (text: string): Out => ({ text, isError: true })

function describe(err: unknown, path: string): string {
  const code = (err as NodeJS.ErrnoException).code
  if (code === 'ENOENT') return `"${path}" does not exist.`
  if (code === 'EPERM' || code === 'EACCES') return `Access to "${path}" is denied.`
  if (code === 'EBUSY') return `"${path}" is in use by another program.`
  return `Couldn't read "${path}": ${(err as Error).message}`
}

const problemText = (e: unknown): string => (e instanceof PdfProblem ? e.message : `This PDF could not be read: ${(e as Error)?.message ?? String(e)}`)

const NETWORK_OR_DEVICE = /^[\\/]{2}/
const MAX_LINKS = 40

/**
 * Follows the links in a path one at a time, reading only where each one points. realpath alone would contact a share
 * a link leads to (\\host\share) before it could be refused, and contacting it signs in to that host.
 */
async function followLinks(full: string, refuse: (target: string) => Error): Promise<string> {
  let path = full
  for (let hop = 0; hop <= MAX_LINKS; hop++) {
    let at = parse(path).root
    const rest = path.slice(at.length).split(/[\\/]+/).filter(Boolean)
    let next: string | null = null
    for (let i = 0; i < rest.length && next === null; i++) {
      const here = join(at, rest[i])
      if ((await fs.lstat(here)).isSymbolicLink()) {
        // "\\?\C:\..." is a local path in its long form.
        const target = (await fs.readlink(here)).replace(/^\\\\\?\\(?=[a-z]:[\\/])/i, '')
        if (NETWORK_OR_DEVICE.test(target)) throw refuse(target)
        next = resolve(at, target, ...rest.slice(i + 1))
        if (NETWORK_OR_DEVICE.test(next)) throw refuse(next)
      }
      at = here
    }
    if (next === null) return path
    path = next
  }
  throw new Error(`"${full}" goes through too many links.`)
}

/** A path made safe to read: no data streams, and no link that leads to a network share or device. Returns the real file. */
async function localFile(full: string): Promise<string> {
  if (/^[a-z]:/i.test(full) && full.indexOf(':', 2) >= 0) throw new Error(`"${full}" names a data stream; give the file's own path.`)
  const refuse = (to: string): Error => new Error(`"${full}" leads to a network or device path (${to}); only files on this computer can be read.`)
  // A drive letter that stands for a share only shows up here.
  const real = await fs.realpath(await followLinks(full, refuse))
  if (NETWORK_OR_DEVICE.test(real)) throw refuse(real)
  return real
}

/** Whole-number option, or the default when it is missing. */
function wholeNumber(v: unknown, what: string): number | undefined {
  if (v === undefined || v === null || v === '') return undefined
  const n = Number(v)
  if (!Number.isInteger(n) || n < 0) throw new PdfProblem('range', `${what} must be a whole number of 0 or more.`)
  return n
}

export async function readPdfTool(full: string, args: { pages?: unknown; query?: unknown; offset?: unknown }, reader: PdfReader | undefined, signal?: AbortSignal): Promise<Out> {
  if (!reader) return fail('PDF reading is not available here.')
  let real: string
  let size: number
  try {
    real = await localFile(full)
    const st = await fs.stat(real)
    if (st.isDirectory()) return fail(`"${full}" is a folder, not a PDF; use list_directory.`)
    if (!st.isFile()) return fail(`"${full}" is not a regular file.`)
    size = st.size
  } catch (e) {
    return fail(e instanceof Error && !(e as NodeJS.ErrnoException).code ? e.message : describe(e, full))
  }
  if (size > MAX_PDF_BYTES) return fail(`"${full}" is ${mb(size)}; read_pdf reads PDFs of up to ${mb(MAX_PDF_BYTES)}.`)

  let offset: number | undefined
  let query = ''
  try {
    offset = wholeNumber(args.offset, 'offset')
    query = typeof args.query === 'string' ? args.query.trim() : args.query === undefined || args.query === null ? '' : String(args.query).trim()
    const spec = args.pages === undefined || args.pages === null || args.pages === '' ? null : String(args.pages).trim()
    if (offset && query) throw new PdfProblem('range', 'offset and query cannot be combined: offset continues reading a long page.')
    if (query.length > MAX_QUERY_CHARS) throw new PdfProblem('range', `query can be at most ${MAX_QUERY_CHARS} characters long.`)
    const bytes = await fs.readFile(real)
    if (bytes.length > MAX_PDF_BYTES) return fail(`"${full}" is larger than ${mb(MAX_PDF_BYTES)}; read_pdf reads PDFs of up to ${mb(MAX_PDF_BYTES)}.`)
    if (!hasPdfMagic(bytes)) return fail(`"${full}" is not a PDF file (it does not start with %PDF-).`)
    const name = basename(full)
    if (query) {
      const r = await reader.run(bytes, { type: 'search', pages: spec, query }, signal)
      return { text: renderSearch({ name, pageCount: r.pageCount, title: r.title }, query, spec ?? '', r.hits ?? [], r.searched ?? 0), isError: false }
    }
    const r = await reader.run(bytes, { type: 'pages', pages: spec, maxChars: RESULT_BUDGET + (offset ?? 0) }, signal)
    const asked = parsePageRange(spec, r.pageCount)
    const out = renderPages({ name, pageCount: r.pageCount, title: r.title }, asked, r.pages ?? [], { offset })
    return { text: out.text, isError: false }
  } catch (e) {
    if (signal?.aborted) return fail('The tool call was stopped.')
    if (e instanceof PdfProblem || (e as NodeJS.ErrnoException).code === undefined) return fail(problemText(e))
    return fail(describe(e, full))
  }
}

/** The text of a PDF the user attached: checks size and type again here, since the window is not trusted. */
export async function readPdfAttachment(name: unknown, bytes: unknown, reader: PdfReader | undefined, signal?: AbortSignal): Promise<PdfAttachmentResult> {
  const label = typeof name === 'string' && name.trim() ? basename(name.trim()) : 'The PDF'
  if (!reader) return { ok: false, error: 'PDF reading is not available here.' }
  if (!ArrayBuffer.isView(bytes)) return { ok: false, error: `${label} could not be read.` }
  const data = new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  if (data.length > MAX_PDF_ATTACH_BYTES) return { ok: false, error: `${label} is larger than ${mb(MAX_PDF_ATTACH_BYTES)}` }
  if (!hasPdfMagic(data)) return { ok: false, error: `${label} is not a PDF file.` }
  try {
    const r = await reader.run(data, { type: 'pages', pages: null, maxChars: MAX_ATTACH_CHARS }, signal)
    const pages = r.pages ?? []
    const text = attachmentText({ name: label, pageCount: r.pageCount, title: r.title }, pages)
    if (text.noText) {
      return { ok: false, error: `${label} has no readable text; it looks like a scan, and there is no OCR here. Nothing was attached.` }
    }
    return { ok: true, text: text.text, pageCount: r.pageCount, pagesIncluded: text.pagesIncluded, cut: text.cut }
  } catch (e) {
    return { ok: false, error: `${label}: ${problemText(e)}` }
  }
}
