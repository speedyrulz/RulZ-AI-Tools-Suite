// The read_pdf tool of the agent: a project path (through the same guard as read_file) or "attachment:<id>".
// An attached PDF lives in the app's data folder; its real path never reaches the model, so everything that
// comes back names it by file name.
import { promises as fs } from 'node:fs'
import { dirname, sep } from 'node:path'
import { PDF_ID } from '@shared/pdf'
import type { PdfReader } from '../pdf/reader'
import { readPdfTool } from '../pdf/tool'
import type { AttachmentStore } from './attachments'
import type { ToolOutcome } from './fileTools'
import { PathError, resolveInProject } from './pathGuard'

export interface PdfTools {
  reader: PdfReader
  attachments: Pick<AttachmentStore, 'resolve' | 'remove' | 'list' | 'sweep'>
}

const PREFIX = /^attachment:/i
const fail = (text: string): ToolOutcome => ({ text, isError: true })

/** Every spelling of a path (and the folders above it that hold attachments) replaced by `to`, so none is left in a text. */
function scrub(text: string, hide: string[], to: string): string {
  let out = text
  for (const h of hide.filter(Boolean).sort((a, b) => b.length - a.length)) {
    out = out.split(h).join(to).split(h.split(sep).join('/')).join(to)
  }
  return out
}

export async function runReadPdf(root: string, args: Record<string, unknown>, pdf: PdfTools | undefined, signal: AbortSignal): Promise<ToolOutcome> {
  if (!pdf) return fail('PDF reading is not available here.')
  if (typeof args.path !== 'string' || !args.path.trim()) return fail('Give the path of the PDF, like "docs/manual.pdf", or "attachment:<id>" for an attached one.')
  const raw = args.path.trim().replace(/^["']|["']$/g, '')

  if (PREFIX.test(raw)) {
    const id = raw.slice('attachment:'.length)
    if (!raw.startsWith('attachment:') || !PDF_ID.test(id)) {
      return fail('That is not an attachment id. Use "attachment:" followed by the 16 characters (0-9, a-f) listed with the attached PDF.')
    }
    const found = await pdf.attachments.resolve(root, id)
    if (!found) return fail(`There is no attached PDF with the id ${id} in this project. It may have been removed together with its message.`)
    const idDir = dirname(found.path)
    const hide = [found.path, idDir, dirname(idDir)]
    const real = await fs.realpath(found.path).catch(() => '')
    if (real) hide.push(real, dirname(real), dirname(dirname(real)))
    try {
      const out = await readPdfTool(found.path, args, pdf.reader, signal)
      return { text: scrub(out.text, hide, found.name), isError: out.isError }
    } catch (e) {
      if (signal.aborted) return fail('The tool call was stopped.')
      return fail(scrub(`The PDF could not be read: ${(e as Error).message}`, hide, found.name))
    }
  }

  let target
  try {
    target = await resolveInProject(root, raw)
  } catch (e) {
    return fail(e instanceof PathError ? e.message : `Couldn't use "${raw}": ${(e as Error).message}`)
  }
  if (!target.rel) return fail('That is the project folder, not a PDF. Use list_files to see what is in it.')
  const out = await readPdfTool(target.abs, args, pdf.reader, signal)
  return { text: out.text.split(target.abs).join(target.rel), isError: out.isError }
}
