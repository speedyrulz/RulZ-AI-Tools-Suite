import { MAX_PDF_ATTACH_BYTES } from '@shared/pdf'
import type { Attachment } from '@shared/types'
import { errorMessage, invoke } from './api'
import { uid } from './format'

const MAX_IMAGE_BYTES = 20 * 1024 * 1024
const MAX_TEXT_BYTES = 2 * 1024 * 1024

const TEXT_EXTENSIONS = new Set(
  (
    'txt md markdown csv tsv json jsonl yaml yml toml ini cfg conf log xml html htm css scss less ' +
    'js jsx ts tsx mjs cjs py rb go rs java kt kts c h cpp hpp cc cs php sh bash zsh ps1 bat cmd sql ' +
    'r lua dart swift scala vue svelte tex rst gradle cmake mk env gitignore dockerfile'
  ).split(' ')
)

function isTextFile(file: File): boolean {
  if (file.type.startsWith('text/') || /json|xml|javascript|typescript|yaml|toml/.test(file.type)) return true
  const ext = file.name.includes('.') ? file.name.split('.').pop()!.toLowerCase() : file.name.toLowerCase()
  return TEXT_EXTENSIONS.has(ext)
}

function readDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const r = new FileReader()
    r.onload = () => resolve(String(r.result))
    r.onerror = () => reject(r.error)
    r.readAsDataURL(file)
  })
}

/**
 * The text of a PDF as an attachment. The main process reads it (in a worker thread), so a big file does not
 * hold up the window. `note` is set when the text had to be cut; `error` when there is nothing to attach. `readKey`
 * lets cancelPdfRead stop it.
 */
export async function pdfToAttachment(file: File, readKey: string): Promise<{ attachment?: Attachment; error?: string; note?: string }> {
  const name = file.name || 'document.pdf'
  if (file.size > MAX_PDF_ATTACH_BYTES) return { error: `${name} is larger than 50 MB` }
  try {
    const r = await invoke('attachments:readPdf', name, new Uint8Array(await file.arrayBuffer()), readKey)
    if (!r.ok) return { error: r.error }
    return {
      attachment: { id: uid(), kind: 'text', name, mime: 'application/pdf', sizeBytes: file.size, text: r.text },
      note: r.cut ? `${name}: only the first ${r.pagesIncluded} of ${r.pageCount} pages fit in an attachment (400,000 characters of text), so the rest is left out.` : undefined
    }
  } catch (e) {
    return { error: `${name}: ${errorMessage(e)}` }
  }
}

export function cancelPdfRead(readKey: string): void {
  void invoke('attachments:cancelPdf', readKey).catch(() => undefined)
}

/** Turns dropped/pasted/picked files (other than PDFs, see pdfToAttachment) into attachments; returns errors for unsupported ones. */
export async function filesToAttachments(files: File[]): Promise<{ attachments: Attachment[]; errors: string[] }> {
  const attachments: Attachment[] = []
  const errors: string[] = []
  for (const file of files) {
    const name = file.name || 'pasted-image.png'
    if (file.type.startsWith('image/')) {
      if (file.size > MAX_IMAGE_BYTES) {
        errors.push(`${name} is larger than 20 MB`)
        continue
      }
      attachments.push({ id: uid(), kind: 'image', name, mime: file.type, sizeBytes: file.size, dataUrl: await readDataUrl(file) })
    } else if (isTextFile(file)) {
      if (file.size > MAX_TEXT_BYTES) {
        errors.push(`${name} is larger than 2 MB`)
        continue
      }
      attachments.push({ id: uid(), kind: 'text', name, mime: file.type || 'text/plain', sizeBytes: file.size, text: await file.text() })
    } else {
      errors.push(`${name}: only images, PDFs and text/code files can be attached`)
    }
  }
  return { attachments, errors }
}
