import type { Attachment } from '@shared/types'
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

/** Turns dropped/pasted/picked files into attachments; returns errors for unsupported ones. */
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
      errors.push(`${name}: only images and text/code files can be attached`)
    }
  }
  return { attachments, errors }
}
