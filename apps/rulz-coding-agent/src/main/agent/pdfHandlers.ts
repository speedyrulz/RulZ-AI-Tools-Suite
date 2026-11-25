// The handlers of the pdf:* channels and the checks of task:start's attachments. The phone is not trusted: every
// argument is checked here, and every project path must be the open project.
import type { PdfAttachment } from '@shared/pdf'
import type { AttachmentStore } from './attachments'
import type { ProjectSession } from './session'

export interface PdfHandlerDeps {
  store: Pick<AttachmentStore, 'attachPath' | 'begin' | 'chunk' | 'end'>
  /** The open project's session for this path; throws "That project is not open." for any other. */
  session: (projectPath: string) => ProjectSession
  /** The caller of the request: 'window', or a phone connection's id. */
  caller: () => string
}

const text = (v: unknown, what: string, max: number): string => {
  if (typeof v !== 'string' || !v || v.length > max || v.includes('\0')) throw new Error(`${what} is missing or not valid.`)
  return v
}

export function pdfHandlers(deps: PdfHandlerDeps) {
  const open = (projectPath: unknown): ProjectSession => deps.session(text(projectPath, 'The project path', 32767))
  return {
    attachPath: async (projectPath: unknown, path: unknown): Promise<PdfAttachment> => {
      const s = open(projectPath)
      return deps.store.attachPath(s.root, text(path, 'The PDF path', 32767))
    },
    uploadBegin: async (projectPath: unknown, name: unknown, bytes: unknown): Promise<{ uploadId: string; chunkBytes: number }> => {
      const owner = deps.caller()
      const s = open(projectPath)
      if (typeof name !== 'string' || name.length > 1000) throw new Error('The file name is not valid.')
      if (typeof bytes !== 'number') throw new Error('The size must be a number.')
      return deps.store.begin(s.root, name, bytes, owner)
    },
    uploadChunk: async (uploadId: unknown, index: unknown, base64: unknown): Promise<void> => {
      const owner = deps.caller()
      if (typeof uploadId !== 'string' || typeof index !== 'number' || !Number.isSafeInteger(index) || index < 0 || typeof base64 !== 'string') {
        throw new Error('The upload chunk is not valid.')
      }
      await deps.store.chunk(uploadId, index, base64, owner)
    },
    uploadEnd: async (uploadId: unknown): Promise<PdfAttachment> => {
      const owner = deps.caller()
      if (typeof uploadId !== 'string') throw new Error('The upload is not valid.')
      return deps.store.end(uploadId, owner)
    },
    remove: async (projectPath: unknown, id: unknown): Promise<void> => {
      await open(projectPath).removePdf(id)
    }
  }
}

/** The task:start arguments, checked: the text a string, the attachments a valid list of this project's ids. */
export async function startArguments(s: ProjectSession, text: unknown, attachments: unknown): Promise<{ text: string; attachments: PdfAttachment[] }> {
  if (typeof text !== 'string') throw new Error('The message must be text.')
  const pdfs = await s.attachmentsFor(attachments)
  return { text, attachments: pdfs }
}
