// PDFs the owner is attaching to the next message, kept per project: they stay while the owner moves between screens
// and after a start that failed, and are cleared only once a message carrying them started.
import { create } from 'zustand'
import { toast } from 'sonner'
import { MAX_PDF_ATTACH_BYTES, MAX_PDFS_PER_MESSAGE } from '@shared/pdf'
import { errorMessage, invoke } from '../lib/api'

export interface Chip {
  /** Local key, there from the moment the owner picks the file. */
  key: string
  name: string
  pages?: number
  /** The attachment's id once it is read; a chip without one is still being read. */
  id?: string
  busy: boolean
}

interface AttachmentsState {
  byProject: Record<string, Chip[]>
  /** Attaches files from a drop or a picker (their path when the window has it, otherwise uploaded in chunks). */
  attachFiles: (projectPath: string, files: File[]) => void
  /** Attaches PDFs by their path on this computer. */
  attachPaths: (projectPath: string, paths: string[]) => void
  remove: (projectPath: string, key: string) => Promise<void>
  /** Drops the chips that were sent (by attachment id). */
  clearSent: (projectPath: string, ids: string[]) => void
}

const isPdfName = (name: string): boolean => /\.pdf$/i.test(name)
const NONE: Chip[] = []
export const chipsOf = (s: AttachmentsState, projectPath: string | undefined): Chip[] => (projectPath ? (s.byProject[projectPath] ?? NONE) : NONE)

/** One at a time: a phone or window may have only one upload under way. */
let queue: Promise<void> = Promise.resolve()

function toBase64(bytes: Uint8Array): string {
  let text = ''
  for (let i = 0; i < bytes.length; i += 0x8000) text += String.fromCharCode(...bytes.subarray(i, i + 0x8000))
  return btoa(text)
}

async function upload(projectPath: string, file: File): Promise<{ id: string; name: string; pages: number }> {
  const { uploadId, chunkBytes } = await invoke('pdf:uploadBegin', projectPath, file.name, file.size)
  for (let at = 0, index = 0; at < file.size; at += chunkBytes, index++) {
    const bytes = new Uint8Array(await file.slice(at, at + chunkBytes).arrayBuffer())
    await invoke('pdf:uploadChunk', uploadId, index, toBase64(bytes))
  }
  return invoke('pdf:uploadEnd', uploadId)
}

export const useAttachments = create<AttachmentsState>((set, get) => {
  const update = (project: string, fn: (chips: Chip[]) => Chip[]): void =>
    set((s) => ({ byProject: { ...s.byProject, [project]: fn(s.byProject[project] ?? []) } }))

  /** Adds busy chips for what fits, then reads each one in turn. */
  const enqueue = (project: string, items: { name: string; read: () => Promise<{ id: string; name: string; pages: number }> }[]): void => {
    const room = MAX_PDFS_PER_MESSAGE - (get().byProject[project] ?? []).length
    if (items.length > room) toast.error(`At most ${MAX_PDFS_PER_MESSAGE} PDFs can go with one message.`)
    for (const item of items.slice(0, Math.max(0, room))) {
      const key = crypto.randomUUID()
      update(project, (c) => [...c, { key, name: item.name, busy: true }])
      queue = queue.then(async () => {
        try {
          const a = await item.read()
          update(project, (c) => c.map((x) => (x.key === key ? { key, id: a.id, name: a.name, pages: a.pages, busy: false } : x)))
        } catch (e) {
          update(project, (c) => c.filter((x) => x.key !== key))
          toast.error(errorMessage(e))
        }
      })
    }
  }

  return {
    byProject: {},
    attachFiles: (project, files) => {
      const pdfs = files.filter((f) => isPdfName(f.name) || f.type === 'application/pdf')
      if (pdfs.length < files.length) toast.error('Only PDFs can be attached.')
      const big = pdfs.filter((f) => f.size > MAX_PDF_ATTACH_BYTES)
      if (big.length) toast.error(`${big.length === 1 ? `“${big[0].name}” is` : 'Some PDFs are'} bigger than 50 MB, so ${big.length === 1 ? 'it' : 'they'} can’t be attached.`)
      enqueue(
        project,
        pdfs
          .filter((f) => f.size <= MAX_PDF_ATTACH_BYTES)
          .map((f) => ({
            name: f.name,
            read: async () => {
              // The computer's window knows a dropped file's path; a file without one is sent in chunks.
              const path = window.rulz.pathForFile(f)
              return path ? invoke('pdf:attachPath', project, path) : upload(project, f)
            }
          }))
      )
    },
    attachPaths: (project, paths) => {
      const pdfs = paths.filter(isPdfName)
      if (pdfs.length < paths.length) toast.error('Only PDFs can be attached.')
      enqueue(
        project,
        pdfs.map((p) => ({ name: p.split(/[\\/]/).pop() ?? p, read: () => invoke('pdf:attachPath', project, p) }))
      )
    },
    remove: async (project, key) => {
      const chip = (get().byProject[project] ?? []).find((c) => c.key === key)
      if (!chip || chip.busy) return
      try {
        if (chip.id) await invoke('pdf:remove', project, chip.id)
        update(project, (c) => c.filter((x) => x.key !== key))
      } catch (e) {
        toast.error(errorMessage(e))
      }
    },
    clearSent: (project, ids) => update(project, (c) => c.filter((x) => !x.id || !ids.includes(x.id)))
  }
})
