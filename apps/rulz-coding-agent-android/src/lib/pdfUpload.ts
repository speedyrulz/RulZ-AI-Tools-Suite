// Attaching PDFs from the phone: the upload queue and its rules, as plain code so it can be tested
// without a phone or a computer. The computer's side is src/main/agent/attachments.ts in the desktop app.
import { create, type StoreApi, type UseBoundStore } from 'zustand'
import { MAX_PDF_ATTACH_BYTES, MAX_PDFS_PER_MESSAGE, type PdfAttachment } from '@shared/pdf'
import { isUnknownChannel } from './modeLogic'

/** Something to upload: a File from the system picker, read slice by slice. */
export interface PdfSource {
  name: string
  size: number
  type?: string
  slice: (start: number, end: number) => Promise<Uint8Array>
}

export interface PdfDeps {
  begin: (projectPath: string, name: string, bytes: number) => Promise<{ uploadId: string; chunkBytes: number }>
  chunk: (uploadId: string, index: number, base64: string) => Promise<void>
  end: (uploadId: string) => Promise<PdfAttachment>
  remove: (projectPath: string, id: string) => Promise<void>
}

export type ChipState = 'queued' | 'uploading' | 'ready' | 'error'

export interface PdfChip {
  key: string
  name: string
  bytes: number
  /** Bytes sent so far (while uploading). */
  sent: number
  state: ChipState
  error?: string
  /** A reason a ready chip could not be removed on the computer. */
  note?: string
  attachment?: PdfAttachment
}

const mb = (n: number): string => `${Math.round(n / 1024 / 1024)} MB`

/** The chunks of a file: in order, the last one short. */
export function planChunks(size: number, chunkBytes: number): { index: number; start: number; end: number }[] {
  const out: { index: number; start: number; end: number }[] = []
  if (!(size > 0) || !(chunkBytes > 0)) return out
  for (let start = 0, index = 0; start < size; start += chunkBytes, index++) out.push({ index, start, end: Math.min(size, start + chunkBytes) })
  return out
}

/** Base64 of bytes, without spreading a big array into one call (that overflows the stack). Blocks are multiples of 3 bytes, so they join without padding. */
export function base64Of(bytes: Uint8Array): string {
  const BLOCK = 3 * 4096
  let out = ''
  for (let i = 0; i < bytes.length; i += BLOCK) {
    const part = bytes.subarray(i, i + BLOCK)
    let binary = ''
    for (let j = 0; j < part.length; j++) binary += String.fromCharCode(part[j])
    out += btoa(binary)
  }
  return out
}

const isPdf = (f: Pick<PdfSource, 'name' | 'type'>): boolean => f.type === 'application/pdf' || /\.pdf$/i.test(f.name)

/**
 * Which of the picked files can be attached: size, type and the count per message are checked before
 * any upload starts. `have` is how many PDFs the message already has.
 */
export function checkPicked(files: Pick<PdfSource, 'name' | 'size' | 'type'>[], have: number): { accepted: number[]; problems: string[] } {
  const accepted: number[] = []
  const problems: string[] = []
  let count = have
  files.forEach((f, i) => {
    if (!isPdf(f)) problems.push(`${f.name} is not a PDF.`)
    else if (!(f.size > 0)) problems.push(`${f.name} is empty.`)
    else if (f.size > MAX_PDF_ATTACH_BYTES) problems.push(`${f.name} is bigger than ${mb(MAX_PDF_ATTACH_BYTES)}.`)
    else if (count >= MAX_PDFS_PER_MESSAGE) problems.push(`A message can have at most ${MAX_PDFS_PER_MESSAGE} PDFs, so ${f.name} was left out.`)
    else {
      accepted.push(i)
      count++
    }
  })
  return { accepted, problems }
}

/** A sentence for a failed upload; `again` says whether trying again can help. */
export function uploadError(e: unknown): { text: string; again: boolean } {
  const m = e instanceof Error ? e.message : String(e)
  if (/connection|Not connected|took too long/i.test(m)) return { text: 'The connection to your computer dropped, so this PDF did not go through.', again: true }
  if (/Too many uploads/i.test(m)) return { text: 'Your computer is still finishing an earlier upload. Try again in a couple of minutes.', again: true }
  return { text: m || 'This PDF could not be sent.', again: true }
}

/** Where a task's attachments may be sent: the computer and project they were added in. */
export interface PdfScope {
  computer: string
  project: string
}

export interface PdfState {
  scope: PdfScope | null
  chips: PdfChip[]
  /** The computer has no PDF channels: no button. Remembered for that computer in memory. */
  older: boolean
  setScope: (scope: PdfScope | null) => void
  markOlder: () => void
  /** Adds picked files; returns the sentences to show for the ones left out. */
  add: (files: PdfSource[]) => string[]
  /** Stops an upload, or removes a chip (asking the computer to drop an uploaded one). */
  remove: (key: string) => Promise<void>
  retry: (key: string) => void
  /** Attachment ids of uploaded chips; empty while uploads are running (Send waits). */
  ids: () => string[]
  busy: () => boolean
  /** Send waits: an upload is running, or one failed and has not been retried or removed. */
  blocked: () => boolean
  /** After the task started: the chips are done with. */
  clearSent: (ids: string[]) => void
}

export type PdfStore = UseBoundStore<StoreApi<PdfState>>

export function createPdfStore(deps: PdfDeps): PdfStore {
  /** Bumped when the scope changes: whatever an older upload still does is dropped. */
  let epoch = 0
  let seq = 0
  /** Chips whose upload was stopped by the owner: no more chunks go out for them. */
  const stopped = new Set<string>()
  const sources = new Map<string, PdfSource>()
  let pumping = false

  return create<PdfState>((set, get) => {
    const patch = (key: string, p: Partial<PdfChip>): void => set((s) => ({ chips: s.chips.map((c) => (c.key === key ? { ...c, ...p } : c)) }))
    const has = (key: string): boolean => get().chips.some((c) => c.key === key)

    const upload = async (chip: PdfChip, mine: number): Promise<void> => {
      const source = sources.get(chip.key)
      const scope = get().scope
      if (!source || !scope) return
      const alive = (): boolean => mine === epoch && !stopped.has(chip.key) && has(chip.key)
      patch(chip.key, { state: 'uploading', sent: 0, error: undefined })
      try {
        const { uploadId, chunkBytes } = await deps.begin(scope.project, chip.name, chip.bytes)
        for (const c of planChunks(chip.bytes, chunkBytes)) {
          if (!alive()) return
          const bytes = await source.slice(c.start, c.end)
          if (!alive()) return
          await deps.chunk(uploadId, c.index, base64Of(bytes))
          if (!alive()) return
          patch(chip.key, { sent: c.end })
        }
        if (!alive()) return
        const attachment = await deps.end(uploadId)
        if (!alive()) return
        sources.delete(chip.key)
        patch(chip.key, { state: 'ready', sent: chip.bytes, attachment })
      } catch (e) {
        if (!alive()) return
        if (isUnknownChannel(e)) {
          // A computer older than this app: no PDFs from the phone. Hide the button for it.
          set((s) => ({ older: true, chips: s.chips.filter((c) => c.key !== chip.key) }))
          sources.delete(chip.key)
          return
        }
        patch(chip.key, { state: 'error', error: uploadError(e).text })
      }
    }

    /** One file after another, never two at once. */
    const pump = async (): Promise<void> => {
      if (pumping) return
      pumping = true
      try {
        for (;;) {
          const mine = epoch
          const next = get().chips.find((c) => c.state === 'queued')
          if (!next) return
          await upload(next, mine)
        }
      } finally {
        pumping = false
      }
    }

    return {
      scope: null,
      chips: [],
      older: false,

      setScope: (scope) => {
        const cur = get().scope
        if (cur && scope && cur.computer === scope.computer && cur.project === scope.project) return
        epoch++
        stopped.clear()
        sources.clear()
        // Another computer may know PDFs; the chips never go to another project or computer.
        set({ scope, chips: [], older: cur?.computer === scope?.computer ? get().older : false })
      },

      markOlder: () => set({ older: true }),

      add: (files) => {
        const scope = get().scope
        if (!scope || get().older) return []
        // Failed ones count too: they may be tried again.
        const have = get().chips.length
        const { accepted, problems } = checkPicked(files, have)
        const added: PdfChip[] = accepted.map((i) => {
          const key = `pdf${++seq}`
          sources.set(key, files[i])
          return { key, name: files[i].name, bytes: files[i].size, sent: 0, state: 'queued' }
        })
        if (added.length) {
          set((s) => ({ chips: [...s.chips, ...added] }))
          void pump()
        }
        return problems
      },

      remove: async (key) => {
        const chip = get().chips.find((c) => c.key === key)
        const scope = get().scope
        if (!chip || !scope) return
        if (chip.state === 'ready' && chip.attachment) {
          try {
            await deps.remove(scope.project, chip.attachment.id)
          } catch (e) {
            const m = e instanceof Error ? e.message : String(e)
            if (get().scope === scope) patch(key, { note: `Could not remove it on your computer: ${m}` })
            return
          }
        }
        // An upload in progress just stops sending; the computer drops it when the next one begins, or after its timeout.
        stopped.add(key)
        sources.delete(key)
        set((s) => ({ chips: s.chips.filter((c) => c.key !== key) }))
      },

      retry: (key) => {
        const chip = get().chips.find((c) => c.key === key)
        if (!chip || chip.state !== 'error' || !sources.has(key)) return
        stopped.delete(key)
        patch(key, { state: 'queued', error: undefined, sent: 0 })
        void pump()
      },

      ids: () => (get().busy() ? [] : get().chips.flatMap((c) => (c.state === 'ready' && c.attachment ? [c.attachment.id] : []))),

      busy: () => get().chips.some((c) => c.state === 'queued' || c.state === 'uploading'),

      blocked: () => get().chips.some((c) => c.state !== 'ready'),

      clearSent: (ids) => {
        // Only what went along: a PDF picked while the task was starting stays for the next message.
        set((s) => ({ chips: s.chips.filter((c) => !c.attachment || !ids.includes(c.attachment.id)) }))
      }
    }
  })
}
