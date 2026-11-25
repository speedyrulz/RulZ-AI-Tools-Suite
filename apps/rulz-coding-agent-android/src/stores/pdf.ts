import { useEffect } from 'react'
import { invoke } from '../lib/api'
import { isUnknownChannel } from '../lib/modeLogic'
import { createPdfStore, type PdfSource } from '../lib/pdfUpload'
import { useLink } from './link'
import { useWorkspace } from './workspace'

export const usePdf = createPdfStore({
  begin: (project, name, bytes) => invoke('pdf:uploadBegin', project, name, bytes),
  chunk: (id, index, base64) => invoke('pdf:uploadChunk', id, index, base64),
  end: (id) => invoke('pdf:uploadEnd', id),
  remove: (project, id) => invoke('pdf:remove', project, id)
})

/** A picked File, read one slice at a time (never the whole file in memory). */
export const sourceOf = (f: File): PdfSource => ({
  name: f.name,
  size: f.size,
  type: f.type,
  slice: async (start, end) => new Uint8Array(await f.slice(start, end).arrayBuffer())
})

/** Computers that answered the PDF channels (in memory): they are not asked again. */
const known = new Set<string>()

/**
 * Keeps the chips tied to the computer and project they were added in, and finds out before the first
 * try whether the computer knows PDFs: `pdf:remove` with an id that is nothing is harmless, and an older
 * computer answers it with "Unknown request".
 */
export function usePdfScope(): void {
  const computer = useLink((s) => s.currentId)
  const project = useWorkspace((s) => s.project?.path ?? null)
  const linked = useLink((s) => s.state.kind === 'connected')
  useEffect(() => {
    usePdf.getState().setScope(computer && project ? { computer, project } : null)
  }, [computer, project])
  useEffect(() => {
    if (!computer || !project || !linked || known.has(computer)) return
    invoke('pdf:remove', project, '0000000000000000').then(
      () => {
        known.add(computer)
        // Its app may have been updated since it refused the channel.
        if (useLink.getState().currentId === computer) usePdf.setState({ older: false })
      },
      (e: unknown) => {
        if (isUnknownChannel(e)) {
          if (useLink.getState().currentId === computer) usePdf.getState().markOlder()
        } else if (!/connection|Not connected|took too long/i.test(e instanceof Error ? e.message : String(e))) known.add(computer)
      }
    )
  }, [computer, project, linked])
}

