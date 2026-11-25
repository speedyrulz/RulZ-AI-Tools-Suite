import { describe, expect, it } from 'vitest'
import { MAX_PDF_ATTACH_BYTES, MAX_PDFS_PER_MESSAGE, type PdfAttachment } from '../src/shared/pdf'
import { base64Of, checkPicked, createPdfStore, planChunks, uploadError, type PdfDeps, type PdfSource } from '../src/lib/pdfUpload'

const CHUNK = 524288
const tick = (n = 5): Promise<void> => new Promise((r) => (n <= 0 ? r() : setTimeout(() => void tick(n - 1).then(r), 0)))
const bytesOf = (n: number): Uint8Array => Uint8Array.from({ length: n }, (_, i) => (i * 7 + 3) & 255)
const source = (name: string, size: number): PdfSource => {
  const data = bytesOf(size)
  return { name, size, type: 'application/pdf', slice: async (s, e) => data.slice(s, e) }
}

function fake(opts: { chunkBytes?: number; failChunk?: (index: number, call: number) => Error | null; beginError?: Error; hold?: Promise<void> } = {}) {
  const log: string[] = []
  const sent: { id: string; index: number; base64: string }[] = []
  let n = 0
  let chunks = 0
  const deps: PdfDeps = {
    begin: async (project, name, bytes) => {
      if (opts.beginError) throw opts.beginError
      log.push(`begin ${project} ${name} ${bytes}`)
      return { uploadId: `u${++n}`, chunkBytes: opts.chunkBytes ?? CHUNK }
    },
    chunk: async (id, index, base64) => {
      if (id === 'u1') await opts.hold
      const err = opts.failChunk?.(index, chunks++)
      if (err) throw err
      sent.push({ id, index, base64 })
    },
    end: async (id): Promise<PdfAttachment> => {
      log.push(`end ${id}`)
      return { id: String(id).padEnd(16, '0'), name: 'x.pdf', pages: 3, bytes: 1 }
    },
    remove: async (project, id) => void log.push(`remove ${project} ${id}`)
  }
  return { deps, log, sent }
}

describe('chunks and base64', () => {
  it('plans chunks in order with a short last one', () => {
    expect(planChunks(CHUNK * 2 + 10, CHUNK)).toEqual([
      { index: 0, start: 0, end: CHUNK },
      { index: 1, start: CHUNK, end: CHUNK * 2 },
      { index: 2, start: CHUNK * 2, end: CHUNK * 2 + 10 }
    ])
    expect(planChunks(CHUNK, CHUNK)).toHaveLength(1)
    expect(planChunks(0, CHUNK)).toEqual([])
  })
  it('encodes exactly what Node does, for every length near a block, and for a big chunk', () => {
    for (const n of [0, 1, 2, 3, 4, 12287, 12288, 12289, 24577]) expect(base64Of(bytesOf(n))).toBe(Buffer.from(bytesOf(n)).toString('base64'))
    const big = bytesOf(CHUNK)
    expect(base64Of(big)).toBe(Buffer.from(big).toString('base64'))
  })
})

describe('limits', () => {
  it('turns down big, empty, non-PDF files and a count over the limit with plain sentences', () => {
    const r = checkPicked(
      [
        { name: 'a.pdf', size: 10 },
        { name: 'big.pdf', size: MAX_PDF_ATTACH_BYTES + 1 },
        { name: 'e.pdf', size: 0 },
        { name: 'notes.txt', size: 5, type: 'text/plain' },
        { name: 'b.PDF', size: 10 }
      ],
      0
    )
    expect(r.accepted).toEqual([0, 4])
    expect(r.problems).toEqual(['big.pdf is bigger than 50 MB.', 'e.pdf is empty.', 'notes.txt is not a PDF.'])
  })
  it('counts the PDFs already waiting', () => {
    const files = Array.from({ length: 3 }, (_, i) => ({ name: `f${i}.pdf`, size: 1 }))
    const r = checkPicked(files, MAX_PDFS_PER_MESSAGE - 1)
    expect(r.accepted).toEqual([0])
    expect(r.problems[0]).toMatch(/at most 10 PDFs/)
  })
})

describe('the upload queue', () => {
  const scope = { computer: 'pc1', project: 'C:/p1' }

  it('uploads one file after another, chunks in order, and ends each', async () => {
    const f = fake({ chunkBytes: 100 })
    const s = createPdfStore(f.deps)
    s.getState().setScope(scope)
    expect(s.getState().add([source('a.pdf', 250), source('b.pdf', 100)])).toEqual([])
    await tick()
    expect(f.log).toEqual(['begin C:/p1 a.pdf 250', 'end u1', 'begin C:/p1 b.pdf 100', 'end u2'])
    expect(f.sent.map((c) => `${c.id}:${c.index}`)).toEqual(['u1:0', 'u1:1', 'u1:2', 'u2:0'])
    const data = bytesOf(250)
    expect(f.sent[2].base64).toBe(Buffer.from(data.slice(200, 250)).toString('base64'))
    expect(s.getState().chips.map((c) => c.state)).toEqual(['ready', 'ready'])
    expect(s.getState().ids()).toHaveLength(2)
    expect(s.getState().blocked()).toBe(false)
  })

  it('does not hand out ids while an upload runs', async () => {
    const f = fake()
    const s = createPdfStore(f.deps)
    s.getState().setScope(scope)
    s.getState().add([source('a.pdf', 10)])
    expect(s.getState().busy()).toBe(true)
    expect(s.getState().ids()).toEqual([])
    await tick()
    expect(s.getState().ids()).toHaveLength(1)
  })

  it('stops sending chunks when cancelled, and the next file still goes', async () => {
    let release = (): void => undefined
    const f = fake({ chunkBytes: 10, hold: new Promise<void>((r) => (release = r)) })
    const s = createPdfStore(f.deps)
    s.getState().setScope(scope)
    s.getState().add([source('a.pdf', 100), source('b.pdf', 10)])
    await tick(1)
    const first = s.getState().chips[0].key
    await s.getState().remove(first)
    release()
    await tick()
    expect(s.getState().chips.map((c) => c.name)).toEqual(['b.pdf'])
    expect(f.sent.filter((c) => c.id === 'u1').length).toBeLessThanOrEqual(1)
    expect(s.getState().chips[0].state).toBe('ready')
  })

  it('asks the computer to remove an uploaded PDF, and keeps the chip with a reason if that fails', async () => {
    const f = fake()
    const s = createPdfStore(f.deps)
    s.getState().setScope(scope)
    s.getState().add([source('a.pdf', 10)])
    await tick()
    const chip = s.getState().chips[0]
    await s.getState().remove(chip.key)
    expect(f.log.at(-1)).toBe(`remove C:/p1 ${chip.attachment!.id}`)
    expect(s.getState().chips).toEqual([])

    const bad = { ...f.deps, remove: async () => Promise.reject(new Error('Not connected to pc1.')) }
    const s2 = createPdfStore(bad)
    s2.getState().setScope(scope)
    s2.getState().add([source('a.pdf', 10)])
    await tick()
    await s2.getState().remove(s2.getState().chips[0].key)
    expect(s2.getState().chips[0].note).toMatch(/Could not remove it/)
  })

  it('fails the file with a sentence on a lost connection, blocks Send, and retries from the start', async () => {
    let fail = true
    const f = fake({ chunkBytes: 10, failChunk: (i) => (fail && i === 1 ? new Error('Lost the connection to pc1.') : null) })
    const s = createPdfStore(f.deps)
    s.getState().setScope(scope)
    s.getState().add([source('a.pdf', 30)])
    await tick()
    const chip = s.getState().chips[0]
    expect(chip.state).toBe('error')
    expect(chip.error).toMatch(/connection to your computer dropped/)
    expect(s.getState().blocked()).toBe(true)
    fail = false
    s.getState().retry(chip.key)
    await tick()
    expect(s.getState().chips[0].state).toBe('ready')
    expect(f.sent.filter((c) => c.id === 'u2').map((c) => c.index)).toEqual([0, 1, 2])
  })

  it('never sends chips to another project or computer', async () => {
    let release = (): void => undefined
    const f = fake({ chunkBytes: 10, hold: new Promise<void>((r) => (release = r)) })
    const s = createPdfStore(f.deps)
    s.getState().setScope(scope)
    s.getState().add([source('a.pdf', 100), source('b.pdf', 10)])
    await tick(1)
    s.getState().setScope({ computer: 'pc1', project: 'C:/p2' })
    release()
    await tick()
    expect(s.getState().chips).toEqual([])
    expect(f.log.some((l) => l.includes('b.pdf'))).toBe(false)
    expect(f.sent.filter((c) => c.id === 'u1').length).toBeLessThanOrEqual(1)
    // The same scope again keeps the chips.
    s.getState().add([source('c.pdf', 10)])
    s.getState().setScope({ computer: 'pc1', project: 'C:/p2' })
    expect(s.getState().chips).toHaveLength(1)
    await tick()
    expect(f.log).toContain('begin C:/p2 c.pdf 10')
  })

  it('hides the button for an older computer after "Unknown request", and asks again for another computer', async () => {
    const f = fake({ beginError: new Error('Unknown request "pdf:uploadBegin".') })
    const s = createPdfStore(f.deps)
    s.getState().setScope(scope)
    s.getState().add([source('a.pdf', 10)])
    await tick()
    expect(s.getState().older).toBe(true)
    expect(s.getState().chips).toEqual([])
    expect(s.getState().add([source('b.pdf', 10)])).toEqual([])
    expect(s.getState().chips).toEqual([])
    s.getState().setScope({ computer: 'pc2', project: 'C:/p1' })
    expect(s.getState().older).toBe(false)
    s.getState().setScope({ computer: 'pc2', project: 'C:/p9' })
    s.getState().markOlder()
    s.getState().setScope({ computer: 'pc2', project: 'C:/p8' })
    expect(s.getState().older).toBe(true)
  })

  it('says what the computer says, or a plain sentence when it is still finishing an earlier upload', () => {
    expect(uploadError(new Error('Too many uploads are in progress.')).text).toMatch(/still finishing an earlier upload/)
    expect(uploadError(new Error('PDF must be between 1 byte and 50 MB.')).text).toBe('PDF must be between 1 byte and 50 MB.')
  })

  it('lets a message with no text go with PDFs: ids come from ready chips only', async () => {
    const f = fake()
    const s = createPdfStore(f.deps)
    s.getState().setScope(scope)
    s.getState().add([source('a.pdf', 10)])
    await tick()
    expect(s.getState().ids()).toHaveLength(1)
    s.getState().clearSent(s.getState().ids())
    expect(s.getState().chips).toEqual([])
  })
})
