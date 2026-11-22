// Stage 4 in the file store: novel:record, recordsInfo, deleteRecords and exportRecords on temp project folders.
import { existsSync, mkdirSync, mkdtempSync, promises as fsp, readdirSync, readFileSync, renameSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { NovelChange, RecordAlt, RecordLine, RecordsExportOptions } from '@shared/novel'
import { NovelStore } from '../src/main/novel/store'

afterEach(() => {
  vi.restoreAllMocks()
})

const sleep = (ms: number): Promise<void> => new Promise((r) => setTimeout(r, ms))
const NOW = new Date(2026, 10, 22, 9, 30, 15)
const DAY = '2026-11-22.jsonl'
const GEN = 'The rain came down over the harbour in long grey sheets.'
const AT = '2026-11-22T08:30:15.000Z'

function setup(trashDelay = 0) {
  const base = mkdtempSync(join(tmpdir(), 'rulz-novel-rec-'))
  const events: NovelChange[] = []
  const recycled: string[] = []
  const store = new NovelStore({
    rootSetting: () => '',
    documents: () => join(base, 'Documents'),
    trashItem: async (p) => {
      if (trashDelay) await sleep(trashDelay)
      const to = join(base, `recycle-${recycled.length}`)
      renameSync(p, to)
      recycled.push(to)
    },
    emit: (c) => events.push(c),
    watch: false,
    now: () => NOW,
    appVersion: () => '9.9.9'
  })
  return { base, events, recycled, store }
}

async function project(title = 'Records', trashDelay = 0) {
  const s = setup(trashDelay)
  const st = await s.store.createProject(title)
  const recDir = join(st.folder, '.rulz', 'records')
  const file = join(recDir, DAY)
  const out = join(s.base, 'out')
  mkdirSync(out)
  return { ...s, st, folder: st.folder, recDir, file, out }
}

const hex = (n: number): string => n.toString(16).padStart(16, '0')
const alt = (index: number, generated: string, finish: RecordAlt['finish'] = 'stop'): RecordAlt => ({ index, seed: 100 + index, generated, finish })

function request(n: number, over: Record<string, unknown> = {}): RecordLine {
  return {
    format: 1,
    type: 'request',
    id: hex(n),
    at: AT,
    app: '0.23.0',
    action: 'continue',
    sceneId: 's-1',
    chapterId: 'c-1',
    model: { id: 'm1', name: 'Model One' },
    sampling: { temperature: 0.8 },
    adapters: [],
    messages: [
      { role: 'system', content: 'You write fiction.' },
      { role: 'user', content: `Continue ${n}.` }
    ],
    alts: [alt(0, GEN)],
    ...over
  } as unknown as RecordLine
}
const outcomeOf = (n: number, kind: string, over: Record<string, unknown> = {}): RecordLine =>
  ({ format: 1, type: 'outcome', id: hex(n), at: AT, kind, alt: kind === 'rejected' || kind === 'abandoned' ? null : 0, region: kind === 'rejected' || kind === 'abandoned' ? null : { start: 0, end: 50 }, replaced: '', ...over }) as unknown as RecordLine
const finalOf = (n: number, text: string | null, reason = 'idle'): RecordLine => ({ format: 1, type: 'final', id: hex(n), at: AT, text, reason }) as unknown as RecordLine
const viewedOf = (n: number, ...alts: number[]): RecordLine => ({ format: 1, type: 'viewed', id: hex(n), at: AT, alts }) as unknown as RecordLine

const jl = (lines: RecordLine[]): string => lines.map((l) => JSON.stringify(l)).join('\n') + '\n'
const read = (path: string): string => readFileSync(path, 'utf8')
const rows = (path: string): Record<string, any>[] =>
  read(path)
    .split('\n')
    .filter(Boolean)
    .map((l) => JSON.parse(l))

/** Writes the given lines / raw chunks as the day's records file. */
function seed(p: { recDir: string; file: string }, parts: (RecordLine | string | Buffer)[]): void {
  mkdirSync(p.recDir, { recursive: true })
  const bufs = parts.map((x) => (Buffer.isBuffer(x) ? x : Buffer.from(typeof x === 'string' ? x : `${JSON.stringify(x)}\n`, 'utf8')))
  writeFileSync(p.file, Buffer.concat(bufs))
}

async function exported(p: Awaited<ReturnType<typeof project>>, parts: (RecordLine | string | Buffer)[], opts: RecordsExportOptions = { editPairs: false }) {
  seed(p, parts)
  const r = await p.store.exportRecords(p.folder, p.out, opts)
  return {
    r,
    train: rows(join(r.outDir, 'train.jsonl')),
    pairs: rows(join(r.outDir, 'pairs.jsonl')),
    manifest: JSON.parse(read(join(r.outDir, 'manifest.json'))) as Record<string, any>
  }
}

describe('novel:record', () => {
  it('appends one line per call to today\'s file, built from the local date', async () => {
    const p = await project()
    expect(await p.store.record(p.folder, request(1))).toBe(true)
    expect(await p.store.record(p.folder, outcomeOf(1, 'accepted'))).toBe(true)
    expect(readdirSync(p.recDir)).toEqual([DAY])
    const lines = read(p.file).split('\n')
    expect(lines).toHaveLength(3)
    expect(lines[2]).toBe('')
    expect(JSON.parse(lines[0])).toMatchObject({ type: 'request', id: hex(1) })
    expect(JSON.parse(lines[1])).toMatchObject({ type: 'outcome', kind: 'accepted' })
  })

  it('is off when recording is false and on again when it is switched back; absent means on', async () => {
    const p = await project()
    expect(p.st.project.recording).toBeUndefined()
    await p.store.projectPatch(p.folder, { recording: false })
    expect(await p.store.record(p.folder, request(1))).toBe(false)
    expect(existsSync(p.recDir)).toBe(false)
    const back = await p.store.projectPatch(p.folder, { recording: true })
    expect(back.recording).toBe(true)
    expect(await p.store.record(p.folder, request(1))).toBe(true)
    expect(JSON.parse(read(join(p.folder, 'project.json'))).recording).toBe(true)
  })

  it('writes nothing for a blocked project', async () => {
    const p = await project()
    writeFileSync(join(p.folder, 'project.json'), '{ not json')
    const state = await p.store.openProject(p.folder)
    expect(state.blocked).toBeTruthy()
    expect(await p.store.record(p.folder, request(1))).toBe(false)
    expect(existsSync(p.recDir)).toBe(false)
    await expect(p.store.deleteRecords(p.folder)).rejects.toThrow(/read-only/)
  })

  it('writes nothing for a project that is not open or was closed', async () => {
    const p = await project()
    expect(await p.store.record(join(p.base, 'nowhere'), request(1))).toBe(false)
    expect(await p.store.record(undefined as unknown as string, request(1))).toBe(false)
    await p.store.closeProject(p.folder)
    expect(await p.store.record(p.folder, request(1))).toBe(false)
    expect(existsSync(p.recDir)).toBe(false)
  })

  it('puts a newline in front when the file does not end with one (torn last line)', async () => {
    const p = await project()
    seed(p, ['{"format":1,"type":"request","id":"aaaaaaaa'])
    expect(await p.store.record(p.folder, request(2))).toBe(true)
    const lines = read(p.file).split('\n')
    expect(lines[0]).toBe('{"format":1,"type":"request","id":"aaaaaaaa')
    expect(JSON.parse(lines[1]).id).toBe(hex(2))
    expect(lines[2]).toBe('')
    // a file that ends properly gets no blank line
    expect(await p.store.record(p.folder, request(3))).toBe(true)
    expect(read(p.file).split('\n')).toHaveLength(4)
  })

  it('rejects lines of the wrong shape without writing', async () => {
    const p = await project()
    for (const bad of [{ ...request(1), type: 'card' }, { ...request(1), id: 'XYZ' }, { ...request(1), alts: [] }, { ...outcomeOf(1, 'accepted'), kind: 'merged' }, null, 'text', 42]) {
      expect(await p.store.record(p.folder, bad as unknown as RecordLine)).toBe(false)
    }
    expect(existsSync(p.recDir)).toBe(false)
  })

  it('rejects a line over 8 MB', async () => {
    const p = await project()
    const big = request(1, { model: { id: 'm', name: 'x'.repeat(8 * 1024 * 1024 + 10) } })
    expect(await p.store.record(p.folder, big)).toBe(false)
    expect(existsSync(p.recDir)).toBe(false)
  })

  it('retries a lock error once and then writes; gives up and returns false when it never clears', async () => {
    const p = await project()
    const real = fsp.open.bind(fsp)
    let calls = 0
    vi.spyOn(fsp, 'open').mockImplementation(((path: any, ...rest: any[]) => {
      if (String(path).endsWith(DAY) && calls++ === 0) return Promise.reject(Object.assign(new Error('busy'), { code: 'EBUSY' }))
      return (real as any)(path, ...rest)
    }) as any)
    expect(await p.store.record(p.folder, request(1))).toBe(true)
    expect(calls).toBeGreaterThanOrEqual(2)
    expect(rows(p.file)).toHaveLength(1)
    vi.restoreAllMocks()

    const warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined)
    vi.spyOn(fsp, 'open').mockImplementation((() => Promise.reject(Object.assign(new Error('locked'), { code: 'EPERM' }))) as any)
    expect(await p.store.record(p.folder, request(2))).toBe(false)
    expect(warn).toHaveBeenCalled()
    vi.restoreAllMocks()
    expect(rows(p.file)).toHaveLength(1)
  }, 15000)

  it('does not throw for a non-lock I/O error', async () => {
    const p = await project()
    vi.spyOn(console, 'warn').mockImplementation(() => undefined)
    vi.spyOn(fsp, 'mkdir').mockRejectedValue(Object.assign(new Error('disk full'), { code: 'ENOSPC' }))
    expect(await p.store.record(p.folder, request(1))).toBe(false)
  })

  it('closing the project waits for an append that is in flight, and none starts after', async () => {
    const p = await project()
    const real = fsp.open.bind(fsp)
    // Delays the write itself, which is after the last "still open" check.
    let writing!: () => void
    const started = new Promise<void>((r) => (writing = r))
    vi.spyOn(fsp, 'open').mockImplementation((async (...a: any[]) => {
      if (String(a[0]).endsWith(DAY)) {
        writing()
        await sleep(150)
      }
      return (real as any)(...a)
    }) as any)
    const rec = p.store.record(p.folder, request(1))
    // the write has begun, past the last "still open" check
    await started
    await p.store.closeProject(p.folder)
    // close returned only after the append finished
    expect(existsSync(p.file)).toBe(true)
    expect(await rec).toBe(true)
    vi.restoreAllMocks()
    expect(await p.store.record(p.folder, request(2))).toBe(false)
    expect(rows(p.file)).toHaveLength(1)
  })

  it('delete project waits for appends and records nothing into a deleted project', async () => {
    const p = await project()
    const real = fsp.open.bind(fsp)
    // Delays the write itself, which is after the last "still open" check.
    let writing!: () => void
    const started = new Promise<void>((r) => (writing = r))
    vi.spyOn(fsp, 'open').mockImplementation((async (...a: any[]) => {
      if (String(a[0]).endsWith(DAY)) {
        writing()
        await sleep(150)
      }
      return (real as any)(...a)
    }) as any)
    const inFlight = p.store.record(p.folder, request(1))
    await started
    const late = p.store.record(p.folder, request(2))
    await p.store.deleteProject(p.folder)
    // Both were admitted before the delete: they finish into the project, which then goes to the Recycle Bin whole.
    const results = [await inFlight, await late]
    expect(results).toEqual([true, true])
    vi.restoreAllMocks()
    // the project went to the Recycle Bin whole; no .rulz\records appeared at the old place
    expect(existsSync(p.folder)).toBe(false)
    expect(p.recycled).toHaveLength(1)
    expect(rows(join(p.recycled[0], '.rulz', 'records', DAY)).map((r) => r.id)).toEqual([hex(1), hex(2)])
    expect(await p.store.record(p.folder, request(3))).toBe(false)
    expect(existsSync(p.folder)).toBe(false)
  })

  it('a failed delete project lets recording go on', async () => {
    const p = await project()
    const store2 = new NovelStore({
      rootSetting: () => '',
      documents: () => join(p.base, 'Documents'),
      trashItem: async () => {
        throw new Error('no recycle bin')
      },
      emit: () => undefined,
      watch: false,
      now: () => NOW
    })
    await store2.openProject(p.folder)
    await expect(store2.deleteProject(p.folder)).rejects.toThrow(/Nothing was deleted/)
    expect(await store2.record(p.folder, request(1))).toBe(true)
  })
})

describe('novel:recordsInfo and novel:deleteRecords', () => {
  it('counts the .jsonl files and their bytes; 0 / 0 when the folder is missing', async () => {
    const p = await project()
    expect(await p.store.recordsInfo(p.folder)).toEqual({ files: 0, bytes: 0 })
    await p.store.record(p.folder, request(1))
    writeFileSync(join(p.recDir, '2026-11-21.jsonl'), 'abc\n')
    writeFileSync(join(p.recDir, 'notes.txt'), 'ignored')
    const info = await p.store.recordsInfo(p.folder)
    expect(info.files).toBe(2)
    expect(info.bytes).toBe(Buffer.byteLength(read(p.file)) + 4)
    await expect(p.store.recordsInfo(join(p.base, 'nowhere'))).rejects.toThrow(/not open/)
  })

  it('moves the records folder to the Recycle Bin, and a missing folder is a no-op', async () => {
    const p = await project()
    await p.store.deleteRecords(p.folder)
    expect(p.recycled).toHaveLength(0)
    await p.store.record(p.folder, request(1))
    await p.store.deleteRecords(p.folder)
    expect(existsSync(p.recDir)).toBe(false)
    expect(p.recycled).toHaveLength(1)
    expect(existsSync(join(p.recycled[0], DAY))).toBe(true)
    expect(await p.store.recordsInfo(p.folder)).toEqual({ files: 0, bytes: 0 })
  })

  it('waits for an append in flight and holds new appends off until the move is done', async () => {
    const p = await project('Records', 120)
    await p.store.record(p.folder, request(1))
    const real = fsp.open.bind(fsp)
    vi.spyOn(fsp, 'open').mockImplementation((async (...a: any[]) => {
      if (String(a[0]).endsWith(DAY)) await sleep(100)
      return (real as any)(...a)
    }) as any)
    const inFlight = p.store.record(p.folder, request(2))
    await sleep(10)
    const del = p.store.deleteRecords(p.folder)
    await sleep(10)
    const after = p.store.record(p.folder, request(3))
    expect(await Promise.all([inFlight, del, after])).toEqual([true, undefined, true])
    vi.restoreAllMocks()
    // the old records (lines 1 and 2) went to the bin; the line written during the move is in a fresh folder
    expect(rows(join(p.recycled[0], DAY)).map((r) => r.id)).toEqual([hex(1), hex(2)])
    expect(rows(p.file).map((r) => r.id)).toEqual([hex(3)])
  })

  it('an append waiting for Delete records is not admitted when the project closes meanwhile', async () => {
    const p = await project('Records', 120)
    await p.store.record(p.folder, request(1))
    const del = p.store.deleteRecords(p.folder)
    await sleep(10)
    const waiting = p.store.record(p.folder, request(2))
    await sleep(10)
    const close = p.store.closeProject(p.folder)
    expect(await Promise.all([del, waiting, close])).toEqual([undefined, false, undefined])
    // the records went to the bin, and nothing re-created a records folder in the closed project
    expect(rows(join(p.recycled[0], DAY)).map((r) => r.id)).toEqual([hex(1)])
    expect(existsSync(p.recDir)).toBe(false)
  })
})

describe('novel:exportRecords', () => {
  const FINAL_TEXT = 'The rain came down over the harbour, grey and slow, without end.'

  it('exports accepted with a final, accepted without a final, and mixed with and without a final', async () => {
    const p = await project()
    const { r, train, pairs, manifest } = await exported(p, [
      request(1), outcomeOf(1, 'accepted'), finalOf(1, FINAL_TEXT),
      request(2), outcomeOf(2, 'accepted'),
      request(3), outcomeOf(3, 'mixed'), finalOf(3, `${GEN} and a bit more`),
      request(4), outcomeOf(4, 'mixed')
    ])
    expect(r.train).toBe(3)
    expect(r.pairs).toBe(0)
    expect(r.dropped).toEqual({ noFinal: 1 })
    const by = Object.fromEntries(train.map((t) => [t.meta.id, t]))
    expect(by[hex(1)].messages.at(-1)).toEqual({ role: 'assistant', content: FINAL_TEXT })
    expect(by[hex(1)].meta).toMatchObject({ final: true, kind: 'accepted', action: 'continue' })
    expect(by[hex(2)].messages.at(-1).content).toBe(GEN)
    expect(by[hex(2)].meta.final).toBe(false)
    expect(by[hex(3)].meta).toMatchObject({ final: true, kind: 'mixed' })
    expect(by[hex(4)]).toBeUndefined()
    expect(pairs).toEqual([])
    expect(manifest).toMatchObject({ format: 1, app: '9.9.9', project: 'Records', dateRange: { from: '2026-11-22', to: '2026-11-22' }, files: { 'train.jsonl': 3, 'pairs.jsonl': 0 }, options: { editPairs: false } })
    expect(manifest.actions.continue).toEqual({ train: 3, pairs: 0 })
    expect(manifest.dropped).toEqual({ noFinal: 1 })
    expect(typeof manifest.created).toBe('string')
    // no temp files are left in the export folder
    expect(readdirSync(r.outDir).sort()).toEqual(['manifest.json', 'pairs.jsonl', 'train.jsonl'])
  })

  it('drops rejected, abandoned, requests without an outcome, and reverted finals', async () => {
    const p = await project()
    const sel = 'The rain fell on the harbour.'
    const { r, train } = await exported(p, [
      request(1), outcomeOf(1, 'rejected'),
      request(2), outcomeOf(2, 'abandoned'),
      request(3),
      request(4, { action: 'rewrite' }), outcomeOf(4, 'accepted', { replaced: sel }), finalOf(4, sel),
      request(5), outcomeOf(5, 'accepted')
    ])
    expect(train.map((t) => t.meta.id)).toEqual([hex(5)])
    expect(r.dropped).toEqual({ rejected: 2, abandoned: 1, noOutcome: 1 })
  })

  it('filters pairs to viewed alternatives that finished with stop and differ; length alts are never in pairs', async () => {
    const p = await project()
    const alts = [alt(0, GEN), alt(1, 'The harbour lay silent while the rain kept coming down.'), alt(2, 'An alternative nobody opened, long enough to matter.'), alt(3, 'A cut-off alternative that was viewed but ran out of room', 'length'), alt(4, `  ${GEN}  `)]
    const { train, pairs, r } = await exported(p, [request(1, { alts }), viewedOf(1, 0, 1, 3, 4), outcomeOf(1, 'accepted')])
    expect(train).toHaveLength(1)
    expect(pairs).toHaveLength(1)
    expect(pairs[0]).toEqual({ prompt: (request(1) as any).messages, chosen: GEN, rejected: alts[1].generated, meta: { id: hex(1), source: 'alts' } })
    expect(r.pairs).toBe(1)
  })

  it('merges viewed lines of one id', async () => {
    const p = await project()
    const alts = [alt(0, GEN), alt(1, 'The harbour lay silent while the rain kept coming down.'), alt(2, 'The gulls wheeled over the water, crying out to nobody at all.')]
    const { pairs } = await exported(p, [request(1, { alts }), viewedOf(1, 0), viewedOf(1, 1, 2), outcomeOf(1, 'accepted')])
    expect(pairs.map((x) => x.rejected)).toEqual([alts[1].generated, alts[2].generated])
  })

  it('makes edit pairs only with the option, and skips an append-only edit', async () => {
    const p = await project()
    const lines = [
      request(1), outcomeOf(1, 'accepted'), finalOf(1, FINAL_TEXT),
      request(2), outcomeOf(2, 'accepted'), finalOf(2, `${GEN} She turned from the window.`),
      request(3), outcomeOf(3, 'accepted')
    ]
    const off = await exported(p, lines)
    expect(off.pairs).toEqual([])
    const on = await exported(p, lines, { editPairs: true })
    expect(on.pairs).toEqual([{ prompt: (request(1) as any).messages, chosen: FINAL_TEXT, rejected: GEN, meta: { id: hex(1), source: 'edit' } }])
    expect(on.manifest.options).toEqual({ editPairs: true })
    expect(on.manifest.actions.continue).toEqual({ train: 3, pairs: 1 })
    // two exports from the same clock never share a folder
    expect(on.r.outDir).not.toBe(off.r.outDir)
  })

  it('counts cards as dropped and never exports them', async () => {
    const p = await project()
    const { train, pairs, r, manifest } = await exported(p, [
      request(1, { action: 'summarizeScene' }), outcomeOf(1, 'accepted'),
      request(2, { action: 'outlineChapters' }), outcomeOf(2, 'edited', { text: 'x', alt: undefined, region: undefined }),
      request(3), outcomeOf(3, 'accepted')
    ])
    expect(r.dropped).toEqual({ cards: 2 })
    expect(train.map((t) => t.meta.id)).toEqual([hex(3)])
    expect(pairs).toEqual([])
    expect(manifest.actions.summarizeScene).toBeUndefined()
  })

  it('the first outcome wins and the last final wins, also across files', async () => {
    const p = await project()
    seed(p, [request(1), outcomeOf(1, 'accepted'), finalOf(1, 'An early final that is long enough to count.')])
    writeFileSync(join(p.recDir, '2026-11-23.jsonl'), jl([outcomeOf(1, 'rejected'), finalOf(1, FINAL_TEXT, 'left')]))
    const r = await p.store.exportRecords(p.folder, p.out, { editPairs: false })
    const train = rows(join(r.outDir, 'train.jsonl'))
    expect(train).toHaveLength(1)
    expect(train[0].messages.at(-1).content).toBe(FINAL_TEXT)
    expect(train[0].meta.kind).toBe('accepted')
    expect(r.dropped).toEqual({})
    expect(JSON.parse(read(join(r.outDir, 'manifest.json'))).dateRange).toEqual({ from: '2026-11-22', to: '2026-11-23' })
  })

  it('exports a repeated request id once', async () => {
    const p = await project()
    const { train, r } = await exported(p, [request(1), outcomeOf(1, 'accepted'), request(1)])
    expect(train).toHaveLength(1)
    expect(r.dropped).toEqual({ duplicate: 1 })
  })

  it('counts bad UTF-8, bad JSON, torn, wrongly shaped and over-long lines, and goes on', async () => {
    const p = await project()
    const { train, r, manifest } = await exported(p, [
      request(1), outcomeOf(1, 'accepted'),
      Buffer.concat([Buffer.from('{"a":"'), Buffer.from([0xff, 0xfe]), Buffer.from('"}\n')]),
      'not json at all\n',
      '{"format":1,"type":"outcome","id":"zzzz","at":"x","kind":"accepted"}\n',
      '\n',
      '   \n',
      request(2), outcomeOf(2, 'accepted'),
      '{"format":1,"type":"request","id":"aaaaaaaaaaaaaaaa","at":"2026'
    ])
    expect(train.map((t) => t.meta.id)).toEqual([hex(1), hex(2)])
    expect(r.dropped).toEqual({ badUtf8: 1, badJson: 2, badShape: 1 })
    expect(manifest.dropped).toEqual(r.dropped)
  })

  it('skips a line longer than 8 MB', async () => {
    const p = await project()
    const huge = `{"format":1,"type":"final","id":"${hex(1)}","at":"${AT}","reason":"idle","text":"${'x'.repeat(8 * 1024 * 1024 + 100)}"}\n`
    const { train, r } = await exported(p, [request(1), outcomeOf(1, 'accepted'), huge, finalOf(1, FINAL_TEXT)])
    expect(r.dropped).toEqual({ tooLong: 1 })
    expect(train[0].messages.at(-1).content).toBe(FINAL_TEXT)
  }, 30000)

  it('counts outcome lines whose request is missing', async () => {
    const p = await project()
    const { r } = await exported(p, [outcomeOf(7, 'accepted'), request(1), outcomeOf(1, 'accepted')])
    expect(r.train).toBe(1)
    expect(r.dropped).toEqual({ noRequest: 1 })
  })

  it('writes empty files and a manifest when there are no records', async () => {
    const p = await project()
    const r = await p.store.exportRecords(p.folder, p.out, { editPairs: false })
    expect(r).toMatchObject({ train: 0, pairs: 0, dropped: {} })
    expect(read(join(r.outDir, 'train.jsonl'))).toBe('')
    expect(JSON.parse(read(join(r.outDir, 'manifest.json'))).dateRange).toEqual({ from: null, to: null })
  })

  it('refuses an output folder inside the project or one that is missing', async () => {
    const p = await project()
    seed(p, [request(1), outcomeOf(1, 'accepted')])
    const inside = join(p.folder, 'manuscript')
    for (const dir of [p.folder, inside, p.recDir, join(p.folder, 'not-yet')]) {
      await expect(p.store.exportRecords(p.folder, dir, { editPairs: false })).rejects.toThrow(/outside the project|does not exist/)
    }
    if (process.platform === 'win32') await expect(p.store.exportRecords(p.folder, inside.toUpperCase(), { editPairs: false })).rejects.toThrow(/outside the project/)
    await expect(p.store.exportRecords(p.folder, join(p.base, 'missing'), { editPairs: false })).rejects.toThrow(/does not exist/)
    await expect(p.store.exportRecords(p.folder, 'relative', { editPairs: false })).rejects.toThrow(/Choose a folder/)
    expect(readdirSync(p.out)).toEqual([])
    expect(readdirSync(inside)).not.toContain('train.jsonl')
  })

  it('never writes into an existing subfolder: a free name with -2, -3 is used', async () => {
    const p = await project()
    const name = 'records-20261122-093015'
    mkdirSync(join(p.out, name))
    writeFileSync(join(p.out, name, 'train.jsonl'), 'keep me')
    seed(p, [request(1), outcomeOf(1, 'accepted')])
    const a = await p.store.exportRecords(p.folder, p.out, { editPairs: false })
    const b = await p.store.exportRecords(p.folder, p.out, { editPairs: false })
    expect(a.outDir).toBe(join(p.out, `${name}-2`))
    expect(b.outDir).toBe(join(p.out, `${name}-3`))
    expect(read(join(p.out, name, 'train.jsonl'))).toBe('keep me')
    expect(readdirSync(join(p.out, name))).toEqual(['train.jsonl'])
  })

  it('leaves no half-written export behind when writing fails', async () => {
    const p = await project()
    seed(p, [request(1), outcomeOf(1, 'accepted')])
    const real = fsp.rename.bind(fsp)
    vi.spyOn(fsp, 'rename').mockImplementation((async (from: any, to: any) => {
      if (String(to).endsWith('pairs.jsonl')) throw Object.assign(new Error('disk gone'), { code: 'EIO' })
      return (real as any)(from, to)
    }) as any)
    await expect(p.store.exportRecords(p.folder, p.out, { editPairs: false })).rejects.toThrow(/export failed/)
    vi.restoreAllMocks()
    expect(readdirSync(p.out)).toEqual([])
    // the records themselves are untouched
    expect(rows(p.file)).toHaveLength(2)
  })

  it('does not change the records files', async () => {
    const p = await project()
    seed(p, [request(1), outcomeOf(1, 'accepted'), 'garbage\n'])
    const before = readFileSync(p.file)
    await p.store.exportRecords(p.folder, p.out, { editPairs: true })
    expect(readFileSync(p.file).equals(before)).toBe(true)
  })
})
