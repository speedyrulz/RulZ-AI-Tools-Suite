// Stage 3 in the file store: state.json (novel:statePatch) and .rulz/checks.json (novel:dismissFinding).
import { existsSync, mkdirSync, mkdtempSync, promises as fsp, readdirSync, readFileSync, renameSync, rmSync, symlinkSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { NovelChange, StateFact, StateOp } from '@shared/novel'
import { sha256 } from '../src/main/novel/files'
import { NovelStore } from '../src/main/novel/store'

afterEach(() => {
  vi.restoreAllMocks()
})

function setup() {
  const base = mkdtempSync(join(tmpdir(), 'rulz-novel-st-'))
  const events: NovelChange[] = []
  const store = new NovelStore({
    rootSetting: () => '',
    documents: () => join(base, 'Documents'),
    trashItem: async (p) => renameSync(p, join(base, `recycle-${Date.now()}`)),
    emit: (c) => events.push(c),
    watch: false
  })
  return { base, events, store }
}

const at = (s: string): string => s.replace(/\\/g, '/')

/** A project with two scenes in two chapters and two character entries. */
async function project() {
  const s = setup()
  const st = await s.store.createProject('State')
  const cur = await s.store.outlinePatch(st.folder, [{ op: 'addChapter', chapterId: 'c-two2222', sceneId: 's-aaa1111', title: 'Two', index: null }])
  const s1 = cur.outline.chapters[0].scenes[0].id
  const s2 = 's-aaa1111'
  const mara = (await s.store.createEntry(st.folder, 'Mara Venn', 'character')).id
  const oskar = (await s.store.createEntry(st.folder, 'Oskar Lind', 'character')).id
  const file = join(st.folder, 'state.json')
  const read = (): { format: number; facts: Record<string, unknown>[]; [k: string]: unknown } => JSON.parse(readFileSync(file, 'utf8'))
  let n = 0
  const mk = (over: Partial<StateFact> = {}): StateFact => ({ id: `f${(++n).toString(16).padStart(5, '0')}`, entity: mara, kind: 'carries', value: 'the brass key', about: null, from: s1, until: null, ...over })
  const patch = (ops: StateOp[]) => s.store.statePatch(st.folder, ops)
  return { ...s, st, folder: st.folder, s1, s2, mara, oskar, file, read, mk, patch }
}

describe('state.json: ops', () => {
  it('is not created with the project, and the first write creates it', async () => {
    const p = await project()
    expect(existsSync(p.file)).toBe(false)
    expect(p.st.state).toEqual({ facts: [], invalid: [], problem: null, canReset: false })
    expect(p.st.dismissed).toEqual({})
    expect(p.st.checksProblem).toBeNull()
    const f = p.mk()
    const r = await p.patch([{ op: 'add', fact: f }])
    expect(r.rejected).toEqual([])
    expect(r.state.state.facts).toEqual([f])
    expect(p.read()).toEqual({ format: 1, facts: [f] })
  })

  it('keeps unknown keys on the file and on facts through add, end and edit', async () => {
    const p = await project()
    const a = p.mk({ value: 'a' })
    writeFileSync(p.file, JSON.stringify({ format: 1, tool: 'newer', facts: [{ ...a, note: 'kept', nested: { x: 1 } }] }))
    await p.store.openProject(p.folder)
    const b = p.mk({ value: 'b' })
    const r = await p.patch([
      { op: 'add', fact: b },
      { op: 'end', factId: a.id, until: p.s2 },
      { op: 'edit', factId: a.id, changes: { value: 'a2' } }
    ])
    expect(r.rejected).toEqual([])
    const out = p.read()
    expect(out.tool).toBe('newer')
    expect(out.facts[0]).toMatchObject({ id: a.id, value: 'a2', until: p.s2, note: 'kept', nested: { x: 1 } })
    expect(out.facts[1]).toMatchObject({ id: b.id, value: 'b' })
  })

  it('applies the valid ops and returns the rejected ones with reasons, by index', async () => {
    const p = await project()
    const a = p.mk({ value: 'a' })
    await p.patch([{ op: 'add', fact: a }])
    const good = p.mk({ value: 'good' })
    const r = await p.patch([
      { op: 'add', fact: { ...a, id: 'fnew001' } }, // duplicate
      { op: 'add', fact: p.mk({ entity: 'nobody' }) }, // bad entity
      { op: 'add', fact: p.mk({ from: 'sX' }) }, // bad scene
      { op: 'add', fact: p.mk({ kind: 'relationship', about: null }) }, // relationship without about
      { op: 'add', fact: p.mk({ kind: 'carries', about: p.oskar }) }, // about on a non-relationship
      { op: 'add', fact: p.mk({ value: 'x\ny' }) }, // two lines
      { op: 'add', fact: p.mk({ value: '   ' }) }, // empty
      { op: 'add', fact: p.mk({ value: 'z'.repeat(301) }) }, // too long
      { op: 'add', fact: p.mk({ from: p.s2, until: p.s1 }) }, // ends before it starts
      { op: 'add', fact: { ...good, id: a.id } }, // id taken
      { op: 'add', fact: good }, // fine
      { op: 'end', factId: 'fmissing', until: p.s2 },
      { op: 'end', factId: a.id, until: 'sX' },
      { op: 'end', factId: a.id, until: p.s2 }, // fine
      { op: 'end', factId: a.id, until: p.s2 }, // already ended
      { op: 'remove', factId: 'fmissing' },
      { op: 'bogus' } as unknown as StateOp
    ])
    expect(r.rejected.map((x) => x.index)).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 14, 15, 16])
    for (const x of r.rejected) expect(x.reason.length).toBeGreaterThan(3)
    expect(r.state.state.facts.map((f) => f.value)).toEqual(['a', 'good'])
    expect(r.state.state.facts[0].until).toBe(p.s2)
  })

  it('trims a value, and edit can fix a detached fact by changing its entity', async () => {
    const p = await project()
    const f = p.mk({ entity: 'deleted-entry', value: 'old' })
    writeFileSync(p.file, JSON.stringify({ format: 1, facts: [f] }))
    await p.store.openProject(p.folder)
    // Changing only the value does not need the (detached) entity to exist.
    let r = await p.patch([{ op: 'edit', factId: f.id, changes: { value: '  new  ' } }])
    expect(r.rejected).toEqual([])
    expect(r.state.state.facts[0]).toMatchObject({ entity: 'deleted-entry', value: 'new' })
    r = await p.patch([
      { op: 'edit', factId: f.id, changes: { entity: 'still-nobody' } },
      { op: 'edit', factId: f.id, changes: { entity: p.oskar } },
      { op: 'edit', factId: f.id, changes: {} },
      { op: 'edit', factId: f.id, changes: { kind: 'relationship' } }
    ])
    expect(r.rejected.map((x) => x.index)).toEqual([0, 2, 3])
    expect(r.state.state.facts[0].entity).toBe(p.oskar)
  })

  it('an edit that moves until before from is refused', async () => {
    const p = await project()
    const f = p.mk({ from: p.s2 })
    await p.patch([{ op: 'add', fact: f }])
    const r = await p.patch([{ op: 'edit', factId: f.id, changes: { until: p.s1 } }])
    expect(r.rejected).toHaveLength(1)
    expect(p.read().facts[0].until).toBeNull()
  })

  it('writes nothing when no op applied', async () => {
    const p = await project()
    const before = JSON.stringify({ format: 1, facts: [p.mk()] }, null, 1)
    writeFileSync(p.file, before)
    await p.store.openProject(p.folder)
    const r = await p.patch([{ op: 'remove', factId: 'fnone' }])
    expect(r.rejected).toHaveLength(1)
    expect(readFileSync(p.file, 'utf8')).toBe(before)
    expect(existsSync(join(p.folder, '.rulz', 'history', 'state.json'))).toBe(false)
  })

  it('rejects a bad argument', async () => {
    const p = await project()
    await expect(p.store.statePatch(p.folder, 'nope')).rejects.toThrow(/list/)
  })
})

describe('state.json: unreadable facts', () => {
  it('keeps a fact that does not validate verbatim through a write, and removes it by hash', async () => {
    const p = await project()
    const bad = { id: 'fbad001', entity: p.mara, kind: 'wishes', value: 'to leave', from: p.s1, extra: [1, 2] }
    const good = p.mk({ value: 'ok' })
    writeFileSync(p.file, JSON.stringify({ format: 1, facts: [bad, good, 'junk'] }))
    const st = await p.store.openProject(p.folder)
    expect(st.state.facts).toEqual([good])
    expect(st.state.invalid.map((i) => i.index)).toEqual([0, 2])
    expect(st.state.invalid[0]).toMatchObject({ index: 0, raw: JSON.stringify(bad), hash: sha256(JSON.stringify(bad)) })
    expect(st.state.invalid[0].reason).toMatch(/wishes/)

    const r = await p.patch([{ op: 'add', fact: p.mk({ value: 'new' }) }])
    expect(r.rejected).toEqual([])
    expect(p.read().facts[0]).toEqual(bad)
    expect(p.read().facts[2]).toBe('junk')

    // A wrong hash removes nothing; a right one removes exactly that item (the other invalid index still resolves).
    const inv = r.state.state.invalid
    const r2 = await p.patch([
      { op: 'removeInvalid', index: inv[0].index, hash: 'f'.repeat(64) },
      { op: 'removeInvalid', index: inv[1].index, hash: inv[1].hash },
      { op: 'removeInvalid', index: inv[0].index, hash: inv[0].hash }
    ])
    expect(r2.rejected.map((x) => x.index)).toEqual([0])
    expect(p.read().facts.map((f) => f.id)).toEqual([good.id, expect.any(String)])
    expect(r2.state.state.invalid).toEqual([])
  })

  it('treats a duplicate id as unreadable instead of dropping it', async () => {
    const p = await project()
    const a = p.mk({ value: 'a' })
    writeFileSync(p.file, JSON.stringify({ format: 1, facts: [a, { ...a, value: 'second' }] }))
    const st = await p.store.openProject(p.folder)
    expect(st.state.facts.map((f) => f.value)).toEqual(['a'])
    expect(st.state.invalid).toHaveLength(1)
    await p.patch([{ op: 'add', fact: p.mk({ value: 'b' }) }])
    expect(p.read().facts.map((f) => f.value)).toEqual(['a', 'second', 'b'])
  })
})

describe('state.json: damaged or missing', () => {
  for (const [name, content, reset] of [
    ['not JSON', '{ nope', false],
    ['a newer format', JSON.stringify({ format: 2, facts: [] }), false],
    ['no facts array', JSON.stringify({ format: 1, things: [] }), false],
    ['not an object', '[1, 2]', false],
    ['0 bytes', '', true]
  ] as const) {
    it(`${name}: a problem, every op refused, nothing written, reset only for 0 bytes`, async () => {
      const p = await project()
      writeFileSync(p.file, content)
      const st = await p.store.openProject(p.folder)
      expect(st.state.problem).toBeTruthy()
      expect(st.state.canReset).toBe(reset)
      expect(st.state.facts).toEqual([])
      const r = await p.patch([{ op: 'add', fact: p.mk() }, { op: 'remove', factId: 'x' }, { op: 'reset' }])
      if (reset) {
        expect(r.rejected.map((x) => x.index)).toEqual([0, 1])
        expect(r.state.state.problem).toBeNull()
        expect(p.read()).toEqual({ format: 1, facts: [] })
      } else {
        expect(r.state.state.problem).toBeTruthy()
        expect(r.rejected.map((x) => x.index)).toEqual([0, 1, 2])
        expect(readFileSync(p.file, 'utf8')).toBe(content)
        expect(existsSync(join(p.folder, '.rulz', 'history', 'state.json'))).toBe(false)
      }
    })
  }

  it('missing at open is an empty state; missing later is a problem that reset can fix', async () => {
    const p = await project()
    const f = p.mk()
    await p.patch([{ op: 'add', fact: f }])
    rmSync(p.file)
    const st = await p.store.openProject(p.folder)
    expect(st.state).toMatchObject({ facts: [], problem: expect.stringMatching(/missing/), canReset: true })
    const r = await p.patch([{ op: 'add', fact: p.mk() }])
    expect(r.rejected).toHaveLength(1)
    expect(existsSync(p.file)).toBe(false)
    const r2 = await p.patch([{ op: 'reset' }, { op: 'add', fact: p.mk({ value: 'after reset' }) }])
    expect(r2.rejected).toEqual([])
    expect(p.read().facts.map((x) => x.value)).toEqual(['after reset'])
    expect(r2.state.state.problem).toBeNull()
  })

  it('reset is refused when the file is fine', async () => {
    const p = await project()
    await p.patch([{ op: 'add', fact: p.mk() }])
    const r = await p.patch([{ op: 'reset' }])
    expect(r.rejected).toHaveLength(1)
    expect(p.read().facts).toHaveLength(1)
  })

  it('an outside fix clears the problem and an outside break sets it, with a state event', async () => {
    const p = await project()
    await p.patch([{ op: 'add', fact: p.mk() }])
    writeFileSync(p.file, '{ broken')
    p.events.length = 0
    await p.store.onExternalChange(p.folder, 'state.json')
    const ev = p.events.find((e) => e.kind === 'state')
    expect(ev && ev.kind === 'state' && ev.state.state.problem).toMatch(/could not be read/)
    writeFileSync(p.file, JSON.stringify({ format: 1, facts: [p.mk({ value: 'fixed' })] }))
    p.events.length = 0
    await p.store.onExternalChange(p.folder, 'state.json')
    const ev2 = p.events.find((e) => e.kind === 'state')
    expect(ev2 && ev2.kind === 'state' && ev2.state.state).toMatchObject({ problem: null, facts: [{ value: 'fixed' }] })
  })

  it('a blocked project refuses every op, reset included', async () => {
    const p = await project()
    writeFileSync(join(p.folder, 'outline.json'), '{ broken')
    const r = await p.patch([{ op: 'add', fact: p.mk() }, { op: 'reset' }])
    expect(r.rejected.map((x) => x.index)).toEqual([0, 1])
    expect(r.rejected[0].reason).toMatch(/read-only/)
    expect(existsSync(p.file)).toBe(false)
  })
})

describe('state.json: files, history, races', () => {
  it('snapshots the old content before overwriting, and state.json has a history', async () => {
    const p = await project()
    const first = JSON.stringify({ format: 1, facts: [p.mk({ value: 'first' })] }, null, 4)
    writeFileSync(p.file, first)
    await p.store.openProject(p.folder)
    await p.patch([{ op: 'add', fact: p.mk({ value: 'second' }) }])
    const items = await p.store.history(p.folder, 'state.json')
    expect(items).toHaveLength(1)
    expect(await p.store.readSnapshot(p.folder, items[0].snapshot)).toBe(first)
    expect(p.read().facts).toHaveLength(2)
  })

  it('an outside edit between the read and the write: the ops are applied again to the new content', async () => {
    const p = await project()
    const a = p.mk({ value: 'a' })
    await p.patch([{ op: 'add', fact: a }])
    const ext = p.mk({ value: 'added outside' })
    const real = fsp.open.bind(fsp)
    let hit = false
    vi.spyOn(fsp, 'open').mockImplementation((async (path: Parameters<typeof fsp.open>[0], ...rest: unknown[]) => {
      if (!hit && at(String(path)).includes('/.rulz/tmp/')) {
        hit = true
        writeFileSync(p.file, JSON.stringify({ format: 1, facts: [a, ext] }))
      }
      return (real as (...x: unknown[]) => Promise<unknown>)(path, ...rest)
    }) as never)
    const mine = p.mk({ value: 'mine' })
    const r = await p.patch([{ op: 'add', fact: mine }])
    vi.restoreAllMocks()
    expect(hit).toBe(true)
    expect(r.rejected).toEqual([])
    expect(p.read().facts.map((f) => f.value)).toEqual(['a', 'added outside', 'mine'])
    expect(readdirSync(join(p.folder, '.rulz', 'tmp'))).toEqual([])
    // The version that was overwritten outside got a snapshot too.
    const hist = await p.store.history(p.folder, 'state.json')
    const texts = await Promise.all(hist.map((h) => p.store.readSnapshot(p.folder, h.snapshot)))
    expect(texts.some((t) => t.includes('added outside'))).toBe(true)
  })

  it('an outside edit that breaks the file between the read and the write: the ops are refused, nothing overwritten', async () => {
    const p = await project()
    await p.patch([{ op: 'add', fact: p.mk({ value: 'a' }) }])
    const real = fsp.open.bind(fsp)
    let hit = false
    vi.spyOn(fsp, 'open').mockImplementation((async (path: Parameters<typeof fsp.open>[0], ...rest: unknown[]) => {
      if (!hit && at(String(path)).includes('/.rulz/tmp/')) {
        hit = true
        writeFileSync(p.file, '{ half written')
      }
      return (real as (...x: unknown[]) => Promise<unknown>)(path, ...rest)
    }) as never)
    const r = await p.patch([{ op: 'add', fact: p.mk({ value: 'mine' }) }])
    vi.restoreAllMocks()
    expect(r.rejected).toHaveLength(1)
    expect(readFileSync(p.file, 'utf8')).toBe('{ half written')
    expect(r.state.state.problem).toBeTruthy()
  })

  it('a state.json that is a link out of the project is not written through', async () => {
    const p = await project()
    const outside = join(p.base, 'outside')
    mkdirSync(outside)
    try {
      symlinkSync(outside, join(p.folder, 'state.json'), 'junction')
    } catch {
      return
    }
    const opened = await p.store.openProject(p.folder)
    expect(opened.state.problem).toMatch(/outside the project/)
    expect(opened.state.canReset).toBe(false)
    const r = await p.patch([{ op: 'add', fact: p.mk() }, { op: 'reset' }])
    expect(r.rejected).toHaveLength(2)
    expect(readdirSync(outside)).toEqual([])
  })

  it('detached facts survive a deleted entry, and come back with it from the trash', async () => {
    const p = await project()
    const f = p.mk()
    await p.patch([{ op: 'add', fact: f }])
    await p.store.deleteEntry(p.folder, p.mara)
    expect(p.read().facts).toHaveLength(1)
    const r = await p.patch([{ op: 'edit', factId: f.id, changes: { value: 'still editable' } }])
    expect(r.rejected).toEqual([])
  })
})

describe('state.json: review fixes', () => {
  it('a write that throws leaves the held state unchanged, and the same op applies on retry', async () => {
    const p = await project()
    const a = p.mk({ value: 'a' })
    await p.patch([{ op: 'add', fact: a }])
    const real = fsp.rename.bind(fsp)
    vi.spyOn(fsp, 'rename').mockImplementation(async (from, to) => {
      if (at(String(to)).endsWith('/state.json')) throw Object.assign(new Error('disk full'), { code: 'ENOSPC' })
      return real(from, to)
    })
    await expect(p.patch([{ op: 'end', factId: a.id, until: p.s2 }, { op: 'edit', factId: a.id, changes: { value: 'changed' } }])).rejects.toThrow(/disk full/)
    vi.restoreAllMocks()
    expect((await p.store.openProject(p.folder)).state.facts[0]).toMatchObject({ value: 'a', until: null })
    const r = await p.patch([{ op: 'end', factId: a.id, until: p.s2 }])
    expect(r.rejected).toEqual([])
    expect(p.read().facts[0]).toMatchObject({ value: 'a', until: p.s2 })
  })

  it('an outside edit during a rename retry wait is not overwritten: the ops are applied again', async () => {
    const p = await project()
    const a = p.mk({ value: 'a' })
    await p.patch([{ op: 'add', fact: a }])
    const ext = p.mk({ value: 'added outside' })
    const real = fsp.rename.bind(fsp)
    let hit = false
    vi.spyOn(fsp, 'rename').mockImplementation(async (from, to) => {
      if (!hit && at(String(to)).endsWith('/state.json')) {
        hit = true
        writeFileSync(p.file, JSON.stringify({ format: 1, facts: [a, ext] }))
        throw Object.assign(new Error('busy'), { code: 'EBUSY' })
      }
      return real(from, to)
    })
    const r = await p.patch([{ op: 'add', fact: p.mk({ value: 'mine' }) }])
    vi.restoreAllMocks()
    expect(hit).toBe(true)
    expect(r.rejected).toEqual([])
    expect(p.read().facts.map((f) => f.value)).toEqual(['a', 'added outside', 'mine'])
    expect(readdirSync(join(p.folder, '.rulz', 'tmp'))).toEqual([])
  })

  it('a state.json that is a file link out of the project is a problem, never read or written', async () => {
    const p = await project()
    const outside = join(p.base, 'outside.json')
    writeFileSync(outside, JSON.stringify({ format: 1, facts: [p.mk({ value: 'outside' })] }))
    try {
      symlinkSync(outside, p.file, 'file')
    } catch {
      return
    }
    const st = await p.store.openProject(p.folder)
    expect(st.state.problem).toMatch(/outside the project/)
    expect(st.state.canReset).toBe(false)
    expect(st.state.facts).toEqual([])
    const r = await p.patch([{ op: 'add', fact: p.mk() }, { op: 'reset' }])
    expect(r.rejected).toHaveLength(2)
    expect(JSON.parse(readFileSync(outside, 'utf8')).facts).toHaveLength(1)
  })

  it('invalid UTF-8 in state.json is a problem that cannot be reset, and nothing is written', async () => {
    const p = await project()
    const bytes = Buffer.concat([Buffer.from('{"format":1,"facts":[],"note":"'), Buffer.from([0xff, 0xfe]), Buffer.from('"}')])
    writeFileSync(p.file, bytes)
    const st = await p.store.openProject(p.folder)
    expect(st.state.problem).toMatch(/could not be read/)
    expect(st.state.canReset).toBe(false)
    const r = await p.patch([{ op: 'add', fact: p.mk() }])
    expect(r.rejected).toHaveLength(1)
    expect(readFileSync(p.file).equals(bytes)).toBe(true)
  })

  it('a UTF-8 BOM is accepted', async () => {
    const p = await project()
    writeFileSync(p.file, Buffer.concat([Buffer.from([0xef, 0xbb, 0xbf]), Buffer.from(JSON.stringify({ format: 1, facts: [p.mk({ value: 'bom' })] }))]))
    const st = await p.store.openProject(p.folder)
    expect(st.state.problem).toBeNull()
    expect(st.state.facts.map((f) => f.value)).toEqual(['bom'])
  })

  it('an untouched fact is written exactly as read (key order, extra keys), and a padded value is unreadable and kept', async () => {
    const p = await project()
    const odd = { until: null, value: 'odd order', extra: { b: 1, a: 2 }, kind: 'carries', id: 'fodd001', from: p.s1, entity: p.mara }
    const padded = { id: 'fpad001', entity: p.mara, kind: 'knows', value: ' padded ', from: p.s1 }
    const multi = { id: 'fmul001', entity: p.mara, kind: 'knows', value: 'two\nlines', from: p.s1 }
    writeFileSync(p.file, JSON.stringify({ format: 1, facts: [odd, padded, multi] }))
    const st = await p.store.openProject(p.folder)
    expect(st.state.facts.map((f) => f.id)).toEqual(['fodd001'])
    expect(st.state.invalid.map((i) => i.index)).toEqual([1, 2])
    await p.patch([{ op: 'add', fact: p.mk({ value: 'unrelated' }) }])
    const out = p.read().facts
    expect(JSON.stringify(out[0])).toBe(JSON.stringify(odd))
    expect(out[1]).toEqual(padded)
    expect(out[2]).toEqual(multi)
    // An edit rewrites only that fact, keeping its extra keys.
    await p.patch([{ op: 'edit', factId: 'fodd001', changes: { value: 'edited' } }])
    expect(p.read().facts[0]).toMatchObject({ value: 'edited', extra: { b: 1, a: 2 } })
  })

  it('an unreadable fact with an id makes a later valid fact with that id a duplicate', async () => {
    const p = await project()
    const bad = { id: 'fsame01', entity: p.mara, kind: 'nope', value: 'x', from: p.s1 }
    const good = p.mk({ id: 'fsame01', value: 'later' })
    writeFileSync(p.file, JSON.stringify({ format: 1, facts: [bad, good] }))
    const st = await p.store.openProject(p.folder)
    expect(st.state.facts).toEqual([])
    expect(st.state.invalid.map((i) => i.index)).toEqual([0, 1])
  })

  it('edit: undefined values are ignored, only changed fields are checked, until needs an existing from', async () => {
    const p = await project()
    const f = p.mk({ entity: 'deleted-entry', value: 'old' })
    const inv = p.mk({ value: 'inverted', from: p.s2 })
    const orphan = p.mk({ value: 'orphan', from: 'sGone' })
    writeFileSync(p.file, JSON.stringify({ format: 1, facts: [f, { ...inv, until: p.s1 }, orphan] }))
    await p.store.openProject(p.folder)
    // A whole-form edit that changes only the value goes through on a detached and on an inverted fact.
    let r = await p.patch([
      { op: 'edit', factId: f.id, changes: { entity: f.entity, kind: f.kind, value: 'new', about: null, from: f.from, until: undefined } },
      { op: 'edit', factId: inv.id, changes: { entity: inv.entity, value: 'inverted too', from: inv.from, until: p.s1 } }
    ])
    expect(r.rejected).toEqual([])
    expect(r.state.state.facts[0]).toMatchObject({ value: 'new', until: null })
    expect(r.state.state.facts[1]).toMatchObject({ value: 'inverted too', until: p.s1 })
    // Nothing really changed: refused.
    r = await p.patch([{ op: 'edit', factId: f.id, changes: { value: 'new', until: undefined } }])
    expect(r.rejected).toHaveLength(1)
    // Setting until on a fact whose from scene is gone is refused; changing from too makes it fine.
    r = await p.patch([
      { op: 'edit', factId: orphan.id, changes: { until: p.s2 } },
      { op: 'edit', factId: orphan.id, changes: { from: p.s1, until: p.s2 } }
    ])
    expect(r.rejected.map((x) => x.index)).toEqual([0])
    expect(r.rejected[0].reason).toMatch(/starts no longer exists/)
    expect(r.state.state.facts[2]).toMatchObject({ from: p.s1, until: p.s2 })
  })
})

describe('.rulz/checks.json', () => {
  it('invalid UTF-8 in checks.json is a problem and the file is not written', async () => {
    const p = await project()
    mkdirSync(join(p.folder, '.rulz'), { recursive: true })
    const bytes = Buffer.concat([Buffer.from('{"format":1,"dismissed":{},"x":"'), Buffer.from([0xc3, 0x28]), Buffer.from('"}')])
    writeFileSync(join(p.folder, '.rulz', 'checks.json'), bytes)
    const st = await p.store.openProject(p.folder)
    expect(st.checksProblem).toMatch(/checks\.json/)
    await expect(p.store.dismissFinding(p.folder, p.s1, 'k', true)).rejects.toThrow()
    expect(readFileSync(join(p.folder, '.rulz', 'checks.json')).equals(bytes)).toBe(true)
  })

  const checksFile = (p: { folder: string }): string => join(p.folder, '.rulz', 'checks.json')
  const readChecks = (p: { folder: string }): { format: number; dismissed: Record<string, string[]>; [k: string]: unknown } => JSON.parse(readFileSync(checksFile(p), 'utf8'))

  it('dismisses and un-dismisses a finding key, keeping unknown keys', async () => {
    const p = await project()
    let st = await p.store.dismissFinding(p.folder, p.s1, 'k1', true)
    expect(st.dismissed).toEqual({ [p.s1]: ['k1'] })
    expect(readChecks(p)).toEqual({ format: 1, dismissed: { [p.s1]: ['k1'] } })
    writeFileSync(checksFile(p), JSON.stringify({ format: 1, future: true, dismissed: { [p.s1]: ['k1'] } }))
    st = await p.store.dismissFinding(p.folder, p.s2, 'k2', true)
    st = await p.store.dismissFinding(p.folder, p.s1, 'k3', true)
    expect(st.dismissed).toEqual({ [p.s1]: ['k1', 'k3'], [p.s2]: ['k2'] })
    expect(readChecks(p).future).toBe(true)
    st = await p.store.dismissFinding(p.folder, p.s1, 'k1', false)
    st = await p.store.dismissFinding(p.folder, p.s2, 'k2', false)
    expect(st.dismissed).toEqual({ [p.s1]: ['k3'] })
    expect(readChecks(p).dismissed).toEqual({ [p.s1]: ['k3'] })
  })

  it('writes nothing when the key is already in the wanted state', async () => {
    const p = await project()
    const st = await p.store.dismissFinding(p.folder, p.s1, 'gone', false)
    expect(st.dismissed).toEqual({})
    expect(existsSync(checksFile(p))).toBe(false)
    await p.store.dismissFinding(p.folder, p.s1, 'k', true)
    const before = readFileSync(checksFile(p), 'utf8')
    await p.store.dismissFinding(p.folder, p.s1, 'k', true)
    expect(readFileSync(checksFile(p), 'utf8')).toBe(before)
  })

  it('checks the arguments: scene must exist, key at most 64 characters', async () => {
    const p = await project()
    await expect(p.store.dismissFinding(p.folder, 'sX', 'k', true)).rejects.toThrow(/scene/)
    await expect(p.store.dismissFinding(p.folder, p.s1, '', true)).rejects.toThrow(/key/)
    await expect(p.store.dismissFinding(p.folder, p.s1, 'k'.repeat(65), true)).rejects.toThrow(/key/)
    await p.store.dismissFinding(p.folder, p.s1, 'k'.repeat(64), true)
    await expect(p.store.dismissFinding(p.folder, p.s1, 'k', 'yes' as unknown as boolean)).rejects.toThrow()
  })

  it('a damaged checks.json is reported, never overwritten, and dismissals are refused', async () => {
    const p = await project()
    mkdirSync(join(p.folder, '.rulz'), { recursive: true })
    writeFileSync(checksFile(p), '{ nope')
    const st = await p.store.openProject(p.folder)
    expect(st.checksProblem).toMatch(/checks\.json/)
    expect(st.dismissed).toEqual({})
    await expect(p.store.dismissFinding(p.folder, p.s1, 'k', true)).rejects.toThrow(/checks\.json/)
    expect(readFileSync(checksFile(p), 'utf8')).toBe('{ nope')
    // Fixed outside: dismissals work again.
    writeFileSync(checksFile(p), JSON.stringify({ format: 1, dismissed: {} }))
    const st2 = await p.store.dismissFinding(p.folder, p.s1, 'k', true)
    expect(st2.checksProblem).toBeNull()
    expect(st2.dismissed).toEqual({ [p.s1]: ['k'] })
  })

  for (const [name, content] of [
    ['a dismissed value that is not a list', JSON.stringify({ format: 1, dismissed: { s: 'x' } })],
    ['a newer format', JSON.stringify({ format: 3, dismissed: {} })]
  ] as const) {
    it(`${name} counts as damaged`, async () => {
      const p = await project()
      writeFileSync(checksFile(p), content)
      const st = await p.store.openProject(p.folder)
      expect(st.checksProblem).toBeTruthy()
      await expect(p.store.dismissFinding(p.folder, p.s1, 'k', true)).rejects.toThrow()
      expect(readFileSync(checksFile(p), 'utf8')).toBe(content)
    })
  }

  it('is refused on a blocked project', async () => {
    const p = await project()
    writeFileSync(join(p.folder, 'project.json'), '{ broken')
    await expect(p.store.dismissFinding(p.folder, p.s1, 'k', true)).rejects.toThrow(/read-only/)
    expect(existsSync(checksFile(p))).toBe(false)
  })

  it('an outside edit between the read and the write is merged, not overwritten', async () => {
    const p = await project()
    await p.store.dismissFinding(p.folder, p.s1, 'k1', true)
    const real = fsp.open.bind(fsp)
    let hit = false
    vi.spyOn(fsp, 'open').mockImplementation((async (path: Parameters<typeof fsp.open>[0], ...rest: unknown[]) => {
      if (!hit && at(String(path)).includes('/.rulz/tmp/')) {
        hit = true
        writeFileSync(checksFile(p), JSON.stringify({ format: 1, dismissed: { [p.s1]: ['k1', 'outside'] } }))
      }
      return (real as (...x: unknown[]) => Promise<unknown>)(path, ...rest)
    }) as never)
    const st = await p.store.dismissFinding(p.folder, p.s1, 'k2', true)
    vi.restoreAllMocks()
    expect(hit).toBe(true)
    expect(st.dismissed[p.s1]).toEqual(['k1', 'outside', 'k2'])
    expect(readChecks(p).dismissed[p.s1]).toEqual(['k1', 'outside', 'k2'])
  })

  it('a junction at .rulz: nothing is written outside, the dismissal is refused', async () => {
    const p = await project()
    const outside = join(p.base, 'outside')
    mkdirSync(outside)
    rmSync(join(p.folder, '.rulz'), { recursive: true, force: true })
    try {
      symlinkSync(outside, join(p.folder, '.rulz'), 'junction')
    } catch {
      return
    }
    await expect(p.store.dismissFinding(p.folder, p.s1, 'k', true)).rejects.toThrow()
    expect(readdirSync(outside)).toEqual([])
  })
})
