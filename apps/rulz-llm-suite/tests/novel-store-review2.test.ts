// Second round of review fixes for the Writing mode's file store (races, links, crash safety).
import { existsSync, mkdirSync, mkdtempSync, promises as fsp, readdirSync, readFileSync, renameSync, rmSync, symlinkSync, utimesSync, writeFileSync, statSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { NovelChange, NovelProjectState } from '@shared/novel'
import { compareStamps, takeSnapshot, thinSnapshots } from '../src/main/novel/history'
import { NovelStore } from '../src/main/novel/store'

afterEach(() => {
  vi.restoreAllMocks()
})

function setup() {
  const base = mkdtempSync(join(tmpdir(), 'rulz-novel-r2-'))
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
type Ctx = ReturnType<typeof setup>

const sceneRel = (st: NovelProjectState): string => `manuscript/${st.outline.chapters[0].scenes[0].file}`
const abs = (st: NovelProjectState, rel: string): string => join(st.folder, ...rel.split('/'))
const at = (s: string): string => s.replace(/\\/g, '/')

describe('stamp order', () => {
  it('sorts the collision suffix as a number: -2 before -10, plain first', () => {
    const base = '20261122T101500'
    const list = [`${base}-10`, `${base}-2`, base, '20261122T101501'].sort(compareStamps)
    expect(list).toEqual([base, `${base}-2`, `${base}-10`, '20261122T101501'])
  })

  it('thinning keeps suffixed stamps of the last hour and orders them numerically', () => {
    const now = new Date('2026-11-22T10:20:00Z')
    const base = '20261122T101500'
    const kept = thinSnapshots([`${base}-10`, `${base}-2`, base], now)
    expect(kept).toEqual([base, `${base}-2`, `${base}-10`])
  })

  it('the newest-snapshot lookup picks -10 over -2', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'rulz-hist-'))
    const rel = 'manuscript/a.md'
    const when = new Date()
    // Twelve different contents in the same second.
    for (let i = 0; i < 12; i++) await takeSnapshot(dir, rel, Buffer.from(`v${i}`), when)
    // The newest one holds v11, so snapshotting v11 again adds nothing, but v10 does.
    expect(await takeSnapshot(dir, rel, Buffer.from('v11'), when)).toBeNull()
    expect(await takeSnapshot(dir, rel, Buffer.from('v10'), when)).not.toBeNull()
  })
})

describe('outline race during a delete', () => {
  it('does not overwrite an outline.json edited outside while the files move, and puts the files back', async () => {
    const s = setup()
    const st = await s.store.createProject('Race')
    let cur = await s.store.outlinePatch(st.folder, [{ op: 'addChapter', chapterId: 'c-two2222', sceneId: 's-aaa1111', title: 'Two', index: null }])
    const file = cur.outline.chapters[1].scenes[0].file
    writeFileSync(abs(cur, `manuscript/${file}`), 'words\n')
    const external = JSON.parse(readFileSync(join(st.folder, 'outline.json'), 'utf8'))
    external.chapters[0].mood = 'added outside'
    const real = fsp.rename.bind(fsp)
    let done = false
    vi.spyOn(fsp, 'rename').mockImplementation(async (from, to) => {
      if (!done && at(String(to)).includes('/.rulz/trash/')) {
        done = true
        writeFileSync(join(st.folder, 'outline.json'), JSON.stringify(external))
      }
      return real(from, to)
    })
    await expect(s.store.deleteChapter(st.folder, 'c-two2222')).rejects.toThrow(/changed on disk/)
    vi.restoreAllMocks()
    expect(JSON.parse(readFileSync(join(st.folder, 'outline.json'), 'utf8')).chapters[0].mood).toBe('added outside')
    expect(readFileSync(abs(cur, `manuscript/${file}`), 'utf8')).toBe('words\n')
    cur = await s.store.openProject(st.folder)
    expect(cur.outline.chapters).toHaveLength(2)
  })

  it('restores the in-memory outline and the files when the outline write fails', async () => {
    const s = setup()
    const st = await s.store.createProject('Write fails')
    const cur = await s.store.outlinePatch(st.folder, [{ op: 'addChapter', chapterId: 'c-two2222', sceneId: 's-aaa1111', title: 'Two', index: null }])
    const file = cur.outline.chapters[1].scenes[0].file
    writeFileSync(abs(cur, `manuscript/${file}`), 'words\n')
    const real = fsp.rename.bind(fsp)
    vi.spyOn(fsp, 'rename').mockImplementation(async (from, to) => {
      if (at(String(to)).endsWith('/outline.json')) throw Object.assign(new Error('disk full'), { code: 'ENOSPC' })
      return real(from, to)
    })
    await expect(s.store.deleteChapter(st.folder, 'c-two2222')).rejects.toThrow(/disk full/)
    vi.restoreAllMocks()
    expect(readFileSync(abs(cur, `manuscript/${file}`), 'utf8')).toBe('words\n')
    expect((await s.store.openProject(st.folder)).outline.chapters).toHaveLength(2)
  })
})

describe('saves and deletes', () => {
  it('a save does not bring back a scene that is no longer in the outline', async () => {
    const s = setup()
    const st = await s.store.createProject('Late save')
    const r = await s.store.saveFile({ folder: st.folder, path: 'manuscript/zzz-gone.md', text: 'late\n', baseHash: '' })
    expect(r).toMatchObject({ ok: false, reason: 'error' })
    expect(existsSync(join(st.folder, 'manuscript', 'zzz-gone.md'))).toBe(false)
  })

  it('a save racing a chapter delete never leaves the file behind', async () => {
    const s = setup()
    const st = await s.store.createProject('Race save')
    let cur = await s.store.outlinePatch(st.folder, [{ op: 'addChapter', chapterId: 'c-two2222', sceneId: 's-aaa1111', title: 'Two', index: null }])
    const rel = `manuscript/${cur.outline.chapters[1].scenes[0].file}`
    writeFileSync(abs(cur, rel), 'one\n')
    const f = await s.store.readFile(st.folder, rel)
    const del = s.store.deleteChapter(st.folder, 'c-two2222')
    const save = s.store.saveFile({ folder: st.folder, path: rel, text: 'two\n', baseHash: f.hash })
    await Promise.all([del, save])
    expect(existsSync(abs(cur, rel))).toBe(false)
    cur = await s.store.openProject(st.folder)
    expect(cur.orphans).toEqual([])
  })
})

describe('links out of the project', () => {
  function link(target: string, path: string): boolean {
    try {
      symlinkSync(target, path, 'junction')
      return true
    } catch {
      return false
    }
  }

  it('refuses to read, save or list through a manuscript folder that is a link to elsewhere', async () => {
    const s = setup()
    const st = await s.store.createProject('Links')
    const rel = sceneRel(st)
    const outside = join(s.base, 'outside')
    mkdirSync(outside)
    writeFileSync(join(outside, st.outline.chapters[0].scenes[0].file), 'secret\n')
    rmSync(join(st.folder, 'manuscript'), { recursive: true })
    if (!link(outside, join(st.folder, 'manuscript'))) return // links not permitted here
    await expect(s.store.readFile(st.folder, rel)).rejects.toThrow(/outside the project/)
    expect(await s.store.saveFile({ folder: st.folder, path: rel, text: 'x', baseHash: '' })).toMatchObject({ ok: false, reason: 'error', error: expect.stringMatching(/outside the project/) })
    expect(readFileSync(join(outside, st.outline.chapters[0].scenes[0].file), 'utf8')).toBe('secret\n')
    expect((await s.store.openProject(st.folder)).words).toEqual({})
  })

  it('allows a link that stays inside the project', async () => {
    const s = setup()
    const st = await s.store.createProject('Inner link')
    mkdirSync(join(st.folder, 'real-bible'))
    rmSync(join(st.folder, 'bible'), { recursive: true })
    if (!link(join(st.folder, 'real-bible'), join(st.folder, 'bible'))) return
    const e = await s.store.createEntry(st.folder, 'Mara', 'character')
    expect(existsSync(join(st.folder, 'real-bible', `${e.id}.md`))).toBe(true)
  })
})

describe('unedited and conflicting saves', () => {
  it('does not rewrite a file with mixed line endings when the text is unchanged', async () => {
    const s = setup()
    const st = await s.store.createProject('Mixed')
    const rel = sceneRel(st)
    writeFileSync(abs(st, rel), 'a\r\nb\nc\r\nd\r\n')
    const old = new Date('2020-01-01T00:00:00Z')
    utimesSync(abs(st, rel), old, old)
    const f = await s.store.readFile(st.folder, rel)
    const r = await s.store.saveFile({ folder: st.folder, path: rel, text: f.text, baseHash: f.hash })
    expect(r).toEqual({ ok: true, hash: f.hash })
    expect(readFileSync(abs(st, rel), 'utf8')).toBe('a\r\nb\nc\r\nd\r\n')
    expect(statSync(abs(st, rel)).mtimeMs).toBe(old.getTime())
  })

  it('reports a conflict when the file changes while the temp file is being written', async () => {
    const s = setup()
    const st = await s.store.createProject('Window')
    const rel = sceneRel(st)
    writeFileSync(abs(st, rel), 'v1\n')
    const f = await s.store.readFile(st.folder, rel)
    const real = fsp.open.bind(fsp)
    let hit = false
    vi.spyOn(fsp, 'open').mockImplementation((async (path: Parameters<typeof fsp.open>[0], ...rest: unknown[]) => {
      if (!hit && at(String(path)).includes('/.rulz/tmp/')) {
        hit = true
        writeFileSync(abs(st, rel), 'external\n')
      }
      return (real as (...a: unknown[]) => Promise<unknown>)(path, ...rest)
    }) as never)
    const r = await s.store.saveFile({ folder: st.folder, path: rel, text: 'mine\n', baseHash: f.hash })
    vi.restoreAllMocks()
    expect(r).toMatchObject({ ok: false, reason: 'conflict', diskText: 'external\n' })
    expect(readFileSync(abs(st, rel), 'utf8')).toBe('external\n')
    expect(readdirSync(join(st.folder, '.rulz', 'tmp'))).toEqual([])
  })

  it('falls back to a temp file next to the target when the rename crosses disks', async () => {
    const s = setup()
    const st = await s.store.createProject('Cross disk')
    const rel = sceneRel(st)
    const real = fsp.rename.bind(fsp)
    let first = true
    vi.spyOn(fsp, 'rename').mockImplementation(async (from, to) => {
      if (first && at(String(from)).includes('/.rulz/tmp/')) {
        first = false
        throw Object.assign(new Error('cross-device'), { code: 'EXDEV' })
      }
      return real(from, to)
    })
    const f = await s.store.readFile(st.folder, rel)
    const r = await s.store.saveFile({ folder: st.folder, path: rel, text: 'saved anyway\n', baseHash: f.hash })
    vi.restoreAllMocks()
    expect(r.ok).toBe(true)
    expect(readFileSync(abs(st, rel), 'utf8')).toBe('saved anyway\n')
    expect(readdirSync(join(st.folder, 'manuscript')).filter((n) => n.includes('rulz-tmp'))).toEqual([])
    expect(readdirSync(join(st.folder, '.rulz', 'tmp'))).toEqual([])
  })
})

describe('outside changes that cannot be read', () => {
  it('says nothing when the read fails, and reports a delete only for a missing file', async () => {
    const s = setup()
    const st = await s.store.createProject('Locked')
    const rel = sceneRel(st)
    writeFileSync(abs(st, rel), 'text\n')
    await s.store.readFile(st.folder, rel)
    s.events.length = 0
    const real = fsp.readFile.bind(fsp)
    vi.spyOn(fsp, 'readFile').mockImplementation((async (path: Parameters<typeof fsp.readFile>[0], ...rest: unknown[]) => {
      if (at(String(path)).endsWith(rel)) throw Object.assign(new Error('io'), { code: 'EIO' })
      return (real as (...a: unknown[]) => Promise<unknown>)(path, ...rest)
    }) as never)
    await s.store.onExternalChange(st.folder, rel)
    vi.restoreAllMocks()
    expect(s.events).toEqual([])
    renameSync(abs(st, rel), join(st.folder, 'moved.txt'))
    await s.store.onExternalChange(st.folder, rel)
    expect(s.events.map((e) => e.kind)).toEqual(['deleted', 'state'])
  })
})

describe('trash items left by a crash', () => {
  it('writes the manifest before moving, and an interrupted item is listed and restorable', async () => {
    const s = setup()
    const st = await s.store.createProject('Crash')
    let cur = await s.store.outlinePatch(st.folder, [{ op: 'addChapter', chapterId: 'c-two2222', sceneId: 's-aaa1111', title: 'Two', index: null }])
    const file = cur.outline.chapters[1].scenes[0].file
    writeFileSync(abs(cur, `manuscript/${file}`), 'words\n')
    cur = await s.store.deleteChapter(st.folder, 'c-two2222')
    const [item] = await s.store.trash(st.folder)
    const manifestPath = join(st.folder, '.rulz', 'trash', item.id, 'manifest.json')
    const m = JSON.parse(readFileSync(manifestPath, 'utf8'))
    // As a crash would leave it: status "moving" and a planned file that never arrived.
    m.status = 'moving'
    m.files.push({ from: 'manuscript/ghost.md', to: `.rulz/trash/${item.id}/manuscript/ghost.md` })
    writeFileSync(manifestPath, JSON.stringify(m))
    expect((await s.store.trash(st.folder))[0].label).toMatch(/\(interrupted\)$/)
    cur = await s.store.restoreTrash(st.folder, item.id)
    expect(readFileSync(abs(cur, `manuscript/${file}`), 'utf8')).toBe('words\n')
    expect(existsSync(join(st.folder, 'manuscript', 'ghost.md'))).toBe(false)
  })

  it('a failed move leaves no trash item and the file where it was', async () => {
    const s = setup()
    const st = await s.store.createProject('Move fails')
    await s.store.createEntry(st.folder, 'X', 'lore')
    const real = fsp.rename.bind(fsp)
    vi.spyOn(fsp, 'rename').mockImplementation(async (from, to) => {
      if (at(String(to)).includes('/.rulz/trash/') && at(String(to)).endsWith('/x.md')) throw Object.assign(new Error('no'), { code: 'ENOSPC' })
      return real(from, to)
    })
    await expect(s.store.deleteEntry(st.folder, 'x')).rejects.toThrow()
    vi.restoreAllMocks()
    expect(existsSync(join(st.folder, 'bible', 'x.md'))).toBe(true)
    expect(await s.store.trash(st.folder)).toEqual([])
  })
})

void (null as unknown as Ctx)

describe('links inside .rulz', () => {
  function junction(target: string, path: string): boolean {
    try {
      symlinkSync(target, path, 'junction')
      return true
    } catch {
      return false
    }
  }

  async function project(s: Ctx, title: string) {
    const st = await s.store.createProject(title)
    const outside = join(s.base, 'outside')
    mkdirSync(outside)
    return { st, outside }
  }

  it('a junction at .rulz/history out of the project: save with snapshot is refused, nothing lands outside', async () => {
    const s = setup()
    const { st, outside } = await project(s, 'History link')
    const rel = sceneRel(st)
    writeFileSync(abs(st, rel), 'v1\n')
    if (!junction(outside, join(st.folder, '.rulz', 'history'))) return
    vi.spyOn(console, 'warn').mockImplementation(() => undefined)
    const f = await s.store.readFile(st.folder, rel)
    const r = await s.store.saveFile({ folder: st.folder, path: rel, text: 'v2\n', baseHash: f.hash, snapshot: true })
    expect(r).toMatchObject({ ok: false, reason: 'error', error: expect.stringMatching(/link|outside/) })
    expect(readFileSync(abs(st, rel), 'utf8')).toBe('v1\n')
    expect(readdirSync(outside)).toEqual([])
    await expect(s.store.history(st.folder, rel)).rejects.toThrow()
  })

  it('a junction at .rulz/unsaved: the journal is refused, nothing lands outside', async () => {
    const s = setup()
    const { st, outside } = await project(s, 'Unsaved link')
    if (!junction(outside, join(st.folder, '.rulz', 'unsaved'))) return
    await expect(s.store.journal(st.folder, sceneRel(st), 'typed\n', '')).rejects.toThrow(/link|outside/)
    expect(readdirSync(outside)).toEqual([])
  })

  it('a junction at .rulz/trash: deleting a scene is refused, the file stays, nothing lands outside', async () => {
    const s = setup()
    const { st, outside } = await project(s, 'Trash link')
    const cur = await s.store.outlinePatch(st.folder, [{ op: 'addScene', chapterId: st.outline.chapters[0].id, sceneId: 's-two2222', title: 'Two', index: null }])
    const two = cur.outline.chapters[0].scenes[1]
    writeFileSync(abs(cur, `manuscript/${two.file}`), 'two\n')
    if (!junction(outside, join(st.folder, '.rulz', 'trash'))) return
    await expect(s.store.deleteScene(st.folder, two.id)).rejects.toThrow(/link|outside/)
    expect(readFileSync(abs(cur, `manuscript/${two.file}`), 'utf8')).toBe('two\n')
    expect(readdirSync(outside)).toEqual([])
    expect((await s.store.openProject(st.folder)).outline.chapters[0].scenes).toHaveLength(2)
    await expect(s.store.emptyTrash(st.folder)).rejects.toThrow(/link|outside/)
  })

  it('refuses a .rulz subfolder that is a link even when it points inside the project', async () => {
    const s = setup()
    const { st } = await project(s, 'Inner link')
    mkdirSync(join(st.folder, 'elsewhere'))
    if (!junction(join(st.folder, 'elsewhere'), join(st.folder, '.rulz', 'unsaved'))) return
    await expect(s.store.journal(st.folder, sceneRel(st), 'typed\n', '')).rejects.toThrow(/link/)
    expect(readdirSync(join(st.folder, 'elsewhere'))).toEqual([])
  })

  it('a junction at .rulz/tmp stops every write', async () => {
    const s = setup()
    const { st, outside } = await project(s, 'Tmp link')
    const rel = sceneRel(st)
    writeFileSync(abs(st, rel), 'v1\n')
    const f = await s.store.readFile(st.folder, rel)
    rmSync(join(st.folder, '.rulz', 'tmp'), { recursive: true, force: true })
    if (!junction(outside, join(st.folder, '.rulz', 'tmp'))) return
    const r = await s.store.saveFile({ folder: st.folder, path: rel, text: 'v2\n', baseHash: f.hash })
    expect(r).toMatchObject({ ok: false, reason: 'error' })
    expect(readFileSync(abs(st, rel), 'utf8')).toBe('v1\n')
    expect(readdirSync(outside)).toEqual([])
  })
})

describe('links in a trashed journal path', () => {
  function junction(target: string, path: string): boolean {
    try {
      symlinkSync(target, path, 'junction')
      return true
    } catch {
      return false
    }
  }

  async function discarded(s: Ctx, title: string) {
    const st = await s.store.createProject(title)
    const rel = sceneRel(st)
    await s.store.journal(st.folder, rel, 'typed\n', '')
    await s.store.discardJournal(st.folder, rel)
    const [item] = await s.store.trash(st.folder)
    const outside = join(s.base, 'outside')
    mkdirSync(outside)
    return { st, item, outside, trashDir: join(st.folder, '.rulz', 'trash') }
  }

  const listing = (dir: string): string[] => readdirSync(dir, { recursive: true }).map(String).sort()

  it('a junction at .rulz/trash/<id>: restore, list and empty are refused, the outside dir is untouched', async () => {
    const s = setup()
    const t = await discarded(s, 'Item link')
    const itemDir = join(t.trashDir, t.item.id)
    const moved = join(t.outside, 'item')
    renameSync(itemDir, moved)
    if (!junction(moved, itemDir)) return
    const before = listing(t.outside)
    await expect(s.store.restoreTrash(t.st.folder, t.item.id)).rejects.toThrow(/link|outside|unreadable/)
    expect(await s.store.trash(t.st.folder)).toEqual([])
    await expect(s.store.emptyTrash(t.st.folder)).rejects.toThrow(/link/)
    expect(listing(t.outside)).toEqual(before)
    expect(existsSync(join(t.st.folder, '.rulz', 'unsaved', `${sceneRel(t.st).replace(/\//g, '__')}.json`))).toBe(false)
  })

  it('a junction at .rulz/trash: restore, list and empty are refused, the outside dir is untouched', async () => {
    const s = setup()
    const t = await discarded(s, 'Trash root link')
    renameSync(t.trashDir, join(t.outside, 'trash'))
    if (!junction(join(t.outside, 'trash'), t.trashDir)) return
    const before = listing(t.outside)
    await expect(s.store.restoreTrash(t.st.folder, t.item.id)).rejects.toThrow(/link|outside/)
    await expect(s.store.trash(t.st.folder)).rejects.toThrow(/link|outside/)
    await expect(s.store.emptyTrash(t.st.folder)).rejects.toThrow(/link|outside/)
    expect(listing(t.outside)).toEqual(before)
  })

  it('refuses a manifest "to" that goes into .rulz/history inside the item, and any second .rulz part but the journal layout', async () => {
    const s = setup()
    const t = await discarded(s, 'Inner rulz')
    const manifestPath = join(t.trashDir, t.item.id, 'manifest.json')
    const m = JSON.parse(readFileSync(manifestPath, 'utf8'))
    const from = m.files[0].from
    for (const to of [`.rulz/trash/${t.item.id}/.rulz/history/x`, `.rulz/trash/${t.item.id}/.rulz/unsaved/sub/x.json`, `.rulz/trash/${t.item.id}/a/.rulz/unsaved/x.json`]) {
      writeFileSync(manifestPath, JSON.stringify({ ...m, files: [{ from, to }] }))
      await expect(s.store.restoreTrash(t.st.folder, t.item.id)).rejects.toThrow(/damaged/)
    }
    writeFileSync(manifestPath, JSON.stringify(m))
    await s.store.restoreTrash(t.st.folder, t.item.id)
    expect(existsSync(join(t.st.folder, '.rulz', 'unsaved', `${sceneRel(t.st).replace(/\//g, '__')}.json`))).toBe(true)
  })
});
