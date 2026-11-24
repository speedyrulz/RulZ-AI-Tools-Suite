import { existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, renameSync, statSync, utimesSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'
import { DEFAULT_SAMPLING } from '@shared/defaults'
import type { NovelChange, NovelProjectState } from '@shared/novel'
import { defaultProject } from '@shared/novelPrompts'
import { decodeBytes, encodeText } from '../src/main/novel/files'
import { stampOf, thinSnapshots } from '../src/main/novel/history'
import { NovelStore } from '../src/main/novel/store'

const MIN = 60_000
const HOUR = 60 * MIN
const DAY = 24 * HOUR

function setup() {
  const base = mkdtempSync(join(tmpdir(), 'rulz-novel-'))
  const documents = join(base, 'Documents')
  const recycle = join(base, 'recycle')
  mkdirSync(recycle)
  const events: NovelChange[] = []
  const store = new NovelStore({
    rootSetting: () => '',
    documents: () => documents,
    // The Recycle Bin stand-in: moves the folder aside.
    trashItem: async (p) => renameSync(p, join(recycle, `${Date.now()}-${Math.random().toString(36).slice(2)}`)),
    emit: (c) => events.push(c),
    watch: false
  })
  return { base, documents, recycle, events, store }
}

async function newProject(s: ReturnType<typeof setup>, title = 'The Salt Road'): Promise<NovelProjectState> {
  return s.store.createProject(title)
}

const sceneRel = (st: NovelProjectState): string => `manuscript/${st.outline.chapters[0].scenes[0].file}`
const abs = (st: NovelProjectState, rel: string): string => join(st.folder, ...rel.split('/'))

describe('projects', () => {
  it('creates the default root and a project with every file and folder', async () => {
    const s = setup()
    const st = await newProject(s)
    expect(st.folder).toBe(join(s.documents, 'RulZ Novels', 'The Salt Road'))
    for (const p of ['project.json', 'outline.json', 'style.md', 'bible', 'manuscript', '.rulz']) expect(existsSync(join(st.folder, p))).toBe(true)
    expect(st.project.title).toBe('The Salt Road')
    expect(st.outline.chapters).toHaveLength(1)
    expect(st.outline.chapters[0]).toMatchObject({ title: 'Chapter 1' })
    expect(st.outline.chapters[0].scenes[0]).toMatchObject({ title: 'Scene 1' })
    expect(existsSync(abs(st, sceneRel(st)))).toBe(true)
    expect(st.blocked).toBeNull()
    expect(st.missing).toEqual([])
    expect(st.orphans).toEqual([])
  })

  it('gives each project its own unique folder name', async () => {
    const s = setup()
    const a = await newProject(s, 'CON')
    const b = await newProject(s, 'CON')
    const c = await newProject(s, 'What? A: title/')
    expect(new Set([a.folder, b.folder, c.folder]).size).toBe(3)
    expect(a.folder.endsWith('CON novel')).toBe(true)
    expect(c.folder.endsWith('What A title')).toBe(true)
  })

  it('lists projects, and lists one with a broken project.json by its folder name', async () => {
    const s = setup()
    const st = await newProject(s)
    await s.store.closeProject(st.folder)
    mkdirSync(join(s.documents, 'RulZ Novels', 'Broken one'))
    writeFileSync(join(s.documents, 'RulZ Novels', 'Broken one', 'project.json'), '{ nope')
    mkdirSync(join(s.documents, 'RulZ Novels', 'Not a project'))
    const list = await s.store.listProjects()
    expect(list.map((p) => p.title).sort()).toEqual(['Broken one', 'The Salt Road'])
    expect(list.find((p) => p.title === 'Broken one')?.id).toBe('')
    const open = await s.store.openProject(join(s.documents, 'RulZ Novels', 'Broken one'))
    expect(open.blocked).toMatch(/project\.json could not be read/)
  })
})

describe('bytes and encoding', () => {
  it('decodes and re-encodes BOM + CRLF exactly', () => {
    const bytes = Buffer.concat([Buffer.from([0xef, 0xbb, 0xbf]), Buffer.from('one\r\ntwo\r\nthree\r\n', 'utf8')])
    const d = decodeBytes(bytes)
    expect(d).toMatchObject({ text: 'one\ntwo\nthree\n', meta: { bom: true, eol: 'crlf' }, readOnly: false })
    expect(encodeText(d.text, d.meta).equals(bytes)).toBe(true)
    expect(decodeBytes(Buffer.from('no breaks')).meta).toEqual({ bom: false, eol: 'lf' })
    expect(decodeBytes(Buffer.from('a\nb\nc\r\nd', 'utf8')).meta.eol).toBe('lf')
  })

  it('round-trips a CRLF + BOM scene through an edit; only the edit changes', async () => {
    const s = setup()
    const st = await newProject(s)
    const rel = sceneRel(st)
    const original = Buffer.concat([Buffer.from([0xef, 0xbb, 0xbf]), Buffer.from('The ferry left.\r\n\r\nNobody waved.\r\n', 'utf8')])
    writeFileSync(abs(st, rel), original)
    const f = await s.store.readFile(st.folder, rel)
    expect(f.text).toBe('The ferry left.\n\nNobody waved.\n')
    expect(f.meta).toEqual({ bom: true, eol: 'crlf' })
    const r = await s.store.saveFile({ folder: st.folder, path: rel, text: f.text.replace('left', 'sailed'), baseHash: f.hash })
    expect(r.ok).toBe(true)
    const after = readFileSync(abs(st, rel))
    expect(after.equals(Buffer.concat([Buffer.from([0xef, 0xbb, 0xbf]), Buffer.from('The ferry sailed.\r\n\r\nNobody waved.\r\n', 'utf8')]))).toBe(true)
    expect(r.ok && r.hash).toBe(decodeBytes(after).hash)
  })

  it('does not write a document that was not edited', async () => {
    const s = setup()
    const st = await newProject(s)
    const rel = sceneRel(st)
    writeFileSync(abs(st, rel), 'Same text.\r\n')
    const old = new Date('2020-01-01T00:00:00Z')
    utimesSync(abs(st, rel), old, old)
    const f = await s.store.readFile(st.folder, rel)
    const r = await s.store.saveFile({ folder: st.folder, path: rel, text: f.text, baseHash: f.hash })
    expect(r).toEqual({ ok: true, hash: f.hash })
    expect(statSync(abs(st, rel)).mtimeMs).toBe(old.getTime())
  })

  it('opens a file that is not valid UTF-8 read-only and refuses to save it', async () => {
    const s = setup()
    const st = await newProject(s)
    const rel = sceneRel(st)
    const bad = Buffer.from([0x68, 0x69, 0xff, 0xfe, 0x0a])
    writeFileSync(abs(st, rel), bad)
    const f = await s.store.readFile(st.folder, rel)
    expect(f.readOnly).toBe(true)
    expect(await s.store.saveFile({ folder: st.folder, path: rel, text: 'x', baseHash: f.hash, force: true })).toEqual({ ok: false, reason: 'readOnly' })
    expect(readFileSync(abs(st, rel)).equals(bad)).toBe(true)
  })

  it('writes new files with LF and no BOM, and leaves nothing in the temp folder', async () => {
    const s = setup()
    const st = await newProject(s)
    const r = await s.store.saveFile({ folder: st.folder, path: 'style.md', text: 'Close third.\nPast tense.\n', baseHash: (await s.store.readFile(st.folder, 'style.md')).hash })
    expect(r.ok).toBe(true)
    expect(readFileSync(join(st.folder, 'style.md'), 'utf8')).toBe('Close third.\nPast tense.\n')
    expect(readdirSync(join(st.folder, '.rulz', 'tmp'))).toEqual([])
  })
})

describe('conflicts and history', () => {
  it('refuses a save when the file changed on disk, and Keep mine snapshots the disk version first', async () => {
    const s = setup()
    const st = await newProject(s)
    const rel = sceneRel(st)
    writeFileSync(abs(st, rel), 'v1\n')
    const f = await s.store.readFile(st.folder, rel)
    writeFileSync(abs(st, rel), 'edited elsewhere\n')
    const c = await s.store.saveFile({ folder: st.folder, path: rel, text: 'mine\n', baseHash: f.hash })
    expect(c).toMatchObject({ ok: false, reason: 'conflict', diskText: 'edited elsewhere\n' })
    expect(readFileSync(abs(st, rel), 'utf8')).toBe('edited elsewhere\n')
    const diskHash = c.ok === false && c.reason === 'conflict' ? c.diskHash : ''
    const before = await s.store.history(st.folder, rel)
    const k = await s.store.saveFile({ folder: st.folder, path: rel, text: 'mine\n', baseHash: diskHash, force: true })
    expect(k.ok).toBe(true)
    expect(readFileSync(abs(st, rel), 'utf8')).toBe('mine\n')
    const after = await s.store.history(st.folder, rel)
    expect(after.length).toBeGreaterThan(before.length)
    const texts = await Promise.all(after.map((h) => s.store.readSnapshot(st.folder, h.snapshot)))
    expect(texts).toContain('edited elsewhere\n')
    expect(texts).toContain('v1\n')
  })

  it('treats a new file that appeared on disk as a conflict', async () => {
    const s = setup()
    const st = await newProject(s)
    writeFileSync(join(st.folder, 'bible', 'ghost.md'), 'there already\n')
    const r = await s.store.saveFile({ folder: st.folder, path: 'bible/ghost.md', text: 'new\n', baseHash: '' })
    expect(r).toMatchObject({ ok: false, reason: 'conflict', diskText: 'there already\n' })
  })

  it('snapshots on first read, on request, and throttles the rest', async () => {
    const s = setup()
    const st = await newProject(s)
    const rel = sceneRel(st)
    writeFileSync(abs(st, rel), 'first\n')
    let f = await s.store.readFile(st.folder, rel)
    expect((await s.store.history(st.folder, rel)).map((h) => h.snapshot.startsWith(`.rulz/history/${rel}/`))).toEqual([true])
    await s.store.readFile(st.folder, rel)
    expect(await s.store.history(st.folder, rel)).toHaveLength(1)
    // A plain save inside the 10-minute gap needs no new snapshot of the same content; a requested one does.
    let r = await s.store.saveFile({ folder: st.folder, path: rel, text: 'second\n', baseHash: f.hash })
    expect(r.ok).toBe(true)
    f = await s.store.readFile(st.folder, rel)
    r = await s.store.saveFile({ folder: st.folder, path: rel, text: 'third\n', baseHash: f.hash, snapshot: true })
    expect(r.ok).toBe(true)
    const items = await s.store.history(st.folder, rel)
    const texts = await Promise.all(items.map((h) => s.store.readSnapshot(st.folder, h.snapshot)))
    expect(texts).toContain('second\n')
    expect(texts).toContain('first\n')
    expect(texts).not.toContain('third\n')
    expect(items.map((i) => i.stamp)).toEqual([...items.map((i) => i.stamp)].sort().reverse())
  })

  it('keeps two snapshots taken in the same second apart', async () => {
    const s = setup()
    const st = await newProject(s)
    const rel = sceneRel(st)
    writeFileSync(abs(st, rel), 'a\n')
    let f = await s.store.readFile(st.folder, rel)
    for (const t of ['b\n', 'c\n', 'd\n']) {
      const r = await s.store.saveFile({ folder: st.folder, path: rel, text: t, baseHash: f.hash, snapshot: true })
      expect(r.ok).toBe(true)
      f = await s.store.readFile(st.folder, rel)
    }
    const items = await s.store.history(st.folder, rel)
    // a (first read), b, c: the snapshot before the last save would repeat what the newest one holds only if unchanged.
    expect(items).toHaveLength(3)
    expect(new Set(items.map((i) => i.stamp)).size).toBe(3)
  })
})

describe('thinSnapshots', () => {
  const now = new Date('2026-11-22T12:00:00Z')
  const at = (ago: number): string => stampOf(new Date(now.getTime() - ago))

  it('keeps everything from the last hour', () => {
    const stamps = [0, 1, 5, 20, 40, 59].map((m) => at(m * MIN))
    expect(thinSnapshots(stamps, now)).toEqual([...stamps].sort())
  })

  it('keeps one per 10 minutes for a day, one per hour for a week, one per day after', () => {
    const day = [71, 72, 75, 95, 99, 130].map((m) => at(m * MIN))
    const keptDay = thinSnapshots(day, now)
    // Slots are clock-aligned: 71/72/75 min ago share one, 95/99 another, 130 a third.
    expect(keptDay).toHaveLength(3)
    const week = [at(3 * DAY + 5 * MIN), at(3 * DAY + 20 * MIN), at(3 * DAY + 2 * HOUR)]
    expect(thinSnapshots(week, now)).toHaveLength(2)
    const old = [at(20 * DAY), at(20 * DAY + 10 * MIN), at(20 * DAY + 3 * HOUR), at(21 * DAY + 3 * HOUR)]
    expect(thinSnapshots(old, now).length).toBeLessThanOrEqual(2)
    expect(thinSnapshots(old, now).length).toBeGreaterThanOrEqual(1)
  })

  it('keeps the newest of a slot and is stable when run twice', () => {
    const stamps = [at(81 * MIN), at(82 * MIN), at(83 * MIN)]
    const kept = thinSnapshots(stamps, now)
    expect(kept).toEqual([at(81 * MIN)])
    expect(thinSnapshots(kept, now)).toEqual(kept)
  })

  it('caps at 200, dropping the oldest, and leaves unknown names alone', () => {
    const stamps = Array.from({ length: 250 }, (_, i) => stampOf(new Date(now.getTime() - i * 1000)))
    const kept = thinSnapshots(stamps, now)
    expect(kept).toHaveLength(200)
    expect(kept).toContain(stamps[0])
    expect(kept).not.toContain(stamps[249])
    expect(thinSnapshots(['odd-name'], now)).toEqual(['odd-name'])
  })

  it('parses a collision-suffixed stamp as the same second', () => {
    const base = at(5 * MIN)
    expect(thinSnapshots([base, `${base}-2`], now)).toEqual([base, `${base}-2`])
  })
})

describe('journal', () => {
  it('offers a journal whatever the file time is, and a save deletes it', async () => {
    const s = setup()
    const st = await newProject(s)
    const rel = sceneRel(st)
    writeFileSync(abs(st, rel), 'saved text\n')
    let f = await s.store.readFile(st.folder, rel)
    await s.store.journal(st.folder, rel, 'unsaved text\n', f.hash)
    const journalFile = join(st.folder, '.rulz', 'unsaved', `${rel.replace(/\//g, '__')}.json`)
    expect(existsSync(journalFile)).toBe(true)
    // Journal is newer than the file: offered.
    const past = new Date(Date.now() - 60_000)
    utimesSync(abs(st, rel), past, past)
    f = await s.store.readFile(st.folder, rel)
    expect(f.journal).toMatchObject({ text: 'unsaved text\n', baseHash: f.hash })
    // File touched after the journal (sync clients do this): still offered, text differs from the file.
    const future = new Date(Date.now() + 60_000)
    utimesSync(abs(st, rel), future, future)
    f = await s.store.readFile(st.folder, rel)
    expect(f.journal).toMatchObject({ text: 'unsaved text\n' })
    expect(existsSync(journalFile)).toBe(true)
    // A successful save removes it.
    const r = await s.store.saveFile({ folder: st.folder, path: rel, text: 'unsaved text\n', baseHash: f.hash })
    expect(r.ok).toBe(true)
    expect(existsSync(journalFile)).toBe(false)
  })

  it('discarding a journal moves it to the trash', async () => {
    const s = setup()
    const st = await newProject(s)
    const rel = sceneRel(st)
    await s.store.journal(st.folder, rel, 'lost?\n', '')
    await s.store.discardJournal(st.folder, rel)
    expect(readdirSync(join(st.folder, '.rulz', 'unsaved'))).toEqual([])
    const trash = await s.store.trash(st.folder)
    expect(trash).toHaveLength(1)
    expect(trash[0].label).toContain(rel)
  })

  it('does not offer a journal identical to the file', async () => {
    const s = setup()
    const st = await newProject(s)
    const rel = sceneRel(st)
    writeFileSync(abs(st, rel), 'same\n')
    await s.store.journal(st.folder, rel, 'same\n', '')
    expect((await s.store.readFile(st.folder, rel)).journal).toBeNull()
  })
})

describe('blocked projects', () => {
  it('opens read-only when outline.json is broken and never writes anything', async () => {
    const s = setup()
    const st = await newProject(s)
    await s.store.closeProject(st.folder)
    const outlinePath = join(st.folder, 'outline.json')
    const broken = Buffer.from('{ "format": 1, "chapters": [ oops')
    writeFileSync(outlinePath, broken)
    const projectBefore = readFileSync(join(st.folder, 'project.json'))
    const filesBefore = readdirSync(join(st.folder, 'manuscript'))
    const open = await s.store.openProject(st.folder)
    expect(open.blocked).toMatch(/outline\.json could not be read/)
    await expect(s.store.outlinePatch(st.folder, [{ op: 'addChapter', chapterId: 'c-x', sceneId: 's-x', title: 'X', index: null }])).rejects.toThrow(/outline\.json could not be read/)
    await expect(s.store.projectPatch(st.folder, { title: 'New' })).rejects.toThrow(/read-only/)
    await expect(s.store.deleteChapter(st.folder, 'whatever')).rejects.toThrow()
    await expect(s.store.createEntry(st.folder, 'Mara', 'character')).rejects.toThrow()
    expect(await s.store.saveFile({ folder: st.folder, path: 'style.md', text: 'x', baseHash: '' })).toEqual({ ok: false, reason: 'blocked' })
    await s.store.journal(st.folder, 'style.md', 'x', '')
    expect(readFileSync(outlinePath).equals(broken)).toBe(true)
    expect(readFileSync(join(st.folder, 'project.json')).equals(projectBefore)).toBe(true)
    expect(readdirSync(join(st.folder, 'manuscript'))).toEqual(filesBefore)
    expect(existsSync(join(st.folder, '.rulz', 'unsaved'))).toBe(false)
    // No fallback copy was made either.
    expect(readdirSync(st.folder).filter((n) => n.includes('corrupt'))).toEqual([])
  })

  it('unblocks when the file is fixed outside the app', async () => {
    const s = setup()
    const st = await newProject(s)
    const good = readFileSync(join(st.folder, 'outline.json'))
    writeFileSync(join(st.folder, 'outline.json'), 'not json')
    await s.store.onExternalChange(st.folder, 'outline.json')
    expect(s.events.at(-1)).toMatchObject({ kind: 'state', state: { blocked: expect.stringMatching(/outline\.json/) } })
    writeFileSync(join(st.folder, 'outline.json'), good)
    await s.store.onExternalChange(st.folder, 'outline.json')
    expect(s.events.at(-1)).toMatchObject({ kind: 'state', state: { blocked: null } })
  })

  it('blocks on a scene file name that could leave manuscript\\', async () => {
    const s = setup()
    const st = await newProject(s)
    await s.store.closeProject(st.folder)
    const o = JSON.parse(readFileSync(join(st.folder, 'outline.json'), 'utf8'))
    o.chapters[0].scenes[0].file = '../../evil.md'
    writeFileSync(join(st.folder, 'outline.json'), JSON.stringify(o))
    expect((await s.store.openProject(st.folder)).blocked).toMatch(/no valid file name/)
  })

  it('keeps unknown keys in project.json and outline.json when it writes them', async () => {
    const s = setup()
    const st = await newProject(s)
    await s.store.closeProject(st.folder)
    const pj = JSON.parse(readFileSync(join(st.folder, 'project.json'), 'utf8'))
    pj.futureField = { a: 1 }
    writeFileSync(join(st.folder, 'project.json'), JSON.stringify(pj))
    const oj = JSON.parse(readFileSync(join(st.folder, 'outline.json'), 'utf8'))
    oj.chapters[0].mood = 'dark'
    writeFileSync(join(st.folder, 'outline.json'), JSON.stringify(oj))
    await s.store.openProject(st.folder)
    await s.store.projectPatch(st.folder, { title: 'Renamed' })
    await s.store.outlinePatch(st.folder, [{ op: 'setChapter', chapterId: oj.chapters[0].id, field: 'title', value: 'Opening' }])
    const pj2 = JSON.parse(readFileSync(join(st.folder, 'project.json'), 'utf8'))
    const oj2 = JSON.parse(readFileSync(join(st.folder, 'outline.json'), 'utf8'))
    expect(pj2).toMatchObject({ futureField: { a: 1 }, title: 'Renamed' })
    expect(oj2.chapters[0]).toMatchObject({ mood: 'dark', title: 'Opening' })
  })
})

describe('outline, trash and restore', () => {
  it('applies patches, creates scene files, and reports orphans and missing files', async () => {
    const s = setup()
    const st = await newProject(s)
    const chapterId = st.outline.chapters[0].id
    const next = await s.store.outlinePatch(st.folder, [
      { op: 'addScene', chapterId, sceneId: 's-new111', title: 'Night crossing', index: null, beats: 'Mara bribes the ferryman' }
    ])
    const added = next.outline.chapters[0].scenes[1]
    expect(added).toMatchObject({ id: 's-new111', title: 'Night crossing', status: 'beats' })
    expect(existsSync(abs(next, `manuscript/${added.file}`))).toBe(true)
    expect(JSON.parse(readFileSync(join(st.folder, 'outline.json'), 'utf8')).chapters[0].scenes).toHaveLength(2)
    writeFileSync(join(st.folder, 'manuscript', 'stray.md'), 'one two three four\n')
    renameSync(abs(next, `manuscript/${added.file}`), join(st.folder, 'manuscript', 'moved-away.txt'))
    const again = await s.store.openProject(st.folder)
    expect(again.orphans).toEqual(['manuscript/stray.md'])
    expect(again.missing).toEqual([`manuscript/${added.file}`])
    expect(again.words['manuscript/stray.md']).toBe(4)
  })

  it('never overwrites an existing file when a new scene gets the same name', async () => {
    const s = setup()
    const st = await newProject(s)
    const clash = join(st.folder, 'manuscript', '02-scene-1-1111.md')
    writeFileSync(clash, 'precious\n')
    await s.store.outlinePatch(st.folder, [{ op: 'addScene', chapterId: st.outline.chapters[0].id, sceneId: 's-aaa1111', title: 'Scene 1', index: null }])
    expect(readFileSync(clash, 'utf8')).toBe('precious\n')
  })

  it('refuses to delete the last scene of a chapter', async () => {
    const s = setup()
    const st = await newProject(s)
    await expect(s.store.deleteScene(st.folder, st.outline.chapters[0].scenes[0].id)).rejects.toThrow(/Delete the chapter instead/)
    expect(existsSync(abs(st, sceneRel(st)))).toBe(true)
  })

  it('deletes a scene to the trash and restores it at its old position', async () => {
    const s = setup()
    const st = await newProject(s)
    const chapterId = st.outline.chapters[0].id
    let cur = await s.store.outlinePatch(st.folder, [
      { op: 'addScene', chapterId, sceneId: 's-two2222', title: 'Two', index: null },
      { op: 'addScene', chapterId, sceneId: 's-thr3333', title: 'Three', index: null }
    ])
    const two = cur.outline.chapters[0].scenes[1]
    writeFileSync(abs(cur, `manuscript/${two.file}`), 'text of two\n')
    cur = await s.store.deleteScene(st.folder, two.id)
    expect(cur.outline.chapters[0].scenes.map((x) => x.id)).not.toContain('s-two2222')
    expect(existsSync(abs(cur, `manuscript/${two.file}`))).toBe(false)
    const items = await s.store.trash(st.folder)
    expect(items).toHaveLength(1)
    expect(items[0].label).toBe('Scene "Two"')
    cur = await s.store.restoreTrash(st.folder, items[0].id)
    expect(cur.outline.chapters[0].scenes.map((x) => x.id)).toEqual([st.outline.chapters[0].scenes[0].id, 's-two2222', 's-thr3333'])
    expect(readFileSync(abs(cur, `manuscript/${two.file}`), 'utf8')).toBe('text of two\n')
    expect(await s.store.trash(st.folder)).toEqual([])
  })

  it('deletes a chapter with all its files in one trash folder and restores it', async () => {
    const s = setup()
    const st = await newProject(s)
    let cur = await s.store.outlinePatch(st.folder, [
      { op: 'addChapter', chapterId: 'c-two2222', sceneId: 's-aaa1111', title: 'The ferry', index: null },
      { op: 'addScene', chapterId: 'c-two2222', sceneId: 's-bbb2222', title: 'Landing', index: null }
    ])
    const chapter = cur.outline.chapters[1]
    for (const sc of chapter.scenes) writeFileSync(abs(cur, `manuscript/${sc.file}`), `words in ${sc.title}\n`)
    cur = await s.store.deleteChapter(st.folder, 'c-two2222')
    expect(cur.outline.chapters.map((c) => c.id)).toEqual([st.outline.chapters[0].id])
    for (const sc of chapter.scenes) expect(existsSync(abs(cur, `manuscript/${sc.file}`))).toBe(false)
    const items = await s.store.trash(st.folder)
    expect(items).toHaveLength(1)
    expect(items[0].label).toBe('Chapter "The ferry" (2 scenes)')
    expect(items[0].files).toHaveLength(2)
    cur = await s.store.restoreTrash(st.folder, items[0].id)
    expect(cur.outline.chapters.map((c) => c.id)).toEqual([st.outline.chapters[0].id, 'c-two2222'])
    expect(cur.outline.chapters[1].scenes.map((x) => x.id)).toEqual(['s-aaa1111', 's-bbb2222'])
    for (const sc of chapter.scenes) expect(readFileSync(abs(cur, `manuscript/${sc.file}`), 'utf8')).toBe(`words in ${sc.title}\n`)
    expect(readdirSync(join(st.folder, '.rulz', 'trash'))).toEqual([])
  })

  it('refuses to restore over an existing file and moves nothing', async () => {
    const s = setup()
    const st = await newProject(s)
    let cur = await s.store.outlinePatch(st.folder, [{ op: 'addChapter', chapterId: 'c-two2222', sceneId: 's-aaa1111', title: 'Two', index: null }])
    const file = cur.outline.chapters[1].scenes[0].file
    writeFileSync(abs(cur, `manuscript/${file}`), 'original\n')
    cur = await s.store.deleteChapter(st.folder, 'c-two2222')
    writeFileSync(abs(cur, `manuscript/${file}`), 'someone else\n')
    const [item] = await s.store.trash(st.folder)
    await expect(s.store.restoreTrash(st.folder, item.id)).rejects.toThrow(/already exists/)
    expect(readFileSync(abs(cur, `manuscript/${file}`), 'utf8')).toBe('someone else\n')
    expect(await s.store.trash(st.folder)).toHaveLength(1)
    expect((await s.store.openProject(st.folder)).outline.chapters).toHaveLength(1)
  })

  it('creates bible entries with unique ids, deletes and restores them, and parses them into the state', async () => {
    const s = setup()
    const st = await newProject(s)
    const a = await s.store.createEntry(st.folder, 'Mara Venn', 'character')
    const b = await s.store.createEntry(st.folder, 'Mara Venn', 'place')
    expect([a.id, b.id]).toEqual(['mara-venn', 'mara-venn-2'])
    expect(a).toMatchObject({ name: 'Mara Venn', kind: 'character', problems: [] })
    let cur = await s.store.openProject(st.folder)
    expect(cur.bible.map((e) => e.id).sort()).toEqual(['mara-venn', 'mara-venn-2'])
    cur = await s.store.deleteEntry(st.folder, 'mara-venn')
    expect(cur.bible.map((e) => e.id)).toEqual(['mara-venn-2'])
    const [item] = await s.store.trash(st.folder)
    cur = await s.store.restoreTrash(st.folder, item.id)
    expect(cur.bible.map((e) => e.id).sort()).toEqual(['mara-venn', 'mara-venn-2'])
  })

  it('empties the trash through trashItem and deletes a project through it', async () => {
    const s = setup()
    const st = await newProject(s)
    await s.store.createEntry(st.folder, 'X', 'lore')
    await s.store.deleteEntry(st.folder, 'x')
    await s.store.emptyTrash(st.folder)
    expect(existsSync(join(st.folder, '.rulz', 'trash'))).toBe(false)
    expect(readdirSync(s.recycle)).toHaveLength(1)
    await s.store.deleteProject(st.folder)
    expect(existsSync(st.folder)).toBe(false)
    expect(s.store.hasOpenProject()).toBe(false)
    await expect(s.store.readFile(st.folder, 'style.md')).rejects.toThrow(/not open/)
  })

  it('deletes nothing when moving the project folder fails', async () => {
    const base = mkdtempSync(join(tmpdir(), 'rulz-novel-'))
    const store = new NovelStore({
      rootSetting: () => '',
      documents: () => base,
      trashItem: async () => {
        throw new Error('locked')
      },
      emit: () => undefined,
      watch: false
    })
    const st = await store.createProject('Keep me')
    await expect(store.deleteProject(st.folder)).rejects.toThrow(/Nothing was deleted/)
    expect(existsSync(join(st.folder, 'project.json'))).toBe(true)
    expect(store.hasOpenProject()).toBe(true)
  })

  it('refuses to delete a folder outside the projects root', async () => {
    const s = setup()
    await newProject(s)
    const other = join(s.base, 'elsewhere')
    mkdirSync(other)
    writeFileSync(join(other, 'project.json'), '{}')
    await expect(s.store.deleteProject(other)).rejects.toThrow(/Only a project folder/)
    await expect(s.store.deleteProject(join(s.documents, 'RulZ Novels'))).rejects.toThrow(/Only a project folder/)
  })
})

describe('paths from the renderer', () => {
  it('refuses anything that escapes the project or points into .rulz', async () => {
    const s = setup()
    const st = await newProject(s)
    const bad = ['../x.md', '..\\x.md', 'manuscript/../../x.md', 'C:\\Windows\\win.ini', 'C:/x.md', '/etc/passwd', '\\\\server\\share\\x.md', '.rulz/trash/x.md', '.RULZ/history/a.md', 'manuscript/a.md:stream', 'project.json', 'outline.json', 'manuscript/', 'manuscript//a.md', 'notes.txt', '']
    for (const path of bad) {
      await expect(s.store.readFile(st.folder, path), `read ${path}`).rejects.toThrow()
      await expect(s.store.saveFile({ folder: st.folder, path, text: 'x', baseHash: '' }), `save ${path}`).rejects.toThrow()
      await expect(s.store.journal(st.folder, path, 'x', ''), `journal ${path}`).rejects.toThrow()
    }
    await expect(s.store.readSnapshot(st.folder, '../project.json')).rejects.toThrow()
    await expect(s.store.readSnapshot(st.folder, 'manuscript/x.md')).rejects.toThrow(/not a history snapshot/)
    await expect(s.store.readSnapshot(st.folder, '.rulz/trash/a/manifest.json')).rejects.toThrow()
    await expect(s.store.history(st.folder, '../x.md')).rejects.toThrow()
    await expect(s.store.deleteEntry(st.folder, '../../project')).rejects.toThrow()
    await expect(s.store.restoreTrash(st.folder, '../x')).rejects.toThrow()
    expect(existsSync(join(st.folder, '..', 'x.md'))).toBe(false)
  })

  it('refuses a folder that is not an open project', async () => {
    const s = setup()
    const st = await newProject(s)
    await expect(s.store.readFile(join(s.base, 'other'), 'style.md')).rejects.toThrow(/not open/)
    await s.store.closeProject(st.folder)
    await expect(s.store.saveFile({ folder: st.folder, path: 'style.md', text: 'x', baseHash: '' })).rejects.toThrow(/not open/)
  })

  it('allows history of outline.json and project.json', async () => {
    const s = setup()
    const st = await newProject(s)
    await s.store.projectPatch(st.folder, { title: 'A' })
    await s.store.projectPatch(st.folder, { title: 'B' })
    const items = await s.store.history(st.folder, 'project.json')
    expect(items.length).toBeGreaterThanOrEqual(1)
    expect(await s.store.readSnapshot(st.folder, items[0].snapshot)).toContain('"title"')
  })
})

describe('changes made outside the app', () => {
  it('reports a changed or deleted file once, and not the app\'s own writes', async () => {
    const s = setup()
    const st = await newProject(s)
    const rel = sceneRel(st)
    writeFileSync(abs(st, rel), 'v1\n')
    const f = await s.store.readFile(st.folder, rel)
    s.events.length = 0
    await s.store.saveFile({ folder: st.folder, path: rel, text: 'v2\n', baseHash: f.hash })
    // The save itself is announced once (so phones hear of it); the watcher's echo of it adds nothing.
    const own = [{ kind: 'text', folder: st.folder, path: rel, hash: decodeBytes(readFileSync(abs(st, rel))).hash }]
    expect(s.events).toEqual(own)
    await s.store.onExternalChange(st.folder, rel)
    expect(s.events).toEqual(own)
    s.events.length = 0
    writeFileSync(abs(st, rel), 'v3 from Obsidian\n')
    await s.store.onExternalChange(st.folder, rel)
    await s.store.onExternalChange(st.folder, rel)
    expect(s.events.filter((e) => e.kind === 'text')).toEqual([{ kind: 'text', folder: st.folder, path: rel, hash: decodeBytes(readFileSync(abs(st, rel))).hash }])
    expect(s.events.filter((e) => e.kind === 'state')).toHaveLength(1)
    s.events.length = 0
    renameSync(abs(st, rel), join(st.folder, 'gone.txt'))
    await s.store.onExternalChange(st.folder, rel)
    expect(s.events.map((e) => e.kind)).toEqual(['deleted', 'state'])
    expect(s.events[1]).toMatchObject({ state: { missing: [rel] } })
  })

  it('sends the text and the new state for an edited style.md or bible entry', async () => {
    const s = setup()
    const st = await newProject(s)
    await s.store.createEntry(st.folder, 'Mara', 'character')
    await s.store.openProject(st.folder)
    s.events.length = 0
    writeFileSync(join(st.folder, 'style.md'), 'New voice\n')
    await s.store.onExternalChange(st.folder, 'style.md')
    expect(s.events.map((e) => e.kind)).toEqual(['text', 'state'])
    expect(s.events[1]).toMatchObject({ state: { style: { text: 'New voice\n' } } })
    s.events.length = 0
    writeFileSync(join(st.folder, 'bible', 'mara.md'), '---\nname: Mara Venn\ntriggers: [witch]\n---\nChanged\n')
    await s.store.onExternalChange(st.folder, 'bible/mara.md')
    expect(s.events.map((e) => e.kind)).toEqual(['text', 'state'])
    expect(s.events[1]).toMatchObject({ state: { bible: [{ id: 'mara', name: 'Mara Venn', triggers: ['witch'] }] } })
  })

  it('ignores .rulz, temp files and other extensions', async () => {
    const s = setup()
    const st = await newProject(s)
    s.events.length = 0
    writeFileSync(join(st.folder, 'manuscript', 'notes.txt'), 'x')
    await s.store.onExternalChange(st.folder, 'manuscript/notes.txt')
    await s.store.onExternalChange(st.folder, '.rulz/history/x.md')
    await s.store.onExternalChange(st.folder, 'manuscript')
    expect(s.events).toEqual([])
  })
})

describe('the sampling of a new project', () => {
  const own = { ...DEFAULT_SAMPLING, temperature: 1.7, topK: 11, jsonSchema: '' }

  function withSampling(newProjectSampling?: () => typeof own | null) {
    const base = mkdtempSync(join(tmpdir(), 'rulz-novel-'))
    return new NovelStore({ rootSetting: () => '', documents: () => join(base, 'Documents'), trashItem: async () => undefined, emit: () => undefined, watch: false, newProjectSampling })
  }

  it('is the fiction preset without a start sampling, or when it gives none', async () => {
    const a = await withSampling().createProject('A')
    const b = await withSampling(() => null).createProject('B')
    expect(a.project.sampling).toEqual(defaultProject('A').sampling)
    expect(b.project.sampling).toEqual(a.project.sampling)
  })

  it("is the model's sampling when one is given, saved with the project, and as a copy", async () => {
    let shared = own
    const store = withSampling(() => shared)
    const st = await store.createProject('Own')
    expect(st.project.sampling).toEqual(own)
    shared.stop.push('X')
    expect(st.project.sampling.stop).not.toContain('X')
    const reopened = await store.openProject(st.folder)
    expect(reopened.project.sampling.topK).toBe(11)
    shared = { ...own, topK: 12 }
    expect((await store.createProject('Second')).project.sampling.topK).toBe(12)
  })

  it('a start sampling that throws never stops the project from being created', async () => {
    const st = await withSampling(() => {
      throw new Error('boom')
    }).createProject('Safe')
    expect(st.project.sampling).toEqual(defaultProject('Safe').sampling)
  })
})
