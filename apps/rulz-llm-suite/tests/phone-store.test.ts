import { mkdirSync, mkdtempSync, readdirSync, readFileSync, renameSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'
import type { NovelChange, NovelProjectState } from '@shared/novel'
import { decodeBytes } from '../src/main/novel/files'
import { NovelStore, runAs, WINDOW_OWNER } from '../src/main/novel/store'

// Phone access, store side: saves are announced to every listener, and open projects are counted by owner.

function setup(watch = false) {
  const base = mkdtempSync(join(tmpdir(), 'rulz-phone-'))
  const recycle = join(base, 'recycle')
  mkdirSync(recycle)
  const events: { change: NovelChange; origin: string | undefined }[] = []
  const store = new NovelStore({
    rootSetting: () => '',
    documents: () => join(base, 'Documents'),
    trashItem: async (p) => renameSync(p, join(recycle, `${Date.now()}-${Math.random().toString(36).slice(2)}`)),
    emit: (change, origin) => events.push({ change, origin }),
    watch
  })
  const kinds = (): string[] => events.map((e) => e.change.kind)
  return { store, events, kinds }
}

const sceneRel = (st: NovelProjectState): string => `manuscript/${st.outline.chapters[0].scenes[0].file}`
const sceneId = (st: NovelProjectState): string => st.outline.chapters[0].scenes[0].id

describe('saves are announced', () => {
  it('announces a saved scene with its new hash, tagged with who asked', async () => {
    const s = setup()
    const st = await s.store.createProject('Book')
    const rel = sceneRel(st)
    const f = await s.store.readFile(st.folder, rel)
    s.events.length = 0
    const r = await runAs('phone:1', () => s.store.saveFile({ folder: st.folder, path: rel, text: 'Hello there.\n', baseHash: f.hash }))
    expect(r.ok).toBe(true)
    const hash = decodeBytes(readFileSync(join(st.folder, ...rel.split('/')))).hash
    // The text first, then the state that follows from it (word counts), both as the phone's change.
    expect(s.events[0]).toEqual({ change: { kind: 'text', folder: st.folder, path: rel, hash }, origin: 'phone:1' })
    expect(s.kinds()).toEqual(['text', 'state'])
    const count = s.events.length
    // The watcher's echo of the same write adds nothing.
    await s.store.onExternalChange(st.folder, rel)
    expect(s.events).toHaveLength(count)
  })

  it('has no origin for a change that nobody asked for', async () => {
    const s = setup()
    const st = await s.store.createProject('Book')
    const rel = sceneRel(st)
    await s.store.readFile(st.folder, rel)
    s.events.length = 0
    writeFileSync(join(st.folder, ...rel.split('/')), 'from another editor\n')
    await s.store.onExternalChange(st.folder, rel)
    expect(s.events.map((e) => e.origin)).toEqual([undefined, undefined])
  })

  it('does not announce a save that changes nothing, once the app knows the file', async () => {
    const s = setup()
    const st = await s.store.createProject('Book')
    const rel = sceneRel(st)
    const f = await s.store.readFile(st.folder, rel)
    s.events.length = 0
    expect((await s.store.saveFile({ folder: st.folder, path: rel, text: '', baseHash: f.hash })).ok).toBe(true)
    expect(s.events).toEqual([])
  })

  it('announces the state after outline, project, entry, delete and restore changes', async () => {
    const s = setup()
    const st = await s.store.createProject('Book')
    s.events.length = 0
    await runAs('phone:2', () => s.store.outlinePatch(st.folder, [{ op: 'setScene', sceneId: sceneId(st), field: 'title', value: 'Opening' }]))
    expect(s.kinds()).toEqual(['state'])
    expect(s.events[0].change).toMatchObject({ state: { outline: { chapters: [{ scenes: [{ title: 'Opening' }] }] } } })
    expect(s.events[0].origin).toBe('phone:2')

    s.events.length = 0
    await runAs('phone:2', () => s.store.projectPatch(st.folder, { title: 'Renamed' }))
    expect(s.kinds()).toEqual(['state'])
    expect(s.events[0].change).toMatchObject({ state: { project: { title: 'Renamed' } } })

    s.events.length = 0
    const entry = await runAs('phone:2', () => s.store.createEntry(st.folder, 'Mara', 'character'))
    expect(s.kinds()).toEqual(['state'])
    expect(s.events[0].change).toMatchObject({ state: { bible: [{ id: entry.id }] } })

    s.events.length = 0
    await runAs('phone:2', () => s.store.deleteEntry(st.folder, entry.id))
    expect(s.kinds()).toEqual(['deleted', 'state'])
    expect(s.events[0].change).toMatchObject({ path: `bible/${entry.id}.md` })

    // A second scene, then delete it and bring it back from the trash.
    const added = await s.store.outlinePatch(st.folder, [{ op: 'addScene', chapterId: st.outline.chapters[0].id, sceneId: 's-second', title: 'Second', index: null }])
    const second = added.outline.chapters[0].scenes[1]
    const secondRel = `manuscript/${second.file}`
    s.events.length = 0
    await runAs('phone:2', () => s.store.deleteScene(st.folder, second.id))
    expect(s.kinds()).toEqual(['deleted', 'state'])
    expect(s.events[0].change).toMatchObject({ kind: 'deleted', path: secondRel })
    const [item] = await s.store.trash(st.folder)
    s.events.length = 0
    await runAs('phone:2', () => s.store.restoreTrash(st.folder, item.id))
    expect(s.kinds()).toEqual(['text', 'state'])
    expect(s.events[0].change).toMatchObject({ kind: 'text', path: secondRel })
  })

  it('announces a state change from statePatch, and from a dismissed finding', async () => {
    const s = setup()
    const st = await s.store.createProject('Book')
    const mara = await s.store.createEntry(st.folder, 'Mara', 'character')
    s.events.length = 0
    await runAs('phone:3', () => s.store.dismissFinding(st.folder, sceneId(st), 'abc', true))
    expect(s.kinds()).toEqual(['state'])
    s.events.length = 0
    const fact = { id: 'f1', entity: mara.id, kind: 'condition' as const, value: 'tired', about: null, from: sceneId(st), until: null }
    const out = await runAs('phone:3', () => s.store.statePatch(st.folder, [{ op: 'add', fact }]))
    expect(out.rejected).toEqual([])
    expect(s.kinds()).toEqual(['state'])
    expect(s.events[0].origin).toBe('phone:3')
    // Nothing applied: nothing announced.
    s.events.length = 0
    const none = await s.store.statePatch(st.folder, [{ op: 'remove', factId: 'nope' }])
    expect(none.rejected).toHaveLength(1)
    expect(s.events).toEqual([])
  })

  it('a listener that throws does not fail the save', async () => {
    const base = mkdtempSync(join(tmpdir(), 'rulz-phone-'))
    const store = new NovelStore({
      rootSetting: () => '',
      documents: () => base,
      trashItem: async () => undefined,
      emit: () => {
        throw new Error('listener broke')
      },
      watch: false
    })
    const st = await store.createProject('Book')
    const rel = sceneRel(st)
    const f = await store.readFile(st.folder, rel)
    expect((await store.saveFile({ folder: st.folder, path: rel, text: 'x\n', baseHash: f.hash })).ok).toBe(true)
  })
})

describe('open projects are counted by owner', () => {
  it('keeps a project the window has open when a phone closes it', async () => {
    const s = setup()
    const st = await s.store.createProject('Book')
    expect(s.store.ownersOf(st.folder)).toEqual([WINDOW_OWNER])
    await s.store.openProject(st.folder, 'phone:1')
    expect(s.store.ownersOf(st.folder).sort()).toEqual(['phone:1', WINDOW_OWNER].sort())
    await s.store.closeProject(st.folder, 'phone:1')
    expect(s.store.ownersOf(st.folder)).toEqual([WINDOW_OWNER])
    // Still open: reading works.
    await expect(s.store.readFile(st.folder, sceneRel(st))).resolves.toBeTruthy()
  })

  it('keeps a project a phone has open when the window closes it, and drops it after the last one', async () => {
    const s = setup()
    const st = await s.store.createProject('Book')
    await s.store.openProject(st.folder, 'phone:1')
    await s.store.closeProject(st.folder) // the window
    await expect(s.store.readFile(st.folder, sceneRel(st))).resolves.toBeTruthy()
    await s.store.closeProject(st.folder, 'phone:1')
    await expect(s.store.readFile(st.folder, sceneRel(st))).rejects.toThrow('not open')
    expect(s.store.hasOpenProject()).toBe(false)
  })

  it('releaseOwner closes every project that phone held, and only its own share', async () => {
    const s = setup()
    const a = await s.store.createProject('A')
    const b = await s.store.createProject('B')
    await s.store.openProject(a.folder, 'phone:1')
    await s.store.openProject(b.folder, 'phone:1')
    await s.store.openProject(b.folder, 'phone:2')
    await s.store.closeProject(a.folder) // the window lets go of A
    await s.store.closeProject(b.folder)
    await s.store.releaseOwner('phone:1')
    await expect(s.store.readFile(a.folder, sceneRel(a))).rejects.toThrow('not open')
    expect(s.store.ownersOf(b.folder)).toEqual(['phone:2'])
    await s.store.releaseOwner('phone:1') // twice is fine
    await s.store.releaseOwner('phone:2')
    expect(s.store.hasOpenProject()).toBe(false)
  })

  it('stops the watcher only when no owner is left', async () => {
    const s = setup(true)
    const st = await s.store.createProject('Book')
    await s.store.openProject(st.folder, 'phone:1')
    await s.store.closeProject(st.folder)
    const rel = sceneRel(st)
    await s.store.closeProject(st.folder, 'phone:1')
    // After the last owner left, an outside edit is not heard (the project is gone, nothing watches).
    s.events.length = 0
    writeFileSync(join(st.folder, ...rel.split('/')), 'late\n')
    await s.store.onExternalChange(st.folder, rel)
    expect(s.events).toEqual([])
  })

  it('opening twice by the same owner and closing once drops it', async () => {
    const s = setup()
    const st = await s.store.createProject('Book')
    await s.store.openProject(st.folder, WINDOW_OWNER)
    await s.store.closeProject(st.folder)
    expect(s.store.hasOpenProject()).toBe(false)
  })

  it('a project created by a phone belongs to that phone', async () => {
    const s = setup()
    const st = await s.store.createProject('Phone book', 'phone:9')
    expect(s.store.ownersOf(st.folder)).toEqual(['phone:9'])
  })
})

describe('a phone save and the desktop', () => {
  it("leaves the desktop editor's journal alone, while the window's own save clears it", async () => {
    const s = setup()
    const st = await s.store.createProject('Book')
    const rel = sceneRel(st)
    const f = await s.store.readFile(st.folder, rel)
    await s.store.journal(st.folder, rel, 'unsaved desktop edits', f.hash)
    expect((await s.store.readFile(st.folder, rel)).journal).toMatchObject({ text: 'unsaved desktop edits' })
    const r = await runAs('phone:1', () => s.store.saveFile({ folder: st.folder, path: rel, text: 'phone text\n', baseHash: f.hash }))
    expect(r.ok).toBe(true)
    expect((await s.store.readFile(st.folder, rel)).journal).toMatchObject({ text: 'unsaved desktop edits' })
    // The no-op phone save keeps it too.
    const g = await s.store.readFile(st.folder, rel)
    await runAs('phone:1', () => s.store.saveFile({ folder: st.folder, path: rel, text: 'phone text\n', baseHash: g.hash }))
    expect((await s.store.readFile(st.folder, rel)).journal).not.toBeNull()
    // The window saving clears it as before.
    await runAs(WINDOW_OWNER, () => s.store.saveFile({ folder: st.folder, path: rel, text: 'desktop text\n', baseHash: g.hash }))
    expect((await s.store.readFile(st.folder, rel)).journal).toBeNull()
  })

  it('announces the new state after a phone save, but not after the windows own', async () => {
    const s = setup()
    const st = await s.store.createProject('Book')
    const entry = await s.store.createEntry(st.folder, 'Mara', 'character')
    const rel = `bible/${entry.id}.md`
    const f = await s.store.readFile(st.folder, rel)
    s.events.length = 0
    const renamed = f.text.replace('name: Mara', 'name: Marina')
    expect(renamed).not.toBe(f.text)
    await runAs('phone:1', () => s.store.saveFile({ folder: st.folder, path: rel, text: renamed, baseHash: f.hash }))
    expect(s.kinds()).toEqual(['text', 'state'])
    expect(s.events[1].change).toMatchObject({ state: { bible: [{ name: 'Marina' }] } })
    const g = await s.store.readFile(st.folder, rel)
    s.events.length = 0
    await runAs(WINDOW_OWNER, () => s.store.saveFile({ folder: st.folder, path: rel, text: g.text + 'more\n', baseHash: g.hash }))
    expect(s.kinds()).toEqual(['text'])
  })
})

describe('an outside change found by a read, before the watcher reports it', () => {
  it('announces it and snapshots before the next overwrite', async () => {
    const s = setup()
    const st = await s.store.createProject('Book')
    const rel = sceneRel(st)
    const first = await s.store.readFile(st.folder, rel)
    // Take the baseline snapshot, as a first save would.
    await s.store.saveFile({ folder: st.folder, path: rel, text: 'mine\n', baseHash: first.hash })
    const mine = await s.store.readFile(st.folder, rel)
    writeFileSync(join(st.folder, ...rel.split('/')), 'from another editor\n')
    s.events.length = 0
    // A phone reads inside the watcher's delay.
    const seen = await runAs('phone:1', () => s.store.readFile(st.folder, rel))
    expect(seen.text).toBe('from another editor\n')
    // The state follows a moment later (it is built after the read returns).
    for (let i = 0; i < 200 && s.events.length < 2; i++) await new Promise((r) => setTimeout(r, 25))
    expect(s.kinds()).toEqual(['text', 'state'])
    expect(s.events[0]).toEqual({ change: { kind: 'text', folder: st.folder, path: rel, hash: seen.hash }, origin: undefined })
    expect(s.events[1].origin).toBeUndefined()
    // The watcher's own event now finds nothing new.
    s.events.length = 0
    await s.store.onExternalChange(st.folder, rel)
    expect(s.events).toEqual([])
    // The next overwrite snapshots the outside version first.
    const before = readdirSync(join(st.folder, '.rulz', 'history'), { recursive: true }).length
    await s.store.saveFile({ folder: st.folder, path: rel, text: 'later\n', baseHash: seen.hash })
    const after = readdirSync(join(st.folder, '.rulz', 'history'), { recursive: true }).length
    expect(after).toBeGreaterThan(before)
    expect(mine.hash).not.toBe(seen.hash)
  })

  it('announces a project file changed outside when a second owner opens the project', async () => {
    const s = setup()
    const st = await s.store.createProject('Book')
    const file = join(st.folder, 'outline.json')
    const outline = JSON.parse(readFileSync(file, 'utf8'))
    outline.chapters[0].title = 'Edited elsewhere'
    writeFileSync(file, JSON.stringify(outline))
    s.events.length = 0
    const opened = await s.store.openProject(st.folder, 'phone:2')
    expect(opened.outline.chapters[0].title).toBe('Edited elsewhere')
    expect(s.kinds()).toEqual(['state'])
    expect(s.events[0].origin).toBeUndefined()
  })
})

