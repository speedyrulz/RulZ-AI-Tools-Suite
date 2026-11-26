// Audit 2026-11-26: deleting a project drains in-flight saves; emptying the trash runs on the project chain.
import { existsSync, mkdirSync, mkdtempSync, readFileSync, renameSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'
import { serialized } from '../src/main/novel/files'
import { NovelStore } from '../src/main/novel/store'

function setup(onTrash?: (p: string) => void) {
  const base = mkdtempSync(join(tmpdir(), 'rulz-novel-audit-'))
  const recycle = join(base, 'recycle')
  mkdirSync(recycle)
  const trashed: string[] = []
  const store = new NovelStore({
    rootSetting: () => '',
    documents: () => join(base, 'Documents'),
    trashItem: async (p) => {
      trashed.push(p)
      onTrash?.(p)
      renameSync(p, join(recycle, `${Date.now()}-${Math.random().toString(36).slice(2)}`))
    },
    emit: () => undefined,
    watch: false
  })
  return { base, recycle, store, trashed }
}
const gate = (): { wait: Promise<void>; open: () => void } => {
  let open!: () => void
  const wait = new Promise<void>((r) => (open = r))
  return { wait, open }
}
const tick = (ms = 60): Promise<void> => new Promise((r) => setTimeout(r, ms))

describe('deleteProject waits for saves in flight', () => {
  it('a queued save finishes before the folder moves, and the folder is not recreated', async () => {
    let sceneAtTrash = ''
    let rel = ''
    let dirHolder = ''
    const s = setup((p) => {
      sceneAtTrash = readFileSync(join(p, ...rel.split('/')), 'utf8')
      dirHolder = p
    })
    const st = await s.store.createProject('Drain')
    rel = `manuscript/${st.outline.chapters[0].scenes[0].file}`
    const target = join(st.folder, ...rel.split('/'))
    const f = await s.store.readFile(st.folder, rel)
    const g = gate()
    // Something already running on the scene's queue (a slow earlier save).
    const blocker = serialized(target, () => g.wait)
    const save = s.store.saveFile({ folder: st.folder, path: rel, text: 'new text\n', baseHash: f.hash })
    await tick()
    let done = false
    const del = s.store.deleteProject(st.folder).then(() => (done = true))
    await tick(150)
    expect(s.trashed).toEqual([])
    expect(done).toBe(false)
    g.open()
    await blocker
    expect((await save).ok).toBe(true)
    await del
    expect(sceneAtTrash).toBe('new text\n')
    expect(dirHolder).toBe(st.folder)
    expect(existsSync(st.folder)).toBe(false)
  })

  it('a save that arrives once the delete has started is refused', async () => {
    const s = setup()
    const st = await s.store.createProject('Refuse')
    const rel = `manuscript/${st.outline.chapters[0].scenes[0].file}`
    const f = await s.store.readFile(st.folder, rel)
    const g = gate()
    const blocker = serialized(join(st.folder, ...rel.split('/')), () => g.wait)
    const del = s.store.deleteProject(st.folder)
    await tick()
    const late = await s.store.saveFile({ folder: st.folder, path: rel, text: 'late\n', baseHash: f.hash })
    expect(late.ok).toBe(false)
    g.open()
    await blocker
    await del
    expect(existsSync(st.folder)).toBe(false)
  })

  it('a project edit or a read that arrives once the delete has started is refused', async () => {
    const s = setup()
    const st = await s.store.createProject('Late')
    const rel = `manuscript/${st.outline.chapters[0].scenes[0].file}`
    const g = gate()
    const blocker = serialized(join(st.folder, ...rel.split('/')), () => g.wait)
    const del = s.store.deleteProject(st.folder)
    await tick()
    await expect(s.store.projectPatch(st.folder, { title: 'Renamed' })).rejects.toThrow(/being deleted/)
    await expect(s.store.readFile(st.folder, rel)).rejects.toThrow(/being deleted/)
    g.open()
    await blocker
    await del
    expect(existsSync(st.folder)).toBe(false)
  })

  it('a project folder whose name starts the same is not waited for', async () => {
    const s = setup()
    const a = await s.store.createProject('Same')
    const b = await s.store.createProject('Same 2')
    const relB = `manuscript/${b.outline.chapters[0].scenes[0].file}`
    const g = gate()
    const blocker = serialized(join(b.folder, ...relB.split('/')), () => g.wait)
    await s.store.deleteProject(a.folder)
    expect(existsSync(a.folder)).toBe(false)
    g.open()
    await blocker
  })
})

describe('emptyTrash runs on the project chain', () => {
  it('waits for a trash move that is under way', async () => {
    const s = setup()
    const st = await s.store.createProject('Trash')
    const chapterId = st.outline.chapters[0].id
    await s.store.deleteChapter(st.folder, chapterId)
    const g = gate()
    const p = (s.store as unknown as { get(f: string): unknown; exclusive(p: unknown, fn: () => Promise<void>): Promise<void> }).get(st.folder)
    const held = (s.store as unknown as { exclusive(p: unknown, fn: () => Promise<void>): Promise<void> }).exclusive(p, () => g.wait)
    const empty = s.store.emptyTrash(st.folder)
    await tick(150)
    expect(s.trashed).toEqual([])
    g.open()
    await held
    await empty
    expect(s.trashed.length).toBe(1)
  })
})
