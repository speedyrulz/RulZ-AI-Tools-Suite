// Tests for the review fixes of the Writing mode's file store (kept apart from novel-store.test.ts).
import { existsSync, mkdirSync, mkdtempSync, readFileSync, renameSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'
import type { NovelChange, NovelProjectState } from '@shared/novel'
import { removeTrashFolder } from '../src/main/novel/trash'
import { NovelStore } from '../src/main/novel/store'

function setup() {
  const base = mkdtempSync(join(tmpdir(), 'rulz-novel-fix-'))
  const recycle = join(base, 'recycle')
  mkdirSync(recycle)
  const events: NovelChange[] = []
  const store = new NovelStore({
    rootSetting: () => '',
    documents: () => join(base, 'Documents'),
    trashItem: async (p) => renameSync(p, join(recycle, `${Date.now()}-${Math.random().toString(36).slice(2)}`)),
    emit: (c) => events.push(c),
    watch: false
  })
  return { base, events, store }
}
type Ctx = ReturnType<typeof setup>

const sceneRel = (st: NovelProjectState): string => `manuscript/${st.outline.chapters[0].scenes[0].file}`
const abs = (st: NovelProjectState, rel: string): string => join(st.folder, ...rel.split('/'))
const snapshotTexts = async (s: Ctx, st: NovelProjectState, rel: string): Promise<string[]> => {
  const items = await s.store.history(st.folder, rel)
  return Promise.all(items.map((h) => s.store.readSnapshot(st.folder, h.snapshot)))
}

describe('history order', () => {
  it('A then B then A inside one second keeps three snapshots, newest first', async () => {
    const s = setup()
    const st = await s.store.createProject('Order')
    const rel = sceneRel(st)
    writeFileSync(abs(st, rel), 'A\n')
    let f = await s.store.readFile(st.folder, rel)
    for (const t of ['B\n', 'A\n', 'C\n']) {
      expect((await s.store.saveFile({ folder: st.folder, path: rel, text: t, baseHash: f.hash, snapshot: true })).ok).toBe(true)
      f = await s.store.readFile(st.folder, rel)
    }
    // Snapshots taken: A (first read), B, A. Before this fix the "-2" file sorted before the plain one
    // and the duplicate check compared against the wrong "newest", dropping the second A.
    expect(await snapshotTexts(s, st, rel)).toEqual(['A\n', 'B\n', 'A\n'])
  })
})

describe('outside edits and snapshots', () => {
  it('an outside edit clears the snapshot gap, so the next save snapshots the outside version', async () => {
    const s = setup()
    const st = await s.store.createProject('Gap')
    const rel = sceneRel(st)
    writeFileSync(abs(st, rel), 'v1\n')
    let f = await s.store.readFile(st.folder, rel)
    expect((await s.store.saveFile({ folder: st.folder, path: rel, text: 'v2\n', baseHash: f.hash })).ok).toBe(true)
    writeFileSync(abs(st, rel), 'outside edit\n')
    await s.store.onExternalChange(st.folder, rel)
    f = await s.store.readFile(st.folder, rel)
    expect((await s.store.saveFile({ folder: st.folder, path: rel, text: 'mine\n', baseHash: f.hash })).ok).toBe(true)
    expect(await snapshotTexts(s, st, rel)).toContain('outside edit\n')
  })

  it('a first read in a blocked project takes no snapshot, so the first read after the fix does', async () => {
    const s = setup()
    const st = await s.store.createProject('Blocked first read')
    writeFileSync(join(st.folder, 'style.md'), 'voice\n')
    const good = readFileSync(join(st.folder, 'outline.json'))
    writeFileSync(join(st.folder, 'outline.json'), 'broken')
    await s.store.onExternalChange(st.folder, 'outline.json')
    await s.store.readFile(st.folder, 'style.md')
    expect(existsSync(join(st.folder, '.rulz', 'history', 'style.md'))).toBe(false)
    writeFileSync(join(st.folder, 'outline.json'), good)
    await s.store.onExternalChange(st.folder, 'outline.json')
    await s.store.readFile(st.folder, 'style.md')
    expect(await s.store.history(st.folder, 'style.md')).toHaveLength(1)
  })

  it('a save or an outside edit drops the cached parse of a bible entry', async () => {
    const s = setup()
    const st = await s.store.createProject('Cache')
    const e = await s.store.createEntry(st.folder, 'Mara', 'character')
    const rel = `bible/${e.id}.md`
    await s.store.openProject(st.folder)
    const f = await s.store.readFile(st.folder, rel)
    await s.store.saveFile({ folder: st.folder, path: rel, text: `${f.text}Aaaa`, baseHash: f.hash })
    expect((await s.store.openProject(st.folder)).bible[0].body).toBe('Aaaa')
    const f2 = await s.store.readFile(st.folder, rel)
    // Same length as before: size alone cannot tell the cache the file changed.
    await s.store.saveFile({ folder: st.folder, path: rel, text: f2.text.replace('Aaaa', 'Bbbb'), baseHash: f2.hash })
    expect((await s.store.openProject(st.folder)).bible[0].body).toBe('Bbbb')
    writeFileSync(abs(st, rel), f2.text.replace('Aaaa', 'Cccc'))
    await s.store.onExternalChange(st.folder, rel)
    const last = s.events.filter((x) => x.kind === 'state').at(-1) as { state: NovelProjectState }
    expect(last.state.bible[0].body).toBe('Cccc')
  })
})

describe('missing scene files', () => {
  it('reads as empty text with hash "" and the first save creates the file', async () => {
    const s = setup()
    const st = await s.store.createProject('Missing')
    const rel = sceneRel(st)
    renameSync(abs(st, rel), join(st.folder, 'gone.txt'))
    const f = await s.store.readFile(st.folder, rel)
    expect(f).toMatchObject({ text: '', hash: '', readOnly: false, journal: null })
    expect((await s.store.openProject(st.folder)).missing).toEqual([rel])
    const r = await s.store.saveFile({ folder: st.folder, path: rel, text: 'new words\n', baseHash: '' })
    expect(r.ok).toBe(true)
    expect(readFileSync(abs(st, rel), 'utf8')).toBe('new words\n')
    expect((await s.store.openProject(st.folder)).missing).toEqual([])
  })
})

describe('adopting files', () => {
  it('offers and adopts only Markdown files, none starting with a dot', async () => {
    const s = setup()
    const st = await s.store.createProject('Adopt')
    writeFileSync(join(st.folder, 'manuscript', 'notes.txt'), 'x')
    writeFileSync(join(st.folder, 'manuscript', '.sync-temp.md'), 'x')
    writeFileSync(join(st.folder, 'manuscript', 'Stray One.md'), 'a b')
    const cur = await s.store.openProject(st.folder)
    expect(cur.orphans).toEqual(['manuscript/Stray One.md'])
    const chapterId = cur.outline.chapters[0].id
    const next = await s.store.outlinePatch(st.folder, [{ op: 'adoptFile', file: 'stray one.MD', chapterId, sceneId: 's-adopt', index: null }])
    expect(next.blocked).toBeNull()
    expect(next.outline.chapters[0].scenes[1].file).toBe('Stray One.md')
    await expect(s.store.outlinePatch(st.folder, [{ op: 'adoptFile', file: 'notes.txt', chapterId, sceneId: 's-bad', index: null }])).rejects.toThrow(/Markdown file/)
  })
})

describe('blocked projects', () => {
  it('emptyTrash refuses', async () => {
    const s = setup()
    const st = await s.store.createProject('Blocked trash')
    await s.store.createEntry(st.folder, 'X', 'lore')
    await s.store.deleteEntry(st.folder, 'x')
    writeFileSync(join(st.folder, 'outline.json'), 'broken')
    await s.store.onExternalChange(st.folder, 'outline.json')
    await expect(s.store.emptyTrash(st.folder)).rejects.toThrow(/read-only/)
    expect(existsSync(join(st.folder, '.rulz', 'trash'))).toBe(true)
  })

  it('an identical journal is left in place while blocked', async () => {
    const s = setup()
    const st = await s.store.createProject('Blocked journal')
    const rel = sceneRel(st)
    writeFileSync(abs(st, rel), 'same\n')
    await s.store.journal(st.folder, rel, 'same\n', '')
    const journalFile = join(st.folder, '.rulz', 'unsaved', `${rel.replace(/\//g, '__')}.json`)
    expect(existsSync(journalFile)).toBe(true)
    writeFileSync(join(st.folder, 'outline.json'), 'broken')
    await s.store.onExternalChange(st.folder, 'outline.json')
    expect((await s.store.readFile(st.folder, rel)).journal).toBeNull()
    expect(existsSync(journalFile)).toBe(true)
  })
})

describe('tampered trash manifests', () => {
  async function trashed(s: Ctx) {
    const st = await s.store.createProject('Tamper')
    let cur = await s.store.outlinePatch(st.folder, [{ op: 'addChapter', chapterId: 'c-two2222', sceneId: 's-aaa1111', title: 'Two', index: null }])
    const file = cur.outline.chapters[1].scenes[0].file
    writeFileSync(abs(cur, `manuscript/${file}`), 'original\n')
    cur = await s.store.deleteChapter(st.folder, 'c-two2222')
    const [item] = await s.store.trash(st.folder)
    const manifestPath = join(st.folder, '.rulz', 'trash', item.id, 'manifest.json')
    return { st, file, item, manifestPath, manifest: JSON.parse(readFileSync(manifestPath, 'utf8')) }
  }

  it('refuses a "to" outside the item and a "from" outside the user files, moving nothing', async () => {
    const s = setup()
    const t = await trashed(s)
    const outside = join(s.base, 'secret.txt')
    writeFileSync(outside, 'keep')
    const goodTo = t.manifest.files[0].to
    for (const files of [
      [{ from: `manuscript/${t.file}`, to: '../../../../secret.txt' }],
      [{ from: `manuscript/${t.file}`, to: '.rulz/trash/other/x.md' }],
      [{ from: '../evil.md', to: goodTo }],
      [{ from: 'project.json', to: goodTo }],
      [{ from: '.rulz/history/x.md', to: goodTo }]
    ]) {
      writeFileSync(t.manifestPath, JSON.stringify({ ...t.manifest, files }))
      await expect(s.store.restoreTrash(t.st.folder, t.item.id)).rejects.toThrow(/damaged|not inside the project|managed by the app/)
    }
    expect(readFileSync(outside, 'utf8')).toBe('keep')
    expect(existsSync(join(t.st.folder, '.rulz', 'trash', t.item.id, 'manuscript', t.file))).toBe(true)
    expect((await s.store.openProject(t.st.folder)).outline.chapters).toHaveLength(1)
  })

  it('refuses an outline fragment with a bad file name or kind', async () => {
    const s = setup()
    const t = await trashed(s)
    const bad = structuredClone(t.manifest)
    bad.outline.chapter.scenes[0].file = '../../evil.md'
    writeFileSync(t.manifestPath, JSON.stringify(bad))
    await expect(s.store.restoreTrash(t.st.folder, t.item.id)).rejects.toThrow(/damaged/)
    writeFileSync(t.manifestPath, JSON.stringify({ ...t.manifest, outline: { kind: 'other', index: 0 } }))
    await expect(s.store.restoreTrash(t.st.folder, t.item.id)).rejects.toThrow(/damaged/)
    expect((await s.store.openProject(t.st.folder)).outline.chapters).toHaveLength(1)
    expect(existsSync(join(t.st.folder, '.rulz', 'trash', t.item.id, 'manuscript', t.file))).toBe(true)
  })

  it('a manifest cannot claim another id, and an untouched one still restores', async () => {
    const s = setup()
    const t = await trashed(s)
    writeFileSync(t.manifestPath, JSON.stringify({ ...t.manifest, id: 'something-else' }))
    const cur = await s.store.restoreTrash(t.st.folder, t.item.id)
    expect(cur.outline.chapters).toHaveLength(2)
    expect(readFileSync(abs(cur, `manuscript/${t.file}`), 'utf8')).toBe('original\n')
  })

  it('removeTrashFolder keeps the manifest while files remain in the item', async () => {
    const s = setup()
    const t = await trashed(s)
    await removeTrashFolder(t.st.folder, t.item.id)
    expect(existsSync(t.manifestPath)).toBe(true)
    expect((await s.store.trash(t.st.folder)).map((i) => i.id)).toEqual([t.item.id])
  })
})
