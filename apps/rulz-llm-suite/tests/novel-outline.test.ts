import { describe, expect, it } from 'vitest'
import type { NovelOutline } from '@shared/novel'
import { applyOutlineOps, emptyOutline, findScene, newId, removeChapter, removeScene, reinsert, sceneFileName, slugify } from '@shared/novelOutline'

function sample(): NovelOutline {
  const scene = (id: string, file: string) => ({ id, title: id, file, beats: '', summary: '', summaryOf: '', status: 'idea' as const, pov: null, present: [], sticky: [], targetWords: null })
  return {
    format: 1,
    chapters: [
      { id: 'c1', title: 'One', act: '', summary: '', summaryOf: '', scenes: [scene('s1', '01-a-s1.md'), scene('s2', '02-b-s2.md')] },
      { id: 'c2', title: 'Two', act: '', summary: '', summaryOf: '', scenes: [scene('s3', '03-c-s3.md')] }
    ]
  }
}

describe('ids and names', () => {
  it('makes prefixed ids', () => {
    expect(newId('s')).toMatch(/^s-[0-9a-z]{6}$/)
    expect(newId('c')).not.toBe(newId('c'))
  })

  it('slugifies to safe lowercase ASCII', () => {
    expect(slugify('Night Crossing!')).toBe('night-crossing')
    expect(slugify('Café Ünïcode')).toBe('cafe-unicode')
    expect(slugify('')).toBe('untitled')
    expect(slugify('日本語')).toBe('untitled')
    expect(slugify('CON')).toBe('con-1')
    expect(slugify('a'.repeat(80)).length).toBe(40)
    expect(slugify('x'.repeat(39) + ' yyy')).toBe('x'.repeat(39))
  })

  it('names scene files by position, slug and id tail, unique ignoring case', () => {
    expect(sceneFileName(0, 'Night crossing', 's-k3f9x2', [])).toBe('01-night-crossing-f9x2.md')
    expect(sceneFileName(11, 'A', 's-k3f9x2', [])).toBe('12-a-f9x2.md')
    expect(sceneFileName(0, 'A', 's-k3f9x2', ['01-A-F9X2.MD'])).toBe('01-a-f9x2-2.md')
    expect(sceneFileName(0, 'A', 's-k3f9x2', ['01-a-f9x2.md', '01-a-f9x2-2.md'])).toBe('01-a-f9x2-3.md')
    expect(sceneFileName(0, 'con', 's-aaaaaa', [])).toBe('01-con-1-aaaa.md')
  })
})

describe('applyOutlineOps', () => {
  it('sets fields and does not touch the input', () => {
    const o = sample()
    const { outline, createdFiles } = applyOutlineOps(o, [
      { op: 'setScene', sceneId: 's2', field: 'beats', value: 'Mara bribes the ferryman' },
      { op: 'setScene', sceneId: 's2', field: 'targetWords', value: 2500 },
      { op: 'setChapter', chapterId: 'c2', field: 'act', value: 'Act I' }
    ], [])
    expect(createdFiles).toEqual([])
    expect(findScene(outline, 's2')?.scene).toMatchObject({ beats: 'Mara bribes the ferryman', targetWords: 2500 })
    expect(outline.chapters[1].act).toBe('Act I')
    expect(findScene(o, 's2')?.scene.beats).toBe('')
  })

  it('adds chapters and scenes with unique file names that avoid existing files', () => {
    const { outline, createdFiles } = applyOutlineOps(sample(), [
      { op: 'addChapter', chapterId: 'c3', sceneId: 's-new001', title: 'Three', index: 1 },
      { op: 'addScene', chapterId: 'c1', sceneId: 's-new002', title: 'Scene 1', index: 0, beats: 'plan' }
    ], ['04-scene-1-w001.md'])
    expect(outline.chapters.map((c) => c.id)).toEqual(['c1', 'c3', 'c2'])
    expect(outline.chapters[1].scenes).toHaveLength(1)
    expect(outline.chapters[0].scenes[0]).toMatchObject({ id: 's-new002', beats: 'plan', status: 'beats' })
    expect(createdFiles).toHaveLength(2)
    expect(new Set(createdFiles.map((f) => f.toLowerCase())).size).toBe(2)
    expect(createdFiles).not.toContain('04-scene-1-w001.md')
    expect(outline.chapters[0].scenes[0].file).toBe(createdFiles[1])
  })

  it('moves chapters and scenes, clamping the index', () => {
    const { outline } = applyOutlineOps(sample(), [
      { op: 'moveChapter', chapterId: 'c2', index: 0 },
      { op: 'moveScene', sceneId: 's1', toChapterId: 'c2', index: 99 }
    ], [])
    expect(outline.chapters.map((c) => c.id)).toEqual(['c2', 'c1'])
    expect(outline.chapters[0].scenes.map((s) => s.id)).toEqual(['s3', 's1'])
  })

  it('refuses to move the last scene out of a chapter', () => {
    expect(() => applyOutlineOps(sample(), [{ op: 'moveScene', sceneId: 's3', toChapterId: 'c1', index: 0 }], [])).toThrow(/at least one scene/)
  })

  it('throws clear errors for unknown ids, bad values and reused ids', () => {
    const run = (op: Parameters<typeof applyOutlineOps>[1][number]) => () => applyOutlineOps(sample(), [op], [])
    expect(run({ op: 'setScene', sceneId: 'nope', field: 'title', value: 'x' })).toThrow(/no scene with id "nope"/)
    expect(run({ op: 'setChapter', chapterId: 'nope', field: 'title', value: 'x' })).toThrow(/no chapter/)
    expect(run({ op: 'setScene', sceneId: 's1', field: 'status', value: 'done' })).toThrow(/not a scene status/)
    expect(run({ op: 'setScene', sceneId: 's1', field: 'title', value: '  ' })).toThrow(/cannot be empty/)
    expect(run({ op: 'setScene', sceneId: 's1', field: 'present', value: 'mara' })).toThrow(/list of ids/)
    expect(run({ op: 'addScene', chapterId: 'c1', sceneId: 's1', title: 'x', index: null })).toThrow(/already used/)
    expect(run({ op: 'moveScene', sceneId: 's1', toChapterId: 'nope', index: 0 })).toThrow(/no chapter/)
    expect(run({ op: 'bogus' } as never)).toThrow(/not an outline operation/)
  })

  it('stops at the first bad op and leaves the input unchanged', () => {
    const o = sample()
    expect(() => applyOutlineOps(o, [{ op: 'setScene', sceneId: 's1', field: 'title', value: 'Changed' }, { op: 'setScene', sceneId: 'zzz', field: 'title', value: 'x' }], [])).toThrow()
    expect(findScene(o, 's1')?.scene.title).toBe('s1')
  })

  it('adopts an orphan file once', () => {
    const ops = [{ op: 'adoptFile' as const, file: '07-the-storm.md', chapterId: 'c1', sceneId: 's-adopt', index: null }]
    const { outline, createdFiles } = applyOutlineOps(sample(), ops, ['07-the-storm.md'])
    expect(createdFiles).toEqual([])
    expect(outline.chapters[0].scenes[2]).toMatchObject({ id: 's-adopt', file: '07-the-storm.md', title: 'the storm', status: 'draft' })
    expect(() => applyOutlineOps(sample(), ops, [])).toThrow(/not a Markdown file in the manuscript/)
    expect(() => applyOutlineOps(sample(), [{ ...ops[0], file: '01-A-S1.md' }], ['01-a-s1.md'])).toThrow(/already a scene/)
  })

  it('keeps unknown keys on chapters and scenes', () => {
    const o = sample()
    ;(o.chapters[0].scenes[0] as unknown as Record<string, unknown>).future = { x: 1 }
    const { outline } = applyOutlineOps(o, [{ op: 'setScene', sceneId: 's1', field: 'title', value: 'New' }], [])
    expect((outline.chapters[0].scenes[0] as unknown as Record<string, unknown>).future).toEqual({ x: 1 })
  })
})

describe('adoptFile rules', () => {
  const adopt = (file: string, existing: string[]) => applyOutlineOps(sample(), [{ op: 'adoptFile', file, chapterId: 'c1', sceneId: 's-adopt', index: null }], existing)

  it('takes only Markdown files and uses the name as it is on disk', () => {
    const { outline } = adopt('07-THE-storm.MD', ['07-The-Storm.md'])
    expect(outline.chapters[0].scenes[2].file).toBe('07-The-Storm.md')
    expect(() => adopt('notes.txt', ['notes.txt'])).toThrow(/Markdown file/)
    expect(() => adopt('.hidden.md', ['.hidden.md'])).toThrow(/Markdown file/)
  })
})

describe('removal helpers', () => {
  it('refuses to remove the last scene of a chapter, removes others, and reinserts at the old place', () => {
    expect(() => removeScene(sample(), 's3')).toThrow(/Delete the chapter instead/)
    const r = removeScene(sample(), 's1')
    expect(r.outline.chapters[0].scenes.map((s) => s.id)).toEqual(['s2'])
    const back = reinsert(r.outline, { kind: 'scene', chapterId: r.chapterId, index: r.index, scene: r.scene })
    expect(back).toEqual(sample())
  })

  it('removes and reinserts a chapter, clamping the position', () => {
    const r = removeChapter(sample(), 'c1')
    expect(r.outline.chapters.map((c) => c.id)).toEqual(['c2'])
    expect(reinsert(r.outline, { kind: 'chapter', index: 50, chapter: r.chapter }).chapters.map((c) => c.id)).toEqual(['c2', 'c1'])
    expect(() => reinsert(sample(), { kind: 'chapter', index: 0, chapter: r.chapter })).toThrow(/same id/)
    expect(emptyOutline().chapters).toEqual([])
  })
})
