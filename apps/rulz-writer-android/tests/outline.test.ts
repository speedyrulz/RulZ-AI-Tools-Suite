import { describe, expect, it } from 'vitest'
import type { BibleEntry, NovelOutline, NovelProjectState, NovelScene, StateFact } from '../src/shared/novel'
import { applyOutlineOps } from '../src/shared/novelOutline'
import { locateQuote, findingKey } from '../src/shared/novelChecks'
import { appendVoiceNote } from '../src/shared/novelVoice'
import {
  addChapterOps,
  addSceneOps,
  moveChapterOps,
  moveSceneOps,
  neighbourScene,
  proposalOps,
  resolveEntryRef,
  resolveSceneRefs,
  scenePath,
  stickyOps
} from '../src/lib/outlineOps'
import { addFactOps, endFactOps, factLine, factsForScene, otherEntry, validateFact } from '../src/lib/stateOps'
import { actionProblem, findAction, proseRequest } from '../src/lib/runs'
import { onlyAdditions, trimContext, wordDiff } from '../src/lib/wordDiff'

const scene = (id: string, title: string, extra: Partial<NovelScene> = {}): NovelScene => ({
  id,
  title,
  file: `${id}.md`,
  beats: '',
  summary: '',
  summaryOf: '',
  status: 'idea',
  pov: null,
  present: [],
  sticky: [],
  targetWords: null,
  ...extra
})

const outline: NovelOutline = {
  format: 1,
  chapters: [
    { id: 'c1', title: 'One', act: '', summary: '', summaryOf: '', scenes: [scene('s1', 'Ferry'), scene('s2', 'Lighthouse')] },
    { id: 'c2', title: 'Two', act: '', summary: '', summaryOf: '', scenes: [scene('s3', 'Storm')] }
  ]
}

const entry = (id: string, name: string, aliases: string[] = []): BibleEntry => ({ id, name, kind: 'character', aliases, triggers: [], priority: 50, always: false, body: '', problems: [] })
const bible = [entry('mara', 'Mara Venn', ['the keeper']), entry('oskar', 'Oskar Lind')]

describe('outline operations', () => {
  it('moves scenes and chapters one place, and not past the ends', () => {
    expect(moveSceneOps(outline, 's2', -1)).toEqual([{ op: 'moveScene', sceneId: 's2', toChapterId: 'c1', index: 0 }])
    expect(moveSceneOps(outline, 's1', -1)).toBeNull()
    expect(moveSceneOps(outline, 's2', 1)).toBeNull()
    expect(moveChapterOps(outline, 'c1', 1)).toEqual([{ op: 'moveChapter', chapterId: 'c1', index: 1 }])
    expect(moveChapterOps(outline, 'c2', 1)).toBeNull()
  })

  it('makes ops the computer’s own outline code accepts', () => {
    const ch = addChapterOps('Three')
    const sc = addSceneOps('c1', 'New scene')
    const applied = applyOutlineOps(outline, [...ch.ops, ...sc.ops, ...moveSceneOps(outline, 's2', -1)!], [])
    expect(applied.outline.chapters.map((c) => c.title)).toEqual(['One', 'Two', 'Three'])
    expect(applied.outline.chapters[0].scenes.map((s) => s.id)).toEqual(['s2', 's1', sc.sceneId])
  })

  it('finds neighbours in reading order across chapters', () => {
    expect(neighbourScene(outline, 's2', 1)?.id).toBe('s3')
    expect(neighbourScene(outline, 's3', -1)?.id).toBe('s2')
    expect(neighbourScene(outline, 's1', -1)).toBeNull()
    expect(neighbourScene(outline, 's3', 1)).toBeNull()
  })

  it('adds new sticky entries only when there are new ones', () => {
    expect(stickyOps(outline, 's1', ['mara'])).toEqual([{ op: 'setScene', sceneId: 's1', field: 'sticky', value: ['mara'] }])
    const withSticky: NovelOutline = { ...outline, chapters: [{ ...outline.chapters[0], scenes: [scene('s1', 'Ferry', { sticky: ['mara'] }), scene('s2', 'L')] }, outline.chapters[1]] }
    expect(stickyOps(withSticky, 's1', ['mara'])).toBeNull()
  })

  it('names the scene file the way the computer does', () => {
    expect(scenePath(scene('s1', 'x', { file: '01-ferry-k3f9.md' }))).toBe('manuscript/01-ferry-k3f9.md')
  })
})

describe('summary and outline proposals', () => {
  it('sets the scene summary and the hash it was made from', () => {
    expect(proposalOps({ kind: 'summary', target: 'scene', targetId: 's1', summaryOf: 'hash', reply: { summary: '  Mara crosses.  ' } }, bible)).toEqual([
      { op: 'setScene', sceneId: 's1', field: 'summary', value: 'Mara crosses.' },
      { op: 'setScene', sceneId: 's1', field: 'summaryOf', value: 'hash' }
    ])
    expect(proposalOps({ kind: 'summary', target: 'chapter', targetId: 'c1', summaryOf: 'k', reply: { summary: 'X' } }, bible).map((o) => o.op)).toEqual(['setChapter', 'setChapter'])
  })

  it('turns proposed chapters into addChapter ops the computer applies', () => {
    const ops = proposalOps({ kind: 'chapters', unresolved: [], reply: { chapters: [{ title: 'Arrival', act: 'Act I', summary: 'She comes.' }, { title: ' ', act: '', summary: '' }] } }, bible)
    const applied = applyOutlineOps(outline, ops, [])
    expect(applied.outline.chapters.map((c) => c.title)).toEqual(['One', 'Two', 'Arrival', 'Untitled chapter'])
    expect(applied.outline.chapters[2]).toMatchObject({ act: 'Act I', summary: 'She comes.' })
  })

  it('turns proposed scenes into scenes with beats, POV and present resolved to entry ids', () => {
    const ops = proposalOps(
      { kind: 'scenes', chapterId: 'c2', unresolved: ['Ghost'], reply: { scenes: [{ title: 'Wreck', beats: 'The boat breaks.', pov: 'mara venn', present: ['The Keeper', 'Ghost', 'oskar'] }] } },
      bible
    )
    const applied = applyOutlineOps(outline, ops, [])
    const added = applied.outline.chapters[1].scenes[1]
    expect(added).toMatchObject({ title: 'Wreck', beats: 'The boat breaks.', pov: 'mara', present: ['mara', 'oskar'], status: 'beats' })
  })

  it('resolves names and aliases, and lists what it could not', () => {
    expect(resolveEntryRef(bible, 'THE KEEPER')).toBe('mara')
    expect(resolveEntryRef(bible, 'oskar')).toBe('oskar')
    expect(resolveEntryRef(bible, 'Nobody')).toBeNull()
    expect(resolveSceneRefs(bible, [{ pov: 'Nobody', present: ['Nobody', 'Mara Venn'] }])).toEqual({ scenes: [{ pov: null, present: ['mara'] }], unresolved: ['Nobody'] })
  })

  it('gives nothing for results that are not outline changes', () => {
    expect(proposalOps({ kind: 'interview', characterId: 'mara', reply: 'Hi' }, bible)).toEqual([])
  })
})

describe('tracked state', () => {
  const fact = (id: string, over: Partial<StateFact>): StateFact => ({ id, entity: 'mara', kind: 'location', value: 'the ferry', about: null, from: 's1', until: null, ...over })
  const state = {
    outline,
    bible,
    state: { facts: [fact('f1', {}), fact('f2', { kind: 'carries', value: 'a key', from: 's2' }), fact('f3', { kind: 'condition', value: 'tired', from: 's1', until: 's2' })], invalid: [], problem: null, canReset: false }
  } as unknown as NovelProjectState

  it('splits the facts around a scene', () => {
    const f = factsForScene(state, 's2')
    expect(f.before.map((x) => x.id).sort()).toEqual(['f1', 'f3'])
    expect(f.added.map((x) => x.id)).toEqual(['f2'])
    expect(f.ended.map((x) => x.id)).toEqual(['f3'])
    expect(f.after.map((x) => x.id).sort()).toEqual(['f1', 'f2'])
  })

  it('words a fact with names', () => {
    expect(factLine(fact('f', { kind: 'relationship', about: 'oskar', value: 'distrusts him' }), bible)).toBe('Mara Venn: relationship with Oskar Lind — distrusts him')
  })

  it('never picks the same entry for About and With', () => {
    expect(otherEntry(bible, 'mara', 'mara')).toBe('oskar')
    expect(otherEntry(bible, 'mara', 'oskar')).toBe('oskar')
    expect(otherEntry(bible, 'mara', '')).toBe('oskar')
    expect(otherEntry([bible[0]], 'mara', 'mara')).toBe('')
  })

  it('checks new facts and builds the ops', () => {
    expect(validateFact({ entity: '', kind: 'location', value: 'x', about: null })).toMatch(/who or what/)
    expect(validateFact({ entity: 'mara', kind: 'location', value: '  ', about: null })).toMatch(/Write the fact/)
    expect(validateFact({ entity: 'mara', kind: 'relationship', value: 'x', about: null })).toMatch(/other character/)
    expect(validateFact({ entity: 'mara', kind: 'location', value: 'x'.repeat(301), about: null })).toMatch(/300/)
    expect(validateFact({ entity: 'mara', kind: 'location', value: 'the lighthouse', about: null })).toBeNull()
    const [op] = addFactOps({ entity: 'mara', kind: 'location', value: ' the   lighthouse ', about: 'oskar' }, 's2')
    expect(op).toMatchObject({ op: 'add', fact: { entity: 'mara', kind: 'location', value: 'the lighthouse', about: null, from: 's2', until: null } })
    expect(endFactOps('f1', 's3')).toEqual([{ op: 'end', factId: 'f1', until: 's3' }])
  })
})

describe('AI action requests', () => {
  const snap = { text: 'He waited. She came.', baseHash: 'h1', selectionStart: 3, selectionEnd: 3 }

  it('builds a prose request from the editor with cursor and selection', () => {
    const req = proseRequest('F', 's1', findAction('continue'), snap, '  slower ', 3)
    expect(req).toMatchObject({ folder: 'F', action: 'continue', sceneId: 's1', text: 'He waited. She came.', baseHash: 'h1', cursor: 3, selection: null, note: 'slower', alternatives: 3 })
    const sel = proseRequest('F', 's1', findAction('rewrite:vivid'), { ...snap, selectionStart: 3, selectionEnd: 10 }, '', 9)
    expect(sel).toMatchObject({ action: 'rewrite', variant: 'vivid', selection: { from: 3, to: 10 }, alternatives: 4 })
  })

  it('asks for a selection or a note where the action needs one', () => {
    expect(actionProblem(findAction('expand'), snap, '')).toMatch(/Select/)
    expect(actionProblem(findAction('expand'), { ...snap, selectionEnd: 8 }, '')).toBeNull()
    expect(actionProblem(findAction('rewrite:tone'), { ...snap, selectionEnd: 8 }, ' ')).toMatch(/note/)
    expect(actionProblem(findAction('continue'), { ...snap, text: '' }, '')).toBeNull()
  })
})

describe('shared helpers the phone relies on', () => {
  it('finds a check finding’s quote through curly quotes and case', () => {
    const text = 'She said, “Don’t go.” Then she left.'
    const at = locateQuote(text, "don't go.")
    expect(at && text.slice(at.from, at.to)).toBe('Don’t go.')
    expect(locateQuote(text, 'not there')).toBeNull()
    expect(findingKey({ type: 'tense', quote: 'Don’t go' })).toBe(findingKey({ type: 'tense', quote: "don't go" }))
  })

  it('adds a voice note to the Voice section and keeps the rest of the file', () => {
    const file = '---\nname: Mara\nkind: character\n---\nA keeper.\n'
    const out = appendVoiceNote(file, 'Short, dry sentences.')
    expect(out).toBe('---\nname: Mara\nkind: character\n---\nA keeper.\n\n## Voice\n\n- Short, dry sentences.\n')
    expect(appendVoiceNote(out, 'Never swears.')).toContain('- Short, dry sentences.\n- Never swears.\n')
  })
})

describe('word diff', () => {
  it('shows what an alternative adds and removes', () => {
    const parts = wordDiff('The ferry left at dusk.', 'The ferry left at dawn, slowly.')
    expect(parts.map((p) => p.kind)).toContain('del')
    expect(parts.filter((p) => p.kind === 'add').map((p) => p.text).join('')).toContain('dawn,')
    expect(parts.map((p) => p.text).filter((_, i) => parts[i].kind !== 'add').join('')).toBe('The ferry left at dusk.')
    expect(parts.filter((p) => p.kind !== 'del').map((p) => p.text).join('')).toBe('The ferry left at dawn, slowly.')
  })

  it('shows a continue as additions only', () => {
    const parts = wordDiff('He waited.', 'He waited. She came.')
    expect(onlyAdditions(parts)).toBe(true)
    expect(parts.find((p) => p.kind === 'add')?.text).toBe(' She came.')
  })

  it('is empty for two empty texts and one part for equal text', () => {
    expect(wordDiff('', '')).toEqual([])
    expect(wordDiff('same', 'same')).toEqual([{ kind: 'same', text: 'same' }])
  })

  it('shortens long unchanged stretches around a change', () => {
    const long = 'word '.repeat(200)
    const trimmed = trimContext(wordDiff(`${long}old${long}`, `${long}new${long}`))
    const total = trimmed.reduce((n, p) => n + p.text.length, 0)
    expect(total).toBeLessThan(600)
    expect(trimmed[0].text.startsWith('…')).toBe(true)
  })
})
