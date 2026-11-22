import { describe, expect, it } from 'vitest'
import type { BibleEntry, NovelOutline, StateFact } from '@shared/novel'
import { factStatus, factsAtScene, formatStateLines, scenePositions, stateAfter, stateBefore } from '@shared/novelState'

const scene = (id: string) => ({ id, title: id, file: `${id}.md`, beats: '', summary: '', summaryOf: '', status: 'idea' as const, pov: null, present: [], sticky: [], targetWords: null })
const outline: NovelOutline = {
  format: 1,
  chapters: [
    { id: 'c1', title: 'One', act: '', summary: '', summaryOf: '', scenes: [scene('s1'), scene('s2')] },
    { id: 'c2', title: 'Two', act: '', summary: '', summaryOf: '', scenes: [scene('s3'), scene('s4')] }
  ]
}
const entry = (id: string, name: string): BibleEntry => ({ id, name, kind: 'character', aliases: [], triggers: [], priority: 50, always: false, body: '', problems: [] })
const bible = [entry('mara', 'Mara Venn'), entry('oskar', 'Oskar Lind')]
const ids = new Set(bible.map((b) => b.id))

let n = 0
const fact = (over: Partial<StateFact>): StateFact => ({ id: `f${++n}`, entity: 'mara', kind: 'carries', value: 'the key', about: null, from: 's1', until: null, ...over })

describe('scenePositions', () => {
  it('numbers scenes in reading order across chapters', () => {
    expect([...scenePositions(outline)]).toEqual([['s1', 0], ['s2', 1], ['s3', 2], ['s4', 3]])
  })
})

describe('factStatus', () => {
  const pos = scenePositions(outline)
  it('is ok for a fact whose references all exist', () => {
    expect(factStatus(fact({ until: 's3' }), pos, ids)).toBe('ok')
    expect(factStatus(fact({ kind: 'relationship', about: 'oskar' }), pos, ids)).toBe('ok')
  })
  it('is detached when the entity, about, from or until is unknown', () => {
    expect(factStatus(fact({ entity: 'gone' }), pos, ids)).toBe('detached')
    expect(factStatus(fact({ kind: 'relationship', about: 'gone' }), pos, ids)).toBe('detached')
    expect(factStatus(fact({ from: 'sX' }), pos, ids)).toBe('detached')
    expect(factStatus(fact({ until: 'sX' }), pos, ids)).toBe('detached')
  })
  it('is inverted when until comes before from, and ok when they are the same scene', () => {
    expect(factStatus(fact({ from: 's3', until: 's1' }), pos, ids)).toBe('inverted')
    expect(factStatus(fact({ from: 's2', until: 's2' }), pos, ids)).toBe('ok')
  })
})

describe('stateBefore / stateAfter', () => {
  it('before a scene: from strictly earlier, until at or after the scene', () => {
    const a = fact({ value: 'a', from: 's1' })
    const b = fact({ value: 'b', from: 's2' })
    const c = fact({ value: 'c', from: 's1', until: 's2' })
    const d = fact({ value: 'd', from: 's1', until: 's1' })
    const facts = [a, b, c, d]
    expect(stateBefore(facts, 's1', outline, bible)).toEqual([])
    expect(stateBefore(facts, 's2', outline, bible).map((f) => f.value)).toEqual(['a', 'c'])
    expect(stateBefore(facts, 's3', outline, bible).map((f) => f.value)).toEqual(['a', 'b'])
  })

  it('after a scene: from at or before, until strictly after', () => {
    const a = fact({ value: 'a', from: 's1' })
    const b = fact({ value: 'b', from: 's2' })
    const c = fact({ value: 'c', from: 's1', until: 's2' })
    const facts = [a, b, c]
    expect(stateAfter(facts, 's1', outline, bible).map((f) => f.value)).toEqual(['a', 'c'])
    expect(stateAfter(facts, 's2', outline, bible).map((f) => f.value)).toEqual(['a', 'b'])
  })

  it('uses only ok facts and returns nothing for an unknown scene', () => {
    const facts = [fact({ value: 'ok' }), fact({ value: 'detached', entity: 'gone' }), fact({ value: 'inverted', from: 's3', until: 's1' })]
    expect(stateBefore(facts, 's4', outline, bible).map((f) => f.value)).toEqual(['ok'])
    expect(stateBefore(facts, 'sX', outline, bible)).toEqual([])
    expect(stateAfter(facts, 'sX', outline, bible)).toEqual([])
  })

  it('location is single-valued per entity: the latest from wins, without an until', () => {
    const l1 = fact({ kind: 'location', value: 'the ferry', from: 's1' })
    const l2 = fact({ kind: 'location', value: 'the lighthouse', from: 's3' })
    const lo = fact({ kind: 'location', value: 'the inn', from: 's2', entity: 'oskar' })
    const other1 = fact({ kind: 'carries', value: 'a key', from: 's1' })
    const other2 = fact({ kind: 'carries', value: 'a lamp', from: 's2' })
    const facts = [l1, l2, lo, other1, other2]
    expect(stateBefore(facts, 's3', outline, bible).map((f) => f.value)).toEqual(['the ferry', 'the inn', 'a key', 'a lamp'])
    expect(stateBefore(facts, 's4', outline, bible).map((f) => f.value)).toEqual(['the lighthouse', 'the inn', 'a key', 'a lamp'])
    expect(stateAfter(facts, 's3', outline, bible).map((f) => f.value)).toEqual(['the lighthouse', 'the inn', 'a key', 'a lamp'])
  })

  it('an ended latest location leaves no location; the older open one does not resurface', () => {
    const a = fact({ kind: 'location', value: 'A', from: 's1' })
    const b = fact({ kind: 'location', value: 'B', from: 's3', until: 's4' })
    expect(stateBefore([a, b], 's2', outline, bible).map((f) => f.value)).toEqual(['A'])
    expect(stateBefore([a, b], 's4', outline, bible).map((f) => f.value)).toEqual(['B'])
    expect(stateBefore([a, b], 's4', outline, bible)).toHaveLength(1)
    const c = fact({ kind: 'location', value: 'C', from: 's2', until: 's3' })
    expect(stateBefore([a, c], 's4', outline, bible)).toEqual([])
    expect(stateAfter([a, c], 's2', outline, bible).map((f) => f.value)).toEqual(['C'])
    expect(stateAfter([a, c], 's3', outline, bible)).toEqual([])
  })

  it('a location tie on from goes to the later fact', () => {
    const l1 = fact({ kind: 'location', value: 'first', from: 's1' })
    const l2 = fact({ kind: 'location', value: 'second', from: 's1' })
    expect(stateBefore([l1, l2], 's2', outline, bible).map((f) => f.value)).toEqual(['second'])
  })
})

describe('factsAtScene', () => {
  it('lists what was added and ended at a scene', () => {
    const a = fact({ from: 's2' })
    const b = fact({ from: 's1', until: 's2' })
    const c = fact({ from: 's1' })
    expect(factsAtScene([a, b, c], 's2')).toEqual({ added: [a], ended: [b] })
    expect(factsAtScene([a, b, c], 's4')).toEqual({ added: [], ended: [] })
  })
})

describe('formatStateLines', () => {
  it('writes kind and value, and resolves the relationship target to its name', () => {
    const lines = formatStateLines(
      [fact({ kind: 'location', value: 'the lighthouse' }), fact({ kind: 'relationship', about: 'oskar', value: 'distrusts him' }), fact({ kind: 'knows', value: 'the code' })],
      bible
    )
    expect(lines).toBe('- location: the lighthouse\n- relationship (Oskar Lind): distrusts him\n- knows: the code')
  })
  it('falls back to the id when the target is gone, and is empty for no facts', () => {
    expect(formatStateLines([fact({ kind: 'relationship', about: 'ghost', value: 'fears' })], bible)).toBe('- relationship (ghost): fears')
    expect(formatStateLines([], bible)).toBe('')
  })
})
