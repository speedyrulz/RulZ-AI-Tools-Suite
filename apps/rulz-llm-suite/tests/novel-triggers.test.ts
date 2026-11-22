import { describe, expect, it } from 'vitest'
import { matchEntries, voiceTargets } from '../src/shared/novelTriggers'
import type { BibleEntry, NovelScene } from '../src/shared/novel'

function entry(id: string, name: string, over: Partial<BibleEntry> = {}): BibleEntry {
  return { id, name, kind: 'character', aliases: [], triggers: [], priority: 50, always: false, body: '', problems: [], ...over }
}

function scene(over: Partial<NovelScene> = {}): NovelScene {
  return {
    id: 's-1',
    title: 'Scene',
    file: '01-scene.md',
    beats: '',
    summary: '',
    summaryOf: '',
    status: 'beats',
    pov: null,
    present: [],
    sticky: [],
    targetWords: null,
    ...over
  }
}

const ids = (list: ReturnType<typeof matchEntries>): string[] => list.map((c) => c.entry.id)

describe('matchEntries keywords', () => {
  it('matches whole words only', () => {
    const mara = entry('mara', 'Mara')
    expect(ids(matchEntries([mara], 'Mara walked in.', null))).toEqual(['mara'])
    expect(ids(matchEntries([mara], 'Maraud and Samara', null))).toEqual([])
    expect(ids(matchEntries([mara], 'Mara-lee and "Mara".', null))).toEqual(['mara'])
    expect(ids(matchEntries([mara], "Mara's boat", null))).toEqual(['mara'])
    expect(ids(matchEntries([mara], 'Mara2', null))).toEqual([])
  })

  it('is case-insensitive and reports the word as written in the entry', () => {
    const result = matchEntries([entry('mara', 'Mara Venn', { triggers: ['Witch'] })], 'the WITCH came', null)
    expect(result).toHaveLength(1)
    expect(result[0].reason).toBe('trigger: Witch')
    expect(result[0].fixed).toBe(false)
    expect(result[0].priority).toBe(50)
  })

  it('matches names, aliases and triggers', () => {
    const e = entry('mara', 'Mara Venn', { aliases: ['Marie'], triggers: ['salt'] })
    expect(matchEntries([e], 'Mara Venn', null)[0].reason).toBe('trigger: Mara Venn')
    expect(matchEntries([e], 'said marie', null)[0].reason).toBe('trigger: Marie')
    expect(matchEntries([e], 'so much salt', null)[0].reason).toBe('trigger: salt')
    expect(matchEntries([e], 'Venn alone', null)).toEqual([])
  })

  it('uses Unicode word boundaries', () => {
    const elan = entry('elan', 'Élan')
    const strasse = entry('strasse', 'Straße')
    expect(ids(matchEntries([elan, strasse], 'élan! and STRAßE.', null)).sort()).toEqual(['elan', 'strasse'])
    // Letters on either side make it a different word, also non-ASCII ones (\b would get this wrong).
    expect(ids(matchEntries([elan], 'xÉlan', null))).toEqual([])
    expect(ids(matchEntries([elan], 'Élanté', null))).toEqual([])
    expect(ids(matchEntries([entry('cafe', 'Café')], 'the Café stood', null))).toEqual(['cafe'])
    expect(ids(matchEntries([entry('cafe', 'Café')], 'Cafés', null))).toEqual([])
  })

  it('matches multi-word aliases as phrases with flexible whitespace', () => {
    const witch = entry('mara', 'Mara', { aliases: ['the Salt Witch'] })
    expect(ids(matchEntries([witch], 'They feared THE salt   witch.', null))).toEqual(['mara'])
    expect(ids(matchEntries([witch], 'the salt\nwitch', null))).toEqual(['mara'])
    expect(ids(matchEntries([witch], 'the salt and witch', null))).toEqual([])
    expect(ids(matchEntries([witch], 'bathe salt witch', null))).toEqual([])
  })

  it('treats regex characters in a trigger literally', () => {
    const e = entry('x', 'C++ (old)', { triggers: ['a.b', '[x]'] })
    expect(ids(matchEntries([e], 'uses C++ (old) here', null))).toEqual(['x'])
    expect(ids(matchEntries([e], 'axb', null))).toEqual([])
    expect(ids(matchEntries([e], 'the [x] mark', null))).toEqual(['x'])
  })

  it('ignores empty triggers and aliases', () => {
    const e = entry('x', 'Real', { aliases: ['', '   '], triggers: [''] })
    expect(matchEntries([e], 'anything at all', null)).toEqual([])
    expect(matchEntries([entry('e', '')], 'anything', null)).toEqual([])
  })

  it('matches nothing in empty text', () => {
    expect(matchEntries([entry('mara', 'Mara')], '', null)).toEqual([])
    expect(matchEntries([], 'Mara', null)).toEqual([])
  })
})

describe('matchEntries fixed entries', () => {
  it('marks always, pov and present as fixed, with their reason', () => {
    const entries = [
      entry('a', 'Always', { always: true }),
      entry('p', 'Pov'),
      entry('r', 'Present'),
      entry('s', 'Sticky'),
      entry('o', 'Other')
    ]
    const result = matchEntries(entries, '', scene({ pov: 'p', present: ['r'], sticky: ['s'] }))
    expect(Object.fromEntries(result.map((c) => [c.entry.id, [c.reason, c.fixed]]))).toEqual({
      a: ['always', true],
      p: ['pov', true],
      r: ['present', true],
      s: ['sticky', false]
    })
  })

  it('treats always as fixed without a scene and ignores pov/present then', () => {
    const entries = [entry('a', 'Always', { always: true }), entry('p', 'Pov')]
    expect(matchEntries(entries, '', null).map((c) => c.entry.id)).toEqual(['a'])
  })

  it('lets fixed win over a keyword match, once per entry', () => {
    const mara = entry('mara', 'Mara', { priority: 20 })
    const result = matchEntries([mara], 'Mara Mara Mara', scene({ pov: 'mara' }))
    expect(result).toHaveLength(1)
    expect(result[0]).toMatchObject({ reason: 'pov', fixed: true, priority: 20 })
    // Also when it is pov, present and sticky: pov is named, it is listed once.
    const both = matchEntries([mara], '', scene({ pov: 'mara', present: ['mara'], sticky: ['mara'] }))
    expect(both).toHaveLength(1)
    expect(both[0].reason).toBe('pov')
  })

  it('does not follow links from fixed entries', () => {
    const a = entry('a', 'Anna', { always: true, body: 'Loves Bob.' })
    const b = entry('b', 'Bob')
    expect(ids(matchEntries([a, b], '', null))).toEqual(['a'])
  })
})

describe('matchEntries depth-1 pass', () => {
  it('pulls in entries named in the body of a matched entry at half priority', () => {
    const mara = entry('mara', 'Mara', { body: 'She sails with Oskar Venn, called Ox.' })
    const oskar = entry('oskar', 'Oskar Venn', { aliases: ['Ox'], priority: 61 })
    const result = matchEntries([mara, oskar], 'Mara waits.', null)
    expect(result.map((c) => [c.entry.id, c.reason, c.fixed, c.priority])).toEqual([
      ['mara', 'trigger: Mara', false, 50],
      ['oskar', 'via: Mara', false, 30]
    ])
  })

  it('scans names and aliases only, not triggers', () => {
    const mara = entry('mara', 'Mara', { body: 'Afraid of the tide.' })
    const sea = entry('sea', 'The sea', { triggers: ['tide'] })
    expect(ids(matchEntries([mara, sea], 'Mara', null))).toEqual(['mara'])
  })

  it('goes one level deep only', () => {
    const a = entry('a', 'Anna', { body: 'Knows Bob.' })
    const b = entry('b', 'Bob', { body: 'Knows Cyd.' })
    const c = entry('c', 'Cyd')
    expect(ids(matchEntries([a, b, c], 'Anna', null))).toEqual(['a', 'b'])
  })

  it('does not replace a direct match or a fixed entry with a via match', () => {
    const a = entry('a', 'Anna', { body: 'Knows Bob and Cyd.' })
    const b = entry('b', 'Bob')
    const c = entry('c', 'Cyd', { always: true })
    const result = matchEntries([a, b, c], 'Anna and Bob', null)
    const reasons = Object.fromEntries(result.map((r) => [r.entry.id, r.reason]))
    expect(reasons).toEqual({ a: 'trigger: Anna', b: 'trigger: Bob', c: 'always' })
  })

  it('rounds half priority down', () => {
    const a = entry('a', 'Anna', { body: 'Bob.' })
    const b = entry('b', 'Bob', { priority: 51 })
    expect(matchEntries([a, b], 'Anna', null)[1].priority).toBe(25)
  })

  it('uses the first matched source as the reason', () => {
    const a = entry('a', 'Anna', { body: 'Cyd.' })
    const b = entry('b', 'Bob', { body: 'Cyd too.' })
    const c = entry('c', 'Cyd')
    const result = matchEntries([a, b, c], 'Anna Bob', null)
    expect(result.find((r) => r.entry.id === 'c')?.reason).toBe('via: Anna')
  })
})

describe('matchEntries order', () => {
  it('lists fixed first, then by priority descending, then by name', () => {
    const entries = [
      entry('t1', 'Zoe', { priority: 80 }),
      entry('t2', 'Amy', { priority: 80 }),
      entry('t3', 'Max', { priority: 90 }),
      entry('f1', 'Fixed low', { priority: 5, always: true }),
      entry('f2', 'Fixed high', { priority: 70, always: true })
    ]
    const result = matchEntries(entries, 'Zoe Amy Max', null)
    expect(ids(result)).toEqual(['f2', 'f1', 't3', 't2', 't1'])
  })
})

describe('matchEntries sticky entries', () => {
  it('are triggered entries, not fixed: reason sticky at full priority, even without a keyword', () => {
    const sticky = entry('s', 'Sticky', { priority: 70 })
    const result = matchEntries([sticky, entry('o', 'Other')], 'nothing here', scene({ sticky: ['s'] }))
    expect(result).toHaveLength(1)
    expect(result[0]).toMatchObject({ reason: 'sticky', fixed: false, priority: 70 })
  })

  it('stay sticky when the text also names them, and fixed still wins over sticky', () => {
    const a = entry('a', 'Anna')
    expect(matchEntries([a], 'Anna', scene({ sticky: ['a'] }))[0].reason).toBe('sticky')
    expect(matchEntries([a], '', scene({ sticky: ['a'], present: ['a'] }))[0]).toMatchObject({ reason: 'present', fixed: true })
  })

  it('are scanned for links like keyword matches, and sort with the triggered entries', () => {
    const a = entry('a', 'Anna', { body: 'Knows Bob.', priority: 60 })
    const b = entry('b', 'Bob', { priority: 80 })
    const f = entry('f', 'Fixed', { always: true, priority: 1 })
    const result = matchEntries([a, b, f], '', scene({ sticky: ['a'] }))
    expect(result.map((r) => [r.entry.id, r.reason, r.fixed])).toEqual([
      ['f', 'always', true],
      ['a', 'sticky', false],
      ['b', 'via: Anna', false]
    ])
  })
})

describe('voiceTargets', () => {
  const bible = [
    entry('mara', 'Mara Venn', { aliases: ['the Salt Witch'] }),
    entry('oskar', 'Oskar Lind'),
    entry('ferry', 'The ferry', { kind: 'place' }),
    entry('tilda', 'Tilda', { aliases: ['Til'] })
  ]

  it('gives the POV and present characters for dialogue, in bible order', () => {
    const s = scene({ pov: 'oskar', present: ['tilda', 'mara'] })
    expect(voiceTargets('dialogue', s, '', bible)).toEqual(['mara', 'oskar', 'tilda'])
    // The selection does not matter for dialogue.
    expect(voiceTargets('dialogue', s, 'Tilda', bible)).toEqual(['mara', 'oskar', 'tilda'])
  })

  it('leaves out places and entries that are not characters, and ids that do not exist', () => {
    const s = scene({ pov: 'ferry', present: ['ghost', 'mara'] })
    expect(voiceTargets('dialogue', s, '', bible)).toEqual(['mara'])
    expect(voiceTargets('dialogue', null, '', bible)).toEqual([])
    expect(voiceTargets('dialogue', scene(), '', bible)).toEqual([])
  })

  it('finds characters named in the selection for other actions (whole word, any case, aliases)', () => {
    const s = scene({ pov: 'oskar' })
    expect(voiceTargets('rewrite', s, 'MARA VENN looked at the ferry.', bible)).toEqual(['mara'])
    expect(voiceTargets('expand', s, 'Til said the salt witch was late.', bible)).toEqual(['mara', 'tilda'])
    expect(voiceTargets('condense', s, 'Tilda’s boat; Oskar Lindgren waved.', bible)).toEqual(['tilda'])
    expect(voiceTargets('rewrite', s, 'Maraud and Until', bible)).toEqual([])
  })

  it('gives nothing without a selection, or for a place named in it', () => {
    expect(voiceTargets('rewrite', scene({ pov: 'mara' }), '   ', bible)).toEqual([])
    expect(voiceTargets('rewrite', null, 'The ferry creaked.', bible)).toEqual([])
    expect(voiceTargets('continue', scene({ pov: 'mara' }), '', bible)).toEqual([])
  })
})
