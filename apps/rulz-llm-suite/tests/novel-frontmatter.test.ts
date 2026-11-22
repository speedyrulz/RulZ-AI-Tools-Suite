import { describe, expect, it } from 'vitest'
import { formatEntry, parseEntry } from '@shared/frontMatter'

const GOOD = `---
name: Mara Venn
kind: character            # character | place | lore | other
aliases: [Mara, "the Salt Witch", "Venn, M."]
triggers: [witch, Venn]
priority: 70
always: true
---
Tall, salt-stained coat.

Speaks slowly.
`

describe('parseEntry', () => {
  it('reads a complete front matter and keeps the body as it is', () => {
    const e = parseEntry('mara-venn', GOOD)
    expect(e).toMatchObject({
      id: 'mara-venn',
      name: 'Mara Venn',
      kind: 'character',
      aliases: ['Mara', 'the Salt Witch', 'Venn, M.'],
      triggers: ['witch', 'Venn'],
      priority: 70,
      always: true,
      problems: []
    })
    expect(e.body).toBe('Tall, salt-stained coat.\n\nSpeaks slowly.\n')
  })

  it('treats a file without front matter as an entry named after the file', () => {
    const e = parseEntry('old-ferry', 'Just notes.\n')
    expect(e).toMatchObject({ name: 'old-ferry', kind: 'other', aliases: [], triggers: [], priority: 50, always: false, body: 'Just notes.\n', problems: [] })
  })

  it('never throws and reports bad values, using defaults for them', () => {
    const e = parseEntry('x', '---\nname: X\nkind: monster\npriority: lots\nalways: maybe\naliases: [a, b\n---\nbody')
    expect(e.name).toBe('X')
    expect(e.kind).toBe('other')
    expect(e.priority).toBe(50)
    expect(e.always).toBe(false)
    expect(e.aliases).toEqual(['a', 'b'])
    expect(e.problems.length).toBe(4)
    expect(e.body).toBe('body')
  })

  it('shows a file whose front matter is not closed as body, with a problem', () => {
    const e = parseEntry('x', '---\nname: X\nstill going')
    expect(e.body).toContain('still going')
    expect(e.problems).toHaveLength(1)
    expect(e.name).toBe('x')
  })

  it('reads CRLF files, a BOM, block lists and clamps the priority', () => {
    const e = parseEntry('x', '﻿---\r\nname: X\r\naliases:\r\n  - one\r\n  - "two"\r\npriority: 250\r\n---\r\nBody\r\n')
    expect(e.aliases).toEqual(['one', 'two'])
    expect(e.priority).toBe(100)
    expect(e.problems).toHaveLength(1)
    expect(e.body).toBe('Body\n')
  })
})

describe('formatEntry', () => {
  it('writes known keys, then the unknown lines of the original in their order, then the body', () => {
    const original = '---\nvoice: dry, short\nname: Old\nmood:\n  - tired\n  - proud\nkind: place\n---\nOld body'
    const out = formatEntry({ name: 'New: name', kind: 'character', aliases: ['A', 'b, c'], triggers: [], priority: 80, always: false, body: 'New body\n' }, original)
    expect(out).toBe('---\nname: "New: name"\nkind: character\naliases: [A, "b, c"]\ntriggers: []\npriority: 80\nalways: false\nvoice: dry, short\nmood:\n  - tired\n  - proud\n---\nNew body\n')
  })

  it('round-trips through parseEntry', () => {
    const a = parseEntry('mara-venn', GOOD)
    const { id: _id, problems: _problems, ...rest } = a
    const b = parseEntry('mara-venn', formatEntry(rest, GOOD))
    expect(b).toEqual(a)
  })

  it('quotes values that would not read back the same', () => {
    for (const name of ['true', '42', ' padded ', '#hash', 'a # b', '[x]', '"q"', '-dash', '']) {
      const text = formatEntry({ name, kind: 'lore', aliases: [name], triggers: [], priority: 1, always: false, body: '' })
      const e = parseEntry('f', text)
      expect(e.name).toBe(name === '' ? 'f' : name)
      if (name !== '') expect(e.aliases).toEqual([name])
    }
  })

  it('round-trips names, aliases and triggers that contain quotes', () => {
    const values = ["Mara's ring", 'say "hi"', 'it\'s "both"', "Oskar's boat"]
    for (const v of values) {
      const text = formatEntry({ name: v, kind: 'lore', aliases: [v, 'plain'], triggers: [v], priority: 1, always: false, body: '' })
      const e = parseEntry('f', text)
      expect(e.name).toBe(v)
      expect(e.aliases).toEqual([v, 'plain'])
      expect(e.triggers).toEqual([v])
      expect(e.problems).toEqual([])
    }
  })

  it('reads a hand-written list with apostrophes as separate items', () => {
    const e = parseEntry('x', '---\nname: X\naliases: [Mara\'s ring, Oskar\'s boat]\ntriggers: ["a, b", it\'s, \'c d\']  # note\n---\n')
    expect(e.aliases).toEqual(["Mara's ring", "Oskar's boat"])
    expect(e.triggers).toEqual(['a, b', "it's", 'c d'])
  })

  it('does not take a # inside a quoted item for a comment', () => {
    const e = parseEntry('x', '---\nname: X\naliases: ["a # b", c] # real comment\n---\n')
    expect(e.aliases).toEqual(['a # b', 'c'])
  })

  it('keeps comment lines between keys when the body and the name are edited', () => {
    const original = '---\nname: Mara\n# voice: terse\nkind: character\n# note about priority\npriority: 60\nvoice: dry\n---\nOld body\n'
    const e = parseEntry('mara', original)
    expect(e.problems).toEqual([])
    const { id: _id, problems: _problems, ...rest } = e
    const out = formatEntry({ ...rest, name: 'Mara Venn', body: 'New body\n' }, original)
    expect(out).toBe('---\nname: Mara Venn\nkind: character\naliases: []\ntriggers: []\npriority: 60\nalways: false\n# voice: terse\n# note about priority\nvoice: dry\n---\nNew body\n')
    expect(parseEntry('mara', out).name).toBe('Mara Venn')
  })

  it('works without an original', () => {
    expect(formatEntry({ name: 'N', kind: 'lore', aliases: [], triggers: [], priority: 50, always: false, body: '' })).toBe(
      '---\nname: N\nkind: lore\naliases: []\ntriggers: []\npriority: 50\nalways: false\n---\n'
    )
  })
})
