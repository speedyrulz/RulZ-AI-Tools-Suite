import { describe, expect, it } from 'vitest'
import { findingKey, locateQuote, normaliseQuote } from '../src/shared/novelChecks'

const slice = (text: string, hit: { from: number; to: number } | null): string | null =>
  hit ? text.slice(hit.from, hit.to) : null

describe('locateQuote', () => {
  it('finds an exact quote, first occurrence, with offsets in the original text', () => {
    const text = 'Mara paused. Mara left the room.'
    expect(locateQuote(text, 'Mara left')).toEqual({ from: 13, to: 22 })
    expect(slice(text, locateQuote(text, 'Mara'))).toBe('Mara')
    expect(locateQuote(text, 'Mara')).toEqual({ from: 0, to: 4 })
  })

  it('ignores case', () => {
    const text = 'She Whispered the NAME.'
    expect(slice(text, locateQuote(text, 'whispered the name'))).toBe('Whispered the NAME')
  })

  it('matches curly quotes against straight ones both ways', () => {
    const curly = 'He said, “Don’t go,” and left.'
    expect(slice(curly, locateQuote(curly, '"Don\'t go,"'))).toBe('“Don’t go,”')
    const straight = 'He said, "Don\'t go," and left.'
    expect(slice(straight, locateQuote(straight, '“Don’t go,”'))).toBe('"Don\'t go,"')
  })

  it('folds dashes and the ellipsis character', () => {
    const text = 'It was late — very late… and cold.'
    expect(slice(text, locateQuote(text, 'late - very late... and'))).toBe('late — very late… and')
    expect(slice(text, locateQuote(text, 'late – very'))).toBe('late — very')
  })

  it('matches a quote that has a line break where the text has a space, and the reverse', () => {
    const text = 'The lighthouse stood\nat the edge of\nthe cliff.'
    expect(slice(text, locateQuote(text, 'lighthouse stood at the edge'))).toBe('lighthouse stood\nat the edge')
    expect(slice(text, locateQuote(text, 'stood\n   at   the edge\nof the cliff'))).toBe('stood\nat the edge of\nthe cliff')
    expect(locateQuote('one two', 'one\ntwo')).toEqual({ from: 0, to: 7 })
  })

  it('works on CRLF text and returns offsets into it', () => {
    const text = 'First line.\r\nSecond line here.\r\n\r\nThird.'
    const hit = locateQuote(text, 'line. Second line')
    expect(slice(text, hit)).toBe('line.\r\nSecond line')
    expect(slice(text, locateQuote(text, 'here. Third'))).toBe('here.\r\n\r\nThird')
  })

  it('ignores whitespace around the quote', () => {
    const text = 'a quiet room'
    expect(slice(text, locateQuote(text, '  quiet  \n'))).toBe('quiet')
  })

  it('returns null when the quote is not in the text, or empty', () => {
    expect(locateQuote('Mara left.', 'Oskar left.')).toBeNull()
    expect(locateQuote('Mara left.', '')).toBeNull()
    expect(locateQuote('Mara left.', ' \n ')).toBeNull()
    expect(locateQuote('', 'x')).toBeNull()
    // Not fuzzy: a missing word is not found.
    expect(locateQuote('the old key', 'the key')).toBeNull()
  })

  it('maps offsets back correctly after astral characters and expanding folds', () => {
    const text = 'x 😀 y… target here'
    expect(slice(text, locateQuote(text, 'target'))).toBe('target')
    const ellipsis = 'wait… then go'
    expect(slice(ellipsis, locateQuote(ellipsis, 'then go'))).toBe('then go')
  })
})

describe('findingKey', () => {
  it('is a stable short hex string', () => {
    const key = findingKey({ type: 'continuity', quote: 'Mara left.' })
    expect(key).toMatch(/^[0-9a-f]{14}$/)
    expect(findingKey({ type: 'continuity', quote: 'Mara left.' })).toBe(key)
  })

  it('does not change with case, quote style, dashes or whitespace', () => {
    const base = findingKey({ type: 'tense', quote: 'He said, "Don\'t go" — now.' })
    expect(findingKey({ type: 'tense', quote: 'he SAID,\n“Don’t   go” - now.  ' })).toBe(base)
    expect(normaliseQuote('  A\n B ')).toBe('a b')
  })

  it('changes with the type or the quote', () => {
    const base = findingKey({ type: 'tense', quote: 'It rains.' })
    expect(findingKey({ type: 'pov', quote: 'It rains.' })).not.toBe(base)
    expect(findingKey({ type: 'tense', quote: 'It rained.' })).not.toBe(base)
  })
})
