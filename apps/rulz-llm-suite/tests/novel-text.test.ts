import { describe, expect, it } from 'vitest'
import { sceneHashKey, splitParagraphs, wordCount } from '../src/shared/novelText'

describe('wordCount', () => {
  it('counts runs of letters and digits', () => {
    expect(wordCount('')).toBe(0)
    expect(wordCount('   \n ')).toBe(0)
    expect(wordCount('One two  three\nfour')).toBe(4)
    expect(wordCount('Chapter 12 begins')).toBe(3)
  })

  it('keeps apostrophes and hyphens inside a word', () => {
    expect(wordCount("don't well-made Élan’s")).toBe(3)
    expect(wordCount("rock 'n' roll")).toBe(3)
  })

  it('counts Unicode letters', () => {
    expect(wordCount('Straße über naïve 東京 café')).toBe(5)
  })

  it('ignores Markdown markers and bare punctuation', () => {
    expect(wordCount('## Heading\n\n**bold** and *italic*')).toBe(4)
    expect(wordCount('---')).toBe(0)
    expect(wordCount('a - b -- c — d')).toBe(4)
    expect(wordCount('- list item\n> quote text')).toBe(4)
    expect(wordCount("' - '")).toBe(0)
  })
})

describe('splitParagraphs', () => {
  it('splits on one or more blank lines and drops empty blocks', () => {
    expect(splitParagraphs('a\n\nb\n\n\n\nc')).toEqual(['a', 'b', 'c'])
    expect(splitParagraphs('\n\na\n\n')).toEqual(['a'])
    expect(splitParagraphs('')).toEqual([])
    expect(splitParagraphs('\n \n\t\n')).toEqual([])
  })

  it('keeps single line breaks inside a block', () => {
    expect(splitParagraphs('first\nline\n\nsecond')).toEqual(['first\nline', 'second'])
  })

  it('treats whitespace-only lines as blank and CRLF like LF', () => {
    expect(splitParagraphs('a\n  \t\nb')).toEqual(['a', 'b'])
    expect(splitParagraphs('a\r\n\r\nb\r\nc')).toEqual(['a', 'b\r\nc'])
  })

  it('trims only the end of a block', () => {
    expect(splitParagraphs('  indented  \n\nnext   ')).toEqual(['  indented', 'next'])
  })

  it('restores the text up to whitespace when joined with a blank line', () => {
    const text = 'One.\n\nTwo\nlines.\n\n\n\nThree.\n'
    expect(splitParagraphs(text).join('\n\n')).toBe('One.\n\nTwo\nlines.\n\nThree.')
  })
})

describe('sceneHashKey', () => {
  it('is stable and has no node:crypto dependency', () => {
    expect(sceneHashKey(['a', 'b'])).toBe(sceneHashKey(['a', 'b']))
    expect(sceneHashKey(['a', 'b'])).toMatch(/^[0-9a-f]{14}$/)
  })

  it('changes with any hash, with the order and with the number of hashes', () => {
    const keys = new Set([
      sceneHashKey(['a', 'b']),
      sceneHashKey(['b', 'a']),
      sceneHashKey(['a', 'c']),
      sceneHashKey(['a']),
      sceneHashKey(['a', 'b', 'c'])
    ])
    expect(keys.size).toBe(5)
  })

  it('does not confuse joined values', () => {
    expect(sceneHashKey(['a,b'])).not.toBe(sceneHashKey(['a', 'b']))
    expect(sceneHashKey(['ab', ''])).not.toBe(sceneHashKey(['a', 'b']))
  })

  it('gives "" for no scenes (no summary yet) and spreads many different inputs', () => {
    expect(sceneHashKey([])).toBe('')
    const seen = new Set<string>()
    for (let i = 0; i < 2000; i++) seen.add(sceneHashKey([`hash-${i}`]))
    expect(seen.size).toBe(2000)
  })
})
