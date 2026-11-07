import { describe, expect, it } from 'vitest'
import { parseNumberDraft } from '../src/renderer/src/lib/input'
import { once } from '../src/renderer/src/lib/once'

describe('parseNumberDraft', () => {
  it('applies a typed number only once editing ends, and only if it changed', () => {
    expect(parseNumberDraft('16384', 4096)).toBe(16384)
    expect(parseNumberDraft(' 8 ', 4)).toBe(8)
    expect(parseNumberDraft('4096', 4096)).toBeUndefined()
  })

  it('maps an emptied field to null, unless it was already empty', () => {
    expect(parseNumberDraft('', 4096)).toBeNull()
    expect(parseNumberDraft('  ', 4096)).toBeNull()
    expect(parseNumberDraft('', null)).toBeUndefined()
  })

  it('ignores text that is not a number', () => {
    expect(parseNumberDraft('abc', 1)).toBeUndefined()
    expect(parseNumberDraft('Infinity', 1)).toBeUndefined()
  })
})

describe('once', () => {
  it('runs the setup a single time and shares its promise', async () => {
    let runs = 0
    const init = once(async () => {
      runs++
      return runs
    })
    const [a, b] = await Promise.all([init(), init()])
    expect(await init()).toBe(1)
    expect([a, b, runs]).toEqual([1, 1, 1])
  })
})
