import { describe, expect, it } from 'vitest'
import { appendLogLines } from '../src/shared/log'

describe('appendLogLines', () => {
  it('adds lines per app, a progress line replacing the one before', () => {
    const logs = appendLogLines(
      { a: ['start'] },
      [
        { appId: 'a', line: 'Receiving objects:  10%' },
        { appId: 'b', line: 'other app' },
        { appId: 'a', line: 'Receiving objects:  90%', replace: true }
      ],
      100
    )
    expect(logs).toEqual({ a: ['start', 'Receiving objects:  90%'], b: ['other app'] })
  })

  it('keeps the newest lines once a log is full', () => {
    const lines = Array.from({ length: 7 }, (_, i) => ({ appId: 'a', line: `line ${i}` }))
    expect(appendLogLines({ a: ['old 1', 'old 2'] }, lines, 5).a).toEqual(['line 2', 'line 3', 'line 4', 'line 5', 'line 6'])
  })

  it('leaves the arrays it was given unchanged (they are React state)', () => {
    const before = { a: ['x'], b: ['y'] }
    const after = appendLogLines(before, [{ appId: 'a', line: 'z' }], 10)
    expect(before.a).toEqual(['x'])
    expect(after.a).not.toBe(before.a)
    expect(after.b).toBe(before.b)
  })
})
