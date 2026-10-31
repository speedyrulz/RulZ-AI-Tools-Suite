import { describe, expect, it, vi } from 'vitest'

vi.mock('electron', () => ({ shell: {} }))

const { mergeLatest, readLatestCache } = await import('../src/main/apps')

const info = (version: string) => ({ version, label: version, date: null, downloadSize: null, url: null })

describe('mergeLatest', () => {
  it('keeps the known version when a refresh fails, with the error for display', () => {
    const prev = { info: info('v1.2.0'), error: null, release: null, at: 1000 }
    const next = { info: null, error: "GitHub's request limit was reached.", release: null }
    const merged = mergeLatest(prev, next)
    expect(merged.info?.version).toBe('v1.2.0')
    expect(merged.error).toMatch(/request limit/)
    expect(merged.at).toBe(1000)
  })

  it('takes a successful answer as it is', () => {
    const next = { info: info('v1.3.0'), error: null, release: null, at: 2000 }
    expect(mergeLatest({ info: info('v1.2.0'), error: null, release: null, at: 1 }, next)).toBe(next)
  })

  it('has nothing to keep without an earlier answer', () => {
    const next = { info: null, error: 'offline', release: null }
    expect(mergeLatest(undefined, next)).toBe(next)
  })
})

describe('readLatestCache', () => {
  it('reads the per-app format', () => {
    const m = readLatestCache({ entries: { a: { info: info('v1'), at: 5 } } })
    expect(m.get('a')).toMatchObject({ info: { version: 'v1' }, at: 5 })
  })

  it("reads the older launcher's format (one time for every app)", () => {
    const m = readLatestCache({ at: 7, entries: { a: info('v1'), b: info('v2') } })
    expect(m.get('b')).toMatchObject({ info: { version: 'v2' }, at: 7 })
  })

  it('copes with no cache', () => {
    expect(readLatestCache(null).size).toBe(0)
  })
})
