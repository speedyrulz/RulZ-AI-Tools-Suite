import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { CloudLedger, localDay, RESERVE_WAIT_MS } from '../src/main/cloud/ledger'
import { Clock, Sleeper } from './helpers/cloudKit'

let dir: string
let path: string
beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), 'cloud-ledger-'))
  path = join(dir, 'cloud-spend.json')
})
afterEach(() => rmSync(dir, { recursive: true, force: true }))

const disk = (): any => JSON.parse(readFileSync(path, 'utf8'))

interface Opts {
  limit?: number | null
  write?: (p: string, d: unknown) => Promise<void>
}
function make(o: Opts = {}) {
  const clock = new Clock()
  const sleeper = new Sleeper()
  let limit = o.limit === undefined ? 1_000_000 : o.limit
  let n = 0
  const ledger = new CloudLedger({ path, now: clock.now, sleep: sleeper.sleep, limitMicro: () => limit, write: o.write, newId: () => `r${++n}` })
  return { ledger, clock, sleeper, setLimit: (l: number | null) => void (limit = l) }
}

describe('reserve and settle', () => {
  it('refuses and writes nothing before the record is read, so a request at startup cannot replace it', async () => {
    writeFileSync(path, JSON.stringify({ version: 1, days: { [localDay(new Clock().t)]: { spent: 900_000, models: {} } }, reservations: [] }))
    const before = readFileSync(path, 'utf8')
    const { ledger } = make()
    expect(await ledger.reserve('a/b', 1)).toMatchObject({ ok: false, kind: 'error' })
    expect(readFileSync(path, 'utf8')).toBe(before)
    await ledger.load()
    expect(ledger.spendError).toBeNull()
    expect(await ledger.reserve('a/b', 200_000)).toMatchObject({ ok: false, kind: 'limit' })
    expect(await ledger.reserve('a/b', 50_000)).toMatchObject({ ok: true })
  })

  it('books the worst case on disk before returning, settles into integer micro-dollars', async () => {
    const { ledger } = make()
    await ledger.load()
    const r = await ledger.reserve('a/b', 250_000)
    expect(r).toEqual({ ok: true, id: 'r1' })
    expect(disk().reservations).toEqual([expect.objectContaining({ id: 'r1', slug: 'a/b', amount: 250_000, day: localDay(Date.now() * 0 + new Clock().t) })])
    expect(ledger.snapshot()).toMatchObject({ spent: 0, reserved: 250_000 })
    expect(await ledger.settle('r1', 1234.2)).toBe(true)
    expect(ledger.snapshot()).toMatchObject({ spent: 1235, reserved: 0, models: { 'a/b': 1235 } })
    expect(disk()).toMatchObject({ reservations: [], days: { [localDay(new Clock().t)]: { spent: 1235 } } })
  })

  it('settles exactly once', async () => {
    const { ledger } = make()
    await ledger.load()
    await ledger.reserve('a/b', 100)
    const results = await Promise.all([ledger.settle('r1', 50), ledger.settle('r1', 50), ledger.settle('r1', 999)])
    expect(results.filter(Boolean)).toHaveLength(1)
    expect(ledger.snapshot().spent).toBe(50)
    expect(await ledger.settle('nope', 5)).toBe(false)
  })

  it('a cost above the reservation is recorded as it is; zero releases only', async () => {
    const { ledger } = make()
    await ledger.load()
    await ledger.reserve('a/b', 100)
    await ledger.settle('r1', 5000)
    await ledger.reserve('a/b', 100)
    await ledger.settle('r2', 0)
    await ledger.reserve('a/c', 100)
    await ledger.settle('r3', NaN)
    expect(ledger.snapshot()).toMatchObject({ spent: 5000, reserved: 0, models: { 'a/b': 5000 } })
  })

  it('keeps the generation id on disk', async () => {
    const { ledger } = make()
    await ledger.load()
    await ledger.reserve('a/b', 100)
    await ledger.setGenerationId('r1', 'gen-9')
    expect(disk().reservations[0].generationId).toBe('gen-9')
    expect(ledger.get('r1')?.generationId).toBe('gen-9')
  })

  it('limit null: no limit, still recorded', async () => {
    const { ledger } = make({ limit: null })
    await ledger.load()
    const r = await ledger.reserve('a/b', 9_000_000_000)
    expect(r.ok).toBe(true)
    await ledger.settle('r1', 8_000_000_000)
    expect(ledger.snapshot().spent).toBe(8_000_000_000)
  })
})

describe('the limit', () => {
  it('refuses for good when spent + this > limit, with the numbers', async () => {
    const { ledger } = make({ limit: 1_000_000 })
    await ledger.load()
    await ledger.reserve('a/b', 100)
    await ledger.settle('r1', 900_000)
    const r = await ledger.reserve('a/b', 230_000)
    expect(r.ok).toBe(false)
    if (!r.ok) {
      expect(r.kind).toBe('limit')
      expect(r.message).toBe("This request could cost up to $0.23, and $0.10 is left of today's $1.00 limit. Shorten it, lower the model's reply or context limit, or raise the limit on the OpenRouter page.")
    }
    expect(ledger.reservations()).toEqual([])
  })

  it('exactly the limit is allowed', async () => {
    const { ledger } = make({ limit: 1000 })
    await ledger.load()
    expect((await ledger.reserve('a/b', 1000)).ok).toBe(true)
  })

  it('only open reservations in the way: waits for them, then goes', async () => {
    const { ledger } = make({ limit: 1000 })
    await ledger.load()
    await ledger.reserve('a/b', 700)
    let second: unknown = null
    const p = ledger.reserve('a/b', 500).then((r) => (second = r))
    await new Promise((r) => setTimeout(r, 20))
    expect(second).toBeNull()
    await ledger.settle('r1', 100)
    await p
    expect(second).toEqual({ ok: true, id: 'r2' })
  })

  it('a waiting request that no longer fits after the others settled is refused for good', async () => {
    const { ledger } = make({ limit: 1000 })
    await ledger.load()
    await ledger.reserve('a/b', 700)
    const p = ledger.reserve('a/b', 500)
    await new Promise((r) => setTimeout(r, 10))
    await ledger.settle('r1', 700)
    const r = await p
    expect(r).toMatchObject({ ok: false, kind: 'limit' })
  })

  it('waits at most ten minutes, then says it is busy', async () => {
    const { ledger, sleeper } = make({ limit: 1000 })
    await ledger.load()
    await ledger.reserve('a/b', 700)
    const p = ledger.reserve('a/b', 500)
    await vi.waitFor(() => expect(sleeper.waits).toHaveLength(1))
    expect(sleeper.waits[0].ms).toBe(RESERVE_WAIT_MS)
    expect(RESERVE_WAIT_MS).toBe(600_000)
    sleeper.waits[0].release()
    const r = await p
    expect(r).toMatchObject({ ok: false, kind: 'busy' })
    if (!r.ok) expect(r.message).toMatch(/busy/)
    expect(ledger.reservations()).toHaveLength(1)
  })

  it('the wait is abortable', async () => {
    const { ledger } = make({ limit: 1000 })
    await ledger.load()
    await ledger.reserve('a/b', 700)
    const ctl = new AbortController()
    const p = ledger.reserve('a/b', 500, ctl.signal)
    await new Promise((r) => setTimeout(r, 10))
    ctl.abort()
    expect(await p).toMatchObject({ ok: false, kind: 'aborted' })
    // Already aborted: refused at once.
    expect(await ledger.reserve('a/b', 1, ctl.signal)).toMatchObject({ ok: false, kind: 'aborted' })
  })

  it('ten concurrent requests never exceed the limit by reservations', async () => {
    const { ledger } = make({ limit: 1_000_000 })
    await ledger.load()
    let open = 0
    let maxOpen = 0
    let maxReserved = 0
    const finishLater: (() => Promise<void>)[] = []
    const done: Promise<void>[] = []
    for (let i = 0; i < 10; i++) {
      done.push(
        (async () => {
          const r = await ledger.reserve('a/b', 300_000)
          if (!r.ok) throw new Error(r.message)
          open++
          maxOpen = Math.max(maxOpen, open)
          const snap = ledger.snapshot()
          maxReserved = Math.max(maxReserved, snap.reserved + snap.spent)
          await new Promise((res) => setTimeout(res, 5))
          open--
          await ledger.settle(r.id, 50_000)
        })()
      )
    }
    await Promise.all(done)
    void finishLater
    expect(maxOpen).toBeLessThanOrEqual(3)
    expect(maxReserved).toBeLessThanOrEqual(1_000_000)
    expect(ledger.snapshot()).toMatchObject({ spent: 500_000, reserved: 0 })
  })
})

describe('crash leftovers, corrupt file, day change', () => {
  it('a missing file is a fresh start', async () => {
    const { ledger } = make()
    await ledger.load()
    expect(ledger.spendError).toBeNull()
    expect(ledger.snapshot()).toMatchObject({ spent: 0, reserved: 0, days: [] })
  })

  it('reservations left by a crash are read back', async () => {
    const a = make()
    await a.ledger.load()
    await a.ledger.reserve('a/b', 4000)
    await a.ledger.setGenerationId('r1', 'gen-x')
    const b = make()
    await b.ledger.load()
    expect(b.ledger.reservations()).toEqual([expect.objectContaining({ id: 'r1', slug: 'a/b', amount: 4000, generationId: 'gen-x' })])
    expect(b.ledger.snapshot().reserved).toBe(4000)
    await b.ledger.settle('r1', 12)
    expect(b.ledger.snapshot()).toMatchObject({ spent: 12, reserved: 0 })
  })

  it.each([['not json {'], ['[]'], ['{"days":[],"reservations":[]}'], ['{"days":{"2026-01-01":{"spent":-1,"models":{}}},"reservations":[]}'], ['{"days":{},"reservations":[{"id":1}]}']])('corrupt file %s: spendError, refused, never reset', async (text) => {
    writeFileSync(path, text)
    const { ledger } = make()
    await ledger.load()
    expect(ledger.spendError).toMatch(/spending record/)
    const r = await ledger.reserve('a/b', 1)
    expect(r).toMatchObject({ ok: false, kind: 'error' })
    expect(readFileSync(path, 'utf8')).toBe(text)
    // Settling something does not overwrite it either.
    expect(await ledger.settle('x', 1)).toBe(false)
    expect(readFileSync(path, 'utf8')).toBe(text)
  })

  it('an unreadable path is an error too, not a fresh start', async () => {
    // A directory where the file should be.
    const { mkdirSync } = await import('node:fs')
    mkdirSync(path)
    const { ledger } = make()
    await ledger.load()
    expect(ledger.spendError).toBeTruthy()
    expect(await ledger.reserve('a/b', 1)).toMatchObject({ ok: false, kind: 'error' })
  })

  it('a day change: new day starts at zero, the reservation settles into its own day, history is kept', async () => {
    const { ledger, clock } = make({ limit: 1000 })
    await ledger.load()
    await ledger.reserve('a/b', 900)
    await ledger.settle('r1', 900)
    expect((await ledger.reserve('a/b', 200)).ok).toBe(false)
    await ledger.reserve('a/b', 50) // r2 today
    const yesterday = localDay(clock.t)
    clock.advance(24 * 3600_000)
    expect((await ledger.reserve('a/b', 900)).ok).toBe(true) // r3 on the new day
    await ledger.settle('r2', 40) // booked on the day it started
    const snap = ledger.snapshot()
    expect(snap.day).toBe(localDay(clock.t))
    expect(snap.spent).toBe(0)
    expect(snap.reserved).toBe(900)
    expect(snap.days).toEqual([{ day: yesterday, spent: 940 }])
    expect(snap.models).toEqual({})
  })

  it('keeps at most 90 earlier days', async () => {
    const days: Record<string, { spent: number; models: Record<string, number> }> = {}
    for (let i = 1; i <= 120; i++) days[`2025-${String((i % 12) + 1).padStart(2, '0')}-${String((i % 28) + 1).padStart(2, '0')}x${i}`] = { spent: i, models: {} }
    writeFileSync(path, JSON.stringify({ version: 1, days, reservations: [] }))
    const { ledger } = make()
    await ledger.load()
    expect(ledger.snapshot().days.length).toBeLessThanOrEqual(90)
    await ledger.reserve('a/b', 1)
    expect(Object.keys(disk().days).length).toBeLessThanOrEqual(91)
  })
})

describe('write failures', () => {
  it('a reservation that cannot be saved is not granted; the next write that works clears the error', async () => {
    let fail = true
    const { ledger } = make({
      write: async (p, d) => {
        if (fail) throw new Error('disk full')
        writeFileSync(p, JSON.stringify(d))
      }
    })
    await ledger.load()
    const r = await ledger.reserve('a/b', 5)
    expect(r).toMatchObject({ ok: false, kind: 'error' })
    expect(ledger.spendError).toMatch(/disk full/)
    expect(ledger.reservations()).toEqual([])
    fail = false
    expect((await ledger.reserve('a/b', 5)).ok).toBe(true)
    expect(ledger.spendError).toBeNull()
  })

  it('a settlement whose write fails stays in memory and is saved by the next write', async () => {
    let fail = false
    const { ledger } = make({
      write: async (p, d) => {
        if (fail) throw new Error('locked')
        writeFileSync(p, JSON.stringify(d))
      }
    })
    await ledger.load()
    await ledger.reserve('a/b', 5)
    fail = true
    expect(await ledger.settle('r1', 77)).toBe(true)
    expect(ledger.snapshot().spent).toBe(77)
    expect(ledger.spendError).toMatch(/locked/)
    fail = false
    await ledger.reserve('a/b', 1)
    expect(ledger.spendError).toBeNull()
    expect(disk().days[localDay(new Clock().t)].spent).toBe(77)
  })
})

describe('notifications', () => {
  it('tells listeners about every change', async () => {
    const { ledger } = make()
    const seen = vi.fn()
    ledger.onChange(seen)
    await ledger.load()
    await ledger.reserve('a/b', 1)
    await ledger.settle('r1', 1)
    expect(seen.mock.calls.length).toBeGreaterThanOrEqual(3)
  })
})
