// The spending record (cloud-spend.json): integer micro-dollars, persisted reservations, one queue for every change
// (docs/cloud-design.md, section 5). A record that cannot be read is never replaced by a fresh one: requests are refused.
import { randomUUID } from 'node:crypto'
import { promises as fs } from 'node:fs'
import { formatUsd, type CloudSpend } from '@shared/cloud'
import { writeJsonAtomic } from '../util/jsonFile'

export interface Reservation {
  id: string
  slug: string
  /** YYYY-MM-DD at the time of the reservation: it settles into that day. */
  day: string
  /** Micro-dollars. */
  amount: number
  generationId?: string
  at: number
}

interface DayTotals {
  spent: number
  models: Record<string, number>
}

interface LedgerFile {
  version: 1
  days: Record<string, DayTotals>
  reservations: Reservation[]
}

const KEEP_DAYS = 90
/** Longest a request waits for running requests to settle before it says the limit is busy. */
export const RESERVE_WAIT_MS = 10 * 60_000

export function localDay(ms: number): string {
  const d = new Date(ms)
  const p = (n: number): string => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}`
}

export type ReserveResult =
  | { ok: true; id: string }
  | {
      ok: false
      /** limit: over the limit by itself; busy: waited for the running ones in vain; error: record unusable; aborted: the caller gave up. */
      kind: 'limit' | 'busy' | 'error' | 'aborted'
      message: string
    }

export interface LedgerDeps {
  path: string
  now: () => number
  sleep: (ms: number, signal?: AbortSignal) => Promise<void>
  /** Micro-dollars; null = no limit. Read at every reservation, so a change applies to the next request. */
  limitMicro: () => number | null
  write?: (path: string, data: unknown) => Promise<void>
  waitMs?: number
  newId?: () => string
}

const isNum = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v) && v >= 0

export class CloudLedger {
  private days = new Map<string, DayTotals>()
  private open = new Map<string, Reservation>()
  /** A load failure, or not read yet: never cleared by a write, and nothing is written over the file meanwhile. */
  private loadError: string | null = 'The spending record has not been read yet. Try again in a moment.'
  /** A write failure: cleared by the next write that works. */
  private writeError: string | null = null
  private queue: Promise<unknown> = Promise.resolve()
  private waiters = new Set<() => void>()
  private listeners = new Set<() => void>()

  constructor(private readonly deps: LedgerDeps) {}

  get spendError(): string | null {
    return this.loadError ?? this.writeError
  }

  onChange(cb: () => void): () => void {
    this.listeners.add(cb)
    return () => this.listeners.delete(cb)
  }

  private version = 0

  private emit(): void {
    this.version++
    for (const cb of this.listeners) cb()
    for (const w of [...this.waiters]) w()
  }

  private enqueue<T>(fn: () => Promise<T>): Promise<T> {
    const run = this.queue.then(fn, fn)
    this.queue = run.catch(() => undefined)
    return run
  }

  /** Reads the file. A missing file is a fresh start; one that cannot be read or understood is an error and stays one. */
  load(): Promise<void> {
    return this.enqueue(async () => {
      this.days.clear()
      this.open.clear()
      let text: string
      try {
        text = await fs.readFile(this.deps.path, 'utf8')
      } catch (err) {
        if ((err as NodeJS.ErrnoException).code === 'ENOENT') {
          this.loadError = null
          return
        }
        this.loadError = `The spending record (${this.deps.path}) cannot be read: ${(err as Error).message}. Cloud requests are refused until it can.`
        return
      }
      try {
        const j = JSON.parse(text.replace(/^﻿/, '')) as Partial<LedgerFile>
        if (!j || typeof j !== 'object' || !j.days || typeof j.days !== 'object' || Array.isArray(j.days) || !Array.isArray(j.reservations)) throw new Error('unexpected layout')
        const days = new Map<string, DayTotals>()
        for (const [day, t] of Object.entries(j.days)) {
          if (!t || !isNum(t.spent) || !t.models || typeof t.models !== 'object') throw new Error(`bad day ${day}`)
          for (const v of Object.values(t.models)) if (!isNum(v)) throw new Error(`bad model total in ${day}`)
          days.set(day, { spent: t.spent, models: { ...t.models } })
        }
        const open = new Map<string, Reservation>()
        for (const r of j.reservations) {
          if (!r || typeof r.id !== 'string' || typeof r.slug !== 'string' || typeof r.day !== 'string' || !isNum(r.amount) || !isNum(r.at)) throw new Error('bad reservation')
          open.set(r.id, { id: r.id, slug: r.slug, day: r.day, amount: r.amount, at: r.at, ...(typeof r.generationId === 'string' ? { generationId: r.generationId } : {}) })
        }
        this.days = days
        this.open = open
        this.loadError = null
      } catch (err) {
        this.loadError = `The spending record (${this.deps.path}) is damaged (${(err as Error).message}). It was left as it is; cloud requests are refused until it is fixed or removed and the app is started again.`
      }
    }).finally(() => this.emit())
  }

  private fileData(): LedgerFile {
    const keep = [...this.days.keys()].sort().reverse().slice(0, KEEP_DAYS + 1)
    const days: Record<string, DayTotals> = {}
    for (const d of keep) days[d] = this.days.get(d)!
    return { version: 1, days, reservations: [...this.open.values()] }
  }

  private async persist(): Promise<void> {
    if (this.loadError) throw new Error(this.loadError)
    const before = this.writeError
    try {
      await (this.deps.write ?? writeJsonAtomic)(this.deps.path, this.fileData())
      this.writeError = null
    } catch (err) {
      this.writeError = `The spending record (${this.deps.path}) cannot be written: ${(err as Error).message}. Cloud requests are refused until it can.`
      throw err
    } finally {
      // The page shows the error, and that it is gone, at once (not a change of what is booked: nobody waiting is woken).
      if (this.writeError !== before) for (const cb of this.listeners) cb()
    }
  }

  private reservedOn(day: string): number {
    let n = 0
    for (const r of this.open.values()) if (r.day === day) n += r.amount
    return n
  }

  private tryReserve(slug: string, amount: number): { status: 'ok'; id: string } | { status: 'wait' } | { status: 'fail'; kind: 'limit' | 'error'; message: string } {
    if (this.loadError) return { status: 'fail', kind: 'error', message: this.loadError }
    const at = this.deps.now()
    const day = localDay(at)
    const limit = this.deps.limitMicro()
    if (limit !== null) {
      const spent = this.days.get(day)?.spent ?? 0
      if (spent + amount > limit) {
        return {
          status: 'fail',
          kind: 'limit',
          message: `This request could cost up to ${formatUsd(amount)}, and ${formatUsd(Math.max(0, limit - spent))} is left of today's ${formatUsd(limit)} limit. Shorten it, lower the model's reply or context limit, or raise the limit on the OpenRouter page.`
        }
      }
      if (spent + this.reservedOn(day) + amount > limit) return { status: 'wait' }
    }
    const id = this.deps.newId?.() ?? randomUUID()
    this.open.set(id, { id, slug, day, amount, at })
    return { status: 'ok', id }
  }

  /**
   * Books the worst case of a request before it leaves, on disk. Over the limit by itself: refused. Only the running
   * requests in the way: waits (abortable, RESERVE_WAIT_MS at most) for them to settle.
   */
  async reserve(slug: string, amount: number, signal?: AbortSignal): Promise<ReserveResult> {
    const wait = new AbortController()
    let timeout: Promise<'timeout' | 'cancelled'> | null = null
    try {
      for (;;) {
        if (signal?.aborted) return { ok: false, kind: 'aborted', message: 'Cancelled.' }
        const seen = this.version
        const r = await this.enqueue(async () => {
          const t = this.tryReserve(slug, amount)
          if (t.status !== 'ok') return t
          try {
            await this.persist()
          } catch (err) {
            this.open.delete(t.id)
            return { status: 'fail' as const, kind: 'error' as const, message: this.spendError ?? (err as Error).message }
          }
          return t
        })
        if (r.status === 'ok') {
          this.emit()
          return { ok: true, id: r.id }
        }
        if (r.status === 'fail') return { ok: false, kind: r.kind, message: r.message }
        // A settlement since the attempt started may have made room already.
        if (this.version !== seen) continue
        // Wait for any settlement, the timeout or the caller.
        timeout ??= this.deps.sleep(this.deps.waitMs ?? RESERVE_WAIT_MS, wait.signal).then(
          () => 'timeout' as const,
          () => 'cancelled' as const
        )
        const hooks: { onAbort?: () => void; wake?: () => void } = {}
        const aborted = new Promise<'aborted'>((res) => {
          if (!signal) return
          hooks.onAbort = () => res('aborted')
          signal.addEventListener('abort', hooks.onAbort, { once: true })
        })
        const changed = new Promise<'changed'>((res) => {
          hooks.wake = () => res('changed')
          this.waiters.add(hooks.wake)
        })
        const why = await Promise.race([changed, timeout, aborted])
        if (hooks.wake) this.waiters.delete(hooks.wake)
        if (hooks.onAbort) signal?.removeEventListener('abort', hooks.onAbort)
        if (why === 'aborted') return { ok: false, kind: 'aborted', message: 'Cancelled.' }
        if (why === 'timeout') {
          const day = localDay(this.deps.now())
          const limit = this.deps.limitMicro()
          return {
            ok: false,
            kind: 'busy',
            message: `This request could cost up to ${formatUsd(amount)}, but other running requests may use up today's ${limit === null ? '' : formatUsd(limit) + ' '}limit (${formatUsd(this.reservedOn(day))} is booked for them). OpenRouter is busy with them: try again when they have finished.`
          }
        }
      }
    } finally {
      wait.abort()
    }
  }

  get(id: string): Reservation | undefined {
    const r = this.open.get(id)
    return r ? { ...r } : undefined
  }

  reservations(): Reservation[] {
    return [...this.open.values()].map((r) => ({ ...r }))
  }

  /** Remembers the generation id of a running request (headers or first chunk), on disk. */
  setGenerationId(id: string, generationId: string): Promise<void> {
    return this.enqueue(async () => {
      const r = this.open.get(id)
      if (!r || r.generationId === generationId) return
      r.generationId = generationId
      await this.persist().catch(() => undefined)
    })
  }

  /**
   * Turns a reservation into spending of `micro` (0 = nothing was charged). True when this call settled it: a reservation
   * that is gone (settled before, or never booked) does nothing.
   */
  settle(id: string, micro: number): Promise<boolean> {
    return this.enqueue(async () => {
      const r = this.open.get(id)
      if (!r) return false
      this.open.delete(id)
      const cost = Number.isFinite(micro) && micro > 0 ? Math.ceil(micro) : 0
      if (cost > 0) {
        const t = this.days.get(r.day) ?? { spent: 0, models: {} }
        t.spent += cost
        t.models[r.slug] = (t.models[r.slug] ?? 0) + cost
        this.days.set(r.day, t)
      }
      // The numbers stay in memory when the write fails; the next write that works saves them.
      await this.persist().catch(() => undefined)
      return true
    }).finally(() => this.emit())
  }

  snapshot(): CloudSpend {
    const day = localDay(this.deps.now())
    const today = this.days.get(day)
    const earlier = [...this.days.entries()]
      .filter(([d]) => d !== day)
      .sort(([a], [b]) => (a < b ? 1 : -1))
      .slice(0, KEEP_DAYS)
      .map(([d, t]) => ({ day: d, spent: t.spent }))
    return { day, spent: today?.spent ?? 0, reserved: this.reservedOn(day), models: { ...(today?.models ?? {}) }, days: earlier }
  }
}
