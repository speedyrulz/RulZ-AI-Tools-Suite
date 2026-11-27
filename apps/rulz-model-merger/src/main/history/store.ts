// History of finished merges, bakes and LoRA merges: <folder>/history.json (HistoryFile). Grids are not stored here.
// Writes are atomic (temp file + rename) and serialised, so two jobs finishing at once cannot lose an entry.
import { mkdir, readFile, rename, rm, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import { HISTORY_MAX_ENTRIES, type HistoryFile } from '@shared/history'

export type StoredEntry = HistoryFile['entries'][number]

const FINAL = new Set(['done', 'error', 'cancelled'])

export interface HistoryStoreDeps {
  /** Folder that holds history.json (created when needed). */
  dir: string
  now?: () => number
}

function isEntry(v: unknown): v is StoredEntry {
  if (v === null || typeof v !== 'object') return false
  const e = v as { kind?: unknown; job?: unknown }
  if (e.kind !== 'merge' && e.kind !== 'loraMerge') return false
  const job = e.job as { id?: unknown; recipe?: unknown; createdAt?: unknown } | null
  return (
    job !== null &&
    typeof job === 'object' &&
    typeof job.id === 'string' &&
    job.recipe !== null &&
    typeof job.recipe === 'object' &&
    typeof job.createdAt === 'number' &&
    Number.isFinite(job.createdAt)
  )
}

export class HistoryStore {
  private entries: StoredEntry[] = []
  private loaded: Promise<void> | null = null
  private queue: Promise<unknown> = Promise.resolve()

  constructor(private readonly d: HistoryStoreDeps) {}

  private get file(): string {
    return join(this.d.dir, 'history.json')
  }

  private enqueue<T>(job: () => Promise<T>): Promise<T> {
    const result = this.queue.then(job, job)
    this.queue = result.catch(() => undefined)
    return result
  }

  /** Set when a corrupt history.json could not be moved aside: nothing may be written over it this session. */
  private writeBlocked: Error | null = null

  /** A failed load is not cached, so a later call tries again. */
  private load(): Promise<void> {
    if (!this.loaded) {
      const p = this.loadNow()
      this.loaded = p
      p.catch(() => {
        if (this.loaded === p) this.loaded = null
      })
    }
    return this.loaded
  }

  private async loadNow(): Promise<void> {
    let text: string
    try {
      text = await readFile(this.file, 'utf8')
    } catch (e) {
      // Only a missing file means an empty history; any other error (locked, no access) must not reset it.
      if ((e as NodeJS.ErrnoException).code === 'ENOENT') {
        this.entries = []
        return
      }
      throw new Error(`The history file could not be read: ${(e as Error).message}`)
    }
    try {
      const raw = JSON.parse(text.replace(/^﻿/, '')) as unknown
      const list = (raw as { entries?: unknown } | null)?.entries
      if (raw === null || typeof raw !== 'object' || (raw as { version?: unknown }).version !== 1 || !Array.isArray(list)) throw new Error('bad shape')
      this.entries = list.filter(isEntry)
    } catch {
      // Keep the unreadable file for the user; never overwrite it.
      try {
        await rename(this.file, `${this.file}.bad-${(this.d.now ?? Date.now)()}`)
      } catch (e) {
        this.writeBlocked = new Error(`The history file is damaged and could not be moved aside, so history is not saved: ${(e as Error).message}`)
      }
      this.entries = []
    }
  }

  private assertWritable(): void {
    if (this.writeBlocked) throw this.writeBlocked
  }

  private async write(): Promise<void> {
    await mkdir(this.d.dir, { recursive: true })
    const data: HistoryFile = { version: 1, entries: this.entries }
    const tmp = `${this.file}.${process.pid}.tmp`
    try {
      await writeFile(tmp, `${JSON.stringify(data, null, 2)}\n`, 'utf8')
      await rename(tmp, this.file)
    } catch (e) {
      await rm(tmp, { force: true }).catch(() => undefined)
      throw e
    }
  }

  /** Stores a merge or LoRA merge that reached a final status. Returns true only when the stored history changed;
   *  false for a non-final entry or an update identical to what is stored. */
  record(entry: StoredEntry): Promise<boolean> {
    if (!isEntry(entry) || !FINAL.has(entry.job.status)) return Promise.resolve(false)
    const copy = structuredClone(entry)
    copy.job.progress = null
    return this.enqueue(async () => {
      await this.load()
      this.assertWritable()
      const prev = this.entries
      const at =prev.findIndex((e) => e.kind === copy.kind && e.job.id === copy.job.id)
      let next: StoredEntry[]
      if (at >= 0) {
        // An update of a known job (e.g. Move into models): same place, and nothing to write when nothing changed.
        if (JSON.stringify(prev[at]) === JSON.stringify(copy)) return false
        next = prev.slice()
        next[at] = copy
      } else {
        // New entry: newest createdAt first; among equal times the newer record goes first.
        const i = prev.findIndex((e) => e.job.createdAt <= copy.job.createdAt)
        next = i < 0 ? [...prev, copy] : [...prev.slice(0, i), copy, ...prev.slice(i)]
        // Older than everything stored in a full list: it would be cut at once, so store nothing.
        if (next.indexOf(copy) >= HISTORY_MAX_ENTRIES) return false
      }
      this.entries = next.slice(0, HISTORY_MAX_ENTRIES)
      try {
        await this.write()
      } catch (e) {
        this.entries = prev
        throw e
      }
      return true
    })
  }

  list(): Promise<StoredEntry[]> {
    return this.enqueue(async () => {
      await this.load()
      return structuredClone(this.entries)
    })
  }

  remove(kind: 'merge' | 'loraMerge', id: string): Promise<void> {
    return this.enqueue(async () => {
      await this.load()
      this.assertWritable()
      const prev = this.entries
      const next = prev.filter((e) => !(e.kind === kind && e.job.id === id))
      if (next.length === prev.length) return
      this.entries = next
      try {
        await this.write()
      } catch (e) {
        this.entries = prev
        throw e
      }
    })
  }
}
