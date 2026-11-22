// History snapshots: .rulz/history/<relative path>/<stamp>.<ext> (data safety rule 5).
import { promises as fs } from 'node:fs'
import { extname, join } from 'node:path'
import type { NovelHistoryItem } from '@shared/novel'
import { assertAppPath, readBytes, writeBytesAtomic } from './files'

const MAX_SNAPSHOTS = 200
const MIN = 60_000
const HOUR = 60 * MIN
const DAY = 24 * HOUR

/** "20261122T101500" as in jsonFile's corruptName (UTC, no colons). */
export function stampOf(date: Date): string {
  return date.toISOString().replace(/[-:]/g, '').replace(/\..*$/, '')
}

/** The time a stamp (with or without a "-2" collision suffix) stands for; null if it is not one. */
export function stampTime(stamp: string): number | null {
  const m = /^(\d{4})(\d\d)(\d\d)T(\d\d)(\d\d)(\d\d)/.exec(stamp)
  if (!m) return null
  return Date.UTC(+m[1], +m[2] - 1, +m[3], +m[4], +m[5], +m[6])
}

/** Orders stamps by time, then by the collision suffix as a number ("-2" before "-10"). */
export function compareStamps(a: string, b: string): number {
  const ta = stampTime(a) ?? -1
  const tb = stampTime(b) ?? -1
  if (ta !== tb) return ta - tb
  const sa = Number(/-(\d+)$/.exec(a)?.[1] ?? 0)
  const sb = Number(/-(\d+)$/.exec(b)?.[1] ?? 0)
  return sa !== sb ? sa - sb : a < b ? -1 : a > b ? 1 : 0
}

/**
 * Which stamps to keep: all from the last hour, one per 10 minutes for a day, one per hour for a week,
 * one per day after that (the newest of each slot), and at most 200 (oldest dropped). Returned oldest first.
 * A stamp that is not a time is kept untouched.
 */
export function thinSnapshots(stamps: string[], now: Date): string[] {
  const sorted = [...stamps].sort(compareStamps)
  const keep = new Set<string>()
  const slots = new Map<string, string>()
  const timed: string[] = []
  for (const stamp of sorted) {
    const t = stampTime(stamp)
    if (t === null) {
      keep.add(stamp)
      continue
    }
    timed.push(stamp)
    const age = now.getTime() - t
    if (age <= HOUR) keep.add(stamp)
    else {
      const [tier, size] = age <= DAY ? ['10m', 10 * MIN] : age <= 7 * DAY ? ['1h', HOUR] : ['1d', DAY]
      // Sorted oldest first, so a later stamp in the same slot replaces the earlier one.
      slots.set(`${tier}:${Math.floor(t / (size as number))}`, stamp)
    }
  }
  for (const s of slots.values()) keep.add(s)
  let kept = sorted.filter((s) => keep.has(s))
  if (kept.length > MAX_SNAPSHOTS) kept = kept.slice(kept.length - MAX_SNAPSHOTS)
  return kept
}

const historyDir = (folder: string, rel: string): string => join(folder, '.rulz', 'history', ...rel.split('/'))
const relSnapshot = (rel: string, name: string): string => `.rulz/history/${rel}/${name}`

async function names(folder: string, dir: string): Promise<string[]> {
  await assertAppPath(folder, dir)
  try {
    return (await fs.readdir(dir, { withFileTypes: true })).filter((e) => e.isFile()).map((e) => e.name)
  } catch {
    return []
  }
}

const stampOfName = (name: string): string => name.slice(0, name.length - extname(name).length)

/** Snapshots of a file, newest first. */
export async function listHistory(folder: string, rel: string): Promise<NovelHistoryItem[]> {
  const dir = historyDir(folder, rel)
  await assertAppPath(folder, dir)
  const items: NovelHistoryItem[] = []
  for (const name of await names(folder, dir)) {
    const stamp = stampOfName(name)
    if (stampTime(stamp) === null) continue
    const st = await fs.stat(join(dir, name)).catch(() => null)
    if (st) items.push({ stamp, snapshot: relSnapshot(rel, name), bytes: st.size })
  }
  return items.sort((a, b) => compareStamps(b.stamp, a.stamp))
}

/**
 * Stores `bytes` as a snapshot of `rel` and thins the folder. Skips it when the newest snapshot is
 * identical. Returns the stamp, or null when nothing was written.
 */
export async function takeSnapshot(folder: string, rel: string, bytes: Uint8Array, now = new Date()): Promise<string | null> {
  const dir = historyDir(folder, rel)
  await assertAppPath(folder, dir)
  const ext = extname(rel) || '.txt'
  const existing = await names(folder, dir)
  const newest = existing.filter((n) => stampTime(stampOfName(n)) !== null).sort((a, b) => compareStamps(stampOfName(a), stampOfName(b))).pop()
  if (newest) {
    const prev = await readBytes(folder, join(dir, newest)).catch(() => null)
    if (prev && Buffer.compare(prev, Buffer.from(bytes)) === 0) return null
  }
  const base = stampOf(now)
  let stamp = base
  const taken = new Set(existing.map((n) => n.toLowerCase()))
  for (let n = 2; taken.has(`${stamp}${ext}`.toLowerCase()); n++) stamp = `${base}-${n}`
  await writeBytesAtomic(folder, join(dir, `${stamp}${ext}`), join(folder, '.rulz', 'tmp'), bytes)
  const all = [...existing.map(stampOfName), stamp]
  const kept = new Set(thinSnapshots(all, now))
  for (const name of [...existing, `${stamp}${ext}`]) {
    if (!kept.has(stampOfName(name)) && stampTime(stampOfName(name)) !== null) await fs.rm(join(dir, name), { force: true }).catch(() => undefined)
  }
  return stamp
}
