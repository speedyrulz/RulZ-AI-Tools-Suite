// reasoning_details across tool rounds (docs/cloud-design.md, section 4): the details of every reply that made tool calls
// are remembered in memory, keyed by slug + a hash of the calls, and put back when a later request contains that message.
import { createHash } from 'node:crypto'

export interface CallKey {
  id: string
  name: string
  arguments: string
}

/** JSON with sorted keys, so re-serialised arguments (other spacing or key order) give the same text. */
export function canonicalJson(text: string): string {
  const sort = (v: unknown): unknown => {
    if (Array.isArray(v)) return v.map(sort)
    if (v && typeof v === 'object') {
      const o = v as Record<string, unknown>
      return Object.fromEntries(Object.keys(o).sort().map((k) => [k, sort(o[k])]))
    }
    return v
  }
  try {
    return JSON.stringify(sort(JSON.parse(text)))
  } catch {
    return text
  }
}

export function callsKey(slug: string, calls: CallKey[]): string {
  const h = createHash('sha256')
  h.update(JSON.stringify([slug, calls.map((c) => [c.id, c.name, canonicalJson(c.arguments)])]))
  return h.digest('hex')
}

export const REASONING_CACHE_BYTES = 32 * 1024 * 1024

/** Least recently used, bounded by the size of the stored text. */
export class ReasoningCache {
  private readonly map = new Map<string, { size: number; details: unknown[] }>()
  private bytes = 0

  constructor(private readonly maxBytes = REASONING_CACHE_BYTES) {}

  put(slug: string, calls: CallKey[], details: unknown[]): void {
    if (!calls.length || !details.length) return
    const key = callsKey(slug, calls)
    const size = JSON.stringify(details).length + key.length
    const old = this.map.get(key)
    if (old) {
      this.bytes -= old.size
      this.map.delete(key)
    }
    if (size > this.maxBytes) return
    this.map.set(key, { size, details })
    this.bytes += size
    while (this.bytes > this.maxBytes) {
      const first = this.map.keys().next()
      if (first.done) break
      this.bytes -= this.map.get(first.value)!.size
      this.map.delete(first.value)
    }
  }

  get(slug: string, calls: CallKey[]): unknown[] | undefined {
    if (!calls.length) return undefined
    const key = callsKey(slug, calls)
    const hit = this.map.get(key)
    if (!hit) return undefined
    this.map.delete(key)
    this.map.set(key, hit)
    return hit.details
  }

  get size(): number {
    return this.bytes
  }
  clear(): void {
    this.map.clear()
    this.bytes = 0
  }
}
