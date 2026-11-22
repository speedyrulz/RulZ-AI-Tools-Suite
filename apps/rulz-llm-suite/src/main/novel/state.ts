// state.json and .rulz/checks.json: parsing, validation and op application (stage 3). Pure apart from the hash;
// the store does the file I/O. Rules: docs/writing-mode-design.md, "Stage 3 design".
import type { NovelStateFile, StateFact, StateKind, StateOp } from '@shared/novel'
import { sha256 } from './files'

export const STATE_KINDS: readonly StateKind[] = ['location', 'knows', 'carries', 'condition', 'relationship', 'other']
const VALUE_MAX = 300

const isObject = (v: unknown): v is Record<string, unknown> => typeof v === 'object' && v !== null && !Array.isArray(v)
const errText = (err: unknown): string => (err instanceof Error ? err.message : String(err))

/** One entry of the file's facts array: a fact that validated, or anything else kept verbatim. */
export interface StateItem {
  /** The value as found in the file (unknown keys of a valid fact live here). */
  orig: unknown
  /** The validated fact, or null for an item that could not be read. */
  fact: StateFact | null
  reason: string | null
  /** Position in the file's facts array when it was parsed (what `invalid[].index` reports). */
  index: number
  /** Added or changed by an op: written from the normalised fact. Untouched facts are written exactly as read. */
  touched?: boolean
}

export interface StateDoc {
  /** Top-level keys of the file other than `facts` (format and anything a newer version added). */
  top: Record<string, unknown>
  items: StateItem[]
}

export const emptyDoc = (): StateDoc => ({ top: { format: 1 }, items: [] })

export interface ParsedState {
  doc: StateDoc
  problem: string | null
  canReset: boolean
}

/** Checks the fields of a fact read from disk. Returns the normalised fact or the reason it is unusable. */
export function readFact(raw: unknown): { fact: StateFact } | { reason: string } {
  if (!isObject(raw)) return { reason: 'It is not an object.' }
  if (typeof raw.id !== 'string' || !raw.id) return { reason: 'It has no id.' }
  if (typeof raw.entity !== 'string' || !raw.entity) return { reason: 'It has no entity.' }
  if (typeof raw.kind !== 'string' || !STATE_KINDS.includes(raw.kind as StateKind)) return { reason: `"${String(raw.kind)}" is not a kind of fact.` }
  const kind = raw.kind as StateKind
  if (typeof raw.value !== 'string') return { reason: 'Its value is not text.' }
  const value = raw.value
  if (!value.trim()) return { reason: 'Its value is empty.' }
  if (value !== value.trim()) return { reason: 'Its value has spaces at the start or end.' }
  if (value.length > VALUE_MAX) return { reason: `Its value is longer than ${VALUE_MAX} characters.` }
  if (/[\r\n]/.test(value)) return { reason: 'Its value has more than one line.' }
  const about = raw.about === undefined ? null : raw.about
  if (about !== null && (typeof about !== 'string' || !about)) return { reason: 'Its "about" is not an entry id.' }
  if (kind === 'relationship' && about === null) return { reason: 'A relationship needs an "about".' }
  if (kind !== 'relationship' && about !== null) return { reason: 'Only a relationship has an "about".' }
  if (typeof raw.from !== 'string' || !raw.from) return { reason: 'It has no "from" scene.' }
  const until = raw.until === undefined ? null : raw.until
  if (until !== null && (typeof until !== 'string' || !until)) return { reason: 'Its "until" is not a scene id.' }
  return { fact: { id: raw.id, entity: raw.entity, kind, value, about: about as string | null, from: raw.from, until: until as string | null } }
}

/**
 * Reads state.json. `bytes` null = the file does not exist; `existed` = it was there earlier in this session
 * (then missing is a problem, otherwise it is just an empty state). Never throws.
 */
export function parseStateBytes(bytes: Buffer | null, existed: boolean): ParsedState {
  const bad = (problem: string, canReset = false): ParsedState => ({ doc: emptyDoc(), problem, canReset })
  if (!bytes) return existed ? bad('state.json is missing.', true) : { doc: emptyDoc(), problem: null, canReset: false }
  if (bytes.length === 0) return bad('state.json is empty.', true)
  let parsed: unknown
  try {
    parsed = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(bytes))
  } catch (err) {
    return bad(`state.json could not be read: ${errText(err)}`)
  }
  if (!isObject(parsed)) return bad('state.json could not be read: it is not a JSON object.')
  if (typeof parsed.format === 'number' && parsed.format > 1) return bad(`state.json is from a newer version of the app (format ${parsed.format}), so it is not changed here.`)
  if (!Array.isArray(parsed.facts)) return bad('state.json could not be read: it has no list of facts.')
  const { facts, ...top } = parsed
  if (typeof top.format !== 'number') top.format = 1
  const seen = new Set<string>()
  const items: StateItem[] = (facts as unknown[]).map((orig, index) => {
    const r = readFact(orig)
    if ('reason' in r) {
      if (isObject(orig) && typeof orig.id === 'string' && orig.id) seen.add(orig.id)
      return { orig, fact: null, reason: r.reason, index }
    }
    if (seen.has(r.fact.id)) return { orig, fact: null, reason: `Its id "${r.fact.id}" is used by an earlier fact.`, index }
    seen.add(r.fact.id)
    return { orig, fact: r.fact, reason: null, index }
  })
  return { doc: { top, items }, problem: null, canReset: false }
}

/** The JSON to write: unknown keys of the file and of each fact kept, invalid items verbatim. */
export function serializeDoc(doc: StateDoc): unknown {
  return {
    ...doc.top,
    facts: doc.items.map((i) => {
      if (!i.fact || !i.touched) return i.orig
      const base = isObject(i.orig) ? i.orig : {}
      const f = i.fact as StateFact
      return { ...base, id: f.id, entity: f.entity, kind: f.kind, value: f.value, about: f.about, from: f.from, until: f.until }
    })
  }
}

export const rawOf = (item: StateItem): string => JSON.stringify(item.orig) ?? 'null'

export function stateFileOf(doc: StateDoc, problem: string | null, canReset: boolean): NovelStateFile {
  if (problem) return { facts: [], invalid: [], problem, canReset }
  return {
    facts: doc.items.flatMap((i) => (i.fact ? [i.fact] : [])),
    invalid: doc.items.flatMap((i) => {
      if (i.fact) return []
      const raw = rawOf(i)
      return [{ index: i.index, reason: i.reason ?? 'Unreadable.', hash: sha256(raw), raw }]
    }),
    problem: null,
    canReset: false
  }
}

// ---------------------------------------------------------------------------------------------
// Ops
// ---------------------------------------------------------------------------------------------

export interface OpContext {
  /** Scene id → reading-order index. */
  pos: Map<string, number>
  entryIds: Set<string>
}

const FIELDS = ['entity', 'kind', 'value', 'about', 'from', 'until'] as const

const sameFact = (a: StateFact, b: StateFact): boolean => a.entity === b.entity && a.kind === b.kind && a.value === b.value && (a.about ?? null) === (b.about ?? null) && a.from === b.from

/**
 * Checks the fields of a fact as a new fact would be. `only` limits the reference checks (entity, about, from,
 * until existing) to those fields, so an edit that does not touch a detached field can still go through.
 */
function checkFact(f: StateFact, ctx: OpContext, only?: ReadonlySet<string>): string | null {
  const need = (k: string): boolean => !only || only.has(k)
  if (!STATE_KINDS.includes(f.kind)) return `"${String(f.kind)}" is not a kind of fact.`
  if (typeof f.value !== 'string' || !f.value) return 'The value is empty.'
  if (f.value.length > VALUE_MAX) return `The value is longer than ${VALUE_MAX} characters.`
  if (/[\r\n]/.test(f.value)) return 'The value must be one line.'
  if (need('entity')) {
    if (typeof f.entity !== 'string' || !ctx.entryIds.has(f.entity)) return `"${String(f.entity)}" is not a bible entry.`
  }
  if (f.kind === 'relationship') {
    if (typeof f.about !== 'string' || !f.about) return 'A relationship needs an "about" entry.'
    if (need('about') && !ctx.entryIds.has(f.about)) return `"${f.about}" is not a bible entry.`
  } else if (f.about != null) return 'Only a relationship has an "about".'
  const from = typeof f.from === 'string' ? ctx.pos.get(f.from) : undefined
  if (need('from') && from === undefined) return `"${String(f.from)}" is not a scene.`
  if (f.until != null) {
    const until = typeof f.until === 'string' ? ctx.pos.get(f.until) : undefined
    if (need('until') && until === undefined) return `"${String(f.until)}" is not a scene.`
    if (need('until') && from === undefined) return 'The scene where this fact starts no longer exists. Edit the fact first.'
    if ((need('from') || need('until')) && from !== undefined && until !== undefined && until < from) return 'The fact would end before it starts.'
  }
  return null
}

function normalised(raw: unknown): StateFact | string {
  if (!isObject(raw)) return 'The fact is not an object.'
  if (typeof raw.id !== 'string' || !raw.id) return 'The fact has no id.'
  if (typeof raw.value !== 'string') return 'The value is empty.'
  return {
    id: raw.id,
    entity: raw.entity as string,
    kind: raw.kind as StateKind,
    value: raw.value.trim(),
    about: raw.about === undefined ? null : (raw.about as string | null),
    from: raw.from as string,
    until: raw.until === undefined ? null : (raw.until as string | null)
  }
}

export interface ApplyResult {
  items: StateItem[]
  /** The file as it is after the ops (a reset makes it valid again). */
  problem: string | null
  canReset: boolean
  applied: number
  rejected: { index: number; reason: string }[]
}

/**
 * Applies the valid ops in order to a copy of `parsed` and lists the rest with reasons. With a problem set only
 * `reset` (when allowed) goes through. Nothing here touches disk.
 */
export function applyStateOps(parsed: ParsedState, ops: unknown[], ctx: OpContext): ApplyResult {
  // Copies of the items: a write that fails or goes stale must leave the held state untouched.
  let items = parsed.doc.items.map((i) => ({ ...i }))
  let problem = parsed.problem
  let canReset = parsed.canReset
  const rejected: { index: number; reason: string }[] = []
  let applied = 0
  const byOrigIndex = (idx: number): StateItem | undefined => items.find((i) => !i.fact && i.index === idx)
  const findFact = (id: unknown): StateItem | undefined => (typeof id === 'string' ? items.find((i) => i.fact?.id === id) : undefined)

  ops.forEach((raw, index) => {
    const reject = (reason: string): void => void rejected.push({ index, reason })
    if (!isObject(raw) || typeof raw.op !== 'string') return reject('That is not a change.')
    const op = raw as unknown as StateOp
    if (op.op === 'reset') {
      if (!problem) return reject('state.json can be read, so there is nothing to reset.')
      if (!canReset) return reject('state.json has content that cannot be read, so it is not replaced. Fix or move the file first.')
      items = []
      problem = null
      canReset = false
      applied++
      return
    }
    if (problem) return reject(problem)
    switch (op.op) {
      case 'add': {
        const f = normalised(op.fact)
        if (typeof f === 'string') return reject(f)
        if (items.some((i) => i.fact?.id === f.id || (!i.fact && isObject(i.orig) && i.orig.id === f.id))) return reject(`The id "${f.id}" is already used.`)
        const why = checkFact(f, ctx)
        if (why) return reject(why)
        if (items.some((i) => i.fact && sameFact(i.fact, f))) return reject('That fact is already recorded.')
        items.push({ orig: {}, fact: f, reason: null, index: -1, touched: true })
        applied++
        return
      }
      case 'end': {
        const it = findFact(op.factId)
        if (!it?.fact) return reject('That fact no longer exists.')
        if (it.fact.until != null) return reject('That fact has already ended.')
        const until = typeof op.until === 'string' ? ctx.pos.get(op.until) : undefined
        if (until === undefined) return reject(`"${String(op.until)}" is not a scene.`)
        const from = ctx.pos.get(it.fact.from)
        if (from === undefined) return reject('The scene where this fact starts no longer exists. Edit the fact first.')
        if (until < from) return reject('The fact would end before it starts.')
        it.fact = { ...it.fact, until: op.until }
        it.touched = true
        applied++
        return
      }
      case 'edit': {
        const it = findFact(op.factId)
        if (!it?.fact) return reject('That fact no longer exists.')
        if (!isObject(op.changes)) return reject('There is nothing to change.')
        const changes = op.changes as Record<string, unknown>
        const cur = it.fact as unknown as Record<string, unknown>
        const merged: Record<string, unknown> = { ...cur }
        const keys: string[] = []
        for (const k of FIELDS) {
          if (!(k in changes) || changes[k] === undefined) continue
          const v = k === 'value' && typeof changes[k] === 'string' ? (changes[k] as string).trim() : changes[k]
          if (v === cur[k]) continue
          merged[k] = v
          keys.push(k)
        }
        if (!keys.length) return reject('There is nothing to change.')
        const next = merged as unknown as StateFact
        const why = checkFact(next, ctx, new Set(keys))
        if (why) return reject(why)
        if (keys.some((k) => k !== 'until') && items.some((i) => i !== it && i.fact && sameFact(i.fact, next))) return reject('That would repeat a fact that is already recorded.')
        it.fact = next
        it.touched = true
        applied++
        return
      }
      case 'remove': {
        const it = findFact(op.factId)
        if (!it) return reject('That fact no longer exists.')
        items = items.filter((i) => i !== it)
        applied++
        return
      }
      case 'removeInvalid': {
        const it = typeof op.index === 'number' ? byOrigIndex(op.index) : undefined
        if (!it) return reject('That entry is no longer in the file.')
        if (sha256(rawOf(it)) !== op.hash) return reject('That entry changed since it was shown. Nothing was removed.')
        items = items.filter((i) => i !== it)
        applied++
        return
      }
      default:
        return reject('That is not a known change.')
    }
  })
  return { items, problem, canReset, applied, rejected }
}

// ---------------------------------------------------------------------------------------------
// .rulz/checks.json
// ---------------------------------------------------------------------------------------------

export interface ChecksDoc {
  top: Record<string, unknown>
  dismissed: Record<string, string[]>
}

export const emptyChecks = (): ChecksDoc => ({ top: { format: 1 }, dismissed: {} })

/** Reads checks.json. Missing or 0 bytes = empty; anything unreadable is a problem (dismissals ignored, never written). */
export function parseChecksBytes(bytes: Buffer | null): { doc: ChecksDoc; problem: string | null } {
  const bad = (problem: string): { doc: ChecksDoc; problem: string } => ({ doc: emptyChecks(), problem })
  if (!bytes || bytes.length === 0) return { doc: emptyChecks(), problem: null }
  let parsed: unknown
  try {
    parsed = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(bytes))
  } catch (err) {
    return bad(`.rulz/checks.json could not be read: ${errText(err)}`)
  }
  if (!isObject(parsed)) return bad('.rulz/checks.json could not be read: it is not a JSON object.')
  if (typeof parsed.format === 'number' && parsed.format > 1) return bad(`.rulz/checks.json is from a newer version of the app (format ${parsed.format}).`)
  const { dismissed, ...top } = parsed
  if (typeof top.format !== 'number') top.format = 1
  if (dismissed === undefined) return { doc: { top, dismissed: {} }, problem: null }
  if (!isObject(dismissed)) return bad('.rulz/checks.json could not be read: "dismissed" is not an object.')
  const out: Record<string, string[]> = {}
  for (const [k, v] of Object.entries(dismissed)) {
    if (!Array.isArray(v) || v.some((x) => typeof x !== 'string')) return bad(`.rulz/checks.json could not be read: the entry for "${k}" is not a list of keys.`)
    Object.defineProperty(out, k, { value: v.slice(), enumerable: true, writable: true, configurable: true })
  }
  return { doc: { top, dismissed: out }, problem: null }
}

export const serializeChecks = (doc: ChecksDoc): unknown => ({ ...doc.top, dismissed: doc.dismissed })
