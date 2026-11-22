// Recording of Writing AI actions and their export as training data (design: docs/writing-mode-design.md, "Stage 4
// design"). The pure parts (line validation, the join rules, the train and pair rows) have no disk access, so they
// are unit-tested alone; the file parts take a path that the store has already guarded. No Electron here.
import { createReadStream, promises as fs } from 'node:fs'
import {
  RECORD_FORMAT,
  type NovelAction,
  type RecordAlt,
  type RecordFinalLine,
  type RecordFinalReason,
  type RecordFinish,
  type RecordLine,
  type RecordMessage,
  type RecordOutcomeKind,
  type RecordOutcomeLine,
  type RecordRegion,
  type RecordRequestLine,
  type RecordViewedLine,
  type RecordsExportOptions
} from '@shared/novel'

/** One record line may be at most this many bytes (serialized, without the newline). */
export const MAX_RECORD_BYTES = 8 * 1024 * 1024
/** Assistant text shorter than this (after trimming) is not training data. */
export const MIN_TRAIN_CHARS = 20

export const PROSE_ACTIONS: readonly NovelAction[] = ['continue', 'rewrite', 'expand', 'condense', 'dialogue', 'draftScene']
export const CARD_ACTIONS: readonly NovelAction[] = ['summarizeScene', 'summarizeChapter', 'outlineChapters', 'outlineScenes']
const RECORDED_ACTIONS = new Set<string>([...PROSE_ACTIONS, ...CARD_ACTIONS])
const FINISHES = new Set<string>(['stop', 'length', 'stopped', 'error'])
const ROLES = new Set<string>(['system', 'user', 'assistant'])
const OUTCOME_KINDS = new Set<string>(['accepted', 'mixed', 'rejected', 'abandoned', 'edited'])
const FINAL_REASONS = new Set<string>(['idle', 'left', 'action', 'close', 'reloaded', 'outside'])
const ID = /^[0-9a-f]{16}$/

const isObject = (v: unknown): v is Record<string, unknown> => typeof v === 'object' && v !== null && !Array.isArray(v)
const isIndex = (v: unknown): v is number => typeof v === 'number' && Number.isInteger(v) && v >= 0
const isStr = (v: unknown): v is string => typeof v === 'string'

export const isCardAction = (a: string): boolean => (CARD_ACTIONS as readonly string[]).includes(a)

// ---------------------------------------------------------------------------------------------
// Validation (strict, per type). Returns a normalized copy that holds only the known fields, or null.
// ---------------------------------------------------------------------------------------------

function validAt(v: unknown): v is string {
  return isStr(v) && v.length > 0 && v.length <= 64 && !Number.isNaN(Date.parse(v))
}

function cleanAlts(raw: unknown): RecordAlt[] | null {
  if (!Array.isArray(raw) || raw.length === 0 || raw.length > 64) return null
  const out: RecordAlt[] = []
  const seen = new Set<number>()
  for (const a of raw) {
    if (!isObject(a) || !isIndex(a.index) || seen.has(a.index)) return null
    if (typeof a.seed !== 'number' || !Number.isFinite(a.seed) || !isStr(a.generated) || !isStr(a.finish) || !FINISHES.has(a.finish)) return null
    seen.add(a.index)
    out.push({ index: a.index, seed: a.seed, generated: a.generated, finish: a.finish as RecordFinish })
  }
  return out
}

function cleanMessages(raw: unknown): RecordMessage[] | null {
  if (!Array.isArray(raw) || raw.length === 0) return null
  const out: RecordMessage[] = []
  for (const m of raw) {
    if (!isObject(m) || !isStr(m.role) || !ROLES.has(m.role) || !isStr(m.content)) return null
    out.push({ role: m.role as RecordMessage['role'], content: m.content })
  }
  return out
}

function cleanRegion(raw: unknown): RecordRegion | null | undefined {
  if (raw === undefined) return undefined
  if (raw === null) return null
  if (!isObject(raw) || !isIndex(raw.start) || !isIndex(raw.end) || raw.end < raw.start) throw new Error('region')
  return { start: raw.start, end: raw.end }
}

export function validateRecordLine(raw: unknown): RecordLine | null {
  try {
    if (!isObject(raw) || raw.format !== RECORD_FORMAT || !isStr(raw.id) || !ID.test(raw.id) || !validAt(raw.at)) return null
    const base = { format: 1 as const, id: raw.id, at: raw.at }
    switch (raw.type) {
      case 'request': {
        if (!isStr(raw.app) || raw.app.length > 64 || !isStr(raw.action) || !RECORDED_ACTIONS.has(raw.action)) return null
        if (raw.variant !== undefined && !isStr(raw.variant)) return null
        if (!(raw.sceneId === null || isStr(raw.sceneId)) || !(raw.chapterId === null || isStr(raw.chapterId))) return null
        if (!isObject(raw.model) || !isStr(raw.model.id) || !isStr(raw.model.name)) return null
        if (!isObject(raw.sampling)) return null
        if (!Array.isArray(raw.adapters)) return null
        const adapters: { name: string; scale: number }[] = []
        for (const a of raw.adapters) {
          if (!isObject(a) || !isStr(a.name) || typeof a.scale !== 'number' || !Number.isFinite(a.scale)) return null
          adapters.push({ name: a.name, scale: a.scale })
        }
        const messages = cleanMessages(raw.messages)
        const alts = cleanAlts(raw.alts)
        if (!messages || !alts) return null
        const line: RecordRequestLine = {
          ...base,
          type: 'request',
          app: raw.app,
          action: raw.action as NovelAction,
          ...(raw.variant !== undefined ? { variant: raw.variant } : {}),
          sceneId: raw.sceneId,
          chapterId: raw.chapterId,
          model: { id: raw.model.id, name: raw.model.name },
          sampling: raw.sampling as unknown as RecordRequestLine['sampling'],
          adapters,
          messages,
          alts
        }
        return line
      }
      case 'viewed': {
        if (!Array.isArray(raw.alts) || raw.alts.length > 64 || !raw.alts.every(isIndex)) return null
        const line: RecordViewedLine = { ...base, type: 'viewed', alts: raw.alts as number[] }
        return line
      }
      case 'outcome': {
        if (!isStr(raw.kind) || !OUTCOME_KINDS.has(raw.kind)) return null
        if (raw.alt !== undefined && raw.alt !== null && !isIndex(raw.alt)) return null
        if (raw.replaced !== undefined && !isStr(raw.replaced)) return null
        if (raw.text !== undefined && !isStr(raw.text)) return null
        const region = cleanRegion(raw.region)
        const line: RecordOutcomeLine = {
          ...base,
          type: 'outcome',
          kind: raw.kind as RecordOutcomeKind,
          ...(raw.alt !== undefined ? { alt: raw.alt } : {}),
          ...(region !== undefined ? { region } : {}),
          ...(raw.replaced !== undefined ? { replaced: raw.replaced } : {}),
          ...(raw.text !== undefined ? { text: raw.text } : {})
        }
        return line
      }
      case 'final': {
        if (!(raw.text === null || isStr(raw.text)) || !isStr(raw.reason) || !FINAL_REASONS.has(raw.reason)) return null
        const line: RecordFinalLine = { ...base, type: 'final', text: raw.text, reason: raw.reason as RecordFinalReason }
        return line
      }
      default:
        return null
    }
  } catch {
    return null
  }
}

// ---------------------------------------------------------------------------------------------
// Dates
// ---------------------------------------------------------------------------------------------

const two = (n: number): string => String(n).padStart(2, '0')

/** YYYY-MM-DD in local time (the name of a day's records file). */
export const localDate = (d: Date): string => `${d.getFullYear()}-${two(d.getMonth() + 1)}-${two(d.getDate())}`

/** YYYYMMDD-HHMMSS in local time (the export subfolder's suffix). */
export const localStamp = (d: Date): string => `${d.getFullYear()}${two(d.getMonth() + 1)}${two(d.getDate())}-${two(d.getHours())}${two(d.getMinutes())}${two(d.getSeconds())}`

// ---------------------------------------------------------------------------------------------
// Appending
// ---------------------------------------------------------------------------------------------

const sleep = (ms: number): Promise<void> => new Promise((r) => setTimeout(r, ms))
const TRANSIENT = new Set(['EBUSY', 'EPERM', 'EACCES'])
/** About 2 s in all. */
export const APPEND_WAITS = [50, 100, 150, 200, 300, 400, 400, 400]

/** Runs `fn`, retrying EBUSY / EPERM / EACCES after each wait; throws the last error. */
export async function retryLock<T>(fn: () => Promise<T>, waits: readonly number[] = APPEND_WAITS): Promise<T> {
  for (let attempt = 0; ; attempt++) {
    try {
      return await fn()
    } catch (err) {
      const code = (err as NodeJS.ErrnoException).code ?? ''
      if (!TRANSIENT.has(code) || attempt >= waits.length) throw err
      await sleep(waits[attempt])
    }
  }
}

/**
 * Appends `text` + '\n' to `path`, after a '\n' when the file does not end with one (a torn last line from a crash
 * must not swallow the new line). The folder must exist. Each attempt is a whole open / look / write / close.
 */
export function appendRecordText(path: string, text: string, waits: readonly number[] = APPEND_WAITS): Promise<void> {
  return retryLock(async () => {
    const fh = await fs.open(path, 'a+')
    try {
      const { size } = await fh.stat()
      let prefix = ''
      if (size > 0) {
        const last = Buffer.alloc(1)
        await fh.read(last, 0, 1, size - 1)
        if (last[0] !== 0x0a) prefix = '\n'
      }
      await fh.appendFile(`${prefix}${text}\n`, 'utf8')
    } finally {
      await fh.close()
    }
  }, waits)
}

// ---------------------------------------------------------------------------------------------
// Reading: raw lines, so each one can be checked as strict UTF-8
// ---------------------------------------------------------------------------------------------

/** Lines of a file as bytes (without the '\n'); null for a line longer than the limit (skipped to its end). */
export async function* readRawLines(path: string, maxBytes = MAX_RECORD_BYTES): AsyncGenerator<Buffer | null> {
  const stream = createReadStream(path, { highWaterMark: 1 << 20 })
  let pending: Buffer[] = []
  let pendingLen = 0
  let overflow = false
  for await (const chunk of stream as AsyncIterable<Buffer>) {
    let start = 0
    for (;;) {
      const nl = chunk.indexOf(0x0a, start)
      const part = nl < 0 ? chunk.subarray(start) : chunk.subarray(start, nl)
      if (!overflow) {
        pending.push(part)
        pendingLen += part.length
        if (pendingLen > maxBytes) {
          overflow = true
          pending = []
          pendingLen = 0
        }
      }
      if (nl < 0) break
      yield overflow ? null : Buffer.concat(pending)
      overflow = false
      pending = []
      pendingLen = 0
      start = nl + 1
    }
  }
  if (overflow) yield null
  else if (pendingLen > 0) yield Buffer.concat(pending)
}

export type LineResult = { ok: true; line: RecordLine } | { ok: false; reason: 'tooLong' | 'badUtf8' | 'badJson' | 'badShape' } | { ok: 'blank' }

/** One raw line → a validated record, or the reason it was dropped. */
export function parseRecordBytes(bytes: Buffer | null): LineResult {
  if (bytes === null) return { ok: false, reason: 'tooLong' }
  let text: string
  try {
    text = new TextDecoder('utf-8', { fatal: true }).decode(bytes)
  } catch {
    return { ok: false, reason: 'badUtf8' }
  }
  if (text.replace(/^﻿/, '').trim() === '') return { ok: 'blank' }
  let raw: unknown
  try {
    raw = JSON.parse(text)
  } catch {
    return { ok: false, reason: 'badJson' }
  }
  const line = validateRecordLine(raw)
  return line ? { ok: true, line } : { ok: false, reason: 'badShape' }
}

// ---------------------------------------------------------------------------------------------
// The join and the export rows
// ---------------------------------------------------------------------------------------------

/** What pass 1 keeps per id: the first outcome, every viewed index, the last final. */
export interface RecordMeta {
  outcome: RecordOutcomeLine | null
  viewed: Set<number>
  final: RecordFinalLine | null
}

export class RecordJoin {
  private metas = new Map<string, RecordMeta>()

  get size(): number {
    return this.metas.size
  }

  /** Outcome: the first wins. Viewed: merged. Final: the last wins. Requests are ignored here. */
  add(line: RecordLine): void {
    if (line.type === 'request') return
    let m = this.metas.get(line.id)
    if (!m) {
      m = { outcome: null, viewed: new Set(), final: null }
      this.metas.set(line.id, m)
    }
    if (line.type === 'outcome') m.outcome ??= line
    else if (line.type === 'viewed') for (const i of line.alts) m.viewed.add(i)
    else m.final = line
  }

  get(id: string): RecordMeta | undefined {
    return this.metas.get(id)
  }

  ids(): string[] {
    return [...this.metas.keys()]
  }
}

export interface TrainRow {
  messages: RecordMessage[]
  meta: { id: string; action: NovelAction; at: string; model: { id: string; name: string }; kind: RecordOutcomeKind; final: boolean }
}

export interface PairRow {
  prompt: RecordMessage[]
  chosen: string
  rejected: string
  meta: { id: string; source: 'alts' | 'edit' }
}

export interface RequestExport {
  train: TrainRow | null
  pairs: PairRow[]
  /** Why the request gave no train row (also set when only the pairs were affected), or null. */
  drop: string | null
}

/**
 * The export rows of one request (prose actions only; cards are the caller's business).
 * Drop reasons: noOutcome, rejected (also a reverted final), abandoned, badOutcome, noFinal, tooShort.
 */
export function exportRequest(req: RecordRequestLine, meta: RecordMeta | undefined, opts: RecordsExportOptions): RequestExport {
  const none = (drop: string): RequestExport => ({ train: null, pairs: [], drop })
  const outcome = meta?.outcome
  if (!meta || !outcome) return none('noOutcome')
  if (outcome.kind === 'rejected') return none('rejected')
  if (outcome.kind === 'abandoned') return none('abandoned')
  if (outcome.kind !== 'accepted' && outcome.kind !== 'mixed') return none('badOutcome')
  const accepted = outcome.kind === 'accepted'
  const alt = typeof outcome.alt === 'number' ? req.alts.find((a) => a.index === outcome.alt) : undefined
  if (accepted && !alt) return none('badOutcome')

  const finalText = meta.final && meta.final.text !== null ? meta.final.text : null
  // A final equal to the text the region replaced means the user undid the whole change: it counts as rejected.
  if (finalText !== null && finalText === (outcome.replaced ?? '')) return none('rejected')

  let content: string
  let hasFinal: boolean
  if (finalText !== null) {
    content = finalText
    hasFinal = true
  } else if (accepted && alt) {
    content = alt.generated
    hasFinal = false
  } else {
    return none('noFinal')
  }

  const out: RequestExport = { train: null, pairs: [], drop: null }
  if (content.trim().length < MIN_TRAIN_CHARS) out.drop = 'tooShort'
  else {
    out.train = {
      messages: [...req.messages, { role: 'assistant', content }],
      meta: { id: req.id, action: req.action, at: req.at, model: req.model, kind: outcome.kind, final: hasFinal }
    }
  }

  // Pairs only from an accepted alternative that ended normally: a cut-off one is never chosen or rejected.
  if (accepted && alt && alt.finish === 'stop') {
    const chosen = alt.generated
    for (const other of req.alts) {
      if (other.index === alt.index) continue
      // Only alternatives the user looked at, that ended normally (a cut-off one says nothing about quality).
      if (!meta.viewed.has(other.index) || other.finish !== 'stop') continue
      if (other.generated.trim() === chosen.trim()) continue
      if (chosen.trim().length < MIN_TRAIN_CHARS || other.generated.trim().length < MIN_TRAIN_CHARS) continue
      out.pairs.push({ prompt: req.messages, chosen, rejected: other.generated, meta: { id: req.id, source: 'alts' } })
    }
    if (opts.editPairs && finalText !== null) {
      const same = finalText.trim() === chosen.trim()
      // Text only added after the accepted text is a continuation, not a correction.
      const appendOnly = finalText.length > chosen.length && finalText.startsWith(chosen)
      const long = finalText.trim().length >= MIN_TRAIN_CHARS && chosen.trim().length >= MIN_TRAIN_CHARS
      if (!same && !appendOnly && long) out.pairs.push({ prompt: req.messages, chosen: finalText, rejected: chosen, meta: { id: req.id, source: 'edit' } })
    }
  }
  return out
}

/** Adds one to a counter. */
export function bump(counts: Record<string, number>, key: string, by = 1): void {
  counts[key] = (counts[key] ?? 0) + by
}

// ---------------------------------------------------------------------------------------------
// Output files: written to a temp name in the same folder, then renamed
// ---------------------------------------------------------------------------------------------

export class LineSink {
  private fh: import('node:fs/promises').FileHandle | null = null
  private buf: string[] = []
  private bufLen = 0
  count = 0

  constructor(
    private final: string,
    private tmp: string
  ) {}

  async open(): Promise<void> {
    this.fh = await fs.open(this.tmp, 'wx')
  }

  async write(text: string): Promise<void> {
    this.buf.push(text, '\n')
    this.bufLen += text.length + 1
    this.count++
    if (this.bufLen >= 1 << 20) await this.flush()
  }

  private async flush(): Promise<void> {
    if (!this.fh || this.buf.length === 0) return
    const data = this.buf.join('')
    this.buf = []
    this.bufLen = 0
    await this.fh.writeFile(data, 'utf8')
  }

  async commit(): Promise<void> {
    if (!this.fh) throw new Error('The output file is not open.')
    await this.flush()
    await this.fh.sync()
    await this.fh.close()
    this.fh = null
    await retryLock(() => fs.rename(this.tmp, this.final))
  }

  async abort(): Promise<void> {
    await this.fh?.close().catch(() => undefined)
    this.fh = null
    await fs.rm(this.tmp, { force: true }).catch(() => undefined)
  }
}
