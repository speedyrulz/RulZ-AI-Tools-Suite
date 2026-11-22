// Writing mode, renderer side of stage 4: the pure parts of recording (ids, outcome, region, final text,
// line builders) and the fire-and-forget send. Design: docs/writing-mode-design.md, "Stage 4 design".
// Nothing here may block, delay or change the user's writing: a failed send is told once, quietly, and ignored.
import { toast } from 'sonner'
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
  type RecordViewedLine
} from '@shared/novel'
import type { SamplingParams } from '@shared/types'
import { invoke } from './api'

/** 16 lowercase hex characters, from the platform's random source. */
export function recordId(): string {
  const bytes = new Uint8Array(8)
  crypto.getRandomValues(bytes)
  return Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('')
}

export const isRecordId = (s: string): boolean => /^[0-9a-f]{16}$/.test(s)

// ---------------------------------------------------------------------------------------------
// Outcome and geometry
// ---------------------------------------------------------------------------------------------

export type ProseOutcome = 'accepted' | 'mixed' | 'rejected'

/**
 * What the user did with a prose change: nothing kept (final is the base), everything kept (final is the
 * active alternative's whole document), or some hunks. The base wins when both are equal: a change that
 * alters nothing is not an acceptance.
 */
export function classifyOutcome(base: string, final: string, altDoc: string): ProseOutcome {
  if (final === base) return 'rejected'
  if (final === altDoc) return 'accepted'
  return 'mixed'
}

export interface Geometry {
  /** Where the alternative's generated text starts in its document. */
  start: number
  /** Where it ends in the alternative's document. */
  end: number
  /** The base text the region replaced: the selection for rewrite, '' for continue, the old text for a draft. */
  replaced: string
}

/**
 * Where an alternative's generated text was placed, from the same rules as `applyResult` (not from a diff):
 * Draft scene replaces the whole text, rewrite / expand / condense replace the selection, the others go at
 * the cursor after any blank-line lead.
 */
export function regionOf(action: NovelAction, text: string, cursor: number, selection: { from: number; to: number } | null, generated: string): Geometry {
  const gen = generated.trim()
  if (action === 'draftScene') return { start: 0, end: gen.length, replaced: text }
  if ((action === 'rewrite' || action === 'expand' || action === 'condense') && selection) {
    return { start: selection.from, end: selection.from + gen.length, replaced: text.slice(selection.from, selection.to) }
  }
  const at = Math.min(cursor, text.length)
  const before = text.slice(0, at)
  const lead = before.length === 0 || before.endsWith('\n\n') ? '' : before.endsWith('\n') ? '\n' : '\n\n'
  const start = at + lead.length
  return { start, end: start + gen.length, replaced: '' }
}

/**
 * The region in the document the user ended with. Accepted: the alternative's own region. Mixed: the same
 * start, and the end is the final length minus the length of the text after the region in the alternative's
 * document (every hunk lies inside the region).
 */
export function outcomeRegion(kind: 'accepted' | 'mixed', geo: Geometry, altDocLength: number, finalLength: number): RecordRegion {
  if (kind === 'accepted') return { start: geo.start, end: geo.end }
  const tail = altDocLength - geo.end
  return { start: geo.start, end: Math.max(geo.start, finalLength - tail) }
}

/**
 * The region's text after the user's later edits, or null when it cannot be told: the later document must
 * start with accepted[0:start] and end with accepted[end:].
 */
export function locateFinal(accepted: string, region: RecordRegion, later: string): string | null {
  const head = accepted.slice(0, region.start)
  const tail = accepted.slice(region.end)
  if (later.length < head.length + tail.length) return null
  if (!later.startsWith(head) || !later.endsWith(tail)) return null
  return later.slice(head.length, later.length - tail.length)
}

// ---------------------------------------------------------------------------------------------
// Finish reasons
// ---------------------------------------------------------------------------------------------

/**
 * The finish of one alternative from the chat path's stop reason: `length` when the server cut the reply at
 * the token limit, `stopped` when the user pressed Stop, `error` on an error, else `stop`.
 */
export function finishOf(stopReason: string | undefined, o: { stopped: boolean; error: string | null }): RecordFinish {
  if (o.error) return 'error'
  if (o.stopped || stopReason === 'stopped') return 'stopped'
  if (stopReason === 'length') return 'length'
  return 'stop'
}

// ---------------------------------------------------------------------------------------------
// Line builders
// ---------------------------------------------------------------------------------------------

export interface RequestInput {
  id: string
  at: string
  app: string
  action: NovelAction
  variant?: string
  sceneId: string | null
  chapterId: string | null
  model: { id: string; name: string }
  sampling: SamplingParams
  adapters: { name: string; scale: number }[]
  messages: { role: string; content: unknown }[]
  alts: RecordAlt[]
}

export function requestLine(i: RequestInput): RecordRequestLine {
  const messages: RecordMessage[] = i.messages.map((m) => ({ role: m.role as RecordMessage['role'], content: typeof m.content === 'string' ? m.content : JSON.stringify(m.content) }))
  const line: RecordRequestLine = {
    format: RECORD_FORMAT,
    type: 'request',
    id: i.id,
    at: i.at,
    app: i.app,
    action: i.action,
    sceneId: i.sceneId,
    chapterId: i.chapterId,
    model: i.model,
    sampling: i.sampling,
    adapters: i.adapters,
    messages,
    alts: i.alts
  }
  if (i.variant) line.variant = i.variant
  return line
}

export function viewedLine(id: string, at: string, alts: number[]): RecordViewedLine {
  return { format: RECORD_FORMAT, type: 'viewed', id, at, alts: [...new Set(alts)].sort((a, b) => a - b) }
}

export function outcomeLine(id: string, at: string, kind: RecordOutcomeKind, extra: { alt?: number | null; region?: RecordRegion | null; replaced?: string; text?: string } = {}): RecordOutcomeLine {
  const line: RecordOutcomeLine = { format: RECORD_FORMAT, type: 'outcome', id, at, kind }
  if (extra.alt !== undefined) line.alt = extra.alt
  if (extra.region !== undefined) line.region = extra.region
  if (extra.replaced !== undefined) line.replaced = extra.replaced
  if (extra.text !== undefined) line.text = extra.text
  return line
}

export function finalLine(id: string, at: string, text: string | null, reason: RecordFinalReason): RecordFinalLine {
  return { format: RECORD_FORMAT, type: 'final', id, at, text, reason }
}

// ---------------------------------------------------------------------------------------------
// Sending
// ---------------------------------------------------------------------------------------------

let told = false

/** Forgets that the failure was told (tests; a new session starts with it forgotten anyway). */
export function resetRecordWarning(): void {
  told = false
}

function tellOnce(): void {
  if (told) return
  told = true
  try {
    toast.message('Recording failed. Your writing is not affected.')
  } catch {
    // A toast that cannot show is not worth a second failure.
  }
}

/**
 * Sends one line to main. Never throws and never rejects: resolves true when it was written, false when not
 * (a false answer or a throw shows one quiet toast per session). Callers do not await it on the writing path.
 */
export function sendRecord(folder: string, line: RecordLine): Promise<boolean> {
  let p: Promise<boolean>
  try {
    p = Promise.resolve(invoke('novel:record', folder, line))
  } catch {
    tellOnce()
    return Promise.resolve(false)
  }
  return p.then(
    (ok) => {
      // main answers false when recording is off or the project is closed: only an explicit false is a failure.
      if (ok === false) {
        tellOnce()
        return false
      }
      return true
    },
    () => {
      tellOnce()
      return false
    }
  )
}

let appVersion: string | null = null

/** The app's version, asked once; '' when it cannot be had. */
/** The app version if it was already fetched, else '' (appVersionText fetches it). */
export const appVersionNow = (): string => appVersion ?? ''

export async function appVersionText(): Promise<string> {
  if (appVersion !== null) return appVersion
  try {
    const info = await invoke('app:info')
    appVersion = typeof info?.version === 'string' ? info.version : ''
  } catch {
    appVersion = ''
  }
  return appVersion
}
