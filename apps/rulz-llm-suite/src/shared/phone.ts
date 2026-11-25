// Phone access for RulZ Writer for Android (apps/rulz-writer-android).
// Design: docs/phone-access-design.md. This file is the contract between the desktop and the phone app;
// the phone copies it verbatim with its sync-shared script, so it may only import from other src/shared files.
//
// Transport (same shape as RulZ Coding Agent's phone access): a WebSocket on this computer's Tailscale
// address only, path PHONE_PATH, port setting phonePort (default PHONE_DEFAULT_PORT).
//
// Phone → computer:  { t: 'hello', token, device, protocol }   then   { t: 'invoke', id, channel, args }
// Computer → phone:  { t: 'welcome', protocol, version, name }  { t: 'result', id, ok, value | error }
//                    { t: 'event', channel, payload }           { t: 'denied', reason }
//
// Only the channels in PHONE_INVOKE are answered, and only the events in PHONE_EVENTS are sent.
import { PHONE_ADVENTURE_CHANNELS, type AdventureIpcEvents, type AdventureIpcInvoke } from './adventure'
import type {
  BibleEntry,
  CheckFinding,
  NovelAction,
  NovelChange,
  NovelIpcInvoke,
  OutlineChaptersReply,
  OutlineScenesReply,
  RewriteVariant,
  StateFact,
  StateKind,
  SummaryReply
} from './novel'

export const PHONE_PROTOCOL = 1
export const PHONE_PATH = '/rulz-writer'
/** RulZ Coding Agent uses 7420, so both can be on at once. */
export const PHONE_DEFAULT_PORT = 7421
/** Pairing link scheme: rulzwriter://pair?host=…&port=…&token=…&name=… */
export const PHONE_LINK_SCHEME = 'rulzwriter'

// ---------------------------------------------------------------------------------------------
// Wire messages
// ---------------------------------------------------------------------------------------------

export type PhoneToComputer =
  | { t: 'hello'; token: string; device: string; protocol: number }
  | { t: 'invoke'; id: string; channel: string; args: unknown[] }

export type ComputerToPhone =
  | { t: 'welcome'; protocol: number; version: string; name: string }
  | { t: 'result'; id: string; ok: true; value: unknown }
  | { t: 'result'; id: string; ok: false; error: string }
  | { t: 'event'; channel: PhoneEventChannel; payload: unknown }
  | { t: 'denied'; reason: string }

// ---------------------------------------------------------------------------------------------
// Status and pairing (computer side, shown in LLM Suite's Settings)
// ---------------------------------------------------------------------------------------------

export interface PhoneClient {
  id: string
  device: string
  address: string
  connectedAt: number
}

export interface PhoneStatus {
  enabled: boolean
  state: 'off' | 'starting' | 'on' | 'no-tailscale' | 'error'
  port: number
  /** This computer's Tailscale addresses (IPv4). */
  addresses: string[]
  /** This computer's Tailscale (MagicDNS) name, when known. */
  dnsName?: string
  clients: PhoneClient[]
  error?: string
}

/** What the phone needs to connect: a link (also shown as a QR code) and the parts to type by hand. */
export interface PhonePairing {
  url: string
  qrSvg: string
  host: string
  port: number
  token: string
}

/** Desktop-only IPC for the Settings page (never answered over the phone socket). */
export interface PhoneIpcInvoke {
  'phone:status': () => PhoneStatus
  'phone:setEnabled': (enabled: boolean) => PhoneStatus
  'phone:pairing': () => PhonePairing
  /** Makes a new pairing code; every paired phone must pair again. */
  'phone:newToken': () => PhonePairing
  'phone:disconnect': (clientId: string) => void
}

export interface PhoneIpcEvents {
  'phone:changed': PhoneStatus
}

// ---------------------------------------------------------------------------------------------
// Writing runs: AI actions started from the phone, run as jobs in main
// ---------------------------------------------------------------------------------------------

/** What the phone asks for. The text is the phone's buffer (may hold unsaved edits); baseHash is the hash of the
 *  disk text that buffer was loaded from or last saved as. */
export interface WritingRunRequest {
  folder: string
  action: NovelAction
  /** Scene actions; null for project-level actions (outlineChapters). */
  sceneId: string | null
  /** summarizeChapter, outlineScenes and a chapter check. */
  chapterId?: string
  text: string
  baseHash: string
  cursor: number
  selection: { from: number; to: number } | null
  note: string
  variant?: RewriteVariant
  /** Prose actions: alternatives to write, 1–4. */
  alternatives?: number
  /** Wanted chapters / scenes for the outline actions. */
  count?: number
  /** interview: the character and the conversation so far (the new question is the last user turn). */
  interview?: { characterId: string; turns: { role: 'user' | 'assistant'; text: string }[] }
}

export type WritingJobStatus = 'queued' | 'preparing' | 'waiting' | 'writing' | 'done' | 'error' | 'stopped'

/** One alternative of a prose action. `text` is the whole scene if it is accepted (applyResult on the request's
 *  text), `generated` what the model wrote. */
export interface WritingAlt {
  text: string
  generated: string
  seed: number
}

/** A state change the model proposed, resolved against the bible and the facts (nothing written yet). */
export interface WritingStateItem {
  key: string
  label: string
  op: 'add' | 'end'
  /** add: the fact to add (id filled in by main on apply). */
  fact?: Omit<StateFact, 'id'>
  /** end: the fact to end. */
  factId?: string
  reason?: string
}

export type WritingResult =
  | { kind: 'prose'; alts: WritingAlt[]; newSticky: string[] }
  | { kind: 'summary'; target: 'scene' | 'chapter'; targetId: string; reply: SummaryReply; summaryOf: string }
  | { kind: 'chapters'; reply: OutlineChaptersReply; unresolved: string[] }
  | { kind: 'scenes'; chapterId: string; reply: OutlineScenesReply; unresolved: string[] }
  | { kind: 'check'; sceneId: string; findings: CheckFinding[] }
  | { kind: 'state'; sceneId: string; textHash: string; items: WritingStateItem[]; left: string[] }
  | { kind: 'interview'; characterId: string; reply: string }

export interface WritingJob {
  id: string
  folder: string
  request: Omit<WritingRunRequest, 'text'>
  /** Short label for the action, e.g. "Continue", "Draft scene". */
  label: string
  status: WritingJobStatus
  /** Which alternative is being written (1-based) and how many. */
  alt: number
  of: number
  /** Prompt processing 0..1 while the model reads the request. */
  progress: number | null
  /** Text written so far for the alternative being written (raw, for live display). */
  partial: string
  result: WritingResult | null
  error: string | null
  /** Warnings from building the request (cut entries, estimated counts, stale summaries). */
  warnings: string[]
  startedAt: number
  endedAt: number | null
}

/** Phone channels for runs. Main keeps every job (per folder) until it is dismissed, so a phone that slept or
 *  reconnected gets finished results back with writing:jobs. */
export interface WritingIpcInvoke {
  /** Starts a job; refused while another phone job runs on the computer, or while the desktop is generating. */
  'writing:run': (req: WritingRunRequest) => WritingJob
  'writing:jobs': (folder: string) => WritingJob[]
  'writing:job': (jobId: string) => WritingJob | null
  'writing:abort': (jobId: string) => void
  /** Forgets a finished job (accepted, rejected or seen). */
  'writing:dismiss': (jobId: string) => void
  /** Applies chosen items of a 'state' job to state.json; refused when the scene's disk text no longer hashes to the
   *  job's textHash. Returns the result of novel:statePatch. */
  'writing:applyState': (jobId: string, itemKeys: string[]) => import('./novel').NovelStatePatchResult
  /** The model library, for the project's model picker (the project's modelId is set with novel:projectPatch). A model on a
   *  rental has `rental` (label null = it is not loaded on one now). */
  'writing:models': () => PhoneModel[]
  /** What applies to a mode with a model: its own settings, and whether the loaded model runs with others. modelId null =
   *  the model a job with no model of its own would use (the newest loaded one). */
  'writing:mode': (mode: PhoneMode, modelId: string | null) => PhoneModeInfo
  /** Loads the model again with the mode's load settings; refused for a rental, while the computer is busy or while
   *  RulZ Coding Agent uses the model. Returns the mode's info as it is right after the start. */
  'writing:reloadMode': (mode: PhoneMode, modelId: string) => PhoneModeInfo
}

export interface PhoneModel {
  id: string
  name: string
  loaded: boolean
  rental?: { label: string | null; reconnecting: boolean }
}

export type PhoneMode = 'writing' | 'adventure'
export const PHONE_MODES: readonly PhoneMode[] = ['writing', 'adventure']

export interface PhoneModeInfo {
  mode: PhoneMode
  /** The model that would write: the given one, else the newest loaded one; null when there is none. */
  modelId: string | null
  name: string
  loaded: boolean
  rental?: { label: string | null; reconnecting: boolean }
  ownLoad: boolean
  ownSampling: boolean
  hasPrompt: boolean
  /** Set when it runs with other settings than the mode's. `blocked`: why it can't be reloaded from here right now. */
  reload: { label: string; blocked: string | null } | null
}

export interface WritingIpcEvents {
  /** Sent on every status change and, throttled to about 4 per second, while text streams. */
  'writing:job': WritingJob
}

// ---------------------------------------------------------------------------------------------
// What the phone may call
// ---------------------------------------------------------------------------------------------

/** novel:* channels the phone may call, answered by the same handlers as the window. Everything else is refused.
 *  Left out on purpose: createProject is allowed, but exportRecords/deleteRecords/record/recordsInfo (desktop-only
 *  training data), deleteProject/emptyTrash (destructive, desktop only), journal/discardJournal (one journal per
 *  file belongs to the desktop editor; the phone keeps its own unsaved draft), flushDone, root. */
export const PHONE_NOVEL_CHANNELS = [
  'novel:listProjects',
  'novel:createProject',
  'novel:openProject',
  'novel:closeProject',
  'novel:readFile',
  'novel:saveFile',
  'novel:outlinePatch',
  'novel:projectPatch',
  'novel:createEntry',
  'novel:deleteScene',
  'novel:deleteChapter',
  'novel:deleteEntry',
  'novel:history',
  'novel:readSnapshot',
  'novel:trash',
  'novel:restoreTrash',
  'novel:statePatch',
  'novel:dismissFinding'
] as const satisfies readonly (keyof NovelIpcInvoke)[]

export const PHONE_WRITING_CHANNELS = [
  'writing:run',
  'writing:jobs',
  'writing:job',
  'writing:abort',
  'writing:dismiss',
  'writing:applyState',
  'writing:models',
  'writing:mode',
  'writing:reloadMode'
] as const satisfies readonly (keyof WritingIpcInvoke)[]

// Adventure mode (src/shared/adventure.ts): PHONE_ADVENTURE_CHANNELS leaves out the desktop-only deletes and the save
// dialog. A computer older than the phone refuses these channels ("That is not available from the phone."); the
// phone says to update RulZ LLM Suite then. PHONE_PROTOCOL stays 1, so older phones keep working with newer computers.
export const PHONE_INVOKE: ReadonlySet<string> = new Set<string>([...PHONE_NOVEL_CHANNELS, ...PHONE_WRITING_CHANNELS, ...PHONE_ADVENTURE_CHANNELS])

export type PhoneEventChannel = 'novel:changed' | 'writing:job' | 'adventure:job' | 'adventure:changed'
export const PHONE_EVENTS: ReadonlySet<string> = new Set<PhoneEventChannel>(['novel:changed', 'writing:job', 'adventure:job', 'adventure:changed'])

/** Typed view of what the phone can call (for the phone's remote.invoke). */
export type PhoneInvoke = Pick<NovelIpcInvoke, (typeof PHONE_NOVEL_CHANNELS)[number]> &
  WritingIpcInvoke &
  Pick<AdventureIpcInvoke, (typeof PHONE_ADVENTURE_CHANNELS)[number]>
export type PhoneInvokeChannel = keyof PhoneInvoke

export interface PhoneEvents extends AdventureIpcEvents {
  'novel:changed': NovelChange
  'writing:job': WritingJob
}

// Re-exported so the phone needs only this file's imports for the common types.
export type { BibleEntry, StateKind }
