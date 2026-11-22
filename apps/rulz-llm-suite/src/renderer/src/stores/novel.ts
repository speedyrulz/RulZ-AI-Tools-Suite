// Writing mode state: the open project, the text buffers with autosave / journal / conflict handling,
// AI actions with pending changes, and structured proposals. All file work goes through the novel:*
// IPC channels (rules: docs/writing-mode-design.md, "Data safety rules"); nothing here writes a
// whole outline or project object back, only patches.
import { create } from 'zustand'
import { toast } from 'sonner'
import type {
  BibleEntry,
  BibleKind,
  CheckFinding,
  NovelAction,
  NovelChange,
  NovelFileMeta,
  NovelProject,
  NovelProjectState,
  NovelProjectSummary,
  NovelSaveResult,
  NovelScene,
  NovelStatePatchResult,
  NovelTextFile,
  OutlineChaptersReply,
  OutlineOp,
  OutlineScenesReply,
  ProjectPatch,
  RecordAlt,
  RecordFinalReason,
  RecordLine,
  RecordRegion,
  RewriteVariant,
  StateFact,
  StateOp,
  SummaryReply
} from '@shared/novel'
import type { SamplingParams, WireMessage } from '@shared/types'
import { parseEntry } from '@shared/frontMatter'
import { findingKey, locateQuote } from '@shared/novelChecks'
import { newId } from '@shared/novelOutline'
import {
  CHECK_SCHEMA,
  OUTLINE_CHAPTERS_SCHEMA,
  OUTLINE_SCENES_SCHEMA,
  SUMMARY_SCHEMA,
  UPDATE_STATE_SCHEMA,
  parseCheckReplyResult,
  parseOutlineChapters,
  parseOutlineScenes,
  parseSummary,
  parseUpdateStateReplyResult
} from '@shared/novelPrompts'
import { stateBefore } from '@shared/novelState'
import { sceneHashKey, wordCount } from '@shared/novelText'
import { appendVoiceNote } from '@shared/novelVoice'
import { errorMessage, invoke, on } from '../lib/api'
import { uid } from '../lib/format'
import type { EditorHandle } from '../lib/novelEditor'
import { once } from '../lib/once'
import { aiBusyReason } from '../lib/aiBusy'
import { chatAdapters } from '../lib/adapters'
import {
  appVersionNow,
  appVersionText,
  classifyOutcome,
  finalLine,
  finishOf,
  locateFinal,
  outcomeLine,
  outcomeRegion,
  recordId,
  regionOf,
  requestLine,
  sendRecord,
  viewedLine
} from '../lib/novelRecord'
import {
  ACTION_LABELS,
  STALE_PENDING,
  applyResult,
  buildInterviewSystem,
  clearCountCache,
  finishText,
  fitInterview,
  fitRequest,
  hashText,
  interviewKey,
  ipcCounter,
  makeFindingsEntry,
  pendingStale,
  resolveSceneRefs,
  resolveStateChanges,
  samplingFor,
  scenePath,
  seedsFor,
  startGeneration,
  stateOpsFor,
  summaryHealth,
  templateOverhead,
  voiceNoteBlock,
  wholeSceneProblem,
  type BuiltRequest,
  type FindingsEntry,
  type GenHandle,
  type InterviewTurn,
  type PendingAlt,
  type PendingChange,
  type RunContext,
  type StateCardItem
} from '../lib/novelRun'
import { readyInstance, useModels } from './models'

export type PanelTab = 'scene' | 'context' | 'checks' | 'state' | 'interview' | 'project'
export type Target = { kind: 'scene'; id: string } | { kind: 'entry'; id: string } | null

export const STYLE_PATH = 'style.md'
export const entryPath = (id: string): string => `bible/${id}.md`

export interface Buffer {
  path: string
  /** Text in the editor (LF). */
  text: string
  /** Text as the disk has it from the last read or save; `dirty` means `text` differs. */
  savedText: string
  /** Hash of the disk bytes the text is based on ('' for a file that does not exist). */
  baseHash: string
  /** Counts every change of `text`; pending AI changes record it (data safety rule 3). */
  version: number
  dirty: boolean
  saving: boolean
  error: string | null
  conflict: { diskHash: string; diskText: string } | null
  changedOnDisk: boolean
  readOnly: boolean
  meta: NovelFileMeta
  /** Leftover journal found on open; offered as Restore / Discard. */
  journal: NovelTextFile['journal']
}

export interface GenState {
  path: string | null
  action: NovelAction
  label: string
  phase: 'preparing' | 'waiting' | 'writing'
  alt: number
  of: number
  /** Prompt processing 0..1 while the model reads the request. */
  progress: number | null
  text: string
}

export type ProposalKind = 'summaryScene' | 'summaryChapter' | 'chapters' | 'scenes'

export interface Proposal {
  id: string
  kind: ProposalKind
  /** Scene or chapter the proposal is for (null for the chapter plan). */
  targetId: string | null
  title: string
  status: 'running' | 'ready' | 'invalid'
  raw: string
  summary?: SummaryReply
  chapters?: OutlineChaptersReply
  scenes?: OutlineScenesReply
  error: string | null
  /** Hash (scene) or combined key (chapter) of the text the summary was made from. */
  summaryOf: string
  /** For Retry. */
  again: { count: number; note: string }
  /** Stage 4: the id of the record this run writes (16 hex). */
  recId?: string
}

/** The card of an Update state request: changes to accept or reject; nothing is written before Apply. */
export interface StateProposal {
  id: string
  sceneId: string
  title: string
  status: 'running' | 'ready' | 'invalid' | 'applied'
  /** hashText of the scene text the request read: Apply is refused when the text is not this any more. */
  textHash: string
  raw: string
  error: string | null
  items: StateCardItem[]
  /** Changes left out, with why (names not in the bible, unknown fact ids). */
  left: string[]
  /** Set when Apply was refused. */
  refused: string | null
  /** Ops main did not apply, with the change they belonged to. */
  rejected: { label: string; reason: string }[]
  applied: number
  applying: boolean
}

/** A Check that could not be read, with what the model wrote (nothing was changed). */
export interface CheckIssue {
  sceneId: string
  error: string
  raw: string
}

/** A chapter check going scene by scene. */
export interface CheckRun {
  chapterId: string | null
  sceneIds: string[]
  /** Index of the scene being checked. */
  index: number
}

export interface LastRequest {
  source: 'preview' | 'run'
  action: NovelAction
  at: number
  built: BuiltRequest
  /** Tokens of the whole request as the template counts it, when known. */
  actual: number | null
  tooBig: boolean
  contextLength: number
}

interface NovelState {
  ready: boolean
  root: string
  projects: NovelProjectSummary[]
  folder: string | null
  state: NovelProjectState | null
  opening: boolean
  openError: string | null
  target: Target
  buffers: Record<string, Buffer>
  /** Latest known disk hash of each scene file (relative path), for stale-summary checks. */
  hashes: Record<string, string>
  panelOpen: boolean
  panelTab: PanelTab
  selection: { from: number; to: number }
  gen: GenState | null
  pending: PendingChange | null
  /** Set when an Accept was refused; shown with Regenerate. */
  pendingRefused: boolean
  proposal: Proposal | null
  /** Stage 3: Check findings per scene (memory only), with the text hash they were made from. */
  findings: Record<string, FindingsEntry>
  checkRun: CheckRun | null
  checkIssue: CheckIssue | null
  /** Stage 3: the Update state card. */
  stateProposal: StateProposal | null
  /** The scene last open in the editor, for the Interview tab while an entry is open. */
  lastSceneId: string | null
  /** Stage 3: interview history per character + scene (interviewKey), memory only. */
  interviews: Record<string, InterviewTurn[]>
  /** Turns left out of the last request of an interview, by key. */
  interviewDropped: Record<string, number>
  /** The interview being answered (its key), so the stream shows only under that character and scene. */
  interviewRun: { key: string } | null
  /** Scenes still to summarize in a "Summarize all" run. */
  summarizeQueue: string[]
  lastRequest: LastRequest | null
  contextBusy: boolean

  init: () => Promise<void>
  togglePanel: () => void
  setPanelTab: (tab: PanelTab) => void
  setSelection: (sel: { from: number; to: number }) => void
  refreshProjects: () => Promise<void>
  createProject: (title: string) => Promise<void>
  openProject: (folder: string) => Promise<void>
  /** False (project kept open, after a toast) when some unsaved text could be neither saved nor journalled. */
  closeProject: () => Promise<boolean>
  deleteProject: () => Promise<void>
  reloadState: () => Promise<void>
  selectScene: (sceneId: string) => Promise<void>
  selectEntry: (entryId: string) => Promise<void>
  edit: (path: string, text: string) => void
  save: (path: string, opts?: { force?: boolean; snapshot?: boolean }) => Promise<boolean>
  reloadFromDisk: (path: string) => Promise<void>
  restoreJournal: (path: string) => void
  discardJournal: (path: string) => Promise<void>
  restoreSnapshot: (path: string, snapshot: string) => Promise<void>
  createMissingFile: (path: string) => Promise<void>
  patch: (ops: OutlineOp[]) => Promise<boolean>
  setScene: (sceneId: string, field: Extract<OutlineOp, { op: 'setScene' }>['field'], value: unknown) => Promise<void>
  patchProject: (patch: ProjectPatch) => Promise<void>
  createEntry: (name: string, kind: BibleKind) => Promise<void>
  deleteScene: (sceneId: string) => Promise<void>
  deleteChapter: (chapterId: string) => Promise<void>
  deleteEntry: (entryId: string) => Promise<void>
  restoreTrash: (id: string) => Promise<void>
  addChapter: (title: string) => Promise<void>
  addScene: (chapterId: string, title: string) => Promise<void>
  adoptFile: (file: string, chapterId: string) => Promise<void>
  ensureHashes: () => Promise<void>
  runAction: (opts: RunOptions) => Promise<void>
  stop: () => void
  selectAlt: (index: number) => void
  acceptAll: () => Promise<void>
  rejectAll: () => Promise<void>
  resolvePending: (finalText: string) => Promise<void>
  regenerate: () => Promise<void>
  summarizeScene: (sceneId: string) => Promise<void>
  summarizeChapter: (chapterId: string) => Promise<void>
  summarizeAll: () => Promise<void>
  planChapters: (count: number, note: string) => Promise<void>
  planScenes: (chapterId: string, count: number, note: string) => Promise<void>
  acceptProposal: (edited?: Proposal) => Promise<void>
  rejectProposal: () => void
  retryProposal: () => Promise<void>
  previewContext: () => Promise<void>
  removeSticky: (sceneId: string, entryId: string) => Promise<void>

  // Stage 3
  checkScene: (sceneId: string) => Promise<void>
  checkChapter: (chapterId: string) => Promise<void>
  dismissFinding: (sceneId: string, finding: CheckFinding, dismissed?: boolean) => Promise<void>
  restoreDismissed: (sceneId: string) => Promise<void>
  jumpToFinding: (sceneId: string, finding: CheckFinding) => void
  fixFinding: (sceneId: string, finding: CheckFinding) => Promise<void>
  clearCheckIssue: () => void
  updateState: (sceneId: string) => Promise<void>
  setStateItem: (key: number, accepted: boolean) => void
  applyStateProposal: () => Promise<void>
  discardStateProposal: () => void
  /** Sends ops for state.json; null when nothing was sent. Rejected ops come back in the result. */
  statePatch: (ops: StateOp[]) => Promise<NovelStatePatchResult | null>
  askInterview: (characterId: string, question: string) => Promise<boolean>
  clearInterview: (key: string) => void
  addVoiceNote: (entryId: string, note: string) => Promise<boolean>
}

export interface RunOptions {
  action: NovelAction
  variant?: RewriteVariant
  note?: string
  /** Alternatives to write, 1–4. */
  alternatives?: number
  /** Where to work; default is the editor's selection or cursor. */
  at?: { cursor: number; selection: { from: number; to: number } | null }
}

// ---------------------------------------------------------------------------------------------
// Module state that is not rendered
// ---------------------------------------------------------------------------------------------

const AUTOSAVE_MS = 1000
const JOURNAL_MS = 300
const RETRY_MS = 5000

const autosaveTimers = new Map<string, ReturnType<typeof setTimeout>>()
const journalTimers = new Map<string, ReturnType<typeof setTimeout>>()
const retryTimers = new Map<string, ReturnType<typeof setTimeout>>()
const saving = new Map<string, Promise<boolean>>()
const editors = new Map<string, EditorHandle>()
let activeGen: GenHandle | null = null
let stopRequested = false
/** Counts project opens and closes; a run started in an earlier one must not touch the current one. */
let epoch = 0

// ---- stage 4: recording (docs/writing-mode-design.md, "Recording") -----------------------------
// Every send is fire-and-forget: nothing on the writing path waits for it, and a failure is ignored
// (sendRecord tells the user once, quietly).

/** How long a scene must be left alone after an edit before its region's text is recorded as a final. */
export const FINAL_IDLE_MS = 10_000

interface FinalTracker {
  recId: string
  folder: string
  path: string
  /** The document as accepted, and where the generated text sits in it. */
  accepted: string
  region: RecordRegion
  timer: ReturnType<typeof setTimeout> | null
}

/** One tracker per scene path, from an accept until the scene is left, a new action starts, or the project closes. */
const trackers = new Map<string, FinalTracker>()
/** Runs (record ids) whose request line was sent, and those whose outcome was sent: nothing more is written for them. */
const requestedRuns = new Set<string>()
const resolvedRuns = new Set<string>()

const nowIso = (): string => new Date().toISOString()

/** Sends a line for `folder` if that project is still the open one and its recording is on. Never awaited on the writing path. */
function recordTo(folder: string, line: RecordLine): Promise<boolean> {
  const s = useNovel.getState()
  if (s.folder !== folder || s.state?.project.recording === false) return Promise.resolve(false)
  return sendRecord(folder, line)
}

function writeFinal(t: FinalTracker, reason: RecordFinalReason, forceNull: boolean): Promise<boolean> | null {
  const b = useNovel.getState().buffers[t.path]
  let text: string | null = null
  let why = reason
  if (!forceNull) {
    // The scene's buffer is gone (deleted): there is no text to read, so no line.
    if (!b) return null
    text = locateFinal(t.accepted, t.region, b.text)
    if (text === null) why = 'outside'
  }
  return recordTo(t.folder, finalLine(t.recId, nowIso(), text, why))
}

/** Ends the tracking of a scene and writes its last final with `reason`. Returns the send, or null when nothing was tracked or sent. */
function endTracker(path: string, reason: RecordFinalReason, forceNull = false): Promise<boolean> | null {
  const t = trackers.get(path)
  if (!t) return null
  trackers.delete(path)
  if (t.timer) clearTimeout(t.timer)
  return writeFinal(t, reason, forceNull)
}

/** A user edit of a tracked scene: the idle timer starts again. */
function touchTracker(path: string): void {
  const t = trackers.get(path)
  if (!t) return
  if (t.timer) clearTimeout(t.timer)
  t.timer = setTimeout(() => {
    t.timer = null
    if (trackers.get(path) === t) void writeFinal(t, 'idle', false)
  }, FINAL_IDLE_MS)
}

function startTracker(t: Omit<FinalTracker, 'timer'>): void {
  const old = trackers.get(t.path)
  if (old?.timer) clearTimeout(old.timer)
  trackers.set(t.path, { ...t, timer: null })
}

/**
 * Writes the current final of every tracked scene (reason 'close'). Tracking goes on: a close that succeeds drops
 * the trackers, and one that fails (unsaved text) leaves the project open and still followed. The promise settles
 * when the sends did.
 */
function flushFinals(): Promise<unknown> {
  const sends: Promise<boolean>[] = []
  for (const t of [...trackers.values()]) {
    if (t.timer) clearTimeout(t.timer)
    t.timer = null
    const p = writeFinal(t, 'close', false)
    if (p) sends.push(p)
  }
  return Promise.all(sends)
}

function dropTrackers(): void {
  for (const t of trackers.values()) if (t.timer) clearTimeout(t.timer)
  trackers.clear()
}

/** The outcome 'abandoned' for a prose change or card the user never resolved. Once per run. */
function abandonRun(folder: string | null, recId: string | undefined): void {
  if (!folder || !recId || !requestedRuns.has(recId) || resolvedRuns.has(recId)) return
  resolvedRuns.add(recId)
  void recordTo(folder, outcomeLine(recId, nowIso(), 'abandoned', { alt: null, region: null }))
}

/**
 * Writes viewed + outcome for a prose change the user resolved (accepted, mixed or rejected) and, for the
 * first two, starts following the region. Once per run; nothing for a run whose request was not recorded.
 */
function recordProseOutcome(p: PendingChange, finalText: string): void {
  const folder = useNovel.getState().folder
  const id = p.recId
  if (!folder || !id || !requestedRuns.has(id) || resolvedRuns.has(id)) return
  resolvedRuns.add(id)
  const alt = p.alts[p.active]
  const kind = alt ? classifyOutcome(p.baseText, finalText, alt.text) : finalText === p.baseText ? 'rejected' : 'mixed'
  const at = nowIso()
  const viewed = p.viewed ?? []
  void recordTo(folder, viewedLine(id, at, viewed.includes(p.active) ? viewed : [...viewed, p.active]))
  if (kind === 'rejected' || !alt) {
    void recordTo(folder, outcomeLine(id, at, 'rejected', { alt: null, region: null }))
    return
  }
  const geo = regionOf(p.action, p.baseText, p.again.cursor, p.again.selection, alt.generated)
  const region = outcomeRegion(kind, geo, alt.text.length, finalText.length)
  void recordTo(folder, outcomeLine(id, at, kind, { alt: p.active, region, replaced: geo.replaced }))
  startTracker({ recId: id, folder, path: p.path, accepted: finalText, region })
}

/** Outcome of a card the user resolved: accepted, edited (with the text) or rejected. Once per run. */
function recordCardOutcome(recId: string | undefined, kind: 'accepted' | 'edited' | 'rejected', text?: string): void {
  const folder = useNovel.getState().folder
  if (!folder || !recId || !requestedRuns.has(recId) || resolvedRuns.has(recId)) return
  resolvedRuns.add(recId)
  void recordTo(folder, outcomeLine(recId, nowIso(), kind, kind === 'edited' ? { text } : {}))
}

/** What a card holds, as text, to tell an edited proposal from the one the model wrote. */
function cardText(p: Pick<Proposal, 'summary' | 'chapters' | 'scenes'>): string {
  if (p.summary) return p.summary.summary
  return JSON.stringify(p.chapters ?? p.scenes ?? null)
}

/** The adapters a request goes out with: each loaded adapter at the project's strength. */
function requestAdapters(instanceId: string, project: NovelProject): { name: string; scale: number }[] {
  const inst = useModels.getState().instances.find((i) => i.instanceId === instanceId)
  return chatAdapters(inst?.adapters ?? [], project.adapters).map((a) => ({ name: a.name, scale: a.scale }))
}

interface RequestRecord {
  recId: string
  action: NovelAction
  variant?: string
  sceneId: string | null
  chapterId: string | null
  model: { instanceId: string; modelId: string }
  project: NovelProject
  /** The sampling actually sent for the first alternative (each alternative's own seed is in alts). */
  sampling: SamplingParams
  messages: { role: string; content: unknown }[]
  alts: RecordAlt[]
}

/** Sends the request line of a run that ended with at least one alternative. Marks the run as recorded at once, so its outcome follows. */
function sendRequestRecord(folder: string, r: RequestRecord): void {
  if (useNovel.getState().state?.project.recording === false) return
  requestedRuns.add(r.recId)
  // Built and sent now, with no wait before it, so it cannot land after a close or in a reopened project.
  void appVersionText()
  {
    const app = appVersionNow()
    void recordTo(
      folder,
      requestLine({
        id: r.recId,
        at: nowIso(),
        app,
        action: r.action,
        variant: r.variant,
        sceneId: r.sceneId,
        chapterId: r.chapterId,
        model: { id: r.model.modelId, name: modelName(r.model.modelId) },
        sampling: r.sampling,
        adapters: requestAdapters(r.model.instanceId, r.project),
        messages: r.messages,
        alts: r.alts
      })
    )
  }
}

function modelName(modelId: string): string {
  const m = useModels.getState()
  return m.models.find((x) => x.id === modelId)?.name ?? m.instances.find((i) => i.modelId === modelId)?.modelName ?? modelId
}

export function registerEditor(path: string, handle: EditorHandle | null): void {
  if (handle) editors.set(path, handle)
  else editors.delete(path)
}

function clearTimer(map: Map<string, ReturnType<typeof setTimeout>>, path: string): void {
  const t = map.get(path)
  if (t) clearTimeout(t)
  map.delete(path)
}

function clearAllTimers(): void {
  for (const m of [autosaveTimers, journalTimers, retryTimers]) {
    for (const t of m.values()) clearTimeout(t)
    m.clear()
  }
}

function remember(key: string, value: string | null): void {
  try {
    if (value === null) localStorage.removeItem(key)
    else localStorage.setItem(key, value)
  } catch {
    // Storage may be blocked; the page works without it.
  }
}

function recall(key: string): string | null {
  try {
    return localStorage.getItem(key)
  } catch {
    return null
  }
}

const LAST_PROJECT = 'rulz.writing.project'
const PANEL_KEY = 'rulz.writing.panel'
const sceneKey = (folder: string): string => `rulz.writing.scene:${folder}`

function makeBuffer(f: NovelTextFile): Buffer {
  return {
    path: f.path,
    text: f.text,
    savedText: f.text,
    baseHash: f.hash,
    version: 0,
    dirty: false,
    saving: false,
    error: null,
    conflict: null,
    changedOnDisk: false,
    readOnly: f.readOnly,
    meta: f.meta,
    journal: f.journal && f.journal.text !== f.text ? f.journal : null
  }
}

/** Live word count of a scene: the editor's text when it is open, else what the disk had. */
export function liveWords(s: Pick<NovelState, 'buffers' | 'state'>, path: string): number {
  const b = s.buffers[path]
  if (b) return wordCount(b.text)
  return s.state?.words[path] ?? 0
}

export function sceneOf(state: NovelProjectState | null, sceneId: string): { scene: NovelScene; chapterId: string } | null {
  if (!state) return null
  for (const c of state.outline.chapters) {
    const scene = c.scenes.find((s) => s.id === sceneId)
    if (scene) return { scene, chapterId: c.id }
  }
  return null
}

function firstSceneId(state: NovelProjectState): string | null {
  return state.outline.chapters.flatMap((c) => c.scenes)[0]?.id ?? null
}

const wire = (messages: { role: string; content: string }[]): WireMessage[] => messages as WireMessage[]

// ---------------------------------------------------------------------------------------------
// The store
// ---------------------------------------------------------------------------------------------

export const useNovel = create<NovelState>((set, get) => {
  const patchBuffer = (path: string, fn: (b: Buffer) => Partial<Buffer>): void => {
    set((s) => {
      const b = s.buffers[path]
      return b ? { buffers: { ...s.buffers, [path]: { ...b, ...fn(b) } } } : s
    })
  }

  const busy = (): boolean => !!get().gen || !!get().checkRun

  // ---- buffers ---------------------------------------------------------------------------

  const scheduleJournal = (path: string): void => {
    clearTimer(journalTimers, path)
    journalTimers.set(
      path,
      setTimeout(() => {
        journalTimers.delete(path)
        const { folder, buffers } = get()
        const b = buffers[path]
        if (!folder || !b || !b.dirty || b.readOnly) return
        void invoke('novel:journal', folder, path, b.text, b.baseHash).catch(() => undefined)
      }, JOURNAL_MS)
    )
  }

  const scheduleAutosave = (path: string): void => {
    clearTimer(autosaveTimers, path)
    autosaveTimers.set(
      path,
      setTimeout(() => {
        autosaveTimers.delete(path)
        void get().save(path)
      }, AUTOSAVE_MS)
    )
  }

  const scheduleRetry = (path: string): void => {
    clearTimer(retryTimers, path)
    retryTimers.set(
      path,
      setTimeout(() => {
        retryTimers.delete(path)
        const b = get().buffers[path]
        if (b?.dirty && b.error && !b.conflict) void get().save(path)
      }, RETRY_MS)
    )
  }

  const doSave = async (path: string, opts: { force?: boolean; snapshot?: boolean }): Promise<boolean> => {
    const folder = get().folder
    const b = get().buffers[path]
    if (!folder || !b) return false
    if (b.readOnly) return false
    if (!b.dirty && !opts.force && !opts.snapshot) return true
    if (b.conflict && !opts.force) return false
    clearTimer(autosaveTimers, path)
    const text = b.text
    patchBuffer(path, () => ({ saving: true }))
    let result: NovelSaveResult
    try {
      result = await invoke('novel:saveFile', { folder, path, text, baseHash: b.baseHash, force: opts.force, snapshot: opts.snapshot })
    } catch (e) {
      result = { ok: false, reason: 'error', error: errorMessage(e) }
    }
    // The project may have been closed or the buffer dropped while the save ran.
    if (get().folder !== folder || !get().buffers[path]) return result.ok
    if (result.ok) {
      clearTimer(retryTimers, path)
      patchBuffer(path, (cur) => ({
        saving: false,
        baseHash: result.hash,
        savedText: text,
        dirty: cur.text !== text,
        error: null,
        conflict: null,
        changedOnDisk: false,
        journal: null
      }))
      set((s) => ({
        hashes: path.startsWith('manuscript/') ? { ...s.hashes, [path]: result.hash } : s.hashes,
        state: s.state && path.startsWith('manuscript/') ? { ...s.state, words: { ...s.state.words, [path]: wordCount(text) } } : s.state
      }))
      // Edited while the save ran (or undone back to the old text): what is in the editor is not on disk.
      if (get().buffers[path]?.dirty) {
        scheduleAutosave(path)
        scheduleJournal(path)
      }
      return true
    }
    if (result.reason === 'conflict') {
      clearTimer(retryTimers, path)
      patchBuffer(path, () => ({ saving: false, conflict: { diskHash: result.diskHash, diskText: result.diskText }, error: null }))
      return false
    }
    if (result.reason === 'readOnly' || result.reason === 'blocked') {
      patchBuffer(path, () => ({
        saving: false,
        error: result.reason === 'blocked' ? 'The project is read-only, so nothing was saved.' : 'This file is read-only, so nothing was saved.'
      }))
      return false
    }
    const message = 'error' in result ? result.error : 'The save failed.'
    patchBuffer(path, () => ({ saving: false, error: message }))
    scheduleRetry(path)
    return false
  }

  /** Reads a file into a new buffer; null (after a toast) if it cannot be read. */
  const loadBuffer = async (path: string): Promise<Buffer | null> => {
    const folder = get().folder
    if (!folder) return null
    try {
      return makeBuffer(await invoke('novel:readFile', folder, path))
    } catch (e) {
      toast.error(`Could not open ${path}: ${errorMessage(e)}`)
      return null
    }
  }

  const ensureBuffer = async (path: string): Promise<boolean> => {
    if (get().buffers[path]) return true
    const epoch0 = epoch
    const b = await loadBuffer(path)
    if (!b) return false
    // The project was closed or switched while the file was read: its buffer belongs to nobody now.
    if (epoch !== epoch0) return false
    set((s) => ({ buffers: { ...s.buffers, [path]: s.buffers[path] ?? b } }))
    if (path.startsWith('manuscript/')) set((s) => ({ hashes: { ...s.hashes, [path]: b.baseHash } }))
    return true
  }

  /** Drops a clean buffer nobody is working on, so it is read fresh the next time. */
  const dropIfIdle = (path: string): void => {
    const s = get()
    const b = s.buffers[path]
    if (!b || b.dirty || b.saving || b.conflict || b.error) return
    if (s.pending?.path === path || s.gen?.path === path) return
    const keep = s.target && ((s.target.kind === 'scene' && sceneOf(s.state, s.target.id) && scenePath(sceneOf(s.state, s.target.id)!.scene) === path) || (s.target.kind === 'entry' && entryPath(s.target.id) === path))
    if (keep || path === STYLE_PATH) return
    clearTimer(journalTimers, path)
    set((st) => {
      const rest = { ...st.buffers }
      delete rest[path]
      return { buffers: rest }
    })
  }

  const targetPath = (t: Target, state: NovelProjectState | null): string | null => {
    if (!t) return null
    if (t.kind === 'entry') return entryPath(t.id)
    const f = sceneOf(state, t.id)
    return f ? scenePath(f.scene) : null
  }

  /** A change on disk: reload a buffer only when nothing of the user's is in it, else mark it. */
  const externalChange = async (path: string): Promise<void> => {
    const folder = get().folder
    const b = get().buffers[path]
    if (!folder || !b) return
    // style.md is an input of every request, so it is held still while anything is pending or running.
    const styleHeld = path === STYLE_PATH && (!!get().gen || !!get().pending)
    const working = b.dirty || b.saving || !!b.conflict || styleHeld || get().pending?.path === path || get().gen?.path === path
    let file: NovelTextFile
    try {
      file = await invoke('novel:readFile', folder, path)
    } catch {
      return
    }
    const cur = get().buffers[path]
    if (!cur || file.hash === cur.baseHash) return
    if (path.startsWith('manuscript/')) set((s) => ({ hashes: { ...s.hashes, [path]: file.hash } }))
    const stillWorking = working || cur.dirty || cur.saving || !!cur.conflict || (path === STYLE_PATH && (!!get().gen || !!get().pending)) || get().pending?.path === path || get().gen?.path === path
    if (stillWorking) return patchBuffer(path, () => ({ changedOnDisk: true }))
    void endTracker(path, 'reloaded', true)
    patchBuffer(path, (x) => ({
      text: file.text,
      savedText: file.text,
      baseHash: file.hash,
      version: x.version + 1,
      dirty: false,
      error: null,
      readOnly: file.readOnly,
      meta: file.meta,
      changedOnDisk: false
    }))
  }

  const applyState = (next: NovelProjectState): void => {
    const prev = get().state
    set((s) => ({ state: next, hashes: s.hashes }))
    // A scene that left the outline can't stay the open one.
    const t = get().target
    if (t?.kind === 'scene' && !sceneOf(next, t.id)) {
      const first = firstSceneId(next)
      set({ target: null })
      if (first) void get().selectScene(first)
    }
    if (t?.kind === 'entry' && !next.bible.some((e) => e.id === t.id)) set({ target: null })
    // style.md: reload when it changed on disk and the user has no edit in it.
    const sb = get().buffers[STYLE_PATH]
    if (!sb) set((s) => ({ buffers: { ...s.buffers, [STYLE_PATH]: makeBuffer(next.style) } }))
    else if (next.style.hash !== sb.baseHash) void externalChange(STYLE_PATH)
    // Hashes of files that left the outline are of no use.
    if (prev && prev.folder === next.folder) {
      const keep = new Set(next.outline.chapters.flatMap((c) => c.scenes.map(scenePath)))
      set((s) => ({ hashes: Object.fromEntries(Object.entries(s.hashes).filter(([p]) => keep.has(p))) }))
    }
  }

  const onChanged = (e: NovelChange): void => {
    if (e.folder !== get().folder) return
    if (e.kind === 'state') return applyState(e.state)
    if (e.kind === 'text') {
      if (e.path.startsWith('manuscript/')) set((s) => ({ hashes: { ...s.hashes, [e.path]: e.hash } }))
      const b = get().buffers[e.path]
      // Our own write comes back as a change too.
      if (!b || e.hash === b.baseHash) return
      void externalChange(e.path)
      return
    }
    if (e.kind === 'deleted') {
      if (get().buffers[e.path]) patchBuffer(e.path, () => ({ changedOnDisk: true }))
      void get().reloadState()
    }
  }

  const recheckOpen = (): void => {
    const { folder, buffers } = get()
    if (!folder) return
    for (const path of Object.keys(buffers)) void externalChange(path)
  }

  // ---- generation helpers ----------------------------------------------------------------

  const resolveModel = async (): Promise<{ modelId: string; contextLength: number; instanceId: string } | null> => {
    const project = get().state?.project
    if (!project) return null
    const modelId = project.modelId ?? useModels.getState().activeModelId
    if (!modelId) {
      toast.error('No model is loaded. Load one (Ctrl+L) or choose a model in the Project tab.')
      return null
    }
    let inst = readyInstance(modelId)
    if (!inst) {
      if (project.modelId === null) {
        toast.error('The active model is not ready. Wait for it to finish loading, or load one (Ctrl+L).')
        return null
      }
      inst = (await useModels.getState().ensure(modelId)) ?? undefined
      if (!inst) return null
    }
    return { modelId: inst.modelId, contextLength: inst.contextLength ?? 8192, instanceId: inst.instanceId }
  }

  /** Text and disk hash of a scene: from its buffer (saved first) when open, else read from disk. */
  const sceneSource = async (sceneId: string, saveFirst: boolean): Promise<{ text: string; hash: string } | null> => {
    const found = sceneOf(get().state, sceneId)
    const folder = get().folder
    if (!found || !folder) return null
    const path = scenePath(found.scene)
    if (get().buffers[path]) {
      if (saveFirst && !(await get().save(path))) return null
      const b = get().buffers[path]
      return b ? { text: b.text, hash: b.baseHash } : null
    }
    try {
      const f = await invoke('novel:readFile', folder, path)
      set((s) => ({ hashes: { ...s.hashes, [path]: f.hash } }))
      return { text: f.text, hash: f.hash }
    } catch {
      return null
    }
  }

  const previousSceneText = async (sceneId: string): Promise<string> => {
    const all = (get().state?.outline.chapters ?? []).flatMap((c) => c.scenes)
    const i = all.findIndex((s) => s.id === sceneId)
    for (let j = i - 1; j >= 0; j--) {
      const src = await sceneSource(all[j].id, false)
      if (src?.text.trim()) return src.text
    }
    return ''
  }

  const baseContext = async (
    model: { modelId: string; contextLength: number; instanceId: string },
    over: Partial<RunContext> & Pick<RunContext, 'action'>,
    estimated: { value: boolean }
  ): Promise<{ ctx: RunContext; count: ReturnType<typeof ipcCounter> } | null> => {
    const s = get()
    if (!s.state) return null
    const count = ipcCounter(model.modelId, () => (estimated.value = true))
    const overhead = await templateOverhead(model.instanceId, model.modelId, count)
    const ctx: RunContext = {
      project: s.state.project,
      style: s.buffers[STYLE_PATH]?.text ?? s.state.style.text,
      outline: s.state.outline,
      bible: s.state.bible,
      sceneId: null,
      sceneHashes: { ...s.hashes },
      facts: s.state.state?.facts ?? [],
      text: '',
      cursor: 0,
      selection: null,
      note: '',
      contextLength: model.contextLength,
      templateOverhead: overhead,
      ...over
    }
    return { ctx, count }
  }

  const countWhole = (modelId: string) => (messages: { role: string; content: string }[]): Promise<number | null> =>
    invoke('chat:countTokens', modelId, wire(messages), null).catch(() => null)

  let throttleTimer: ReturnType<typeof setTimeout> | null = null
  let throttleText = ''
  const pushStreamText = (text: string): void => {
    throttleText = text
    if (throttleTimer) return
    throttleTimer = setTimeout(() => {
      throttleTimer = null
      set((s) => (s.gen ? { gen: { ...s.gen, text: throttleText, phase: 'writing', progress: null } } : s))
    }, 120)
  }

  // ---- structured proposals --------------------------------------------------------------

  const runStructured = async (
    kind: ProposalKind,
    action: NovelAction,
    over: Partial<RunContext>,
    meta: { targetId: string | null; title: string; summaryOf: string; again: { count: number; note: string } }
  ): Promise<void> => {
    if (busy() || get().pending) return void toast.message('Wait for the current request to finish first.')
    const why = aiBusyReason('writing')
    if (why) return void toast.message(why)
    // A card still on screen is replaced: it is abandoned.
    abandonRun(get().folder, get().proposal?.recId)
    const proposal: Proposal = { id: uid(), kind, targetId: meta.targetId, title: meta.title, status: 'running', raw: '', error: null, summaryOf: meta.summaryOf, again: meta.again, recId: recordId() }
    set({ proposal, gen: { path: null, action, label: ACTION_LABELS[action], phase: 'preparing', alt: 1, of: 1, progress: null, text: '' }, panelOpen: true, panelTab: 'scene' })
    stopRequested = false
    // The project open now: a result that arrives after it was closed is dropped.
    const epoch0 = epoch
    const folder0 = get().folder
    const stale = (): boolean => epoch !== epoch0 || get().folder !== folder0
    const done = (patch: Partial<Proposal> | null): void => {
      if (stale()) return
      set((s) => ({ gen: null, proposal: patch && s.proposal?.id === proposal.id ? { ...s.proposal, ...patch } : patch === null ? null : s.proposal }))
    }
    try {
      const model = await resolveModel()
      if (!model || stale()) return done(null)
      const estimated = { value: false }
      const base = await baseContext(model, { action, ...over }, estimated)
      if (!base || stale()) return done(null)
      const fit = await fitRequest(base.ctx, base.count, countWhole(model.modelId), estimated)
      if (stale()) return done(null)
      set({ lastRequest: { source: 'run', action, at: Date.now(), built: fit.built, actual: fit.actual, tooBig: fit.tooBig, contextLength: model.contextLength } })
      if (fit.tooBig) {
        toast.error('This request does not fit in the context. The Context tab shows what is using it.')
        return done({ status: 'invalid', error: 'The request does not fit in the context.' })
      }
      const schema = action === 'outlineChapters' ? OUTLINE_CHAPTERS_SCHEMA : action === 'outlineScenes' ? OUTLINE_SCENES_SCHEMA : SUMMARY_SCHEMA
      const sampling = base.ctx.project.sampling
      const seed = seedsFor(sampling.seed, 1)[0]
      const maxTokens = Math.max(1024, base.ctx.project.budget.replyTokens)
      const gen = startGeneration({
        modelId: model.modelId,
        messages: fit.built.result.messages,
        sampling,
        adapters: base.ctx.project.adapters,
        action,
        seed,
        maxTokens,
        schema,
        onProgress: (p) => set((s) => (s.gen ? { gen: { ...s.gen, phase: 'waiting', progress: p } } : s)),
        onDelta: pushStreamText
      })
      activeGen = gen
      const res = await gen.done
      if (epoch === epoch0) activeGen = null
      if (stale() || get().proposal?.id !== proposal.id) return done(null)
      if (res.stopped && !res.text.trim()) return done(null)
      // The run's request: one alternative, the raw reply (also an unreadable one: it is what the model wrote).
      if (res.text.trim() && proposal.recId && folder0) {
        sendRequestRecord(folder0, {
          recId: proposal.recId,
          action,
          sceneId: over.sceneId ?? null,
          chapterId: over.chapterId ?? null,
          model,
          project: base.ctx.project,
          sampling: samplingFor(sampling, { seed, maxTokens, schema, action }),
          messages: fit.built.result.messages,
          alts: [{ index: 0, seed, generated: res.text, finish: finishOf(res.stopReason, res) }]
        })
      }
      if (res.error) return done({ status: 'invalid', raw: res.text, error: res.error })
      const raw = res.text.trim()
      if (kind === 'chapters') {
        const p = parseOutlineChapters(raw)
        return done(p.ok ? { status: 'ready', raw, chapters: p.value } : { status: 'invalid', raw, error: p.error })
      }
      if (kind === 'scenes') {
        const p = parseOutlineScenes(raw)
        return done(p.ok ? { status: 'ready', raw, scenes: p.value } : { status: 'invalid', raw, error: p.error })
      }
      const p = parseSummary(raw)
      return done(p.ok ? { status: 'ready', raw, summary: p.value } : { status: 'invalid', raw, error: p.error })
    } catch (e) {
      if (epoch === epoch0) activeGen = null
      done({ status: 'invalid', error: errorMessage(e) })
    }
  }

  // ---- stage 3 helpers -------------------------------------------------------------------

  const factsNow = (): StateFact[] => get().state?.state?.facts ?? []

  type WholeResult =
    | { status: 'done'; raw: string; text: string; textHash: string }
    | { status: 'empty' }
    | { status: 'refused' }
    | { status: 'stopped' }
    | { status: 'stale' }
    | { status: 'failed'; error: string; raw: string }

  /**
   * One structured request over the whole scene (Check, Update state). The scene is held still while it runs
   * (gen.path), every await re-checks the project open now, and the request is refused when the scene does not fit whole.
   */
  const runWholeScene = async (sceneId: string, action: 'checkScene' | 'updateState', label: string): Promise<WholeResult> => {
    const s0 = get()
    const found = sceneOf(s0.state, sceneId)
    if (!s0.state || !s0.folder || !found) return { status: 'refused' }
    const path = scenePath(found.scene)
    const epoch0 = epoch
    const folder0 = s0.folder
    const stale = (): boolean => epoch !== epoch0 || get().folder !== folder0
    set({ gen: { path, action, label, phase: 'preparing', alt: 1, of: 1, progress: null, text: '' } })
    try {
      const b0 = get().buffers[path]
      if (b0?.dirty && !b0.readOnly && !(await get().save(path))) {
        toast.error('Save the text first: it could not be saved, so nothing was sent.')
        return { status: 'refused' }
      }
      if (stale()) return { status: 'stale' }
      if (stopRequested) return { status: 'stopped' }
      const src = await sceneSource(sceneId, false)
      if (stale()) return { status: 'stale' }
      if (stopRequested) return { status: 'stopped' }
      if (!src) return { status: 'refused' }
      if (!src.text.trim()) return { status: 'empty' }
      const version = get().buffers[path]?.version
      // The text moved since it was read: what was sent (or is about to be) is not what is in the scene.
      const moved = (): boolean => {
        const b = get().buffers[path]
        return !!b && (b.version !== version || b.text !== src.text)
      }
      const model = await resolveModel()
      if (stale()) return { status: 'stale' }
      if (stopRequested) return { status: 'stopped' }
      if (!model) return { status: 'refused' }
      await get().ensureHashes()
      if (stale()) return { status: 'stale' }
      if (stopRequested) return { status: 'stopped' }
      const estimated = { value: false }
      const base = await baseContext(model, { action, sceneId, text: src.text, cursor: src.text.length }, estimated)
      if (!base || stale()) return { status: 'stale' }
      if (stopRequested) return { status: 'stopped' }
      const fit = await fitRequest(base.ctx, base.count, countWhole(model.modelId), estimated)
      if (stale() || moved()) return { status: 'stale' }
      if (stopRequested) return { status: 'stopped' }
      set({ lastRequest: { source: 'run', action, at: Date.now(), built: fit.built, actual: fit.actual, tooBig: fit.tooBig, contextLength: model.contextLength } })
      if (fit.tooBig) {
        set({ panelOpen: true, panelTab: 'context' })
        toast.error('This request does not fit in the context. The Context tab shows what is using it.')
        return { status: 'refused' }
      }
      const problem = wholeSceneProblem(fit.built)
      if (problem) {
        set({ panelOpen: true, panelTab: 'context' })
        toast.error(problem)
        return { status: 'refused' }
      }
      const sampling = base.ctx.project.sampling
      if (stopRequested) return { status: 'stopped' }
      const gen = startGeneration({
        modelId: model.modelId,
        messages: fit.built.result.messages,
        sampling,
        adapters: base.ctx.project.adapters,
        action,
        seed: seedsFor(sampling.seed, 1)[0],
        maxTokens: Math.max(1024, base.ctx.project.budget.replyTokens),
        schema: action === 'checkScene' ? CHECK_SCHEMA : UPDATE_STATE_SCHEMA,
        onProgress: (p) => set((x) => (x.gen ? { gen: { ...x.gen, phase: 'waiting', progress: p } } : x)),
        onDelta: pushStreamText
      })
      activeGen = gen
      const res = await gen.done
      if (epoch === epoch0) activeGen = null
      if (stale() || moved()) return { status: 'stale' }
      if (res.stopped) return { status: 'stopped' }
      if (res.error) return { status: 'failed', error: res.error, raw: res.text }
      return { status: 'done', raw: res.text.trim(), text: src.text, textHash: hashText(src.text) }
    } catch (e) {
      return { status: 'failed', error: errorMessage(e), raw: '' }
    } finally {
      if (epoch === epoch0) {
        activeGen = null
        set({ gen: null })
      }
    }
  }

  /** Checks one scene and keeps its findings. 'refused' leaves the scene unchecked (already told); 'failed' / 'stopped' end a chapter run. */
  const checkOne = async (sceneId: string, label: string): Promise<'done' | 'empty' | 'refused' | 'failed' | 'stopped'> => {
    const epoch0 = epoch
    const r = await runWholeScene(sceneId, 'checkScene', label)
    if (epoch !== epoch0) return 'stopped'
    if (r.status === 'stale' || r.status === 'stopped') return 'stopped'
    if (r.status === 'empty' || r.status === 'refused') return r.status
    if (r.status === 'failed') {
      set({ checkIssue: { sceneId, error: r.error, raw: r.raw }, panelOpen: true, panelTab: 'checks' })
      toast.error(`The check failed: ${r.error}`)
      return 'failed'
    }
    const parsed = parseCheckReplyResult(r.raw)
    if (!parsed.ok) {
      set({ checkIssue: { sceneId, error: parsed.error, raw: r.raw }, panelOpen: true, panelTab: 'checks' })
      return 'failed'
    }
    const entry = makeFindingsEntry(r.text, parsed.value.findings)
    set((s) => ({ findings: { ...s.findings, [sceneId]: entry }, checkIssue: s.checkIssue?.sceneId === sceneId ? null : s.checkIssue }))
    return 'done'
  }

  /** Where a finding's quote is now: the editor's mark (it follows edits), else found again in the buffer. */
  const findingRange = (sceneId: string, finding: CheckFinding): { from: number; to: number } | null => {
    const found = sceneOf(get().state, sceneId)
    if (!found) return null
    const path = scenePath(found.scene)
    const live = editors.get(path)?.findingRange(findingKey(finding))
    if (live) return live
    const b = get().buffers[path]
    return b ? locateQuote(b.text, finding.quote) : null
  }

  const sendStatePatch = async (ops: StateOp[]): Promise<NovelStatePatchResult | null> => {
    const s = get()
    if (!s.folder || !s.state || ops.length === 0) return null
    if (s.state.blocked) {
      toast.error('The project is read-only, so the state was not changed.')
      return null
    }
    const st = s.state.state
    const onlyReset = ops.length === 1 && ops[0].op === 'reset'
    if (st?.problem && !(onlyReset && st.canReset)) {
      toast.error(`state.json cannot be used: ${st.problem}`)
      return null
    }
    const folder0 = s.folder
    const epoch0 = epoch
    try {
      const res = await invoke('novel:statePatch', folder0, ops)
      if (epoch === epoch0 && get().folder === folder0) applyState(res.state)
      return res
    } catch (e) {
      toast.error(errorMessage(e))
      return null
    }
  }

  // ---- the store -------------------------------------------------------------------------

  return {
    ready: false,
    root: '',
    projects: [],
    folder: null,
    state: null,
    opening: false,
    openError: null,
    target: null,
    buffers: {},
    hashes: {},
    panelOpen: recall(PANEL_KEY) !== '0',
    panelTab: 'scene',
    selection: { from: 0, to: 0 },
    gen: null,
    pending: null,
    pendingRefused: false,
    proposal: null,
    findings: {},
    checkRun: null,
    checkIssue: null,
    stateProposal: null,
    lastSceneId: null,
    interviews: {},
    interviewDropped: {},
    interviewRun: null,
    summarizeQueue: [],
    lastRequest: null,
    contextBusy: false,

    init: once(async () => {
      on('novel:changed', onChanged)
      on('novel:flush', () => {
        void flushAll().finally(() => void invoke('novel:flushDone').catch(() => undefined))
      })
      window.addEventListener('focus', recheckOpen)
      void appVersionText()
      set({ ready: true })
    }),

    togglePanel: () => {
      const open = !get().panelOpen
      remember(PANEL_KEY, open ? '1' : '0')
      set({ panelOpen: open })
    },
    setPanelTab: (panelTab) => set({ panelTab, panelOpen: true }),
    setSelection: (selection) => {
      const cur = get().selection
      if (cur.from !== selection.from || cur.to !== selection.to) set({ selection })
    },

    refreshProjects: async () => {
      try {
        const [root, projects] = await Promise.all([invoke('novel:root'), invoke('novel:listProjects')])
        set({ root, projects })
      } catch (e) {
        toast.error(`Could not list projects: ${errorMessage(e)}`)
      }
    },

    createProject: async (title) => {
      try {
        if (!(await get().closeProject())) return
        const st = await invoke('novel:createProject', title)
        await enterProject(st)
        void get().refreshProjects()
      } catch (e) {
        toast.error(`Could not create the project: ${errorMessage(e)}`)
      }
    },

    openProject: async (folder) => {
      if (get().opening) return
      // The folder that is already open: refresh its state, leave buffers and timers alone.
      if (get().folder === folder) return void (await get().reloadState())
      set({ opening: true, openError: null })
      try {
        // The open project is saved or journalled first; if that fails it stays open.
        if (get().folder && !(await get().closeProject())) return
        const st = await invoke('novel:openProject', folder)
        await enterProject(st)
      } catch (e) {
        set({ openError: errorMessage(e) })
        toast.error(`Could not open the project: ${errorMessage(e)}`)
      } finally {
        set({ opening: false })
      }
    },

    closeProject: async () => {
      const folder = get().folder
      if (!folder) return true
      // A running request belongs to this project: stop it, and ignore whatever it still delivers.
      stopRequested = true
      activeGen?.abort()
      // Whatever is unsaved is saved or journalled first; if some of it can't be, the project stays open.
      const flushed = await flushAll()
      if (!flushed.ok) {
        toast.error('Some unsaved text could not be saved or kept, so the project stays open: ' + flushed.failed.join(', '))
        return false
      }
      // A change or card still waiting for the user's answer is abandoned (flushAll wrote the finals).
      abandonRun(folder, get().pending?.recId)
      abandonRun(folder, get().proposal?.recId)
      dropTrackers()
      epoch++
      activeGen = null
      clearAllTimers()
      editors.clear()
      set({ folder: null, state: null, target: null, buffers: {}, hashes: {}, gen: null, pending: null, pendingRefused: false, proposal: null, findings: {}, checkRun: null, checkIssue: null, stateProposal: null, lastSceneId: null, interviews: {}, interviewDropped: {}, interviewRun: null, summarizeQueue: [], lastRequest: null })
      await invoke('novel:closeProject', folder).catch(() => undefined)
      return true
    },

    deleteProject: async () => {
      const folder = get().folder
      if (!folder) return
      if (!(await get().closeProject())) return
      try {
        await invoke('novel:deleteProject', folder)
        remember(LAST_PROJECT, null)
        toast.success('The project folder is in the Recycle Bin.')
      } catch (e) {
        toast.error(`Nothing was deleted: ${errorMessage(e)}`)
      }
      void get().refreshProjects()
    },

    reloadState: async () => {
      const folder = get().folder
      if (!folder) return
      const epoch0 = epoch
      try {
        const st = await invoke('novel:openProject', folder)
        // The project was closed or switched meanwhile: the open above added the window as an owner again.
        if (epoch !== epoch0 || get().folder !== folder) {
          if (get().folder !== folder) await invoke('novel:closeProject', folder).catch(() => undefined)
          return
        }
        applyState(st)
      } catch {
        // Keep what is shown; the next change event retries.
      }
    },

    selectScene: async (sceneId) => {
      const s = get()
      const found = sceneOf(s.state, sceneId)
      if (!found || !s.folder) return
      const path = scenePath(found.scene)
      const old = targetPath(s.target, s.state)
      if (old && old !== path) void get().save(old)
      if (!(await ensureBuffer(path))) return
      set({ target: { kind: 'scene', id: sceneId }, lastSceneId: sceneId, selection: { from: 0, to: 0 } })
      remember(sceneKey(s.folder), sceneId)
      if (old && old !== path) {
        void endTracker(old, 'left')
        dropIfIdle(old)
      }
    },

    selectEntry: async (entryId) => {
      const s = get()
      if (!s.folder) return
      const path = entryPath(entryId)
      const old = targetPath(s.target, s.state)
      if (old && old !== path) void get().save(old)
      if (!(await ensureBuffer(path))) return
      set({ target: { kind: 'entry', id: entryId } })
      if (old && old !== path) {
        void endTracker(old, 'left')
        dropIfIdle(old)
      }
    },

    edit: (path, text) => {
      const s = get()
      const b = s.buffers[path]
      if (!b || b.readOnly || text === b.text) return
      // The text is held still while a change is pending or being written.
      if (s.pending?.path === path || s.gen?.path === path) return
      // A save in flight writes the text it was sent: until it ends, undoing back to the old text is not "clean".
      const inflight = b.saving || saving.has(path)
      const dirty = text !== b.savedText || inflight
      patchBuffer(path, (x) => ({ text, version: x.version + 1, dirty }))
      touchTracker(path)
      if (dirty) {
        scheduleJournal(path)
        if (!b.conflict) scheduleAutosave(path)
        if (b.error) scheduleRetry(path)
      } else {
        clearTimer(autosaveTimers, path)
        clearTimer(journalTimers, path)
        const folder = s.folder
        if (folder) void invoke('novel:discardJournal', folder, path).catch(() => undefined)
      }
    },

    save: (path, opts = {}) => {
      const prev = saving.get(path) ?? Promise.resolve(true)
      const p = prev.then(() => doSave(path, opts))
      saving.set(path, p)
      void p.finally(() => {
        if (saving.get(path) === p) saving.delete(path)
      })
      return p
    },

    reloadFromDisk: async (path) => {
      const folder = get().folder
      if (!folder) return
      // The editor stays editable while the file is read; an edit made meanwhile wins over the reply.
      const version = get().buffers[path]?.version
      try {
        const f = await invoke('novel:readFile', folder, path)
        const cur = get().buffers[path]
        if (get().folder !== folder || !cur) return
        if (cur.version !== version) return void toast.message('You edited the text while it was being reloaded, so it was kept. Choose Reload from disk again to replace it.')
        clearTimer(autosaveTimers, path)
        clearTimer(journalTimers, path)
        clearTimer(retryTimers, path)
        void endTracker(path, 'reloaded', true)
        patchBuffer(path, (x) => ({
          text: f.text,
          savedText: f.text,
          baseHash: f.hash,
          version: x.version + 1,
          dirty: false,
          error: null,
          conflict: null,
          changedOnDisk: false,
          readOnly: f.readOnly,
          meta: f.meta,
          journal: null
        }))
        if (path.startsWith('manuscript/')) set((s) => ({ hashes: { ...s.hashes, [path]: f.hash } }))
        await invoke('novel:discardJournal', folder, path).catch(() => undefined)
      } catch (e) {
        toast.error(`Could not reload: ${errorMessage(e)}`)
      }
    },

    restoreJournal: (path) => {
      const b = get().buffers[path]
      if (!b?.journal) return
      const j = b.journal
      void endTracker(path, 'reloaded', true)
      // Based on the hash the journal was made against: if the file moved on since, the save asks first.
      patchBuffer(path, (x) => ({ text: j.text, version: x.version + 1, dirty: j.text !== x.savedText, baseHash: j.baseHash || x.baseHash, journal: null }))
      scheduleAutosave(path)
    },

    discardJournal: async (path) => {
      const folder = get().folder
      patchBuffer(path, () => ({ journal: null }))
      if (folder) await invoke('novel:discardJournal', folder, path).catch(() => undefined)
    },

    restoreSnapshot: async (path, snapshot) => {
      const folder = get().folder
      const b = get().buffers[path]
      if (!folder || !b) return
      try {
        const text = await invoke('novel:readSnapshot', folder, snapshot)
        void endTracker(path, 'reloaded', true)
        patchBuffer(path, (x) => ({ text, version: x.version + 1, dirty: text !== x.savedText }))
        const ok = await get().save(path, { snapshot: true })
        if (ok) toast.success('Restored. The text from before is in the history.')
      } catch (e) {
        toast.error(`Could not restore: ${errorMessage(e)}`)
      }
    },

    createMissingFile: async (path) => {
      const folder = get().folder
      if (!folder) return
      const r = await invoke('novel:saveFile', { folder, path, text: '', baseHash: '' }).catch((e) => ({ ok: false as const, reason: 'error' as const, error: errorMessage(e) }))
      if (!r.ok) return void toast.error(r.reason === 'conflict' ? 'A file appeared there meanwhile; reload the project.' : 'Could not create the file.')
      set((s) => {
        const rest = { ...s.buffers }
        delete rest[path]
        return { buffers: rest }
      })
      await ensureBuffer(path)
      await get().reloadState()
    },

    patch: async (ops) => {
      const folder = get().folder
      if (!folder || get().state?.blocked) return false
      try {
        applyState(await invoke('novel:outlinePatch', folder, ops))
        return true
      } catch (e) {
        toast.error(errorMessage(e))
        return false
      }
    },

    setScene: async (sceneId, field, value) => {
      await get().patch([{ op: 'setScene', sceneId, field, value }])
    },

    patchProject: async (patch) => {
      const folder = get().folder
      if (!folder) return
      // Show the change at once so a field doesn't snap back while the save runs.
      set((s) => (s.state ? { state: { ...s.state, project: { ...s.state.project, ...patch } as NovelProject } } : s))
      try {
        const project = await invoke('novel:projectPatch', folder, patch)
        set((s) => (s.state ? { state: { ...s.state, project } } : s))
      } catch (e) {
        toast.error(errorMessage(e))
        await get().reloadState()
      }
    },

    createEntry: async (name, kind) => {
      const folder = get().folder
      if (!folder) return
      try {
        const entry: BibleEntry = await invoke('novel:createEntry', folder, name, kind)
        set((s) => (s.state ? { state: { ...s.state, bible: [...s.state.bible.filter((e) => e.id !== entry.id), entry] } } : s))
        await get().selectEntry(entry.id)
      } catch (e) {
        toast.error(`Could not create the entry: ${errorMessage(e)}`)
      }
    },

    deleteScene: async (sceneId) => {
      const folder = get().folder
      const found = sceneOf(get().state, sceneId)
      if (!folder || !found) return
      const path = scenePath(found.scene)
      try {
        await get().save(path)
        const st = await invoke('novel:deleteScene', folder, sceneId)
        set((s) => {
          const rest = { ...s.buffers }
          delete rest[path]
          return { buffers: rest }
        })
        applyState(st)
        toast.success('Moved to the project trash.')
      } catch (e) {
        toast.error(`Nothing was deleted: ${errorMessage(e)}`)
      }
    },

    deleteChapter: async (chapterId) => {
      const folder = get().folder
      const chapter = get().state?.outline.chapters.find((c) => c.id === chapterId)
      if (!folder || !chapter) return
      try {
        for (const sc of chapter.scenes) await get().save(scenePath(sc))
        const st = await invoke('novel:deleteChapter', folder, chapterId)
        set((s) => {
          const rest = { ...s.buffers }
          for (const sc of chapter.scenes) delete rest[scenePath(sc)]
          return { buffers: rest }
        })
        applyState(st)
        toast.success('Moved to the project trash.')
      } catch (e) {
        toast.error(`Nothing was deleted: ${errorMessage(e)}`)
      }
    },

    deleteEntry: async (entryId) => {
      const folder = get().folder
      if (!folder) return
      try {
        await get().save(entryPath(entryId))
        const st = await invoke('novel:deleteEntry', folder, entryId)
        set((s) => {
          const rest = { ...s.buffers }
          delete rest[entryPath(entryId)]
          return { buffers: rest, target: s.target?.kind === 'entry' && s.target.id === entryId ? null : s.target }
        })
        applyState(st)
        toast.success('Moved to the project trash.')
      } catch (e) {
        toast.error(`Nothing was deleted: ${errorMessage(e)}`)
      }
    },

    restoreTrash: async (id) => {
      const folder = get().folder
      if (!folder) return
      try {
        applyState(await invoke('novel:restoreTrash', folder, id))
        toast.success('Restored.')
      } catch (e) {
        toast.error(`Could not restore: ${errorMessage(e)}`)
      }
    },

    addChapter: async (title) => {
      await get().patch([{ op: 'addChapter', chapterId: newId('c'), sceneId: newId('s'), title, index: null }])
    },

    addScene: async (chapterId, title) => {
      const sceneId = newId('s')
      const ok = await get().patch([{ op: 'addScene', chapterId, sceneId, title, index: null }])
      if (ok) await get().selectScene(sceneId)
    },

    adoptFile: async (file, chapterId) => {
      const sceneId = newId('s')
      const ok = await get().patch([{ op: 'adoptFile', file, chapterId, sceneId, index: null }])
      if (ok) await get().selectScene(sceneId)
    },

    ensureHashes: async () => {
      const { folder, state, hashes } = get()
      if (!folder || !state) return
      const missing = state.outline.chapters.flatMap((c) => c.scenes.map(scenePath)).filter((p) => hashes[p] === undefined && !state.missing.includes(p))
      for (let i = 0; i < missing.length; i += 8) {
        const got = await Promise.all(
          missing.slice(i, i + 8).map((p) =>
            invoke('novel:readFile', folder, p).then(
              (f) => [p, f.hash] as const,
              () => null
            )
          )
        )
        set((s) => ({ hashes: { ...s.hashes, ...Object.fromEntries(got.filter((x): x is readonly [string, string] => !!x)) } }))
      }
    },

    // ---- AI actions --------------------------------------------------------------------

    runAction: async (opts) => {
      const s = get()
      const t = s.target
      if (!s.state || s.state.blocked || t?.kind !== 'scene') return
      if (busy() || s.pending) return
      const found = sceneOf(s.state, t.id)
      if (!found) return
      const path = scenePath(found.scene)
      const handle = editors.get(path)
      if (!handle) return
      const why = aiBusyReason('writing')
      if (why) return void toast.message(why)
      // Where to work is read now, before anything is awaited: the user may move on meanwhile.
      const sel = handle.selection()
      const at = opts.at ?? { cursor: sel.from, selection: sel.from === sel.to ? null : sel }
      const needsSelection = opts.action === 'rewrite' || opts.action === 'expand' || opts.action === 'condense'
      if (needsSelection && !at.selection) return void toast.message('Select the text to change first.')
      const alternatives = Math.min(4, Math.max(1, opts.alternatives ?? 1))
      const label = opts.action === 'rewrite' && opts.variant ? `${ACTION_LABELS.rewrite}: ${opts.variant}` : ACTION_LABELS[opts.action]
      // The run's identity: this project open, this file. Every await below re-checks it.
      const epoch0 = epoch
      const folder0 = s.folder
      const stale = (): boolean => epoch !== epoch0 || get().folder !== folder0 || !get().buffers[path]
      stopRequested = false
      // A new action on the scene ends the tracking of the last accepted change in it.
      void endTracker(path, 'action')
      const recId = recordId()
      // gen is set first: it holds the text still (no edits, no reload) from here to the end.
      set({ gen: { path, action: opts.action, label, phase: 'preparing', alt: 1, of: alternatives, progress: null, text: '' }, pendingRefused: false })
      const finish = (): void => {
        // A run of an earlier project must not clear the state of the current one.
        if (epoch !== epoch0) return
        activeGen = null
        set({ gen: null })
      }
      // A clean buffer first: the pending change is made against saved text.
      if (!(await get().save(path))) {
        finish()
        return void toast.error('Save the text first: it could not be saved, so nothing was written.')
      }
      const buf = get().buffers[path]
      if (stale() || !buf || buf.readOnly) return finish()
      try {
        const model = await resolveModel()
        if (!model || stale()) return finish()
        await get().ensureHashes()
        const previousText = opts.action === 'draftScene' && !buf.text.trim() ? await previousSceneText(t.id) : ''
        if (stale() || get().buffers[path]?.version !== buf.version) return finish()
        const estimated = { value: false }
        const base = await baseContext(
          model,
          {
            action: opts.action,
            sceneId: t.id,
            text: buf.text,
            cursor: Math.min(at.cursor, buf.text.length),
            selection: at.selection,
            note: opts.note ?? '',
            variant: opts.variant,
            previousText
          },
          estimated
        )
        if (!base || stale()) return finish()
        const fit = await fitRequest(base.ctx, base.count, countWhole(model.modelId), estimated)
        if (stale() || get().buffers[path]?.version !== buf.version) return finish()
        set({ lastRequest: { source: 'run', action: opts.action, at: Date.now(), built: fit.built, actual: fit.actual, tooBig: fit.tooBig, contextLength: model.contextLength } })
        if (fit.tooBig) {
          set({ panelOpen: true, panelTab: 'context' })
          toast.error('This request does not fit in the context. The Context tab shows what is using it.')
          return finish()
        }
        const project = base.ctx.project
        const seeds = seedsFor(project.sampling.seed, alternatives)
        const alts: PendingAlt[] = []
        const finishes: RecordAlt['finish'][] = []
        let failure: string | null = null
        const mk = (): PendingChange => ({
          id: uid(),
          recId,
          viewed: [0],
          path,
          sceneId: t.id,
          action: opts.action,
          label,
          baseText: buf.text,
          baseHash: buf.baseHash,
          version: buf.version,
          alts: [...alts],
          active: 0,
          newSticky: fit.built.newSticky,
          again: { action: opts.action, variant: opts.variant, note: opts.note ?? '', selection: at.selection, cursor: at.cursor, count: alternatives }
        })
        for (let i = 0; i < alternatives && !stopRequested && !stale(); i++) {
          set((x) => (x.gen ? { gen: { ...x.gen, alt: i + 1, phase: 'waiting', progress: null, text: '' } } : x))
          const gen = startGeneration({
            modelId: model.modelId,
            messages: fit.built.result.messages,
            sampling: project.sampling,
            adapters: project.adapters,
            action: opts.action,
            seed: seeds[i],
            maxTokens: project.budget.replyTokens,
            onProgress: (p) => set((x) => (x.gen ? { gen: { ...x.gen, phase: 'waiting', progress: p } } : x)),
            onDelta: pushStreamText
          })
          activeGen = gen
          const res = await gen.done
          if (epoch === epoch0) activeGen = null
          // The project was closed or the buffer is gone: this result belongs to nobody now.
          if (stale()) break
          const generated = finishText(opts.action, res.text)
          if (res.error) {
            failure = res.error
            break
          }
          if (!generated) {
            if (!res.stopped) failure = 'The model returned no text.'
            break
          }
          alts.push({ text: applyResult(opts.action, buf.text, Math.min(at.cursor, buf.text.length), at.selection, generated), generated, seed: seeds[i] })
          finishes.push(finishOf(res.stopReason, res))
          set((x) => ({ pending: x.pending ? { ...x.pending, alts: [...alts] } : mk() }))
          if (res.stopped) break
        }
        if (failure && !stale()) toast.error(alts.length ? `Stopped after ${alts.length} of ${alternatives}: ${failure}` : failure)
        // The run's request, once, when it produced at least one alternative.
        if (alts.length && folder0 && !stale()) {
          sendRequestRecord(folder0, {
            recId,
            action: opts.action,
            variant: opts.variant,
            sceneId: t.id,
            chapterId: found.chapterId,
            model,
            project,
            sampling: samplingFor(project.sampling, { seed: seeds[0], maxTokens: project.budget.replyTokens, action: opts.action }),
            messages: fit.built.result.messages,
            alts: alts.map((a, i) => ({ index: i, seed: a.seed, generated: a.generated, finish: finishes[i] ?? 'stop' }))
          })
        }
      } catch (e) {
        toast.error(errorMessage(e))
      } finally {
        finish()
      }
    },

    stop: () => {
      stopRequested = true
      activeGen?.abort()
    },

    selectAlt: (index) => {
      set((s) =>
        s.pending && index >= 0 && index < s.pending.alts.length
          ? { pending: { ...s.pending, active: index, viewed: s.pending.viewed && !s.pending.viewed.includes(index) ? [...s.pending.viewed, index] : s.pending.viewed } }
          : s
      )
    },

    acceptAll: async () => {
      const p = get().pending
      if (!p) return
      const text = editors.get(p.path)?.pendingAcceptAll() ?? p.alts[p.active]?.text ?? p.baseText
      await get().resolvePending(text)
    },

    rejectAll: async () => {
      const p = get().pending
      if (!p) return
      const text = editors.get(p.path)?.pendingRejectAll() ?? p.baseText
      await get().resolvePending(text)
    },

    resolvePending: async (finalText) => {
      const p = get().pending
      if (!p) return
      const b = get().buffers[p.path]
      if (!b) {
        abandonRun(get().folder, p.recId)
        return void set({ pending: null })
      }
      if (finalText === p.baseText) {
        recordProseOutcome(p, finalText)
        return void set({ pending: null, pendingRefused: false })
      }
      if (pendingStale(p, b)) {
        // The change can no longer be accepted: abandoned (once; Regenerate or Reject after it add nothing).
        abandonRun(get().folder, p.recId)
        toast.error(STALE_PENDING)
        return void set({ pendingRefused: true })
      }
      // The buffer takes the text, then one save with a history snapshot (Accept, data safety rule 5).
      set((s) => ({ pending: null, pendingRefused: false, buffers: { ...s.buffers, [p.path]: { ...b, text: finalText, version: b.version + 1, dirty: finalText !== b.savedText } } }))
      clearTimer(autosaveTimers, p.path)
      recordProseOutcome(p, finalText)
      // The accepted text is in the journal at once, even if the save has to wait behind another one.
      const folder = get().folder
      if (folder) void invoke('novel:journal', folder, p.path, finalText, b.baseHash).catch(() => undefined)
      scheduleJournal(p.path)
      const ok = await get().save(p.path, { snapshot: true })
      const found = sceneOf(get().state, p.sceneId)
      if (!found) return
      const ops: OutlineOp[] = []
      const fresh = p.newSticky.filter((id) => !found.scene.sticky.includes(id))
      if (fresh.length) ops.push({ op: 'setScene', sceneId: p.sceneId, field: 'sticky', value: [...found.scene.sticky, ...fresh] })
      if (p.action === 'draftScene' && (found.scene.status === 'idea' || found.scene.status === 'beats')) ops.push({ op: 'setScene', sceneId: p.sceneId, field: 'status', value: 'draft' })
      if (ops.length) await get().patch(ops)
      if (!ok && !get().buffers[p.path]?.conflict) toast.error('Accepted, but saving failed. The text stays in the editor and will be saved again.')
    },

    regenerate: async () => {
      const p = get().pending
      if (!p) return
      const again = p.again
      abandonRun(get().folder, p.recId)
      set({ pending: null, pendingRefused: false })
      await get().runAction({ action: again.action, variant: again.variant, note: again.note, alternatives: again.count, at: { cursor: again.cursor, selection: again.selection } })
    },

    // ---- structured proposals ------------------------------------------------------------

    summarizeScene: async (sceneId) => {
      const found = sceneOf(get().state, sceneId)
      if (!found) return
      const src = await sceneSource(sceneId, true)
      if (!src) return void toast.error('Save the scene first: it could not be saved.')
      if (!src.text.trim()) return void toast.message('The scene has no text to summarize.')
      await get().ensureHashes()
      await runStructured(
        'summaryScene',
        'summarizeScene',
        { sceneId, text: src.text, cursor: src.text.length },
        { targetId: sceneId, title: `Summary of "${found.scene.title}"`, summaryOf: src.hash, again: { count: 1, note: '' } }
      )
    },

    summarizeChapter: async (chapterId) => {
      const chapter = get().state?.outline.chapters.find((c) => c.id === chapterId)
      if (!chapter) return
      for (const sc of chapter.scenes) {
        const path = scenePath(sc)
        if (get().buffers[path]) await get().save(path)
      }
      await get().ensureHashes()
      const hashes = get().hashes
      const key = sceneHashKey(chapter.scenes.map((sc) => hashes[scenePath(sc)] ?? ''))
      await runStructured('summaryChapter', 'summarizeChapter', { chapterId }, { targetId: chapterId, title: `Summary of "${chapter.title}"`, summaryOf: key, again: { count: 1, note: '' } })
    },

    summarizeAll: async () => {
      const s = get()
      if (!s.state) return
      await get().ensureHashes()
      const h = summaryHealth(s.state.outline, get().hashes, s.state.words)
      const queue = [...h.missing, ...h.stale]
      if (!queue.length) return void toast.message('Every written scene has a current summary.')
      set({ summarizeQueue: queue.slice(1) })
      await get().summarizeScene(queue[0])
    },

    planChapters: async (count, note) => {
      await runStructured('chapters', 'outlineChapters', { note, count }, { targetId: null, title: `Plan ${count} chapters`, summaryOf: '', again: { count, note } })
    },

    planScenes: async (chapterId, count, note) => {
      const chapter = get().state?.outline.chapters.find((c) => c.id === chapterId)
      if (!chapter) return
      await runStructured('scenes', 'outlineScenes', { chapterId, note, count }, { targetId: chapterId, title: `Plan ${count} scenes for "${chapter.title}"`, summaryOf: '', again: { count, note } })
    },

    acceptProposal: async (edited) => {
      const p = edited ?? get().proposal
      if (!p || p.status !== 'ready') return
      const ops: OutlineOp[] = []
      if (p.kind === 'summaryScene' && p.targetId && p.summary) {
        ops.push({ op: 'setScene', sceneId: p.targetId, field: 'summary', value: p.summary.summary.trim() })
        ops.push({ op: 'setScene', sceneId: p.targetId, field: 'summaryOf', value: p.summaryOf })
      } else if (p.kind === 'summaryChapter' && p.targetId && p.summary) {
        ops.push({ op: 'setChapter', chapterId: p.targetId, field: 'summary', value: p.summary.summary.trim() })
        ops.push({ op: 'setChapter', chapterId: p.targetId, field: 'summaryOf', value: p.summaryOf })
      } else if (p.kind === 'chapters' && p.chapters) {
        for (const c of p.chapters.chapters) {
          const chapterId = newId('c')
          ops.push({ op: 'addChapter', chapterId, sceneId: newId('s'), title: c.title.trim() || 'Untitled chapter', index: null })
          if (c.act.trim()) ops.push({ op: 'setChapter', chapterId, field: 'act', value: c.act.trim() })
          if (c.summary.trim()) ops.push({ op: 'setChapter', chapterId, field: 'summary', value: c.summary.trim() })
        }
      } else if (p.kind === 'scenes' && p.targetId && p.scenes) {
        const refs = resolveSceneRefs(get().state?.bible ?? [], p.scenes.scenes).scenes
        for (const [i, sc] of p.scenes.scenes.entries()) {
          const sceneId = newId('s')
          ops.push({ op: 'addScene', chapterId: p.targetId, sceneId, title: sc.title.trim() || 'Untitled scene', index: null, beats: sc.beats })
          if (refs[i].pov) ops.push({ op: 'setScene', sceneId, field: 'pov', value: refs[i].pov })
          const present = refs[i].present
          if (present.length) ops.push({ op: 'setScene', sceneId, field: 'present', value: present })
          ops.push({ op: 'setScene', sceneId, field: 'status', value: sc.beats.trim() ? 'beats' : 'idea' })
        }
      }
      if (!ops.length) return void set({ proposal: null })
      const ok = await get().patch(ops)
      if (!ok) return
      // Resolved: accepted as written, or edited (the edited text is recorded).
      const original = get().proposal
      if (edited && original && cardText(edited) !== cardText(original)) recordCardOutcome(original.recId, 'edited', cardText(edited))
      else recordCardOutcome((original ?? p).recId, 'accepted')
      set({ proposal: null })
      const queue = get().summarizeQueue
      if (queue.length) {
        set({ summarizeQueue: queue.slice(1) })
        await get().summarizeScene(queue[0])
      }
    },

    rejectProposal: () => {
      recordCardOutcome(get().proposal?.recId, 'rejected')
      set({ proposal: null, summarizeQueue: [] })
    },

    retryProposal: async () => {
      const p = get().proposal
      if (!p) return
      abandonRun(get().folder, p.recId)
      set({ proposal: null })
      if (p.kind === 'summaryScene' && p.targetId) await get().summarizeScene(p.targetId)
      else if (p.kind === 'summaryChapter' && p.targetId) await get().summarizeChapter(p.targetId)
      else if (p.kind === 'chapters') await get().planChapters(p.again.count, p.again.note)
      else if (p.kind === 'scenes' && p.targetId) await get().planScenes(p.targetId, p.again.count, p.again.note)
    },

    previewContext: async () => {
      const s = get()
      const t = s.target
      if (!s.state || t?.kind !== 'scene' || s.gen || s.contextBusy) return
      const found = sceneOf(s.state, t.id)
      if (!found) return
      const path = scenePath(found.scene)
      const buf = s.buffers[path]
      if (!buf) return
      set({ contextBusy: true })
      try {
        const modelId = s.state.project.modelId ?? useModels.getState().activeModelId
        const inst = readyInstance(modelId)
        const model = inst ? { modelId: inst.modelId, contextLength: inst.contextLength ?? 8192, instanceId: inst.instanceId } : { modelId: modelId ?? '', contextLength: 8192, instanceId: '' }
        await get().ensureHashes()
        const empty = !buf.text.trim()
        const action: NovelAction = empty ? 'draftScene' : 'continue'
        const estimated = { value: !inst }
        const previousText = empty ? await previousSceneText(t.id) : ''
        const cursor = Math.min(editors.get(path)?.selection().from ?? buf.text.length, buf.text.length)
        // Without a loaded model there is nothing to count with: the counts are estimates.
        const base = await baseContext(model, { action, sceneId: t.id, text: buf.text, cursor, previousText }, estimated)
        if (!base) return
        const count = inst ? base.count : async (text: string): Promise<number> => Math.ceil(text.length / 3)
        const fit = await fitRequest(base.ctx, count, inst ? countWhole(model.modelId) : async () => null, estimated)
        set({ lastRequest: { source: 'preview', action, at: Date.now(), built: { ...fit.built, estimated: estimated.value }, actual: fit.actual, tooBig: fit.tooBig, contextLength: model.contextLength } })
      } catch {
        // A preview that fails shows nothing; running an action reports the reason.
      } finally {
        set({ contextBusy: false })
      }
    },

    removeSticky: async (sceneId, entryId) => {
      const found = sceneOf(get().state, sceneId)
      if (!found) return
      await get().setScene(sceneId, 'sticky', found.scene.sticky.filter((id) => id !== entryId))
    },

    // ---- stage 3: checks -----------------------------------------------------------------

    checkScene: async (sceneId) => {
      const s = get()
      const found = sceneOf(s.state, sceneId)
      if (!s.state || !found) return
      if (busy() || s.pending) return void toast.message('Wait for the current request to finish first.')
      const why = aiBusyReason('writing')
      if (why) return void toast.message(why)
      stopRequested = false
      const r = await checkOne(sceneId, `Check: ${found.scene.title}`)
      if (r === 'empty') return void toast.message('The scene has no text to check.')
      if (r === 'done') {
        set({ panelOpen: true, panelTab: 'checks' })
        const n = get().findings[sceneId]?.items.length ?? 0
        if (n === 0) toast.success('No problems found in this scene.')
      }
    },

    checkChapter: async (chapterId) => {
      const s = get()
      const chapter = s.state?.outline.chapters.find((c) => c.id === chapterId)
      if (!chapter) return
      if (busy() || s.pending) return void toast.message('Wait for the current request to finish first.')
      const why = aiBusyReason('writing')
      if (why) return void toast.message(why)
      const ids = chapter.scenes.map((sc) => sc.id)
      const epoch0 = epoch
      stopRequested = false
      set({ checkRun: { chapterId, sceneIds: ids, index: 0 } })
      let checked = 0
      const tooLong: string[] = []
      try {
        for (let i = 0; i < ids.length; i++) {
          if (stopRequested || epoch !== epoch0) break
          set((x) => (x.checkRun ? { checkRun: { ...x.checkRun, index: i } } : x))
          const title = sceneOf(get().state, ids[i])?.scene.title ?? ''
          const r = await checkOne(ids[i], `Check ${i + 1} of ${ids.length}: ${title}`)
          if (r === 'done') checked++
          else if (r === 'refused') tooLong.push(title)
          else if (r === 'failed' || r === 'stopped') break
        }
      } finally {
        if (epoch === epoch0) set({ checkRun: null })
      }
      if (epoch !== epoch0) return
      if (checked > 0) set({ panelOpen: true, panelTab: 'checks' })
      const total = Object.entries(get().findings).filter(([id]) => ids.includes(id)).reduce((n, [, e]) => n + e.items.length, 0)
      toast.message(`Checked ${checked} of ${ids.length} scenes in "${chapter.title}": ${total} ${total === 1 ? 'finding' : 'findings'}.${tooLong.length ? ` Not sent: ${tooLong.join(', ')}.` : ''}`)
    },

    dismissFinding: async (sceneId, finding, dismissed = true) => {
      const { folder, state } = get()
      if (!folder || !state) return
      if (state.blocked || state.checksProblem) return void toast.error(state.checksProblem ? `Dismissals cannot be saved: ${state.checksProblem}` : 'The project is read-only.')
      const epoch0 = epoch
      try {
        const next = await invoke('novel:dismissFinding', folder, sceneId, findingKey(finding), dismissed)
        if (epoch === epoch0 && get().folder === folder) applyState(next)
      } catch (e) {
        toast.error(errorMessage(e))
      }
    },

    restoreDismissed: async (sceneId) => {
      const { folder, state } = get()
      if (!folder || !state || state.blocked || state.checksProblem) return
      const keys = state.dismissed?.[sceneId] ?? []
      const epoch0 = epoch
      try {
        let last: NovelProjectState | null = null
        for (const key of keys) last = await invoke('novel:dismissFinding', folder, sceneId, key, false)
        if (last && epoch === epoch0 && get().folder === folder) applyState(last)
      } catch (e) {
        toast.error(errorMessage(e))
      }
    },

    jumpToFinding: (sceneId, finding) => {
      const found = sceneOf(get().state, sceneId)
      const range = findingRange(sceneId, finding)
      if (!found || !range) return void toast.message('That text is not in the scene any more.')
      editors.get(scenePath(found.scene))?.reveal(range.from, range.to)
    },

    fixFinding: async (sceneId, finding) => {
      const s = get()
      if (s.target?.kind !== 'scene' || s.target.id !== sceneId) return void toast.message('Open the scene first.')
      const range = findingRange(sceneId, finding)
      if (!range) return void toast.message('That text is not in the scene any more.')
      // A normal Rewrite → Custom: it becomes a pending change to accept or reject.
      await get().runAction({ action: 'rewrite', variant: 'custom', note: finding.suggestion.trim() || finding.problem, alternatives: 1, at: { cursor: range.from, selection: range } })
    },

    clearCheckIssue: () => set({ checkIssue: null }),

    // ---- stage 3: tracked state ----------------------------------------------------------

    updateState: async (sceneId) => {
      const s = get()
      const found = sceneOf(s.state, sceneId)
      if (!s.state || !found) return
      if (s.state.blocked) return void toast.error('The project is read-only.')
      if (s.state.state?.problem) return void toast.error(`state.json cannot be used: ${s.state.state.problem}`)
      if (busy() || s.pending) return void toast.message('Wait for the current request to finish first.')
      const why = aiBusyReason('writing')
      if (why) return void toast.message(why)
      if (s.stateProposal && (s.stateProposal.status === 'ready' || s.stateProposal.status === 'running')) return void toast.message('Apply or reject the state update on the card first.')
      const id = uid()
      set({
        stateProposal: { id, sceneId, title: `State update for "${found.scene.title}"`, status: 'running', textHash: '', raw: '', error: null, items: [], left: [], refused: null, rejected: [], applied: 0, applying: false },
        panelOpen: true
      })
      stopRequested = false
      const epoch0 = epoch
      const r = await runWholeScene(sceneId, 'updateState', `Update state: ${found.scene.title}`)
      if (epoch !== epoch0 || get().stateProposal?.id !== id) return
      const patch = (p: Partial<StateProposal>): void => set((x) => (x.stateProposal?.id === id ? { stateProposal: { ...x.stateProposal, ...p } } : x))
      if (r.status === 'done') {
        const parsed = parseUpdateStateReplyResult(r.raw)
        if (!parsed.ok) return patch({ status: 'invalid', raw: r.raw, error: parsed.error })
        const st = get().state
        const before = st ? stateBefore(factsNow(), sceneId, st.outline, st.bible) : []
        const res = resolveStateChanges(parsed.value.changes, st?.bible ?? [], before)
        return patch({ status: 'ready', textHash: r.textHash, raw: r.raw, items: res.items, left: res.left })
      }
      if (r.status === 'failed') return patch({ status: 'invalid', raw: r.raw, error: r.error })
      set({ stateProposal: null })
    },

    setStateItem: (key, accepted) => {
      set((s) => (s.stateProposal ? { stateProposal: { ...s.stateProposal, items: s.stateProposal.items.map((i) => (i.key === key ? { ...i, accepted } : i)) } } : s))
    },

    applyStateProposal: async () => {
      const p = get().stateProposal
      if (!p || p.status !== 'ready' || p.applying) return
      const patch = (x: Partial<StateProposal>): void => set((s) => (s.stateProposal?.id === p.id ? { stateProposal: { ...s.stateProposal, ...x } } : s))
      if (!p.items.some((i) => i.accepted)) return void toast.message('Nothing is accepted. Accept at least one change, or reject the card.')
      // Locked and identified before anything is awaited: a second click does nothing, and a close or switch during the read sends nothing.
      const epoch0 = epoch
      const folder0 = get().folder
      patch({ applying: true, refused: null })
      const gone = (): boolean => epoch !== epoch0 || get().folder !== folder0 || get().stateProposal?.id !== p.id
      // The card belongs to the scene text it read: if that moved, nothing is written.
      const src = await sceneSource(p.sceneId, false)
      if (gone()) return
      if (!src || hashText(src.text) !== p.textHash) {
        const msg = 'The scene text changed since this was made. Run Update state again.'
        toast.error(msg)
        return patch({ refused: msg, applying: false })
      }
      const { ops, labels } = stateOpsFor(p.items, p.sceneId)
      const res = await sendStatePatch(ops)
      if (epoch !== epoch0 || get().stateProposal?.id !== p.id) return
      if (!res) return patch({ applying: false })
      const rejected = res.rejected.map((r) => ({ label: labels[r.index] ?? 'A change', reason: r.reason }))
      if (rejected.length === 0) {
        set({ stateProposal: null })
        return void toast.success(`State updated: ${ops.length} ${ops.length === 1 ? 'change' : 'changes'}.`)
      }
      patch({ status: 'applied', applying: false, rejected, applied: ops.length - rejected.length })
    },

    discardStateProposal: () => {
      const p = get().stateProposal
      if (p?.status === 'running') {
        stopRequested = true
        activeGen?.abort()
      }
      set({ stateProposal: null })
    },

    statePatch: (ops) => sendStatePatch(ops),

    // ---- stage 3: interview and voice notes ----------------------------------------------

    askInterview: async (characterId, question) => {
      const s = get()
      const q = question.trim()
      const st = s.state
      const entry = st?.bible.find((e) => e.id === characterId)
      const sceneId = s.target?.kind === 'scene' ? s.target.id : s.lastSceneId
      if (!st || !q || !entry) return false
      if (!sceneId || !sceneOf(st, sceneId)) {
        toast.message('Open a scene first: the character knows what happened before it.')
        return false
      }
      if (busy()) {
        toast.message('Wait for the current request to finish first.')
        return false
      }
      const why = aiBusyReason('writing')
      if (why) {
        toast.message(why)
        return false
      }
      const key = interviewKey(characterId, sceneId)
      const epoch0 = epoch
      const folder0 = s.folder
      const stale = (): boolean => epoch !== epoch0 || get().folder !== folder0
      stopRequested = false
      // Its own request: no scene buffer is held or touched, and no pending change is made. The key says whose answer is streaming.
      set({ gen: { path: null, action: 'interview', label: `Interview: ${entry.name}`, phase: 'preparing', alt: 1, of: 1, progress: null, text: '' }, interviewRun: { key } })
      try {
        const model = await resolveModel()
        if (!model || stale() || stopRequested) return false
        const cur = get().state
        if (!cur) return false
        const buf = get().buffers[entryPath(characterId)]
        const body = buf ? parseEntry(characterId, buf.text).body : entry.body
        const system = buildInterviewSystem({ entry, body, facts: factsNow(), sceneId, outline: cur.outline, bible: cur.bible })
        const project = cur.project
        const limit = model.contextLength - project.budget.replyTokens - Math.max(256, Math.ceil(model.contextLength * 0.03))
        const fit = await fitInterview(system, get().interviews[key] ?? [], q, countWhole(model.modelId), limit)
        if (stale() || stopRequested) return false
        if (fit.tooBig) {
          toast.error('This does not fit in the context. Shorten the character notes, or load a model with a longer context.')
          return false
        }
        const gen = startGeneration({
          modelId: model.modelId,
          messages: fit.messages,
          sampling: project.sampling,
          adapters: project.adapters,
          action: 'interview',
          prose: true,
          seed: seedsFor(project.sampling.seed, 1)[0],
          maxTokens: project.budget.replyTokens,
          onProgress: (p) => set((x) => (x.gen ? { gen: { ...x.gen, phase: 'waiting', progress: p } } : x)),
          onDelta: pushStreamText
        })
        activeGen = gen
        const res = await gen.done
        if (epoch === epoch0) activeGen = null
        if (stale()) return false
        if (res.error) {
          toast.error(res.error)
          return false
        }
        const answer = res.text.trim()
        if (!answer) {
          if (!res.stopped) toast.error('The model returned no text.')
          return false
        }
        set((x) => ({
          interviews: { ...x.interviews, [key]: [...(x.interviews[key] ?? []), { role: 'user', content: q }, { role: 'assistant', content: answer }] },
          interviewDropped: { ...x.interviewDropped, [key]: fit.dropped }
        }))
        return true
      } catch (e) {
        if (!stale()) toast.error(errorMessage(e))
        return false
      } finally {
        if (epoch === epoch0) {
          activeGen = null
          set({ gen: null, interviewRun: null })
        }
      }
    },

    clearInterview: (key) => {
      set((s) => {
        const { [key]: _a, ...rest } = s.interviews
        const { [key]: _b, ...dropped } = s.interviewDropped
        return { interviews: rest, interviewDropped: dropped }
      })
    },

    addVoiceNote: async (entryId, note) => {
      const path = entryPath(entryId)
      if (!get().folder || !note.trim()) return false
      const refuse = (): boolean => {
        const why = voiceNoteBlock(get(), path)
        if (why) toast.error(`Not added to the voice notes. ${why}`)
        return !!why
      }
      if (refuse()) return false
      const epoch0 = epoch
      const folder0 = get().folder
      if (!(await ensureBuffer(path))) return false
      if (epoch !== epoch0 || get().folder !== folder0) return false
      if (refuse()) return false
      const b = get().buffers[path]
      if (!b) return false
      const next = appendVoiceNote(b.text, note)
      if (next === b.text) return false
      // The entry's own buffer and its normal save path (journal, autosave, conflict checks).
      get().edit(path, next)
      toast.success('Added to the voice notes.')
      return true
    }
  }

  // Opens a state the main process returned: buffers for style.md, last scene or first scene.
  async function enterProject(st: NovelProjectState): Promise<void> {
    epoch++
    dropTrackers()
    clearAllTimers()
    clearCountCache()
    set({
      folder: st.folder,
      state: st,
      target: null,
      buffers: { [STYLE_PATH]: makeBuffer(st.style) },
      hashes: {},
      gen: null,
      pending: null,
      pendingRefused: false,
      proposal: null,
      findings: {},
      checkRun: null,
      checkIssue: null,
      stateProposal: null,
      lastSceneId: null,
      interviews: {},
      interviewDropped: {},
      interviewRun: null,
      summarizeQueue: [],
      lastRequest: null,
      openError: null,
      selection: { from: 0, to: 0 }
    })
    remember(LAST_PROJECT, st.folder)
    const last = recall(sceneKey(st.folder))
    const pick = last && sceneOf(st, last) ? last : firstSceneId(st)
    if (pick) await get().selectScene(pick)
    void get().ensureHashes()
  }
})

/** The folder opened last, to reopen it when the page is shown again. */
export function lastProjectFolder(): string | null {
  return recall(LAST_PROJECT)
}

/** Resolves to `fallback` if `p` has not settled within `ms`. */
function within<T>(p: Promise<T>, ms: number, fallback: T): Promise<T> {
  return new Promise((resolve) => {
    const t = setTimeout(() => resolve(fallback), Math.max(0, ms))
    p.then(
      (v) => {
        clearTimeout(t)
        resolve(v)
      },
      () => {
        clearTimeout(t)
        resolve(fallback)
      }
    )
  })
}

/** Total time a flush may take: under the 3 s the main process waits for the answer to novel:flush. */
export const FLUSH_BUDGET_MS = 2600
const JOURNAL_WAIT_MS = 1000

export interface FlushResult {
  ok: boolean
  /** Paths whose current text is neither saved nor journalled. */
  failed: string[]
}

/**
 * Saves or journals every dirty buffer: first the journal (the safe copy, each wait bounded), then a
 * save with what time is left. A save already in flight is waited for (the queue does that), and text
 * typed during the flush gets another round. `ok` is false when some buffer's current text is neither
 * on disk nor in the journal, so the caller must not close the project or the window on it.
 */
export async function flushAll(): Promise<FlushResult> {
  const deadline = Date.now() + FLUSH_BUDGET_MS
  const left = (): number => deadline - Date.now()
  clearAllTimers()
  if (!useNovel.getState().folder) return { ok: true, failed: [] }
  // The last final of every followed scene goes out at once (fire and forget; awaited only briefly at the end).
  const finals = flushFinals()
  const safe = new Map<string, string>()
  for (let round = 0; round < 3 && left() > 0; round++) {
    const { folder, buffers } = useNovel.getState()
    if (!folder) break
    const todo = Object.values(buffers).filter((b) => b.dirty && !b.readOnly && safe.get(b.path) !== b.text)
    if (todo.length === 0) break
    await Promise.all(
      todo.map(async (b) => {
        const journalled = await within(
          invoke('novel:journal', folder, b.path, b.text, b.baseHash).then(() => true),
          Math.min(JOURNAL_WAIT_MS, left()),
          false
        )
        if (journalled) safe.set(b.path, b.text)
        if (b.conflict) return
        await within(useNovel.getState().save(b.path), left(), false)
      })
    )
    clearAllTimers()
  }
  await within(finals, Math.min(500, Math.max(0, left())), null)
  const { buffers } = useNovel.getState()
  const failed = Object.values(buffers)
    .filter((b) => b.dirty && !b.readOnly && safe.get(b.path) !== b.text)
    .map((b) => b.path)
  return { ok: failed.length === 0, failed }
}

/** Whether a scene's summary is missing / stale, for the Scene tab note. */
export function summaryState(scene: NovelScene, hash: string | undefined): 'none' | 'missing' | 'stale' | 'current' {
  if (!scene.summary.trim()) return 'missing'
  if (hash !== undefined && scene.summaryOf !== hash) return 'stale'
  return 'current'
}

