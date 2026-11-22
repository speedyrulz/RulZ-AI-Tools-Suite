// Writing mode: types shared by the main process (file store) and the renderer (Writing page).
// Design and data-safety rules: docs/writing-mode-design.md. This file is the contract; change it
// only together with both sides.
import type { LoadAdapter, SamplingParams } from './types'

export const NOVEL_FORMAT = 1

// ---------------------------------------------------------------------------------------------
// Project files
// ---------------------------------------------------------------------------------------------

export interface NovelBudget {
  /** Sent as max_tokens and kept free in the context. */
  replyTokens: number
  /** Share of what is left after system/style/task/scene for the prose tail (0–1). */
  proseShare: number
  /** Share of the available context for bible entries, fixed + triggered (0–1). */
  bibleShare: number
}

/** project.json. Unknown keys found on disk are kept by the store when it rewrites the file. */
export interface NovelProject {
  format: number
  id: string
  title: string
  createdAt: string
  updatedAt: string
  /** null = the active chat model. */
  modelId: string | null
  sampling: SamplingParams
  adapters: LoadAdapter[] | null
  budget: NovelBudget
  /** Free-form POV/tense line shown to the model next to style.md, e.g. "Close third, past tense". */
  povTense: string
  /** Stage 4: record AI actions in .rulz/records. Absent = on. */
  recording?: boolean
}

export type SceneStatus = 'idea' | 'beats' | 'draft' | 'revised'

export interface NovelScene {
  id: string
  title: string
  /** File name inside manuscript\, fixed at creation; never renamed on reorder. */
  file: string
  /** The plan for the scene. */
  beats: string
  /** Recap of what was written (accepted by the user). */
  summary: string
  /** Byte hash (see NovelTextFile.hash) of the scene file the summary was made from; '' = none. */
  summaryOf: string
  status: SceneStatus
  /** Bible entry id or null. */
  pov: string | null
  /** Bible entry ids ticked as present. */
  present: string[]
  /** Triggered entries kept for this scene (sticky), entry ids. */
  sticky: string[]
  targetWords: number | null
}

export interface NovelChapter {
  id: string
  title: string
  /** Optional grouping label such as "Act I". */
  act: string
  summary: string
  /** Combined hash of the chapter's scene hashes the summary was made from; '' = none. */
  summaryOf: string
  /** At least one scene. */
  scenes: NovelScene[]
}

/** outline.json */
export interface NovelOutline {
  format: number
  chapters: NovelChapter[]
}

export type BibleKind = 'character' | 'place' | 'lore' | 'other'

export interface BibleEntry {
  /** File name without .md; also the id used in NovelScene.pov/present/sticky. */
  id: string
  name: string
  kind: BibleKind
  aliases: string[]
  triggers: string[]
  /** 0–100; higher stays when the budget is tight. */
  priority: number
  /** true = in every request. */
  always: boolean
  /** Markdown body after the front matter. */
  body: string
  /** Front-matter problems, shown on the entry; never thrown. */
  problems: string[]
}

// ---------------------------------------------------------------------------------------------
// Text files as the store hands them over (bytes ↔ text rules: design, data safety rule 2)
// ---------------------------------------------------------------------------------------------

export interface NovelFileMeta {
  bom: boolean
  eol: 'crlf' | 'lf'
}

/** A user-text file: scene, bible entry or style.md. `text` is LF-normalised, without BOM. */
export interface NovelTextFile {
  /** Path relative to the project folder with forward slashes, e.g. "manuscript/01-ferry-k3f9.md". */
  path: string
  text: string
  /** sha256 hex of the raw bytes on disk ('' when the file does not exist). */
  hash: string
  meta: NovelFileMeta
  /** Not valid UTF-8: shown read-only, never saved. */
  readOnly: boolean
  /** Unsaved-edit journal newer than the file, if any (data safety rule 4). */
  journal: { text: string; baseHash: string; savedAt: string } | null
}

export interface NovelProjectSummary {
  id: string
  title: string
  folder: string
  updatedAt: string
  words: number
}

/** Everything the Writing page needs to show a project (scene texts are loaded on demand). */
export interface NovelProjectState {
  folder: string
  project: NovelProject
  outline: NovelOutline
  bible: BibleEntry[]
  style: NovelTextFile
  /** Words per scene file (relative path → count), for the outline tree. */
  words: Record<string, number>
  /** Scene files referenced by the outline but missing on disk. */
  missing: string[]
  /** .md files in manuscript\ not referenced by the outline. */
  orphans: string[]
  /** project.json or outline.json could not be parsed: the project is read-only (data safety rule 8). */
  blocked: string | null
  /** Tracked state (state.json), stage 3. Statuses are computed by the renderer with novelState.ts. */
  state: NovelStateFile
  /** Dismissed check findings per scene id (.rulz/checks.json): finding keys. */
  dismissed: Record<string, string[]>
  /** .rulz/checks.json could not be read: dismissals are ignored and not written. */
  checksProblem: string | null
}

// ---------------------------------------------------------------------------------------------
// Stage 3: tracked state (state.json), checks, voice. Design: docs/writing-mode-design.md, "Stage 3 design".
// ---------------------------------------------------------------------------------------------

export type StateKind = 'location' | 'knows' | 'carries' | 'condition' | 'relationship' | 'other'

export interface StateFact {
  /** 'f' + random, made by the renderer. */
  id: string
  /** Bible entry id. */
  entity: string
  kind: StateKind
  /** Trimmed, one line, 1-300 characters. */
  value: string
  /** The other entry's id; required for 'relationship', null for every other kind. */
  about: string | null
  /** Scene id where the fact became true. */
  from: string
  /** Scene id where it stopped being true, or null. */
  until: string | null
}

/** A fact in state.json that did not validate; kept verbatim in the file, listed for removal. */
export interface StateInvalidFact {
  index: number
  reason: string
  /** sha256 of the raw JSON of the fact, so a removal hits the fact that was shown. */
  hash: string
  raw: string
}

export interface NovelStateFile {
  facts: StateFact[]
  invalid: StateInvalidFact[]
  /** Missing mid-session, 0 bytes, not JSON, format > 1 or no facts array: read-only, never written. */
  problem: string | null
  /** True when problem is a missing or 0-byte file, so 'reset' is allowed. */
  canReset: boolean
}

export type StateFactStatus = 'ok' | 'detached' | 'inverted'

export type StateOp =
  | { op: 'add'; fact: StateFact }
  | { op: 'end'; factId: string; until: string }
  | { op: 'edit'; factId: string; changes: Partial<Omit<StateFact, 'id'>> }
  | { op: 'remove'; factId: string }
  | { op: 'removeInvalid'; index: number; hash: string }
  | { op: 'reset' }

export interface NovelStatePatchResult {
  state: NovelProjectState
  /** Ops that were not applied, by their index in the request. */
  rejected: { index: number; reason: string }[]
}

export type FindingType = 'continuity' | 'character' | 'timeline' | 'pov' | 'tense' | 'other'

export interface CheckFinding {
  type: FindingType
  quote: string
  problem: string
  conflictsWith: string
  suggestion: string
}

export interface CheckReply {
  findings: CheckFinding[]
}

export type UpdateStateChange =
  | { op: 'add'; entity: string; kind: StateKind; value: string; about: string | null }
  | { op: 'end'; factId: string; reason: string }

export interface UpdateStateReply {
  changes: UpdateStateChange[]
}

/** State lines of one entity, before the current scene, for assembly. */
export interface AssemblyStateGroup extends Counted {
  /** Bible entry id. */
  entity: string
  name: string
}

// ---------------------------------------------------------------------------------------------
// Saving
// ---------------------------------------------------------------------------------------------

export interface NovelSaveRequest {
  folder: string
  path: string
  text: string
  /** Hash the editor's text was based on; '' for a new file. */
  baseHash: string
  /** "Keep mine" after a conflict: write even though the disk changed (the disk version is snapshotted first). */
  force?: boolean
  /** Take a history snapshot before writing (Accept, Restore, Keep mine). */
  snapshot?: boolean
}

export type NovelSaveResult =
  | { ok: true; hash: string }
  | { ok: false; reason: 'conflict'; diskHash: string; diskText: string }
  | { ok: false; reason: 'readOnly' | 'blocked' }
  | { ok: false; reason: 'error'; error: string }

export interface NovelHistoryItem {
  /** Compact stamp as used in the file name, e.g. "20261122T101500". */
  stamp: string
  /** Relative path of the snapshot inside the project. */
  snapshot: string
  bytes: number
}

export interface NovelTrashItem {
  id: string
  deletedAt: string
  /** What was deleted, e.g. 'Chapter "The ferry" (3 scenes)'. */
  label: string
  files: string[]
}

// ---------------------------------------------------------------------------------------------
// Outline and project changes: applied in order by the main process (data safety rule 7)
// ---------------------------------------------------------------------------------------------

export type SceneField = 'title' | 'beats' | 'summary' | 'summaryOf' | 'status' | 'pov' | 'present' | 'sticky' | 'targetWords'
export type ChapterField = 'title' | 'act' | 'summary' | 'summaryOf'

export type OutlineOp =
  | { op: 'setScene'; sceneId: string; field: SceneField; value: unknown }
  | { op: 'setChapter'; chapterId: string; field: ChapterField; value: unknown }
  /** Adds a chapter with one empty scene; `index` null = at the end. File names are made by main. */
  | { op: 'addChapter'; chapterId: string; sceneId: string; title: string; index: number | null }
  | { op: 'addScene'; chapterId: string; sceneId: string; title: string; index: number | null; beats?: string }
  | { op: 'moveChapter'; chapterId: string; index: number }
  | { op: 'moveScene'; sceneId: string; toChapterId: string; index: number }
  /** Adds an orphan file from manuscript\ as a scene. */
  | { op: 'adoptFile'; file: string; chapterId: string; sceneId: string; index: number | null }

export type ProjectPatch = Partial<Pick<NovelProject, 'title' | 'modelId' | 'sampling' | 'adapters' | 'budget' | 'povTense' | 'recording'>>

// ---------------------------------------------------------------------------------------------
// Request assembly (pure: src/shared/novelAssembly.ts)
// ---------------------------------------------------------------------------------------------

export type NovelAction =
  | 'continue'
  | 'rewrite'
  | 'expand'
  | 'condense'
  | 'dialogue'
  | 'draftScene'
  | 'summarizeScene'
  | 'summarizeChapter'
  | 'outlineChapters'
  | 'outlineScenes'
  | 'checkScene'
  | 'updateState'
  | 'interview'

export type RewriteVariant = 'tighter' | 'vivid' | 'tone' | 'show' | 'custom'

export type AssemblyPartKind = 'system' | 'task' | 'scene' | 'prose' | 'fixedEntries' | 'triggeredEntries' | 'storySoFar' | 'state'

/** A piece of text the assembler may include, with its real token count (counted by the renderer). */
export interface Counted {
  text: string
  tokens: number
}

export interface AssemblyEntry extends Counted {
  id: string
  name: string
  priority: number
  /** Why it is a candidate: "always", "pov", "present", "sticky", "trigger: witch", "via: Mara Venn". */
  reason: string
  fixed: boolean
  /** The entry cut to its first paragraph (plus its voice section for a voice target), used before dropping it. */
  short: Counted
  /** Stage 3: a voice target (character for Dialogue, or named in the selection); sorted first in its group. */
  voice?: boolean
}

export interface AssemblySummary extends Counted {
  /** "Chapter 2: The ferry" or "Scene: Night crossing". */
  label: string
  kind: 'chapter' | 'scene'
  /** Missing or stale summary replaced by beats. */
  fallback: 'none' | 'missing' | 'stale'
}

export interface AssemblyInput {
  action: NovelAction
  /** Model context length (per slot) and a measured template overhead in tokens. */
  contextLength: number
  templateOverhead: number
  budget: NovelBudget
  system: Counted
  task: Counted
  scene: Counted
  /** Candidate entries, fixed ones included. */
  entries: AssemblyEntry[]
  /** Oldest first. */
  summaries: AssemblySummary[]
  /** Prose before the cursor split into paragraph blocks, oldest first, each counted. */
  proseBlocks: Counted[]
  /** Prose after the cursor (insertion mid-text), counted; may be empty. */
  proseAfter: Counted
  /** Anchor step in tokens (about a quarter of the tail budget) for a stable prose start. */
  anchorStep: number
  /**
   * Stage 3: state before the current scene, one group per entity with 'ok' facts. Groups of fixed entities go in
   * the 'state' part; groups of triggered entities ride inside their entry. Optional: missing = no state.
   */
  stateGroups?: AssemblyStateGroup[]
}

export interface AssemblyPart {
  kind: AssemblyPartKind
  tokens: number
  cap: number | null
}

export interface AssemblyDropped {
  kind: AssemblyPartKind
  label: string
  /** 'dropped' or 'cut' (entry shortened to its first paragraph). */
  how: 'dropped' | 'cut'
  reason: string
}

export interface AssemblyResult {
  /** OpenAI-style messages ready for chat:start with prefitted: true. */
  messages: { role: 'system' | 'user' | 'assistant'; content: string }[]
  parts: AssemblyPart[]
  dropped: AssemblyDropped[]
  /** Included entry ids (for the sticky list and the Context panel). */
  includedEntries: string[]
  /** Index into proseBlocks of the first block included. */
  proseStart: number
  totalTokens: number
  available: number
  warnings: string[]
}

/** Structured reply schemas (JSON-schema output, validated with zod in novelPrompts.ts). */
export interface OutlineChaptersReply {
  chapters: { title: string; act: string; summary: string }[]
}
export interface OutlineScenesReply {
  scenes: { title: string; beats: string; pov: string | null; present: string[] }[]
}
export interface SummaryReply {
  summary: string
}

// ---------------------------------------------------------------------------------------------
// Stage 4: recording (.rulz/records/<YYYY-MM-DD>.jsonl) and training-data export.
// Design: docs/writing-mode-design.md, "Stage 4 design".
// ---------------------------------------------------------------------------------------------

export const RECORD_FORMAT = 1

/** 'length' = cut off at max tokens; 'stopped' = the user pressed Stop. */
export type RecordFinish = 'stop' | 'length' | 'stopped' | 'error'

export interface RecordMessage {
  role: 'system' | 'user' | 'assistant'
  content: string
}

export interface RecordAlt {
  index: number
  seed: number
  /** Prose: the new text only (PendingAlt.generated). Cards: the raw reply. */
  generated: string
  finish: RecordFinish
}

/** Written once when a run ends with at least one alternative. */
export interface RecordRequestLine {
  format: 1
  type: 'request'
  /** 16 lowercase hex characters, made by the renderer when the run starts. */
  id: string
  /** ISO time. */
  at: string
  /** App version. */
  app: string
  action: NovelAction
  variant?: string
  sceneId: string | null
  chapterId: string | null
  model: { id: string; name: string }
  /** The sampling values actually sent. */
  sampling: SamplingParams
  adapters: { name: string; scale: number }[]
  /** Exactly as sent. */
  messages: RecordMessage[]
  alts: RecordAlt[]
}

/** Prose: the alternatives the user displayed (the first active one included). Written with the outcome. */
export interface RecordViewedLine {
  format: 1
  type: 'viewed'
  id: string
  at: string
  alts: number[]
}

/** Prose: accepted | mixed | rejected | abandoned. Cards: accepted | edited | rejected | abandoned. */
export type RecordOutcomeKind = 'accepted' | 'mixed' | 'rejected' | 'abandoned' | 'edited'

export interface RecordRegion {
  /** Offsets of the generated text in the accepted document (from the apply geometry, not a diff). */
  start: number
  end: number
}

export interface RecordOutcomeLine {
  format: 1
  type: 'outcome'
  id: string
  at: string
  kind: RecordOutcomeKind
  /** Prose: the accepted (or, for mixed, active) alt index; null when rejected or abandoned. */
  alt?: number | null
  /** Prose accepted / mixed: where the generated text sits in the accepted document. */
  region?: RecordRegion | null
  /** Prose accepted / mixed: the base text the region replaced (the selection for rewrite, '' for continue). */
  replaced?: string
  /** Cards, kind 'edited': the edited proposal. */
  text?: string
}

export type RecordFinalReason = 'idle' | 'left' | 'action' | 'close' | 'reloaded' | 'outside'

/** Prose after accepted / mixed: the region after the user's later edits. Repeats allowed; the last one wins. */
export interface RecordFinalLine {
  format: 1
  type: 'final'
  id: string
  at: string
  /** null when the region could not be located ('outside') or the text was reloaded ('reloaded'). */
  text: string | null
  reason: RecordFinalReason
}

export type RecordLine = RecordRequestLine | RecordViewedLine | RecordOutcomeLine | RecordFinalLine

export interface RecordsInfo {
  files: number
  bytes: number
}

export interface RecordsExportOptions {
  /** Pairs of (final text, accepted text). Off by default. */
  editPairs: boolean
}

export interface RecordsExportResult {
  /** The new subfolder that was written. */
  outDir: string
  train: number
  pairs: number
  /** Reason → count. */
  dropped: Record<string, number>
}

// ---------------------------------------------------------------------------------------------
// IPC (merged into IpcInvoke / IpcEvents in ipc.ts). `folder` is the project's absolute folder.
// ---------------------------------------------------------------------------------------------

export interface NovelIpcInvoke {
  /** Projects in the projects root (setting writingRoot; '' = <Documents>\RulZ Novels). */
  'novel:listProjects': () => NovelProjectSummary[]
  'novel:root': () => string
  'novel:createProject': (title: string) => NovelProjectState
  /** Opens (and starts watching) a project folder, inside the root or anywhere. */
  'novel:openProject': (folder: string) => NovelProjectState
  'novel:closeProject': (folder: string) => void
  'novel:readFile': (folder: string, path: string) => NovelTextFile
  'novel:saveFile': (req: NovelSaveRequest) => NovelSaveResult
  /** Unsaved-edit journal (debounced by the renderer); a successful save removes it. */
  'novel:journal': (folder: string, path: string, text: string, baseHash: string) => void
  'novel:discardJournal': (folder: string, path: string) => void
  /** Applies ops in order and writes outline.json; creates empty scene files for new scenes. */
  'novel:outlinePatch': (folder: string, ops: OutlineOp[]) => NovelProjectState
  'novel:projectPatch': (folder: string, patch: ProjectPatch) => NovelProject
  /** Creates bible/<id>.md with front matter; returns the new entry. */
  'novel:createEntry': (folder: string, name: string, kind: BibleKind) => BibleEntry
  'novel:deleteScene': (folder: string, sceneId: string) => NovelProjectState
  'novel:deleteChapter': (folder: string, chapterId: string) => NovelProjectState
  'novel:deleteEntry': (folder: string, entryId: string) => NovelProjectState
  'novel:history': (folder: string, path: string) => NovelHistoryItem[]
  'novel:readSnapshot': (folder: string, snapshot: string) => string
  'novel:trash': (folder: string) => NovelTrashItem[]
  'novel:restoreTrash': (folder: string, id: string) => NovelProjectState
  'novel:emptyTrash': (folder: string) => void
  /** Moves the whole project folder to the Recycle Bin. */
  'novel:deleteProject': (folder: string) => void
  /** Renderer's answer to the 'novel:flush' event: every dirty buffer is saved or journalled. */
  'novel:flushDone': () => void
  /** Stage 3: applies the valid ops to state.json, returns the rejected ones with reasons. */
  'novel:statePatch': (folder: string, ops: StateOp[]) => NovelStatePatchResult
  /** Stage 3: (un)dismisses a check finding for a scene in .rulz/checks.json. */
  'novel:dismissFinding': (folder: string, sceneId: string, key: string, dismissed: boolean) => NovelProjectState
  /** Stage 4: appends one record line; false when not written (not open, blocked, recording off, I/O error). */
  'novel:record': (folder: string, line: RecordLine) => boolean
  'novel:recordsInfo': (folder: string) => RecordsInfo
  /** Writes train.jsonl, pairs.jsonl and manifest.json into a new subfolder of outDir. */
  'novel:exportRecords': (folder: string, outDir: string, opts: RecordsExportOptions) => RecordsExportResult
  /** Moves .rulz/records to the Recycle Bin. */
  'novel:deleteRecords': (folder: string) => void
  /** Token count of a text on the loaded model (cached per instance); null if the model can't count. */
  'chat:countText': (modelId: string, text: string) => number | null
}

export type NovelChange =
  | { kind: 'text'; folder: string; path: string; hash: string }
  | { kind: 'deleted'; folder: string; path: string }
  | { kind: 'state'; folder: string; state: NovelProjectState }

export interface NovelIpcEvents {
  /** A file changed on disk outside the app, or the project state changed (outline/project/bible). */
  'novel:changed': NovelChange
  /** The window is closing: save or journal every dirty buffer, then invoke 'novel:flushDone'. */
  'novel:flush': undefined
}
