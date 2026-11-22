// Writing mode, renderer side: builds the request for an AI action (candidates, real token counts,
// the shared assembler), checks that it fits, runs it through chat:start and shapes the result.
// The building part is pure (token counts come in through `CountFn`), so tests run it without the app.
import type {
  AssemblyEntry,
  AssemblyInput,
  AssemblyResult,
  AssemblySummary,
  BibleEntry,
  Counted,
  NovelAction,
  NovelChapter,
  NovelOutline,
  NovelProject,
  NovelScene,
  RewriteVariant
} from '@shared/novel'
import type { ChatRequest, LoadAdapter, SamplingParams, StreamEvent, WireMessage } from '@shared/types'
import { assembleRequest } from '@shared/novelAssembly'
import { findScene } from '@shared/novelOutline'
import { cleanProse, entryListText, entryShortText, entryText, isProseAction, resolveEntryRef, summaryText, sceneText, taskText, systemText } from '@shared/novelPrompts'
import { matchEntries } from '@shared/novelTriggers'
import { sceneHashKey, splitParagraphs } from '@shared/novelText'
import { errorMessage, invoke, on } from './api'
import { uid } from './format'

// ---------------------------------------------------------------------------------------------
// Token counts
// ---------------------------------------------------------------------------------------------

export type CountFn = (text: string) => Promise<number>

/** A fast 53-bit string hash (cyrb53), for cache keys. */
export function hashText(text: string): string {
  let h1 = 0xdeadbeef
  let h2 = 0x41c6ce57
  for (let i = 0; i < text.length; i++) {
    const c = text.charCodeAt(i)
    h1 = Math.imul(h1 ^ c, 2654435761)
    h2 = Math.imul(h2 ^ c, 1597334677)
  }
  h1 = Math.imul(h1 ^ (h1 >>> 16), 2246822507) ^ Math.imul(h2 ^ (h2 >>> 13), 3266489909)
  h2 = Math.imul(h2 ^ (h2 >>> 16), 2246822507) ^ Math.imul(h1 ^ (h1 >>> 13), 3266489909)
  return (4294967296 * (2097151 & h2) + (h1 >>> 0)).toString(36)
}

/** Used when the model cannot count: slightly high on purpose, so the request errs on the small side. */
export function estimateTokens(text: string): number {
  return Math.ceil(text.length / 3)
}

const COUNT_CACHE_MAX = 20000
const countCache = new Map<string, number>()

export function clearCountCache(): void {
  countCache.clear()
}

/** Counts through `raw` and remembers each answer by model and text. A failed count falls back to an estimate. */
export function cachedCounter(modelId: string, raw: (modelId: string, text: string) => Promise<number | null>, onEstimate?: () => void): CountFn {
  return async (text) => {
    if (!text) return 0
    const key = `${modelId}\u0000${text.length}:${hashText(text)}`
    const hit = countCache.get(key)
    if (hit !== undefined) return hit
    let n: number | null = null
    try {
      n = await raw(modelId, text)
    } catch {
      n = null
    }
    if (n === null) {
      onEstimate?.()
      return estimateTokens(text)
    }
    if (countCache.size >= COUNT_CACHE_MAX) countCache.delete(countCache.keys().next().value as string)
    countCache.set(key, n)
    return n
  }
}

export function ipcCounter(modelId: string, onEstimate?: () => void): CountFn {
  return cachedCounter(modelId, (m, t) => invoke('chat:countText', m, t), onEstimate)
}

const overheadCache = new Map<string, number>()

/** The chat template's own tokens around a system + user pair: whole-request count minus the parts' counts. Cached per instance. */
export async function templateOverhead(instanceId: string, modelId: string, count: CountFn): Promise<number> {
  const known = overheadCache.get(instanceId)
  if (known !== undefined) return known
  const sys = 'You are a writer.'
  const usr = 'Write one sentence.'
  let total: number | null = null
  try {
    total = await invoke('chat:countTokens', modelId, [{ role: 'system', content: sys }, { role: 'user', content: usr }], null)
  } catch {
    total = null
  }
  // No count: assume a typical template, and don't cache so the next request tries again.
  if (total === null) return 48
  const parts = (await count(sys)) + (await count(usr))
  const overhead = Math.max(0, total - parts)
  overheadCache.set(instanceId, overhead)
  return overhead
}

async function countAll(texts: string[], count: CountFn): Promise<number[]> {
  const out: number[] = []
  for (let i = 0; i < texts.length; i += 16) out.push(...(await Promise.all(texts.slice(i, i + 16).map(count))))
  return out
}

// ---------------------------------------------------------------------------------------------
// Building a request
// ---------------------------------------------------------------------------------------------

export const scenePath = (scene: Pick<NovelScene, 'file'>): string => `manuscript/${scene.file}`

export interface RunContext {
  action: NovelAction
  project: NovelProject
  /** style.md text. */
  style: string
  outline: NovelOutline
  bible: BibleEntry[]
  /** Scene (scene actions) or null for project-level actions. */
  sceneId: string | null
  /** Chapter for chapter actions (summarizeChapter, outlineScenes). */
  chapterId?: string
  /** Current byte hash of each scene file (relative path → hash), for the stale-summary check. */
  sceneHashes: Record<string, string>
  /** The scene's text in the editor. */
  text: string
  cursor: number
  selection: { from: number; to: number } | null
  note: string
  variant?: RewriteVariant
  /** Wanted chapters / scenes for the outline actions. */
  count?: number
  /** Text of the previous scene, for Draft scene on an empty scene. */
  previousText?: string
  contextLength: number
  templateOverhead: number
}

export interface CandidateInfo {
  name: string
  reason: string
  fixed: boolean
}

export interface BuiltRequest {
  input: AssemblyInput
  result: AssemblyResult
  /** Every candidate entry by id, with why it was one. */
  candidates: Record<string, CandidateInfo>
  /** How many summaries in "story so far" were replaced by beats. */
  fallbacks: { missing: number; stale: number }
  /** Triggered entries this request included that the scene doesn't hold yet (for the sticky list). */
  newSticky: string[]
  /** The model could not count some text; counts are estimates. */
  estimated: boolean
}

/** The text before and after the edit point: the selection's edges, or the cursor. */
export function splitAt(ctx: Pick<RunContext, 'text' | 'cursor' | 'selection'>): { before: string; selected: string; after: string } {
  const from = ctx.selection ? ctx.selection.from : ctx.cursor
  const to = ctx.selection ? ctx.selection.to : ctx.cursor
  return { before: ctx.text.slice(0, from), selected: ctx.text.slice(from, to), after: ctx.text.slice(to) }
}

/**
 * Words asked for in the task. Draft scene uses the scene's target; Continue is sized to the reply budget
 * (about 0.6 words per reply token, to the nearest 50), not to the whole scene's target.
 */
export function targetWordsFor(action: NovelAction, sceneTarget: number | null, replyTokens: number): number | null {
  if (action === 'draftScene') return sceneTarget
  if (action === 'continue') return Math.max(50, Math.round((replyTokens * 0.6) / 50) * 50)
  return null
}

/** What the trigger scan looks at: the beats, the last ~2000 characters before the edit point, the selection and the note. */
export function scanTextFor(ctx: Pick<RunContext, 'text' | 'cursor' | 'selection' | 'note'>, scene: NovelScene | null): string {
  const { before, selected } = splitAt(ctx)
  return [scene?.beats ?? '', before.slice(-2000), selected, ctx.note].filter((s) => s.trim()).join('\n')
}

/** A short piece of the text after the edit point, cut at a paragraph: context for inserting mid-text. */
export function afterPiece(after: string): string {
  const blocks = splitParagraphs(after.trimStart())
  let out = ''
  for (const b of blocks.slice(0, 3)) {
    if (out && out.length + b.length > 1500) break
    out += (out ? '\n\n' : '') + b
  }
  return out.slice(0, 2000)
}

interface RawSummary {
  label: string
  kind: 'chapter' | 'scene'
  content: string
  fallback: 'none' | 'missing' | 'stale'
}

const planLine = (s: NovelScene): string => {
  const body = s.summary.trim() || s.beats.trim()
  return body ? `${s.title}: ${body}` : ''
}

/** The "story so far" items, oldest first: earlier chapters, then (when `sceneIndex` is given) earlier scenes of the chapter. */
export function collectSummaries(
  outline: NovelOutline,
  hashes: Record<string, string>,
  chapterIndex: number,
  sceneIndex: number | null
): RawSummary[] {
  const out: RawSummary[] = []
  outline.chapters.slice(0, chapterIndex).forEach((c, i) => {
    const key = sceneHashKey(c.scenes.map((s) => hashes[scenePath(s)] ?? ''))
    const has = c.summary.trim() !== ''
    // A chapter whose scenes' hashes are not all known is not called stale on a guess.
    const known = c.scenes.every((s) => hashes[scenePath(s)] !== undefined)
    const stale = has && known && c.summaryOf !== key
    const label = `Chapter ${i + 1}: ${c.title}`
    if (has && !stale) return void out.push({ label, kind: 'chapter', content: c.summary.trim(), fallback: 'none' })
    const plan = c.scenes.map(planLine).filter(Boolean).join('\n')
    if (plan) out.push({ label: `${label} (plan)`, kind: 'chapter', content: plan, fallback: has ? 'stale' : 'missing' })
    else if (has) out.push({ label, kind: 'chapter', content: c.summary.trim(), fallback: 'none' })
  })
  if (sceneIndex !== null) {
    const chapter = outline.chapters[chapterIndex]
    chapter?.scenes.slice(0, sceneIndex).forEach((s) => {
      const hash = hashes[scenePath(s)]
      const has = s.summary.trim() !== ''
      const stale = has && hash !== undefined && s.summaryOf !== hash
      const label = `Scene: ${s.title}`
      if (has && !stale) return void out.push({ label, kind: 'scene', content: s.summary.trim(), fallback: 'none' })
      if (s.beats.trim()) out.push({ label: `${label} (plan)`, kind: 'scene', content: s.beats.trim(), fallback: has ? 'stale' : 'missing' })
      else if (has) out.push({ label, kind: 'scene', content: s.summary.trim(), fallback: 'none' })
    })
  }
  return out
}

/** Scenes whose text exists but whose summary is missing or out of date, for the "Summarize all" note. */
export function summaryHealth(
  outline: NovelOutline,
  hashes: Record<string, string>,
  words: Record<string, number>
): { missing: string[]; stale: string[] } {
  const missing: string[] = []
  const stale: string[] = []
  for (const c of outline.chapters) {
    for (const s of c.scenes) {
      const p = scenePath(s)
      if ((words[p] ?? 0) === 0) continue
      if (!s.summary.trim()) missing.push(s.id)
      else if (hashes[p] !== undefined && s.summaryOf !== hashes[p]) stale.push(s.id)
    }
  }
  return { missing, stale }
}

/** Last paragraphs of a text, newest last, up to about `tokens` by a rough count: draft scene's lead-in from the previous scene. */
export function tailBlocks(text: string, maxChars: number): string[] {
  const blocks = splitParagraphs(text)
  const out: string[] = []
  let chars = 0
  for (let i = blocks.length - 1; i >= 0; i--) {
    if (out.length && chars + blocks[i].length > maxChars) break
    out.unshift(blocks[i])
    chars += blocks[i].length
  }
  return out
}

function chapterOverview(outline: NovelOutline): string {
  if (!outline.chapters.length) return 'No chapters yet.'
  return ['Existing chapters:', ...outline.chapters.map((c, i) => `${i + 1}. ${c.title}${c.act ? ` (${c.act})` : ''}${c.summary.trim() ? `: ${c.summary.trim()}` : ''}`)].join('\n')
}

function entryIdList(bible: BibleEntry[]): string {
  const list = bible.filter((e) => e.kind === 'character' || e.kind === 'place')
  if (!list.length) return ''
  return ['Bible entries (use their ids for pov and present):', entryListText(list)].join('\n')
}

/**
 * Maps the pov and present names of a scene plan to bible entry ids (id, then name or alias). References
 * that match nothing are dropped and returned in `unresolved`, so the user can see what was left out.
 */
export function resolveSceneRefs(
  bible: BibleEntry[],
  scenes: { pov: string | null; present: string[] }[]
): { scenes: { pov: string | null; present: string[] }[]; unresolved: string[] } {
  const unresolved: string[] = []
  const miss = (ref: string): void => {
    if (ref.trim() && !unresolved.includes(ref.trim())) unresolved.push(ref.trim())
  }
  const out = scenes.map((sc) => {
    let pov: string | null = null
    if (sc.pov) {
      pov = resolveEntryRef(bible, sc.pov)
      if (!pov) miss(sc.pov)
    }
    const present: string[] = []
    for (const ref of sc.present) {
      const id = resolveEntryRef(bible, ref)
      if (!id) miss(ref)
      else if (!present.includes(id)) present.push(id)
    }
    return { pov, present }
  })
  return { scenes: out, unresolved }
}

function chapterText(chapter: NovelChapter, outline: NovelOutline, bible: BibleEntry[]): string {
  const idx = outline.chapters.indexOf(chapter)
  return [
    `Chapter ${idx + 1}: ${chapter.title}`,
    chapter.act && `Act: ${chapter.act}`,
    chapter.summary.trim() && `Summary so far: ${chapter.summary.trim()}`,
    chapter.scenes.length ? ['Scenes:', ...chapter.scenes.map((s) => `- ${planLine(s) || s.title}`)].join('\n') : '',
    entryIdList(bible)
  ]
    .filter(Boolean)
    .join('\n')
}

function summarySource(chapter: NovelChapter): string {
  return [`Chapter: ${chapter.title}`, ...chapter.scenes.map((s) => `Scene: ${s.title}\n${s.summary.trim() || s.beats.trim() || '(nothing written)'}`)].join('\n\n')
}

/**
 * Builds and assembles the request for an action. Pure apart from `count`. Throws an Error with a
 * plain message when the target scene or chapter does not exist.
 */
export async function buildRequest(ctx: RunContext, count: CountFn, estimated: { value: boolean } = { value: false }): Promise<BuiltRequest> {
  const { action, outline, bible, project } = ctx
  const entriesById = new Map(bible.map((e) => [e.id, e]))

  let scene: NovelScene | null = null
  let chapter: NovelChapter | null = null
  let chapterIndex = outline.chapters.length
  let sceneIndex: number | null = null
  if (ctx.sceneId) {
    const found = findScene(outline, ctx.sceneId)
    if (!found) throw new Error('The scene is not in the outline any more.')
    scene = found.scene
    chapter = found.chapter
    chapterIndex = found.chapterIndex
    sceneIndex = found.sceneIndex
  } else if (ctx.chapterId) {
    const i = outline.chapters.findIndex((c) => c.id === ctx.chapterId)
    if (i < 0) throw new Error('The chapter is not in the outline any more.')
    chapter = outline.chapters[i]
    chapterIndex = i
  }

  const { before, selected, after } = splitAt(ctx)

  // --- text parts
  const systemStr = systemText(action, ctx.style, project.povTense)
  const taskStr = taskText(action, {
    variant: ctx.variant,
    note: ctx.note,
    selection: selected || undefined,
    targetWords: targetWordsFor(action, scene?.targetWords ?? null, project.budget.replyTokens),
    count: ctx.count
  })
  let sceneStr = ''
  if (action === 'outlineChapters') sceneStr = [chapterOverview(outline), entryIdList(bible)].filter(Boolean).join('\n\n')
  else if (action === 'outlineScenes' && chapter) sceneStr = chapterText(chapter, outline, bible)
  else if (action === 'summarizeChapter' && chapter) sceneStr = summarySource(chapter)
  else if (scene && chapter) sceneStr = sceneText(scene, chapter, entriesById)

  // --- summaries
  let raw: RawSummary[] = []
  if (action === 'outlineChapters') raw = collectSummaries(outline, ctx.sceneHashes, outline.chapters.length, null)
  else if (action === 'outlineScenes' || action === 'summarizeChapter') raw = collectSummaries(outline, ctx.sceneHashes, chapterIndex, null)
  else if (scene) raw = collectSummaries(outline, ctx.sceneHashes, chapterIndex, sceneIndex)

  // --- entries
  const scan =
    action === 'outlineChapters'
      ? [outline.chapters.map((c) => `${c.title} ${c.summary}`).join('\n'), ctx.note].join('\n')
      : scene
        ? scanTextFor(ctx, scene)
        : [chapter?.title, chapter?.summary, ctx.note].filter(Boolean).join('\n')
  const candidates = matchEntries(bible, scan, scene)

  // --- prose
  let proseTexts: string[] = []
  let afterStr = ''
  if (action === 'summarizeScene') proseTexts = splitParagraphs(ctx.text)
  else if (isProseAction(action)) {
    if (action === 'draftScene' && !before.trim()) proseTexts = tailBlocks(ctx.previousText ?? '', 2400)
    else proseTexts = splitParagraphs(before)
    if (action !== 'draftScene') afterStr = afterPiece(after)
  }

  // --- count everything that is not yet counted
  const entryTexts = candidates.map((c) => ({ full: entryText(c.entry), short: entryShortText(c.entry) }))
  const summaryTexts = raw.map((r) => summaryText(r.label, r.content))
  const counts = await countAll(
    [systemStr, taskStr, sceneStr, afterStr, ...entryTexts.flatMap((e) => [e.full, e.short]), ...summaryTexts, ...proseTexts],
    (t) => count(t)
  )
  let at = 0
  const take = (): number => counts[at++]
  const sys: Counted = { text: systemStr, tokens: take() }
  const task: Counted = { text: taskStr, tokens: take() }
  const sceneC: Counted = { text: sceneStr, tokens: take() }
  const proseAfter: Counted = { text: afterStr, tokens: take() }
  const entries: AssemblyEntry[] = candidates.map((c, i) => ({
    id: c.entry.id,
    name: c.entry.name,
    priority: c.priority,
    reason: c.reason,
    fixed: c.fixed,
    text: entryTexts[i].full,
    tokens: take(),
    short: { text: entryTexts[i].short, tokens: take() }
  }))
  const summaries: AssemblySummary[] = raw.map((r, i) => ({ label: r.label, kind: r.kind, fallback: r.fallback, text: summaryTexts[i], tokens: take() }))
  const proseBlocks: Counted[] = proseTexts.map((t) => ({ text: t, tokens: take() }))

  const reply = project.budget.replyTokens
  const available0 = Math.max(512, ctx.contextLength - reply - Math.max(256, Math.ceil(ctx.contextLength * 0.03)))
  const input: AssemblyInput = {
    action,
    contextLength: ctx.contextLength,
    templateOverhead: ctx.templateOverhead,
    budget: project.budget,
    system: sys,
    task,
    scene: sceneC,
    entries,
    summaries,
    proseBlocks,
    proseAfter,
    // Depends on the model and budget only, so the prose start stays put between presses of Continue.
    anchorStep: Math.max(128, Math.round((available0 * project.budget.proseShare) / 4))
  }
  const result = assembleRequest(input)

  const info: Record<string, CandidateInfo> = {}
  for (const c of candidates) info[c.entry.id] = { name: c.entry.name, reason: c.reason, fixed: c.fixed }
  const held = new Set(scene?.sticky ?? [])
  const newSticky = scene
    ? candidates.filter((c) => !c.fixed && result.includedEntries.includes(c.entry.id) && !held.has(c.entry.id)).map((c) => c.entry.id)
    : []
  return {
    input,
    result,
    candidates: info,
    fallbacks: {
      missing: summaries.filter((s) => s.fallback === 'missing').length,
      stale: summaries.filter((s) => s.fallback === 'stale').length
    },
    newSticky,
    estimated: estimated.value
  }
}

/** Whether the assembled request is over the budget: content tokens against what is available. */
export function doesNotFit(built: Pick<BuiltRequest, 'result' | 'input'>): boolean {
  return built.result.totalTokens - built.input.templateOverhead > built.result.available
}

export interface FitOutcome {
  built: BuiltRequest
  /** Tokens of the whole request as the model's template counts it; null if it could not count. */
  actual: number | null
  /** The request still does not fit after the one retry. */
  tooBig: boolean
  retried: boolean
}

/**
 * Final check with the real chat template: count the whole request; if it is over what the context
 * allows (drift between summed parts and the template), assemble once more with the drift added to
 * the overhead, which shortens the prose tail by the amount needed.
 */
export async function fitRequest(
  ctx: RunContext,
  count: CountFn,
  countWhole: (messages: AssemblyResult['messages']) => Promise<number | null>,
  /** Set to true by the counter when it had to estimate; shared with the caller's `count`. */
  estimated: { value: boolean } = { value: false }
): Promise<FitOutcome> {
  const counter: CountFn = async (t) => count(t)
  let built = await buildRequest(ctx, counter, estimated)
  if (doesNotFit(built)) return { built, actual: null, tooBig: true, retried: false }
  const limit = ctx.contextLength - built.input.budget.replyTokens
  let actual = await countWhole(built.result.messages)
  if (actual === null || actual <= limit) return { built, actual, tooBig: false, retried: false }
  const drift = actual - built.result.totalTokens + Math.max(32, Math.ceil(ctx.contextLength * 0.01))
  built = await buildRequest({ ...ctx, templateOverhead: ctx.templateOverhead + Math.max(1, drift) }, counter, estimated)
  actual = doesNotFit(built) ? null : await countWhole(built.result.messages)
  return { built, actual, tooBig: doesNotFit(built) || (actual !== null && actual > limit), retried: true }
}

// ---------------------------------------------------------------------------------------------
// What an AI result does to the text
// ---------------------------------------------------------------------------------------------

/**
 * The whole document with the generated text put in: replaces the selection for rewrite, expand and
 * condense; goes at the cursor for continue and dialogue (a blank line apart from its neighbours);
 * is the whole text for a draft of an empty scene.
 */
export function applyResult(action: NovelAction, text: string, cursor: number, selection: { from: number; to: number } | null, generated: string): string {
  const gen = generated.trim()
  if (action === 'draftScene') return gen
  if ((action === 'rewrite' || action === 'expand' || action === 'condense') && selection) {
    return text.slice(0, selection.from) + gen + text.slice(selection.to)
  }
  const before = text.slice(0, cursor)
  const after = text.slice(cursor)
  const lead = before.length === 0 || before.endsWith('\n\n') ? '' : before.endsWith('\n') ? '\n' : '\n\n'
  const tail = after.length === 0 ? '' : after.startsWith('\n\n') ? '' : after.startsWith('\n') ? '\n' : '\n\n'
  return before + lead + gen + tail + after
}

export interface PendingAlt {
  /** The whole document if this alternative is accepted. */
  text: string
  /** What the model wrote, after cleaning. */
  generated: string
  seed: number
}

export interface PendingChange {
  id: string
  path: string
  sceneId: string
  action: NovelAction
  label: string
  /** The buffer when the request was made; the baseline of the diff. */
  baseText: string
  /** Buffer hash and version the request was made against (data safety rule 3). */
  baseHash: string
  version: number
  alts: PendingAlt[]
  active: number
  /** Triggered entries to add to the scene's sticky list when this is accepted. */
  newSticky: string[]
  /** Request that made it, for Regenerate. */
  again: { action: NovelAction; variant?: RewriteVariant; note: string; selection: { from: number; to: number } | null; cursor: number; count: number }
}

/** True when the buffer moved since the pending change was made: Accept must be refused. */
export function pendingStale(p: Pick<PendingChange, 'baseHash' | 'version'>, b: { baseHash: string; version: number }): boolean {
  return p.baseHash !== b.baseHash || p.version !== b.version
}

export const STALE_PENDING = 'The text changed since this was written.'

// ---------------------------------------------------------------------------------------------
// Running a request
// ---------------------------------------------------------------------------------------------

export interface GenResult {
  text: string
  error: string | null
  stopped: boolean
}

export interface GenHandle {
  requestId: string
  done: Promise<GenResult>
  abort: () => void
}

/** How long an aborted request may take to report its end before it is treated as stopped. */
const ABORT_GRACE_MS = 5000
const streams = new Map<string, (e: StreamEvent) => void>()
let subscribed = false

function ensureSubscribed(): void {
  if (subscribed) return
  subscribed = true
  on('chat:event', (e) => streams.get(e.requestId)?.(e))
}

export interface GenInput {
  modelId: string
  messages: AssemblyResult['messages']
  sampling: SamplingParams
  adapters: LoadAdapter[] | null
  action: NovelAction
  seed: number
  maxTokens: number
  /** JSON schema for structured actions. */
  schema?: object
  onDelta?: (text: string) => void
  onProgress?: (fraction: number | null) => void
}

/** The sampling a request goes out with: project settings, a given seed and reply size, and the schema for structured actions. */
export function samplingFor(base: SamplingParams, input: Pick<GenInput, 'seed' | 'maxTokens' | 'schema' | 'action'>): SamplingParams {
  const structured = !isProseAction(input.action)
  return {
    ...base,
    seed: input.seed,
    maxTokens: input.maxTokens,
    jsonSchema: input.schema ? JSON.stringify(input.schema) : '',
    // Structured actions may think; prose never does (it would eat the reply budget).
    thinking: structured ? (base.thinking === 'off' ? 'auto' : base.thinking) : 'off'
  }
}

export function startGeneration(input: GenInput): GenHandle {
  ensureSubscribed()
  const requestId = uid()
  let finish: (r: GenResult) => void = () => undefined
  const done = new Promise<GenResult>((resolve) => (finish = resolve))
  let text = ''
  let ended = false
  const end = (r: GenResult): void => {
    if (ended) return
    ended = true
    streams.delete(requestId)
    finish(r)
  }
  streams.set(requestId, (e) => {
    if (e.type === 'progress') input.onProgress?.(e.total ? e.processed / e.total : null)
    else if (e.type === 'delta') {
      if (e.content) {
        text += e.content
        input.onDelta?.(text)
      }
    } else if (e.type === 'error') end({ text, error: e.error, stopped: false })
    else if (e.type === 'done') end({ text, error: null, stopped: e.stats.stopReason === 'stopped' })
  })
  const req: ChatRequest = {
    requestId,
    modelId: input.modelId,
    messages: input.messages as WireMessage[],
    sampling: samplingFor(input.sampling, input),
    adapters: input.adapters,
    overflow: 'stop',
    disableThinking: isProseAction(input.action),
    prefitted: true
  }
  invoke('chat:start', req).catch((e) => end({ text, error: errorMessage(e), stopped: false }))
  return {
    requestId,
    done,
    abort: () => {
      void invoke('chat:abort', requestId).catch(() => undefined)
      // The server normally ends the stream with a 'stopped' event; if none comes, don't wait for ever.
      setTimeout(() => end({ text, error: null, stopped: true }), ABORT_GRACE_MS)
    }
  }
}

/** Seeds for alternatives: the project's seed plus the index, or random ones that differ from each other. */
export function seedsFor(base: number | null, n: number, random: () => number = Math.random): number[] {
  const first = base ?? Math.floor(random() * 2_000_000_000)
  return Array.from({ length: n }, (_, i) => first + i)
}

/** Cleans prose output; structured output is left as the model wrote it. */
export function finishText(action: NovelAction, raw: string): string {
  return isProseAction(action) ? cleanProse(raw) : raw.trim()
}

export const ACTION_LABELS: Record<NovelAction, string> = {
  continue: 'Continue',
  rewrite: 'Rewrite',
  expand: 'Expand',
  condense: 'Condense',
  dialogue: 'Dialogue',
  draftScene: 'Draft scene',
  summarizeScene: 'Summarize scene',
  summarizeChapter: 'Summarize chapter',
  outlineChapters: 'Plan chapters',
  outlineScenes: 'Plan scenes'
}

export const VARIANT_LABELS: Record<RewriteVariant, string> = {
  tighter: 'Tighter',
  vivid: 'More vivid',
  tone: 'Different tone…',
  show: "Show, don't tell",
  custom: 'Custom…'
}
