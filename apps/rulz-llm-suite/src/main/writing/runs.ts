// Writing mode's AI actions as jobs in main, for the phone app (docs/phone-access-design.md, "Writing runs").
// The desktop renderer runs the same actions in src/renderer/src/stores/novel.ts; this service reads the project
// from the store instead of an editor buffer, builds and fits the request with the same shared helpers
// (src/shared/novelRun.ts), runs it on the chat service and shapes a WritingResult. Everything it touches is
// injected (store, chat, models, clock, emit), so tests run it with fakes and no model.
import type {
  BibleEntry,
  CheckFinding,
  NovelProjectState,
  NovelStatePatchResult,
  NovelTextFile,
  StateOp,
  UpdateStateChange
} from '@shared/novel'
import {
  CHECK_SCHEMA,
  OUTLINE_CHAPTERS_SCHEMA,
  OUTLINE_SCENES_SCHEMA,
  SUMMARY_SCHEMA,
  UPDATE_STATE_SCHEMA,
  isProseAction,
  parseCheckReplyResult,
  parseOutlineChapters,
  parseOutlineScenes,
  parseSummary,
  parseUpdateStateReplyResult
} from '@shared/novelPrompts'
import { modePromptExtra, type AppMode, type ModelModeDefaults } from '@shared/modes'
import { findChapter, findScene } from '@shared/novelOutline'
import { stateBefore } from '@shared/novelState'
import { sceneHashKey } from '@shared/novelText'
import {
  ACTION_LABELS,
  applyResult,
  buildInterviewSystem,
  dedupeFindings,
  finishText,
  fitInterview,
  fitRequest,
  hashText,
  makeIpcCounter,
  resolveSceneRefs,
  resolveStateChanges,
  scenePath,
  seedsFor,
  startGeneration,
  stateOpsFor,
  templateOverheadWith,
  wholeSceneProblem,
  type GenBackend,
  type GenInput,
  type GenResult,
  type RunContext,
  type StateCardItem
} from '@shared/novelRun'
import type { ChatRequest, StreamEvent, WireMessage } from '@shared/types'
import type { PhoneModel, WritingAlt, WritingJob, WritingJobStatus, WritingResult, WritingRunRequest, WritingStateItem } from '@shared/phone'
import type { ModelGateLike } from '../modelGate'

// ---------------------------------------------------------------------------------------------
// What the service needs from its surroundings
// ---------------------------------------------------------------------------------------------

/** The slice of NovelStore a run uses. `projectState` is the store's state for an open project (see adapters.ts). */
export interface WritingStore {
  /** Keeps the project open for `owner` ('job:<id>') so it stays open while a job runs, even if the phone leaves. */
  hold(folder: string, owner: string): Promise<void>
  release(folder: string, owner: string): Promise<void>
  projectState(folder: string): Promise<NovelProjectState>
  readFile(folder: string, rel: string): Promise<NovelTextFile>
  statePatch(folder: string, ops: StateOp[]): Promise<NovelStatePatchResult>
}

/** Main's chat service, as a run uses it. `subscribe` taps the stream events of one request. */
export interface WritingChat {
  countText(modelId: string, text: string): Promise<number | null>
  countTokens(modelId: string, messages: WireMessage[]): Promise<number | null>
  start(req: ChatRequest): Promise<void>
  abort(requestId: string): void
  subscribe(requestId: string, handler: (e: StreamEvent) => void): () => void
}

export interface WritingModel {
  modelId: string
  instanceId: string
  contextLength: number | null
  /** The rental it runs on; absent = this PC. */
  rental?: { label: string; reconnecting: boolean }
}

export interface WritingModels {
  /** The model library. */
  list(): { id: string; name: string; remote?: boolean }[]
  /** The ready instance of a model; with null, any ready instance (the desktop's "active model"). */
  ready(modelId: string | null): WritingModel | undefined
  /** Loads the model as the desktop does (instances:ensure) and resolves when it is ready. */
  ensure(modelId: string, mode?: AppMode): Promise<WritingModel>
}

export interface WritingRunsDeps {
  store: WritingStore
  chat: WritingChat
  models: WritingModels
  /** Called with a copy of the job on every status change and, throttled, while text streams. */
  emit: (job: WritingJob) => void
  /** Why something else holds the model right now (the desktop is generating), or null. */
  busyReason?: () => string | null
  /** The one-model-job-at-a-time gate shared with adventure jobs (src/main/modelGate.ts): taken where a job starts. */
  gate?: ModelGateLike
  now?: () => number
  newId?: () => string
  /** Least gap between two streaming emits; default 250 (about 4 a second). */
  throttleMs?: number
  /** Finished and running jobs kept per folder; default 50. */
  maxJobsPerFolder?: number
  /** Whether an owner ('phone:<clientId>') currently holds a project open. A phone that reconnects has a new owner id, so
   *  holding the job's folder is what gives it its jobs back. Without it only the starting owner may use a job. */
  holds?: (owner: string, folder: string) => boolean
  /** The model's own settings per mode, read for each request (the Writing prompt text); none = the built-in prompts. */
  modeDefaults?: (modelId: string) => ModelModeDefaults | undefined
}

const PROSE_ALT_MAX = 4
const DEFAULT_MAX_JOBS = 50
const RUNNING: WritingJobStatus[] = ['queued', 'preparing', 'waiting', 'writing']

const errorMessage = (e: unknown): string => {
  const msg = e instanceof Error ? e.message : String(e)
  return msg.replace(/^Error invoking remote method '[^']+':\s*(Error:\s*)?/, '')
}

interface JobRec {
  job: WritingJob
  stopRequested: boolean
  abort: (() => void) | null
  /** Gives the model gate back; called once, where the job ends. */
  release?: () => void
  lastEmit: number
  timer: ReturnType<typeof setTimeout> | null
  /** Update state: what the result's items would do, for applyState. */
  /** Who started it ('phone:<clientId>'); undefined for a call made inside main. */
  owner?: string
  /** dismiss() forgot the job while it still ran: it keeps counting as the running job until execute ends. */
  dismissed?: boolean
  /** applyState calls of this job run one after the other. */
  applyChain: Promise<unknown>
  state?: { sceneId: string; textHash: string; items: StateCardItem[]; applied: Set<string> }
}

/** Where a run's reading of the project ended up. */
interface Loaded {
  st: NovelProjectState
  hashes: Record<string, string>
  files: Map<string, NovelTextFile>
}

class Stopped extends Error {}

export class WritingRuns {
  private jobs = new Map<string, JobRec>()
  /** The job that is running, kept until its execute ends even when dismissed. */
  private current: JobRec | null = null
  private readonly now: () => number
  private readonly newId: () => string
  private readonly throttleMs: number
  private readonly maxJobs: number
  private readonly countText: ReturnType<typeof makeIpcCounter>

  constructor(private readonly deps: WritingRunsDeps) {
    this.now = deps.now ?? Date.now
    this.newId = deps.newId ?? (() => globalThis.crypto.randomUUID())
    this.throttleMs = deps.throttleMs ?? 250
    this.maxJobs = deps.maxJobsPerFolder ?? DEFAULT_MAX_JOBS
    this.countText = makeIpcCounter((modelId, text) => deps.chat.countText(modelId, text))
  }

  // -------------------------------------------------------------------------------------------
  // The phone's channels
  // -------------------------------------------------------------------------------------------

  /** Starts a job and returns it at once (status 'preparing'); progress and the result come as events. */
  run(req: WritingRunRequest, owner?: string): WritingJob {
    if (owner !== undefined && !this.holdsFolder(owner, req?.folder)) throw new Error('That project is not open on this phone.')
    if (!req || typeof req.folder !== 'string' || !req.folder) throw new Error('The request has no project folder.')
    if (!Object.hasOwn(ACTION_LABELS, req.action)) throw new Error('That action is not known.')
    if (typeof req.text !== 'string') throw new Error('The request has no text.')
    if (this.activeJob()) throw new Error('The computer is already working on another AI action from a phone. Wait for it to finish or stop it first.')
    const why = this.deps.busyReason?.() ?? null
    if (why) throw new Error(`${why} Try again when it has finished.`)
    const prose = isProseAction(req.action)
    const of = prose ? Math.min(PROSE_ALT_MAX, Math.max(1, Math.round(req.alternatives ?? 1))) : 1
    const { text: _text, ...request } = req
    const rec: JobRec = {
      job: {
        id: this.newId(),
        folder: req.folder,
        request,
        label: req.action === 'rewrite' && req.variant ? `${ACTION_LABELS.rewrite}: ${req.variant}` : ACTION_LABELS[req.action],
        status: 'preparing',
        alt: 1,
        of,
        progress: null,
        partial: '',
        result: null,
        error: null,
        warnings: [],
        startedAt: this.now(),
        endedAt: null
      },
      stopRequested: false,
      abort: null,
      lastEmit: 0,
      timer: null,
      owner,
      applyChain: Promise.resolve()
    }
    if (this.deps.gate) {
      const got = this.deps.gate.take(`writing:${rec.job.id}`)
      if (typeof got === 'string') throw new Error(`${got} Try again when it has finished.`)
      rec.release = got
    }
    this.trim(req.folder)
    this.jobs.set(rec.job.id, rec)
    this.current = rec
    this.send(rec)
    void this.execute(rec, req, of).catch((e) => this.fail(rec, errorMessage(e)))
    return this.snapshot(rec)
  }

  private holdsFolder(owner: string, folder: unknown): boolean {
    return typeof folder === 'string' && this.deps.holds?.(owner, folder) === true
  }

  /** The job, or null when it does not exist; throws when the caller may not use it. */
  private owned(jobId: string, owner: string | undefined): JobRec | null {
    const rec = this.jobs.get(jobId)
    if (!rec) return null
    if (owner !== undefined && rec.owner !== owner && !this.holdsFolder(owner, rec.job.folder)) throw new Error('That job belongs to another phone.')
    return rec
  }

  list(folder: string, owner?: string): WritingJob[] {
    if (owner !== undefined && !this.holdsFolder(owner, folder)) throw new Error('That project is not open on this phone.')
    return [...this.jobs.values()].filter((r) => r.job.folder === folder).map((r) => this.snapshot(r))
  }

  get(jobId: string, owner?: string): WritingJob | null {
    const rec = this.owned(jobId, owner)
    return rec ? this.snapshot(rec) : null
  }

  abort(jobId: string, owner?: string): void {
    const rec = this.owned(jobId, owner)
    if (!rec || !RUNNING.includes(rec.job.status)) return
    rec.stopRequested = true
    rec.abort?.()
  }

  /** Forgets a job. A running one is stopped first. */
  dismiss(jobId: string, owner?: string): void {
    const rec = this.owned(jobId, owner)
    if (!rec) return
    if (RUNNING.includes(rec.job.status)) {
      this.abort(jobId)
      rec.dismissed = true
    }
    if (rec.timer) clearTimeout(rec.timer)
    rec.timer = null
    this.jobs.delete(jobId)
  }

  /** Applies the chosen items of a finished Update state job, after checking the scene's disk text is the text it read. */
  async applyState(jobId: string, itemKeys: string[], owner?: string): Promise<NovelStatePatchResult> {
    const rec = this.owned(jobId, owner)
    if (!rec) throw new Error('That is not a finished state update.')
    // One call at a time per job: the second sees which items the first applied.
    const next = rec.applyChain.then(() => this.applyStateNow(rec, itemKeys))
    rec.applyChain = next.catch(() => undefined)
    return next
  }

  private async applyStateNow(rec: JobRec, itemKeys: string[]): Promise<NovelStatePatchResult> {
    if (!this.jobs.has(rec.job.id)) throw new Error('That is not a finished state update.')
    if (!rec || !rec.state || rec.job.result?.kind !== 'state') throw new Error('That is not a finished state update.')
    if (RUNNING.includes(rec.job.status)) throw new Error('The state update is still running.')
    if (!Array.isArray(itemKeys)) throw new Error('The chosen changes must be a list.')
    const { sceneId, textHash, items, applied } = rec.state
    const keys = new Set(itemKeys.map(String).filter((k) => !applied.has(k)))
    const chosen = items.filter((i) => keys.has(String(i.key))).map((i) => ({ ...i, accepted: true }))
    const chosenKeys = chosen.map((i) => String(i.key))
    if (!chosen.length) throw new Error('Nothing is accepted. Choose at least one change that has not been applied yet.')
    const folder = rec.job.folder
    const st = await this.deps.store.projectState(folder)
    if (st.blocked) throw new Error('The project is read-only, so the state was not changed.')
    if (st.state?.problem) throw new Error(`state.json cannot be used: ${st.state.problem}`)
    const found = findScene(st.outline, sceneId)
    if (!found) throw new Error('The scene is not in the outline any more.')
    // The proposal belongs to the scene text it read: if that moved, nothing is written.
    const file = await this.deps.store.readFile(folder, scenePath(found.scene)).catch(() => null)
    if (!file || hashText(file.text) !== textHash) throw new Error('The scene text changed since this was made. Run Update state again.')
    const { ops } = stateOpsFor(chosen, sceneId)
    const res = await this.deps.store.statePatch(folder, ops)
    // Only what was written counts as applied; a rejected op stays available (the phone sees it in `rejected`).
    const rejected = new Set(res.rejected.map((x) => x.index))
    chosenKeys.forEach((k, i) => {
      if (!rejected.has(i)) applied.add(k)
    })
    return res
  }

  models(): PhoneModel[] {
    return this.deps.models.list().map((m) => {
      const ready = this.deps.models.ready(m.id)
      // A rental's model: label null = it is not loaded on a rental now.
      return {
        id: m.id,
        name: m.name,
        loaded: !!ready,
        ...(m.remote ? { rental: { label: ready?.rental?.label ?? null, reconnecting: !!ready?.rental?.reconnecting } } : {})
      }
    })
  }

  // -------------------------------------------------------------------------------------------
  // Jobs: bookkeeping and events
  // -------------------------------------------------------------------------------------------

  private activeJob(): JobRec | undefined {
    const c = this.current
    return c && (RUNNING.includes(c.job.status) || c.dismissed) ? c : undefined
  }

  private snapshot(rec: JobRec): WritingJob {
    return { ...rec.job, request: { ...rec.job.request }, warnings: [...rec.job.warnings] }
  }

  /** Keeps at most maxJobs per folder: the oldest finished ones go to make room. */
  private trim(folder: string): void {
    const mine = [...this.jobs.values()].filter((r) => r.job.folder === folder)
    let over = mine.length - this.maxJobs + 1
    for (const r of mine) {
      if (over <= 0) break
      if (RUNNING.includes(r.job.status)) continue
      this.jobs.delete(r.job.id)
      over--
    }
  }

  private send(rec: JobRec): void {
    if (rec.timer) clearTimeout(rec.timer)
    rec.timer = null
    rec.lastEmit = this.now()
    this.deps.emit(this.snapshot(rec))
  }

  /** A status change goes out at once; streaming text and progress at most once per throttle gap. */
  private touch(rec: JobRec, immediate = false): void {
    if (!this.jobs.has(rec.job.id)) return
    const wait = this.throttleMs - (this.now() - rec.lastEmit)
    if (immediate || wait <= 0) return this.send(rec)
    if (rec.timer) return
    rec.timer = setTimeout(() => {
      rec.timer = null
      if (this.jobs.has(rec.job.id)) this.send(rec)
    }, wait)
    rec.timer.unref?.()
  }

  private setStatus(rec: JobRec, status: WritingJobStatus, over: Partial<WritingJob> = {}): void {
    Object.assign(rec.job, over, { status })
    this.touch(rec, true)
  }

  private finish(rec: JobRec, status: 'done' | 'stopped' | 'error', over: Partial<WritingJob>): void {
    rec.abort = null
    if (this.current === rec && !rec.dismissed) this.current = null
    this.setStatus(rec, status, { progress: null, partial: '', endedAt: this.now(), ...over })
  }

  private fail(rec: JobRec, error: string): void {
    if (!RUNNING.includes(rec.job.status)) return
    if (rec.stopRequested) return this.finish(rec, 'stopped', {})
    this.finish(rec, 'error', { error })
  }

  private checkStop(rec: JobRec): void {
    if (rec.stopRequested) throw new Stopped()
  }

  // -------------------------------------------------------------------------------------------
  // Reading the project
  // -------------------------------------------------------------------------------------------

  /** State, style, bible, outline and facts from the store, and every scene file's disk hash. */
  private async load(folder: string): Promise<Loaded> {
    const st = await this.deps.store.projectState(folder)
    const paths = st.outline.chapters.flatMap((c) => c.scenes.map(scenePath)).filter((p) => !st.missing.includes(p))
    const files = new Map<string, NovelTextFile>()
    const hashes: Record<string, string> = {}
    for (let i = 0; i < paths.length; i += 8) {
      const got = await Promise.all(paths.slice(i, i + 8).map((p) => this.deps.store.readFile(folder, p).then((f) => [p, f] as const, (e) => {
        // A file that is gone is left out; a project that is not open any more fails the job.
        if (/not open/i.test(errorMessage(e))) throw e
        return null
      })))
      for (const g of got) {
        if (!g) continue
        files.set(g[0], g[1])
        hashes[g[0]] = g[1].hash
      }
    }
    return { st, hashes, files }
  }

  private async resolveModel(project: NovelProjectState['project']): Promise<{ modelId: string; contextLength: number; instanceId: string }> {
    const { models } = this.deps
    let inst = models.ready(project.modelId)
    if (!inst) {
      if (project.modelId === null) throw new Error('No model is loaded. Load one in LLM Suite, or choose the project\'s model.')
      try {
        inst = await models.ensure(project.modelId, 'writing')
      } catch (e) {
        throw new Error(`The project's model could not be loaded: ${errorMessage(e)} Load it in LLM Suite first.`)
      }
    }
    return { modelId: inst.modelId, contextLength: inst.contextLength ?? 8192, instanceId: inst.instanceId }
  }

  private previousText(loaded: Loaded, sceneId: string): string {
    const all = loaded.st.outline.chapters.flatMap((c) => c.scenes)
    const i = all.findIndex((s) => s.id === sceneId)
    for (let j = i - 1; j >= 0; j--) {
      const f = loaded.files.get(scenePath(all[j]))
      if (f?.text.trim()) return f.text
    }
    return ''
  }

  private backend(): GenBackend {
    const { chat } = this.deps
    return {
      start: (req) => chat.start(req),
      abort: (id) => chat.abort(id),
      subscribe: (id, handler) => chat.subscribe(id, handler),
      newId: this.newId
    }
  }

  /** One generation, wired to the job: progress and streamed text go to its status. */
  private async generate(rec: JobRec, input: Omit<GenInput, 'onDelta' | 'onProgress'>): Promise<GenResult> {
    this.checkStop(rec)
    this.setStatus(rec, 'waiting', { progress: null, partial: '' })
    const gen = startGeneration(
      {
        ...input,
        onProgress: (p) => {
          if (rec.job.status !== 'waiting') return
          rec.job.progress = p
          this.touch(rec)
        },
        onDelta: (text) => {
          rec.job.partial = text
          if (rec.job.status !== 'writing') this.setStatus(rec, 'writing', { progress: null })
          else this.touch(rec)
        }
      },
      this.backend()
    )
    rec.abort = gen.abort
    try {
      return await gen.done
    } finally {
      rec.abort = null
    }
  }

  // -------------------------------------------------------------------------------------------
  // One job
  // -------------------------------------------------------------------------------------------

  private async execute(rec: JobRec, req: WritingRunRequest, of: number): Promise<void> {
    const holder = `job:${rec.job.id}`
    let held = false
    try {
      // The job holds its project open for its whole run, so a phone that leaves does not close it underneath.
      try {
        await this.deps.store.hold(req.folder, holder)
        held = true
      } catch (e) {
        throw new Error(`The project could not be opened: ${errorMessage(e)}`)
      }
      await this.executeInner(rec, req, of)
    } catch (e) {
      if (e instanceof Stopped || rec.stopRequested) this.finish(rec, 'stopped', {})
      else this.fail(rec, errorMessage(e))
    } finally {
      if (held) await this.deps.store.release(req.folder, holder).catch(() => undefined)
      rec.release?.()
      if (this.current === rec) this.current = null
    }
  }

  private async executeInner(rec: JobRec, req: WritingRunRequest, of: number): Promise<void> {
    const { action } = req
    const loaded = await this.load(req.folder)
    this.checkStop(rec)
    const { st } = loaded
    if (st.blocked) throw new Error(`The project is read-only: ${st.blocked}`)

    // ---- what the action works on, as the desktop checks it before it sends anything
    const needsScene = action !== 'outlineChapters' && action !== 'outlineScenes' && action !== 'summarizeChapter'
    // Chapter and project actions work on no scene, whatever the phone sent.
    const scene = needsScene && req.sceneId ? findScene(st.outline, req.sceneId) : null
    if (needsScene && req.sceneId && !scene) throw new Error('The scene is not in the outline any more.')
    if (req.chapterId && !findChapter(st.outline, req.chapterId)) throw new Error('The chapter is not in the outline any more.')
    if (needsScene && !scene) {
      throw new Error(action === 'checkScene' && req.chapterId ? 'Check one scene at a time: choose a scene.' : 'Choose a scene first.')
    }
    if ((action === 'outlineScenes' || action === 'summarizeChapter') && !req.chapterId) throw new Error('Choose a chapter first.')
    const text = req.text
    // Summarize scene, Check and Update state read the whole scene: the desktop sends them with the cursor at the end,
    // no selection and no note.
    const wholeScene = action === 'summarizeScene' || action === 'checkScene' || action === 'updateState'
    const cursor = wholeScene ? text.length : Math.max(0, Math.min(Math.round(req.cursor) || 0, text.length))
    let selection: { from: number; to: number } | null = null
    if (req.selection && !wholeScene) {
      const from = Math.max(0, Math.min(req.selection.from, text.length))
      const to = Math.max(from, Math.min(req.selection.to, text.length))
      selection = from === to ? null : { from, to }
    }
    if ((action === 'rewrite' || action === 'expand' || action === 'condense') && !selection) throw new Error('Select the text to change first.')
    if (action === 'summarizeScene' && !text.trim()) throw new Error('The scene has no text to summarize.')
    if (action === 'checkScene' && !text.trim()) throw new Error('The scene has no text to check.')
    if (action === 'updateState') {
      if (!text.trim()) throw new Error('The scene has no text to read.')
      if (st.state?.problem) throw new Error(`state.json cannot be used: ${st.state.problem}`)
    }
    let interview: { entry: BibleEntry; history: { role: 'user' | 'assistant'; content: string }[]; question: string } | null = null
    if (action === 'interview') {
      const turns = req.interview?.turns ?? []
      const last = turns[turns.length - 1]
      const entry = st.bible.find((e) => e.id === req.interview?.characterId)
      if (!entry) throw new Error('That character is not in the bible.')
      if (!last || last.role !== 'user' || !last.text.trim()) throw new Error('Ask a question first.')
      interview = { entry, history: turns.slice(0, -1).map((t) => ({ role: t.role, content: t.text })), question: last.text.trim() }
    }

    const model = await this.resolveModel(st.project)
    this.checkStop(rec)
    const estimated = { value: false }
    const count = this.countText(model.modelId, () => (estimated.value = true))
    const countTokens = (m: string, messages: WireMessage[]): Promise<number | null> => this.deps.chat.countTokens(m, messages).catch(() => null)
    const countWhole = (messages: { role: string; content: string }[]): Promise<number | null> => countTokens(model.modelId, messages as WireMessage[])
    const project = st.project
    const sampling = project.sampling
    // The model's own Writing prompt goes after the built-in one, through the system text, so it is counted and fitted with it.
    const promptExtra = modePromptExtra(this.deps.modeDefaults?.(model.modelId), 'writing')

    // ---- the interview is its own request: no scene buffer, no assembler
    if (interview) {
      const sceneId = scene!.scene.id
      const system = buildInterviewSystem({ entry: interview.entry, body: interview.entry.body, facts: st.state?.facts ?? [], sceneId, outline: st.outline, bible: st.bible, promptExtra })
      const limit = model.contextLength - project.budget.replyTokens - Math.max(256, Math.ceil(model.contextLength * 0.03))
      const fit = await fitInterview(system, interview.history, interview.question, countWhole, limit)
      this.checkStop(rec)
      if (fit.tooBig) throw new Error('This does not fit in the context. Shorten the character notes, or load a model with a longer context.')
      if (fit.dropped) rec.job.warnings.push(`${fit.dropped} earlier ${fit.dropped === 1 ? 'turn was' : 'turns were'} left out to fit the context.`)
      const res = await this.generate(rec, {
        modelId: model.modelId,
        messages: fit.messages,
        sampling,
        adapters: project.adapters,
        action: 'interview',
        prose: true,
        seed: seedsFor(sampling.seed, 1)[0],
        maxTokens: project.budget.replyTokens
      })
      if (res.error) throw new Error(res.error)
      const reply = res.text.trim()
      if (!reply) {
        if (res.stopped || rec.stopRequested) return this.finish(rec, 'stopped', {})
        throw new Error('The model returned no text.')
      }
      return this.finish(rec, rec.stopRequested ? 'stopped' : 'done', { result: { kind: 'interview', characterId: interview.entry.id, reply } })
    }

    // ---- the request, built and fitted exactly as the desktop does
    const over: Partial<RunContext> = { sceneId: scene?.scene.id ?? null }
    if (action === 'outlineScenes' || action === 'summarizeChapter') over.chapterId = req.chapterId
    if (action === 'outlineChapters') over.count = clampCount(req.count, 8)
    if (action === 'outlineScenes') over.count = clampCount(req.count, 5)
    // A note goes with the prose actions and the outline plans only (the desktop's summaries, check and state send none).
    if ((isProseAction(action) && scene) || action === 'outlineChapters' || action === 'outlineScenes') over.note = req.note ?? ''
    const overhead = await templateOverheadWith(countTokens, model.instanceId, model.modelId, count)
    this.checkStop(rec)
    const ctx: RunContext = {
      action,
      project,
      style: st.style.text,
      outline: st.outline,
      bible: st.bible,
      sceneId: null,
      sceneHashes: loaded.hashes,
      facts: st.state?.facts ?? [],
      text: scene ? text : '',
      cursor: scene ? cursor : 0,
      selection: scene ? selection : null,
      note: '',
      variant: req.variant,
      previousText: action === 'draftScene' && scene && !text.trim() ? this.previousText(loaded, scene.scene.id) : '',
      contextLength: model.contextLength,
      templateOverhead: overhead,
      promptExtra,
      ...over
    }
    const fit = await fitRequest(ctx, count, countWhole, estimated)
    this.checkStop(rec)
    if (estimated.value) rec.job.warnings.push('The model could not count tokens, so sizes are estimates.')
    rec.job.warnings.push(...fit.built.result.warnings)
    const { missing, stale } = fit.built.fallbacks
    if (missing) rec.job.warnings.push(`${missing} earlier ${missing === 1 ? 'summary is' : 'summaries are'} missing: the plan was used instead.`)
    if (stale) rec.job.warnings.push(`${stale} earlier ${stale === 1 ? 'summary is' : 'summaries are'} out of date: the plan was used instead.`)
    if (fit.tooBig) throw new Error('The request does not fit in the context. Shorten the style notes, entries or summaries, or load a model with a longer context.')
    if (action === 'checkScene' || action === 'updateState') {
      const problem = wholeSceneProblem(fit.built)
      if (problem) throw new Error(problem)
    }
    const messages = fit.built.result.messages

    // ---- prose: one to four alternatives, one after the other
    if (isProseAction(action)) {
      const seeds = seedsFor(sampling.seed, of)
      const alts: WritingAlt[] = []
      let failure: string | null = null
      for (let i = 0; i < of && !rec.stopRequested; i++) {
        rec.job.alt = i + 1
        const res = await this.generate(rec, {
          modelId: model.modelId,
          messages,
          sampling,
          adapters: project.adapters,
          action,
          seed: seeds[i],
          maxTokens: project.budget.replyTokens
        })
        const generated = finishText(action, res.text)
        if (res.error) {
          failure = res.error
          break
        }
        if (!generated) {
          if (!res.stopped) failure = 'The model returned no text.'
          break
        }
        alts.push({ text: applyResult(action, text, cursor, selection, generated), generated, seed: seeds[i] })
        if (res.stopped) break
      }
      if (!alts.length) {
        if (failure) throw new Error(failure)
        return this.finish(rec, 'stopped', {})
      }
      if (failure) rec.job.warnings.push(`Stopped after ${alts.length} of ${of}: ${failure}`)
      const result: WritingResult = { kind: 'prose', alts, newSticky: fit.built.newSticky }
      return this.finish(rec, rec.stopRequested ? 'stopped' : 'done', { result })
    }

    // ---- structured replies: summaries, outline proposals, check, update state
    const schema =
      action === 'outlineChapters'
        ? OUTLINE_CHAPTERS_SCHEMA
        : action === 'outlineScenes'
          ? OUTLINE_SCENES_SCHEMA
          : action === 'checkScene'
            ? CHECK_SCHEMA
            : action === 'updateState'
              ? UPDATE_STATE_SCHEMA
              : SUMMARY_SCHEMA
    const res = await this.generate(rec, {
      modelId: model.modelId,
      messages,
      sampling,
      adapters: project.adapters,
      action,
      seed: seedsFor(sampling.seed, 1)[0],
      maxTokens: Math.max(1024, project.budget.replyTokens),
      schema
    })
    if (res.stopped && (!res.text.trim() || action === 'checkScene' || action === 'updateState')) return this.finish(rec, 'stopped', {})
    if (res.error) throw new Error(res.error)
    const raw = res.text.trim()

    if (action === 'summarizeScene' || action === 'summarizeChapter') {
      const p = parseSummary(raw)
      if (!p.ok) throw new Error(p.error)
      if (action === 'summarizeScene') {
        const s = scene!.scene
        const file = loaded.files.get(scenePath(s))
        // The summary belongs to the text it read: with unsaved edits there is no disk hash to tie it to.
        let summaryOf = file?.hash ?? ''
        if (!file || file.text !== text) {
          summaryOf = ''
          rec.job.warnings.push('The scene has edits that are not saved, so the summary is marked out of date until it is made from the saved text.')
        }
        return this.finish(rec, rec.stopRequested ? 'stopped' : 'done', { result: { kind: 'summary', target: 'scene', targetId: s.id, reply: p.value, summaryOf } })
      }
      const chapter = findChapter(st.outline, req.chapterId!)!.chapter
      const summaryOf = sceneHashKey(chapter.scenes.map((sc) => loaded.hashes[scenePath(sc)] ?? ''))
      return this.finish(rec, rec.stopRequested ? 'stopped' : 'done', { result: { kind: 'summary', target: 'chapter', targetId: chapter.id, reply: p.value, summaryOf } })
    }
    if (action === 'outlineChapters') {
      const p = parseOutlineChapters(raw)
      if (!p.ok) throw new Error(p.error)
      return this.finish(rec, rec.stopRequested ? 'stopped' : 'done', { result: { kind: 'chapters', reply: p.value, unresolved: [] } })
    }
    if (action === 'outlineScenes') {
      const p = parseOutlineScenes(raw)
      if (!p.ok) throw new Error(p.error)
      // pov and present come back as bible ids (names that match nothing are dropped and listed), so the phone can
      // build the outline ops without the bible.
      const refs = resolveSceneRefs(st.bible, p.value.scenes)
      const reply = { scenes: p.value.scenes.map((sc, i) => ({ ...sc, pov: refs.scenes[i].pov, present: refs.scenes[i].present })) }
      return this.finish(rec, rec.stopRequested ? 'stopped' : 'done', { result: { kind: 'scenes', chapterId: req.chapterId!, reply, unresolved: refs.unresolved } })
    }
    if (action === 'checkScene') {
      const p = parseCheckReplyResult(raw)
      if (!p.ok) throw new Error(p.error)
      const findings: CheckFinding[] = dedupeFindings(p.value.findings)
      return this.finish(rec, 'done', { result: { kind: 'check', sceneId: scene!.scene.id, findings } })
    }
    // updateState
    const p = parseUpdateStateReplyResult(raw)
    if (!p.ok) throw new Error(p.error)
    const sceneId = scene!.scene.id
    const before = stateBefore(st.state?.facts ?? [], sceneId, st.outline, st.bible)
    const resolved = resolveStateChanges(p.value.changes as UpdateStateChange[], st.bible, before)
    const textHash = hashText(text)
    rec.state = { sceneId, textHash, items: resolved.items, applied: new Set() }
    const items: WritingStateItem[] = resolved.items.map((it) =>
      it.planned.op === 'add'
        ? { key: String(it.key), label: it.label, op: 'add', fact: { entity: it.planned.entity, kind: it.planned.kind, value: it.planned.value, about: it.planned.about, from: sceneId, until: null } }
        : { key: String(it.key), label: it.label, op: 'end', factId: it.planned.factId }
    )
    return this.finish(rec, 'done', { result: { kind: 'state', sceneId, textHash, items, left: resolved.left } })
  }
}

function clampCount(n: number | undefined, fallback: number): number {
  if (typeof n !== 'number' || !Number.isFinite(n)) return fallback
  return Math.min(50, Math.max(1, Math.round(n)))
}
