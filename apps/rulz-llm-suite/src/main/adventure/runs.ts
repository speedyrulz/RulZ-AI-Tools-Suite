// Adventure mode's generation as jobs in main (docs/adventure-mode-design.md, "A turn"). The desktop window and the
// phone start, watch and stop the same jobs. Like WritingRuns, everything the service touches is injected (store,
// chat, models, gate, clock, ids, emit), so tests run it with fakes and no model.
import { DEFAULT_SAMPLING } from '@shared/defaults'
import {
  PASSAGE_WORDS,
  emptyState,
  pathTo,
  type Adventure,
  type AdventureJob,
  type AdventureJobKind,
  type AdventureJobStatus,
  type AdventureSetup,
  type NodeInput,
  type PlayRequest,
  type RetryRequest
} from '@shared/adventure'
import { choosingFinal, choosingSchema, cleanPassage, fitStory, parseChoosing, passageFinal, systemMessage, type StoryFit } from '@shared/adventurePrompts'
import { startGeneration, type GenBackend, type GenInput, type GenResult } from '@shared/novelRun'
import type { WireMessage } from '@shared/types'
import type { ModelGateLike } from '../modelGate'
import type { WritingChat, WritingModels } from '../writing/runs'
import { checkAdventureId, type AdventureStore } from './store'

export interface AdventureRunsDeps {
  store: AdventureStore
  chat: WritingChat
  models: WritingModels
  /** One model job at a time on the whole computer (src/main/modelGate.ts); taken for a job from start to end. */
  gate: ModelGateLike
  /** Called with a copy of the job on every status change and, throttled, while text streams. */
  emit: (job: AdventureJob) => void
  /** Why something else holds the model right now (the desktop is generating, a benchmark runs, the gate), or null. */
  busyReason?: () => string | null
  now?: () => number
  newId?: () => string
  /** A seed for each request; default random. */
  seed?: () => number
  /** Least gap between two streaming emits; default 250 (about 4 a second). */
  throttleMs?: number
  /** Finished jobs kept per adventure; default 20. */
  maxFinishedPerAdventure?: number
}

const RUNNING: AdventureJobStatus[] = ['queued', 'preparing', 'waiting', 'writing', 'choosing']
const DEFAULT_MAX_FINISHED = 20
/** Reply budget of the choosing request: a short JSON object, with room for a model that thinks first. */
const CHOOSING_TOKENS = 1536

const errorMessage = (e: unknown): string => {
  const msg = e instanceof Error ? e.message : String(e)
  return msg.replace(/^Error invoking remote method '[^']+':\s*(Error:\s*)?/, '')
}

class Stopped extends Error {}

/** What a job does, decided (and checked) before it starts. */
interface Plan {
  kind: AdventureJobKind
  /** play: the parent (null = the opening). retryPassage / retryChoices: the node retried. */
  nodeId: string | null
  input: NodeInput | null
  /** Where a new passage goes (play, retryPassage); null = the opening. */
  parentId: string | null
  /** Nodes the job works on; deleteBranch is refused while a running job uses one of them. */
  uses: Set<string>
}

interface JobRec {
  job: AdventureJob
  stopRequested: boolean
  abort: (() => void) | null
  lastEmit: number
  timer: ReturnType<typeof setTimeout> | null
  /** Gives the model gate back; called once, as soon as the job ends. */
  release: () => void
  uses: Set<string>
}

export class AdventureRuns {
  private jobs = new Map<string, JobRec>()
  /** Jobs whose execute has not ended, dismissed or not: they keep the gate and their branch. */
  private running = new Set<JobRec>()
  private readonly now: () => number
  private readonly newId: () => string
  private readonly seed: () => number
  private readonly throttleMs: number
  private readonly maxFinished: number

  constructor(private readonly deps: AdventureRunsDeps) {
    this.now = deps.now ?? Date.now
    this.newId = deps.newId ?? (() => globalThis.crypto.randomUUID())
    this.seed = deps.seed ?? (() => Math.floor(Math.random() * 2_000_000_000))
    this.throttleMs = deps.throttleMs ?? 250
    this.maxFinished = deps.maxFinishedPerAdventure ?? DEFAULT_MAX_FINISHED
  }

  // -------------------------------------------------------------------------------------------
  // The channels
  // -------------------------------------------------------------------------------------------

  /** Starts a job that writes a new passage under `parentId`; returns it at once, progress and result come as events. */
  async play(req: PlayRequest): Promise<AdventureJob> {
    if (!req || typeof req !== 'object') throw new Error('The request is missing.')
    return this.begin(req.adventureId, (adv) => this.planPlay(adv, req))
  }

  /** Starts a job that writes a passage again (a new sibling) or writes new choices for a passage. */
  async retry(req: RetryRequest): Promise<AdventureJob> {
    if (!req || typeof req !== 'object') throw new Error('The request is missing.')
    return this.begin(req.adventureId, (adv) => this.planRetry(adv, req))
  }

  list(adventureId: string): AdventureJob[] {
    checkAdventureId(adventureId)
    return [...this.jobs.values()].filter((r) => r.job.adventureId === adventureId).map((r) => this.snapshot(r))
  }

  abort(jobId: string): void {
    const rec = this.jobs.get(jobId)
    if (!rec || !RUNNING.includes(rec.job.status)) return
    rec.stopRequested = true
    rec.abort?.()
  }

  /** Forgets a job. A running one is stopped first (it keeps the model until it has ended). */
  dismiss(jobId: string): void {
    const rec = this.jobs.get(jobId)
    if (!rec) return
    if (RUNNING.includes(rec.job.status)) this.abort(jobId)
    if (rec.timer) clearTimeout(rec.timer)
    rec.timer = null
    this.jobs.delete(jobId)
  }

  models(): { id: string; name: string; loaded: boolean }[] {
    return this.deps.models.list().map((m) => ({ id: m.id, name: m.name, loaded: !!this.deps.models.ready(m.id) }))
  }

  /** A job is running on this adventure (deleting the adventure is refused meanwhile). */
  hasRunning(adventureId: string): boolean {
    for (const r of this.running) if (r.job.adventureId === adventureId) return true
    return false
  }

  /** A running job works on one of these passages (deleting their branch is refused meanwhile). */
  branchInUse(adventureId: string, nodeIds: Set<string>): boolean {
    for (const r of this.running) {
      if (r.job.adventureId !== adventureId) continue
      for (const id of r.uses) if (nodeIds.has(id)) return true
      if (r.job.resultNodeId && nodeIds.has(r.job.resultNodeId)) return true
    }
    return false
  }

  // -------------------------------------------------------------------------------------------
  // Starting
  // -------------------------------------------------------------------------------------------

  private planPlay(adv: Adventure, req: PlayRequest): Plan {
    const input = req.input as NodeInput | undefined
    if (!input || typeof input !== 'object') throw new Error('The request has no step.')
    if (req.parentId === null) {
      if (input.kind !== 'start') throw new Error('Only the opening has no passage before it.')
      if (adv.rootId !== null) throw new Error('This adventure already has its opening.')
      return { kind: 'play', nodeId: null, input, parentId: null, uses: new Set() }
    }
    if (typeof req.parentId !== 'string' || !Object.hasOwn(adv.nodes, req.parentId)) throw new Error('That passage was not found.')
    const parent = adv.nodes[req.parentId]
    if (input.kind === 'start') throw new Error('The opening can only be written first.')
    if (input.kind === 'choice') {
      if (parent.ended) throw new Error('The story ended here. Type what you do to carry on.')
      const choice = parent.choices.find((c) => c.id === input.choiceId)
      if (!choice) throw new Error('That choice is not offered here any more. Reload the passage.')
      // The text kept on the node is the parent's own, not whatever the caller sent.
      return { kind: 'play', nodeId: parent.id, input: { kind: 'choice', choiceId: choice.id, text: choice.text }, parentId: parent.id, uses: new Set([parent.id]) }
    }
    if (input.kind !== 'action') throw new Error('That step is not known.')
    const text = typeof input.text === 'string' ? input.text.trim().slice(0, 2000) : ''
    if (!text) throw new Error('Type what you do first.')
    return { kind: 'play', nodeId: parent.id, input: { kind: 'action', text }, parentId: parent.id, uses: new Set([parent.id]) }
  }

  private planRetry(adv: Adventure, req: RetryRequest): Plan {
    if (typeof req.nodeId !== 'string' || !Object.hasOwn(adv.nodes, req.nodeId)) throw new Error('That passage was not found.')
    const node = adv.nodes[req.nodeId]
    if (req.what === 'choices') return { kind: 'retryChoices', nodeId: node.id, input: null, parentId: null, uses: new Set([node.id]) }
    if (req.what !== 'passage') throw new Error('That retry is not known.')
    if (node.parentId === null) throw new Error('The opening cannot be written again from here. Edit it, or start a new adventure.')
    return { kind: 'retryPassage', nodeId: node.id, input: node.input, parentId: node.parentId, uses: new Set([node.id, node.parentId]) }
  }

  /** Checks the gate and the adventure, then starts the job. The gate is taken first, so two starts cannot both pass. */
  private async begin(adventureId: unknown, plan: (adv: Adventure) => Plan): Promise<AdventureJob> {
    const id = checkAdventureId(adventureId)
    const why = this.deps.busyReason?.() ?? null
    if (why) throw new Error(`${why} Try again when it has finished.`)
    const jobId = this.newId()
    const got = this.deps.gate.take(`adventure:${jobId}`)
    if (typeof got === 'string') throw new Error(`${got} Try again when it has finished.`)
    let started = false
    try {
      const adv = await this.deps.store.get(id)
      const p = plan(adv)
      const rec: JobRec = {
        job: {
          id: jobId,
          adventureId: id,
          kind: p.kind,
          nodeId: p.nodeId,
          input: p.input,
          status: 'preparing',
          progress: null,
          text: '',
          resultNodeId: null,
          error: null,
          warnings: [],
          startedAt: this.now(),
          endedAt: null
        },
        stopRequested: false,
        abort: null,
        lastEmit: 0,
        timer: null,
        release: got,
        uses: p.uses
      }
      this.trim(id)
      this.jobs.set(jobId, rec)
      this.running.add(rec)
      started = true
      this.send(rec)
      void this.execute(rec, p).catch((e) => this.fail(rec, errorMessage(e)))
      return this.snapshot(rec)
    } finally {
      if (!started) got()
    }
  }

  // -------------------------------------------------------------------------------------------
  // Jobs: bookkeeping and events (as WritingRuns)
  // -------------------------------------------------------------------------------------------

  private snapshot(rec: JobRec): AdventureJob {
    return { ...rec.job, input: rec.job.input ? { ...rec.job.input } : null, warnings: [...rec.job.warnings] }
  }

  /** Keeps at most maxFinished finished jobs per adventure: the oldest go to make room. */
  private trim(adventureId: string): void {
    const finished = [...this.jobs.values()].filter((r) => r.job.adventureId === adventureId && !RUNNING.includes(r.job.status))
    for (let i = 0; i < finished.length - this.maxFinished + 1; i++) {
      const r = finished[i]
      if (r.timer) clearTimeout(r.timer)
      this.jobs.delete(r.job.id)
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

  private setStatus(rec: JobRec, status: AdventureJobStatus, over: Partial<AdventureJob> = {}): void {
    Object.assign(rec.job, over, { status })
    this.touch(rec, true)
  }

  private finish(rec: JobRec, status: 'done' | 'stopped' | 'error', over: Partial<AdventureJob> = {}): void {
    if (!RUNNING.includes(rec.job.status)) return
    rec.abort = null
    // Back before the last event goes out, so a client that starts the next job on seeing it is not refused.
    rec.release()
    this.running.delete(rec)
    // A passage that was never saved is not kept as text: it would look like one that was.
    const text = status !== 'done' && !rec.job.resultNodeId ? '' : rec.job.text
    this.setStatus(rec, status, { progress: null, text, endedAt: this.now(), ...over })
  }

  private fail(rec: JobRec, error: string): void {
    if (!RUNNING.includes(rec.job.status)) return
    if (rec.stopRequested) return this.finish(rec, 'stopped')
    this.finish(rec, 'error', { error })
  }

  private checkStop(rec: JobRec): void {
    if (rec.stopRequested) throw new Stopped()
  }

  private warn(rec: JobRec, text: string): void {
    if (!rec.job.warnings.includes(text)) rec.job.warnings.push(text)
  }

  // -------------------------------------------------------------------------------------------
  // One job
  // -------------------------------------------------------------------------------------------

  private async execute(rec: JobRec, plan: Plan): Promise<void> {
    try {
      await this.run(rec, plan)
    } catch (e) {
      if (e instanceof Stopped || rec.stopRequested) this.finish(rec, 'stopped')
      else this.fail(rec, errorMessage(e))
    } finally {
      rec.release()
      this.running.delete(rec)
    }
  }

  private async resolveModel(setup: AdventureSetup): Promise<{ modelId: string; instanceId: string; contextLength: number }> {
    const { models } = this.deps
    let inst = models.ready(setup.modelId)
    if (!inst) {
      if (setup.modelId === null) throw new Error('No model is loaded. Load one in LLM Suite, or choose a model in the adventure\'s settings.')
      try {
        inst = await models.ensure(setup.modelId)
      } catch (e) {
        throw new Error(`The adventure's model could not be loaded: ${errorMessage(e)} Load it in LLM Suite first.`)
      }
    }
    return { modelId: inst.modelId, instanceId: inst.instanceId, contextLength: inst.contextLength ?? 8192 }
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

  /**
   * One request, wired to the job. A passage goes waiting (prompt progress), then writing (the text streams into the
   * job). The choices request keeps the status 'choosing': only its progress shows.
   */
  private async generate(rec: JobRec, input: Omit<GenInput, 'onDelta' | 'onProgress'>, passage: boolean): Promise<GenResult> {
    this.checkStop(rec)
    if (passage) this.setStatus(rec, 'waiting', { progress: null, text: '' })
    else if (rec.job.status !== 'choosing') this.setStatus(rec, 'choosing', { progress: null })
    const waitingStatus = passage ? 'waiting' : 'choosing'
    const gen = startGeneration(
      {
        ...input,
        onProgress: (p) => {
          if (rec.job.status !== waitingStatus) return
          rec.job.progress = p
          this.touch(rec)
        },
        onDelta: (text) => {
          if (!passage) {
            if (rec.job.progress !== null) rec.job.progress = null
            return
          }
          rec.job.text = text
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

  private noteFit(rec: JobRec, fit: StoryFit): void {
    if (fit.estimated) this.warn(rec, 'The model could not count tokens, so sizes are estimates.')
    if (fit.dropped > 0) this.warn(rec, 'Older passages were left out to fit the context.')
    if (fit.tooBig) throw new Error('The story does not fit in the context. Load a model with a longer context.')
  }

  private async run(rec: JobRec, plan: Plan): Promise<void> {
    const { store, chat } = this.deps
    let adv = await store.get(rec.job.adventureId)
    this.checkStop(rec)
    const setup = adv.setup
    const model = await this.resolveModel(setup)
    this.checkStop(rec)
    const sampling = setup.temperature === null ? DEFAULT_SAMPLING : { ...DEFAULT_SAMPLING, temperature: setup.temperature }
    const system = systemMessage(setup)
    const margin = Math.max(256, Math.ceil(model.contextLength * 0.03))
    const countMessages = (m: { role: string; content: string }[]): Promise<number | null> => chat.countTokens(model.modelId, m as WireMessage[]).catch(() => null)

    let nodeId: string
    if (plan.kind !== 'retryChoices') {
      // ---- Writing: the passage streams as prose, no thinking, no schema
      const input = plan.input as NodeInput
      const parent = plan.parentId ? adv.nodes[plan.parentId] : undefined
      if (plan.parentId && !parent) throw new Error('The passage this continues from is gone.')
      const replyTokens = Math.max(256, PASSAGE_WORDS[setup.length] * 2)
      const fit = await fitStory({
        system,
        path: pathTo(adv, plan.parentId),
        final: passageFinal(input, parent?.state ?? emptyState()),
        countMessages,
        limit: model.contextLength - replyTokens - margin
      })
      this.checkStop(rec)
      this.noteFit(rec, fit)
      const res = await this.generate(rec, { modelId: model.modelId, messages: fit.messages, sampling, adapters: null, action: 'continue', prose: true, seed: this.seed(), maxTokens: replyTokens }, true)
      if (res.error) throw new Error(res.error)
      // Stopped (or cut short by Stop) before the passage finished: nothing is saved.
      if (res.stopped) return this.finish(rec, 'stopped')
      const text = cleanPassage(res.text)
      if (!text) throw new Error('The model returned no text.')
      if (res.stopReason === 'length') this.warn(rec, 'The passage was cut off at the reply limit.')
      // Saved at once, with no choices: a stop or a failure while the choices are written keeps the passage.
      const saved = await store.addNode(adv.id, { parentId: plan.parentId, input, text })
      nodeId = saved.node.id
      this.setStatus(rec, 'choosing', { progress: null, text, resultNodeId: nodeId })
    } else {
      nodeId = plan.nodeId as string
      this.setStatus(rec, 'choosing', { progress: null, resultNodeId: nodeId })
    }

    // ---- Choosing: a short structured request about the passage that is now saved
    this.checkStop(rec)
    adv = await store.get(adv.id)
    const node = adv.nodes[nodeId]
    if (!node) throw new Error('That passage is gone.')
    // The passage as the choosing request reads it: a hand edit made meanwhile moves the rev past this one.
    const baseRev = node.rev
    const parent = node.parentId ? adv.nodes[node.parentId] : undefined
    // A small context cannot spare the whole budget: a quarter of it, at least 512.
    const chooseTokens = Math.min(CHOOSING_TOKENS, Math.max(512, Math.floor(model.contextLength / 4)))
    const fit = await fitStory({
      system,
      path: pathTo(adv, nodeId),
      final: choosingFinal(setup, parent?.state ?? emptyState(), node.parentId === null),
      countMessages,
      limit: model.contextLength - chooseTokens - margin,
      keepNewest: true
    })
    this.checkStop(rec)
    this.noteFit(rec, fit)
    // prose: true only turns thinking off, the schema still applies. A thinking model would reason through the whole
    // small budget and send back no JSON at all; the schema already makes the reply the structured part.
    const res = await this.generate(
      rec,
      { modelId: model.modelId, messages: fit.messages, sampling, adapters: null, action: 'checkScene', prose: true, seed: this.seed(), maxTokens: chooseTokens, schema: choosingSchema(setup.choiceCount) },
      false
    )
    if (res.stopped) return this.finish(rec, 'stopped')
    if (res.error) throw new Error(res.error)
    const parsed = parseChoosing(res.text, setup.choiceCount)
    if (!parsed.ok) {
      const why = res.stopReason === 'length' ? 'the reply was cut off at its limit' : parsed.error.replace(/\.$/, '').replace(/^The /, 'the ')
      this.warn(rec, `The choices could not be read (${why}). Use New choices, or type what you do.`)
      return this.finish(rec, 'done')
    }
    for (const w of parsed.warnings) this.warn(rec, w)
    const saved = await store.setChoosing(adv.id, nodeId, parsed.value, baseRev)
    if (!saved) this.warn(rec, 'The passage was edited while its choices were written, so they were not saved. Use New choices.')
    this.finish(rec, 'done')
  }
}
