// The practice pool: runs a batch of practice tasks over lanes (one lane = one slot of one loaded instance), each
// lane running one unattended session at a time (docs/vast-design.md, section 8). One run at a time. The pool
// reaches everything else (instances, rentals, the chat service, MCP, the conversation store) through PoolDeps, so
// tests run it against fakes and index.ts wires the real parts.
import {
  batchAdapters,
  cleanBatchOptions,
  itemResult,
  planBatch,
  sameBatchAdapters,
  sessionStats,
  type BatchItem,
  type BatchRun,
  type BatchStop
} from '@shared/batch'
import { RE_MIN_CONTEXT } from '@shared/re'
import { cleanBatchTarget, type BatchTarget } from '@shared/remote'
import type { BatchStartRequest } from '@shared/ipc'
import type { Conversation, ConversationSummary, McpServerStatus, ToolResult } from '@shared/types'
import { runSession, type InstanceState, type SessionChat, type SessionDeps, type SessionResult, type SessionSettings, type SessionSpec } from './session'

/** An instance a lane can use, as the wiring prepared it (the model is loaded). */
export interface PoolHost {
  instanceId: string
  /** "This PC", "RTX 4090 (12345)". */
  hostLabel: string
  /** 0 for this PC. */
  dollarsPerHour: number
  rentalId?: number
  modelName: string
  contextLength: number | null
  adapters: { name: string; scale: number }[] | undefined
}

/** What a lane needs of an MCP scope. */
export interface PoolScope {
  /** A new task starts: nothing of the last one stays open. */
  beginTask(): void
  callTool(call: { name: string; arguments: string }, signal?: AbortSignal): Promise<ToolResult>
  close(): Promise<void>
}

export interface PoolEvents {
  batchChanged(run: BatchRun, active: boolean, stopping: BatchStop | null): void
  poolChats(ids: string[]): void
  chatLive(conversation: Conversation): void
  conversationSaved(summary: ConversationSummary): void
}

export interface PoolDeps {
  now(): number
  sleep(ms: number): Promise<void>
  uid(): string
  store: {
    save(run: BatchRun): Promise<void>
    get(id: string): Promise<BatchRun | undefined>
    setActive?(fn: () => string | null): void
  }
  conversations: {
    own(id: string): void
    release(id: string): void
    savePool(conversation: Conversation): Promise<ConversationSummary>
    get(id: string): Promise<Conversation | null>
  }
  events: PoolEvents
  /** A sentence when something stops a run from starting on these hosts (a benchmark, a phone job, a chat reply on this PC). */
  busy(hosts: BatchTarget['hosts']): string | null
  /**
   * Makes the model ready on every host of the target (loading it where needed; `load` says how: null = as it is
   * loaded or with its defaults). Rejects with a sentence; never rents or starts anything.
   */
  prepareHosts(target: BatchTarget, modelId: string, load: { parallel: number; contextLength: number } | null): Promise<PoolHost[]>
  /** Marks the host's instance (and rental) as in use by the run; call the result to release it. */
  useHost(instanceId: string): () => void
  /** The shared MCP connections' status. */
  servers(): McpServerStatus[]
  /**
   * One scope per lane; throws a sentence when REA cannot run privately. `alone`: this is the run's only lane, so
   * with no local REA program (an HTTP server, or none) the scope falls back to the shared connection, as the
   * old loop did.
   */
  openScope(opts: { alone: boolean }): PoolScope
  settings(): SessionSettings
  chat: SessionChat
  instance(instanceId: string): { hasVision: boolean } | undefined
  instanceState(instanceId: string): InstanceState
  prepare(kitDir: string, taskId: string): Promise<{ target: string; prompt: string; family: string; tier: string }>
  grade: SessionDeps['grade']
  /** A run ended: the rentals to stop when the target asked for it (empty otherwise). */
  onEnded(run: BatchRun, rentalIds: number[]): void
  warn?(message: string): void
  /** Test hooks. */
  runSession?: (deps: SessionDeps, spec: SessionSpec) => Promise<SessionResult>
  /** Lanes start this far apart (the first Ghidra analysis is heavy on the CPU). */
  staggerMs?: number
}

export const LANE_STAGGER_MS = 3000
/** Saves of the run are coalesced to at most one a second. */
export const SAVE_INTERVAL_MS = 1000
/** A task whose instance was lost this many times becomes an error instead of going back to pending. */
export const MAX_LOST_PER_ITEM = 2

interface Lane {
  n: number
  host: PoolHost
  hostIndex: number
  scope: PoolScope
  ended: boolean
}

interface HostTime {
  active: number
  since: number
  accrued: number
  release: () => void
}

interface Running {
  ctrl: AbortController
  itemIndex: number
}

const errorText = (e: unknown): string => (e instanceof Error ? e.message : String(e))

function adapterLabel(a: { name: string; scale: number }[] | undefined): string {
  return a?.length ? a.map((x) => `+ ${x.name} x${Math.round(x.scale * 1000) / 1000}`).join(', ') : ''
}

export class PracticePool {
  private run: BatchRun | null = null
  private stopMode: BatchStop | null = null
  private running = new Map<string, Running>()
  private owned = new Set<string>()
  private watchedId: string | null = null
  private lanes: Lane[] = []
  private hostTimes: HostTime[] = []
  private baseCost = 0
  private finishedPromise: Promise<void> = Promise.resolve()
  private starting = false
  /** The run has ended and its last save is being written: no new run may start meanwhile. */
  private settling = false
  private disposed = false
  private saveTimer: ReturnType<typeof setTimeout> | null = null
  private lastSaveAt = 0
  private saveChain: Promise<void> = Promise.resolve()
  private hostLabels = ''

  constructor(private readonly deps: PoolDeps) {
    deps.store.setActive?.(() => this.activeId)
  }

  /** The run in progress. */
  get activeId(): string | null {
    return this.run && !this.runEnded ? this.run.id : null
  }
  private runEnded = true

  get stopping(): BatchStop | null {
    return this.stopMode
  }

  /** True while a run is starting or running. */
  get busy(): boolean {
    return this.starting || this.settling || !this.runEnded
  }

  /** The hosts' instance ids of the run in progress; this PC's model is not free for other work while they are. */
  get activeInstances(): string[] {
    return this.runEnded ? [] : this.lanes.map((l) => l.host.instanceId)
  }

  /** The chats the pool owns now. */
  poolChats(): string[] {
    return [...this.owned]
  }

  /** Resolves when the run in progress (if any) has ended and been saved. */
  whenIdle(): Promise<void> {
    return this.finishedPromise
  }

  // ---------- starting ----------

  async start(req: BatchStartRequest): Promise<BatchRun> {
    this.guardStart()
    const options = cleanBatchOptions(req.options)
    const items = planBatch(req.tasks, options, new Set(req.solved))
    if (!items.length) throw new Error('No practice tasks match this selection.')
    if (!req.modelId) throw new Error('Load a model first, or choose one under New session.')
    const target = cleanBatchTarget(req.target)
    this.starting = true
    try {
      const hosts = await this.prepare(target, req.modelId)
      // The app began closing while the model loaded: nothing is acquired and no run begins.
      if (this.disposed) throw new Error('RulZ LLM Suite is closing.')
      const first = hosts[0]
      const run: BatchRun = {
        id: this.deps.uid(),
        label: req.label.trim().slice(0, 80),
        createdAt: this.deps.now(),
        status: 'running',
        modelId: req.modelId,
        modelName: first.modelName,
        contextLength: first.contextLength,
        maxToolRounds: this.deps.settings().re.maxToolRounds,
        kitDir: req.kitDir,
        options,
        items,
        target,
        hosts: hosts.map((h) => ({ label: h.hostLabel, dollarsPerHour: h.dollarsPerHour })),
        ...(batchAdapters(first.adapters) ? { adapters: batchAdapters(first.adapters) } : {})
      }
      return this.begin(run, target, hosts, 0)
    } finally {
      this.starting = false
    }
  }

  async resume(runId: string, rawTarget: BatchTarget): Promise<BatchRun> {
    this.guardStart()
    // Before the first await: a second resume (or a start) must find the pool taken.
    this.starting = true
    try {
      const saved = await this.deps.store.get(runId)
      if (!saved) throw new Error('That batch run is gone.')
      const items = saved.items.map((i): BatchItem => (i.status === 'running' ? { ...i, status: 'pending', startedAt: undefined } : i))
      if (!items.some((i) => i.status === 'pending')) throw new Error('Every task of this run has been tried.')
      if (!saved.modelId) throw new Error('This run has no model recorded.')
      const target = cleanBatchTarget(rawTarget)
      const hosts = await this.prepare(target, saved.modelId)
      if (this.disposed) throw new Error('RulZ LLM Suite is closing.')
      // Results already recorded were made with the run's adapters; resuming with others would mix them.
      if (!sameBatchAdapters(saved.adapters, hosts[0].adapters)) {
        const used = adapterLabel(saved.adapters) || 'no adapters'
        throw new Error(`This run used other adapters (${used}). Load the model with the same adapters, or start a new run.`)
      }
      const run: BatchRun = {
        ...saved,
        status: 'running',
        finishedAt: undefined,
        items,
        target,
        hosts: hosts.map((h) => ({ label: h.hostLabel, dollarsPerHour: h.dollarsPerHour })),
        modelName: hosts[0].modelName,
        contextLength: hosts[0].contextLength
      }
      return this.begin(run, target, hosts, saved.cost ?? 0)
    } finally {
      this.starting = false
    }
  }

  private guardStart(): void {
    if (this.disposed) throw new Error('RulZ LLM Suite is closing.')
    if (this.busy) throw new Error('A batch run is already running.')
  }

  /** Makes the model ready on the target's hosts and checks they agree. */
  private async prepare(target: BatchTarget, modelId: string): Promise<PoolHost[]> {
    const why = this.deps.busy(target.hosts)
    if (why) throw new Error(why)
    const n = target.sessionsPerHost
    const load =
      n === 1 && target.contextPerSession === null ? null : { parallel: n, contextLength: (target.contextPerSession ?? RE_MIN_CONTEXT) * n }
    const hosts = await this.deps.prepareHosts(target, modelId, load)
    if (!hosts.length) throw new Error('No model is ready to run the tasks.')
    for (const h of hosts) {
      if (!sameBatchAdapters(hosts[0].adapters, h.adapters)) {
        throw new Error(`The hosts have different adapters loaded (${h.hostLabel}), so their results would not be comparable.`)
      }
    }
    return hosts
  }

  private begin(run: BatchRun, target: BatchTarget, hosts: PoolHost[], baseCost: number): BatchRun {
    // One scope per lane, opened (not yet started) here so a run that cannot work fails before it exists.
    const lanes: Lane[] = []
    const releases: (() => void)[] = []
    const alone = hosts.length * target.sessionsPerHost === 1
    try {
      for (const [hostIndex, host] of hosts.entries()) {
        releases.push(this.deps.useHost(host.instanceId))
        for (let slot = 0; slot < target.sessionsPerHost; slot++) {
          lanes.push({ n: lanes.length, host, hostIndex, scope: this.deps.openScope({ alone }), ended: false })
        }
      }
    } catch (e) {
      for (const l of lanes) void l.scope.close().catch(() => undefined)
      for (const r of releases) r()
      throw e
    }
    this.lanes = lanes
    this.run = run
    this.runEnded = false
    this.stopMode = null
    this.baseCost = baseCost
    const t = this.deps.now()
    this.hostTimes = hosts.map((_h, i) => ({ active: lanes.filter((l) => l.hostIndex === i).length, since: t, accrued: 0, release: releases[i] }))
    this.hostLabels = hosts.map((h) => h.hostLabel).join(', ')
    this.emit()
    this.saveSoon(true)
    this.finishedPromise = this.runLanes().catch((e) => this.deps.warn?.(`The batch run stopped: ${errorText(e)}`))
    return this.run
  }

  // ---------- the run ----------

  private async runLanes(): Promise<void> {
    await Promise.all(this.lanes.map((lane) => this.laneLoop(lane)))
    await this.finish()
  }

  /** Estimated rental cost so far: every host counted once, for as long as any of its lanes ran. */
  private cost(): number {
    const t = this.deps.now()
    let total = this.baseCost
    this.hostTimes.forEach((h, i) => {
      const dph = this.lanes.find((l) => l.hostIndex === i)?.host.dollarsPerHour ?? 0
      const ms = h.accrued + (h.active > 0 ? t - h.since : 0)
      total += (dph * ms) / 3_600_000
    })
    return total
  }

  private laneEnded(lane: Lane): void {
    if (lane.ended) return
    lane.ended = true
    const h = this.hostTimes[lane.hostIndex]
    h.active--
    if (h.active <= 0) {
      h.accrued += this.deps.now() - h.since
      h.release()
    }
    void lane.scope.close().catch(() => undefined)
  }

  private async laneLoop(lane: Lane): Promise<void> {
    try {
      // Lanes start apart: the first analysis in REA is heavy on the CPU.
      const wait = lane.n * (this.deps.staggerMs ?? LANE_STAGGER_MS)
      if (wait > 0) await this.deps.sleep(wait)
      while (!this.stopMode && !this.disposed) {
        const state = await this.waitForHost(lane)
        if (state !== 'ready') {
          this.deps.warn?.(`${lane.host.hostLabel} is not reachable; its lane ends.`)
          break
        }
        const index = this.claim(lane)
        if (index < 0) break
        const keepGoing = await this.runItem(lane, index)
        if (!keepGoing) break
      }
    } catch (e) {
      this.deps.warn?.(`A lane of the batch run failed: ${errorText(e)}`)
    } finally {
      this.laneEnded(lane)
    }
  }

  /** The state of the lane's instance; waits while it reconnects (up to the same 3 minutes a session waits). */
  private async waitForHost(lane: Lane): Promise<InstanceState> {
    const since = this.deps.now()
    for (;;) {
      const s = this.deps.instanceState(lane.host.instanceId)
      if (s !== 'reconnecting' || this.stopMode || this.deps.now() - since >= 180_000) return s
      await this.deps.sleep(1000)
    }
  }

  /** Takes the next pending item and marks it running, in one step: no other lane can take it too. */
  private claim(lane: Lane): number {
    const run = this.run
    if (!run) return -1
    const index = run.items.findIndex((i) => i.status === 'pending')
    if (index < 0) return -1
    this.patchItem(index, { status: 'running', startedAt: this.deps.now(), error: undefined, host: lane.host.hostLabel })
    return index
  }

  private patchItem(index: number, patch: Partial<BatchItem>): void {
    const run = this.run
    // Nothing changes a run that has ended (or that the app closing already saved).
    if (!run || this.runEnded) return
    this.run = { ...run, items: run.items.map((it, i) => (i === index ? { ...it, ...patch } : it)) }
    this.emit()
    this.saveSoon(false)
  }

  /** Runs one task to its end and records what became of it. False when the lane is over. */
  private async runItem(lane: Lane, index: number): Promise<boolean> {
    const run = this.run!
    const item = run.items[index]
    const startedAt = item.startedAt ?? this.deps.now()
    let prepared: Awaited<ReturnType<PoolDeps['prepare']>>
    try {
      prepared = await this.deps.prepare(run.kitDir, item.taskId)
    } catch (e) {
      this.patchItem(index, { status: 'error', error: errorText(e), ms: this.deps.now() - startedAt })
      return true
    }
    if (this.stopMode) {
      // Asked to stop while the task was being prepared: it has not been tried.
      this.patchItem(index, { status: 'pending', startedAt: undefined })
      return false
    }
    const conversationId = this.deps.uid()
    this.own(conversationId)
    this.patchItem(index, { conversationId })
    const ctrl = new AbortController()
    this.running.set(conversationId, { ctrl, itemIndex: index })
    lane.scope.beginTask()
    let result: SessionResult
    try {
      const runner = this.deps.runSession ?? runSession
      result = await runner(this.sessionDeps(lane), {
        conversationId,
        target: prepared.target,
        goal: prepared.prompt,
        modelId: run.modelId,
        instanceId: lane.host.instanceId,
        practice: { taskId: item.taskId, family: prepared.family, tier: prepared.tier, kitDir: run.kitDir },
        minutesPerTask: run.options.minutesPerTask,
        collect: this.deps.settings().re.collect ?? true,
        signal: ctrl.signal
      } satisfies SessionSpec)
    } catch (e) {
      this.running.delete(conversationId)
      this.release(conversationId)
      this.patchItem(index, { status: 'error', error: errorText(e), ms: this.deps.now() - startedAt })
      return true
    }
    this.running.delete(conversationId)
    // Its last save has been written (the session waits for it), so the window may have the chat back.
    this.release(conversationId)

    if (result.aborted === 'stopNow' && !result.timedOut) {
      // "Stop now" cut it off: it has not had its try, so a resumed run starts it again (in a new chat; the
      // cut-off one stays in the chat list).
      this.patchItem(index, { status: 'pending', startedAt: undefined, conversationId: undefined })
      return false
    }
    if (result.lost) {
      const lostCount = (this.run?.items[index]?.lost ?? 0) + 1
      if (lostCount >= MAX_LOST_PER_ITEM) {
        this.patchItem(index, {
          status: 'error',
          error: `Not counted: its instance was lost ${lostCount} times (${lane.host.hostLabel}).`,
          lost: lostCount,
          ms: this.deps.now() - startedAt
        })
      } else {
        this.patchItem(index, { status: 'pending', startedAt: undefined, conversationId: undefined, lost: lostCount })
      }
      // The lane goes on only if its instance came back.
      return this.deps.instanceState(lane.host.instanceId) === 'ready'
    }
    if (result.saveError) {
      // The chat is not on disk, so a grade recorded here would point at nothing.
      this.patchItem(index, { status: 'error', error: `The chat could not be saved: ${result.saveError}`, ms: this.deps.now() - (result.replyAt ?? startedAt) })
      return true
    }
    const conv = result.conversation
    this.patchItem(index, {
      ...itemResult(conv, result.end, { timedOut: result.timedOut }),
      ms: this.deps.now() - (result.replyAt ?? startedAt),
      ...sessionStats(conv)
    })
    return true
  }

  private sessionDeps(lane: Lane): SessionDeps {
    const d = this.deps
    return {
      now: () => d.now(),
      sleep: (ms) => d.sleep(ms),
      uid: () => d.uid(),
      settings: () => d.settings(),
      servers: () => d.servers(),
      instance: (id) => d.instance(id),
      instanceState: (id) => d.instanceState(id),
      chat: d.chat,
      callTool: (call, signal) => lane.scope.callTool(call, signal),
      grade: (kitDir, taskId, target, text) => d.grade(kitDir, taskId, target, text),
      save: async (conv) => {
        const summary = await d.conversations.savePool(conv)
        d.events.conversationSaved(summary)
      },
      onLive: (conv) => d.events.chatLive(conv),
      watched: (id) => id === this.watchedId,
      warn: (m) => d.warn?.(m)
    }
  }

  private async finish(): Promise<void> {
    const run = this.run
    // dispose() has saved the run already.
    if (!run || this.disposed) return
    // Held until the last save is written (the window may already be told the run ended): a new run cannot
    // begin in between and be damaged by this one's clean-up.
    this.settling = true
    try {
      await this.finishRun(run, this.lanes)
    } finally {
      this.settling = false
    }
  }

  private async finishRun(run: BatchRun, lanes: Lane[]): Promise<void> {
    if (this.saveTimer) clearTimeout(this.saveTimer)
    this.saveTimer = null
    // A task cut off by a failure has not been tried.
    const items = run.items.map((i): BatchItem => (i.status === 'running' ? { ...i, status: 'pending', startedAt: undefined } : i))
    const cost = this.cost()
    const ended: BatchRun = {
      ...run,
      items,
      status: items.some((i) => i.status === 'pending') ? 'stopped' : 'done',
      finishedAt: this.deps.now(),
      ...(cost > 0 ? { cost: Math.round(cost * 10000) / 10000 } : {})
    }
    this.run = ended
    const rentalIds = [...new Set(lanes.flatMap((l) => (l.host.rentalId !== undefined ? [l.host.rentalId] : [])))]
    const stopAfter = ended.target?.stopRentalsAfter === true
    // The run is over before its last save is written, so the window never sees it still running.
    this.runEnded = true
    this.stopMode = null
    this.emit()
    try {
      await this.writeRun()
    } finally {
      for (const l of lanes) {
        if (!l.ended) this.laneEnded(l)
      }
    }
    try {
      this.deps.onEnded(ended, stopAfter ? rentalIds : [])
    } catch (e) {
      this.deps.warn?.(`After the run: ${errorText(e)}`)
    }
  }

  // ---------- stopping ----------

  /** 'after': let the tasks in progress finish. 'now': stop them too (they go back to pending). */
  stop(mode: BatchStop): void {
    if (this.runEnded) return
    // 'now' outranks 'after'; asking for 'after' later does not take 'now' back.
    this.stopMode = this.stopMode === 'now' ? 'now' : mode
    if (this.stopMode === 'now') for (const r of this.running.values()) r.ctrl.abort('stopNow')
    this.emit()
  }

  /** Stops one session; its task is marked stopped and the lane takes the next. */
  skip(conversationId: string): void {
    this.running.get(conversationId)?.ctrl.abort('skip')
  }

  // ---------- watching ----------

  /** The window shows this chat; it gets live updates. Replies with the chat as saved (null = none). */
  async watch(conversationId: string | null): Promise<Conversation | null> {
    this.watchedId = conversationId
    if (!conversationId) return null
    return this.deps.conversations.get(conversationId).catch(() => null)
  }

  private own(id: string): void {
    this.deps.conversations.own(id)
    this.owned.add(id)
    this.deps.events.poolChats([...this.owned])
  }

  private release(id: string): void {
    this.deps.conversations.release(id)
    this.owned.delete(id)
    this.deps.events.poolChats([...this.owned])
  }

  // ---------- saving and telling the window ----------

  private emit(): void {
    if (this.run) this.deps.events.batchChanged(this.run, !this.runEnded, this.stopMode)
  }

  /** Saves the run now or soon: at most once a second, and `now` skips the wait. */
  private saveSoon(now: boolean): void {
    if (this.saveTimer) return
    const wait = now ? 0 : Math.max(0, this.lastSaveAt + SAVE_INTERVAL_MS - this.deps.now())
    if (wait <= 0) {
      void this.writeRun()
      return
    }
    this.saveTimer = setTimeout(() => {
      this.saveTimer = null
      void this.writeRun()
    }, wait)
    this.saveTimer.unref?.()
  }

  private writeRun(): Promise<void> {
    const run = this.run
    if (!run) return Promise.resolve()
    this.lastSaveAt = this.deps.now()
    const cost = this.runEnded ? run.cost : this.cost()
    const copy: BatchRun = cost && cost > 0 ? { ...run, cost: Math.round(cost * 10000) / 10000 } : run
    this.saveChain = this.saveChain.then(() => this.deps.store.save(copy)).catch((e) => this.deps.warn?.(`Could not save the batch run: ${errorText(e)}`))
    return this.saveChain
  }

  // ---------- app quit ----------

  /** The app is closing: every session is stopped, running tasks go back to pending, and the run is saved. */
  async dispose(): Promise<void> {
    this.disposed = true
    if (this.runEnded || !this.run) return
    for (const r of this.running.values()) r.ctrl.abort('stopNow')
    if (this.saveTimer) clearTimeout(this.saveTimer)
    this.saveTimer = null
    const run = this.run
    const items = run.items.map((i): BatchItem => (i.status === 'running' ? { ...i, status: 'pending', startedAt: undefined } : i))
    this.run = { ...run, items, status: 'stopped', finishedAt: this.deps.now(), ...(this.cost() > 0 ? { cost: Math.round(this.cost() * 10000) / 10000 } : {}) }
    this.runEnded = true
    await this.writeRun()
  }

  /** Labels of the hosts of the run, for messages. */
  get hosts(): string {
    return this.hostLabels
  }
}
