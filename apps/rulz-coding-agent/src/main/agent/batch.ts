// Practice batch runs: the saved runs (scoreboard file) and the loop that runs the tasks one after
// another in the open project with nobody watching. It lives in the main process, so it carries on
// when the phone disconnects or the window reloads. Everything it touches comes in through
// BatchDeps, so tests run it without the app.
import { randomUUID } from 'node:crypto'
import { resolve } from 'node:path'
import { cleanBatchOptions, planBatch, settleRun } from '@shared/batch'
import type { AgentEvent, BatchItem, BatchOptions, BatchRun, BatchState, ReOverview, TaskRecord } from '@shared/types'
import { readJson, writeJsonAtomic } from '../util/jsonFile'
import type { ReSessionInfo } from '../rulz/client'
import { practiceTask } from './reTasks'
import type { ProjectSession } from './session'

export const BATCH_ACTIVE_MESSAGE = 'A batch run is going. Stop it first.'

/** Older runs beyond this are dropped when a new one is saved. */
export const MAX_BATCH_RUNS = 200

const isRun = (r: unknown): r is BatchRun => {
  const x = r as Partial<BatchRun> | null
  return !!x && typeof x.id === 'string' && typeof x.createdAt === 'number' && Array.isArray(x.items) && !!x.options
}

/** Saved runs, newest first, in one JSON file. Writes are queued so two saves cannot overwrite each other. */
export class BatchStore {
  private runs: BatchRun[] = []
  private loaded: Promise<void> | null = null
  private queue: Promise<unknown> = Promise.resolve()

  constructor(
    private readonly path: string,
    private readonly maxRuns = MAX_BATCH_RUNS
  ) {}

  /** Reads the file once. Nothing is running when it is first read: a run still marked so was cut off. */
  load(): Promise<void> {
    this.loaded ??= readJson<unknown>(this.path, []).then((raw) => {
      const list = Array.isArray(raw) ? raw.filter(isRun) : []
      this.runs = list.map(settleRun).sort((a, b) => b.createdAt - a.createdAt)
    })
    return this.loaded
  }

  /** The runs as they are now (call load() first). */
  list(): BatchRun[] {
    return this.runs
  }

  get(id: string): BatchRun | undefined {
    return this.runs.find((r) => r.id === id)
  }

  /** Adds or replaces a run in memory at once and saves it; the promise ends when it is on disk. */
  put(run: BatchRun): Promise<void> {
    const runs = this.runs.filter((r) => r.id !== run.id)
    runs.push(run)
    runs.sort((a, b) => b.createdAt - a.createdAt)
    this.runs = runs.slice(0, this.maxRuns)
    return this.write()
  }

  remove(id: string): Promise<void> {
    this.runs = this.runs.filter((r) => r.id !== id)
    return this.write()
  }

  private write(): Promise<void> {
    const next = this.queue.catch(() => undefined).then(() => writeJsonAtomic(this.path, this.runs))
    this.queue = next
    return next
  }

  /** Waits for queued saves (tests, quitting). */
  flush(): Promise<void> {
    return this.queue.then(
      () => undefined,
      () => undefined
    )
  }
}

export interface BatchDeps {
  client: {
    reOverview(): Promise<ReOverview>
    preparePractice(taskId: string): Promise<{ target: string; prompt: string; family: string; tier: string }>
    describeReSession(id: string, info: ReSessionInfo): Promise<boolean>
  }
  store: BatchStore
  /** The open project's session, if any. */
  session: () => ProjectSession | null
  /** The model the agent uses now. */
  modelName: () => string
  broadcast: (state: BatchState) => void
  /** Milliseconds in a "minute per task"; tests make it small. */
  msPerMinute?: number
  now?: () => number
}

const errorText = (e: unknown): string => (e instanceof Error ? e.message : String(e))

export class BatchRunner {
  private activeId: string | null = null
  private stopping: 'after' | 'now' | null = null
  /** Read fresh after an await: a stop can be asked for at any moment. */
  private stopNowAsked(): boolean {
    return this.stopping === 'now'
  }
  /** Called by onAgentEvent when the task in progress ends. */
  private waiter: ((task: TaskRecord) => void) | null = null
  /** A task of this run is running in the session now (so Stop now has something to stop). */
  private inTask = false
  private timedOut = false
  private loop: Promise<void> = Promise.resolve()

  constructor(private readonly deps: BatchDeps) {}

  private now = (): number => (this.deps.now ?? Date.now)()

  get active(): boolean {
    return this.activeId !== null
  }

  /** For the handlers that start a task or change the project: refuses while a run is going. */
  assertIdle(): void {
    if (this.activeId) throw new Error(BATCH_ACTIVE_MESSAGE)
  }

  async state(): Promise<BatchState> {
    await this.deps.store.load()
    return this.snapshot()
  }

  private snapshot(): BatchState {
    return { runs: this.deps.store.list(), activeId: this.activeId, stopping: this.stopping }
  }

  private changed(): BatchState {
    const s = this.snapshot()
    this.deps.broadcast(s)
    return s
  }

  /** Waits until the run loop has ended (tests). */
  idle(): Promise<void> {
    return this.loop
  }

  /** Waits for the saved runs to be written. */
  flush(): Promise<void> {
    return this.deps.store.flush()
  }

  /** The index.ts hook: tells the runner a task of the session ended. */
  onAgentEvent(e: AgentEvent): void {
    if (e.type === 'task-end') this.waiter?.(e.task)
  }

  async start(projectPath: string, label: string, options: BatchOptions, record: boolean): Promise<BatchState> {
    await this.deps.store.load()
    if (this.activeId) throw new Error('A batch run is already going.')
    const session = this.deps.session()
    if (!session || resolve(session.root).toLowerCase() !== resolve(projectPath).toLowerCase()) throw new Error('That project is not open.')
    if (session.isRunning) throw new Error('A task is already running. Wait for it to finish or press Stop.')
    const opts = cleanBatchOptions(options)
    const overview = await this.deps.client.reOverview()
    const tasks = overview.practice.tasks
    const items = planBatch(tasks, opts, new Set(tasks.filter((t) => t.solved > 0).map((t) => t.id)))
    if (!items.length) throw new Error('No tasks match.')
    // The overview took a moment: check again that nothing else got in.
    if (this.activeId) throw new Error('A batch run is already going.')
    if (session.isRunning || this.deps.session() !== session) throw new Error('A task is already running. Wait for it to finish or press Stop.')
    const run: BatchRun = {
      id: randomUUID(),
      label: label.trim().slice(0, 80),
      createdAt: this.now(),
      status: 'running',
      projectPath: session.root,
      modelName: this.deps.modelName(),
      kitDir: overview.practice.kitDir,
      record,
      options: opts,
      items
    }
    this.activeId = run.id
    this.stopping = null
    void this.save(run)
    const state = this.changed()
    this.loop = this.runLoop(run.id)
    return state
  }

  stop(mode: 'after' | 'now'): BatchState {
    if (!this.activeId) return this.snapshot()
    // "Stop now" outranks "after", never the other way round.
    if (this.stopping !== 'now') this.stopping = mode
    if (mode === 'now' && this.inTask) this.deps.session()?.stop()
    return this.changed()
  }

  async delete(id: string): Promise<BatchState> {
    await this.deps.store.load()
    if (id === this.activeId) throw new Error('That run is going now. Stop it first.')
    await this.deps.store.remove(id)
    return this.changed()
  }

  private save(run: BatchRun): Promise<void> {
    return this.deps.store.put(run).catch((e) => console.error('[batch] could not save the run:', e))
  }

  private patchRun(fn: (r: BatchRun) => BatchRun): BatchRun | undefined {
    const run = this.activeId ? this.deps.store.get(this.activeId) : undefined
    if (!run) return undefined
    const next = fn(run)
    void this.save(next)
    this.changed()
    return next
  }

  private patchItem(index: number, patch: Partial<BatchItem>): BatchRun | undefined {
    return this.patchRun((r) => ({ ...r, items: r.items.map((it, i) => (i === index ? { ...it, ...patch } : it)) }))
  }

  private async runLoop(id: string): Promise<void> {
    try {
      for (;;) {
        if (this.stopping) break
        const run = this.deps.store.get(id)
        if (!run) break
        const index = run.items.findIndex((i) => i.status === 'pending')
        if (index < 0) break
        await this.runItem(index)
      }
    } catch (e) {
      // Not expected (runItem catches its own failures); the run ends stopped, with what it has.
      this.patchRun((r) => ({ ...r, items: r.items.map((i) => (i.status === 'running' ? { ...i, status: 'error', error: errorText(e) } : i)) }))
    } finally {
      this.waiter = null
      this.inTask = false
      this.patchRun((r) => {
        // A task cut off above has not been tried.
        const items = r.items.map((i): BatchItem => (i.status === 'running' ? { ...i, status: 'pending', startedAt: undefined } : i))
        return { ...r, items, status: items.some((i) => i.status === 'pending') ? 'stopped' : 'done', finishedAt: this.now() }
      })
      this.activeId = null
      this.stopping = null
      this.changed()
    }
  }

  /** Runs one task to its end and records what became of it. */
  private async runItem(index: number): Promise<void> {
    const run = this.deps.store.get(this.activeId!)!
    const item = run.items[index]
    const startedAt = this.now()
    this.patchItem(index, { status: 'running', startedAt, error: undefined })
    const fail = (error: string): void => void this.patchItem(index, { status: 'error', error, ms: this.now() - startedAt })
    const session = this.deps.session()
    if (!session || resolve(session.root).toLowerCase() !== resolve(run.projectPath).toLowerCase()) return fail('The project was closed.')

    let started: ReturnType<typeof practiceTask>
    try {
      const prep = await this.deps.client.preparePractice(item.taskId)
      started = practiceTask(prep, item.taskId, run.record)
    } catch (e) {
      return fail(errorText(e))
    }
    if (this.stopping === 'now') {
      // "Stop now" while the task was being prepared: it has not been tried. ("After" lets it have its try.)
      this.patchItem(index, { status: 'pending', startedAt: undefined })
      return
    }

    this.timedOut = false
    const ended = new Promise<TaskRecord>((resolveEnd) => (this.waiter = resolveEnd))
    let timer: NodeJS.Timeout | undefined
    try {
      session.start(started.text, { shown: started.shown, re: started.info, batch: true })
    } catch (e) {
      this.waiter = null
      return fail(errorText(e))
    }
    this.inTask = true
    const taskRecord = session.data.tasks.at(-1)!
    this.patchItem(index, { taskRecordId: taskRecord.id })
    const minutes = run.options.minutesPerTask
    if (minutes > 0) {
      timer = setTimeout(() => {
        this.timedOut = true
        session.stop()
      }, minutes * (this.deps.msPerMinute ?? 60_000))
    }
    // "Stop now" may have been asked for between the check above and the task's start.
    if (this.stopNowAsked()) session.stop()
    const task = await ended
    clearTimeout(timer)
    this.waiter = null
    this.inTask = false
    const ms = this.now() - startedAt
    const timedOut = this.timedOut
    const stopNow = this.stopNowAsked()

    const p = task.re?.practice
    const graded = task.outcome === 'done' && p?.passed !== undefined
    const rounds = toolRounds(session, task)
    const common = { ms, toolRounds: rounds, ...(p?.answer ? { answer: p.answer } : {}), ...(p?.expected ? { expected: p.expected } : {}) }

    if (!graded && stopNow) {
      // "Stop now" cut it off: it has not had its try, so a resumed run starts it again.
      this.patchItem(index, { status: 'pending', startedAt: undefined, taskRecordId: undefined })
      return
    }
    if (!graded && timedOut) {
      // A stopped task is not graded, so tell the Suite what became of its recorded session.
      if (run.record) {
        await this.deps.client
          .describeReSession(`agent-${task.id}`, {
            outcome: 'failed',
            notes: `Batch run: stopped at the ${minutes}-minute limit.`,
            practice: { taskId: item.taskId, family: item.family, tier: item.tier }
          })
          .catch(() => undefined)
      }
      this.patchItem(index, { status: 'timeout', ms, toolRounds: rounds })
      return
    }
    if (task.outcome === 'stopped') {
      // The owner pressed Stop on this task: it does not count, and the run goes on with the next one.
      this.patchItem(index, { status: 'stopped', ms, toolRounds: rounds })
      return
    }
    if (graded) {
      const status = p!.passed ? 'solved' : p!.answer ? 'wrong' : 'no-answer'
      this.patchItem(index, { status, ...common })
      return
    }
    this.patchItem(index, { status: 'error', ...common, error: lastProblem(session, task) })
  }
}

/** Model replies that called tools, in the task's part of the conversation. */
function toolRounds(session: ProjectSession, task: TaskRecord): number {
  const msgs = session.data.messages
  let start = -1
  for (let i = msgs.length - 1; i >= 0; i--) {
    if (msgs[i].role === 'user' && msgs[i].task === task.id) {
      start = i
      break
    }
  }
  if (start < 0) return 0
  return msgs.slice(start + 1).filter((m) => m.role === 'assistant' && m.tool_calls?.length).length
}

/** Why a task ended without a grade: the last error (or warning) the conversation shows for it. */
function lastProblem(session: ProjectSession, task: TaskRecord): string {
  const notices = session.data.timeline.filter((i) => i.kind === 'notice' && i.taskId === task.id)
  const pick = [...notices].reverse().find((i) => i.kind === 'notice' && i.tone === 'error') ?? [...notices].reverse().find((i) => i.kind === 'notice' && i.tone === 'warn')
  const text = pick?.kind === 'notice' ? pick.text : ''
  return (text || (task.outcome === 'error' ? 'The task failed.' : 'The answer was not graded.')).replace(/\s+/g, ' ').slice(0, 500)
}
