// One open project: its conversation with the model, the agent loop, approvals, undo and diffs.
import { randomUUID } from 'node:crypto'
import { promises as fs } from 'node:fs'
import type {
  AgentEvent,
  AgentStatus,
  AppSettings,
  ChangedFile,
  ContextUsage,
  DistributiveOmit,
  FileDiff,
  ModelActivity,
  ModelInfo,
  PendingPrompt,
  ProjectState,
  PromptAnswer,
  Step,
  TaskRecord,
  TimelineItem,
  WireMessage,
  WireToolCall
} from '@shared/types'
import { DEFAULT_MAX_OUTPUT_TOKENS, MAX_INSTRUCTION_CHARS, samplingRequest } from '@shared/defaults'
import { splitThink } from '@shared/think'
import { friendlyServerError, pickDefaultModel, type RulzClient, type TurnResult } from '../rulz/client'
import { addDraft } from '../rulz/draft'
import type { Snapshot, Snapshotter } from '../project/snapshots'
import type { ProjectData, ProjectStore } from '../project/store'
import { BackgroundCommands, commandHints, commandReport, headTail, runCommand } from './commands'
import {
  CHARS_PER_TOKEN,
  CONTEXT_SHARE,
  estimateTokens,
  fitMessages,
  mechanicalSummary,
  SMALL_CONTEXT,
  SUMMARY_PROMPT,
  summaryRange,
  tokenScale,
  totalTokens,
  transcript
} from './context'
import * as files from './fileTools'
import type { ToolOutcome } from './fileTools'
import { LoopGuard, type GuardTrip } from './loopGuard'
import { formatPageReport, pageHasProblems, pageTarget, type PageReport, type PageTarget } from './pageCheck'
import { resolveInProject } from './pathGuard'
import { HOW_TO_RUN_TASK, parseRunLine, parseShareLine, systemPrompt } from './prompt'
import { approvalKey, classifyCommand, needsApproval, type ActionKind } from './safety'
import { describeStep, EXTRA_TOOL_DEFS, EXTRA_TOOL_NAMES, parseToolArgs, serverLabel, TOOL_DEFS, TOOL_NAMES } from './tools'
import { toolSummary } from '../mcp/config'
import type { ToolInfo, ToolServers } from '../mcp/toolServers'

/** Loaded tool-server tools may use at most this share of the context; the oldest are unloaded first. */
const LOADED_TOOLS_SHARE = 0.15
const DEFAULT_CTX = 32768
const MAX_STEP_OUTPUT_LIVE = 40_000
const MAX_DIFF_BYTES = 400 * 1024
/** How many times a reply may come back as plain-text tool calls before it counts as stuck. */
const MAX_TEXT_CALLS = 3

export interface SessionDeps {
  client: RulzClient
  settings: () => AppSettings
  store: ProjectStore
  emit: (e: AgentEvent) => void
  activity: (a: ModelActivity) => void
  /** Last known model list (to tell whether the model is loaded). */
  models: () => ModelInfo[]
  refreshModels: () => void
  snapshotter: (kind: 'git' | 'copy' | 'auto') => Promise<Snapshotter>
  trash: (abs: string) => Promise<void>
  /** Opens a page in a hidden window (check_page); missing where there is no Electron. */
  checkPage?: (target: PageTarget, waitMs: number, signal: AbortSignal) => Promise<PageReport>
  /** Tool servers set up in RulZ LLM Suite (such as REA). */
  toolServers?: Pick<ToolServers, 'configured' | 'start' | 'status' | 'find' | 'search' | 'instructions' | 'call'>
}

/** Ends the task without counting as an error (owner chose Stop in a prompt, or pressed Stop). */
class StopTask extends Error {}

/** An error whose message is already written for the owner. */
class FriendlyError extends Error {}

function cleanContent(text: string): string {
  return splitThink(text).answer
}

function stepDetail(name: string, args: Record<string, unknown>): string | undefined {
  // Tool-server tools: the full name and arguments.
  if (name.includes('__')) return `${name} ${JSON.stringify(args)}`.slice(0, 400)
  if (typeof args.command === 'string') return args.command
  if (typeof args.page === 'string') return args.page
  if (typeof args.path === 'string') return args.path
  if (typeof args.pattern === 'string') return `${args.pattern}${args.path ? ` in ${String(args.path)}` : ''}`
  if (typeof args.query === 'string') return `"${args.query}"${args.glob ? ` in ${String(args.glob)}` : ''}`
  if (typeof args.id === 'string') return args.id
  return name === 'list_files' ? '(whole project)' : undefined
}

export class ProjectSession {
  readonly background = new BackgroundCommands()
  private running = false
  private abort: AbortController | null = null
  private pending: { request: PendingPrompt; resolve: (a: PromptAnswer | null) => void } | null = null
  private allowed = new Set<string>()
  /** Index in data.messages of the current task's first message. */
  private taskStart = 0
  private saveTimer: NodeJS.Timeout | null = null
  private statusNow: AgentStatus = { phase: 'idle' }
  /** Real tokens per estimated token, learnt from the server's counts (see tokenScale). */
  private scale = 1
  private warnedSmallContext = false
  /** Tool servers set up in RulZ LLM Suite, read at the start of each task. */
  private extraServers: string[] = []
  /** Tool-server tools the model has loaded, oldest first. */
  private loaded = new Map<string, ToolInfo>()
  private nCtx = DEFAULT_CTX

  constructor(
    public data: ProjectData,
    private readonly deps: SessionDeps
  ) {}

  get root(): string {
    return this.data.path
  }

  get isRunning(): boolean {
    return this.running
  }

  state(name: string, lastOpened: number): ProjectState {
    return {
      project: { path: this.root, name, lastOpened },
      timeline: this.data.timeline,
      tasks: this.data.tasks,
      running: this.running,
      prompt: this.pending?.request ?? null,
      runCommand: this.data.runCommand,
      memory: this.data.memory,
      context: this.data.context
    }
  }

  /** Saves the owner's "Things to always remember"; the next request includes them. */
  async setMemory(text: string): Promise<void> {
    this.data.memory = text.trim().slice(0, MAX_INSTRUCTION_CHARS)
    await this.saveNow()
  }

  get status(): AgentStatus {
    return this.statusNow
  }

  // ---------- events ----------

  private emit(e: DistributiveOmit<AgentEvent, 'projectPath'>): void {
    this.deps.emit({ ...e, projectPath: this.root } as AgentEvent)
  }

  private setStatus(status: AgentStatus): void {
    this.statusNow = status
    this.emit({ type: 'status', status })
  }

  private addItem(item: TimelineItem): void {
    this.data.timeline.push(item)
    this.emit({ type: 'item', item })
    this.scheduleSave()
  }

  private notice(text: string, tone: 'info' | 'ok' | 'warn' | 'error', taskId?: string, reveal?: string): void {
    this.addItem({ id: randomUUID(), kind: 'notice', text, tone, taskId, ...(reveal ? { reveal } : {}) })
  }

  private addStep(task: TaskRecord, step: Omit<Step, 'id' | 'taskId'>): Step {
    const full: Step = { ...step, id: randomUUID(), taskId: task.id }
    task.steps.push(full)
    this.emit({ type: 'step', step: full })
    return full
  }

  private updateStep(step: Step, patch: Partial<Step>): void {
    Object.assign(step, patch)
    this.emit({ type: 'step', step: { ...step } })
    this.scheduleSave()
  }

  private scheduleSave(): void {
    if (this.saveTimer) return
    this.saveTimer = setTimeout(() => {
      this.saveTimer = null
      void this.deps.store.save(this.data).catch(() => undefined)
    }, 1500)
  }

  async saveNow(): Promise<void> {
    if (this.saveTimer) clearTimeout(this.saveTimer)
    this.saveTimer = null
    await this.deps.store.save(this.data)
  }

  // ---------- public actions ----------

  /** `shown` is what the conversation shows when the model gets a longer text (e.g. "Fix this"). */
  start(text: string, opts: { howToRun?: boolean; share?: boolean; shown?: string } = {}): void {
    const trimmed = text.trim()
    if (!trimmed) return
    if (this.running) throw new Error('A task is already running. Wait for it to finish or press Stop.')
    const visible = opts.shown?.trim() || trimmed
    const title = visible.split('\n')[0].slice(0, 80) + (visible.split('\n')[0].length > 80 ? '…' : '')
    const task: TaskRecord = { id: randomUUID(), title: opts.howToRun ? 'Finding out how to run the app' : title, startedAt: Date.now(), outcome: 'running', steps: [], files: [] }
    const item: TimelineItem = { id: randomUUID(), kind: 'user', text: opts.howToRun ? 'How do I run this app?' : visible, taskId: task.id }
    this.data.tasks.push(task)
    this.data.timeline.push(item)
    this.running = true
    this.abort = new AbortController()
    this.emit({ type: 'task-start', task, item })
    void this.run(task, trimmed, opts.howToRun ?? false, opts.share ?? false)
  }

  stop(): void {
    this.abort?.abort()
  }

  answer(promptId: string, answer: PromptAnswer): void {
    if (this.pending?.request.id === promptId) this.pending.resolve(answer)
  }

  async close(): Promise<void> {
    this.stop()
    await this.background.stopAll()
    await this.saveNow()
  }

  canUndo(): TaskRecord | undefined {
    return [...this.data.tasks].reverse().find((t) => !t.undone && t.snapshot && t.outcome !== 'running')
  }

  /** "Undo last task" with no id; History's "Go back to before this" with one (that task and every later one are undone). */
  async undo(taskId?: string): Promise<void> {
    if (this.running) throw new Error('Wait for the current task to finish (or press Stop) before undoing.')
    const target = taskId ? this.data.tasks.find((t) => t.id === taskId) : this.canUndo()
    if (!target) throw new Error('There is nothing to undo.')
    if (target.undone) throw new Error(`"${target.title}" is already undone.`)
    if (!target.snapshot) throw new Error(`There is no restore point from before "${target.title}", so the project can't go back to it.`)
    const tasks = this.data.tasks.slice(this.data.tasks.indexOf(target)).filter((t) => !t.undone && t.outcome !== 'running')
    // A git restore point holds the whole project, so going back to it is enough. Copied files only
    // hold what each task changed, so each later task is put back in turn, newest first.
    const restores = target.snapshot.kind === 'git' ? [target] : tasks.slice().reverse()
    const missing = restores.find((t) => !t.snapshot)
    if (missing) throw new Error(`There is no restore point from before "${missing.title}", so the project can't go back past it.`)
    // Running programs can hold files open.
    await this.background.stopAll()
    for (const t of restores) {
      const snap = await this.deps.snapshotter(t.snapshot!.kind)
      await snap.restore(t.snapshot!, t.title)
    }
    for (const t of tasks) t.undone = true
    const later = tasks.slice(1).map((t) => `"${t.title}"`)
    this.data.notes.push(
      (taskId
        ? `The owner used History to go back, so the project files are back to how they were before the task "${target.title}"` +
          (later.length ? ` (the later tasks ${later.join(', ')} were undone too)` : '')
        : `The owner clicked "Undo last task", so the project files are back to how they were before the task "${target.title}"`) +
        '. Everything those tasks created or changed is gone; check the files again before relying on earlier results.'
    )
    this.notice(
      `Undone: "${target.title}"${later.length ? ` and ${later.length} later task${later.length > 1 ? 's' : ''}` : ''}. The project is back to how it was before that task.`,
      'ok',
      target.id
    )
    this.emit({ type: 'tasks', tasks: this.data.tasks })
    await this.saveNow()
  }

  async diff(taskId: string, rel: string): Promise<FileDiff> {
    const task = this.data.tasks.find((t) => t.id === taskId)
    const target = await resolveInProject(this.root, rel)
    const out: FileDiff = { path: target.rel, before: null, after: null, binary: false, tooLarge: false }
    const decode = (buf: Buffer | null | undefined): string | null => {
      if (!buf) return null
      if (buf.length > MAX_DIFF_BYTES) out.tooLarge = true
      if (buf.subarray(0, 8000).includes(0)) out.binary = true
      return out.binary || out.tooLarge ? null : buf.toString('utf8')
    }
    if (task?.snapshot) {
      const snap = await this.deps.snapshotter(task.snapshot.kind)
      out.before = decode(await snap.fileBefore(task.snapshot, target.rel))
    }
    out.after = decode(await fs.readFile(target.abs).catch(() => null))
    return out
  }

  // ---------- the task ----------

  private async run(task: TaskRecord, text: string, howToRun: boolean, share: boolean): Promise<void> {
    const signal = this.abort!.signal
    this.allowed = new Set()
    let snap: Snapshotter | null = null
    try {
      this.setStatus({ phase: 'working', text: 'Saving a restore point…' })
      try {
        snap = await this.deps.snapshotter('auto')
        task.snapshot = await snap.begin(task.id, task.title)
      } catch (e) {
        this.notice(`I couldn't save a restore point, so "Undo last task" won't work for this task. (${(e as Error).message})`, 'warn', task.id)
      }
      if (signal.aborted) throw new StopTask()

      const model = this.resolveModel()
      const nCtx = await this.prepareModel(model, signal)
      this.nCtx = nCtx
      this.extraServers = (await this.deps.toolServers?.configured().catch(() => [])) ?? []

      const notes = this.data.notes.splice(0)
      const body = howToRun ? HOW_TO_RUN_TASK : text
      const content = notes.length ? `${notes.map((n) => `[Note: ${n}]`).join('\n')}\n\n${body}` : body
      this.taskStart = this.data.messages.length
      this.data.messages.push({ role: 'user', content })

      await this.loop(task, model, nCtx, signal, snap)
      task.outcome = 'done'
      if (howToRun) {
        const last = [...this.data.messages].reverse().find((m) => m.role === 'assistant' && m.content.trim())
        const cmd = last ? parseRunLine(last.content) : null
        if (cmd) {
          this.data.runCommand = cmd
          this.notice('Saved how to run this app. Click **Run my app** to start it.', 'ok', task.id)
        } else this.notice("I couldn't work out how to run this app.", 'warn', task.id)
      }
      if (share) await this.reportShare(task)
    } catch (e) {
      if (e instanceof StopTask || signal.aborted) {
        task.outcome = 'stopped'
        this.notice(
          `Stopped. What was done so far is kept${task.snapshot ? '; use **Undo last task** to go back to before this task' : ''}.`,
          'info',
          task.id
        )
      } else {
        task.outcome = 'error'
        this.notice(e instanceof FriendlyError ? e.message : friendlyServerError(e), 'error', task.id)
      }
    } finally {
      this.closeOpenCalls()
      for (const s of task.steps) if (s.status === 'running' || s.status === 'waiting') this.updateStep(s, { status: 'skipped' })
      task.endedAt = Date.now()
      if (snap) await this.refreshFiles(task, snap).catch(() => undefined)
      this.running = false
      this.abort = null
      this.setStatus({ phase: 'idle' })
      this.deps.activity('idle')
      this.deps.refreshModels()
      this.emit({ type: 'task-end', task })
      await this.saveNow().catch(() => undefined)
    }
  }

  /** Gives every tool call without a result a "stopped" result, so the conversation stays valid. */
  private closeOpenCalls(): void {
    const msgs = this.data.messages
    for (let i = msgs.length - 1; i >= 0; i--) {
      const m = msgs[i]
      if (m.role === 'assistant' && m.tool_calls?.length) {
        const answered = new Set(msgs.slice(i + 1).filter((x) => x.role === 'tool').map((x) => x.tool_call_id))
        const missing = m.tool_calls.filter((c) => !answered.has(c.id))
        msgs.push(...missing.map((c) => ({ role: 'tool' as const, tool_call_id: c.id, content: 'Not run: the owner stopped the task.' })))
        return
      }
      if (m.role === 'user') return
    }
  }

  private resolveModel(): string {
    const chosen = this.deps.settings().model
    const models = this.deps.models()
    if (chosen && (!models.length || models.some((m) => m.id === chosen))) return chosen
    const pick = pickDefaultModel(models)
    if (!pick) throw new FriendlyError('RulZ LLM Suite has no models yet. Add one on its My Models page, then try again.')
    return pick
  }

  /** After "Share my app": points the owner at the packaged file the agent named, if it is really there. */
  private async reportShare(task: TaskRecord): Promise<void> {
    const last = [...this.data.messages].reverse().find((m) => m.role === 'assistant' && m.content.trim())
    const rel = last ? parseShareLine(last.content) : null
    const target = rel ? await resolveInProject(this.root, rel).catch(() => null) : null
    const st = target ? await fs.stat(target.abs).catch(() => null) : null
    if (target && st?.isFile()) {
      const size = st.size >= 1024 * 1024 ? `${(st.size / 1024 / 1024).toFixed(1)} MB` : `${Math.max(1, Math.round(st.size / 1024))} KB`
      this.notice(`Your app is packaged: **${target.rel}** (${size}). This is the file to give to other people.`, 'ok', task.id, target.abs)
    } else {
      this.notice("I couldn't find a packaged file to share, so it may not have worked. Read my last message for what happened.", 'warn', task.id)
    }
  }

  /** Makes sure the model is loaded and learns its context length and chat template. */
  private async prepareModel(model: string, signal: AbortSignal): Promise<number> {
    const info = this.deps.models().find((m) => m.id === model)
    if (!info?.loaded) {
      this.setStatus({ phase: 'loading', text: 'Loading the model… (this can take a minute)' })
      this.deps.activity('loading')
    }
    let nCtx: number | undefined
    try {
      nCtx = (await this.deps.client.modelProps(model, signal)).nCtx
    } catch (e) {
      if (signal.aborted) throw e
      const status = (e as { status?: number }).status
      // A 503 means loading failed; other errors only cost us the extra details.
      if (status === 503) throw e
    }
    this.deps.activity('working')
    this.deps.refreshModels()
    const size = nCtx ?? info?.loadedContext ?? DEFAULT_CTX
    if (size <= SMALL_CONTEXT && !this.warnedSmallContext) {
      this.warnedSmallContext = true
      this.notice(
        `The model is loaded with a small working memory (${Math.round(size / 1024)}K tokens), so in bigger projects I'll have to tidy up my notes often and may forget details. ` +
          'For bigger projects, load the model in RulZ LLM Suite with a larger context length (32K or more).',
        'warn'
      )
    }
    return size
  }

  private async loop(task: TaskRecord, model: string, nCtx: number, signal: AbortSignal, snap: Snapshotter | null): Promise<void> {
    const settings = this.deps.settings()
    let limit = settings.maxSteps
    let steps = 0
    let textCalls = 0
    const guard = new LoopGuard()

    for (;;) {
      if (signal.aborted) throw new StopTask()
      const request = await this.buildRequest(model, nCtx, signal)
      const turn = await this.streamTurn(task, model, request, nCtx, signal)
      const content = cleanContent(turn.content)
      const calls = turn.toolCalls
      this.data.messages.push({ role: 'assistant', content, ...(calls.length ? { tool_calls: calls } : {}) })

      if (!calls.length) {
        // Tool calls that came through as plain text: ask for a proper call.
        if (/<tool_call>|<function=/.test(content) && ++textCalls <= MAX_TEXT_CALLS) {
          this.data.messages.push({
            role: 'user',
            content: '[Note: Your last tool call came through as plain text, so nothing ran. Call the tool again using the tool-calling format.]'
          })
          continue
        }
        if (turn.finishReason === 'length') {
          this.data.messages.push({
            role: 'user',
            content: '[Note: Your last reply was cut off because it was too long. Continue from where you stopped; write big files in smaller parts.]'
          })
          if (++textCalls <= MAX_TEXT_CALLS) continue
        }
        return
      }

      let trip: GuardTrip | null = null
      let lastTitle = ''
      for (const call of calls) {
        if (signal.aborted) break
        const { result, title } = await this.execute(task, call, signal, snap, turn)
        this.data.messages.push({ role: 'tool', tool_call_id: call.id, content: result.text })
        steps++
        lastTitle = title
        trip = guard.record(call.function.name, call.function.arguments, result.isError, result.text) ?? trip
      }
      if (signal.aborted) throw new StopTask()

      if (trip) {
        const message =
          trip === 'repeating'
            ? `I seem to be going round in circles: I've done "${lastTitle}" three times in a row. Should I keep going, try a different approach, or stop?`
            : `I've hit five problems in a row (the last one: ${guard.lastError.split('\n')[0].slice(0, 160)}). Should I keep going, try a different approach, or stop?`
        await this.handleStuck(task, message, trip, signal)
        guard.reset()
      }
      if (steps >= limit) {
        await this.handleStuck(task, `I've taken ${steps} steps on this task so far. Should I keep going?`, 'limit', signal)
        limit = steps + settings.maxSteps
      }
    }
  }

  private async handleStuck(task: TaskRecord, message: string, reason: GuardTrip | 'limit', signal: AbortSignal): Promise<void> {
    this.notice(message, 'warn', task.id)
    const answer = await this.ask({ kind: 'stuck', message, reason }, signal)
    if (!answer || answer.kind !== 'stuck' || answer.decision === 'stop') throw new StopTask()
    if (answer.decision === 'change') {
      const said = answer.text?.trim()
      if (said) this.addItem({ id: randomUUID(), kind: 'user', text: said, taskId: task.id })
      this.data.messages.push({
        role: 'user',
        content: said
          ? `[The owner paused you because you seemed stuck and says:] ${said}`
          : '[The owner paused you because you seemed stuck and asks you to try a different approach. Think about why the last attempts failed and change course.]'
      })
    }
  }

  private ask(req: DistributiveOmit<PendingPrompt, 'id'>, signal: AbortSignal): Promise<PromptAnswer | null> {
    const request = { ...req, id: randomUUID() } as PendingPrompt
    const previous = this.statusNow
    return new Promise<PromptAnswer | null>((resolve) => {
      const onAbort = (): void => resolve(null)
      signal.addEventListener('abort', onAbort, { once: true })
      this.pending = {
        request,
        resolve: (a) => {
          signal.removeEventListener('abort', onAbort)
          resolve(a)
        }
      }
      this.emit({ type: 'prompt', prompt: request })
      this.setStatus({ phase: 'waiting', text: request.kind === 'question' ? 'Waiting for your answer' : 'Waiting for you' })
    }).finally(() => {
      this.pending = null
      this.emit({ type: 'prompt', prompt: null })
      this.setStatus(previous.phase === 'idle' ? { phase: 'working', text: 'Working…' } : previous)
    })
  }

  // ---------- talking to the model ----------

  private async buildRequest(model: string, nCtx: number, signal: AbortSignal): Promise<WireMessage[]> {
    // The budget is in estimated tokens, corrected by what the server counted last time.
    const budget = Math.floor((nCtx * CONTEXT_SHARE) / this.scale)
    for (let attempt = 0; ; attempt++) {
      const system = systemPrompt({
        projectPath: this.root,
        projectName: this.root.split(/[\\/]/).pop() ?? this.root,
        date: new Date().toDateString(),
        summary: this.data.summary,
        memory: this.data.memory,
        instructions: this.deps.settings().instructions,
        background: this.background.running(),
        extraTools: this.extraServers.length
          ? { servers: this.extraServers.map(serverLabel), advice: this.deps.toolServers?.instructions().map((i) => ({ ...i, server: serverLabel(i.server) })) ?? [] }
          : undefined
      })
      const fit = fitMessages(this.data.messages, budget, estimateTokens(system) + estimateTokens(JSON.stringify(this.toolDefs())))
      const range = fit.fits || attempt >= 3 ? null : summaryRange(this.data.messages, this.taskStart)
      if (!range) return [{ role: 'system' as const, content: system }, ...fit.messages]
      await this.summarise(model, range, budget, signal)
    }
  }

  private async summarise(model: string, range: { from: number; to: number }, budget: number, signal: AbortSignal): Promise<void> {
    this.setStatus({ phase: 'thinking', text: 'Tidying up my notes to save space…' })
    const part = this.data.messages.slice(range.from, range.to)
    const earlier = this.data.summary.trim() ? `Summary of even earlier work:\n${this.data.summary}\n\nTranscript:\n` : ''
    const maxChars = Math.max(4000, Math.floor((budget - 2500) * CHARS_PER_TOKEN) - earlier.length)
    let summary: string
    try {
      summary = await this.deps.client.complete(
        model,
        [
          { role: 'system', content: SUMMARY_PROMPT },
          { role: 'user', content: earlier + transcript(part, maxChars) }
        ],
        1500,
        signal
      )
      if (!summary.trim()) throw new Error('empty summary')
    } catch (e) {
      if (signal.aborted) throw e
      summary = `${this.data.summary ? `${this.data.summary}\n` : ''}${mechanicalSummary(part)}`
    }
    this.data.summary = cleanContent(summary)
    this.data.messages.splice(range.from, range.to - range.from)
    if (range.to <= this.taskStart) this.taskStart -= range.to - range.from
  }

  private async streamTurn(task: TaskRecord, model: string, messages: WireMessage[], nCtx: number, signal: AbortSignal): Promise<TurnResult> {
    this.setStatus({ phase: 'thinking', text: 'Thinking…' })
    this.deps.activity('working')
    let item: Extract<TimelineItem, { kind: 'assistant' }> | null = null
    const tools = this.toolDefs()
    const estimated = totalTokens(messages) + estimateTokens(JSON.stringify(tools))
    const own = this.deps.settings().modelSampling[model]
    const cap = own?.maxOutputTokens ?? DEFAULT_MAX_OUTPUT_TOKENS
    const maxTokens = Math.max(Math.min(1024, cap), Math.min(cap, nCtx - Math.ceil(estimated * this.scale) - 256))
    const sampling = samplingRequest(model, own)
    const result = await this.deps.client.streamTurn({ model, messages, tools, max_tokens: maxTokens, ...sampling }, signal, {
      onProgress: (done, total) => {
        if (total > 4000) this.setStatus({ phase: 'thinking', text: `Reading the project notes… ${Math.round((done / total) * 100)}%` })
      },
      onReasoning: (text) => {
        if (!item) {
          item = { id: randomUUID(), kind: 'assistant', text: '', taskId: task.id }
          this.addItem(item)
        }
        item.reasoning = (item.reasoning ?? '') + text
        this.emit({ type: 'reasoning', itemId: item.id, text })
      },
      onContent: (text) => {
        if (!item) {
          item = { id: randomUUID(), kind: 'assistant', text: '', taskId: task.id }
          this.addItem(item)
        }
        if (!item.text) this.setStatus({ phase: 'thinking', text: 'Writing…' })
        item.text += text
        this.emit({ type: 'delta', itemId: item.id, text })
      },
      onToolDelta: () => {
        if (this.statusNow.phase !== 'working') this.setStatus({ phase: 'working', text: 'Preparing the next step…' })
      }
    })
    const finished = item as Extract<TimelineItem, { kind: 'assistant' }> | null
    if (finished) {
      // Inline <think> text joins the separately sent reasoning.
      const { reasoning, answer } = splitThink(finished.text)
      finished.text = /<tool_call>|<function=/.test(answer) ? '' : answer
      const thinking = [finished.reasoning?.trim(), reasoning].filter(Boolean).join('\n\n')
      if (thinking) finished.reasoning = thinking
      else delete finished.reasoning
      if (!finished.text && !thinking) {
        this.data.timeline = this.data.timeline.filter((i) => i.id !== finished.id)
        this.emit({ type: 'item', item: { ...finished, text: '' } })
      } else this.emit({ type: 'item', item: { ...finished } })
    }
    this.measure(result, estimated, nCtx)
    this.countDrafts(task, model, result)
    return result
  }

  /** Updates the Memory meter, and the estimate's correction, from the server's token counts. */
  private measure(result: TurnResult, estimated: number, nCtx: number): void {
    let context: ContextUsage
    if (result.usage) {
      this.scale = tokenScale(estimated, result.usage.prompt)
      context = { used: result.usage.prompt + result.usage.completion, size: nCtx, exact: true }
    } else {
      const reply = result.content + JSON.stringify(result.toolCalls)
      context = { used: Math.ceil((estimated + estimateTokens(reply)) * this.scale), size: nCtx, exact: false }
    }
    this.data.context = context
    this.emit({ type: 'context', context })
    this.scheduleSave()
  }

  /** Adds the turn's speculative-decoding numbers to the task's draft stats. */
  private countDrafts(task: TaskRecord, model: string, result: TurnResult): void {
    if (!result.gen) return
    const draft = addDraft(task.draft, result.gen, this.deps.client.noDraftBaseline(model))
    if (!draft || draft === task.draft) return
    task.draft = draft
    this.emit({ type: 'draft', taskId: task.id, draft })
    this.scheduleSave()
  }

  // ---------- running tools ----------

  /** The tools offered in this request: the built-in ones, plus find/load and loaded tool-server tools when there are tool servers. */
  private toolDefs(): unknown[] {
    if (!this.extraServers.length) return TOOL_DEFS
    const loaded = [...this.loaded.values()].map((t) => ({
      type: 'function' as const,
      function: { name: t.qualifiedName, description: t.description ?? `${t.name} (from ${t.server})`, parameters: t.inputSchema }
    }))
    return [...TOOL_DEFS, ...EXTRA_TOOL_DEFS, ...loaded]
  }

  /** A tool-server tool by name, starting the servers first if needed (the model may call one it remembers). */
  private async extraTool(name: string): Promise<ToolInfo | undefined> {
    const servers = this.deps.toolServers
    if (!servers || !this.extraServers.length || !name.includes('__')) return undefined
    const found = servers.find(name)
    if (found) return found
    await this.startToolServers()
    return servers.find(name)
  }

  private async startToolServers(): Promise<void> {
    const servers = this.deps.toolServers!
    const waiting = this.extraServers.filter((n) => servers.status().servers.find((s) => s.name === n)?.state !== 'connected')
    if (!waiting.length) return
    this.setStatus({ phase: 'working', text: `Starting ${waiting.map(serverLabel).join(' and ')}… (the first time can take a minute)` })
    await servers.start()
  }

  /** Adds tools to the loaded set (newest last), unloading the oldest while they take more than their share of the context. */
  private loadTools(tools: ToolInfo[]): ToolInfo[] {
    for (const t of tools) {
      this.loaded.delete(t.qualifiedName)
      this.loaded.set(t.qualifiedName, t)
    }
    const budget = Math.floor(this.nCtx * LOADED_TOOLS_SHARE)
    const cost = (t: ToolInfo): number => estimateTokens(JSON.stringify({ name: t.qualifiedName, description: t.description, parameters: t.inputSchema }))
    const unloaded: ToolInfo[] = []
    let total = [...this.loaded.values()].reduce((n, t) => n + cost(t), 0)
    for (const t of [...this.loaded.values()]) {
      if (total <= budget || this.loaded.size <= 1) break
      this.loaded.delete(t.qualifiedName)
      unloaded.push(t)
      total -= cost(t)
    }
    return unloaded
  }

  private async execute(
    task: TaskRecord,
    call: WireToolCall,
    signal: AbortSignal,
    snap: Snapshotter | null,
    turn: TurnResult
  ): Promise<{ result: ToolOutcome; title: string }> {
    const name = call.function.name
    const parsed = parseToolArgs(call.function.arguments)
    const args = 'args' in parsed ? parsed.args : {}
    const title = describeStep(name, args)
    const step = this.addStep(task, { title, tool: name, status: 'running', detail: stepDetail(name, args) })
    this.setStatus({ phase: 'working', text: `${title}…` })

    const finish = (result: ToolOutcome, status?: Step['status'], note?: string): { result: ToolOutcome; title: string } => {
      const output = step.output ? step.output : result.text
      this.updateStep(step, { status: status ?? (result.isError ? 'error' : 'done'), output: headTail(output, MAX_STEP_OUTPUT_LIVE), note })
      return { result, title }
    }

    if ('error' in parsed) {
      const cut = turn.finishReason === 'length'
        ? ' Your message was cut off because it was too long, so the arguments are incomplete. Write large files in smaller parts: create the file with the first part, then add the rest with edit_file.'
        : ''
      return finish({ text: parsed.error + cut, isError: true })
    }
    const known = TOOL_NAMES.has(name) || (this.extraServers.length > 0 && EXTRA_TOOL_NAMES.has(name))
    const extra = known ? undefined : await this.extraTool(name).catch(() => undefined)
    if (!known && !extra) {
      const more = this.extraServers.length ? ' Tools from the tool servers are found with find_tools and loaded with load_tools.' : ''
      return finish({ text: `There is no tool named "${name}". The tools are: ${[...TOOL_NAMES].join(', ')}.${more}`, isError: true })
    }

    try {
      const result = extra ? await this.runExtraTool(step, extra, args, signal) : await this.runTool(task, step, name, args, signal, snap)
      if ('denied' in result) return finish(result.outcome, 'skipped', "You didn't allow this.")
      if ('refused' in result) return finish(result.outcome, 'refused', result.refused)
      if (result.changed?.length || name === 'run_command' || name === 'start_background_command') {
        if (snap) await this.refreshFiles(task, snap, result.changed).catch(() => undefined)
      }
      return finish(result)
    } catch (e) {
      if (signal.aborted) {
        finish({ text: 'Stopped by the owner.', isError: true }, 'skipped')
        throw new StopTask()
      }
      return finish({ text: `The tool failed: ${(e as Error).message}`, isError: true })
    }
  }

  /** Asks the owner when the approval level requires it. */
  private async approve(
    step: Step,
    action: ActionKind,
    what: { action: string; detail: string; risk?: string },
    signal: AbortSignal
  ): Promise<boolean> {
    if (!needsApproval(this.deps.settings().approval, action, this.allowed)) return true
    this.updateStep(step, { status: 'waiting' })
    const lastText = [...this.data.messages].reverse().find((m) => m.role === 'assistant')?.content.trim() ?? ''
    const why = lastText ? (lastText.length > 500 ? `${lastText.slice(0, 500)}…` : lastText) : `${step.title}.`
    const answer = await this.ask({ kind: 'approval', action: what.action, why, risk: what.risk, detail: what.detail }, signal)
    if (!answer) throw new StopTask()
    this.updateStep(step, { status: 'running' })
    if (answer.kind !== 'approval' || answer.decision === 'deny') return false
    if (answer.decision === 'allow-task') this.allowed.add(approvalKey(action))
    return true
  }

  private async runTool(
    task: TaskRecord,
    step: Step,
    name: string,
    args: Record<string, unknown>,
    signal: AbortSignal,
    snap: Snapshotter | null
  ): Promise<ToolOutcome | { denied: true; outcome: ToolOutcome } | { refused: string; outcome: ToolOutcome }> {
    const root = this.root
    const denied = (what: string): { denied: true; outcome: ToolOutcome } => ({
      denied: true,
      outcome: {
        text: `The owner did not allow this (${what}). Don't try the same thing another way. Continue without it if you can, or use ask_user to ask what they would like instead.`,
        isError: true
      }
    })
    const beforeChange = async (rel: string): Promise<void> => {
      if (snap && task.snapshot) await snap.beforeChange(task.snapshot as Snapshot, rel)
    }

    switch (name) {
      case 'list_files':
        return files.listFiles(root, args)
      case 'find_files':
        return files.findFiles(root, args)
      case 'search_text':
        return files.searchText(root, args)
      case 'read_file':
        return files.readFile(root, args)

      case 'write_file':
      case 'edit_file': {
        const target = await resolveInProject(root, args.path, { write: true }).catch((e: Error) => e)
        if (target instanceof Error) return { text: target.message, isError: true }
        const exists = await fs.stat(target.abs).then(() => true, () => false)
        const verb = name === 'edit_file' ? 'Change' : exists ? 'Replace' : 'Create'
        const preview = name === 'write_file' ? String(args.content ?? '').slice(0, 3000) : `Replace:\n${String(args.old_text ?? '')}\n\nWith:\n${String(args.new_text ?? '')}`.slice(0, 3000)
        const ok = await this.approve(step, { type: 'write' }, { action: `${verb} the file ${target.rel}`, detail: `${target.rel}\n\n${preview}` }, signal)
        if (!ok) return denied(`${verb.toLowerCase()} ${target.rel}`)
        await beforeChange(target.rel)
        return name === 'write_file' ? files.writeFile(root, args) : files.editFile(root, args)
      }

      case 'delete_path': {
        const target = await resolveInProject(root, args.path, { write: true }).catch((e: Error) => e)
        if (target instanceof Error) return { text: target.message, isError: true }
        if (!(await fs.stat(target.abs).then(() => true, () => false))) return { text: `"${target.rel}" does not exist.`, isError: true }
        const ok = await this.approve(step, { type: 'delete' }, { action: `Move ${target.rel} to the Recycle Bin`, detail: target.abs, risk: 'This deletes files (you can get them back from the Recycle Bin).' }, signal)
        if (!ok) return denied(`delete ${target.rel}`)
        await beforeChange(target.rel)
        return files.deletePath(root, args, this.deps.trash)
      }

      case 'run_command':
      case 'start_background_command': {
        const command = typeof args.command === 'string' ? args.command.trim() : ''
        if (!command) return { text: 'Give the command to run.', isError: true }
        const verdict = classifyCommand(command, root)
        if (verdict.verdict === 'refuse') {
          this.notice(`I refused to run a command: ${verdict.reason}`, 'warn', task.id)
          return {
            refused: verdict.reason,
            outcome: {
              text: `Refused: ${verdict.reason} This app never allows that. Don't try to do it another way; if it is truly needed, explain to the owner what they would have to do themselves and why.`,
              isError: true
            }
          }
        }
        const risk = verdict.verdict === 'ask' ? verdict.kind : undefined
        const ok = await this.approve(
          step,
          { type: 'command', risk },
          { action: name === 'start_background_command' ? `${step.title} (keeps running in the background)` : step.title, detail: command, risk: verdict.verdict === 'ask' ? verdict.reason : undefined },
          signal
        )
        if (!ok) return denied(`run "${command}"`)
        const onOutput = (chunk: string): void => {
          step.output = ((step.output ?? '') + chunk).slice(-MAX_STEP_OUTPUT_LIVE)
          this.emit({ type: 'output', stepId: step.id, chunk })
        }
        if (name === 'start_background_command') {
          const id = this.background.run(command, root, onOutput)
          return { text: `Started in the background with id "${id}". Use check_background_command to see its output.`, isError: false }
        }
        const minutes = this.deps.settings().commandTimeoutMin
        const seconds = typeof args.timeout_seconds === 'number' && args.timeout_seconds > 0 ? args.timeout_seconds : minutes * 60
        const timeoutMs = Math.min(seconds, 60 * 60) * 1000
        const r = await runCommand(command, { cwd: root, timeoutMs, signal, onOutput })
        if (r.stopped) throw new StopTask()
        const report = commandReport(r, timeoutMs)
        report.text += commandHints(r.output, this.background.running())
        step.output = r.output
        return report
      }

      case 'check_background_command': {
        const wait = typeof args.wait_seconds === 'number' ? Math.min(60, Math.max(0, args.wait_seconds)) : 5
        const checked = await this.background.check(String(args.id ?? ''), wait * 1000, signal)
        return { ...checked, text: checked.text + commandHints(checked.text) }
      }
      case 'stop_background_command':
        return this.background.stop(String(args.id ?? ''))

      case 'check_page': {
        if (!this.deps.checkPage) return { text: 'Checking pages is not available here.', isError: true }
        const target = await pageTarget(root, args.page ?? args.url ?? args.path).catch((e: Error) => e)
        if (target instanceof Error) return { text: target.message, isError: true }
        const wait = typeof args.wait_seconds === 'number' ? Math.min(30, Math.max(0, args.wait_seconds)) : 2
        const report = await this.deps.checkPage(target, wait * 1000, signal)
        if (signal.aborted) throw new StopTask()
        return { text: formatPageReport(report), isError: pageHasProblems(report) }
      }

      case 'find_tools': {
        const servers = this.deps.toolServers
        if (!servers) return { text: 'There are no extra tools here.', isError: true }
        const query = typeof args.query === 'string' ? args.query.trim() : ''
        if (!query) return { text: 'Give some keywords to search for, e.g. "decompile function".', isError: true }
        await this.startToolServers()
        const states = servers.status().servers.filter((s) => this.extraServers.includes(s.name))
        const broken = states.filter((s) => s.state === 'error').map((s) => `${serverLabel(s.name)} could not start: ${s.error ?? 'unknown problem'}`)
        const limit = typeof args.limit === 'number' ? Math.min(30, Math.max(1, Math.round(args.limit))) : 10
        const hits = servers.search(query, limit)
        const lines = hits.map((t) => `- ${t.qualifiedName}: ${toolSummary(t.description)}${t.destructive ? ' (can change things: the owner is asked separately)' : ''}${this.loaded.has(t.qualifiedName) ? ' [loaded]' : ''}`)
        const running = states.filter((s) => s.state === 'connected').map((s) => `${serverLabel(s.name)} (${s.tools} tools)`)
        const text = [
          ...broken,
          hits.length ? `Tools matching "${query}":\n${lines.join('\n')}\n\nLoad the ones you need with load_tools, then call them.` : `No tools matched "${query}". Try other words.`,
          running.length ? `Tool servers running: ${running.join(', ')}.` : ''
        ]
          .filter(Boolean)
          .join('\n\n')
        return { text, isError: !hits.length && broken.length > 0 }
      }

      case 'load_tools': {
        const servers = this.deps.toolServers
        if (!servers) return { text: 'There are no extra tools here.', isError: true }
        const names = (Array.isArray(args.names) ? args.names : typeof args.names === 'string' ? args.names.split(/[,\s]+/) : []).map(String).filter(Boolean)
        if (!names.length) return { text: 'Give the tool names to load, e.g. ["rea__open_binary"].', isError: true }
        await this.startToolServers()
        const found = names.map((n) => ({ n, t: servers.find(n.trim()) }))
        const tools = found.flatMap((f) => (f.t ? [f.t] : []))
        const missing = found.filter((f) => !f.t).map((f) => f.n)
        const unloaded = this.loadTools(tools)
        const now = tools.filter((t) => this.loaded.has(t.qualifiedName))
        const text = [
          now.length ? `Loaded: ${now.map((t) => t.qualifiedName).join(', ')}. You can call them from your next step.` : '',
          unloaded.length ? `Unloaded to make room: ${unloaded.map((t) => t.qualifiedName).join(', ')} (load them again if you need them).` : '',
          missing.length ? `Not found: ${missing.join(', ')}. Use find_tools to get the exact names.` : ''
        ]
          .filter(Boolean)
          .join('\n')
        return { text, isError: !now.length }
      }

      case 'ask_user': {
        const question = typeof args.question === 'string' ? args.question.trim() : ''
        if (!question) return { text: 'Give the question to ask.', isError: true }
        this.addItem({ id: randomUUID(), kind: 'assistant', text: question, taskId: task.id })
        const answer = await this.ask({ kind: 'question', question }, signal)
        if (!answer || answer.kind !== 'question') throw new StopTask()
        this.addItem({ id: randomUUID(), kind: 'user', text: answer.text, taskId: task.id })
        return { text: `The owner answered: ${answer.text}`, isError: false }
      }
    }
    return { text: `Unknown tool "${name}".`, isError: true }
  }

  /** Runs a tool from a tool server (such as REA) after the owner's OK, loading it if the model hadn't. */
  private async runExtraTool(
    step: Step,
    tool: ToolInfo,
    args: Record<string, unknown>,
    signal: AbortSignal
  ): Promise<ToolOutcome | { denied: true; outcome: ToolOutcome }> {
    const label = serverLabel(tool.server)
    const ok = await this.approve(
      step,
      { type: 'tool', server: tool.server, tool: tool.name, destructive: tool.destructive },
      {
        action: step.title,
        detail: `${tool.qualifiedName}\n${JSON.stringify(args, null, 2)}`.slice(0, 3000),
        risk: tool.destructive ? `${label} marks this tool as one that can change things or run programs.` : undefined
      },
      signal
    )
    if (!ok) {
      return {
        denied: true,
        outcome: {
          text: `The owner did not allow this (${tool.qualifiedName}). Don't try the same thing another way. Continue without it if you can, or use ask_user to ask what they would like instead.`,
          isError: true
        }
      }
    }
    if (!this.loaded.has(tool.qualifiedName)) this.loadTools([tool])
    const result = await this.deps.toolServers!.call(tool, args, signal)
    if (signal.aborted) throw new StopTask()
    return { text: headTail(result.text), isError: result.isError }
  }

  /** Updates the task's changed-files list (from the restore point, or from what the tools changed). */
  private async refreshFiles(task: TaskRecord, snap: Snapshotter, changed?: ChangedFile[]): Promise<void> {
    let list: ChangedFile[] | null = null
    if (task.snapshot) list = await snap.changedSince(task.snapshot)
    if (!list) {
      const map = new Map(task.files.map((f) => [f.path, f]))
      for (const c of changed ?? []) {
        const prev = map.get(c.path)
        map.set(c.path, prev?.change === 'created' && c.change === 'changed' ? prev : prev?.change === 'created' && c.change === 'deleted' ? { ...c, change: 'deleted' } : c)
      }
      list = [...map.values()].sort((a, b) => a.path.localeCompare(b.path))
    }
    task.files = list
    this.emit({ type: 'files', taskId: task.id, files: list })
  }
}
