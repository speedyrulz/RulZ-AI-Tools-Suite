// One open project: its conversation with the model, the agent loop, approvals, undo and diffs.
import { randomUUID } from 'node:crypto'
import { promises as fs } from 'node:fs'
import type {
  AgentEvent,
  AgentStatus,
  AppSettings,
  ChangedFile,
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
import { friendlyServerError, pickDefaultModel, type RulzClient, type TurnResult } from '../rulz/client'
import type { Snapshot, Snapshotter } from '../project/snapshots'
import type { ProjectData, ProjectStore } from '../project/store'
import { BackgroundCommands, commandHints, commandReport, headTail, runCommand } from './commands'
import {
  CHARS_PER_TOKEN,
  CONTEXT_SHARE,
  estimateTokens,
  fitMessages,
  mechanicalSummary,
  SUMMARY_PROMPT,
  summaryRange,
  transcript
} from './context'
import * as files from './fileTools'
import type { ToolOutcome } from './fileTools'
import { LoopGuard, type GuardTrip } from './loopGuard'
import { formatPageReport, pageHasProblems, pageTarget, type PageReport, type PageTarget } from './pageCheck'
import { resolveInProject } from './pathGuard'
import { HOW_TO_RUN_TASK, parseRunLine, systemPrompt } from './prompt'
import { approvalKey, classifyCommand, needsApproval, type ActionKind } from './safety'
import { describeStep, parseToolArgs, TOOL_DEFS, TOOL_NAMES } from './tools'

const TOOLS_TOKENS = estimateTokens(JSON.stringify(TOOL_DEFS))
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
}

/** Ends the task without counting as an error (owner chose Stop in a prompt, or pressed Stop). */
class StopTask extends Error {}

/** An error whose message is already written for the owner. */
class FriendlyError extends Error {}

const isQwenCoder = (model: string): boolean => /qwen3[-_.]?coder/i.test(model)

function cleanContent(text: string): string {
  return text.replace(/<think>[\s\S]*?<\/think>/g, '').trim()
}

function stepDetail(name: string, args: Record<string, unknown>): string | undefined {
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
      runCommand: this.data.runCommand
    }
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

  private notice(text: string, tone: 'info' | 'ok' | 'warn' | 'error', taskId?: string): void {
    this.addItem({ id: randomUUID(), kind: 'notice', text, tone, taskId })
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
  start(text: string, opts: { howToRun?: boolean; shown?: string } = {}): void {
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
    void this.run(task, trimmed, opts.howToRun ?? false)
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

  private async run(task: TaskRecord, text: string, howToRun: boolean): Promise<void> {
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
    return nCtx ?? info?.loadedContext ?? DEFAULT_CTX
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
    const budget = Math.floor(nCtx * CONTEXT_SHARE)
    for (let attempt = 0; ; attempt++) {
      const system = systemPrompt({
        projectPath: this.root,
        projectName: this.root.split(/[\\/]/).pop() ?? this.root,
        date: new Date().toDateString(),
        summary: this.data.summary,
        background: this.background.running()
      })
      const fit = fitMessages(this.data.messages, budget, estimateTokens(system) + TOOLS_TOKENS)
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
    const used = messages.reduce((n, m) => n + estimateTokens(m.content) + estimateTokens(JSON.stringify(m.tool_calls ?? '')), TOOLS_TOKENS)
    const maxTokens = Math.max(1024, Math.min(16384, nCtx - used - 256))
    const sampling = isQwenCoder(model) ? { temperature: 0.7, top_p: 0.8, top_k: 20, repeat_penalty: 1.05 } : {}
    const result = await this.deps.client.streamTurn({ model, messages, tools: TOOL_DEFS, max_tokens: maxTokens, ...sampling }, signal, {
      onProgress: (done, total) => {
        if (total > 4000) this.setStatus({ phase: 'thinking', text: `Reading the project notes… ${Math.round((done / total) * 100)}%` })
      },
      onContent: (text) => {
        if (!item) {
          item = { id: randomUUID(), kind: 'assistant', text: '', taskId: task.id }
          this.addItem(item)
          this.setStatus({ phase: 'thinking', text: 'Writing…' })
        }
        item.text += text
        this.emit({ type: 'delta', itemId: item.id, text })
      },
      onToolDelta: () => {
        if (this.statusNow.phase !== 'working') this.setStatus({ phase: 'working', text: 'Preparing the next step…' })
      }
    })
    const finished = item as Extract<TimelineItem, { kind: 'assistant' }> | null
    if (finished) {
      finished.text = cleanContent(finished.text)
      if (!finished.text || /<tool_call>|<function=/.test(finished.text)) {
        this.data.timeline = this.data.timeline.filter((i) => i.id !== finished.id)
        this.emit({ type: 'item', item: { ...finished, text: '' } })
      } else this.emit({ type: 'item', item: { ...finished } })
    }
    return result
  }

  // ---------- running tools ----------

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
    if (!TOOL_NAMES.has(name)) {
      return finish({ text: `There is no tool named "${name}". The tools are: ${[...TOOL_NAMES].join(', ')}.`, isError: true })
    }

    try {
      const result = await this.runTool(task, step, name, args, signal, snap)
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
