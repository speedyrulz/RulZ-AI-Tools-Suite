// Practice batch runs: practice tasks run one after another with nobody watching, and a score per
// run, so two models (or a model before and after fine-tuning) can be compared on the same tasks.
import type { Conversation, RePracticeTask } from './types'

export type BatchItemStatus =
  | 'pending'
  | 'running'
  /** The final ANSWER line passed the task's check. */
  | 'solved'
  /** An ANSWER line that did not pass. */
  | 'wrong'
  /** A final reply without an ANSWER line. */
  | 'no-answer'
  /** Still calling tools at the round limit. */
  | 'rounds'
  /** Still working at the time limit. */
  | 'timeout'
  /** It could not run or be graded (model failed to load, request error, task missing). */
  | 'error'
  /** The user stopped this session in the chat; the run went on to the next task. */
  | 'stopped'

export const BATCH_ITEM_STATUSES: readonly BatchItemStatus[] = [
  'pending',
  'running',
  'solved',
  'wrong',
  'no-answer',
  'rounds',
  'timeout',
  'error',
  'stopped'
]

export interface BatchItem {
  taskId: string
  family: string
  tier: string
  /** 1 for the first try of this task in the run. */
  attempt: number
  status: BatchItemStatus
  /** The chat it ran in. */
  conversationId?: string
  startedAt?: number
  ms?: number
  /** Model replies that called tools. */
  toolRounds?: number
  /** Largest prompt of the session: how full the context got. */
  promptTokens?: number
  /** Tokens the model wrote over the whole session. */
  outputTokens?: number
  /** Tool calls refused because they needed approval and nobody was there to give it. */
  denied?: number
  answer?: string
  error?: string
}

export interface BatchOptions {
  /** Only this task family; '' = every family. */
  family: string
  /** Leave out tasks an earlier session already solved. */
  onlyUnsolved: boolean
  /** At most this many tasks, spread over the families; 0 = all. */
  maxTasks: number
  /** Tries per task. */
  attempts: number
  /** A task still running after this long is stopped; 0 = no limit. */
  minutesPerTask: number
}

export const DEFAULT_BATCH_OPTIONS: BatchOptions = { family: '', onlyUnsolved: false, maxTasks: 0, attempts: 1, minutesPerTask: 30 }

export interface BatchRun {
  id: string
  /** The user's note for telling runs apart ("base", "lora v2"). */
  label: string
  createdAt: number
  finishedAt?: number
  /** 'running' in a saved run that is not running means the app closed during it. */
  status: 'running' | 'done' | 'stopped'
  modelId: string | null
  modelName: string
  contextLength: number | null
  maxToolRounds: number
  kitDir: string
  options: BatchOptions
  items: BatchItem[]
  /** Adapters the model had loaded when the run started, with the strengths its sessions used. Absent = none or an older run. */
  adapters?: { name: string; scale: number }[]
}

/** How a turn in the chat ended (see onTurnEnd in the chat store). */
export type TurnEndReason = 'done' | 'stopped' | 'error' | 'rounds' | 'failed'

/** Result text of a tool call refused in an unattended session; the model reads it and carries on. */
export const UNATTENDED_DENIAL =
  'Not run: this tool needs the user’s approval and this session runs unattended. Continue with the tools that read the target.'

const clampInt = (v: unknown, min: number, max: number, fallback: number): number => {
  const n = typeof v === 'number' && Number.isFinite(v) ? Math.round(v) : fallback
  return Math.min(max, Math.max(min, n))
}

export function cleanBatchOptions(o: Partial<BatchOptions> | null | undefined): BatchOptions {
  const d = DEFAULT_BATCH_OPTIONS
  return {
    family: typeof o?.family === 'string' ? o.family : d.family,
    onlyUnsolved: typeof o?.onlyUnsolved === 'boolean' ? o.onlyUnsolved : d.onlyUnsolved,
    maxTasks: clampInt(o?.maxTasks, 0, 100000, d.maxTasks),
    attempts: clampInt(o?.attempts, 1, 10, d.attempts),
    minutesPerTask: clampInt(o?.minutesPerTask, 0, 1440, d.minutesPerTask)
  }
}

/**
 * The run's items, in the order they run. With a task limit the tasks are taken family by family
 * in turn, so a short run still covers every family. Every task gets its first try before any
 * gets a second, so a run stopped early has tried as many tasks as it could.
 */
export function planBatch(tasks: RePracticeTask[], options: BatchOptions, solved: ReadonlySet<string>): BatchItem[] {
  const o = cleanBatchOptions(options)
  const byFamily = new Map<string, RePracticeTask[]>()
  const seen = new Set<string>()
  for (const t of tasks) {
    if (seen.has(t.id)) continue
    seen.add(t.id)
    if (o.family && t.family !== o.family) continue
    if (o.onlyUnsolved && solved.has(t.id)) continue
    const list = byFamily.get(t.family) ?? []
    list.push(t)
    byFamily.set(t.family, list)
  }
  const families = [...byFamily.keys()].sort((a, b) => a.localeCompare(b))
  for (const f of families) byFamily.get(f)!.sort((a, b) => a.id.localeCompare(b.id, undefined, { numeric: true }))
  const total = families.reduce((n, f) => n + byFamily.get(f)!.length, 0)
  const want = o.maxTasks > 0 ? Math.min(o.maxTasks, total) : total
  const picked: RePracticeTask[] = []
  for (let row = 0; picked.length < want; row++) {
    for (const f of families) {
      const t = byFamily.get(f)![row]
      if (t && picked.length < want) picked.push(t)
    }
  }
  const items: BatchItem[] = []
  for (let attempt = 1; attempt <= o.attempts; attempt++) {
    for (const t of picked) items.push({ taskId: t.id, family: t.family, tier: t.tier, attempt, status: 'pending' })
  }
  return items
}

/** Numbers read off a finished session's chat. */
export function sessionStats(conv: Pick<Conversation, 'messages'>): Pick<BatchItem, 'toolRounds' | 'promptTokens' | 'outputTokens' | 'denied'> {
  let toolRounds = 0
  let promptTokens = 0
  let outputTokens = 0
  let denied = 0
  for (const m of conv.messages) {
    if (m.role !== 'assistant') continue
    if (m.toolCalls?.length) toolRounds++
    promptTokens = Math.max(promptTokens, m.stats?.promptTokens ?? 0)
    outputTokens += m.stats?.predictedTokens ?? 0
    for (const c of m.toolCalls ?? []) if (c.status === 'denied' && c.result === UNATTENDED_DENIAL) denied++
  }
  return { toolRounds, promptTokens, outputTokens, denied }
}

/**
 * What became of one task. `end` is how its turn ended; `timedOut` says the run stopped it at
 * the time limit. A session that ended on its own is read from its grade.
 */
export function itemResult(
  conv: Pick<Conversation, 'messages' | 're'> | null | undefined,
  end: TurnEndReason | null,
  flags: { timedOut?: boolean } = {}
): Pick<BatchItem, 'status' | 'answer' | 'error'> {
  if (flags.timedOut) return { status: 'timeout' }
  if (!conv) return { status: 'error', error: 'The session’s chat could not be read.' }
  const lastError = conv.messages.findLast((m) => m.role === 'assistant')?.error
  switch (end) {
    case 'rounds':
      return { status: 'rounds' }
    case 'stopped':
      return { status: 'stopped' }
    case 'error':
    case 'failed':
    case null:
      return { status: 'error', error: lastError ?? (end === 'failed' ? 'The session could not start (no model, or it failed to load).' : 'The session ended without a reply.') }
    case 'done': {
      const re = conv.re
      const answer = re?.practice?.answer
      if (re?.outcome === 'solved') return { status: 'solved', answer }
      if (re?.outcome === 'failed' && re.autoGraded) return answer ? { status: 'wrong', answer } : { status: 'no-answer' }
      return { status: 'error', error: 'The final reply could not be graded.' }
    }
  }
}

/** Counts toward the score: the model had its chance. Errors and stops say nothing about the model. */
export function isScored(status: BatchItemStatus): boolean {
  return status === 'solved' || status === 'wrong' || status === 'no-answer' || status === 'rounds' || status === 'timeout'
}

export function isFinished(status: BatchItemStatus): boolean {
  return status !== 'pending' && status !== 'running'
}

export function median(values: number[]): number | null {
  const v = values.filter((x) => Number.isFinite(x)).sort((a, b) => a - b)
  if (!v.length) return null
  const mid = v.length >> 1
  return v.length % 2 ? v[mid] : (v[mid - 1] + v[mid]) / 2
}

export interface BatchScore {
  solved: number
  /** Tries that count (see isScored). */
  scored: number
  /** solved / scored, or null before anything counted. */
  rate: number | null
}

export interface BatchSummary extends BatchScore {
  total: number
  finished: number
  /** Tasks solved in at least one try, of the tasks with a try that counted. */
  tasksSolved: number
  tasksScored: number
  counts: Record<BatchItemStatus, number>
  /** Of solved tries. */
  medianRounds: number | null
  medianMs: number | null
  outputTokens: number
  denied: number
  families: { family: string; score: BatchScore }[]
}

const score = (items: BatchItem[]): BatchScore => {
  const scored = items.filter((i) => isScored(i.status)).length
  const solved = items.filter((i) => i.status === 'solved').length
  return { solved, scored, rate: scored ? solved / scored : null }
}

export function summarize(run: Pick<BatchRun, 'items'>): BatchSummary {
  const items = run.items
  const counts = Object.fromEntries(BATCH_ITEM_STATUSES.map((s) => [s, 0])) as Record<BatchItemStatus, number>
  for (const i of items) counts[i.status]++
  const solvedItems = items.filter((i) => i.status === 'solved')
  const tasksScored = new Set(items.filter((i) => isScored(i.status)).map((i) => i.taskId))
  const tasksSolved = new Set(solvedItems.map((i) => i.taskId))
  const families = [...new Set(items.map((i) => i.family))]
    .sort((a, b) => a.localeCompare(b))
    .map((family) => ({ family, score: score(items.filter((i) => i.family === family)) }))
  return {
    ...score(items),
    total: items.length,
    finished: items.filter((i) => isFinished(i.status)).length,
    tasksSolved: tasksSolved.size,
    tasksScored: tasksScored.size,
    counts,
    medianRounds: median(solvedItems.flatMap((i) => (i.toolRounds === undefined ? [] : [i.toolRounds]))),
    medianMs: median(solvedItems.flatMap((i) => (i.ms === undefined ? [] : [i.ms]))),
    outputTokens: items.reduce((n, i) => n + (i.outputTokens ?? 0), 0),
    denied: items.reduce((n, i) => n + (i.denied ?? 0), 0),
    families
  }
}

/** A run as saved: one left 'running' by a closed app is shown as stopped, its running item as pending. */
export function settleRun(run: BatchRun): BatchRun {
  if (run.status !== 'running') return run
  return { ...run, status: 'stopped', items: run.items.map((i) => (i.status === 'running' ? { ...i, status: 'pending' } : i)) }
}

const str = (v: unknown, max: number): string => (typeof v === 'string' ? v.slice(0, max) : '')
const optNum = (v: unknown): number | undefined => (typeof v === 'number' && Number.isFinite(v) && v >= 0 ? v : undefined)

/** A run read from disk or from the renderer, with anything unexpected dropped; null if unusable. */
export function cleanBatchRun(raw: unknown): BatchRun | null {
  if (!raw || typeof raw !== 'object') return null
  const r = raw as Record<string, unknown>
  if (typeof r.id !== 'string' || !/^[\w-]{1,64}$/.test(r.id) || !Array.isArray(r.items)) return null
  const items: BatchItem[] = []
  for (const x of r.items.slice(0, 100000)) {
    if (!x || typeof x !== 'object') continue
    const i = x as Record<string, unknown>
    if (typeof i.taskId !== 'string' || !i.taskId) continue
    const status = BATCH_ITEM_STATUSES.includes(i.status as BatchItemStatus) ? (i.status as BatchItemStatus) : 'pending'
    const conversationId = str(i.conversationId, 80)
    const answer = str(i.answer, 500)
    const error = str(i.error, 2000)
    items.push({
      taskId: i.taskId.slice(0, 80),
      family: str(i.family, 80),
      tier: str(i.tier, 80),
      attempt: clampInt(i.attempt, 1, 10, 1),
      status,
      ...(conversationId ? { conversationId } : {}),
      ...(optNum(i.startedAt) !== undefined ? { startedAt: optNum(i.startedAt) } : {}),
      ...(optNum(i.ms) !== undefined ? { ms: optNum(i.ms) } : {}),
      ...(optNum(i.toolRounds) !== undefined ? { toolRounds: optNum(i.toolRounds) } : {}),
      ...(optNum(i.promptTokens) !== undefined ? { promptTokens: optNum(i.promptTokens) } : {}),
      ...(optNum(i.outputTokens) !== undefined ? { outputTokens: optNum(i.outputTokens) } : {}),
      ...(optNum(i.denied) !== undefined ? { denied: optNum(i.denied) } : {}),
      ...(answer ? { answer } : {}),
      ...(error ? { error } : {})
    })
  }
  const status = r.status === 'done' || r.status === 'stopped' || r.status === 'running' ? r.status : 'stopped'
  return {
    id: r.id,
    label: str(r.label, 80),
    createdAt: optNum(r.createdAt) ?? 0,
    ...(optNum(r.finishedAt) !== undefined ? { finishedAt: optNum(r.finishedAt) } : {}),
    status,
    modelId: typeof r.modelId === 'string' && r.modelId ? r.modelId.slice(0, 500) : null,
    modelName: str(r.modelName, 300),
    contextLength: optNum(r.contextLength) ?? null,
    maxToolRounds: clampInt(r.maxToolRounds, 1, 2000, 80),
    kitDir: str(r.kitDir, 1000),
    options: cleanBatchOptions(r.options as Partial<BatchOptions> | undefined),
    items
  }
}
