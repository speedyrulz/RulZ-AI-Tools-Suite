// Practice batch runs: planning a run and scoring it. Pure functions, shared by the computer's app
// and the phone app (ported from RulZ LLM Suite's shared/batch.ts).
import type { BatchItem, BatchItemStatus, BatchOptions, BatchRun, RePracticeTaskStats } from './types'

export const BATCH_ITEM_STATUSES: readonly BatchItemStatus[] = ['pending', 'running', 'solved', 'wrong', 'no-answer', 'timeout', 'error', 'stopped']

export const DEFAULT_BATCH_OPTIONS: BatchOptions = { family: '', onlyUnsolved: false, maxTasks: 0, attempts: 1, minutesPerTask: 30 }

/** Tool replies to approval questions in a batch task: nobody is watching, so they are approved. */
export const BATCH_APPROVAL_NOTE = 'Approved automatically: this task is part of a batch run.'

const clampInt = (v: unknown, min: number, max: number, fallback: number): number => {
  const n = typeof v === 'number' && Number.isFinite(v) ? Math.round(v) : fallback
  return Math.min(max, Math.max(min, n))
}

export function cleanBatchOptions(o: Partial<BatchOptions> | null | undefined): BatchOptions {
  const d = DEFAULT_BATCH_OPTIONS
  return {
    family: typeof o?.family === 'string' ? o.family.slice(0, 80) : d.family,
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
 * `solved` holds the ids of tasks an earlier session solved (for onlyUnsolved).
 */
export function planBatch(tasks: RePracticeTaskStats[], options: BatchOptions, solved: ReadonlySet<string>): BatchItem[] {
  const o = cleanBatchOptions(options)
  const byFamily = new Map<string, RePracticeTaskStats[]>()
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
  const picked: RePracticeTaskStats[] = []
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

/** The practice task "Start next task" picks: unsolved, fewest attempts first, then by id. */
export function nextPracticeTask(tasks: RePracticeTaskStats[]): RePracticeTaskStats | null {
  const open = tasks.filter((t) => t.solved === 0)
  open.sort((a, b) => a.attempts - b.attempts || a.id.localeCompare(b.id, undefined, { numeric: true }))
  return open[0] ?? null
}

/** Per family: tasks, tasks tried at least once, tasks solved at least once (the family tiles). */
export function familyCounts(tasks: RePracticeTaskStats[]): { family: string; total: number; tried: number; solved: number }[] {
  const out = new Map<string, { total: number; tried: number; solved: number }>()
  for (const t of tasks) {
    const f = out.get(t.family) ?? { total: 0, tried: 0, solved: 0 }
    f.total++
    if (t.attempts > 0) f.tried++
    if (t.solved > 0) f.solved++
    out.set(t.family, f)
  }
  return [...out.entries()].sort(([a], [b]) => a.localeCompare(b)).map(([family, c]) => ({ family, ...c }))
}

/** Counts toward the score: the model had its chance. Errors and stops say nothing about the model. */
export function isScored(status: BatchItemStatus): boolean {
  return status === 'solved' || status === 'wrong' || status === 'no-answer' || status === 'timeout'
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
    families
  }
}

/** A run as saved: one left 'running' by a closed app is shown as stopped, its running item as pending. */
export function settleRun(run: BatchRun): BatchRun {
  if (run.status !== 'running') return run
  return { ...run, status: 'stopped', items: run.items.map((i) => (i.status === 'running' ? { ...i, status: 'pending' } : i)) }
}
