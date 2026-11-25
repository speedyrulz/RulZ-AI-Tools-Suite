// Practice results per model: which RE practice tasks a model has tried and solved. Shared by the window and the
// API server (GET /v1/re/overview?model=), so both count the same way. Models are told apart by name (modelKey).
import { isScored, type BatchRun } from './batch'
import { isCloudModelId } from './cloud'
import type { ReTraceInfo } from './types'

/** One practice session: which task, how it was graded, and by which models (see modelKey; empty = not known). */
export interface PracticeRecord {
  taskId: string
  outcome: string
  models: string[]
}

/** What the records are read from; `models` is the library, for the names of the models the chats used. */
export interface PracticeSources {
  summaries: { id: string; rePracticeTask?: string; reOutcome?: string; modelId?: string | null }[]
  traces: Pick<ReTraceInfo, 'conversationId' | 'api' | 'models'>[]
  runs: Pick<BatchRun, 'modelId' | 'modelName' | 'items'>[]
  models: { id: string; name: string }[]
}

/**
 * The same key for a model on this PC and its copy on a rental, and after its file moved to another folder: models
 * are told apart by name. Takes a name, a library id (a path) or a remote id. An OpenRouter model is recorded under its
 * whole id (openrouter/<slug>): it is never the same as a model on this PC of that name.
 */
export function modelKey(nameOrId: string): string {
  if (isCloudModelId(nameOrId)) return nameOrId.toLowerCase()
  return (nameOrId.split(/[\\/]/).at(-1) ?? '')
    .toLowerCase()
    .replace(/\.gguf$/, '')
    .replace(/-\d{5}-of-\d{5}$/, '')
    .replace(/[^a-z0-9._-]+/g, '-')
}

/** The key of a model id, by its name in the library when it is still there. */
export function keyOfModel(modelId: string, models: PracticeSources['models']): string {
  if (isCloudModelId(modelId)) return modelKey(modelId)
  return modelKey(models.find((m) => m.id === modelId)?.name ?? modelId)
}

/** Tasks a practice session (here or in another app) has solved. */
export function solvedTasks(records: PracticeRecord[]): Set<string> {
  const out = new Set<string>()
  for (const r of records) if (r.outcome === 'solved') out.add(r.taskId)
  return out
}

/**
 * Practice sessions from the chat list, from other apps' recorded sessions, and from batch runs whose chats were
 * deleted since (their results are still on the scoreboard).
 */
export function practiceRecords({ summaries, traces, runs, models }: PracticeSources): PracticeRecord[] {
  const out: PracticeRecord[] = []
  const chatIds = new Set(summaries.map((s) => s.id))
  // A batch run's chat counts for the run's model, whoever answered a follow-up question in it later.
  const runModel = new Map<string, string>()
  for (const run of runs) {
    const key = run.modelName ? modelKey(run.modelName) : run.modelId ? keyOfModel(run.modelId, models) : ''
    for (const i of run.items) {
      if (i.conversationId && key) runModel.set(i.conversationId, key)
      if (!isScored(i.status) || (i.conversationId && chatIds.has(i.conversationId))) continue
      out.push({ taskId: i.taskId, outcome: i.status === 'solved' ? 'solved' : 'failed', models: key ? [key] : [] })
    }
  }
  for (const s of summaries) {
    if (!s.rePracticeTask) continue
    const key = runModel.get(s.id) ?? (s.modelId ? keyOfModel(s.modelId, models) : '')
    out.push({ taskId: s.rePracticeTask, outcome: s.reOutcome ?? 'unrated', models: key ? [key] : [] })
  }
  for (const t of traces) {
    const p = t.api?.practice
    if (p && !chatIds.has(t.conversationId)) out.push({ taskId: p.taskId, outcome: t.api!.outcome, models: t.models.map(modelKey) })
  }
  return out
}

/** The records of one model; with no model, all of them. */
export function practiceOf(modelId: string | null, sources: PracticeSources): PracticeRecord[] {
  const all = practiceRecords(sources)
  if (!modelId) return all
  const key = keyOfModel(modelId, sources.models)
  return all.filter((r) => r.models.includes(key))
}

export interface TaskRecord {
  attempts: number
  /** 'tried': only sessions that were not graded, or graded as partial. */
  status: 'solved' | 'failed' | 'tried'
}

/** Per task: how many sessions, and the best that came of them. */
export function taskRecords(records: PracticeRecord[]): Map<string, TaskRecord> {
  const out = new Map<string, TaskRecord>()
  for (const r of records) {
    const e = out.get(r.taskId) ?? { attempts: 0, status: 'tried' }
    e.attempts++
    if (r.outcome === 'solved') e.status = 'solved'
    else if (r.outcome === 'failed' && e.status !== 'solved') e.status = 'failed'
    out.set(r.taskId, e)
  }
  return out
}
