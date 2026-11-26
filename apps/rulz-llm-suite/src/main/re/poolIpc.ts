// The IPC handlers of the practice pool: input checked here, the work done by PracticePool. No electron import:
// index.ts passes each entry to its `handle()`, and tests call them directly.
import { cleanBatchOptions, type BatchRun } from '@shared/batch'
import type { BatchStartRequest, IpcInvoke } from '@shared/ipc'
import { cleanBatchTarget, type BatchTarget } from '@shared/remote'
import type { Conversation, RePracticeTask } from '@shared/types'
import type { PracticePool } from './pool'

/** Most tasks one start request may carry (the kit has a few hundred; this only stops absurd input). */
export const MAX_START_TASKS = 20000
const TASK_ID = /^[\w-]{1,80}$/
const CHAT_ID = /^[\w-]{1,64}$/

const str = (v: unknown, max: number): string | null => (typeof v === 'string' && v.length <= max ? v : null)

function cleanTask(raw: unknown): RePracticeTask | null {
  if (!raw || typeof raw !== 'object') return null
  const t = raw as Record<string, unknown>
  const id = str(t.id, 80)
  if (!id || !TASK_ID.test(id)) return null
  return { id, family: str(t.family, 80) ?? '', tier: str(t.tier, 80) ?? '', binary: str(t.binary, 300) ?? '', prompt: str(t.prompt, 20000) ?? '', set: str(t.set, 40) ?? 'original' }
}

/** A start request from the window with anything unusable refused (a sentence) or replaced. */
export function cleanStartRequest(raw: unknown): BatchStartRequest {
  if (!raw || typeof raw !== 'object') throw new Error('Not a batch run request.')
  const r = raw as Record<string, unknown>
  if (!Array.isArray(r.tasks)) throw new Error('The request has no tasks.')
  if (r.tasks.length > MAX_START_TASKS) throw new Error(`Too many tasks (most ${MAX_START_TASKS}).`)
  const tasks = r.tasks.flatMap((t) => {
    const c = cleanTask(t)
    return c ? [c] : []
  })
  const kitDir = str(r.kitDir, 1000)
  if (!kitDir?.trim()) throw new Error('The request has no practice kit folder.')
  const modelId = str(r.modelId, 500)
  if (!modelId) throw new Error('The request has no model.')
  const solved = Array.isArray(r.solved) ? r.solved.flatMap((x) => (typeof x === 'string' && TASK_ID.test(x) ? [x] : [])).slice(0, MAX_START_TASKS) : []
  return {
    tasks,
    options: cleanBatchOptions(r.options as Partial<BatchRunOptions> | undefined),
    label: (str(r.label, 400) ?? '').trim().slice(0, 80),
    kitDir,
    modelId,
    solved,
    target: cleanBatchTarget(r.target)
  }
}
type BatchRunOptions = BatchRun['options']

type PoolChannel = 're:batchStart' | 're:batchResume' | 're:batchStop' | 're:batchSkip' | 're:watch' | 're:poolChats' | 're:deleteBatch'

export type PoolHandlers = {
  [K in PoolChannel]: (...args: Parameters<IpcInvoke[K]>) => Promise<Awaited<ReturnType<IpcInvoke[K]>>> | Awaited<ReturnType<IpcInvoke[K]>>
}

/** Handlers for the pool's channels. `deleteRun` removes a run from the store. */
export function poolHandlers(pool: PracticePool, deleteRun: (id: string) => Promise<void>): PoolHandlers {
  return {
    're:batchStart': async (req) => pool.start(cleanStartRequest(req)),
    're:batchResume': async (runId, target: BatchTarget) => {
      const id = str(runId, 64)
      if (!id || !CHAT_ID.test(id)) throw new Error('Not a batch run id.')
      return pool.resume(id, cleanBatchTarget(target))
    },
    're:batchStop': (mode) => {
      if (mode !== 'after' && mode !== 'now') throw new Error('Stop how? "after" or "now".')
      pool.stop(mode)
    },
    're:batchSkip': (conversationId) => {
      const id = str(conversationId, 64)
      if (!id || !CHAT_ID.test(id)) throw new Error('Not a chat id.')
      pool.skip(id)
    },
    're:watch': async (conversationId): Promise<Conversation | null> => {
      if (conversationId === null) return pool.watch(null)
      const id = str(conversationId, 64)
      if (!id || !CHAT_ID.test(id)) throw new Error('Not a chat id.')
      return pool.watch(id)
    },
    're:poolChats': () => pool.poolChats(),
    're:deleteBatch': async (id) => {
      if (pool.activeId === id) throw new Error('This batch run is running; stop it first.')
      await deleteRun(id)
    }
  }
}
