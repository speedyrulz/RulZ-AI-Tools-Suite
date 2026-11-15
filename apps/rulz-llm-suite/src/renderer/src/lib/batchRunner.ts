// The loop of a practice batch run: one task after another, each to its end, with a time limit and
// two ways to stop. Everything it touches comes in through `BatchDeps`, so it runs in tests without
// the app.
import { itemResult, sessionStats, type BatchItem, type BatchRun, type TurnEndReason } from '@shared/batch'
import type { Conversation, ReSession } from '@shared/types'

export type BatchStop = 'after' | 'now'

export interface BatchSessionOptions {
  target: string
  goal: string
  modelId: string | null
  title: string
  practice: NonNullable<ReSession['practice']>
  /** Show the new chat on the Chat page: the user was watching the session before it. */
  follow: boolean
}

export interface BatchDeps {
  now: () => number
  sleep: (ms: number) => Promise<void>
  /** 'after': finish the task in progress, then stop. 'now': stop that task too. */
  stopping: () => BatchStop | null
  /** A reply is streaming, running tools, or waiting for its model to load. */
  chatBusy: () => boolean
  /** Nothing is replying and no handoff is being written: a new session can start. */
  chatIdle: () => boolean
  /** The chat a reply is being written into, if any. */
  replyingIn: () => string | null
  /** The chat's final reply is in and being graded: its end is about to be reported. */
  settling: (conversationId: string) => boolean
  /** The chat on screen on the Chat page. */
  currentChat: () => string | null
  stopReply: () => void
  prepare: (kitDir: string, taskId: string) => Promise<{ target: string; prompt: string; family: string; tier: string }>
  /** Starts the session unattended; null if another reply got in first. */
  startSession: (opts: BatchSessionOptions) => Promise<string | null>
  /** Ends unattended mode (the session started in it). */
  attended: () => void
  failPractice: (conversationId: string, notes: string) => Promise<void>
  snapshot: (conversationId: string) => Promise<Conversation | null>
  onTurnEnd: (fn: (conversationId: string, reason: TurnEndReason) => void) => () => void
  /** The run as it is now; undefined once it was removed. */
  getRun: () => BatchRun | undefined
  /** Changes the run and saves it; undefined once it was removed. */
  patchRun: (fn: (run: BatchRun) => BatchRun) => BatchRun | undefined
}

const POLL_MS = 200

/** Waits until `test` holds (true) or `timeoutMs` passed (false). */
async function until(deps: Pick<BatchDeps, 'now' | 'sleep'>, test: () => boolean, timeoutMs = Infinity): Promise<boolean> {
  const deadline = deps.now() + timeoutMs
  for (;;) {
    if (test()) return true
    if (deps.now() >= deadline) return false
    await deps.sleep(POLL_MS)
  }
}

const errorText = (e: unknown): string => (e instanceof Error ? e.message : String(e))

/** Runs one task to its end and records what became of it. */
async function runItem(
  deps: BatchDeps,
  index: number,
  ends: Map<string, TurnEndReason>,
  follow: boolean
): Promise<{ conversationId: string | null; retry: boolean }> {
  const patchItem = (patch: Partial<BatchItem>): BatchRun | undefined =>
    deps.patchRun((r) => ({ ...r, items: r.items.map((it, i) => (i === index ? { ...it, ...patch } : it)) }))

  const startedAt = deps.now()
  const run = patchItem({ status: 'running', startedAt, error: undefined })
  if (!run) return { conversationId: null, retry: false }
  const item = run.items[index]
  let conversationId: string | null
  try {
    const p = await deps.prepare(run.kitDir, item.taskId)
    if (deps.stopping()) {
      // Asked to stop while the task was being prepared: it has not been tried.
      patchItem({ status: 'pending', startedAt: undefined })
      return { conversationId: null, retry: false }
    }
    conversationId = await deps.startSession({
      target: p.target,
      goal: p.prompt,
      modelId: run.modelId,
      title: `Practice: ${item.taskId}`,
      practice: { taskId: item.taskId, family: p.family, tier: p.tier, kitDir: run.kitDir },
      follow
    })
  } catch (e) {
    deps.attended()
    patchItem({ status: 'error', error: errorText(e), ms: deps.now() - startedAt })
    return { conversationId: null, retry: false }
  }
  if (!conversationId) {
    // Another reply started in the moment between tasks; this task waits for it.
    patchItem({ status: 'pending', startedAt: undefined })
    return { conversationId: null, retry: true }
  }
  const id = conversationId
  patchItem({ conversationId: id })

  const minutes = run.options.minutesPerTask
  // The limit counts from the first reply: loading the model is not the model's working time.
  let replyAt: number | null = null
  const overTime = (): boolean => {
    if (replyAt === null && deps.replyingIn() === id) replyAt = deps.now()
    return minutes > 0 && replyAt !== null && deps.now() >= replyAt + minutes * 60_000
  }
  let timedOut = false
  let userStopped = false
  // A final reply that is being graded is left to finish: it beat the limit and the stop.
  await until(deps, () => ends.has(id) || ((overTime() || deps.stopping() === 'now') && !deps.settling(id)))
  if (!ends.has(id)) {
    if (deps.stopping() === 'now') userStopped = true
    else timedOut = true
    // Stop it, and keep at it: a model still loading has no reply to stop yet.
    while (!ends.has(id)) {
      if (deps.replyingIn() === id) deps.stopReply()
      else if (!deps.chatBusy() && !deps.settling(id)) break
      await until(deps, () => ends.has(id), 3000)
    }
  }
  const workedFrom = replyAt ?? startedAt
  const end = ends.get(id) ?? null
  deps.attended()

  if (userStopped) {
    // "Stop now" cut it off: it has not had its try, so a resumed run starts it again (in a new
    // chat; the cut-off one stays in the chat list).
    patchItem({ status: 'pending', startedAt: undefined, conversationId: undefined })
    return { conversationId: id, retry: false }
  }
  if (timedOut || end === 'rounds') {
    await deps
      .failPractice(id, timedOut ? `Batch run: stopped at the ${minutes}-minute limit.` : 'Batch run: no answer within the tool round limit.')
      .catch(() => undefined)
  }
  const conv = await deps.snapshot(id).catch(() => null)
  // The user can switch models in the middle of a session; its result then says nothing about this run's model.
  const otherModel = conv?.modelId && run.modelId && conv.modelId !== run.modelId ? conv.modelId : null
  patchItem({
    ...(otherModel
      ? { status: 'error' as const, error: `Not counted: the session ran on ${otherModel}, not on this run’s model.` }
      : itemResult(conv, end, { timedOut })),
    ms: deps.now() - workedFrom,
    ...(conv ? sessionStats(conv) : {})
  })
  return { conversationId: id, retry: false }
}

/**
 * Runs the run's pending tasks until none are left or a stop is asked for, then marks the run
 * done or stopped. Never throws: a failure ends the run as stopped and is returned.
 */
export async function runBatchLoop(deps: BatchDeps): Promise<{ run: BatchRun | undefined; error?: string }> {
  const ends = new Map<string, TurnEndReason>()
  const off = deps.onTurnEnd((id, reason) => void ends.set(id, reason))
  let previous: string | null = null
  let first = true
  let error: string | undefined
  try {
    for (;;) {
      if (deps.stopping()) break
      const run = deps.getRun()
      if (!run) break
      const index = run.items.findIndex((i) => i.status === 'pending')
      if (index < 0) break
      await until(deps, () => !!deps.stopping() || deps.chatIdle())
      if (deps.stopping()) break
      const follow = first || (previous !== null && deps.currentChat() === previous)
      const r = await runItem(deps, index, ends, follow)
      if (r.retry) {
        await deps.sleep(500)
        continue
      }
      first = false
      if (r.conversationId) previous = r.conversationId
    }
  } catch (e) {
    error = errorText(e)
  } finally {
    off()
    deps.attended()
  }
  const run = deps.patchRun((r) => {
    // A task cut off by a failure above has not been tried.
    const items = r.items.map((i): BatchItem => (i.status === 'running' ? { ...i, status: 'pending' } : i))
    return { ...r, items, status: items.some((i) => i.status === 'pending') ? 'stopped' : 'done', finishedAt: deps.now() }
  })
  return { run, error }
}
