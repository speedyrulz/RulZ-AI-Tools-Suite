// One unattended RE practice session, run to its end in the main process: the chat store's loop (start, tool
// rounds, approvals, round limit, grading) without a window. The practice pool runs several of these at once
// (docs/vast-design.md, section 8). Everything it touches comes in through SessionDeps, so tests run it against
// a scripted model.
import { DEFAULT_SAMPLING } from '@shared/defaults'
import { isRemoteModelId } from '@shared/remote'
import { REMOTE_DENIAL, UNATTENDED_DENIAL, type TurnEndReason } from '@shared/batch'
import {
  RE_STARTER_TOOLS,
  RE_SYSTEM_PROMPT,
  reFirstMessage,
  remoteRoots,
  reSampling,
  runsWithoutAsking,
  shouldGradePractice
} from '@shared/re'
import {
  availableToolsOf,
  effectiveToolModeOf,
  loadToolNames,
  runLoaderTool,
  tokensOfNames,
  toolBudgetOf,
  toolRequestOf
} from '@shared/toolSession'
import { META_TOOLS, resolveTool } from '@shared/tools'
import type {
  AppSettings,
  ChatMessage,
  ChatRequest,
  Conversation,
  McpServerStatus,
  ReSession,
  ReSettings,
  StreamEvent,
  ToolCall,
  ToolResult
} from '@shared/types'
import { buildWire } from '@shared/wire'

/** Why a session was stopped from outside. */
export type AbortReason = 'stopNow' | 'skip' | 'timeout'

/** What the pool knows of the instance a session is pinned to. */
export type InstanceState = 'ready' | 'reconnecting' | 'gone'

/** The settings a session reads (live, at each use, as the chat store reads them). */
export type SessionSettings = Pick<
  AppSettings,
  'toolsEnabled' | 'toolMode' | 'toolBudgetTokens' | 'disabledTools' | 'confirmToolCalls' | 'allowedTools' | 'defaultSampling' | 'contextOverflow' | 'compactToolResults' | 'keepFullToolResults'
> & { re: Pick<ReSettings, 'maxToolRounds' | 'toolBudgetTokens' | 'kitDir' | 'autoApproveStatic' | 'collect'> }

/** ChatService, as a session uses it. */
export interface SessionChat {
  /** Runs one request. Its events arrive through `onEvent`; resolves when the request is over. */
  start(req: ChatRequest, onEvent: (e: StreamEvent) => void): Promise<void>
  abort(requestId: string): void
  /** The model is no longer needed for the tool round that followed this request. */
  releaseTurn(requestId: string): void
}

export interface SessionDeps {
  now(): number
  sleep(ms: number): Promise<void>
  uid(): string
  settings(): SessionSettings
  /** The shared MCP connections' status (tools by server). */
  servers(): McpServerStatus[]
  /** The instance the session is pinned to, if it still exists. */
  instance(instanceId: string): { hasVision: boolean } | undefined
  instanceState(instanceId: string): InstanceState
  chat: SessionChat
  /** Runs a tool call on the lane's MCP scope. */
  callTool(call: { name: string; arguments: string }, signal: AbortSignal): Promise<ToolResult>
  grade(kitDir: string, taskId: string, target: string, finalText: string): Promise<{ answer: string | null; passed: boolean; expected: string }>
  /** Writes the chat (it carries `poolRev`). Saves of one session arrive in order, one at a time. */
  save(conversation: Conversation, poolRev: number): Promise<void>
  /** The chat as it is now, for a window that watches it. Called at most 4 times a second. */
  onLive(conversation: Conversation): void
  watched(conversationId: string): boolean
  /** A save failed, or something else worth a line in the log. */
  warn?(message: string): void
}

export interface SessionSpec {
  /** Chosen by the caller so it can own the chat from the first moment. */
  conversationId: string
  target: string
  goal: string
  modelId: string | null
  /** The instance every request and round is pinned to. */
  instanceId: string
  practice: NonNullable<ReSession['practice']>
  /** 0 = no limit. Counted from the first request: loading is not the model's working time. */
  minutesPerTask: number
  collect: boolean
  /** The instance is a rented one: its tool calls are held to the task's folder. A remote model id says so too. */
  remote?: boolean
  /** Aborting it with 'stopNow', 'skip' or 'timeout' as the reason stops the session. */
  signal: AbortSignal
  /** Rounds of tool calls before the session must answer; default settings.re.maxToolRounds. */
  maxToolRounds?: number
}

export interface SessionResult {
  conversation: Conversation
  end: TurnEndReason
  timedOut: boolean
  /** The instance could not be reached for 3 minutes (or is gone): the task has not been tried. */
  lost: boolean
  aborted: AbortReason | null
  /** When the first request went out (epoch ms); null if none did. The task's working time counts from here. */
  replyAt: number | null
  /** The chat's last save failed (after retries): what is on disk is older than the session. */
  saveError?: string
}

/** A failed save is tried again this many times, with these waits (ms) in between. */
export const SAVE_RETRY_WAITS_MS: readonly number[] = [500, 1500, 4000]

/** How long a round is retried while the instance is unreachable. */
export const TRANSPORT_RETRY_MS = 180_000
const TRANSPORT_PAUSE_MS = 2000
const RECONNECT_POLL_MS = 1000
const LIVE_INTERVAL_MS = 250

const errorText = (e: unknown): string => (e instanceof Error ? e.message : String(e))

/** How one request to the model ended. */
type Round =
  | { kind: 'final'; messageId: string; stopped: boolean }
  | { kind: 'tools'; messageId: string; calls: ToolCall[]; requestId: string }
  | { kind: 'error'; messageId: string }
  | { kind: 'transport'; messageId: string }

function abortReasonOf(signal: AbortSignal): AbortReason {
  const r: unknown = signal.reason
  return r === 'stopNow' || r === 'skip' || r === 'timeout' ? r : 'skip'
}

export async function runSession(deps: SessionDeps, spec: SessionSpec): Promise<SessionResult> {
  const first = deps.settings()
  const starterTools = availableToolsOf(deps.servers(), first, { clone: false, practice: true })
  const starter = RE_STARTER_TOOLS.flatMap((n) => {
    const t = resolveTool(starterTools, n)
    return t ? [t.qualifiedName] : []
  })
  const startedAt = deps.now()
  let conv: Conversation = {
    id: spec.conversationId,
    title: `Practice: ${spec.practice.taskId}`.slice(0, 60),
    folder: null,
    pinned: false,
    createdAt: startedAt,
    updatedAt: startedAt,
    modelId: spec.modelId,
    systemPrompt: RE_SYSTEM_PROMPT,
    sampling: reSampling(first.defaultSampling ?? DEFAULT_SAMPLING),
    adapters: null,
    presetId: null,
    activeTools: starter,
    re: { target: spec.target, goal: spec.goal, collect: spec.collect, outcome: 'unrated', practice: spec.practice, clone: undefined },
    messages: [{ id: deps.uid(), role: 'user', content: reFirstMessage(spec.goal, spec.target), attachments: undefined, createdAt: startedAt }]
  }

  // ---- stop, time limit ----
  const internal = new AbortController()
  let abortReason: AbortReason | null = null
  /** The final reply is being graded: it beat the limit and the stop, so both are left alone. */
  let grading = false
  let finished = false
  let currentRequest: string | null = null
  let limitAt: number | null = null
  let replyAt: number | null = null
  let limitTimer: ReturnType<typeof setTimeout> | null = null
  let timedOut = false

  const halt = (reason: AbortReason): void => {
    if (abortReason || grading || finished) return
    abortReason = reason
    if (reason === 'timeout') timedOut = true
    internal.abort()
    if (currentRequest) deps.chat.abort(currentRequest)
  }
  const armLimit = (): void => {
    if (limitTimer) clearTimeout(limitTimer)
    limitTimer = null
    if (limitAt === null || finished) return
    limitTimer = setTimeout(() => halt('timeout'), Math.max(0, limitAt - deps.now()))
    limitTimer.unref?.()
  }
  const onSpecAbort = (): void => halt(abortReasonOf(spec.signal))
  if (spec.signal.aborted) onSpecAbort()
  else spec.signal.addEventListener('abort', onSpecAbort, { once: true })

  // ---- saving and live updates ----
  let rev = conv.poolRev ?? 0
  let saveChain: Promise<void> = Promise.resolve()
  /** Set while the newest save has failed for good; cleared when a later one is written. */
  let saveError: string | null = null
  /** Writes one revision, trying again after a failure; a newer revision queued meanwhile supersedes a failed one. */
  const writeSave = async (copy: Conversation, n: number): Promise<void> => {
    for (let attempt = 0; ; attempt++) {
      try {
        await deps.save(copy, n)
        saveError = null
        return
      } catch (e) {
        // The next revision holds everything this one did: it takes over.
        if (rev > n) return
        if (attempt >= SAVE_RETRY_WAITS_MS.length) {
          saveError = errorText(e)
          deps.warn?.(`Could not save the practice chat ${conv.id}: ${errorText(e)}`)
          return
        }
        await deps.sleep(SAVE_RETRY_WAITS_MS[attempt])
      }
    }
  }
  /** Saves `c` as the chat's next revision. Saves go out one after another, in order. */
  const save = (c: Conversation): void => {
    rev++
    const copy: Conversation = { ...c, poolRev: rev }
    const n = rev
    conv = { ...conv, poolRev: n }
    saveChain = saveChain.then(() => writeSave(copy, n))
  }
  let liveAt = 0
  const live = (force: boolean): void => {
    if (!deps.watched(conv.id)) return
    const t = deps.now()
    if (!force && t - liveAt < LIVE_INTERVAL_MS) return
    liveAt = t
    flush()
    try {
      deps.onLive(conv)
    } catch (e) {
      deps.warn?.(`A live update failed: ${errorText(e)}`)
    }
  }

  // ---- the assistant message being written ----
  let pendingContent = ''
  let pendingReasoning = ''
  const patchMessage = (id: string, fn: (m: ChatMessage) => ChatMessage): void => {
    conv = { ...conv, messages: conv.messages.map((m) => (m.id === id ? fn(m) : m)) }
  }
  let streamingId: string | null = null
  function flush(): void {
    if (!streamingId || (!pendingContent && !pendingReasoning)) return
    const content = pendingContent
    const reasoning = pendingReasoning
    pendingContent = ''
    pendingReasoning = ''
    patchMessage(streamingId, (m) => ({ ...m, content: m.content + content, reasoning: reasoning ? (m.reasoning ?? '') + reasoning : m.reasoning }))
  }
  const updateCall = (messageId: string, id: string, patch: Partial<ToolCall>): void =>
    patchMessage(messageId, (m) => ({ ...m, toolCalls: m.toolCalls?.map((c) => (c.id === id ? { ...c, ...patch } : c)) }))

  // ---- one round: a request to the model ----
  let held: string | null = null
  const releaseHeld = (): void => {
    if (!held) return
    const id = held
    held = null
    deps.chat.releaseTurn(id)
  }

  const runRound = async (): Promise<Round> => {
    const s = deps.settings()
    const servers = deps.servers()
    const requestId = deps.uid()
    const messageId = deps.uid()
    const history = conv.messages
    conv = {
      ...conv,
      messages: [...history, { id: messageId, role: 'assistant', content: '', createdAt: deps.now() }],
      modelId: spec.modelId,
      updatedAt: deps.now()
    }
    // The new request is the session's work from here on: the limit counts from the first one.
    replyAt ??= deps.now()
    if (limitAt === null && spec.minutesPerTask > 0) {
      limitAt = deps.now() + spec.minutesPerTask * 60_000
      armLimit()
    }
    // The chat as the model is about to see it, saved so the chat list is current during a long reply.
    save({ ...conv, messages: conv.messages.filter((m) => m.id !== messageId || m.content) })
    pendingContent = ''
    pendingReasoning = ''
    streamingId = messageId
    live(true)

    const hasVision = deps.instance(spec.instanceId)?.hasVision ?? false
    const req: ChatRequest = {
      requestId,
      modelId: spec.modelId ?? '',
      instanceId: spec.instanceId,
      messages: buildWire(history, conv.systemPrompt, hasVision, s.compactToolResults ? s.keepFullToolResults : null, true),
      sampling: conv.sampling,
      adapters: conv.adapters ?? null,
      overflow: s.contextOverflow ?? 'truncateMiddle',
      tools: toolRequestOf(servers, s, conv.activeTools, { re: true, practice: true }),
      record: conv.re?.collect ? { conversationId: conv.id } : undefined
    }
    currentRequest = requestId
    const end = await new Promise<StreamEvent>((resolve) => {
      let settled = false
      const settle = (e: StreamEvent): void => {
        if (settled) return
        settled = true
        resolve(e)
      }
      const onEvent = (e: StreamEvent): void => {
        if (e.requestId !== requestId || settled) return
        if (e.type === 'delta') {
          if (e.content) pendingContent += e.content
          if (e.reasoning) pendingReasoning += e.reasoning
          live(false)
        } else if (e.type === 'done' || e.type === 'error') {
          // A final reply that is in beats the limit and the stop: it is graded, not cut off.
          if (e.type === 'done' && !e.toolCalls?.length && e.stats.stopReason !== 'stopped' && !abortReason) grading = true
          settle(e)
        }
      }
      deps.chat.start(req, onEvent).then(
        () => settle({ type: 'error', requestId, error: 'The reply ended without a result.' }),
        (e) => settle({ type: 'error', requestId, error: errorText(e) })
      )
      if (abortReason) deps.chat.abort(requestId)
    })
    currentRequest = null
    // The tools of the round before have been run and this round has started (or is over).
    releaseHeld()
    flush()
    streamingId = null

    if (end.type === 'error') {
      if (end.transport) return { kind: 'transport', messageId }
      patchMessage(messageId, (m) => ({ ...m, stats: end.stats, error: end.error }))
      return { kind: 'error', messageId }
    }
    if (end.type !== 'done') return { kind: 'error', messageId }
    if (end.toolCalls?.length && end.stats.stopReason !== 'stopped') {
      const calls: ToolCall[] = end.toolCalls.map((c) => ({ id: c.id, name: c.function.name, arguments: c.function.arguments, status: 'pending' }))
      patchMessage(messageId, (m) => ({ ...m, stats: end.stats, reasoningMs: end.reasoningMs ?? m.reasoningMs, toolCalls: calls }))
      // The model stays in use until the next round starts.
      held = requestId
      live(true)
      return { kind: 'tools', messageId, calls, requestId }
    }
    patchMessage(messageId, (m) => ({ ...m, stats: end.stats, reasoningMs: end.reasoningMs ?? m.reasoningMs, error: undefined }))
    return { kind: 'final', messageId, stopped: end.stats.stopReason === 'stopped' }
  }

  // ---- tools ----
  // Decided once, not read off the instance, which may be gone by the time a call is checked.
  const remote = spec.remote === true || isRemoteModelId(spec.modelId ?? '')
  /** Marks calls that have not finished as stopped. */
  const stopCalls = (messageId: string): void =>
    patchMessage(messageId, (m) => ({
      ...m,
      toolCalls: m.toolCalls?.map((c) => (c.status === 'pending' || c.status === 'running' ? { ...c, status: 'denied', result: 'Stopped by the user.' } : c))
    }))

  /** Runs the round's calls. False when a stop cut it short. */
  const runTools = async (messageId: string, calls: ToolCall[]): Promise<boolean> => {
    for (let call of calls) {
      if (abortReason) {
        stopCalls(messageId)
        return false
      }
      const s = deps.settings()
      const servers = deps.servers()
      const tools = availableToolsOf(servers, s, { clone: false, practice: true })
      const tokensOf = (names: string[]): number => tokensOfNames(servers, names)
      const budget = toolBudgetOf(s, { re: true })
      if (META_TOOLS.has(call.name)) {
        const r = runLoaderTool(call, tools, conv.activeTools ?? [], budget, tokensOf)
        conv = { ...conv, activeTools: r.active }
        updateCall(messageId, call.id, { status: r.result.isError ? 'error' : 'done', result: r.result.text })
        live(false)
        continue
      }
      // The model may call a tool by its bare name; map it to the real one.
      const target = resolveTool(tools, call.name)
      if (!target) {
        updateCall(messageId, call.id, { status: 'error', result: `Tool "${call.name}" is not available or is switched off.` })
        continue
      }
      if (target.qualifiedName !== call.name) {
        call = { ...call, name: target.qualifiedName }
        updateCall(messageId, call.id, { name: call.name })
      }
      // Calling a tool keeps it loaded and marks it recently used.
      if (effectiveToolModeOf(servers, s, { re: true, practice: true }) === 'dynamic') {
        conv = { ...conv, activeTools: loadToolNames(conv.activeTools ?? [], [target.qualifiedName], budget, tokensOf).active }
      }
      // Static analysis runs without asking; anything else is refused, because nobody is there to ask.
      if (!runsWithoutAsking(s, target, true, remote ? { args: call.arguments, roots: remoteRoots(conv.re), rea: 'private' } : undefined)) {
        updateCall(messageId, call.id, { status: 'denied', result: remote ? REMOTE_DENIAL : UNATTENDED_DENIAL })
        continue
      }
      updateCall(messageId, call.id, { status: 'running' })
      live(false)
      const r = await deps.callTool({ name: call.name, arguments: call.arguments }, internal.signal).catch((e): ToolResult => ({ text: errorText(e), isError: true }))
      if (abortReason) {
        stopCalls(messageId)
        return false
      }
      updateCall(messageId, call.id, { status: r.isError ? 'error' : 'done', result: r.text })
      live(false)
    }
    return true
  }

  /** Tool rounds the model made in a row at the end of the chat. */
  const trailingRounds = (): number => {
    let rounds = 0
    for (let i = conv.messages.length - 1; i >= 0 && conv.messages[i].role === 'assistant'; i--) {
      if (conv.messages[i].toolCalls?.length) rounds++
    }
    return rounds
  }

  // ---- grading ----
  const grade = async (messageId: string): Promise<void> => {
    const re = conv.re
    const practice = re?.practice
    const text = conv.messages.find((m) => m.id === messageId)?.content ?? ''
    if (!re || !practice || !shouldGradePractice(re, text)) return
    const kitDir = practice.kitDir || deps.settings().re.kitDir
    try {
      const r = await deps.grade(kitDir, practice.taskId, re.target, text)
      conv = {
        ...conv,
        re: { ...re, outcome: r.passed ? 'solved' : 'failed', autoGraded: true, practice: { ...practice, answer: r.answer ?? undefined, expected: r.expected } }
      }
      save(conv)
    } catch (e) {
      // Left ungraded: the item then reads "could not be graded".
      deps.warn?.(`Could not grade practice ${practice.taskId}: ${errorText(e)}`)
    }
  }

  /** A session that ran out of rounds or time failed, unless something graded it. */
  const failUngraded = (notes: string): void => {
    if (conv.re?.practice && conv.re.outcome === 'unrated') {
      conv = { ...conv, re: { ...conv.re, outcome: 'failed', autoGraded: true, notes: conv.re.notes ?? notes } }
      save(conv)
    }
  }

  // ---- waiting for an unreachable instance ----
  /** 'retry' when the round may be sent again, 'lost' when the instance did not come back in time, 'aborted'. */
  const waitForInstance = async (since: number): Promise<'retry' | 'lost' | 'aborted'> => {
    const waitStart = deps.now()
    // The model is not working while it waits: the limit stands still.
    if (limitTimer) clearTimeout(limitTimer)
    limitTimer = null
    let outcome: 'retry' | 'lost' | 'aborted'
    for (;;) {
      if (abortReason) {
        outcome = 'aborted'
        break
      }
      const state = deps.instanceState(spec.instanceId)
      if (state === 'gone' || deps.now() - since >= TRANSPORT_RETRY_MS) {
        outcome = 'lost'
        break
      }
      if (state === 'ready') {
        // A short pause, so a server that keeps refusing is not asked a hundred times a second.
        await deps.sleep(TRANSPORT_PAUSE_MS)
        outcome = 'retry'
        break
      }
      await deps.sleep(RECONNECT_POLL_MS)
    }
    // The model was not working meanwhile: the time limit does not run.
    if (limitAt !== null) {
      limitAt += deps.now() - waitStart
      armLimit()
    }
    return outcome
  }

  // ---- the session ----
  let end: TurnEndReason = 'error'
  let lost = false
  let transportSince: number | null = null
  const maxRounds = spec.maxToolRounds ?? first.re.maxToolRounds ?? 80
  try {
    if (abortReason) {
      end = 'stopped'
    } else {
      for (;;) {
        if (abortReason) {
          end = 'stopped'
          break
        }
        const r = await runRound()
        if (r.kind === 'transport') {
          // Not the model's fault: the partial reply is dropped and the round sent again.
          conv = { ...conv, messages: conv.messages.filter((m) => m.id !== r.messageId) }
          transportSince ??= deps.now()
          const w = await waitForInstance(transportSince)
          if (w === 'retry') continue
          if (w === 'lost') {
            lost = true
            end = 'error'
          } else end = 'stopped'
          break
        }
        transportSince = null
        if (r.kind === 'error') {
          end = 'error'
          conv = { ...conv, updatedAt: deps.now() }
          save(conv)
          live(true)
          break
        }
        if (r.kind === 'final') {
          conv = { ...conv, updatedAt: deps.now() }
          save(conv)
          if (r.stopped) {
            end = 'stopped'
            live(true)
            break
          }
          // Graded before the session counts as over; a stop or limit that comes now is ignored.
          grading = true
          try {
            await grade(r.messageId)
          } finally {
            grading = false
          }
          end = 'done'
          live(true)
          break
        }
        // Tool calls.
        let toolsDone: boolean
        try {
          toolsDone = await runTools(r.messageId, r.calls)
        } catch (e) {
          // A failure in the round must not leave the session hanging.
          const error = errorText(e)
          patchMessage(r.messageId, (m) => ({
            ...m,
            error,
            toolCalls: m.toolCalls?.map((c) => (c.status === 'pending' || c.status === 'running' ? { ...c, status: 'error', result: `Not run: ${error}` } : c))
          }))
          save(conv)
          end = 'error'
          break
        }
        if (!toolsDone) {
          save(conv)
          end = 'stopped'
          live(true)
          break
        }
        if (trailingRounds() >= maxRounds) {
          save(conv)
          end = 'rounds'
          live(true)
          break
        }
        // The next round saves the chat with these results.
      }
    }
  } catch (e) {
    // Nothing above should throw; if it does the session ends as an error with the reason on its last reply.
    const error = errorText(e)
    const lastAssistant = conv.messages.findLast((m) => m.role === 'assistant')
    if (lastAssistant) patchMessage(lastAssistant.id, (m) => ({ ...m, error }))
    save(conv)
    end = 'error'
  }

  releaseHeld()
  finished = true
  if (limitTimer) clearTimeout(limitTimer)
  spec.signal.removeEventListener('abort', onSpecAbort)
  if (!lost && (timedOut || end === 'rounds')) {
    failUngraded(timedOut ? `Batch run: stopped at the ${spec.minutesPerTask}-minute limit.` : 'Batch run: no answer within the tool round limit.')
  }
  await saveChain
  return { conversation: conv, end, timedOut, lost, aborted: abortReason, replyAt, ...(saveError ? { saveError } : {}) }
}
