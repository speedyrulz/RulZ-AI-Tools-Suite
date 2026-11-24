import { create } from 'zustand'
import { toast } from 'sonner'
import type {
  Attachment,
  ChatMessage,
  Conversation,
  FileToolScope,
  ReClone,
  ConversationSummary,
  Preset,
  ReOutcome,
  ReSession,
  SamplingParams,
  StreamEvent,
  ToolCall,
  ToolResult,
  WireMessage
} from '@shared/types'
import { DEFAULT_SAMPLING } from '@shared/defaults'
import { type AllModeDefaults, modePromptExtra, modeSampling, withPromptExtra } from '@shared/modes'
import { errorMessage, invoke, on } from '../lib/api'
import { uid } from '../lib/format'
import { type QueuedNote, takeQueued } from '../lib/interject'
import { editMessages, idsToDelete, removeMessages, restoreMessages } from '../lib/messages'
import { chatDefaultsFor, followChatDefaults, isFollowable, modesFor } from '../lib/modes'
import { once } from '../lib/once'
import { WBENCH_BUSY, wbenchRunning } from '../lib/wbenchFlag'
import { buildWire } from '../lib/wire'
import { isRemoteModelId } from '@shared/remote'
import { useApp } from './app'
import { availableTools, effectiveToolMode, tokensOf, toolBudget, toolRequestFor } from './mcp'
import {
  CLONE_STARTER_TOOLS,
  cloneFirstMessage,
  clonePrompt,
  runsWithoutAsking,
  RE_MIN_CONTEXT,
  RE_STARTER_TOOLS,
  RE_SYSTEM_PROMPT,
  reFirstMessage,
  remoteRoots,
  shouldGradePractice,
  unquotePath
} from '@shared/re'
import { modeDefaultsOf, readyInstance, useModels } from './models'
import { META_TOOLS, resolveTool } from '@shared/tools'
import { loadToolNames, runLoaderTool } from '@shared/toolSession'
import { isStaleChatError } from '@shared/batch'

export const NEW_CHAT_TITLE = 'New Chat'

interface Generation {
  requestId: string
  conversationId: string
  messageId: string
  promptProgress: number | null
  /** 'tools' while the model's tool calls are being approved and run. */
  phase?: 'generating' | 'tools'
}

/** Stops runaway loops where the model keeps calling tools without answering. */
const MAX_TOOL_ROUNDS = 12

export type ToolDecision = 'deny' | 'once' | 'always'

export interface ToolApproval {
  call: ToolCall
  /** The model runs on a rented instance: "always" then means in this chat, until the app closes. */
  remote: boolean
  resolve: (decision: ToolDecision) => void
}

/** Tools the user allowed for one chat with a model on a rented instance ("<chat id> <tool>"); never saved. */
const allowedInChat = new Set<string>()
/** The chat whose program the shared REA process has open; null = not known (nothing yet, an open failed, another open tool or a practice lane ran). */
let reaOpener: string | null = null

/** Share of the context window at which the user is offered a handoff. */
export const HANDOFF_THRESHOLD = 0.85

const HANDOFF_PROMPT = `This conversation is about to continue in a new chat with a fresh context window, and you will not see anything above this message there.

Write a detailed handoff message that will be sent as the first message of the new chat. Write it in the first person, as me (the user), so it can be sent exactly as written. Include:
- the overall goal and the context behind it;
- key facts, decisions, preferences and constraints we have established;
- important specifics verbatim where they matter (names, numbers, code, file names, wording);
- what has been done so far and the current state;
- open questions, and what I want to happen next.

Be thorough but do not pad. Output only the handoff message, with no preamble.`

export interface HandoffState {
  requestId: string
  /** Snapshot of the chat being handed off; the new chat copies its settings. */
  source: Conversation
  modelId: string
  text: string
  status: 'loading' | 'writing' | 'error'
  promptProgress: number | null
  error?: string
}

interface ChatState {
  loaded: boolean
  summaries: ConversationSummary[]
  folders: string[]
  presets: Preset[]
  current: Conversation | null
  /** Conversation being generated into; can differ from `current` if the user switched chats. */
  genConversation: Conversation | null
  generation: Generation | null
  /** Model being loaded automatically before a send. */
  autoLoadingModelId: string | null
  tokenCount: number | null
  handoff: HandoffState | null
  /** Which handoff confirmation is open: from the button, or the automatic context warning. */
  handoffConfirm: 'manual' | 'auto' | null
  /** Tool call waiting for the user's approval. */
  toolApproval: ToolApproval | null
  /** Notes written while a reply streamed, sent one by one as replies end. */
  queued: QueuedNote[]
  /** Chats the practice pool is running now (the main process writes them; they are read-only here). */
  poolChats: string[]

  init: () => Promise<void>
  newChat: () => Promise<Conversation>
  open: (id: string) => Promise<void>
  mutateCurrent: (fn: (c: Conversation) => Conversation) => void
  send: (text: string, attachments: Attachment[]) => Promise<void>
  /** Stops the reply streaming in the current chat, keeps what it had, and sends the note right away. */
  sendNow: (text: string, attachments: Attachment[]) => void
  /** Queues a note to send once the current chat's reply ends. */
  queueNote: (text: string, attachments: Attachment[]) => void
  cancelQueued: (id: string) => void
  /** Sends a queued note now: interrupts the reply if one is streaming. */
  sendQueued: (id: string) => void
  regenerate: (messageId: string) => Promise<void>
  continueMessage: (messageId: string) => Promise<void>
  editMessage: (messageId: string, content: string, resend: boolean) => Promise<void>
  deleteMessage: (messageId: string) => void
  branchFrom: (messageId: string) => Promise<void>
  stop: () => void
  rename: (id: string, title: string) => Promise<void>
  togglePin: (id: string) => Promise<void>
  moveToFolder: (id: string, folder: string | null) => Promise<void>
  duplicate: (id: string) => Promise<void>
  remove: (id: string) => Promise<void>
  addFolder: (name: string) => Promise<void>
  removeFolder: (name: string) => Promise<void>
  savePreset: (name: string) => Promise<void>
  applyPreset: (id: string | null) => void
  deletePreset: (id: string) => Promise<void>
  refreshTokenCount: () => void
  setHandoffConfirm: (mode: 'manual' | 'auto' | null) => void
  dismissHandoffPrompt: () => void
  startHandoff: () => Promise<void>
  cancelHandoff: () => void
  /**
   * Starts a reverse-engineering session in a new chat and sends its first message. Returns the
   * chat's id, or null if nothing was started because another reply is still running.
   */
  startRe: (opts: StartReOptions) => Promise<string | null>
  setReOutcome: (id: string, outcome: ReOutcome, notes?: string) => Promise<void>
  setReCollect: (id: string, collect: boolean) => Promise<void>
}

/** Is the practice pool writing this chat? (A selector for the UI: `useChat((s) => isPoolChat(s, id))`.) */
export function isPoolChat(s: { poolChats: string[] }, id: string | null | undefined): boolean {
  return !!id && s.poolChats.includes(id)
}

export interface StartReOptions {
  target: string
  /** Rebuild part or all of the target as new code (a clone session). */
  clone?: ReClone
  /** First message; {target} is replaced by the target path. */
  goal: string
  collect: boolean
  /** Model to run it on; null = the loaded model. */
  modelId: string | null
  practice?: ReSession['practice']
  title?: string
}

function makeTitle(text: string): string {
  const line = text.trim().split(/\r?\n/)[0].replace(/\s+/g, ' ')
  return line.length > 48 ? `${line.slice(0, 47)}…` : line || NEW_CHAT_TITLE
}

/** How many recent tool results to keep whole, or null to send every result in full. */
function keepFullToolResults(): number | null {
  const s = useApp.getState().settings
  return s?.compactToolResults ? s.keepFullToolResults : null
}

/** Converts chat history to the wire format, compacting old tool results as the settings say. */
export function toWire(
  messages: ChatMessage[],
  systemPrompt: string,
  vision: boolean,
  keepFull: number | null = keepFullToolResults(),
  keepReasoning = false
): WireMessage[] {
  return buildWire(messages, systemPrompt, vision, keepFull, keepReasoning)
}

/** Where a clone session's write/read file tools may work; undefined for other chats. */
export function fileScope(c: Conversation | null | undefined): FileToolScope | undefined {
  const clone = c?.re?.clone
  return clone?.outputDir ? { writeRoot: clone.outputDir, target: c!.re!.target } : undefined
}

/** Tool options for a chat: RE sessions use the RE budget, clone sessions get the file tools. */
function toolOpts(c: Conversation | null | undefined): { re: boolean; files?: FileToolScope; practice?: boolean } {
  const files = fileScope(c)
  return { re: !!c?.re, ...(files ? { files } : {}), ...(c?.re?.practice ? { practice: true } : {}) }
}

/** A copy of an RE session's details for a new chat (branch, copy, handoff): it needs grading again. */
function reCopy(re: ReSession | undefined): ReSession | undefined {
  return re ? { ...re, outcome: 'unrated', autoGraded: undefined } : undefined
}

/** Last part of a path. */
function baseName(path: string): string {
  return path.split(/[\\/]/).filter(Boolean).at(-1) ?? path
}

// ---- Persistence (debounced per conversation) ----
const saveTimers = new Map<string, ReturnType<typeof setTimeout>>()
/** Chats deleted this session. A reply or tool call still finishing in one must not save it back. */
const deletedIds = new Set<string>()
/**
 * The newest copy of each chat whose save is waiting (debounced) or still being written. Reading the
 * chat from disk meanwhile would show an older copy, and the next edit would save that over the newer one.
 */
const unsaved = new Map<string, Conversation>()

function upsertSummary(list: ConversationSummary[], s: ConversationSummary): ConversationSummary[] {
  return [s, ...list.filter((x) => x.id !== s.id)].sort((a, b) => b.updatedAt - a.updatedAt)
}

function cancelSave(id: string): void {
  const t = saveTimers.get(id)
  if (t) clearTimeout(t)
  saveTimers.delete(id)
}

/** Chats being re-read from disk after a refused save, so one stale copy makes one reload and no toast flood. */
const reloading = new Set<string>()

/** The window's copy of a chat was refused (a batch run wrote it meanwhile): show what is on disk instead. */
async function reloadStale(id: string): Promise<void> {
  if (reloading.has(id)) return
  reloading.add(id)
  try {
    // Never a copy that is still waiting to be saved: that is what was just refused.
    unsaved.delete(id)
    cancelSave(id)
    if (useChat.getState().current?.id !== id) return
    const fresh = await invoke('conversations:get', id).catch(() => null)
    if (fresh && useChat.getState().current?.id === id && !useChat.getState().generation) {
      liveEpoch++
      useChat.setState({ current: fresh, tokenCount: null })
    }
  } finally {
    reloading.delete(id)
  }
}

async function persistNow(conv: Conversation): Promise<void> {
  cancelSave(conv.id)
  if (deletedIds.has(conv.id)) return
  unsaved.set(conv.id, conv)
  try {
    const summary = await invoke('conversations:save', conv).finally(() => {
      // A newer copy may have been handed over meanwhile; that one is still on its way.
      if (unsaved.get(conv.id) === conv) unsaved.delete(conv.id)
    })
    // Deleted while the save was in flight: keep it out of the list.
    if (deletedIds.has(conv.id)) return
    useChat.setState((s) => ({ summaries: upsertSummary(s.summaries, summary) }))
  } catch (e) {
    // Main refuses saves of deleted chats; that's the intended outcome, not a failure to report.
    if (deletedIds.has(conv.id)) return
    // A batch run owns the chat or wrote it after this copy was made: the run's copy wins, quietly.
    if (isStaleChatError(errorMessage(e))) return void reloadStale(conv.id)
    toast.error(`Could not save chat: ${errorMessage(e)}`)
  }
}

function persistSoon(conv: Conversation): void {
  if (deletedIds.has(conv.id)) return
  cancelSave(conv.id)
  unsaved.set(conv.id, conv)
  saveTimers.set(conv.id, setTimeout(() => void persistNow(conv), 500))
}

/** The pool chat the main process streams to this window (re:watch); null = none. */
let watching: string | null = null

/** Counts the newer copies of the watched chat that reached the window (live updates, reloads); a reply to re:watch older than one is not shown. */
let liveEpoch = 0
/** The newest re:watch request; an answer to an earlier one is dropped. */
let watchSeq = 0

/** Whether a copy of a chat from main may replace what is on screen: it must not be older than it. */
export function isNewerOrSame(incoming: { poolRev?: number }, shown: { poolRev?: number } | null | undefined): boolean {
  return (incoming.poolRev ?? 0) >= (shown?.poolRev ?? 0)
}

/** Tells main which pool chat is on screen: it streams live updates of that one only. */
function syncWatch(): void {
  const { current, poolChats } = useChat.getState()
  const want = current && poolChats.includes(current.id) ? current.id : null
  if (want === watching) return
  watching = want
  const epoch = liveEpoch
  const seq = ++watchSeq
  void invoke('re:watch', want)
    .then((conv) => {
      // The chat as main has it now (the window's copy may be older than the run's last save), unless this
      // answer is stale: another chat is watched since, a live update or reload came first, or it is older.
      if (!want || !conv || watching !== want || seq !== watchSeq || epoch !== liveEpoch) return
      const shown = useChat.getState().current
      if (shown?.id === want && isNewerOrSame(conv, shown)) useChat.setState({ current: conv })
    })
    .catch(() => undefined)
}

/** A reply is streaming, running tools, or waiting for its model to load. */
export function isChatBusy(s: { generation: unknown; autoLoadingModelId: string | null }): boolean {
  return !!s.generation || !!s.autoLoadingModelId
}

// ---- Streaming deltas are batched: every update re-renders the message's Markdown, so ~11 per second ----
const FLUSH_MS = 90
let pending = { content: '', reasoning: '' }
let flushQueued = false

function applyToGenMessage(fn: (m: ChatMessage) => ChatMessage): void {
  const { genConversation, generation, current } = useChat.getState()
  if (!genConversation || !generation) return
  const next: Conversation = {
    ...genConversation,
    messages: genConversation.messages.map((m) => (m.id === generation.messageId ? fn(m) : m))
  }
  useChat.setState({ genConversation: next, current: current?.id === next.id ? next : current })
}

function flushPending(): void {
  flushQueued = false
  if (!pending.content && !pending.reasoning) return
  const { content, reasoning } = pending
  pending = { content: '', reasoning: '' }
  applyToGenMessage((m) => ({
    ...m,
    content: m.content + content,
    reasoning: reasoning ? (m.reasoning ?? '') + reasoning : m.reasoning
  }))
}

let tokenTimer: ReturnType<typeof setTimeout> | null = null

/** A "Send now" note waiting for its interrupted reply to finish stopping. */
let interject: { requestId: string; text: string; attachments: Attachment[] } | null = null

function markInterrupted(conv: Conversation, messageId: string): Conversation {
  return { ...conv, messages: conv.messages.map((m) => (m.id === messageId ? { ...m, interrupted: true, error: undefined } : m)) }
}

export const useChat = create<ChatState>((set, get) => {
  const handleHandoffEvent = (ho: HandoffState, e: StreamEvent): void => {
    switch (e.type) {
      case 'progress':
        set({ handoff: { ...ho, promptProgress: e.total ? e.processed / e.total : null } })
        break
      case 'delta':
        if (e.content) set({ handoff: { ...ho, text: ho.text + e.content, promptProgress: null } })
        break
      case 'error':
        set({ handoff: { ...ho, status: 'error', error: e.error } })
        break
      case 'done': {
        if (e.stats.stopReason === 'stopped') return set({ handoff: null })
        // Some models put their reasoning inline; keep only the final answer.
        const text = ho.text.replace(/^\s*<think>[\s\S]*?<\/think>/, '').trim()
        if (!text) return set({ handoff: { ...ho, status: 'error', error: 'The model returned an empty handoff message.' } })
        void finishHandoff(ho, text)
        break
      }
    }
  }

  /** Opens a new chat with the source chat's settings and sends the handoff message in it. */
  const finishHandoff = async (ho: HandoffState, text: string): Promise<void> => {
    const src = ho.source
    const now = Date.now()
    const conv: Conversation = {
      id: uid(),
      title: `Handoff: ${src.title}`.slice(0, 60),
      folder: src.folder,
      pinned: false,
      createdAt: now,
      updatedAt: now,
      modelId: ho.modelId,
      systemPrompt: src.systemPrompt,
      sampling: src.sampling,
      adapters: src.adapters ?? null,
      presetId: src.presetId,
      activeTools: src.activeTools,
      re: reCopy(src.re),
      messages: []
    }
    await withConversation(src.id, (c) => ({ ...c, handoffPrompted: true }))
    set({ handoff: null, current: conv, tokenCount: null })
    await persistNow(conv)
    toast.success('Handed off to a new chat')
    await get().send(text, [])
  }

  const handleEvent = (e: StreamEvent): void => {
    // Main keeps the model in use after a reply that asked for tools; a reply whose tools won't
    // run here lets it go at once.
    const releaseUnused = (): void => {
      if (e.type === 'done' && e.toolCalls?.length) void invoke('chat:releaseTurn', e.requestId)
    }
    const ho = get().handoff
    if (ho && e.requestId === ho.requestId) {
      releaseUnused()
      return handleHandoffEvent(ho, e)
    }
    const gen = get().generation
    if (!gen || e.requestId !== gen.requestId) return releaseUnused()
    if (e.type === 'done' && e.toolCalls?.length && e.stats.stopReason !== 'stopped') {
      flushPending()
      const calls: ToolCall[] = e.toolCalls.map((c) => ({
        id: c.id,
        name: c.function.name,
        arguments: c.function.arguments,
        status: 'pending'
      }))
      applyToGenMessage((m) => ({ ...m, stats: e.stats, reasoningMs: e.reasoningMs ?? m.reasoningMs, toolCalls: calls }))
      set({ generation: { ...gen, phase: 'tools', promptProgress: null } })
      void runTools(gen.requestId, calls)
      return
    }
    switch (e.type) {
      case 'progress':
        set({ generation: { ...gen, promptProgress: e.total ? e.processed / e.total : null } })
        break
      case 'delta':
        if (e.content) pending.content += e.content
        if (e.reasoning) pending.reasoning += e.reasoning
        if (gen.promptProgress !== null) set({ generation: { ...gen, promptProgress: null } })
        if (!flushQueued) {
          flushQueued = true
          setTimeout(flushPending, FLUSH_MS)
        }
        break
      case 'done':
      case 'error': {
        flushPending()
        applyToGenMessage((m) => ({
          ...m,
          stats: e.stats,
          reasoningMs: e.type === 'done' ? (e.reasoningMs ?? m.reasoningMs) : m.reasoningMs,
          error: e.type === 'error' ? e.error : undefined
        }))
        const conv = get().genConversation
        set({ generation: null, genConversation: null })
        const note = interject?.requestId === gen.requestId ? interject : null
        interject = null
        if (conv && note) {
          void appendAndGenerate(markInterrupted(conv, gen.messageId), note.text, note.attachments)
          break
        }
        if (conv) void persistNow({ ...conv, updatedAt: Date.now() })
        get().refreshTokenCount()
        if (conv && e.type === 'done' && e.stats.stopReason !== 'stopped') {
          if (conv.re?.practice) void gradePractice(conv, gen.messageId)
          sendNextQueued(conv.id)
        }
        break
      }
    }
  }

  const updateCall = (id: string, patch: Partial<ToolCall>): void =>
    applyToGenMessage((m) => ({ ...m, toolCalls: m.toolCalls?.map((c) => (c.id === id ? { ...c, ...patch } : c)) }))

  /** Keys (`requestId:callId`) of MCP tool calls in flight, so Stop can cancel them in main. */
  const runningToolCalls = new Set<string>()

  /** True while the tool phase started by `requestId` is still the active one (not stopped). */
  const inToolPhase = (requestId: string): boolean => {
    const g = get().generation
    return g?.requestId === requestId && g.phase === 'tools'
  }

  /** Updates the loaded-tools list of the conversation being generated into. */
  const setActiveTools = (fn: (active: string[]) => string[]): void => {
    const g = get().genConversation
    if (!g) return
    const next = { ...g, activeTools: fn(g.activeTools ?? []) }
    set({ genConversation: next, current: get().current?.id === next.id ? next : get().current })
  }

  /** Marks tools as loaded and most recently used, then evicts the least recently used over budget. */
  const loadTools = (names: string[]): string[] => {
    const budget = toolBudget({ re: !!get().genConversation?.re })
    let evicted: string[] = []
    setActiveTools((active) => {
      const r = loadToolNames(active, names, budget, tokensOf)
      evicted = r.evicted
      return r.active
    })
    return evicted
  }

  /** Runs one of the built-in loader tools; these change chat state only, so they need no approval. */
  const runMetaTool = (call: ToolCall): ToolResult => {
    const gen = get().genConversation
    const tools = availableTools({ clone: !!fileScope(gen), practice: !!gen?.re?.practice })
    const active = gen?.activeTools ?? []
    const r = runLoaderTool(call, tools, active, toolBudget({ re: !!gen?.re }), tokensOf)
    if (r.active !== active) setActiveTools(() => r.active)
    return r.result
  }

  /** Approves and runs each tool call, then continues the reply with the results. */
  const runTools = async (requestId: string, calls: ToolCall[]): Promise<void> => {
    try {
      await runToolRound(requestId, calls)
    } catch (e) {
      // A failure in the round must not leave the chat stuck in the tools phase, showing Stop.
      if (inToolPhase(requestId)) {
        const error = errorMessage(e)
        applyToGenMessage((m) => ({
          ...m,
          error,
          toolCalls: m.toolCalls?.map((c) =>
            c.status === 'pending' || c.status === 'running' ? { ...c, status: 'error', result: `Not run: ${error}` } : c
          )
        }))
        const conv = get().genConversation
        set({ generation: null, genConversation: null, toolApproval: null })
        if (conv) void persistNow(conv)
      } else {
        toast.error(`The tool round failed: ${errorMessage(e)}`)
      }
    } finally {
      // By now the next round has started (and holds the model itself) or the turn has ended.
      void invoke('chat:releaseTurn', requestId)
    }
  }

  const runToolRound = async (requestId: string, calls: ToolCall[]): Promise<void> => {
    for (let call of calls) {
      if (!inToolPhase(requestId)) return
      if (META_TOOLS.has(call.name)) {
        const r = runMetaTool(call)
        updateCall(call.id, { status: r.isError ? 'error' : 'done', result: r.text })
        continue
      }
      // The model may call a tool by its bare name; map it to the real one.
      const scope = fileScope(get().genConversation)
      const target = resolveTool(availableTools({ clone: !!scope, practice: !!get().genConversation?.re?.practice }), call.name)
      if (!target) {
        updateCall(call.id, { status: 'error', result: `Tool "${call.name}" is not available or is switched off.` })
        continue
      }
      if (target.qualifiedName !== call.name) {
        call = { ...call, name: target.qualifiedName }
        updateCall(call.id, { name: call.name })
      }
      const isRe = !!get().genConversation?.re
      // Calling a tool keeps it loaded and marks it recently used.
      if (effectiveToolMode({ re: isRe, files: scope, practice: !!get().genConversation?.re?.practice }) === 'dynamic') loadTools([target.qualifiedName])
      const s = useApp.getState().settings
      const gen = get().genConversation
      const remote = isRemoteModelId(gen?.modelId ?? '')
      const chatKey = `${gen?.id ?? ''} ${call.name}`
      // RE sessions make dozens of read-only analysis calls; those run without asking.
      let approved =
        (!s && !remote) ||
        (!!s && runsWithoutAsking(s, target, isRe, remote ? { args: call.arguments, roots: remoteRoots(gen?.re), rea: !!gen && reaOpener === gen.id ? 'opened' : 'other' } : undefined)) ||
        (remote && allowedInChat.has(chatKey))
      if (!approved) {
        const decision = await new Promise<ToolDecision>((resolve) => set({ toolApproval: { call, remote, resolve } }))
        set({ toolApproval: null })
        if (!inToolPhase(requestId)) return
        if (decision === 'always' && remote) allowedInChat.add(chatKey)
        else if (decision === 'always' && s) {
          await useApp.getState().updateSettings({ allowedTools: [...new Set([...s.allowedTools, call.name])] })
          // Stop may have been pressed while that saved; the tool must not run after it.
          if (!inToolPhase(requestId)) return
        }
        approved = decision !== 'deny'
      }
      if (!approved) {
        updateCall(call.id, { status: 'denied', result: 'The user declined to run this tool.' })
        continue
      }
      updateCall(call.id, { status: 'running' })
      const callKey = `${requestId}:${call.id}`
      runningToolCalls.add(callKey)
      const r = await invoke('mcp:callTool', call.name, call.arguments, callKey, scope)
        .catch((e) => ({ text: errorMessage(e), isError: true }))
        .finally(() => runningToolCalls.delete(callKey))
      if (target.server === 'rea' && target.name.startsWith('open_')) reaOpener = target.name === 'open_binary' && !r.isError ? (gen?.id ?? null) : null
      if (!inToolPhase(requestId)) return
      updateCall(call.id, { status: r.isError ? 'error' : 'done', result: r.text })
    }

    const conv = get().genConversation
    set({ generation: null, genConversation: null })
    if (!conv) return

    let rounds = 0
    for (let i = conv.messages.length - 1; i >= 0 && conv.messages[i].role === 'assistant'; i--) {
      if (conv.messages[i].toolCalls?.length) rounds++
    }
    const reSettings = useApp.getState().settings?.re
    const maxRounds = conv.re ? ((conv.re.clone ? reSettings?.cloneMaxToolRounds : reSettings?.maxToolRounds) ?? 80) : MAX_TOOL_ROUNDS
    if (rounds >= maxRounds) {
      void persistNow(conv)
      toast.warning(`Stopped after ${maxRounds} rounds of tool calls without a final answer.`)
      return
    }
    // startGeneration saves the chat with these results (on every path), so it is not saved here
    // too: a long RE chat is several MB, written in full each time.
    await startGeneration(conv, 'new')
  }

  /** Shows `conv` if it is the chat on screen. A chat replying in the background stays there. */
  const showIfCurrent = (conv: Conversation): Partial<ChatState> => (get().current?.id === conv.id ? { current: conv } : {})

  /** Ensures a model is ready, then streams a reply into `conv`. */
  const startGeneration = async (conv: Conversation, mode: 'new' | 'continue'): Promise<void> => {
    const settings = useApp.getState().settings
    const modelId = useModels.getState().activeModelId ?? conv.modelId
    set(showIfCurrent(conv))
    if (!modelId) {
      void persistNow(conv)
      toast('Load a model to start chatting')
      useApp.getState().openPicker()
      return
    }
    if (conv.sampling.jsonSchema.trim()) {
      try {
        JSON.parse(conv.sampling.jsonSchema)
      } catch {
        void persistNow(conv)
        toast.error('The structured output schema is not valid JSON.')
        return
      }
    }

    let inst = readyInstance(modelId)
    if (!inst) {
      void persistNow(conv)
      set({ autoLoadingModelId: modelId })
      // A load still running (from the picker, with its own settings) is waited for, not replaced.
      inst = (await useModels.getState().ensure(modelId, conv.re ? 're' : 'chat')) ?? undefined
      set({ autoLoadingModelId: null })
      if (!inst || deletedIds.has(conv.id)) return
      const latest = get().current
      if (latest?.id === conv.id) conv = latest
    }

    let messageId: string
    let history: ChatMessage[]
    if (mode === 'continue') {
      const last = conv.messages.at(-1)!
      messageId = last.id
      history = conv.messages
      conv = { ...conv, messages: conv.messages.map((m) => (m.id === last.id ? { ...m, error: undefined } : m)) }
    } else {
      history = conv.messages
      messageId = uid()
      conv = {
        ...conv,
        messages: [...conv.messages, { id: messageId, role: 'assistant', content: '', createdAt: Date.now() }]
      }
    }
    conv = { ...conv, modelId, updatedAt: Date.now() }

    if (conv.re && inst.contextLength && inst.contextLength < RE_MIN_CONTEXT && !history.some((m) => m.role === 'assistant')) {
      toast.warning(
        `${inst.modelName} is loaded with a ${inst.contextLength.toLocaleString()}-token context. RE sessions fill that after a few decompiled functions; ${RE_MIN_CONTEXT.toLocaleString()} or more works much better (Load settings, Context length).`,
        { duration: 12000 }
      )
    }
    const hasImages = history.some((m) => m.attachments?.some((a) => a.kind === 'image'))
    if (hasImages && !inst.hasVision) toast.warning(`${inst.modelName} has no vision projector loaded; images are ignored.`)

    // Save the user's message (and new title) now so the chat list is current during long replies.
    void persistNow({ ...conv, messages: conv.messages.filter((m) => m.id !== messageId || m.content) })

    const requestId = uid()
    pending = { content: '', reasoning: '' }
    set({ ...showIfCurrent(conv), genConversation: conv, generation: { requestId, conversationId: conv.id, messageId, promptProgress: null } })
    try {
      await invoke('chat:start', {
        requestId,
        modelId,
        sessionKey: conv.id,
        messages: toWire(history, conv.systemPrompt, inst.hasVision, undefined, !!conv.re),
        sampling: conv.sampling,
        adapters: conv.adapters ?? null,
        overflow: settings?.contextOverflow ?? 'truncateMiddle',
        tools: toolRequestFor(conv.activeTools, toolOpts(conv)),
        record: conv.re?.collect ? { conversationId: conv.id } : undefined
      })
    } catch (e) {
      handleEvent({ type: 'error', requestId, error: errorMessage(e) })
    }
  }

  /** Adds the user's message to `cur` and streams a reply to it. */
  const appendAndGenerate = async (cur: Conversation, text: string, attachments: Attachment[]): Promise<void> => {
    const msg: ChatMessage = {
      id: uid(),
      role: 'user',
      content: text,
      attachments: attachments.length ? attachments : undefined,
      createdAt: Date.now()
    }
    const conv: Conversation = {
      ...cur,
      title: cur.title === NEW_CHAT_TITLE && cur.messages.length === 0 ? makeTitle(text || attachments[0]?.name || '') : cur.title,
      messages: [...cur.messages, msg]
    }
    await startGeneration(conv, 'new')
  }

  /**
   * Sends the next note queued for a chat whose reply just ended, if the user is still in that chat.
   * True if a note went out (the turn goes on).
   */
  const sendNextQueued = (conversationId: string): boolean => {
    if (get().current?.id !== conversationId) return false
    const { note, rest } = takeQueued(get().queued, conversationId)
    if (!note) return false
    set({ queued: rest })
    void get().send(note.text, note.attachments)
    return true
  }

  /** Stops the reply in progress. */
  const stopTurn = (): void => {
    interject = null
    const gen = get().generation
    if (!gen) return
    if (gen.phase !== 'tools') return void invoke('chat:abort', gen.requestId)
    // Stopping during the tool phase: cancel the running call, skip the rest and end the turn here.
    for (const key of runningToolCalls) void invoke('mcp:cancelTool', key)
    get().toolApproval?.resolve('deny')
    applyToGenMessage((m) => ({
      ...m,
      toolCalls: m.toolCalls?.map((c) =>
        c.status === 'pending' || c.status === 'running' ? { ...c, status: 'denied', result: 'Stopped by the user.' } : c
      )
    }))
    const conv = get().genConversation
    set({ generation: null, genConversation: null, toolApproval: null })
    if (conv) void persistNow(conv)
  }

  /** True (after a toast) when the practice pool is writing this chat: it is read-only here until the run is done with it. */
  const poolBlocked = (id: string | null | undefined): boolean => {
    if (!id || !get().poolChats.includes(id)) return false
    toast('The batch run is using this chat.')
    return true
  }

  /** What the empty chat on screen was last given from its model's defaults (sampling and prompt followed separately). */
  let followed: { id: string; sampling: SamplingParams; systemPrompt: string } | null = null
  const appChatDefaults = (): { defaultSampling: SamplingParams; defaultSystemPrompt: string } => {
    const s = useApp.getState().settings
    return { defaultSampling: s?.defaultSampling ?? DEFAULT_SAMPLING, defaultSystemPrompt: s?.defaultSystemPrompt ?? '' }
  }

  /**
   * An empty chat whose sampling or prompt nobody has touched follows the active model's chat defaults (separately for
   * each); one that was edited, has messages, belongs to the pool or is busy is left alone. `before` = the mode settings
   * as they were when the change happened, for a chat whose starting point was not recorded here.
   */
  const followEmptyChat = (before?: AllModeDefaults): void => {
    const cur = get().current
    if (!cur || !isFollowable(cur) || deletedIds.has(cur.id)) return
    if (get().generation || get().handoff || get().autoLoadingModelId || get().poolChats.includes(cur.id)) return
    // With no model loaded the chat keeps its own: sending a message loads it.
    const activeId = useModels.getState().activeModelId ?? cur.modelId
    const app = appChatDefaults()
    const next = chatDefaultsFor(modeDefaultsOf(activeId), app)
    const base =
      followed?.id === cur.id ? followed : { id: cur.id, ...chatDefaultsFor(modesFor(before ?? useModels.getState().modeDefaults, cur.modelId), app) }
    const r = followChatDefaults(cur, base, next)
    if (!r.followed.sampling && !r.followed.prompt) return
    followed = {
      id: cur.id,
      sampling: r.followed.sampling ? next.sampling : base.sampling,
      systemPrompt: r.followed.prompt ? next.systemPrompt : base.systemPrompt
    }
    if (r.conv === cur && cur.modelId === activeId) return
    // Not a change by the user: updatedAt stays.
    const out = { ...r.conv, modelId: activeId }
    set({ current: out, tokenCount: null })
    persistSoon(out)
    get().refreshTokenCount()
  }

  /**
   * Reads a saved chat. null: it no longer exists. undefined: it exists but couldn't be read
   * (locked by another program, say); that is reported here and the chat stays in the list.
   */
  const readConversation = async (id: string): Promise<Conversation | null | undefined> => {
    // A save still on its way holds a newer copy than the disk.
    const waiting = unsaved.get(id)
    if (waiting) return waiting
    return invoke('conversations:get', id).catch((e) => {
      toast.error(`Could not open chat: ${errorMessage(e)}`)
      return undefined
    })
  }

  /** Checks a practice session's final reply against the task's answer and records the outcome. */
  const gradePractice = async (conv: Conversation, messageId: string): Promise<void> => {
    const re = conv.re
    const practice = re?.practice
    const text = conv.messages.find((m) => m.id === messageId)?.content ?? ''
    if (!re || !practice || !shouldGradePractice(re, text)) return
    const kitDir = practice.kitDir || (useApp.getState().settings?.re.kitDir ?? '')
    try {
      const r = await invoke('re:practiceGrade', kitDir, practice.taskId, re.target, text)
      await withConversation(conv.id, (c) =>
        c.re
          ? {
              ...c,
              re: {
                ...c.re,
                outcome: r.passed ? 'solved' : 'failed',
                autoGraded: true,
                practice: { ...practice, answer: r.answer ?? undefined, expected: r.expected }
              }
            }
          : c
      )
      if (r.passed) toast.success(`Practice ${practice.taskId}: solved (${r.answer})`)
      else if (!r.answer) toast.warning(`Practice ${practice.taskId}: no ANSWER line in the final reply, graded as failed.`)
      else toast.warning(`Practice ${practice.taskId}: wrong answer (${r.answer}; expected ${r.expected}).`)
    } catch (e) {
      toast.error(`Could not grade practice ${practice.taskId}: ${errorMessage(e)}`)
    }
  }

  /** Loads a conversation (or reuses `current`), applies `fn`, saves it. */
  const withConversation = async (id: string, fn: (c: Conversation) => Conversation): Promise<void> => {
    if (poolBlocked(id)) return
    const { current, genConversation } = get()
    const base = current?.id === id ? current : genConversation?.id === id ? genConversation : await readConversation(id)
    if (!base) return
    const next = fn(base)
    if (get().current?.id === id) set({ current: next })
    if (get().genConversation?.id === id) set({ genConversation: next })
    await persistNow(next)
  }

  return {
    loaded: false,
    summaries: [],
    folders: [],
    presets: [],
    current: null,
    genConversation: null,
    generation: null,
    autoLoadingModelId: null,
    tokenCount: null,
    handoff: null,
    handoffConfirm: null,
    toolApproval: null,
    queued: [],
    poolChats: [],

    init: once(async () => {
      on('chat:event', handleEvent)
      on('mcp:reaTaken', () => {
        reaOpener = null
      })
      // The practice pool writes its chats in the main process: list entries, the chats it owns, and the one on screen.
      on('conversations:saved', (summary) => set((st) => ({ summaries: upsertSummary(st.summaries, summary) })))
      on('re:poolChats', (ids) => {
        const before = get().poolChats
        set({ poolChats: ids })
        const cur = get().current?.id
        // A chat on screen that the run has just finished with: read its final state.
        if (cur && before.includes(cur) && !ids.includes(cur)) {
          void invoke('conversations:get', cur)
            .catch(() => null)
            .then((fresh) => {
              if (fresh && get().current?.id === cur && !get().generation) {
                liveEpoch++
                set({ current: fresh, tokenCount: null })
              }
            })
        }
        syncWatch()
      })
      on('re:chatLive', (conv) => {
        if (get().current?.id === conv.id && get().poolChats.includes(conv.id)) {
          liveEpoch++
          set({ current: conv })
        }
      })
      useChat.subscribe((st, prev) => {
        if (st.current?.id !== prev.current?.id) syncWatch()
      })
      useModels.subscribe((st, prev) => {
        if (st.activeModelId !== prev.activeModelId || st.modeDefaults !== prev.modeDefaults) followEmptyChat(prev.modeDefaults)
      })
      const [summaries, folders, presets, poolChats] = await Promise.all([
        invoke('conversations:list'),
        invoke('folders:list'),
        invoke('presets:list'),
        invoke('re:poolChats').catch(() => [] as string[])
      ])
      set({ summaries, folders, presets, poolChats, loaded: true })
      if (summaries[0]) await get().open(summaries[0].id)
    }),

    newChat: async () => {
      const { current } = get()
      if (current && current.messages.length === 0) {
        followEmptyChat()
        return get().current ?? current
      }
      const now = Date.now()
      const modelId = useModels.getState().activeModelId
      const defaults = chatDefaultsFor(modeDefaultsOf(modelId), appChatDefaults())
      const conv: Conversation = {
        id: uid(),
        title: NEW_CHAT_TITLE,
        folder: null,
        pinned: false,
        createdAt: now,
        updatedAt: now,
        modelId,
        systemPrompt: defaults.systemPrompt,
        sampling: defaults.sampling,
        adapters: null,
        presetId: null,
        messages: []
      }
      followed = { id: conv.id, sampling: defaults.sampling, systemPrompt: defaults.systemPrompt }
      set({ current: conv, tokenCount: null })
      await persistNow(conv)
      // It may have followed another model's defaults while it was saved.
      const latest = get().current
      return latest?.id === conv.id ? latest : conv
    },

    open: async (id) => {
      if (get().current?.id === id) return
      const gen = get().genConversation
      const conv = gen?.id === id ? gen : await readConversation(id)
      if (conv === undefined) return
      if (!conv) {
        set((s) => ({ summaries: s.summaries.filter((x) => x.id !== id) }))
        return
      }
      set({ current: conv, tokenCount: null })
      get().refreshTokenCount()
    },

    mutateCurrent: (fn) => {
      const cur = get().current
      if (!cur || poolBlocked(cur.id)) return
      const next = { ...fn(cur), updatedAt: Date.now() }
      set({ current: next })
      if (get().genConversation?.id === next.id) set({ genConversation: next })
      else persistSoon(next)
    },

    send: async (text, attachments) => {
      if (isChatBusy(get())) return
      if (wbenchRunning()) return void toast(WBENCH_BUSY)
      const cur = get().current ?? (await get().newChat())
      if (poolBlocked(cur.id)) return
      await appendAndGenerate(cur, text, attachments)
    },

    sendNow: (text, attachments) => {
      const gen = get().generation
      if (!gen) return void get().send(text, attachments)
      if (gen.conversationId !== get().current?.id) return
      // Already stopping for an earlier note: this one waits its turn.
      if (interject) return get().queueNote(text, attachments)
      if (gen.phase === 'tools') {
        // Tool calls stop at once; the turn so far stays, and the note follows it.
        stopTurn()
        const conv = get().current
        if (conv) void appendAndGenerate(markInterrupted(conv, gen.messageId), text, attachments)
        return
      }
      // The note goes out when the stopped reply's final event arrives (see handleEvent).
      interject = { requestId: gen.requestId, text, attachments }
      void invoke('chat:abort', gen.requestId)
    },

    queueNote: (text, attachments) => {
      const cur = get().current
      if (!cur || poolBlocked(cur.id)) return
      if (!get().generation) return void get().send(text, attachments)
      set((s) => ({ queued: [...s.queued, { id: uid(), conversationId: cur.id, text, attachments }] }))
    },

    cancelQueued: (id) => set((s) => ({ queued: s.queued.filter((q) => q.id !== id) })),

    sendQueued: (id) => {
      const note = get().queued.find((q) => q.id === id)
      if (!note || note.conversationId !== get().current?.id || interject || get().autoLoadingModelId) return
      set((s) => ({ queued: s.queued.filter((q) => q.id !== id) }))
      get().sendNow(note.text, note.attachments)
    },

    regenerate: async (messageId) => {
      const cur = get().current
      if (!cur || isChatBusy(get()) || poolBlocked(cur.id)) return
      if (wbenchRunning()) return void toast(WBENCH_BUSY)
      const idx = cur.messages.findIndex((m) => m.id === messageId)
      if (idx < 0) return
      const keep = cur.messages[idx].role === 'assistant' ? idx : idx + 1
      await startGeneration({ ...cur, messages: cur.messages.slice(0, keep) }, 'new')
    },

    continueMessage: async (messageId) => {
      const cur = get().current
      if (!cur || isChatBusy(get()) || poolBlocked(cur.id)) return
      if (wbenchRunning()) return void toast(WBENCH_BUSY)
      const last = cur.messages.at(-1)
      if (last?.id !== messageId || last.role !== 'assistant') return
      // A turn that called tools is complete; writing more into it would mix new calls with the
      // old results. Continuing means the model's next turn, after those results.
      await startGeneration(cur, last.toolCalls?.length ? 'new' : 'continue')
    },

    editMessage: async (messageId, content, resend) => {
      const cur = get().current
      if (!cur || poolBlocked(cur.id)) return
      const target = cur.messages.find((m) => m.id === messageId)
      if (!target) return
      if (resend && target.role === 'user' && !isChatBusy(get())) {
        if (wbenchRunning()) return void toast(WBENCH_BUSY)
        await startGeneration({ ...cur, messages: editMessages(cur.messages, messageId, content, true) }, 'new')
      } else {
        get().mutateCurrent((c) => ({ ...c, messages: editMessages(c.messages, messageId, content, false) }))
      }
    },

    deleteMessage: (messageId) => {
      const cur = get().current
      if (!cur || poolBlocked(cur.id)) return
      const ids = idsToDelete(cur.messages, messageId)
      const gen = get().generation
      if (!ids.length || (gen && ids.includes(gen.messageId))) return
      const { kept, removed } = removeMessages(cur.messages, ids)
      get().mutateCurrent((c) => ({ ...c, messages: kept }))
      get().refreshTokenCount()
      toast(removed.length > 1 ? 'Deleted the message and its reply' : 'Deleted the message', {
        action: {
          label: 'Undo',
          onClick: () => {
            if (get().current?.id !== cur.id) return
            get().mutateCurrent((c) => ({ ...c, messages: restoreMessages(c.messages, removed) }))
            get().refreshTokenCount()
          }
        }
      })
    },

    branchFrom: async (messageId) => {
      const cur = get().current
      if (!cur) return
      const idx = cur.messages.findIndex((m) => m.id === messageId)
      if (idx < 0) return
      const now = Date.now()
      const branch: Conversation = {
        ...cur,
        id: uid(),
        title: `${cur.title} (branch)`,
        pinned: false,
        re: reCopy(cur.re),
        createdAt: now,
        updatedAt: now,
        messages: cur.messages.slice(0, idx + 1).map((m) => ({ ...m, id: uid() }))
      }
      await persistNow(branch)
      set({ current: branch, tokenCount: null })
      get().refreshTokenCount()
      toast.success('Branched into a new chat')
    },

    stop: () => stopTurn(),

    rename: (id, title) => withConversation(id, (c) => ({ ...c, title: title.trim() || c.title })),
    togglePin: (id) => withConversation(id, (c) => ({ ...c, pinned: !c.pinned })),
    moveToFolder: (id, folder) => withConversation(id, (c) => ({ ...c, folder })),

    duplicate: async (id) => {
      const src = get().current?.id === id ? get().current : await readConversation(id)
      if (!src) return
      const now = Date.now()
      const copy: Conversation = {
        ...src,
        id: uid(),
        title: `${src.title} (copy)`,
        pinned: false,
        createdAt: now,
        updatedAt: now,
        re: reCopy(src.re)
      }
      await persistNow(copy)
    },

    remove: async (id) => {
      if (poolBlocked(id)) return
      // Mark it first: a reply stopped here still sends a final event, which would save the chat again.
      deletedIds.add(id)
      cancelSave(id)
      if (get().generation?.conversationId === id) get().stop()
      set((s) => ({ queued: s.queued.filter((q) => q.conversationId !== id) }))
      try {
        await invoke('conversations:delete', id)
      } catch (e) {
        deletedIds.delete(id)
        toast.error(errorMessage(e))
        // The chat is still there; keep what was on screen (including a reply that ended meanwhile)
        // or the edit whose save was waiting.
        const cur = get().current
        const keep = cur?.id === id ? cur : unsaved.get(id)
        if (keep) void persistNow(keep)
        return
      }
      unsaved.delete(id)
      const summaries = get().summaries.filter((s) => s.id !== id)
      set({ summaries })
      if (get().current?.id === id) {
        set({ current: null, tokenCount: null })
        if (summaries[0]) await get().open(summaries[0].id)
      }
    },

    addFolder: async (name) => {
      const n = name.trim()
      if (!n || get().folders.includes(n)) return
      const before = get().folders
      const folders = [...before, n].sort((a, b) => a.localeCompare(b))
      set({ folders })
      try {
        await invoke('folders:save', folders)
      } catch (e) {
        set({ folders: before })
        toast.error(`Could not save folders: ${errorMessage(e)}`)
      }
    },

    removeFolder: async (name) => {
      const folders = get().folders.filter((f) => f !== name)
      // Saved first: if the list can't be saved, the chats stay in the folder too.
      try {
        await invoke('folders:save', folders)
      } catch (e) {
        toast.error(`Could not save folders: ${errorMessage(e)}`)
        return
      }
      set({ folders })
      for (const s of get().summaries.filter((x) => x.folder === name)) await get().moveToFolder(s.id, null)
    },

    savePreset: async (name) => {
      const cur = get().current
      if (!cur) return
      const existing = get().presets.find((p) => p.name.toLowerCase() === name.trim().toLowerCase())
      const preset: Preset = {
        id: existing?.id ?? uid(),
        name: name.trim(),
        systemPrompt: cur.systemPrompt,
        sampling: cur.sampling,
        updatedAt: Date.now()
      }
      let presets: Preset[]
      try {
        presets = await invoke('presets:save', preset)
      } catch (e) {
        toast.error(`Could not save the preset: ${errorMessage(e)}`)
        return
      }
      set({ presets })
      get().mutateCurrent((c) => ({ ...c, presetId: preset.id }))
      toast.success(`Saved preset "${preset.name}"`)
    },

    applyPreset: (id) => {
      const preset = get().presets.find((p) => p.id === id)
      get().mutateCurrent((c) =>
        preset ? { ...c, presetId: preset.id, systemPrompt: preset.systemPrompt, sampling: preset.sampling } : { ...c, presetId: null }
      )
    },

    deletePreset: async (id) => {
      try {
        set({ presets: await invoke('presets:delete', id) })
      } catch (e) {
        toast.error(`Could not delete the preset: ${errorMessage(e)}`)
        return
      }
      if (get().current?.presetId === id) get().mutateCurrent((c) => ({ ...c, presetId: null }))
    },

    setHandoffConfirm: (handoffConfirm) => set({ handoffConfirm }),

    dismissHandoffPrompt: () => {
      set({ handoffConfirm: null })
      get().mutateCurrent((c) => ({ ...c, handoffPrompted: true }))
    },

    startHandoff: async () => {
      const conv = get().current
      set({ handoffConfirm: null })
      if (!conv || get().generation || get().handoff || get().autoLoadingModelId || poolBlocked(conv.id)) return
      if (wbenchRunning()) return void toast(WBENCH_BUSY)
      if (!conv.messages.length) {
        toast('There is nothing to hand off yet.')
        return
      }
      const modelId = useModels.getState().activeModelId ?? conv.modelId
      if (!modelId) {
        toast('Load a model to write the handoff')
        useApp.getState().openPicker()
        return
      }
      const requestId = uid()
      set({ handoff: { requestId, source: conv, modelId, text: '', status: 'loading', promptProgress: null } })

      let inst = readyInstance(modelId)
      if (!inst) inst = (await useModels.getState().ensure(modelId, conv.re ? 're' : 'chat')) ?? undefined
      if (get().handoff?.requestId !== requestId) return // cancelled while loading
      if (!inst) return set({ handoff: null })

      set((s) => ({ handoff: s.handoff && { ...s.handoff, status: 'writing' } }))
      try {
        await invoke('chat:start', {
          requestId,
          modelId,
          messages: [...toWire(conv.messages, conv.systemPrompt, inst.hasVision), { role: 'user', content: HANDOFF_PROMPT }],
          sampling: { ...conv.sampling, temperature: Math.min(conv.sampling.temperature, 0.5), maxTokens: null, jsonSchema: '' },
          adapters: conv.adapters ?? null,
          // The chat is near the limit by definition, so always make room rather than refuse.
          overflow: 'truncateMiddle',
          // Little context is left at this point; spend it on the handoff, not on reasoning.
          disableThinking: true
        })
      } catch (e) {
        set((s) => ({ handoff: s.handoff && { ...s.handoff, status: 'error', error: errorMessage(e) } }))
      }
    },

    cancelHandoff: () => {
      const ho = get().handoff
      if (!ho) return
      if (ho.status === 'writing') void invoke('chat:abort', ho.requestId)
      set({ handoff: null })
    },

    startRe: async (input) => {
      if (wbenchRunning()) {
        toast(WBENCH_BUSY)
        return null
      }
      if (isChatBusy(get()) || get().handoff) {
        toast('Wait for the current reply to finish first.')
        return null
      }
      // Pasted paths may keep their quotes (Explorer's "Copy as path"); the tools need them without.
      const opts = {
        ...input,
        target: unquotePath(input.target),
        ...(input.clone ? { clone: { ...input.clone, outputDir: unquotePath(input.clone.outputDir) } } : {})
      }
      const s = useApp.getState().settings
      const models = useModels.getState()
      const modelId = opts.modelId && models.models.some((m) => m.id === opts.modelId) ? opts.modelId : models.activeModelId
      // Messages go to the active model, so the session's model becomes it (loaded on send if needed).
      if (modelId) models.setActive(modelId)
      const available = availableTools({ clone: !!opts.clone, practice: !!opts.practice })
      const starter = [...RE_STARTER_TOOLS, ...(opts.clone ? CLONE_STARTER_TOOLS : [])].flatMap((n) => {
        const t = resolveTool(available, n)
        return t ? [t.qualifiedName] : []
      })
      const now = Date.now()
      const conv: Conversation = {
        id: uid(),
        title: (opts.title ?? `${opts.clone ? 'Clone' : 'RE'}: ${baseName(opts.target)}`).slice(0, 60),
        folder: null,
        pinned: false,
        createdAt: now,
        updatedAt: now,
        modelId,
        systemPrompt: withPromptExtra(
          opts.clone ? `${RE_SYSTEM_PROMPT}\n\n${clonePrompt(opts.clone)}` : RE_SYSTEM_PROMPT,
          modePromptExtra(modeDefaultsOf(modelId), 're')
        ),
        sampling: modeSampling('re', modeDefaultsOf(modelId), s?.defaultSampling ?? DEFAULT_SAMPLING),
        adapters: null,
        presetId: null,
        activeTools: starter,
        re: { target: opts.target, goal: opts.goal, collect: opts.collect, outcome: 'unrated', practice: opts.practice, clone: opts.clone },
        messages: []
      }
      set({ current: conv, tokenCount: null })
      useApp.getState().setPage('chat')
      // Nothing is awaited between the busy check above and the reply starting here, so no other
      // reply can start in between. startGeneration saves the new chat on every path.
      await appendAndGenerate(conv, opts.clone ? cloneFirstMessage(opts.clone, opts.target) : reFirstMessage(opts.goal, opts.target), [])
      return conv.id
    },

    setReOutcome: (id, outcome, notes) =>
      withConversation(id, (c) => (c.re ? { ...c, re: { ...c.re, outcome, autoGraded: false, notes: notes ?? c.re.notes } } : c)),

    setReCollect: (id, collect) => withConversation(id, (c) => (c.re ? { ...c, re: { ...c.re, collect } } : c)),

    refreshTokenCount: () => {
      if (tokenTimer) clearTimeout(tokenTimer)
      tokenTimer = setTimeout(async () => {
        const conv = get().current
        const modelId = useModels.getState().activeModelId ?? conv?.modelId ?? null
        const inst = readyInstance(modelId)
        if (!conv || !inst) return set({ tokenCount: null })
        const tools = toolRequestFor(conv.activeTools, toolOpts(conv)) ?? null
        const wire = toWire(conv.messages, conv.systemPrompt, inst.hasVision, undefined, !!conv.re)
        const n = await invoke('chat:countTokens', inst.modelId, wire, tools).catch(
          () => null
        )
        if (get().current?.id === conv.id) set({ tokenCount: n })
      }, 600)
    }
  }
})
