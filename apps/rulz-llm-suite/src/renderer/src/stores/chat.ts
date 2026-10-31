import { create } from 'zustand'
import { toast } from 'sonner'
import type {
  Attachment,
  ChatMessage,
  Conversation,
  ConversationSummary,
  Preset,
  StreamEvent,
  ToolCall,
  ToolResult,
  WireMessage
} from '@shared/types'
import { DEFAULT_SAMPLING } from '@shared/defaults'
import { errorMessage, invoke, on } from '../lib/api'
import { uid } from '../lib/format'
import { type QueuedNote, takeQueued } from '../lib/interject'
import { editMessages, idsToDelete, removeMessages, restoreMessages } from '../lib/messages'
import { buildWire } from '../lib/wire'
import { useApp } from './app'
import { availableTools, effectiveToolMode, tokensOf, toolRequestFor } from './mcp'
import { readyInstance, useModels } from './models'
import { META_LOAD, META_SEARCH, META_TOOLS, resolveTool, searchTools } from '@shared/tools'

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
  resolve: (decision: ToolDecision) => void
}

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
  keepFull: number | null = keepFullToolResults()
): WireMessage[] {
  return buildWire(messages, systemPrompt, vision, keepFull)
}

// ---- Persistence (debounced per conversation) ----
const saveTimers = new Map<string, ReturnType<typeof setTimeout>>()

function upsertSummary(list: ConversationSummary[], s: ConversationSummary): ConversationSummary[] {
  return [s, ...list.filter((x) => x.id !== s.id)].sort((a, b) => b.updatedAt - a.updatedAt)
}

async function persistNow(conv: Conversation): Promise<void> {
  const t = saveTimers.get(conv.id)
  if (t) clearTimeout(t)
  saveTimers.delete(conv.id)
  try {
    const summary = await invoke('conversations:save', conv)
    useChat.setState((s) => ({ summaries: upsertSummary(s.summaries, summary) }))
  } catch (e) {
    toast.error(`Could not save chat: ${errorMessage(e)}`)
  }
}

function persistSoon(conv: Conversation): void {
  const t = saveTimers.get(conv.id)
  if (t) clearTimeout(t)
  saveTimers.set(conv.id, setTimeout(() => void persistNow(conv), 500))
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
      presetId: src.presetId,
      activeTools: src.activeTools,
      messages: []
    }
    await withConversation(src.id, (c) => ({ ...c, handoffPrompted: true }))
    set({ handoff: null, current: conv, tokenCount: null })
    await persistNow(conv)
    toast.success('Handed off to a new chat')
    await get().send(text, [])
  }

  const handleEvent = (e: StreamEvent): void => {
    const ho = get().handoff
    if (ho && e.requestId === ho.requestId) return handleHandoffEvent(ho, e)
    const gen = get().generation
    if (!gen || e.requestId !== gen.requestId) return
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
        if (conv && e.type === 'done' && e.stats.stopReason !== 'stopped') sendNextQueued(conv.id)
        break
      }
    }
  }

  const updateCall = (id: string, patch: Partial<ToolCall>): void =>
    applyToGenMessage((m) => ({ ...m, toolCalls: m.toolCalls?.map((c) => (c.id === id ? { ...c, ...patch } : c)) }))

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
    const budget = useApp.getState().settings?.toolBudgetTokens ?? 6000
    let evicted: string[] = []
    setActiveTools((active) => {
      const next = [...active.filter((n) => !names.includes(n)), ...names]
      evicted = []
      while (next.length > names.length && tokensOf(next) > budget) evicted.push(next.shift()!)
      return next
    })
    return evicted
  }

  /** Tool names from a list, or from text such as "a, b" or "['a', 'b']" that some models send. */
  const toolNameList = (value: unknown): string[] => {
    const raw = Array.isArray(value) ? value.map(String) : typeof value === 'string' ? value.split(/[\s,;]+/) : []
    return raw.map((n) => n.replace(/^[\s[\]'"`]+|[\s[\]'"`]+$/g, '')).filter(Boolean)
  }

  /** Runs one of the built-in loader tools; these change chat state only, so they need no approval. */
  const runMetaTool = (call: ToolCall): ToolResult => {
    let args: { query?: string; limit?: number; names?: unknown } = {}
    try {
      args = call.arguments.trim() ? JSON.parse(call.arguments) : {}
    } catch {
      return { text: `The arguments were not valid JSON: ${call.arguments}`, isError: true }
    }
    const tools = availableTools()
    const active = new Set(get().genConversation?.activeTools ?? [])
    if (call.name === META_SEARCH) {
      const hits = searchTools(tools, String(args.query ?? ''), Number(args.limit) || 10)
      if (!hits.length) return { text: `No tools match "${args.query ?? ''}".`, isError: false }
      return {
        text: hits
          .map((h) => `- ${h.name} (${h.server})${active.has(resolveTool(tools, `${h.server}__${h.name}`)?.qualifiedName ?? '') ? ' [loaded]' : ''}: ${h.summary}`)
          .join('\n'),
        isError: false
      }
    }
    const names = toolNameList(args.names)
    if (!names.length) {
      return {
        text: 'No tool names arrived: "names" was empty. Pass a JSON list with double quotes, for example ["analyze_javascript_application"].',
        isError: true
      }
    }
    const resolved = names.map((n) => ({ n, t: resolveTool(tools, n) }))
    const unknown = resolved.filter((r) => !r.t).map((r) => {
      const guess = searchTools(tools, r.n, 3).map((h) => h.name)
      return guess.length ? `${r.n} (did you mean ${guess.join(', ')}?)` : r.n
    })
    const found = [...new Set(resolved.flatMap((r) => (r.t ? [r.t.qualifiedName] : [])))]
    const lines: string[] = []
    if (call.name === META_LOAD) {
      if (found.length) {
        const evicted = loadTools(found)
        lines.push(`Loaded: ${found.join(', ')}. You can call them in your next step.`)
        if (evicted.length) lines.push(`Unloaded to stay within the tool budget (least recently used): ${evicted.join(', ')}.`)
      }
    } else if (found.length) {
      setActiveTools((a) => a.filter((n) => !found.includes(n)))
      lines.push(`Unloaded: ${found.join(', ')}.`)
    }
    if (unknown.length) lines.push(`Unknown tools: ${unknown.join('; ')}.`)
    return { text: lines.join('\n'), isError: found.length === 0 }
  }

  /** Approves and runs each tool call, then continues the reply with the results. */
  const runTools = async (requestId: string, calls: ToolCall[]): Promise<void> => {
    for (let call of calls) {
      if (!inToolPhase(requestId)) return
      if (META_TOOLS.has(call.name)) {
        const r = runMetaTool(call)
        updateCall(call.id, { status: r.isError ? 'error' : 'done', result: r.text })
        continue
      }
      // The model may call a tool by its bare name; map it to the real one.
      const target = resolveTool(availableTools(), call.name)
      if (!target) {
        updateCall(call.id, { status: 'error', result: `Tool "${call.name}" is not available or is switched off.` })
        continue
      }
      if (target.qualifiedName !== call.name) {
        call = { ...call, name: target.qualifiedName }
        updateCall(call.id, { name: call.name })
      }
      // Calling a tool keeps it loaded and marks it recently used.
      if (effectiveToolMode() === 'dynamic') loadTools([target.qualifiedName])
      const s = useApp.getState().settings
      let approved = !s?.confirmToolCalls || s.allowedTools.includes(call.name)
      if (!approved) {
        const decision = await new Promise<ToolDecision>((resolve) => set({ toolApproval: { call, resolve } }))
        set({ toolApproval: null })
        if (!inToolPhase(requestId)) return
        if (decision === 'always' && s) {
          await useApp.getState().updateSettings({ allowedTools: [...new Set([...s.allowedTools, call.name])] })
        }
        approved = decision !== 'deny'
      }
      if (!approved) {
        updateCall(call.id, { status: 'denied', result: 'The user declined to run this tool.' })
        continue
      }
      updateCall(call.id, { status: 'running' })
      const r = await invoke('mcp:callTool', call.name, call.arguments).catch((e) => ({ text: errorMessage(e), isError: true }))
      if (!inToolPhase(requestId)) return
      updateCall(call.id, { status: r.isError ? 'error' : 'done', result: r.text })
    }

    const conv = get().genConversation
    set({ generation: null, genConversation: null })
    if (!conv) return
    await persistNow(conv)

    let rounds = 0
    for (let i = conv.messages.length - 1; i >= 0 && conv.messages[i].role === 'assistant'; i--) {
      if (conv.messages[i].toolCalls?.length) rounds++
    }
    if (rounds >= MAX_TOOL_ROUNDS) {
      toast.warning(`Stopped after ${MAX_TOOL_ROUNDS} rounds of tool calls without a final answer.`)
      return
    }
    await startGeneration(conv, 'new')
  }

  /** Ensures a model is ready, then streams a reply into `conv`. */
  const startGeneration = async (conv: Conversation, mode: 'new' | 'continue'): Promise<void> => {
    const settings = useApp.getState().settings
    const modelId = useModels.getState().activeModelId ?? conv.modelId
    set({ current: conv })
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
      inst = (await useModels.getState().load(modelId, null)) ?? undefined
      set({ autoLoadingModelId: null })
      if (!inst) return
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

    const hasImages = history.some((m) => m.attachments?.some((a) => a.kind === 'image'))
    if (hasImages && !inst.hasVision) toast.warning(`${inst.modelName} has no vision projector loaded; images are ignored.`)

    // Save the user's message (and new title) now so the chat list is current during long replies.
    void persistNow({ ...conv, messages: conv.messages.filter((m) => m.id !== messageId || m.content) })

    const requestId = uid()
    pending = { content: '', reasoning: '' }
    set({ current: conv, genConversation: conv, generation: { requestId, conversationId: conv.id, messageId, promptProgress: null } })
    try {
      await invoke('chat:start', {
        requestId,
        modelId,
        messages: toWire(history, conv.systemPrompt, inst.hasVision),
        sampling: conv.sampling,
        overflow: settings?.contextOverflow ?? 'truncateMiddle',
        tools: toolRequestFor(conv.activeTools)
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

  /** Sends the next note queued for a chat whose reply just ended, if the user is still in that chat. */
  const sendNextQueued = (conversationId: string): void => {
    if (get().current?.id !== conversationId) return
    const { note, rest } = takeQueued(get().queued, conversationId)
    if (!note) return
    set({ queued: rest })
    void get().send(note.text, note.attachments)
  }

  /** Loads a conversation (or reuses `current`), applies `fn`, saves it. */
  const withConversation = async (id: string, fn: (c: Conversation) => Conversation): Promise<void> => {
    const { current, genConversation } = get()
    const base = current?.id === id ? current : genConversation?.id === id ? genConversation : await invoke('conversations:get', id)
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

    init: async () => {
      on('chat:event', handleEvent)
      const [summaries, folders, presets] = await Promise.all([
        invoke('conversations:list'),
        invoke('folders:list'),
        invoke('presets:list')
      ])
      set({ summaries, folders, presets, loaded: true })
      if (summaries[0]) await get().open(summaries[0].id)
    },

    newChat: async () => {
      const { current } = get()
      if (current && current.messages.length === 0) return current
      const s = useApp.getState().settings
      const now = Date.now()
      const conv: Conversation = {
        id: uid(),
        title: NEW_CHAT_TITLE,
        folder: null,
        pinned: false,
        createdAt: now,
        updatedAt: now,
        modelId: useModels.getState().activeModelId,
        systemPrompt: s?.defaultSystemPrompt ?? '',
        sampling: s?.defaultSampling ?? DEFAULT_SAMPLING,
        presetId: null,
        messages: []
      }
      set({ current: conv, tokenCount: null })
      await persistNow(conv)
      return conv
    },

    open: async (id) => {
      if (get().current?.id === id) return
      const gen = get().genConversation
      const conv = gen?.id === id ? gen : await invoke('conversations:get', id)
      if (!conv) {
        set((s) => ({ summaries: s.summaries.filter((x) => x.id !== id) }))
        return
      }
      set({ current: conv, tokenCount: null })
      get().refreshTokenCount()
    },

    mutateCurrent: (fn) => {
      const cur = get().current
      if (!cur) return
      const next = { ...fn(cur), updatedAt: Date.now() }
      set({ current: next })
      if (get().genConversation?.id === next.id) set({ genConversation: next })
      else persistSoon(next)
    },

    send: async (text, attachments) => {
      if (get().generation || get().autoLoadingModelId) return
      const cur = get().current ?? (await get().newChat())
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
        get().stop()
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
      if (!cur) return
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
      if (!cur || get().generation) return
      const idx = cur.messages.findIndex((m) => m.id === messageId)
      if (idx < 0) return
      const keep = cur.messages[idx].role === 'assistant' ? idx : idx + 1
      await startGeneration({ ...cur, messages: cur.messages.slice(0, keep) }, 'new')
    },

    continueMessage: async (messageId) => {
      const cur = get().current
      if (!cur || get().generation) return
      if (cur.messages.at(-1)?.id !== messageId || cur.messages.at(-1)?.role !== 'assistant') return
      await startGeneration(cur, 'continue')
    },

    editMessage: async (messageId, content, resend) => {
      const cur = get().current
      if (!cur) return
      const target = cur.messages.find((m) => m.id === messageId)
      if (!target) return
      if (resend && target.role === 'user' && !get().generation) {
        await startGeneration({ ...cur, messages: editMessages(cur.messages, messageId, content, true) }, 'new')
      } else {
        get().mutateCurrent((c) => ({ ...c, messages: editMessages(c.messages, messageId, content, false) }))
      }
    },

    deleteMessage: (messageId) => {
      const cur = get().current
      if (!cur) return
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
        createdAt: now,
        updatedAt: now,
        messages: cur.messages.slice(0, idx + 1).map((m) => ({ ...m, id: uid() }))
      }
      await persistNow(branch)
      set({ current: branch, tokenCount: null })
      get().refreshTokenCount()
      toast.success('Branched into a new chat')
    },

    stop: () => {
      interject = null
      const gen = get().generation
      if (!gen) return
      if (gen.phase !== 'tools') return void invoke('chat:abort', gen.requestId)
      // Stopping during the tool phase: skip the remaining calls and end the turn here.
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
    },

    rename: (id, title) => withConversation(id, (c) => ({ ...c, title: title.trim() || c.title })),
    togglePin: (id) => withConversation(id, (c) => ({ ...c, pinned: !c.pinned })),
    moveToFolder: (id, folder) => withConversation(id, (c) => ({ ...c, folder })),

    duplicate: async (id) => {
      const src = get().current?.id === id ? get().current : await invoke('conversations:get', id)
      if (!src) return
      const now = Date.now()
      const copy: Conversation = { ...src, id: uid(), title: `${src.title} (copy)`, pinned: false, createdAt: now, updatedAt: now }
      await persistNow(copy)
    },

    remove: async (id) => {
      if (get().generation?.conversationId === id) get().stop()
      set((s) => ({ queued: s.queued.filter((q) => q.conversationId !== id) }))
      await invoke('conversations:delete', id).catch((e) => toast.error(errorMessage(e)))
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
      const folders = [...get().folders, n].sort((a, b) => a.localeCompare(b))
      set({ folders })
      await invoke('folders:save', folders)
    },

    removeFolder: async (name) => {
      for (const s of get().summaries.filter((x) => x.folder === name)) await get().moveToFolder(s.id, null)
      const folders = get().folders.filter((f) => f !== name)
      set({ folders })
      await invoke('folders:save', folders)
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
      const presets = await invoke('presets:save', preset)
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
      set({ presets: await invoke('presets:delete', id) })
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
      if (!conv || get().generation || get().handoff || get().autoLoadingModelId) return
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
      if (!inst) inst = (await useModels.getState().load(modelId, null)) ?? undefined
      if (get().handoff?.requestId !== requestId) return // cancelled while loading
      if (!inst) return set({ handoff: null })

      set((s) => ({ handoff: s.handoff && { ...s.handoff, status: 'writing' } }))
      try {
        await invoke('chat:start', {
          requestId,
          modelId,
          messages: [...toWire(conv.messages, conv.systemPrompt, inst.hasVision), { role: 'user', content: HANDOFF_PROMPT }],
          sampling: { ...conv.sampling, temperature: Math.min(conv.sampling.temperature, 0.5), maxTokens: null, jsonSchema: '' },
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

    refreshTokenCount: () => {
      if (tokenTimer) clearTimeout(tokenTimer)
      tokenTimer = setTimeout(async () => {
        const conv = get().current
        const modelId = useModels.getState().activeModelId ?? conv?.modelId ?? null
        const inst = readyInstance(modelId)
        if (!conv || !inst) return set({ tokenCount: null })
        const tools = toolRequestFor(conv.activeTools) ?? null
        const n = await invoke('chat:countTokens', inst.modelId, toWire(conv.messages, conv.systemPrompt, inst.hasVision), tools).catch(
          () => null
        )
        if (get().current?.id === conv.id) set({ tokenCount: n })
      }, 600)
    }
  }
})
