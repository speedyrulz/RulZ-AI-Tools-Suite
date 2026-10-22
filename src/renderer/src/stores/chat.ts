import { create } from 'zustand'
import { toast } from 'sonner'
import type {
  Attachment,
  ChatMessage,
  Conversation,
  ConversationSummary,
  Preset,
  StreamEvent,
  WireMessage
} from '@shared/types'
import { DEFAULT_SAMPLING } from '@shared/defaults'
import { errorMessage, invoke, on } from '../lib/api'
import { uid } from '../lib/format'
import { useApp } from './app'
import { readyInstance, useModels } from './models'

export const NEW_CHAT_TITLE = 'New Chat'

interface Generation {
  requestId: string
  conversationId: string
  messageId: string
  promptProgress: number | null
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

  init: () => Promise<void>
  newChat: () => Promise<Conversation>
  open: (id: string) => Promise<void>
  mutateCurrent: (fn: (c: Conversation) => Conversation) => void
  send: (text: string, attachments: Attachment[]) => Promise<void>
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

/** Converts chat history to the OpenAI message format llama-server expects. */
export function toWire(messages: ChatMessage[], systemPrompt: string, vision: boolean): WireMessage[] {
  const out: WireMessage[] = []
  if (systemPrompt.trim()) out.push({ role: 'system', content: systemPrompt })
  for (const m of messages) {
    if (m.role === 'assistant' && !m.content && m.error) continue
    let text = m.content
    for (const a of m.attachments ?? []) {
      if (a.kind === 'text' && a.text !== undefined) text += `\n\n\`\`\`${a.name}\n${a.text}\n\`\`\``
    }
    const images = vision ? (m.attachments ?? []).filter((a) => a.kind === 'image' && a.dataUrl) : []
    if (m.role === 'user' && images.length) {
      out.push({
        role: 'user',
        content: [
          { type: 'text', text },
          ...images.map((i) => ({ type: 'image_url' as const, image_url: { url: i.dataUrl! } }))
        ]
      })
    } else {
      out.push({ role: m.role, content: text })
    }
  }
  return out
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

// ---- Streaming deltas are batched per animation frame ----
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
          requestAnimationFrame(flushPending)
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
        if (conv) void persistNow({ ...conv, updatedAt: Date.now() })
        get().refreshTokenCount()
        break
      }
    }
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
        overflow: settings?.contextOverflow ?? 'truncateMiddle'
      })
    } catch (e) {
      handleEvent({ type: 'error', requestId, error: errorMessage(e) })
    }
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
      const idx = cur.messages.findIndex((m) => m.id === messageId)
      if (idx < 0) return
      const edited = cur.messages.map((m) => (m.id === messageId ? { ...m, content } : m))
      if (resend && cur.messages[idx].role === 'user' && !get().generation) {
        await startGeneration({ ...cur, messages: edited.slice(0, idx + 1) }, 'new')
      } else {
        get().mutateCurrent((c) => ({ ...c, messages: edited }))
      }
    },

    deleteMessage: (messageId) => {
      if (get().generation?.messageId === messageId) return
      get().mutateCurrent((c) => ({ ...c, messages: c.messages.filter((m) => m.id !== messageId) }))
      get().refreshTokenCount()
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
      const gen = get().generation
      if (gen) void invoke('chat:abort', gen.requestId)
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
        const n = await invoke('chat:countTokens', inst.modelId, toWire(conv.messages, conv.systemPrompt, inst.hasVision)).catch(() => null)
        if (get().current?.id === conv.id) set({ tokenCount: n })
      }, 600)
    }
  }
})
