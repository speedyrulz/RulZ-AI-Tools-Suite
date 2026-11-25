// Per-model, per-mode defaults on the renderer side: the pure helpers (empty chats following the active model, the picker's
// scope, what "remember" saves, when the reload bar shows) and the chat store's newChat / startRe with a fake window API.
import { beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'
import { DEFAULT_LOAD_CONFIG, DEFAULT_SAMPLING } from '@shared/defaults'
import type { AllModeDefaults, ModeDefaults } from '@shared/modes'
import { RE_SYSTEM_PROMPT, reSampling } from '@shared/re'
import type { AppSettings, Conversation, LoadConfig, ModelEntry } from '@shared/types'
import {
  chatDefaultsFor,
  cloudPickerMode,
  deepEqual,
  followChatDefaults,
  isFollowable,
  modesFor,
  pickerScope,
  reloadOffer,
  samplingSave,
  scopeFromValue,
  scopeLoad,
  scopeValue,
  startLoad,
  startScope,
  withModeLoad,
  withModeSaves
} from '../src/renderer/src/lib/modes'

const calls: { channel: string; args: unknown[] }[] = []
const listeners = new Map<string, ((p: unknown) => void)[]>()

vi.stubGlobal('window', {
  matchMedia: () => ({ matches: false, addEventListener: () => undefined }),
  rulz: {
    invoke: (channel: string, ...args: unknown[]) => {
      calls.push({ channel, args })
      if (channel === 'conversations:save') {
        const c = args[0] as Conversation
        return Promise.resolve({ id: c.id, title: c.title, folder: null, pinned: false, createdAt: c.createdAt, updatedAt: c.updatedAt, messageCount: 0 })
      }
      if (channel === 'conversations:list' || channel === 'folders:list' || channel === 'presets:list' || channel === 're:poolChats') return Promise.resolve([])
      if (channel === 'models:modeDefaults') return Promise.resolve({})
      return Promise.resolve(null)
    },
    on: (channel: string, fn: (p: unknown) => void) => {
      listeners.set(channel, [...(listeners.get(channel) ?? []), fn])
      return () => undefined
    }
  }
})

const APP = { defaultSampling: DEFAULT_SAMPLING, defaultSystemPrompt: 'APP PROMPT' }
const own = (over: Partial<ModeDefaults> = {}): ModeDefaults => ({ load: null, sampling: null, systemPrompt: '', ...over })
const conv = (over: Partial<Conversation> = {}): Conversation =>
  ({
    id: 'c1',
    title: 'New Chat',
    folder: null,
    pinned: false,
    createdAt: 0,
    updatedAt: 5,
    modelId: null,
    systemPrompt: 'APP PROMPT',
    sampling: DEFAULT_SAMPLING,
    adapters: null,
    presetId: null,
    messages: [],
    ...over
  }) as Conversation
const loadWith = (over: Partial<LoadConfig>): LoadConfig => ({ ...DEFAULT_LOAD_CONFIG, ...over })

describe('modesFor', () => {
  it('looks a model up by own property only', () => {
    const all: AllModeDefaults = { a: { chat: own({ systemPrompt: 'x' }) } }
    expect(modesFor(all, 'a')?.chat?.systemPrompt).toBe('x')
    expect(modesFor(all, 'constructor')).toBeUndefined()
    expect(modesFor(all, '__proto__')).toBeUndefined()
    expect(modesFor(all, null)).toBeUndefined()
    expect(modesFor(undefined, 'a')).toBeUndefined()
  })
})

describe('chat defaults and following', () => {
  const modelA = { chat: own({ sampling: { ...DEFAULT_SAMPLING, temperature: 0.2 }, systemPrompt: 'Be A.' }) }
  const modelB = { chat: own({ systemPrompt: 'Be B.' }) }

  it('a model without chat settings gives the app defaults', () => {
    const d = chatDefaultsFor(undefined, APP)
    expect(d.systemPrompt).toBe('APP PROMPT')
    expect(deepEqual(d.sampling, DEFAULT_SAMPLING)).toBe(true)
  })

  it('moves an untouched empty chat, sampling and prompt separately', () => {
    const base = chatDefaultsFor(modelA, APP)
    const next = chatDefaultsFor(modelB, APP)
    const c = conv({ sampling: base.sampling, systemPrompt: base.systemPrompt })
    const r = followChatDefaults(c, base, next)
    expect(r.conv.systemPrompt).toBe('Be B.')
    expect(r.conv.sampling.temperature).toBe(DEFAULT_SAMPLING.temperature)
    expect(r.followed).toEqual({ sampling: true, prompt: true })
    // An edited prompt stays; the sampling still follows.
    const edited = followChatDefaults({ ...c, systemPrompt: 'mine' }, base, next)
    expect(edited.conv.systemPrompt).toBe('mine')
    expect(edited.conv.sampling.temperature).toBe(DEFAULT_SAMPLING.temperature)
    expect(edited.followed).toEqual({ sampling: true, prompt: false })
    // An edited sampling stays; the prompt follows.
    const tuned = followChatDefaults({ ...c, sampling: { ...base.sampling, topK: 7 } }, base, next)
    expect(tuned.conv.sampling.topK).toBe(7)
    expect(tuned.conv.systemPrompt).toBe('Be B.')
  })

  it('returns the same chat when nothing needs to change', () => {
    const base = chatDefaultsFor(modelB, APP)
    const c = conv({ sampling: base.sampling, systemPrompt: base.systemPrompt })
    expect(followChatDefaults(c, base, chatDefaultsFor(modelB, APP)).conv).toBe(c)
  })

  it('never touches chats with messages, reverse-engineering sessions or presets', () => {
    const base = chatDefaultsFor(modelA, APP)
    const next = chatDefaultsFor(modelB, APP)
    const mk = (over: Partial<Conversation>): Conversation => conv({ sampling: base.sampling, systemPrompt: base.systemPrompt, ...over })
    const withMessage = mk({ messages: [{ id: 'm', role: 'user', content: 'hi' } as never] })
    const re = mk({ re: { target: 'x' } as never })
    const preset = mk({ presetId: 'p' })
    for (const c of [withMessage, re, preset]) {
      expect(isFollowable(c)).toBe(false)
      expect(followChatDefaults(c, base, next).conv).toBe(c)
    }
  })
})

describe('picker scope', () => {
  it('starts from where the user is', () => {
    expect(pickerScope('chat', false)).toBe('chat')
    expect(pickerScope('chat', true)).toBe('re')
    expect(pickerScope('re', false)).toBe('re')
    expect(pickerScope('adventure', false)).toBe('adventure')
    expect(pickerScope('writing', false)).toBe('writing')
    expect(pickerScope('models', false)).toBeNull()
    expect(pickerScope('settings', true)).toBeNull()
  })

  it('maps the Select value both ways', () => {
    expect(scopeValue(null)).toBe('all')
    expect(scopeFromValue('all')).toBeNull()
    expect(scopeFromValue('writing')).toBe('writing')
    expect(scopeFromValue('nonsense')).toBeNull()
  })

  it('shows a mode\'s own load settings, else the model\'s default ones', () => {
    const base = loadWith({ contextLength: 4096 })
    const mine = loadWith({ contextLength: 32768 })
    const modes = { chat: own({ load: mine }) }
    expect(scopeLoad(base, modes, 'chat')).toEqual(mine)
    const left = { ...mine, sampling: { ...mine.sampling, temperature: 1.9 } }
    expect(scopeLoad(base, { chat: own({ load: left }) }, 'chat').sampling).toBe(base.sampling)
    expect(scopeLoad(base, modes, 're')).toBe(base)
    expect(scopeLoad(base, modes, null)).toBe(base)
  })
})

describe('what the picker first shows', () => {
  const base: LoadConfig = { ...DEFAULT_LOAD_CONFIG, contextLength: 1000 }
  const re: LoadConfig = { ...DEFAULT_LOAD_CONFIG, contextLength: 2000 }
  const adv: LoadConfig = { ...DEFAULT_LOAD_CONFIG, contextLength: 3000 }
  const byHand: LoadConfig = { ...DEFAULT_LOAD_CONFIG, contextLength: 4000 }
  const modes = { re: own({ load: re }), adventure: own({ load: adv }), chat: own({ systemPrompt: 'x' }) }
  it("starts with the page's mode only when the model has load settings for it", () => {
    expect(startScope('re', modes)).toBe('re')
    expect(startScope('chat', modes)).toBeNull()
    expect(startScope('re', undefined)).toBeNull()
    expect(startScope(null, modes)).toBeNull()
  })
  it("never shows one scope's running settings under another, so Remember cannot overwrite it", () => {
    expect(startLoad(base, modes, 'adventure', re)).toEqual(adv)
    expect(startLoad(base, modes, null, re)).toBe(base)
  })
  it('all modes shows settings the model was loaded with by hand', () => {
    expect(startLoad(base, modes, null, byHand)).toBe(byHand)
    expect(startLoad(base, undefined, null, undefined)).toBe(base)
  })
})

describe('remember and reset for a mode', () => {
  const sampling = { ...DEFAULT_SAMPLING, temperature: 0.3 }
  it('saves the load settings and keeps the mode\'s sampling and prompt', () => {
    const load = loadWith({ contextLength: 8192 })
    expect(withModeLoad(own({ sampling, systemPrompt: 'hi' }), load)).toEqual({ load, sampling, systemPrompt: 'hi' })
    expect(withModeLoad(undefined, load)).toEqual({ load, sampling: null, systemPrompt: '' })
  })

  it('reset forgets only the load settings', () => {
    expect(withModeLoad(own({ load: loadWith({}), sampling, systemPrompt: 'hi' }), null)).toEqual({ load: null, sampling, systemPrompt: 'hi' })
  })

  it('an entry that says nothing is dropped (null)', () => {
    expect(withModeLoad(own({ load: loadWith({}) }), null)).toBeNull()
    expect(withModeLoad(undefined, null)).toBeNull()
  })
})

describe("API sampling in a mode's stored load", () => {
  const baseS = { ...DEFAULT_LOAD_CONFIG.sampling, temperature: 0.7 }
  const withS = (over: Partial<LoadConfig['sampling']>, ctx = 8192): LoadConfig => ({ ...DEFAULT_LOAD_CONFIG, contextLength: ctx, sampling: { ...DEFAULT_LOAD_CONFIG.sampling, ...over } })

  it("keeps what an entry has and gives a new one the default load settings' sampling", () => {
    const existing = own({ load: withS({ temperature: 1 }, 4096) })
    expect(withModeLoad(existing, withS({ temperature: 0.2 }, 16384), baseS)?.load).toMatchObject({ contextLength: 16384, sampling: { temperature: 1 } })
    expect(withModeLoad(undefined, withS({ temperature: 0.2 }, 16384), baseS)?.load?.sampling).toEqual(baseS)
  })

  it('startLoad does not take a mode that differs only in sampling for a hand-made load', () => {
    const base = withS({ temperature: 0.7 })
    const modes = { re: own({ load: withS({ temperature: 1 }) }) }
    expect(startLoad(base, modes, null, withS({ temperature: 1 }))).toBe(base)
  })
})

describe('what the picker saves for a mode in one call', () => {
  const baseS = { ...DEFAULT_LOAD_CONFIG.sampling, temperature: 0.7 }
  const cfg = { ...DEFAULT_LOAD_CONFIG, contextLength: 16384, sampling: baseS }
  const stored = { ...DEFAULT_LOAD_CONFIG, contextLength: 4096, sampling: { ...DEFAULT_LOAD_CONFIG.sampling, temperature: 1 } }
  const mine = { ...DEFAULT_SAMPLING, temperature: 0.3, jsonSchema: '{"a":1}' }

  it('saves nothing but what changed, and keeps the prompt', () => {
    const existing = own({ load: stored, sampling: mine, systemPrompt: 'hi' })
    expect(withModeSaves(existing, baseS, {})).toEqual(existing)
    expect(withModeSaves(existing, baseS, { load: cfg })).toEqual({ ...existing, load: { ...cfg, sampling: stored.sampling } })
    const next = { ...DEFAULT_SAMPLING, temperature: 0.9 }
    expect(withModeSaves(existing, baseS, { sampling: next })).toEqual({ load: { ...stored, sampling: baseS }, sampling: next, systemPrompt: 'hi' })
    expect(withModeSaves(existing, baseS, { sampling: null })).toEqual({ ...existing, sampling: null })
  })

  it('merges the load and the sampling in one entry; a saved jsonSchema is always empty', () => {
    const got = withModeSaves(undefined, baseS, { load: cfg, sampling: mine })
    expect(got).toEqual({ load: { ...cfg, sampling: baseS }, sampling: { ...mine, jsonSchema: '' }, systemPrompt: '' })
  })

  it('own sampling alone leaves a mode without its own load without one', () => {
    expect(withModeSaves(undefined, baseS, { sampling: mine })?.load).toBeNull()
  })

  it('is null when nothing is left', () => {
    expect(withModeSaves(own({ sampling: mine }), baseS, { sampling: null })).toBeNull()
    expect(withModeSaves(undefined, baseS, {})).toBeNull()
  })
})

describe("the picker's OpenRouter panel", () => {
  const mine = { ...DEFAULT_SAMPLING, temperature: 0.3 }

  it('starts with the mode of the page, and with Chat where the page has none', () => {
    expect(cloudPickerMode('chat', false)).toBe('chat')
    expect(cloudPickerMode('chat', true)).toBe('re')
    expect(cloudPickerMode('writing', false)).toBe('writing')
    expect(cloudPickerMode('openrouter', false)).toBe('chat')
    expect(cloudPickerMode('models', true)).toBe('chat')
  })

  it('saves the sampling and keeps the prompt, and nothing is left when it is all off', () => {
    expect(samplingSave(undefined, { on: true, sampling: mine })).toEqual({ load: null, sampling: { ...mine, jsonSchema: '' }, systemPrompt: '' })
    const existing = own({ sampling: mine, systemPrompt: 'hi' })
    expect(samplingSave(existing, { on: true, sampling: { ...mine, temperature: 1 } })).toEqual({ ...existing, sampling: { ...mine, jsonSchema: '', temperature: 1 } })
    expect(samplingSave(existing, { on: false, sampling: mine })).toEqual({ ...existing, sampling: null })
    expect(samplingSave(own({ sampling: mine }), { on: false, sampling: mine })).toBeNull()
    expect(samplingSave(undefined, { on: true, sampling: null })).toBeNull()
  })
})

describe('when the reload bar shows', () => {
  const base = loadWith({ contextLength: 4096 })
  const wanted = loadWith({ contextLength: 32768 })
  const modes = { writing: own({ load: wanted }) }

  it('offers the mode\'s settings when the model runs with others', () => {
    expect(reloadOffer(base, base, modes, 'writing', false)).toEqual(wanted)
  })
  it('stays away when the model already runs with them', () => {
    expect(reloadOffer({ ...wanted }, base, modes, 'writing', false)).toBeNull()
  })
  it('stays away for a mode without own load settings when the model runs its default ones', () => {
    expect(reloadOffer(base, base, modes, 'chat', false)).toBeNull()
  })
  it('offers the default settings when it runs with another mode\'s settings', () => {
    expect(reloadOffer(wanted, base, modes, 'chat', false)).toBe(base)
  })
  it('is hidden once kept as is, and without a model, a running config or a base', () => {
    expect(reloadOffer(base, base, modes, 'writing', true)).toBeNull()
    expect(reloadOffer(null, base, modes, 'writing', false)).toBeNull()
    expect(reloadOffer(base, null, modes, 'writing', false)).toBeNull()
    expect(reloadOffer(base, base, undefined, 'writing', false)).toBeNull()
  })
})

describe('the chat store', () => {
  let useChat: typeof import('../src/renderer/src/stores/chat').useChat
  let useModels: typeof import('../src/renderer/src/stores/models').useModels
  let useApp: typeof import('../src/renderer/src/stores/app').useApp

  const entry = (id: string): ModelEntry => ({ id, name: id, path: id, sizeBytes: 1, files: [], modifiedAt: 0 }) as unknown as ModelEntry
  const settings = { defaultSampling: DEFAULT_SAMPLING, defaultSystemPrompt: 'APP PROMPT', re: { modelId: '' } } as unknown as AppSettings
  const modes: AllModeDefaults = {
    A: { chat: own({ sampling: { ...DEFAULT_SAMPLING, temperature: 0.11 }, systemPrompt: 'Be A.' }), re: own({ systemPrompt: 'RE note for A.', sampling: { ...DEFAULT_SAMPLING, temperature: 0.22 } }) },
    B: { chat: own({ systemPrompt: 'Be B.' }) }
  }
  const tick = (): Promise<void> => new Promise((r) => setTimeout(r, 0))

  beforeAll(async () => {
    ;({ useChat } = await import('../src/renderer/src/stores/chat'))
    ;({ useModels } = await import('../src/renderer/src/stores/models'))
    ;({ useApp } = await import('../src/renderer/src/stores/app'))
    await useChat.getState().init()
  })

  beforeEach(() => {
    calls.length = 0
    useApp.setState({ settings })
    useModels.setState({ models: [entry('A'), entry('B')], instances: [], modeDefaults: modes, activeModelId: null })
    useChat.setState({ current: null, generation: null, handoff: null, autoLoadingModelId: null, poolChats: [] })
  })

  it('a new chat starts with the active model\'s chat sampling and prompt', async () => {
    useModels.setState({ activeModelId: 'A' })
    const c = await useChat.getState().newChat()
    expect(c.systemPrompt).toBe('Be A.')
    expect(c.sampling.temperature).toBe(0.11)
    expect(c.modelId).toBe('A')
  })

  it('a model without chat settings gives the app defaults', async () => {
    const c = await useChat.getState().newChat()
    expect(c.systemPrompt).toBe('APP PROMPT')
    expect(deepEqual(c.sampling, DEFAULT_SAMPLING)).toBe(true)
  })

  it('an untouched empty chat follows the active model, without bumping updatedAt', async () => {
    useModels.setState({ activeModelId: 'A' })
    const first = await useChat.getState().newChat()
    useModels.setState({ activeModelId: 'B' })
    const cur = useChat.getState().current!
    expect(cur.id).toBe(first.id)
    expect(cur.systemPrompt).toBe('Be B.')
    expect(deepEqual(cur.sampling, DEFAULT_SAMPLING)).toBe(true)
    expect(cur.updatedAt).toBe(first.updatedAt)
    expect(cur.modelId).toBe('B')
  })

  it('follows a change of the model\'s chat defaults too', async () => {
    useModels.setState({ activeModelId: 'B' })
    await useChat.getState().newChat()
    useModels.setState({ modeDefaults: { ...modes, B: { chat: own({ systemPrompt: 'Be B2.' }) } } })
    expect(useChat.getState().current?.systemPrompt).toBe('Be B2.')
  })

  it('a new chat that follows another model while it is being saved is returned as it is then', async () => {
    useModels.setState({ activeModelId: 'A' })
    const made = useChat.getState().newChat()
    useModels.setState({ activeModelId: 'B' })
    const c = await made
    expect(c.systemPrompt).toBe('Be B.')
    expect(c.modelId).toBe('B')
  })

  it('keeps its model when nothing is loaded any more, so sending a message loads it', async () => {
    useModels.setState({ activeModelId: 'A' })
    await useChat.getState().newChat()
    useModels.setState({ activeModelId: null })
    const cur = useChat.getState().current!
    expect(cur.modelId).toBe('A')
    expect(cur.systemPrompt).toBe('Be A.')
  })

  it('leaves an edited prompt alone but still moves the untouched sampling', async () => {
    useModels.setState({ activeModelId: 'A' })
    await useChat.getState().newChat()
    useChat.getState().mutateCurrent((c) => ({ ...c, systemPrompt: 'mine' }))
    useModels.setState({ activeModelId: 'B' })
    const cur = useChat.getState().current!
    expect(cur.systemPrompt).toBe('mine')
    expect(cur.sampling.temperature).toBe(DEFAULT_SAMPLING.temperature)
  })

  it('leaves chats with messages and pool chats alone', async () => {
    useModels.setState({ activeModelId: 'A' })
    const c = await useChat.getState().newChat()
    useChat.setState({ current: { ...c, messages: [{ id: 'm', role: 'user', content: 'hi' } as never] } })
    useModels.setState({ activeModelId: 'B' })
    expect(useChat.getState().current?.systemPrompt).toBe('Be A.')

    useChat.setState({ current: { ...c, id: 'pool1' }, poolChats: ['pool1'] })
    useModels.setState({ activeModelId: 'A' })
    expect(useChat.getState().current?.systemPrompt).toBe('Be A.')
  })

  it('newChat returning the existing empty chat leaves it right', async () => {
    useModels.setState({ activeModelId: 'A' })
    const first = await useChat.getState().newChat()
    // The model changed while a generation blocked following; the chat is empty again afterwards.
    useChat.setState({ generation: { requestId: 'r' } as never })
    useModels.setState({ activeModelId: 'B' })
    expect(useChat.getState().current?.systemPrompt).toBe('Be A.')
    useChat.setState({ generation: null })
    const again = await useChat.getState().newChat()
    expect(again.id).toBe(first.id)
    expect(again.systemPrompt).toBe('Be B.')
  })

  it('startRe uses the model\'s reverse-engineering sampling and adds its prompt text, and loads for that mode', async () => {
    useModels.setState({ activeModelId: 'A' })
    const id = await useChat.getState().startRe({ target: 'C:\\x\\app.exe', goal: 'Find it', collect: false, modelId: 'A' })
    await tick()
    expect(id).toBeTruthy()
    const cur = useChat.getState().current!
    expect(cur.systemPrompt).toBe(`${RE_SYSTEM_PROMPT}\n\nRE note for A.`)
    expect(cur.sampling.temperature).toBe(0.22)
    const ensure = calls.find((c) => c.channel === 'instances:ensure')
    expect(ensure?.args).toEqual(['A', 're'])
  })

  it('startRe without own settings keeps the built-in prompt and sampling', async () => {
    useModels.setState({ activeModelId: 'B' })
    await useChat.getState().startRe({ target: 'C:\\x\\app.exe', goal: 'Find it', collect: false, modelId: 'B' })
    await tick()
    const cur = useChat.getState().current!
    expect(cur.systemPrompt).toBe(RE_SYSTEM_PROMPT)
    expect(deepEqual(cur.sampling, reSampling(DEFAULT_SAMPLING))).toBe(true)
  })
})
