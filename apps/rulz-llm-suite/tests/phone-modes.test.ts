// The phone's writing:mode and writing:reloadMode, and the rentals in writing:models (docs/app-modes-design.md, 1.3).
import { describe, expect, it } from 'vitest'
import { DEFAULT_LOAD_CONFIG, DEFAULT_SAMPLING } from '@shared/defaults'
import type { ModelModeDefaults } from '@shared/modes'
import { PHONE_INVOKE, PHONE_WRITING_CHANNELS } from '@shared/phone'
import type { LoadConfig, LoadedInstance, ModelEntry } from '@shared/types'
import type { InstanceManager } from '../src/main/engine/instances'
import { PhoneHandlers } from '../src/main/phone/registry'
import type { ModeReloadResult } from '../src/main/server/apiServer'
import { modelsAdapter } from '../src/main/writing/adapters'
import { writingHandlers } from '../src/main/writing/ipc'
import { LEASED, PhoneModes, RENTAL_RELOAD, type PhoneModesDeps } from '../src/main/writing/modeInfo'
import { WritingRuns } from '../src/main/writing/runs'

const entry = (id: string, remote = false): ModelEntry => ({ id, name: id.toUpperCase(), folder: '', path: id, files: [], sizeBytes: 1, info: null, modifiedAt: 0, ...(remote ? { remote: true } : {}) })
const LOCAL = entry('local')
const RENTED = entry('vast/r', true)
const WRITING_LOAD: LoadConfig = { ...DEFAULT_LOAD_CONFIG, contextLength: 32768 }

const inst = (modelId: string, over: Partial<LoadedInstance> = {}): LoadedInstance => ({
  instanceId: `i-${modelId}`,
  modelId,
  modelName: modelId,
  port: 1,
  pid: 1,
  status: 'ready',
  progress: 1,
  stage: '',
  config: DEFAULT_LOAD_CONFIG,
  startedAt: 1,
  contextLength: 8192,
  hasVision: false,
  estimate: null,
  adapters: [],
  ...over
})

function setup(over: { modes?: ModelModeDefaults; running?: LoadedInstance[]; block?: string | null; leased?: boolean; result?: ModeReloadResult } = {}) {
  const calls: { model: string; mode: string; also: string | null }[] = []
  const state = { running: over.running ?? [inst('local')], block: over.block ?? null, leased: over.leased ?? false, result: over.result ?? ({ status: 'started' } as ModeReloadResult) }
  const modes: ModelModeDefaults = over.modes ?? { writing: { load: WRITING_LOAD, sampling: DEFAULT_SAMPLING, systemPrompt: 'Write well.' } }
  const deps: PhoneModesDeps = {
    getEntry: (id) => [LOCAL, RENTED].find((m) => m.id === id),
    newestReady: () => [...state.running].sort((a, b) => b.startedAt - a.startedAt)[0]?.modelId ?? null,
    readyFor: (id) => state.running.find((i) => i.modelId === id && i.status === 'ready'),
    modeDefaults: (id) => (id === 'local' ? modes : {}),
    loadDefaults: () => DEFAULT_LOAD_CONFIG,
    reloadBlock: () => state.block,
    leased: () => state.leased,
    reload: (model, mode, also) => {
      calls.push({ model: model.id, mode, also: also() })
      return state.result
    }
  }
  return { phone: new PhoneModes(deps), calls, state }
}

describe('writing:mode', () => {
  it('says what applies to the mode and offers a reload when the model runs with other settings', () => {
    const t = setup()
    expect(t.phone.info('writing', 'local')).toEqual({
      mode: 'writing',
      modelId: 'local',
      name: 'LOCAL',
      loaded: true,
      ownLoad: true,
      ownSampling: true,
      hasPrompt: true,
      reload: { label: 'Writing settings', blocked: null }
    })
    // Adventure has nothing of its own for this model.
    expect(t.phone.info('adventure', 'local')).toMatchObject({ ownLoad: false, ownSampling: false, hasPrompt: false, reload: null })
  })

  it('offers nothing when the model already runs with the mode\'s settings, or is not loaded', () => {
    expect(setup({ running: [inst('local', { config: WRITING_LOAD })] }).phone.info('writing', 'local').reload).toBeNull()
    const none = setup({ running: [] }).phone.info('writing', 'local')
    expect(none).toMatchObject({ loaded: false, reload: null, ownLoad: true })
  })

  it('names why a reload cannot be done now: the app\'s reason first, then a lease', () => {
    expect(setup({ block: 'LLM Suite is loading a model.', leased: true }).phone.info('writing', 'local').reload).toEqual({ label: 'Writing settings', blocked: 'LLM Suite is loading a model.' })
    expect(setup({ leased: true }).phone.info('writing', 'local').reload).toEqual({ label: 'Writing settings', blocked: LEASED })
    expect(LEASED).toBe('RulZ Coding Agent is using the model.')
  })

  it('uses the newest loaded model when none is given, and says so when there is none', () => {
    const t = setup({ running: [inst('local', { startedAt: 1 }), inst('vast/r', { startedAt: 5, host: { kind: 'vast', rentalId: 1, label: 'RTX 4090' } })] })
    expect(t.phone.info('writing', null)).toMatchObject({ modelId: 'vast/r', loaded: true, rental: { label: 'RTX 4090', reconnecting: false } })
    expect(setup({ running: [] }).phone.info('writing', null)).toEqual({ mode: 'writing', modelId: null, name: '', loaded: false, ownLoad: false, ownSampling: false, hasPrompt: false, reload: null })
  })

  it('shows a rental with its label, null while it is not loaded there, and never lets a phone reload it', () => {
    const offline = setup({ running: [] }).phone.info('writing', 'vast/r')
    expect(offline).toMatchObject({ loaded: false, rental: { label: null, reconnecting: false } })
    const on = setup({ running: [inst('vast/r', { host: { kind: 'vast', rentalId: 1, label: 'RTX 4090' }, reconnecting: true })], modes: {} })
    expect(on.phone.info('writing', 'vast/r').rental).toEqual({ label: 'RTX 4090', reconnecting: true })
    // On a rental that runs with settings the mode would change, the phone is told where to do it.
    const other = setup({ running: [inst('vast/r', { host: { kind: 'vast', rentalId: 1, label: 'RTX 4090' }, config: { ...DEFAULT_LOAD_CONFIG, contextLength: 4096 } })] })
    const deps = (other.phone as unknown as { deps: PhoneModesDeps }).deps
    deps.modeDefaults = () => ({ writing: { load: WRITING_LOAD, sampling: null, systemPrompt: '' } })
    expect(other.phone.info('writing', 'vast/r').reload).toEqual({ label: 'Writing settings', blocked: RENTAL_RELOAD })
  })
})

describe('writing:reloadMode', () => {
  it('starts the reload and answers that the model is loading', () => {
    const t = setup()
    const info = t.phone.reloadMode('writing', 'local')
    expect(t.calls).toEqual([{ model: 'local', mode: 'writing', also: null }])
    expect(info).toMatchObject({ modelId: 'local', loaded: false, reload: null })
  })

  it('hands the lease to the API\'s reload as the phone\'s own reason, and says why when it is refused', () => {
    const t = setup({ leased: true, result: { status: 'blocked', why: LEASED } })
    expect(() => t.phone.reloadMode('writing', 'local')).toThrow(LEASED)
    expect(t.calls[0].also).toBe(LEASED)
    expect(() => setup({ result: { status: 'blocked', why: 'The model is in use.' } }).phone.reloadMode('adventure', 'local')).toThrow('The model is in use.')
  })

  it('refuses a rental\'s model and a model that is not in the library, without asking the API', () => {
    const t = setup()
    expect(() => t.phone.reloadMode('writing', 'vast/r')).toThrow(RENTAL_RELOAD)
    expect(() => t.phone.reloadMode('writing', 'missing')).toThrow(/not in the library/)
    expect(t.calls).toEqual([])
  })

  it('returns the plain info when nothing needed to happen', () => {
    const t = setup({ result: { status: 'matches' } })
    expect(t.phone.reloadMode('writing', 'local')).toMatchObject({ loaded: true, reload: { label: 'Writing settings' } })
  })
})

describe('the phone channels', () => {
  const runs = {} as WritingRuns

  it('are phone channels, and any phone may call them without a project', async () => {
    expect(PHONE_WRITING_CHANNELS).toContain('writing:mode')
    expect(PHONE_WRITING_CHANNELS).toContain('writing:reloadMode')
    expect(PHONE_INVOKE.has('writing:mode')).toBe(true)
    expect(PHONE_INVOKE.has('writing:reloadMode')).toBe(true)
    const t = setup()
    const h = writingHandlers(runs, t.phone)
    const registry = new PhoneHandlers({ projectFolders: async () => [], ownsProject: () => false })
    for (const channel of ['writing:mode', 'writing:reloadMode'] as const) registry.register(channel, ((owner: string, ...args: unknown[]) => h[channel](owner, ...args)) as never)
    const info = (await registry.invoke('phone:1', 'writing:mode', ['writing', null])) as { modelId: string }
    expect(info.modelId).toBe('local')
    expect(((await registry.invoke('phone:1', 'writing:reloadMode', ['adventure', 'local'])) as { mode: string }).mode).toBe('adventure')
  })

  it('check their arguments: the mode is exactly one of two strings, the model a string or null of a sensible length', async () => {
    const h = writingHandlers(runs, setup().phone)
    for (const mode of ['chat', 'coding', 're', 'Writing', '', null, undefined, 1, { mode: 'writing' }, ['writing']]) {
      await expect(h['writing:mode']('phone:1', mode, null)).rejects.toThrow(/mode must be/)
      await expect(h['writing:reloadMode']('phone:1', mode, 'local')).rejects.toThrow(/mode must be/)
    }
    for (const id of [undefined, 5, {}, [], '', 'x'.repeat(1025)]) await expect(h['writing:mode']('phone:1', 'writing', id)).rejects.toThrow(/model/)
    // reloadMode needs a real id: null is refused there.
    await expect(h['writing:reloadMode']('phone:1', 'writing', null)).rejects.toThrow(/model/)
    await expect(h['writing:mode']('phone:1', 'writing', 'x'.repeat(1024))).resolves.toMatchObject({ loaded: false })
    await expect(h['writing:mode']('phone:1', 'adventure', null)).resolves.toMatchObject({ modelId: 'local' })
  })

  it('are not available when main did not plug the modes in', async () => {
    await expect(writingHandlers(runs)['writing:mode']('phone:1', 'writing', null)).rejects.toThrow(/not available/)
  })
})

describe('writing:models with rentals', () => {
  const fake = (running: LoadedInstance[]): InstanceManager => ({ readyFor: (id: string) => running.find((i) => i.modelId === id && i.status === 'ready'), list: () => running }) as unknown as InstanceManager
  const make = (running: LoadedInstance[]): WritingRuns =>
    new WritingRuns({ store: {} as never, chat: {} as never, models: modelsAdapter({ list: () => [LOCAL, RENTED] }, fake(running)), emit: () => undefined })

  it('adds a rental to a rented model: its label while it is loaded there, null when it is not', () => {
    expect(make([]).models()).toEqual([
      { id: 'local', name: 'LOCAL', loaded: false },
      { id: 'vast/r', name: 'VAST/R', loaded: false, rental: { label: null, reconnecting: false } }
    ])
    const host = { kind: 'vast' as const, rentalId: 2, label: 'RTX 5090' }
    expect(make([inst('vast/r', { host }), inst('local')]).models()).toEqual([
      { id: 'local', name: 'LOCAL', loaded: true },
      { id: 'vast/r', name: 'VAST/R', loaded: true, rental: { label: 'RTX 5090', reconnecting: false } }
    ])
    expect(make([inst('vast/r', { host, reconnecting: true })]).models()[1].rental).toEqual({ label: 'RTX 5090', reconnecting: true })
  })
})
