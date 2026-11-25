import { describe, expect, it } from 'vitest'
import { practiceScope, reloadOffer, rentalOf, triesText } from '../src/shared/modes'
import type { AppSettings, ConnectionInfo, ModelInfo, ModelModes, ModeSettings } from '../src/shared/types'
import { DEFAULT_SETTINGS } from '../src/shared/defaults'
import { batchRentalGate, chosenRentalGone, createModesStore, isUnknownChannel, openingSettingsFor, type ModesDeps } from '../src/lib/modeLogic'

const mode = (p: Partial<ModeSettings> = {}): ModeSettings => ({ ownLoad: false, sampling: null, thinking: null, systemPrompt: '', matches: true, ...p })
const modes = (instance: string, coding: Partial<ModeSettings> = {}, re: Partial<ModeSettings> = {}): ModelModes => ({
  id: 'm',
  state: 'ready',
  instance,
  apiOverride: null,
  modes: { coding: mode(coding), re: mode(re) }
})

/** Answers the test releases one by one, so it can decide which arrives first. */
function fake() {
  const asked: string[] = []
  const waiting: { model: string; resolve: (m: ModelModes | null) => void; reject: (e: Error) => void }[] = []
  const reloads: { model: string; mode: string; resolve: (m: ModelModes) => void; reject: (e: Error) => void }[] = []
  const deps: ModesDeps = {
    modes: (model) =>
      new Promise((resolve, reject) => {
        asked.push(model)
        waiting.push({ model, resolve, reject })
      }),
    reloadMode: (model, m) => new Promise((resolve, reject) => reloads.push({ model, mode: m, resolve, reject }))
  }
  return { deps, asked, waiting, reloads }
}
const tick = (): Promise<void> => new Promise((r) => setTimeout(r, 0))

describe('the reload bar offer and "Keep as is"', () => {
  it('is offered only when the settings do not match', () => {
    expect(reloadOffer(modes('a', { matches: false }), 'coding', undefined)).toEqual({ label: 'Reload with Coding settings' })
    expect(reloadOffer(modes('a', { matches: true }), 'coding', undefined)).toBeNull()
    expect(reloadOffer(modes('a', {}, { matches: false }), 're', undefined)).toEqual({ label: 'Reload with Reverse engineering settings' })
    expect(reloadOffer(null, 'coding', undefined)).toBeNull()
  })
  it('stays away until the instance changes, per model, in memory', async () => {
    const f = fake()
    const store = createModesStore(f.deps)
    store.getState().want('m')
    const p = store.getState().refresh('m', true)
    f.waiting[0].resolve(modes('a', { matches: false }))
    await p
    expect(reloadOffer(store.getState().byModel.m ?? null, 'coding', store.getState().kept.m)).not.toBeNull()
    store.getState().keep('m')
    expect(reloadOffer(store.getState().byModel.m ?? null, 'coding', store.getState().kept.m)).toBeNull()
    const p2 = store.getState().refresh('m', true)
    f.waiting[1].resolve(modes('b', { matches: false }))
    await p2
    expect(reloadOffer(store.getState().byModel.m ?? null, 'coding', store.getState().kept.m)).not.toBeNull()
  })
  it('forgets the choice when another computer is picked', async () => {
    const f = fake()
    const store = createModesStore(f.deps)
    store.getState().forComputer('pc1')
    store.getState().want('m')
    const p = store.getState().refresh('m', true)
    f.waiting[0].resolve(modes('a', { matches: false }))
    await p
    store.getState().keep('m')
    store.getState().forComputer('pc2')
    expect(store.getState().kept).toEqual({})
    expect(store.getState().byModel).toEqual({})
  })
})

describe('asking for the modes', () => {
  it('never runs two questions at once, and a question for another model waits', async () => {
    const f = fake()
    const store = createModesStore(f.deps)
    store.getState().want('a')
    const first = store.getState().refresh('a', true)
    void store.getState().refresh('a', true)
    expect(f.asked).toEqual(['a'])
    store.getState().want('b')
    void store.getState().refresh('b', true)
    expect(f.asked).toEqual(['a'])
    f.waiting[0].resolve(modes('x'))
    await first
    await tick()
    expect(f.asked).toEqual(['a', 'b'])
  })
  it('does not ask again within the gap unless forced', async () => {
    const f = fake()
    let t = 1_000_000
    const store = createModesStore(f.deps, () => t)
    store.getState().want('a')
    const p = store.getState().refresh('a')
    f.waiting[0].resolve(modes('x'))
    await p
    await store.getState().refresh('a')
    expect(f.asked).toEqual(['a'])
    t += 2500
    void store.getState().refresh('a')
    expect(f.asked).toEqual(['a', 'a'])
  })
  it('drops an answer for the model that is no longer shown', async () => {
    const f = fake()
    const store = createModesStore(f.deps)
    store.getState().want('a')
    const p = store.getState().refresh('a', true)
    store.getState().want('b')
    f.waiting[0].resolve(modes('x'))
    await p
    expect(store.getState().byModel.a).toBeUndefined()
  })
  it('drops an answer that was asked on another computer', async () => {
    const f = fake()
    const store = createModesStore(f.deps)
    store.getState().forComputer('pc1')
    store.getState().want('a')
    const p = store.getState().refresh('a', true)
    store.getState().forComputer('pc2')
    f.waiting[0].resolve(modes('x'))
    await p
    expect(store.getState().byModel.a).toBeUndefined()
  })
  it('asks the new computer once the old one\'s question has ended, for the same model too', async () => {
    const f = fake()
    const store = createModesStore(f.deps)
    store.getState().forComputer('pc1')
    store.getState().want('a')
    void store.getState().refresh('a', true)
    store.getState().forComputer('pc2')
    void store.getState().refresh('a', true)
    expect(f.asked).toEqual(['a'])
    f.waiting[0].reject(new Error('The connection was closed.'))
    await tick()
    expect(f.asked).toEqual(['a', 'a'])
    f.waiting[1].resolve(modes('pc2'))
    await tick()
    expect(store.getState().byModel.a?.instance).toBe('pc2')
  })
  it('lets a reload win over a slower, older question', async () => {
    const f = fake()
    const store = createModesStore(f.deps)
    store.getState().want('a')
    const q = store.getState().refresh('a', true)
    const r = store.getState().reload('a', 'coding')
    f.reloads[0].resolve(modes('new'))
    await r
    f.waiting[0].resolve(modes('old', { matches: false }))
    await q
    expect(store.getState().byModel.a?.instance).toBe('new')
  })
  it('passes the computer’s refusal of a reload on', async () => {
    const f = fake()
    const store = createModesStore(f.deps)
    const r = store.getState().reload('a', 're')
    f.reloads[0].reject(new Error('Wait for it to finish or press Stop, then reload the model.'))
    await expect(r).rejects.toThrow('press Stop')
  })
  it('keeps what was known when a question fails', async () => {
    const f = fake()
    const store = createModesStore(f.deps)
    store.getState().want('a')
    const p = store.getState().refresh('a', true)
    f.waiting[0].resolve(modes('x'))
    await p
    const p2 = store.getState().refresh('a', true)
    f.waiting[1].reject(new Error('Your computer took too long to answer.'))
    await p2
    expect(store.getState().byModel.a?.instance).toBe('x')
    expect(store.getState().older).toBe(false)
  })
})

describe('an older computer or LLM Suite', () => {
  it('knows an unknown channel by the computer’s words', () => {
    expect(isUnknownChannel(new Error('Unknown request "models:modes".'))).toBe(true)
    expect(isUnknownChannel(new Error('Not connected to my-pc.'))).toBe(false)
  })
  it('hides the controls and stops asking after "Unknown request"', async () => {
    const f = fake()
    const store = createModesStore(f.deps)
    store.getState().want('a')
    const p = store.getState().refresh('a', true)
    f.waiting[0].reject(new Error('Unknown request "models:modes".'))
    await p
    expect(store.getState().older).toBe(true)
    expect(store.getState().byModel.a).toBeNull()
    await store.getState().refresh('a', true)
    expect(f.asked).toEqual(['a'])
    // Its app may have been updated by the time the phone connects again.
    void store.getState().reconnected()
    expect(store.getState().older).toBe(false)
    expect(f.asked).toEqual(['a', 'a'])
    f.waiting[1].reject(new Error('Unknown request "models:modes".'))
    await tick()
    expect(store.getState().older).toBe(true)
    // Another computer may be newer.
    store.getState().forComputer('pc2')
    expect(store.getState().older).toBe(false)
  })
  it('treats a LLM Suite without modes (null) as nothing to offer', async () => {
    const f = fake()
    const store = createModesStore(f.deps)
    store.getState().want('a')
    const p = store.getState().refresh('a', true)
    f.waiting[0].resolve(null)
    await p
    expect(store.getState().byModel.a).toBeNull()
    expect(openingSettingsFor(store.getState().byModel.a)).toBe('all')
    expect(reloadOffer(store.getState().byModel.a ?? null, 'coding', undefined)).toBeNull()
  })
  it('counts all models together when the answer has no model', () => {
    expect(practiceScope({ kitDir: 'k', tasks: [] })).toBe('All models together')
    expect(practiceScope({ kitDir: 'k', tasks: [], model: 'x/qwen.gguf' }, (s) => s.replace(/^.*\//, ''))).toBe('For qwen.gguf')
  })
})

describe('"Settings for" opening choice', () => {
  it('opens on Coding when the model has its own Coding load settings, else on all modes', () => {
    expect(openingSettingsFor(modes('a', { ownLoad: true }))).toBe('coding')
    expect(openingSettingsFor(modes('a', { ownLoad: false }, { ownLoad: true }))).toBe('all')
    expect(openingSettingsFor(null)).toBe('all')
    expect(openingSettingsFor(undefined)).toBe('all')
  })
})

describe('the rental yes for a batch run', () => {
  const rental = { label: 'RTX 5090', reconnecting: false }
  it('needs the switch on before Start, and sends rentalOk only then', () => {
    expect(batchRentalGate(rental, false)).toEqual({ needsYes: true, canStart: false, rentalOk: false })
    expect(batchRentalGate(rental, true)).toEqual({ needsYes: true, canStart: true, rentalOk: true })
  })
  it('asks nothing for a local model, even with the switch left on', () => {
    expect(batchRentalGate(undefined, true)).toEqual({ needsYes: false, canStart: true, rentalOk: false })
  })
})

describe('a chosen rental that is gone', () => {
  const info = (id: string, rental?: ModelInfo['rental']): ModelInfo => ({ id, loaded: true, ...(rental ? { rental } : {}) }) as ModelInfo
  const conn = (models: ModelInfo[], status: ConnectionInfo['status'] = 'connected'): ConnectionInfo => ({ status, url: '', models }) as ConnectionInfo
  const settings = (model: string, modelRental: boolean): AppSettings => ({ ...DEFAULT_SETTINGS, model, modelRental })

  it('says so, and only then', () => {
    expect(chosenRentalGone(settings('vast/qwen', true), conn([info('local')]))).toMatch(/is not loaded on a rental now/)
    expect(chosenRentalGone(settings('vast/qwen', true), conn([info('vast/qwen', { label: null, reconnecting: false })]))).toBeNull()
    expect(chosenRentalGone(settings('local-gone', false), conn([info('local')]))).toBeNull()
    expect(chosenRentalGone(settings('moved', false), conn([info('moved', { label: null, reconnecting: false })]))).toMatch(/Choose it again/)
    expect(chosenRentalGone(settings('vast/qwen', true), conn([info('local')], 'unreachable' as ConnectionInfo['status']))).toBeNull()
    expect(chosenRentalGone(settings('', true), conn([info('local')]))).toBeNull()
  })
  it('reads a rental from the model list, or from a vast/ path', () => {
    expect(rentalOf({ rental: { label: 'A', reconnecting: true } })).toEqual({ label: 'A', reconnecting: true })
    expect(rentalOf({ path: 'vast/123/model.gguf' })).toEqual({ label: null, reconnecting: false })
    expect(rentalOf({ path: 'C:/models/a.gguf' })).toBeUndefined()
  })
})

describe('practice tries', () => {
  it('shows failed tries when LLM Suite sends them', () => {
    expect(triesText({ attempts: 5, solved: 2, failed: 1 })).toBe('solved 2/5 · 1 failed')
    expect(triesText({ attempts: 3, solved: 0 })).toBe('0/3')
  })
})
