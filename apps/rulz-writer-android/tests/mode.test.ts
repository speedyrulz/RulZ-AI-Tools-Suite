import { describe, expect, it, vi } from 'vitest'
import type { PhoneModeInfo, PhoneModel } from '../src/shared/phone'
import { applies, canPickModel, cloudLabel, isCloudId, LOADING_LIMIT_MS, missingModelLabel, ModeController, rentalLabel } from '../src/lib/mode'

const base = (): PhoneModeInfo => ({ mode: 'writing', modelId: 'm', name: 'Model', loaded: true, ownLoad: true, ownSampling: true, hasPrompt: true, reload: { label: 'Writing settings', blocked: null } })
const deferred = <T,>() => { let resolve!: (value: T) => void; let reject!: (error: Error) => void; const promise = new Promise<T>((a, b) => { resolve = a; reject = b }); return { promise, resolve, reject } }
describe('OpenRouter models on the phone', () => {
  const cloud: PhoneModel = { id: 'openrouter/vendor/m', name: 'Model M', loaded: true, cloud: { free: true } }
  it('can always be picked, and is labelled as OpenRouter, never as a rental', () => {
    expect(canPickModel(cloud, null)).toBe(true)
    expect(canPickModel({ ...cloud, cloud: { free: false } }, 'other')).toBe(true)
    expect(rentalLabel(cloud)).toBe('OpenRouter, Free')
    expect(cloudLabel({ cloud: { free: false } })).toBe('OpenRouter, Paid')
    // Rentals and local models are as before.
    expect(canPickModel({ id: 'l', name: 'L', loaded: false }, null)).toBe(true)
    expect(canPickModel({ id: 'r', name: 'R', loaded: false, rental: { label: null, reconnecting: false } }, null)).toBe(false)
    expect(rentalLabel({ id: 'r', name: 'R', loaded: false, rental: { label: 'GPU', reconnecting: false } })).toBe('GPU')
  })
  it('survives hostile values from the computer: anything but free: true is paid', () => {
    for (const cloudValue of ['x', {}, { free: 'yes' }, { free: 1 }, null, [], { provider: 'other' }]) {
      expect(cloudLabel({ cloud: cloudValue })).toBe('OpenRouter, Paid')
    }
    expect(cloudLabel({ cloud: { free: true } })).toBe('OpenRouter, Free')
  })
  it('a chosen model that is no longer listed is named, not replaced: an OpenRouter one says so', () => {
    expect(isCloudId('openrouter/vendor/m')).toBe(true)
    expect(isCloudId('Qwen3-8B')).toBe(false)
    expect(missingModelLabel('openrouter/vendor/m')).toBe('vendor/m (OpenRouter, not available now)')
    expect(missingModelLabel('Qwen3-8B')).toBe('Qwen3-8B')
  })
})

describe('mode information', () => {
  it('never reloads a cloud model', async () => {
    const info = { ...base(), cloud: { free: false } }
    const reload = vi.fn(async () => info)
    const c = new ModeController('cloud', 'writing', 'm', async () => info, reload, () => {})
    await c.refresh()
    await c.reload()
    expect(reload).not.toHaveBeenCalled()
  })
  it('shows applicable settings', () => {
    expect(applies(base())).toBe('own load settings, own prompt')
    expect(applies({ ...base(), mode: 'adventure' })).toBe('own load settings, own sampling, own prompt')
  })
  it('keeps until unloaded or reload disappears, separately per computer', async () => {
    let info = base()
    const make = (id: string) => new ModeController(id, 'writing', 'm', async () => info, async () => info, () => {})
    const c = make('keep-test'); await c.refresh(); c.keep(); expect(c.isKept).toBe(true)
    const other = make('other-test'); await other.refresh(); expect(other.isKept).toBe(false)
    info = { ...base(), loaded: false }; await c.refresh(); info = base(); await c.refresh(); expect(c.isKept).toBe(false)
    c.keep(); info = { ...base(), reload: null }; await c.refresh(); info = base(); await c.refresh(); expect(c.isKept).toBe(false)
  })
  it('drops stale answers and stops on unknown channel until reconnect', async () => {
    const d = deferred<PhoneModeInfo>(); const ask = vi.fn().mockReturnValueOnce(d.promise).mockRejectedValueOnce(new Error('That is not available from the phone.')).mockResolvedValue(base())
    const c = new ModeController('stale', 'writing', null, ask, async () => base(), () => {})
    const pending = c.refresh(); c.invalidate(); d.resolve(base()); await pending; expect(c.info).toBeNull()
    await c.refresh(); expect(c.unsupported).toBe(true); await c.refresh(); expect(ask).toHaveBeenCalledTimes(2)
    c.reconnect(); await c.refresh(); expect(c.info?.name).toBe('Model')
  })
  it('refuses double reloads and returns the refusal', async () => {
    const d = deferred<PhoneModeInfo>(); const reload = vi.fn(() => d.promise)
    const c = new ModeController('reload', 'writing', 'm', async () => base(), reload, () => {})
    await c.refresh(); const first = c.reload(); await c.reload(); expect(reload).toHaveBeenCalledTimes(1)
    d.reject(new Error('Model is busy.')); expect(await first).toBe('Model is busy.')
  })
  it('takes a reload press while a background question is on its way, and drops that answer', async () => {
    const d = deferred<PhoneModeInfo>()
    const loading = { ...base(), loaded: false, reload: null }
    const ask = vi.fn().mockResolvedValueOnce(base()).mockReturnValueOnce(d.promise).mockResolvedValue(loading)
    const reload = vi.fn(async () => loading)
    const c = new ModeController('press', 'writing', 'm', ask, reload, () => {})
    await c.refresh(); const pending = c.refresh(); await c.reload()
    expect(reload).toHaveBeenCalledTimes(1); expect(c.loading).toBe(true)
    d.resolve(base()); await pending; expect(c.info?.loaded).toBe(false); expect(c.loading).toBe(true)
  })
  it('keeps saying it loads, and tells a refusal, when the link flaps during the reload', async () => {
    const d = deferred<PhoneModeInfo>(); const d2 = deferred<PhoneModeInfo>()
    const loading = { ...base(), loaded: false, reload: null }
    const c = new ModeController('flap', 'writing', 'm', vi.fn().mockResolvedValueOnce(base()).mockResolvedValue(loading), vi.fn().mockReturnValueOnce(d.promise).mockReturnValueOnce(d2.promise), () => {})
    await c.refresh(); const first = c.reload(); c.reconnect(); d.resolve(loading); expect(await first).toBeNull(); expect(c.loading).toBe(true)
    await c.refresh(); c.info = base(); const second = c.reload(); c.invalidate(); d2.reject(new Error('Model is busy.')); expect(await second).toBe('Model is busy.')
  })
  it('stops saying it is loading when the model never comes back', async () => {
    vi.useFakeTimers()
    const gone = { ...base(), loaded: false, reload: null }
    const c = new ModeController('gone', 'writing', 'm', vi.fn().mockResolvedValueOnce(base()).mockResolvedValue(gone), async () => gone, () => {})
    await c.refresh(); await c.reload(); await c.refresh(); expect(c.loading).toBe(true)
    vi.advanceTimersByTime(LOADING_LIMIT_MS + 1); await c.refresh(); expect(c.loading).toBe(false)
    vi.useRealTimers()
  })
  it('handles rentals and older computers', () => {
    const old = { id: 'old', name: 'Old', loaded: false }
    expect(canPickModel(old, null)).toBe(true)
    const rental = { ...old, rental: { label: null, reconnecting: false } }
    expect(canPickModel(rental, null)).toBe(false); expect(canPickModel(rental, 'old')).toBe(true)
    expect(rentalLabel(rental)).toBe('Not loaded on a rental')
    expect(rentalLabel({ ...rental, rental: { label: 'GPU', reconnecting: true } })).toBe('GPU')
  })
})
