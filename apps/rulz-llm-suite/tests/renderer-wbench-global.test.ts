/// <reference lib="dom" />
// Writing benchmark setup, "all candidates" settings: the merge order (shared base, own wins), the
// load diff and the names shown. No component is rendered.
import { describe, expect, it, vi } from 'vitest'
import { DEFAULT_LOAD_CONFIG } from '../src/shared/defaults'
import { FICTION_SAMPLING } from '../src/shared/novelPrompts'
import type { LoadConfig, SamplingParams } from '../src/shared/types'

vi.mock('sonner', () => ({ toast: Object.assign(vi.fn(), { error: vi.fn(), message: vi.fn(), success: vi.fn(), warning: vi.fn() }) }))
vi.mock('../src/renderer/src/lib/api', () => ({
  invoke: async () => undefined,
  on: () => () => undefined,
  errorMessage: (e: unknown) => (e instanceof Error ? e.message : String(e))
}))

interface SetupFormApi {
  effectiveSampling: (base: SamplingParams, all: Partial<SamplingParams>, own: Partial<SamplingParams>) => SamplingParams
  effectiveLoad: (saved: LoadConfig, all: Partial<LoadConfig>, own: Partial<LoadConfig>) => LoadConfig | null
  diffLoad: (next: LoadConfig, shown: LoadConfig) => Partial<LoadConfig>
  roundLoad: (config: LoadConfig) => LoadConfig
  settingName: (key: string) => string
  ownNames: (own: { sampling: Partial<SamplingParams>; load: Partial<LoadConfig> }) => string[]
}

// The app store reads window and localStorage when it loads.
;(globalThis as unknown as { window: unknown }).window = { rulz: { invoke: async () => undefined, on: () => () => undefined }, addEventListener: () => undefined, matchMedia: () => ({ matches: true, addEventListener: () => undefined }) }
;(globalThis as unknown as { localStorage: unknown }).localStorage = { getItem: () => null, setItem: () => undefined, removeItem: () => undefined }

// Loaded by a computed path: the node tsconfig has no JSX setting, so it cannot type-check a .tsx import.
const setupFormPath = '../src/renderer/src/components/wbench/SetupForm.tsx'
const { diffLoad, effectiveLoad, effectiveSampling, ownNames, roundLoad, settingName } = (await import(/* @vite-ignore */ setupFormPath)) as SetupFormApi

const saved = { ...DEFAULT_LOAD_CONFIG, contextLength: 8192, threads: 8 }

describe('effectiveSampling', () => {
  it('uses the select choice when nothing is set', () => {
    expect(effectiveSampling(FICTION_SAMPLING, {}, {})).toEqual(FICTION_SAMPLING)
  })
  it('applies all-candidates values over the choice', () => {
    expect(effectiveSampling(FICTION_SAMPLING, { temperature: 0.5 }, {}).temperature).toBe(0.5)
  })
  it('lets the candidate win, setting by setting', () => {
    const s = effectiveSampling(FICTION_SAMPLING, { temperature: 0.5, topK: 10 }, { temperature: 1.2 })
    expect(s.temperature).toBe(1.2)
    expect(s.topK).toBe(10)
    expect(s.minP).toBe(FICTION_SAMPLING.minP)
  })
})

describe('effectiveLoad', () => {
  it('is null (the model\'s saved settings) when nobody changed a load setting', () => {
    expect(effectiveLoad(saved, {}, {})).toBeNull()
  })
  it('layers saved, all and own, with no adapters', () => {
    const l = effectiveLoad({ ...saved, adapters: [{ id: 'a', scale: 1 }] } as typeof saved, { contextLength: 16384, threads: 4 }, { contextLength: 4096 })
    expect(l?.contextLength).toBe(4096)
    expect(l?.threads).toBe(4)
    expect(l?.adapters).toEqual([])
  })
  it('applies all-candidates values when the candidate changed nothing', () => {
    expect(effectiveLoad(saved, { threads: 2 }, {})?.threads).toBe(2)
    expect(effectiveLoad(saved, { threads: 2 }, {})?.contextLength).toBe(8192)
  })
  it('applies a candidate\'s own change when nothing is set for all', () => {
    expect(effectiveLoad(saved, {}, { contextLength: 2048 })?.contextLength).toBe(2048)
  })
  it('rounds whole-number fields', () => {
    expect(effectiveLoad(saved, { contextLength: 4096.4 }, {})?.contextLength).toBe(4096)
  })
})

describe('diffLoad', () => {
  it('records only the changed fields', () => {
    expect(diffLoad({ ...saved, threads: 12 }, saved)).toEqual({ threads: 12 })
  })
  it('ignores adapters', () => {
    expect(diffLoad({ ...saved, adapters: [{ id: 'a', scale: 1 }] } as typeof saved, saved)).toEqual({})
  })
  it('rounds before comparing', () => {
    expect(diffLoad({ ...saved, contextLength: 8192.2 }, saved)).toEqual({})
    expect(diffLoad({ ...saved, contextLength: 4096.6 }, saved)).toEqual({ contextLength: 4097 })
  })
  it('compares nested values by content', () => {
    expect(diffLoad({ ...saved, sampling: { ...saved.sampling } }, saved)).toEqual({})
  })
})

describe('roundLoad and names', () => {
  it('rounds only the whole-number fields', () => {
    const r = roundLoad({ ...saved, contextLength: 100.5, threads: 3.2 })
    expect(r.contextLength).toBe(101)
    expect(r.threads).toBe(3)
  })
  it('turns keys into words', () => {
    expect(settingName('contextLength')).toBe('context length')
    expect(settingName('temperature')).toBe('temperature')
  })
  it('lists a candidate\'s own changes', () => {
    expect(ownNames({ sampling: { temperature: 1 }, load: { contextLength: 4096 } })).toEqual(['temperature', 'context length'])
  })
})
